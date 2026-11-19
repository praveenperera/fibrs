//! The CUDA backend: the bricks each launch reads go up, centrality and seeds are computed per tile on the device,
//! and only the centrality bricks holding a nonzero value come back, with the seed list
//!
//! The family's tiles (the reference's 128³ tiles in its order, with their 8-voxel halo) are grouped into units:
//! runs of up to [`UNIT_TILES`] adjacent tiles along x sharing their z and y tile rows. A unit uploads the bricks
//! of its box of the brick grid from the four volumes the kernels read (both pooled predictions, nx and ny), as one
//! pool of payloads and a table per volume, and runs five kernels over its tiles: gather, centrality, seeds, then
//! slots and pack, which pack the nonzero centrality bricks densely for the download. Two units are in flight at a
//! time on two streams, so the host stages one unit's bricks and installs another's results while the device works

use std::sync::Arc;

use color_eyre::eyre::{Result, WrapErr, bail, ensure, eyre};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PinnedHostSlice, PushKernelArg,
    ValidAsZeroBits, sys,
};
use cudarc::nvrtc::Ptx;
use rayon::prelude::*;

use crate::fields::{FamilyFields, Seed, TILE, TILE_HALO, tiles};
use crate::stage1::{Arith, Fused};
use crate::vol::{BRICK, EDGE, SHIFT, Shape, Vol};

/// The kernels for sm_80, which the driver compiles for the GPU it runs on; `scripts/build-ptx.sh` builds it
const PTX: &str = include_str!("../../ptx/fibrs.sm80.ptx");

/// The kernels' table entry for a brick with no payload
const NO_BRICK: u32 = u32::MAX;
/// u32 fields per tile descriptor, as the kernels read them
const DESC: usize = 16;
/// How far beyond its core the seed test reads a tile's centrality, so the band `fb_centrality` covers
const BAND_REACH: usize = 3;
/// Threads per block of every launch
const BLOCK: u32 = 256;
/// Most tiles in one unit: bounds the pinned host memory each of the two units in flight holds (at most about
/// 16 MB of bricks and 8 MB of centrality per tile, far less in a band's mostly empty bricks)
const UNIT_TILES: usize = 32;
/// Oldest driver fibrs opens, as `cuDriverGetVersion` reports it (1000 × major + 10 × minor): cudarc's bindings are
/// CUDA 12's
const MIN_DRIVER: i32 = 12000;
/// Most device memory one tile can need in a slot, `fit`'s spare half included: the brick pool and tables of the
/// four volumes over its region's bricks, three u8 and one u16 scratch arrays over its region, the u16 output and
/// its packed copy, the seed list, and the per-brick flags, slots and ids
const TILE_DEVICE_BYTES: usize = {
    let side = TILE + 2 * TILE_HALO;
    let region = side.pow(3);
    let region_bricks = (side / EDGE + 1).pow(3);
    let core = TILE.pow(3);
    let bytes =
        4 * region_bricks * (BRICK + 4) + 5 * region + 2 * 2 * core + 8 * core.div_ceil(2) + 3 * 4 * core / BRICK;
    bytes + bytes / 2
};

/// One PTX parameter's kind
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Param {
    Ptr,
    U32,
}

use Param::{Ptr, U32};

/// Each kernel's PTX params, in the order every launch below passes them; checked against the embedded PTX when
/// the device opens, so a stale PTX fails there instead of launching with mismatched arguments
const KERNELS: [(&str, &[Param]); 5] = [
    ("fb_gather", &[Ptr, Ptr, Ptr, U32, U32, U32, U32, U32, U32, U32, Ptr, Ptr, Ptr]),
    ("fb_centrality", &[Ptr, Ptr, Ptr, Ptr, Ptr, Ptr]),
    ("fb_seeds", &[Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, U32, U32, Ptr, Ptr, Ptr, U32, Ptr]),
    ("fb_slots", &[Ptr, U32, Ptr, Ptr, Ptr]),
    ("fb_pack", &[Ptr, U32, Ptr, Ptr]),
];

/// The params of kernel `name` in `ptx`, in order, read from the PTX text
fn entry_params(ptx: &str, name: &str) -> Result<Vec<Param>> {
    let entry = format!(".entry {name}(");
    let Some(start) = ptx.find(&entry) else {
        bail!("kernel {name} missing from the embedded PTX; run scripts/build-ptx.sh");
    };
    let start = start + entry.len();
    let end = ptx[start..].find(')').ok_or_else(|| eyre!("kernel {name}: unterminated PTX param list"))?;
    ptx[start..start + end]
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(|line| {
            let ty = line.strip_prefix(".param ").and_then(|rest| rest.split_whitespace().next());
            match (ty, line.contains(" .ptr ")) {
                (Some(".u64"), true) => Ok(Ptr),
                (Some(".u32"), false) => Ok(U32),
                _ => Err(eyre!("kernel {name}: unexpected PTX param `{line}`")),
            }
        })
        .collect()
}

struct Kernels {
    gather: CudaFunction,
    centrality: CudaFunction,
    seeds: CudaFunction,
    slots: CudaFunction,
    pack: CudaFunction,
}

/// An open CUDA device with the kernels loaded
pub struct Gpu {
    ctx: Arc<CudaContext>,
    /// The device's index, model and compute capability, for the log
    name: String,
    k: Kernels,
    /// Device memory free when it opened, which picks the device
    free: usize,
    /// `decode_dir` in the reference's CUDA rounding for every `nx << 8 | ny`, four float64 per entry
    dirs: CudaSlice<f64>,
}

impl Gpu {
    /// Open CUDA device `device` (an index among the visible devices), or the first usable visible device, and load
    /// the kernels
    pub fn open(device: Option<usize>) -> Result<Self> {
        // SAFETY: probing for the driver library has no preconditions; cudarc panics on first use without it
        if !unsafe { sys::is_culib_present() } {
            bail!("no NVIDIA driver found (libcuda.so)");
        }
        // cudarc loads each driver function on first call and panics when the driver lacks it, so an older driver
        // is turned away before any of them runs
        let mut version = 0;
        // SAFETY: cuDriverGetVersion only writes the version through the pointer, and exists in every driver
        let status = unsafe { sys::cuDriverGetVersion(&mut version) };
        ensure!(status == sys::CUresult::CUDA_SUCCESS, "read the NVIDIA driver version: {status:?}");
        ensure!(
            version >= MIN_DRIVER,
            "the NVIDIA driver supports CUDA {}.{}; fibrs needs a driver for CUDA {}.{} or newer",
            version / 1000,
            version % 1000 / 10,
            MIN_DRIVER / 1000,
            MIN_DRIVER % 1000 / 10
        );
        if let Some(device) = device {
            return Self::open_device(device);
        }

        // a host can list an older or busier GPU first, so each visible device gets its turn: the first with room
        // for a full tile in each slot wins, and failing that the one with the most free memory, which a small
        // volume may fit
        let count = CudaContext::device_count().map_err(|e| eyre!("count the CUDA devices: {e:?}"))?;
        let mut reasons = Vec::new();
        let mut cramped = None;
        for device in 0..count as usize {
            match Self::open_device(device) {
                Ok(gpu) if gpu.free >= 2 * TILE_DEVICE_BYTES => return Ok(gpu),
                Ok(gpu) if cramped.as_ref().is_none_or(|c: &Gpu| gpu.free > c.free) => cramped = Some(gpu),
                Ok(_) => {}
                Err(e) => reasons.push(format!("{e:#}")),
            }
        }
        if let Some(gpu) = cramped {
            return Ok(gpu);
        }
        match reasons.len() {
            0 => bail!("no CUDA device is visible"),
            1 => bail!("{}", reasons[0]),
            _ => bail!("no CUDA device is usable: {}", reasons.join("; ")),
        }
    }

    fn open_device(device: usize) -> Result<Self> {
        // cudarc passes the index to the driver as an i32, so a larger one would wrap to another device
        ensure!(i32::try_from(device).is_ok(), "CUDA device index {device} is out of range");
        let ctx = CudaContext::new(device).map_err(|e| {
            let reason = e.error_string().map_or_else(|_| format!("{e:?}"), |s| s.to_string_lossy().into_owned());
            eyre!("cannot open CUDA device {device}: {reason}")
        })?;
        let (major, minor) = ctx.compute_capability()?;
        ensure!(major >= 8, "CUDA device {device} has compute capability {major}.{minor}; fibrs needs 8.0 or newer");
        for (name, want) in KERNELS {
            let got = entry_params(PTX, name)?;
            ensure!(got == want, "kernel {name}: PTX params {got:?}, expected {want:?}; run scripts/build-ptx.sh");
        }
        let module = ctx.load_module(Ptx::from_src(PTX)).wrap_err("load the fibrs CUDA kernels")?;
        let function = |name: &str| module.load_function(name).wrap_err_with(|| format!("kernel {name}"));
        let k = Kernels {
            gather: function("fb_gather")?,
            centrality: function("fb_centrality")?,
            seeds: function("fb_seeds")?,
            slots: function("fb_slots")?,
            pack: function("fb_pack")?,
        };
        let table: Vec<f64> = Fused::dir_table().iter().flat_map(|d| [d[0], d[1], d[2], 0.0]).collect();
        let stream = ctx.default_stream();
        let dirs = stream.clone_htod(&table)?;
        stream.synchronize()?;
        let (free, _) = ctx.mem_get_info()?;
        let name = format!("CUDA device {device}, {} (compute capability {major}.{minor})", ctx.name()?);
        Ok(Self { ctx, name, k, free, dirs })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Most tiles per unit, so both slots' buffers fit in half the device memory free now, as other processes may
    /// have taken some since the device opened
    ///
    /// The other half is left for the driver, other processes and later stages. A device too full for even one tile
    /// per unit fails in `centrality` unless the volume's tiles are small, which `--device auto` answers by running
    /// on the CPU
    fn unit_tiles(&self) -> Result<usize> {
        let (free, _) = self.ctx.mem_get_info()?;
        Ok((free / 2 / (2 * TILE_DEVICE_BYTES)).clamp(1, UNIT_TILES))
    }

    /// One family's centrality and seeds (1 = V, 2 = H), equal to `fields::centrality::<Fused>` of the family's
    /// presence
    pub fn centrality(
        &self,
        tracing: &[Vol<u8>; 2],
        family: usize,
        nx: &Vol<u8>,
        ny: &Vol<u8>,
    ) -> Result<FamilyFields> {
        let shape = nx.shape();
        let vols = [&tracing[0], &tracing[1], nx, ny];
        ensure!(vols.iter().all(|v| v.shape() == shape), "the pooled predictions and direction field differ in shape");
        // the kernels' defaults for bricks without a payload
        ensure!(vols.iter().zip([0, 0, 128, 128]).all(|(v, d)| v.default_value() == d), "unexpected volume defaults");
        ensure!(
            shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d)).is_some_and(|n| n <= Seed::MAX_VOXELS)
                && shape.iter().all(|&n| u32::try_from(n).is_ok()),
            "volume of {shape:?} voxels is too large to index seeds"
        );
        let units = units(&tracing[family - 1], shape, self.unit_tiles()?);
        let mut cent = Vol::new(shape, 0u16);
        let mut seeds = Vec::new();
        let mut slots = [Slot::new(&self.ctx)?, Slot::new(&self.ctx)?];
        // unit i stages and launches on slot i % 2 while unit i - 1 finishes on the other; a slot's previous
        // unit was collected (its stream synchronized) before the slot stages again
        let mut pending: Option<(usize, Meta)> = None;
        for (i, unit) in units.iter().enumerate() {
            let s = i % 2;
            let meta = slots[s].stage(unit, &vols, shape)?;
            slots[s].launch(self, &meta, family, shape)?;
            if let Some((p, m)) = pending.replace((s, meta)) {
                slots[p].collect(&m, &mut cent, &mut seeds)?;
            }
        }
        if let Some((p, m)) = pending {
            slots[p].collect(&m, &mut cent, &mut seeds)?;
        }
        // the catalogue reads seeds in C order over the whole volume; indices are unique, so any sort agrees
        seeds.sort_unstable();
        seeds.shrink_to_fit();
        Ok(FamilyFields { cent, seeds })
    }
}

/// A tile core's origin and end
type TileSpan = (Shape, Shape);

/// The region a tile's kernels read: its core and halo, clipped to the volume, as origin and extent
fn region((start, end): TileSpan, shape: Shape) -> (Shape, Shape) {
    let lo = start.map(|v| v.saturating_sub(TILE_HALO));
    let hi = [0, 1, 2].map(|a| (end[a] + TILE_HALO).min(shape[a]));
    (lo, [0, 1, 2].map(|a| hi[a] - lo[a]))
}

/// The family's tiles in the reference's order, grouped into units of up to `unit_tiles` adjacent tiles along x
///
/// A tile whose region holds no brick of the family's own pooled prediction has presence 0 throughout, so no
/// centrality and no seeds, and is left out; the CPU path skips those tiles and every other tile under presence 64
fn units(own: &Vol<u8>, shape: Shape, unit_tiles: usize) -> Vec<Vec<TileSpan>> {
    let all = tiles(shape, TILE);
    let keep: Vec<bool> = all
        .par_iter()
        .map(|&t| {
            let (lo, dims) = region(t, shape);
            own.any_allocated_in(lo, dims, |_, _, _| true)
        })
        .collect();
    let mut units: Vec<Vec<TileSpan>> = Vec::new();
    let mut prev: Option<Shape> = None;
    for (&t, keep) in all.iter().zip(keep) {
        if !keep {
            prev = None;
            continue;
        }
        let start = t.0;
        let adjacent = prev.is_some_and(|p| p[0] == start[0] && p[1] == start[1] && p[2] + TILE == start[2]);
        match units.last_mut() {
            Some(unit) if adjacent && unit.len() < unit_tiles => unit.push(t),
            _ => units.push(vec![t]),
        }
        prev = Some(start);
    }
    units
}

/// What a staged unit's launches and collection need besides its device buffers
struct Meta {
    tiles: u32,
    /// The unit's box of bricks: first brick and extent, z, y, x
    b0: Shape,
    nb: Shape,
    /// Voxels of every tile's region, the scratch arrays' length
    region_voxels: usize,
    /// The largest region, band and core of any one tile, which size the per-tile launches
    max_region: usize,
    max_band: usize,
    max_core: usize,
    /// Each output brick's index in the volume's brick grid, in the order the tiles' output bricks lie on device
    bricks: Vec<usize>,
    /// Room in the seed list: every even voxel of every core, more than the seed test can pass
    cap: usize,
}

/// One unit in flight: a stream with the device buffers and pinned host buffers it uses, grown as units need
struct Slot {
    stream: Arc<CudaStream>,
    desc: CudaSlice<u32>,
    table: CudaSlice<u32>,
    pool: CudaSlice<u8>,
    pres: CudaSlice<u8>,
    nx: CudaSlice<u8>,
    ny: CudaSlice<u8>,
    cent: CudaSlice<u16>,
    out: CudaSlice<u16>,
    flags: CudaSlice<u32>,
    slots: CudaSlice<u32>,
    ids: CudaSlice<u32>,
    packed: CudaSlice<u16>,
    seeds: CudaSlice<u64>,
    /// The seed list's length and the number of packed bricks
    count: CudaSlice<u32>,
    /// Write-combined, as the host only writes it
    staging: PinnedHostSlice<u8>,
    host_count: PinnedHostSlice<u32>,
    host_packed: PinnedHostSlice<u16>,
    host_ids: PinnedHostSlice<u32>,
    host_seeds: PinnedHostSlice<u64>,
    /// The table and descriptors of the unit in flight: the driver may read pageable memory after an asynchronous
    /// copy returns, so they live until the slot's stream is synchronized
    host_table: Vec<u32>,
    host_desc: Vec<u32>,
}

/// Grow a device buffer to hold `n` elements, by half again beyond, so slowly growing units reallocate rarely
fn fit<T: DeviceRepr + ValidAsZeroBits>(stream: &Arc<CudaStream>, buf: &mut CudaSlice<T>, n: usize) -> Result<()> {
    if buf.len() < n {
        *buf = stream.alloc_zeros(n + n / 2).wrap_err("allocate GPU memory")?;
    }
    Ok(())
}

/// `n` elements of pinned host memory, zeroed, so the typed slices made over it never cover uninitialized memory
fn pinned<T: DeviceRepr + ValidAsZeroBits>(ctx: &Arc<CudaContext>, n: usize, flags: u32) -> Result<PinnedHostSlice<T>> {
    // SAFETY: the allocation is zeroed below, before any slice is made over it
    let mut buf = unsafe { ctx.alloc_pinned_with_flags::<T>(n, flags) }.wrap_err("allocate pinned host memory")?;
    // SAFETY: the pointer is valid for `n` elements, and all-zero bytes are a valid `T`
    unsafe { buf.as_mut_ptr()?.write_bytes(0, n) };
    Ok(buf)
}

/// Grow a pinned host buffer to hold `n` elements, as [`fit`] does
///
/// The copies take plain slices of these buffers, which cudarc does not track, so the stream is synchronized
/// before the old buffer is freed
fn fit_host<T: DeviceRepr + ValidAsZeroBits>(
    stream: &Arc<CudaStream>,
    buf: &mut PinnedHostSlice<T>,
    n: usize,
    flags: u32,
) -> Result<()> {
    if buf.len() < n {
        stream.synchronize()?;
        *buf = pinned(stream.context(), n + n / 2, flags)?;
    }
    Ok(())
}

fn grid_1d(n: usize) -> Result<u32> {
    u32::try_from(n.div_ceil(BLOCK as usize)).map_err(|_| eyre!("launch of {n} threads is too large"))
}

impl Slot {
    fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        let stream = ctx.new_stream()?;
        Ok(Self {
            staging: pinned(ctx, 1, sys::CU_MEMHOSTALLOC_WRITECOMBINED)?,
            host_count: pinned(ctx, 2, 0)?,
            host_packed: pinned(ctx, 1, 0)?,
            host_ids: pinned(ctx, 1, 0)?,
            host_seeds: pinned(ctx, 1, 0)?,
            host_table: Vec::new(),
            host_desc: Vec::new(),
            desc: stream.alloc_zeros(1)?,
            table: stream.alloc_zeros(1)?,
            pool: stream.alloc_zeros(1)?,
            pres: stream.alloc_zeros(1)?,
            nx: stream.alloc_zeros(1)?,
            ny: stream.alloc_zeros(1)?,
            cent: stream.alloc_zeros(1)?,
            out: stream.alloc_zeros(1)?,
            flags: stream.alloc_zeros(1)?,
            slots: stream.alloc_zeros(1)?,
            ids: stream.alloc_zeros(1)?,
            packed: stream.alloc_zeros(1)?,
            seeds: stream.alloc_zeros(1)?,
            count: stream.alloc_zeros(2)?,
            stream,
        })
    }

    /// Build a unit's brick pool, tables and tile descriptors and start their upload
    fn stage(&mut self, unit: &[TileSpan], vols: &[&Vol<u8>; 4], shape: Shape) -> Result<Meta> {
        let grid = vols[0].grid();
        let regions: Vec<(Shape, Shape)> = unit.iter().map(|&t| region(t, shape)).collect();
        // the tiles share their z and y rows and run along x, so the first and last regions bound the unit
        let lo = regions[0].0;
        let (last_lo, last_dims) = regions[regions.len() - 1];
        let hi = [0, 1, 2].map(|a| last_lo[a] + last_dims[a]);
        let b0 = lo.map(|v| v >> SHIFT);
        let nb = [0, 1, 2].map(|a| ((hi[a] - 1) >> SHIFT) + 1 - b0[a]);
        let n_box = nb.iter().product::<usize>();
        let mut table = vec![NO_BRICK; 4 * n_box];
        let mut payloads: Vec<&[u8; BRICK]> = Vec::new();
        let mut entry = table.iter_mut();
        for vol in vols {
            for bz in b0[0]..b0[0] + nb[0] {
                for by in b0[1]..b0[1] + nb[1] {
                    let row = (bz * grid[1] + by) * grid[2];
                    for bx in b0[2]..b0[2] + nb[2] {
                        let slot = entry.next().expect("the table has 4 entries per brick of the box");
                        if let Some(b) = vol.brick_at(row + bx) {
                            *slot = payloads.len() as u32;
                            payloads.push(b);
                        }
                    }
                }
            }
        }
        ensure!(payloads.len() < NO_BRICK as usize, "a unit holds too many bricks");

        let mut desc = Vec::with_capacity(unit.len() * DESC);
        let mut bricks = Vec::new();
        let (mut base, mut out, mut cap) = (0usize, 0usize, 0usize);
        let (mut max_region, mut max_band, mut max_core) = (0, 0, 0);
        for (&(start, end), &(rlo, dims)) in unit.iter().zip(&regions) {
            let off = [0, 1, 2].map(|a| start[a] - rlo[a]);
            let core = [0, 1, 2].map(|a| end[a] - start[a]);
            let band =
                [0, 1, 2].map(|a| (off[a] + core[a] + BAND_REACH).min(dims[a]) - off[a].saturating_sub(BAND_REACH));
            let fields = [rlo, dims, off, core].concat();
            desc.extend(fields.iter().map(|&v| v as u32));
            desc.extend([base as u32, out as u32, 0, 0]);
            let cb = core.map(|n| n.div_ceil(EDGE));
            let sb = start.map(|v| v >> SHIFT);
            for z in sb[0]..sb[0] + cb[0] {
                for y in sb[1]..sb[1] + cb[1] {
                    bricks.extend((sb[2]..sb[2] + cb[2]).map(|x| (z * grid[1] + y) * grid[2] + x));
                }
            }
            let region_voxels = dims.iter().product::<usize>();
            base += region_voxels;
            out += cb.iter().product::<usize>() * BRICK;
            cap += core.iter().product::<usize>().div_ceil(2);
            max_region = max_region.max(region_voxels);
            max_band = max_band.max(band.iter().product());
            max_core = max_core.max(core.iter().product());
        }
        // the kernels index the scratch arrays and the output in u32
        ensure!(base <= u32::MAX as usize && out <= u32::MAX as usize, "a unit is too large for the kernels' indices");

        let n_pool = payloads.len() * BRICK;
        fit_host(&self.stream, &mut self.staging, n_pool, sys::CU_MEMHOSTALLOC_WRITECOMBINED)?;
        // the pool gathers the scattered bricks into pinned memory, so its upload is one asynchronous copy
        let host = &mut self.staging.as_mut_slice()?[..n_pool];
        host.par_chunks_mut(BRICK).zip(payloads.par_iter()).for_each(|(d, s)| d.copy_from_slice(&s[..]));
        fit(&self.stream, &mut self.pool, n_pool)?;
        fit(&self.stream, &mut self.table, table.len())?;
        fit(&self.stream, &mut self.desc, desc.len())?;
        // the slot's previous unit was collected, so its stream no longer reads the buffers replaced here
        self.host_table = table;
        self.host_desc = desc;
        self.stream.memcpy_htod(&self.host_table, &mut self.table.slice_mut(..self.host_table.len()))?;
        self.stream.memcpy_htod(&self.host_desc, &mut self.desc.slice_mut(..self.host_desc.len()))?;
        if n_pool > 0 {
            // pinned: the copy runs after this returns, and the staging buffer is not written again until this
            // slot's stream is synchronized in `collect`
            self.stream.memcpy_htod(&self.staging.as_slice()?[..n_pool], &mut self.pool.slice_mut(..n_pool))?;
        }
        Ok(Meta { tiles: unit.len() as u32, b0, nb, region_voxels: base, max_region, max_band, max_core, bricks, cap })
    }

    /// Queue a staged unit's kernels and the copy of its counts
    fn launch(&mut self, gpu: &Gpu, m: &Meta, family: usize, shape: Shape) -> Result<()> {
        let n_out = m.bricks.len() * BRICK;
        let n_bricks = m.bricks.len();
        let stream = self.stream.clone();
        for buf in [&mut self.pres, &mut self.nx, &mut self.ny] {
            fit(&stream, buf, m.region_voxels)?;
        }
        fit(&stream, &mut self.cent, m.region_voxels)?;
        fit(&stream, &mut self.out, n_out)?;
        fit(&stream, &mut self.packed, n_out)?;
        for buf in [&mut self.flags, &mut self.slots, &mut self.ids] {
            fit(&stream, buf, n_bricks)?;
        }
        fit(&stream, &mut self.seeds, m.cap)?;
        // the output's bricks past a core's edge are never written, and read as centrality 0
        stream.memset_zeros(&mut self.out.slice_mut(..n_out))?;
        stream.memset_zeros(&mut self.flags.slice_mut(..n_bricks))?;
        stream.memset_zeros(&mut self.count)?;

        let per_tile = |n: usize| -> Result<LaunchConfig> {
            Ok(LaunchConfig { grid_dim: (grid_1d(n)?, m.tiles, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 })
        };
        let flat = |n: usize| -> Result<LaunchConfig> {
            Ok(LaunchConfig { grid_dim: (grid_1d(n)?, 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 })
        };
        let nb_total = (m.nb.iter().product::<usize>()) as u32;
        let [bz0, by0, bx0] = m.b0.map(|v| v as u32);
        let (nby, nbx) = (m.nb[1] as u32, m.nb[2] as u32);
        let family = family as u32;
        let (shape_h, shape_w) = (shape[1] as u32, shape[2] as u32);
        let cap = u32::try_from(m.cap).map_err(|_| eyre!("a unit's seed list is too long"))?;
        let (n_bricks_u32, n_out_u32) = (n_bricks as u32, n_out as u32);
        let dirs = &gpu.dirs;
        // SAFETY: each launch passes the kernel's params in its PTX order (checked in `Gpu::open`), every buffer
        // holds what the kernel indexes (sized above from the descriptors), and all run in order on this stream
        unsafe {
            stream
                .launch_builder(&gpu.k.gather)
                .arg(&self.desc)
                .arg(&self.pool)
                .arg(&self.table)
                .arg(&nb_total)
                .arg(&bz0)
                .arg(&by0)
                .arg(&bx0)
                .arg(&nby)
                .arg(&nbx)
                .arg(&family)
                .arg(&mut self.pres)
                .arg(&mut self.nx)
                .arg(&mut self.ny)
                .launch(per_tile(m.max_region)?)?;
            stream
                .launch_builder(&gpu.k.centrality)
                .arg(&self.desc)
                .arg(&self.pres)
                .arg(&self.nx)
                .arg(&self.ny)
                .arg(dirs)
                .arg(&mut self.cent)
                .launch(per_tile(m.max_band)?)?;
            stream
                .launch_builder(&gpu.k.seeds)
                .arg(&self.desc)
                .arg(&self.pres)
                .arg(&self.nx)
                .arg(&self.ny)
                .arg(dirs)
                .arg(&self.cent)
                .arg(&shape_h)
                .arg(&shape_w)
                .arg(&mut self.out)
                .arg(&mut self.flags)
                .arg(&mut self.seeds)
                .arg(&cap)
                .arg(&mut self.count)
                .launch(per_tile(m.max_core)?)?;
            let mut brick_count = self.count.slice_mut(1..2);
            stream
                .launch_builder(&gpu.k.slots)
                .arg(&self.flags)
                .arg(&n_bricks_u32)
                .arg(&mut self.slots)
                .arg(&mut self.ids)
                .arg(&mut brick_count)
                .launch(flat(n_bricks)?)?;
            stream
                .launch_builder(&gpu.k.pack)
                .arg(&self.out)
                .arg(&n_out_u32)
                .arg(&self.slots)
                .arg(&mut self.packed)
                .launch(flat(n_out)?)?;
        }
        stream.memcpy_dtoh(&self.count, &mut self.host_count.as_mut_slice()?[..2])?;
        Ok(())
    }

    /// Wait for a launched unit, then install its nonzero centrality bricks and append its seeds
    fn collect(&mut self, m: &Meta, cent: &mut Vol<u16>, seeds: &mut Vec<Seed>) -> Result<()> {
        self.stream.synchronize().wrap_err("run the centrality kernels")?;
        let (n_seeds, n_packed) = {
            let c = self.host_count.as_slice()?;
            (c[0] as usize, c[1] as usize)
        };
        ensure!(n_seeds <= m.cap, "{n_seeds} seeds overflow a unit's list of {}", m.cap);
        ensure!(n_packed <= m.bricks.len(), "{n_packed} packed bricks of {}", m.bricks.len());
        let n_values = n_packed * BRICK;
        fit_host(&self.stream, &mut self.host_packed, n_values, 0)?;
        fit_host(&self.stream, &mut self.host_ids, n_packed, 0)?;
        fit_host(&self.stream, &mut self.host_seeds, n_seeds, 0)?;
        if n_packed > 0 {
            self.stream
                .memcpy_dtoh(&self.packed.slice(..n_values), &mut self.host_packed.as_mut_slice()?[..n_values])?;
            self.stream.memcpy_dtoh(&self.ids.slice(..n_packed), &mut self.host_ids.as_mut_slice()?[..n_packed])?;
        }
        if n_seeds > 0 {
            self.stream.memcpy_dtoh(&self.seeds.slice(..n_seeds), &mut self.host_seeds.as_mut_slice()?[..n_seeds])?;
        }
        self.stream.synchronize().wrap_err("copy the centrality back")?;
        let packed = &self.host_packed.as_slice()?[..n_values];
        for (&id, values) in self.host_ids.as_slice()?[..n_packed].iter().zip(packed.as_chunks::<BRICK>().0) {
            cent.set_brick(m.bricks[id as usize], values);
        }
        seeds.extend(self.host_seeds.as_slice()?[..n_seeds].iter().map(|&raw| Seed::from_raw(raw)));
        Ok(())
    }
}

impl Drop for Slot {
    /// Wait for the slot's queued copies before its host buffers are freed: cudarc does not track the plain
    /// slices they were given, and an error can leave a copy queued after `stage` or `launch` returns early
    fn drop(&mut self) {
        // a failed sync means the context is broken, and freeing then is all that is left to do
        let _ = self.stream.synchronize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fields::{Presence, centrality};

    #[test]
    fn launches_match_the_embedded_ptx() {
        for (name, want) in KERNELS {
            assert_eq!(entry_params(PTX, name).unwrap(), want, "{name}");
        }
        assert!(PTX.contains("\n.target sm_80\n"));
    }

    /// A small xorshift, so the test volumes need no dependency
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    /// Pooled predictions and a direction field with ridges in both families, noise, and an empty slab, over a
    /// shape whose tiles end mid-brick on every axis
    ///
    /// Tiles z 0, y 1 have empty cores, and tiles z 0, x 0, y 1 and 2 are empty to the edge of every brick their
    /// regions touch, so the GPU path leaves them out and places the tiles after each gap
    fn volumes() -> ([Vol<u8>; 2], Vol<u8>, Vol<u8>) {
        let shape = [150, 260, 270];
        let mut tracing = [Vol::new(shape, 0u8), Vol::new(shape, 0u8)];
        let (mut nx, mut ny) = (Vol::new(shape, 128u8), Vol::new(shape, 128u8));
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for z in 0..shape[0] {
            for y in 0..shape[1] {
                for x in 0..shape[2] {
                    let noise = (rng.next() % 41) as f64 - 20.0;
                    let empty_cores = z < 128 && (128..256).contains(&y);
                    let empty_tile = z < 144 && y >= 112 && x < 144;
                    if empty_cores || empty_tile {
                        continue;
                    }
                    let (zf, yf, xf) = (z as f64, y as f64, x as f64);
                    let v = 255.0 * (0.5 + 0.5 * (0.31 * xf + 0.07 * yf + 0.05 * zf).sin()).powi(3) + noise;
                    let h = 255.0 * (0.5 + 0.5 * (0.29 * yf - 0.04 * xf + 0.11 * zf).sin()).powi(3) + noise;
                    tracing[0].set(z, y, x, v.clamp(0.0, 255.0) as u8);
                    tracing[1].set(z, y, x, h.clamp(0.0, 255.0) as u8);
                    let a = 128.0 + 120.0 * (0.013 * xf + 0.021 * zf).sin() + noise;
                    let b = 128.0 + 120.0 * (0.017 * yf - 0.011 * zf).cos() - noise;
                    nx.set(z, y, x, a.clamp(0.0, 255.0) as u8);
                    ny.set(z, y, x, b.clamp(0.0, 255.0) as u8);
                }
            }
        }
        (tracing, nx, ny)
    }

    /// The fixture leaves exactly tiles z 0, x 0, y 1 and 2 out of the GPU path, in both families
    #[test]
    fn fixture_leaves_out_two_tiles() {
        let (tracing, nx, _) = volumes();
        let shape = nx.shape();
        let gaps = [[0, TILE, 0], [0, 2 * TILE, 0]];
        let want: Vec<TileSpan> = tiles(shape, TILE).into_iter().filter(|t| !gaps.contains(&t.0)).collect();
        for own in &tracing {
            assert_eq!(units(own, shape, UNIT_TILES).concat(), want);
        }
    }

    /// The kernels against `stage1::<Fused>`, the same computation on the CPU: every centrality voxel and every
    /// seed, both families
    #[test]
    #[ignore = "needs an NVIDIA GPU; run through the GPU queue"]
    fn gpu_matches_fused_cpu() {
        let gpu = Gpu::open(None).unwrap();
        let (tracing, nx, ny) = volumes();
        let shape = nx.shape();
        for family in [1, 2] {
            let got = gpu.centrality(&tracing, family, &nx, &ny).unwrap();
            let want = centrality::<Fused>(&Presence::new(&tracing, family), &nx, &ny);
            let (a, b) = (got.cent.read_block([0; 3], shape), want.cent.read_block([0; 3], shape));
            let differ = a.iter().zip(&b).filter(|(a, b)| a != b).count();
            assert_eq!(differ, 0, "family {family}: {differ} centrality voxels differ");
            assert!(b.iter().filter(|&&v| v != 0).count() > 100_000, "family {family}: too little centrality to test");
            assert!(want.seeds.len() > 1000, "family {family}: {} seeds, too few to test", want.seeds.len());
            assert!(
                got.seeds == want.seeds,
                "family {family}: {} seeds, expected {}",
                got.seeds.len(),
                want.seeds.len()
            );
        }
    }
}
