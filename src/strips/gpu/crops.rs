//! The GPU input stage: the device CT cache, the strip and frame crops, and the tube channel
//!
//! [`ChunkCache`] keeps 128^3 uint8 CT chunks resident in a device pool of 2 MiB slots, indexed by a dense table
//! over the volume's chunk grid. Before each crop launch the host works out, per crop, every chunk its samples can
//! read ([`CropGrid::chunks`]), uploads the missing ones from vc_reader's disk cache through pinned staging on a
//! copy stream, evicting the least recently used chunks no crop of the launch needs, and makes the compute stream
//! wait for the uploads. The copy stream in turn waits for the last sampler launch, so a slot is never rewritten
//! under a crop still reading it. A launch needing a chunk the disk cache lacks fails before any upload, with every
//! such chunk in its [`Uncached`] error.
//!
//! [`Crops`] samples the strip model's input, NDHWC `(B, 144, 128, 128, 2)` float32 with the CT in channel 0 and
//! villa's `render_tube` in channel 1, and the frame model's `(B, 32, 32, 32)` float32 crops, with vc_reader's CUDA
//! sampler arithmetic and villa's float64 z-score (`strips-kernels/src/crops.rs`). The frame crop is written
//! z-scored, in two sampling passes. The strip crop takes one: its CT channel is written raw beside each crop's
//! statistics, and its readers (the CNN's first conv and the head's profile tokens) z-score it as they load it, to
//! the same bits; [`StripInput`] says which of the two a strip-model input is

use std::ffi::c_void;
use std::sync::Arc;

use color_eyre::eyre::{Result, WrapErr, bail, ensure, eyre};
use cudarc::driver::result as cur;
use cudarc::driver::{CudaContext, CudaEvent, CudaStream};
use volkit::gpu::{Arg, Buf, Gpu};

use self::prefetch::Prefetcher;
use super::Device;
use crate::strips::ct::{Uncached, Volume};

mod prefetch;
use crate::strips::types::{Mat3, TUBE_POINTS};

/// Table entry of a chunk the source holds none of (strips-kernels `FILL`)
const FILL: u32 = 0xFFFF_FFFE;
/// Table entry of a chunk that is not resident (strips-kernels `ABSENT`)
const ABSENT: u32 = 0xFFFF_FFFF;
/// The chunk edge the kernels address by shifts
const CHUNK: usize = 128;
const CHUNK_BYTES: usize = CHUNK * CHUNK * CHUNK;
/// Threads reading a launch's missing chunks from the disk cache: the reads, one at a time on the engine thread, took
/// as long as every kernel of the run together (13.3 of 26.3 s on a 200-fibre PHerc0211 set, nsys)
const READERS: usize = 16;
/// Words per crop grid record (strips-kernels `GRID_WORDS`)
const GRID_WORDS: usize = 16;
/// Threads per sampler block (strips-kernels `SAMPLE_THREADS`): a 2·[`STEPS`] x 8 x 16 tile
const SAMPLE_THREADS: u32 = 256;
/// d-steps of two samples each per sampler thread
const STEPS: u32 = 8;
/// Margin, in CT voxels, around a crop's sample positions within which its trilinear reads fall (they reach at
/// most one voxel past a sample; the rest covers float32 rounding of the grid)
const READ_MARGIN: f64 = 1.5;

/// One crop's sampling grid, villa's `CropSpec`: `depth` samples along f behind and ahead of the centre, `height`
/// along v and `width` along u, `spacing` trace voxels apart, `behind` samples behind the centre
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CropSpec {
    pub depth: usize,
    pub height: usize,
    pub width: usize,
    pub behind: f64,
    pub spacing: f64,
}

/// The strip model's crop (`explainer` fine crop: depth 144, width 128, behind 72, spacing 0.5)
pub const STRIP_CROP: CropSpec = CropSpec { depth: 144, height: 128, width: 128, behind: 72.0, spacing: 0.5 };
/// The frame model's crop (`heading_model.pt` patch: depth 32, width 32, behind 4, spacing 1.25)
pub const FRAME_CROP: CropSpec = CropSpec { depth: 32, height: 32, width: 32, behind: 4.0, spacing: 1.25 };

impl CropSpec {
    /// Samples per crop
    pub fn samples(&self) -> usize {
        self.depth * self.height * self.width
    }

    /// The first sample's offsets (lateral u, vertical v, forward f) in trace voxels, as villa's
    /// `lateral_coords[0]`, `vertical_coords[0]`, `forward_coords[0]`
    fn first(&self) -> [f64; 3] {
        [
            (0.0 - (self.width as f64 - 1.0) / 2.0) * self.spacing,
            (0.0 - (self.height as f64 - 1.0) / 2.0) * self.spacing,
            (0.0 - self.behind) * self.spacing,
        ]
    }

    /// Sampler tiles per crop along (w, h, d)
    fn tiles(&self) -> Result<[u32; 3]> {
        ensure!(
            self.width.is_multiple_of(16)
                && self.height.is_multiple_of(8)
                && self.depth.is_multiple_of(2 * STEPS as usize),
            "crop {self:?}: the sampler tiles 16 x 8 x {} (w, h, d)",
            2 * STEPS
        );
        Ok([(self.width / 16) as u32, (self.height / 8) as u32, (self.depth / (2 * STEPS as usize)) as u32])
    }
}

/// One crop's affine grid as vc_reader samples it: sample `(d, h, w)` at CT zyx `base + c`,
/// `c[a] = ((origin[a] + axes[a][0]·d) + axes[a][1]·h) + axes[a][2]·w` in float32
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CropGrid {
    /// Integer origin, the floor of the first sample clamped to [0, 2^30] (vc_reader `localOrigin`)
    pub base: [i64; 3],
    /// The first sample relative to `base`
    pub origin: [f32; 3],
    /// Rows CT z, y, x; columns the steps along d (f), h (v), w (u)
    pub axes: [[f32; 3]; 3],
}

impl CropGrid {
    /// The grid of a crop centred at `pos` (trace xyz) along `frame` (columns u, v, f), `scale` CT voxels per
    /// trace voxel: villa's `crop_grids`, then vc_reader's split of the float64 origin
    pub fn new(spec: &CropSpec, pos: [f64; 3], frame: &Mat3, scale: f64) -> Self {
        let f = |r: usize, c: usize| frame[r][c] * scale;
        let first = spec.first();
        let mut grid = CropGrid { base: [0; 3], origin: [0.0; 3], axes: [[0.0; 3]; 3] };
        for a in 0..3 {
            let i = 2 - a;
            let o = pos[i] * scale + (f(i, 0) * first[0] + f(i, 1) * first[1] + f(i, 2) * first[2]);
            grid.base[a] = o.floor().clamp(0.0, (1u64 << 30) as f64) as i64;
            grid.origin[a] = (o - grid.base[a] as f64) as f32;
            for (c, axis) in grid.axes[a].iter_mut().enumerate() {
                *axis = (f(i, 2 - c) * spec.spacing) as f32;
            }
        }
        grid
    }

    /// The kernels' record: origin, axes row-major, base, padding
    fn record(&self) -> [u32; GRID_WORDS] {
        let mut r = [0u32; GRID_WORDS];
        for a in 0..3 {
            r[a] = self.origin[a].to_bits();
            for c in 0..3 {
                r[3 + 3 * a + c] = self.axes[a][c].to_bits();
            }
            r[12 + a] = self.base[a] as i32 as u32;
        }
        r
    }

    /// Appends to `out` the table index of every chunk of a volume of `shape` (`grid` chunks) whose voxels a
    /// sample of this crop can read: the chunks, grown by [`READ_MARGIN`], that meet the crop's parallelepiped on
    /// the volume's three axes and the crop's three face normals (a superset of the exact set, which would also
    /// test the nine edge-pair axes)
    pub fn chunks(&self, spec: &CropSpec, shape: [usize; 3], grid: [usize; 3], out: &mut Vec<u32>) {
        let p0: [f64; 3] = std::array::from_fn(|a| self.base[a] as f64 + self.origin[a] as f64);
        let a: [[f64; 3]; 3] = std::array::from_fn(|r| std::array::from_fn(|c| self.axes[r][c] as f64));
        let n = [(spec.depth - 1) as f64, (spec.height - 1) as f64, (spec.width - 1) as f64];
        let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for corner in 0..8 {
            let u = [n[0] * (corner & 1) as f64, n[1] * ((corner >> 1) & 1) as f64, n[2] * ((corner >> 2) & 1) as f64];
            for r in 0..3 {
                let v = p0[r] + a[r][0] * u[0] + a[r][1] * u[1] + a[r][2] * u[2];
                lo[r] = lo[r].min(v);
                hi[r] = hi[r].max(v);
            }
        }
        let mut range = [(0usize, 0usize); 3];
        for r in 0..3 {
            let (l, h) = (lo[r] - READ_MARGIN, hi[r] + READ_MARGIN);
            if h < 0.0 || l >= shape[r] as f64 {
                return;
            }
            let first = (l.max(0.0) as usize) / CHUNK;
            let last = ((h as usize).min(shape[r] - 1)) / CHUNK;
            range[r] = (first, last.min(grid[r] - 1));
        }

        // the inverse axes take a CT position to sample coordinates; the margin grows by each row's 1-norm
        let inv = invert(&a);
        let slack: [f64; 3] =
            std::array::from_fn(|i| READ_MARGIN * inv.map_or(0.0, |m| m[i].iter().map(|x| x.abs()).sum()));
        for cz in range[0].0..=range[0].1 {
            for cy in range[1].0..=range[1].1 {
                for cx in range[2].0..=range[2].1 {
                    let keep = match inv {
                        None => true,
                        Some(m) => {
                            let k = [cz, cy, cx];
                            let (mut ulo, mut uhi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
                            for corner in 0..8 {
                                let v: [f64; 3] = std::array::from_fn(|r| {
                                    (k[r] * CHUNK + ((corner >> r) & 1) * (CHUNK - 1)) as f64 - p0[r]
                                });
                                for i in 0..3 {
                                    let u = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
                                    ulo[i] = ulo[i].min(u);
                                    uhi[i] = uhi[i].max(u);
                                }
                            }
                            (0..3).all(|i| uhi[i] >= -slack[i] && ulo[i] <= n[i] + slack[i])
                        }
                    };
                    if keep {
                        out.push(((cz * grid[1] + cy) * grid[2] + cx) as u32);
                    }
                }
            }
        }
    }
}

/// The inverse of `m`, or `None` when it is singular
fn invert(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let c = |r: usize, k: usize| {
        let (r1, r2, k1, k2) = ((r + 1) % 3, (r + 2) % 3, (k + 1) % 3, (k + 2) % 3);
        m[r1][k1] * m[r2][k2] - m[r1][k2] * m[r2][k1]
    };
    let det = m[0][0] * c(0, 0) + m[0][1] * c(0, 1) + m[0][2] * c(0, 2);
    if det.abs() < 1e-12 {
        return None;
    }
    Some(std::array::from_fn(|i| std::array::from_fn(|j| c(j, i) / det)))
}

/// Page-locked host memory with an event marking its last queued copy
pub(crate) struct Pinned<T: Copy> {
    ptr: *mut T,
    len: usize,
    /// Recorded after each copy that reads or writes the buffer; the host waits on it before touching the buffer
    pub(crate) done: CudaEvent,
}

impl<T: Copy> Pinned<T> {
    pub(crate) fn new(ctx: &Arc<CudaContext>, len: usize) -> Result<Self> {
        ctx.bind_to_thread()?;
        let bytes = (len * size_of::<T>()).max(1);
        // SAFETY: a plain page-locked allocation, freed in Drop
        let ptr = unsafe { cur::malloc_host(bytes, 0) }.wrap_err_with(|| format!("pin {bytes} host bytes"))? as *mut T;
        // SAFETY: zeroed so every element is initialized; T is plain data
        unsafe { std::ptr::write_bytes(ptr as *mut u8, 0, bytes) };
        Ok(Self { ptr, len, done: ctx.new_event(None)? })
    }

    /// The buffer, once every copy queued on it has finished
    pub(crate) fn slice_mut(&mut self) -> Result<&mut [T]> {
        self.done.synchronize()?;
        // SAFETY: our allocation of `len` initialized elements, and no copy is in flight
        Ok(unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) })
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }
}

impl<T: Copy> Drop for Pinned<T> {
    fn drop(&mut self) {
        let _ = self.done.synchronize();
        // SAFETY: our allocation, and no copy is in flight
        let _ = unsafe { cur::free_host(self.ptr as *mut c_void) };
    }
}

// SAFETY: the pointer is owned and only reached through &mut self
unsafe impl<T: Copy + Send> Send for Pinned<T> {}

/// Queues a copy of `src` to device address `dst` on `stream` and marks `buf` busy until it finishes
fn upload<T: Copy>(stream: &CudaStream, dst: u64, buf: &mut Pinned<T>, len: usize) -> Result<()> {
    ensure!(len <= buf.len, "upload of {len} elements from a pinned buffer of {}", buf.len);
    if len == 0 {
        return Ok(());
    }
    // SAFETY: `buf` stays allocated and untouched until `done` completes, which the host waits for before reuse
    unsafe {
        let src = std::slice::from_raw_parts(buf.ptr, len);
        cur::memcpy_htod_async(dst, src, stream.cu_stream())?;
    }
    buf.done.record(stream)?;
    Ok(())
}

/// What a [`ChunkCache`] did, summed over its calls
#[derive(Clone, Copy, Debug, Default)]
pub struct CacheCounts {
    /// Chunks the launches needed, counted once per launch
    pub needed: u64,
    /// Chunks read from disk and uploaded
    pub uploads: u64,
    /// Chunks the source holds none of, which read as the fill value without a slot
    pub fills: u64,
    /// Resident chunks evicted to make room
    pub evictions: u64,
    /// The most chunks one launch needed
    pub most: u64,
    /// Uploads whose chunk [`ChunkCache::prefetch`] had read ahead
    pub prefetched: u64,
}

/// The device CT cache: a pool of chunk slots, the dense chunk table the sampler reads, and the host's LRU
pub struct ChunkCache {
    vol: Arc<Volume>,
    shape: [usize; 3],
    grid: [usize; 3],
    fill: u8,
    table: Buf,
    host_table: Vec<u32>,
    pool: Buf,
    slots: usize,
    owner: Vec<u32>,
    last_use: Vec<u64>,
    free: Vec<u32>,
    epoch: u64,
    copy: Arc<CudaStream>,
    staging: Vec<Pinned<u8>>,
    entries: Pinned<u32>,
    uploaded: CudaEvent,
    sampled: CudaEvent,
    prefetch: Prefetcher,
    /// Totals since the cache was made
    pub counts: CacheCounts,
}

impl ChunkCache {
    /// A cache of `slots` chunks (2 MiB each) of `vol`, uploading through `staging` pinned chunk buffers, and as
    /// many again for chunks read ahead
    pub fn new(dev: &Device, vol: Arc<Volume>, slots: usize, staging: usize) -> Result<Self> {
        ensure!(vol.chunk_shape() == [CHUNK; 3], "the device CT cache needs 128^3 chunks, not {:?}", vol.chunk_shape());
        ensure!(slots > 0 && slots < FILL as usize && staging > 0, "cache of {slots} slots, {staging} staging");
        let shape = vol.shape();
        let grid = shape.map(|s| s.div_ceil(CHUNK));
        let cells = grid.iter().product::<usize>();
        ensure!(cells < u32::MAX as usize, "CT chunk grid {grid:?} too large for the table");
        let gpu = &dev.gpu;
        let ctx = gpu.inner.context();
        let table = gpu.alloc_fill(cells * 4, 0xFF).wrap_err("allocate the CT chunk table")?;
        let pool = gpu.alloc(slots * CHUNK_BYTES).wrap_err_with(|| format!("allocate {slots} CT chunk slots"))?;
        // the table fill runs on the compute stream, the updates on the copy stream
        gpu.sync()?;
        let prefetch = Prefetcher::new(ctx, vol.clone(), staging, READERS)?;
        let staging = (0..staging).map(|_| Pinned::new(ctx, CHUNK_BYTES)).collect::<Result<Vec<_>>>()?;
        Ok(Self {
            fill: vol.fill(),
            vol,
            shape,
            grid,
            table,
            host_table: vec![ABSENT; cells],
            pool,
            slots,
            owner: vec![u32::MAX; slots],
            last_use: vec![0; slots],
            free: (0..slots as u32).rev().collect(),
            epoch: 0,
            copy: ctx.new_stream()?,
            staging,
            entries: Pinned::new(ctx, 1 << 16)?,
            uploaded: ctx.new_event(None)?,
            sampled: ctx.new_event(None)?,
            prefetch,
            counts: CacheCounts::default(),
        })
    }

    /// The table indices of every chunk the crops `grids` of `spec` read, sorted and unique
    pub fn needed(&self, spec: &CropSpec, grids: &[CropGrid]) -> Vec<u32> {
        let mut out = Vec::new();
        for g in grids {
            g.chunks(spec, self.shape, self.grid, &mut out);
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Starts reading, on background threads, the chunks of `needed` (table indices) that are not resident, for a
    /// later [`ChunkCache::prepare`]; call it once the current launch is queued, with the next launch's chunks
    pub fn prefetch(&mut self, needed: &[u32]) -> Result<()> {
        let missing: Vec<(u32, [usize; 3])> = needed
            .iter()
            .filter(|&&idx| self.host_table[idx as usize] == ABSENT)
            .map(|&idx| (idx, self.coords(idx)))
            .collect();
        self.prefetch.start(&missing)
    }

    /// Makes the chunks `needed` (table indices) resident and the compute stream wait for their uploads
    pub fn prepare(&mut self, gpu: &Gpu, needed: &[u32]) -> Result<()> {
        self.epoch += 1;
        self.counts.needed += needed.len() as u64;
        self.counts.most = self.counts.most.max(needed.len() as u64);
        gpu.inner.context().bind_to_thread()?;
        // slots and table entries change below: wait for every crop already queued to finish reading them
        self.copy.wait(&self.sampled)?;
        // pin the resident chunks first, so no eviction below takes one this launch needs
        for &idx in needed {
            let e = self.host_table[idx as usize];
            if e < FILL {
                self.last_use[e as usize] = self.epoch;
            }
        }

        let missing: Vec<u32> = needed.iter().copied().filter(|&idx| self.host_table[idx as usize] == ABSENT).collect();
        // every chunk the disk cache lacks, before any slot or table entry changes, so the launch fails whole and
        // names them all for one fetch
        let uncached: Vec<[usize; 3]> =
            missing.iter().map(|&idx| self.coords(idx)).filter(|&c| !self.vol.cached(c)).collect();
        if !uncached.is_empty() {
            return Err(Uncached(uncached).into());
        }
        let mut updates: Vec<(u32, u32)> = Vec::new();
        for group in missing.chunks(self.staging.len()) {
            // chunks read ahead come from their prefetch buffers, the rest are read now into staging buffers `0..`;
            // either way the group is uploaded in order, so slots and evictions do not depend on what was read ahead
            let ahead: Vec<Option<(usize, bool)>> = group.iter().map(|&idx| self.prefetch.take(idx)).collect();
            let rest: Vec<u32> = group.iter().zip(&ahead).filter(|(_, a)| a.is_none()).map(|(&idx, _)| idx).collect();
            let read = if rest.is_empty() { Vec::new() } else { self.read_group(&rest)? };
            let mut read = read.into_iter().enumerate();
            for (&idx, ahead) in group.iter().zip(ahead) {
                let (present, from) = match ahead {
                    Some((buf, present)) => (present, Ok(buf)),
                    None => {
                        let (k, present) = read.next().ok_or_else(|| eyre!("chunk read count mismatch"))?;
                        (present, Err(k))
                    }
                };
                if !present {
                    self.host_table[idx as usize] = FILL;
                    updates.push((idx, FILL));
                    self.counts.fills += 1;
                    continue;
                }
                let slot = self.take_slot(&mut updates)?;
                let buf = match from {
                    Ok(buf) => {
                        self.counts.prefetched += 1;
                        self.prefetch.buf_mut(buf)
                    }
                    Err(k) => &mut self.staging[k],
                };
                upload(&self.copy, self.pool.ptr() + (slot as u64) * CHUNK_BYTES as u64, buf, CHUNK_BYTES)?;
                self.host_table[idx as usize] = slot;
                self.owner[slot as usize] = idx;
                self.last_use[slot as usize] = self.epoch;
                updates.push((idx, slot));
                self.counts.uploads += 1;
            }
        }
        self.write_entries(&mut updates)?;
        self.uploaded.record(&self.copy)?;
        gpu.inner.stream().wait(&self.uploaded)?;
        Ok(())
    }

    /// Reads the chunks `group` (table indices, at most one per staging buffer) into staging buffers `0..`, in
    /// parallel; per chunk, false where the source holds none
    fn read_group(&mut self, group: &[u32]) -> Result<Vec<bool>> {
        let coords: Vec<[usize; 3]> = group.iter().map(|&idx| self.coords(idx)).collect();
        let mut bufs = Vec::with_capacity(group.len());
        for staging in &mut self.staging[..group.len()] {
            bufs.push(staging.slice_mut()?);
        }

        let vol = &*self.vol;
        let mut present = vec![false; group.len()];
        let per = group.len().div_ceil(READERS);
        std::thread::scope(|scope| -> Result<()> {
            let readers: Vec<_> = bufs
                .chunks_mut(per)
                .zip(coords.chunks(per))
                .zip(present.chunks_mut(per))
                .map(|((bufs, coords), present)| {
                    scope.spawn(move || -> Result<()> {
                        for ((buf, &c), present) in bufs.iter_mut().zip(coords).zip(present) {
                            *present = vol.read_chunk(c, buf)?;
                        }
                        Ok(())
                    })
                })
                .collect();
            for reader in readers {
                reader.join().map_err(|_| eyre!("a chunk reader panicked"))??;
            }
            Ok(())
        })?;
        Ok(present)
    }

    /// Marks the end of the crops that read the cache, queued on the compute stream after their kernels
    pub fn sampled(&mut self, gpu: &Gpu) -> Result<()> {
        self.sampled.record(gpu.inner.stream())?;
        Ok(())
    }

    /// Chunk-grid zyx of table index `idx`
    fn coords(&self, idx: u32) -> [usize; 3] {
        let idx = idx as usize;
        [idx / (self.grid[1] * self.grid[2]), (idx / self.grid[2]) % self.grid[1], idx % self.grid[2]]
    }

    /// A free slot, or the least recently used one this launch does not need, whose chunk is then marked absent
    fn take_slot(&mut self, updates: &mut Vec<(u32, u32)>) -> Result<u32> {
        if let Some(slot) = self.free.pop() {
            return Ok(slot);
        }
        let (slot, used) = self.last_use.iter().enumerate().min_by_key(|(_, u)| **u).map(|(s, u)| (s, *u)).unwrap();
        if used >= self.epoch {
            bail!(
                "the device CT cache's {} slots cannot hold the chunks of one launch; raise the slot count",
                self.slots
            );
        }
        let old = self.owner[slot];
        self.host_table[old as usize] = ABSENT;
        updates.push((old, ABSENT));
        self.counts.evictions += 1;
        Ok(slot as u32)
    }

    /// Queues the table updates on the copy stream, one copy per run of consecutive entries; a later update of an
    /// entry replaces an earlier one
    fn write_entries(&mut self, updates: &mut [(u32, u32)]) -> Result<()> {
        if updates.is_empty() {
            return Ok(());
        }
        // stable, so the last update of an entry stays last
        updates.sort_by_key(|u| u.0);
        let mut merged: Vec<(u32, u32)> = Vec::with_capacity(updates.len());
        for &(idx, v) in updates.iter() {
            match merged.last_mut() {
                Some(last) if last.0 == idx => last.1 = v,
                _ => merged.push((idx, v)),
            }
        }
        for part in merged.chunks(self.entries.len()) {
            let buf = self.entries.slice_mut()?;
            for (slot, &(_, v)) in buf.iter_mut().zip(part) {
                *slot = v;
            }
            let mut start = 0;
            while start < part.len() {
                let mut end = start + 1;
                while end < part.len() && part[end].0 == part[end - 1].0 + 1 {
                    end += 1;
                }
                // SAFETY: the entries buffer stays untouched until `done`, recorded below, completes
                unsafe {
                    let src = std::slice::from_raw_parts(self.entries.ptr.add(start), end - start);
                    cur::memcpy_htod_async(self.table.ptr() + 4 * part[start].0 as u64, src, self.copy.cu_stream())?;
                }
                start = end;
            }
            self.entries.done.record(&self.copy)?;
        }
        Ok(())
    }
}

/// One crop request: the centre (trace xyz) and the frame (columns u, v, f) that orients it
#[derive(Clone, Copy, Debug)]
pub struct CropAt {
    pub pos: [f64; 3],
    pub frame: Mat3,
}

/// The strip model's input crop, NDHWC `(B, 144, 128, 128, 2)` float32 (CT, tube), as its readers take it
#[derive(Clone, Copy)]
pub enum StripInput<'a> {
    /// The CT channel z-scored, as the reference crops are
    ZScored(&'a Buf),
    /// The CT channel raw (`value / 255`) as [`Crops::strip`] writes it, z-scored on load with each crop's
    /// `[mean_hi, mean_lo, inv_hi, inv_lo]` in `stats`
    Raw { crop: &'a Buf, stats: &'a Buf },
}

impl<'a> StripInput<'a> {
    /// The crop buffer
    pub fn crop(self) -> &'a Buf {
        match self {
            Self::ZScored(crop) | Self::Raw { crop, .. } => crop,
        }
    }

    /// The readers' `stats` kernel argument: empty for a z-scored crop
    pub fn stats(self) -> Arg {
        match self {
            Self::ZScored(_) => Arg::EMPTY,
            Self::Raw { stats, .. } => stats.arg::<f32>(),
        }
    }
}

impl<'a> From<&'a Buf> for StripInput<'a> {
    fn from(crop: &'a Buf) -> Self {
        Self::ZScored(crop)
    }
}

/// What [`Crops::sample`] writes
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Pass {
    /// The z-scored crop, sampled twice: once for its statistics, once to write it
    ZScored,
    /// The raw crop and its statistics in one sampling pass; needs the two-channel layout
    Raw,
    /// The z-scored write's ceiling: the same launch reading no CT
    WriteCeiling,
}

/// The sampler's device buffers for up to `capacity` crops per launch
pub struct Crops {
    capacity: usize,
    scale: f64,
    grids: Buf,
    partials: Buf,
    /// The z-scored crops' statistics, used within their launch
    stats: Buf,
    /// The raw strip crops' statistics, read with the crop by the strip model until the next strip launch
    strip_stats: Buf,
    points: Buf,
    counts: Buf,
    offsets: Buf,
    noff: u32,
    host: Pinned<u32>,
}

/// villa's `tube_offsets(sigma=1)`: the integer offsets within 3.5 samples of a point, `(d, h, w)`
fn tube_offsets() -> Vec<[i32; 3]> {
    let mut out = Vec::new();
    for d in -3i32..=3 {
        for h in -3i32..=3 {
            for w in -3i32..=3 {
                if ((d * d + h * h + w * w) as f32) <= 3.5f32 * 3.5 {
                    out.push([d, h, w]);
                }
            }
        }
    }
    out
}

impl Crops {
    /// Buffers for up to `capacity` crops per launch of a CT at `scale` voxels per trace voxel (villa
    /// `input_scale`)
    pub fn new(dev: &Device, capacity: usize, scale: f64) -> Result<Self> {
        let gpu = &dev.gpu;
        let tiles = STRIP_CROP.tiles()?.iter().product::<u32>().max(FRAME_CROP.tiles()?.iter().product()) as usize;
        let offsets = tube_offsets();
        ensure!(offsets.len() == 179, "tube stencil of {} offsets, expected 179", offsets.len());
        let flat: Vec<i32> = offsets.iter().flatten().copied().collect();
        let host_words = capacity * (GRID_WORDS + 1 + 3 * TUBE_POINTS);
        Ok(Self {
            capacity,
            scale,
            grids: gpu.alloc(capacity * GRID_WORDS * 4)?,
            partials: gpu.alloc(capacity * tiles * 16)?,
            stats: gpu.alloc(capacity * 16)?,
            strip_stats: gpu.alloc(capacity * 16)?,
            points: gpu.alloc(capacity * TUBE_POINTS * 12)?,
            counts: gpu.alloc(capacity * 4)?,
            offsets: gpu.upload(&flat)?,
            noff: offsets.len() as u32,
            host: Pinned::new(gpu.inner.context(), host_words)?,
        })
    }

    /// The strip model's input for `at` into `out`: NDHWC `(B, 144, 128, 128, 2)` float32, channel 0 the raw
    /// CT (`value / 255`), channel 1 the tube of `tubes[b]` (crop-local (u, v, f) points, at most
    /// [`TUBE_POINTS`]); [`Crops::strip_input`] pairs `out` with the crops' statistics for the strip model
    pub fn strip(
        &mut self,
        dev: &Device,
        cache: &mut ChunkCache,
        at: &[CropAt],
        tubes: &[&[[f32; 3]]],
        out: &Buf,
    ) -> Result<()> {
        ensure!(tubes.len() == at.len(), "{} tubes for {} crops", tubes.len(), at.len());
        self.sample(dev, cache, &STRIP_CROP, at, Some(tubes), out, Pass::Raw)
    }

    /// `crop`, written by the last [`Crops::strip`], as the strip model's input: valid until the next one
    pub fn strip_input<'a>(&'a self, crop: &'a Buf) -> StripInput<'a> {
        StripInput::Raw { crop, stats: &self.strip_stats }
    }

    /// The frame model's crops for `at` (prior frames) into `out`: `(B, 32, 32, 32)` float32, z-scored
    pub fn frame(&mut self, dev: &Device, cache: &mut ChunkCache, at: &[CropAt], out: &Buf) -> Result<()> {
        self.sample(dev, cache, &FRAME_CROP, at, None, out, Pass::ZScored)
    }

    /// Starts reading ahead the CT chunks that crops of `spec` at `at` need and the cache lacks, for a later launch
    /// on them; see [`ChunkCache::prefetch`]
    pub fn prefetch(&self, cache: &mut ChunkCache, spec: &CropSpec, at: &[CropAt]) -> Result<()> {
        if at.is_empty() {
            return Ok(());
        }
        let grids: Vec<CropGrid> = at.iter().map(|c| CropGrid::new(spec, c.pos, &c.frame, self.scale)).collect();
        let needed = cache.needed(spec, &grids);
        cache.prefetch(&needed)
    }

    /// Stages the grids (and tube points), makes their chunks resident and launches the sampler for `pass`, writing
    /// one channel without `tubes` and two with them
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sample(
        &mut self,
        dev: &Device,
        cache: &mut ChunkCache,
        spec: &CropSpec,
        at: &[CropAt],
        tubes: Option<&[&[[f32; 3]]]>,
        out: &Buf,
        pass: Pass,
    ) -> Result<()> {
        let b = at.len();
        ensure!(b > 0 && b <= self.capacity, "{b} crops for a sampler of {}", self.capacity);
        let channels = if tubes.is_some() { 2 } else { 1 };
        ensure!(pass != Pass::Raw || channels == 2, "a raw crop needs the two-channel layout");
        let need = b * spec.samples() * channels * 4;
        ensure!(out.bytes() >= need, "crop output of {} bytes, {need} needed", out.bytes());
        let grids: Vec<CropGrid> = at.iter().map(|c| CropGrid::new(spec, c.pos, &c.frame, self.scale)).collect();
        let needed = cache.needed(spec, &grids);
        cache.prepare(&dev.gpu, &needed)?;

        let gpu = &dev.gpu;
        let stream = gpu.inner.stream().clone();
        let host = self.host.slice_mut()?;
        for (k, g) in grids.iter().enumerate() {
            host[k * GRID_WORDS..(k + 1) * GRID_WORDS].copy_from_slice(&g.record());
        }
        let mut used = b * GRID_WORDS;
        if let Some(tubes) = tubes {
            for (k, t) in tubes.iter().enumerate() {
                ensure!(t.len() <= TUBE_POINTS, "decision {k}: {} tube points, at most {TUBE_POINTS}", t.len());
                host[used + k] = t.len() as u32;
            }
            used += b;
            for (k, t) in tubes.iter().enumerate() {
                let dst = &mut host[used + k * 3 * TUBE_POINTS..];
                for (i, p) in t.iter().enumerate() {
                    dst[3 * i..3 * i + 3].copy_from_slice(&p.map(f32::to_bits));
                }
            }
        }
        // three copies from one staging buffer: grids, counts, points
        let host_ptr = self.host.ptr;
        // SAFETY: the staging buffer is idle (slice_mut waited) and stays untouched until `done` completes
        unsafe {
            cur::memcpy_htod_async(
                self.grids.ptr(),
                std::slice::from_raw_parts(host_ptr, b * GRID_WORDS),
                stream.cu_stream(),
            )?;
            if tubes.is_some() {
                let counts = std::slice::from_raw_parts(host_ptr.add(b * GRID_WORDS), b);
                cur::memcpy_htod_async(self.counts.ptr(), counts, stream.cu_stream())?;
                let points = std::slice::from_raw_parts(host_ptr.add(b * GRID_WORDS + b), b * 3 * TUBE_POINTS);
                cur::memcpy_htod_async(self.points.ptr(), points, stream.cu_stream())?;
            }
        }
        self.host.done.record(&stream)?;

        let [tw, th, td] = spec.tiles()?;
        let tiles = tw * th * td;
        let ct = [
            Arg::Ptr(cache.table.ptr()),
            Arg::Ptr(cache.pool.ptr()),
            Arg::U32(cache.shape[0] as u32),
            Arg::U32(cache.shape[1] as u32),
            Arg::U32(cache.shape[2] as u32),
            Arg::U32(cache.grid[1] as u32),
            Arg::U32(cache.grid[2] as u32),
            Arg::F32(cache.fill as f32),
            Arg::Ptr(self.grids.ptr()),
            Arg::U32(tw),
            Arg::U32(th),
            Arg::U32(STEPS),
        ];
        let k = &dev.kernels;
        let block = (SAMPLE_THREADS, 1, 1);
        let dims = [spec.depth, spec.height, spec.width].map(|d| Arg::U32(d as u32));
        let mut args = ct.to_vec();
        args.push(Arg::Ptr(self.partials.ptr()));
        if pass == Pass::Raw {
            args.extend(dims);
            args.push(Arg::Ptr(out.ptr()));
            k.launch(gpu, "sc_raw", (tiles, b as u32, 1), block, 0, &args)?;
        } else {
            k.launch(gpu, "sc_stats", (tiles, b as u32, 1), block, 0, &args)?;
        }
        let stats = if pass == Pass::Raw { &self.strip_stats } else { &self.stats };
        k.launch(
            gpu,
            "sc_finalize",
            (b as u32, 1, 1),
            (256, 1, 1),
            0,
            &[Arg::Ptr(self.partials.ptr()), Arg::U32(tiles), Arg::U32(spec.samples() as u32), Arg::Ptr(stats.ptr())],
        )?;
        if pass != Pass::Raw {
            let mut args = ct.to_vec();
            args.push(Arg::Ptr(stats.ptr()));
            args.extend(dims);
            args.extend([
                Arg::U32(channels as u32),
                Arg::U32(u32::from(pass == Pass::WriteCeiling)),
                Arg::Ptr(out.ptr()),
            ]);
            k.launch(gpu, "sc_write", (tiles, b as u32, 1), block, 0, &args)?;
        }
        if tubes.is_some() {
            let n = TUBE_POINTS as u32 * self.noff;
            k.launch(
                gpu,
                "sc_tube",
                (n.div_ceil(256), b as u32, 1),
                (256, 1, 1),
                0,
                &[
                    Arg::Ptr(self.points.ptr()),
                    Arg::Ptr(self.counts.ptr()),
                    Arg::U32(TUBE_POINTS as u32),
                    Arg::Ptr(self.offsets.ptr()),
                    Arg::U32(self.noff),
                    Arg::U32(spec.depth as u32),
                    Arg::U32(spec.height as u32),
                    Arg::U32(spec.width as u32),
                    Arg::F32(spec.spacing as f32),
                    Arg::F32(spec.behind as f32),
                    Arg::F32(((spec.height - 1) as f64 / 2.0) as f32),
                    Arg::F32(((spec.width - 1) as f64 / 2.0) as f32),
                    // villa's 2·sigma² with tube_sigma 1
                    Arg::F32(2.0),
                    Arg::Ptr(out.ptr()),
                ],
            )?;
        }
        cache.sampled(gpu)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl ChunkCache {
        /// Device bytes the pool and the table hold
        pub fn device_bytes(&self) -> usize {
            self.pool.bytes() + self.table.bytes()
        }
    }

    /// villa's `crop_grids` + vc_reader's `localOrigin` on a rotated frame: the record and the base split
    #[test]
    fn grid_matches_villa_layout() {
        let (c, s) = (0.6f64, 0.8f64);
        // columns u, v, f
        let frame = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
        let pos = [3000.25, 2000.5, 4000.75];
        let g = CropGrid::new(&STRIP_CROP, pos, &frame, 2.0);
        // first sample, xyz: pos*2 + F*2 @ (-31.75, -31.75, -36)
        let x = pos[0] * 2.0 + (2.0 * c * -31.75 + 2.0 * -s * -31.75);
        let y = pos[1] * 2.0 + (2.0 * s * -31.75 + 2.0 * c * -31.75);
        let z = pos[2] * 2.0 + 2.0 * -36.0;
        for (a, o) in [z, y, x].into_iter().enumerate() {
            assert_eq!(g.base[a], o.floor() as i64);
            assert_eq!(g.origin[a], (o - o.floor()) as f32);
        }
        // rows zyx, columns d (f), h (v), w (u), each frame*2*0.5
        assert_eq!(g.axes[0], [1.0, 0.0, 0.0]);
        assert_eq!(g.axes[1], [0.0, c as f32, s as f32]);
        assert_eq!(g.axes[2], [0.0, -s as f32, c as f32]);
    }

    /// The chunk set covers every voxel any sample's trilinear cell touches, checked by brute force over a
    /// small crop with an oblique frame
    #[test]
    fn chunks_cover_every_read() {
        let spec = CropSpec { depth: 48, height: 40, width: 40, behind: 24.0, spacing: 2.0 };
        let (a, b) = (0.3f64.cos(), 0.3f64.sin());
        let f = [[a, 0.0, -b], [0.0, 1.0, 0.0], [b, 0.0, a]];
        let (r, q) = (0.7f64.cos(), 0.7f64.sin());
        let g2 = [[r, -q, 0.0], [q, r, 0.0], [0.0, 0.0, 1.0]];
        let mut frame = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                frame[i][j] = (0..3).map(|k| f[i][k] * g2[k][j]).sum();
            }
        }
        let shape = [1000, 900, 800];
        let grid = shape.map(|s: usize| s.div_ceil(CHUNK));
        let g = CropGrid::new(&spec, [200.0, 190.0, 260.0], &frame, 2.0);
        let mut got = Vec::new();
        g.chunks(&spec, shape, grid, &mut got);
        got.sort_unstable();
        let mut want = Vec::new();
        for d in 0..spec.depth {
            for h in 0..spec.height {
                for w in 0..spec.width {
                    let c: [f32; 3] = std::array::from_fn(|k| {
                        g.origin[k] + g.axes[k][0] * d as f32 + g.axes[k][1] * h as f32 + g.axes[k][2] * w as f32
                    });
                    for corner in 0..8 {
                        let v: [i64; 3] =
                            std::array::from_fn(|k| g.base[k] + c[k].floor() as i64 + ((corner >> k) & 1));
                        if (0..3).all(|k| v[k] >= 0 && (v[k] as usize) < shape[k]) {
                            let k = v.map(|x| x as usize / CHUNK);
                            want.push(((k[0] * grid[1] + k[1]) * grid[2] + k[2]) as u32);
                        }
                    }
                }
            }
        }
        want.sort_unstable();
        want.dedup();
        for w in &want {
            assert!(got.binary_search(w).is_ok(), "chunk {w} read but not in the set {got:?}");
        }
        // a superset, but not the whole bounding box
        assert!(got.len() < want.len() * 3, "{} chunks for {} read", got.len(), want.len());
    }

    #[test]
    fn tube_stencil_has_179_offsets() {
        assert_eq!(tube_offsets().len(), 179);
    }
}
