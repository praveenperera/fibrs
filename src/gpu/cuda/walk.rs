//! The fibre walk on the GPU: a family's fields resident on the device, and chunks of seeds traced there ahead of
//! the CPU's commit point against a mirror of the coverage (`fibrs-kernels/src/walk_dev.rs`)
//!
//! [`Gpu::walk`] uploads the family's presence (combined from both pooled predictions, as `Presence` reads it),
//! centrality and packed direction bytes as brick pools with a table over the full brick grid, and the
//! catalogue's seeds. [`Device::update`] applies the CPU's commits to the device's coverage on a stream of its own;
//! [`Device::launch`] queues one chunk of seeds on a free chunk slot and [`Device::collect`] waits for the oldest and
//! returns its results ([`ChunkOut`]). The kernels read the coverage while updates change it, and every result
//! records the values it read, which the host checks against its own coverage before committing it
//!
//! Kernels take raw device pointers, so cudarc's events order none of their launches: the uploads are
//! synchronized before the walk is handed out, each chunk waits on an event recorded after the last update so it
//! reads every change sent before it, and each slot's buffers are reused only after its stream is synchronized

use std::collections::VecDeque;
use std::ops::Range;
use std::sync::Arc;

use color_eyre::eyre::{Result, WrapErr, bail, ensure, eyre};
use cudarc::driver::{
    CudaEvent, CudaSlice, CudaStream, DevicePtr, DeviceRepr, LaunchConfig, PinnedHostSlice, PushKernelArg,
    ValidAsZeroBits, sys,
};
use rayon::prelude::*;

use super::{Gpu, NO_BRICK, fit_host, pinned};
use crate::fields::{Catalogue, Presence};
use crate::gpu::walk::{CHUNK, Caps, IN_FLIGHT, STEP_CAP};
use crate::kernels::{Change, DirField, decode_packed};
use crate::lookahead::{ChunkOut, Device, RawCheck, SeedOut, Status};
use crate::vol::{BRICK, Vol};

// the host reads only part of the kernels' layout
#[path = "../../../fibrs-kernels/src/walk_layout.rs"]
#[allow(dead_code)]
mod layout;

use layout::{
    CHECK_WORDS, CU_WORDS, MAX_CAP, PREFIX_CHECKS, RESULT_WORDS, SPAN_CAP, SURV_WORDS, T_CHECKS, T_CURVES, T_PTS,
    T_SEEDS, TOTALS,
};

/// Coverage bricks per device allocation of the mirror: 32 MiB
const SEGMENT_BRICKS: usize = 4096;
/// Pinned staging for the volume uploads
const STAGING_BYTES: usize = 64 << 20;
/// Device memory left free beyond the walk's own, for the driver and other stages
const HEADROOM: usize = 1 << 30;

/// Device bytes of one chunk slot with room `caps`
fn slot_bytes(caps: &Caps) -> usize {
    let width = (2 * STEP_CAP + 1) as usize;
    let surv = caps.survivors as usize;
    let per_seed = 4 * 4 + PREFIX_CHECKS as usize * CHECK_WORDS * 8 + RESULT_WORDS * 4;
    let per_surv = 4
        + CU_WORDS * 8
        + SURV_WORDS * 4
        + 2 * width * 12
        + 2 * STEP_CAP as usize * CHECK_WORDS * 8
        + width * 8
        + SPAN_CAP as usize * 4;
    CHUNK * per_seed
        + surv * per_surv
        + caps.checks as usize * CHECK_WORDS * 8
        + caps.pts as usize * 2 * 12
        + caps.curves as usize * 8
}

/// A device pointer, as the kernels take it
fn raw<T>(buf: &CudaSlice<T>, stream: &CudaStream) -> u64 {
    // cudarc records this read before the launch that makes it, so the event orders nothing; the walk orders its
    // own launches (see the module docs)
    buf.device_ptr(stream).0
}

/// One resident volume: its brick table over the full grid and the pool of payloads
struct Pool<T> {
    table: CudaSlice<u32>,
    pool: CudaSlice<T>,
}

/// Upload the bricks `present` lists (volume brick indices) as a pool, each filled by `fill(brick, payload)`, and
/// a table over all `n_bricks` bricks; synchronous
fn upload_pool<T: DeviceRepr + ValidAsZeroBits + Copy + Default + Send + Sync>(
    stream: &Arc<CudaStream>,
    n_bricks: usize,
    present: &[u32],
    fill: impl Fn(usize, &mut [T]) + Sync,
) -> Result<Pool<T>> {
    let mut table = vec![NO_BRICK; n_bricks];
    for (k, &b) in present.iter().enumerate() {
        table[b as usize] = k as u32;
    }
    let table = stream.clone_htod(&table).wrap_err("upload a brick table")?;
    let mut pool = stream.alloc_zeros::<T>(present.len().max(1) * BRICK).wrap_err("allocate a brick pool")?;
    let per = (STAGING_BYTES / (BRICK * size_of::<T>())).max(1);
    let mut staging = pinned::<T>(stream.context(), per.min(present.len()).max(1) * BRICK, 0)?;
    for (i, batch) in present.chunks(per).enumerate() {
        // the previous batch's copy has finished reading the staging buffer
        stream.synchronize()?;
        let host = &mut staging.as_mut_slice()?[..batch.len() * BRICK];
        host.par_chunks_mut(BRICK).zip(batch.par_iter()).for_each(|(dst, &b)| fill(b as usize, dst));
        let at = i * per * BRICK;
        stream.memcpy_htod(
            &staging.as_slice()?[..batch.len() * BRICK],
            &mut pool.slice_mut(at..at + batch.len() * BRICK),
        )?;
    }
    stream.synchronize().wrap_err("upload a brick pool")?;
    Ok(Pool { table, pool })
}

/// The bricks of a grid of `n` for which `keep` holds, in order
fn bricks_where(n: usize, keep: impl Fn(usize) -> bool + Sync) -> Vec<u32> {
    (0..n).into_par_iter().filter(|&b| keep(b)).map(|b| b as u32).collect()
}

/// The family's fields on the device, which every chunk reads
struct Resident {
    dims: [u32; 3],
    gy: u32,
    gx: u32,
    pres: Pool<u8>,
    cent: Pool<u16>,
    dirs: Pool<u16>,
    decode: CudaSlice<f64>,
    ring: CudaSlice<f64>,
    /// z, y, x and a primary flag per catalogue seed
    seeds: CudaSlice<i32>,
    n_seeds: usize,
}

/// The device's coverage: a table of brick payload addresses (0 while a brick reads as 0) and the payloads, in
/// segments allocated as commits mark new bricks
struct Mirror {
    stream: Arc<CudaStream>,
    table: CudaSlice<u64>,
    /// The table as the host last wrote it
    host: Vec<u64>,
    segments: Vec<CudaSlice<u16>>,
    /// Bricks handed out of the newest segment
    used: usize,
    new_host: PinnedHostSlice<u64>,
    set_host: PinnedHostSlice<u32>,
    new_dev: CudaSlice<u64>,
    set_dev: CudaSlice<u32>,
    /// Recorded after each update, so a chunk launched later reads every change sent before it
    applied: CudaEvent,
}

/// One chunk slot: a stream and the device and pinned host buffers one chunk uses
struct ChunkSlot {
    stream: Arc<CudaStream>,
    state: CudaSlice<u32>,
    pre: CudaSlice<u64>,
    n_checks: CudaSlice<u32>,
    n_pts: CudaSlice<u32>,
    n_curves: CudaSlice<u32>,
    totals: CudaSlice<u32>,
    surv_seed: CudaSlice<u32>,
    surv_cu: CudaSlice<f64>,
    surv_len: CudaSlice<u32>,
    path_pts: CudaSlice<f32>,
    path_tans: CudaSlice<f32>,
    surv_checks: CudaSlice<u64>,
    stack: CudaSlice<u32>,
    cum: CudaSlice<f32>,
    spans: CudaSlice<u32>,
    out_checks: CudaSlice<u64>,
    out_pts: CudaSlice<f32>,
    out_tans: CudaSlice<f32>,
    out_curves: CudaSlice<u32>,
    results: CudaSlice<u32>,
    host_totals: PinnedHostSlice<u32>,
    host_results: PinnedHostSlice<u32>,
    host_checks: PinnedHostSlice<u64>,
    host_pts: PinnedHostSlice<f32>,
    host_tans: PinnedHostSlice<f32>,
    host_curves: PinnedHostSlice<u32>,
}

impl ChunkSlot {
    fn new(gpu: &Gpu, caps: &Caps) -> Result<Self> {
        let stream = gpu.ctx.new_stream()?;
        let width = (2 * STEP_CAP + 1) as usize;
        let surv = caps.survivors as usize;
        let (checks, pts, curves) = (caps.checks as usize, caps.pts as usize, caps.curves as usize);
        let z = |n: usize| -> Result<CudaSlice<u32>> { Ok(stream.alloc_zeros::<u32>(n.max(1))?) };
        Ok(Self {
            state: z(CHUNK)?,
            pre: stream.alloc_zeros(CHUNK * PREFIX_CHECKS as usize * CHECK_WORDS)?,
            n_checks: z(CHUNK)?,
            n_pts: z(CHUNK)?,
            n_curves: z(CHUNK)?,
            totals: z(TOTALS)?,
            surv_seed: z(surv)?,
            surv_cu: stream.alloc_zeros(surv.max(1) * CU_WORDS)?,
            surv_len: z(surv * SURV_WORDS)?,
            path_pts: stream.alloc_zeros(surv.max(1) * width * 3)?,
            path_tans: stream.alloc_zeros(surv.max(1) * width * 3)?,
            surv_checks: stream.alloc_zeros(surv.max(1) * 2 * STEP_CAP as usize * CHECK_WORDS)?,
            stack: z(surv * width)?,
            cum: stream.alloc_zeros(surv.max(1) * width)?,
            spans: z(surv * SPAN_CAP as usize)?,
            out_checks: stream.alloc_zeros(checks.max(1) * CHECK_WORDS)?,
            out_pts: stream.alloc_zeros(pts.max(1) * 3)?,
            out_tans: stream.alloc_zeros(pts.max(1) * 3)?,
            out_curves: z(curves * 2)?,
            results: z(CHUNK * RESULT_WORDS)?,
            host_totals: pinned(&gpu.ctx, TOTALS, 0)?,
            host_results: pinned(&gpu.ctx, CHUNK * RESULT_WORDS, 0)?,
            host_checks: pinned(&gpu.ctx, 1, 0)?,
            host_pts: pinned(&gpu.ctx, 1, 0)?,
            host_tans: pinned(&gpu.ctx, 1, 0)?,
            host_curves: pinned(&gpu.ctx, 1, 0)?,
            stream,
        })
    }
}

impl Drop for ChunkSlot {
    /// Wait for queued work before the buffers it reads and writes are freed
    fn drop(&mut self) {
        // a failed sync means the context is broken, and freeing then is all that is left to do
        let _ = self.stream.synchronize();
    }
}

/// A family's walk on the device, from [`Gpu::walk`]
pub struct Walk<'g> {
    gpu: &'g Gpu,
    caps: Caps,
    fields: Resident,
    cov: Mirror,
    slots: Vec<ChunkSlot>,
    /// Slots free to launch on
    free: Vec<usize>,
    /// Launched chunks, oldest first
    flying: VecDeque<(usize, Range<usize>)>,
}

impl Gpu {
    /// Put a family's fields on the device for a walk over the seeds of `cat`: `tracing` the pooled predictions
    /// (presence is combined from them as `Presence` does), `cent` the family's centrality, `dirs` the direction
    /// field in either form, `ring` the tracer's ring exponentials (`kernels::ring`)
    ///
    /// Fails, leaving the walk to the CPU, when the device lacks the memory for the fields and the chunk slots
    pub fn walk<D: DirField>(
        &self,
        tracing: &[Vol<u8>; 2],
        family: usize,
        cent: &Vol<u16>,
        dirs: &D,
        cat: &Catalogue,
        ring: &[f64; 13],
    ) -> Result<Walk<'_>> {
        self.walk_with(tracing, family, cent, dirs, cat, ring, Caps::default())
    }

    /// [`Gpu::walk`] with chunk slots of `caps`
    #[allow(clippy::too_many_arguments)]
    pub fn walk_with<D: DirField>(
        &self,
        tracing: &[Vol<u8>; 2],
        family: usize,
        cent: &Vol<u16>,
        dirs: &D,
        cat: &Catalogue,
        ring: &[f64; 13],
        caps: Caps,
    ) -> Result<Walk<'_>> {
        ensure!(family == 1 || family == 2, "family {family} is neither V (1) nor H (2)");
        let shape = cent.shape();
        ensure!(
            tracing.iter().all(|v| v.shape() == shape) && dirs.shape() == shape,
            "the walk's fields differ in shape"
        );
        ensure!(
            tracing.iter().all(|v| v.default_value() == 0) && cent.default_value() == 0,
            "unexpected field defaults"
        );
        ensure!(
            shape.iter().all(|&n| n < i32::MAX as usize / 2),
            "volume of {shape:?} voxels is too large for the walk"
        );
        ensure!(STEP_CAP <= MAX_CAP, "step cap {STEP_CAP} above the kernels' {MAX_CAP}");
        let grid = cent.grid();
        let n_bricks = grid.iter().product::<usize>();
        ensure!(n_bricks < NO_BRICK as usize, "a brick grid of {grid:?} is too large for the walk's tables");
        let n_seeds = cat.coords.len();
        ensure!(n_seeds < u32::MAX as usize, "{n_seeds} seeds is more than one walk can trace");

        let (own, other) = (&tracing[family - 1], &tracing[2 - family]);
        let pres_bricks = bricks_where(n_bricks, |b| own.brick_at(b).is_some());
        let cent_bricks = bricks_where(n_bricks, |b| cent.brick_at(b).is_some());
        let dir_bricks = bricks_where(n_bricks, |b| dirs.has_brick(b));

        // everything the walk holds, checked before the first allocation so a walk that cannot fit fails at once
        let fields_bytes = 3 * 4 * n_bricks
            + pres_bricks.len() * BRICK
            + 2 * (cent_bricks.len() + dir_bricks.len()) * BRICK
            + (1 << 16) * 4 * 8
            + n_seeds * 16;
        let cov_bytes = 8 * n_bricks + SEGMENT_BRICKS * BRICK * 2;
        let need = fields_bytes + cov_bytes + IN_FLIGHT * slot_bytes(&caps) + HEADROOM;
        let (free, _) = self.ctx.mem_get_info()?;
        if need > free {
            let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
            bail!("the GPU walk needs {:.1} GiB of device memory, {:.1} GiB is free", gib(need), gib(free));
        }

        let stream = self.ctx.new_stream()?;
        let pres = upload_pool::<u8>(&stream, n_bricks, &pres_bricks, |b, dst| {
            let own = own.brick_at(b).expect("listed bricks of the own prediction are allocated");
            match other.brick_at(b) {
                Some(oth) => {
                    dst.iter_mut().zip(own.iter().zip(oth)).for_each(|(d, (&a, &o))| *d = Presence::combine(a, o))
                }
                None => dst.iter_mut().zip(own).for_each(|(d, &a)| *d = Presence::combine(a, 0)),
            }
        })?;
        let cent_pool = upload_pool::<u16>(&stream, n_bricks, &cent_bricks, |b, dst| {
            dst.copy_from_slice(cent.brick_at(b).expect("listed centrality bricks are allocated"));
        })?;
        let dirs = upload_pool::<u16>(&stream, n_bricks, &dir_bricks, |b, dst| {
            dirs.packed_brick(b, dst.try_into().expect("brick-sized chunks"));
        })?;
        let table: Vec<f64> = (0..=u16::MAX)
            .flat_map(|d| {
                let [a, b, c] = decode_packed(d);
                [a, b, c, 0.0]
            })
            .collect();
        let decode = stream.clone_htod(&table)?;
        let ring = stream.clone_htod(&ring[..])?;
        let seeds_host = seed_table(cat);
        let seeds = stream.clone_htod(&seeds_host)?;
        let cov_table = stream.alloc_zeros::<u64>(n_bricks)?;
        stream.synchronize().wrap_err("upload the walk's fields")?;
        let fields = Resident {
            dims: shape.map(|n| n as u32),
            gy: grid[1] as u32,
            gx: grid[2] as u32,
            pres,
            cent: cent_pool,
            dirs,
            decode,
            ring,
            seeds,
            n_seeds,
        };

        let cov_stream = self.ctx.new_stream()?;
        let cov = Mirror {
            table: cov_table,
            host: vec![0; n_bricks],
            segments: Vec::new(),
            used: SEGMENT_BRICKS,
            new_host: pinned(&self.ctx, 1, 0)?,
            set_host: pinned(&self.ctx, 1, 0)?,
            new_dev: cov_stream.alloc_zeros(1)?,
            set_dev: cov_stream.alloc_zeros(1)?,
            // never recorded until the first update, which a wait on it then passes at once
            applied: self.ctx.new_event(Some(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING))?,
            stream: cov_stream,
        };
        let slots = (0..IN_FLIGHT).map(|_| ChunkSlot::new(self, &caps)).collect::<Result<Vec<_>>>()?;
        Ok(Walk { gpu: self, caps, fields, cov, slots, free: (0..IN_FLIGHT).rev().collect(), flying: VecDeque::new() })
    }
}

/// z, y, x and a primary flag per seed of the catalogue, in its order
fn seed_table(cat: &Catalogue) -> Vec<i32> {
    let mut out = Vec::with_capacity(cat.coords.len() * 4);
    for (r, run) in cat.starts.windows(2).enumerate() {
        let primary = cat.primary[r];
        for (k, &[z, y, x]) in cat.coords[run[0]..run[1]].iter().enumerate() {
            out.extend([z, y, x, (k < primary) as i32]);
        }
    }
    out
}

/// A launch of `n` threads in blocks of `block`
fn grid(n: usize, block: u32) -> Result<LaunchConfig> {
    let blocks =
        u32::try_from(n.div_ceil(block as usize).max(1)).map_err(|_| eyre!("launch of {n} threads is too large"))?;
    Ok(LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 })
}

impl Walk<'_> {
    /// Device bytes the walk holds now, its coverage mirror included
    pub fn device_bytes(&self) -> usize {
        let f = &self.fields;
        let pools = f.pres.pool.len() + 2 * (f.cent.pool.len() + f.dirs.pool.len());
        let tables = 4 * (f.pres.table.len() + f.cent.table.len() + f.dirs.table.len()) + 8 * self.cov.table.len();
        pools + tables + self.cov.segments.len() * SEGMENT_BRICKS * BRICK * 2 + IN_FLIGHT * slot_bytes(&self.caps)
    }

    /// [`Device::update`]
    fn apply(&mut self, changes: &[Change]) -> Result<()> {
        if changes.is_empty() {
            return Ok(());
        }
        let m = &mut self.cov;
        // the previous batch's copies have read the staging buffers
        m.stream.synchronize().wrap_err("update the GPU coverage")?;
        let mut new = Vec::new();
        for c in changes {
            let b = c.at.slot();
            ensure!(b < m.host.len(), "coverage change outside the volume: {c:?}");
            if m.host[b] != 0 {
                continue;
            }
            if m.used == SEGMENT_BRICKS {
                // zeroed on the update stream, before the table points at it
                let seg = m.stream.alloc_zeros::<u16>(SEGMENT_BRICKS * BRICK).wrap_err("grow the GPU coverage")?;
                m.segments.push(seg);
                m.used = 0;
            }
            let seg = m.segments.last().expect("a segment with room");
            let addr = raw(seg, &m.stream) + (m.used * BRICK * 2) as u64;
            m.used += 1;
            m.host[b] = addr;
            new.extend([b as u64, addr]);
        }
        let table = raw(&m.table, &m.stream);
        if !new.is_empty() {
            let n = new.len() / 2;
            fit_host(&m.stream, &mut m.new_host, new.len(), 0)?;
            if m.new_dev.len() < new.len() {
                m.new_dev = m.stream.alloc_zeros(new.len() + new.len() / 2)?;
            }
            m.new_host.as_mut_slice()?[..new.len()].copy_from_slice(&new);
            m.stream.memcpy_htod(&m.new_host.as_slice()?[..new.len()], &mut m.new_dev.slice_mut(..new.len()))?;
            let list = raw(&m.new_dev, &m.stream);
            let n32 = n as u32;
            // SAFETY: params in the PTX's order (checked when the device opened); the table holds every listed
            // brick and the list `n` pairs
            unsafe {
                m.stream
                    .launch_builder(&self.gpu.k.walk.cov_bricks)
                    .arg(&table)
                    .arg(&list)
                    .arg(&n32)
                    .launch(grid(n, 256)?)?;
            }
        }
        let words = 2 * changes.len();
        fit_host(&m.stream, &mut m.set_host, words, 0)?;
        if m.set_dev.len() < words {
            m.set_dev = m.stream.alloc_zeros(words + words / 2)?;
        }
        let host = &mut m.set_host.as_mut_slice()?[..words];
        for (pair, c) in host.as_chunks_mut::<2>().0.iter_mut().zip(changes) {
            pair[0] = c.at.slot() as u32;
            pair[1] = c.at.offset() as u32 | (c.value as u32) << 16;
        }
        m.stream.memcpy_htod(&m.set_host.as_slice()?[..words], &mut m.set_dev.slice_mut(..words))?;
        let list = raw(&m.set_dev, &m.stream);
        let n32 = changes.len() as u32;
        // SAFETY: as above; every listed brick now has a payload, written earlier on this stream
        unsafe {
            m.stream
                .launch_builder(&self.gpu.k.walk.cov_set)
                .arg(&table)
                .arg(&list)
                .arg(&n32)
                .launch(grid(changes.len(), 256)?)?;
        }
        m.applied.record(&m.stream)?;
        Ok(())
    }

    /// [`Device::launch`]
    fn start(&mut self, seeds: Range<usize>) -> Result<()> {
        let n = seeds.len();
        ensure!(n > 0 && n <= CHUNK, "a chunk of {n} seeds; chunks hold 1 to {CHUNK}");
        ensure!(seeds.end <= self.fields.n_seeds, "seeds {seeds:?} beyond the catalogue's {}", self.fields.n_seeds);
        let Some(s) = self.free.pop() else { bail!("{IN_FLIGHT} chunks are in flight; collect one first") };
        if let Err(e) = self.queue(s, &seeds) {
            self.free.push(s);
            return Err(e);
        }
        self.flying.push_back((s, seeds));
        Ok(())
    }

    fn queue(&mut self, s: usize, seeds: &Range<usize>) -> Result<()> {
        let (f, k, c) = (&self.fields, &self.gpu.k, &self.caps);
        let slot = &mut self.slots[s];
        let st = slot.stream.clone();
        let n = seeds.len();
        // the updates sent so far land before this chunk reads the coverage; chunks already running may still
        // read either value of a voxel they change
        st.wait(&self.cov.applied)?;
        st.memset_zeros(&mut slot.totals)?;
        st.memset_zeros(&mut slot.n_pts.slice_mut(..n))?;
        st.memset_zeros(&mut slot.n_curves.slice_mut(..n))?;
        let p = |b: &CudaSlice<u32>| raw(b, &st);
        let [d, h, w] = f.dims;
        let (gy, gx) = (f.gy, f.gx);
        let (pres_t, pres_p) = (raw(&f.pres.table, &st), raw(&f.pres.pool, &st));
        let (cent_t, cent_p) = (raw(&f.cent.table, &st), raw(&f.cent.pool, &st));
        let (dir_t, dir_p) = (raw(&f.dirs.table, &st), raw(&f.dirs.pool, &st));
        let (decode, ring, seed_tab) = (raw(&f.decode, &st), raw(&f.ring, &st), raw(&f.seeds, &st));
        let cov = raw(&self.cov.table, &st);
        let (first, n32, cap) = (seeds.start as u32, n as u32, STEP_CAP);
        let (state, pre, n_checks, n_pts, n_curves) =
            (p(&slot.state), raw(&slot.pre, &st), p(&slot.n_checks), p(&slot.n_pts), p(&slot.n_curves));
        let totals = p(&slot.totals);
        let at = |i: usize| totals + (i * 4) as u64;
        let (surv_seed, surv_cu, surv_len) = (p(&slot.surv_seed), raw(&slot.surv_cu, &st), p(&slot.surv_len));
        let (path_pts, path_tans, surv_checks) =
            (raw(&slot.path_pts, &st), raw(&slot.path_tans, &st), raw(&slot.surv_checks, &st));
        let (stack, cum, spans) = (p(&slot.stack), raw(&slot.cum, &st), p(&slot.spans));
        let (out_checks, out_pts, out_tans) =
            (raw(&slot.out_checks, &st), raw(&slot.out_pts, &st), raw(&slot.out_tans, &st));
        let (out_curves, results) = (p(&slot.out_curves), p(&slot.results));
        let one_block = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
        let surv = c.survivors as usize;
        // SAFETY: every launch passes its kernel's params in the PTX's order (checked when the device opened); each
        // buffer holds what the kernels index, sized from `caps`, `CHUNK` and `STEP_CAP` as `walk_layout.rs` lays
        // them out, and the kernels bound every write by the caps and counts passed; all run in order on this
        // slot's stream, whose previous chunk was collected (synchronized) before the slot was free again
        unsafe {
            st.launch_builder(&k.walk.seeds)
                .arg(&d)
                .arg(&h)
                .arg(&w)
                .arg(&gy)
                .arg(&gx)
                .arg(&pres_t)
                .arg(&pres_p)
                .arg(&cent_t)
                .arg(&cent_p)
                .arg(&dir_t)
                .arg(&dir_p)
                .arg(&decode)
                .arg(&ring)
                .arg(&cov)
                .arg(&seed_tab)
                .arg(&first)
                .arg(&n32)
                .arg(&c.survivors)
                .arg(&state)
                .arg(&pre)
                .arg(&n_checks)
                .arg(&totals)
                .arg(&surv_seed)
                .arg(&surv_cu)
                .launch(grid(n, 128)?)?;
            st.launch_builder(&k.walk.trace)
                .arg(&d)
                .arg(&h)
                .arg(&w)
                .arg(&gy)
                .arg(&gx)
                .arg(&pres_t)
                .arg(&pres_p)
                .arg(&cent_t)
                .arg(&cent_p)
                .arg(&dir_t)
                .arg(&dir_p)
                .arg(&decode)
                .arg(&ring)
                .arg(&cov)
                .arg(&cap)
                .arg(&c.survivors)
                .arg(&totals)
                .arg(&surv_cu)
                .arg(&surv_len)
                .arg(&path_pts)
                .arg(&path_tans)
                .arg(&surv_checks)
                .launch(grid(surv, 32)?)?;
            st.launch_builder(&k.walk.spans)
                .arg(&cap)
                .arg(&c.survivors)
                .arg(&totals)
                .arg(&seed_tab)
                .arg(&first)
                .arg(&surv_seed)
                .arg(&surv_len)
                .arg(&path_pts)
                .arg(&stack)
                .arg(&cum)
                .arg(&spans)
                .arg(&n_checks)
                .arg(&n_pts)
                .arg(&n_curves)
                .launch(grid(surv, 32)?)?;
            for (data, total) in [(n_checks, T_CHECKS), (n_pts, T_PTS), (n_curves, T_CURVES)] {
                st.launch_builder(&k.walk.scan)
                    .arg(&data)
                    .arg(&at(T_SEEDS))
                    .arg(&n32)
                    .arg(&at(total))
                    .launch(one_block)?;
            }
            st.launch_builder(&k.walk.gather)
                .arg(&cap)
                .arg(&n32)
                .arg(&state)
                .arg(&pre)
                .arg(&n_checks)
                .arg(&n_pts)
                .arg(&n_curves)
                .arg(&surv_len)
                .arg(&surv_checks)
                .arg(&path_pts)
                .arg(&path_tans)
                .arg(&spans)
                .arg(&c.checks)
                .arg(&c.pts)
                .arg(&c.curves)
                .arg(&out_checks)
                .arg(&out_pts)
                .arg(&out_tans)
                .arg(&out_curves)
                .launch(grid(n, 128)?)?;
            st.launch_builder(&k.walk.results)
                .arg(&n32)
                .arg(&state)
                .arg(&surv_len)
                .arg(&n_checks)
                .arg(&n_pts)
                .arg(&n_curves)
                .arg(&totals)
                .arg(&c.checks)
                .arg(&c.pts)
                .arg(&c.curves)
                .arg(&results)
                .launch(grid(n, 256)?)?;
        }
        st.memcpy_dtoh(&slot.totals, &mut slot.host_totals.as_mut_slice()?[..TOTALS])?;
        Ok(())
    }

    /// [`Device::collect`]
    fn finish(&mut self) -> Result<Option<ChunkOut>> {
        let Some((s, seeds)) = self.flying.pop_front() else { return Ok(None) };
        let out = self.download(s, seeds);
        // the slot's stream was synchronized, or its context is broken and the walk with it
        self.free.push(s);
        out.map(Some)
    }

    fn download(&mut self, s: usize, seeds: Range<usize>) -> Result<ChunkOut> {
        let c = self.caps;
        let slot = &mut self.slots[s];
        let st = slot.stream.clone();
        st.synchronize().wrap_err("run the GPU walk's kernels")?;
        let t: [u32; TOTALS] = slot.host_totals.as_slice()?[..TOTALS].try_into().expect("TOTALS counts");
        ensure!(t[T_SEEDS] as usize == seeds.len(), "the GPU walk traced {} seeds of {}", t[T_SEEDS], seeds.len());
        let n = seeds.len();
        let n_checks = t[T_CHECKS].min(c.checks) as usize;
        let n_pts = t[T_PTS].min(c.pts) as usize;
        let n_curves = t[T_CURVES].min(c.curves) as usize;
        let words = n * RESULT_WORDS;
        fit_host(&st, &mut slot.host_checks, n_checks * CHECK_WORDS, 0)?;
        fit_host(&st, &mut slot.host_pts, n_pts * 3, 0)?;
        fit_host(&st, &mut slot.host_tans, n_pts * 3, 0)?;
        fit_host(&st, &mut slot.host_curves, n_curves * 2, 0)?;
        st.memcpy_dtoh(&slot.results.slice(..words), &mut slot.host_results.as_mut_slice()?[..words])?;
        if n_checks > 0 {
            let w = n_checks * CHECK_WORDS;
            st.memcpy_dtoh(&slot.out_checks.slice(..w), &mut slot.host_checks.as_mut_slice()?[..w])?;
        }
        if n_pts > 0 {
            st.memcpy_dtoh(&slot.out_pts.slice(..n_pts * 3), &mut slot.host_pts.as_mut_slice()?[..n_pts * 3])?;
            st.memcpy_dtoh(&slot.out_tans.slice(..n_pts * 3), &mut slot.host_tans.as_mut_slice()?[..n_pts * 3])?;
        }
        if n_curves > 0 {
            st.memcpy_dtoh(
                &slot.out_curves.slice(..n_curves * 2),
                &mut slot.host_curves.as_mut_slice()?[..n_curves * 2],
            )?;
        }
        st.synchronize().wrap_err("copy the GPU walk's results back")?;

        let results = slot.host_results.as_slice()?[..words]
            .as_chunks::<RESULT_WORDS>()
            .0
            .iter()
            .map(|r| SeedOut {
                status: if r[0] == 0 { Status::Done } else { Status::Cpu },
                checks: r[1]..r[2],
                curves: r[3]..r[4],
                tans: r[5]..r[6],
                // the host marks the coverage from the points when it uses a result
                marks: 0..0,
            })
            .collect();
        let checks = slot.host_checks.as_slice()?[..n_checks * CHECK_WORDS]
            .as_chunks::<CHECK_WORDS>()
            .0
            .par_iter()
            .map(|r| RawCheck {
                slot: r[0] as u32,
                offset: (r[0] >> 32) as u16,
                seen: (r[0] >> 48) as u16,
                same_dir: (r[1] != 0).then(|| [f64::from_bits(r[2]), f64::from_bits(r[3]), f64::from_bits(r[4])]),
            })
            .collect();
        let xyz = |v: &[f32]| -> Vec<[f32; 3]> { v.as_chunks::<3>().0.to_vec() };
        let pts = xyz(&slot.host_pts.as_slice()?[..n_pts * 3]);
        let tans = xyz(&slot.host_tans.as_slice()?[..n_pts * 3]);
        let curves =
            slot.host_curves.as_slice()?[..n_curves * 2].as_chunks::<2>().0.iter().map(|r| r[0]..r[1]).collect();
        Ok(ChunkOut { seeds, results, checks, pts, tans, curves, marks: Vec::new() })
    }
}

/// The walk as the producer's look-ahead drives it, from a thread of its own
impl Device for Walk<'_> {
    /// Apply coverage changes the CPU committed to the device's mirror, on its own stream; chunks in flight may
    /// read the mirror while it changes (each u16 read is the old value or the new one, and results record what
    /// they read)
    ///
    /// A batch may hold one voxel more than once: coverage only goes from 0 to a direction to 1 (a count of two),
    /// and the device applies a batch in that order whatever order it lists the changes in
    fn update(&mut self, changes: &[Change]) -> Result<()> {
        self.gpu.ctx.bind_to_thread()?;
        self.apply(changes)
    }

    /// Queue the catalogue's seeds `seeds` (at most [`CHUNK`]) as one chunk; at most [`IN_FLIGHT`] chunks may be
    /// launched and not yet collected
    fn launch(&mut self, seeds: Range<usize>) -> Result<()> {
        self.gpu.ctx.bind_to_thread()?;
        self.start(seeds)
    }

    /// The oldest launched chunk's results, waiting for it; None when no chunk is in flight
    fn collect(&mut self) -> Result<Option<ChunkOut>> {
        self.gpu.ctx.bind_to_thread()?;
        self.finish()
    }

    fn in_flight(&self) -> usize {
        self.flying.len()
    }
}

impl Drop for Walk<'_> {
    /// Wait for queued chunks and coverage updates before the fields, mirror and staging buffers they read are freed
    ///
    /// The kernels take raw pointers, so cudarc's events do not hold a buffer's free behind them
    fn drop(&mut self) {
        // a failed sync means the context is broken, and freeing then is all that is left to do
        for slot in &self.slots {
            let _ = slot.stream.synchronize();
        }
        let _ = self.cov.stream.synchronize();
    }
}
