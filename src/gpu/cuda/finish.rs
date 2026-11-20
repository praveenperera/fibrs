//! Trace finishing on the GPU, bit for bit `FinishCtx::finish` for a whole batch of raw curves
//!
//! Both pooled predictions go up once per family, a table and a pool of bricks each. The native predictions go up
//! slab by slab into a slab cache sized from the device memory free then. Each batch lists every slab a sample of
//! its curves can read, decodes the ones the cache lacks on the CPU through the `NativeCache`, in parallel on the
//! finishing pool, and uploads them in place of the least recently used. `fs_segments` then flags each raw segment,
//! the host cuts the curves into pieces as `finish` does, `fs_fit` smooths and resamples each piece of at least
//! [`MIN_FIT_POINTS`] points, and `fs_segments` tests each fitted piece whole. A batch the kernels flag, or one that
//! reads more slabs than the cache holds, comes back as [`Finished::OnCpu`] for the CPU to finish, so the output is
//! the CPU's either way

use std::ops::RangeInclusive;
use std::sync::Arc;

use color_eyre::eyre::{Result, WrapErr, ensure, eyre};
use cudarc::driver::{
    CudaEvent, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchArgs, LaunchConfig, PinnedHostSlice,
    PushKernelArg, result, sys,
};
use rayon::ThreadPool;
use rayon::prelude::*;

use super::{BLOCK, Gpu, NO_BRICK, fit, grid_1d, pinned};
use crate::finish::{NativeCache, SlabLayout, valid_runs};
use crate::gpu::{CpuReason, Finished, Pieces};
use crate::kernels::{MIN_FIT_POINTS, RawCurve};
use crate::vol::{BRICK, Shape, Vol};

/// Pinned staging for uploads, in two halves so one fills while the other's copies run; host memory, not copy
/// speed, bounds it, so it stays small
const STAGING_BYTES: usize = 64 << 20;
/// Most device memory the slab cache takes, however much is free: at 256 KiB a slab (8 planes of a 128³ chunk,
/// both channels) that is 32,768 slabs, far more than a batch of curves reads
const SLAB_CACHE_MAX: usize = 8 << 30;
/// Device bytes a batch's buffers take per raw point, at most: 12 for the point, 12 for its smoothed copy and 1
/// for its flag; 27 for its share of the fitted room, at most 2.25 points a raw point (9 for a 4-point piece) at
/// the tracer's step of 0.75; 8 for a curve's first slot and length and 5 for a fitted piece's five words, as a
/// curve holds a point at least and a fitted piece 4. Twice that covers the spare half `fit` leaves when it grows
/// a buffer and the smaller buffer it replaces
const BATCH_BYTES_PER_POINT: usize = 2 * (12 + 12 + 1 + 27 + 8 + 5);
/// How far, in tracing voxels, a sample of a fitted segment can lie from the raw segments of its piece: fitting
/// moves a point at most 0.3 voxels, and the Catmull-Rom resampling overshoots the smoothed points by under 0.1 of
/// the neighbouring segments' lengths (about 0.15 at the tracer's step of 0.75). The kernels flag a read of a slab
/// outside the listing, so a sample beyond this sends its batch to the CPU instead of reading a wrong value
const REACH: f32 = 1.0;
/// The slab table's entry for a slab not in the cache
const NO_SLAB: u32 = u32::MAX;
/// `err` bit the kernels set when a sample reads a slab not in the cache
const ERR_SLAB: u32 = 1;
/// `err` bit `fs_fit` sets when a fitted piece outgrows its room in `fitted`
const ERR_CAPACITY: u32 = 2;

/// Finishes batches of one family's raw curves on the GPU: the pooled predictions and the native slab cache on
/// the device, and the buffers each batch reuses
pub struct GpuFinisher {
    /// `fs_segments` and `fs_fit`
    kernels: [CudaFunction; 2],
    stream: Arc<CudaStream>,
    /// 1 = V, 2 = H
    family: u32,
    tracing: Tracing,
    slabs: SlabCache,
    staging: Staging,
    bufs: Buffers,
}

impl Gpu {
    /// A finisher of family `family` (1 = V, 2 = H) that reads `tracing` and native predictions laid out as
    /// `layout`, for batches of about `batch_points` raw points at most; uploads both pooled predictions and sizes
    /// the slab cache from the device memory free after them, less what such a batch's buffers take
    pub fn finisher(
        &self,
        tracing: &[Vol<u8>; 2],
        family: usize,
        layout: SlabLayout,
        batch_points: usize,
    ) -> Result<GpuFinisher> {
        self.finisher_with(tracing, family, layout, batch_points, SLAB_CACHE_MAX)
    }

    /// [`Gpu::finisher`] with a slab cache of at most `cache_bytes`
    fn finisher_with(
        &self,
        tracing: &[Vol<u8>; 2],
        family: usize,
        layout: SlabLayout,
        batch_points: usize,
        cache_bytes: usize,
    ) -> Result<GpuFinisher> {
        let shape = tracing[0].shape();
        ensure!(tracing[1].shape() == shape, "the pooled predictions differ in shape");
        // the kernels' default for bricks without a payload
        ensure!(tracing.iter().all(|v| v.default_value() == 0), "unexpected pooled prediction defaults");
        ensure!(family == 1 || family == 2, "family {family} is neither V (1) nor H (2)");
        let grid = tracing[0].grid();
        ensure!(
            kernels_index_pooled(shape, grid),
            "pooled predictions of {shape:?} voxels in a {grid:?} brick grid are too large for the finishing \
             kernels' u32 indices"
        );
        ensure!(
            kernels_index(&layout),
            "native predictions of {:?} voxels in chunks of {:?}, {} planes a slab, are too large for the finishing \
             kernels' u32 indices",
            layout.shape,
            layout.edge,
            layout.planes
        );
        let stream = self.ctx.new_stream()?;
        let mut staging = Staging::new(&stream, layout.slab_bytes())?;
        let tracing = Tracing::upload(&stream, &mut staging, tracing)?;
        let reserve = batch_points.saturating_mul(BATCH_BYTES_PER_POINT);
        let slabs = SlabCache::new(self, &stream, layout, reserve, cache_bytes)?;
        let bufs = Buffers::new(&stream)?;
        let kernels = [self.k.fs_segments.clone(), self.k.fs_fit.clone()];
        Ok(GpuFinisher { kernels, stream, family: family as u32, tracing, slabs, staging, bufs })
    }
}

impl GpuFinisher {
    /// Slabs the cache holds at once
    pub fn slots(&self) -> usize {
        self.slabs.held.len()
    }

    /// Slabs uploaded so far, counting uploads again after eviction
    pub fn uploaded(&self) -> u64 {
        self.slabs.uploaded
    }

    /// Finish `batch`, each curve's pieces equal to `FinishCtx::finish`'s, in batch order; or say why the CPU
    /// must finish it instead. Decoding runs on `pool`, through `cache`, which must lie as the finisher's layout
    ///
    /// An error is a CUDA error, after which the finisher should not be used again
    pub fn finish(&mut self, pool: &ThreadPool, cache: &NativeCache, batch: &[RawCurve]) -> Result<Finished> {
        if batch.is_empty() {
            return Ok(Finished::Pieces(Vec::new()));
        }
        let layout = self.slabs.layout;
        ensure!(cache.layout() == layout, "the native cache lies as {:?}, the GPU's as {layout:?}", cache.layout());
        let needed = pool.install(|| list_slabs(&layout, batch));
        if needed.len() > self.slots() {
            return Ok(Finished::OnCpu(CpuReason::Slabs));
        }
        if !self.slabs.load(&self.stream, &mut self.staging, pool, cache, &needed)? {
            return Ok(Finished::OnCpu(CpuReason::Decode));
        }

        let raw = Raw::pack(batch)?;
        let (valid, err) = self.raw_pass(&raw)?;
        if let Some(reason) = flagged(err) {
            return Ok(Finished::OnCpu(reason));
        }
        let cuts = pool.install(|| cut(batch, &raw.start, &valid));
        let plan = pool.install(|| Plan::new(batch, &raw.start, &cuts))?;
        let fitted = if plan.is_empty() { Fitted::default() } else { self.fit_pass(&plan, raw.points())? };
        if let Some(reason) = flagged(fitted.err) {
            return Ok(Finished::OnCpu(reason));
        }
        // a piece the kernels left too short or too long means their output cannot be trusted
        let lengths_ok =
            fitted.flen.iter().zip(plan.fstart.windows(2)).all(|(&n, room)| (2..=room[1] - room[0]).contains(&n));
        if !lengths_ok {
            return Ok(Finished::OnCpu(CpuReason::Capacity));
        }
        Ok(Finished::Pieces(pool.install(|| assemble(batch, &cuts, &plan, &fitted))))
    }

    /// Upload the raw curves, run `fs_segments` over them and bring back each point slot's flag (1 where the
    /// segment from it to the next point is valid) and `err`
    fn raw_pass(&mut self, raw: &Raw) -> Result<(Vec<u8>, u32)> {
        let (n, curves) = (raw.points(), raw.len.len());
        let stream = self.stream.clone();
        let b = &mut self.bufs;
        fit(&stream, &mut b.pts, 3 * n)?;
        fit(&stream, &mut b.start, curves + 1)?;
        fit(&stream, &mut b.len, curves)?;
        fit(&stream, &mut b.valid, n)?;
        let mut valid = vec![0u8; n];
        let mut err = [0u32];
        // after the host memory the copies read and write, so an early return waits for them before it is freed
        let pending = Pending(&stream);
        stream.memcpy_htod(&raw.pts, &mut b.pts.slice_mut(..3 * n))?;
        stream.memcpy_htod(&raw.start, &mut b.start.slice_mut(..curves + 1))?;
        stream.memcpy_htod(&raw.len, &mut b.len.slice_mut(..curves))?;
        // a curve's last slot starts no segment and is never written
        stream.memset_zeros(&mut b.valid.slice_mut(..n))?;
        stream.memset_zeros(&mut b.err)?;

        let segments = Segments { mode: 0, curves: u32_of(curves)?, slots: u32_of(n)?, family: self.family };
        let curves_in = (&b.pts, &b.start, &b.len);
        segments.queue(
            &stream,
            &self.kernels[0],
            &self.tracing,
            &self.slabs,
            curves_in,
            (&mut b.valid, &mut b.ok, &mut b.err),
        )?;

        stream.memcpy_dtoh(&b.valid.slice(..n), &mut valid)?;
        stream.memcpy_dtoh(&b.err, &mut err[..])?;
        // the pageable sources above are read by then too
        pending.wait("run the finishing kernels")?;
        Ok((valid, err[0]))
    }

    /// Run `fs_fit` over the pieces of `plan`, whose raw points are already on the device among `raw_points`, then
    /// `fs_segments` over the fitted pieces, and bring back what they wrote
    fn fit_pass(&mut self, plan: &Plan, raw_points: usize) -> Result<Fitted> {
        let pieces = plan.plen.len();
        let room = *plan.fstart.last().expect("fstart holds one more entry than the pieces") as usize;
        let stream = self.stream.clone();
        let b = &mut self.bufs;
        fit(&stream, &mut b.pstart, pieces)?;
        fit(&stream, &mut b.plen, pieces)?;
        fit(&stream, &mut b.fstart, pieces + 1)?;
        fit(&stream, &mut b.flen, pieces)?;
        fit(&stream, &mut b.ok, pieces)?;
        fit(&stream, &mut b.smooth, 3 * raw_points)?;
        fit(&stream, &mut b.fitted, 3 * room)?;
        // the kernel clears a piece's flag when one of its segments fails
        let ones = vec![1u32; pieces];
        let mut out = Fitted { points: vec![0.0; 3 * room], flen: vec![0; pieces], ok: vec![0; pieces], err: 0 };
        let mut err = [0u32];
        // after the host memory the copies read and write, so an early return waits for them before it is freed
        let pending = Pending(&stream);
        stream.memcpy_htod(&plan.pstart, &mut b.pstart.slice_mut(..pieces))?;
        stream.memcpy_htod(&plan.plen, &mut b.plen.slice_mut(..pieces))?;
        stream.memcpy_htod(&plan.fstart, &mut b.fstart.slice_mut(..pieces + 1))?;
        stream.memcpy_htod(&ones, &mut b.ok.slice_mut(..pieces))?;
        stream.memset_zeros(&mut b.flen.slice_mut(..pieces))?;
        stream.memset_zeros(&mut b.err)?;

        let n_pieces = u32_of(pieces)?;
        {
            let mut args = stream.launch_builder(&self.kernels[1]);
            args.arg(&b.pts).arg(&b.pstart).arg(&b.plen).arg(&n_pieces).arg(&mut b.smooth).arg(&mut b.fitted);
            args.arg(&b.fstart).arg(&mut b.flen).arg(&mut b.err).arg(&self.family);
            self.tracing.push(&mut args);
            // SAFETY: the params are `fs_fit`'s in its PTX order (checked in `Gpu::open`); every piece's raw points
            // lie in `pts` and `smooth` (3 × raw_points floats each) and its room in `fitted` (3 × room floats), as
            // `plan` lays them out, and the launch runs in order on this stream after the uploads above
            unsafe { args.launch(flat(pieces)?) }?;
        }

        let segments = Segments { mode: 1, curves: n_pieces, slots: u32_of(room)?, family: self.family };
        let curves_in = (&b.fitted, &b.fstart, &b.flen);
        segments.queue(
            &stream,
            &self.kernels[0],
            &self.tracing,
            &self.slabs,
            curves_in,
            (&mut b.valid, &mut b.ok, &mut b.err),
        )?;

        stream.memcpy_dtoh(&b.flen.slice(..pieces), &mut out.flen)?;
        stream.memcpy_dtoh(&b.ok.slice(..pieces), &mut out.ok)?;
        stream.memcpy_dtoh(&b.err, &mut err[..])?;
        stream.memcpy_dtoh(&b.fitted.slice(..3 * room), &mut out.points)?;
        pending.wait("run the fitting kernels")?;
        out.err = err[0];
        Ok(out)
    }
}

/// Waits, when dropped, for the work queued on a stream, so host memory its copies read or write outlives them
/// when an error returns early: cudarc does not track plain host slices, and the driver may read pageable memory
/// after an asynchronous copy returns. Locals drop in reverse order, so it guards the memory declared before it.
/// It waits through [`drain`], so an error cudarc stored cannot skip the wait
struct Pending<'a>(&'a CudaStream);

impl Pending<'_> {
    /// Wait for the queued work, saying on failure that it was `what`
    fn wait(self, what: &'static str) -> Result<()> {
        let stream = self.0;
        // waited for here, so the drop has nothing left to wait for
        std::mem::forget(self);
        drain(stream).wrap_err(what)
    }
}

impl Drop for Pending<'_> {
    fn drop(&mut self) {
        // a failed wait means the context is broken, and freeing then is all that is left to do
        let _ = drain(self.0);
    }
}

/// Wait for everything queued on `stream`, then return any error cudarc stored before, as
/// `CudaStream::synchronize` would; every wait that keeps host memory alive for a copy goes through here
///
/// `CudaStream::synchronize` returns a stored error, such as a failed record of a copy's tracking event after the
/// copy was queued, before it reaches the driver, so it could return while that copy still reads memory about to
/// be freed. This waits through the driver first, whatever is stored
fn drain(stream: &CudaStream) -> Result<(), DriverError> {
    let ctx = stream.context();
    // SAFETY: the context and stream handles are live for as long as `stream` is borrowed, and neither call frees
    // or changes them; the stream is still waited for when making its context current fails
    let (bound, waited) =
        unsafe { (result::ctx::set_current(ctx.cu_ctx()), result::stream::synchronize(stream.cu_stream())) };
    ctx.check_err()?;
    bound.and(waited)
}

/// The reason in a kernel's `err` word, if it set one
fn flagged(err: u32) -> Option<CpuReason> {
    match err {
        0 => None,
        e if e & ERR_SLAB != 0 => Some(CpuReason::Unlisted),
        e if e & ERR_CAPACITY != 0 => Some(CpuReason::Capacity),
        // a bit no kernel sets still means their output cannot be trusted
        _ => Some(CpuReason::Capacity),
    }
}

fn u32_of(n: usize) -> Result<u32> {
    u32::try_from(n).map_err(|_| eyre!("a batch of {n} is too large for the finishing kernels' indices"))
}

fn flat(n: usize) -> Result<LaunchConfig> {
    Ok(LaunchConfig { grid_dim: (grid_1d(n)?, 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 })
}

/// One `fs_segments` launch: mode 0 flags each raw segment, mode 1 clears the flag of each fitted piece one of
/// whose segments fails
struct Segments {
    mode: u32,
    curves: u32,
    /// Point slots of every curve: the threads
    slots: u32,
    family: u32,
}

impl Segments {
    /// Queue the launch over the curves `(pts, start, len)`, writing `(valid, ok, err)`
    #[allow(clippy::type_complexity)]
    fn queue(
        &self,
        stream: &Arc<CudaStream>,
        kernel: &CudaFunction,
        tracing: &Tracing,
        slabs: &SlabCache,
        (pts, start, len): (&CudaSlice<f32>, &CudaSlice<u32>, &CudaSlice<u32>),
        (valid, ok, err): (&mut CudaSlice<u8>, &mut CudaSlice<u32>, &mut CudaSlice<u32>),
    ) -> Result<()> {
        let mut args = stream.launch_builder(kernel);
        args.arg(pts).arg(start).arg(len).arg(&self.curves).arg(&self.slots).arg(&self.mode);
        args.arg(valid).arg(ok).arg(err).arg(&self.family);
        tracing.push(&mut args);
        slabs.push(&mut args);
        // SAFETY: the params are `fs_segments`' in its PTX order (checked in `Gpu::open`). `start` holds
        // `curves + 1` offsets ending at `slots` and `len` a length per curve, so every point a thread reads lies in
        // `pts`; `valid` holds a flag per slot in mode 0 and `ok` a flag per curve in mode 1 (the other is never
        // touched); every slab the table names holds a slot of the pool. The launch runs in order on this stream
        unsafe { args.launch(flat(self.slots as usize)?) }?;
        Ok(())
    }
}

/// A batch's raw curves packed for the device: float32 x, y, z per point, and each curve's first slot and length
struct Raw {
    pts: Vec<f32>,
    /// One more entry than the curves, the last the number of points
    start: Vec<u32>,
    len: Vec<u32>,
}

impl Raw {
    fn pack(batch: &[RawCurve]) -> Result<Self> {
        let n = batch.iter().map(|c| c.pts.len()).sum::<usize>();
        // the kernels index point coordinates in u32
        u32_of(3 * n)?;
        let mut pts = Vec::with_capacity(3 * n);
        let mut start = Vec::with_capacity(batch.len() + 1);
        let mut len = Vec::with_capacity(batch.len());
        for c in batch {
            start.push((pts.len() / 3) as u32);
            len.push(c.pts.len() as u32);
            pts.extend_from_slice(c.pts.as_flattened());
        }
        start.push(n as u32);
        Ok(Self { pts, start, len })
    }

    fn points(&self) -> usize {
        self.pts.len() / 3
    }
}

/// Each curve's raw pieces, as ranges of its points, from the flags `fs_segments` wrote at each segment's first
/// point slot; a curve's last slot starts no segment, so its flag is never read
fn cut(batch: &[RawCurve], start: &[u32], valid: &[u8]) -> Vec<Vec<RangeInclusive<usize>>> {
    batch
        .par_iter()
        .zip(start)
        .map(|(c, &s)| {
            let flags = &valid[s as usize..s as usize + c.pts.len()];
            valid_runs(c.pts.len().saturating_sub(1), |i| flags[i] != 0).collect()
        })
        .collect()
}

fn points_in(r: &RangeInclusive<usize>) -> usize {
    r.end() + 1 - r.start()
}

/// Room for the fitted piece of raw piece `pts`: `fit_curve` resamples at `arc / 0.8 + 1.5` points at most, and
/// moves each point at most 0.3 voxels from its raw place (plus float32 rounding, under 0.002 voxels below 16,384),
/// so each smoothed segment is at most 0.6 voxels longer than its raw one; 0.65 a segment and 3 more points spare
/// that rounding and the float32 sum of the arc. `fs_fit` flags a piece that does not fit anyway
fn fitted_room(pts: &[[f32; 3]]) -> usize {
    let arc = pts
        .windows(2)
        .map(|s| (0..3).map(|k| (s[1][k] as f64 - s[0][k] as f64).powi(2)).sum::<f64>().sqrt())
        .sum::<f64>();
    ((arc + 0.65 * (pts.len() - 1) as f64) / 0.8).ceil() as usize + 3
}

/// The pieces `fs_fit` smooths, those of at least [`MIN_FIT_POINTS`] points, in batch order
struct Plan {
    /// Each piece's first raw point slot and length
    pstart: Vec<u32>,
    plen: Vec<u32>,
    /// Each piece's first slot in `fitted`, and one more entry, the room of every piece; `fstart[p + 1] -
    /// fstart[p]` is piece p's room
    fstart: Vec<u32>,
    /// Each curve's first piece among those fitted: the fitted pieces of the curves before it
    first: Vec<usize>,
}

impl Plan {
    fn new(batch: &[RawCurve], start: &[u32], cuts: &[Vec<RangeInclusive<usize>>]) -> Result<Self> {
        let rooms: Vec<Vec<usize>> = batch
            .par_iter()
            .zip(cuts)
            .map(|(c, rs)| {
                let fitted = rs.iter().filter(|r| points_in(r) >= MIN_FIT_POINTS);
                fitted.map(|r| fitted_room(&c.pts[r.clone()])).collect()
            })
            .collect();
        let mut plan =
            Plan { pstart: Vec::new(), plen: Vec::new(), fstart: vec![0], first: Vec::with_capacity(batch.len()) };
        let mut room = 0usize;
        for ((rs, &s), curve_rooms) in cuts.iter().zip(start).zip(&rooms) {
            plan.first.push(plan.plen.len());
            for r in rs.iter().filter(|r| points_in(r) >= MIN_FIT_POINTS) {
                plan.pstart.push(s + *r.start() as u32);
                plan.plen.push(points_in(r) as u32);
            }
            for &n in curve_rooms {
                room += n;
                plan.fstart.push(u32_of(room)?);
            }
        }
        // the kernels index fitted coordinates in u32
        u32_of(3 * room)?;
        Ok(plan)
    }

    fn is_empty(&self) -> bool {
        self.plen.is_empty()
    }
}

/// What the fitting pass wrote, for each piece of a [`Plan`]
#[derive(Default)]
struct Fitted {
    /// float32 x, y, z of every piece's room
    points: Vec<f32>,
    /// Each fitted piece's length in points
    flen: Vec<u32>,
    /// 1 where every segment of the fitted piece is valid
    ok: Vec<u32>,
    err: u32,
}

/// Each curve's finished pieces: a piece's fitted points when it was fitted and every fitted segment is valid,
/// else its raw points, as `FinishCtx::finish` chooses
fn assemble(batch: &[RawCurve], cuts: &[Vec<RangeInclusive<usize>>], plan: &Plan, fitted: &Fitted) -> Vec<Pieces> {
    let points = fitted.points.as_chunks::<3>().0;
    batch
        .par_iter()
        .zip(cuts)
        .zip(&plan.first)
        .map(|((c, rs), &first)| {
            let mut p = first;
            let mut piece = |r: &RangeInclusive<usize>| {
                let raw = c.pts[r.clone()].to_vec();
                if points_in(r) < MIN_FIT_POINTS {
                    return raw;
                }
                let k = p;
                p += 1;
                if fitted.ok[k] == 0 {
                    return raw;
                }
                let at = plan.fstart[k] as usize;
                points[at..at + fitted.flen[k] as usize].to_vec()
            };
            rs.iter().map(&mut piece).collect()
        })
        .collect()
}

/// Every slab of `layout` a sample of the batch can read, as sorted unique table indices
///
/// Every raw segment is sampled, and every fitted segment lies within [`REACH`] of the raw segments of its piece,
/// so each raw segment's box grown by [`REACH`] covers both
fn list_slabs(layout: &SlabLayout, batch: &[RawCurve]) -> Vec<u32> {
    let mut all: Vec<u32> = batch.par_iter().flat_map_iter(|c| curve_slabs(layout, &c.pts)).collect();
    all.par_sort_unstable();
    all.dedup();
    all
}

/// The slabs of one curve, as [`list_slabs`]
fn curve_slabs(layout: &SlabLayout, pts: &[[f32; 3]]) -> Vec<u32> {
    let mut out = Vec::new();
    for seg in pts.windows(2) {
        let (lo, hi) = native_box(layout.shape, seg[0], seg[1]);
        out.extend(layout.indices_in(lo, hi).map(|i| i as u32));
    }
    // consecutive segments mostly read the same slabs, so the batch's sort sees each about once per curve
    out.sort_unstable();
    out.dedup();
    out
}

/// The first and last native voxel, z, y, x, that a sample within [`REACH`] of the segment from `a` to `b` (float32
/// x, y, z in tracing voxels) can read: `NativeSampler::sample` reads the cell at `2 q + 0.5`, clipped to the array
fn native_box(shape: Shape, a: [f32; 3], b: [f32; 3]) -> (Shape, Shape) {
    let axis = |n: usize, k: usize| {
        let top = (n - 1) as f64;
        // NaN clips to 0 here as `sample`'s `max(0.0)` clips it
        let native = |v: f32| (v as f64 * 2.0 + 0.5).max(0.0).min(top).floor() as usize;
        let lo = native(a[k].min(b[k]) - REACH);
        let hi = (native(a[k].max(b[k]) + REACH) + 1).min(n - 1);
        (lo, hi)
    };
    let [z, y, x] = [axis(shape[0], 2), axis(shape[1], 1), axis(shape[2], 0)];
    ([z.0, y.0, x.0], [z.1, y.1, x.1])
}

/// Both pooled predictions on the device, a table from each brick of the grid to its payload in a pool, or
/// [`NO_BRICK`], per volume
struct Tracing {
    /// V then H
    tables: [CudaSlice<u32>; 2],
    pools: [CudaSlice<u8>; 2],
    /// The shape, then the brick grid, z, y, x
    dims: [u32; 6],
}

impl Tracing {
    fn upload(stream: &Arc<CudaStream>, staging: &mut Staging, tracing: &[Vol<u8>; 2]) -> Result<Self> {
        let (table_v, pool_v) = Self::upload_vol(stream, staging, &tracing[0])?;
        let (table_h, pool_h) = Self::upload_vol(stream, staging, &tracing[1])?;
        let (shape, grid) = (tracing[0].shape(), tracing[0].grid());
        let dims = [shape[0], shape[1], shape[2], grid[0], grid[1], grid[2]].map(|n| n as u32);
        Ok(Self { tables: [table_v, table_h], pools: [pool_v, pool_h], dims })
    }

    /// One volume's table and pool, the payloads through `staging` a half at a time
    fn upload_vol(
        stream: &Arc<CudaStream>,
        staging: &mut Staging,
        vol: &Vol<u8>,
    ) -> Result<(CudaSlice<u32>, CudaSlice<u8>)> {
        let n_grid = vol.grid().iter().product::<usize>();
        let mut table = vec![NO_BRICK; n_grid];
        let mut payloads: Vec<&[u8; BRICK]> = Vec::new();
        for (bi, slot) in table.iter_mut().enumerate() {
            if let Some(b) = vol.brick_at(bi) {
                *slot = payloads.len() as u32;
                payloads.push(b);
            }
        }
        ensure!(payloads.len() < NO_BRICK as usize, "a pooled prediction holds too many bricks");

        let mut pool = stream.alloc_zeros::<u8>((payloads.len() * BRICK).max(1)).wrap_err("allocate GPU memory")?;
        let per_half = staging.half() / BRICK;
        for (k, part) in payloads.chunks(per_half).enumerate() {
            let len = part.len() * BRICK;
            let host = &mut staging.take()?[..len];
            host.par_chunks_mut(BRICK).zip(part.par_iter()).for_each(|(d, s)| d.copy_from_slice(&s[..]));
            let at = k * per_half * BRICK;
            stream.memcpy_htod(&*host, &mut pool.slice_mut(at..at + len))?;
            staging.queued(stream)?;
        }
        let device_table = stream.clone_htod(&table)?;
        // the driver may read pageable memory after the copy returns, and `table` is freed here
        drain(stream).wrap_err("upload the pooled predictions")?;
        Ok((device_table, pool))
    }

    /// Push the params every finishing kernel takes for the pooled predictions, in PTX order
    fn push<'a>(&'a self, args: &mut LaunchArgs<'a>) {
        args.arg(&self.tables[0]).arg(&self.tables[1]).arg(&self.pools[0]).arg(&self.pools[1]);
        for v in &self.dims {
            args.arg(v);
        }
    }
}

/// Pinned host memory for uploads in two halves: one half fills while the copies from the other run
struct Staging {
    buf: PinnedHostSlice<u8>,
    /// The copies queued from each half, which must finish before it is written again
    queued: [Option<CudaEvent>; 2],
    next: usize,
    /// The stream every copy from `buf` is queued on
    stream: Arc<CudaStream>,
}

impl Staging {
    /// Halves of [`STAGING_BYTES`], or of `most` bytes, one upload, when that is larger
    fn new(stream: &Arc<CudaStream>, most: usize) -> Result<Self> {
        let half = (STAGING_BYTES / 2).max(most);
        // write-combined, as the host only writes it
        let buf = pinned(stream.context(), 2 * half, sys::CU_MEMHOSTALLOC_WRITECOMBINED)?;
        Ok(Self { buf, queued: [None, None], next: 0, stream: stream.clone() })
    }

    fn half(&self) -> usize {
        self.buf.len() / 2
    }

    /// The next half to fill, once the copies queued from it before have finished
    fn take(&mut self) -> Result<&mut [u8]> {
        if let Some(done) = self.queued[self.next].take() {
            done.synchronize()?;
        }
        let half = self.half();
        Ok(&mut self.buf.as_mut_slice()?[self.next * half..][..half])
    }

    /// Record that the copies queued on `stream` so far read the half [`Staging::take`] gave, and move to the other
    fn queued(&mut self, stream: &Arc<CudaStream>) -> Result<()> {
        self.queued[self.next] = Some(stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING))?);
        self.next ^= 1;
        Ok(())
    }
}

impl Drop for Staging {
    /// Wait for every copy queued from `buf` before it is freed: cudarc does not track the plain slices the copies
    /// were given, and an error, while the finisher is set up or later, can leave copies queued that no event in
    /// `queued` covers yet
    fn drop(&mut self) {
        // a failed wait means the context is broken, and freeing then is all that is left to do
        let _ = drain(&self.stream);
    }
}

/// The native predictions' slabs on the device: a pool of slots, each holding one slab in
/// `NativeCache::copy_slab`'s layout, and a table from each slab of the layout to its slot, or [`NO_SLAB`]
///
/// Slots are reused least recently read first across batches; a batch's own slabs are never evicted for it
struct SlabCache {
    layout: SlabLayout,
    /// The native shape, the chunk edge, planes per slab, the chunk grid's y and x, and slabs per chunk
    params: [u32; 10],
    /// Each slab's slot, or [`NO_SLAB`]: what `table` holds once written
    host: Vec<u32>,
    table: CudaSlice<u32>,
    pool: CudaSlice<u8>,
    /// Each slot's slab, as its table index, or [`NO_SLAB`] while the slot is empty
    held: Vec<u32>,
    /// The batch that last read each slot
    used: Vec<u64>,
    batch: u64,
    /// The span of `host` changed since `table` was last written
    dirty: Option<(usize, usize)>,
    uploaded: u64,
}

impl SlabCache {
    /// A cache as [`slab_slots`] sizes it from the device memory free now, leaving room for its table and
    /// `reserve` bytes for a batch's buffers, and holding at most `most` bytes
    fn new(gpu: &Gpu, stream: &Arc<CudaStream>, layout: SlabLayout, reserve: usize, most: usize) -> Result<Self> {
        let (free, _) = gpu.ctx.mem_get_info()?;
        let bytes = layout.slab_bytes();
        let slots = slab_slots(free, reserve, most, bytes, layout.len());
        ensure!(slots > 0, "{} MiB free on the GPU leave no room for native slabs", free >> 20);
        // the pool first, so a failed allocation leaves no copy queued
        let pool = stream.alloc_zeros::<u8>(slots * bytes).wrap_err("allocate GPU memory")?;
        let host = vec![NO_SLAB; layout.len()];
        let table = stream.clone_htod(&host)?;
        // the driver may read pageable memory after the copy returns, and `host` is freed with the cache, which an
        // error before the first batch's table write can drop
        drain(stream).wrap_err("upload the slab table")?;
        let [_, ncy, ncx] = layout.chunks();
        let (s, e) = (layout.shape, layout.edge);
        let params =
            [s[0], s[1], s[2], e[0], e[1], e[2], layout.planes, ncy, ncx, layout.per_chunk()].map(|n| n as u32);
        Ok(Self {
            layout,
            params,
            host,
            table,
            pool,
            held: vec![NO_SLAB; slots],
            used: vec![0; slots],
            batch: 0,
            dirty: None,
            uploaded: 0,
        })
    }

    /// Make every slab of `needed` (sorted unique table indices, no more than the slots) resident, decoding the
    /// missing ones through `cache` on `pool` and uploading them; false when one fails to decode
    fn load(
        &mut self,
        stream: &Arc<CudaStream>,
        staging: &mut Staging,
        pool: &ThreadPool,
        cache: &NativeCache,
        needed: &[u32],
    ) -> Result<bool> {
        self.batch += 1;
        let mut missing = Vec::new();
        for &i in needed {
            match self.host[i as usize] {
                NO_SLAB => missing.push(i),
                slot => self.used[slot as usize] = self.batch,
            }
        }
        if missing.is_empty() {
            return Ok(true);
        }

        let slots = self.victims(missing.len());
        for &slot in &slots {
            let old = std::mem::replace(&mut self.held[slot as usize], NO_SLAB);
            if old != NO_SLAB {
                self.set(old as usize, NO_SLAB);
            }
        }
        let bytes = self.layout.slab_bytes();
        let per_half = staging.half() / bytes;
        let layout = self.layout;
        let mut decoded = true;
        for (keys, slots) in missing.chunks(per_half).zip(slots.chunks(per_half)) {
            let host = &mut staging.take()?[..keys.len() * bytes];
            let copied = pool.install(|| {
                host.par_chunks_mut(bytes)
                    .zip(keys)
                    .try_for_each(|(out, &i)| cache.copy_slab(layout.key(i as usize), out))
            });
            // the CPU finishes the batch, reading only the slabs it needs, and reports the error if one is this
            if copied.is_err() {
                decoded = false;
                break;
            }
            for ((&i, &slot), data) in keys.iter().zip(slots).zip(host.chunks_exact(bytes)) {
                let at = slot as usize * bytes;
                stream.memcpy_htod(data, &mut self.pool.slice_mut(at..at + bytes))?;
                self.held[slot as usize] = i;
                self.used[slot as usize] = self.batch;
                self.set(i as usize, slot);
            }
            staging.queued(stream)?;
            self.uploaded += keys.len() as u64;
        }
        // the evictions reach the device even when a decode failed, so no entry names a slot that changed
        self.write_table(stream)?;
        Ok(decoded)
    }

    /// `n` slots for new slabs: empty ones first, then the least recently read, never one this batch reads
    fn victims(&self, n: usize) -> Vec<u32> {
        let mut free: Vec<u32> = (0..self.held.len() as u32).filter(|&s| self.used[s as usize] != self.batch).collect();
        // an empty slot was last read in batch 0, before any slot that holds a slab
        let age = |&s: &u32| if self.held[s as usize] == NO_SLAB { 0 } else { self.used[s as usize] };
        if n < free.len() {
            free.select_nth_unstable_by_key(n, age);
        }
        free.truncate(n);
        free
    }

    fn set(&mut self, i: usize, slot: u32) {
        self.host[i] = slot;
        self.dirty = Some(self.dirty.map_or((i, i), |(lo, hi)| (lo.min(i), hi.max(i))));
    }

    /// Copy the changed span of the table to the device
    fn write_table(&mut self, stream: &Arc<CudaStream>) -> Result<()> {
        let Some((lo, hi)) = self.dirty.take() else { return Ok(()) };
        stream.memcpy_htod(&self.host[lo..=hi], &mut self.table.slice_mut(lo..=hi))?;
        // the driver may read pageable memory after the copy returns, and the next batch changes `host`
        drain(stream).wrap_err("upload the native slabs")?;
        Ok(())
    }

    /// Push the params `fs_segments` takes for the native predictions, in PTX order
    fn push<'a>(&'a self, args: &mut LaunchArgs<'a>) {
        args.arg(&self.table).arg(&self.pool);
        for v in &self.params {
            args.arg(v);
        }
    }
}

/// Whether the kernels can index native predictions laid out as `layout`: `Native::value`
/// (fibrs-kernels/src/finish.rs) takes the shape and chunk edge as u32 and computes in u32 a slab's table index,
/// which must stay below [`NO_SLAB`], and a slab's bytes per channel, `planes · ey · ex`, which bound a voxel's
/// offset in it; the slot's offset in the pool it computes in usize
fn kernels_index(layout: &SlabLayout) -> bool {
    let fits = |n: usize| u32::try_from(n).is_ok();
    let [_, ey, ex] = layout.edge;
    let per_channel = layout.planes.checked_mul(ey).and_then(|n| n.checked_mul(ex));
    layout.shape.iter().chain(&layout.edge).all(|&n| fits(n))
        && layout.len() < NO_SLAB as usize
        && per_channel.is_some_and(fits)
}

/// Whether the kernels can index pooled predictions of `shape` in a brick grid of `grid`: `Pooled::get`
/// (fibrs-kernels/src/finish.rs) takes the shape and grid as u32 and computes in u32 a voxel's brick index, at most
/// one below the grid's brick count, and its offset in the brick; the clamped voxel indices stay below the shape,
/// and the slot's offset in the pool it computes in usize
///
/// Each axis is also at most [`POOLED_AXIS_MAX`] voxels: fit's cell clamp rounds `n - 1.001` to float32, which
/// past that reaches `n - 1` and puts the cell's far corner outside the volume
fn kernels_index_pooled(shape: Shape, grid: Shape) -> bool {
    let fits = |n: usize| u32::try_from(n).is_ok();
    let bricks = grid.iter().try_fold(1usize, |acc, &n| acc.checked_mul(n));
    shape.iter().all(|&n| n <= POOLED_AXIS_MAX) && bricks.is_some_and(|n| n == 0 || fits(n - 1))
}

/// Longest pooled axis the GPU finishes, 2^15 voxels, three times the longest pooled axis of the scrolls traced so far (about 10,500)
const POOLED_AXIS_MAX: usize = 1 << 15;

/// Slots of `bytes` each for the slab cache: three quarters of the device memory `free` beyond `reserve` and the
/// slab table, the other quarter left for the driver and other processes, up to `most` bytes and no more slots than
/// the `slabs` the native predictions hold, which is below [`NO_SLAB`] (checked in [`Gpu::finisher`])
fn slab_slots(free: usize, reserve: usize, most: usize, bytes: usize, slabs: usize) -> usize {
    // the table, a u32 per slab, is allocated after the pool, so the pool must leave room for it; on small
    // chunks it can outgrow the pool
    let table = slabs.saturating_mul(size_of::<u32>());
    let budget = (free.saturating_sub(reserve).saturating_sub(table) / 4 * 3).min(most);
    (budget / bytes).min(slabs)
}

/// The device buffers each batch reuses, grown as batches need
struct Buffers {
    /// The raw curves' points and each curve's first slot and length
    pts: CudaSlice<f32>,
    start: CudaSlice<u32>,
    len: CudaSlice<u32>,
    /// A flag per raw point slot
    valid: CudaSlice<u8>,
    /// A flag per fitted piece
    ok: CudaSlice<u32>,
    err: CudaSlice<u32>,
    /// The fitted pieces' raw points, room and fitted length, and their smoothed and fitted points
    pstart: CudaSlice<u32>,
    plen: CudaSlice<u32>,
    fstart: CudaSlice<u32>,
    flen: CudaSlice<u32>,
    smooth: CudaSlice<f32>,
    fitted: CudaSlice<f32>,
}

impl Buffers {
    fn new(stream: &Arc<CudaStream>) -> Result<Self> {
        Ok(Self {
            pts: stream.alloc_zeros(1)?,
            start: stream.alloc_zeros(1)?,
            len: stream.alloc_zeros(1)?,
            valid: stream.alloc_zeros(1)?,
            ok: stream.alloc_zeros(1)?,
            err: stream.alloc_zeros(1)?,
            pstart: stream.alloc_zeros(1)?,
            plen: stream.alloc_zeros(1)?,
            fstart: stream.alloc_zeros(1)?,
            flen: stream.alloc_zeros(1)?,
            smooth: stream.alloc_zeros(1)?,
            fitted: stream.alloc_zeros(1)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::fields::Presence;
    use crate::finish::CachePools;
    use crate::finish::test_support::store_with;
    use crate::finish::{FinishCtx, NativeSampler};
    use crate::gpu::Stages;
    use crate::kernels::{evaluate_curve, fit_curve};
    use crate::vol::EDGE;

    /// A small xorshift, so the fixtures need no dependency
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        /// Uniform in [0, 1)
        fn unit(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// The tracing shape of the fixtures; the native predictions are twice as large on each axis
    const SHAPE: Shape = [40, 72, 72];

    /// V's lead over H at a tracing-space point: positive and negative bands crossing the volume obliquely, so
    /// curves cross between the families' regions
    fn lead(z: f64, y: f64, x: f64) -> f64 {
        (0.11 * x + 0.05 * y + 0.07 * z).sin() + 0.4 * (0.09 * y - 0.13 * z).cos()
    }

    /// A V and H pair from `lead` with noise, summing to about 255 so presence is high where the family leads
    fn pair(rng: &mut Rng, l: f64) -> [u8; 2] {
        let noise = (rng.next() % 31) as f64 - 15.0;
        let v = (127.5 + 90.0 * l + noise).clamp(0.0, 255.0);
        [v as u8, (255.0 - v + noise / 2.0).clamp(0.0, 255.0) as u8]
    }

    /// Pooled predictions over [`SHAPE`] from `lead`, with an empty corner block that reads as 0 in both
    fn volumes() -> [Vol<u8>; 2] {
        let mut tracing = [Vol::new(SHAPE, 0u8), Vol::new(SHAPE, 0u8)];
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        for z in 0..SHAPE[0] {
            for y in 0..SHAPE[1] {
                for x in 0..SHAPE[2] {
                    if z < 16 && y < 16 && x < 16 {
                        continue;
                    }
                    let [v, h] = pair(&mut rng, lead(z as f64, y as f64, x as f64));
                    tracing[0].set(z, y, x, v);
                    tracing[1].set(z, y, x, h);
                }
            }
        }
        tracing
    }

    /// Random walks at the tracer's step of 0.75 voxels with slowly turning directions, of 1 to 150 points, some
    /// running past the volume's edges, and a few with steps of up to 2 voxels that turn sharply
    fn curves(n: usize) -> Vec<RawCurve> {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        (0..n)
            .map(|c| {
                let len = 1 + (rng.next() % 150) as usize;
                let (step, turn) = if c % 10 == 9 { (2.0, 1.5) } else { (0.75, 0.25) };
                let mut p = [0, 1, 2].map(|k| rng.unit() * SHAPE[2 - k] as f64);
                let mut d = [0, 1, 2].map(|_| rng.unit() - 0.5);
                let mut pts = Vec::with_capacity(len);
                for _ in 0..len {
                    pts.push(p.map(|v| v as f32));
                    for v in &mut d {
                        *v += turn * (rng.unit() - 0.5);
                    }
                    let norm = d.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-9);
                    for k in 0..3 {
                        // the walks may leave the volume by up to 2 voxels, where sampling clamps
                        p[k] = (p[k] + step * d[k] / norm).clamp(-2.0, SHAPE[2 - k] as f64 + 1.0);
                    }
                }
                RawCurve { pts }
            })
            .collect()
    }

    /// Each segment's samples, as `valid_segments` takes them, every one of them
    fn samples(pts: &[[f32; 3]]) -> Vec<[f32; 3]> {
        let mut out = Vec::new();
        for seg in pts.windows(2) {
            let d = [seg[1][0] - seg[0][0], seg[1][1] - seg[0][1], seg[1][2] - seg[0][2]];
            let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let n = ((len / 0.25f32).ceil() as i64).max(1);
            let step = 1.0 / n as f64;
            for j in 0..=n {
                let t = if j == n { 1.0f32 } else { (j as f64 * step) as f32 };
                out.push([0, 1, 2].map(|k| seg[0][k] + d[k] * t));
            }
        }
        out
    }

    /// The native voxels `NativeSampler::sample` reads for a sample at tracing point `q`
    fn corners(shape: Shape, q: [f32; 3]) -> [Shape; 8] {
        let p = q.map(|v| v * 2.0f32 + 0.5f32);
        let p = [p[2], p[1], p[0]];
        let c: [f64; 3] = [0, 1, 2].map(|a| {
            let v = p[a].max(0.0);
            let hi = (shape[a] - 1) as f64;
            if hi < v as f64 { hi } else { v as f64 }
        });
        let lo = c.map(|v| v.floor() as i64);
        std::array::from_fn(|corner| {
            let o = [(corner >> 2) & 1, (corner >> 1) & 1, corner & 1];
            [0, 1, 2].map(|a| ((lo[a] + o[a] as i64) as usize).min(shape[a] - 1))
        })
    }

    /// Pieces of at least 4 points a curve may be cut into: the whole curve, and a few runs inside it
    fn runs(rng: &mut Rng, n: usize) -> Vec<RangeInclusive<usize>> {
        if n < MIN_FIT_POINTS {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(4);
        out.push(0..=n - 1);
        for _ in 0..3 {
            let a = (rng.next() as usize) % (n - MIN_FIT_POINTS + 1);
            let b = a + MIN_FIT_POINTS - 1 + (rng.next() as usize) % (n - a - MIN_FIT_POINTS + 1);
            out.push(a..=b);
        }
        out
    }

    /// The kernels' flags are read at each curve's own point slots, and a curve's last slot, which starts no
    /// segment, is ignored whatever it holds
    #[test]
    fn cuts_read_each_curves_flags_at_its_slots() {
        let curve = |n: usize| RawCurve { pts: vec![[0.0; 3]; n] };
        let batch = [curve(4), curve(1), curve(5), curve(3)];
        let start = [0, 4, 5, 10, 13];
        let valid = [1, 0, 1, 1, /* one point */ 1, /* five */ 1, 1, 0, 1, 1, /* three */ 0, 0, 1];
        let cuts = cut(&batch, &start, &valid);
        assert_eq!(cuts, [vec![0..=1, 2..=3], vec![], vec![0..=2, 3..=4], vec![]]);
    }

    /// The slabs listed for a curve hold every native voxel any sample of its raw segments reads, and any sample
    /// of the fitted segments of every piece it might be cut into, on slabs small enough that a missed margin
    /// shows
    #[test]
    fn listed_slabs_cover_every_voxel_a_sample_reads() {
        let tracing = volumes();
        let layout = SlabLayout { shape: SHAPE.map(|n| 2 * n), edge: [16, 16, 16], planes: 8 };
        let mut rng = Rng(7);
        let mut checked = 0;
        for family in [1, 2] {
            let pres = Presence::new(&tracing, family);
            for c in curves(300) {
                let listed = curve_slabs(&layout, &c.pts);
                let mut reads = samples(&c.pts);
                for r in runs(&mut rng, c.pts.len()) {
                    let (smoothed, n_out) = fit_curve(&c.pts[r], &pres);
                    reads.extend(samples(&evaluate_curve(&smoothed, n_out, SHAPE)));
                }
                for q in reads {
                    for [z, y, x] in corners(layout.shape, q) {
                        let [ez, ey, ex] = layout.edge;
                        let i = layout.index([z / ez, y / ey, x / ex, z % ez / layout.planes]) as u32;
                        assert!(listed.binary_search(&i).is_ok(), "voxel {z} {y} {x} of {q:?} is in no listed slab");
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 100_000, "{checked} voxels checked, too few to test");
    }

    /// Every piece's fitted points fit the room the host gives it, including on the sharply turning curves where
    /// fitting moves points furthest
    #[test]
    fn fitted_pieces_fit_their_room() {
        let tracing = volumes();
        let mut rng = Rng(11);
        let mut checked = 0;
        for family in [1, 2] {
            let pres = Presence::new(&tracing, family);
            for c in curves(300) {
                for r in runs(&mut rng, c.pts.len()) {
                    let piece = &c.pts[r];
                    let (_, n_out) = fit_curve(piece, &pres);
                    let room = fitted_room(piece);
                    assert!(n_out <= room, "{n_out} fitted points of a {}-point piece in room for {room}", piece.len());
                    checked += 1;
                }
            }
        }
        assert!(checked > 1000, "{checked} pieces checked, too few to test");
    }

    /// The slab cache takes no more slots than the native predictions have slabs however much is free, and the
    /// reserve for a run's batches (of about 2 Mi raw points) leaves slots on a GPU with under 1 GiB free
    #[test]
    fn slab_slots_fit_the_predictions_and_the_device() {
        const GIB: usize = 1 << 30;
        let reserve = (2 << 20) * BATCH_BYTES_PER_POINT;
        // a 64³ volume in one 512 KiB slab
        let small = SlabLayout { shape: [64; 3], edge: [64; 3], planes: 64 };
        assert_eq!((small.len(), small.slab_bytes()), (1, 512 << 10));
        assert_eq!(slab_slots(12 * GIB, reserve, SLAB_CACHE_MAX, small.slab_bytes(), small.len()), 1);

        // 128³ chunks in slabs of 8 planes, over a band larger than the cache
        let band = SlabLayout { shape: [512, 4096, 4096], edge: [128; 3], planes: 8 };
        let bytes = band.slab_bytes();
        let low = slab_slots(768 << 20, reserve, SLAB_CACHE_MAX, bytes, band.len());
        assert!(low * bytes > 256 << 20, "{low} slots in 768 MiB free");
        let high = slab_slots(24 * GIB, reserve, SLAB_CACHE_MAX, bytes, band.len());
        assert_eq!(high * bytes, SLAB_CACHE_MAX);
        assert_eq!(slab_slots(reserve / 2, reserve, SLAB_CACHE_MAX, bytes, band.len()), 0);

        // 8³ chunks of a 4096³ volume: 128 Mi slabs of 1 KiB, whose 512 MiB table must fit beside the pool
        let fine = SlabLayout { shape: [4096; 3], edge: [8; 3], planes: 8 };
        assert_eq!((fine.len(), fine.slab_bytes()), (1 << 27, 1 << 10));
        let slots = slab_slots(GIB, reserve, SLAB_CACHE_MAX, fine.slab_bytes(), fine.len());
        assert!(slots > 0);
        let table = fine.len() * size_of::<u32>();
        assert!(slots * fine.slab_bytes() + table + reserve <= GIB, "{slots} slots and a {table}-byte table in 1 GiB");
    }

    /// The kernels index a slab of up to u32::MAX bytes per channel, so a uint8 chunk of 1024 × 2048 × 2048 decoded
    /// whole, 4 GiB a channel, goes to the CPU, while the same chunk decoded by planes stays on the GPU; and a
    /// pooled brick grid of up to 2³² bricks, so one plane of bricks more than 1024 × 2048 × 2048 goes to the CPU
    /// although every axis fits
    #[test]
    fn kernels_index_within_u32() {
        let whole = SlabLayout { shape: [1024, 2048, 2048], edge: [1024, 2048, 2048], planes: 1024 };
        assert_eq!(whole.slab_bytes(), 8 << 30);
        assert!(!kernels_index(&whole));
        assert!(kernels_index(&SlabLayout { planes: 8, ..whole }));

        // 65535 × 65537 is u32::MAX
        let edge = SlabLayout { shape: [1, 65535, 65537], edge: [1, 65535, 65537], planes: 1 };
        assert!(kernels_index(&edge));
        assert!(!kernels_index(&SlabLayout { shape: [1, 65535, 65538], edge: [1, 65535, 65538], ..edge }));

        let grid = |bz: usize| [bz, 2048, 2048];
        let shape = |bz: usize| grid(bz).map(|n| n * EDGE);
        assert!(kernels_index_pooled(shape(1024), grid(1024)));
        // float32 rounds 32768.999 up to 32769, the last voxel of an axis of 32770
        assert!(((32770 - 1) as f64 - 0.001) as f32 == 32769.0);
        assert!(!kernels_index_pooled([16, 16, 32770], [1, 1, 2049]));
        assert!(!kernels_index_pooled(shape(1025), grid(1025)));
        assert!(!kernels_index_pooled([1 << 32, 16, 16], [1 << 28, 1, 1]));
    }

    /// Bit patterns of each curve's pieces, so 0.0 and -0.0 differ
    fn bits(pieces: &[Pieces]) -> Vec<Vec<Vec<[u32; 3]>>> {
        pieces.iter().map(|c| c.iter().map(|p| p.iter().map(|q| q.map(f32::to_bits)).collect()).collect()).collect()
    }

    /// The GPU's pieces against `FinishCtx::finish`, bit for bit, for both families, over native stores decoded
    /// by slabs and by whole chunks (whose chunks at the array's edge are clipped), through a slab cache small
    /// enough that later batches evict earlier ones' slabs; a batch reading more slabs than the cache holds comes
    /// back for the CPU
    #[test]
    #[ignore = "needs an NVIDIA GPU; run through the GPU queue"]
    fn gpu_finish_matches_cpu() {
        const BATCH: usize = 10;
        let gpu = Gpu::open(None, Stages::DirectionField).unwrap();
        let tracing = volumes();
        let native = SHAPE.map(|n| 2 * n);
        let edge = 32;
        let value = |c: usize, z: usize, y: usize, x: usize| {
            // a hash of the voxel for noise, so V and H disagree near the bands' edges
            let mut rng = Rng(((z * 7919 + y) * 104_729 + x) as u64 * 2 + 1);
            rng.next();
            let at = |v: usize| (v as f64 - 0.5) / 2.0;
            pair(&mut rng, lead(at(z), at(y), at(x)))[c]
        };
        // batches of curves that start near each other, as the tracer's are, so each reads a part of the store
        let mut batch = curves(600);
        batch.sort_by(|a, b| a.pts[0][0].total_cmp(&b.pts[0][0]));
        let workers = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        let blocks = [("finish-sliced", 8 * edge * edge), ("finish-whole", 2 * edge * edge * edge)];
        for (name, blocksize) in blocks {
            let (dir, pred) = store_with(name, native, edge, |_| blocksize, value);
            let pools = CachePools::new();
            let cache = NativeCache::new(&pred, 64 << 20, &pools);
            let layout = cache.layout();
            let per_batch = batch.chunks(BATCH).map(|b| list_slabs(&layout, b).len()).max().unwrap();
            let all = list_slabs(&layout, &batch).len();
            let room = per_batch + per_batch / 4;
            assert!(all > room, "{name}: {all} slabs in all, {per_batch} in a batch; too few to evict");
            for family in [1, 2] {
                let pres = Presence::new(&tracing, family);
                let ctx = FinishCtx { tracing: &tracing, pres: &pres, family };
                let mut sampler = NativeSampler::new(&cache);
                let want: Vec<Pieces> = batch.iter().map(|c| ctx.finish(&c.pts, &mut sampler).unwrap()).collect();
                let pieces = want.iter().map(Vec::len).sum::<usize>();
                assert!(pieces > batch.len(), "{name} {family}: {pieces} pieces of {} curves", batch.len());

                // the fixture's curves hold 150 points at most
                let (points, cache_bytes) = (BATCH * 150, room * layout.slab_bytes());
                let mut finisher = gpu.finisher_with(&tracing, family, layout, points, cache_bytes).unwrap();
                assert!(finisher.slots() < all, "{name} {family}: the cache holds every slab");
                for (i, (curves, want)) in batch.chunks(BATCH).zip(want.chunks(BATCH)).enumerate() {
                    match finisher.finish(&workers, &cache, curves).unwrap() {
                        Finished::Pieces(got) => {
                            assert!(bits(&got) == bits(want), "{name} {family}: batch {i} differs")
                        }
                        Finished::OnCpu(why) => panic!("{name} {family}: batch {i} came back for the CPU ({why:?})"),
                    }
                }
                assert!(finisher.uploaded() > finisher.slots() as u64, "{name} {family}: no slab was evicted");
                assert!(matches!(
                    finisher.finish(&workers, &cache, &batch).unwrap(),
                    Finished::OnCpu(CpuReason::Slabs)
                ));
            }
            let _ = fs::remove_dir_all(&dir);
        }
    }
}
