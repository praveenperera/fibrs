//! The direction field on the GPU, bit for bit `axes::direction_field`
//!
//! The tiles are the CPU's: 64³ cores with an 11-voxel halo, every filter run over the core's box (core and halo,
//! clipped to the volume) with the CPU's edge rules and roundings, so every value the kernels compute is the CPU's.
//! Tiles whose core holds a brick of either pooled prediction are grouped into units of up to [`AXES_UNIT_TILES`]
//! adjacent along x. A unit uploads its box of bricks of both predictions and runs gather, which lists each family's
//! present voxels, then for each family three smoothing passes, the gradient, two passes smoothing the gradient
//! products and the eigen kernel. That kernel smooths the last axis at each present voxel and certifies the voxel's
//! quantized direction with a Jacobi solve and an error bound (`fibrs-kernels/src/eigen.rs`). Voxels it cannot
//! certify come back as their tensors, and the host solves them with `dsyevd` as the CPU path does. Two units are
//! in flight at a time on two streams, as for centrality

use std::sync::Arc;

use color_eyre::eyre::{Result, WrapErr, ensure, eyre};
use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, DevicePtrMut, LaunchConfig, PinnedHostSlice, PushKernelArg, result,
};
use rayon::prelude::*;

use super::{
    BLOCK, DESC, Gpu, Inputs, TileSpan, brick_box, core_bricks, fit, fit_host, fit_host_upto, grid_1d, group, pinned,
    region,
};
use crate::axes::{CORE, HALO, Lapack};
use crate::fields::tiles;
use crate::gpu::AxesCounts;
use crate::vol::{BRICK, EDGE, Shape, Vol};

/// Most tiles in one unit, however much device memory is free
const AXES_UNIT_TILES: usize = 32;
/// Most fallback records copied back and solved at once, so a unit where most voxels fall back (a volume of flat
/// predictions) holds 32 MiB of pinned memory per slot instead of 32 bytes per voxel
const FB_CHUNK: usize = 1 << 20;
/// Most device memory one tile can need in a slot, `fit`'s spare half included: the brick pool and tables of both
/// pooled predictions over its box's bricks, the predictions and twelve float32 channels over its box, and over its
/// core the presence lists, the fallback records (which have room for every core voxel) and both output arrays;
/// about 68 MB for an interior tile
pub(super) const AXES_TILE_DEVICE_BYTES: usize = {
    let side = CORE + 2 * HALO;
    let bricks = (side.div_ceil(EDGE) + 1).pow(3);
    let core = CORE.pow(3);
    let bytes = 2 * bricks * (BRICK + 4) + (2 + 12 * 4) * side.pow(3) + (2 * 4 + RECORD * 4 + 2) * core;
    bytes + bytes / 2
};
/// u32 per fallback record: the tile in its unit, the voxel's index in the tile's core, and the bits of the six
/// float32 tensor values
const RECORD: usize = 8;

impl Gpu {
    /// Most direction field tiles per unit, so both slots' buffers fit in half the device memory free now
    fn axes_unit_tiles(&self) -> Result<usize> {
        let (free, _) = self.ctx.mem_get_info()?;
        Ok((free / 2 / (2 * AXES_TILE_DEVICE_BYTES)).clamp(1, AXES_UNIT_TILES))
    }

    /// The direction field (nx, ny), equal to `axes::direction_field`, and how many present voxels the host
    /// solved because the kernels could not certify them
    pub fn direction_field(&self, tracing: &[Vol<u8>; 2], lapack: &Lapack) -> Result<(Vol<u8>, Vol<u8>, AxesCounts)> {
        let shape = tracing[0].shape();
        ensure!(tracing[1].shape() == shape, "the pooled predictions differ in shape");
        // the kernels' default for bricks without a payload
        ensure!(tracing.iter().all(|v| v.default_value() == 0), "unexpected pooled prediction defaults");
        ensure!(shape.iter().all(|&n| u32::try_from(n).is_ok()), "volume of {shape:?} voxels is too large");
        let mut field = Field { nx: Vol::new(shape, 128u8), ny: Vol::new(shape, 128u8), counts: AxesCounts::default() };
        let mut slots = [Slot::new(&self.ctx)?, Slot::new(&self.ctx)?];
        // unit i stages and launches on slot i % 2 while unit i - 1 finishes on the other; a slot's previous
        // unit was collected (its stream synchronized) before the slot stages again
        let mut pending: Option<(usize, Meta)> = None;
        for (i, unit) in units(tracing, shape, self.axes_unit_tiles()?).iter().enumerate() {
            let s = i % 2;
            let meta = slots[s].stage(unit, tracing, shape)?;
            slots[s].launch(self, &meta)?;
            if let Some((p, m)) = pending.replace((s, meta)) {
                slots[p].collect(&m, lapack, &mut field)?;
            }
        }
        if let Some((p, m)) = pending {
            slots[p].collect(&m, lapack, &mut field)?;
        }
        Ok((field.nx, field.ny, field.counts))
    }
}

/// The CPU's tiles in its order, grouped into units of up to `unit_tiles` adjacent tiles along x
///
/// A tile whose core holds no brick of either pooled prediction has no present voxel, so the CPU leaves its core
/// at the default and so does this, by leaving it out
fn units(tracing: &[Vol<u8>; 2], shape: Shape, unit_tiles: usize) -> Vec<Vec<TileSpan>> {
    let all = tiles(shape, CORE);
    let keep: Vec<bool> = all
        .par_iter()
        .map(|&(start, end)| {
            let dims = [0, 1, 2].map(|a| end[a] - start[a]);
            tracing.iter().any(|v| v.any_allocated_in(start, dims, |_, _, _| true))
        })
        .collect();
    group(&all, &keep, unit_tiles)
}

/// The direction field as units finish, and the counts so far
struct Field {
    nx: Vol<u8>,
    ny: Vol<u8>,
    counts: AxesCounts,
}

/// What a staged unit's launches and collection need besides its device buffers
struct Meta {
    /// The unit's box of bricks: first brick and extent, z, y, x
    b0: Shape,
    nb: Shape,
    /// Each tile's core origin and extent, which place its fallback voxels
    cores: Vec<(Shape, Shape)>,
    /// Voxels of every tile's box: the length of one channel of the scratch arrays
    total: usize,
    /// The largest box of any one tile, which sizes the per-tile launches
    max_box: usize,
    /// The largest core of any one tile: the room in each presence list
    cap: usize,
    /// Room in the fallback list: every core voxel of the unit, so it cannot overflow
    fb_cap: usize,
    /// Each output brick's index in the volume's brick grid, in the order the tiles' output bricks lie on device
    bricks: Vec<usize>,
}

/// One unit in flight: a stream with the device buffers and pinned host buffers it uses, grown as units need
struct Slot {
    stream: Arc<CudaStream>,
    inputs: Inputs,
    /// Both pooled predictions over every tile's box, channel-major
    raw: CudaSlice<u8>,
    /// Present voxels per tile and family (`2 t + f`), and their core indices, `cap` per list
    count: CudaSlice<u32>,
    list: CudaSlice<u32>,
    /// Six channels of `total`: the gradient products smoothed along z
    p: CudaSlice<f32>,
    /// Six channels of `total`: first the smoothed block (channel 0), the pass between its axes (1) and the
    /// gradients (2 to 4), which are all read before the tensor smoothed along z and y overwrites them
    q: CudaSlice<f32>,
    out_nx: CudaSlice<u8>,
    out_ny: CudaSlice<u8>,
    fb: CudaSlice<u32>,
    fb_count: CudaSlice<u32>,
    /// The counts, then the fallback count
    host_count: PinnedHostSlice<u32>,
    host_nx: PinnedHostSlice<u8>,
    host_ny: PinnedHostSlice<u8>,
    host_fb: PinnedHostSlice<u32>,
}

/// Set the first `n` bytes of `buf` to `value`, in the stream's order
fn fill(stream: &Arc<CudaStream>, buf: &mut CudaSlice<u8>, n: usize, value: u8) -> Result<()> {
    stream.context().bind_to_thread()?;
    let mut view = buf.slice_mut(..n);
    let (ptr, _record) = view.device_ptr_mut(stream);
    // SAFETY: `ptr` addresses `n` bytes of a live allocation made on `stream`, any byte is a valid u8, and
    // `device_ptr_mut` made the stream wait for the buffer's earlier work and records this write when dropped
    unsafe { result::memset_d8_async(ptr, value, n, stream.cu_stream()) }?;
    Ok(())
}

impl Slot {
    fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        let stream = ctx.new_stream()?;
        Ok(Self {
            inputs: Inputs::new(&stream)?,
            host_count: pinned(ctx, 1, 0)?,
            host_nx: pinned(ctx, 1, 0)?,
            host_ny: pinned(ctx, 1, 0)?,
            host_fb: pinned(ctx, 1, 0)?,
            raw: stream.alloc_zeros(1)?,
            count: stream.alloc_zeros(1)?,
            list: stream.alloc_zeros(1)?,
            p: stream.alloc_zeros(1)?,
            q: stream.alloc_zeros(1)?,
            out_nx: stream.alloc_zeros(1)?,
            out_ny: stream.alloc_zeros(1)?,
            fb: stream.alloc_zeros(1)?,
            fb_count: stream.alloc_zeros(1)?,
            stream,
        })
    }

    /// Build a unit's brick pool, tables and tile descriptors and start their upload
    fn stage(&mut self, unit: &[TileSpan], tracing: &[Vol<u8>; 2], shape: Shape) -> Result<Meta> {
        let grid = tracing[0].grid();
        let boxes: Vec<(Shape, Shape)> = unit.iter().map(|&t| region(t, shape, HALO)).collect();
        let mut desc = Vec::with_capacity(unit.len() * DESC);
        let mut cores = Vec::with_capacity(unit.len());
        let mut bricks = Vec::new();
        let (mut total, mut max_box, mut cap, mut fb_cap) = (0usize, 0usize, 0usize, 0usize);
        for (&(start, end), &(lo, dims)) in unit.iter().zip(&boxes) {
            let off = [0, 1, 2].map(|a| start[a] - lo[a]);
            let core = [0, 1, 2].map(|a| end[a] - start[a]);
            let fields = [lo, dims, off, core].concat();
            desc.extend(fields.iter().map(|&v| v as u32));
            desc.extend([total as u32, (bricks.len() * BRICK) as u32, 0, 0]);
            core_bricks(start, core, grid, &mut bricks);
            cores.push((start, core));
            let box_voxels = dims.iter().product::<usize>();
            let core_voxels = core.iter().product::<usize>();
            total += box_voxels;
            max_box = max_box.max(box_voxels);
            cap = cap.max(core_voxels);
            fb_cap += core_voxels;
        }
        // the kernels index the six-channel scratch arrays, the lists, the fallback records and the output in u32
        let fits = |n: usize| u32::try_from(n).is_ok();
        ensure!(
            fits(6 * total) && fits(2 * unit.len() * cap) && fits(RECORD * fb_cap) && fits(bricks.len() * BRICK),
            "a unit is too large for the kernels' indices"
        );
        let (b0, nb) = brick_box(&boxes);
        let vols = [&tracing[0], &tracing[1]];
        self.inputs.upload(&self.stream, &vols, (b0, nb), desc)?;
        Ok(Meta { b0, nb, cores, total, max_box, cap, fb_cap, bricks })
    }

    /// Queue a staged unit's kernels and the copies of its counts and output bricks
    fn launch(&mut self, gpu: &Gpu, m: &Meta) -> Result<()> {
        let stream = self.stream.clone();
        let t = m.total;
        let n_count = 2 * m.cores.len();
        let n_out = m.bricks.len() * BRICK;
        fit(&stream, &mut self.raw, 2 * t)?;
        fit(&stream, &mut self.count, n_count)?;
        fit(&stream, &mut self.list, n_count * m.cap)?;
        fit(&stream, &mut self.p, 6 * t)?;
        fit(&stream, &mut self.q, 6 * t)?;
        fit(&stream, &mut self.out_nx, n_out)?;
        fit(&stream, &mut self.out_ny, n_out)?;
        fit(&stream, &mut self.fb, RECORD * m.fb_cap)?;
        // grown before the launches, as growing waits for the stream
        fit_host(&stream, &mut self.host_count, n_count + 1, 0)?;
        fit_host(&stream, &mut self.host_nx, n_out, 0)?;
        fit_host(&stream, &mut self.host_ny, n_out, 0)?;
        stream.memset_zeros(&mut self.count.slice_mut(..n_count))?;
        stream.memset_zeros(&mut self.fb_count)?;
        // the voxels the eigen kernel does not write, absent or past a core's edge, keep the default
        fill(&stream, &mut self.out_nx, n_out, 128)?;
        fill(&stream, &mut self.out_ny, n_out, 128)?;

        let tiles = m.cores.len() as u32;
        let per_tile = |n: usize| -> Result<LaunchConfig> {
            Ok(LaunchConfig { grid_dim: (grid_1d(n)?, tiles, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 })
        };
        let nb_total = m.nb.iter().product::<usize>() as u32;
        let [bz0, by0, bx0] = m.b0.map(|v| v as u32);
        let (nby, nbx) = (m.nb[1] as u32, m.nb[2] as u32);
        let (total, cap, fb_cap) = (t as u32, m.cap as u32, m.fb_cap as u32);
        let (axis_y, axis_x) = (1u32, 2u32);
        let k = &gpu.k;
        let inputs = &self.inputs;
        // SAFETY: each launch passes the kernel's params in its PTX order (checked in `Gpu::open`), every buffer
        // holds what the kernel indexes (sized above from the descriptors), and all run in order on this stream
        unsafe {
            stream
                .launch_builder(&k.ax_gather)
                .arg(&inputs.desc)
                .arg(&inputs.pool)
                .arg(&inputs.table)
                .arg(&nb_total)
                .arg(&bz0)
                .arg(&by0)
                .arg(&bx0)
                .arg(&nby)
                .arg(&nbx)
                .arg(&total)
                .arg(&cap)
                .arg(&mut self.raw)
                .arg(&mut self.count)
                .arg(&mut self.list)
                .launch(per_tile(m.max_box)?)?;
        }
        for family in [0u32, 1] {
            // the smoothed block, the pass between its axes and the gradients live in q until tensor_y
            // overwrites it, so the scratch is twelve channels rather than seventeen
            let (mut s, mut rest) = self.q.split_at_mut(t);
            let (mut tmp, mut g) = rest.split_at_mut(t);
            // SAFETY: as above; s, tmp and g are disjoint views of q
            unsafe {
                stream
                    .launch_builder(&k.ax_smooth0)
                    .arg(&inputs.desc)
                    .arg(&self.count)
                    .arg(&family)
                    .arg(&total)
                    .arg(&self.raw)
                    .arg(&gpu.unit)
                    .arg(&mut s)
                    .launch(per_tile(m.max_box)?)?;
                stream
                    .launch_builder(&k.ax_smooth)
                    .arg(&inputs.desc)
                    .arg(&self.count)
                    .arg(&family)
                    .arg(&axis_y)
                    .arg(&s.slice(..))
                    .arg(&mut tmp)
                    .launch(per_tile(m.max_box)?)?;
                stream
                    .launch_builder(&k.ax_smooth)
                    .arg(&inputs.desc)
                    .arg(&self.count)
                    .arg(&family)
                    .arg(&axis_x)
                    .arg(&tmp.slice(..))
                    .arg(&mut s)
                    .launch(per_tile(m.max_box)?)?;
                stream
                    .launch_builder(&k.ax_grad)
                    .arg(&inputs.desc)
                    .arg(&self.count)
                    .arg(&family)
                    .arg(&total)
                    .arg(&s.slice(..))
                    .arg(&mut g)
                    .launch(per_tile(m.max_box)?)?;
                stream
                    .launch_builder(&k.ax_tensor_z)
                    .arg(&inputs.desc)
                    .arg(&self.count)
                    .arg(&family)
                    .arg(&total)
                    .arg(&g.slice(..))
                    .arg(&mut self.p)
                    .launch(per_tile(m.max_box)?)?;
            }
            // SAFETY: as above
            unsafe {
                stream
                    .launch_builder(&k.ax_tensor_y)
                    .arg(&inputs.desc)
                    .arg(&self.count)
                    .arg(&family)
                    .arg(&total)
                    .arg(&self.p)
                    .arg(&mut self.q)
                    .launch(per_tile(m.max_box)?)?;
                stream
                    .launch_builder(&k.ax_eigen)
                    .arg(&inputs.desc)
                    .arg(&self.count)
                    .arg(&family)
                    .arg(&total)
                    .arg(&cap)
                    .arg(&self.list)
                    .arg(&self.q)
                    .arg(&mut self.out_nx)
                    .arg(&mut self.out_ny)
                    .arg(&mut self.fb)
                    .arg(&fb_cap)
                    .arg(&mut self.fb_count)
                    .launch(per_tile(m.cap)?)?;
            }
        }

        let host_count = self.host_count.as_mut_slice()?;
        stream.memcpy_dtoh(&self.count.slice(..n_count), &mut host_count[..n_count])?;
        stream.memcpy_dtoh(&self.fb_count, &mut host_count[n_count..n_count + 1])?;
        stream.memcpy_dtoh(&self.out_nx.slice(..n_out), &mut self.host_nx.as_mut_slice()?[..n_out])?;
        stream.memcpy_dtoh(&self.out_ny.slice(..n_out), &mut self.host_ny.as_mut_slice()?[..n_out])?;
        Ok(())
    }

    /// Wait for a launched unit, then install its direction bricks and solve and install its fallback voxels
    fn collect(&mut self, m: &Meta, lapack: &Lapack, field: &mut Field) -> Result<()> {
        self.stream.synchronize().wrap_err("run the direction field kernels")?;
        let n_count = 2 * m.cores.len();
        let counts = &self.host_count.as_slice()?[..n_count + 1];
        for (t, (&[f0, f1], &(_, core))) in counts[..n_count].as_chunks::<2>().0.iter().zip(&m.cores).enumerate() {
            let core_voxels = core.iter().product::<usize>();
            let present = f0 as usize + f1 as usize;
            ensure!(present <= core_voxels, "tile {t} lists {present} present voxels of {core_voxels}");
        }
        let present = counts[..n_count].iter().map(|&n| n as usize).sum::<usize>();
        let n_fb = counts[n_count] as usize;
        ensure!(n_fb <= m.fb_cap, "{n_fb} fallbacks overflow a unit's list of {}", m.fb_cap);
        ensure!(n_fb <= present, "{n_fb} fallbacks of {present} present voxels");

        let n_out = m.bricks.len() * BRICK;
        let nx = self.host_nx.as_slice()?[..n_out].as_chunks::<BRICK>().0;
        let ny = self.host_ny.as_slice()?[..n_out].as_chunks::<BRICK>().0;
        for ((&bi, a), b) in m.bricks.iter().zip(nx).zip(ny) {
            field.nx.set_brick(bi, a);
            field.ny.set_brick(bi, b);
        }

        // the voxels the kernels could not certify, finished as the CPU path finishes every voxel
        for start in (0..n_fb).step_by(FB_CHUNK) {
            let n_records = RECORD * FB_CHUNK.min(n_fb - start);
            let from = RECORD * start;
            fit_host_upto(&self.stream, &mut self.host_fb, n_records, RECORD * FB_CHUNK, 0)?;
            let records = &mut self.host_fb.as_mut_slice()?[..n_records];
            self.stream.memcpy_dtoh(&self.fb.slice(from..from + n_records), records)?;
            self.stream.synchronize().wrap_err("copy the fallback voxels back")?;

            let solved: Vec<(Shape, (u8, u8))> = records
                .par_chunks_exact(RECORD)
                .map_init(|| (Vec::new(), Vec::new()), |(work, iwork), r| fallback(r, m, lapack, work, iwork))
                .collect::<Result<_>>()?;
            for ([z, y, x], (a, b)) in solved {
                field.nx.set(z, y, x, a);
                field.ny.set(z, y, x, b);
            }
        }
        field.counts.present += present;
        field.counts.fallback += n_fb;
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

/// A fallback record's voxel and its direction from `dsyevd`
fn fallback(
    r: &[u32],
    m: &Meta,
    lapack: &Lapack,
    work: &mut Vec<f64>,
    iwork: &mut Vec<i64>,
) -> Result<(Shape, (u8, u8))> {
    let tile = r[0] as usize;
    let &(start, core) = m.cores.get(tile).ok_or_else(|| eyre!("a fallback names tile {tile} of {}", m.cores.len()))?;
    let local = r[1] as usize;
    let core_voxels = core.iter().product::<usize>();
    ensure!(local < core_voxels, "a fallback names voxel {local} of tile {tile}'s {core_voxels}");
    let at = [start[0] + local / (core[1] * core[2]), start[1] + local / core[2] % core[1], start[2] + local % core[2]];
    let t: [f32; 6] = std::array::from_fn(|k| f32::from_bits(r[2 + k]));
    // the channels are the upper triangle a00, a01, a02, a11, a12, a22, axes z, y, x
    let full = [t[0], t[1], t[2], t[1], t[3], t[4], t[2], t[4], t[5]];
    Ok((at, lapack.direction(full, work, iwork)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::axes::{direction_field, find_lapack};
    use crate::gpu::Stages;

    /// A small xorshift, so the test volume needs no dependency
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    /// Pooled predictions with ridges in both families and noise over a shape whose tiles end mid-brick on every
    /// axis, with an empty tile row and a saturated block wider than the halo, whose zero tensors the kernels
    /// cannot certify
    fn volumes() -> [Vol<u8>; 2] {
        let shape = [150, 200, 270];
        let mut tracing = [Vol::new(shape, 0u8), Vol::new(shape, 0u8)];
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for z in 0..shape[0] {
            for y in 0..shape[1] {
                for x in 0..shape[2] {
                    let noise = (rng.next() % 41) as f64 - 20.0;
                    // tiles z 0, y 1 hold no prediction, so the GPU path leaves them out
                    if z < 64 && (64..128).contains(&y) {
                        continue;
                    }
                    if (70..120).contains(&z) && (20..70).contains(&y) && (100..160).contains(&x) {
                        tracing[0].set(z, y, x, 255);
                        tracing[1].set(z, y, x, 255);
                        continue;
                    }
                    let (zf, yf, xf) = (z as f64, y as f64, x as f64);
                    let v = 255.0 * (0.5 + 0.5 * (0.31 * xf + 0.07 * yf + 0.05 * zf).sin()).powi(3) + noise;
                    let h = 255.0 * (0.5 + 0.5 * (0.29 * yf - 0.04 * xf + 0.11 * zf).sin()).powi(3) + noise;
                    tracing[0].set(z, y, x, v.clamp(0.0, 255.0) as u8);
                    tracing[1].set(z, y, x, h.clamp(0.0, 255.0) as u8);
                }
            }
        }
        tracing
    }

    /// The kernels and their fallbacks against `axes::direction_field`: every nx and ny voxel
    #[test]
    #[ignore = "needs an NVIDIA GPU and numpy's OpenBLAS (FIBRS_LAPACK); run through the GPU queue"]
    fn gpu_direction_matches_cpu() {
        let gpu = Gpu::open(None, Stages::DirectionField).unwrap();
        let lapack = Lapack::init(&find_lapack().unwrap()).unwrap();
        let tracing = volumes();
        let shape = tracing[0].shape();
        let (nx, ny, counts) = gpu.direction_field(&tracing, lapack).unwrap();
        let (want_nx, want_ny) = direction_field(&tracing, lapack).unwrap();
        for (name, got, want) in [("nx", &nx, &want_nx), ("ny", &ny, &want_ny)] {
            let (a, b) = (got.read_block([0; 3], shape), want.read_block([0; 3], shape));
            let differ = a.iter().zip(&b).filter(|(a, b)| a != b).count();
            assert_eq!(differ, 0, "{name}: {differ} voxels differ");
            let set = b.iter().filter(|&&v| v != 128).count();
            assert!(set > 1_000_000, "{name}: {set} voxels away from the default, too few to test");
        }

        let (c0, c1) = (tracing[0].read_block([0; 3], shape), tracing[1].read_block([0; 3], shape));
        let present = c0.iter().zip(&c1).filter(|&(&a, &b)| a as u16 + b as u16 >= 40).count();
        assert_eq!(counts.present, present);
        assert!(counts.fallback > 0, "no voxel fell back to dsyevd");
    }
}
