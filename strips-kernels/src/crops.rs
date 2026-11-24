//! Oriented crops of the GPU-resident CT, their per-crop z-score, and the tube channel (`sc_`)
//!
//! The CT lives in a device pool of 128^3 uint8 chunks (2 MiB slots) indexed by a dense table over the volume's
//! chunk grid: the entry of chunk `(cz, cy, cx)` at `(cz * gy + cy) * gx + cx` is its slot, [`FILL`] for a chunk the
//! source holds none of (every voxel the fill value), or [`ABSENT`] for one the host did not make resident, which
//! samples as NaN so a residency bug fails the parity gates instead of reading stale bytes.
//!
//! A crop is vc_reader's affine crop as `CudaReader.read_affine_batch(order=1)` samples it
//! (`reader_cuda/kernels.cu` `sample_affine`, `sampling_math.hpp`): sample `(d, h, w)` lies at CT zyx
//! `base + c`, `c[a] = ((o[a] + A[a][0]·d) + A[a][1]·h) + A[a][2]·w` with every product and sum rounded on its own
//! (no FMA in the coordinate), the integer `base` and float32 `o` split on the host. Points with any
//! `c[a] <= -1 - base[a]` or `c[a] >= shape[a] - base[a]` read the fill value; the rest are trilinear over the 8
//! corners, corners outside the volume reading the fill value. The weights and their sum follow vc_reader's
//! `trilinear`: weight `k` (bit 0 z, bit 1 y, bit 2 x) is `(wz · wy) · wx`, summed in corner order as nvcc
//! contracts `v += w * value`, one FMA per corner. vc_reader's generic path for cells with a zero fraction skips
//! those axes; their weights are exactly 1 and 0 here, so the sum is the same.
//!
//! The model input is that value times float32(1/255), z-scored per crop in float64 (villa
//! `zscore_crops_float64`): `(x - mean) · (1 / max(std, 1e-6))` rounded to float32. [`sc_stats`] samples every
//! crop once for its sum and sum of squares (float32 pairs per thread, float64 across the block, partials in a
//! fixed order), [`sc_finalize`] turns each crop's partials into its mean and reciprocal deviation as float32
//! pairs, and [`sc_write`] samples again and writes the z-scored value: the frame model's `(B, D, H, W)` input. The
//! strip model's NDHWC `(B, D, H, W, 2)` float32 input takes one sampling pass instead: [`sc_raw`] writes the raw
//! `x` as channel 0 (channel 1 zeroed for [`sc_tube`]) beside [`sc_stats`]' partials, and the crop's readers
//! apply [`zscore`] as they load it, to the bits [`sc_write`] would have written. [`sc_tube`] then renders villa's
//! `render_tube` into channel 1
//!
//! Index math is 32-bit per crop; the host keeps a crop under 2^32 samples and the pool's offsets in 64 bits

use core::ptr::addr_of_mut;

use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::float::{
    add_rn_f32, add_rn_f64, div_rn_f32, div_rn_f64, fma_rn_f32, mul_rn_f32, mul_rn_f64, sqrt_rn_f64,
};
use cuda_device::thread::{__unroll_config, blockDim_x, blockIdx_x, blockIdx_y, gridDim_x, sync_threads, threadIdx_x};
use cuda_device::vector::F32x2;
use cuda_device::warp::shuffle_xor_f32;
use cuda_device::{SharedArray, kernel};

unsafe extern "C" {
    fn __nv_floorf(x: f32) -> f32;
    fn __nv_rintf(x: f32) -> f32;
    fn __nv_expf(x: f32) -> f32;
}

/// Table entry of a chunk the source does not hold: every voxel reads the fill value
pub const FILL: u32 = 0xFFFF_FFFE;
/// Table entry of a chunk that is not resident: its voxels sample as NaN
pub const ABSENT: u32 = 0xFFFF_FFFF;
/// log2 of the chunk edge
pub const CHUNK_SHIFT: u32 = 7;
/// Words per crop grid record: origin (3 f32), axes (9 f32, row-major zyx x (d, h, w)), base (3 i32), padding
pub const GRID_WORDS: usize = 16;
/// Threads per block of the sampling kernels: a 2 x 8 x 16 (d, h, w) tile of warps of 2 x 4 x 4
pub const SAMPLE_THREADS: u32 = 256;
/// float32(1/255), villa's `np.float32(1/255)`
const INV255: f32 = 0.003_921_569;
const CHUNK_MASK: i32 = (1 << CHUNK_SHIFT) - 1;
const SLOT_SHIFT: u32 = 3 * CHUNK_SHIFT;

/// The resident CT: chunk table, slot pool and the volume's extent
#[derive(Clone, Copy)]
struct Ct {
    table: *const u32,
    pool: *const u8,
    shape: [u32; 3],
    gy: u32,
    gx: u32,
    fill: f32,
}

impl Ct {
    /// Voxel `(z, y, x)`: the fill value outside the volume or in a chunk the source lacks, NaN in a chunk that is
    /// not resident
    #[inline(always)]
    unsafe fn voxel(&self, z: i32, y: i32, x: i32) -> f32 {
        if (z as u32 >= self.shape[0]) | (y as u32 >= self.shape[1]) | (x as u32 >= self.shape[2]) {
            return self.fill;
        }
        let at = ((z >> CHUNK_SHIFT) as u32 * self.gy + (y >> CHUNK_SHIFT) as u32) * self.gx + (x >> CHUNK_SHIFT) as u32;
        // SAFETY: the coordinates are inside the volume, so `at` is inside the table the host sized for its grid
        let slot = unsafe { *self.table.add(at as usize) };
        if slot >= FILL {
            return if slot == FILL { self.fill } else { f32::NAN };
        }
        let local = (((z & CHUNK_MASK) << (2 * CHUNK_SHIFT)) | ((y & CHUNK_MASK) << CHUNK_SHIFT) | (x & CHUNK_MASK)) as u64;
        // SAFETY: a resident slot holds a whole chunk
        unsafe { *self.pool.add((((slot as u64) << SLOT_SHIFT) + local) as usize) as f32 }
    }

    /// The 8 corners of the cell at `lo`, corner `k` one step up along z for bit 0, y for bit 1, x for bit 2
    #[inline(always)]
    unsafe fn corners(&self, lo: [i32; 3]) -> [f32; 8] {
        let inner = (lo[0] >= 0)
            & (lo[1] >= 0)
            & (lo[2] >= 0)
            & ((lo[0] + 1) < self.shape[0] as i32)
            & ((lo[1] + 1) < self.shape[1] as i32)
            & ((lo[2] + 1) < self.shape[2] as i32)
            & ((lo[0] & CHUNK_MASK) != CHUNK_MASK)
            & ((lo[1] & CHUNK_MASK) != CHUNK_MASK)
            & ((lo[2] & CHUNK_MASK) != CHUNK_MASK);
        if inner {
            let at =
                ((lo[0] >> CHUNK_SHIFT) as u32 * self.gy + (lo[1] >> CHUNK_SHIFT) as u32) * self.gx + (lo[2] >> CHUNK_SHIFT) as u32;
            // SAFETY: the cell is inside the volume and inside one chunk
            let slot = unsafe { *self.table.add(at as usize) };
            if slot >= FILL {
                let v = if slot == FILL { self.fill } else { f32::NAN };
                return [v; 8];
            }
            let local =
                (((lo[0] & CHUNK_MASK) << (2 * CHUNK_SHIFT)) | ((lo[1] & CHUNK_MASK) << CHUNK_SHIFT) | (lo[2] & CHUNK_MASK)) as u64;
            // SAFETY: a resident slot holds a whole chunk, and the cell's far corner is inside it
            unsafe {
                let p = self.pool.add((((slot as u64) << SLOT_SHIFT) + local) as usize);
                const Z: usize = 1 << (2 * CHUNK_SHIFT);
                const Y: usize = 1 << CHUNK_SHIFT;
                return [
                    *p as f32,
                    *p.add(Z) as f32,
                    *p.add(Y) as f32,
                    *p.add(Z + Y) as f32,
                    *p.add(1) as f32,
                    *p.add(Z + 1) as f32,
                    *p.add(Y + 1) as f32,
                    *p.add(Z + Y + 1) as f32,
                ];
            }
        }
        // SAFETY: `voxel` checks every corner against the volume
        unsafe {
            [
                self.voxel(lo[0], lo[1], lo[2]),
                self.voxel(lo[0] + 1, lo[1], lo[2]),
                self.voxel(lo[0], lo[1] + 1, lo[2]),
                self.voxel(lo[0] + 1, lo[1] + 1, lo[2]),
                self.voxel(lo[0], lo[1], lo[2] + 1),
                self.voxel(lo[0] + 1, lo[1], lo[2] + 1),
                self.voxel(lo[0], lo[1] + 1, lo[2] + 1),
                self.voxel(lo[0] + 1, lo[1] + 1, lo[2] + 1),
            ]
        }
    }
}

/// One crop's affine grid, from its [`GRID_WORDS`] record
#[derive(Clone, Copy)]
struct Grid {
    origin: [f32; 3],
    axes: [f32; 9],
    base: [i32; 3],
}

impl Grid {
    #[inline(always)]
    unsafe fn load(grids: *const u32, crop: u32) -> Self {
        // SAFETY: the host passes one record per crop of the launch
        let r = unsafe { grids.add(crop as usize * GRID_WORDS) };
        let f = |i: usize| f32::from_bits(unsafe { *r.add(i) });
        let b = |i: usize| unsafe { *r.add(i) } as i32;
        Grid {
            origin: [f(0), f(1), f(2)],
            axes: [f(3), f(4), f(5), f(6), f(7), f(8), f(9), f(10), f(11)],
            base: [b(12), b(13), b(14)],
        }
    }

    /// The CT value at sample `(d, h, w)`, as `sample_affine` with `order=1` computes it
    #[inline(always)]
    unsafe fn sample(&self, ct: &Ct, d: u32, h: u32, w: u32) -> f32 {
        let (d, h, w) = (d as f32, h as f32, w as f32);
        let mut lo = [0i32; 3];
        let mut frac = [0f32; 3];
        let mut outside = false;
        for a in 0..3 {
            __unroll_config::<0>();
            let c = add_rn_f32(
                add_rn_f32(add_rn_f32(self.origin[a], mul_rn_f32(self.axes[3 * a], d)), mul_rn_f32(self.axes[3 * a + 1], h)),
                mul_rn_f32(self.axes[3 * a + 2], w),
            );
            outside |= !((c > (-1 - self.base[a]) as f32) & (c < (ct.shape[a] as i32 - self.base[a]) as f32));
            // SAFETY: libdevice's floorf, a pure function
            let low = unsafe { __nv_floorf(c) };
            frac[a] = add_rn_f32(c, -low);
            lo[a] = self.base[a] + low as i32;
        }
        if outside {
            return ct.fill;
        }

        // SAFETY: `corners` checks the cell against the volume and the table
        let v = unsafe { ct.corners(lo) };
        let g = [add_rn_f32(1.0, -frac[0]), add_rn_f32(1.0, -frac[1]), add_rn_f32(1.0, -frac[2])];
        let zy = [
            mul_rn_f32(g[0], g[1]),
            mul_rn_f32(frac[0], g[1]),
            mul_rn_f32(g[0], frac[1]),
            mul_rn_f32(frac[0], frac[1]),
        ];
        let mut acc = 0.0f32;
        for k in 0..8 {
            __unroll_config::<0>();
            let wx = if k & 4 == 0 { g[2] } else { frac[2] };
            acc = fma_rn_f32(mul_rn_f32(zy[k & 3], wx), v[k], acc);
        }
        acc
    }
}

/// `a + b` and its rounding error, exactly (Knuth's TwoSum)
#[inline(always)]
fn two_sum(a: f32, b: f32) -> (f32, f32) {
    let s = add_rn_f32(a, b);
    let bb = add_rn_f32(s, -a);
    (s, add_rn_f32(add_rn_f32(a, -add_rn_f32(s, -bb)), add_rn_f32(b, -bb)))
}

/// villa's z-score of `x` from a crop's float32-pair statistics `[mean_hi, mean_lo, inv_hi, inv_lo]`
/// ([`sc_finalize`]): `x - mean` exactly as a pair, then the pair product with `inv`, rounded once to float32.
/// [`sc_write`] applies it as it writes a crop; readers of an [`sc_raw`] crop apply it as they load one, to the
/// same bits
#[inline(always)]
pub fn zscore(x: f32, s: [f32; 4]) -> f32 {
    let [mh, ml, ih, il] = s;
    let (dh, de) = two_sum(x, -mh);
    let dl = add_rn_f32(de, -ml);
    let ph = mul_rn_f32(dh, ih);
    let pe = fma_rn_f32(dh, ih, -ph);
    add_rn_f32(ph, add_rn_f32(pe, fma_rn_f32(dh, il, mul_rn_f32(dl, ih))))
}

/// An unevaluated float32 sum `hi + lo`
#[derive(Clone, Copy)]
struct Pair {
    hi: f32,
    lo: f32,
}

impl Pair {
    const ZERO: Pair = Pair { hi: 0.0, lo: 0.0 };

    /// Adds the pair `b + b_lo`, keeping the rounding error of the high parts in `lo`
    #[inline(always)]
    fn add(self, b: f32, b_lo: f32) -> Pair {
        let (s, err) = two_sum(self.hi, b);
        Pair { hi: s, lo: add_rn_f32(add_rn_f32(self.lo, b_lo), err) }
    }

    #[inline(always)]
    fn to_f64(self) -> f64 {
        add_rn_f64(self.hi as f64, self.lo as f64)
    }
}

/// The sampling kernels' thread layout: block `blockIdx.x` of crop `blockIdx.y` covers a (2·steps) x 8 x 16
/// (d, h, w) tile; each warp a 2 x 4 x 4 brick of it per step, so a warp's corners stay within a few chunk rows
#[derive(Clone, Copy)]
struct Tile {
    d0: u32,
    h: u32,
    w: u32,
}

impl Tile {
    #[inline(always)]
    fn of(tiles_w: u32, tiles_h: u32, steps: u32) -> Tile {
        let t = blockIdx_x();
        let (lane, warp) = (threadIdx_x() & 31, threadIdx_x() >> 5);
        let tw = t % tiles_w;
        let th = (t / tiles_w) % tiles_h;
        let td = t / (tiles_w * tiles_h);
        Tile {
            d0: td * 2 * steps + ((lane >> 4) & 1),
            h: th * 8 + ((warp >> 2) & 1) * 4 + ((lane >> 2) & 3),
            w: tw * 16 + (warp & 3) * 4 + (lane & 3),
        }
    }
}

/// Samples block `blockIdx.x`'s tile of crop `blockIdx.y` and writes the tile's sum and sum of squares of
/// `x = value · float32(1/255)` as two float64 partials at `partials[2 · (crop · tiles + tile)]`; with a non-null
/// `out`, also writes every `x` raw as channel 0 of the NDHWC `(B, cd, ch, cw, 2)` float32 `out`, channel 1 zeroed.
/// [`sc_stats`] and [`sc_raw`] share it, so their partials are the same bits
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn tile_stats(
    ct: &Ct,
    grids: *const u32,
    tiles_w: u32,
    tiles_h: u32,
    steps: u32,
    partials: *mut f64,
    out: *mut f32,
    dims: [u32; 3],
) {
    static mut PART: SharedArray<f64, 16> = SharedArray::UNINIT;
    let crop = blockIdx_y();
    // SAFETY: one grid record per crop
    let grid = unsafe { Grid::load(grids, crop) };
    let tile = Tile::of(tiles_w, tiles_h, steps);
    let [cd, ch, cw] = dims;
    let per_crop = (cd * ch * cw) as u64;
    let mut sum = Pair::ZERO;
    let mut sq = Pair::ZERO;
    for i in 0..steps {
        let d = tile.d0 + 2 * i;
        // SAFETY: `sample` bounds every read
        let v = unsafe { grid.sample(ct, d, tile.h, tile.w) };
        let x = mul_rn_f32(v, INV255);
        if !out.is_null() {
            let idx = crop as u64 * per_crop + ((d * ch + tile.h) * cw + tile.w) as u64;
            // SAFETY: the host sizes `out` at two floats per sample of every crop
            unsafe { *(out.add(2 * idx as usize) as *mut F32x2) = F32x2::new([x, 0.0]) };
        }
        let x2 = mul_rn_f32(x, x);
        sum = sum.add(x, 0.0);
        sq = sq.add(x2, fma_rn_f32(x, x, -x2));
    }

    let mut off = 16u32;
    while off > 0 {
        let (sh, sl) = (shuffle_xor_f32(sum.hi, off), shuffle_xor_f32(sum.lo, off));
        let (qh, ql) = (shuffle_xor_f32(sq.hi, off), shuffle_xor_f32(sq.lo, off));
        sum = sum.add(sh, sl);
        sq = sq.add(qh, ql);
        off >>= 1;
    }
    let (lane, warp) = (threadIdx_x() & 31, threadIdx_x() >> 5);
    let part = unsafe { &mut *addr_of_mut!(PART) };
    if lane == 0 {
        part[2 * warp as usize] = sum.to_f64();
        part[2 * warp as usize + 1] = sq.to_f64();
    }
    sync_threads();
    if threadIdx_x() == 0 {
        let (mut s, mut q) = (0.0f64, 0.0f64);
        for k in 0..(SAMPLE_THREADS / 32) as usize {
            s = add_rn_f64(s, part[2 * k]);
            q = add_rn_f64(q, part[2 * k + 1]);
        }
        let tiles = blockIdx_x() as usize + crop as usize * gridDim_x() as usize;
        // SAFETY: the host sizes `partials` at two per tile of the launch
        unsafe {
            *partials.add(2 * tiles) = s;
            *partials.add(2 * tiles + 1) = q;
        }
    }
}

/// Pass 1: each block's sum and sum of squares of `x = value · float32(1/255)` over its tile, as two float64
/// partials at `partials[2 · (crop · tiles + tile)]`
///
/// Grid: x over a crop's `tiles_w · tiles_h · tiles_d` tiles, y over crops; [`SAMPLE_THREADS`] threads. The crop
/// is `cd x ch x cw` with `cw % 16 == 0`, `ch % 8 == 0` and `cd % (2 · steps) == 0`
#[kernel]
#[allow(clippy::too_many_arguments)]
pub unsafe fn sc_stats(
    table: *const u32,
    pool: *const u8,
    sz: u32,
    sy: u32,
    sx: u32,
    gy: u32,
    gx: u32,
    fill: f32,
    grids: *const u32,
    tiles_w: u32,
    tiles_h: u32,
    steps: u32,
    partials: *mut f64,
) {
    let ct = Ct { table, pool, shape: [sz, sy, sx], gy, gx, fill };
    // SAFETY: as the kernel's contract; no raw output
    unsafe { tile_stats(&ct, grids, tiles_w, tiles_h, steps, partials, core::ptr::null_mut(), [0; 3]) }
}

/// The strip crop in one sampling pass: [`sc_stats`]' partials, and every sample's raw `x = value ·
/// float32(1/255)` as channel 0 of the NDHWC `(B, cd, ch, cw, 2)` float32 `out` with channel 1 zeroed for
/// [`sc_tube`]. The crop's readers z-score channel 0 as they load it ([`zscore`] with [`sc_finalize`]'s
/// statistics), so the second sampling pass of [`sc_write`] is not needed
///
/// Grid and tile as [`sc_stats`]
#[kernel]
#[allow(clippy::too_many_arguments)]
pub unsafe fn sc_raw(
    table: *const u32,
    pool: *const u8,
    sz: u32,
    sy: u32,
    sx: u32,
    gy: u32,
    gx: u32,
    fill: f32,
    grids: *const u32,
    tiles_w: u32,
    tiles_h: u32,
    steps: u32,
    partials: *mut f64,
    cd: u32,
    ch: u32,
    cw: u32,
    out: *mut f32,
) {
    let ct = Ct { table, pool, shape: [sz, sy, sx], gy, gx, fill };
    // SAFETY: as the kernel's contract
    unsafe { tile_stats(&ct, grids, tiles_w, tiles_h, steps, partials, out, [cd, ch, cw]) }
}

/// Each crop's mean and `1 / max(std, 1e-6)` (population deviation) from its `tiles` partials, as float32 pairs
/// `stats[4 · crop ..] = (mean_hi, mean_lo, inv_hi, inv_lo)`; `n` is the crop's sample count
///
/// Grid: x over crops; 256 threads. Partials are summed in a fixed order, so a crop's statistics do not depend
/// on the launch
#[kernel]
pub unsafe fn sc_finalize(partials: *const f64, tiles: u32, n: u32, stats: *mut f32) {
    let n = n as f64;
    static mut SUM: SharedArray<f64, 512> = SharedArray::UNINIT;
    let crop = blockIdx_x();
    let t = threadIdx_x();
    let base = 2 * crop as usize * tiles as usize;
    let (mut s, mut q) = (0.0f64, 0.0f64);
    let mut i = t;
    while i < tiles {
        // SAFETY: two partials per tile of the crop
        unsafe {
            s = add_rn_f64(s, *partials.add(base + 2 * i as usize));
            q = add_rn_f64(q, *partials.add(base + 2 * i as usize + 1));
        }
        i += blockDim_x();
    }
    let sum = unsafe { &mut *addr_of_mut!(SUM) };
    sum[2 * t as usize] = s;
    sum[2 * t as usize + 1] = q;
    sync_threads();
    let mut half = 128u32;
    while half > 0 {
        if t < half {
            let (a, b) = (2 * t as usize, 2 * (t + half) as usize);
            sum[a] = add_rn_f64(sum[a], sum[b]);
            sum[a + 1] = add_rn_f64(sum[a + 1], sum[b + 1]);
        }
        sync_threads();
        half >>= 1;
    }
    if t == 0 {
        let mean = div_rn_f64(sum[0], n);
        let var = add_rn_f64(div_rn_f64(sum[1], n), -mul_rn_f64(mean, mean));
        let std = sqrt_rn_f64(if var > 0.0 { var } else { 0.0 });
        let inv = div_rn_f64(1.0, if std > 1e-6 { std } else { 1e-6 });
        let (mh, ih) = (mean as f32, inv as f32);
        // SAFETY: four floats per crop
        unsafe {
            let o = stats.add(4 * crop as usize);
            *o = mh;
            *o.add(1) = add_rn_f64(mean, -(mh as f64)) as f32;
            *o.add(2) = ih;
            *o.add(3) = add_rn_f64(inv, -(ih as f64)) as f32;
        }
    }
}

/// Pass 2: every sample's z-scored value `(x - mean) · inv`, rounded once to float32 from a float32-pair product,
/// written to crop `blockIdx.y` of `out`: channel 0 of an NDHWC `(B, cd, ch, cw, 2)` buffer with channel 1 zeroed
/// when `channels` is 2, or `(B, cd, ch, cw)` when it is 1
///
/// `mode` 1 is the write ceiling: no CT reads, a cheap per-sample value written in their place. Grid and tile as
/// [`sc_stats`]
#[kernel]
#[allow(clippy::too_many_arguments)]
pub unsafe fn sc_write(
    table: *const u32,
    pool: *const u8,
    sz: u32,
    sy: u32,
    sx: u32,
    gy: u32,
    gx: u32,
    fill: f32,
    grids: *const u32,
    tiles_w: u32,
    tiles_h: u32,
    steps: u32,
    stats: *const f32,
    cd: u32,
    ch: u32,
    cw: u32,
    channels: u32,
    mode: u32,
    out: *mut f32,
) {
    let ct = Ct { table, pool, shape: [sz, sy, sx], gy, gx, fill };
    let crop = blockIdx_y();
    // SAFETY: one grid record and four statistics per crop
    let grid = unsafe { Grid::load(grids, crop) };
    let st = unsafe {
        let s = stats.add(4 * crop as usize);
        [*s, *s.add(1), *s.add(2), *s.add(3)]
    };
    let tile = Tile::of(tiles_w, tiles_h, steps);
    let per_crop = (cd * ch * cw) as u64;
    for i in 0..steps {
        let d = tile.d0 + 2 * i;
        let at = (d * ch + tile.h) * cw + tile.w;
        let x = if mode == 1 {
            at as f32 * 1e-9
        } else {
            // SAFETY: `sample` bounds every read
            mul_rn_f32(unsafe { grid.sample(&ct, d, tile.h, tile.w) }, INV255)
        };
        let y = zscore(x, st);
        let idx = crop as u64 * per_crop + at as u64;
        // SAFETY: the host sizes `out` at `channels` floats per sample of every crop
        unsafe {
            if channels == 2 {
                *(out.add(2 * idx as usize) as *mut F32x2) = F32x2::new([y, 0.0]);
            } else {
                *out.add(idx as usize) = y;
            }
        }
    }
}

/// villa's `render_tube` into channel 1 of the NDHWC `(B, cd, ch, cw, 2)` input: each of decision `blockIdx.y`'s
/// `counts[b]` crop-local `(u, v, f)` points (`points[b · npts + p]`) takes index
/// `(f / spacing + behind, v / spacing + hc, u / spacing + wc)`, and each of the `noff` integer offsets around its
/// rounded (half to even) index gets weight `exp(-|s - c|² / denom)` (`denom` = 2 sigma²), kept where larger;
/// offsets outside the crop are skipped. The weights are positive and max is order-free, so the result is the
/// reference's exactly. `out` is the input buffer's bits; [`sc_write`] zeroed channel 1
///
/// Grid: x over `npts · noff` in blocks of 256, y over decisions
#[kernel]
#[allow(clippy::too_many_arguments)]
pub unsafe fn sc_tube(
    points: *const f32,
    counts: *const u32,
    npts: u32,
    offsets: *const i32,
    noff: u32,
    cd: u32,
    ch: u32,
    cw: u32,
    spacing: f32,
    behind: f32,
    hc: f32,
    wc: f32,
    denom: f32,
    out: *mut u32,
) {
    let b = blockIdx_y();
    let i = blockIdx_x() * blockDim_x() + threadIdx_x();
    let p = i / noff;
    let k = i - p * noff;
    // SAFETY: one count per decision
    if p >= unsafe { *counts.add(b as usize) } {
        return;
    }
    // SAFETY: `p < counts[b] <= npts`, three floats per point, three offsets per entry
    let (pt, off) = unsafe {
        let q = points.add(3 * (b * npts + p) as usize);
        let o = offsets.add(3 * k as usize);
        ([*q, *q.add(1), *q.add(2)], [*o, *o.add(1), *o.add(2)])
    };
    let c = [
        add_rn_f32(div_rn_f32(pt[2], spacing), behind),
        add_rn_f32(div_rn_f32(pt[1], spacing), hc),
        add_rn_f32(div_rn_f32(pt[0], spacing), wc),
    ];
    let dims = [cd as i32, ch as i32, cw as i32];
    let mut s = [0i32; 3];
    let mut sq = [0f32; 3];
    let mut inside = true;
    #[unroll]
    for a in 0..3 {
        // SAFETY: libdevice's rintf, a pure function
        s[a] = unsafe { __nv_rintf(c[a]) } as i32 + off[a];
        let diff = add_rn_f32(s[a] as f32, -c[a]);
        sq[a] = mul_rn_f32(diff, diff);
        inside &= (s[a] >= 0) & (s[a] < dims[a]);
    }
    if !inside {
        return;
    }
    // torch's CUDA `sum(-1)` over three values folds its lanes as (0 + 2) + 1, which render_tube's bits follow
    let sum = add_rn_f32(add_rn_f32(sq[0], sq[2]), sq[1]);
    // SAFETY: libdevice's expf, a pure function
    let weight = unsafe { __nv_expf(div_rn_f32(-sum, denom)) };
    let at = ((b * cd + s[0] as u32) * ch + s[1] as u32) * cw + s[2] as u32;
    // SAFETY: inside the crop; device-scope atomics, and the max of non-negative floats is the max of their bits
    unsafe {
        DeviceAtomicU32::from_ptr(out.add(2 * at as usize + 1)).fetch_max(weight.to_bits(), AtomicOrdering::Relaxed);
    }
}
