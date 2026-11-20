//! Trace finishing on the GPU (`FinishCtx::finish`, src/finish.rs): [`fs_segments`] tests each segment of a
//! batch of curves against the pooled and the native predictions (`valid_segments`), and [`fs_fit`] smooths
//! and resamples each valid piece (`fit_curve`, then `evaluate_curve`, src/kernels.rs)
//!
//! Every float32 and float64 add, multiply, divide and square root is an explicit round-to-nearest instruction
//! grouped as the CPU's expression is, so no product is fused into an add and the bytes equal the CPU's. The
//! loops whose trip count depends on the data (a segment's samples with their early exit, the ten in-place
//! smoothing passes of a piece, the resampling's monotone scan) run sequentially in one thread, as on the CPU
//!
//! The pooled volumes arrive as the host's 16³ bricks over the whole brick grid (`Vol::brick_index`,
//! src/vol.rs), and the native predictions as slabs of planes of each chunk (`NativeCache::slab_at`,
//! src/finish.rs), V then H in each slot. A native voxel whose slab the host did not upload reads as 0 and sets
//! bit 1 of `err`; the host then finishes the batch on the CPU

use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::float::{
    add_rn_f32, add_rn_f64, div_rn_f32, div_rn_f64, mul_rn_f32, mul_rn_f64, sqrt_rn_f32, sqrt_rn_f64,
};
use cuda_device::kernel;
use cuda_device::thread::{blockDim_x, blockIdx_x, threadIdx_x};

use crate::{BRICK, MASK, NO_BRICK, SHIFT, py_max, py_min};

/// A slab table entry for a slab the host did not upload
pub const NO_SLAB: u32 = u32::MAX;

/// `err` bit set by a native read from a slab missing from the slab cache
pub const ERR_SLAB: u32 = 1;
/// `err` bit set by a fitted piece longer than its capacity in `fitted`
pub const ERR_CAPACITY: u32 = 2;

#[inline(always)]
fn fadd(a: f32, b: f32) -> f32 {
    add_rn_f32(a, b)
}

#[inline(always)]
fn fsub(a: f32, b: f32) -> f32 {
    add_rn_f32(a, -b)
}

#[inline(always)]
fn fmul(a: f32, b: f32) -> f32 {
    mul_rn_f32(a, b)
}

#[inline(always)]
fn dadd(a: f64, b: f64) -> f64 {
    add_rn_f64(a, b)
}

#[inline(always)]
fn dsub(a: f64, b: f64) -> f64 {
    add_rn_f64(a, -b)
}

#[inline(always)]
fn dmul(a: f64, b: f64) -> f64 {
    mul_rn_f64(a, b)
}

/// `f64::floor`, which core lacks; exact for every input, signed zeros included
#[inline(always)]
fn floor64(x: f64) -> f64 {
    // from 2^52 up every float64 is an integer, and NaN and the infinities floor to themselves
    let mag = if x < 0.0 { -x } else { x };
    if x.is_nan() || mag >= 4503599627370496.0 {
        return x;
    }
    let t = x as i64 as f64;
    if t == x {
        x
    } else if t > x {
        dsub(t, 1.0)
    } else {
        t
    }
}

/// `x.ceil() as i64` for a float32
#[inline(always)]
fn ceil_i64(x: f32) -> i64 {
    // from 2^23 up every float32 is an integer, and the cast saturates the infinities and maps NaN to 0, as the
    // CPU's cast of its ceil does
    let mag = if x < 0.0 { -x } else { x };
    if x.is_nan() || mag >= 8388608.0 {
        return x as i64;
    }
    let t = x as i32;
    if (t as f32) < x { (t + 1) as i64 } else { t as i64 }
}

/// `x - y` of two float32 points
#[inline(always)]
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [fsub(a[0], b[0]), fsub(a[1], b[1]), fsub(a[2], b[2])]
}

/// `norm2_f32(v).sqrt()` (src/kernels.rs): the squares summed left to right in float32
#[inline(always)]
fn norm32(v: [f32; 3]) -> f32 {
    sqrt_rn_f32(fadd(fadd(fmul(v[0], v[0]), fmul(v[1], v[1])), fmul(v[2], v[2])))
}

/// Point `i` of a buffer of float32 x, y, z triples
#[inline(always)]
unsafe fn point(p: *const f32, i: u32) -> [f32; 3] {
    let at = 3 * i as usize;
    unsafe { [*p.add(at), *p.add(at + 1), *p.add(at + 2)] }
}

#[inline(always)]
unsafe fn set_point(p: *mut f32, i: u32, v: [f32; 3]) {
    let at = 3 * i as usize;
    unsafe {
        *p.add(at) = v[0];
        *p.add(at + 1) = v[1];
        *p.add(at + 2) = v[2];
    }
}

/// One pooled volume (`Vol<u8>`, default 0): a pool slot per brick of the full brick grid, C order, or
/// [`NO_BRICK`]
#[derive(Clone, Copy)]
struct Pooled {
    table: *const u32,
    pool: *const u8,
    /// Bricks per z plane and per y row of the grid
    gy: u32,
    gx: u32,
}

impl Pooled {
    /// `Vol::get` at an in-bounds voxel
    #[inline(always)]
    unsafe fn get(&self, z: u32, y: u32, x: u32) -> u8 {
        let b = ((z >> SHIFT) * self.gy + (y >> SHIFT)) * self.gx + (x >> SHIFT);
        let slot = unsafe { *self.table.add(b as usize) };
        if slot == NO_BRICK {
            return 0;
        }
        let at = ((z & MASK) << (2 * SHIFT)) | ((y & MASK) << SHIFT) | (x & MASK);
        unsafe { *self.pool.add(slot as usize * BRICK as usize + at as usize) }
    }
}

/// Both pooled volumes, V and H, and their shape z, y, x
#[derive(Clone, Copy)]
struct Tracing {
    v: Pooled,
    h: Pooled,
    shape: [u32; 3],
}

impl Tracing {
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn new(
        table_v: *const u32,
        table_h: *const u32,
        pool_v: *const u8,
        pool_h: *const u8,
        shape: [u32; 3],
        gy: u32,
        gx: u32,
    ) -> Self {
        Self {
            v: Pooled { table: table_v, pool: pool_v, gy, gx },
            h: Pooled { table: table_h, pool: pool_h, gy, gx },
            shape,
        }
    }

    /// The family's own volume and the other one (`Presence::new`)
    #[inline(always)]
    fn own_other(&self, family: u32) -> (Pooled, Pooled) {
        if family == 1 { (self.v, self.h) } else { (self.h, self.v) }
    }

    /// `sample_tracing` (src/finish.rs): SciPy's order-1 `map_coordinates` in "nearest" mode of both volumes
    /// at a point given as float32 x, y, z, returned as (V, H)
    #[inline(always)]
    unsafe fn sample(&self, p: [f32; 3]) -> (f32, f32) {
        let (z0, z1, wz0, wz1) = axis_weights(p[2] as f64, self.shape[0]);
        let (y0, y1, wy0, wy1) = axis_weights(p[1] as f64, self.shape[1]);
        let (x0, x1, wx0, wx1) = axis_weights(p[0] as f64, self.shape[2]);
        let channel = |vol: Pooled| {
            // the corners in kz, ky, kx order, each value times its z, y then x weight, summed in float64
            let term = |acc: f64, z: u32, y: u32, x: u32, wz: f64, wy: f64, wx: f64| {
                let coeff = unsafe { vol.get(z, y, x) } as f64;
                dadd(acc, dmul(dmul(dmul(coeff, wz), wy), wx))
            };
            let mut acc = 0.0f64;
            acc = term(acc, z0, y0, x0, wz0, wy0, wx0);
            acc = term(acc, z0, y0, x1, wz0, wy0, wx1);
            acc = term(acc, z0, y1, x0, wz0, wy1, wx0);
            acc = term(acc, z0, y1, x1, wz0, wy1, wx1);
            acc = term(acc, z1, y0, x0, wz1, wy0, wx0);
            acc = term(acc, z1, y0, x1, wz1, wy0, wx1);
            acc = term(acc, z1, y1, x0, wz1, wy1, wx0);
            acc = term(acc, z1, y1, x1, wz1, wy1, wx1);
            acc as f32
        };
        (channel(self.v), channel(self.h))
    }
}

/// One axis of `sample_tracing`: the two clamped indices and their weights `1 - t` and `1 - (1 - t)`, from
/// the unclamped coordinate
#[inline(always)]
fn axis_weights(c: f64, len: u32) -> (u32, u32, f64, f64) {
    let start = floor64(c);
    let t = dsub(c, start);
    let w0 = dsub(1.0, t);
    let w1 = dsub(1.0, w0);
    let s = start as i64;
    let hi = len as i64 - 1;
    // `clamp` asserts its bounds; max then min clamps the same without the trap
    let i0 = s.max(0).min(hi) as u32;
    let i1 = s.wrapping_add(1).max(0).min(hi) as u32;
    (i0, i1, w0, w1)
}

/// The native predictions: a slot of `slab_pool` per slab of the slab grid, or [`NO_SLAB`]
#[derive(Clone, Copy)]
struct Native {
    table: *const u32,
    pool: *const u8,
    err: *mut u32,
    /// The native shape z, y, x
    shape: [u32; 3],
    /// Chunk edge z, y, x and planes per slab, each at least 1 so the divisions cannot trap
    edge: [u32; 3],
    planes: u32,
    ncy: u32,
    ncx: u32,
    /// Slabs per chunk
    spc: u32,
}

impl Native {
    /// V and H at an in-bounds voxel (`NativeSampler::value`)
    #[inline(always)]
    unsafe fn value(&self, z: u32, y: u32, x: u32) -> (u8, u8) {
        let [ez, ey, ex] = self.edge;
        let (cz, cy, cx) = (z / ez, y / ey, x / ex);
        let s = (z - cz * ez) / self.planes;
        let key = ((cz * self.ncy + cy) * self.ncx + cx) * self.spc + s;
        let slot = unsafe { *self.table.add(key as usize) };
        if slot == NO_SLAB {
            // SAFETY: `err` is one u32 the host zeroed; a relaxed device-scope or makes concurrent flags defined
            unsafe { DeviceAtomicU32::from_ptr(self.err).fetch_or(ERR_SLAB, AtomicOrdering::Relaxed) };
            return (0, 0);
        }
        // the slab's planes start at plane s·planes of the chunk, each plane `ey·ex` bytes, V then H
        let half = self.planes * ey * ex;
        let at = (z - (cz * ez + s * self.planes)) * ey * ex + (y - cy * ey) * ex + (x - cx * ex);
        let base = slot as usize * 2 * half as usize;
        unsafe { (*self.pool.add(base + at as usize), *self.pool.add(base + half as usize + at as usize)) }
    }

    /// `NativeSampler::sample` (src/finish.rs) of a point given as float32 x, y, z in native voxels, returned as
    /// (V, H)
    #[inline(always)]
    unsafe fn sample(&self, p: [f32; 3]) -> (f32, f32) {
        let (lz, fz) = native_axis(p[2], self.shape[0]);
        let (ly, fy) = native_axis(p[1], self.shape[1]);
        let (lx, fx) = native_axis(p[0], self.shape[2]);
        let (mut v, mut h) = (0.0f32, 0.0f32);
        let mut corner = 0u32;
        while corner < 8 {
            let (cz, cy, cx) = ((corner >> 2) & 1, (corner >> 1) & 1, corner & 1);
            let iz = (lz + cz).min(self.shape[0] - 1);
            let iy = (ly + cy).min(self.shape[1] - 1);
            let ix = (lx + cx).min(self.shape[2] - 1);
            let wz = if cz == 1 { fz } else { dsub(1.0, fz) };
            let wy = if cy == 1 { fy } else { dsub(1.0, fy) };
            let wx = if cx == 1 { fx } else { dsub(1.0, fx) };
            let weight = dmul(dmul(wz, wy), wx);
            let (a, b) = unsafe { self.value(iz, iy, ix) };
            v = dadd(v as f64, dmul(a as f64, weight)) as f32;
            h = dadd(h as f64, dmul(b as f64, weight)) as f32;
            corner += 1;
        }
        (v, h)
    }
}

/// One axis of `NativeSampler::sample`: `np.clip(q, 0, len - 1)` in float64, then its floor and fraction
#[inline(always)]
fn native_axis(q: f32, len: u32) -> (u32, f64) {
    // `q.max(0.0)` takes 0 for NaN; for -0.0 either zero gives the same sample, the fraction weighting
    // nothing then
    let v = if q > 0.0 { q } else { 0.0 };
    let hi = (len - 1) as f64;
    let q = if hi < v as f64 { hi } else { v as f64 };
    // q is at least 0, so truncation is the floor
    let lo = q as u32;
    (lo, dsub(q, lo as f64))
}

/// The family's own and other values of a (V, H) pair
#[inline(always)]
fn own_other(family: u32, vh: (f32, f32)) -> (f32, f32) {
    if family == 1 { vh } else { (vh.1, vh.0) }
}

/// One segment of `valid_segments` (src/finish.rs): whether every sample from `a` to `b` favours the family in
/// both the pooled and the native predictions, stopping at the first sample that does not
#[inline(always)]
unsafe fn segment_valid(a: [f32; 3], b: [f32; 3], tracing: &Tracing, native: &Native, family: u32) -> bool {
    let d = sub3(b, a);
    let len = norm32(d);
    let n = ceil_i64(div_rn_f32(len, 0.25)).max(1);
    let step = div_rn_f64(1.0, n as f64);
    let mut j = 0i64;
    while j <= n {
        // np.linspace(0, 1, n + 1, dtype="f4"): j * step in float64, the last exactly 1
        let t = if j == n { 1.0f32 } else { dmul(j as f64, step) as f32 };
        let q = [fadd(a[0], fmul(d[0], t)), fadd(a[1], fmul(d[1], t)), fadd(a[2], fmul(d[2], t))];
        let (own, other) = own_other(family, unsafe { tracing.sample(q) });
        let mut good = fadd(own, 1e-4) >= other;
        if good {
            let nq = [fadd(fmul(q[0], 2.0), 0.5), fadd(fmul(q[1], 2.0), 0.5), fadd(fmul(q[2], 2.0), 0.5)];
            let (own, other) = own_other(family, unsafe { native.sample(nq) });
            good = fadd(own, 1e-4) >= other;
        }
        if !good {
            return false;
        }
        j += 1;
    }
    true
}

/// The curve holding point slot `i`: the last `c` with `start[c] <= i`, `start` having `n_curves + 1`
/// nondecreasing entries from 0 with `i < start[n_curves]`; empty curves share their start with the next
#[inline(always)]
unsafe fn curve_of(start: *const u32, n_curves: u32, i: u32) -> u32 {
    let (mut lo, mut hi) = (0u32, n_curves);
    while hi - lo > 1 {
        let mid = lo + ((hi - lo) >> 1);
        if unsafe { *start.add(mid as usize) } <= i {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Test segment `i → i + 1` of each curve `start[c] .. start[c] + len[c]` of `pts` (float32 x, y, z): in
/// mode 0 (raw curves) `valid[i]` becomes 1 or 0; in mode 1 (fitted pieces, `valid_segments` with `all`) a
/// failing segment clears `ok[c]`, which the host sets to 1. Slots with no segment after them write nothing
///
/// The pooled volumes: `table_v`/`table_h` hold a slot of `pool_v`/`pool_h` (16³ bytes each, brick order) per
/// brick of the `grid_z × grid_y × grid_x` brick grid of the `shape_z × shape_y × shape_x` volumes. The native
/// predictions (`native_z × native_y × native_x`): `slab_table[((cz·ncy + cy)·ncx + cx)·spc + s]` is the slot of
/// slab `s` (planes `s·planes ..` of chunk cz, cy, cx of `edge_z × edge_y × edge_x`) in `slab_pool`, whose slots
/// hold `2·planes·edge_y·edge_x` bytes, V then H, each plane-major from the slab's first voxel, or [`NO_SLAB`]
///
/// Grid: x over the `n_slots` point slots
#[kernel]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fs_segments(
    pts: *const f32,
    start: *const u32,
    len: *const u32,
    n_curves: u32,
    n_slots: u32,
    mode: u32,
    valid: *mut u8,
    ok: *mut u32,
    err: *mut u32,
    family: u32,
    table_v: *const u32,
    table_h: *const u32,
    pool_v: *const u8,
    pool_h: *const u8,
    shape_z: u32,
    shape_y: u32,
    shape_x: u32,
    grid_z: u32,
    grid_y: u32,
    grid_x: u32,
    slab_table: *const u32,
    slab_pool: *const u8,
    native_z: u32,
    native_y: u32,
    native_x: u32,
    edge_z: u32,
    edge_y: u32,
    edge_x: u32,
    planes: u32,
    ncy: u32,
    ncx: u32,
    spc: u32,
) {
    // every voxel read is inside the volumes, so the grid's depth bounds no lookup
    let _ = grid_z;
    let i = blockIdx_x() * blockDim_x() + threadIdx_x();
    if i >= n_slots {
        return;
    }
    let c = unsafe { curve_of(start, n_curves, i) };
    let end = unsafe { *start.add(c as usize) + *len.add(c as usize) };
    if i + 1 >= end {
        return;
    }
    // SAFETY: in mode 1 `ok` holds one u32 per curve; relaxed atomics make the concurrent reads and clears
    // defined. Mode 0 never touches it
    let ok_c = || unsafe { DeviceAtomicU32::from_ptr(ok.add(c as usize)) };
    // a piece another segment already failed needs no more samples
    if mode == 1 && ok_c().load(AtomicOrdering::Relaxed) == 0 {
        return;
    }

    let tracing = Tracing::new(table_v, table_h, pool_v, pool_h, [shape_z, shape_y, shape_x], grid_y, grid_x);
    let native = Native {
        table: slab_table,
        pool: slab_pool,
        err,
        shape: [native_z, native_y, native_x],
        edge: [edge_z.max(1), edge_y.max(1), edge_x.max(1)],
        planes: planes.max(1),
        ncy,
        ncx,
        spc,
    };
    let (a, b) = unsafe { (point(pts, i), point(pts, i + 1)) };
    let good = unsafe { segment_valid(a, b, &tracing, &native, family) };
    if mode == 0 {
        unsafe { *valid.add(i as usize) = good as u8 };
    } else if !good {
        ok_c().store(0, AtomicOrdering::Relaxed);
    }
}

/// The clamp and cell split of `trilin_u8` (`cell`, src/kernels.rs) on one axis of `n` voxels: the cell's low
/// index and the float32 fraction
#[inline(always)]
fn cell_axis(mut v: f64, n: u32) -> (u32, f32) {
    let hi = dsub(n as f64, 1.001);
    if v < 0.0 {
        v = 0.0;
    } else if v > hi {
        v = hi as f32 as f64;
    }
    // the CPU's `as i64`: truncation toward zero, NaN to 0; every value here fits a u32 or truncates to 0
    let i = v as u32;
    (i, dsub(v, i as f64) as f32)
}

/// `Presence::combine` (src/fields.rs): `min(own + other, 255)`, zero where `own < other`
#[inline(always)]
unsafe fn presence(own: &Pooled, other: &Pooled, z: u32, y: u32, x: u32) -> u8 {
    let a = unsafe { own.get(z, y, x) };
    // an own value of 0 gives presence 0 whatever the other is
    if a == 0 {
        return 0;
    }
    let b = unsafe { other.get(z, y, x) };
    if a < b { 0 } else { (a as u32 + b as u32).min(255) as u8 }
}

/// `trilin_u8` (src/kernels.rs) of the family's presence at x, y, z; `fx · value` is a float32 product
#[inline(always)]
unsafe fn trilin_presence(own: &Pooled, other: &Pooled, shape: [u32; 3], x: f64, y: f64, z: f64) -> f64 {
    let (ix, fx) = cell_axis(x, shape[2]);
    let (iy, fy) = cell_axis(y, shape[1]);
    let (iz, fz) = cell_axis(z, shape[0]);
    let at = |dz: u32, dy: u32, dx: u32| unsafe { presence(own, other, iz + dz, iy + dy, ix + dx) };
    let lerp = |dz: u32, dy: u32| {
        let (c0, c1) = (at(dz, dy, 0), at(dz, dy, 1));
        dadd(dmul(dsub(1.0, fx as f64), c0 as f64), fmul(fx, c1 as f32) as f64)
    };
    let p00 = lerp(0, 0);
    let p01 = lerp(0, 1);
    let p10 = lerp(1, 0);
    let p11 = lerp(1, 1);
    let (fy, fz) = (fy as f64, fz as f64);
    let lo = dadd(dmul(dsub(1.0, fy), p00), dmul(fy, p01));
    let hi = dadd(dmul(dsub(1.0, fy), p10), dmul(fy, p11));
    dadd(dmul(dsub(1.0, fz), lo), dmul(fz, hi))
}

/// One coordinate of a smoothing step at an end-adjacent point: `0.25·b[i−1] + 0.5·b[i] + 0.25·b[i+1]`
#[inline(always)]
fn smooth_end(bm1: f32, b0: f32, bp1: f32) -> f64 {
    dadd(dadd(dmul(0.25, bm1 as f64), dmul(0.5, b0 as f64)), dmul(0.25, bp1 as f64))
}

/// One coordinate of a smoothing step inside: `0.5·b[i] + 0.3·lap + 0.2·bih`, the Laplacian's sum in float32
#[inline(always)]
fn smooth_inner(bm2: f32, bm1: f32, b0: f32, bp1: f32, bp2: f32) -> f64 {
    let lap = dmul(0.5, fadd(bm1, bp1) as f64);
    let bih = div_rn_f64(dsub(dadd(dadd(-bm2 as f64, dmul(4.0, bm1 as f64)), dmul(4.0, bp1 as f64)), bp2 as f64), 6.0);
    dadd(dadd(dmul(0.5, b0 as f64), dmul(0.3, lap)), dmul(0.2, bih))
}

/// Fit and resample each piece `pts[pstart[p] .. pstart[p] + plen[p]]` (at least 4 points): `fit_curve`'s ten
/// in-place smoothing passes on a copy in `smooth[pstart[p] ..]`, then `evaluate_curve` of its resampled count
/// into `fitted[fstart[p] ..]`, writing `flen[p]`. A piece whose count exceeds its capacity
/// `fstart[p + 1] − fstart[p]` gets `flen[p] = 0` and sets bit 2 of `err`. The pooled volumes as for
/// [`fs_segments`]
///
/// Grid: x over the `n_pieces` pieces
#[kernel]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fs_fit(
    pts: *const f32,
    pstart: *const u32,
    plen: *const u32,
    n_pieces: u32,
    smooth: *mut f32,
    fitted: *mut f32,
    fstart: *const u32,
    flen: *mut u32,
    err: *mut u32,
    family: u32,
    table_v: *const u32,
    table_h: *const u32,
    pool_v: *const u8,
    pool_h: *const u8,
    shape_z: u32,
    shape_y: u32,
    shape_x: u32,
    grid_z: u32,
    grid_y: u32,
    grid_x: u32,
) {
    // every voxel read is inside the volumes, so the grid's depth bounds no lookup
    let _ = grid_z;
    let p = blockIdx_x() * blockDim_x() + threadIdx_x();
    if p >= n_pieces {
        return;
    }
    let shape = [shape_z, shape_y, shape_x];
    let tracing = Tracing::new(table_v, table_h, pool_v, pool_h, shape, grid_y, grid_x);
    let (own, other) = tracing.own_other(family);
    let (base, m) = unsafe { (*pstart.add(p as usize), *plen.add(p as usize)) };
    let raw = unsafe { pts.add(3 * base as usize) };
    let buf = unsafe { smooth.add(3 * base as usize) };

    let mut i = 0;
    while i < m {
        unsafe { set_point(buf, i, point(raw, i)) };
        i += 1;
    }

    let arc = unsafe { fit(raw, buf, m, &own, &other, shape) };
    let n_out = (dadd(div_rn_f64(arc as f64, 0.8), 0.5) as i64).wrapping_add(1).max(2);
    let (out, cap) = unsafe { (*fstart.add(p as usize), *fstart.add(p as usize + 1) - *fstart.add(p as usize)) };
    if n_out > cap as i64 {
        unsafe {
            *flen.add(p as usize) = 0;
            // SAFETY: `err` is one u32 the host zeroed; a relaxed device-scope or makes concurrent flags defined
            DeviceAtomicU32::from_ptr(err).fetch_or(ERR_CAPACITY, AtomicOrdering::Relaxed);
        }
        return;
    }

    let n_out = n_out as u32;
    unsafe {
        evaluate(buf, m, arc, fitted.add(3 * out as usize), n_out, shape);
        *flen.add(p as usize) = n_out;
    }
}

/// `fit_curve`'s smoothing (src/kernels.rs, `max_dev` 0.3) of the `m ≥ 4` points of `buf`, a copy of `raw`, in
/// place: ten passes, each updating the points in order from the ones already updated. Returns the smoothed
/// arc length in float32
#[inline(always)]
unsafe fn fit(raw: *const f32, buf: *mut f32, m: u32, own: &Pooled, other: &Pooled, shape: [u32; 3]) -> f32 {
    const MAX_DEV: f64 = 0.3;
    let mut pass = 0;
    while pass < 10 {
        let mut i = 1;
        while i < m - 1 {
            let (bm1, b0, bp1) = unsafe { (point(buf, i - 1), point(buf, i), point(buf, i + 1)) };
            let mut p = if i == 1 || i == m - 2 {
                [
                    smooth_end(bm1[0], b0[0], bp1[0]),
                    smooth_end(bm1[1], b0[1], bp1[1]),
                    smooth_end(bm1[2], b0[2], bp1[2]),
                ]
            } else {
                let (bm2, bp2) = unsafe { (point(buf, i - 2), point(buf, i + 2)) };
                [
                    smooth_inner(bm2[0], bm1[0], b0[0], bp1[0], bp2[0]),
                    smooth_inner(bm2[1], bm1[1], b0[1], bp1[1], bp2[1]),
                    smooth_inner(bm2[2], bm1[2], b0[2], bp1[2], bp2[2]),
                ]
            };

            let o = unsafe { point(raw, i) };
            let (ox, oy, oz) = (o[0] as f64, o[1] as f64, o[2] as f64);
            let (dx, dy, dz) = (dsub(p[0], ox), dsub(p[1], oy), dsub(p[2], oz));
            let d = sqrt_rn_f64(dadd(dadd(dmul(dx, dx), dmul(dy, dy)), dmul(dz, dz)));
            if d > MAX_DEV {
                let s = div_rn_f64(MAX_DEV, d);
                p = [dadd(ox, dmul(dx, s)), dadd(oy, dmul(dy, s)), dadd(oz, dmul(dz, s))];
            }
            let p_orig = unsafe { trilin_presence(own, other, shape, ox, oy, oz) };
            let p_new = unsafe { trilin_presence(own, other, shape, p[0], p[1], p[2]) };
            if p_orig >= 76.5 && p_new < py_max(77.0, dsub(p_orig, 6.0)) {
                p = [
                    dmul(0.5, fadd(o[0], b0[0]) as f64),
                    dmul(0.5, fadd(o[1], b0[1]) as f64),
                    dmul(0.5, fadd(o[2], b0[2]) as f64),
                ];
            }
            unsafe { set_point(buf, i, [p[0] as f32, p[1] as f32, p[2] as f32]) };
            i += 1;
        }
        pass += 1;
    }

    let mut arc = 0.0f32;
    let mut i = 1;
    while i < m {
        arc = fadd(arc, norm32(sub3(unsafe { point(buf, i) }, unsafe { point(buf, i - 1) })));
        i += 1;
    }
    arc
}

/// `evaluate_curve` (src/kernels.rs): `n_out ≥ 2` Catmull-Rom samples of the `m ≥ 4` smoothed points, evenly
/// spaced in arc length, clamped to the volume of `shape` (z, y, x)
///
/// `tot` is `fit`'s arc length, the same float32 sum `evaluate_curve` builds as `cum_s[m − 1]`. The running
/// sums are formed as the scan reaches them instead of stored: `cum_s[k + 1]` is `cum_s[k]` plus segment k's
/// length either way
#[inline(always)]
unsafe fn evaluate(s: *const f32, m: u32, tot: f32, out: *mut f32, n_out: u32, shape: [u32; 3]) {
    let seg = |k: u32| norm32(sub3(unsafe { point(s, k + 1) }, unsafe { point(s, k) }));
    unsafe {
        set_point(out, 0, point(s, 0));
        set_point(out, n_out - 1, point(s, m - 1));
    }
    let upper =
        [(shape[2] as i64 - 1) as f32 as f64, (shape[1] as i64 - 1) as f32 as f64, (shape[0] as i64 - 1) as f32 as f64];
    let (mut seg_k, mut c0) = (0u32, 0.0f32);
    let mut c1 = fadd(c0, seg(0));
    let mut j = 1;
    while j < n_out - 1 {
        let target = fmul(div_rn_f32(j as f32, (n_out - 1) as f32), tot);
        while seg_k + 2 < m && c1 < target {
            seg_k += 1;
            c0 = c1;
            c1 = fadd(c1, seg(seg_k));
        }
        let ds = fsub(c1, c0);
        let floor = 1e-06f32;
        let u = div_rn_f32(fsub(target, c0), if floor > ds { floor } else { ds });
        let u2 = fmul(u, u);
        let u3 = fmul(u2, u);
        let (u, u2, u3) = (u as f64, u2 as f64, u3 as f64);
        let i0 = seg_k.saturating_sub(1);
        let i3 = (m - 1).min(seg_k + 2);
        let w0 = dsub(dadd(dmul(-0.5, u3), u2), dmul(0.5, u));
        let w1 = dadd(dsub(dmul(1.5, u3), dmul(2.5, u2)), 1.0);
        let w2 = dadd(dadd(dmul(-1.5, u3), dmul(2.0, u2)), dmul(0.5, u));
        let w3 = dsub(dmul(0.5, u3), dmul(0.5, u2));
        let (s0, s1, s2, s3) = unsafe { (point(s, i0), point(s, seg_k), point(s, seg_k + 1), point(s, i3)) };
        let val = |d: usize| {
            let v = dadd(
                dadd(dadd(dmul(w0, s0[d] as f64), dmul(w1, s1[d] as f64)), dmul(w2, s2[d] as f64)),
                dmul(w3, s3[d] as f64),
            );
            py_min(py_max(v, 0.0), upper[d]) as f32
        };
        unsafe { set_point(out, j, [val(0), val(1), val(2)]) };
        j += 1;
    }
}
