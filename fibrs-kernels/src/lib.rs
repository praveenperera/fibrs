//! cuda-oxide device kernels for `fibrs --device cuda`: the reference's centrality and ridge seeds
//! (`centrality_kernel` and `seed_kernel`, cuda.py:102-203 of AVF_SPLINE_GENERATOR at bee798f7)
//!
//! `scripts/build-ptx.sh` compiles this crate to `ptx/fibrs.sm80.ptx`, which fibrs embeds and loads by kernel
//! name. The kernels take raw pointers and scalars (C-like entries); fibrs's `src/gpu.rs` checks every launch
//! against the PTX param lists.
//!
//! The volumes arrive as the host's 16³ bricks: one pool of brick payloads and, per volume, a table from each
//! brick of a box of the brick grid to its slot in the pool, or [`NO_BRICK`] where every voxel reads as the
//! volume's default. Per tile, [`fb_gather`] writes the tile's region (the reference's 128³ tile with its 8-voxel
//! halo) densely, [`fb_centrality`] computes centrality over it, and [`fb_seeds`] tests the tile core for seeds,
//! appends them to a list, copies the core's centrality out in brick order and flags the bricks holding a nonzero
//! value. [`fb_slots`] and [`fb_pack`] then pack the flagged bricks densely, so the host downloads only those:
//! most of a band's centrality is zero. Coordinates and clamps are local to the region, as in the reference.
//!
//! The arithmetic is the reference's CUDA rounding, operation by operation: every float64 multiply, add and
//! multiply-add is an explicit round-to-nearest instruction (`mul.rn`, `add.rn`, `fma.rn`), which neither LLVM
//! nor ptxas contracts, placed where Numba's PTX and the SASS ptxas builds from it put theirs. The host's
//! `stage1::<Fused>` (src/stage1.rs) is the same computation on the CPU, and the GPU tests hold the two equal
//!
//! The `fb_ax_*` kernels compute the direction field (`axes::direction_field`, src/axes.rs) on the CPU's tiles:
//! 64³ cores with an 11-voxel halo, each box clipped to the volume. [`fb_ax_gather`] writes both tracing channels
//! of each box densely and lists each tile's present core voxels per family; per family, [`fb_ax_smooth0`] and
//! [`fb_ax_smooth`] apply the σ 0.9 Gaussian, [`fb_ax_grad`] takes `np.gradient`, [`fb_ax_tensor_z`] and
//! [`fb_ax_tensor_y`] smooth the six gradient products along z and y, and [`fb_ax_eigen`] smooths them along x at
//! each listed voxel and certifies its quantized eigenvector ([`eigen::certify`]), handing the voxels it cannot
//! certify back to the host's `dsyevd`. Each filter runs over the whole box with the CPU's edge rules, and every
//! float32 and float64 operation is an explicit round-to-nearest instruction in the CPU's order, so the bytes
//! equal the CPU's
//!
//! The `fb_walk_*` kernels trace chunks of a family's seeds ahead of the host's commit point (`src/gpu/cuda/walk.rs`
//! drives them, `src/lookahead.rs` uses the results). The per-seed tracer is [`walk`], generic over its arithmetic
//! and its reads, and the host includes the same source (`src/walk.rs`), where the tests hold it equal to
//! `kernels::trace_seed` seed by seed. Per chunk, [`walk_dev::fb_walk_seeds`] runs each seed's start and lists the
//! survivors, [`walk_dev::fb_walk_trace`] traces both halves of each survivor up to a step cap,
//! [`walk_dev::fb_walk_spans`] cuts the joined path into accepted curves, three [`walk_dev::fb_walk_scan`] runs lay
//! the chunk's checks, points and curves out in seed order for [`walk_dev::fb_walk_gather`], and
//! [`walk_dev::fb_walk_results`] gives each seed its ranges, or hands it back to the CPU when a half reached the
//! cap, its spans failed or a buffer ran out. The host marks the coverage from a result's points itself. Every coverage read is recorded with the value it
//! saw, so the host can check a result against the coverage it has committed since. Float64 and float32
//! operations carry explicit rounding modes, as the CPU tracer's plain Rust arithmetic never contracts

#![no_std]
#![allow(clippy::missing_safety_doc)]

use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::float::{add_rn_f32, add_rn_f64, div_rn_f64, fma_rn_f64, mul_rn_f32, mul_rn_f64, sqrt_rn_f64};
use cuda_device::kernel;
use cuda_device::thread::{__unroll_config, blockDim_x, blockIdx_x, blockIdx_y, threadIdx_x};

pub mod eigen;

/// Full unrolling of the loop whose body it starts, for the walk core shared with the host (which defines it as
/// nothing)
macro_rules! unroll_full {
    () => {
        cuda_device::thread::__unroll_config::<0>()
    };
}

pub mod walk;
mod walk_dev;
pub mod walk_layout;

use eigen::{Ops, certify};

/// Brick edge as a shift; the host's `vol::SHIFT`
const SHIFT: u32 = 4;
const MASK: u32 = (1 << SHIFT) - 1;
/// Voxels per brick
const BRICK: u32 = 1 << (3 * SHIFT);

/// A table entry for a brick with no payload: every voxel reads as the volume's default
pub const NO_BRICK: u32 = u32::MAX;

/// u32 fields per tile descriptor
pub const DESC: usize = 16;

/// One tile of a launch, from its descriptor: region origin (global), region extent, the core's offset in the
/// region and extent, the region's first voxel in the scratch arrays and the core's first element in the
/// output bricks
#[derive(Clone, Copy)]
struct Tile {
    lo: [u32; 3],
    dims: [i32; 3],
    off: [i32; 3],
    core: [i32; 3],
    base: u32,
    out: u32,
}

#[inline(always)]
unsafe fn tile(desc: *const u32, t: u32) -> Tile {
    let d = unsafe { desc.add(t as usize * DESC) };
    let at = |i: usize| unsafe { *d.add(i) };
    Tile {
        lo: [at(0), at(1), at(2)],
        dims: [at(3) as i32, at(4) as i32, at(5) as i32],
        off: [at(6) as i32, at(7) as i32, at(8) as i32],
        core: [at(9) as i32, at(10) as i32, at(11) as i32],
        base: at(12),
        out: at(13),
    }
}

impl Tile {
    /// Scratch index of the region voxel at `z, y, x`
    #[inline(always)]
    fn idx(&self, z: i32, y: i32, x: i32) -> usize {
        (self.base + ((z * self.dims[1] + y) * self.dims[2] + x) as u32) as usize
    }
}

#[inline(always)]
fn mul(a: f64, b: f64) -> f64 {
    mul_rn_f64(a, b)
}

#[inline(always)]
fn add(a: f64, b: f64) -> f64 {
    add_rn_f64(a, b)
}

#[inline(always)]
fn sub(a: f64, b: f64) -> f64 {
    add_rn_f64(a, -b)
}

/// `a * b + c`, rounded once
#[inline(always)]
fn mad(a: f64, b: f64, c: f64) -> f64 {
    fma_rn_f64(a, b, c)
}

#[inline(always)]
fn abs(v: f64) -> f64 {
    if v < 0.0 { -v } else { v }
}

/// Python's two-argument `max`: the second argument wins only when strictly larger
#[inline(always)]
fn py_max(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

/// Python's two-argument `min`: the second argument wins only when strictly smaller
#[inline(always)]
fn py_min(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

/// `decode_dir` from the host's table of all byte pairs (`nx << 8 | ny`, four float64 per entry), computed in
/// the reference's CUDA rounding
#[inline(always)]
unsafe fn dir(dirs: *const f64, nx: u8, ny: u8) -> [f64; 3] {
    let i = ((nx as usize) << 8 | ny as usize) * 4;
    unsafe { [*dirs.add(i), *dirs.add(i + 1), *dirs.add(i + 2)] }
}

/// The clamp and cell split of the trilinear helpers (cuda.py:26-44)
#[inline(always)]
fn cell(t: &Tile, mut x: f64, mut y: f64, mut z: f64) -> ([i32; 3], [f32; 3]) {
    let ww = sub(t.dims[2] as f64, 1.001);
    let hh = sub(t.dims[1] as f64, 1.001);
    let dd = sub(t.dims[0] as f64, 1.001);
    if x < 0.0 {
        x = 0.0;
    } else if x > ww {
        x = ww as f32 as f64;
    }
    if y < 0.0 {
        y = 0.0;
    } else if y > hh {
        y = hh as f32 as f64;
    }
    if z < 0.0 {
        z = 0.0;
    } else if z > dd {
        z = dd as f32 as f64;
    }
    let (ix, iy, iz) = (x as i32, y as i32, z as i32);
    let fx = sub(x, ix as f64) as f32;
    let fy = sub(y, iy as f64) as f32;
    let fz = sub(z, iz as f64) as f32;
    ([iz, iy, ix], [fz, fy, fx])
}

/// `(1 - fz) * ((1 - fy) * p00 + fy * p01) + fz * ((1 - fy) * p10 + fy * p11)`
#[inline(always)]
fn blend(p00: f64, p01: f64, p10: f64, p11: f64, fy: f32, fz: f32) -> f64 {
    let (fy, fz) = (fy as f64, fz as f64);
    let lo = mad(sub(1.0, fy), p00, mul(p01, fy));
    let hi = mad(sub(1.0, fy), p10, mul(p11, fy));
    mad(sub(1.0, fz), lo, mul(hi, fz))
}

/// One x-lerp of `trilin_scalar_u8`; `fx * arr[..]` is a float32 product
#[inline(always)]
unsafe fn lerp_u8(a: *const u8, t: &Tile, z: i32, y: i32, ix: i32, fx: f32) -> f64 {
    let (a0, a1) = unsafe { (*a.add(t.idx(z, y, ix)), *a.add(t.idx(z, y, ix + 1))) };
    mad(sub(1.0, fx as f64), a0 as f64, mul_rn_f32(fx, a1 as f32) as f64)
}

/// `trilin_scalar_u8` (cuda.py:25)
#[inline(always)]
unsafe fn trilin_u8(a: *const u8, t: &Tile, x: f64, y: f64, z: f64) -> f64 {
    let ([iz, iy, ix], [fz, fy, fx]) = cell(t, x, y, z);
    unsafe {
        let p00 = lerp_u8(a, t, iz, iy, ix, fx);
        let p01 = lerp_u8(a, t, iz, iy + 1, ix, fx);
        let p10 = lerp_u8(a, t, iz + 1, iy, ix, fx);
        let p11 = lerp_u8(a, t, iz + 1, iy + 1, ix, fx);
        blend(p00, p01, p10, p11, fy, fz)
    }
}

/// One x-lerp of `trilin_scalar_u16`, all float64
#[inline(always)]
unsafe fn lerp_u16(a: *const u16, t: &Tile, z: i32, y: i32, ix: i32, fx: f64) -> f64 {
    let (a0, a1) = unsafe { (*a.add(t.idx(z, y, ix)), *a.add(t.idx(z, y, ix + 1))) };
    mad(sub(1.0, fx), a0 as f64, mul(fx, a1 as f64))
}

/// `trilin_scalar_u16` (cuda.py:53) before its final `0.01` scale, which the seed test fuses into a sum
#[inline(always)]
unsafe fn trilin_u16_unscaled(a: *const u16, t: &Tile, x: f64, y: f64, z: f64) -> f64 {
    let ([iz, iy, ix], [fz, fy, fx]) = cell(t, x, y, z);
    let fx = fx as f64;
    unsafe {
        let p00 = lerp_u16(a, t, iz, iy, ix, fx);
        let p01 = lerp_u16(a, t, iz, iy + 1, ix, fx);
        let p10 = lerp_u16(a, t, iz + 1, iy, ix, fx);
        let p11 = lerp_u16(a, t, iz + 1, iy + 1, ix, fx);
        blend(p00, p01, p10, p11, fy, fz)
    }
}

/// `float32(0.01)`, the scale of `trilin_scalar_u16`
const CENT_SCALE: f64 = 0.01f32 as f64;

/// `ortho_basis` (cuda.py:83); ptxas fuses each cross-product component's first product
#[inline(always)]
fn ortho_basis(ux: f64, uy: f64, uz: f64) -> [f64; 6] {
    let (e1x, e1y, e1z) = if abs(uz) < 0.85 { (-uy, ux, 0.0) } else { (0.0, -uz, uy) };
    let n1 = sqrt_rn_f64(mad(e1z, e1z, mad(e1x, e1x, mul(e1y, e1y))));
    let (e1x, e1y, e1z) = (div_rn_f64(e1x, n1), div_rn_f64(e1y, n1), div_rn_f64(e1z, n1));
    let e2x = mad(uy, e1z, -mul(uz, e1y));
    let e2y = mad(uz, e1x, -mul(ux, e1z));
    let e2z = mad(ux, e1y, -mul(uy, e1x));
    [e1x, e1y, e1z, e2x, e2y, e2z]
}

// the reference's literals, rounded to float32 as Numba does
#[allow(clippy::approx_constant)]
const COS_A: [f32; 8] = [1.0, 0.70710678, 0.0, -0.70710678, -1.0, -0.70710678, 0.0, 0.70710678];
#[allow(clippy::approx_constant)]
const SIN_A: [f32; 8] = [0.0, 0.70710678, 1.0, 0.70710678, 0.0, -0.70710678, -1.0, -0.70710678];
/// Ring radii and their weights over 8 (`wr / 8.0`, which the reference computes as `wr * 0.125`)
const RADII: [f64; 4] = [1.0, 1.8, 2.7, 3.6];
const WEIGHTS: [f64; 4] = [0.5, 0.28, 0.14, 0.06];

/// Direction `k` of the ring in the plane of `e1`, `e2`
#[inline(always)]
fn ring_dir(b: &[f64; 6], k: usize) -> [f64; 3] {
    let (ca, sa) = (COS_A[k] as f64, SIN_A[k] as f64);
    [mad(b[0], ca, mul(b[3], sa)), mad(b[1], ca, mul(b[4], sa)), mad(b[2], ca, mul(b[5], sa))]
}

/// The volumes' brick tables: `table[v * nb + brick]` is volume `v`'s pool slot for a brick of the box of
/// `nb` bricks from brick `b0` (z, y, x), `nby` x `nbx` bricks per z row
#[derive(Clone, Copy)]
struct Bricks {
    pool: *const u8,
    table: *const u32,
    nb: u32,
    b0: [u32; 3],
    nby: u32,
    nbx: u32,
}

impl Bricks {
    #[inline(always)]
    unsafe fn get(&self, v: u32, z: u32, y: u32, x: u32, default: u8) -> u8 {
        let (bz, by, bx) = ((z >> SHIFT) - self.b0[0], (y >> SHIFT) - self.b0[1], (x >> SHIFT) - self.b0[2]);
        let b = (bz * self.nby + by) * self.nbx + bx;
        let slot = unsafe { *self.table.add((v * self.nb + b) as usize) };
        if slot == NO_BRICK {
            return default;
        }
        let at = ((z & MASK) << (2 * SHIFT)) | ((y & MASK) << SHIFT) | (x & MASK);
        unsafe { *self.pool.add(slot as usize * BRICK as usize + at as usize) }
    }
}

/// Write each tile's region densely: the family's presence (`min(own + other, 255)`, zero where `own < other`)
/// from the two pooled predictions (volumes 0 and 1, default 0) and the direction bytes (volumes 2 and 3,
/// default 128); every tile's region lies in the tables' box
///
/// Grid: x over region voxels in blocks, y over tiles
#[kernel]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fb_gather(
    desc: *const u32,
    pool: *const u8,
    table: *const u32,
    nb: u32,
    bz0: u32,
    by0: u32,
    bx0: u32,
    nby: u32,
    nbx: u32,
    family: u32,
    pres: *mut u8,
    nx: *mut u8,
    ny: *mut u8,
) {
    let t = unsafe { tile(desc, blockIdx_y()) };
    let i = blockIdx_x() * blockDim_x() + threadIdx_x();
    let (h, w) = (t.dims[1] as u32, t.dims[2] as u32);
    if i >= t.dims[0] as u32 * h * w {
        return;
    }
    let z = i / (h * w);
    let y = i / w - z * h;
    let x = i - (z * h + y) * w;
    let (gz, gy, gx) = (t.lo[0] + z, t.lo[1] + y, t.lo[2] + x);
    let b = Bricks { pool, table, nb, b0: [bz0, by0, bx0], nby, nbx };
    let at = (t.base + i) as usize;
    unsafe {
        let own = b.get(family - 1, gz, gy, gx, 0);
        let p = if own == 0 {
            0
        } else {
            let other = b.get(2 - family, gz, gy, gx, 0);
            if own < other { 0 } else { (own as u32 + other as u32).min(255) as u8 }
        };
        *pres.add(at) = p;
        *nx.add(at) = b.get(2, gz, gy, gx, 128);
        *ny.add(at) = b.get(3, gz, gy, gx, 128);
    }
}

/// Centrality over each tile's band, the voxels within 3 of its core, which are all the seed test reads
/// (`centrality_kernel`, cuda.py:103); voxels below presence 64 get 0
///
/// Grid: x over band voxels in blocks, y over tiles
#[kernel]
pub unsafe fn fb_centrality(desc: *const u32, pres: *const u8, nx: *const u8, ny: *const u8, dirs: *const f64, cent: *mut u16) {
    let t = unsafe { tile(desc, blockIdx_y()) };
    let lo = [0, 1, 2].map(|a| (t.off[a] - 3).max(0));
    let ext = [0, 1, 2].map(|a| (t.off[a] + t.core[a] + 3).min(t.dims[a]) - lo[a]);
    let i = blockIdx_x() * blockDim_x() + threadIdx_x();
    let (eh, ew) = (ext[1] as u32, ext[2] as u32);
    if i >= ext[0] as u32 * eh * ew {
        return;
    }
    let bz = i / (eh * ew);
    let by = i / ew - bz * eh;
    let bx = i - (bz * eh + by) * ew;
    let (z, y, x) = (lo[0] + bz as i32, lo[1] + by as i32, lo[2] + bx as i32);
    let v = unsafe { centrality_at(&t, pres, nx, ny, dirs, z, y, x) };
    unsafe { *cent.add(t.idx(z, y, x)) = v };
}

#[inline(always)]
unsafe fn centrality_at(t: &Tile, pres: *const u8, nx: *const u8, ny: *const u8, dirs: *const f64, z: i32, y: i32, x: i32) -> u16 {
    let c = t.idx(z, y, x);
    let p0 = unsafe { *pres.add(c) } as f32;
    if p0 < 64.0 {
        return 0;
    }
    let (wf, hf, df) = (sub(t.dims[2] as f64, 1.001), sub(t.dims[1] as f64, 1.001), sub(t.dims[0] as f64, 1.001));
    let (xf, yf, zf) = (x as f64, y as f64, z as f64);
    let [ux, uy, uz] = unsafe { dir(dirs, *nx.add(c), *ny.add(c)) };
    let p_p = unsafe { trilin_u8(pres, t, add(xf, ux), add(yf, uy), add(zf, uz)) };
    let p_m = unsafe { trilin_u8(pres, t, sub(xf, ux), sub(yf, uy), sub(zf, uz)) };
    let p_par = mad(p_m, 0.25, mad(p0 as f64, 0.5, mul(p_p, 0.25)));
    let basis = ortho_basis(ux, uy, uz);
    let mut c_sum = 0.0f64;
    let mut k = 0;
    while k < 8 {
        let [dx, dy, dz] = ring_dir(&basis, k);
        let mut j = 0;
        while j < 4 {
            let r = RADII[j];
            let xx = py_min(py_max(mad(dx, r, xf), 0.0), wf);
            let yy = py_min(py_max(mad(dy, r, yf), 0.0), hf);
            let zz = py_min(py_max(mad(dz, r, zf), 0.0), df);
            let pr = unsafe { trilin_u8(pres, t, xx, yy, zz) };
            let n = t.idx(add(zz, 0.5) as i32, add(yy, 0.5) as i32, add(xx, 0.5) as i32);
            let [ux2, uy2, uz2] = unsafe { dir(dirs, *nx.add(n), *ny.add(n)) };
            let dr = abs(mad(uz, uz2, mad(uy, uy2, mul(ux, ux2))));
            if dr > 0.85 {
                c_sum = mad(mul(mul(WEIGHTS[j], 0.125), pr), div_rn_f64(sub(dr, 0.85), 0.15), c_sum);
            }
            j += 1;
        }
        k += 1;
    }
    py_min(py_max(mad(add(p_par, c_sum), 100.0, 0.5), 0.0), 65535.0) as u16
}

/// Seeds of each tile's core (`seed_kernel`, cuda.py:157, seed stride 2), appended to `seeds` as
/// `global C-order index << 8 | priority` through `count` (entries past `cap` are counted, not written, and the
/// host fails the run); also copies the core's centrality to `out` in brick order (`ceil(core / 16)` bricks per
/// axis from the tile's `out` element, 4096 elements each) and sets `flags[brick]` for each of those bricks
/// holding a nonzero value
///
/// Grid: x over core voxels in blocks, y over tiles
#[kernel]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fb_seeds(
    desc: *const u32,
    pres: *const u8,
    nx: *const u8,
    ny: *const u8,
    dirs: *const f64,
    cent: *const u16,
    shape_h: u32,
    shape_w: u32,
    out: *mut u16,
    flags: *mut u32,
    seeds: *mut u64,
    cap: u32,
    count: *mut u32,
) {
    let t = unsafe { tile(desc, blockIdx_y()) };
    let (ch, cw) = (t.core[1] as u32, t.core[2] as u32);
    let i = blockIdx_x() * blockDim_x() + threadIdx_x();
    if i >= t.core[0] as u32 * ch * cw {
        return;
    }
    let cz = i / (ch * cw);
    let cy = i / cw - cz * ch;
    let cx = i - (cz * ch + cy) * cw;
    let (z, y, x) = (t.off[0] + cz as i32, t.off[1] + cy as i32, t.off[2] + cx as i32);
    let c = unsafe { *cent.add(t.idx(z, y, x)) };
    let (nby, nbx) = ((ch + MASK) >> SHIFT, (cw + MASK) >> SHIFT);
    let brick = ((cz >> SHIFT) * nby + (cy >> SHIFT)) * nbx + (cx >> SHIFT);
    let at = ((cz & MASK) << (2 * SHIFT)) | ((cy & MASK) << SHIFT) | (cx & MASK);
    let first = t.out / BRICK;
    unsafe { *out.add(t.out as usize + (brick * BRICK + at) as usize) = c };
    if c != 0 {
        // many threads mark the same brick; a relaxed atomic store makes those concurrent writes well defined
        // SAFETY: `flags` holds one u32 per output brick of the unit, zeroed by the host
        unsafe { DeviceAtomicU32::from_ptr(flags.add((first + brick) as usize)).store(1, AtomicOrdering::Relaxed) };
    }
    let interior = z >= 1 && y >= 1 && x >= 1 && z < t.dims[0] - 1 && y < t.dims[1] - 1 && x < t.dims[2] - 1;
    if !interior {
        return;
    }
    let s = unsafe { seed_at(&t, pres, nx, ny, dirs, cent, z, y, x) };
    if s == 0 {
        return;
    }
    let (gz, gy, gx) = ((t.lo[0] + z as u32) as u64, (t.lo[1] + y as u32) as u64, (t.lo[2] + x as u32) as u64);
    let index = (gz * shape_h as u64 + gy) * shape_w as u64 + gx;
    // SAFETY: `count` is one u32 the host zeroed; device-scope relaxed atomics only hand out distinct slots
    let slot = unsafe { DeviceAtomicU32::from_ptr(count).fetch_add(1, AtomicOrdering::Relaxed) };
    if slot < cap {
        unsafe { *seeds.add(slot as usize) = index << 8 | s as u64 };
    }
}

#[inline(always)]
unsafe fn seed_at(
    t: &Tile,
    pres: *const u8,
    nx: *const u8,
    ny: *const u8,
    dirs: *const f64,
    cent: *const u16,
    z: i32,
    y: i32,
    x: i32,
) -> u8 {
    let c = t.idx(z, y, x);
    if unsafe { *pres.add(c) } < 79 || (x + y + z) & 1 != 0 {
        return 0;
    }
    let c0 = mul_rn_f32(unsafe { *cent.add(c) } as f32, 0.01f32);
    let c0d = c0 as f64;
    let (xf, yf, zf) = (x as f64, y as f64, z as f64);
    let [ux, uy, uz] = unsafe { dir(dirs, *nx.add(c), *ny.add(c)) };
    let basis = ortho_basis(ux, uy, uz);
    let mut n_max = 0;
    let mut k = 0;
    while k < 4 {
        let [dx, dy, dz] = ring_dir(&basis, k);
        k += 1;
        let tp = unsafe { trilin_u16_unscaled(cent, t, mad(dx, 1.15, xf), mad(dy, 1.15, yf), mad(dz, 1.15, zf)) };
        let tm = unsafe { trilin_u16_unscaled(cent, t, mad(dx, -1.15, xf), mad(dy, -1.15, yf), mad(dz, -1.15, zf)) };
        if !(c0d >= mul(CENT_SCALE, tp) && c0d >= mul(CENT_SCALE, tm)) {
            continue;
        }
        let (dx2, dy2, dz2) = (add(dx, dx), add(dy, dy), add(dz, dz));
        let tp2 = unsafe { trilin_u16_unscaled(cent, t, add(xf, dx2), add(yf, dy2), add(zf, dz2)) };
        let tm2 = unsafe { trilin_u16_unscaled(cent, t, sub(xf, dx2), sub(yf, dy2), sub(zf, dz2)) };
        // 0.5 * (cp2 + cm2) + 2.0, with cp2's scale fused into the sum
        if c0d >= mad(mad(tp2, CENT_SCALE, mul(tm2, CENT_SCALE)), 0.5, 2.0) {
            n_max += 1;
        }
    }
    if n_max == 4 {
        2
    } else if n_max == 3 && c0 >= 185.0 {
        1
    } else {
        0
    }
}

/// A slot in the packed output for each flagged brick, in no particular order: `slots[b]` is brick `b`'s slot or
/// [`NO_BRICK`], and `ids[slot]` is the brick; `count` ends as the number of flagged bricks
///
/// Grid: x over the `n` bricks
#[kernel]
pub unsafe fn fb_slots(flags: *const u32, n: u32, slots: *mut u32, ids: *mut u32, count: *mut u32) {
    let b = blockIdx_x() * blockDim_x() + threadIdx_x();
    if b >= n {
        return;
    }
    if unsafe { *flags.add(b as usize) } == 0 {
        unsafe { *slots.add(b as usize) = NO_BRICK };
        return;
    }
    // SAFETY: `count` is one u32 the host zeroed; device-scope relaxed atomics only hand out distinct slots
    let slot = unsafe { DeviceAtomicU32::from_ptr(count).fetch_add(1, AtomicOrdering::Relaxed) };
    unsafe {
        *slots.add(b as usize) = slot;
        *ids.add(slot as usize) = b;
    }
}

/// Copy each flagged brick of `out` to its slot of `packed`
///
/// Grid: x over the `n` voxels of `out` (4096 per brick)
#[kernel]
pub unsafe fn fb_pack(out: *const u16, n: u32, slots: *const u32, packed: *mut u16) {
    let i = blockIdx_x() * blockDim_x() + threadIdx_x();
    if i >= n {
        return;
    }
    let slot = unsafe { *slots.add((i / BRICK) as usize) };
    if slot != NO_BRICK {
        unsafe { *packed.add((slot * BRICK + i % BRICK) as usize) = *out.add(i as usize) };
    }
}

/// SciPy's Gaussian weights for σ 0.9 (radius 4) and σ 1.5 (radius 6), outermost first, centre last: the float64
/// bit patterns of `W09` and `W15` in src/axes.rs
const W09: [f64; 5] = [
    f64::from_bits(0x3ef7dff10d1ee066),
    f64::from_bits(0x3f5c138a6b624c17),
    f64::from_bits(0x3fa336a4e8e9e375),
    f64::from_bits(0x3fce9aeb020a6f44),
    f64::from_bits(0x3fdc5e85af5bea50),
];
const W15: [f64; 7] = [
    f64::from_bits(0x3f17638649a0fcd1),
    f64::from_bits(0x3f50d891ab53680a),
    f64::from_bits(0x3f7f1e741f80c9b8),
    f64::from_bits(0x3fa26dd9e7a284e1),
    f64::from_bits(0x3fbbfdc88289323c),
    f64::from_bits(0x3fcb42851d1c8a96),
    f64::from_bits(0x3fd1058ef29a59d7),
];

/// The certification's arithmetic in explicit round-to-nearest instructions
struct Rn;

impl Ops for Rn {
    #[inline(always)]
    fn mul(a: f64, b: f64) -> f64 {
        mul_rn_f64(a, b)
    }

    #[inline(always)]
    fn add(a: f64, b: f64) -> f64 {
        add_rn_f64(a, b)
    }

    #[inline(always)]
    fn div(a: f64, b: f64) -> f64 {
        div_rn_f64(a, b)
    }

    #[inline(always)]
    fn sqrt(a: f64) -> f64 {
        sqrt_rn_f64(a)
    }

    #[inline(always)]
    fn mul32(a: f32, b: f32) -> f32 {
        mul_rn_f32(a, b)
    }

    #[inline(always)]
    fn add32(a: f32, b: f32) -> f32 {
        add_rn_f32(a, b)
    }
}

impl Tile {
    /// Voxels in the box
    #[inline(always)]
    fn len(&self) -> u32 {
        (self.dims[0] * self.dims[1] * self.dims[2]) as u32
    }

    /// Box coordinates of the box voxel `i`
    #[inline(always)]
    fn voxel(&self, i: u32) -> (i32, i32, i32) {
        let (h, w) = (self.dims[1] as u32, self.dims[2] as u32);
        let z = i / (h * w);
        let y = i / w - z * h;
        let x = i - (z * h + y) * w;
        (z as i32, y as i32, x as i32)
    }

    /// Core coordinates of the core voxel with core-local index `i`, `(cz·ch + cy)·cw + cx`
    #[inline(always)]
    fn core_voxel(&self, i: u32) -> (u32, u32, u32) {
        let (ch, cw) = (self.core[1] as u32, self.core[2] as u32);
        let cz = i / (ch * cw);
        let cy = i / cw - cz * ch;
        let cx = i - (cz * ch + cy) * cw;
        (cz, cy, cx)
    }

    /// The core voxel's element in the output bricks, as [`fb_seeds`] lays them out
    #[inline(always)]
    fn out_at(&self, cz: u32, cy: u32, cx: u32) -> usize {
        let (ch, cw) = (self.core[1] as u32, self.core[2] as u32);
        let (nby, nbx) = ((ch + MASK) >> SHIFT, (cw + MASK) >> SHIFT);
        let brick = ((cz >> SHIFT) * nby + (cy >> SHIFT)) * nbx + (cx >> SHIFT);
        let at = ((cz & MASK) << (2 * SHIFT)) | ((cy & MASK) << SHIFT) | (cx & MASK);
        self.out as usize + (brick * BRICK + at) as usize
    }
}

/// Whether tile `t` lists any present voxel of `family`; the per-family kernels skip a tile that lists none
#[inline(always)]
unsafe fn has_family(count: *const u32, t: u32, family: u32) -> bool {
    unsafe { *count.add((2 * t + family) as usize) != 0 }
}

/// The sample SciPy's "reflect" mode reads for position `i` of a line of `n` samples: half-sample symmetric,
/// repeated with period `2n` (`reflect` in src/axes.rs), so a line shorter than the kernel radius reflects more
/// than once; each step mirrors about the nearer edge, which maps `i` to the same sample as `i mod 2n` does
#[inline(always)]
fn reflect(mut i: i32, n: i32) -> i32 {
    while i < 0 || i >= n {
        i = if i < 0 { -1 - i } else { 2 * n - 1 - i };
    }
    i
}

/// One output of SciPy's `correlate1d` with the symmetric kernel `w` (outermost first, centre last) at position
/// `c` of a line of `n` samples, `read(k)` giving sample `k`: `line[c]·centre`, then `(line[c − off] +
/// line[c + off])·w[j]` added for each outer weight in turn, in float64, rounded to float32 at the end
/// (`correlate_rows` in src/axes.rs)
///
/// `w` is a reference to a constant, a read-only global: taken by value, the array stays in local memory when
/// the tap loop is not unrolled
#[inline(always)]
fn correlate<const N: usize>(w: &'static [f64; N], c: i32, n: i32, read: impl Fn(i32) -> f32) -> f32 {
    let r = N - 1;
    let mut acc = mul_rn_f64(read(c) as f64, w[r]);
    let mut j = 0;
    while j < r {
        __unroll_config::<0>();
        let off = (r - j) as i32;
        let pair = add_rn_f64(read(reflect(c - off, n)) as f64, read(reflect(c + off, n)) as f64);
        acc = add_rn_f64(acc, mul_rn_f64(pair, w[j]));
        j += 1;
    }
    acc as f32
}

/// Write both tracing channels of each tile's box densely (`raw[c·total + base + i]`, volumes 0 and 1 of the
/// tables, default 0), and list each tile's present core voxels, those with `c0 + c1 ≥ 40`, per family (1 where
/// `c1 > c0`, else 0): `list[(2t + f)·cap + k]` is the core-local index `(cz·ch + cy)·cw + cx` of the family's
/// `k`th voxel, in no particular order, and `count[2t + f]` (zeroed by the host) ends as the number listed. `cap`
/// is at least every tile's core size, so a list cannot overflow; were it smaller, voxels past `cap` would be
/// counted, not written, and [`fb_ax_eigen`], whose grid covers `cap` entries, would read only written ones
///
/// Grid: x over box voxels in blocks, y over tiles
#[kernel]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fb_ax_gather(
    desc: *const u32,
    pool: *const u8,
    table: *const u32,
    nb: u32,
    bz0: u32,
    by0: u32,
    bx0: u32,
    nby: u32,
    nbx: u32,
    total: u32,
    cap: u32,
    raw: *mut u8,
    count: *mut u32,
    list: *mut u32,
) {
    let ti = blockIdx_y();
    let t = unsafe { tile(desc, ti) };
    let i = blockIdx_x() * blockDim_x() + threadIdx_x();
    if i >= t.len() {
        return;
    }
    let (z, y, x) = t.voxel(i);
    let (gz, gy, gx) = (t.lo[0] + z as u32, t.lo[1] + y as u32, t.lo[2] + x as u32);
    let b = Bricks { pool, table, nb, b0: [bz0, by0, bx0], nby, nbx };
    let (c0, c1) = unsafe { (b.get(0, gz, gy, gx, 0), b.get(1, gz, gy, gx, 0)) };
    let at = (t.base + i) as usize;
    unsafe {
        *raw.add(at) = c0;
        *raw.add(total as usize + at) = c1;
    }

    let (cz, cy, cx) = (z - t.off[0], y - t.off[1], x - t.off[2]);
    let core = cz >= 0 && cy >= 0 && cx >= 0 && cz < t.core[0] && cy < t.core[1] && cx < t.core[2];
    if !core || (c0 as u32 + c1 as u32) < 40 {
        return;
    }
    let slot = 2 * ti + (c1 > c0) as u32;
    // SAFETY: `count` holds 2 u32 per tile the host zeroed; device-scope relaxed atomics only hand out distinct
    // slots
    let k = unsafe { DeviceAtomicU32::from_ptr(count.add(slot as usize)).fetch_add(1, AtomicOrdering::Relaxed) };
    if k >= cap {
        return;
    }
    let local = (cz * t.core[1] + cy) * t.core[2] + cx;
    unsafe { *list.add((slot * cap + k) as usize) = local as u32 };
}

/// The σ 0.9 Gaussian along z of the family's channel as float32 (`unit[v] = v as f32 / 255.0`, the host's
/// table): `raw` box → `dst` box
///
/// Grid: x over box voxels in blocks, y over tiles
#[kernel]
pub unsafe fn fb_ax_smooth0(
    desc: *const u32,
    count: *const u32,
    family: u32,
    total: u32,
    raw: *const u8,
    unit: *const f32,
    dst: *mut f32,
) {
    let ti = blockIdx_y();
    if unsafe { !has_family(count, ti, family) } {
        return;
    }
    let t = unsafe { tile(desc, ti) };
    let i = blockIdx_x() * blockDim_x() + threadIdx_x();
    if i >= t.len() {
        return;
    }
    let (z, y, x) = t.voxel(i);
    let channel = unsafe { raw.add((family * total) as usize) };
    let v = correlate(&W09, z, t.dims[0], |k| unsafe { *unit.add(*channel.add(t.idx(k, y, x)) as usize) });
    unsafe { *dst.add((t.base + i) as usize) = v };
}

/// The σ 0.9 Gaussian along `axis` (1 or 2) of a float32 box: `src` → `dst`
///
/// Grid: x over box voxels in blocks, y over tiles
#[kernel]
pub unsafe fn fb_ax_smooth(
    desc: *const u32,
    count: *const u32,
    family: u32,
    axis: u32,
    src: *const f32,
    dst: *mut f32,
) {
    let ti = blockIdx_y();
    if unsafe { !has_family(count, ti, family) } {
        return;
    }
    let t = unsafe { tile(desc, ti) };
    let i = blockIdx_x() * blockDim_x() + threadIdx_x();
    if i >= t.len() {
        return;
    }
    let (z, y, x) = t.voxel(i);
    let v = if axis == 1 {
        correlate(&W09, y, t.dims[1], |k| unsafe { *src.add(t.idx(z, k, x)) })
    } else {
        correlate(&W09, x, t.dims[2], |k| unsafe { *src.add(t.idx(z, y, k)) })
    };
    unsafe { *dst.add((t.base + i) as usize) = v };
}

/// `np.gradient` along one axis at index `k` of a line of `n` samples, `f(j)` giving sample `j`: one-sided at
/// the ends, central inside, in float32 (`gradient` in src/axes.rs; its `/ 2.0` is the same correctly rounded
/// value as `· 0.5`, and its `/ 1.0` is exact)
///
/// A box axis is 1 voxel only on a 1-voxel volume axis, which `np.gradient` refuses and the CPU's `gradient`
/// indexes past; it gives 0 here rather than reading outside the box
#[inline(always)]
fn gradient(k: i32, n: i32, f: impl Fn(i32) -> f32) -> f32 {
    if n < 2 {
        0.0
    } else if k == 0 {
        add_rn_f32(f(1), -f(0))
    } else if k == n - 1 {
        add_rn_f32(f(k), -f(k - 1))
    } else {
        mul_rn_f32(add_rn_f32(f(k + 1), -f(k - 1)), 0.5)
    }
}

/// The gradient of the smoothed box `s` along each axis `a`: `g[a·total + base + i]`
///
/// Grid: x over box voxels in blocks, y over tiles
#[kernel]
pub unsafe fn fb_ax_grad(desc: *const u32, count: *const u32, family: u32, total: u32, s: *const f32, g: *mut f32) {
    let ti = blockIdx_y();
    if unsafe { !has_family(count, ti, family) } {
        return;
    }
    let t = unsafe { tile(desc, ti) };
    let i = blockIdx_x() * blockDim_x() + threadIdx_x();
    if i >= t.len() {
        return;
    }
    let (z, y, x) = t.voxel(i);
    let gz = gradient(z, t.dims[0], |k| unsafe { *s.add(t.idx(k, y, x)) });
    let gy = gradient(y, t.dims[1], |k| unsafe { *s.add(t.idx(z, k, x)) });
    let gx = gradient(x, t.dims[2], |k| unsafe { *s.add(t.idx(z, y, k)) });
    let at = (t.base + i) as usize;
    let total = total as usize;
    unsafe {
        *g.add(at) = gz;
        *g.add(total + at) = gy;
        *g.add(2 * total + at) = gx;
    }
}

/// The six gradient products `g_u·g_v` in float32 at scratch index `at`, (u, v) in (0,0), (0,1), (0,2), (1,1),
/// (1,2), (2,2) order, widened to float64 for the Gaussian's sums
#[inline(always)]
unsafe fn products(g: *const f32, total: usize, at: usize) -> [f64; 6] {
    let (g0, g1, g2) = unsafe { (*g.add(at), *g.add(total + at), *g.add(2 * total + at)) };
    [
        mul_rn_f32(g0, g0) as f64,
        mul_rn_f32(g0, g1) as f64,
        mul_rn_f32(g0, g2) as f64,
        mul_rn_f32(g1, g1) as f64,
        mul_rn_f32(g1, g2) as f64,
        mul_rn_f32(g2, g2) as f64,
    ]
}

/// `acc[k] + (lo[k] + hi[k])·w` for each channel, one Gaussian step of [`correlate`] on six lines at once
#[inline(always)]
fn step6(acc: [f64; 6], lo: [f64; 6], hi: [f64; 6], w: f64) -> [f64; 6] {
    let f = |a: f64, l: f64, h: f64| add_rn_f64(a, mul_rn_f64(add_rn_f64(l, h), w));
    [
        f(acc[0], lo[0], hi[0]),
        f(acc[1], lo[1], hi[1]),
        f(acc[2], lo[2], hi[2]),
        f(acc[3], lo[3], hi[3]),
        f(acc[4], lo[4], hi[4]),
        f(acc[5], lo[5], hi[5]),
    ]
}

/// The σ 1.5 Gaussian along z of the six gradient products of `g`: `p[k·total + base + i]` for product `k`; the
/// products are formed at each of the 13 z positions as they are read, which gives the same values as smoothing
/// a stored product box
///
/// Grid: x over box voxels in blocks, y over tiles
#[kernel]
pub unsafe fn fb_ax_tensor_z(desc: *const u32, count: *const u32, family: u32, total: u32, g: *const f32, p: *mut f32) {
    let ti = blockIdx_y();
    if unsafe { !has_family(count, ti, family) } {
        return;
    }
    let t = unsafe { tile(desc, ti) };
    let i = blockIdx_x() * blockDim_x() + threadIdx_x();
    if i >= t.len() {
        return;
    }
    let (z, y, x) = t.voxel(i);
    let (n, total) = (t.dims[0], total as usize);
    let r = W15.len() - 1;
    let c = unsafe { products(g, total, t.idx(z, y, x)) };
    let w = W15[r];
    let mut acc = [
        mul_rn_f64(c[0], w),
        mul_rn_f64(c[1], w),
        mul_rn_f64(c[2], w),
        mul_rn_f64(c[3], w),
        mul_rn_f64(c[4], w),
        mul_rn_f64(c[5], w),
    ];
    // a counted range loop, which `#[unroll]` needs
    #[allow(clippy::needless_range_loop)]
    #[unroll]
    for j in 0..6 {
        let off = (r - j) as i32;
        let lo = unsafe { products(g, total, t.idx(reflect(z - off, n), y, x)) };
        let hi = unsafe { products(g, total, t.idx(reflect(z + off, n), y, x)) };
        acc = step6(acc, lo, hi, W15[j]);
    }

    let at = (t.base + i) as usize;
    unsafe {
        *p.add(at) = acc[0] as f32;
        *p.add(total + at) = acc[1] as f32;
        *p.add(2 * total + at) = acc[2] as f32;
        *p.add(3 * total + at) = acc[3] as f32;
        *p.add(4 * total + at) = acc[4] as f32;
        *p.add(5 * total + at) = acc[5] as f32;
    }
}

/// The σ 1.5 Gaussian along y of each of the six product channels: `p` → `q`, both `[k·total + base + i]`
///
/// Grid: x over box voxels in blocks, y over tiles
#[kernel]
pub unsafe fn fb_ax_tensor_y(desc: *const u32, count: *const u32, family: u32, total: u32, p: *const f32, q: *mut f32) {
    let ti = blockIdx_y();
    if unsafe { !has_family(count, ti, family) } {
        return;
    }
    let t = unsafe { tile(desc, ti) };
    let i = blockIdx_x() * blockDim_x() + threadIdx_x();
    if i >= t.len() {
        return;
    }
    let (z, y, x) = t.voxel(i);
    let at = (t.base + i) as usize;
    let mut k = 0;
    while k < 6 {
        let channel = unsafe { p.add(k * total as usize) };
        let v = correlate(&W15, y, t.dims[1], |j| unsafe { *channel.add(t.idx(z, j, x)) });
        unsafe { *q.add(k * total as usize + at) = v };
        k += 1;
    }
}

/// Finish each listed voxel of the family: the σ 1.5 Gaussian along x of the six `q` channels at the voxel gives
/// its tensor m = [a00, a01, a02, a11, a12, a22]; a certified (nx, ny) goes to the core voxel's element of
/// `out_nx` and `out_ny` (brick order, as [`fb_seeds`] writes `out`), and an uncertified voxel to the next
/// fallback record `fb[8j..8j + 8] = [tile, core-local index, m as float32 bits]` through `fb_count` (records
/// past `fb_cap` are counted, not written, and the host fails the run)
///
/// Grid: x over list entries (`cap`) in blocks, y over tiles
#[kernel]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fb_ax_eigen(
    desc: *const u32,
    count: *const u32,
    family: u32,
    total: u32,
    cap: u32,
    list: *const u32,
    q: *const f32,
    out_nx: *mut u8,
    out_ny: *mut u8,
    fb: *mut u32,
    fb_cap: u32,
    fb_count: *mut u32,
) {
    let ti = blockIdx_y();
    let slot = 2 * ti + family;
    let k = blockIdx_x() * blockDim_x() + threadIdx_x();
    if k >= unsafe { *count.add(slot as usize) } {
        return;
    }
    let t = unsafe { tile(desc, ti) };
    let local = unsafe { *list.add((slot * cap + k) as usize) };
    let (cz, cy, cx) = t.core_voxel(local);
    let (z, y, x) = (t.off[0] + cz as i32, t.off[1] + cy as i32, t.off[2] + cx as i32);
    let n = t.dims[2];
    let channel = |c: usize| {
        let src = unsafe { q.add(c * total as usize) };
        correlate(&W15, x, n, |j| unsafe { *src.add(t.idx(z, y, j)) })
    };
    let m = [channel(0), channel(1), channel(2), channel(3), channel(4), channel(5)];

    if let Some((nx, ny)) = certify::<Rn>(m) {
        let at = t.out_at(cz, cy, cx);
        unsafe {
            *out_nx.add(at) = nx;
            *out_ny.add(at) = ny;
        }
        return;
    }

    // SAFETY: `fb_count` is one u32 the host zeroed; device-scope relaxed atomics only hand out distinct records
    let j = unsafe { DeviceAtomicU32::from_ptr(fb_count).fetch_add(1, AtomicOrdering::Relaxed) };
    if j >= fb_cap {
        return;
    }
    let rec = unsafe { fb.add(8 * j as usize) };
    unsafe {
        *rec = ti;
        *rec.add(1) = local;
        *rec.add(2) = m[0].to_bits();
        *rec.add(3) = m[1].to_bits();
        *rec.add(4) = m[2].to_bits();
        *rec.add(5) = m[3].to_bits();
        *rec.add(6) = m[4].to_bits();
        *rec.add(7) = m[5].to_bits();
    }
}
