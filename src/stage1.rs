//! Centrality and ridge seeds of one tile, in either of the reference's two roundings
//!
//! The reference has two implementations of this stage from the same Python source: Numba's CPU kernel
//! (`stage1_centrality_and_ridge`, core.py:254) and its CUDA kernels (`centrality_kernel` and `seed_kernel`,
//! cuda.py:102-203), picked by `--device`. Both keep the float32/float64 typing described in `kernels.rs`, but
//! they round differently: LLVM on the CPU rounds every product and sum, while NVVM (CUDA's default `-fma=1`,
//! which Numba keeps without `fastmath`) and then ptxas fuse chosen multiply-adds into one `fma`. Every such
//! site is written here as [`Arith::mad`], with the operand order the fused instruction takes, so one source
//! gives the reference's CPU output with [`Separate`] and its CUDA output with [`Fused`]. The fused sites and
//! orders come from the PTX Numba 0.64 emits for cuda.py and the SASS ptxas 13.2 builds from it for sm_120; the
//! kernels of `--device cuda` (`fibrs-kernels/src/lib.rs`) fuse the same sites in the same order

use std::sync::OnceLock;

use crate::kernels::{Dense, Grid};
use crate::pyfmt::{py_max, py_min};

/// How a multiply-add site rounds
pub trait Arith: Send + Sync + 'static {
    /// `a * b + c`
    fn mad(a: f64, b: f64, c: f64) -> f64;

    /// `decode_dir` (core.py:8, cuda.py:10) for every byte pair, as `nx << 8 | ny`
    fn dir_table() -> &'static [[f64; 3]];
}

/// The reference's CPU rounding: each product and each sum rounded on its own
pub struct Separate;

/// The reference's CUDA rounding: the product and the sum rounded once, as one `fma`
#[cfg_attr(not(all(target_os = "linux", target_arch = "x86_64")), allow(dead_code))]
pub struct Fused;

impl Arith for Separate {
    #[inline(always)]
    fn mad(a: f64, b: f64, c: f64) -> f64 {
        a * b + c
    }

    fn dir_table() -> &'static [[f64; 3]] {
        static TABLE: OnceLock<Vec<[f64; 3]>> = OnceLock::new();
        TABLE.get_or_init(dir_table::<Self>)
    }
}

impl Arith for Fused {
    #[inline(always)]
    fn mad(a: f64, b: f64, c: f64) -> f64 {
        a.mul_add(b, c)
    }

    fn dir_table() -> &'static [[f64; 3]] {
        static TABLE: OnceLock<Vec<[f64; 3]>> = OnceLock::new();
        TABLE.get_or_init(dir_table::<Self>)
    }
}

fn dir_table<A: Arith>() -> Vec<[f64; 3]> {
    (0..=u16::MAX).map(|i| decode_dir::<A>((i >> 8) as u8, i as u8)).collect()
}

/// `decode_dir`: all float64 after the float32 conversion of each byte; `1 / sqrt(r2)` is CUDA's `rcp.rn`
/// of `sqrt.rn`, the same correctly rounded value
fn decode_dir<A: Arith>(nx: u8, ny: u8) -> [f64; 3] {
    let mut dx = (nx as f32 as f64 - 128.0) / 127.0;
    let mut dy = (ny as f32 as f64 - 128.0) / 127.0;
    let r2 = A::mad(dx, dx, dy * dy);
    if r2 <= 1.0 {
        return [dx, dy, (1.0f32 as f64 - r2).sqrt()];
    }
    let inv = 1.0f32 as f64 / r2.sqrt();
    dx *= inv;
    dy *= inv;
    [dx, dy, 0.0f32 as f64]
}

/// The direction of a voxel from its packed bytes ([`crate::kernels::pack_dir`]), which index the table
#[inline(always)]
fn dir<A: Arith>(packed: u16) -> [f64; 3] {
    A::dir_table()[packed as usize]
}

/// The clamp and cell split shared by the trilinear helpers (core.py:36-53, cuda.py:26-44)
#[inline(always)]
fn cell(d: [i64; 3], mut x: f64, mut y: f64, mut z: f64) -> ([i64; 3], [f32; 3]) {
    let (dd, hh, ww) = (d[0] as f64, d[1] as f64, d[2] as f64);
    if x < 0.0 {
        x = 0.0;
    } else if x > ww - 1.001 {
        x = (ww - 1.001) as f32 as f64;
    }
    if y < 0.0 {
        y = 0.0;
    } else if y > hh - 1.001 {
        y = (hh - 1.001) as f32 as f64;
    }
    if z < 0.0 {
        z = 0.0;
    } else if z > dd - 1.001 {
        z = (dd - 1.001) as f32 as f64;
    }
    let (ix, iy, iz) = (x as i64, y as i64, z as i64);
    let fx = (x - ix as f64) as f32;
    let fy = (y - iy as f64) as f32;
    let fz = (z - iz as f64) as f32;
    ([iz, iy, ix], [fz, fy, fx])
}

/// The y then z blend of four x-lerps: `(1 - fz) * ((1 - fy) * p00 + fy * p01) + fz * (...)`
#[inline(always)]
fn blend<A: Arith>(p: [f64; 4], fy: f32, fz: f32) -> f64 {
    let (fy, fz) = (fy as f64, fz as f64);
    let lo = A::mad(1.0 - fy, p[0], p[1] * fy);
    let hi = A::mad(1.0 - fy, p[2], p[3] * fy);
    A::mad(1.0 - fz, lo, hi * fz)
}

/// `trilin_scalar_u8`: `fx * arr[..]` is a float32 product
#[inline]
fn trilin_u8<A: Arith>(arr: &Dense<u8>, x: f64, y: f64, z: f64) -> f64 {
    let ([iz, iy, ix], [fz, fy, fx]) = cell(arr.dims(), x, y, z);
    let lerp = |z: i64, y: i64| {
        let (a0, a1) = (arr.at(z, y, ix), arr.at(z, y, ix + 1));
        A::mad(1.0 - fx as f64, a0 as f64, (fx * a1 as f32) as f64)
    };
    blend::<A>([lerp(iz, iy), lerp(iz, iy + 1), lerp(iz + 1, iy), lerp(iz + 1, iy + 1)], fy, fz)
}

/// `trilin_scalar_u16` before its final `0.01` scale, which the seed test fuses into a sum
#[inline]
fn trilin_u16_unscaled<A: Arith>(arr: &Dense<u16>, x: f64, y: f64, z: f64) -> f64 {
    let ([iz, iy, ix], [fz, fy, fx]) = cell(arr.dims(), x, y, z);
    let fx = fx as f64;
    let lerp = |z: i64, y: i64| A::mad(1.0 - fx, arr.at(z, y, ix) as f64, fx * arr.at(z, y, ix + 1) as f64);
    blend::<A>([lerp(iz, iy), lerp(iz, iy + 1), lerp(iz + 1, iy), lerp(iz + 1, iy + 1)], fy, fz)
}

/// `float32(0.01)`, the scale of `trilin_scalar_u16`
const CENT_SCALE: f64 = 0.01f32 as f64;

/// `ortho_basis`; ptxas fuses each cross-product component's first product
#[inline(always)]
fn ortho_basis<A: Arith>(ux: f64, uy: f64, uz: f64) -> [f64; 6] {
    let (e1x, e1y, e1z) = if uz.abs() < 0.85 { (-uy, ux, 0.0) } else { (0.0, -uz, uy) };
    let n1 = A::mad(e1z, e1z, A::mad(e1x, e1x, e1y * e1y)).sqrt();
    let (e1x, e1y, e1z) = (e1x / n1, e1y / n1, e1z / n1);
    let e2x = A::mad(uy, e1z, -(uz * e1y));
    let e2y = A::mad(uz, e1x, -(ux * e1z));
    let e2z = A::mad(ux, e1y, -(uy * e1x));
    [e1x, e1y, e1z, e2x, e2y, e2z]
}

// the reference's literals, rounded to float32 as Numba does; not the library constant
#[allow(clippy::approx_constant)]
const COS_A: [f32; 8] = [1.0, 0.70710678, 0.0, -0.70710678, -1.0, -0.70710678, 0.0, 0.70710678];
#[allow(clippy::approx_constant)]
const SIN_A: [f32; 8] = [0.0, 0.70710678, 1.0, 0.70710678, 0.0, -0.70710678, -1.0, -0.70710678];
/// Ring radii and weights; `wr / 8.0` is `wr * 0.125` exactly
const RADII: [(f64, f64); 4] = [(1.0, 0.5), (1.8, 0.28), (2.7, 0.14), (3.6, 0.06)];

/// Direction `k` of the ring in the plane of `e1`, `e2`
#[inline(always)]
fn ring_dir<A: Arith>(b: &[f64; 6], k: usize) -> [f64; 3] {
    let (ca, sa) = (COS_A[k] as f64, SIN_A[k] as f64);
    [A::mad(b[0], ca, b[3] * sa), A::mad(b[1], ca, b[4] * sa), A::mad(b[2], ca, b[5] * sa)]
}

/// The centrality of the voxel at `z, y, x`, or None below presence 64
#[inline]
pub fn centrality_at<A: Arith>(pres: &Dense<u8>, dirs: &Dense<u16>, z: i64, y: i64, x: i64) -> Option<u16> {
    let p0 = pres.at(z, y, x) as f32;
    if p0 < 64.0 {
        return None;
    }
    let [d, h, w] = pres.dims();
    let (df, hf, wf) = (d as f64, h as f64, w as f64);
    let (xf, yf, zf) = (x as f64, y as f64, z as f64);
    let [ux, uy, uz] = dir::<A>(dirs.at(z, y, x));
    let p_p = trilin_u8::<A>(pres, xf + ux, yf + uy, zf + uz);
    let p_m = trilin_u8::<A>(pres, xf - ux, yf - uy, zf - uz);
    let p_par = A::mad(p_m, 0.25, A::mad(p0 as f64, 0.5, p_p * 0.25));
    let basis = ortho_basis::<A>(ux, uy, uz);
    let mut c_sum = 0.0f64;
    for k in 0..8 {
        let [dx, dy, dz] = ring_dir::<A>(&basis, k);
        for (r, wr) in RADII {
            let xx = py_min(py_max(A::mad(dx, r, xf), 0.0), wf - 1.001);
            let yy = py_min(py_max(A::mad(dy, r, yf), 0.0), hf - 1.001);
            let zz = py_min(py_max(A::mad(dz, r, zf), 0.0), df - 1.001);
            let pr = trilin_u8::<A>(pres, xx, yy, zz);
            let (iz, iy, ix) = ((zz + 0.5) as i64, (yy + 0.5) as i64, (xx + 0.5) as i64);
            let [ux2, uy2, uz2] = dir::<A>(dirs.at(iz, iy, ix));
            let dr = A::mad(uz, uz2, A::mad(uy, uy2, ux * ux2)).abs();
            if dr > 0.85 {
                c_sum = A::mad(wr * 0.125 * pr, (dr - 0.85) / 0.15, c_sum);
            }
        }
    }
    Some(py_min(py_max(A::mad(p_par + c_sum, 100.0, 0.5), 0.0), 65535.0) as u16)
}

/// The seed priority (0, 1 or 2) of an interior voxel at `z, y, x` with stride-2 parity
#[inline]
pub fn seed_at<A: Arith>(pres: &Dense<u8>, dirs: &Dense<u16>, cent: &Dense<u16>, z: i64, y: i64, x: i64) -> u8 {
    if pres.at(z, y, x) < 79 || (x + y + z) & 1 != 0 {
        return 0;
    }
    let c0 = cent.at(z, y, x) as f32 * 0.01f32;
    let c0d = c0 as f64;
    let (xf, yf, zf) = (x as f64, y as f64, z as f64);
    let [ux, uy, uz] = dir::<A>(dirs.at(z, y, x));
    let basis = ortho_basis::<A>(ux, uy, uz);
    let mut n_max = 0;
    for k in 0..4 {
        let [dx, dy, dz] = ring_dir::<A>(&basis, k);
        let cp = CENT_SCALE
            * trilin_u16_unscaled::<A>(cent, A::mad(dx, 1.15, xf), A::mad(dy, 1.15, yf), A::mad(dz, 1.15, zf));
        let cm = CENT_SCALE
            * trilin_u16_unscaled::<A>(cent, A::mad(dx, -1.15, xf), A::mad(dy, -1.15, yf), A::mad(dz, -1.15, zf));
        if !(c0d >= cp && c0d >= cm) {
            continue;
        }
        let tp2 = trilin_u16_unscaled::<A>(cent, xf + 2.0 * dx, yf + 2.0 * dy, zf + 2.0 * dz);
        let tm2 = trilin_u16_unscaled::<A>(cent, xf - 2.0 * dx, yf - 2.0 * dy, zf - 2.0 * dz);
        // 0.5 * (cp2 + cm2) + 2.0, with cp2's scale fused into the sum
        if c0d >= A::mad(A::mad(tp2, CENT_SCALE, tm2 * CENT_SCALE), 0.5, 2.0) {
            n_max += 1;
        }
    }
    match n_max {
        4 => 2,
        3 if c0d >= 185.0 => 1,
        _ => 0,
    }
}

/// Centrality and ridge seeds of one tile (seed stride 2); every coordinate and clamp is local to the tile, as
/// in the reference
pub fn stage1<A: Arith>(pres: &Dense<u8>, dirs: &Dense<u16>) -> (Vec<u16>, Vec<u8>) {
    let [d, h, w] = pres.dims();
    let n = pres.data.len();
    let mut cent = Dense { dims: pres.dims, data: vec![0u16; n] };
    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                if let Some(c) = centrality_at::<A>(pres, dirs, z, y, x) {
                    cent.data[((z * h + y) * w + x) as usize] = c;
                }
            }
        }
    }
    let mut seeds = vec![0u8; n];
    for z in 1..d - 1 {
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                seeds[((z * h + y) * w + x) as usize] = seed_at::<A>(pres, dirs, &cent, z, y, x);
            }
        }
    }
    (cent.data, seeds)
}
