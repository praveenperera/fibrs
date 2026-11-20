//! The fibre walk's per-seed tracer, `kernels::trace_seed` (src/kernels.rs) expression by expression, shared by
//! the `fb_walk_*` kernels and the host's tests
//!
//! Every float64 and float32 operation goes through [`Ops`], in the CPU's order and typing, so the kernels run it
//! as explicit round-to-nearest instructions, which neither LLVM nor ptxas contracts, and the host runs the same
//! code in plain Rust, which never contracts. Volumes, coverage, path history and span scratch are reached through
//! traits, so the device reads its brick pools and the host its own volumes. Coverage reads go through [`Reads`],
//! which records each one: a result is exactly `trace_seed`'s for any coverage that holds the values its reads
//! recorded. `ring()`'s exponentials come from the host ([`Fields::ring`]), never from a device `exp`.
//!
//! The code keeps register arrays indexed by constants only: every loop over one is fully unrolled
//! (`unroll_full!`, which the including crate defines: cuda-oxide's unroll hint on the device, nothing on the
//! host), and loops are `while` loops, whose counters stay out of local memory

/// The arithmetic the tracer runs in: round-to-nearest, never contracted
pub trait Ops {
    fn mul(a: f64, b: f64) -> f64;
    fn add(a: f64, b: f64) -> f64;
    fn div(a: f64, b: f64) -> f64;
    fn sqrt(a: f64) -> f64;
    fn mul32(a: f32, b: f32) -> f32;
    fn add32(a: f32, b: f32) -> f32;
    fn div32(a: f32, b: f32) -> f32;
    fn sqrt32(a: f32) -> f32;
}

/// cos(55°): the turn limit used throughout the tracer
pub const COS55: f64 = 0.5735764363510462;

/// Half-path capacity of `trace_seed_range` (`kernels::HALF`): a half reaching it is refused
pub const HALF: u32 = 32768;

/// A coverage read of `Coverage::is_covered`
pub const IS_COVERED: u8 = 0;
/// A coverage read of `Coverage::covered_same_dir`, with its tangent
pub const SAME_DIR: u8 = 1;

/// Coverage values: a count of 0, a count of 2, otherwise a count of 1 with its direction
const UNCOVERED: u16 = 0;
const COVERED_TWICE: u16 = 1;

/// The ring `recenter_normal_plane` samples, in its order
pub const RING_U1: [f32; 13] =
    [0.0, 0.55, 0.0, -0.55, 0.0, 1.15, 0.81317, 0.0, -0.81317, -1.15, -0.81317, 0.0, 0.81317];
pub const RING_U2: [f32; 13] =
    [0.0, 0.0, 0.55, 0.0, -0.55, 0.0, 0.81317, 1.15, 0.81317, 0.0, -0.81317, -1.15, -0.81317];

/// The read-only fields one family traces over, all of shape [`Fields::dims`]
pub trait Fields {
    /// Extent as D, H, W
    fn dims(&self) -> [i32; 3];
    /// Presence at the eight corners of the in-bounds cell whose lowest corner is `z, y, x`, `[z][y][x]` flattened
    fn pres_corners(&self, z: i32, y: i32, x: i32) -> [u8; 8];
    /// Centrality at the eight corners of an in-bounds cell, as [`Fields::pres_corners`]
    fn cent_corners(&self, z: i32, y: i32, x: i32) -> [u16; 8];
    /// Packed direction bytes `nx << 8 | ny` at an in-bounds voxel
    fn dir(&self, z: i32, y: i32, x: i32) -> u16;
    /// Packed direction bytes at the eight corners of an in-bounds cell, as [`Fields::pres_corners`]
    fn dir_corners(&self, z: i32, y: i32, x: i32) -> [u16; 8];
    /// `kernels::decode_packed` of packed direction bytes
    fn decode(&self, d: u16) -> [f64; 3];
    /// `exp(-0.5 * r2)` of ring point `i`, `kernels::ring()[i]`
    fn ring(&self, i: usize) -> f64;
}

/// The family's coverage as one seed's trace reads it
pub trait Reads {
    /// The coverage value at an in-bounds voxel, recorded as a read of `kind` with tangent `t` (zero for
    /// [`IS_COVERED`])
    fn read(&mut self, z: i32, y: i32, x: i32, kind: u8, t: [f64; 3]) -> u16;
}

/// A seed's two half-paths, `pts_f`/`tans_f` (half 0) and `pts_b`/`tans_b` (half 1) of `trace_seed`
///
/// One type holds both, so the device inlines one `trace_half` for both halves
pub trait Path {
    /// Point `i` of `half`, already pushed
    fn pt(&self, half: u32, i: u32) -> [f32; 3];
    /// Set point and tangent `i` of `half`, `i` below the cap the half is traced with
    fn push(&mut self, half: u32, i: u32, p: [f32; 3], t: [f32; 3]);
}

/// A seed's joined path `pts_c` and the scratch `valid_curve_spans` needs
pub trait Curve {
    /// Point `i` of the joined path
    fn pt(&self, i: u32) -> [f32; 3];
    /// Push a pending piece `[a, b)`; false when the stack is full
    fn push(&mut self, a: u32, b: u32) -> bool;
    /// Pop the newest pending piece, or `u32::MAX` in both halves when none is left
    fn pop(&mut self) -> (u32, u32);
    /// `cum_s[i]` of `first_curve_split`
    fn cum(&self, i: u32) -> f32;
    fn set_cum(&mut self, i: u32, v: f32);
    /// Keep an accepted span `[a, b)`; false when there is no room
    fn accept(&mut self, a: u32, b: u32) -> bool;
}

#[inline(always)]
fn sub<O: Ops>(a: f64, b: f64) -> f64 {
    O::add(a, -b)
}

#[inline(always)]
fn sub32<O: Ops>(a: f32, b: f32) -> f32 {
    O::add32(a, -b)
}

/// `|a|` by clearing the sign bit, as `f64::abs` does
#[inline(always)]
fn abs(a: f64) -> f64 {
    f64::from_bits(a.to_bits() & !(1 << 63))
}

/// `a[0] * b[0] + a[1] * b[1] + a[2] * b[2]`, left to right
#[inline(always)]
fn dot<O: Ops>(a: [f64; 3], b: [f64; 3]) -> f64 {
    O::add(O::add(O::mul(a[0], b[0]), O::mul(a[1], b[1])), O::mul(a[2], b[2]))
}

#[inline(always)]
fn norm<O: Ops>(a: [f64; 3]) -> f64 {
    O::sqrt(dot::<O>(a, a))
}

#[inline(always)]
fn sub3<O: Ops>(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [sub::<O>(a[0], b[0]), sub::<O>(a[1], b[1]), sub::<O>(a[2], b[2])]
}

#[inline(always)]
fn div3<O: Ops>(a: [f64; 3], n: f64) -> [f64; 3] {
    [O::div(a[0], n), O::div(a[1], n), O::div(a[2], n)]
}

#[inline(always)]
fn f64x3(p: [f32; 3]) -> [f64; 3] {
    [p[0] as f64, p[1] as f64, p[2] as f64]
}

#[inline(always)]
fn f32x3(p: [f64; 3]) -> [f32; 3] {
    [p[0] as f32, p[1] as f32, p[2] as f32]
}

#[inline(always)]
fn sub3_32<O: Ops>(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [sub32::<O>(a[0], b[0]), sub32::<O>(a[1], b[1]), sub32::<O>(a[2], b[2])]
}

#[inline(always)]
fn dot32<O: Ops>(a: [f32; 3], b: [f32; 3]) -> f32 {
    O::add32(O::add32(O::mul32(a[0], b[0]), O::mul32(a[1], b[1])), O::mul32(a[2], b[2]))
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

/// `decode_dir` of the direction at an in-bounds voxel
#[inline(always)]
fn dir_at<F: Fields>(f: &F, z: i32, y: i32, x: i32) -> [f64; 3] {
    f.decode(f.dir(z, y, x))
}

/// One axis of the trilinear helpers' clamp: below 0 to 0, above `n - 1.001` to that bound in float32
#[inline(always)]
fn clamp_axis<O: Ops>(v: f64, n: f64) -> f64 {
    if v < 0.0 {
        0.0
    } else if v > sub::<O>(n, 1.001) {
        sub::<O>(n, 1.001) as f32 as f64
    } else {
        v
    }
}

/// The clamp and cell split shared by the trilinear helpers (`kernels::cell`)
#[inline(always)]
fn cell<O: Ops>(d: [i32; 3], x: f64, y: f64, z: f64) -> ([i32; 3], [f32; 3]) {
    let x = clamp_axis::<O>(x, d[2] as f64);
    let y = clamp_axis::<O>(y, d[1] as f64);
    let z = clamp_axis::<O>(z, d[0] as f64);
    let (ix, iy, iz) = (x as i32, y as i32, z as i32);
    let fx = sub::<O>(x, ix as f64) as f32;
    let fy = sub::<O>(y, iy as f64) as f32;
    let fz = sub::<O>(z, iz as f64) as f32;
    ([iz, iy, ix], [fz, fy, fx])
}

/// `(1 - fz) * ((1 - fy) * p00 + fy * p01) + fz * ((1 - fy) * p10 + fy * p11)`
#[inline(always)]
fn blend<O: Ops>(p: [f64; 4], fy: f64, fz: f64) -> f64 {
    let lo = O::add(O::mul(sub::<O>(1.0, fy), p[0]), O::mul(fy, p[1]));
    let hi = O::add(O::mul(sub::<O>(1.0, fy), p[2]), O::mul(fy, p[3]));
    O::add(O::mul(sub::<O>(1.0, fz), lo), O::mul(fz, hi))
}

/// `trilin_scalar_u8` (`kernels::trilin_u8`): `fx * arr[..]` is a float32 product
#[inline(always)]
pub fn trilin_u8<O: Ops, F: Fields>(f: &F, x: f64, y: f64, z: f64) -> f64 {
    let ([iz, iy, ix], [fz, fy, fx]) = cell::<O>(f.dims(), x, y, z);
    let c = f.pres_corners(iz, iy, ix);
    let w0 = sub::<O>(1.0, fx as f64);
    let lerp = |a: u8, b: u8| O::add(O::mul(w0, a as f64), O::mul32(fx, b as f32) as f64);
    let p = [lerp(c[0], c[1]), lerp(c[2], c[3]), lerp(c[4], c[5]), lerp(c[6], c[7])];
    blend::<O>(p, fy as f64, fz as f64)
}

/// `trilin_scalar_u16` (`kernels::trilin_u16`): float32 times uint16 is float64
#[inline(always)]
pub fn trilin_u16<O: Ops, F: Fields>(f: &F, x: f64, y: f64, z: f64) -> f64 {
    let ([iz, iy, ix], [fz, fy, fx]) = cell::<O>(f.dims(), x, y, z);
    let fx = fx as f64;
    let c = f.cent_corners(iz, iy, ix);
    let w0 = sub::<O>(1.0, fx);
    let lerp = |a: u16, b: u16| O::add(O::mul(w0, a as f64), O::mul(fx, b as f64));
    let p = [lerp(c[0], c[1]), lerp(c[2], c[3]), lerp(c[4], c[5]), lerp(c[6], c[7])];
    O::mul(0.01f32 as f64, blend::<O>(p, fy as f64, fz as f64))
}

/// `trilin_dir_aligned` (`Fields::dir_aligned`)
#[inline(always)]
pub fn dir_aligned<O: Ops, F: Fields>(f: &F, x: f64, y: f64, z: f64, r: [f64; 3]) -> [f64; 3] {
    let ([iz, iy, ix], [fz, fy, fx]) = cell::<O>(f.dims(), x, y, z);
    let (fx, fy, fz) = (fx as f64, fy as f64, fz as f64);
    let pres = f.pres_corners(iz, iy, ix);
    let dirs = f.dir_corners(iz, iy, ix);
    let mut v = [0.0f64; 3];
    let mut k = 0;
    while k < 8 {
        unroll_full!();
        let wz = if k >> 2 == 0 { sub::<O>(1.0, fz) } else { fz };
        let wy = O::mul(wz, if (k >> 1) & 1 == 0 { sub::<O>(1.0, fy) } else { fy });
        let w = O::mul(wy, if k & 1 == 0 { sub::<O>(1.0, fx) } else { fx });
        let pw = O::mul(w, O::add(pres[k] as f32 as f64, 10.0));
        let mut u = f.decode(dirs[k]);
        if dot::<O>(u, r) < 0.0 {
            u = [-u[0], -u[1], -u[2]];
        }
        v = [O::add(v[0], O::mul(pw, u[0])), O::add(v[1], O::mul(pw, u[1])), O::add(v[2], O::mul(pw, u[2]))];
        k += 1;
    }
    let n = norm::<O>(v);
    if n > 1e-08 {
        v = div3::<O>(v, n);
    }
    v
}

/// `ortho_basis` (`kernels::ortho_basis`): e1 then e2
#[inline(always)]
fn ortho_basis<O: Ops>(u: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let [ux, uy, uz] = u;
    let e1 = if abs(uz) < 0.85 { [-uy, ux, 0.0] } else { [0.0, -uz, uy] };
    let e1 = div3::<O>(e1, norm::<O>(e1));
    let e2 = [
        sub::<O>(O::mul(uy, e1[2]), O::mul(uz, e1[1])),
        sub::<O>(O::mul(uz, e1[0]), O::mul(ux, e1[2])),
        sub::<O>(O::mul(ux, e1[1]), O::mul(uy, e1[0])),
    ];
    (e1, e2)
}

/// `c + u1 * e1 + u2 * e2` per axis, left to right
#[inline(always)]
fn plane_point<O: Ops>(c: [f64; 3], u1: f64, u2: f64, e1: [f64; 3], e2: [f64; 3]) -> [f64; 3] {
    [
        O::add(O::add(c[0], O::mul(u1, e1[0])), O::mul(u2, e2[0])),
        O::add(O::add(c[1], O::mul(u1, e1[1])), O::mul(u2, e2[1])),
        O::add(O::add(c[2], O::mul(u1, e1[2])), O::mul(u2, e2[2])),
    ]
}

/// `recenter_normal_plane` (`Fields::recenter`) of the point `p` (x, y, z) in the plane normal to `t`
#[inline(always)]
pub fn recenter<O: Ops, F: Fields>(f: &F, p: [f64; 3], t: [f64; 3], max_shift: f64, n_iter: u32) -> [f64; 3] {
    let [d, h, w] = f.dims();
    let (d, h, w) = (d as f64, h as f64, w as f64);
    let (e1, e2) = ortho_basis::<O>(t);
    let mut c = p;
    let mut vals = [0.0f32; 13];
    let mut it = 0;
    while it < n_iter {
        let mut c_min = 1_000_000_000.0f32;
        let mut c_max = -1_000_000_000.0f32;
        let mut i = 0;
        while i < 13 {
            unroll_full!();
            let [px, py, pz] = plane_point::<O>(c, RING_U1[i] as f64, RING_U2[i] as f64, e1, e2);
            let out = px < 0.0
                || px >= sub::<O>(w, 1.001)
                || py < 0.0
                || py >= sub::<O>(h, 1.001)
                || pz < 0.0
                || pz >= sub::<O>(d, 1.001);
            if out {
                vals[i] = 0.0;
            } else {
                let half = |v: f64| O::add(v, 0.5) as i32;
                let u = dir_at(f, half(pz), half(py), half(px));
                let dt = abs(dot::<O>(u, t));
                vals[i] = if dt < 0.82 { 0.0 } else { O::mul(trilin_u16::<O, F>(f, px, py, pz), dt) as f32 };
            }
            if vals[i] < c_min {
                c_min = vals[i];
            }
            if vals[i] > c_max {
                c_max = vals[i];
            }
            i += 1;
        }
        if (sub32::<O>(c_max, c_min) as f64) < 0.5 {
            break;
        }
        let (mut w_sum, mut du1, mut du2) = (0.0f64, 0.0f64, 0.0f64);
        let mut i = 0;
        while i < 13 {
            unroll_full!();
            let ex = sub32::<O>(vals[i], c_min);
            let wgt = O::mul(O::mul32(ex, ex) as f64, f.ring(i));
            w_sum = O::add(w_sum, wgt);
            du1 = O::add(du1, O::mul(wgt, RING_U1[i] as f64));
            du2 = O::add(du2, O::mul(wgt, RING_U2[i] as f64));
            i += 1;
        }
        if w_sum > 1e-06 {
            du1 = O::div(du1, w_sum);
            du2 = O::div(du2, w_sum);
            c = [
                O::add(c[0], O::add(O::mul(du1, e1[0]), O::mul(du2, e2[0]))),
                O::add(c[1], O::add(O::mul(du1, e1[1]), O::mul(du2, e2[1]))),
                O::add(c[2], O::add(O::mul(du1, e1[2]), O::mul(du2, e2[2]))),
            ];
        }
        it += 1;
    }
    let dv = sub3::<O>(c, p);
    let dist = norm::<O>(dv);
    if dist > max_shift {
        let s = O::div(max_shift, dist);
        c = [O::add(p[0], O::mul(dv[0], s)), O::add(p[1], O::mul(dv[1], s)), O::add(p[2], O::mul(dv[2], s))];
    }
    c
}

/// `is_covered_same_dir` (`Coverage::covered_same_dir`) at `p` (x, y, z): out of bounds reads nothing
#[inline(always)]
fn covered_same_dir<O: Ops, F: Fields, R: Reads>(f: &F, r: &mut R, p: [f64; 3], t: [f64; 3]) -> bool {
    let [d, h, w] = f.dims();
    let ix = O::add(p[0], 0.5) as i32;
    let iy = O::add(p[1], 0.5) as i32;
    let iz = O::add(p[2], 0.5) as i32;
    if ix < 0 || ix >= w || iy < 0 || iy >= h || iz < 0 || iz >= d {
        return false;
    }
    let c = r.read(iz, iy, ix, SAME_DIR, t);
    if c == UNCOVERED {
        return false;
    }
    if c == COVERED_TWICE {
        return true;
    }
    abs(dot::<O>(f.decode(c), t)) > 0.86
}

/// The start of `trace_seed` for the seed at `seed` (z, y, x): the recentred start point and its aligned
/// direction (x, y, z order each), or None when the seed exits before tracing
#[inline(always)]
pub fn prefix<O: Ops, F: Fields, R: Reads>(
    f: &F,
    r: &mut R,
    seed: [i32; 3],
    primary: bool,
) -> Option<([f64; 3], [f64; 3])> {
    let [sz, sy, sx] = seed;
    if !primary && r.read(sz, sy, sx, IS_COVERED, [0.0; 3]) != UNCOVERED {
        return None;
    }
    let u = dir_at(f, sz, sy, sx);
    if covered_same_dir::<O, F, R>(f, r, [sx as f64, sy as f64, sz as f64], u) {
        return None;
    }
    let s = [sx as f32 as f64, sy as f32 as f64, sz as f32 as f64];
    let c = recenter::<O, F>(f, s, u, 1.5, 4);
    if covered_same_dir::<O, F, R>(f, r, c, u) {
        return None;
    }
    if trilin_u8::<O, F>(f, c[0], c[1], c[2]) < 76.5 {
        return None;
    }
    Some((c, dir_aligned::<O, F>(f, c[0], c[1], c[2], u)))
}

/// Whether `p` (x, y, z) lies outside the tracer's margin: `x < 0.25 || x >= W - 1.25`, per axis
#[inline(always)]
fn outside<O: Ops>(dims: [f64; 3], p: [f64; 3]) -> bool {
    let [d, h, w] = dims;
    p[0] < 0.25
        || p[0] >= sub::<O>(w, 1.25)
        || p[1] < 0.25
        || p[1] >= sub::<O>(h, 1.25)
        || p[2] < 0.25
        || p[2] >= sub::<O>(d, 1.25)
}

/// `(v1 · v2) / (l1 * l2) < COS55` with both lengths above 1e-05: a turn sharper than 55°
#[inline(always)]
fn turns<O: Ops>(v1: [f64; 3], v2: [f64; 3]) -> bool {
    let l2 = norm::<O>(v2);
    let l1 = norm::<O>(v1);
    l1 > 1e-05 && l2 > 1e-05 && O::div(dot::<O>(v1, v2), O::mul(l1, l2)) < COS55
}

/// `trace_half_buf` (`kernels::trace_half`) with `step_size = 0.75`, `max_low_steps = 3` and at most `cap` steps:
/// the points kept, and whether the loop ran all `cap` steps
#[inline(always)]
pub fn trace_half<O: Ops, F: Fields, R: Reads, P: Path>(
    f: &F,
    r: &mut R,
    path: &mut P,
    half: u32,
    p0: [f64; 3],
    t0: [f64; 3],
    cap: u32,
) -> (u32, bool) {
    const STEP: f64 = 0.75;
    // `0.5 * STEP`, exact
    const HALF_STEP: f64 = 0.375;
    const MAX_LOW: i32 = 3;
    let [d, h, w] = f.dims();
    let dims = [d as f64, h as f64, w as f64];
    let mut p = p0;
    let mut t = t0;
    let mut n_pts = 0u32;
    let mut low_pres_run = 0i32;
    let mut cov_run = 0i32;
    let mut steps = 0u32;
    while steps < cap {
        steps += 1;
        let m = [
            O::add(p[0], O::mul(HALF_STEP, t[0])),
            O::add(p[1], O::mul(HALF_STEP, t[1])),
            O::add(p[2], O::mul(HALF_STEP, t[2])),
        ];
        if outside::<O>(dims, m) {
            break;
        }
        let mut u = dir_aligned::<O, F>(f, m[0], m[1], m[2], t);
        let dot_u = dot::<O>(u, t);
        let (np, nt);
        if dot_u < COS55 {
            let mut crossed = false;
            let mut look = 0;
            while look < 2 {
                let look_d = if look == 0 { 1.5 } else { 2.2 };
                let l = [
                    O::add(p[0], O::mul(look_d, t[0])),
                    O::add(p[1], O::mul(look_d, t[1])),
                    O::add(p[2], O::mul(look_d, t[2])),
                ];
                let inside = l[0] >= 0.25
                    && l[0] < sub::<O>(dims[2], 1.25)
                    && l[1] >= 0.25
                    && l[1] < sub::<O>(dims[1], 1.25)
                    && l[2] >= 0.25
                    && l[2] < sub::<O>(dims[0], 1.25);
                if inside {
                    let lp = trilin_u8::<O, F>(f, l[0], l[1], l[2]);
                    let lu = dir_aligned::<O, F>(f, l[0], l[1], l[2], t);
                    if lp >= 72.5 && dot::<O>(lu, t) >= COS55 {
                        crossed = true;
                        break;
                    }
                }
                look += 1;
            }
            if !crossed {
                break;
            }
            np = [O::add(p[0], O::mul(STEP, t[0])), O::add(p[1], O::mul(STEP, t[1])), O::add(p[2], O::mul(STEP, t[2]))];
            nt = t;
            u = t;
        } else {
            let nt1 = [
                O::add(O::mul(0.45, t[0]), O::mul(0.55, u[0])),
                O::add(O::mul(0.45, t[1]), O::mul(0.55, u[1])),
                O::add(O::mul(0.45, t[2]), O::mul(0.55, u[2])),
            ];
            nt = div3::<O>(nt1, norm::<O>(nt1));
            let q = [
                O::add(p[0], O::mul(STEP, nt[0])),
                O::add(p[1], O::mul(STEP, nt[1])),
                O::add(p[2], O::mul(STEP, nt[2])),
            ];
            if outside::<O>(dims, q) {
                break;
            }
            np = recenter::<O, F>(f, q, nt, 0.42, 2);
        }
        let dv = sub3::<O>(np, p);
        let dl = norm::<O>(dv);
        if dl < 0.25 || dl > 1.45 {
            break;
        }
        let dv = div3::<O>(dv, dl);
        if dot::<O>(dv, t) < COS55 {
            break;
        }
        if n_pts >= 4 {
            let ch = sub3::<O>(np, f64x3(path.pt(half, n_pts - 4)));
            let chl = norm::<O>(ch);
            if chl > 1e-05 && O::div(dot::<O>(ch, dv), chl) < COS55 {
                break;
            }
        } else {
            let sd = sub3::<O>(np, p0);
            let sl = norm::<O>(sd);
            if sl > 1e-05 && O::div(dot::<O>(sd, t0), sl) < COS55 {
                break;
            }
        }
        if n_pts >= 4 {
            let p4 = path.pt(half, n_pts - 4);
            let q4 = f64x3(p4);
            let v2 = sub3::<O>(np, q4);
            // both operands float32 gives a float32 difference
            let v1 = if n_pts >= 9 { f64x3(sub3_32::<O>(p4, path.pt(half, n_pts - 9))) } else { sub3::<O>(q4, p0) };
            if turns::<O>(v1, v2) {
                break;
            }
        }
        if n_pts >= 9 {
            let p9 = path.pt(half, n_pts - 9);
            let q9 = f64x3(p9);
            let w2 = sub3::<O>(np, q9);
            let w1 = if n_pts >= 19 { f64x3(sub3_32::<O>(p9, path.pt(half, n_pts - 19))) } else { sub3::<O>(q9, p0) };
            if turns::<O>(w1, w2) {
                break;
            }
        }
        let p_new = trilin_u8::<O, F>(f, np[0], np[1], np[2]);
        let mid =
            [O::mul(0.5, O::add(p[0], np[0])), O::mul(0.5, O::add(p[1], np[1])), O::mul(0.5, O::add(p[2], np[2]))];
        let p_mid = trilin_u8::<O, F>(f, mid[0], mid[1], mid[2]);
        if p_new < 76.5 || p_mid < 76.5 {
            if p_new < 52.5 || p_mid < 52.5 {
                break;
            }
            low_pres_run += 1;
            if low_pres_run > MAX_LOW {
                break;
            }
        } else {
            low_pres_run = 0;
        }
        if covered_same_dir::<O, F, R>(f, r, np, nt) {
            cov_run += 1;
            if cov_run >= 2 {
                break;
            }
        } else {
            cov_run = 0;
        }
        let mut near = false;
        let mut j = 0;
        let scan = n_pts.saturating_sub(5);
        while j < scan {
            let dd = sub3::<O>(np, f64x3(path.pt(half, j)));
            if dot::<O>(dd, dd) < 9.0 {
                near = true;
                break;
            }
            j += 1;
        }
        if near {
            break;
        }
        let tv = [
            O::add(O::add(O::mul(0.5, nt[0]), O::mul(0.3, dv[0])), O::mul(0.2, u[0])),
            O::add(O::add(O::mul(0.5, nt[1]), O::mul(0.3, dv[1])), O::mul(0.2, u[1])),
            O::add(O::add(O::mul(0.5, nt[2]), O::mul(0.3, dv[2])), O::mul(0.2, u[2])),
        ];
        t = div3::<O>(tv, norm::<O>(tv));
        p = np;
        path.push(half, n_pts, f32x3(p), f32x3(t));
        n_pts += 1;
    }
    let ran_out = n_pts == cap;
    while n_pts > 0 && cov_run > 0 {
        n_pts -= 1;
        cov_run -= 1;
    }
    while n_pts > 0 {
        let q = f64x3(path.pt(half, n_pts - 1));
        if trilin_u8::<O, F>(f, q[0], q[1], q[2]) >= 76.5 {
            break;
        }
        n_pts -= 1;
    }
    (n_pts, ran_out)
}

/// Whether a half traced with `cap` steps must go back to the CPU: `trace_seed` refuses a half of [`HALF`]
/// points, and a half that ran all of a smaller cap might have gone on
#[inline(always)]
fn refused(n: u32, ran_out: bool, cap: u32) -> bool {
    ran_out && (cap < HALF || n == HALF)
}

/// Both halves of `trace_seed` from the recentred start `c` with aligned direction `u`, into `path`: `(nf, nb)`, or
/// None when the seed goes back to the CPU (a half [`refused`])
///
/// The halves run as two turns of one loop, so the device holds one copy of `trace_half`
#[inline(always)]
pub fn trace_halves<O: Ops, F: Fields, R: Reads, P: Path>(
    f: &F,
    r: &mut R,
    path: &mut P,
    c: [f64; 3],
    u: [f64; 3],
    cap: u32,
) -> Option<(u32, u32)> {
    let mut nf = 0;
    let mut t = u;
    let mut half = 0;
    while half < 2 {
        if half == 1 {
            t = back_dir::<O, P>(path, nf, c, u);
        }
        let (n, ran_out) = trace_half::<O, F, R, P>(f, r, path, half, c, t, cap);
        if refused(n, ran_out, cap) {
            return None;
        }
        if half == 1 {
            return Some((nf, n));
        }
        nf = n;
        half += 1;
    }
    None
}

/// The back half's start direction: `-u`, or away from the forward half's fourth point when it has three or more
#[inline(always)]
fn back_dir<O: Ops, P: Path>(path: &P, nf: u32, c: [f64; 3], u: [f64; 3]) -> [f64; 3] {
    let bu = [-u[0], -u[1], -u[2]];
    if nf < 3 {
        return bu;
    }
    let k_dep = if nf - 1 < 3 { nf - 1 } else { 3 };
    let df = sub3::<O>(f64x3(path.pt(0, k_dep)), c);
    let dfn = norm::<O>(df);
    // a NaN length fails the test, as on the CPU, and takes the fallback
    let long = dfn > 1e-05;
    if !long {
        return bu;
    }
    let df = div3::<O>(df, dfn);
    let b = [
        -O::add(O::mul(0.55, df[0]), O::mul(0.45, u[0])),
        -O::add(O::mul(0.55, df[1]), O::mul(0.45, u[1])),
        -O::add(O::mul(0.55, df[2]), O::mul(0.45, u[2])),
    ];
    div3::<O>(b, norm::<O>(b))
}

/// `|v|²` in float32, left to right
#[inline(always)]
fn norm2_32<O: Ops>(v: [f32; 3]) -> f32 {
    dot32::<O>(v, v)
}

/// `np.dot(delta, delta)` of a float32 3-vector: each product rounded to float32, summed in float64
#[inline(always)]
fn np_dot3<O: Ops>(v: [f32; 3]) -> f32 {
    let mut acc = 0.0f64;
    acc = O::add(acc, O::mul32(v[0], v[0]) as f64);
    acc = O::add(acc, O::mul32(v[1], v[1]) as f64);
    acc = O::add(acc, O::mul32(v[2], v[2]) as f64);
    acc as f32
}

/// `first_curve_split`'s bend test at `l`, `c`, `r` of the piece starting at `a`
#[inline(always)]
fn bend<O: Ops, C: Curve>(cv: &C, a: u32, l: u32, c: u32, r: u32) -> bool {
    let (pl, pc, pr) = (cv.pt(a + l), cv.pt(a + c), cv.pt(a + r));
    let (v1, v2) = (sub3_32::<O>(pc, pl), sub3_32::<O>(pr, pc));
    let (l1, l2) = (O::sqrt32(norm2_32::<O>(v1)), O::sqrt32(norm2_32::<O>(v2)));
    (l1 as f64) > 1e-05 && (l2 as f64) > 1e-05 && (O::div32(dot32::<O>(v1, v2), O::mul32(l1, l2)) as f64) < COS55
}

/// `first_curve_split` (`kernels::first_curve_split`) of the piece `[a, b)` of the joined path, all float32
#[inline(always)]
fn first_curve_split<O: Ops, C: Curve>(cv: &mut C, a: u32, b: u32) -> i32 {
    let n = b - a;
    if n < 2 {
        return 0;
    }
    let mut i = 0;
    while i < n {
        let pi = cv.pt(a + i);
        let mut j = i + 7;
        while j < n {
            if (norm2_32::<O>(sub3_32::<O>(cv.pt(a + j), pi)) as f64) < 10.24 {
                return ((i + j) / 2) as i32;
            }
            j += 1;
        }
        i += 1;
    }
    let mut i = 1;
    while i < n.saturating_sub(1) {
        if bend::<O, C>(cv, a, i - 1, i, i + 1) {
            return i as i32;
        }
        i += 1;
    }
    let mut pass = 0;
    while pass < 2 {
        let s = if pass == 0 { 5 } else { 10 };
        if n > 2 * s {
            let mut i = s;
            while i < n - s {
                if bend::<O, C>(cv, a, i - s, i, i + s) {
                    return i as i32;
                }
                i += 1;
            }
        }
        pass += 1;
    }
    cv.set_cum(0, 0.0);
    let mut i = 0;
    while i < n - 1 {
        let step = O::sqrt32(norm2_32::<O>(sub3_32::<O>(cv.pt(a + i + 1), cv.pt(a + i))));
        let next = O::add32(cv.cum(i), step);
        cv.set_cum(i + 1, next);
        i += 1;
    }
    const WIN: u32 = 24;
    if n > WIN {
        let mut i = 0;
        while i < n - WIN {
            let arc_w = sub32::<O>(cv.cum(i + WIN), cv.cum(i));
            if arc_w as f64 > 1.0 {
                let ch_w = O::sqrt32(norm2_32::<O>(sub3_32::<O>(cv.pt(a + i + WIN), cv.pt(a + i))));
                if (ch_w as f64) < O::mul(0.78, arc_w as f64) {
                    return (i + WIN / 2) as i32;
                }
            }
            i += 1;
        }
    }
    -1
}

/// Why a seed's spans send it back to the CPU
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SpansFail {
    /// `valid_curve_spans` bails: a rejected nontrivial curve without an interior split
    Split,
    /// The pending stack or the accepted spans ran out of room
    Full,
}

/// `valid_curve_spans` (`kernels::valid_curve_spans`) over the joined path of `ntot` points with the seed at
/// `seed_index`, then `trace_seed`'s length filter (`n >= 8`, arc length at least 7.5, or 16 for a seed that is
/// not primary): each kept span goes to [`Curve::accept`] in path order. The CPU computes every span before it
/// filters, and a bail anywhere fails the seed, so filtering each span as it is found gives the same result
#[inline(always)]
pub fn spans<O: Ops, C: Curve>(cv: &mut C, ntot: u32, seed_index: u32, primary: bool) -> Result<(), SpansFail> {
    let req_len = if primary { 7.5 } else { py_max(7.5, 16.0f32 as f64) };
    if !cv.push(0, ntot) {
        return Err(SpansFail::Full);
    }
    loop {
        let (a, b) = cv.pop();
        if a == u32::MAX {
            return Ok(());
        }
        let n = b - a;
        if n < 2 {
            continue;
        }
        let mut cut = first_curve_split::<O, C>(cv, a, b);
        let local_seed = seed_index as i32 - a as i32;
        if cut < 0 && 0 < local_seed && local_seed < n as i32 - 1 {
            let ls = local_seed as u32;
            let mut j = 0;
            'outer: while j < ls {
                let pj = cv.pt(a + j);
                let mut k = ls + 1;
                while k < n {
                    if k - j >= 6 && (np_dot3::<O>(sub3_32::<O>(pj, cv.pt(a + k))) as f64) < 9.0 {
                        cut = ((j + k) / 2) as i32;
                        break 'outer;
                    }
                    k += 1;
                }
                j += 1;
            }
        }
        if cut < 0 {
            if keep::<O, C>(cv, a, b, req_len) && !cv.accept(a, b) {
                return Err(SpansFail::Full);
            }
            continue;
        }
        if !(0 < cut && cut < n as i32 - 1) {
            return Err(SpansFail::Split);
        }
        let split = a + cut as u32;
        if !cv.push(split, b) || !cv.push(a, split + 1) {
            return Err(SpansFail::Full);
        }
    }
}

/// `trace_seed`'s filter on one span `[a, b)`: at least 8 points and a float32 arc length of at least `req_len`
#[inline(always)]
fn keep<O: Ops, C: Curve>(cv: &C, a: u32, b: u32, req_len: f64) -> bool {
    if b - a < 8 {
        return false;
    }
    let mut arc_len = 0.0f32;
    let mut j = a;
    while j < b - 1 {
        let d = sub3_32::<O>(cv.pt(j + 1), cv.pt(j));
        arc_len = O::add32(arc_len, O::sqrt32(norm2_32::<O>(d)));
        j += 1;
    }
    (arc_len as f64) >= req_len
}
