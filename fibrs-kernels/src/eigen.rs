//! Certified eigenvectors for the direction field: a cyclic Jacobi solve of the voxel's 3×3 structure tensor in
//! float64, and a bound on how far LAPACK's `dsyevd` vector can lie from it, which together prove that `dsyevd`
//! plus the CPU's flip and quantization give the same (nx, ny) bytes; a voxel the bound cannot settle goes back
//! to the host, which runs `dsyevd` itself
//!
//! The bound is the step-0 audit's (`check` and `jacobi` in the FIBRS-GPU audit crate) at a LAPACK backward-error
//! constant of 1e3: on 157M real voxels it certified every voxel it could with no mismatch against `dsyevd`.
//! The arithmetic runs through [`Ops`], so the kernels run it in explicit round-to-nearest instructions and the
//! host's tests run the same code in plain Rust, which never contracts. Plain generic `core` code with no loops
//! indexing an array by a runtime counter, so the device build keeps every array in registers

/// The arithmetic the certification runs in: round-to-nearest, never contracted
pub trait Ops {
    fn mul(a: f64, b: f64) -> f64;
    fn add(a: f64, b: f64) -> f64;
    fn div(a: f64, b: f64) -> f64;
    fn sqrt(a: f64) -> f64;
    fn mul32(a: f32, b: f32) -> f32;
    fn add32(a: f32, b: f32) -> f32;
}

const EPS: f64 = f64::EPSILON;

/// The audit's LAPACK backward-error constant `p`: `dsyevd`'s vector is taken to lie within `p·ε·‖A‖/gap` of the
/// true one
const P: f64 = 1e3;

/// A 3×3 float64 matrix, rows of columns
type M3 = [[f64; 3]; 3];

#[inline(always)]
fn sub<O: Ops>(a: f64, b: f64) -> f64 {
    O::add(a, -b)
}

/// `|a|` by clearing the sign bit, as `f64::abs` does
#[inline(always)]
fn abs(a: f64) -> f64 {
    f64::from_bits(a.to_bits() & !(1 << 63))
}

/// `f64::signum` away from NaN: -1 when the sign bit is set (-0.0 included), else 1; the callers never see NaN
/// from finite float32 input
#[inline(always)]
fn signum(a: f64) -> f64 {
    if a.is_sign_negative() { -1.0 } else { 1.0 }
}

/// `a² + b² + c²`, summed left to right
#[inline(always)]
fn sum_sq<O: Ops>(a: f64, b: f64, c: f64) -> f64 {
    O::add(O::add(O::mul(a, a), O::mul(b, b)), O::mul(c, c))
}

/// `x[i]` for `i` in 0..3, by selects rather than an indexed load, which would put `x` in local memory on the
/// device
#[inline(always)]
fn pick(x: [f64; 3], i: usize) -> f64 {
    if i == 0 {
        x[0]
    } else if i == 1 {
        x[1]
    } else {
        x[2]
    }
}

/// One plane rotation of the pair `(x, y)`: `(c·x − s·y, s·x + c·y)`
#[inline(always)]
fn rot<O: Ops>(c: f64, s: f64, x: f64, y: f64) -> (f64, f64) {
    (sub::<O>(O::mul(c, x), O::mul(s, y)), O::add(O::mul(s, x), O::mul(c, y)))
}

/// One Jacobi rotation zeroing `a[P][Q]`: A ← JᵀAJ (columns P, Q, then rows P, Q), V ← VJ; nothing when `a[P][Q]`
/// is already zero
#[inline(always)]
fn rotate<O: Ops, const P: usize, const Q: usize>(mut a: M3, mut v: M3) -> (M3, M3) {
    let apq = a[P][Q];
    if apq == 0.0 {
        return (a, v);
    }

    let theta = O::div(sub::<O>(a[Q][Q], a[P][P]), O::mul(2.0, apq));
    let t = if abs(theta) > 1e150 {
        O::div(0.5, theta)
    } else {
        O::div(signum(theta), O::add(abs(theta), O::sqrt(O::add(O::mul(theta, theta), 1.0))))
    };
    let c = O::div(1.0, O::sqrt(O::add(O::mul(t, t), 1.0)));
    let s = O::mul(t, c);

    // written out per row rather than looped, so every index is a constant
    (a[0][P], a[0][Q]) = rot::<O>(c, s, a[0][P], a[0][Q]);
    (a[1][P], a[1][Q]) = rot::<O>(c, s, a[1][P], a[1][Q]);
    (a[2][P], a[2][Q]) = rot::<O>(c, s, a[2][P], a[2][Q]);
    (a[P][0], a[Q][0]) = rot::<O>(c, s, a[P][0], a[Q][0]);
    (a[P][1], a[Q][1]) = rot::<O>(c, s, a[P][1], a[Q][1]);
    (a[P][2], a[Q][2]) = rot::<O>(c, s, a[P][2], a[Q][2]);
    (v[0][P], v[0][Q]) = rot::<O>(c, s, v[0][P], v[0][Q]);
    (v[1][P], v[1][Q]) = rot::<O>(c, s, v[1][P], v[1][Q]);
    (v[2][P], v[2][Q]) = rot::<O>(c, s, v[2][P], v[2][Q]);
    (a, v)
}

/// Cyclic Jacobi on a symmetric 3×3, at most 12 sweeps: the diagonal it ends with, the eigenvectors as the
/// columns of V, and the off-diagonal Frobenius norm `sqrt(2·(a01² + a02² + a12²))` it leaves
#[inline(always)]
fn jacobi<O: Ops>(mut a: M3) -> ([f64; 3], M3, f64) {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut sweep = 0;
    while sweep < 12 {
        let off = sum_sq::<O>(a[0][1], a[0][2], a[1][2]);
        let dg = sum_sq::<O>(a[0][0], a[1][1], a[2][2]);
        if off == 0.0 || off <= O::mul(1e-36, dg) {
            break;
        }

        (a, v) = rotate::<O, 0, 1>(a, v);
        (a, v) = rotate::<O, 0, 2>(a, v);
        (a, v) = rotate::<O, 1, 2>(a, v);
        sweep += 1;
    }

    let off = O::sqrt(O::mul(2.0, sum_sq::<O>(a[0][1], a[0][2], a[1][2])));
    ([a[0][0], a[1][1], a[2][2]], v, off)
}

/// The indices of the smallest and second smallest of `d` under `f64::total_cmp`, lower index first on ties: the
/// first two of a stable sort of 0..3
#[inline(always)]
fn two_smallest(d: [f64; 3]) -> (usize, usize) {
    let less = |i: usize, j: usize| pick(d, i).total_cmp(&pick(d, j)).is_lt();
    let mut i1 = 0;
    if less(1, i1) {
        i1 = 1;
    }
    if less(2, i1) {
        i1 = 2;
    }

    // the other two in index order; the later one wins only when strictly smaller
    let (j, k) = match i1 {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    let i2 = if less(k, j) { k } else { j };
    (i1, i2)
}

/// The CPU's quantization of one direction component: `round_ties_even((c as f32)·127 + 128)` clamped to 1..=255,
/// in float32
///
/// Clamping before rounding gives the same byte, as both bounds are integers, and leaves a value in [1, 255],
/// where the truncation and the fraction `v − trunc(v)` are exact; ties go to the even neighbour
// spelled out rather than `f32::clamp`, whose bounds assertion is a panic path in the device build
#[allow(clippy::manual_clamp)]
#[inline(always)]
fn q<O: Ops>(c: f64) -> u8 {
    let v = O::add32(O::mul32(c as f32, 127.0), 128.0);
    let v = if v < 1.0 {
        1.0
    } else if v > 255.0 {
        255.0
    } else {
        v
    };
    let t = v as u32;
    let frac = O::add32(v, -(t as f32));
    let up = frac > 0.5 || (frac == 0.5 && t & 1 == 1);
    (t + up as u32) as u8
}

/// The quantized (nx, ny) of the smallest-eigenvalue eigenvector of the symmetric tensor
/// m = [a00, a01, a02, a11, a12, a22] (axes z, y, x), when it is certified equal to what dsyevd + the CPU's
/// quantization give; None when it cannot be certified (the host then runs dsyevd)
///
/// The Jacobi eigenvalue estimates lie within η = offF + 100ε‖A‖ of the true ones, so a positive gap after
/// removing 2η separates the smallest eigenvalue; the Jacobi vector's residual over that gap, plus `dsyevd`'s
/// own error at backward-error constant 1e3, bounds both vectors' distance δ from the true eigenvector. When
/// every component within δ of the Jacobi vector quantizes to the same byte (and the z component's sign, which
/// decides the flip, cannot change), `dsyevd`'s bytes are that byte
// `!(x > 0.0)` and `!(x < 0.5)` are the audit's tests, which NaN fails
#[allow(clippy::neg_cmp_op_on_partial_ord)]
#[inline(always)]
pub fn certify<O: Ops>(m: [f32; 6]) -> Option<(u8, u8)> {
    let [a00, a01, a02, a11, a12, a22] = [m[0] as f64, m[1] as f64, m[2] as f64, m[3] as f64, m[4] as f64, m[5] as f64];
    let a0: M3 = [[a00, a01, a02], [a01, a11, a12], [a02, a12, a22]];
    // ‖A‖_F over all nine entries, row-major
    let sq = |acc: f64, x: f64| O::add(acc, O::mul(x, x));
    let sum = sq(sq(sq(sq(sq(sq(sq(sq(sq(0.0, a00), a01), a02), a01), a11), a12), a02), a12), a22);
    let fro = O::sqrt(sum);
    if fro == 0.0 {
        return None;
    }

    let (d, v, off) = jacobi::<O>(a0);
    let (i1, i2) = two_smallest(d);
    let u = [pick(v[0], i1), pick(v[1], i1), pick(v[2], i1)];
    let nu = O::sqrt(sum_sq::<O>(u[0], u[1], u[2]));
    let u = [O::div(u[0], nu), O::div(u[1], nu), O::div(u[2], nu)];

    // residual of the normalized Jacobi vector against the original matrix
    let row = |r: [f64; 3]| O::add(O::add(O::mul(r[0], u[0]), O::mul(r[1], u[1])), O::mul(r[2], u[2]));
    let au = [row(a0[0]), row(a0[1]), row(a0[2])];
    let lam = O::add(O::add(O::mul(u[0], au[0]), O::mul(u[1], au[1])), O::mul(u[2], au[2]));
    let r = O::sqrt(sum_sq::<O>(
        sub::<O>(au[0], O::mul(lam, u[0])),
        sub::<O>(au[1], O::mul(lam, u[1])),
        sub::<O>(au[2], O::mul(lam, u[2])),
    ));

    let rounding = O::mul(O::mul(100.0, EPS), fro);
    let eta = O::add(off, rounding);
    let gap = sub::<O>(sub::<O>(pick(d, i2), pick(d, i1)), O::mul(2.0, eta));
    if !(gap > 0.0) {
        return None;
    }

    let residual = O::div(O::add(r, rounding), gap);
    let lapack = O::div(O::mul(O::mul(P, EPS), fro), gap);
    let delta = O::add(O::add(residual, lapack), O::mul(8.0, EPS));
    if !(delta < 0.5) || abs(u[0]) <= delta {
        return None;
    }

    // flipped so z > 0, then x = u[2], y = u[1]
    let s = signum(u[0]);
    let (x, y) = (O::mul(s, u[2]), O::mul(s, u[1]));
    let (qx, qy) = (q::<O>(x), q::<O>(y));
    let fixed = |c: f64| q::<O>(sub::<O>(c, delta)) == q::<O>(O::add(c, delta));
    if fixed(x) && fixed(y) { Some((qx, qy)) } else { None }
}
