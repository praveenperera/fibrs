//! NumPy's arithmetic as villa's strip geometry runs it, on xyz points
//!
//! Each helper reproduces the operation order NumPy uses on this platform, measured against NumPy 2.4 with its
//! bundled OpenBLAS: 1-D dots, `norm` and `sum(-1)` add left to right;
//! `(K, 3) @ (3, 3)` products for K >= 2 go through `dgemm` and accumulate with fused multiply-adds, a single row
//! through `dgemv` adds left to right; `einsum('ij,ij->i')` adds `(x0 x0 + x2 x2) + x1 x1`. NumPy's 3x3
//! matrix-vector product and its 13-term vector-matrix products in `dgemv` mix both and are matched to the last bit
//! only most of the time; geometry agrees to an ulp there

/// A point or direction, xyz
pub type V3 = [f64; 3];

/// A 3x3 matrix `m[row][col]`
pub type M3 = [[f64; 3]; 3];

pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

pub fn div(a: V3, s: f64) -> V3 {
    [a[0] / s, a[1] / s, a[2] / s]
}

pub fn neg(a: V3) -> V3 {
    [-a[0], -a[1], -a[2]]
}

/// `a @ b` of two 1-D vectors, `np.dot`, and the `sum(-1)` of their product: left to right
pub fn dot(a: V3, b: V3) -> f64 {
    (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]
}

/// `np.linalg.norm` of a vector (and of rows with `axis=-1`)
pub fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}

/// `np.einsum('ij,ij->i', d, d)` for one row
pub fn einsum_sq(d: V3) -> f64 {
    (d[0] * d[0] + d[2] * d[2]) + d[1] * d[1]
}

/// `np.cross`
pub fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// villa's `normalize`: `v / max(norm(v), 1e-9)`
pub fn normalize(v: V3) -> V3 {
    div(v, norm(v).max(1e-9))
}

/// Column `c` of a matrix
pub fn col(m: &M3, c: usize) -> V3 {
    [m[0][c], m[1][c], m[2][c]]
}

/// The matrix whose columns are `u`, `v`, `w` (`np.stack([u, v, w], axis=1)`)
pub fn from_cols(u: V3, v: V3, w: V3) -> M3 {
    [[u[0], v[0], w[0]], [u[1], v[1], w[1]], [u[2], v[2], w[2]]]
}

/// `v @ m` for a 1-D `v` (`dgemv`, left to right)
pub fn vec_mat(v: V3, m: &M3) -> V3 {
    [dot(v, col(m, 0)), dot(v, col(m, 1)), dot(v, col(m, 2))]
}

/// One row of a `(K, 3) @ (3, 3)` product with K >= 2 (`dgemm`: fused multiply-adds from the first column)
pub fn row_mat_gemm(v: V3, m: &M3) -> V3 {
    let f = |c: usize| v[2].mul_add(m[2][c], v[1].mul_add(m[1][c], v[0] * m[0][c]));
    [f(0), f(1), f(2)]
}

/// `rows @ m` for a `(K, 3)` array, as NumPy picks `dgemv` for one row and `dgemm` for more
pub fn rows_mat(rows: &[V3], m: &M3) -> Vec<V3> {
    match rows.len() {
        1 => vec![vec_mat(rows[0], m)],
        _ => rows.iter().map(|&r| row_mat_gemm(r, m)).collect(),
    }
}

/// `rows @ m` in place, as [`rows_mat`]
pub fn rows_mat_in_place(rows: &mut [V3], m: &M3) {
    if rows.len() == 1 {
        rows[0] = vec_mat(rows[0], m);
        return;
    }
    for r in rows {
        *r = row_mat_gemm(*r, m);
    }
}

/// `m @ v` for a 3x3 `m` and a 1-D `v` (`dgemv`; NumPy fuses the last product into the sum of the first two in
/// most cases, which this follows)
pub fn mat_vec(m: &M3, v: V3) -> V3 {
    let f = |r: usize| m[r][2].mul_add(v[2], m[r][0] * v[0] + m[r][1] * v[1]);
    [f(0), f(1), f(2)]
}

/// `a.T @ b` of two 3x3 matrices (`dgemm`)
pub fn mat_t_mat(a: &M3, b: &M3) -> M3 {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, x) in row.iter_mut().enumerate() {
            *x = a[2][i].mul_add(b[2][j], a[1][i].mul_add(b[1][j], a[0][i] * b[0][j]));
        }
    }
    out
}

/// `np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])`
pub fn arclength(p: &[V3]) -> Vec<f64> {
    let mut s = Vec::with_capacity(p.len());
    let mut acc = 0.0;
    s.push(0.0);
    for w in p.windows(2) {
        acc += norm(sub(w[1], w[0]));
        s.push(acc);
    }
    s
}

/// A polyline sampled by arclength: `np.interp` on each axis
#[derive(Clone, Copy)]
pub struct Curve<'a> {
    /// The vertices
    pub p: &'a [V3],
    /// Their arclength, non-decreasing
    pub s: &'a [f64],
}

impl<'a> Curve<'a> {
    pub fn new(p: &'a [V3], s: &'a [f64]) -> Self {
        debug_assert_eq!(p.len(), s.len());
        Self { p, s }
    }

    /// The last vertex index j with `s[j] <= x`, for `s[0] <= x < s[last]` (NumPy's binary search)
    fn bracket(&self, x: f64) -> usize {
        self.s.partition_point(|&v| v <= x) - 1
    }

    /// `interp_at(p, s, [x])[0]`: `np.interp` per axis, clamped at both ends
    pub fn at(&self, x: f64) -> V3 {
        let (s, p) = (self.s, self.p);
        let last = s.len() - 1;
        // NumPy returns NaN for a NaN abscissa; villa never asks for one
        if x >= s[last] {
            return p[last];
        }
        if x < s[0] {
            return p[0];
        }
        let j = self.bracket(x);
        if s[j] == x {
            return p[j];
        }
        let dx = s[j + 1] - s[j];
        let f = |a: usize| {
            let slope = (p[j + 1][a] - p[j][a]) / dx;
            slope * (x - s[j]) + p[j][a]
        };
        [f(0), f(1), f(2)]
    }

    /// The total arclength
    pub fn length(&self) -> f64 {
        self.s[self.s.len() - 1]
    }
}

/// `np.arange(start, stop, step)` for a positive step: `ceil((stop - start) / step)` values, `start`, `start + step`,
/// then `start + i * delta` with `delta = (start + step) - start` (NumPy's `fill`)
pub fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let len = ((stop - start) / step).ceil();
    if len.is_nan() || len <= 0.0 {
        return Vec::new();
    }
    let len = len as usize;
    let delta = (start + step) - start;
    (0..len)
        .map(|i| match i {
            0 => start,
            1 => start + step,
            _ => start + i as f64 * delta,
        })
        .collect()
}

/// `np.linspace(start, stop, num)` (endpoint included)
pub fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num <= 1 {
        return vec![start; num];
    }
    let div = (num - 1) as f64;
    let delta = stop - start;
    let step = delta / div;
    let mut y: Vec<f64> = if step == 0.0 {
        (0..num).map(|i| i as f64 / div * delta + start).collect()
    } else {
        (0..num).map(|i| i as f64 * step + start).collect()
    };
    y[num - 1] = stop;
    y
}

/// `np.linspace(start, stop, num, endpoint=False)`
pub fn linspace_open(start: f64, stop: f64, num: usize) -> Vec<f64> {
    let step = (stop - start) / num as f64;
    (0..num).map(|i| i as f64 * step + start).collect()
}

/// villa's `dense_line(points, step)`: every segment cut into `ceil(len / step)` equal parts, zero-length
/// segments (<= 1e-9) dropped, both ends and every vertex kept. None for a polyline without length
pub fn dense_line(points: &[V3], step: f64) -> Option<Vec<V3>> {
    let mut out = Vec::new();
    for w in points.windows(2) {
        let delta = sub(w[1], w[0]);
        let mut length = einsum_sq(delta).sqrt();
        let ratio = length / step;
        // where the einsum length could round across either threshold, villa takes the exact norm
        let close = (ratio - ratio.round_ties_even()).abs() <= 1e-9 * ratio.max(1.0) || (length - 1e-9).abs() <= 1e-12;
        if close {
            length = norm(delta);
        }
        if length <= 1e-9 {
            continue;
        }
        let count = ((length / step).ceil() as i64).max(1);
        for k in 0..count {
            let t = k as f64 / count as f64;
            out.push(add(w[0], scale(delta, t)));
        }
    }
    if out.is_empty() {
        return None;
    }
    out.push(points[points.len() - 1]);
    Some(out)
}

/// `np.add.reduce` of a contiguous float64 vector (NumPy's pairwise sum after the identity)
pub fn sum(a: &[f64]) -> f64 {
    crate::pyfmt::np_sum_f64(a)
}

/// NumPy's float32 `np.exp` on x86-64 with AVX2 and FMA (`simd_exp_FLOAT`, loops_exponent_log.dispatch.c.src in
/// NumPy 2.4): Cody-Waite reduction by ln 2 and a degree 5/2 rational polynomial, every step in float32 with fused
/// multiply-adds. NumPy runs this for every float32 exp on such a CPU, strided or not; it differs from a correctly
/// rounded expf in about 40% of inputs
pub fn exp_f32(x: f32) -> f32 {
    // npy_simd_data.h's constants, as the C compiler rounds them to float32
    const XMAX: f32 = f32::from_bits(0x42b17218);
    const XMIN: f32 = f32::from_bits(0xc2cff1b5);
    const LOG2E: f32 = f32::from_bits(0x3fb8aa3b);
    const MAGIC: f32 = f32::from_bits(0x4b400000);
    const C1: f32 = f32::from_bits(0xbf317200);
    const C2: f32 = f32::from_bits(0xb5bfbe8e);
    const P: [f32; 6] = [
        f32::from_bits(0x3f800000),
        f32::from_bits(0x3f39cbd5),
        f32::from_bits(0x3e7d4c58),
        f32::from_bits(0x3d517d8c),
        f32::from_bits(0x3bdd7159),
        f32::from_bits(0x3a053dd8),
    ];
    const Q: [f32; 3] = [f32::from_bits(0x3f800000), f32::from_bits(0xbe8c6857), f32::from_bits(0x3cb0e832)];
    if x.is_nan() {
        return f32::NAN;
    }
    if x >= XMAX {
        return f32::INFINITY;
    }
    if x <= XMIN {
        return 0.0;
    }
    let quadrant = (x * LOG2E + MAGIC) - MAGIC;
    let r = quadrant.mul_add(C1, x);
    let r = quadrant.mul_add(C2, r);
    let r = quadrant.mul_add(0.0, r);
    let num = P[5].mul_add(r, P[4]);
    let num = num.mul_add(r, P[3]);
    let num = num.mul_add(r, P[2]);
    let num = num.mul_add(r, P[1]);
    let num = num.mul_add(r, P[0]);
    let den = Q[2].mul_add(r, Q[1]);
    let den = den.mul_add(r, Q[0]);
    let poly = num / den;
    // poly * 2^quadrant by adding to the exponent bits; below 2^-125 NumPy divides the result down instead
    if quadrant <= -125.0 {
        let diff = -(quadrant + 125.0);
        let scaled = f32::from_bits(poly.to_bits().wrapping_add(((-125i32) << 23) as u32));
        return scaled / (1u32 << diff as u32) as f32;
    }
    f32::from_bits(poly.to_bits().wrapping_add(((quadrant as i32) << 23) as u32))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `np.interp` at the ends, at vertices, between them and past both ends
    #[test]
    fn interp_matches_numpy_cases() {
        let p = [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [3.0, 2.0, -1.0]];
        let s = [0.0, 1.0, 3.0];
        let c = Curve::new(&p, &s);
        assert_eq!(c.at(-1.0), p[0]);
        assert_eq!(c.at(0.0), p[0]);
        assert_eq!(c.at(1.0), p[1]);
        assert_eq!(c.at(3.0), p[2]);
        assert_eq!(c.at(5.0), p[2]);
        assert_eq!(c.at(2.0), [2.0, 2.0, 1.0]);
        assert_eq!(c.at(0.5), [0.5, 1.0, 1.5]);
    }

    /// `np.arange` lengths at the boundary villa's arcs use, and `np.linspace` ends
    #[test]
    fn arange_and_linspace_match_numpy() {
        assert_eq!(arange(12.0, 20.0 + 1e-9, 8.0), vec![12.0, 20.0]);
        assert_eq!(arange(12.0, 19.999 + 1e-9, 8.0), vec![12.0]);
        assert_eq!(arange(1.0, 16.0 + 1e-9, 1.0).len(), 16);
        assert_eq!(linspace(0.0, 1.0, 5), vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        assert_eq!(linspace_open(0.0, 1.0, 4), vec![0.0, 0.25, 0.5, 0.75]);
    }

    /// Densifying keeps every vertex and cuts each segment into equal parts no longer than the step
    #[test]
    fn dense_line_keeps_vertices() {
        let p = [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [2.5, 0.0, 0.0], [2.5, 1.0, 0.0]];
        let d = dense_line(&p, 1.0).unwrap();
        assert_eq!(
            d,
            // villa's a + (k / n) * (b - a), not k * (b - a) / n
            vec![
                [0.0, 0.0, 0.0],
                [(1.0 / 3.0) * 2.5, 0.0, 0.0],
                [(2.0 / 3.0) * 2.5, 0.0, 0.0],
                [2.5, 0.0, 0.0],
                [2.5, 1.0, 0.0]
            ]
        );
        assert!(dense_line(&[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], 1.0).is_none());
    }
}
