//! The CT seed heading (villa `fibers/frames/ct.py` `oriented_seed_heading`): the sheet normal of the 2/8
//! structure tensor of the 77^3 CT cube at the seed, turned into the family's in-sheet axis and signed along the
//! fibre
//!
//! The tensor reproduces `shared/ct_structure.py` exactly: SciPy's `gaussian_filter1d` (sigma 2, truncate 3, so 13
//! taps; mode `nearest`) through `correlate1d`'s symmetric and antisymmetric loops, in float64 and in the same
//! per-sample operation order, so every pass is bitwise SciPy's while the loops run across a row at once; the
//! Gaussian window and the nine weighted sums use NumPy's pairwise summation. The eigenvectors come from a cyclic
//! Jacobi solver, which agrees with NumPy's `eigh` (LAPACK `dsyevd`) to rounding

use std::fmt;

use color_eyre::eyre::Result;

use super::ct::Volume;
use super::np::{M3, V3, cross, dot, norm, scale, sub};

/// villa's `SeedHeadingError`: the seed and family give no usable CT heading; the fibre is listed as failed
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeedHeadingError(pub &'static str);

impl fmt::Display for SeedHeadingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for SeedHeadingError {}

/// The context cube's radius (`CT_CONTEXT_RADIUS`) and side
pub(super) const RADIUS: i64 = 38;
pub(super) const SIDE: usize = 77;
const N: usize = SIDE * SIDE * SIDE;
const PLANE: usize = SIDE * SIDE;
/// Taps on each side of the 13-tap derivative kernels
const HALF: usize = 6;

/// `_gaussian_kernel1d(2, 0, 6)[::-1]`, as SciPy builds it (bits)
const SMOOTH: [u64; 13] = [
    0x3f622be4d4dbc2e6,
    0x3f81f7a6381b2e4a,
    0x3f9babf5688e5b99,
    0x3fb0986222208acd,
    0x3fbf0106659c27a1,
    0x3fc68e26cfc05da8,
    0x3fc98ef890f4b77b,
    0x3fc68e26cfc05da8,
    0x3fbf0106659c27a1,
    0x3fb0986222208acd,
    0x3f9babf5688e5b99,
    0x3f81f7a6381b2e4a,
    0x3f622be4d4dbc2e6,
];

/// `_gaussian_kernel1d(2, 1, 6)[::-1]` (bits)
const DERIV: [u64; 13] = [
    0xbf6b41d73f49a459,
    0xbf86758fc621f9dc,
    0xbf9babf5688e5b99,
    0xbfa8e4933330d034,
    0xbfaf0106659c27a1,
    0xbfa68e26cfc05da8,
    0x0000000000000000,
    0x3fa68e26cfc05da8,
    0x3faf0106659c27a1,
    0x3fa8e4933330d034,
    0x3f9babf5688e5b99,
    0x3f86758fc621f9dc,
    0x3f6b41d73f49a459,
];

/// One `correlate1d` kernel: its centre weight and the weights of offsets -6..-1, with the SciPy loop it takes
#[derive(Clone, Copy)]
struct Kernel {
    centre: f64,
    side: [f64; HALF],
    /// Symmetric (`+`) or antisymmetric (`-`) pairs
    symmetric: bool,
}

impl Kernel {
    fn of(bits: &[u64; 13], symmetric: bool) -> Self {
        Self { centre: f64::from_bits(bits[HALF]), side: std::array::from_fn(|k| f64::from_bits(bits[k])), symmetric }
    }
}

/// Working buffers for one structure tensor, reused across seeds: the cube, its image, and one z plane of each
/// pass. The tensor is computed a plane at a time so a thread's working set stays near its cache (full-volume
/// intermediates made the tensor memory-bound past four threads)
pub struct Workspace {
    pub(super) cube: Vec<u8>,
    img: Vec<f64>,
    /// The z passes of the current plane: derivative and smoothing
    za: Vec<f64>,
    zs: Vec<f64>,
    plane: Vec<f64>,
    /// gz, gy, gx of the current plane
    g: [Vec<f64>; 3],
    line: Vec<f64>,
}

impl Default for Workspace {
    fn default() -> Self {
        Self {
            cube: vec![0; N],
            img: vec![0.0; N],
            za: vec![0.0; PLANE],
            zs: vec![0.0; PLANE],
            plane: vec![0.0; PLANE],
            g: [vec![0.0; PLANE], vec![0.0; PLANE], vec![0.0; PLANE]],
            line: vec![0.0; SIDE + 2 * HALF],
        }
    }
}

/// Plane `z` along z (axis 0): `out = in[z] * w0 + sum over k of (in[z-k] op in[z+k]) * w[-k]`, edges clamped
fn pass_z(input: &[f64], z: usize, o: &mut [f64], k: Kernel) {
    let centre = &input[z * PLANE..(z + 1) * PLANE];
    for (d, &c) in o.iter_mut().zip(centre) {
        *d = c * k.centre;
    }
    for (j, &w) in k.side.iter().enumerate() {
        let off = HALF - j;
        let before = &input[z.saturating_sub(off) * PLANE..][..PLANE];
        let after = &input[(z + off).min(SIDE - 1) * PLANE..][..PLANE];
        accumulate(o, before, after, w, k.symmetric);
    }
}

/// `o += (before op after) * w` over a row, `op` + or -
#[inline(always)]
fn accumulate(o: &mut [f64], before: &[f64], after: &[f64], w: f64, symmetric: bool) {
    if symmetric {
        for ((d, &b), &a) in o.iter_mut().zip(before).zip(after) {
            *d += (b + a) * w;
        }
    } else {
        for ((d, &b), &a) in o.iter_mut().zip(before).zip(after) {
            *d += (b - a) * w;
        }
    }
}

/// Along y (axis 1) within one z plane
fn pass_y(input: &[f64], out: &mut [f64], k: Kernel) {
    for y in 0..SIDE {
        let o = &mut out[y * SIDE..(y + 1) * SIDE];
        for (d, &c) in o.iter_mut().zip(&input[y * SIDE..(y + 1) * SIDE]) {
            *d = c * k.centre;
        }
        for (j, &w) in k.side.iter().enumerate() {
            let off = HALF - j;
            let before = &input[y.saturating_sub(off) * SIDE..][..SIDE];
            let after = &input[(y + off).min(SIDE - 1) * SIDE..][..SIDE];
            accumulate(o, before, after, w, k.symmetric);
        }
    }
}

/// Along x (axis 2) within one z plane, through a line padded with its edge values
fn pass_x(input: &[f64], out: &mut [f64], line: &mut [f64], k: Kernel) {
    for y in 0..SIDE {
        let row = &input[y * SIDE..(y + 1) * SIDE];
        line[..HALF].fill(row[0]);
        line[HALF..HALF + SIDE].copy_from_slice(row);
        line[HALF + SIDE..].fill(row[SIDE - 1]);
        let o = &mut out[y * SIDE..(y + 1) * SIDE];
        for (d, &c) in o.iter_mut().zip(row) {
            *d = c * k.centre;
        }
        for (j, &w) in k.side.iter().enumerate() {
            let off = HALF - j;
            accumulate(o, &line[HALF - off..][..SIDE], &line[HALF + off..][..SIDE], w, k.symmetric);
        }
    }
}

/// NumPy's pairwise sum (`np.add.reduce` without its identity) of `n` values in `L` lanes at once, the values
/// produced in order by `fill`: its blocks of at most 128 are requested in ascending order, so a producer can
/// stream them
fn pairwise<const L: usize>(n: usize, fill: &mut impl FnMut(&mut [[f64; L]]), buf: &mut [[f64; L]; 128]) -> [f64; L] {
    if n > 128 {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        let a = pairwise(n2, fill, buf);
        let b = pairwise(n - n2, fill, buf);
        return std::array::from_fn(|l| a[l] + b[l]);
    }

    let a = &mut buf[..n];
    fill(a);
    if n < 8 {
        let mut res = [-0.0; L];
        for x in a.iter() {
            for l in 0..L {
                res[l] += x[l];
            }
        }
        return res;
    }

    let mut r: [[f64; L]; 8] = std::array::from_fn(|k| a[k]);
    let full = n - n % 8;
    for chunk in a[8..full].as_chunks::<8>().0 {
        for (acc, x) in r.iter_mut().zip(chunk) {
            for l in 0..L {
                acc[l] += x[l];
            }
        }
    }
    let mut res: [f64; L] = std::array::from_fn(|l| {
        ((r[0][l] + r[1][l]) + (r[2][l] + r[3][l])) + ((r[4][l] + r[5][l]) + (r[6][l] + r[7][l]))
    });
    for x in &a[full..] {
        for l in 0..L {
            res[l] += x[l];
        }
    }
    res
}

/// `np.sum` of `n` float64 values per lane, streamed by `fill`
fn np_sum<const L: usize>(n: usize, fill: &mut impl FnMut(&mut [[f64; L]])) -> [f64; L] {
    let sums = pairwise(n, fill, &mut [[0.0; L]; 128]);
    sums.map(|v| 0.0 + v)
}

/// The cube's voxels in row-major (z, y, x) order
#[derive(Default)]
struct Voxels(usize);

impl Voxels {
    fn next(&mut self) -> (usize, usize, usize) {
        let i = self.0;
        self.0 += 1;
        (i / PLANE, i / SIDE % SIDE, i % SIDE)
    }
}

impl Workspace {
    /// `ct_structure_tensor(cube, centre, derivative_sigma=2, integration_sigma=8)`, axes xyz, of the cube in
    /// `self.cube` (zyx, 77^3 uint8)
    pub fn structure_tensor(&mut self, centre_zyx: V3) -> M3 {
        for (d, &c) in self.img.iter_mut().zip(&self.cube) {
            *d = c as f64 / 255.0;
        }
        // the integration window: separable Gaussians, their outer product normalized by its sum
        let window: [[f64; SIDE]; 3] = std::array::from_fn(|a| {
            std::array::from_fn(|i| {
                let t = (i as f64 - centre_zyx[a]) / 8.0;
                (-0.5 * (t * t)).exp()
            })
        });
        let [wz, wy, wx] = &window;
        let mut voxels = Voxels::default();
        let [total] = np_sum(N, &mut |out: &mut [[f64; 1]]| {
            for o in out {
                let (z, y, x) = voxels.next();
                *o = [(wz[z] * wy[y]) * wx[x]];
            }
        });

        // tensor[i][j] = sum((w * g_i) * g_j), zyx, each plane's gradients as the sum reaches it; LAPACK reads the
        // lower triangle, so only it is summed
        let mut voxels = Voxels::default();
        let mut current = usize::MAX;
        let sums = np_sum(N, &mut |out: &mut [[f64; 6]]| {
            for o in out {
                let (z, y, x) = voxels.next();
                if z != current {
                    self.gradients(z);
                    current = z;
                }
                let e = y * SIDE + x;
                let w = ((wz[z] * wy[y]) * wx[x]) / total;
                let g = [self.g[0][e], self.g[1][e], self.g[2][e]];
                let wg = g.map(|v| w * v);
                *o = [wg[0] * g[0], wg[1] * g[0], wg[1] * g[1], wg[2] * g[0], wg[2] * g[1], wg[2] * g[2]];
            }
        });
        let [t00, t10, t11, t20, t21, t22] = sums;
        let t = [[t00, t10, t20], [t10, t11, t21], [t20, t21, t22]];
        // zyx array axes -> world xyz
        std::array::from_fn(|i| std::array::from_fn(|j| t[2 - i][2 - j]))
    }

    /// Plane `z` of gz = x(y(z'(img))), gy = x(y'(s0)) and gx = x'(y(s0)), with s0 = z(img)
    fn gradients(&mut self, z: usize) {
        let smooth = Kernel::of(&SMOOTH, true);
        let deriv = Kernel::of(&DERIV, false);
        pass_z(&self.img, z, &mut self.za, deriv);
        pass_z(&self.img, z, &mut self.zs, smooth);
        let [gz, gy, gx] = &mut self.g;
        pass_y(&self.za, &mut self.plane, smooth);
        pass_x(&self.plane, gz, &mut self.line, smooth);
        pass_y(&self.zs, &mut self.plane, deriv);
        pass_x(&self.plane, gy, &mut self.line, smooth);
        pass_y(&self.zs, &mut self.plane, smooth);
        pass_x(&self.plane, gx, &mut self.line, deriv);
    }
}

/// Eigenvalues (ascending) and eigenvectors (columns) of a symmetric 3x3 matrix, by cyclic Jacobi rotations to
/// convergence
pub fn eigh(m: &M3) -> (V3, M3) {
    let mut a = *m;
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _sweep in 0..64 {
        let off = a[0][1].abs() + a[0][2].abs() + a[1][2].abs();
        if off == 0.0 {
            break;
        }
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            if a[p][q] == 0.0 {
                continue;
            }
            // the rotation that zeroes a[p][q] (Golub and Van Loan's symmetric Schur decomposition)
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let t = if theta == 0.0 { 1.0 } else { t };
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            let r = 3 - p - q;
            let (app, aqq, apq) = (a[p][p], a[q][q], a[p][q]);
            a[p][p] = app - t * apq;
            a[q][q] = aqq + t * apq;
            a[p][q] = 0.0;
            a[q][p] = 0.0;
            let (arp, arq) = (a[r][p], a[r][q]);
            a[r][p] = c * arp - s * arq;
            a[p][r] = a[r][p];
            a[r][q] = s * arp + c * arq;
            a[q][r] = a[r][q];
            for row in &mut v {
                let (vp, vq) = (row[p], row[q]);
                row[p] = c * vp - s * vq;
                row[q] = s * vp + c * vq;
            }
        }
    }
    let mut order = [0usize, 1, 2];
    order.sort_by(|&i, &j| a[i][i].total_cmp(&a[j][j]));
    let values = order.map(|i| a[i][i]);
    let vectors = std::array::from_fn(|r| order.map(|i| v[r][i]));
    (values, vectors)
}

/// The sheet normal (world xyz) of a structure tensor (xyz), as `ct_sheet_normal` with `min_gap` 1e-6
pub fn sheet_normal(tensor_xyz: &M3) -> Result<V3, SeedHeadingError> {
    // villa solves in array-axis order (zyx) and reverses the vector back
    let zyx: M3 = std::array::from_fn(|i| std::array::from_fn(|j| tensor_xyz[2 - i][2 - j]));
    let (values, vectors) = eigh(&zyx);
    if values[2] <= 1e-12 || values[2] - values[1] <= 1e-6 * values[2] {
        return Err(SeedHeadingError("CT seed context has no identifiable sheet normal"));
    }
    Ok([vectors[2][2], vectors[1][2], vectors[0][2]])
}

/// The family's in-sheet axis for a sheet normal (`sheet_heading`)
pub fn sheet_heading(normal: V3, family: &str) -> Result<V3, SeedHeadingError> {
    let family = super::fibre::family_of(family)
        .map_err(|_| SeedHeadingError("CT seed headings require an H or V fiber family"))?;
    let z = [0.0, 0.0, 1.0];
    let axis = match family {
        super::types::Family::V => sub(z, scale(normal, dot(z, normal))),
        super::types::Family::H => cross(z, normal),
    };
    let size = norm(axis);
    if size < 1e-3 {
        return Err(SeedHeadingError("H/V is ambiguous for this sheet orientation"));
    }
    Ok(super::np::div(axis, size))
}

/// The tracer's seed heading at `pos` (trace voxels): the CT sheet axis for the family, signed along `direction`
/// (`oriented_seed_heading`). `input_scale` is CT voxels per trace voxel. A [`SeedHeadingError`] in the report
/// fails the fibre; anything else (a CT chunk missing from the cache) stops the run
pub fn oriented_seed_heading(
    ct: &Volume,
    input_scale: f64,
    pos: V3,
    family: &str,
    direction: V3,
    work: &mut Workspace,
) -> Result<V3> {
    if !pos.iter().all(|v| v.is_finite()) {
        return Err(SeedHeadingError("Seed position must be a finite xyz vector").into());
    }
    let centre = [pos[2] * input_scale, pos[1] * input_scale, pos[0] * input_scale];
    let start: [i64; 3] = std::array::from_fn(|a| centre[a].floor() as i64 - RADIUS);
    let shape = ct.shape();
    if (0..3).any(|a| start[a] < 0 || start[a] + SIDE as i64 > shape[a] as i64) {
        return Err(SeedHeadingError("CT seed context crosses the volume boundary").into());
    }
    let ustart = start.map(|v| v as usize);
    ct.read_region(ustart, [SIDE; 3], &mut work.cube)?;
    let local = std::array::from_fn(|a| centre[a] - start[a] as f64);
    let tensor = work.structure_tensor(local);
    let normal = sheet_normal(&tensor)?;
    let axis = sheet_heading(normal, family)?;
    Ok(if dot(axis, direction) < 0.0 { super::np::neg(axis) } else { axis })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Jacobi eigenpairs reconstruct the matrix and are orthonormal, eigenvalues ascending
    #[test]
    fn jacobi_eigenpairs() {
        let m = [[4.0, 1.0, -2.0], [1.0, 2.0, 0.5], [-2.0, 0.5, 3.0]];
        let (w, v) = eigh(&m);
        assert!(w[0] <= w[1] && w[1] <= w[2]);
        for k in 0..3 {
            let col = [v[0][k], v[1][k], v[2][k]];
            let mv: V3 = std::array::from_fn(|r| dot(m[r], col));
            for r in 0..3 {
                assert!((mv[r] - w[k] * col[r]).abs() < 1e-13, "{k} {r}");
            }
            assert!((norm(col) - 1.0).abs() < 1e-14);
        }
    }

    /// A V fibre's heading is the steepest in-sheet direction; an H fibre's the horizontal one; a vertical normal
    /// leaves H/V ambiguous
    #[test]
    fn sheet_headings() {
        let n = [1.0, 0.0, 0.0];
        assert_eq!(sheet_heading(n, "V").unwrap(), [0.0, 0.0, 1.0]);
        assert_eq!(sheet_heading(n, "H").unwrap(), [0.0, 1.0, 0.0]);
        assert!(sheet_heading([0.0, 0.0, 1.0], "H").is_err());
        assert_eq!(sheet_heading(n, "X").unwrap_err().0, "CT seed headings require an H or V fiber family");
    }
}
