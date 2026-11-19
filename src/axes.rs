//! The direction field: `prepare.tensor_axes` (prepare.py:62) per voxel
//!
//! A voxel's structure tensor reaches 11 voxels (Gaussian σ 0.9 radius 4, gradient 1, Gaussian σ 1.5
//! radius 6), and the reference's 256³ tiles carry a 16-voxel halo, so no output voxel ever sees a
//! tile edge except at the volume edge, where tile and volume edges coincide. Any tiling with an
//! 11-voxel halo therefore gives the reference's values. The filters reproduce SciPy's `correlate1d`
//! (symmetric kernel, float64 sums, float32 after each axis, reflect at the edge) and `np.gradient`;
//! eigenvectors come from the `dsyevd` of numpy's own OpenBLAS on a float64 copy, cast back to float32, which
//! is what `np.linalg.eigh` does with float32 input (`_commonType` always computes in double)

use std::alloc::{Allocator, Global};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use color_eyre::eyre::{Result, bail, eyre};
use libloading::Library;
use rayon::prelude::*;

use crate::arena::BufferPool;
use crate::vol::{Shape, Vol};

/// Halo the tensor needs: 4 + 1 + 6
pub(crate) const HALO: usize = 11;
/// Tile edge for the parallel sweep; a multiple of the brick edge
pub(crate) const CORE: usize = 64;

/// SciPy's Gaussian weights for σ 0.9 (radius 4) and σ 1.5 (radius 6), outermost first, centre last;
/// computed by `scipy.ndimage._gaussian_kernel1d` and stored as their float64 bit patterns
const W09: [u64; 5] =
    [0x3ef7dff10d1ee066, 0x3f5c138a6b624c17, 0x3fa336a4e8e9e375, 0x3fce9aeb020a6f44, 0x3fdc5e85af5bea50];
const W15: [u64; 7] = [
    0x3f17638649a0fcd1,
    0x3f50d891ab53680a,
    0x3f7f1e741f80c9b8,
    0x3fa26dd9e7a284e1,
    0x3fbbfdc88289323c,
    0x3fcb42851d1c8a96,
    0x3fd1058ef29a59d7,
];

type Dsyevd = unsafe extern "C" fn(
    jobz: *const u8,
    uplo: *const u8,
    n: *const i64,
    a: *mut f64,
    lda: *const i64,
    w: *mut f64,
    work: *mut f64,
    lwork: *const i64,
    iwork: *mut i64,
    liwork: *const i64,
    info: *mut i64,
    jobz_len: usize,
    uplo_len: usize,
) -> ();

/// `dsyevd` from numpy's bundled OpenBLAS (ILP64, `scipy_` prefix)
pub struct Lapack {
    _lib: Library,
    dsyevd: Dsyevd,
    lwork: i64,
    liwork: i64,
}

static LAPACK: OnceLock<Lapack> = OnceLock::new();

impl Lapack {
    /// Load the library at `path` once for the process
    pub fn init(path: &Path) -> Result<&'static Lapack> {
        if let Some(l) = LAPACK.get() {
            return Ok(l);
        }
        // SAFETY: loading a shared library runs its initializers; numpy's OpenBLAS is built to be loaded
        // into arbitrary processes
        let lib = unsafe { Library::new(path) }.map_err(|e| eyre!("{}: {e}", path.display()))?;
        // SAFETY: the symbol has the reference LAPACK signature with 64-bit integers
        let dsyevd: Dsyevd = unsafe { *lib.get::<Dsyevd>(b"scipy_dsyevd_64_\0")? };
        let mut lapack = Lapack { _lib: lib, dsyevd, lwork: -1, liwork: -1 };
        // workspace query, as numpy's init_evd does
        let mut a = [0.0f64; 9];
        let mut w = [0.0f64; 3];
        let mut work = [0.0f64; 1];
        let mut iwork = [0i64; 1];
        let info = lapack.call(&mut a, &mut w, &mut work, &mut iwork);
        if info != 0 {
            bail!("dsyevd workspace query failed ({info})");
        }
        lapack.lwork = work[0] as i64;
        lapack.liwork = iwork[0];
        Ok(LAPACK.get_or_init(|| lapack))
    }

    fn call(&self, a: &mut [f64; 9], w: &mut [f64; 3], work: &mut [f64], iwork: &mut [i64]) -> i64 {
        let (n, lda) = (3i64, 3i64);
        let (lwork, liwork) = (self.lwork, self.liwork);
        let mut info = 0i64;
        // SAFETY: buffers are at least as large as the workspace query asked for
        unsafe {
            (self.dsyevd)(
                b"V".as_ptr(),
                b"L".as_ptr(),
                &n,
                a.as_mut_ptr(),
                &lda,
                w.as_mut_ptr(),
                work.as_mut_ptr(),
                &lwork,
                iwork.as_mut_ptr(),
                &liwork,
                &mut info,
                1,
                1,
            );
        }
        info
    }

    /// The eigenvector of the smallest eigenvalue of a symmetric 3×3 matrix (row-major), as
    /// `np.linalg.eigh(m)[1][:, 0]`
    fn smallest(&self, m: [f32; 9], work: &mut Vec<f64>, iwork: &mut Vec<i64>) -> Result<[f32; 3]> {
        work.resize(self.lwork.max(1) as usize, 0.0);
        iwork.resize(self.liwork.max(1) as usize, 0);
        // column-major A(k, i) = m[k][i]; the matrix is symmetric, so this is m itself
        let mut a = [0.0f64; 9];
        for i in 0..3 {
            for k in 0..3 {
                a[k + 3 * i] = m[3 * k + i] as f64;
            }
        }
        let mut w = [0.0f64; 3];
        let info = self.call(&mut a, &mut w, work, iwork);
        if info != 0 {
            bail!("Eigenvalues did not converge");
        }
        // vt.astype(float32)
        Ok([a[0] as f32, a[1] as f32, a[2] as f32])
    }

    /// A voxel's stored direction from its structure tensor `m` (row-major, axes z, y, x): the eigenvector of the
    /// smallest eigenvalue, flipped so z >= 0, its x and y quantized to bytes as `prepare.tensor_axes` stores them
    pub(crate) fn direction(&self, m: [f32; 9], work: &mut Vec<f64>, iwork: &mut Vec<i64>) -> Result<(u8, u8)> {
        let v = self.smallest(m, work, iwork)?;
        // xyz = v[::-1], flipped so z >= 0
        let mut xyz = [v[2], v[1], v[0]];
        if xyz[2] < 0.0 {
            xyz = xyz.map(|c| -c);
        }
        let q = |c: f32| (c * 127.0f32 + 128.0f32).round_ties_even().clamp(1.0, 255.0) as u8;
        Ok((q(xyz[0]), q(xyz[1])))
    }
}

/// Find numpy's OpenBLAS: `FIBRS_LAPACK`, else the venv this executable was started from
/// (`VENV/bin/avf-spline-generator`), else `$VIRTUAL_ENV`
///
/// In each venv, numpy's bundled copy (`numpy.libs`, Linux wheels) comes first, then the `scipy-openblas64`
/// package's, which is the same OpenBLAS build as a standalone wheel. macOS numpy wheels link Accelerate instead
/// and bundle no OpenBLAS, so there the wheel of this tool depends on `scipy-openblas64`
pub fn find_lapack() -> Result<PathBuf> {
    if let Some(p) = std::env::var_os("FIBRS_LAPACK") {
        return Ok(PathBuf::from(p));
    }
    let mut roots = Vec::new();
    if let Some(arg0) = std::env::args_os().next() {
        let arg0 = PathBuf::from(arg0);
        let arg0 = if arg0.components().count() > 1 { Some(arg0) } else { which(&arg0) };
        if let Some(bin) = arg0.as_deref().and_then(Path::parent) {
            roots.push(bin.join(".."));
        }
    }
    if let Some(v) = std::env::var_os("VIRTUAL_ENV") {
        roots.push(PathBuf::from(v));
    }
    for root in roots {
        let Ok(libs) = std::fs::read_dir(root.join("lib")) else { continue };
        for py in libs.flatten() {
            let site = py.path().join("site-packages");
            for dir in [site.join("numpy.libs"), site.join("scipy_openblas64/lib")] {
                if let Some(lib) = openblas_in(&dir) {
                    return Ok(lib);
                }
            }
        }
    }
    bail!(
        "cannot find numpy's OpenBLAS (numpy.libs/libscipy_openblas64_*.so, or the scipy-openblas64 package); \
         install this executable into the venv's bin directory or set FIBRS_LAPACK"
    )
}

/// The `libscipy_openblas64_*` shared library in `dir`, if there is one
fn openblas_in(dir: &Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    entries.flatten().map(|e| e.path()).find(|path| {
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        name.starts_with("libscipy_openblas64_") && (name.ends_with(".so") || name.ends_with(".dylib"))
    })
}

fn which(name: &Path) -> Option<PathBuf> {
    std::env::split_paths(&std::env::var_os("PATH")?).map(|d| d.join(name)).find(|p| p.is_file())
}

/// The GPU kernels' certification of a voxel's direction, which the tests hold against dsyevd on the CPU
#[cfg(test)]
#[path = "../fibrs-kernels/src/eigen.rs"]
mod eigen;

fn weights<const N: usize>(bits: [u64; N]) -> [f64; N] {
    bits.map(f64::from_bits)
}

/// A dense f32 block
struct Block<A: Allocator = Global> {
    dims: Shape,
    data: Vec<f32, A>,
}

impl<A: Allocator> Block<A> {
    fn stride(&self, axis: usize) -> usize {
        match axis {
            0 => self.dims[1] * self.dims[2],
            1 => self.dims[2],
            _ => 1,
        }
    }
}

/// One axis of `scipy.ndimage.correlate1d` with a symmetric kernel `w` (outermost first, centre last)
/// and mode "reflect", in place
///
/// Each output is `line[c] * centre`, then `(line[c - off] + line[c + off]) * w[j]` added for each outer weight
/// in turn, in float64, as SciPy sums it; the loops run that sequence over a whole row of outputs at once so it
/// vectorises, but each output's operations and their order are unchanged, so the result is bitwise the same
fn correlate_axis<A: Allocator>(b: &mut Block<A>, axis: usize, w: &[f64], scratch: &mut Vec<f64>) {
    if axis == 2 {
        correlate_rows(b, w, scratch);
    } else {
        correlate_across_rows(b, axis, w, scratch);
    }
}

/// `correlate_axis` along x, where each line is one contiguous row: the outputs of a row are summed together
fn correlate_rows<A: Allocator>(b: &mut Block<A>, w: &[f64], scratch: &mut Vec<f64>) {
    let r = w.len() - 1;
    let n = b.dims[2];
    let centre = w[r];
    scratch.clear();
    scratch.resize(n + 2 * r + n, 0.0);
    let (line, acc) = scratch.split_at_mut(n + 2 * r);
    for row in b.data.chunks_exact_mut(n) {
        for (l, &v) in line[r..r + n].iter_mut().zip(row.iter()) {
            *l = v as f64;
        }
        for k in 0..r {
            line[r - 1 - k] = line[r + reflect(-1 - k as isize, n)];
            line[r + n + k] = line[r + reflect((n + k) as isize, n)];
        }

        for (a, &l) in acc.iter_mut().zip(&line[r..r + n]) {
            *a = l * centre;
        }
        for (j, &wj) in w[..r].iter().enumerate() {
            let off = r - j;
            let (lo, hi) = (&line[r - off..r - off + n], &line[r + off..r + off + n]);
            for ((a, &l), &h) in acc.iter_mut().zip(lo).zip(hi) {
                *a += (l + h) * wj;
            }
        }

        for (v, &a) in row.iter_mut().zip(acc.iter()) {
            *v = a as f32;
        }
    }
}

/// `correlate_axis` along z or y: the lines of one plane across the axis run side by side as contiguous x rows,
/// so each step of the sum covers a whole x row
fn correlate_across_rows<A: Allocator>(b: &mut Block<A>, axis: usize, w: &[f64], scratch: &mut Vec<f64>) {
    let r = w.len() - 1;
    let n = b.dims[axis];
    let width = b.dims[2];
    let stride = b.stride(axis);
    let other = 1 - axis;
    let (planes, plane_stride) = (b.dims[other], b.stride(other));
    let centre = w[r];
    scratch.clear();
    scratch.resize((n + 2 * r) * width + width, 0.0);
    let (panel, acc) = scratch.split_at_mut((n + 2 * r) * width);
    for p in 0..planes {
        let base = p * plane_stride;
        for k in 0..n {
            let src = &b.data[base + k * stride..][..width];
            for (l, &v) in panel[(r + k) * width..][..width].iter_mut().zip(src) {
                *l = v as f64;
            }
        }
        for k in 0..r {
            let below = r + reflect(-1 - k as isize, n);
            panel.copy_within(below * width..(below + 1) * width, (r - 1 - k) * width);
            let above = r + reflect((n + k) as isize, n);
            panel.copy_within(above * width..(above + 1) * width, (r + n + k) * width);
        }

        for k in 0..n {
            let c = r + k;
            for (a, &l) in acc.iter_mut().zip(&panel[c * width..][..width]) {
                *a = l * centre;
            }
            for (j, &wj) in w[..r].iter().enumerate() {
                let off = r - j;
                let lo = &panel[(c - off) * width..][..width];
                let hi = &panel[(c + off) * width..][..width];
                for ((a, &l), &h) in acc.iter_mut().zip(lo).zip(hi) {
                    *a += (l + h) * wj;
                }
            }

            let out = &mut b.data[base + k * stride..][..width];
            for (v, &a) in out.iter_mut().zip(acc.iter()) {
                *v = a as f32;
            }
        }
    }
}

/// The sample index scipy's "reflect" mode reads for position `i` of a line of `n` samples: half-sample
/// symmetric extension, repeated with period `2n` (`NI_EXTEND_REFLECT`), so a kernel wider than the line
/// reflects more than once
fn reflect(i: isize, n: usize) -> usize {
    let m = i.rem_euclid(2 * n as isize) as usize;
    if m < n { m } else { 2 * n - 1 - m }
}

/// `gaussian_filter(block, sigma)` on float32: axes 0, 1, 2, each rounded to float32
fn gaussian<A: Allocator>(b: &mut Block<A>, w: &[f64], scratch: &mut Vec<f64>) {
    for axis in 0..3 {
        correlate_axis(b, axis, w, scratch);
    }
}

/// `np.gradient` of a float32 block along one axis (unit spacing, first-order edges), in a buffer from the
/// block's own allocator
fn gradient<A: Allocator + Copy>(b: &Block<A>, axis: usize) -> Block<A> {
    let n = b.dims[axis];
    let s = b.stride(axis);
    let mut out = Vec::with_capacity_in(b.data.len(), *b.data.allocator());
    out.resize(b.data.len(), 0.0f32);
    for (i, o) in out.iter_mut().enumerate() {
        let k = (i / s) % n;
        let f = &b.data;
        *o = if k == 0 {
            (f[i + s] - f[i]) / 1.0
        } else if k == n - 1 {
            (f[i] - f[i - s]) / 1.0
        } else {
            (f[i + s] - f[i - s]) / 2.0
        };
    }
    Block { dims: b.dims, data: out }
}

/// One tile's core: origin, extent, nx and ny
type TileAxes = (Shape, Shape, Vec<u8>, Vec<u8>);

/// Compute nx, ny for the whole tracing volume from the two pooled channels
pub fn direction_field(tracing: &[Vol<u8>; 2], lapack: &Lapack) -> Result<(Vol<u8>, Vol<u8>)> {
    let shape = tracing[0].shape();
    let grid = shape.map(|n| n.div_ceil(CORE));
    let tiles: Vec<Shape> =
        (0..grid[0]).flat_map(|z| (0..grid[1]).flat_map(move |y| (0..grid[2]).map(move |x| [z, y, x]))).collect();
    let mut nx = Vol::new(shape, 128u8);
    let mut ny = Vol::new(shape, 128u8);
    // every tile's smoothed block, gradients and products are a few MiB each and mostly of one size, so they come
    // from a pool instead of a fresh mapping each; a thread's next tile takes the buffers its last one freed, so
    // the pool keeps one tile's buffers per thread, up to TILE_POOL_RETAIN
    let threads = rayon::current_num_threads();
    let in_flight = 4 * threads;
    let per_tile = TILE_BUFFERS * BufferPool::footprint(tile_bytes(shape));
    let pool = BufferPool::new((threads * per_tile).min(TILE_POOL_RETAIN));
    // tiles run a few per thread at a time and are written before the next run, so the dense tile results held
    // at once stay small; tile cores are disjoint, so the order they are written in does not matter
    for part in tiles.chunks(in_flight) {
        let results: Vec<Result<Option<TileAxes>>> =
            part.par_iter().map(|&t| tile_axes(tracing, t, lapack, &pool)).collect();
        for r in results {
            let Some((lo, dims, xx, yy)) = r? else { continue };
            nx.write_block(lo, dims, &xx);
            ny.write_block(lo, dims, &yy);
        }
    }
    Ok((nx, ny))
}

/// Most freed tile buffers the direction field keeps for reuse, the bound the README states: one tile's buffers
/// per thread up to 12 threads
const TILE_POOL_RETAIN: usize = 128 << 20;

/// Pooled f32 buffers a tile holds at once: the smoothed block or a product, and three gradients
const TILE_BUFFERS: usize = 4;

/// Bytes of one f32 buffer of an interior tile: the core and its halo on each side, clipped to the volume
fn tile_bytes(shape: Shape) -> usize {
    shape.iter().map(|&n| n.min(CORE + 2 * HALO)).product::<usize>() * size_of::<f32>()
}

fn tile_axes(tracing: &[Vol<u8>; 2], t: Shape, lapack: &Lapack, pool: &BufferPool) -> Result<Option<TileAxes>> {
    let shape = tracing[0].shape();
    let start = t.map(|i| i * CORE);
    let end = [0, 1, 2].map(|a| (start[a] + CORE).min(shape[a]));
    let lo = [0, 1, 2].map(|a| start[a].saturating_sub(HALO));
    let hi = [0, 1, 2].map(|a| (end[a] + HALO).min(shape[a]));
    let core_dims = [0, 1, 2].map(|a| end[a] - start[a]);
    let dims = [0, 1, 2].map(|a| hi[a] - lo[a]);
    // which family each core voxel belongs to, or none: presence = c0 + c1 >= 40, dominant = c1 > c0
    let mut ids: [Vec<(usize, usize)>; 2] = [Vec::new(), Vec::new()];
    let mut rows = [vec![0u8; core_dims[2]], vec![0u8; core_dims[2]]];
    for z in 0..core_dims[0] {
        for y in 0..core_dims[1] {
            let (gz, gy) = (start[0] + z, start[1] + y);
            for (ch, row) in rows.iter_mut().enumerate() {
                tracing[ch].read_row(gz, gy, start[2], row);
            }
            for x in 0..core_dims[2] {
                let gx = start[2] + x;
                let (c0, c1) = (rows[0][x] as u16, rows[1][x] as u16);
                if c0 + c1 < 40 {
                    continue;
                }
                let family = (c1 > c0) as usize;
                let local = ((z * core_dims[1]) + y) * core_dims[2] + x;
                let block = (((gz - lo[0]) * dims[1]) + gy - lo[1]) * dims[2] + gx - lo[2];
                ids[family].push((local, block));
            }
        }
    }
    if ids[0].is_empty() && ids[1].is_empty() {
        return Ok(None);
    }
    let n = core_dims.iter().product();
    let mut xx = vec![128u8; n];
    let mut yy = vec![128u8; n];
    let (w09, w15) = (weights(W09), weights(W15));
    let mut scratch = Vec::new();
    let (mut work, mut iwork) = (Vec::new(), Vec::new());
    for (family, ids) in ids.iter().enumerate() {
        if ids.is_empty() {
            continue;
        }
        let raw = tracing[family].read_block(lo, dims);
        let mut data = Vec::with_capacity_in(raw.len(), pool);
        data.extend(raw.iter().map(|&v| v as f32 / 255.0f32));
        let mut smooth = Block { dims, data };
        gaussian(&mut smooth, &w09, &mut scratch);
        let grad = [0, 1, 2].map(|a| gradient(&smooth, a));
        drop(smooth);
        let mut tensor = vec![[0.0f32; 9]; ids.len()];
        for u in 0..3 {
            for v in u..3 {
                let mut data = Vec::with_capacity_in(grad[u].data.len(), pool);
                data.extend(grad[u].data.iter().zip(&grad[v].data).map(|(a, b)| a * b));
                let mut prod = Block { dims, data };
                gaussian(&mut prod, &w15, &mut scratch);
                for (m, &(_, at)) in tensor.iter_mut().zip(ids) {
                    m[3 * u + v] = prod.data[at];
                    m[3 * v + u] = prod.data[at];
                }
            }
        }
        for (m, &(local, _)) in tensor.iter().zip(ids) {
            (xx[local], yy[local]) = lapack.direction(*m, &mut work, &mut iwork)?;
        }
    }
    Ok(Some((start, core_dims, xx, yy)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_weights_sum_to_one() {
        for w in [weights(W09).to_vec(), weights(W15).to_vec()] {
            let r = w.len() - 1;
            let total = w[r] + 2.0 * w[..r].iter().sum::<f64>();
            assert!((total - 1.0).abs() < 1e-12, "{total}");
        }
    }

    /// `scipy.ndimage.gaussian_filter1d(x, sigma, mode="reflect")` on float32 lines shorter than the
    /// kernel radius (4 voxels is the shortest tracing axis the reference accepts), as f32 bit patterns
    const SCIPY_SHORT_LINES: [(f64, &[u32], &[u32]); 6] = [
        (0.9, &[0x00000000, 0x3f666666, 0x3e800000, 0x3f100000], &[0x3e850000, 0x3ef6f5cd, 0x3ef6d393, 0x3efa036c]),
        (1.5, &[0x00000000, 0x3f666666, 0x3e800000, 0x3f100000], &[0x3eb65875, 0x3ed2ffe2, 0x3eecc526, 0x3ef6af4f]),
        (
            0.9,
            &[0x00000000, 0x3f666666, 0x3e23d70a, 0x3eb851ec, 0x3f23d70a],
            &[0x3e830340, 0x3ee81f81, 0x3ecb6f75, 0x3ed18bda, 0x3f0b4d20],
        ),
        (
            1.5,
            &[0x00000000, 0x3f666666, 0x3e23d70a, 0x3eb851ec, 0x3f23d70a],
            &[0x3eabab6b, 0x3ec01090, 0x3ed3ed9e, 0x3ee6adda, 0x3ef860de],
        ),
        (
            0.9,
            &[
                0x00000000, 0x3f666666, 0x3d4a4588, 0x3de38e39, 0x3e4a4588, 0x3e9e0652, 0x3ee38e39, 0x3f1add3c,
                0x3f4a4588,
            ],
            &[
                0x3e809059, 0x3ed567e6, 0x3e8b11f4, 0x3e1e208e, 0x3e5611b1, 0x3ea3270c, 0x3ee87be8, 0x3f1b2028,
                0x3f3977a5,
            ],
        ),
        (
            1.5,
            &[
                0x00000000, 0x3f666666, 0x3d4a4588, 0x3de38e39, 0x3e4a4588, 0x3e9e0652, 0x3ee38e39, 0x3f1add3c,
                0x3f4a4588,
            ],
            &[
                0x3e9bbe5e, 0x3e9c1c0e, 0x3e8b5652, 0x3e726099, 0x3e839b53, 0x3eae8017, 0x3eeb4e31, 0x3f13d8c4,
                0x3f273cdb,
            ],
        ),
    ];

    #[test]
    fn correlate_matches_scipy_on_short_lines() {
        let mut scratch = Vec::new();
        for (sigma, input, expected) in SCIPY_SHORT_LINES {
            let w = if sigma == 0.9 { weights(W09).to_vec() } else { weights(W15).to_vec() };
            let data = input.iter().map(|&b| f32::from_bits(b)).collect();
            let mut b = Block { dims: [1, 1, input.len()], data };
            correlate_axis(&mut b, 2, &w, &mut scratch);
            let got: Vec<u32> = b.data.iter().map(|v| v.to_bits()).collect();
            assert_eq!(got, expected, "sigma {sigma}, n {}", input.len());
        }
    }

    /// The per-line `correlate_axis` the row-at-a-time version replaced, kept as the bitwise reference
    fn correlate_axis_per_line(b: &mut Block, axis: usize, w: &[f64]) {
        let r = w.len() - 1;
        let n = b.dims[axis];
        let stride = b.stride(axis);
        let centre = w[r];
        let mut line = vec![0.0f64; n + 2 * r];
        let others: Vec<usize> = (0..3).filter(|&a| a != axis).collect();
        let (d0, d1) = (b.dims[others[0]], b.dims[others[1]]);
        let (s0, s1) = (b.stride(others[0]), b.stride(others[1]));
        for i0 in 0..d0 {
            for i1 in 0..d1 {
                let base = i0 * s0 + i1 * s1;
                for k in 0..n {
                    line[r + k] = b.data[base + k * stride] as f64;
                }
                for k in 0..r {
                    line[r - 1 - k] = line[r + reflect(-1 - k as isize, n)];
                    line[r + n + k] = line[r + reflect((n + k) as isize, n)];
                }
                for k in 0..n {
                    let c = r + k;
                    let mut acc = line[c] * centre;
                    for (j, &wj) in w[..r].iter().enumerate() {
                        let off = r - j;
                        acc += (line[c - off] + line[c + off]) * wj;
                    }
                    b.data[base + k * stride] = acc as f32;
                }
            }
        }
    }

    /// Plain Rust arithmetic for the certification: each operation rounds to nearest on its own, as the kernels'
    /// explicit intrinsics do, since Rust never contracts float operations
    struct Plain;

    impl eigen::Ops for Plain {
        fn mul(a: f64, b: f64) -> f64 {
            a * b
        }

        fn add(a: f64, b: f64) -> f64 {
            a + b
        }

        fn div(a: f64, b: f64) -> f64 {
            a / b
        }

        fn sqrt(a: f64) -> f64 {
            a.sqrt()
        }

        fn mul32(a: f32, b: f32) -> f32 {
            a * b
        }

        fn add32(a: f32, b: f32) -> f32 {
            a + b
        }
    }

    /// A small xorshift, so the test tensors need no dependency
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

        /// Uniform in [-1, 1)
        fn signed(&mut self) -> f64 {
            2.0 * self.unit() - 1.0
        }

        /// Uniform over the integers lo..=hi
        fn int(&mut self, lo: i32, hi: i32) -> i32 {
            lo + (self.next() % (hi - lo + 1) as u64) as i32
        }

        /// 10 to a uniform integer power in lo..=hi
        fn scale(&mut self, lo: i32, hi: i32) -> f64 {
            10f64.powi(self.int(lo, hi))
        }

        fn vector(&mut self) -> [f64; 3] {
            [self.signed(), self.signed(), self.signed()]
        }
    }

    fn normalized(v: [f64; 3]) -> [f64; 3] {
        let n = v.iter().map(|c| c * c).sum::<f64>().sqrt();
        v.map(|c| c / n)
    }

    /// An orthonormal basis whose first vector is `u` normalized
    fn basis(u: [f64; 3], rng: &mut Rng) -> [[f64; 3]; 3] {
        let c0 = normalized(u);
        let r = rng.vector();
        let d = r[0] * c0[0] + r[1] * c0[1] + r[2] * c0[2];
        let c1 = normalized([0, 1, 2].map(|i| r[i] - d * c0[i]));
        let c2 = [c0[1] * c1[2] - c0[2] * c1[1], c0[2] * c1[0] - c0[0] * c1[2], c0[0] * c1[1] - c0[1] * c1[0]];
        [c0, c1, c2]
    }

    /// The upper triangle [a00, a01, a02, a11, a12, a22] of the matrix with eigenvectors `cols` and eigenvalues
    /// `lambda`, computed in f64 and rounded to f32 as a smoothed tensor is
    fn spectral(cols: [[f64; 3]; 3], lambda: [f64; 3]) -> [f32; 6] {
        let at = |i: usize, j: usize| (0..3).map(|k| cols[k][i] * lambda[k] * cols[k][j]).sum::<f64>() as f32;
        [at(0, 0), at(0, 1), at(0, 2), at(1, 1), at(1, 2), at(2, 2)]
    }

    /// Three increasing eigenvalues of one scale, well apart, so only the vector under test is hard
    fn separated(rng: &mut Rng) -> [f64; 3] {
        let s = rng.scale(-12, 6);
        let l0 = rng.signed() * s;
        let l1 = l0 + s * (0.1 + rng.unit());
        [l0, l1, l1 + s * (0.1 + rng.unit())]
    }

    fn full(m: [f32; 6]) -> [f32; 9] {
        [m[0], m[1], m[2], m[1], m[3], m[4], m[2], m[4], m[5]]
    }

    /// Test tensors by kind; the kinds after `random_*` are the adversarial ones
    fn tensors() -> Vec<(&'static str, [f32; 6])> {
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let mut out = Vec::new();
        for _ in 0..1_500_000 {
            let s = rng.scale(-20, 20);
            out.push(("random_symmetric", [(); 6].map(|_| (rng.signed() * s) as f32)));
        }
        // structure tensors are smoothed sums of gradient outer products, so positive semidefinite
        for _ in 0..1_500_000 {
            let s = rng.scale(-20, 8);
            let mut a = [0.0f64; 6];
            for _ in 0..rng.int(1, 3) {
                let (g, w) = (rng.vector(), rng.unit());
                let outer = [g[0] * g[0], g[0] * g[1], g[0] * g[2], g[1] * g[1], g[1] * g[2], g[2] * g[2]];
                a = [0, 1, 2, 3, 4, 5].map(|k| a[k] + w * outer[k]);
            }
            out.push(("random_structure", a.map(|v| (v * s) as f32)));
        }
        for _ in 0..1_000_000 {
            let cols = basis(rng.vector(), &mut rng);
            let s = rng.scale(-15, 15);
            out.push(("random_spectral", spectral(cols, [(); 3].map(|_| rng.unit() * s))));
        }

        out.extend([[0.0f32; 6], [-0.0; 6], [0.0, -0.0, 0.0, -0.0, 0.0, -0.0]].map(|m| ("zero", m)));
        // exact diagonals, with ties drawn from a few values, so repeated eigenvalues and an axis-aligned smallest
        // vector (|z| exactly 0 or 1) both occur
        let values = [0.0, 1.0, 2.0, -1.0, 0.5, 1e-30, 1e30, 3.0e-38];
        for _ in 0..100_000 {
            let d = [(); 3].map(|_| values[rng.int(0, values.len() as i32 - 1) as usize] as f32);
            out.push(("diagonal", [d[0], 0.0, 0.0, d[1], 0.0, d[2]]));
            let r = [(); 3].map(|_| (rng.signed() * rng.scale(-10, 10)) as f32);
            out.push(("diagonal", [r[0], 0.0, 0.0, r[1], 0.0, r[2]]));
        }
        for _ in 0..100_000 {
            let cols = basis(rng.vector(), &mut rng);
            let (a, b) = (rng.signed() * rng.scale(-10, 10), rng.signed() * rng.scale(-10, 10));
            for lambda in [[a, a, b], [a, b, b], [a, a, a]] {
                out.push(("repeated", spectral(cols, lambda)));
            }
        }
        for _ in 0..200_000 {
            let cols = basis(rng.vector(), &mut rng);
            let a = rng.unit() * rng.scale(-10, 10);
            let rho = rng.scale(-16, -1) * rng.unit();
            out.push(("near_degenerate", spectral(cols, [a, a * (1.0 + rho), a * (2.0 + rng.unit())])));
        }
        // the smallest vector nearly in the x-y plane, where its z sign decides the flip
        for _ in 0..200_000 {
            let zeta = rng.signed() * rng.scale(-17, -1);
            let cols = basis([zeta, rng.signed(), rng.signed()], &mut rng);
            out.push(("small_z", spectral(cols, separated(&mut rng))));
        }
        for _ in 0..100_000 {
            let s = if rng.next() & 1 == 0 { rng.scale(-46, -36) } else { rng.scale(30, 38) };
            out.push(("extreme_scale", [(); 6].map(|_| (rng.signed() * s) as f32)));
        }
        // the smallest vector's x and y within a hair of a quantization step's edge, (n + 0.5 - 128) / 127
        for _ in 0..300_000 {
            let mut edge = || {
                let n = rng.int(60, 195) as f64;
                let hair = if rng.next().is_multiple_of(4) { 0.0 } else { rng.signed() * rng.scale(-14, -5) };
                (n + 0.5 - 128.0) / 127.0 + hair
            };
            let (x, y) = (edge(), edge());
            let z = (1.0 - x * x - y * y).max(0.0).sqrt();
            let cols = basis([z, y, x], &mut rng);
            out.push(("quantization_edge", spectral(cols, separated(&mut rng))));
        }
        out
    }

    /// Every direction `eigen::certify` certifies equals dsyevd's, quantized as the CPU path stores it, over random
    /// tensors at many scales and adversarial ones; the certification falls back on the hard ones instead
    #[test]
    #[ignore = "needs numpy's OpenBLAS: FIBRS_LAPACK=.../numpy.libs/libscipy_openblas64_*.so cargo test --release \
                certify_matches_dsyevd -- --ignored"]
    fn certify_matches_dsyevd() {
        let lapack = Lapack::init(&find_lapack().unwrap()).unwrap();
        let cases = tensors();
        // per case: whether it was certified, and how it differs from dsyevd when it does
        let results: Vec<(bool, Option<String>)> = cases
            .par_iter()
            .map_init(
                || (Vec::new(), Vec::new()),
                |(work, iwork), &(kind, m)| {
                    let Some(got) = eigen::certify::<Plain>(m) else { return (false, None) };
                    let want = lapack.direction(full(m), work, iwork).unwrap();
                    let bits = m.map(f32::to_bits);
                    (true, (got != want).then(|| format!("{kind} {bits:08x?}: certified {got:?}, dsyevd {want:?}")))
                },
            )
            .collect();

        let mut kinds: Vec<(&str, usize, usize)> = Vec::new();
        for (&(kind, _), &(certified, _)) in cases.iter().zip(&results) {
            if kinds.last().is_none_or(|k| k.0 != kind) {
                kinds.push((kind, 0, 0));
            }
            let k = kinds.last_mut().expect("pushed above");
            k.1 += 1;
            k.2 += certified as usize;
        }
        for (kind, n, certified) in &kinds {
            eprintln!("{kind}: {certified} of {n} certified");
        }
        let mismatches: Vec<&String> = results.iter().filter_map(|r| r.1.as_ref()).collect();
        assert!(
            mismatches.is_empty(),
            "{} certified mismatches, first: {:#?}",
            mismatches.len(),
            &mismatches[..5.min(mismatches.len())]
        );

        let count = |pred: &dyn Fn(&str) -> bool| -> (usize, usize) {
            kinds.iter().filter(|k| pred(k.0)).fold((0, 0), |(n, c), k| (n + k.1, c + k.2))
        };
        let (n, certified) = count(&|k| k.starts_with("random_"));
        assert!(certified as f64 > 0.95 * n as f64, "only {certified} of {n} random tensors certified");
        assert_eq!(count(&|k| k == "zero").1, 0, "a zero tensor was certified");
        let (n, certified) = count(&|k| !k.starts_with("random_"));
        assert!(certified < n, "no adversarial tensor fell back");
    }

    #[test]
    fn correlate_matches_the_per_line_sum_bitwise() {
        // tile-like blocks, including axes shorter than the kernel radius, of pseudo-random tensor-like values
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 24) as f32 * 255.0 - 64.0
        };
        let mut scratch = Vec::new();
        for dims in [[86, 86, 86], [3, 40, 17], [29, 2, 5], [7, 11, 1]] {
            for w in [weights(W09).to_vec(), weights(W15).to_vec()] {
                let data: Vec<f32> = (0..dims.iter().product::<usize>()).map(|_| next()).collect();
                let mut got = Block { dims, data: data.clone() };
                let mut want = Block { dims, data };
                for axis in 0..3 {
                    correlate_axis(&mut got, axis, &w, &mut scratch);
                    correlate_axis_per_line(&mut want, axis, &w);
                    let bits = |b: &Block| b.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                    assert_eq!(bits(&got), bits(&want), "dims {dims:?} axis {axis} radius {}", w.len() - 1);
                }
            }
        }
    }
}
