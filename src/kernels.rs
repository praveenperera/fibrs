//! The reference's Numba kernels (`core.py`), ported expression by expression
//!
//! Every expression keeps the float32/float64 typing Numba infers for it (as Numba's type dump
//! reports it), so results are bitwise the reference's: a float32 times a uint8
//! is float32, a float32 with a float64 literal, an int64 or a uint16 is float64, variables of mixed
//! type are float64, `int()` truncates, and two-argument `min`/`max` keep the first argument on ties.
//! Line numbers refer to `core.py` at bee798f7

use std::sync::OnceLock;

use color_eyre::eyre::{Result, bail};

use crate::arena::{OWN_MAPPING_BYTES, Voxel};
use crate::pyfmt::{py_max, py_min};
use crate::vol::{Place, Shape, SharedVol, SharedVolWriter, Vol};

/// cos(55°): the turn limit used throughout the tracer
pub const COS55: f64 = 0.5735764363510462;

/// Read-only access to a 3D field in z, y, x order
pub trait Grid<T: Copy>: Sync {
    /// Extent as D, H, W
    fn dims(&self) -> [i64; 3];
    /// The value at an in-bounds voxel
    fn at(&self, z: i64, y: i64, x: i64) -> T;

    /// The values at the eight corners of the cell whose lowest corner is `z, y, x`, which must be in bounds
    /// with the cell, in z, y, x order: `[z][y][x]` flattened, so index `4 * dz + 2 * dy + dx`
    #[inline(always)]
    fn corners(&self, z: i64, y: i64, x: i64) -> [T; 8] {
        std::array::from_fn(|k| {
            let k = k as i64;
            self.at(z + (k >> 2), y + (k >> 1 & 1), x + (k & 1))
        })
    }
}

impl<T: Voxel> Grid<T> for Vol<T> {
    #[inline(always)]
    fn dims(&self) -> [i64; 3] {
        self.shape().map(|n| n as i64)
    }

    #[inline(always)]
    fn at(&self, z: i64, y: i64, x: i64) -> T {
        self.get(z as usize, y as usize, x as usize)
    }

    #[inline(always)]
    fn corners(&self, z: i64, y: i64, x: i64) -> [T; 8] {
        Vol::corners(self, z as usize, y as usize, x as usize)
    }
}

/// A dense C-order block, as handed to `stage1_centrality_and_ridge`
pub struct Dense<T> {
    pub dims: [usize; 3],
    pub data: Vec<T>,
}

impl<T: Copy + Sync> Grid<T> for Dense<T> {
    #[inline(always)]
    fn dims(&self) -> [i64; 3] {
        self.dims.map(|n| n as i64)
    }

    #[inline(always)]
    fn at(&self, z: i64, y: i64, x: i64) -> T {
        self.data[((z as usize) * self.dims[1] + y as usize) * self.dims[2] + x as usize]
    }
}

/// `(b as f32 as f64 - 128.0) / 127.0` for every byte, the per-axis part of `decode_dir`
///
/// A table over byte pairs gives the same values but takes 1.5 MiB, and its lookups missed the cache often
/// enough to cost the single-threaded tracer a fifth of its time; the rest of `decode_dir` is a few float operations
const AXIS: [f64; 256] = {
    let mut t = [0.0f64; 256];
    let mut b = 0;
    while b < 256 {
        t[b] = (b as u8 as f32 as f64 - 128.0) / 127.0;
        b += 1;
    }
    t
};

/// `decode_dir` (core.py:8): all float64 after the float32 conversion of each byte
#[inline(always)]
pub fn decode_dir(nx: u8, ny: u8) -> [f64; 3] {
    let mut dx = AXIS[nx as usize];
    let mut dy = AXIS[ny as usize];
    let r2 = dx * dx + dy * dy;
    if r2 <= 1.0 {
        return [dx, dy, (1.0f32 as f64 - r2).sqrt()];
    }
    let inv = 1.0f32 as f64 / r2.sqrt();
    dx *= inv;
    dy *= inv;
    [dx, dy, 0.0f32 as f64]
}

/// The direction bytes `nx`, `ny` of a voxel as one value, `nx << 8 | ny`
#[inline(always)]
pub const fn pack_dir(nx: u8, ny: u8) -> u16 {
    (nx as u16) << 8 | ny as u16
}

/// `decode_dir` of packed direction bytes
#[inline(always)]
pub fn decode_packed(d: u16) -> [f64; 3] {
    decode_dir((d >> 8) as u8, d as u8)
}

/// The direction field as centrality and the tracer read it: each voxel's `nx` and `ny` as one packed value
/// ([`pack_dir`])
///
/// It is held in the form the centrality device reads. On the CPU that is one packed volume, `Vol<u16>`: the tracer
/// is bound by direction reads, and one volume instead of two halves their brick lookups and cache lines. The GPU
/// reads `nx` and `ny` as two volumes, which [`SplitDirs`] reads as packed values. A field is written in its final
/// form ([`Directions`]) and never converted, since converting one into the other would hold both at once
pub trait DirField: Grid<u16> {
    fn shape(&self) -> Shape;

    /// Bytes held by allocated bricks
    fn resident_bytes(&self) -> usize;

    /// A dense C-order block of packed values with origin `lo` and extent `dims`
    fn packed_block(&self, lo: Shape, dims: Shape) -> Vec<u16>;

    /// `nx` and `ny` as two volumes for the GPU's centrality, when the field is held that way
    fn split(&self) -> Option<(&Vol<u8>, &Vol<u8>)>;
}

impl DirField for Vol<u16> {
    fn shape(&self) -> Shape {
        Vol::shape(self)
    }

    fn resident_bytes(&self) -> usize {
        Vol::resident_bytes(self)
    }

    fn packed_block(&self, lo: Shape, dims: Shape) -> Vec<u16> {
        self.read_block(lo, dims)
    }

    fn split(&self) -> Option<(&Vol<u8>, &Vol<u8>)> {
        None
    }
}

/// The form a direction field is written in: [`DirForm::Split`] when the GPU computes centrality, which reads `nx`
/// and `ny` as two volumes, and [`DirForm::Packed`] otherwise
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DirForm {
    Packed,
    Split,
}

/// An owned direction field in one of its two forms ([`DirField`])
pub enum Directions {
    Packed(Vol<u16>),
    Split { nx: Vol<u8>, ny: Vol<u8> },
}

impl Directions {
    /// A field of `shape` in `form` where every voxel reads as straight up, `nx = ny = 128`
    pub fn new(shape: Shape, form: DirForm) -> Self {
        match form {
            DirForm::Packed => Directions::Packed(Vol::new(shape, pack_dir(128, 128))),
            DirForm::Split => Directions::Split { nx: Vol::new(shape, 128), ny: Vol::new(shape, 128) },
        }
    }

    /// Copy dense C-order blocks of `nx` and `ny` bytes with origin `lo` and extent `dims` into the field
    pub fn write_block(&mut self, lo: Shape, dims: Shape, nx: &[u8], ny: &[u8]) {
        match self {
            Directions::Packed(dirs) => {
                let packed: Vec<u16> = nx.iter().zip(ny).map(|(&x, &y)| pack_dir(x, y)).collect();
                dirs.write_block(lo, dims, &packed);
            }
            Directions::Split { nx: to_nx, ny: to_ny } => {
                to_nx.write_block(lo, dims, nx);
                to_ny.write_block(lo, dims, ny);
            }
        }
    }
}

/// A direction field the GPU computed: its `nx` and `ny` volumes, read as packed values
pub struct SplitDirs<'a> {
    pub nx: &'a Vol<u8>,
    pub ny: &'a Vol<u8>,
}

impl Grid<u16> for SplitDirs<'_> {
    #[inline(always)]
    fn dims(&self) -> [i64; 3] {
        self.nx.dims()
    }

    #[inline(always)]
    fn at(&self, z: i64, y: i64, x: i64) -> u16 {
        pack_dir(self.nx.at(z, y, x), self.ny.at(z, y, x))
    }

    #[inline(always)]
    fn corners(&self, z: i64, y: i64, x: i64) -> [u16; 8] {
        let (nx, ny) = (Grid::corners(self.nx, z, y, x), Grid::corners(self.ny, z, y, x));
        std::array::from_fn(|k| pack_dir(nx[k], ny[k]))
    }
}

impl DirField for SplitDirs<'_> {
    fn shape(&self) -> Shape {
        self.nx.shape()
    }

    fn resident_bytes(&self) -> usize {
        self.nx.resident_bytes() + self.ny.resident_bytes()
    }

    fn packed_block(&self, lo: Shape, dims: Shape) -> Vec<u16> {
        let (nx, ny) = (self.nx.read_block(lo, dims), self.ny.read_block(lo, dims));
        nx.iter().zip(&ny).map(|(&x, &y)| pack_dir(x, y)).collect()
    }

    fn split(&self) -> Option<(&Vol<u8>, &Vol<u8>)> {
        Some((self.nx, self.ny))
    }
}

/// `encode_dir` (core.py:23), with float32 tangents
#[inline]
pub fn encode_dir(mut ux: f32, mut uy: f32, uz: f32) -> (u8, u8) {
    if (uz as f64) < 0.0 {
        ux = -ux;
        uy = -uy;
    }
    let b_nx = py_min(py_max(ux as f64 * 127.0 + 128.5, 1.0), 255.0) as i64;
    let b_ny = py_min(py_max(uy as f64 * 127.0 + 128.5, 1.0), 255.0) as i64;
    (b_nx as u8, b_ny as u8)
}

/// The clamp and cell split shared by the trilinear helpers (core.py:36-53)
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
    let ix = x as i64;
    let iy = y as i64;
    let iz = z as i64;
    let fx = (x - ix as f64) as f32;
    let fy = (y - iy as f64) as f32;
    let fz = (z - iz as f64) as f32;
    ([iz, iy, ix], [fz, fy, fx])
}

/// `trilin_scalar_u8` (core.py:34): `fx * arr[..]` is a float32 product
#[inline]
pub fn trilin_u8<G: Grid<u8>>(arr: &G, x: f64, y: f64, z: f64) -> f64 {
    let ([iz, iy, ix], [fz, fy, fx]) = cell(arr.dims(), x, y, z);
    let c = arr.corners(iz, iy, ix);
    let lerp = |k: usize| (1.0 - fx as f64) * c[k] as f64 + (fx * c[k + 1] as f32) as f64;
    let p00 = lerp(0);
    let p01 = lerp(2);
    let p10 = lerp(4);
    let p11 = lerp(6);
    let (fy, fz) = (fy as f64, fz as f64);
    (1.0 - fz) * ((1.0 - fy) * p00 + fy * p01) + fz * ((1.0 - fy) * p10 + fy * p11)
}

/// `trilin_scalar_u16` (core.py:62): float32 times uint16 is float64 in Numba
#[inline]
pub fn trilin_u16<G: Grid<u16>>(arr: &G, x: f64, y: f64, z: f64) -> f64 {
    let ([iz, iy, ix], [fz, fy, fx]) = cell(arr.dims(), x, y, z);
    let fx = fx as f64;
    let c = arr.corners(iz, iy, ix).map(f64::from);
    let lerp = |k: usize| (1.0 - fx) * c[k] + fx * c[k + 1];
    let p00 = lerp(0);
    let p01 = lerp(2);
    let p10 = lerp(4);
    let p11 = lerp(6);
    let (fy, fz) = (fy as f64, fz as f64);
    0.01f32 as f64 * ((1.0 - fz) * ((1.0 - fy) * p00 + fy * p01) + fz * ((1.0 - fy) * p10 + fy * p11))
}

/// The read-only fields one family traces over
pub struct Fields<'a, P, C, D> {
    pub pres: &'a P,
    pub cent: &'a C,
    /// Packed direction bytes, [`pack_dir`]
    pub dirs: &'a D,
}

impl<P: Grid<u8>, C: Grid<u16>, D: Grid<u16>> Fields<'_, P, C, D> {
    #[inline(always)]
    fn dir_at(&self, z: i64, y: i64, x: i64) -> [f64; 3] {
        decode_packed(self.dirs.at(z, y, x))
    }

    /// `trilin_dir_aligned` (core.py:92)
    pub fn dir_aligned(&self, x: f64, y: f64, z: f64, rx: f64, ry: f64, rz: f64) -> [f64; 3] {
        let ([iz, iy, ix], [fz, fy, fx]) = cell(self.pres.dims(), x, y, z);
        let (fx, fy, fz) = (fx as f64, fy as f64, fz as f64);
        let (mut vx, mut vy, mut vz) = (0.0f64, 0.0f64, 0.0f64);
        let pres = self.pres.corners(iz, iy, ix);
        let dirs = self.dirs.corners(iz, iy, ix);
        for dz_i in 0..2 {
            let wz = if dz_i == 0 { 1.0 - fz } else { fz };
            for dy_i in 0..2 {
                let wy = wz * (if dy_i == 0 { 1.0 - fy } else { fy });
                for dx_i in 0..2 {
                    let w = wy * (if dx_i == 0 { 1.0 - fx } else { fx });
                    let k = 4 * dz_i + 2 * dy_i + dx_i;
                    let pw = w * (pres[k] as f32 as f64 + 10.0);
                    let [mut ux, mut uy, mut uz] = decode_packed(dirs[k]);
                    if ux * rx + uy * ry + uz * rz < 0.0 {
                        ux = -ux;
                        uy = -uy;
                        uz = -uz;
                    }
                    vx += pw * ux;
                    vy += pw * uy;
                    vz += pw * uz;
                }
            }
        }
        let n = (vx * vx + vy * vy + vz * vz).sqrt();
        if n > 1e-08 {
            vx /= n;
            vy /= n;
            vz /= n;
        }
        [vx, vy, vz]
    }

    /// `recenter_normal_plane` (core.py:162)
    #[allow(clippy::too_many_arguments)]
    pub fn recenter(
        &self,
        x: f64,
        y: f64,
        z: f64,
        tx: f64,
        ty: f64,
        tz: f64,
        max_shift: f64,
        n_iter: usize,
    ) -> [f64; 3] {
        let [d, h, w] = self.cent.dims().map(|n| n as f64);
        let [e1x, e1y, e1z, e2x, e2y, e2z] = ortho_basis(tx, ty, tz);
        let (mut cx, mut cy, mut cz) = (x, y, z);
        let ring = ring();
        let mut vals = [0.0f32; 13];
        for _ in 0..n_iter {
            let mut c_min = 1_000_000_000.0f32;
            let mut c_max = -1_000_000_000.0f32;
            for i in 0..13 {
                let (u1, u2) = (RING_U1[i] as f64, RING_U2[i] as f64);
                let px = cx + u1 * e1x + u2 * e2x;
                let py = cy + u1 * e1y + u2 * e2y;
                let pz = cz + u1 * e1z + u2 * e2z;
                if px < 0.0 || px >= w - 1.001 || py < 0.0 || py >= h - 1.001 || pz < 0.0 || pz >= d - 1.001 {
                    vals[i] = 0.0;
                } else {
                    let [ux, uy, uz] = self.dir_at((pz + 0.5) as i64, (py + 0.5) as i64, (px + 0.5) as i64);
                    let dot = (ux * tx + uy * ty + uz * tz).abs();
                    vals[i] = if dot < 0.82 { 0.0 } else { (trilin_u16(self.cent, px, py, pz) * dot) as f32 };
                }
                if vals[i] < c_min {
                    c_min = vals[i];
                }
                if vals[i] > c_max {
                    c_max = vals[i];
                }
            }
            if ((c_max - c_min) as f64) < 0.5 {
                break;
            }
            let (mut w_sum, mut du1, mut du2) = (0.0f64, 0.0f64, 0.0f64);
            for i in 0..13 {
                let ex = vals[i] - c_min;
                let wgt = (ex * ex) as f64 * ring[i];
                w_sum += wgt;
                du1 += wgt * RING_U1[i] as f64;
                du2 += wgt * RING_U2[i] as f64;
            }
            if w_sum > 1e-06 {
                du1 /= w_sum;
                du2 /= w_sum;
                cx += du1 * e1x + du2 * e2x;
                cy += du1 * e1y + du2 * e2y;
                cz += du1 * e1z + du2 * e2z;
            }
        }
        let (dx, dy, dz) = (cx - x, cy - y, cz - z);
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if dist > max_shift {
            let s = max_shift / dist;
            cx = x + dx * s;
            cy = y + dy * s;
            cz = z + dz * s;
        }
        [cx, cy, cz]
    }
}

const RING_U1: [f32; 13] = [0.0, 0.55, 0.0, -0.55, 0.0, 1.15, 0.81317, 0.0, -0.81317, -1.15, -0.81317, 0.0, 0.81317];
const RING_U2: [f32; 13] = [0.0, 0.0, 0.55, 0.0, -0.55, 0.0, 0.81317, 1.15, 0.81317, 0.0, -0.81317, -1.15, -0.81317];

/// `exp_arr` of `recenter_normal_plane`: exp(-0.5 * r2) in float64 of a float32 r2
fn ring() -> &'static [f64; 13] {
    static RING: OnceLock<[f64; 13]> = OnceLock::new();
    RING.get_or_init(|| {
        std::array::from_fn(|i| {
            let r2 = RING_U1[i] * RING_U1[i] + RING_U2[i] * RING_U2[i];
            (-0.5 * r2 as f64).exp()
        })
    })
}

/// `ortho_basis` (core.py:142)
#[inline]
pub fn ortho_basis(ux: f64, uy: f64, uz: f64) -> [f64; 6] {
    let (mut e1x, mut e1y, mut e1z) = if uz.abs() < 0.85 { (-uy, ux, 0.0) } else { (0.0, -uz, uy) };
    let n1 = (e1x * e1x + e1y * e1y + e1z * e1z).sqrt();
    e1x /= n1;
    e1y /= n1;
    e1z /= n1;
    let e2x = uy * e1z - uz * e1y;
    let e2y = uz * e1x - ux * e1z;
    let e2z = ux * e1y - uy * e1x;
    [e1x, e1y, e1z, e2x, e2y, e2z]
}

/// Coverage of one family: the reference's count (0, 1, 2) and first marking direction, packed in one `u16` per
/// voxel
///
/// 0 is a count of 0 and 1 a count of 2, whose direction the reference never reads; a count of 1 stores the
/// direction as `nx << 8 | ny`, which is at least 256 because `encode_dir` never returns a 0 byte. One volume
/// instead of three saves a byte per covered voxel.
///
/// Seeds are traced on several threads against this shared volume while one [`CoverageWriter`] marks the curves
/// of committed seeds, in seed order. Every read records the brick it touched in a [`ReadSet`], so the writer can
/// tell whether anything a seed read has changed since it was traced
pub struct Coverage(SharedVol);

const UNCOVERED: u16 = 0;
const COVERED_TWICE: u16 = 1;

impl Coverage {
    pub fn new(shape: [usize; 3]) -> Self {
        Self(SharedVol::new(shape))
    }

    /// Whether a voxel has been marked at all
    #[inline]
    pub fn is_covered(&self, reads: &mut ReadSet, z: i64, y: i64, x: i64) -> bool {
        self.read(reads, z, y, x) != UNCOVERED
    }

    #[inline(always)]
    fn read(&self, reads: &mut ReadSet, z: i64, y: i64, x: i64) -> u16 {
        let at = self.0.place(z as usize, y as usize, x as usize);
        reads.push(at.slot());
        self.0.get(at)
    }

    /// Bytes held by allocated bricks
    pub fn resident_bytes(&self) -> usize {
        self.0.resident_bytes()
    }

    /// `is_covered_same_dir` (core.py:237)
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn covered_same_dir(&self, reads: &mut ReadSet, x: f64, y: f64, z: f64, tx: f64, ty: f64, tz: f64) -> bool {
        let [d, h, w] = self.0.shape().map(|n| n as i64);
        let ix = (x + 0.5) as i64;
        let iy = (y + 0.5) as i64;
        let iz = (z + 0.5) as i64;
        if ix < 0 || ix >= w || iy < 0 || iy >= h || iz < 0 || iz >= d {
            return false;
        }
        let c = self.read(reads, iz, iy, ix);
        if c == UNCOVERED {
            return false;
        }
        if c == COVERED_TWICE {
            return true;
        }
        let [ux, uy, uz] = decode_packed(c);
        (ux * tx + uy * ty + uz * tz).abs() > 0.86
    }

    /// The voxels `mark_curve_coverage` (core.py:527, `core_r = 2.05`) would update for one curve, in its order
    ///
    /// Everything that decides whether a voxel near a point is marked reads only the fields, so it is done here,
    /// on the tracing thread; only the update itself, which reads the voxel's coverage, is left to
    /// [`CoverageWriter::apply`]. `first` is the index of `tans[0]` in the seed's [`Traced::tans`]
    pub fn mark_list<P: Grid<u8>, C: Grid<u16>, D: Grid<u16>>(
        &self,
        f: &Fields<P, C, D>,
        pts: &[[f32; 3]],
        tans: &[[f32; 3]],
        first: usize,
        out: &mut Vec<Mark>,
    ) {
        let [d, h, w] = self.0.shape().map(|n| n as i64);
        let core_r2 = (2.05f64 * 2.05) as f32 as f64;
        for (k, (p, t)) in pts.iter().zip(tans).enumerate() {
            let [px, py, pz] = p.map(|v| v as f64);
            let [tx, ty, tz] = t.map(|v| v as f64);
            let (bnx, bny) = encode_dir(t[0], t[1], t[2]);
            debug_assert!(bnx != 0);
            let dir = pack_dir(bnx, bny);
            let tan = u32::try_from(first + k).expect("a seed's curves hold fewer than 2³² points");
            let c0 = trilin_u16(f.cent, px, py, pz);
            let ix0 = (px + 0.5) as i64;
            let iy0 = (py + 0.5) as i64;
            let iz0 = (pz + 0.5) as i64;
            for zz in iz0 - 3..=iz0 + 3 {
                if zz < 0 || zz >= d {
                    continue;
                }
                for yy in iy0 - 3..=iy0 + 3 {
                    if yy < 0 || yy >= h {
                        continue;
                    }
                    for xx in ix0 - 3..=ix0 + 3 {
                        if xx < 0 || xx >= w {
                            continue;
                        }
                        let (vx, vy, vz) = (xx as f64 - px, yy as f64 - py, zz as f64 - pz);
                        let par = vx * tx + vy * ty + vz * tz;
                        if par.abs() > 0.9 {
                            continue;
                        }
                        let perp2 = vx * vx + vy * vy + vz * vz - par * par;
                        if perp2 > 3.5 * 3.5 {
                            continue;
                        }
                        let [ux, uy, uz] = f.dir_at(zz, yy, xx);
                        if (ux * tx + uy * ty + uz * tz).abs() <= 0.85 {
                            continue;
                        }
                        let suppress = perp2 <= core_r2 || {
                            let cq = (f.cent.at(zz, yy, xx) as f32 * 0.01f32) as f64;
                            let (xf, yf, zf) = (xx as f64, yy as f64, zz as f64);
                            let cm1 = trilin_u16(f.cent, 0.6 * px + 0.4 * xf, 0.6 * py + 0.4 * yf, 0.6 * pz + 0.4 * zf);
                            let cm2 =
                                trilin_u16(f.cent, 0.35 * px + 0.65 * xf, 0.35 * py + 0.65 * yf, 0.35 * pz + 0.65 * zf);
                            c0 + 2.0 >= cm1 && cm1 + 3.0 >= cm2 && cm2 + 3.0 >= cq
                        };
                        if suppress {
                            out.push(Mark { at: self.0.place(zz as usize, yy as usize, xx as usize), dir, tan });
                        }
                    }
                }
            }
        }
    }

    /// The coverage's one writer, which commits seeds in order
    ///
    /// # Panics
    ///
    /// When called a second time on the same coverage
    pub fn writer(&self) -> CoverageWriter<'_> {
        CoverageWriter { vol: self.0.writer(), written: vec![0; WRITTEN_SLOTS].into_boxed_slice() }
    }
}

/// The brick slots a seed's tracing read from the coverage, in read order
///
/// A slot read again right after itself is kept once; steps mostly stay in one brick, so the set stays short
#[derive(Default)]
pub struct ReadSet(Vec<usize>);

impl ReadSet {
    #[inline(always)]
    fn push(&mut self, slot: usize) {
        if self.0.last() != Some(&slot) {
            self.0.push(slot);
        }
    }

    pub fn slots(&self) -> &[usize] {
        &self.0
    }
}

/// One voxel update of [`Coverage::mark_list`]: the encoded direction of the marking point and its tangent's
/// index in [`Traced::tans`]
#[derive(Clone, Copy)]
pub struct Mark {
    at: Place,
    dir: u16,
    tan: u32,
}

/// The `written` table's size, as a power of two
const WRITTEN_BITS: u32 = 16;
const WRITTEN_SLOTS: usize = 1 << WRITTEN_BITS;

/// Applies committed seeds' marks to a [`Coverage`] and checks traced seeds' reads against them
///
/// `written` maps each brick slot, by a hash, to one more than the newest seed that changed a voxel in it. Only
/// the last few dozen seeds matter for a check, so a small table that stays in cache replaces one counter per
/// slot: slots that share an entry can only make a seed look stale, never current, and a stale seed is traced
/// again
pub struct CoverageWriter<'c> {
    vol: SharedVolWriter<'c>,
    written: Box<[u32]>,
}

impl CoverageWriter<'_> {
    #[inline(always)]
    fn entry(slot: usize) -> usize {
        // Fibonacci hashing spreads neighbouring slots over the table
        ((slot as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - WRITTEN_BITS)) as usize
    }

    /// Whether a seed that read `reads` while the first `committed` seeds were committed may have read a value a
    /// later commit changed
    pub fn stale(&self, reads: &ReadSet, committed: u32) -> bool {
        reads.slots().iter().any(|&slot| self.written[Self::entry(slot)] > committed)
    }

    /// Mark seed `seed`'s curves, as `mark_curve_coverage` does curve by curve and point by point
    pub fn apply(&mut self, seed: u32, traced: &Traced) {
        let stamp = seed + 1;
        for m in &traced.marks {
            let c = self.vol.get(m.at);
            let value = match c {
                UNCOVERED => m.dir,
                // a count of 2 stays 2 whichever way the new direction points
                COVERED_TWICE => continue,
                _ => {
                    let [tx, ty, tz] = traced.tans[m.tan as usize].map(|v| v as f64);
                    let [ux0, uy0, uz0] = decode_packed(c);
                    if (ux0 * tx + uy0 * ty + uz0 * tz).abs() > 0.85 {
                        continue;
                    }
                    COVERED_TWICE
                }
            };
            self.vol.set(m.at, value);
            self.written[Self::entry(m.at.slot())] = stamp;
        }
    }
}

/// Half-path capacity of `trace_seed_range` (`pts_f`/`pts_b`)
pub const HALF: usize = 32768;

/// Scratch buffers for one tracing thread
pub struct Scratch {
    pts_f: Vec<[f32; 3]>,
    tans_f: Vec<[f32; 3]>,
    pts_b: Vec<[f32; 3]>,
    tans_b: Vec<[f32; 3]>,
    pts_c: Vec<[f32; 3]>,
    tans_c: Vec<[f32; 3]>,
}

impl Default for Scratch {
    fn default() -> Self {
        let buf = || vec![[0.0f32; 3]; HALF];
        Self {
            pts_f: buf(),
            tans_f: buf(),
            pts_b: buf(),
            tans_b: buf(),
            pts_c: vec![[0.0; 3]; 2 * HALF + 1],
            tans_c: vec![[0.0; 3]; 2 * HALF + 1],
        }
    }
}

/// One accepted raw curve; its tangents only matter for coverage, which [`Traced::marks`] holds
pub struct RawCurve {
    pub pts: Vec<[f32; 3]>,
}

/// What tracing one seed found, before it is committed: the curves it accepted, the coverage updates they make,
/// and the coverage bricks it read to decide
///
/// Kept and cleared between seeds, so a thread's buffers are reused
#[derive(Default)]
pub struct Traced {
    pub curves: Vec<RawCurve>,
    pub marks: Vec<Mark>,
    /// The accepted curves' tangents, which [`Mark::tan`] indexes
    pub tans: Vec<[f32; 3]>,
    pub reads: ReadSet,
}

/// Marks a [`Traced`] keeps room for between seeds: as many as fit a page below the size glibc maps on its own,
/// leaving room for its chunk header
///
/// A seed whose curves are accepted makes thousands, and a window of seeds traced ahead each keeping room for the
/// largest it has held would add hundreds of MiB. Room below that size stays in a thread's heap, where shrinking it
/// after each seed made glibc delete and map again a 64 MiB heap for nearly every curve traced: on a dense
/// PHerc0358 crop that was 110,000 `munmap`s, each one stalling every thread's page faults. Room kept below that
/// size is never freed, and only the rare larger list is mapped and unmapped
const MARKS_KEPT: usize = (OWN_MAPPING_BYTES - 4096) / size_of::<Mark>();

impl Traced {
    /// Give back the room for marks beyond [`MARKS_KEPT`]
    pub fn trim(&mut self) {
        if self.marks.capacity() > MARKS_KEPT {
            self.marks.clear();
            self.marks.shrink_to(MARKS_KEPT);
        }
    }

    fn clear(&mut self) {
        self.curves.clear();
        self.marks.clear();
        self.tans.clear();
        self.reads.0.clear();
    }

    /// Drop a result that will not be committed, keeping no more room than [`Traced::trim`] does
    pub fn discard(&mut self) {
        self.clear();
        self.trim();
    }
}

/// `trace_half_buf` (core.py:322) with `step_size = 0.75`, `max_low_steps = 3`
#[allow(clippy::too_many_arguments)]
fn trace_half<P: Grid<u8>, C: Grid<u16>, D: Grid<u16>>(
    f: &Fields<P, C, D>,
    cov: &Coverage,
    reads: &mut ReadSet,
    x0: f64,
    y0: f64,
    z0: f64,
    tx0: f64,
    ty0: f64,
    tz0: f64,
    pts: &mut [[f32; 3]],
    tans: &mut [[f32; 3]],
) -> usize {
    const STEP: f64 = 0.75;
    const MAX_LOW: i64 = 3;
    let [d, h, w] = f.pres.dims().map(|n| n as f64);
    let out =
        |x: f64, y: f64, z: f64| x < 0.25 || x >= w - 1.25 || y < 0.25 || y >= h - 1.25 || z < 0.25 || z >= d - 1.25;
    let max_steps = pts.len();
    let (mut x, mut y, mut z) = (x0, y0, z0);
    let (mut tx, mut ty, mut tz) = (tx0, ty0, tz0);
    let mut n_pts = 0usize;
    let mut low_pres_run = 0i64;
    let mut cov_run = 0i64;
    let p64 = |p: [f32; 3]| p.map(|v| v as f64);
    'steps: for _ in 0..max_steps {
        let mx = x + 0.5 * STEP * tx;
        let my = y + 0.5 * STEP * ty;
        let mz = z + 0.5 * STEP * tz;
        if out(mx, my, mz) {
            break;
        }
        let [mut ux, mut uy, mut uz] = f.dir_aligned(mx, my, mz, tx, ty, tz);
        let dot_u = ux * tx + uy * ty + uz * tz;
        let (nx_p, ny_p, nz_p, ntx, nty, ntz);
        if dot_u < COS55 {
            let mut crossed = false;
            for look_d in [1.5, 2.2] {
                let lx = x + look_d * tx;
                let ly = y + look_d * ty;
                let lz = z + look_d * tz;
                if lx >= 0.25 && lx < w - 1.25 && ly >= 0.25 && ly < h - 1.25 && lz >= 0.25 && lz < d - 1.25 {
                    let lp = trilin_u8(f.pres, lx, ly, lz);
                    let [lux, luy, luz] = f.dir_aligned(lx, ly, lz, tx, ty, tz);
                    if lp >= 72.5 && lux * tx + luy * ty + luz * tz >= COS55 {
                        crossed = true;
                        break;
                    }
                }
            }
            if !crossed {
                break;
            }
            nx_p = x + STEP * tx;
            ny_p = y + STEP * ty;
            nz_p = z + STEP * tz;
            (ntx, nty, ntz) = (tx, ty, tz);
            (ux, uy, uz) = (tx, ty, tz);
        } else {
            let mut ntx1 = 0.45 * tx + 0.55 * ux;
            let mut nty1 = 0.45 * ty + 0.55 * uy;
            let mut ntz1 = 0.45 * tz + 0.55 * uz;
            let nn = (ntx1 * ntx1 + nty1 * nty1 + ntz1 * ntz1).sqrt();
            ntx1 /= nn;
            nty1 /= nn;
            ntz1 /= nn;
            (ntx, nty, ntz) = (ntx1, nty1, ntz1);
            let px = x + STEP * ntx;
            let py = y + STEP * nty;
            let pz = z + STEP * ntz;
            if out(px, py, pz) {
                break;
            }
            [nx_p, ny_p, nz_p] = f.recenter(px, py, pz, ntx, nty, ntz, 0.42, 2);
        }
        let (mut dx, mut dy, mut dz) = (nx_p - x, ny_p - y, nz_p - z);
        let dl = (dx * dx + dy * dy + dz * dz).sqrt();
        #[allow(clippy::manual_range_contains)]
        if dl < 0.25 || dl > 1.45 {
            break;
        }
        dx /= dl;
        dy /= dl;
        dz /= dl;
        if dx * tx + dy * ty + dz * tz < COS55 {
            break;
        }
        if n_pts >= 4 {
            let p = p64(pts[n_pts - 4]);
            let (chx, chy, chz) = (nx_p - p[0], ny_p - p[1], nz_p - p[2]);
            let chl = (chx * chx + chy * chy + chz * chz).sqrt();
            if chl > 1e-05 && (chx * dx + chy * dy + chz * dz) / chl < COS55 {
                break;
            }
        } else {
            let (sx_d, sy_d, sz_d) = (nx_p - x0, ny_p - y0, nz_p - z0);
            let sl = (sx_d * sx_d + sy_d * sy_d + sz_d * sz_d).sqrt();
            if sl > 1e-05 && (sx_d * tx0 + sy_d * ty0 + sz_d * tz0) / sl < COS55 {
                break;
            }
        }
        if n_pts >= 4 {
            let p4 = pts[n_pts - 4];
            let q4 = p64(p4);
            let (v2x, v2y, v2z) = (nx_p - q4[0], ny_p - q4[1], nz_p - q4[2]);
            let l2 = (v2x * v2x + v2y * v2y + v2z * v2z).sqrt();
            // both operands float32 gives a float32 difference
            let [v1x, v1y, v1z] = if n_pts >= 9 {
                let p9 = pts[n_pts - 9];
                [0, 1, 2].map(|k| (p4[k] - p9[k]) as f64)
            } else {
                [q4[0] - x0, q4[1] - y0, q4[2] - z0]
            };
            let l1 = (v1x * v1x + v1y * v1y + v1z * v1z).sqrt();
            if l1 > 1e-05 && l2 > 1e-05 && (v1x * v2x + v1y * v2y + v1z * v2z) / (l1 * l2) < COS55 {
                break;
            }
        }
        if n_pts >= 9 {
            let p9 = pts[n_pts - 9];
            let q9 = p64(p9);
            let (w2x, w2y, w2z) = (nx_p - q9[0], ny_p - q9[1], nz_p - q9[2]);
            let wl2 = (w2x * w2x + w2y * w2y + w2z * w2z).sqrt();
            let [w1x, w1y, w1z] = if n_pts >= 19 {
                let p19 = pts[n_pts - 19];
                [0, 1, 2].map(|k| (p9[k] - p19[k]) as f64)
            } else {
                [q9[0] - x0, q9[1] - y0, q9[2] - z0]
            };
            let wl1 = (w1x * w1x + w1y * w1y + w1z * w1z).sqrt();
            if wl1 > 1e-05 && wl2 > 1e-05 && (w1x * w2x + w1y * w2y + w1z * w2z) / (wl1 * wl2) < COS55 {
                break;
            }
        }
        let p_new = trilin_u8(f.pres, nx_p, ny_p, nz_p);
        let p_mid = trilin_u8(f.pres, 0.5 * (x + nx_p), 0.5 * (y + ny_p), 0.5 * (z + nz_p));
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
        if cov.covered_same_dir(reads, nx_p, ny_p, nz_p, ntx, nty, ntz) {
            cov_run += 1;
            if cov_run >= 2 {
                break;
            }
        } else {
            cov_run = 0;
        }
        for p in &pts[..n_pts.saturating_sub(5)] {
            let (ddx, ddy, ddz) = (nx_p - p[0] as f64, ny_p - p[1] as f64, nz_p - p[2] as f64);
            if ddx * ddx + ddy * ddy + ddz * ddz < 9.0 {
                break 'steps;
            }
        }
        tx = 0.5 * ntx + 0.3 * dx + 0.2 * ux;
        ty = 0.5 * nty + 0.3 * dy + 0.2 * uy;
        tz = 0.5 * ntz + 0.3 * dz + 0.2 * uz;
        let tn = (tx * tx + ty * ty + tz * tz).sqrt();
        tx /= tn;
        ty /= tn;
        tz /= tn;
        (x, y, z) = (nx_p, ny_p, nz_p);
        pts[n_pts] = [x as f32, y as f32, z as f32];
        tans[n_pts] = [tx as f32, ty as f32, tz as f32];
        n_pts += 1;
    }
    while n_pts > 0 && cov_run > 0 {
        n_pts -= 1;
        cov_run -= 1;
    }
    while n_pts > 0 {
        let [px, py, pz] = p64(pts[n_pts - 1]);
        if trilin_u8(f.pres, px, py, pz) >= 76.5 {
            break;
        }
        n_pts -= 1;
    }
    n_pts
}

/// One iteration of `trace_seed_range` (core.py:882-1016) for seed `i`, with `min_raw_len = 7.5`, into `out`
///
/// The seed reads `cov` but does not mark it: its accepted curves and the marks they make are left in `out` for
/// [`CoverageWriter::apply`]. Given the same coverage values at the bricks in `out.reads`, the result is the same
#[allow(clippy::too_many_arguments)]
pub fn trace_seed<P: Grid<u8>, C: Grid<u16>, D: Grid<u16>>(
    f: &Fields<P, C, D>,
    cov: &Coverage,
    s: &mut Scratch,
    seed: [i32; 3],
    primary: bool,
    out: &mut Traced,
) -> Result<()> {
    out.clear();
    let reads = &mut out.reads;
    let [sz, sy, sx] = seed.map(|v| v as i64);
    if !primary && cov.is_covered(reads, sz, sy, sx) {
        return Ok(());
    }
    let [ux, uy, uz] = f.dir_at(sz, sy, sx);
    let (sxf, syf, szf) = (sx as f64, sy as f64, sz as f64);
    if cov.covered_same_dir(reads, sxf, syf, szf, ux, uy, uz) {
        return Ok(());
    }
    let [cx, cy, cz] = f.recenter(sx as f32 as f64, sy as f32 as f64, sz as f32 as f64, ux, uy, uz, 1.5, 4);
    if cov.covered_same_dir(reads, cx, cy, cz, ux, uy, uz) {
        return Ok(());
    }
    if trilin_u8(f.pres, cx, cy, cz) < 76.5 {
        return Ok(());
    }
    let [ux, uy, uz] = f.dir_aligned(cx, cy, cz, ux, uy, uz);
    let nf = trace_half(f, cov, reads, cx, cy, cz, ux, uy, uz, &mut s.pts_f, &mut s.tans_f);
    if nf == HALF {
        bail!("Full-scroll half-path buffer reached; refusing an artificial endpoint");
    }
    let (mut bux, mut buy, mut buz) = (-ux, -uy, -uz);
    if nf >= 3 {
        let k_dep = 3.min(nf - 1);
        let p = s.pts_f[k_dep];
        let (mut dfx, mut dfy, mut dfz) = (p[0] as f64 - cx, p[1] as f64 - cy, p[2] as f64 - cz);
        let dfn = (dfx * dfx + dfy * dfy + dfz * dfz).sqrt();
        if dfn > 1e-05 {
            dfx /= dfn;
            dfy /= dfn;
            dfz /= dfn;
            bux = -(0.55 * dfx + 0.45 * ux);
            buy = -(0.55 * dfy + 0.45 * uy);
            buz = -(0.55 * dfz + 0.45 * uz);
            let bn = (bux * bux + buy * buy + buz * buz).sqrt();
            bux /= bn;
            buy /= bn;
            buz /= bn;
        }
    }
    let nb = trace_half(f, cov, reads, cx, cy, cz, bux, buy, buz, &mut s.pts_b, &mut s.tans_b);
    if nb == HALF {
        bail!("Full-scroll half-path buffer reached; refusing an artificial endpoint");
    }
    let ntot = nb + 1 + nf;
    if ntot < 8 {
        return Ok(());
    }
    let mut idx = 0;
    for j in (0..nb).rev() {
        s.pts_c[idx] = s.pts_b[j];
        s.tans_c[idx] = s.tans_b[j].map(|v| -v);
        idx += 1;
    }
    s.pts_c[idx] = [cx as f32, cy as f32, cz as f32];
    s.tans_c[idx] = [ux as f32, uy as f32, uz as f32];
    idx += 1;
    s.pts_c[idx..idx + nf].copy_from_slice(&s.pts_f[..nf]);
    s.tans_c[idx..idx + nf].copy_from_slice(&s.tans_f[..nf]);
    let req_len = if primary { 7.5 } else { py_max(7.5, 16.0f32 as f64) };
    for (a, b) in valid_curve_spans(&s.pts_c[..ntot], nb as i64)? {
        let n = b - a;
        if n < 8 {
            continue;
        }
        let mut arc_len = 0.0f32;
        for j in a..b - 1 {
            let (p, q) = (s.pts_c[j], s.pts_c[j + 1]);
            let (dx, dy, dz) = (q[0] - p[0], q[1] - p[1], q[2] - p[2]);
            arc_len += (dx * dx + dy * dy + dz * dz).sqrt();
        }
        if (arc_len as f64) < req_len {
            continue;
        }
        // a later span's marks follow an earlier one's, as `mark_curve_coverage` marks them curve by curve
        cov.mark_list(f, &s.pts_c[a..b], &s.tans_c[a..b], out.tans.len(), &mut out.marks);
        out.tans.extend_from_slice(&s.tans_c[a..b]);
        out.curves.push(RawCurve { pts: s.pts_c[a..b].to_vec() });
    }
    Ok(())
}

#[inline(always)]
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline(always)]
fn norm2_f32(v: [f32; 3]) -> f32 {
    v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
}

#[inline(always)]
fn dot_f32(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// `np.dot` of two float32 3-vectors: OpenBLAS `sdot` rounds each product to float32 and sums them in
/// float64
#[inline(always)]
fn np_dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let mut acc = 0.0f64;
    for k in 0..3 {
        acc += (a[k] * b[k]) as f64;
    }
    acc as f32
}

/// `first_curve_split` (core.py:717), all float32
pub fn first_curve_split(pts: &[[f32; 3]]) -> i64 {
    let n = pts.len();
    if n < 2 {
        return 0;
    }
    for i in 0..n {
        for j in i + 7..n {
            if (norm2_f32(sub(pts[j], pts[i])) as f64) < 10.24 {
                return ((i + j) / 2) as i64;
            }
        }
    }
    let bend = |l: usize, c: usize, r: usize| {
        let (v1, v2) = (sub(pts[c], pts[l]), sub(pts[r], pts[c]));
        let (l1, l2) = (norm2_f32(v1).sqrt(), norm2_f32(v2).sqrt());
        (l1 as f64) > 1e-05 && (l2 as f64) > 1e-05 && ((dot_f32(v1, v2) / (l1 * l2)) as f64) < COS55
    };
    for i in 1..n.saturating_sub(1) {
        if bend(i - 1, i, i + 1) {
            return i as i64;
        }
    }
    for s in [5usize, 10] {
        if n > 2 * s {
            for i in s..n - s {
                if bend(i - s, i, i + s) {
                    return i as i64;
                }
            }
        }
    }
    let mut cum_s = vec![0.0f32; n];
    for i in 0..n - 1 {
        cum_s[i + 1] = cum_s[i] + norm2_f32(sub(pts[i + 1], pts[i])).sqrt();
    }
    const WIN: usize = 24;
    if n > WIN {
        for i in 0..n - WIN {
            let arc_w = cum_s[i + WIN] - cum_s[i];
            if arc_w as f64 > 1.0 {
                let ch_w = norm2_f32(sub(pts[i + WIN], pts[i])).sqrt();
                if (ch_w as f64) < 0.78 * arc_w as f64 {
                    return (i + WIN / 2) as i64;
                }
            }
        }
    }
    -1
}

/// `valid_curve_spans` (core.py:791): contiguous valid pieces in path order, as `[start, end)` pairs
pub fn valid_curve_spans(pts: &[[f32; 3]], seed_index: i64) -> Result<Vec<(usize, usize)>> {
    let mut pending = vec![(0usize, pts.len())];
    let mut spans = Vec::new();
    while let Some((a, b)) = pending.pop() {
        let n = b - a;
        if n < 2 {
            continue;
        }
        let piece = &pts[a..b];
        let mut cut = first_curve_split(piece);
        let local_seed = seed_index - a as i64;
        if cut < 0 && 0 < local_seed && local_seed < n as i64 - 1 {
            let ls = local_seed as usize;
            'outer: for j in 0..ls {
                for k in ls + 1..n {
                    if k - j < 6 {
                        continue;
                    }
                    let delta = sub(piece[j], piece[k]);
                    if (np_dot3(delta, delta) as f64) < 9.0 {
                        cut = ((j + k) / 2) as i64;
                        break 'outer;
                    }
                }
            }
        }
        if cut < 0 {
            spans.push((a, b));
            continue;
        }
        if !(0 < cut && cut < n as i64 - 1) {
            bail!("A rejected nontrivial curve must have an interior split");
        }
        let split = a + cut as usize;
        pending.push((split, b));
        pending.push((a, split + 1));
    }
    Ok(spans)
}

/// Fewest points `fit_curve` smooths and resamples; a shorter curve is finished as it was traced
pub const MIN_FIT_POINTS: usize = 4;

/// `fit_and_resample_splines` (core.py:590) for one curve with `target_step = 0.8`, `smooth_iters = 10`,
/// `max_dev = 0.3` and `min_final_len = 0`: the smoothed points and the resampled count (0 when the
/// curve has fewer than 4 points)
pub fn fit_curve<P: Grid<u8>>(pts: &[[f32; 3]], pres: &P) -> (Vec<[f32; 3]>, usize) {
    const MAX_DEV: f64 = 0.3;
    let m = pts.len();
    if m < MIN_FIT_POINTS {
        return (pts.to_vec(), 0);
    }
    let mut buf = pts.to_vec();
    for _ in 0..10 {
        for i in 1..m - 1 {
            let mut p = [0.0f64; 3];
            if i == 1 || i == m - 2 {
                for k in 0..3 {
                    p[k] = 0.25 * buf[i - 1][k] as f64 + 0.5 * buf[i][k] as f64 + 0.25 * buf[i + 1][k] as f64;
                }
            } else {
                for k in 0..3 {
                    let lap = 0.5 * (buf[i - 1][k] + buf[i + 1][k]) as f64;
                    let bih = (-buf[i - 2][k] as f64 + 4.0 * buf[i - 1][k] as f64 + 4.0 * buf[i + 1][k] as f64
                        - buf[i + 2][k] as f64)
                        / 6.0;
                    p[k] = 0.5 * buf[i][k] as f64 + 0.3 * lap + 0.2 * bih;
                }
            }
            let o = pts[i];
            let [ox, oy, oz] = o.map(|v| v as f64);
            let (dx, dy, dz) = (p[0] - ox, p[1] - oy, p[2] - oz);
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            if d > MAX_DEV {
                let s = MAX_DEV / d;
                p = [ox + dx * s, oy + dy * s, oz + dz * s];
            }
            let p_orig = trilin_u8(pres, ox, oy, oz);
            let p_new = trilin_u8(pres, p[0], p[1], p[2]);
            if p_orig >= 76.5 && p_new < py_max(77.0, p_orig - 6.0) {
                p = [0, 1, 2].map(|k| 0.5 * (o[k] + buf[i][k]) as f64);
            }
            buf[i] = p.map(|v| v as f32);
        }
    }
    let mut arc = 0.0f32;
    for i in 1..m {
        arc += norm2_f32(sub(buf[i], buf[i - 1])).sqrt();
    }
    let n_out = ((arc as f64 / 0.8 + 0.5) as i64 + 1).max(2) as usize;
    (buf, n_out)
}

/// `evaluate_catmull_rom_splines` (core.py:655) for one curve; `shape` is the tracing shape z, y, x
pub fn evaluate_curve(smoothed: &[[f32; 3]], n_out: usize, shape: [usize; 3]) -> Vec<[f32; 3]> {
    let m = smoothed.len();
    let mut cum_s = vec![0.0f32; m];
    for i in 0..m - 1 {
        cum_s[i + 1] = cum_s[i] + norm2_f32(sub(smoothed[i + 1], smoothed[i])).sqrt();
    }
    let tot_s = cum_s[m - 1];
    let mut out = vec![[0.0f32; 3]; n_out];
    out[0] = smoothed[0];
    out[n_out - 1] = smoothed[m - 1];
    let upper: [f64; 3] = [0, 1, 2].map(|d| (shape[2 - d] as i64 - 1) as f32 as f64);
    let mut seg_k = 0usize;
    for (j, slot) in out.iter_mut().enumerate().take(n_out - 1).skip(1) {
        let target_s = j as f32 / (n_out - 1) as f32 * tot_s;
        while seg_k + 2 < m && cum_s[seg_k + 1] < target_s {
            seg_k += 1;
        }
        let ds = cum_s[seg_k + 1] - cum_s[seg_k];
        let floor = 1e-06f32;
        let u = (target_s - cum_s[seg_k]) / (if floor > ds { floor } else { ds });
        let u2 = u * u;
        let u3 = u2 * u;
        let (u, u2, u3) = (u as f64, u2 as f64, u3 as f64);
        let i0 = seg_k.saturating_sub(1);
        let i1 = seg_k;
        let i2 = seg_k + 1;
        let i3 = (m - 1).min(seg_k + 2);
        let w0 = -0.5 * u3 + u2 - 0.5 * u;
        let w1 = 1.5 * u3 - 2.5 * u2 + 1.0;
        let w2 = -1.5 * u3 + 2.0 * u2 + 0.5 * u;
        let w3 = 0.5 * u3 - 0.5 * u2;
        for d in 0..3 {
            let val = w0 * smoothed[i0][d] as f64
                + w1 * smoothed[i1][d] as f64
                + w2 * smoothed[i2][d] as f64
                + w3 * smoothed[i3][d] as f64;
            slot[d] = py_min(py_max(val, 0.0), upper[d]) as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tracer reads the GPU's two volumes and the CPU's packed volume as the same values, and a packed field
    /// split for the GPU's centrality gives back the two volumes, including bricks only one of them had written
    #[test]
    fn both_direction_forms_read_the_same() {
        let shape = [20, 33, 17];
        let (mut nx, mut ny) = (Vol::new(shape, 128u8), Vol::new(shape, 128u8));
        let mut packed = Vol::new(shape, pack_dir(128, 128));
        let (mut written_packed, mut written_split) =
            (Directions::new(shape, DirForm::Packed), Directions::new(shape, DirForm::Split));
        for ([z, y, x], bx, by) in
            [([1, 2, 3], 7, 250), ([1, 2, 4], 9, 128), ([18, 30, 16], 1, 128), ([5, 20, 9], 128, 129)]
        {
            nx.set(z, y, x, bx);
            ny.set(z, y, x, by);
            packed.set(z, y, x, pack_dir(bx, by));
            for written in [&mut written_packed, &mut written_split] {
                written.write_block([z, y, x], [1, 1, 1], &[bx], &[by]);
            }
        }

        let split = SplitDirs { nx: &nx, ny: &ny };
        assert_eq!(split.at(1, 2, 3), pack_dir(7, 250));
        assert_eq!(split.corners(1, 2, 3), packed.corners(1, 2, 3));
        assert_eq!(split.corners(1, 2, 3)[1], pack_dir(9, 128));
        assert_eq!(split.packed_block([0; 3], shape), packed.packed_block([0; 3], shape));

        assert!(packed.split().is_none());
        assert_eq!(
            split.split().map(|(x, y)| (x.read_block([0; 3], shape), y.read_block([0; 3], shape))),
            Some((nx.read_block([0; 3], shape), ny.read_block([0; 3], shape)))
        );

        // the field the CPU writes in either form reads the same values
        let Directions::Packed(written) = written_packed else { panic!("packed form") };
        assert_eq!(written.packed_block([0; 3], shape), packed.packed_block([0; 3], shape));
        let Directions::Split { nx: wx, ny: wy } = written_split else { panic!("split form") };
        assert_eq!(SplitDirs { nx: &wx, ny: &wy }.packed_block([0; 3], shape), packed.packed_block([0; 3], shape));
    }

    #[test]
    fn decode_matches_reference_formula() {
        // nx = ny = 128 is the straight-up direction; 255, 255 has r2 > 1 and is renormalized flat
        assert_eq!(decode_dir(128, 128), [0.0, 0.0, 1.0]);
        let [x, y, z] = decode_dir(255, 255);
        assert_eq!(z, 0.0);
        assert!(((x * x + y * y).sqrt() - 1.0).abs() < 1e-12);
        assert_eq!(encode_dir(0.0, 0.0, 1.0), (128, 128));
        assert_eq!(encode_dir(1.0, -1.0, -0.5), (1, 255));
    }

    #[test]
    fn a_seed_is_stale_once_a_later_commit_changes_a_brick_it_read() {
        let cov = Coverage::new([40, 70, 70]);
        let mut reads = ReadSet::default();
        assert!(!cov.is_covered(&mut reads, 20, 20, 20));
        assert!(!cov.covered_same_dir(&mut reads, 60.0, 60.0, 30.0, 0.0, 0.0, 1.0));
        assert_eq!(reads.slots().len(), 2);

        // seed 3 marks a voxel in the first brick read: a seed traced once 4 seeds were committed saw the mark, one
        // traced when only 3 were did not
        let mut writer = cov.writer();
        let mut traced = Traced::default();
        traced.tans.push([0.0, 0.0, 1.0]);
        traced.marks.push(Mark { at: cov.0.place(21, 22, 23), dir: pack_dir(128, 128), tan: 0 });
        assert!(!writer.stale(&reads, 3));
        writer.apply(3, &traced);
        assert!(writer.stale(&reads, 3));
        assert!(!writer.stale(&reads, 4));
        assert!(cov.is_covered(&mut ReadSet::default(), 21, 22, 23));

        // a count of 1 becomes 2 for a crossing direction and stays 1 for the same one
        traced.marks[0].dir = pack_dir(255, 128);
        traced.tans[0] = [0.0, 0.0, 1.0];
        writer.apply(4, &traced);
        assert_eq!(cov.0.get(cov.0.place(21, 22, 23)), pack_dir(128, 128));
        traced.tans[0] = [1.0, 0.0, 0.0];
        writer.apply(5, &traced);
        assert_eq!(cov.0.get(cov.0.place(21, 22, 23)), COVERED_TWICE);
    }

    #[test]
    fn spans_split_a_hairpin_at_its_vertex() {
        // a straight run that doubles back: the split keeps the vertex on both sides
        let mut pts = Vec::new();
        for i in 0..20 {
            pts.push([i as f32 * 0.75, 0.0, 0.0]);
        }
        for i in 1..20 {
            pts.push([(19 - i) as f32 * 0.75, 4.0, 0.0]);
        }
        let spans = valid_curve_spans(&pts, 5).unwrap();
        assert!(spans.len() >= 2);
        assert_eq!(spans[0].0, 0);
        assert_eq!(spans.last().unwrap().1, pts.len());
        for w in spans.windows(2) {
            assert_eq!(w[0].1, w[1].0 + 1);
        }
    }
}
