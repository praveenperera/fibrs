//! The head's token builders: each samples what a token family reads and writes its projection's bf16 input rows
//! (zero-padded to a multiple of 64 columns: the pad columns are zero from allocation and never written)
//!
//! Feature sampling is villa's `sample_features` (models/follower.py): points outside the crop read zero features
//! and zero support; inside, `grid_sample(cells.float(), bilinear, border, align_corners=True)` on the 36x32x32
//! lattice, with PyTorch's fp32 coordinate arithmetic, corner weights and FMA accumulation order. The tube tokens'
//! fixed points instead apply a precomputed sparse operator (host `gather.rs`) that merges their sub-points'
//! corners per cell; the sector tokens apply the same operator inside their projection (`sector.rs`). Profile sampling is vc_reader's `sample_array` (order 1, zeros) on the fp32 input crop. One
//! warp per sampled point or token; lane `l` carries channels `2 l, 2 l + 1`
//!
//! The per-decision request record (`REQ` floats, laid out by the host's `head.rs`) holds the history, seed, path
//! geometry and orientation

use cuda_device::thread::{__unroll_config, blockIdx_x, blockIdx_y, gridDim_y, threadIdx_x};
use cuda_device::{DisjointSlice, kernel, launch_contract};

use super::util::{hi_f32, lo_f32, math, pack_bf16, require};
use crate::crops::zscore;

/// Floats per decision record, and the offsets of its fields
pub const REQ: usize = 800;
/// `hist` (128, 3)
pub const R_HIST: usize = 0;
/// `hmask` (128)
pub const R_HMASK: usize = 384;
/// `seed` (3)
pub const R_SEED: usize = 512;
/// `seed_mask`
pub const R_SEED_MASK: usize = 515;
/// `seed_age`
pub const R_SEED_AGE: usize = 516;
/// `seed_tangent` (3)
pub const R_SEED_TANGENT: usize = 517;
/// `path_geometry` (29, 8)
pub const R_GEOM: usize = 520;
/// `path_geometry_valid` (29), 0 or 1
pub const R_GEOM_VALID: usize = 752;
/// `orientation` (5)
pub const R_ORIENT: usize = 781;

/// Cell lattice depth, height, width; channel pairs per cell
pub(super) const CD: u32 = 36;
pub(super) const CH: u32 = 32;
pub(super) const CW: u32 = 32;
pub(super) const CP: u32 = 32;
/// u32 pairs per decision's cells
pub(super) const CELLS: usize = (CD * CH * CW * CP) as usize;
/// Crop depth, height, width (2 fp32 channels)
const DD: i32 = 144;
const DH: i32 = 128;
const DW: i32 = 128;
const CROP: usize = (DD * DH * DW * 2) as usize;

/// Sector tokens and tube tokens per decision
pub const SECTORS: u32 = 1404;
pub const TUBES: u32 = 18;
/// Reference tokens (128 history + seed) and path-geometry tokens per decision
pub const REFS: u32 = 129;
pub const GEOMS: u32 = 29;
/// Strip tokens per decision
pub const STRIPS: u32 = 450;
/// Padded input widths of the projections: tube 910, reference 72, geometry 68, strip query 67, orientation 5,
/// refinement 132, profile 100 (the sector projection's is `sector::K_SECTOR`)
pub const K_TUBE: u32 = 960;
pub const K_REF: u32 = 128;
pub const K_GEOM: u32 = 128;
pub const K_QUERY: u32 = 128;
pub const K_ORIENT: u32 = 64;
pub const K_REFINE: u32 = 192;
pub const K_PROFILE: u32 = 128;
/// Key-mask row length and the first reference and geometry keys
pub const LDK: u32 = 2048;
const KEY_REF: u32 = 1422;
const KEY_GEOM: u32 = 1551;

#[inline(always)]
fn bf16_bits(v: f32) -> u16 {
    pack_bf16(v, 0.0) as u16
}

/// `a[i]` (`a[0]` past the end) through selects, so a runtime index doesn't move the array to local memory
#[inline(always)]
fn pick<const N: usize>(a: [f32; N], i: u32) -> f32 {
    let mut r = a[0];
    let mut k = 1;
    while k < N {
        __unroll_config::<0>();
        if i == k as u32 {
            r = a[k];
        }
        k += 1;
    }
    r
}

/// `grid_sample`'s source index on one axis from a crop-local coordinate: `feature_grid` (`2 (p - origin) / (scale
/// (size - 1)) - 1`, op by op) then the align-corners unnormalize and the border clip
#[inline(always)]
fn source_index(p: f32, half: f32, size1: f32) -> f32 {
    let g = (2.0 * (p + half)) / (2.0 * size1) - 1.0;
    let i = ((g + 1.0) * 0.5) * size1;
    i.max(0.0).min(size1)
}

/// Whether `p` lies in the crop (`crop_support`): finite and inside `[-31.75, 31.75]^2 x [-36, 35.5]`
#[inline(always)]
fn supported(x: f32, y: f32, z: f32) -> bool {
    (x >= -31.75) & (x <= 31.75) & (y >= -31.75) & (y <= 31.75) & (z >= -36.0) & (z <= 35.5)
}

/// One lane's channel pair of `sample_features` at crop-local `(x, y, z)` on a decision's cells (`[36, 32, 32, 32]`
/// bf16 pairs), and the point's support. NaN compares false, so a non-finite point is unsupported
///
/// # Safety
/// `cells` points to one decision's whole lattice; `lane < 32`
#[inline(always)]
unsafe fn sample_cells(cells: *const u32, x: f32, y: f32, z: f32, lane: u32) -> (f32, f32, bool) {
    if !supported(x, y, z) {
        return (0.0, 0.0, false);
    }
    let ix = source_index(x, 31.75, (CW - 1) as f32);
    let iy = source_index(y, 31.75, (CH - 1) as f32);
    let iz = source_index(z, 36.0, (CD - 1) as f32);
    let (x0, y0, z0) = (math::floorf(ix), math::floorf(iy), math::floorf(iz));
    let (fx, fy, fz) = (ix - x0, iy - y0, iz - z0);
    let (gx, gy, gz) = ((x0 + 1.0) - ix, (y0 + 1.0) - iy, (z0 + 1.0) - iz);
    let (x0, y0, z0) = (x0 as u32, y0 as u32, z0 as u32);
    // the far corner exists unless the point sits on the last cell of an axis, where its weight is zero
    let (ex, ey, ez) = (x0 + 1 < CW, y0 + 1 < CH, z0 + 1 < CD);
    let weights = [
        gx * gy * gz,
        fx * gy * gz,
        gx * fy * gz,
        fx * fy * gz,
        gx * gy * fz,
        fx * gy * fz,
        gx * fy * fz,
        fx * fy * fz,
    ];
    let (mut v0, mut v1) = (0.0f32, 0.0f32);
    let mut corner = 0u32;
    while corner < 8 {
        __unroll_config::<0>();
        let (bx, by, bz) = (corner & 1, (corner >> 1) & 1, corner >> 2);
        let inside = ((bx == 0) | ex) & ((by == 0) | ey) & ((bz == 0) | ez);
        if inside {
            let at = (((z0 + bz) * CH + (y0 + by)) * CW + (x0 + bx)) * CP + lane;
            // SAFETY: every coordinate is inside the lattice and lane < 32
            let p = unsafe { *cells.add(at as usize) };
            let w = weights[corner as usize];
            v0 = lo_f32(p) * w + v0;
            v1 = hi_f32(p) * w + v1;
        }
        corner += 1;
    }
    (v0, v1, true)
}

/// vc_reader `sample_array` (order 1, zeros boundary) of both channels of a decision's `[144, 128, 128, 2]` fp32
/// crop at array index `(d, h, w)`: zero when any coordinate is `<= -1` or `>= size`, NaN when one isn't finite;
/// else the multilinear sum over the axes with a nonzero fraction, corners z-fastest, weights multiplied in z, y, x
/// order, corners outside the crop zero. With `st`, the crop's CT channel is a raw [`crate::crops::sc_raw`] crop
/// and each corner's value is z-scored with those statistics as it is read, to the z-scored crop's bits
///
/// # Safety
/// `crop` points to one decision's whole crop
#[inline(always)]
unsafe fn sample_crop(crop: *const f32, st: Option<[f32; 4]>, d: f32, h: f32, w: f32) -> (f32, f32) {
    let c = [d, h, w];
    let shape = [DD, DH, DW];
    let mut finite = true;
    let mut outside = false;
    let mut a = 0;
    while a < 3 {
        __unroll_config::<0>();
        finite &= c[a].is_finite();
        outside |= !((c[a] > -1.0) & (c[a] < shape[a] as f32));
        a += 1;
    }
    if !finite {
        return (f32::NAN, f32::NAN);
    }
    if outside {
        return (0.0, 0.0);
    }
    let lo = [math::floorf(d), math::floorf(h), math::floorf(w)];
    let f = [d - lo[0], h - lo[1], w - lo[2]];
    let lo = [lo[0] as i32, lo[1] as i32, lo[2] as i32];
    let (mut v0, mut v1) = (0.0f32, 0.0f32);
    let mut corner = 0u32;
    while corner < 8 {
        __unroll_config::<0>();
        let mut weight = 1.0f32;
        let mut skip = false;
        let mut inside = true;
        let mut at = 0i32;
        let mut a = 0;
        while a < 3 {
            __unroll_config::<0>();
            let bit = (corner >> a) & 1;
            if f[a] != 0.0 {
                weight *= if bit != 0 { f[a] } else { 1.0 - f[a] };
            } else {
                skip |= bit != 0;
            }
            let v = lo[a] + bit as i32;
            inside &= (v >= 0) & (v < shape[a]);
            at = at * shape[a] + v;
            a += 1;
        }
        if !skip && inside {
            // SAFETY: every coordinate is inside the crop
            let p = unsafe { crop.add(at as usize * 2) };
            let (c0, c1) = unsafe { (*p, *p.add(1)) };
            let c0 = match st {
                Some(st) => zscore(c0, st),
                None => c0,
            };
            v0 = weight * c0 + v0;
            v1 = weight * c1 + v1;
        }
        corner += 1;
    }
    (v0, v1)
}

/// Decisions per warp in `sm_tube_tokens`: each operator entry is read once for all of them, and their cell loads
/// are independent
const CELL_DECISIONS: u32 = 4;

/// The tube tokens' projection inputs from the fixed sampling operator (host `gather.rs`; the sector tokens' go
/// straight into their projection, `sector::sm_sector_tokens`): per tube point (`points[4 i..4 i + 4]` = first
/// entry, entry count, `token * 16 + point within the token`, support as f32 bits) the weighted sum of its
/// entries' cells (`entries[2 e..2 e + 2]` = lattice cell, weight as f32 bits) and its support, written at columns
/// `65 p..65 p + 65` of the token's row of `xt` (`[B * 18, 960]`), bf16
///
/// Grid `(ceil(npoints / 4), ceil(batch / 4))`, block 128: one warp per point and 4 decisions
#[kernel]
#[launch_contract(domain = 2, block = (128, 1, 1))]
pub fn sm_tube_tokens(
    mut xt: DisjointSlice<u16>,
    cells: &[u32],
    points: &[u32],
    entries: &[u32],
    npoints: u32,
    batch: u32,
) {
    let nb = batch as usize;
    require(
        (cells.len() >= nb * CELLS)
            & (xt.len() >= nb * (TUBES * K_TUBE) as usize)
            & (points.len() >= npoints as usize * 4),
    );
    let i = blockIdx_x() * 4 + threadIdx_x() / 32;
    let b0 = blockIdx_y() * CELL_DECISIONS;
    if (i >= npoints) | (b0 >= batch) {
        return;
    }
    let lane = threadIdx_x() % 32;
    let nent = (entries.len() / 2) as u32;
    // SAFETY: i < npoints, inside the checked table
    let (first, count, slot, support) = unsafe {
        let d = points.as_ptr().add(i as usize * 4);
        (*d, *d.add(1), *d.add(2), f32::from_bits(*d.add(3)))
    };
    // a table row past the entries reads none of them
    let count = count.min(nent.saturating_sub(first));
    // decisions past the batch reread the last one and are never stored
    let mut base = [0usize; CELL_DECISIONS as usize];
    let mut d = 0;
    while d < CELL_DECISIONS as usize {
        __unroll_config::<0>();
        base[d] = (b0 as usize + d).min(nb - 1) * CELLS + lane as usize;
        d += 1;
    }
    let mut v = [[0.0f32; 2]; CELL_DECISIONS as usize];
    let mut e = 0;
    while e < count {
        // SAFETY: first + e < nent
        let (cell, w) = unsafe {
            let q = entries.as_ptr().add((first + e) as usize * 2);
            (*q, f32::from_bits(*q.add(1)))
        };
        // a cell index past the lattice reads its last cell
        let at = (cell.min(CD * CH * CW - 1) * CP) as usize;
        let mut d = 0;
        while d < CELL_DECISIONS as usize {
            __unroll_config::<0>();
            // SAFETY: base[d] + at lies inside decision b0 + d's (or the batch's last) lattice
            let p = unsafe { *cells.as_ptr().add(base[d] + at) };
            v[d][0] = lo_f32(p) * w + v[d][0];
            v[d][1] = hi_f32(p) * w + v[d][1];
            d += 1;
        }
        e += 1;
    }
    let (token, p) = (slot / 16, slot % 16);
    let (out, col) = (xt.as_mut_ptr(), (p * 65).min(K_TUBE - 65) as usize);
    let mut d = 0;
    while d < CELL_DECISIONS as usize {
        __unroll_config::<0>();
        let b = b0 as usize + d;
        if b < nb {
            // a token below the tube's writes the first tube token's row
            let row = (b * TUBES as usize + token.saturating_sub(SECTORS).min(TUBES - 1) as usize) * K_TUBE as usize;
            // SAFETY: the row lies inside the checked output and col + 65 <= K_TUBE; each column is this lane's alone
            unsafe {
                let o = out.add(row + col);
                *o.add(2 * lane as usize) = bf16_bits(v[d][0]);
                *o.add(2 * lane as usize + 1) = bf16_bits(v[d][1]);
                if lane == 0 {
                    *o.add(64) = bf16_bits(support);
                }
            }
        }
        d += 1;
    }
}

/// The reference tokens' inputs (models/follower.py `reference_points`, `reference_metadata`): per decision 128
/// history points and the seed, masked by their validity and the crop support (masked points zeroed); their
/// sampled features (64) and metadata (position / 16, the seed's tangent, `log1p(clamp(age, 0, 2048)) /
/// log(2049)`, the seed flag; zero when masked) as `xr` rows `[B * 129, 128]` bf16. Writes each reference's key
/// mask byte (`[B, 2048]`, keys 1422..1551)
///
/// Grid `(ceil(129 / 4), B)`, block 128: one warp per reference
#[kernel]
#[launch_contract(domain = 2, block = (128, 1, 1))]
pub fn sm_ref_tokens(
    mut xr: DisjointSlice<u16>,
    mut keymask: DisjointSlice<u8>,
    cells: &[u32],
    req: &[f32],
    inv_log2049: f32,
) {
    let b = blockIdx_y() as usize;
    let nb = gridDim_y() as usize;
    require(
        (cells.len() >= nb * CELLS)
            & (req.len() >= nb * REQ)
            & (xr.len() >= nb * (REFS * K_REF) as usize)
            & (keymask.len() >= nb * LDK as usize),
    );
    let r = blockIdx_x() * 4 + threadIdx_x() / 32;
    if r >= REFS {
        return;
    }
    let lane = threadIdx_x() % 32;
    let rq = req.as_ptr().wrapping_add(b * REQ);
    let seed = r == REFS - 1;
    // SAFETY: every offset lies inside decision b's record
    let (p, valid, age, tangent) = unsafe {
        if seed {
            let s = rq.add(R_SEED);
            let t = rq.add(R_SEED_TANGENT);
            ([*s, *s.add(1), *s.add(2)], *rq.add(R_SEED_MASK) != 0.0, *rq.add(R_SEED_AGE), [*t, *t.add(1), *t.add(2)])
        } else {
            let h = rq.add(R_HIST + 3 * r as usize);
            ([*h, *h.add(1), *h.add(2)], *rq.add(R_HMASK + r as usize) != 0.0, (r + 1) as f32, [0.0; 3])
        }
    };
    let mask = valid & supported(p[0], p[1], p[2]);
    let p = if mask { p } else { [0.0; 3] };
    // SAFETY: decision b's lattice
    let (v0, v1, _) = unsafe { sample_cells(cells.as_ptr().wrapping_add(b * CELLS), p[0], p[1], p[2], lane) };
    let row = (b * REFS as usize + r as usize) * K_REF as usize;
    // SAFETY: the row lies inside the checked output; columns are this lane's alone
    unsafe {
        let o = xr.as_mut_ptr().add(row);
        *o.add(2 * lane as usize) = bf16_bits(v0);
        *o.add(2 * lane as usize + 1) = bf16_bits(v1);
        if lane < 8 {
            let meta = match lane {
                0..=2 => pick(p, lane) * 0.0625,
                3..=5 => pick(tangent, lane - 3),
                6 => math::log1pf(age.max(0.0).min(2048.0)) * inv_log2049,
                _ => {
                    if seed {
                        1.0
                    } else {
                        0.0
                    }
                }
            };
            *o.add(64 + lane as usize) = bf16_bits(if mask { meta } else { 0.0 });
        }
        if lane == 0 {
            *keymask.as_mut_ptr().add(b * LDK as usize + (KEY_REF + r) as usize) = mask as u8;
        }
    }
}

/// The path-geometry tokens' inputs (models/path_geometry.py `PathGeometryTokens`): per row, the geometry zeroed
/// when invalid, then 3 x 10 Fourier pairs of the position (`sin`, `cos` of `pos * 2 pi / 2^(k + 1)`, as `sinpi` of
/// `pos * 2^-k`), the position / 64, the tangent, `log1p(back) / log1p(1024)` and the seed flag, as `xg` rows `[B *
/// 29, 128]` bf16. Writes each row's key mask byte (keys 1551..1580)
///
/// Grid `(ceil(29 / 4), B)`, block 128: one warp per row, lane `l` taking features `l, l + 32, l + 64`
#[kernel]
#[launch_contract(domain = 2, block = (128, 1, 1))]
pub fn sm_geom_tokens(mut xg: DisjointSlice<u16>, mut keymask: DisjointSlice<u8>, req: &[f32], inv_log1p1024: f32) {
    let b = blockIdx_y() as usize;
    let nb = gridDim_y() as usize;
    require(
        (req.len() >= nb * REQ) & (xg.len() >= nb * (GEOMS * K_GEOM) as usize) & (keymask.len() >= nb * LDK as usize),
    );
    let r = blockIdx_x() * 4 + threadIdx_x() / 32;
    if r >= GEOMS {
        return;
    }
    let lane = threadIdx_x() % 32;
    let rq = req.as_ptr().wrapping_add(b * REQ);
    // SAFETY: inside decision b's record
    let (g, valid) = unsafe {
        let p = rq.add(R_GEOM + 8 * r as usize);
        let g = [*p, *p.add(1), *p.add(2), *p.add(3), *p.add(4), *p.add(5), *p.add(6), *p.add(7)];
        (g, *rq.add(R_GEOM_VALID + r as usize) != 0.0)
    };
    let g = if valid { g } else { [0.0; 8] };
    let row = (b * GEOMS as usize + r as usize) * K_GEOM as usize;
    let mut i = lane;
    while i < 68 {
        let v = if i < 60 {
            let (c, j) = (i / 20, i % 20);
            // pos * 2^-k: exact, so sinpi sees the phase over pi without the rounding of 2 pi / 2^(k + 1)
            let x = pick(g, c) * f32::from_bits((127 - (j % 10)) << 23);
            if j < 10 { math::sinpif(x) } else { math::cospif(x) }
        } else if i < 63 {
            pick(g, i - 60) * 0.015625
        } else if i < 66 {
            pick(g, i - 60)
        } else if i == 66 {
            math::log1pf(g[6]) * inv_log1p1024
        } else {
            g[7]
        };
        // SAFETY: the row lies inside the checked output and i < 68; the column is this lane's alone
        unsafe { *xg.as_mut_ptr().add(row + i as usize) = bf16_bits(v) };
        i += 32;
    }
    if lane == 0 {
        // SAFETY: inside decision b's mask row
        unsafe { *keymask.as_mut_ptr().add(b * LDK as usize + (KEY_GEOM + r) as usize) = valid as u8 };
    }
}

/// The strip queries' inputs (sheet_strip/model.py `strip_outputs`): per strip token, the evidence at its anchor
/// (`anchors[3 t..3 t + 3]`, crop-local: features, support) and the anchor's `(v, f) / 16`, as `xq` rows `[B * 450,
/// 128]` bf16; and each decision's orientation rounded to bf16 as `xo` rows `[B, 64]`
///
/// Grid `(ceil(450 / 4), B)`, block 128: one warp per token
#[kernel]
#[launch_contract(domain = 2, block = (128, 1, 1))]
pub fn sm_query_tokens(
    mut xq: DisjointSlice<u16>,
    mut xo: DisjointSlice<u16>,
    cells: &[u32],
    req: &[f32],
    anchors: &[f32],
) {
    let b = blockIdx_y() as usize;
    let nb = gridDim_y() as usize;
    require(
        (cells.len() >= nb * CELLS)
            & (req.len() >= nb * REQ)
            & (anchors.len() >= STRIPS as usize * 3)
            & (xq.len() >= nb * (STRIPS * K_QUERY) as usize)
            & (xo.len() >= nb * K_ORIENT as usize),
    );
    let t = blockIdx_x() * 4 + threadIdx_x() / 32;
    if t >= STRIPS {
        return;
    }
    let lane = threadIdx_x() % 32;
    // SAFETY: t < 450, inside the checked table
    let (x, y, z) = unsafe {
        let a = anchors.as_ptr().add(t as usize * 3);
        (*a, *a.add(1), *a.add(2))
    };
    // SAFETY: decision b's lattice
    let (v0, v1, sup) = unsafe { sample_cells(cells.as_ptr().wrapping_add(b * CELLS), x, y, z, lane) };
    let row = (b * STRIPS as usize + t as usize) * K_QUERY as usize;
    // SAFETY: the row lies inside the checked output; columns are this lane's alone
    unsafe {
        let o = xq.as_mut_ptr().add(row);
        *o.add(2 * lane as usize) = bf16_bits(v0);
        *o.add(2 * lane as usize + 1) = bf16_bits(v1);
        match lane {
            0 => *o.add(64) = bf16_bits(if sup { 1.0 } else { 0.0 }),
            1 => *o.add(65) = bf16_bits(y * 0.0625),
            2 => *o.add(66) = bf16_bits(z * 0.0625),
            _ => {}
        }
        if t == 0 && lane < 5 {
            let v = *req.as_ptr().add(b * REQ + R_ORIENT + lane as usize);
            *xo.as_mut_ptr().add(b * K_ORIENT as usize + lane as usize) = bf16_bits(v);
        }
    }
}

/// The refinement pass's inputs (sheet_strip/model.py `strip_outputs`, `strip_profiles`): per strip token `t =
/// slab * 25 + col` and row `n` of its slab, the point `(h0, v[col], f[2 slab + n])` with `h0` the first pass's
/// height (`h0` `[B * 450, 2]` fp32). `xf` rows `[B * 450, 192]`: the evidence of both points (features, support),
/// then both `h0 / 16`. `xp` rows `[B * 450, 128]`: at `(n * 25 + k) * 2 + c`, channel `c` of the crop (`[B, 144,
/// 128, 128, 2]` fp32) at `u = h0 + offsets[k]`, all bf16. `grid` holds the 25 column `v`, the 36 row `f` and the 25
/// profile offsets. `stats` is empty when the crop's CT channel is z-scored, else each crop's
/// [`crate::crops::zscore`] statistics `[B][4]` for a raw [`crate::crops::sc_raw`] crop
///
/// Grid `(ceil(450 / 4), B)`, block 128: one warp per token
#[kernel]
#[launch_contract(domain = 2, block = (128, 1, 1))]
pub fn sm_refine_tokens(
    mut xf: DisjointSlice<u16>,
    mut xp: DisjointSlice<u16>,
    cells: &[u32],
    crop: &[f32],
    h0: &[f32],
    grid: &[f32],
    stats: &[f32],
) {
    let b = blockIdx_y() as usize;
    let nb = gridDim_y() as usize;
    require(
        (cells.len() >= nb * CELLS)
            & (crop.len() >= nb * CROP)
            & (stats.is_empty() | (stats.len() >= nb * 4))
            & (h0.len() >= nb * STRIPS as usize * 2)
            & (grid.len() >= 86)
            & (xf.len() >= nb * (STRIPS * K_REFINE) as usize)
            & (xp.len() >= nb * (STRIPS * K_PROFILE) as usize),
    );
    let t = blockIdx_x() * 4 + threadIdx_x() / 32;
    if t >= STRIPS {
        return;
    }
    let lane = threadIdx_x() % 32;
    let (slab, col) = ((t / 25) as usize, (t % 25) as usize);
    let tok = b * STRIPS as usize + t as usize;
    // SAFETY: inside the checked tables
    let (u, v, f) = unsafe {
        let h = h0.as_ptr().add(tok * 2);
        let g = grid.as_ptr();
        ([*h, *h.add(1)], *g.add(col), [*g.add(25 + 2 * slab), *g.add(26 + 2 * slab)])
    };
    let cp = cells.as_ptr().wrapping_add(b * CELLS);
    // SAFETY: the rows lie inside the checked outputs; columns are this lane's alone
    unsafe {
        let o = xf.as_mut_ptr().add(tok * K_REFINE as usize);
        let mut n = 0;
        while n < 2 {
            __unroll_config::<0>();
            let (a0, a1, sup) = sample_cells(cp, u[n], v, f[n], lane);
            *o.add(65 * n + 2 * lane as usize) = bf16_bits(a0);
            *o.add(65 * n + 2 * lane as usize + 1) = bf16_bits(a1);
            if lane == 0 {
                *o.add(65 * n + 64) = bf16_bits(if sup { 1.0 } else { 0.0 });
            }
            n += 1;
        }
        if lane < 2 {
            *o.add(130 + lane as usize) = bf16_bits(pick(u, lane) * 0.0625);
        }

        let cr = crop.as_ptr().add(b * CROP);
        let st = if stats.is_empty() {
            None
        } else {
            let p = stats.as_ptr().add(4 * b);
            Some([*p, *p.add(1), *p.add(2), *p.add(3)])
        };
        let o = xp.as_mut_ptr().add(tok * K_PROFILE as usize);
        let mut i = lane;
        while i < 50 {
            let (n, k) = (i / 25, (i % 25) as usize);
            let w = (pick(u, n) + *grid.as_ptr().add(61 + k)) * 2.0 + 63.5;
            let (c0, c1) = sample_crop(cr, st, pick(f, n) * 2.0 + 72.0, v * 2.0 + 63.5, w);
            *o.add(2 * i as usize) = bf16_bits(c0);
            *o.add(2 * i as usize + 1) = bf16_bits(c1);
            i += 32;
        }
    }
}
