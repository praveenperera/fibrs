//! A fibre's decisions stitched into one strip (villa `batched.stitched_strip`, `infer.py`'s `orient_decisions`,
//! `stitch`, `trim_ends`, `stitch_values`, and `infer_all.display_grid`)
//!
//! Each decision's strip points `pos + F (u, v, f)` become rows at fibre arclength `t + f`; overlapping decisions
//! are averaged per (row, column) with weight cos^2(pi f / 32) for |f| < 16, accumulated in NumPy's `np.add.at`
//! order (decision, row, column), so the averages are bitwise villa's

use super::np::{M3, V3, col, dot, sum};
use super::types::{STRIP_COLS, STRIP_ROWS};

/// The strip grid (`strip_grid` through float32): rows f_i = 2i - 35.25 and columns v_j = 2.5j - 30
pub fn strip_f() -> [f64; STRIP_ROWS] {
    std::array::from_fn(|i| 2.0 * i as f64 - 35.25)
}

pub fn strip_v() -> [f64; STRIP_COLS] {
    std::array::from_fn(|j| 2.5 * j as f64 - 30.0)
}

/// The weight of strip row `f` (`np.where(|f| < blend, cos(pi f / (2 blend))^2, 0)`)
fn window(f: f64, blend: f64) -> f64 {
    if f.abs() < blend {
        let c = (std::f64::consts::PI * f / (2.0 * blend)).cos();
        c * c
    } else {
        0.0
    }
}

/// Which decisions turn their v axis to agree with the previous decision's (`orient_decisions`)
pub fn flipped(frames: &[M3]) -> Vec<bool> {
    let mut out = vec![false; frames.len()];
    for k in 1..frames.len() {
        out[k] = out[k - 1] ^ (dot(col(&frames[k], 1), col(&frames[k - 1], 1)) < 0.0);
    }
    out
}

/// A decision's frame after orientation: u and v negated when flipped
fn oriented_frame(frame: &M3, flip: bool) -> M3 {
    let mut f = *frame;
    if flip {
        for row in &mut f {
            row[0] = -row[0];
            row[1] = -row[1];
        }
    }
    f
}

/// The strip row of each decision's grid row: `rint((arcs[k] - arcs[0] + f_i - f[0]) / df)`
fn row_of(arcs: &[f64], k: usize, f: &[f64; STRIP_ROWS], i: usize) -> usize {
    let df = f[1] - f[0];
    ((((arcs[k] - arcs[0]) + f[i]) - f[0]) / df).round_ties_even() as usize
}

/// The stitched rows of oriented decisions: their weighted mean point (trace voxels) and summed weight per node
pub struct Stitched {
    pub rows: Vec<[V3; STRIP_COLS]>,
    pub weight: Vec<[f64; STRIP_COLS]>,
}

/// `stitch(heights, frames, positions, arcs, f, v, blend)` with heights already oriented; `zero` stitches the frame
/// plane (heights 0) instead
fn stitch(
    heights: &[[[f64; STRIP_COLS]; STRIP_ROWS]],
    frames: &[M3],
    positions: &[V3],
    arcs: &[f64],
    blend: f64,
    zero: bool,
) -> Stitched {
    let (f, v) = (strip_f(), strip_v());
    let k_count = arcs.len();
    let rows = row_of(arcs, k_count - 1, &f, STRIP_ROWS - 1) + 1;
    let mut total = vec![[0.0f64; STRIP_COLS]; rows];
    let mut first = vec![[[0.0f64; 3]; STRIP_COLS]; rows];
    for k in 0..k_count {
        let fr = &frames[k];
        for i in 0..STRIP_ROWS {
            let r = row_of(arcs, k, &f, i);
            let w = window(f[i], blend) * 1.0;
            for j in 0..STRIP_COLS {
                let u = if zero { 0.0 } else { heights[k][i][j] };
                let local = [u, v[j], f[i]];
                // einsum('kfvj,kij->kfvi'): (j0 + j2) + j1
                let world: V3 = std::array::from_fn(|a| {
                    positions[k][a] + ((local[0] * fr[a][0] + local[2] * fr[a][2]) + local[1] * fr[a][1])
                });
                total[r][j] += w;
                for a in 0..3 {
                    first[r][j][a] += w * world[a];
                }
            }
        }
    }
    let rows =
        first.iter().zip(&total).map(|(fr, t)| std::array::from_fn(|j| fr[j].map(|x| x / t[j].max(1e-12)))).collect();
    Stitched { rows, weight: total }
}

/// `stitch_values(values, arcs, f, blend)`: the stitch weights' average of a per-point value, NaN without weight
fn stitch_values(values: &[[[f32; STRIP_COLS]; STRIP_ROWS]], arcs: &[f64], blend: f64) -> Vec<[f64; STRIP_COLS]> {
    let f = strip_f();
    let rows = row_of(arcs, arcs.len() - 1, &f, STRIP_ROWS - 1) + 1;
    let mut total = vec![[0.0f64; STRIP_COLS]; rows];
    let mut first = vec![[0.0f64; STRIP_COLS]; rows];
    for (k, value) in values.iter().enumerate() {
        for i in 0..STRIP_ROWS {
            let r = row_of(arcs, k, &f, i);
            let w = window(f[i], blend) * 1.0;
            for j in 0..STRIP_COLS {
                total[r][j] += w;
                first[r][j] += w * f64::from(value[i][j]);
            }
        }
    }
    first
        .iter()
        .zip(&total)
        .map(|(fr, t)| std::array::from_fn(|j| if t[j] > 0.0 { fr[j] / t[j].max(1e-12) } else { f64::NAN }))
        .collect()
}

/// A fibre's stitched strip (`stitched_strip`) and what its npz records
pub struct Strip {
    /// Rows (R, Nv) of points, trace voxels, rows along the fibre at arclength `arcs[0] + f[0] + r * df`
    pub rows: Vec<[V3; STRIP_COLS]>,
    /// Nodes kept: weighted and at least `trim` inside both fibre ends
    pub keep: Vec<[bool; STRIP_COLS]>,
    /// Expected |error| per node, patch voxels (`exp(log scale) / tpp`), NaN without weight
    pub expected: Vec<[f64; STRIP_COLS]>,
    /// Which decisions flipped their v axis
    pub flipped: Vec<bool>,
}

/// The decisions' results stitched: `heights` and `log_scale` from the strip model, `frames` and `positions` the
/// decisions' final frames and heads, `trim` in trace voxels, `tpp` trace voxels per patch voxel
#[allow(clippy::too_many_arguments)]
pub fn stitched_strip(
    heights: &[[[f32; STRIP_COLS]; STRIP_ROWS]],
    log_scale: &[[[f32; STRIP_COLS]; STRIP_ROWS]],
    frames: &[M3],
    positions: &[V3],
    arcs: &[f64],
    length: f64,
    blend: f64,
    trim: f64,
    tpp: f64,
) -> Strip {
    let flipped = flipped(frames);
    let oriented: Vec<[[f64; STRIP_COLS]; STRIP_ROWS]> = heights
        .iter()
        .zip(&flipped)
        .map(|(h, &flip)| {
            std::array::from_fn(|i| {
                std::array::from_fn(|j| if flip { -f64::from(h[i][STRIP_COLS - 1 - j]) } else { f64::from(h[i][j]) })
            })
        })
        .collect();
    let frames: Vec<M3> = frames.iter().zip(&flipped).map(|(fr, &flip)| oriented_frame(fr, flip)).collect();
    let Stitched { rows, weight } = stitch(&oriented, &frames, positions, arcs, blend, false);
    let keep = trim_ends(&weight, arcs, length, trim);
    let error: Vec<[[f32; STRIP_COLS]; STRIP_ROWS]> = log_scale
        .iter()
        .zip(&flipped)
        .map(|(s, &flip)| {
            std::array::from_fn(|i| {
                std::array::from_fn(|j| {
                    let x = if flip { s[i][STRIP_COLS - 1 - j] } else { s[i][j] };
                    // float32 throughout: np.exp on float32, then / tpp as a weak Python float
                    super::np::exp_f32(x) / tpp as f32
                })
            })
        })
        .collect();
    let expected = stitch_values(&error, arcs, blend);
    Strip { rows, keep, expected, flipped }
}

/// `trim_ends(weight > 0, arcs[0] + f[0], df, length, trim)`
fn trim_ends(weight: &[[f64; STRIP_COLS]], arcs: &[f64], length: f64, margin: f64) -> Vec<[bool; STRIP_COLS]> {
    let f = strip_f();
    let first = arcs[0] + f[0];
    let step = f[1] - f[0];
    weight
        .iter()
        .enumerate()
        .map(|(r, w)| {
            let arc = first + r as f64 * step;
            let inside = arc >= margin && arc <= length - margin;
            w.map(|x| x > 0.0 && inside)
        })
        .collect()
}

/// The frame plane (heights 0) stitched as the strip, for the npz's `plane`
pub fn plane(frames: &[M3], positions: &[V3], arcs: &[f64], blend: f64) -> Vec<[V3; STRIP_COLS]> {
    let flipped = flipped(frames);
    let frames: Vec<M3> = frames.iter().zip(&flipped).map(|(fr, &flip)| oriented_frame(fr, flip)).collect();
    stitch(&[], &frames, positions, arcs, blend, true).rows
}

/// The strip as displayed (`display_grid`): a V fibre along the rows, an H fibre along the columns (transposed),
/// each turned so higher z is row 0
pub struct Display {
    /// Grid rows x columns of points (patch voxels once scaled)
    pub rows: usize,
    pub cols: usize,
    /// Row-major points, keep and expected error
    pub points: Vec<V3>,
    pub keep: Vec<bool>,
    pub expected: Vec<f64>,
    /// `fiber_along`: rows or columns
    pub along_columns: bool,
    pub fiber_reversed: bool,
    pub across_reversed: bool,
}

/// `display_grid(rows, keep, expected, family)`; `family` other than "H" lays the fibre along the rows
pub fn display_grid(strip: &Strip, family: &str) -> Display {
    let r_count = strip.rows.len();
    let nodes: Vec<(usize, usize)> =
        (0..r_count).flat_map(|r| (0..STRIP_COLS).map(move |c| (r, c))).filter(|&(r, c)| strip.keep[r][c]).collect();
    let z: Vec<f64> = nodes.iter().map(|&(r, c)| strip.rows[r][c][2]).collect();
    let rising = |index: &dyn Fn(&(usize, usize)) -> usize| -> bool {
        let n = nodes.len() as f64;
        // integer indices sum exactly; z by NumPy's pairwise sum
        let mean_i = nodes.iter().map(|p| index(p) as f64).sum::<f64>() / n;
        let mean_z = sum(&z) / n;
        let products: Vec<f64> =
            nodes.iter().zip(&z).map(|(p, &zz)| (index(p) as f64 - mean_i) * (zz - mean_z)).collect();
        sum(&products) > 0.0
    };
    let along_columns = family == "H";
    let flip_rows = !along_columns && nodes.len() > 1 && rising(&|p| p.0);
    let flip_cols = along_columns && nodes.len() > 1 && rising(&|p| p.1);
    let src = |r: usize, c: usize| {
        let r = if flip_rows { r_count - 1 - r } else { r };
        let c = if flip_cols { STRIP_COLS - 1 - c } else { c };
        (r, c)
    };
    let (rows, cols) = if along_columns { (STRIP_COLS, r_count) } else { (r_count, STRIP_COLS) };
    let mut d = Display {
        rows,
        cols,
        points: Vec::with_capacity(rows * cols),
        keep: Vec::with_capacity(rows * cols),
        expected: Vec::with_capacity(rows * cols),
        along_columns,
        fiber_reversed: flip_rows,
        across_reversed: flip_cols,
    };
    for a in 0..rows {
        for b in 0..cols {
            let (r, c) = if along_columns { src(b, a) } else { src(a, b) };
            d.points.push(strip.rows[r][c]);
            d.keep.push(strip.keep[r][c]);
            d.expected.push(strip.expected[r][c]);
        }
    }
    d
}
