//! The strip model's fixed sampling layout, ported from villa (models/patch_tokens.py `polar_layout`,
//! sheet_strip/model.py `strip_grid`) for the checkpoint's configuration: the polar cell tokens' points and
//! sub-points and their centres, the strip tokens' anchors, rows, columns and profile offsets
//!
//! Coordinates are computed in f64 in villa's operation order and rounded to f32 where villa rounds them, so the
//! tables equal the buffers PyTorch samples with (the GPU parity test compares them)

use std::f64::consts::PI;

/// Crop width and depth in samples, samples behind the centre, crop units per sample
const WIDTH: f64 = 128.0;
const DEPTH: usize = 144;
const BEHIND: f64 = 72.0;
const SPACING: f64 = 0.5;
/// Samples per slab and points per patch axis
const PATCH: usize = 8;
const POINTS: usize = 2;
/// Tube radius, the first rings' widths, ring growth, sectors per turn, pooled sub-point spacing (samples)
const TUBE_RADIUS: f64 = 3.0;
const RING_WIDTHS: [f64; 3] = [3.0, 4.0, 6.0];
const RING_GROWTH: f64 = 1.5;
const SECTORS: usize = 6;
const POOL_SPACING: f64 = 4.0;
/// Strip half width, column spacing, rows per slab, profile reach (crop units)
const STRIP_HALF_WIDTH: f64 = 30.0;
const STRIP_V_SPACING: f64 = 2.5;
const STRIP_PROFILE_REACH: f64 = 6.0;

/// Slabs along the crop axis
pub const SLABS: usize = DEPTH / PATCH;
/// Sector and tube tokens
pub const SECTOR_TOKENS: usize = 1404;
pub const TUBE_TOKENS: usize = 18;
/// Cell tokens
pub const CELL_TOKENS: usize = SECTOR_TOKENS + TUBE_TOKENS;

/// One sampled point of a cell token: the mean of its sub-points
#[derive(Clone, Debug, PartialEq)]
pub struct Point {
    /// The token, layout order: sector tokens slab-major then ring then sector, then the tube tokens by slab
    pub token: u32,
    /// The point within its token (8 per sector token, 14 per tube token)
    pub index: u32,
    /// Its sub-points, crop-local xyz
    pub subs: Vec<[f32; 3]>,
}

/// The cell tokens' layout
pub struct Cells {
    /// Every point, tokens in layout order
    pub points: Vec<Point>,
    /// The tokens' centres (`cell_xyz`), crop-local xyz
    pub centres: Vec<[f32; 3]>,
}

/// `_local`: crop-local xyz of sample indices along width, height and depth
fn local(u: f64, v: f64, d: f64) -> [f64; 3] {
    let middle = (WIDTH - 1.0) / 2.0;
    [(u - middle) * SPACING, (v - middle) * SPACING, (d - BEHIND) * SPACING]
}

/// `_pooled`: sub-point offsets in units of a cell of `extent` samples, at most `POOL_SPACING` apart
fn pooled(extent: f64) -> Vec<f64> {
    let m = ((extent / POOL_SPACING - 1e-9).ceil() as usize).max(1);
    (0..m).map(|i| (i as f64 + 0.5) / m as f64 - 0.5).collect()
}

/// `polar_rings`: inner radius, width and sector count of each ring starting inside the crop's corner radius
fn rings() -> Vec<(f64, f64, usize)> {
    let corner = (WIDTH - 1.0) / 2.0 * 2f64.sqrt();
    let (mut rings, mut inner, mut width) = (Vec::new(), TUBE_RADIUS, RING_WIDTHS[0]);
    while inner < corner {
        // Python's round: halves to even
        let count = SECTORS * ((inner / width).round_ties_even() as usize).max(1);
        rings.push((inner, width, count));
        inner += width;
        width = RING_WIDTHS.get(rings.len()).copied().unwrap_or(width * RING_GROWTH);
    }
    rings
}

/// `crop_support` of an f32 point
fn supported(p: [f32; 3]) -> bool {
    let half = ((WIDTH - 1.0) * SPACING / 2.0) as f32;
    let (lo, hi) = (-(BEHIND * SPACING) as f32, ((DEPTH as f64 - 1.0 - BEHIND) * SPACING) as f32);
    p.iter().all(|v| v.is_finite()) && p[0].abs() <= half && p[1].abs() <= half && p[2] >= lo && p[2] <= hi
}

fn to_f32(p: [f64; 3]) -> [f32; 3] {
    [p[0] as f32, p[1] as f32, p[2] as f32]
}

/// The polar layout: per ring its sectors' points and sub-points (dropping sectors with no supported point in some
/// slab), then the tube
pub fn cells() -> Cells {
    let middle = (WIDTH - 1.0) / 2.0;
    let n = POINTS;
    // slab_depths: per slab the depth sample of each point and the slab centre; pooling leaves one depth sub-point
    let depth = |s: usize, dp: usize| (s * PATCH) as f64 + dp as f64 * (PATCH as f64 / n as f64);
    let depth_centre = |s: usize| (depth(s, 0) + depth(s, 1)) / 2.0;
    let frac = |i: usize| (i as f64 + 0.5) / n as f64;

    // per ring: (sectors kept, per sector its 8 points' sub-points' (u, v) and its centre's (u, v))
    type Sector = (Vec<Vec<(f64, f64)>>, (f32, f32));
    let mut ring_sectors: Vec<Vec<Sector>> = Vec::new();
    for (inner, width, count) in rings() {
        let sub_r = pooled(width / n as f64);
        let sub_a = pooled((inner + width) * 2.0 * PI / count as f64 / n as f64);
        let radius = |i: usize, k: usize| inner + (frac(i) + sub_r[k] / n as f64) * width;
        let mut kept = Vec::new();
        for j in 0..count {
            let angle = |i: usize, k: usize| 2.0 * PI * (j as f64 + frac(i) + sub_a[k] / n as f64) / count as f64;
            // points radial-major then angular; sub-points radial-major then angular
            let mut points = Vec::with_capacity(n * n);
            for ri in 0..n {
                for ai in 0..n {
                    let mut subs = Vec::with_capacity(sub_r.len() * sub_a.len());
                    for kr in 0..sub_r.len() {
                        for ka in 0..sub_a.len() {
                            let (r, a) = (radius(ri, kr), angle(ai, ka));
                            subs.push((middle + r * a.cos(), middle + r * a.sin()));
                        }
                    }
                    points.push(subs);
                }
            }
            // villa builds the centre's (u, v) as Python floats into a float32 tensor, then `_local` runs in f32
            let (mid_a, mid_r) = (2.0 * PI * (j as f64 + 0.5) / count as f64, inner + width / 2.0);
            let centre = ((middle + mid_r * mid_a.cos()) as f32, (middle + mid_r * mid_a.sin()) as f32);
            let keep = (0..SLABS).all(|s| {
                points.iter().any(|subs| {
                    (0..n).any(|dp| subs.iter().any(|&(u, v)| supported(to_f32(local(u, v, depth(s, dp))))))
                })
            });
            if keep {
                kept.push((points, centre));
            }
        }
        ring_sectors.push(kept);
    }

    let mut points = Vec::new();
    let mut centres = Vec::new();
    let mut token = 0u32;
    let half = middle as f32;
    for s in 0..SLABS {
        for sectors in &ring_sectors {
            for (cross, (cu, cv)) in sectors {
                // points depth-major, then the cross-section's radial-major order
                for dp in 0..n {
                    for (k, subs) in cross.iter().enumerate() {
                        let subs = subs.iter().map(|&(u, v)| to_f32(local(u, v, depth(s, dp)))).collect();
                        points.push(Point { token, index: (dp * n * n + k) as u32, subs });
                    }
                }
                let z = ((depth_centre(s) - BEHIND) * SPACING) as f32;
                centres.push([(cu - half) * SPACING as f32, (cv - half) * SPACING as f32, z]);
                token += 1;
            }
        }
    }
    assert_eq!(token as usize, SECTOR_TOKENS, "the polar layout's sector count");

    // tube: its axis and SECTORS points at half its radius, at each depth point
    let turn = |i: usize| i as f64 * 2.0 * PI / SECTORS as f64;
    let tube = |k: usize| {
        if k == 0 {
            (middle, middle)
        } else {
            (TUBE_RADIUS / 2.0 * turn(k - 1).cos() + middle, TUBE_RADIUS / 2.0 * turn(k - 1).sin() + middle)
        }
    };
    for s in 0..SLABS {
        for dp in 0..n {
            for k in 0..=SECTORS {
                let (u, v) = tube(k);
                let index = (dp * (SECTORS + 1) + k) as u32;
                points.push(Point { token, index, subs: vec![to_f32(local(u, v, depth(s, dp)))] });
            }
        }
        centres.push(to_f32(local(middle, middle, depth_centre(s))));
        token += 1;
    }
    Cells { points, centres }
}

/// The strip grid: rows `f` (36, crop-local), columns `v` (25), the profile offsets along u (25), and the strip
/// tokens' anchors `(0, v, slab centre)`, slab-major
pub struct Strip {
    pub f: Vec<f32>,
    pub v: Vec<f32>,
    pub offsets: Vec<f32>,
    pub anchors: Vec<[f32; 3]>,
}

/// `strip_grid`, the anchors and `strip_profile_offsets`
pub fn strip() -> Strip {
    let rows = DEPTH / PATCH * 2;
    let f: Vec<f32> =
        (0..rows).map(|r| (((r as f64 + 0.5) * PATCH as f64 / 2.0 - 0.5 - BEHIND) * SPACING) as f32).collect();
    let columns = (2.0 * STRIP_HALF_WIDTH / STRIP_V_SPACING).round() as usize + 1;
    // torch.linspace in f64: start + i * step for the first half, end - (n - 1 - i) * step for the rest
    let step = 2.0 * STRIP_HALF_WIDTH / (columns - 1) as f64;
    let v: Vec<f32> = (0..columns)
        .map(|i| {
            if i < columns / 2 {
                -STRIP_HALF_WIDTH + i as f64 * step
            } else {
                STRIP_HALF_WIDTH - (columns - 1 - i) as f64 * step
            }
        })
        .map(|x| x as f32)
        .collect();
    let k = (STRIP_PROFILE_REACH / SPACING).round() as i64;
    let offsets = (-k..=k).map(|i| (i as f64 * SPACING) as f32).collect();
    // slab centres: the mean of the slab's two rows, in f32
    let mut anchors = Vec::with_capacity(SLABS * columns);
    for s in 0..SLABS {
        let centre = (f[2 * s] + f[2 * s + 1]) / 2.0;
        for &vc in &v {
            anchors.push([0.0, vc, centre]);
        }
    }
    Strip { f, v, offsets, anchors }
}
