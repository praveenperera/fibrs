//! The cell tokens' sampling as one fixed sparse operator on the cell lattice
//!
//! A cell-token point's input is the mean over its sub-points of `sample_features` (trilinear `grid_sample` with
//! border clamping, zero outside the crop), which is linear in the cells, and every sub-point is a fixed
//! crop-local position. So each point's features are a fixed weighted sum of lattice cells: merging the sub-points'
//! corners per cell takes the gather from ~470k corner loads per decision to ~107k. The slab depths fall within
//! fp32 rounding of lattice planes, so half the corners carry weights near 1e-7; merged weights below `2^-24` are
//! dropped, which moves a point's value by far less than its bf16 rounding
//!
//! Weights follow the device sampler's fp32 coordinate arithmetic (`grid_sample`'s source index, the border clip,
//! the corner weight products), then merge in f64 and round once to f32. Support is fixed too: each point's
//! supported fraction, summed in the device's order

use std::collections::BTreeMap;

use color_eyre::eyre::{Result, ensure};

use super::layout::{Cells, SECTOR_TOKENS};

/// Cell lattice depth, height, width
const CD: u32 = 36;
const CH: u32 = 32;
const CW: u32 = 32;
/// Crop-local half extents of the lattice: x and y, and the depth behind the centre
const HALF: f32 = 31.75;
const BEHIND: f32 = 36.0;
const DEPTH_HI: f32 = 35.5;
/// Merged weights below this are dropped
const DROP: f64 = 1.0 / (1 << 24) as f64;
/// Points per sector token
const SECTOR_POINTS: usize = 8;

/// The operator: per point its entry range, output slot and support; per entry a lattice cell and its weight
pub struct Gather {
    /// Per point: first entry, entry count, `token * 16 + point within the token`, support (f32 bits)
    pub points: Vec<[u32; 4]>,
    /// Per entry: the cell's index in the lattice (`(z * 32 + y) * 32 + x`), its weight (f32 bits)
    pub entries: Vec<[u32; 2]>,
}

impl Gather {
    /// The sector tokens' points by slot, as `sm_sector_tokens` reads them: row `8 token + p` = first entry, entry
    /// count, support (f32 bits), 0
    pub fn sector_table(&self) -> Result<Vec<[u32; 4]>> {
        let mut table = vec![None; SECTOR_TOKENS * SECTOR_POINTS];
        for &[first, count, slot, support] in &self.points {
            let (token, p) = ((slot / 16) as usize, (slot % 16) as usize);
            if token >= SECTOR_TOKENS {
                continue;
            }
            ensure!(p < SECTOR_POINTS, "sector token {token} has a point {p}");
            let row = &mut table[token * SECTOR_POINTS + p];
            ensure!(row.is_none(), "sector token {token} point {p} appears twice");
            *row = Some([first, count, support, 0]);
        }
        table
            .into_iter()
            .enumerate()
            .map(|(i, row)| row.ok_or_else(|| color_eyre::eyre::eyre!("sector slot {i} has no point")))
            .collect()
    }

    /// The tube tokens' points, as `sm_tube_tokens` reads them
    pub fn tube_points(&self) -> Vec<[u32; 4]> {
        self.points.iter().filter(|p| p[2] / 16 >= SECTOR_TOKENS as u32).copied().collect()
    }
}

/// `grid_sample`'s source index on one axis from a crop-local coordinate, as the device computes it
fn source_index(p: f32, half: f32, size1: f32) -> f32 {
    let g = (2.0 * (p + half)) / (2.0 * size1) - 1.0;
    let i = ((g + 1.0) * 0.5) * size1;
    i.max(0.0).min(size1)
}

/// `crop_support`: finite and inside the crop
fn supported([x, y, z]: [f32; 3]) -> bool {
    (-HALF..=HALF).contains(&x) && (-HALF..=HALF).contains(&y) && (-BEHIND..=DEPTH_HI).contains(&z)
}

/// The trilinear corners of a supported point: (cell index, weight) for each corner inside the lattice
fn corners([x, y, z]: [f32; 3]) -> impl Iterator<Item = (u32, f32)> {
    let i = [
        source_index(x, HALF, (CW - 1) as f32),
        source_index(y, HALF, (CH - 1) as f32),
        source_index(z, BEHIND, (CD - 1) as f32),
    ];
    let lo = i.map(f32::floor);
    let f = [i[0] - lo[0], i[1] - lo[1], i[2] - lo[2]];
    let g = [(lo[0] + 1.0) - i[0], (lo[1] + 1.0) - i[1], (lo[2] + 1.0) - i[2]];
    let lo = lo.map(|v| v as u32);
    let size = [CW, CH, CD];
    (0..8u32).filter_map(move |corner| {
        let bit = [corner & 1, (corner >> 1) & 1, corner >> 2];
        let at = [lo[0] + bit[0], lo[1] + bit[1], lo[2] + bit[2]];
        // the far corner past the last cell has zero weight, and the device skips it
        if (0..3).any(|a| at[a] >= size[a]) {
            return None;
        }
        let w = |a: usize| if bit[a] == 0 { g[a] } else { f[a] };
        Some(((at[2] * CH + at[1]) * CW + at[0], w(0) * w(1) * w(2)))
    })
}

/// Builds the operator for the layout's points, in their order
pub fn gather(cells: &Cells) -> Gather {
    let mut points = Vec::with_capacity(cells.points.len());
    let mut entries = Vec::new();
    for pt in &cells.points {
        let m = pt.subs.len();
        let share = 1.0 / m as f64;
        let mut merged = BTreeMap::<u32, f64>::new();
        for &sub in pt.subs.iter().filter(|&&s| supported(s)) {
            for (cell, w) in corners(sub) {
                *merged.entry(cell).or_default() += w as f64 * share;
            }
        }

        // support as the device summed it: a single sub-point as is, else each flag times 1 / m in order
        let inv = 1.0f32 / m as f32;
        let support = if m == 1 {
            f32::from(u8::from(supported(pt.subs[0])))
        } else {
            pt.subs.iter().map(|&s| f32::from(u8::from(supported(s)))).fold(0.0f32, |acc, s| acc + s * inv)
        };

        let first = entries.len() as u32;
        entries
            .extend(merged.into_iter().filter(|&(_, w)| w.abs() >= DROP).map(|(cell, w)| [cell, (w as f32).to_bits()]));
        points.push([first, entries.len() as u32 - first, pt.token * 16 + pt.index, support.to_bits()]);
    }
    Gather { points, entries }
}

#[cfg(test)]
mod tests {
    use super::super::layout;
    use super::*;

    /// A deterministic pseudo-random cell value in [-4, 4)
    fn cell(index: u32, channel: u32) -> f64 {
        let mut x = (index as u64) << 8 | channel as u64;
        x = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        x ^= x >> 29;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 32;
        (x >> 11) as f64 / (1u64 << 53) as f64 * 8.0 - 4.0
    }

    /// The operator against the sub-point mean of the per-corner weights, in f64, on pseudo-random cells: every
    /// point and channel within a bf16 rounding's 1/64, and the support exactly the supported fraction
    #[test]
    fn operator_matches_subpoint_mean() {
        let cells = layout::cells();
        let op = gather(&cells);
        assert_eq!(op.points.len(), cells.points.len());
        let loads: usize = cells
            .points
            .iter()
            .flat_map(|p| p.subs.iter().filter(|&&s| supported(s)))
            .map(|&s| corners(s).count())
            .sum();
        println!("{} points: {} corner loads, {} merged entries", op.points.len(), loads, op.entries.len());
        assert!(op.entries.len() * 4 < loads, "merging cut the loads less than 4x");

        let mut worst = 0.0f64;
        for (pt, &[first, count, slot, support]) in cells.points.iter().zip(&op.points) {
            assert_eq!(slot, pt.token * 16 + pt.index);
            let m = pt.subs.len() as f64;
            let frac = pt.subs.iter().filter(|&&s| supported(s)).count() as f64 / m;
            assert!(
                (f32::from_bits(support) as f64 - frac).abs() < 1e-6,
                "support {} vs {frac}",
                f32::from_bits(support)
            );
            for channel in [0, 17, 63] {
                let want: f64 = pt
                    .subs
                    .iter()
                    .filter(|&&s| supported(s))
                    .flat_map(|&s| corners(s))
                    .map(|(c, w)| w as f64 * cell(c, channel) / m)
                    .sum();
                let got: f64 = op.entries[first as usize..(first + count) as usize]
                    .iter()
                    .map(|&[c, w]| {
                        assert!(c < CD * CH * CW);
                        f32::from_bits(w) as f64 * cell(c, channel)
                    })
                    .sum();
                worst = worst.max((got - want).abs());
            }
        }
        println!("max |operator - sub-point mean|: {worst:.3e}");
        let sector = op.sector_table().expect("every sector token has its 8 points once");
        assert_eq!(sector.len() + op.tube_points().len(), op.points.len());
        // a bf16 value near 4 rounds by up to 2^-7; the operator must stay far inside that
        assert!(worst < 1.0 / 512.0 / 64.0, "operator error {worst}");
    }
}
