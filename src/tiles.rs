//! Tiled tracing (`--tiles-yx`): a window traced as Y-X tiles one after another, and the tiles' fibres joined
//!
//! The window's Y and X extent is split into NY × NX tile cores. Each tile is traced from a crop of the predictions:
//! its core plus a halo of its neighbours' voxels on each side, clipped to the window, so the run's peak memory is
//! its densest tile's rather than the window's. Every core edge and the halo are whole units of [`alignment`], so a
//! crop lies on whole chunks of both prediction arrays and on the window's 2× pooling and centrality grids. The
//! tiles' fibres are then cut to their cores and joined across the seams ([`crate::seams`]) as they are exported

use std::path::Path;

use color_eyre::eyre::{Result, bail};
use serde_json::{Value, json};

use crate::afv::{Curve, tracing_to_native};
use crate::fields;
use crate::input::Predictions;
use crate::run::{CurveReader, gcd};
use crate::seams::{CoreBox, Face, FamilySeams, Point, RADIUS, Reports};

/// Native voxels of the Y and X extent: `lo..hi`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub lo: usize,
    pub hi: usize,
}

/// One tile, in native voxels of the window, Y then X
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tile {
    /// Row and column in the grid
    pub index: [usize; 2],
    /// The voxels whose fibres this tile keeps
    pub core: [Span; 2],
    /// The voxels it is traced from: the core and up to the halo of its neighbours on each side
    pub traced: [Span; 2],
}

impl Tile {
    /// The tile's work directory under the run's: `tile-YY-XX`
    pub fn name(&self) -> String {
        format!("tile-{:02}-{:02}", self.index[0], self.index[1])
    }
}

/// The unit, in native voxels, that tile cores and halos are made of: the least common multiple of
///
/// - the native array's chunk edges, so a native crop lies on whole chunks and the native cache reads it unchanged,
/// - twice the tracing array's chunk edges, since the tracing crop starts at half the native offset,
/// - twice the centrality pass's tile edge: its kernel clamps samples to its tiles, so a crop keeps the window's
///   centrality only when its tracing offset is a whole number of them
///
/// Y and X only; Z is not tiled. NPY input reads in [`crate::input::NPY_BLOCK`]s
pub fn alignment(native: &Predictions, supplied: Option<&Predictions>) -> usize {
    let mut units = vec![2 * fields::TILE];
    let unit = native.read_unit();
    units.extend([unit[1], unit[2]]);
    if let Some(s) = supplied {
        let unit = s.read_unit();
        units.extend([2 * unit[1], 2 * unit[2]]);
    }
    units.into_iter().fold(1, |a, b| a / gcd(a, b) * b)
}

/// The tile grid over a window's Y and X extent
#[derive(Clone, Debug)]
pub struct Tiling {
    grid: [usize; 2],
    /// Halo in native voxels, a whole number of `align` units
    halo: usize,
    align: usize,
    extent: [usize; 2],
    /// Core edges along Y and X: 0, the interior edges, then the extent
    edges: [Vec<usize>; 2],
}

impl Tiling {
    /// `grid` (NY, NX) tile cores over a Y-X `extent` of native voxels, each traced with up to `halo` voxels of its
    /// neighbours on each side; core edges fall at `floor(k × extent / N / align) × align` and the halo is rounded up
    /// to a whole number of `align` units. Every core must hold at least one unit
    pub fn new(grid: [usize; 2], halo: usize, align: usize, extent: [usize; 2]) -> Result<Self> {
        if grid.contains(&0) {
            bail!("--tiles-yx needs at least one tile along each axis");
        }
        // a tile traced from its core alone stops short of its seams, so no fibre would reach a face to be joined
        if halo == 0 && grid != [1, 1] {
            bail!("--tile-halo 0 leaves no fibre reaching a seam; give at least 1 (rounded up to {align})");
        }
        let mut edges = [Vec::new(), Vec::new()];
        for (a, axis) in ["Y", "X"].into_iter().enumerate() {
            let n = grid[a];
            // one tile is the whole extent; more need n whole units, and then consecutive edges differ by at least
            // one unit. Checked before the edges are built, so an absurd grid is refused rather than allocated
            if n > 1 && n > extent[a] / align {
                bail!(
                    "--tiles-yx {} {}: the {axis} extent of {} voxels does not split into {n} tiles of whole \
                     {align}-voxel units",
                    grid[0],
                    grid[1],
                    extent[a]
                );
            }

            edges[a] = (0..n).map(|k| k * extent[a] / (n * align) * align).chain([extent[a]]).collect();
        }

        Ok(Self { grid, halo: halo.div_ceil(align) * align, align, extent, edges })
    }

    /// Tiles along Y and X
    pub fn grid(&self) -> [usize; 2] {
        self.grid
    }

    /// Every tile, row by row
    pub fn tiles(&self) -> Vec<Tile> {
        let span = |a: usize, k: usize| Span { lo: self.edges[a][k], hi: self.edges[a][k + 1] };
        let traced =
            |s: Span, a: usize| Span { lo: s.lo.saturating_sub(self.halo), hi: (s.hi + self.halo).min(self.extent[a]) };
        let mut out = Vec::with_capacity(self.grid[0] * self.grid[1]);
        for y in 0..self.grid[0] {
            for x in 0..self.grid[1] {
                let core = [span(0, y), span(1, x)];
                out.push(Tile { index: [y, x], core, traced: [traced(core[0], 0), traced(core[1], 1)] });
            }
        }
        out
    }

    /// The tiling as `run.json` records it
    pub fn record(&self) -> Value {
        let tiles: Vec<Value> = self
            .tiles()
            .iter()
            .map(|t| {
                json!({
                    "name": t.name(),
                    "core_yx": [[t.core[0].lo, t.core[0].hi], [t.core[1].lo, t.core[1].hi]],
                    "traced_yx": [[t.traced[0].lo, t.traced[0].hi], [t.traced[1].lo, t.traced[1].hi]],
                })
            })
            .collect();
        json!({"grid_yx": self.grid, "halo": self.halo, "align": self.align, "tiles": tiles})
    }

    /// The faces between tile cores in the export's coordinates (x, y, z with the window's origin added)
    fn interior(&self, origin_xyz: [i64; 3]) -> Vec<Face> {
        // the points' axis of the grid's Y and X
        let axes = [1, 0];
        let mut faces = Vec::new();
        for a in 0..2 {
            let interior = &self.edges[a][1..self.grid[a]];
            faces.extend(interior.iter().map(|&e| Face { axis: axes[a], at: (origin_xyz[axes[a]] + e as i64) as f64 }));
        }
        faces
    }

    /// A tile's core box in the export's coordinates; open at the window's edges, which no tile lies beyond, so
    /// a 1 × 1 tiling keeps every point of every fibre
    fn core_box(&self, tile: &Tile, origin_xyz: [i64; 3]) -> CoreBox {
        let mut lo = [f64::NEG_INFINITY; 3];
        let mut hi = [f64::INFINITY; 3];
        for (a, axis) in [(0, 1), (1, 0)] {
            let at = |e: usize| (origin_xyz[axis] + e as i64) as f64;
            let core = tile.core[a];
            if core.lo > 0 {
                lo[axis] = at(core.lo);
            }
            if core.hi < self.extent[a] {
                hi[axis] = at(core.hi);
            }
        }
        CoreBox { lo, hi }
    }
}

/// A tile's fibres of one family in native coordinates, read back from its work directory
struct TileFibres {
    reader: CurveReader,
    /// The native coordinate of the tile's first voxel, x, y, z, for [`tracing_to_native`]
    origin: [f64; 3],
}

impl Iterator for TileFibres {
    type Item = Result<Vec<Point>>;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.reader.next()?;
        Some(item.map(|(_, points)| tracing_to_native(&points, self.origin)))
    }
}

/// The export's curves for a tiled trace, in the order an untiled trace writes them: family by family, every tile's
/// uncut fibres in tile order, then the chains the family's cut runs join into
///
/// Curves are read, converted and cut one at a time, and only one family's cut runs are held, so the export holds
/// no more than the runs that touch a seam
pub fn seam_curves<'a>(
    work: &'a Path,
    tiling: &'a Tiling,
    families: &'a [usize],
    origin_xyz: [i64; 3],
) -> impl Iterator<Item = Result<Curve>> + 'a {
    let interior = tiling.interior(origin_xyz);
    // the tiled trace logs each family's counts; nothing else reads them
    let reports = Reports::default();
    families.iter().flat_map(move |&family| {
        let pieces = tiling.tiles().into_iter().map(move |t| {
            let core = tiling.core_box(&t, origin_xyz);
            let origin = [
                (origin_xyz[0] + t.traced[1].lo as i64) as f64,
                (origin_xyz[1] + t.traced[0].lo as i64) as f64,
                origin_xyz[2] as f64,
            ];
            let reader = CurveReader::open(&work.join(t.name()), family)?;
            Ok((core, TileFibres { reader, origin }))
        });
        FamilySeams::new(family, pieces, interior.clone(), RADIUS, reports.clone())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cores_and_halos_are_whole_units_clipped_to_the_extent() {
        let tiling = Tiling::new([2, 3], 100, 256, [1000, 2000]).unwrap();
        assert_eq!(tiling.halo, 256);
        let tiles = tiling.tiles();
        assert_eq!(tiles.len(), 6);
        let spans = |t: &Tile| (t.core.map(|s| (s.lo, s.hi)), t.traced.map(|s| (s.lo, s.hi)));
        // Y: 1000 / 2 = 500 -> 256; X: 2000 / 3 = 666 -> 512, 4000 / 3 = 1333 -> 1280
        assert_eq!(spans(&tiles[0]), ([(0, 256), (0, 512)], [(0, 512), (0, 768)]));
        assert_eq!(spans(&tiles[1]), ([(0, 256), (512, 1280)], [(0, 512), (256, 1536)]));
        assert_eq!(spans(&tiles[5]), ([(256, 1000), (1280, 2000)], [(0, 1000), (1024, 2000)]));
        assert_eq!(tiles[5].name(), "tile-01-02");

        let one = Tiling::new([1, 1], 256, 256, [300, 77]).unwrap().tiles();
        assert_eq!(spans(&one[0]), ([(0, 300), (0, 77)], [(0, 300), (0, 77)]));
        // a 511-voxel extent holds one whole 256-voxel unit, too few for two cores
        assert!(Tiling::new([2, 1], 256, 256, [511, 600]).is_err());
        assert!(Tiling::new([0, 1], 256, 256, [600, 600]).is_err());
        assert!(Tiling::new([1_000_000_000, 1], 256, 256, [640, 640]).is_err());
        assert!(Tiling::new([1, 2], 0, 256, [600, 600]).is_err());
        assert!(Tiling::new([1, 1], 0, 256, [600, 600]).is_ok());
    }

    /// The core box is open where the core meets the window's edge, and the interior faces are the core edges
    /// between tiles, both with the window's origin added
    #[test]
    fn core_boxes_and_faces_are_in_export_coordinates() {
        let tiling = Tiling::new([2, 2], 1, 256, [600, 700]).unwrap();
        let origin = [1000, 2000, 3000];
        let tiles = tiling.tiles();
        let first = tiling.core_box(&tiles[0], origin);
        assert_eq!(first.lo, [f64::NEG_INFINITY; 3]);
        assert_eq!(first.hi, [1256.0, 2256.0, f64::INFINITY]);
        let last = tiling.core_box(&tiles[3], origin);
        assert_eq!(last.lo, [1256.0, 2256.0, f64::NEG_INFINITY]);
        assert_eq!(last.hi, [f64::INFINITY; 3]);
        assert_eq!(tiling.interior(origin), [Face { axis: 1, at: 2256.0 }, Face { axis: 0, at: 1256.0 }]);
    }
}
