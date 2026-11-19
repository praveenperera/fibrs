//! Per-family fields and the seed catalogue (`prepare.prepare_family`, `prepare.prepare_seeds`)

use rayon::prelude::*;

use crate::kernels::{Dense, Grid, stage1};
use crate::pyfmt::argsort_u16;
use crate::vol::{Shape, Vol};

/// Tile edge and halo of the reference's centrality pass (prepare.py:148-150); the kernel clamps samples
/// to its tile, so these are part of the output, not a tuning choice
const TILE: usize = 128;
const TILE_HALO: usize = 8;

/// One family's centrality and its seeds
pub struct FamilyFields {
    pub cent: Vol<u16>,
    /// Every voxel of seed priority 1 or 2, in C order over the volume; a list, because seeds are a few voxels
    /// in every brick and a volume of them would allocate nearly all of the family's bricks
    pub seeds: Vec<Seed>,
}

/// A seed voxel's C-order index in the volume and its priority, packed in 8 bytes: the index in the high 56
/// bits, so seeds sort in C order, and the priority in the low 8
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Seed(u64);

impl Seed {
    /// Volumes up to this many voxels have every index fit in 56 bits
    const MAX_VOXELS: usize = 1 << 56;

    fn new(at: Shape, shape: Shape, priority: u8) -> Self {
        let index = (at[0] * shape[1] + at[1]) * shape[2] + at[2];
        Self((index as u64) << 8 | u64::from(priority))
    }

    /// The seed's z, y, x in a volume of `shape`
    pub fn coords(self, shape: Shape) -> [i32; 3] {
        let index = (self.0 >> 8) as usize;
        [index / (shape[1] * shape[2]), index / shape[2] % shape[1], index % shape[2]].map(|v| v as i32)
    }

    pub fn priority(self) -> u8 {
        self.0 as u8
    }
}

/// Presence of `family` (1 = V, 2 = H): `min(own + other, 255)`, zero where `own < other`
///
/// Computed from the pooled predictions on every read instead of stored: the pooled predictions stay resident
/// for finishing anyway, so a stored presence would add a byte per voxel of data to the trace's peak
pub struct Presence<'a> {
    own: &'a Vol<u8>,
    other: &'a Vol<u8>,
}

impl<'a> Presence<'a> {
    pub fn new(tracing: &'a [Vol<u8>; 2], family: usize) -> Self {
        Self { own: &tracing[family - 1], other: &tracing[2 - family] }
    }

    pub fn shape(&self) -> Shape {
        self.own.shape()
    }

    /// The value at an in-bounds voxel
    #[inline(always)]
    pub fn get(&self, z: usize, y: usize, x: usize) -> u8 {
        // the pooled predictions read 0 outside their sub-bricks, and an own value of 0 gives presence 0 whatever
        // the other is, so most air reads stop at the first lookup
        let (own, at) = self.own.locate(z, y, x);
        let a = own[at];
        if a == 0 {
            return 0;
        }
        presence(a, self.other.locate(z, y, x).0[at])
    }

    /// A dense C-order block with origin `lo` and extent `dims`
    pub fn read_block(&self, lo: Shape, dims: Shape) -> Vec<u8> {
        let mut out = vec![0u8; dims.iter().product()];
        let mut other = vec![0u8; dims[2]];
        for z in 0..dims[0] {
            for y in 0..dims[1] {
                let row = &mut out[(z * dims[1] + y) * dims[2]..][..dims[2]];
                self.own.read_row(lo[0] + z, lo[1] + y, lo[2], row);
                self.other.read_row(lo[0] + z, lo[1] + y, lo[2], &mut other);
                for (v, &b) in row.iter_mut().zip(&other) {
                    *v = presence(*v, b);
                }
            }
        }
        out
    }

    /// Whether any voxel of the block `[lo, lo + dims)` satisfies `pred`; presence is 0 wherever the family's
    /// own pooled prediction is, so only its allocated bricks are visited
    pub fn any_in(&self, lo: Shape, dims: Shape, pred: impl Fn(u8) -> bool) -> bool {
        self.own.any_allocated_in(lo, dims, |z, y, x| pred(self.get(z, y, x)))
    }
}

impl Grid<u8> for Presence<'_> {
    #[inline(always)]
    fn dims(&self) -> [i64; 3] {
        self.shape().map(|n| n as i64)
    }

    #[inline(always)]
    fn at(&self, z: i64, y: i64, x: i64) -> u8 {
        self.get(z as usize, y as usize, x as usize)
    }
}

/// Presence from the family's own pooled prediction and the other's
#[inline(always)]
fn presence(own: u8, other: u8) -> u8 {
    if own < other { 0 } else { (own as u16 + other as u16).min(255) as u8 }
}

/// The reference's `tiles(shape, size)` order: C order over tile origins
fn tiles(shape: Shape, size: usize) -> Vec<(Shape, Shape)> {
    let grid = shape.map(|n| n.div_ceil(size));
    let mut out = Vec::with_capacity(grid.iter().product());
    for z in 0..grid[0] {
        for y in 0..grid[1] {
            for x in 0..grid[2] {
                let lo = [z * size, y * size, x * size];
                let hi = [0, 1, 2].map(|a| (lo[a] + size).min(shape[a]));
                out.push((lo, hi));
            }
        }
    }
    out
}

/// One tile's core: origin, extent, centrality and seeds
type TileFields = (Shape, Shape, Vec<u16>, Vec<u8>);

/// Centrality and seeds over the reference's 128³ tiles with an 8-voxel halo, tiles in parallel
pub fn centrality(pres: &Presence, nx: &Vol<u8>, ny: &Vol<u8>) -> FamilyFields {
    let shape = pres.shape();
    assert!(
        shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d)).is_some_and(|n| n <= Seed::MAX_VOXELS),
        "volume of {shape:?} voxels is too large to index seeds"
    );
    let jobs = tiles(shape, TILE);
    let mut cent = Vol::new(shape, 0u16);
    let mut seeds = Vec::new();
    // tiles run a few per thread at a time and are written before the next run, so the dense tile results held
    // at once stay small; tile cores are disjoint, so the order they are written in does not matter
    for part in jobs.chunks(4 * rayon::current_num_threads()) {
        let results: Vec<Option<TileFields>> =
            part.par_iter().map(|&(start, end)| centrality_tile(pres, nx, ny, start, end)).collect();
        for (start, inner, c, s) in results.into_iter().flatten() {
            cent.write_block(start, inner, &c);
            let mut i = 0;
            for z in 0..inner[0] {
                for y in 0..inner[1] {
                    for x in 0..inner[2] {
                        if s[i] != 0 {
                            seeds.push(Seed::new([start[0] + z, start[1] + y, start[2] + x], shape, s[i]));
                        }
                        i += 1;
                    }
                }
            }
        }
    }
    // the catalogue reads seeds in C order over the whole volume; indices are unique, so any sort agrees
    seeds.sort_unstable();
    // the list grew by doubling and is held through the catalogue; drop the unused half
    seeds.shrink_to_fit();
    FamilyFields { cent, seeds }
}

/// One tile's core centrality and seeds, or None when no voxel of the tile and its halo reaches presence 64
fn centrality_tile(pres: &Presence, nx: &Vol<u8>, ny: &Vol<u8>, start: Shape, end: Shape) -> Option<TileFields> {
    let shape = pres.shape();
    let lo = start.map(|v| v.saturating_sub(TILE_HALO));
    let hi = [0, 1, 2].map(|a| (end[a] + TILE_HALO).min(shape[a]));
    let dims = [0, 1, 2].map(|a| hi[a] - lo[a]);
    // a tile with no voxel at 64 or more has zero centrality and no seeds
    if !pres.any_in(lo, dims, |p| p >= 64) {
        return None;
    }
    let block = |v: &Vol<u8>| Dense { dims, data: v.read_block(lo, dims) };
    let pres = Dense { dims, data: pres.read_block(lo, dims) };
    let (cent, seeds) = stage1(&pres, &block(nx), &block(ny));
    let inner = [0, 1, 2].map(|a| end[a] - start[a]);
    let off = [0, 1, 2].map(|a| start[a] - lo[a]);
    let mut c = Vec::with_capacity(inner.iter().product());
    let mut s = Vec::with_capacity(c.capacity());
    for z in 0..inner[0] {
        for y in 0..inner[1] {
            let at = ((z + off[0]) * dims[1] + y + off[1]) * dims[2] + off[2];
            c.extend_from_slice(&cent[at..at + inner[2]]);
            s.extend_from_slice(&seeds[at..at + inner[2]]);
        }
    }
    Some((start, inner, c, s))
}

/// The seed catalogue: coordinates in tracing order, grouped by 1024³ region
pub struct Catalogue {
    pub coords: Vec<[i32; 3]>,
    /// `starts[r]..starts[r + 1]` are region r's seeds
    pub starts: Vec<usize>,
    /// The first `primary[r]` seeds of region r are priority 2
    pub primary: Vec<usize>,
}

/// `prepare_seeds` (prepare.py:162): priority 2 then 1, each in descending centrality (reversed
/// unstable argsort), then a stable sort by region
pub fn catalogue(seeds: &[Seed], cent: &Vol<u16>) -> Catalogue {
    let shape = cent.shape();
    let mut by_priority: [Vec<[i32; 3]>; 2] = [Vec::new(), Vec::new()];
    for &seed in seeds {
        let v = seed.priority();
        if v == 2 || v == 1 {
            by_priority[(v == 1) as usize].push(seed.coords(shape));
        }
    }
    let n_primary = by_priority[0].len();
    let mut ordered = Vec::with_capacity(n_primary + by_priority[1].len());
    for raw in &by_priority {
        let values: Vec<u16> = raw.iter().map(|c| cent.get(c[0] as usize, c[1] as usize, c[2] as usize)).collect();
        let order = argsort_u16(&values);
        ordered.extend(order.iter().rev().map(|&i| raw[i]));
    }
    const REGION: usize = 1024;
    let grid = shape.map(|n| n.div_ceil(REGION));
    let key = |c: &[i32; 3]| {
        let r = c.map(|v| v as usize / REGION);
        (r[0] * grid[1] + r[1]) * grid[2] + r[2]
    };
    let keys: Vec<usize> = ordered.iter().map(key).collect();
    let n_regions = grid.iter().product();
    let mut counts = vec![0usize; n_regions];
    let mut primary = vec![0usize; n_regions];
    for (i, &k) in keys.iter().enumerate() {
        counts[k] += 1;
        if i < n_primary {
            primary[k] += 1;
        }
    }
    let mut idx: Vec<usize> = (0..ordered.len()).collect();
    idx.sort_by_key(|&i| keys[i]);
    let coords = idx.iter().map(|&i| ordered[i]).collect();
    let mut starts = vec![0usize; n_regions + 1];
    for r in 0..n_regions {
        starts[r + 1] = starts[r] + counts[r];
    }
    Catalogue { coords, starts, primary }
}
