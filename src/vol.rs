//! Sparse in-memory volumes stored as 16³ bricks of 4³ sub-bricks
//!
//! The band's fields are mostly air: a sub-brick is allocated only once a value other than the volume's
//! default is written to it, and reads from an unallocated one return the default. This replaces the
//! reference's compressed Zarr work store and its paged tile cache with plain memory. Sub-bricks live in a
//! [`BrickStore`] owned by the volume, not one heap allocation each

use std::ops::Range;

use rayon::prelude::*;

use crate::arena::{BrickStore, Voxel};

/// Brick edge, as a shift
///
/// Bricks are the unit of the volume's slot table and of [`Vol::fill_brick_rows`]; each holds the ids of its
/// [`SUB`] sub-bricks in a table of its own, allocated with its first sub-brick
pub const SHIFT: usize = 4;
/// Brick edge in voxels
pub const EDGE: usize = 1 << SHIFT;
/// Sub-brick edge, as a shift
///
/// A sub-brick is allocated whole for its first non-default voxel. The fields hold thin structures (centrality
/// ridges, the tubes coverage marks around each curve), so the smaller the unit, the less air is stored with them:
/// on a dense 1,024-plane PHerc0358 window centrality's non-default voxels filled 9% of the 16³ bricks holding
/// them, and 4³ sub-bricks held them in 39% of the memory, coverage's in 45%. Below 4³ a sub-brick's id (4 bytes)
/// would cost more than the voxels it saves
const SUB_SHIFT: usize = 2;
const SUB_EDGE: usize = 1 << SUB_SHIFT;
const SUB_MASK: usize = SUB_EDGE - 1;
/// Voxels per sub-brick
const SUB: usize = SUB_EDGE * SUB_EDGE * SUB_EDGE;
/// Sub-bricks per brick
const SUBS: usize = BRICK / SUB;
/// Voxels per brick
pub const BRICK: usize = EDGE * EDGE * EDGE;

/// A z, y, x shape
pub type Shape = [usize; 3];

/// A sparse 3D volume of `T` in z, y, x order
pub struct Vol<T: Voxel> {
    shape: Shape,
    grid: Shape,
    default: T,
    /// Each brick's table of sub-brick ids in `tables`; 0, the table of all zeros, while every voxel of the brick
    /// reads as the default
    bricks: Vec<u32>,
    /// The tables, [`SUBS`] sub-brick ids each, 0 for an unallocated sub-brick; id 0 is the table of every
    /// unallocated brick
    tables: BrickStore<u32, SUBS>,
    /// The sub-bricks; id 0 holds the default and stands for every unallocated sub-brick
    store: BrickStore<T, SUB>,
}

// SAFETY: the table and sub-brick ids address `tables` and `store`, which the volume owns and frees only
// when it is dropped. Tables and sub-bricks are written only through `&mut self` (each brick by one writer in
// `fill_brick_rows`) and read through `&self`, so the volume follows the same sharing rules as `Box`ed bricks
unsafe impl<T: Voxel> Send for Vol<T> {}
unsafe impl<T: Voxel> Sync for Vol<T> {}

impl<T: Voxel> Vol<T> {
    /// An empty volume where every voxel reads as `default`
    pub fn new(shape: Shape, default: T) -> Self {
        let grid = shape.map(|n| n.div_ceil(EDGE));
        let slots = grid.iter().product();
        let bricks = vec![0; slots];
        // each store's id 0 is handed out first: the empty table and the default sub-brick
        let mut tables = BrickStore::new(0, slots + 1);
        tables.alloc();
        let mut store = BrickStore::new(default, slots * SUBS + 1);
        store.alloc();
        Self { shape, grid, default, bricks, tables, store }
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    /// Brick grid size in z, y, x
    pub fn grid(&self) -> Shape {
        self.grid
    }

    #[inline(always)]
    fn brick_index(&self, z: usize, y: usize, x: usize) -> usize {
        ((z >> SHIFT) * self.grid[1] + (y >> SHIFT)) * self.grid[2] + (x >> SHIFT)
    }

    /// A voxel's sub-brick within its brick
    #[inline(always)]
    fn sub_index(z: usize, y: usize, x: usize) -> usize {
        let s = SHIFT - SUB_SHIFT;
        let m = (1 << s) - 1;
        ((z >> SUB_SHIFT & m) << (2 * s)) | ((y >> SUB_SHIFT & m) << s) | (x >> SUB_SHIFT & m)
    }

    /// A voxel's offset within its sub-brick
    #[inline(always)]
    fn offset(z: usize, y: usize, x: usize) -> usize {
        ((z & SUB_MASK) << (2 * SUB_SHIFT)) | ((y & SUB_MASK) << SUB_SHIFT) | (x & SUB_MASK)
    }

    /// The sub-brick ids of a brick, all 0 for an unallocated one
    #[inline(always)]
    fn table(&self, bi: usize) -> &[u32; SUBS] {
        // SAFETY: every brick names a table `tables` handed out, which lives as long as `self`
        unsafe { &*self.tables.get(self.bricks[bi]).as_ptr().cast::<[u32; SUBS]>() }
    }

    /// The voxels of a sub-brick, the default one for id 0
    #[inline(always)]
    fn sub(&self, id: u32) -> &[T; SUB] {
        // SAFETY: every id in a table was handed out by `store`, which lives as long as `self`
        unsafe { &*self.store.get(id).as_ptr().cast::<[T; SUB]>() }
    }

    /// The value at an in-bounds voxel
    #[inline(always)]
    pub fn get(&self, z: usize, y: usize, x: usize) -> T {
        let (sub, at) = self.locate(z, y, x);
        sub[at]
    }

    /// The sub-brick holding an in-bounds voxel and the voxel's offset in it; every voxel of an unallocated
    /// sub-brick reads as the default, and offsets agree between volumes, so one offset reads two volumes
    #[inline(always)]
    pub fn locate(&self, z: usize, y: usize, x: usize) -> (&[T; SUB], usize) {
        debug_assert!(z < self.shape[0] && y < self.shape[1] && x < self.shape[2]);
        let id = self.table(self.brick_index(z, y, x))[Self::sub_index(z, y, x)];
        (self.sub(id), Self::offset(z, y, x))
    }

    /// The 2 × 2 × 2 values from an in-bounds voxel to the one past it on every axis, which must be in bounds too,
    /// in z, y, x order: corner `dz * 4 + dy * 2 + dx`
    ///
    /// The tracer's trilinear samples read cells like this; when the cell lies in one sub-brick, it is found once
    #[inline(always)]
    pub fn cell(&self, z: usize, y: usize, x: usize) -> [T; 8] {
        if (z & SUB_MASK) != SUB_MASK && (y & SUB_MASK) != SUB_MASK && (x & SUB_MASK) != SUB_MASK {
            let (sub, at) = self.locate(z, y, x);
            let (dy, dz) = (SUB_EDGE, SUB_EDGE * SUB_EDGE);
            return [0, 1, dy, dy + 1, dz, dz + 1, dz + dy, dz + dy + 1].map(|d| sub[at + d]);
        }
        std::array::from_fn(|c| self.get(z + (c >> 2), y + (c >> 1 & 1), x + (c & 1)))
    }

    /// The value at a voxel given as signed coordinates, which must be in bounds
    #[inline(always)]
    pub fn at(&self, z: i64, y: i64, x: i64) -> T {
        self.get(z as usize, y as usize, x as usize)
    }

    /// Set one voxel, allocating its brick's table and its sub-brick on first non-default write
    #[inline]
    pub fn set(&mut self, z: usize, y: usize, x: usize, value: T) {
        let bi = self.brick_index(z, y, x);
        let table = match self.bricks[bi] {
            0 if value == self.default => return,
            0 => {
                let (id, t) = self.tables.alloc();
                self.bricks[bi] = id;
                t
            }
            id => self.tables.get(id),
        };
        // SAFETY: the table is SUBS ids in `tables`, and `&mut self` makes this the only access to it
        let slot = unsafe { &mut *table.as_ptr().add(Self::sub_index(z, y, x)) };
        let ptr = match *slot {
            0 if value == self.default => return,
            0 => {
                let (id, ptr) = self.store.alloc();
                *slot = id;
                ptr
            }
            id => self.store.get(id),
        };
        // SAFETY: an allocated sub-brick is SUB voxels of `store`, and `&mut self` makes this the only access to it
        unsafe { *ptr.as_ptr().add(Self::offset(z, y, x)) = value };
    }

    /// Bytes held by allocated sub-bricks and the tables that name them
    pub fn resident_bytes(&self) -> usize {
        let tables = self.bricks.iter().filter(|&&t| t != 0).count();
        let subs: usize = (0..self.bricks.len())
            .filter(|&bi| self.bricks[bi] != 0)
            .map(|bi| self.table(bi).iter().filter(|&&id| id != 0).count())
            .sum();
        tables * SUBS * size_of::<u32>() + subs * SUB * size_of::<T>()
    }

    /// Copy a dense C-order block with origin `lo` and extent `dims` into the volume; sub-bricks that would
    /// stay all-default are not allocated
    pub fn write_block(&mut self, lo: Shape, dims: Shape, data: &[T]) {
        assert_eq!(data.len(), dims.iter().product::<usize>());
        for z in 0..dims[0] {
            for y in 0..dims[1] {
                let row = &data[(z * dims[1] + y) * dims[2]..][..dims[2]];
                for (x, &v) in row.iter().enumerate() {
                    self.set(lo[0] + z, lo[1] + y, lo[2] + x, v);
                }
            }
        }
    }

    /// Read a dense C-order block with origin `lo` and extent `dims`
    pub fn read_block(&self, lo: Shape, dims: Shape) -> Vec<T> {
        let mut out = vec![self.default; dims.iter().product()];
        for z in 0..dims[0] {
            for y in 0..dims[1] {
                self.read_row(lo[0] + z, lo[1] + y, lo[2], &mut out[(z * dims[1] + y) * dims[2]..][..dims[2]]);
            }
        }
        out
    }

    /// Read the voxels of row `z`, `y` from `x` on into `out`, finding each sub-brick once instead of per voxel
    #[inline]
    pub fn read_row(&self, z: usize, y: usize, x: usize, out: &mut [T]) {
        debug_assert!(x + out.len() <= self.shape[2]);
        let mut at = 0;
        while at < out.len() {
            let gx = x + at;
            let n = (SUB_EDGE - (gx & SUB_MASK)).min(out.len() - at);
            let (sub, offset) = self.locate(z, y, gx);
            out[at..at + n].copy_from_slice(&sub[offset..offset + n]);
            at += n;
        }
    }

    /// Whether `f(z, y, x)` holds at any voxel of the block `[lo, lo + dims)` that lies in an allocated sub-brick
    pub fn any_allocated_in(&self, lo: Shape, dims: Shape, f: impl Fn(usize, usize, usize) -> bool) -> bool {
        let first = lo.map(|v| v >> SUB_SHIFT);
        let last = [0, 1, 2].map(|a| (lo[a] + dims[a] - 1) >> SUB_SHIFT);
        for sz in first[0]..=last[0] {
            for sy in first[1]..=last[1] {
                for sx in first[2]..=last[2] {
                    let slo = [sz, sy, sx].map(|v| v << SUB_SHIFT);
                    if self.table(self.brick_index(slo[0], slo[1], slo[2]))[Self::sub_index(slo[0], slo[1], slo[2])]
                        == 0
                    {
                        continue;
                    }
                    let a0 = [0, 1, 2].map(|a| slo[a].max(lo[a]));
                    let a1 = [0, 1, 2].map(|a| (slo[a] + SUB_EDGE).min(lo[a] + dims[a]).min(self.shape[a]));
                    for z in a0[0]..a1[0] {
                        for y in a0[1]..a1[1] {
                            for x in a0[2]..a1[2] {
                                if f(z, y, x) {
                                    return true;
                                }
                            }
                        }
                    }
                }
            }
        }
        false
    }

    /// Fill the bricks whose z index lies in `rows` in parallel, leaving the others as they are:
    /// `f(brick_lo, brick_dims, out)` writes one brick's voxels in C order (dims clipped at the volume edge) and
    /// returns whether any differs from the default
    ///
    /// Each row of bricks may be filled once: the stores do not take sub-bricks back, so a brick replaced by a
    /// second fill would stay allocated, uncounted by [`Vol::resident_bytes`], until the volume is dropped
    pub fn fill_brick_rows<F>(&mut self, rows: Range<usize>, f: F)
    where
        F: Fn(Shape, Shape, &mut [T]) -> bool + Sync,
    {
        let grid = self.grid;
        let shape = self.shape;
        let default = self.default;
        let plane = grid[1] * grid[2];
        let (tables, store) = (&self.tables, &self.store);
        let slab = &mut self.bricks[rows.start * plane..rows.end * plane];
        debug_assert!(slab.iter().all(|&t| t == 0), "brick rows {rows:?} are filled a second time");
        // each writer takes tables and sub-bricks from its own runs of the stores and reuses one dense buffer
        let init = || (tables.run(), store.run(), Vec::with_capacity(BRICK));
        slab.par_iter_mut().enumerate().for_each_init(init, |(table_run, run, dense), (i, slot)| {
            let bi = rows.start * plane + i;
            let bz = bi / plane;
            let by = (bi / grid[2]) % grid[1];
            let bx = bi % grid[2];
            let lo = [bz * EDGE, by * EDGE, bx * EDGE];
            let dims = [0, 1, 2].map(|a| EDGE.min(shape[a] - lo[a]));
            dense.clear();
            dense.resize(dims.iter().product(), default);
            *slot = 0;
            if !f(lo, dims, dense) {
                return;
            }
            let mut table: Option<&mut [u32]> = None;
            for s in 0..SUBS {
                let slo = [s >> 4, s >> 2 & 3, s & 3].map(|v| v << SUB_SHIFT);
                let n = [0, 1, 2].map(|a| SUB_EDGE.min(dims[a].saturating_sub(slo[a])));
                // a sub-brick past the volume's edge holds no voxel
                if n.contains(&0) {
                    continue;
                }
                let row =
                    |z: usize, y: usize| &dense[((slo[0] + z) * dims[1] + slo[1] + y) * dims[2] + slo[2]..][..n[2]];
                let any = (0..n[0]).any(|z| (0..n[1]).any(|y| row(z, y).iter().any(|&v| v != default)));
                if !any {
                    continue;
                }
                let ids = table.get_or_insert_with(|| {
                    let (id, t) = table_run.alloc();
                    *slot = id;
                    // SAFETY: the store handed this table to this writer alone, and it lives as long as the volume
                    unsafe { std::slice::from_raw_parts_mut(t.as_ptr(), SUBS) }
                });
                let (id, ptr) = run.alloc();
                ids[s] = id;
                // SAFETY: the store handed this sub-brick to this writer alone, and it lives as long as the volume
                let b = unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr(), SUB) };
                for z in 0..n[0] {
                    for y in 0..n[1] {
                        let at = Self::offset(z, y, 0);
                        b[at..at + n[2]].copy_from_slice(row(z, y));
                    }
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_reads_default_and_round_trips() {
        let mut v = Vol::new([40, 70, 33], 128u8);
        assert_eq!(v.get(39, 69, 32), 128);
        v.set(39, 69, 32, 7);
        v.set(0, 0, 0, 128);
        assert_eq!(v.get(39, 69, 32), 7);
        // one table of sub-brick ids and the one sub-brick written
        assert_eq!(v.resident_bytes(), SUBS * 4 + SUB);
        let block: Vec<u8> = (0..3 * 4 * 5).map(|i| i as u8).collect();
        v.write_block([30, 31, 28], [3, 4, 5], &block);
        assert_eq!(v.read_block([30, 31, 28], [3, 4, 5]), block);
    }

    #[test]
    fn brick_rows_fill_as_voxel_writes_do() {
        // a shape clipped on every axis, a nonzero default and a few scattered voxels, so most sub-bricks of the
        // bricks filled stay unallocated
        let shape = [37, 21, 50];
        let value =
            |z: usize, y: usize, x: usize| if (z * 7 + y * 3 + x).is_multiple_of(97) { (z + y + x) as u8 } else { 9 };
        let mut filled = Vol::new(shape, 9u8);
        let rows = filled.grid()[0];
        filled.fill_brick_rows(0..rows, |lo, dims, out| {
            let mut any = false;
            for z in 0..dims[0] {
                for y in 0..dims[1] {
                    for x in 0..dims[2] {
                        let v = value(lo[0] + z, lo[1] + y, lo[2] + x);
                        out[(z * dims[1] + y) * dims[2] + x] = v;
                        any |= v != 9;
                    }
                }
            }
            any
        });
        let mut set = Vol::new(shape, 9u8);
        let mut nonzero = Vec::new();
        for z in 0..shape[0] {
            for y in 0..shape[1] {
                for x in 0..shape[2] {
                    set.set(z, y, x, value(z, y, x));
                    if value(z, y, x) != 9 {
                        nonzero.push([z, y, x]);
                    }
                }
            }
        }
        let all = filled.read_block([0, 0, 0], shape);
        assert_eq!(all, set.read_block([0, 0, 0], shape));
        assert_eq!(filled.resident_bytes(), set.resident_bytes());
        // only the sub-bricks of the written voxels are allocated
        let subs: std::collections::HashSet<_> = nonzero.iter().map(|p| p.map(|v| v >> SUB_SHIFT)).collect();
        let bricks: std::collections::HashSet<_> = nonzero.iter().map(|p| p.map(|v| v >> SHIFT)).collect();
        assert_eq!(filled.resident_bytes(), bricks.len() * SUBS * 4 + subs.len() * SUB);
        // a cell reads the same values whether it lies in one sub-brick or spans several
        for z in 0..shape[0] - 1 {
            for y in 0..shape[1] - 1 {
                for x in 0..shape[2] - 1 {
                    let each = std::array::from_fn(|c| value(z + (c >> 2), y + (c >> 1 & 1), x + (c & 1)));
                    assert_eq!(filled.cell(z, y, x), each);
                }
            }
        }
        let [z, y, x] = nonzero[3];
        assert!(filled.any_allocated_in([z, y, x], [1, 1, 1], |_, _, _| true));
        assert!(!filled.any_allocated_in([z, y, x], [1, 1, 1], |a, b, c| value(a, b, c) == 9));
    }
}
