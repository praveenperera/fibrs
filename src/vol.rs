//! Sparse in-memory volumes stored as 16³ bricks of 4³ sub-bricks
//!
//! The band's fields are mostly air: a sub-brick is allocated only once a value other than the volume's
//! default is written to it, and reads from an unallocated one return the default. This replaces the
//! reference's compressed Zarr work store and its paged tile cache with plain memory. Sub-bricks live in a
//! [`BrickStore`] owned by the volume, not one heap allocation each

use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU32, Ordering};

use rayon::prelude::*;

use crate::arena::{BrickRun, BrickStore, Voxel};

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

/// The slot of the brick holding voxel `z, y, x` in a brick grid of `grid`
#[inline(always)]
fn brick_index(grid: Shape, z: usize, y: usize, x: usize) -> usize {
    ((z >> SHIFT) * grid[1] + (y >> SHIFT)) * grid[2] + (x >> SHIFT)
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

    /// The value every voxel outside an allocated sub-brick reads as
    #[cfg_attr(not(all(target_os = "linux", target_arch = "x86_64")), allow(dead_code))]
    pub fn default_value(&self) -> T {
        self.default
    }

    #[inline(always)]
    fn brick_index(&self, z: usize, y: usize, x: usize) -> usize {
        brick_index(self.grid, z, y, x)
    }

    #[inline(always)]
    fn sub_index(z: usize, y: usize, x: usize) -> usize {
        sub_index(z, y, x)
    }

    #[inline(always)]
    fn offset(z: usize, y: usize, x: usize) -> usize {
        offset(z, y, x)
    }

    /// The origin of sub-brick `s` within its brick, in voxels
    #[inline(always)]
    fn sub_origin(s: usize) -> Shape {
        let per = EDGE >> SUB_SHIFT;
        [s / (per * per), s / per % per, s % per].map(|v| v << SUB_SHIFT)
    }

    /// The origin of brick `bi` (C order over the brick grid) and its extent, clipped at the volume's edge
    fn brick_extent(grid: Shape, shape: Shape, bi: usize) -> (Shape, Shape) {
        let lo = [bi / (grid[1] * grid[2]), bi / grid[2] % grid[1], bi % grid[2]].map(|v| v * EDGE);
        (lo, [0, 1, 2].map(|a| EDGE.min(shape[a] - lo[a])))
    }

    fn brick_dims(&self, bi: usize) -> (Shape, Shape) {
        Self::brick_extent(self.grid, self.shape, bi)
    }

    /// Write one brick whose table id is `slot` and whose extent is `dims` (clipped at the volume's edge) from
    /// `row(z, y)`, the brick's `dims[2]` voxels of row z, y: the sub-bricks it has are overwritten, and the others
    /// are taken from `store` only where a voxel differs from `default`, with the brick's table from `tables` on
    /// the first
    fn write_brick<'d>(
        slot: &mut u32,
        dims: Shape,
        default: T,
        tables: &mut BrickRun<'_, u32, SUBS>,
        store: &mut BrickRun<'_, T, SUB>,
        row: impl Fn(usize, usize) -> &'d [T],
    ) where
        T: 'd,
    {
        // SAFETY: a table the brick names is SUBS ids, and the caller is the brick's only writer
        let mut table =
            (*slot != 0).then(|| unsafe { std::slice::from_raw_parts_mut(tables.get(*slot).as_ptr(), SUBS) });
        for s in 0..SUBS {
            let slo = Self::sub_origin(s);
            let n = [0, 1, 2].map(|a| SUB_EDGE.min(dims[a].saturating_sub(slo[a])));
            // a sub-brick past the volume's edge holds no voxel
            if n.contains(&0) {
                continue;
            }
            let sub_row = |z: usize, y: usize| &row(slo[0] + z, slo[1] + y)[slo[2]..][..n[2]];
            let have = table.as_ref().map_or(0, |t| t[s]);
            let ptr = if have != 0 {
                store.get(have)
            } else {
                if !(0..n[0]).any(|z| (0..n[1]).any(|y| sub_row(z, y).iter().any(|&v| v != default))) {
                    continue;
                }
                let ids = table.get_or_insert_with(|| {
                    let (id, t) = tables.alloc();
                    *slot = id;
                    // SAFETY: the store handed this table to this writer alone, and it lives as long as the volume
                    unsafe { std::slice::from_raw_parts_mut(t.as_ptr(), SUBS) }
                });
                let (id, ptr) = store.alloc();
                ids[s] = id;
                ptr
            };
            // SAFETY: an allocated sub-brick is SUB voxels of the store, and the caller is the brick's only writer
            let b = unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr(), SUB) };
            for z in 0..n[0] {
                for y in 0..n[1] {
                    let at = Self::offset(z, y, 0);
                    b[at..at + n[2]].copy_from_slice(sub_row(z, y));
                }
            }
        }
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

    /// The values at the eight corners of the cell whose lowest corner is `z, y, x`, in z, y, x order: corner
    /// `4 * dz + 2 * dy + dx`
    ///
    /// The cell must be in bounds. The tracer's trilinear samples read cells like this, so each sub-brick the cell
    /// touches is found once: one for 42% of cells, two for another 42%, and at most eight
    #[inline(always)]
    pub fn corners(&self, z: usize, y: usize, x: usize) -> [T; 8] {
        // the axes on which the cell crosses into the next sub-brick
        let cross = [z, y, x].map(|v| (v & SUB_MASK) == SUB_MASK);
        if cross == [false; 3] {
            let (sub, at) = self.locate(z, y, x);
            let (dy, dz) = (SUB_EDGE, SUB_EDGE * SUB_EDGE);
            return [0, 1, dy, dy + 1, dz, dz + 1, dz + dy, dz + dy + 1].map(|d| sub[at + d]);
        }
        // the sub-brick of corner c is the one its offsets reach on the crossing axes only
        let reach = |c: usize| {
            let d = [c >> 2, c >> 1 & 1, c & 1];
            (d[0] & cross[0] as usize) << 2 | (d[1] & cross[1] as usize) << 1 | (d[2] & cross[2] as usize)
        };
        let mut subs = [self.sub(0); 8];
        for (k, sub) in subs.iter_mut().enumerate() {
            if reach(k) == k {
                *sub = self.locate(z + (k >> 2), y + (k >> 1 & 1), x + (k & 1)).0;
            }
        }
        std::array::from_fn(|c| subs[reach(c)][Self::offset(z + (c >> 2), y + (c >> 1 & 1), x + (c & 1))])
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

    /// Whether brick `bi` (C order over the brick grid) holds an allocated sub-brick; where it does not, every voxel
    /// reads as the default
    #[cfg_attr(not(all(target_os = "linux", target_arch = "x86_64")), allow(dead_code))]
    pub fn has_brick(&self, bi: usize) -> bool {
        self.bricks[bi] != 0
    }

    /// Copy brick `bi` (C order over the brick grid) into `out` in brick order, C order over its 16³ voxels, with
    /// voxels past the volume's edge at the default
    #[cfg_attr(not(all(target_os = "linux", target_arch = "x86_64")), allow(dead_code))]
    pub fn read_brick(&self, bi: usize, out: &mut [T; BRICK]) {
        for (s, &id) in self.table(bi).iter().enumerate() {
            let sub = self.sub(id);
            let slo = Self::sub_origin(s);
            for z in 0..SUB_EDGE {
                for y in 0..SUB_EDGE {
                    let at = ((slo[0] + z) * EDGE + slo[1] + y) * EDGE + slo[2];
                    out[at..at + SUB_EDGE].copy_from_slice(&sub[Self::offset(z, y, 0)..][..SUB_EDGE]);
                }
            }
        }
    }

    /// Replace brick `bi` with `data`, one brick's voxels in brick order; voxels past the volume's edge are ignored,
    /// and sub-bricks that would hold only the default are not allocated
    #[cfg_attr(not(all(target_os = "linux", target_arch = "x86_64")), allow(dead_code))]
    pub fn set_brick(&mut self, bi: usize, data: &[T; BRICK]) {
        let dims = self.brick_dims(bi).1;
        let (mut tables, mut store) = (self.tables.run(), self.store.run());
        Self::write_brick(&mut self.bricks[bi], dims, self.default, &mut tables, &mut store, |z, y| {
            &data[(z * EDGE + y) * EDGE..][..dims[2]]
        });
    }

    /// Fill bricks in parallel, each given as its slot (C order over the brick grid) and its voxels in brick order;
    /// sub-bricks that would hold only the default are not allocated, as in [`Vol::set_brick`]
    ///
    /// Each slot may be filled once and must still read as the default, as in [`Vol::fill_brick_rows`]. A slot given
    /// twice is a caller error: debug builds panic when both bricks hold a non-default voxel, and an all-default
    /// brick is skipped before the slot is checked
    #[cfg_attr(not(all(target_os = "linux", target_arch = "x86_64")), allow(dead_code))]
    pub fn set_bricks<'a, I>(&mut self, bricks: I)
    where
        I: IndexedParallelIterator<Item = (usize, &'a [T; BRICK])>,
    {
        let (grid, shape, default) = (self.grid, self.shape, self.default);
        let (tables, store) = (&self.tables, &self.store);
        // each writer takes tables and sub-bricks from its own runs of the stores, so the brick slots are the only
        // shared state, and they are set below on this thread
        let filled: Vec<(usize, u32)> = bricks
            .map_init(
                || (tables.run(), store.run()),
                |(tables, store), (bi, data)| {
                    let dims = Self::brick_extent(grid, shape, bi).1;
                    let mut slot = 0;
                    Self::write_brick(&mut slot, dims, default, tables, store, |z, y| {
                        &data[(z * EDGE + y) * EDGE..][..dims[2]]
                    });
                    (bi, slot)
                },
            )
            .filter(|&(_, slot)| slot != 0)
            .collect();
        for (bi, slot) in filled {
            debug_assert!(self.bricks[bi] == 0, "brick {bi} is filled a second time");
            self.bricks[bi] = slot;
        }
    }

    /// Whether any voxel of brick `bi` (C order over the brick grid) that lies in an allocated sub-brick satisfies
    /// `f`; the brick's other voxels read as the default and are not tested
    #[cfg_attr(not(all(target_os = "linux", target_arch = "x86_64")), allow(dead_code))]
    pub fn any_allocated_in_brick(&self, bi: usize, f: impl Fn(T) -> bool) -> bool {
        self.table(bi).iter().filter(|&&id| id != 0).any(|&id| self.sub(id).iter().any(|&v| f(v)))
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
        slab.par_iter_mut().enumerate().for_each_init(init, |(tables, store, dense), (i, slot)| {
            let bi = rows.start * plane + i;
            let (lo, dims) = Self::brick_extent(grid, shape, bi);
            dense.clear();
            dense.resize(dims.iter().product(), default);
            if f(lo, dims, dense) {
                Self::write_brick(slot, dims, default, tables, store, |z, y| {
                    &dense[(z * dims[1] + y) * dims[2]..][..dims[2]]
                });
            }
        });
    }
}

/// A sparse `u16` volume, zero by default, that one writer changes while other threads read it
///
/// It is laid out as [`Vol`] is, in 16³ bricks of 4³ sub-bricks, but every brick slot, table entry and voxel is
/// atomic: the writer publishes a table or sub-brick it allocates with a release store, so a reader that sees its
/// id sees its zeros, and a reader racing the writer reads either the old or the new value of a voxel, never a torn
/// one. Readers learn nothing about ordering from a voxel's value; the tracer orders them through its own commit
/// counter
pub struct SharedVol {
    shape: Shape,
    grid: Shape,
    /// Each brick's table id in `tables`, 0 while the whole brick reads as zero
    bricks: Box<[AtomicU32]>,
    /// The tables, [`SUBS`] sub-brick ids each, 0 for an unallocated sub-brick
    tables: BrickStore<u32, SUBS>,
    store: BrickStore<u16, SUB>,
    /// Whether [`SharedVol::writer`] has been called: a second writer would race the first on allocation
    has_writer: AtomicBool,
}

/// Where a voxel of a [`SharedVol`] lives: its brick slot, its sub-brick in the brick and its offset in that
///
/// Only [`SharedVol::place`] makes one, so the sub-brick and offset are always in range: reads and writes through
/// a place index the table and sub-brick without a bounds check
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Place {
    slot: usize,
    sub: u8,
    offset: u8,
}

impl Place {
    /// The brick slot, in C order over the brick grid
    #[inline(always)]
    pub fn slot(self) -> usize {
        self.slot
    }
}

impl SharedVol {
    /// An empty volume where every voxel reads as zero
    pub fn new(shape: Shape) -> Self {
        let grid = shape.map(|n| n.div_ceil(EDGE));
        let slots: usize = grid.iter().product();
        let bricks = (0..slots).map(|_| AtomicU32::new(0)).collect();
        // each store's id 0 is handed out first and never written: the empty table and the zero sub-brick. Both
        // stores' bricks start zeroed (fresh mappings), so neither writes them before they are published
        let mut tables = BrickStore::new(0, slots + 1);
        tables.alloc();
        let mut store = BrickStore::new(0, slots * SUBS + 1);
        store.alloc();
        Self { shape, grid, bricks, tables, store, has_writer: AtomicBool::new(false) }
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    /// The brick slot, sub-brick and offset of an in-bounds voxel
    #[inline(always)]
    pub fn place(&self, z: usize, y: usize, x: usize) -> Place {
        debug_assert!(z < self.shape[0] && y < self.shape[1] && x < self.shape[2]);
        Place { slot: brick_index(self.grid, z, y, x), sub: sub_index(z, y, x) as u8, offset: offset(z, y, x) as u8 }
    }

    /// The table entry of a place's sub-brick, in a published table
    #[inline(always)]
    fn entry(&self, table: u32, at: Place) -> &AtomicU32 {
        // SAFETY: a published table holds SUBS ids of `tables`, which lives as long as `self`, and is only ever
        // accessed atomically once published; `at.sub` is below SUBS
        unsafe { AtomicU32::from_ptr(self.tables.get(table).as_ptr().add(at.sub as usize)) }
    }

    /// A place's voxel, in a published sub-brick
    #[inline(always)]
    fn voxel(&self, sub: u32, at: Place) -> &AtomicU16 {
        // SAFETY: a published sub-brick holds SUB voxels of `store`, which lives as long as `self`, and is only
        // ever accessed atomically once published; `at.offset` is below SUB
        unsafe { AtomicU16::from_ptr(self.store.get(sub).as_ptr().add(at.offset as usize)) }
    }

    /// The value at a place
    #[inline(always)]
    pub fn get(&self, at: Place) -> u16 {
        let table = self.bricks[at.slot].load(Ordering::Acquire);
        if table == 0 {
            return 0;
        }
        let sub = self.entry(table, at).load(Ordering::Acquire);
        if sub == 0 {
            return 0;
        }
        self.voxel(sub, at).load(Ordering::Relaxed)
    }

    /// Bytes held by allocated sub-bricks and the tables that name them
    pub fn resident_bytes(&self) -> usize {
        let mut tables = 0;
        let mut subs = 0;
        for slot in &self.bricks {
            let table = slot.load(Ordering::Acquire);
            if table == 0 {
                continue;
            }
            tables += 1;
            let entry = |sub: usize| self.entry(table, Place { slot: 0, sub: sub as u8, offset: 0 });
            subs += (0..SUBS).filter(|&sub| entry(sub).load(Ordering::Relaxed) != 0).count();
        }
        tables * SUBS * size_of::<u32>() + subs * SUB * size_of::<u16>()
    }

    /// The volume's one writer
    ///
    /// # Panics
    ///
    /// When called a second time on the same volume
    pub fn writer(&self) -> SharedVolWriter<'_> {
        assert!(!self.has_writer.swap(true, Ordering::Relaxed), "a shared volume has one writer");
        SharedVolWriter { vol: self, tables: self.tables.run(), store: self.store.run() }
    }
}

/// The only handle that changes a [`SharedVol`]
pub struct SharedVolWriter<'v> {
    vol: &'v SharedVol,
    tables: BrickRun<'v, u32, SUBS>,
    store: BrickRun<'v, u16, SUB>,
}

impl SharedVolWriter<'_> {
    /// The value at a place; the writer always sees its own writes
    #[inline(always)]
    pub fn get(&self, at: Place) -> u16 {
        self.vol.get(at)
    }

    /// Set one voxel, allocating and publishing its brick's table and its sub-brick on the first write to them
    #[inline]
    pub fn set(&mut self, at: Place, value: u16) {
        let slot = &self.vol.bricks[at.slot];
        let mut table = slot.load(Ordering::Relaxed);
        if table == 0 {
            table = self.tables.alloc().0;
            slot.store(table, Ordering::Release);
        }
        let entry = self.vol.entry(table, at);
        let mut sub = entry.load(Ordering::Relaxed);
        if sub == 0 {
            sub = self.store.alloc().0;
            entry.store(sub, Ordering::Release);
        }
        self.vol.voxel(sub, at).store(value, Ordering::Relaxed);
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

    /// The parallel install leaves a volume as one `set_brick` per brick would: the values read back, and an
    /// all-default brick stays unallocated
    #[test]
    fn set_bricks_matches_set_brick() {
        let shape = [40, 70, 33];
        let data: Vec<[u16; BRICK]> = (0..6u16).map(|k| std::array::from_fn(|i| (i as u16 % 7) * k)).collect();
        let slots = [0, 5, 17, 23, 31, 44];
        let mut one = Vol::new(shape, 0u16);
        for (&bi, d) in slots.iter().zip(&data) {
            one.set_brick(bi, d);
        }

        let mut all = Vol::new(shape, 0u16);
        all.set_bricks(slots.par_iter().copied().zip(&data));
        assert_eq!(all.read_block([0; 3], shape), one.read_block([0; 3], shape));
        // brick 0 holds only zeros, so five of the six are allocated
        assert_eq!(slots.map(|bi| all.has_brick(bi)), [false, true, true, true, true, true]);
        assert_eq!(all.resident_bytes(), one.resident_bytes());
    }

    #[test]
    fn shared_volume_reads_its_writers_values() {
        let v = SharedVol::new([20, 33, 40]);
        let at = v.place(19, 32, 39);
        assert_eq!(v.get(at), 0);
        let mut w = v.writer();
        w.set(at, 0x8080);
        w.set(v.place(19, 32, 38), 1);
        assert_eq!((v.get(at), w.get(v.place(19, 32, 38)), v.get(v.place(0, 0, 0))), (0x8080, 1, 0));
        // both voxels share one sub-brick of one brick
        assert_eq!(v.resident_bytes(), SUBS * 4 + SUB * 2);
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
                    assert_eq!(filled.corners(z, y, x), each);
                }
            }
        }
        let [z, y, x] = nonzero[3];
        assert!(filled.any_allocated_in([z, y, x], [1, 1, 1], |_, _, _| true));
        assert!(!filled.any_allocated_in([z, y, x], [1, 1, 1], |a, b, c| value(a, b, c) == 9));

        // whole bricks in brick order, as the GPU paths move them: copied out and written into a fresh volume, they
        // read back the same voxels and allocate the same sub-bricks, and an all-default brick allocates nothing
        let mut copied = Vol::new(shape, 9u8);
        let mut brick = [0u8; BRICK];
        for bi in 0..filled.bricks.len() {
            assert_eq!(
                filled.has_brick(bi),
                bricks.contains(&Vol::<u8>::brick_extent(filled.grid, shape, bi).0.map(|v| v >> SHIFT))
            );
            brick.fill(9);
            filled.read_brick(bi, &mut brick);
            let (lo, dims) = filled.brick_dims(bi);
            for (i, &v) in brick.iter().enumerate() {
                let at = [i >> (2 * SHIFT), i >> SHIFT & (EDGE - 1), i & (EDGE - 1)];
                let inside = (0..3).all(|a| at[a] < dims[a]);
                assert_eq!(v, if inside { value(lo[0] + at[0], lo[1] + at[1], lo[2] + at[2]) } else { 9 });
            }
            copied.set_brick(bi, &brick);
        }
        assert_eq!(copied.read_block([0, 0, 0], shape), all);
        assert_eq!(copied.resident_bytes(), filled.resident_bytes());
        // writing a brick again overwrites its sub-bricks in place
        let bi = filled.brick_index(z, y, x);
        filled.read_brick(bi, &mut brick);
        brick.iter_mut().filter(|v| **v != 9).for_each(|v| *v = 200);
        copied.set_brick(bi, &brick);
        assert_eq!(copied.get(z, y, x), 200);
        assert_eq!(copied.resident_bytes(), filled.resident_bytes());
    }
}
