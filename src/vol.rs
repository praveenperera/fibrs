//! Sparse in-memory volumes stored as 16³ bricks
//!
//! The band's fields are mostly air: a brick is allocated only once a value other than the volume's
//! default is written to it, and reads from an unallocated brick return the default. This replaces the
//! reference's compressed Zarr work store and its paged tile cache with plain memory. Bricks live in a
//! [`BrickStore`] owned by the volume, not one heap allocation each

use std::ops::Range;
use std::ptr::NonNull;

use rayon::prelude::*;

use crate::arena::{BrickStore, Voxel};

/// Brick edge, as a shift
///
/// A brick is allocated whole for its first non-default voxel, so smaller bricks hold less air where data ends:
/// 16³ took 0.5 GiB (4%) less peak memory than 32³ on a 1024-plane PHerc0826 window, at the same speed, while
/// the table of brick slots (8 bytes each) stays near 17 MB for that window
pub const SHIFT: usize = 4;
/// Brick edge in voxels
pub const EDGE: usize = 1 << SHIFT;
const MASK: usize = EDGE - 1;
/// Voxels per brick
pub const BRICK: usize = EDGE * EDGE * EDGE;

/// A z, y, x shape
pub type Shape = [usize; 3];

/// A sparse 3D volume of `T` in z, y, x order
pub struct Vol<T: Voxel> {
    shape: Shape,
    grid: Shape,
    default: T,
    /// Each brick slot's storage in `store`, or `None` while every voxel of the brick reads as the default; a
    /// slot is one pointer wide, as a boxed brick was
    bricks: Vec<Option<Brick<T>>>,
    store: BrickStore<T>,
}

/// An allocated brick's first voxel in the volume's store
#[derive(Clone, Copy)]
struct Brick<T>(NonNull<T>);

// SAFETY: a brick pointer is only dereferenced through the volume that owns its store, under the volume's own
// sharing rules below
unsafe impl<T: Send> Send for Brick<T> {}
unsafe impl<T: Sync> Sync for Brick<T> {}

// SAFETY: the brick pointers address `store`, which the volume owns and frees only when it is dropped. Bricks are
// written only through `&mut self` (each slot by one writer in `fill_brick_rows`) and read through `&self`, so
// the volume follows the same sharing rules as the `Box`ed bricks it replaced
unsafe impl<T: Voxel> Send for Vol<T> {}
unsafe impl<T: Voxel> Sync for Vol<T> {}

impl<T: Voxel> Vol<T> {
    /// An empty volume where every voxel reads as `default`
    pub fn new(shape: Shape, default: T) -> Self {
        let grid = shape.map(|n| n.div_ceil(EDGE));
        let slots = grid.iter().product();
        let bricks = vec![None; slots];
        Self { shape, grid, default, bricks, store: BrickStore::new(BRICK, default, slots) }
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    /// Brick grid size in z, y, x
    pub fn grid(&self) -> Shape {
        self.grid
    }

    /// The voxels of an allocated brick
    #[inline(always)]
    fn brick(&self, b: Brick<T>) -> &[T] {
        // SAFETY: an allocated slot points at BRICK voxels of `store`, which lives as long as `self`
        unsafe { std::slice::from_raw_parts(b.0.as_ptr(), BRICK) }
    }

    /// The voxels of an allocated brick, for writing
    #[inline(always)]
    fn brick_mut(&mut self, b: Brick<T>) -> &mut [T] {
        // SAFETY: as in `brick`, and `&mut self` makes this the only access to the brick
        unsafe { std::slice::from_raw_parts_mut(b.0.as_ptr(), BRICK) }
    }

    #[inline(always)]
    fn brick_index(&self, z: usize, y: usize, x: usize) -> usize {
        ((z >> SHIFT) * self.grid[1] + (y >> SHIFT)) * self.grid[2] + (x >> SHIFT)
    }

    #[inline(always)]
    fn offset(z: usize, y: usize, x: usize) -> usize {
        ((z & MASK) << (2 * SHIFT)) | ((y & MASK) << SHIFT) | (x & MASK)
    }

    /// The value at an in-bounds voxel
    #[inline(always)]
    pub fn get(&self, z: usize, y: usize, x: usize) -> T {
        debug_assert!(z < self.shape[0] && y < self.shape[1] && x < self.shape[2]);
        match self.bricks[self.brick_index(z, y, x)] {
            Some(b) => self.brick(b)[Self::offset(z, y, x)],
            None => self.default,
        }
    }

    /// The allocated brick holding an in-bounds voxel, or None where every voxel of the brick reads as the
    /// default, and the voxel's offset in it
    #[inline(always)]
    pub fn locate(&self, z: usize, y: usize, x: usize) -> (Option<&[T]>, usize) {
        debug_assert!(z < self.shape[0] && y < self.shape[1] && x < self.shape[2]);
        (self.bricks[self.brick_index(z, y, x)].map(|b| self.brick(b)), Self::offset(z, y, x))
    }

    /// The value at a voxel given as signed coordinates, which must be in bounds
    #[inline(always)]
    pub fn at(&self, z: i64, y: i64, x: i64) -> T {
        self.get(z as usize, y as usize, x as usize)
    }

    /// Set one voxel, allocating its brick on first non-default write
    #[inline]
    pub fn set(&mut self, z: usize, y: usize, x: usize, value: T) {
        let bi = self.brick_index(z, y, x);
        let b = match self.bricks[bi] {
            Some(b) => b,
            None if value == self.default => return,
            None => {
                let b = Brick(self.store.alloc());
                self.bricks[bi] = Some(b);
                b
            }
        };
        self.brick_mut(b)[Self::offset(z, y, x)] = value;
    }

    /// Bytes held by allocated bricks
    pub fn resident_bytes(&self) -> usize {
        self.bricks.iter().filter(|b| b.is_some()).count() * BRICK * std::mem::size_of::<T>()
    }

    /// Copy a dense C-order block with origin `lo` and extent `dims` into the volume; bricks that would
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
                let row = &mut out[(z * dims[1] + y) * dims[2]..][..dims[2]];
                for (x, v) in row.iter_mut().enumerate() {
                    *v = self.get(lo[0] + z, lo[1] + y, lo[2] + x);
                }
            }
        }
        out
    }

    /// Whether `f(z, y, x)` holds at any voxel of the block `[lo, lo + dims)` that lies in an allocated brick
    pub fn any_allocated_in(&self, lo: Shape, dims: Shape, f: impl Fn(usize, usize, usize) -> bool) -> bool {
        let first = lo.map(|v| v >> SHIFT);
        let last = [0, 1, 2].map(|a| (lo[a] + dims[a] - 1) >> SHIFT);
        for bz in first[0]..=last[0] {
            for by in first[1]..=last[1] {
                for bx in first[2]..=last[2] {
                    if self.bricks[(bz * self.grid[1] + by) * self.grid[2] + bx].is_none() {
                        continue;
                    }
                    let blo = [bz * EDGE, by * EDGE, bx * EDGE];
                    let a0 = [0, 1, 2].map(|a| blo[a].max(lo[a]));
                    let a1 = [0, 1, 2].map(|a| (blo[a] + EDGE).min(lo[a] + dims[a]).min(self.shape[a]));
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
    pub fn fill_brick_rows<F>(&mut self, rows: Range<usize>, f: F)
    where
        F: Fn(Shape, Shape, &mut [T]) -> bool + Sync,
    {
        let grid = self.grid;
        let shape = self.shape;
        let default = self.default;
        let plane = grid[1] * grid[2];
        let store = &self.store;
        let slab = &mut self.bricks[rows.start * plane..rows.end * plane];
        // each writer takes bricks from its own run of the store and reuses one dense buffer
        let init = || (store.run(), Vec::with_capacity(BRICK));
        slab.par_iter_mut().enumerate().for_each_init(init, |(run, dense), (i, slot)| {
            let bi = rows.start * plane + i;
            let bz = bi / plane;
            let by = (bi / grid[2]) % grid[1];
            let bx = bi % grid[2];
            let lo = [bz * EDGE, by * EDGE, bx * EDGE];
            let dims = [0, 1, 2].map(|a| EDGE.min(shape[a] - lo[a]));
            dense.clear();
            dense.resize(dims.iter().product(), default);
            if !f(lo, dims, dense) {
                *slot = None;
                return;
            }
            let ptr = store.alloc_in(run);
            // SAFETY: the store handed this brick to this writer alone, and it lives as long as the volume
            let b = unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr(), BRICK) };
            for z in 0..dims[0] {
                for y in 0..dims[1] {
                    let src = &dense[(z * dims[1] + y) * dims[2]..][..dims[2]];
                    let at = Self::offset(z, y, 0);
                    b[at..at + dims[2]].copy_from_slice(src);
                }
            }
            *slot = Some(Brick(ptr));
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
        assert_eq!(v.resident_bytes(), BRICK);
        let block: Vec<u8> = (0..3 * 4 * 5).map(|i| i as u8).collect();
        v.write_block([30, 31, 28], [3, 4, 5], &block);
        assert_eq!(v.read_block([30, 31, 28], [3, 4, 5]), block);
    }
}
