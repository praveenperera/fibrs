//! Memory reuse for the tracer's large and numerous allocations
//!
//! glibc serves each block of 1 MiB or more with its own mapping (see `run::fix_mmap_threshold`), so every native
//! chunk decoded and every direction-field tile computed costs an `mmap`, a `munmap` and a page fault for each of
//! its pages, and it grows its per-thread heaps a few pages at a time with `mprotect` as bricks are added. Each of
//! those calls takes the process's `mmap_lock` for writing, and every page fault on every thread waits for it: on
//! the dense PHerc0358 windows the tracer spent most of its time there, at about half a core. Two owners take
//! these allocations off the system calls:
//!
//! - [`BufferPool`], an [`Allocator`] that keeps freed large buffers by size and hands them out again, for the
//!   decoded native chunks and the direction field's tile buffers
//! - [`BrickStore`], the bump storage behind a [`Vol`](crate::vol::Vol)'s bricks: whole segments mapped at once and
//!   carved into bricks in allocation order, freed together with the volume

use std::alloc::{AllocError, Allocator, Global, Layout, handle_alloc_error};
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::Mutex;

/// Requests below this size go straight to the global allocator: they come from its per-thread caches and never
/// map memory of their own
const MIN_POOLED: usize = 64 << 10;
/// Pooled buffers are kept in size classes this far apart, so buffers of nearly equal size (compressed chunks,
/// clipped tiles) share a class
const GRANULE: usize = 64 << 10;
/// Alignment of every pooled buffer; larger alignments go to the global allocator
const POOL_ALIGN: usize = 64;

/// An allocator that keeps large freed buffers and serves later requests of the same size class from them
///
/// Collections take it by reference (`Vec::with_capacity_in(n, &pool)`), so a buffer goes back to the pool when it
/// is dropped, on whichever thread drops it, and the borrow checker keeps every such buffer from outliving the
/// pool. Freed buffers are kept up to `retain` bytes; the rest, and everything kept, go back to the global
/// allocator when the pool is dropped
pub struct BufferPool {
    free: Mutex<FreeLists>,
    retain: usize,
}

#[derive(Default)]
struct FreeLists {
    by_class: HashMap<usize, Vec<Block>>,
    bytes: usize,
}

/// A pooled buffer's address; the pool hands it to one owner at a time
struct Block(NonNull<u8>);

// SAFETY: a kept block is plain memory owned by the pool, and only the pool's lock reaches it
unsafe impl Send for Block {}

impl BufferPool {
    /// A pool that keeps at most `retain` bytes of freed buffers
    pub fn new(retain: usize) -> Self {
        Self { free: Mutex::default(), retain }
    }

    /// The size class of a pooled request, or `None` for one the global allocator serves directly
    fn class(layout: Layout) -> Option<usize> {
        (layout.size() >= MIN_POOLED && layout.align() <= POOL_ALIGN).then(|| layout.size().next_multiple_of(GRANULE))
    }

    fn class_layout(class: usize) -> Layout {
        Layout::from_size_align(class, POOL_ALIGN).expect("a size class is a valid layout")
    }
}

// SAFETY: a pooled block is allocated with its class's layout, which is at least as large and as aligned as every
// request mapped to that class, and it is handed to one owner at a time; `deallocate` receives the request's own
// layout (the pool returns exactly the requested length), which maps to the same class
unsafe impl Allocator for BufferPool {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let Some(class) = Self::class(layout) else { return Global.allocate(layout) };
        let reused = {
            let mut free = self.free.lock().expect("pool lock");
            let block = free.by_class.get_mut(&class).and_then(Vec::pop);
            if block.is_some() {
                free.bytes -= class;
            }
            block
        };
        let ptr = match reused {
            Some(Block(ptr)) => ptr,
            None => Global.allocate(Self::class_layout(class))?.cast(),
        };
        Ok(NonNull::slice_from_raw_parts(ptr, layout.size()))
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        let Some(class) = Self::class(layout) else {
            // SAFETY: requests outside the pool's classes were served by `Global` with this layout
            return unsafe { Global.deallocate(ptr, layout) };
        };
        {
            let mut free = self.free.lock().expect("pool lock");
            if free.bytes + class <= self.retain {
                free.bytes += class;
                free.by_class.entry(class).or_default().push(Block(ptr));
                return;
            }
        }
        // SAFETY: every pooled block was allocated by `Global` with its class's layout
        unsafe { Global.deallocate(ptr, Self::class_layout(class)) }
    }
}

impl Drop for BufferPool {
    fn drop(&mut self) {
        let free = std::mem::take(self.free.get_mut().expect("pool lock"));
        for (class, blocks) in free.by_class {
            for Block(ptr) in blocks {
                // SAFETY: kept blocks are owned by the pool alone and were allocated with their class's layout
                unsafe { Global.deallocate(ptr, Self::class_layout(class)) }
            }
        }
    }
}

/// A voxel type whose all-zero bit pattern is [`Voxel::ZERO`], so freshly mapped memory reads as that value
pub trait Voxel: Copy + PartialEq + Send + Sync + 'static {
    const ZERO: Self;
}

impl Voxel for u8 {
    const ZERO: Self = 0;
}

impl Voxel for u16 {
    const ZERO: Self = 0;
}

/// Bytes mapped per segment: large enough that the mappings of a dense window number in the hundreds, small
/// enough that the unused tail of the last segment, which is never touched and so never resident, stays virtual
const SEGMENT_BYTES: usize = 64 << 20;
/// Bricks a parallel writer takes from the shared cursor at a time
const RUN_BRICKS: usize = 64;

/// Bump storage for the bricks of one volume
///
/// Segments are anonymous mappings made directly, not through the global allocator, so their pages are zero and
/// untouched until a brick is written, whatever allocator the process runs on. A brick of a zero default is
/// handed out as it is; any other default is written into the brick when it is handed out, so only bricks in use
/// are ever resident. Bricks are never freed one by one: the segments are unmapped when the store is dropped
pub struct BrickStore<T: Voxel> {
    brick: usize,
    default: T,
    segment_bricks: usize,
    cursor: Mutex<Cursor<T>>,
}

/// The unused part of the newest segment, and every segment mapped so far
struct Cursor<T> {
    next: *mut T,
    left: usize,
    segments: Vec<(NonNull<u8>, usize)>,
}

// SAFETY: the cursor's pointers address the store's own mappings, which live as long as the store; the cursor is
// only reached through the store's lock or `&mut self`
unsafe impl<T: Send> Send for Cursor<T> {}

/// Bricks taken from a store for one writer, so parallel writers touch the shared cursor once per
/// `RUN_BRICKS` bricks
pub struct BrickRun<T> {
    next: *mut T,
    left: usize,
}

impl<T: Voxel> BrickStore<T> {
    /// Storage for bricks of `brick` voxels of `T` reading `default` until written; `max_bricks` bounds how many the
    /// volume can hold, so a small volume maps no more than it could use
    pub fn new(brick: usize, default: T, max_bricks: usize) -> Self {
        let per_segment = (SEGMENT_BYTES / (brick * size_of::<T>())).max(1);
        let cursor = Cursor { next: std::ptr::null_mut(), left: 0, segments: Vec::new() };
        Self { brick, default, segment_bricks: per_segment.min(max_bricks.max(1)), cursor: Mutex::new(cursor) }
    }

    /// One brick filled with the default, from a store this thread owns
    pub fn alloc(&mut self) -> NonNull<T> {
        let (brick, segment_bricks) = (self.brick, self.segment_bricks);
        let cursor = self.cursor.get_mut().expect("brick store lock");
        let ptr = cursor.take(1, brick, segment_bricks).next;
        self.fill(ptr)
    }

    /// An empty run for a parallel writer; [`BrickStore::alloc_in`] fills it on first use
    pub fn run(&self) -> BrickRun<T> {
        BrickRun { next: std::ptr::null_mut(), left: 0 }
    }

    /// One brick filled with the default, from `run`, refilled from the shared cursor when it runs out
    pub fn alloc_in(&self, run: &mut BrickRun<T>) -> NonNull<T> {
        if run.left == 0 {
            *run = self.cursor.lock().expect("brick store lock").take(RUN_BRICKS, self.brick, self.segment_bricks);
        }
        let ptr = run.next;
        // SAFETY: the run holds `left` more bricks of `brick` voxels from `next` on, inside one segment
        run.next = unsafe { ptr.add(self.brick) };
        run.left -= 1;
        self.fill(ptr)
    }

    fn fill(&self, ptr: *mut T) -> NonNull<T> {
        if self.default != T::ZERO {
            // SAFETY: the brick is `brick` voxels inside a live segment, handed to this caller only
            unsafe { std::slice::from_raw_parts_mut(ptr, self.brick) }.fill(self.default);
        }
        NonNull::new(ptr).expect("segments are never at address zero")
    }
}

impl<T> Cursor<T> {
    /// Up to `want` bricks from the newest segment, mapping a new one when it is used up
    fn take(&mut self, want: usize, brick: usize, segment_bricks: usize) -> BrickRun<T> {
        if self.left == 0 {
            let bytes = segment_bricks * brick * size_of::<T>();
            let ptr = map_segment(bytes);
            self.segments.push((ptr, bytes));
            self.next = ptr.as_ptr().cast();
            self.left = segment_bricks;
        }
        let n = want.min(self.left);
        let run = BrickRun { next: self.next, left: n };
        // SAFETY: `n` bricks remain in the newest segment from `next` on
        self.next = unsafe { self.next.add(n * brick) };
        self.left -= n;
        run
    }
}

impl<T: Voxel> Drop for BrickStore<T> {
    fn drop(&mut self) {
        let cursor = self.cursor.get_mut().expect("brick store lock");
        for &(ptr, bytes) in &cursor.segments {
            // SAFETY: each segment was mapped by `map_segment` with this length, and no brick outlives the store
            unsafe { libc::munmap(ptr.as_ptr().cast(), bytes) };
        }
    }
}

/// A private anonymous mapping of `bytes`, zero until written
fn map_segment(bytes: usize) -> NonNull<u8> {
    // SAFETY: a fresh private anonymous mapping aliases nothing
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            bytes,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if ptr == libc::MAP_FAILED {
        handle_alloc_error(Layout::from_size_align(bytes, 4096).expect("a segment is a valid layout"));
    }
    NonNull::new(ptr.cast()).expect("mmap never returns address zero on success")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_reuses_a_freed_buffer_of_the_same_class() {
        let pool = BufferPool::new(64 << 20);
        let first = {
            let mut a: Vec<u8, &BufferPool> = Vec::with_capacity_in(1 << 20, &pool);
            a.resize(1 << 20, 7);
            a.as_ptr()
        };
        // a slightly smaller request falls in the same 64 KiB class and gets the same buffer back
        let b: Vec<u32, &BufferPool> = Vec::with_capacity_in(((1 << 20) - 100) / 4, &pool);
        assert_eq!(b.as_ptr().cast::<u8>(), first);
        let small: Vec<u8, &BufferPool> = Vec::with_capacity_in(100, &pool);
        assert_ne!(small.as_ptr(), first);
    }

    #[test]
    fn pool_keeps_no_more_than_retain() {
        let pool = BufferPool::new(1 << 20);
        let bufs: Vec<Vec<u8, &BufferPool>> = (0..3).map(|_| Vec::with_capacity_in(1 << 20, &pool)).collect();
        drop(bufs);
        assert_eq!(pool.free.lock().unwrap().bytes, 1 << 20);
    }

    #[test]
    fn bricks_are_distinct_and_hold_the_default() {
        let mut store = BrickStore::new(4096, 128u8, 10_000);
        let a = store.alloc();
        let b = store.alloc();
        assert_eq!(unsafe { b.as_ptr().offset_from(a.as_ptr()) }, 4096);
        let mut run = store.run();
        let c = store.alloc_in(&mut run);
        let brick = unsafe { std::slice::from_raw_parts(c.as_ptr(), 4096) };
        assert!(brick.iter().all(|&v| v == 128));
        // a store sized for few bricks maps no more than it can hold, and maps again when that is used up
        let mut small = BrickStore::new(4096, 0u16, 2);
        let bricks: Vec<_> = (0..5).map(|_| small.alloc()).collect();
        assert_eq!(small.cursor.lock().unwrap().segments.len(), 3);
        assert!(bricks.windows(2).all(|w| w[0] != w[1]));
    }
}
