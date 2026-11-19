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

    /// The bytes a buffer of `bytes` occupies while it is in use: its size class when the pool serves it
    pub fn footprint(bytes: usize) -> usize {
        Layout::from_size_align(bytes, 1).ok().and_then(Self::class).unwrap_or(bytes)
    }

    /// The size class of a pooled request, or `None` for one the global allocator serves directly
    fn class(layout: Layout) -> Option<usize> {
        if layout.size() < MIN_POOLED || layout.align() > POOL_ALIGN {
            return None;
        }

        // a request just under `isize::MAX` would round up past the largest valid layout; the global allocator
        // refuses such a request itself, where an allocator may not panic
        let class = layout.size().next_multiple_of(GRANULE);
        (class <= isize::MAX as usize).then_some(class)
    }

    fn class_layout(class: usize) -> Layout {
        Layout::from_size_align(class, POOL_ALIGN).expect("`class` admits only valid layouts")
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

/// A voxel type that freshly mapped memory holds valid values of
///
/// # Safety
///
/// Every bit pattern, and in particular all zero bits, must be a valid value of the type, and all zero bits must
/// be [`Voxel::ZERO`]: a [`BrickStore`] hands out zeroed or partly written mappings as slices of the type
pub unsafe trait Voxel: Copy + PartialEq + Send + Sync + 'static {
    const ZERO: Self;
}

// SAFETY: every bit pattern is a valid `u8`, and all zero bits are 0
unsafe impl Voxel for u8 {
    const ZERO: Self = 0;
}

// SAFETY: every bit pattern is a valid `u16`, and all zero bits are 0
unsafe impl Voxel for u16 {
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

/// The unused part of the newest segment, the unused tails of finished runs, and every segment mapped so far
struct Cursor<T> {
    next: *mut T,
    left: usize,
    spare: Vec<(*mut T, usize)>,
    segments: Vec<(NonNull<u8>, usize)>,
}

// SAFETY: the cursor's pointers address the store's own mappings, which live as long as the store; the cursor is
// only reached through the store's lock or `&mut self`
unsafe impl<T: Send> Send for Cursor<T> {}

/// Bricks taken from a store for one writer, so parallel writers touch the shared cursor once per
/// `RUN_BRICKS` bricks
///
/// A run borrows its store, so its bricks cannot outlive their mapping, and gives the bricks it did not hand out
/// back to the store when it is dropped: a parallel fill starts a run for every job it splits into, and most jobs
/// over sparse rows use one brick or none
pub struct BrickRun<'s, T: Voxel> {
    store: &'s BrickStore<T>,
    next: *mut T,
    left: usize,
}

impl<T: Voxel> BrickStore<T> {
    /// Storage for bricks of `brick` voxels of `T` reading `default` until written; `max_bricks` bounds how many the
    /// volume can hold, so a small volume maps no more than it could use
    pub fn new(brick: usize, default: T, max_bricks: usize) -> Self {
        let per_segment = (SEGMENT_BYTES / (brick * size_of::<T>())).max(1);
        let cursor = Cursor { next: std::ptr::null_mut(), left: 0, spare: Vec::new(), segments: Vec::new() };
        Self { brick, default, segment_bricks: per_segment.min(max_bricks.max(1)), cursor: Mutex::new(cursor) }
    }

    /// One brick filled with the default, from a store this thread owns
    pub fn alloc(&mut self) -> NonNull<T> {
        let (brick, segment_bricks) = (self.brick, self.segment_bricks);
        let cursor = self.cursor.get_mut().expect("brick store lock");
        let (ptr, _) = cursor.take(1, brick, segment_bricks);
        self.fill(ptr)
    }

    /// An empty run for a parallel writer; [`BrickRun::alloc`] fills it on first use
    pub fn run(&self) -> BrickRun<'_, T> {
        BrickRun { store: self, next: std::ptr::null_mut(), left: 0 }
    }

    fn fill(&self, ptr: *mut T) -> NonNull<T> {
        if self.default != T::ZERO {
            // SAFETY: the brick is `brick` voxels inside a live segment, handed to this caller only
            unsafe { std::slice::from_raw_parts_mut(ptr, self.brick) }.fill(self.default);
        }
        NonNull::new(ptr).expect("segments are never at address zero")
    }
}

impl<T: Voxel> BrickRun<'_, T> {
    /// One brick filled with the default, refilling the run from its store when it runs out
    pub fn alloc(&mut self) -> NonNull<T> {
        let store = self.store;
        if self.left == 0 {
            let mut cursor = store.cursor.lock().expect("brick store lock");
            (self.next, self.left) = cursor.take(RUN_BRICKS, store.brick, store.segment_bricks);
        }

        let ptr = self.next;
        // SAFETY: the run holds `left` more bricks of `brick` voxels from `next` on, inside one segment
        self.next = unsafe { ptr.add(store.brick) };
        self.left -= 1;
        store.fill(ptr)
    }
}

impl<T: Voxel> Drop for BrickRun<'_, T> {
    fn drop(&mut self) {
        if self.left > 0 {
            self.store.cursor.lock().expect("brick store lock").spare.push((self.next, self.left));
        }
    }
}

impl<T> Cursor<T> {
    /// The start and length of up to `want` bricks, from a finished run's tail or the newest segment, mapping a new
    /// segment when both are used up
    fn take(&mut self, want: usize, brick: usize, segment_bricks: usize) -> (*mut T, usize) {
        if let Some((next, left)) = self.spare.pop() {
            let n = want.min(left);
            if n < left {
                // SAFETY: the tail holds `left` bricks from `next` on, inside one segment
                self.spare.push((unsafe { next.add(n * brick) }, left - n));
            }
            return (next, n);
        }

        if self.left == 0 {
            let bytes = segment_bricks * brick * size_of::<T>();
            let ptr = map_segment(bytes);
            self.segments.push((ptr, bytes));
            self.next = ptr.as_ptr().cast();
            self.left = segment_bricks;
        }
        let n = want.min(self.left);
        let next = self.next;
        // SAFETY: `n` bricks remain in the newest segment from `next` on
        self.next = unsafe { self.next.add(n * brick) };
        self.left -= n;
        (next, n)
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
        let (c, d) = {
            let mut run = store.run();
            (run.alloc(), run.alloc())
        };
        let brick = unsafe { std::slice::from_raw_parts(c.as_ptr(), 4096) };
        assert!(brick.iter().all(|&v| v == 128));
        // a dropped run's unused bricks go to the next run instead of being skipped
        let e = store.run().alloc();
        assert_eq!(unsafe { e.as_ptr().offset_from(d.as_ptr()) }, 4096);
        let fresh = store.run().alloc();
        assert_eq!(unsafe { fresh.as_ptr().offset_from(e.as_ptr()) }, 4096);
        // a store sized for few bricks maps no more than it can hold, and maps again when that is used up
        let mut small = BrickStore::new(4096, 0u16, 2);
        let bricks: Vec<_> = (0..5).map(|_| small.alloc()).collect();
        assert_eq!(small.cursor.lock().unwrap().segments.len(), 3);
        assert!(bricks.windows(2).all(|w| w[0] != w[1]));
    }
}
