//! Memory reuse for the tracer's large and numerous allocations
//!
//! glibc serves each block of 1 MiB or more with its own mapping (see `run::fix_malloc_thresholds`), so every native
//! chunk decoded and every direction-field tile computed costs an `mmap`, a `munmap` and a page fault for each of
//! its pages, and it grows its per-thread heaps a few pages at a time with `mprotect` as bricks are added. Each of
//! those calls takes the process's `mmap_lock` for writing, and every page fault on every thread waits for it: on
//! a dense PHerc0358 crop 40% of the CPU time went to the kernel and a run took twice as long. Two owners take
//! these allocations off the system calls:
//!
//! - [`BufferPool`], an [`Allocator`] that keeps freed large buffers by size and hands them out again, for the
//!   native predictions (decoded slabs and compressed chunks) and the direction field's tile buffers
//! - [`BrickStore`], the bump storage behind a [`Vol`](crate::vol::Vol)'s bricks: one reserved range, opened a segment at a time and
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
/// pool. Freed buffers are kept up to `retain` bytes, and a request no kept buffer fits releases kept buffers of
/// other sizes until its own would fit, so sizes no longer asked for do not hold the pool. A request whose rounded
/// size class is larger than `retain` releases nothing, since its buffer is never kept. The rest, and everything
/// kept, go back to the global allocator when the pool is dropped
pub struct BufferPool {
    free: Mutex<FreeLists>,
    retain: usize,
}

#[derive(Default)]
struct FreeLists {
    by_class: HashMap<usize, Vec<Block>>,
    bytes: usize,
}

impl FreeLists {
    /// Take kept blocks out until a block of `class` would fit under `retain` once freed; the caller frees them
    /// outside the lock. Called when no block of `class` is kept, so every block taken is of another size
    fn release_for(&mut self, class: usize, retain: usize) -> Vec<(usize, Block)> {
        let mut out = Vec::new();
        while self.bytes + class > retain {
            let Some((&c, blocks)) = self.by_class.iter_mut().find(|(_, b)| !b.is_empty()) else { break };
            let block = blocks.pop().expect("a non-empty list");
            self.bytes -= c;
            out.push((c, block));
        }
        self.by_class.retain(|_, b| !b.is_empty());
        out
    }
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
        let (reused, stale) = {
            let mut free = self.free.lock().expect("pool lock");
            match free.by_class.get_mut(&class).and_then(Vec::pop) {
                Some(block) => {
                    free.bytes -= class;
                    (Some(block), Vec::new())
                }
                // a buffer larger than the pool keeps can never be kept, so it makes no room
                None if class > self.retain => (None, Vec::new()),
                None => (None, free.release_for(class, self.retain)),
            }
        };
        for (class, Block(ptr)) in stale {
            // SAFETY: kept blocks are owned by the pool alone and were allocated with their class's layout
            unsafe { Global.deallocate(ptr, Self::class_layout(class)) }
        }
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

// SAFETY: every bit pattern is a valid `u32`, and all zero bits are 0; a volume's tables of sub-brick ids are stored
// as bricks of `u32`
unsafe impl Voxel for u32 {
    const ZERO: Self = 0;
}

/// Bytes opened for writing at a time: large enough that a dense window opens its stores' range in a few hundred
/// steps, small enough that the unused tail of the last step, which is never touched and so never resident, stays
/// virtual
const SEGMENT_BYTES: usize = 64 << 20;
/// Bricks a parallel writer takes from the shared cursor at a time
const RUN_BRICKS: usize = 64;
/// Runs a store makes room for at once beyond its bricks: far more than the writers of one parallel fill
const RUN_SLACK: usize = 4096;

/// Bump storage for the bricks of one volume, each named by a 32-bit id
///
/// The store reserves the address range of every brick it can hold up front, inaccessible, and opens it for
/// writing a segment at a time as bricks are handed out. The range is an anonymous mapping made directly, not
/// through the global allocator, so its pages are zero and untouched until a brick is written, whatever allocator
/// the process runs on, and only the opened segments count against the system's commit limit, as separate mappings
/// would. A brick of a zero default is handed out as it is; any other default is written into the brick when it is
/// handed out, so only bricks in use are ever resident. Bricks are never freed one by one: the range is unmapped
/// when the store is dropped.
///
/// A brick's address is the range's start plus its id times the brick size, so ids, which take half the room of
/// pointers in the tables that hold them, cost no lookup to follow. Id 0 is the first brick handed out
pub struct BrickStore<T: Voxel, const N: usize> {
    default: T,
    /// The reserved range: room for `capacity` bricks
    base: NonNull<T>,
    capacity: usize,
    /// Bricks opened for writing at a time
    segment: usize,
    cursor: Mutex<Cursor>,
}

// SAFETY: the store owns its range; bricks are handed to one writer each, and the cursor is behind its lock
unsafe impl<T: Voxel, const N: usize> Send for BrickStore<T, N> {}
unsafe impl<T: Voxel, const N: usize> Sync for BrickStore<T, N> {}

/// The unused ids of the newest segment, the unused tails of finished runs, and the bricks opened so far
struct Cursor {
    next: usize,
    left: usize,
    spare: Vec<(usize, usize)>,
    opened: usize,
}

/// Bricks taken from a store for one writer, so parallel writers touch the shared cursor once per
/// `RUN_BRICKS` bricks
///
/// A run borrows its store, so its bricks cannot outlive their mapping, and gives the bricks it did not hand out
/// back to the store when it is dropped: a parallel fill starts a run for every job it splits into, and most jobs
/// over sparse rows use one brick or none
pub struct BrickRun<'s, T: Voxel, const N: usize> {
    store: &'s BrickStore<T, N>,
    next: usize,
    left: usize,
}

impl<T: Voxel, const N: usize> BrickStore<T, N> {
    /// Storage for up to `max_bricks` bricks of `N` voxels of `T` reading `default` until written
    ///
    /// # Panics
    ///
    /// When `max_bricks` ids do not fit in 32 bits. Taking more bricks than the store was sized for panics too
    pub fn new(default: T, max_bricks: usize) -> Self {
        let bytes = N * size_of::<T>();
        // runs in flight hold up to RUN_BRICKS ids each that they may not use, so the range has room for that many
        // more bricks than the volume can hold, for up to RUN_SLACK runs at once
        let capacity = max_bricks.max(1) + RUN_SLACK * RUN_BRICKS;
        assert!(capacity <= 1 << 32, "{max_bricks} bricks do not fit 32-bit ids");
        let segment = (SEGMENT_BYTES / bytes).clamp(1, capacity);
        let base = reserve(capacity * bytes).cast();
        let cursor = Cursor { next: 0, left: 0, spare: Vec::new(), opened: 0 };
        Self { default, base, capacity, segment, cursor: Mutex::new(cursor) }
    }

    /// The first voxel of brick `id`, which the store has handed out
    #[inline(always)]
    pub fn get(&self, id: u32) -> NonNull<T> {
        debug_assert!((id as usize) < self.cursor.lock().map_or(usize::MAX, |c| c.opened), "brick {id} is not open");
        // SAFETY: a handed-out id lies in the opened part of the range
        unsafe { self.base.add(id as usize * N) }
    }

    /// One brick filled with the default, from a store this thread owns
    pub fn alloc(&mut self) -> (u32, NonNull<T>) {
        let (base, segment, capacity) = (self.base, self.segment, self.capacity);
        let cursor = self.cursor.get_mut().expect("brick store lock");
        let (id, _) = cursor.take(1, base.cast(), N * size_of::<T>(), segment, capacity);
        (id as u32, self.fill(id))
    }

    /// An empty run for a parallel writer; [`BrickRun::alloc`] fills it on first use
    pub fn run(&self) -> BrickRun<'_, T, N> {
        BrickRun { store: self, next: 0, left: 0 }
    }

    /// Up to `want` bricks from the shared cursor
    fn take(&self, want: usize) -> (usize, usize) {
        let mut cursor = self.cursor.lock().expect("brick store lock");
        cursor.take(want, self.base.cast(), N * size_of::<T>(), self.segment, self.capacity)
    }

    fn fill(&self, id: usize) -> NonNull<T> {
        let ptr = self.get(id as u32);
        if self.default != T::ZERO {
            // SAFETY: the brick is `N` voxels in the opened range, handed to this caller only
            unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr(), N) }.fill(self.default);
        }
        ptr
    }
}

impl Cursor {
    /// The first id and the length of up to `want` bricks, from a finished run's tail or the newest segment,
    /// opening the next segment of the range when both are used up
    fn take(
        &mut self,
        want: usize,
        base: NonNull<u8>,
        bytes: usize,
        segment: usize,
        capacity: usize,
    ) -> (usize, usize) {
        if let Some((next, left)) = self.spare.pop() {
            let n = want.min(left);
            if n < left {
                self.spare.push((next + n, left - n));
            }
            return (next, n);
        }

        if self.left == 0 {
            assert!(self.opened < capacity, "brick store is full");
            let n = segment.min(capacity - self.opened);
            // SAFETY: the bricks from `opened` on lie in the reserved range
            open(unsafe { base.add(self.opened * bytes) }, n * bytes);
            self.next = self.opened;
            self.left = n;
            self.opened += n;
        }
        let n = want.min(self.left);
        let next = self.next;
        self.next += n;
        self.left -= n;
        (next, n)
    }
}

impl<T: Voxel, const N: usize> BrickRun<'_, T, N> {
    /// One brick filled with the default and its id, refilling the run from its store when it runs out
    pub fn alloc(&mut self) -> (u32, NonNull<T>) {
        if self.left == 0 {
            (self.next, self.left) = self.store.take(RUN_BRICKS);
        }

        let id = self.next;
        self.next += 1;
        self.left -= 1;
        (id as u32, self.store.fill(id))
    }

    /// The address of a brick its store handed out, as [`BrickStore::get`]
    #[inline(always)]
    pub fn get(&self, id: u32) -> NonNull<T> {
        self.store.get(id)
    }
}

impl<T: Voxel, const N: usize> Drop for BrickRun<'_, T, N> {
    fn drop(&mut self) {
        if self.left > 0 {
            self.store.cursor.lock().expect("brick store lock").spare.push((self.next, self.left));
        }
    }
}

impl<T: Voxel, const N: usize> Drop for BrickStore<T, N> {
    fn drop(&mut self) {
        // SAFETY: the range was reserved by `reserve` with this length, and no brick outlives the store
        unsafe { libc::munmap(self.base.as_ptr().cast(), self.capacity * N * size_of::<T>()) };
    }
}

/// An inaccessible private anonymous range of `bytes`, which costs address space only
fn reserve(bytes: usize) -> NonNull<u8> {
    // SAFETY: a fresh private anonymous mapping aliases nothing
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            bytes,
            libc::PROT_NONE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
            -1,
            0,
        )
    };
    if ptr == libc::MAP_FAILED {
        handle_alloc_error(Layout::from_size_align(bytes, 4096).expect("a range is a valid layout"));
    }
    NonNull::new(ptr.cast()).expect("mmap never returns address zero on success")
}

/// Open `bytes` of a reserved range for reading and writing; its pages stay zero until written
fn open(at: NonNull<u8>, bytes: usize) {
    // SAFETY: the caller passes a part of a range `reserve` made, which nothing else maps
    if unsafe { libc::mprotect(at.as_ptr().cast(), bytes, libc::PROT_READ | libc::PROT_WRITE) } != 0 {
        handle_alloc_error(Layout::from_size_align(bytes, 4096).expect("a segment is a valid layout"));
    }
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
    fn a_new_size_releases_kept_buffers_of_other_sizes() {
        // two 1 MiB buffers nearly fill a 2.5 MiB pool; a 1.5 MiB request then releases one of them to make room
        // for its own buffer, which the pool keeps when it is freed
        let pool = BufferPool::new(5 << 19);
        let bufs: Vec<Vec<u8, &BufferPool>> = (0..2).map(|_| Vec::with_capacity_in(1 << 20, &pool)).collect();
        drop(bufs);
        let big: Vec<u8, &BufferPool> = Vec::with_capacity_in(3 << 19, &pool);
        let at = big.as_ptr();
        drop(big);
        let free = pool.free.lock().unwrap();
        assert_eq!(free.bytes, (1 << 20) + (3 << 19));
        assert_eq!(free.by_class[&(1 << 20)].len(), 1);
        assert_eq!(free.by_class[&(3 << 19)].len(), 1);
        drop(free);
        let again: Vec<u8, &BufferPool> = Vec::with_capacity_in(3 << 19, &pool);
        assert_eq!(again.as_ptr(), at);
    }

    #[test]
    fn a_buffer_larger_than_the_pool_releases_nothing() {
        let pool = BufferPool::new(5 << 19);
        let bufs: Vec<Vec<u8, &BufferPool>> = (0..2).map(|_| Vec::with_capacity_in(1 << 20, &pool)).collect();
        drop(bufs);
        let huge: Vec<u8, &BufferPool> = Vec::with_capacity_in(4 << 20, &pool);
        assert_eq!(pool.free.lock().unwrap().bytes, 2 << 20);
        drop(huge);
        assert_eq!(pool.free.lock().unwrap().bytes, 2 << 20);
    }

    #[test]
    fn bricks_are_distinct_and_hold_the_default() {
        let mut store = BrickStore::<u8, 4096>::new(128, 10_000);
        let (ia, a) = store.alloc();
        let (ib, b) = store.alloc();
        assert_eq!((ia, ib), (0, 1));
        assert_eq!(unsafe { b.as_ptr().offset_from(a.as_ptr()) }, 4096);
        assert_eq!(store.get(ib), b);
        let ((ic, c), (_, d)) = {
            let mut run = store.run();
            (run.alloc(), run.alloc())
        };
        assert_eq!(store.get(ic), c);
        let brick = unsafe { std::slice::from_raw_parts(c.as_ptr(), 4096) };
        assert!(brick.iter().all(|&v| v == 128));
        // a dropped run's unused bricks go to the next run instead of being skipped
        let (_, e) = store.run().alloc();
        assert_eq!(unsafe { e.as_ptr().offset_from(d.as_ptr()) }, 4096);
        let (_, fresh) = store.run().alloc();
        assert_eq!(unsafe { fresh.as_ptr().offset_from(e.as_ptr()) }, 4096);
        // a store sized for few bricks maps no more than it can hold, and maps again when that is used up; ids
        // keep naming their own bricks across segments
        let mut small = BrickStore::<u16, 2048>::new(0, 5);
        let bricks: Vec<_> = (0..5).map(|_| small.alloc()).collect();
        assert!(bricks.windows(2).all(|w| w[0].1 != w[1].1));
        assert!(bricks.iter().all(|&(id, ptr)| small.get(id) == ptr));
        // the range opens a segment at a time; bricks past the first segment are writable and keep their ids
        let mut tiny = BrickStore::<u8, 64>::new(0, 5_000_000);
        let ids: Vec<_> = (0..(1 << 20) + 3).map(|_| tiny.alloc()).collect();
        assert_eq!(tiny.cursor.lock().unwrap().opened, 2 << 20);
        let (id, last) = ids[ids.len() - 1];
        unsafe { last.as_ptr().write(5) };
        assert_eq!(unsafe { tiny.get(id).as_ptr().read() }, 5);
        assert!(ids.iter().step_by(4099).all(|&(id, ptr)| tiny.get(id) == ptr));
    }
}
