//! The global allocator: the system allocator for large blocks, mimalloc for the rest
//!
//! mimalloc keeps the tracer's many small allocations fast across threads. Large buffers, which the predictor
//! allocates per batch and column, go to the system allocator, which glibc serves with fresh mappings and grows in
//! place with `mremap`, where mimalloc zeroes reused blocks, copies a growing vector and faults fresh pages in
//!
//! Which allocator owns a block follows from its size alone, and a block is always freed with the layout it was
//! allocated or last resized with, so every block is resized and freed by the allocator that made it

use std::alloc::{GlobalAlloc, Layout, System};
use std::ptr;

use mimalloc::MiMalloc;

/// Blocks of at least this many bytes go to the system allocator
const LARGE: usize = 1 << 20;

/// mimalloc for blocks under `LARGE` bytes, the system allocator for the rest
pub struct Dispatch;

#[inline]
fn large(size: usize) -> bool {
    size >= LARGE
}

// SAFETY: a block's owner follows from its size, and GlobalAlloc requires a block be freed or resized with the
// layout it was made with, so every call reaches the allocator that made the block; a resize that crosses `LARGE`
// moves the block to the other allocator
unsafe impl GlobalAlloc for Dispatch {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        unsafe { if large(layout.size()) { System.alloc(layout) } else { MiMalloc.alloc(layout) } }
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        unsafe { if large(layout.size()) { System.alloc_zeroed(layout) } else { MiMalloc.alloc_zeroed(layout) } }
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { if large(layout.size()) { System.dealloc(ptr, layout) } else { MiMalloc.dealloc(ptr, layout) } }
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let (was, will) = (large(layout.size()), large(new_size));
        if was == will {
            return unsafe {
                if will { System.realloc(ptr, layout, new_size) } else { MiMalloc.realloc(ptr, layout, new_size) }
            };
        }

        // SAFETY: the caller guarantees new_size, rounded up to the alignment, does not overflow isize
        let new_layout = unsafe { Layout::from_size_align_unchecked(new_size, layout.align()) };
        let moved = unsafe { self.alloc(new_layout) };
        if moved.is_null() {
            return moved;
        }

        // SAFETY: both blocks are live and hold at least the smaller size; the old one goes back to its owner
        unsafe {
            ptr::copy_nonoverlapping(ptr, moved, layout.size().min(new_size));
            self.dealloc(ptr, layout);
        }
        moved
    }
}
