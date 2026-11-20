//! The global allocator: mimalloc for the tracer, the system allocator for `fibrs predict`
//!
//! mimalloc keeps the tracer's dense windows off the kernel's `mmap_lock`, which glibc malloc holds while it maps
//! and unmaps large blocks. The predictor allocates large buffers per batch and column instead, and there mimalloc
//! zeroes reused blocks, copies a growing vector where glibc remaps it, and faults fresh pages in: on a PHerc0826
//! band it doubled the predictor's kernel time and took 12% longer end to end, with the forward pass unchanged
//!
//! The choice is made at the first allocation, from the command line, and never changes, so every block is freed
//! by the allocator that made it. Where the command line cannot be read without allocating, the binary keeps
//! mimalloc
//!
//! On Linux with glibc the tracer's mimalloc runs without transparent huge pages unless `MIMALLOC_ALLOW_THP` asks
//! for them (`thp`)

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU8, Ordering};

use mimalloc::MiMalloc;

const UNDECIDED: u8 = 0;
const MIMALLOC: u8 = 1;
const SYSTEM: u8 = 2;

/// Which allocator this process uses, once decided
static CHOICE: AtomicU8 = AtomicU8::new(UNDECIDED);

/// mimalloc, or the system allocator in `fibrs predict`
pub struct Dispatch;

impl Dispatch {
    #[inline]
    fn system() -> bool {
        match CHOICE.load(Ordering::Relaxed) {
            UNDECIDED => decide(),
            choice => choice == SYSTEM,
        }
    }
}

/// Settles the choice; a thread that loses the race takes the winner's, so all agree
#[cold]
fn decide() -> bool {
    let choice = if args::predict() { SYSTEM } else { MIMALLOC };
    let settled = match CHOICE.compare_exchange(UNDECIDED, choice, Ordering::Relaxed, Ordering::Relaxed) {
        Ok(_) => choice,
        Err(winner) => winner,
    };
    settled == SYSTEM
}

// SAFETY: each call forwards to one allocator, the same one for every call in the process, so a block is resized
// and freed by the allocator that made it
unsafe impl GlobalAlloc for Dispatch {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        unsafe { if Self::system() { System.alloc(layout) } else { MiMalloc.alloc(layout) } }
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        unsafe { if Self::system() { System.alloc_zeroed(layout) } else { MiMalloc.alloc_zeroed(layout) } }
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { if Self::system() { System.dealloc(ptr, layout) } else { MiMalloc.dealloc(ptr, layout) } }
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        unsafe {
            if Self::system() { System.realloc(ptr, layout, new_size) } else { MiMalloc.realloc(ptr, layout, new_size) }
        }
    }
}

/// The tracer's transparent huge pages: off unless `MIMALLOC_ALLOW_THP` turns them on
///
/// With THP allowed, mimalloc asks the kernel for huge pages on its large regions. On a host with THP set to
/// `madvise`, most of those faults fall back to small pages after the kernel has tried to compact memory, which on
/// a dense PHerc0358 window cost the tracer about a sixth of its wall time and more than doubled its kernel time.
/// Off, mimalloc also disables THP for the whole process with `PR_SET_THP_DISABLE`, exactly as
/// `MIMALLOC_ALLOW_THP=0` does. `fibrs predict` runs on the system allocator and keeps the host's setting
#[cfg(all(target_os = "linux", target_env = "gnu"))]
mod thp {
    use libmimalloc_sys::{mi_option_set_default, mi_option_t};

    /// mimalloc's `allow_thp`, by its place in mimalloc 3.3's `mi_option_e`, which libmimalloc-sys does not name;
    /// `tests/allocator.rs` checks it by the name mimalloc prints
    const ALLOW_THP: mi_option_t = 43;

    /// Makes THP off mimalloc's default for the tracer. mimalloc reads its options once, at process load, so this
    /// runs before that, from the `.init_array` function that captures the command line. A value in the
    /// environment still wins, as mimalloc reads it in place of the default
    pub fn default_off() {
        if super::args::predict() {
            return;
        }

        // SAFETY: mimalloc's option table is a static that needs no initialisation, and setting a default before
        // mimalloc reads its options is what `mi_option_set_default` is for
        unsafe { mi_option_set_default(ALLOW_THP, 0) };
    }
}

/// The process's command line, read without allocating
mod args {
    use std::ffi::{CStr, c_char};

    use crate::predict;

    /// Whether the command line asks for `fibrs predict`, as `predict::requested` decides it
    pub fn predict() -> bool {
        let Some((argc, argv)) = argc_argv() else { return false };
        if argc < 2 || argv.is_null() {
            return false;
        }

        // SAFETY: argv holds argc pointers to NUL-terminated strings that live, unmodified, for the whole process
        let (arg0, arg1) = unsafe { (*argv, *argv.add(1)) };
        if arg0.is_null() || arg1.is_null() {
            return false;
        }

        // SAFETY: as above
        let (arg0, arg1) = unsafe { (CStr::from_ptr(arg0), CStr::from_ptr(arg1)) };
        predict::requested_bytes(arg0.to_bytes(), Some(arg1.to_bytes()))
    }

    // glibc passes argc and argv to `.init_array` functions, which run before any Rust code; std reads its
    // `env::args` the same way. Priority 99 also runs this before mimalloc's own constructor (priority 101 when
    // built with clang, none with gcc), which reads mimalloc's options, so `thp` can set their defaults
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    mod capture {
        use std::ffi::{c_char, c_int};
        use std::ptr;
        use std::sync::atomic::{AtomicIsize, AtomicPtr, Ordering};

        static ARGC: AtomicIsize = AtomicIsize::new(0);
        static ARGV: AtomicPtr<*const c_char> = AtomicPtr::new(ptr::null_mut());

        #[used]
        #[unsafe(link_section = ".init_array.00099")]
        static CAPTURE: extern "C" fn(c_int, *const *const c_char, *const *const c_char) = capture;

        extern "C" fn capture(argc: c_int, argv: *const *const c_char, _envp: *const *const c_char) {
            ARGC.store(argc as isize, Ordering::Relaxed);
            ARGV.store(argv.cast_mut(), Ordering::Relaxed);
            super::super::thp::default_off();
        }

        pub fn get() -> Option<(isize, *const *const c_char)> {
            let argv = ARGV.load(Ordering::Relaxed);
            (!argv.is_null()).then(|| (ARGC.load(Ordering::Relaxed), argv.cast_const()))
        }
    }

    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    fn argc_argv() -> Option<(isize, *const *const c_char)> {
        capture::get()
    }

    #[cfg(target_vendor = "apple")]
    fn argc_argv() -> Option<(isize, *const *const c_char)> {
        unsafe extern "C" {
            fn _NSGetArgc() -> *mut std::ffi::c_int;
            fn _NSGetArgv() -> *mut *mut *mut c_char;
        }

        // SAFETY: dyld sets both before any code of the executable runs, and they stay valid for the process
        unsafe { Some((*_NSGetArgc() as isize, (*_NSGetArgv()).cast_const().cast())) }
    }

    #[cfg(not(any(all(target_os = "linux", target_env = "gnu"), target_vendor = "apple")))]
    fn argc_argv() -> Option<(isize, *const *const c_char)> {
        None
    }
}
