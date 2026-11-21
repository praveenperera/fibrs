//! `fibrs`: a native, multi-threaded drop-in for `avf-spline-generator` 0.2.0
//! (Qualzz/AVF_SPLINE_GENERATOR @ bee798f7, Apache-2.0) that writes the same `.afv`

mod afv;
#[cfg(feature = "mimalloc")]
mod alloc;
mod arena;
mod axes;
mod fields;
mod finish;
mod gpu;
mod input;
mod join;
mod kernels;
mod predict;
mod produce;
mod pyenv;
mod pyfmt;
mod run;
mod seams;
mod stage1;
mod tiles;
mod vol;

use std::ffi::OsString;

use clap::{CommandFactory, Parser};

/// mimalloc in place of the system allocator for the tracer, unless built with `--no-default-features`; `fibrs
/// predict` keeps the system allocator (`alloc`)
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: alloc::Dispatch = alloc::Dispatch;

/// Print one log line as the reference's `logging` setup does: `HH:MM:SS message` on stderr
#[macro_export]
macro_rules! log {
    ($($arg:tt)*) => {
        eprintln!("{} {}", $crate::clock(), format_args!($($arg)*))
    };
}

/// Local wall-clock time as `%H:%M:%S`
pub fn clock() -> String {
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    let now = unsafe { libc::time(std::ptr::null_mut()) };
    // the C runtime's reentrant local time: POSIX's localtime_r, or the MSVC runtime's localtime_s, which takes
    // its arguments the other way round
    #[cfg(unix)]
    let _ = unsafe { libc::localtime_r(&now, &mut tm) };
    #[cfg(windows)]
    let _ = unsafe { libc::localtime_s(&mut tm, &now) };
    format!("{:02}:{:02}:{:02}", tm.tm_hour, tm.tm_min, tm.tm_sec)
}

extern "C" fn on_interrupt(_: libc::c_int) {
    // only async-signal-safe calls here: write(2) and _exit(2). On Windows the C runtime runs the handler on a
    // thread of its own when Ctrl+C arrives, and both calls are as safe there
    const MSG: &[u8] = b"Interrupted. Run the same command with --resume to continue.\n";
    unsafe {
        libc::write(2, MSG.as_ptr().cast(), MSG.len() as _);
        libc::_exit(130);
    }
}

fn main() {
    // Windows' default DLL search includes the current directory and PATH; volkit takes both out before any CUDA
    // library loads by bare name, so a DLL planted in the folder a run starts from is never opened. It must come
    // before anything else in the process loads a library
    #[cfg(all(windows, target_arch = "x86_64"))]
    if let Err(e) = volkit::dll::restrict_search() {
        eprintln!("fibrs: {e}");
        std::process::exit(1);
    }

    let argv: Vec<OsString> = std::env::args_os().collect();
    // the predictor has its own command line and none of the tracer's process setup
    if predict::requested(&argv) {
        std::process::exit(predict::main(&argv[2..]));
    }
    if join::requested(&argv) {
        std::process::exit(join::main(&argv[2..]));
    }

    // numpy's OpenBLAS is loaded only for 3x3 eigenproblems; its own thread pool would only add threads
    // beyond --threads. Set before any thread exists
    unsafe { std::env::set_var("OPENBLAS_NUM_THREADS", "1") };
    unsafe { libc::signal(libc::SIGINT, on_interrupt as extern "C" fn(libc::c_int) as libc::sighandler_t) };
    // argparse runs the help and version actions as it meets them, before it checks required flags
    let args = match run::prescan(&argv[1..]) {
        run::Prescan::Help => {
            let _ = run::Args::command().print_help();
            return;
        }
        run::Prescan::Version => {
            let arg0 = argv.first().map(|a| a.as_encoded_bytes()).unwrap_or_default();
            println!("{}", run::version(arg0));
            return;
        }
        run::Prescan::Error(e) => e.exit(),
        run::Prescan::Args(args) => args,
    };
    let args = run::Args::parse_from(std::iter::once(argv[0].clone()).chain(args));
    if let Err(e) = run::run(args) {
        log!("{e}");
        std::process::exit(1);
    }
}
