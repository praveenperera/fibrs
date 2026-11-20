//! Which allocator a `fibrs` process runs on: the system allocator for `fibrs predict`, mimalloc for everything
//! else, as `src/alloc.rs` chooses it from the command line
#![cfg(all(feature = "mimalloc", any(all(target_os = "linux", target_env = "gnu"), target_vendor = "apple")))]

use std::fs;
use std::os::unix::fs::symlink;
use std::path::Path;
use std::process::Command;

/// Whether mimalloc started in a run of `exe` with `args`. With `MIMALLOC_SHOW_STATS` set, mimalloc prints its
/// statistics, the memory it reserved among them, to stderr at exit, and only if it started
fn mimalloc_started(exe: &Path, args: &[&str]) -> bool {
    let out = Command::new(exe).args(args).env("MIMALLOC_SHOW_STATS", "1").output().unwrap();
    String::from_utf8_lossy(&out.stderr).contains("reserved")
}

#[test]
fn predict_runs_on_the_system_allocator_and_the_rest_on_mimalloc() {
    let fibrs = Path::new(env!("CARGO_BIN_EXE_fibrs"));
    assert!(!mimalloc_started(fibrs, &["predict", "--help"]), "fibrs predict started mimalloc");
    assert!(mimalloc_started(fibrs, &["--version"]), "fibrs --version did not start mimalloc");

    // run as the drop-in, the binary keeps the reference CLI, which has no `predict`
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("allocator");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let dropin = dir.join("avf-spline-generator");
    symlink(fibrs, &dropin).unwrap();
    assert!(mimalloc_started(&dropin, &["predict"]), "avf-spline-generator predict did not start mimalloc");
}
