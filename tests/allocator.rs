//! Which allocator a `fibrs` process runs on: the system allocator for `fibrs predict`, mimalloc for everything
//! else, as `src/alloc.rs` chooses it from the command line
#![cfg(all(feature = "mimalloc", any(all(target_os = "linux", target_env = "gnu"), target_vendor = "apple")))]

use std::fs;
use std::os::unix::fs::symlink;
use std::path::Path;
use std::process::{Command, Output};

/// A run of `exe` with `args` and `MIMALLOC_SHOW_STATS` set, so mimalloc, if it started, prints its statistics, the
/// memory it reserved among them, to stderr at exit
fn run(exe: &Path, args: &[&str]) -> Output {
    Command::new(exe).args(args).env("MIMALLOC_SHOW_STATS", "1").output().unwrap()
}

fn mimalloc_started(out: &Output) -> bool {
    String::from_utf8_lossy(&out.stderr).contains("reserved")
}

#[test]
fn predict_runs_on_the_system_allocator_and_the_rest_on_mimalloc() {
    let fibrs = Path::new(env!("CARGO_BIN_EXE_fibrs"));

    // a run that died before exit would print no statistics either, so it must first have finished its work
    let help = run(fibrs, &["predict", "--help"]);
    assert!(help.status.success() && String::from_utf8_lossy(&help.stdout).contains("--ct"), "{help:?}");
    assert!(!mimalloc_started(&help), "fibrs predict started mimalloc");

    assert!(mimalloc_started(&run(fibrs, &["--version"])), "fibrs --version did not start mimalloc");

    // run as the drop-in, the binary keeps the reference CLI, which has no `predict`
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("allocator");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let dropin = dir.join("avf-spline-generator");
    symlink(fibrs, &dropin).unwrap();
    assert!(mimalloc_started(&run(&dropin, &["predict"])), "avf-spline-generator predict did not start mimalloc");
}
