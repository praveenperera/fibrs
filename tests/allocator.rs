//! Which allocator a `fibrs` process runs on: the system allocator for `fibrs predict`, mimalloc for everything
//! else, as `src/alloc.rs` chooses it from the command line
#![cfg(all(
    feature = "mimalloc",
    any(all(target_os = "linux", target_env = "gnu"), target_vendor = "apple", all(windows, target_env = "msvc"))
))]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// A run of `exe` with `args` and `MIMALLOC_SHOW_STATS` set, so mimalloc, if it started, prints its statistics, the
/// memory it reserved among them, to stderr at exit
fn run(exe: &Path, args: &[&str]) -> Output {
    let mut command = Command::new(exe);

    // other mimalloc settings from the caller's environment, such as one that skips the exit report, would change
    // what the run prints; mimalloc reads their names in any case
    for (name, _) in std::env::vars_os() {
        if name.to_string_lossy().to_ascii_uppercase().starts_with("MIMALLOC_") {
            command.env_remove(name);
        }
    }

    command.args(args).env("MIMALLOC_SHOW_STATS", "1").output().unwrap()
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
    let dropin = dropin(fibrs, &dir);
    assert!(mimalloc_started(&run(&dropin, &["predict"])), "avf-spline-generator predict did not start mimalloc");
}

/// `fibrs` under the drop-in's name in `dir`: a symlink, or on Windows, where a link needs a privilege and the
/// system runs only a name ending in `.exe`, a copy
fn dropin(fibrs: &Path, dir: &Path) -> PathBuf {
    #[cfg(unix)]
    let dropin = {
        let dropin = dir.join("avf-spline-generator");
        std::os::unix::fs::symlink(fibrs, &dropin).unwrap();
        dropin
    };
    #[cfg(windows)]
    let dropin = {
        let dropin = dir.join("avf-spline-generator.exe");
        fs::copy(fibrs, &dropin).unwrap();
        dropin
    };
    dropin
}
