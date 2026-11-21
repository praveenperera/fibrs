//! What `--version` prints, by the name the binary runs as

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// What `exe --version` prints, without its trailing newline, after checking that it exited cleanly
fn version(exe: &Path) -> String {
    let out = Command::new(exe).arg("--version").output().unwrap();
    assert!(out.status.success(), "{out:?}");
    String::from_utf8(out.stdout).unwrap().trim_end().to_owned()
}

/// Run as `fibrs`, the binary prints this package's version; run as the drop-in `avf-spline-generator`, the version
/// of the reference release it stands in for, which callers of the original may check
#[test]
fn version_follows_the_name_the_binary_runs_as() {
    let fibrs = Path::new(env!("CARGO_BIN_EXE_fibrs"));
    assert_eq!(version(fibrs), env!("CARGO_PKG_VERSION"));

    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("version");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let dropin = dropin(fibrs, &dir);
    assert_eq!(version(&dropin), "0.2.0");
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
