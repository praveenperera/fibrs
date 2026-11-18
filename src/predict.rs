//! `fibrs predict`: the AFV fibre model V2 inference, which is Python (`fibrs.predict`, from the `predict` extra)

use std::ffi::{OsStr, OsString};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Whether `argv` asks for `fibrs predict`. Run as the drop-in `avf-spline-generator`, the binary keeps the
/// reference CLI, which has no subcommands
pub fn requested(argv: &[OsString]) -> bool {
    let dropin = argv.first().and_then(|a| Path::new(a).file_name()) == Some(OsStr::new("avf-spline-generator"));
    !dropin && argv.get(1).is_some_and(|a| a == "predict")
}

/// Replace this process with `PYTHON -m fibrs.predict ARGS`; returns only if that can't start
pub fn exec(args: &[OsString]) -> ! {
    let python = python();
    let err = Command::new(&python).args(["-m", "fibrs.predict"]).args(args).exec();
    eprintln!("fibrs predict: could not run {}: {err}", python.display());
    std::process::exit(1);
}

/// `FIBRS_PYTHON`, else the `python` beside this binary (the wheel installs both in the environment's scripts
/// directory), else `python3` on `PATH`
fn python() -> PathBuf {
    if let Some(explicit) = std::env::var_os("FIBRS_PYTHON") {
        return explicit.into();
    }

    let beside = std::env::current_exe().ok().and_then(|exe| exe.parent().map(|dir| dir.join("python")));
    beside.filter(|p| p.is_file()).unwrap_or_else(|| "python3".into())
}
