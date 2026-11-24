//! `fibrs strips`: villa's sheet-strip inference (`fibers/sheet_strip`, checkpoint `scrollprize/fiber_strip_small_v1`)
//! run natively, as STRIPS-SCALE's `strips_run.py` and `strips_fast.py` run it over an AFV's fibres
//!
//! Per fibre the CPU plans every decision (`geom`: the decision items, the CT seed heading and the frame-model
//! request), the GPU runs the frame model on its crops (`gpu`), the CPU turns its output into each decision's
//! final frame and the strip model's inputs, the GPU crops the CT, renders the tube and runs the strip model, and
//! the CPU stitches the decisions into a strip and writes it (`out`). Decisions batch across fibres on the GPU;
//! the CPU work runs on a thread pool. [`types`] holds the records that pass between the two sides

pub mod gpu;
pub mod types;

use std::ffi::OsString;

/// Whether `argv` asks for `fibrs strips`
pub fn requested(argv: &[OsString]) -> bool {
    let Some(arg0) = argv.first() else { return false };
    !crate::run::dropin(arg0.as_encoded_bytes()) && argv.get(1).map(|a| a.as_encoded_bytes()) == Some(b"strips")
}

/// Runs `fibrs strips` on the arguments after the subcommand; the process exit code
pub fn main(_args: &[OsString]) -> i32 {
    eprintln!("fibrs strips: not built yet");
    2
}
