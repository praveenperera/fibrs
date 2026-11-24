//! `fibrs strips`: villa's sheet-strip inference (`fibers/sheet_strip`, checkpoint `scrollprize/fiber_strip_small_v1`)
//! run natively, as STRIPS-SCALE's `strips_run.py` and `strips_fast.py` run it over an AFV's fibres
//!
//! Per fibre the CPU plans every decision (`geom`: the decision items, the CT seed heading and the frame-model
//! request), the GPU runs the frame model on its crops (`gpu`), the CPU turns its output into each decision's
//! final frame and the strip model's inputs, the GPU crops the CT, renders the tube and runs the strip model, and
//! the CPU stitches the decisions into a strip and writes it (`out`). Decisions batch across fibres on the GPU;
//! the CPU work runs on a thread pool. [`types`] holds the records that pass between the two sides

mod bench_model;
pub mod ct;
pub mod gpu;
pub mod refset;
pub mod types;

use std::ffi::OsString;

use clap::Parser;

/// Whether `argv` asks for `fibrs strips`
pub fn requested(argv: &[OsString]) -> bool {
    let Some(arg0) = argv.first() else { return false };
    !crate::run::dropin(arg0.as_encoded_bytes()) && argv.get(1).map(|a| a.as_encoded_bytes()) == Some(b"strips")
}

/// `fibrs strips` subcommands
#[derive(Parser, Debug)]
#[command(name = "fibrs strips", version = env!("CARGO_PKG_VERSION"))]
enum Command {
    /// Time the native strip model and check it against PyTorch's reference
    BenchModel(bench_model::Args),
}

/// Runs `fibrs strips` on the arguments after the subcommand; the process exit code
pub fn main(args: &[OsString]) -> i32 {
    let command = Command::parse_from(std::iter::once(OsString::from("fibrs strips")).chain(args.iter().cloned()));
    let result = match &command {
        Command::BenchModel(args) => bench_model::run(args),
    };
    match result {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("fibrs strips: {e:#}");
            1
        }
    }
}
