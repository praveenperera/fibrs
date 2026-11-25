//! `fibrs strips`: villa's sheet-strip inference (`fibers/sheet_strip`, checkpoint `scrollprize/fiber_strip_small_v1`)
//! run natively over an AFV's fibres, with the Python reference runner's command line and outputs
//!
//! Per fibre the CPU plans every decision (`geom`: the decision items, the CT seed heading and the frame-model
//! request), the GPU runs the frame model on its crops (`gpu`), the CPU turns its output into each decision's
//! final frame and the strip model's inputs, the GPU crops the CT, renders the tube and runs the strip model, and
//! the CPU stitches the decisions into a strip and writes it (`out`). Decisions batch across fibres on the GPU;
//! the CPU work runs on a thread pool. [`types`] holds the records that pass between the two sides

mod bench_cpu;
mod bench_model;
pub mod ct;
pub mod engine;
pub mod fibre;
pub mod geom;
pub mod gpu;
pub mod np;
pub mod out;
pub mod refset;
mod runner;
pub mod seed;
pub mod stitch;
pub mod types;

#[cfg(test)]
mod tests;

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
    /// Run the native strip runner over an AFV's fibres (the Python reference runner's command line and outputs)
    Run(Box<runner::Args>),
    /// Time the native strip model and check it against PyTorch's reference
    BenchModel(bench_model::Args),
}

/// Runs `fibrs strips` on the arguments after the subcommand; the process exit code
pub fn main(args: &[OsString]) -> i32 {
    // bench-cpu parses its own arguments
    let result = if args.first().and_then(|a| a.to_str()) == Some("bench-cpu") {
        bench_cpu::main(&args[1..])
    } else {
        let command = Command::parse_from(std::iter::once(OsString::from("fibrs strips")).chain(args.iter().cloned()));
        match &command {
            Command::Run(args) => runner::run(args),
            Command::BenchModel(args) => bench_model::run(args),
        }
    };
    match result {
        Ok(()) => 0,
        Err(e) if e.downcast_ref::<runner::Deferred>().is_some() => {
            eprintln!("fibrs strips: {e:#}");
            runner::EXIT_UNCACHED
        }
        Err(e) => {
            eprintln!("fibrs strips: {e:#}");
            1
        }
    }
}
