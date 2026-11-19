//! `fibrs predict`: AFV fibre model V2 (Qualzz20/afv_fiber_9um_v2) probabilities for a CT, natively on an
//! NVIDIA GPU
//!
//! The same command line, output group and records as villa `lasagna/qualzz_inference.py` and fibrs's band
//! reference: the CT is cut into 128^3 tiles on a stride-96 lattice, grouped into xy columns of the output
//! (`contract`); the tiles are read ahead on reader threads and batched (`feed`); volkit's `AfvV2` normalises,
//! runs the network and blends and finalises each column on the GPU (`engine`); one writer thread stores the
//! columns and their progress records (`out`). `--mode villa` swaps the air rule for villa's storage-based one
//! (`villa`)

mod contract;
mod ct;
mod engine;
mod feed;
mod out;
mod run;
mod villa;

use std::ffi::{OsStr, OsString};
use std::path::Path;

pub use run::main;

/// Whether `argv` asks for `fibrs predict`. Run as the drop-in `avf-spline-generator`, the binary keeps the
/// reference CLI, which has no subcommands
pub fn requested(argv: &[OsString]) -> bool {
    let dropin = argv.first().and_then(|a| Path::new(a).file_name()) == Some(OsStr::new("avf-spline-generator"));
    !dropin && argv.get(1).is_some_and(|a| a == "predict")
}
