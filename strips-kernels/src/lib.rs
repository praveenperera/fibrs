//! cuda-oxide device kernels for `fibrs strips`, the native sheet-strip runner
//!
//! `scripts/build-strips-ptx.sh` compiles this crate to `ptx/strips.sm80.ptx`, which fibrs embeds and loads through
//! volkit's `Gpu::load_module`; every launch is checked against the PTX param lists. Each stage's kernels live in
//! their own module, prefixed by the stage (`sc_` crops and tube, `sn_` the CNN, `sm_` the strip model's tokens,
//! transformer and heads, `sf_` the frame model)

#![no_std]
#![allow(clippy::missing_safety_doc)]

pub mod crops;
pub mod frame;
