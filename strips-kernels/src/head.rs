//! The strip model's head (prefix `sm_`): the polar cell tokens, reference and path-geometry tokens, the
//! four-layer context transformer, the strip queries, both strip passes and their readouts
//!
//! Numerics follow PyTorch under bf16 autocast: every Linear rounds its input, weight and bias to bf16, accumulates
//! in fp32 and rounds its output to bf16; LayerNorm reads bf16 or fp32 and writes fp32 (here rounded to bf16 where
//! a Linear reads it); feature sampling runs in fp32 on the bf16 cells. Activations are row-major `[tokens,
//! channels]`; attention reads q `[B, 4, nq, 32]` and k/v `[B, 4, ldk, 32]`

pub mod attn;
pub mod linear;
pub mod mma;
pub mod norm;
pub mod tokens;
pub mod util;
