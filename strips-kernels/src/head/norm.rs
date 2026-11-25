//! LayerNorm over the head's width (128), one warp per row: bf16 or fp32 in, fp32 statistics, bf16 and/or fp32 out
//!
//! Two-pass fp32 statistics (mean, then the mean of squared deviations) and `gamma * (rstd * (x - mean)) + beta`
//! with `rstd = rsqrtf(var + eps)`, the output formula of PyTorch's vectorized LayerNorm kernel; PyTorch's own
//! statistics are Welford's, so the two agree to fp32 rounding. Under autocast LayerNorm outputs fp32: `yf` gets
//! that, `yb` the bf16 rounding the next Linear reads

use cuda_device::thread::{__unroll_config, blockIdx_x, threadIdx_x};
use cuda_device::{DisjointSlice, kernel, launch_contract};

use super::util::{hi_f32, lo_f32, math, pack_bf16, require, warp_sum};

/// Width
const C: usize = 128;
/// LayerNorm epsilon (PyTorch's default)
const EPS: f32 = 1e-5;

/// LayerNorm of `rows` rows of 128: from `xb` (bf16 as `u32` pairs, `[rows, 64]`) or `xf` (`[rows, 128]` fp32),
/// whichever isn't empty, to `yb` (bf16 pairs) and/or `yf` (fp32), each skipped when empty
///
/// Grid `ceil(rows / 4)`, block 128: warp `w` of block `x` takes row `4 x + w`; lane `l` takes columns `4 l..4 l + 4`
#[kernel]
#[launch_contract(domain = 1, block = (128, 1, 1))]
#[allow(clippy::too_many_arguments)]
pub fn sm_layernorm(
    mut yb: DisjointSlice<u32>,
    mut yf: DisjointSlice<f32>,
    xb: &[u32],
    xf: &[f32],
    gamma: &[f32],
    beta: &[f32],
    rows: u32,
) {
    let n = rows as usize;
    let from_f32 = !xf.is_empty();
    require(
        (gamma.len() >= C)
            & (beta.len() >= C)
            & (from_f32 | (xb.len() >= n * C / 2))
            & (!from_f32 | (xf.len() >= n * C))
            & ((yb.len() == 0) | (yb.len() >= n * C / 2))
            & ((yf.len() == 0) | (yf.len() >= n * C)),
    );
    let row = blockIdx_x() * 4 + threadIdx_x() / 32;
    if row >= rows {
        return;
    }
    let lane = (threadIdx_x() % 32) as usize;
    let c0 = 4 * lane;

    // SAFETY: row < rows and c0 + 4 <= 128, inside the checked input
    let x: [f32; 4] = unsafe {
        if from_f32 {
            let p = xf.as_ptr().add(row as usize * C + c0);
            [*p, *p.add(1), *p.add(2), *p.add(3)]
        } else {
            let p = xb.as_ptr().add((row as usize * C + c0) / 2);
            let (a, b) = (*p, *p.add(1));
            [lo_f32(a), hi_f32(a), lo_f32(b), hi_f32(b)]
        }
    };
    let mean = warp_sum((x[0] + x[1]) + (x[2] + x[3])) / C as f32;
    let d = [x[0] - mean, x[1] - mean, x[2] - mean, x[3] - mean];
    let var = warp_sum((d[0] * d[0] + d[1] * d[1]) + (d[2] * d[2] + d[3] * d[3])) / C as f32;
    let rstd = math::rsqrtf(var + EPS);
    let mut y = [0.0f32; 4];
    let mut i = 0;
    while i < 4 {
        __unroll_config::<0>();
        // SAFETY: c0 + i < 128 <= gamma.len(), beta.len()
        let (g, b) = unsafe { (*gamma.get_unchecked(c0 + i), *beta.get_unchecked(c0 + i)) };
        y[i] = g * (rstd * d[i]) + b;
        i += 1;
    }
    // SAFETY: each output row's columns c0..c0 + 4 are this lane's alone, inside the checked outputs
    unsafe {
        if yf.len() != 0 {
            let p = yf.as_mut_ptr().add(row as usize * C + c0);
            *p = y[0];
            *p.add(1) = y[1];
            *p.add(2) = y[2];
            *p.add(3) = y[3];
        }
        if yb.len() != 0 {
            let p = yb.as_mut_ptr().add((row as usize * C + c0) / 2);
            *p = pack_bf16(y[0], y[1]);
            *p.add(1) = pack_bf16(y[2], y[3]);
        }
    }
}
