//! The frame model, villa's HeadingNet (`fibers/heading_model/model.py`, `crop_frame_ct_path_family_v4`), in
//! float32 (`sf_`)
//!
//! `features`: four Conv3d(k 4, stride 2, pad 1) with bias, each followed by SiLU: 1 -> 16 -> 32 -> 64 -> 64
//! channels, 32^3 -> 16^3 -> 8^3 -> 4^3 -> 2^3, then AdaptiveAvgPool3d(2) (the identity on 2^3) and a flatten in
//! (C, D, H, W) order. `path`: Linear(128, 64) and SiLU, plus the family embedding. `head` and `normal_head`:
//! Linear(576, 64), SiLU, Linear(64, 3) over the concatenated 512 + 64 features; the heading is the head's output
//! plus (0, 0, 1), normalized, and the normal the normal head's, normalized.
//!
//! Activations are channels-last `(B, S, S, S, C)`. [`sf_conv1`] runs the single-channel first layer from the
//! `(B, 32, 32, 32)` crop; [`sf_conv`] the other three as a direct convolution whose weights the host lays out
//! `[(tap · CI + ci) · CO + co]`; [`sf_head`] the path branch, both heads and the normalization. SiLU is torch's
//! `x / (1 + exp(-x))`. Accumulation order differs from cuDNN's, so outputs track the reference to float32
//! rounding, not bit for bit

use core::ptr::addr_of_mut;

use cuda_device::float::{add_rn_f32, div_rn_f32, fma_rn_f32, mul_rn_f32, sqrt_rn_f32};
use cuda_device::thread::{blockDim_x, blockIdx_x, blockIdx_y, sync_threads, threadIdx_x};
use cuda_device::vector::F32x4;
use cuda_device::{SharedArray, kernel};

unsafe extern "C" {
    fn __nv_expf(x: f32) -> f32;
}

/// Output channels each [`sf_conv`] thread computes
pub const CONV_CO_TILE: u32 = 16;
/// Threads (output positions) per [`sf_conv`] block
pub const CONV_THREADS: u32 = 128;
/// Rows of the weight matrix a [`sf_conv`] block stages at a time
const CONV_K_TILE: u32 = 64;

#[inline(always)]
fn silu(x: f32) -> f32 {
    // SAFETY: libdevice's expf, a pure function
    div_rn_f32(x, add_rn_f32(1.0, unsafe { __nv_expf(-x) }))
}

/// Layer 1: `y[b, z, y, x, co] = silu(bias[co] + Σ_tap w[co · 64 + tap] · crop[b, 2z+kz-1, 2y+ky-1, 2x+kx-1])`
/// over the `(B, 32, 32, 32)` crop, zero outside it; `y` is `(B, 16, 16, 16, 16)`
///
/// Grid: x over `B · 4096 / 256`; 256 threads, one output position each
#[kernel]
pub unsafe fn sf_conv1(crop: *const f32, w: *const f32, bias: *const f32, n: u32, y: *mut f32) {
    static mut W: SharedArray<f32, 1024, 16> = SharedArray::UNINIT;
    let ws = unsafe { &mut *addr_of_mut!(W) };
    let t = threadIdx_x();
    let mut i = t;
    while i < 1024 {
        // SAFETY: 16 x 64 weights
        ws[i as usize] = unsafe { *w.add(i as usize) };
        i += blockDim_x();
    }
    sync_threads();
    let p = blockIdx_x() * blockDim_x() + t;
    if p >= n {
        return;
    }
    let b = p >> 12;
    let (oz, oy, ox) = ((p >> 8) & 15, (p >> 4) & 15, p & 15);
    let mut acc = [0f32; 16];
    // SAFETY: 16 biases
    #[unroll]
    for co in 0..16 {
        acc[co] = unsafe { *bias.add(co) };
    }
    let src = unsafe { crop.add((b as usize) << 15) };
    for kz in 0..4i32 {
        let iz = 2 * oz as i32 + kz - 1;
        for ky in 0..4i32 {
            let iy = 2 * oy as i32 + ky - 1;
            #[unroll]
            for kx in 0..4i32 {
                let ix = 2 * ox as i32 + kx - 1;
                let inside = ((iz as u32) < 32) & ((iy as u32) < 32) & ((ix as u32) < 32);
                // SAFETY: inside the crop
                let v = if inside { unsafe { *src.add(((iz << 10) | (iy << 5) | ix) as usize) } } else { 0.0 };
                let tap = ((kz * 4 + ky) * 4 + kx) as usize;
                #[unroll]
                for co in 0..16 {
                    acc[co] = fma_rn_f32(ws[co * 64 + tap], v, acc[co]);
                }
            }
        }
    }
    // SAFETY: 16 channels per position
    let dst = unsafe { y.add(p as usize * 16) as *mut F32x4 };
    #[unroll]
    for q in 0..4 {
        unsafe {
            *dst.add(q) = F32x4::new([silu(acc[4 * q]), silu(acc[4 * q + 1]), silu(acc[4 * q + 2]), silu(acc[4 * q + 3])]);
        }
    }
}

/// Layers 2-4: `y[b, o, co] = silu(bias[co] + Σ_{tap, ci} w[(tap · ci_n + ci) · co_n + co] · x[b, 2o + k - 1, ci])`
/// from the channels-last `(B, s_in, s_in, s_in, ci_n)` input to `(B, s_in/2, s_in/2, s_in/2, co_n)`, zero padding.
/// `s_in` is `1 << s_shift`, `ci_n` a multiple of 4 and `co_n` of [`CONV_CO_TILE`]; `n` is `B · (s_in/2)^3`
///
/// Grid: x over `n / CONV_THREADS` rounded up, y over `co_n / CONV_CO_TILE`; [`CONV_THREADS`] threads, one output
/// position each, computing the block's [`CONV_CO_TILE`] channels
#[kernel]
#[allow(clippy::too_many_arguments)]
pub unsafe fn sf_conv(
    x: *const f32,
    w: *const f32,
    bias: *const f32,
    s_shift: u32,
    ci_n: u32,
    co_n: u32,
    n: u32,
    y: *mut f32,
) {
    static mut W: SharedArray<f32, { (CONV_K_TILE * CONV_CO_TILE) as usize }, 16> = SharedArray::UNINIT;
    let ws = unsafe { &mut *addr_of_mut!(W) };
    let t = threadIdx_x();
    let co0 = blockIdx_y() * CONV_CO_TILE;
    let p = blockIdx_x() * blockDim_x() + t;
    let live = p < n;
    let o_shift = s_shift - 1;
    let o_mask = (1u32 << o_shift) - 1;
    let b = p >> (3 * o_shift);
    let (oz, oy, ox) = ((p >> (2 * o_shift)) & o_mask, (p >> o_shift) & o_mask, p & o_mask);
    let s_in = 1i32 << s_shift;
    let src = unsafe { x.add(((b as usize) << (3 * s_shift)) * ci_n as usize) };
    let mut acc = [0f32; 16];
    #[unroll]
    for c in 0..16 {
        // SAFETY: `co_n` biases
        acc[c] = unsafe { *bias.add((co0 + c as u32) as usize) };
    }
    let k_total = 64 * ci_n;
    let mut k0 = 0u32;
    while k0 < k_total {
        sync_threads();
        let mut i = t;
        while i < CONV_K_TILE * CONV_CO_TILE {
            let (r, c) = (i >> 4, i & 15);
            // SAFETY: `k0 + r < k_total` since `k_total` is a multiple of the tile
            ws[i as usize] = unsafe { *w.add(((k0 + r) * co_n + co0 + c) as usize) };
            i += blockDim_x();
        }
        sync_threads();
        if live {
            let mut kk = 0u32;
            while kk < CONV_K_TILE {
                let k = k0 + kk;
                let tap = k / ci_n;
                let ci = k - tap * ci_n;
                let (kz, ky, kx) = ((tap >> 4) as i32, ((tap >> 2) & 3) as i32, (tap & 3) as i32);
                let iz = 2 * oz as i32 + kz - 1;
                let iy = 2 * oy as i32 + ky - 1;
                let ix = 2 * ox as i32 + kx - 1;
                let inside = ((iz as u32) < s_in as u32) & ((iy as u32) < s_in as u32) & ((ix as u32) < s_in as u32);
                let v = if inside {
                    let at = ((((iz << s_shift) | iy) << s_shift) | ix) as usize * ci_n as usize + ci as usize;
                    // SAFETY: inside the input, and `ci` is a multiple of 4 with `ci + 4 <= ci_n`
                    unsafe { *(src.add(at) as *const F32x4) }
                } else {
                    F32x4::new([0.0; 4])
                };
                let v = v.to_array();
                #[unroll]
                for j in 0..4 {
                    let row = ((kk + j as u32) * CONV_CO_TILE) as usize;
                    #[unroll]
                    for c in 0..16 {
                        acc[c] = fma_rn_f32(ws[row + c], v[j], acc[c]);
                    }
                }
                kk += 4;
            }
        }
        k0 += CONV_K_TILE;
    }
    if !live {
        return;
    }
    // SAFETY: `co_n` channels per position
    let dst = unsafe { y.add(p as usize * co_n as usize + co0 as usize) as *mut F32x4 };
    #[unroll]
    for q in 0..4 {
        unsafe {
            *dst.add(q) = F32x4::new([silu(acc[4 * q]), silu(acc[4 * q + 1]), silu(acc[4 * q + 2]), silu(acc[4 * q + 3])]);
        }
    }
}

/// The path branch, both heads and the normalization for decision `blockIdx.x`: from the last conv's
/// channels-last `(B, 8, 64)` features (flattened in (C, D, H, W) order as torch's Flatten), the `(B, 128)` path
/// features and the family index, `out[6 b ..] = (heading, normal)`
///
/// `wp`/`bp` the path Linear(128, 64), `emb` the (2, 64) family embedding, `w0`/`b0` and `w2`/`b2` the heading
/// MLP's Linear(576, 64) and Linear(64, 3), `v0`/`c0` and `v2`/`c2` the normal MLP's. Grid: x over decisions;
/// 64 threads
#[kernel]
#[allow(clippy::too_many_arguments)]
pub unsafe fn sf_head(
    feat: *const f32,
    path: *const f32,
    family: *const u32,
    wp: *const f32,
    bp: *const f32,
    emb: *const f32,
    w0: *const f32,
    b0: *const f32,
    w2: *const f32,
    b2: *const f32,
    v0: *const f32,
    c0: *const f32,
    v2: *const f32,
    c2: *const f32,
    out: *mut f32,
) {
    static mut HIDDEN: SharedArray<f32, 576> = SharedArray::UNINIT;
    static mut MID: SharedArray<f32, 128> = SharedArray::UNINIT;
    static mut RAW: SharedArray<f32, 8> = SharedArray::UNINIT;
    let hidden = unsafe { &mut *addr_of_mut!(HIDDEN) };
    let mid = unsafe { &mut *addr_of_mut!(MID) };
    let raw = unsafe { &mut *addr_of_mut!(RAW) };
    let b = blockIdx_x() as usize;
    let j = threadIdx_x() as usize;
    // SAFETY: every read below is inside the tensors' shapes, which the host checks
    unsafe {
        let mut i = j;
        while i < 512 {
            // torch's flatten index c·8 + s from the channels-last [s][c]
            let (c, s) = (i >> 3, i & 7);
            hidden[i] = *feat.add(b * 512 + s * 64 + c);
            i += 64;
        }
        let pb = path.add(b * 128);
        let row = wp.add(j * 128);
        let mut acc = 0f32;
        for i in 0..128 {
            acc = fma_rn_f32(*row.add(i), *pb.add(i), acc);
        }
        let fam = *family.add(b) as usize;
        hidden[512 + j] = add_rn_f32(silu(add_rn_f32(acc, *bp.add(j))), *emb.add(fam * 64 + j));
        sync_threads();

        let (rh, rn) = (w0.add(j * 576), v0.add(j * 576));
        let (mut ah, mut an) = (0f32, 0f32);
        for i in 0..576 {
            let h = hidden[i];
            ah = fma_rn_f32(*rh.add(i), h, ah);
            an = fma_rn_f32(*rn.add(i), h, an);
        }
        mid[j] = silu(add_rn_f32(ah, *b0.add(j)));
        mid[64 + j] = silu(add_rn_f32(an, *c0.add(j)));
        sync_threads();

        if j < 6 {
            let (normal, o) = (j >= 3, j % 3);
            let (w, bias, m) = if normal { (v2, c2, 64) } else { (w2, b2, 0) };
            let mut acc = 0f32;
            for i in 0..64 {
                acc = fma_rn_f32(*w.add(o * 64 + i), mid[m + i], acc);
            }
            let mut v = add_rn_f32(acc, *bias.add(o));
            if !normal && o == 2 {
                v = add_rn_f32(v, 1.0);
            }
            raw[j] = v;
        }
        sync_threads();
        if j < 6 {
            let base = if j >= 3 { 3 } else { 0 };
            let (x, y, z) = (raw[base], raw[base + 1], raw[base + 2]);
            let norm = sqrt_rn_f32(add_rn_f32(add_rn_f32(mul_rn_f32(x, x), mul_rn_f32(y, y)), mul_rn_f32(z, z)));
            *out.add(6 * b + j) = div_rn_f32(raw[j], if norm > 1e-12 { norm } else { 1e-12 });
        }
    }
}
