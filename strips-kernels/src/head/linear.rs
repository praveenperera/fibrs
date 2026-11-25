//! The head's Linear layers as bf16 tensor-core GEMMs, `y = a @ w^T + bias`, with the epilogue each use needs
//!
//! A block computes a 64 x 64 tile of `y` (4 warps of 16 rows, each warp all 64 columns) over the depth in chunks
//! of 64, staged through shared memory; `kp` (the padded input width) is a multiple of 64, and the pad columns of
//! `a` and `w` are zero. Every epilogue first forms `acc + bias` in fp32 and rounds it to bf16, as PyTorch's
//! autocast Linear (cuBLASLt's bias epilogue) does, then applies its own steps; each bf16 result rounds to nearest
//! even. Rows from `m` and columns from `n` are computed on zeros and never stored

use core::ptr::addr_of_mut;

use cuda_device::thread::{__unroll_config, blockIdx_x, blockIdx_y, threadIdx_x};
use cuda_device::{DisjointSlice, SharedArray, kernel, launch_contract, thread};

use super::mma::{SharedTile, mma, sts4};
use super::util::{Bf16, hi_f32, ldg_u4, lo_f32, math, pack_bf16, require, round_bf16};

const THREADS: usize = 128;
/// Rows and columns per block tile, and depth per chunk
const T: usize = 64;
/// Shared rows padded by 8 bf16, which keeps the `ldmatrix` loads bank-conflict free
const LDT: usize = T + 8;
/// 16-byte chunks each thread stages per 64 x 64 tile
const STAGE_ITERS: usize = T * T / 8 / THREADS;
/// Score-free accumulator count per warp: one per 8 columns
const NJ: usize = T / 8;

/// Epilogue activations
pub const ACT_NONE: u32 = 0;
/// Exact (erf) GELU
pub const ACT_GELU: u32 = 1;
/// SiLU
pub const ACT_SILU: u32 = 2;

/// Stages rows `[0, 64)`, columns `[0, 64)` of a bf16 matrix whose row `r` starts at `src + r * ld`, rows from
/// `valid` on as zeros
///
/// # Safety
/// Every thread of the 128-thread block calls it between barriers; rows below `valid` hold 64 elements from `src`,
/// and `src` and `ld` are multiples of 8 elements from a 16-byte aligned buffer
#[inline(always)]
unsafe fn stage(dst: SharedTile<LDT>, src: *const Bf16, ld: u32, valid: u32) {
    let mut x = [[0u32; 4]; STAGE_ITERS];
    for it in 0..STAGE_ITERS {
        __unroll_config::<0>();
        let i = (it * THREADS) as u32 + threadIdx_x();
        let (r, c) = (i / 8, (i % 8) * 8);
        if r < valid {
            // SAFETY: r < valid and c + 8 <= 64, inside the rows the caller vouches for, 16-byte aligned
            x[it] = unsafe { ldg_u4(src.add((r * ld + c) as usize)) };
        }
    }
    for it in 0..STAGE_ITERS {
        __unroll_config::<0>();
        let i = (it * THREADS) as u32 + threadIdx_x();
        let (r, c) = (i / 8, (i % 8) * 8);
        // SAFETY: (r, c) is this thread's alone and r < 64; LDT and c are multiples of 8
        unsafe { sts4(dst.at(r as usize, c as usize), x[it]) };
    }
}

/// The warp's 16 x 64 part of a block tile: rows `[16 warp, 16 warp + 16)` of `a`'s rows from `a0`, all `arows`
/// valid, against `w`'s rows from `w0` (`wrows` valid), over depth `kp`. `acc[j]` holds columns `[8 j, 8 j + 8)`
/// in the m16n8 accumulator layout
///
/// # Safety
/// Every thread of the 128-thread block calls it; `as_` and `ws` are this block's tiles; rows below `arows` of `a0`
/// (stride `lda`) and below `wrows` of `w0` (stride `kp`) hold `kp` elements; `kp`, `lda` are multiples of 64 and
/// 8, and both pointers 16-byte aligned
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_tile(
    as_: SharedTile<LDT>,
    ws: SharedTile<LDT>,
    a0: *const Bf16,
    lda: u32,
    arows: u32,
    w0: *const Bf16,
    kp: u32,
    wrows: u32,
) -> [[f32; 4]; NJ] {
    let wr = (threadIdx_x() as usize / 32) * 16;
    let mut acc = [[0.0f32; 4]; NJ];
    let mut kc = 0u32;
    while kc < kp {
        thread::sync_threads();
        // SAFETY: the caller's contract; columns [kc, kc + 64) lie below kp
        unsafe {
            stage(as_, a0.wrapping_add(kc as usize), lda, arows);
            stage(ws, w0.wrapping_add(kc as usize), kp, wrows);
        }
        thread::sync_threads();
        let mut kk = 0;
        while kk < T / 16 {
            __unroll_config::<0>();
            // SAFETY: warp-uniform offsets inside the tiles, after the barrier
            unsafe {
                let fa = as_.frag_a(wr, kk * 16);
                let mut j = 0;
                while j < NJ {
                    __unroll_config::<0>();
                    let fb = ws.frag_b_pair_nk(j * 8, kk * 16);
                    acc[j] = mma(acc[j], fa, fb[0]);
                    acc[j + 1] = mma(acc[j + 1], fa, fb[1]);
                    j += 2;
                }
            }
            kk += 1;
        }
        kc += T as u32;
    }
    acc
}

/// Calls `f(row, col, y0, y1)` for each pair of columns `col, col + 1` this thread holds that lies inside `m` rows
/// and `n` columns, with `y = bf16(acc + bias)` as f32. `n` is even, so a pair is whole or absent
#[inline(always)]
fn each_pair(acc: &[[f32; 4]; NJ], m0: u32, n0: u32, m: u32, n: u32, bias: &[f32], mut f: impl FnMut(u32, u32, f32, f32)) {
    let lane = threadIdx_x() % 32;
    let (g, t, wr) = (lane / 4, lane % 4, (threadIdx_x() / 32) * 16);
    let mut j = 0;
    while j < NJ {
        __unroll_config::<0>();
        let col = n0 + j as u32 * 8 + 2 * t;
        if col < n {
            // SAFETY: col + 1 < n <= bias.len(), checked by every kernel up front
            let (b0, b1) = unsafe { (*bias.get_unchecked(col as usize), *bias.get_unchecked(col as usize + 1)) };
            let mut r = 0;
            while r < 2 {
                __unroll_config::<0>();
                let row = m0 + wr + g + 8 * r as u32;
                if row < m {
                    f(row, col, round_bf16(acc[j][2 * r] + b0), round_bf16(acc[j][2 * r + 1] + b1));
                }
                r += 1;
            }
        }
        j += 1;
    }
}

/// `act` of a bf16 value, rounded to bf16, as PyTorch's CUDA GELU (exact) and SiLU compute them in fp32
#[inline(always)]
fn activate(y: f32, act: u32) -> f32 {
    match act {
        ACT_GELU => round_bf16(y * 0.5 * (1.0 + math::erff(y * core::f32::consts::FRAC_1_SQRT_2))),
        ACT_SILU => round_bf16(y / (1.0 + math::expf(-y))),
        _ => y,
    }
}

/// The checks every GEMM shares: `a` holds `m` rows of `lda >= kp`, `w` `n` rows of `kp`, `kp` a multiple of 64
#[inline(always)]
fn gemm_ok(a: &[Bf16], w: &[Bf16], bias: &[f32], m: u32, n: u32, kp: u32, lda: u32) -> bool {
    (kp % T as u32 == 0)
        & (kp > 0)
        & (lda >= kp)
        & (lda % 8 == 0)
        & (n % 2 == 0)
        & (m as usize * lda as usize <= a.len())
        & (n as usize * kp as usize <= w.len())
        & (n as usize <= bias.len())
}

macro_rules! tiles {
    () => {{
        static mut AS: SharedArray<u16, { T * LDT }, 16> = SharedArray::UNINIT;
        static mut WS: SharedArray<u16, { T * LDT }, 16> = SharedArray::UNINIT;
        // SAFETY: each array is this kernel's own, 16-byte aligned and 64 rows of LDT
        unsafe {
            (
                SharedTile::<LDT>::new(addr_of_mut!(AS) as *mut u16),
                SharedTile::<LDT>::new(addr_of_mut!(WS) as *mut u16),
            )
        }
    }};
}

/// `y = act(a @ w^T + bias)` in bf16: `out` is `[m, ldo]` bf16 as `u32` pairs, columns `[0, n)` written
///
/// Grid `(ceil(m / 64), ceil(n / 64))`, block 128
#[kernel]
#[launch_contract(domain = 2, block = (128, 1, 1))]
#[allow(clippy::too_many_arguments)]
pub fn sm_linear(
    mut out: DisjointSlice<u32>,
    a: &[Bf16],
    w: &[Bf16],
    bias: &[f32],
    m: u32,
    n: u32,
    kp: u32,
    lda: u32,
    ldo: u32,
    act: u32,
) {
    let (as_, ws) = tiles!();
    let (m0, n0) = (blockIdx_x() * T as u32, blockIdx_y() * T as u32);
    require(
        gemm_ok(a, w, bias, m, n, kp, lda)
            & (m0 < m)
            & (n0 < n)
            & (ldo % 2 == 0)
            & (n <= ldo)
            & (m as usize * ldo as usize / 2 <= out.len()),
    );
    // SAFETY: rows below the valid counts lie inside the checked rows
    let acc = unsafe {
        gemm_tile(
            as_,
            ws,
            a.as_ptr().wrapping_add(m0 as usize * lda as usize),
            lda,
            (m - m0).min(T as u32),
            w.as_ptr().wrapping_add(n0 as usize * kp as usize),
            kp,
            (n - n0).min(T as u32),
        )
    };
    let op = out.as_mut_ptr();
    each_pair(&acc, m0, n0, m, n, bias, |row, col, y0, y1| {
        let p = pack_bf16(activate(y0, act), activate(y1, act));
        // SAFETY: row < m and col + 1 < n <= ldo, inside the checked rows; each pair is this thread's alone
        unsafe { *op.add((row as usize * ldo as usize + col as usize) / 2) = p };
    });
}

/// A residual branch's output projection added to a bf16 residual stream in place: `res = bf16(res + bf16(a @
/// w^T + bias))`, `res` `[m, ldo]` bf16 as `u32` pairs
///
/// Grid `(ceil(m / 64), ceil(n / 64))`, block 128
#[kernel]
#[launch_contract(domain = 2, block = (128, 1, 1))]
#[allow(clippy::too_many_arguments)]
pub fn sm_linear_res(
    mut res: DisjointSlice<u32>,
    a: &[Bf16],
    w: &[Bf16],
    bias: &[f32],
    m: u32,
    n: u32,
    kp: u32,
    lda: u32,
    ldo: u32,
) {
    let (as_, ws) = tiles!();
    let (m0, n0) = (blockIdx_x() * T as u32, blockIdx_y() * T as u32);
    require(
        gemm_ok(a, w, bias, m, n, kp, lda)
            & (m0 < m)
            & (n0 < n)
            & (ldo % 2 == 0)
            & (n <= ldo)
            & (m as usize * ldo as usize / 2 <= res.len()),
    );
    // SAFETY: as in sm_linear
    let acc = unsafe {
        gemm_tile(
            as_,
            ws,
            a.as_ptr().wrapping_add(m0 as usize * lda as usize),
            lda,
            (m - m0).min(T as u32),
            w.as_ptr().wrapping_add(n0 as usize * kp as usize),
            kp,
            (n - n0).min(T as u32),
        )
    };
    let rp = res.as_mut_ptr();
    each_pair(&acc, m0, n0, m, n, bias, |row, col, y0, y1| {
        // SAFETY: as in sm_linear; the pair is read and written only by this thread
        unsafe {
            let at = rp.add((row as usize * ldo as usize + col as usize) / 2);
            let r = *at;
            *at = pack_bf16(lo_f32(r) + y0, hi_f32(r) + y1);
        }
    });
}

/// The same added to an fp32 residual stream in place, then `vec` per column when it isn't empty: `res = (res +
/// bf16(a @ w^T + bias)) + vec`, `res` `[m, ldo]` fp32
///
/// Grid `(ceil(m / 64), ceil(n / 64))`, block 128
#[kernel]
#[launch_contract(domain = 2, block = (128, 1, 1))]
#[allow(clippy::too_many_arguments)]
pub fn sm_linear_resf(
    mut res: DisjointSlice<f32>,
    a: &[Bf16],
    w: &[Bf16],
    bias: &[f32],
    vec: &[f32],
    m: u32,
    n: u32,
    kp: u32,
    lda: u32,
    ldo: u32,
) {
    let (as_, ws) = tiles!();
    let (m0, n0) = (blockIdx_x() * T as u32, blockIdx_y() * T as u32);
    require(
        gemm_ok(a, w, bias, m, n, kp, lda)
            & (m0 < m)
            & (n0 < n)
            & (n <= ldo)
            & (vec.is_empty() | (vec.len() >= n as usize))
            & (m as usize * ldo as usize <= res.len()),
    );
    // SAFETY: as in sm_linear
    let acc = unsafe {
        gemm_tile(
            as_,
            ws,
            a.as_ptr().wrapping_add(m0 as usize * lda as usize),
            lda,
            (m - m0).min(T as u32),
            w.as_ptr().wrapping_add(n0 as usize * kp as usize),
            kp,
            (n - n0).min(T as u32),
        )
    };
    let rp = res.as_mut_ptr();
    let with_vec = !vec.is_empty();
    each_pair(&acc, m0, n0, m, n, bias, |row, col, y0, y1| {
        // SAFETY: as in sm_linear; vec holds n values when it isn't empty
        unsafe {
            let at = rp.add(row as usize * ldo as usize + col as usize);
            let (mut r0, mut r1) = (*at + y0, *at.add(1) + y1);
            if with_vec {
                r0 += *vec.get_unchecked(col as usize);
                r1 += *vec.get_unchecked(col as usize + 1);
            }
            *at = r0;
            *at.add(1) = r1;
        }
    });
}

/// `scale * bf16(a @ w^T + bias)` as fp32: `out` `[m, ldo]` fp32, columns `[0, n)` written (the strip heights, `scale`
/// 8, and the log scales, 1)
///
/// Grid `(ceil(m / 64), ceil(n / 64))`, block 128
#[kernel]
#[launch_contract(domain = 2, block = (128, 1, 1))]
#[allow(clippy::too_many_arguments)]
pub fn sm_linear_f32(
    mut out: DisjointSlice<f32>,
    a: &[Bf16],
    w: &[Bf16],
    bias: &[f32],
    m: u32,
    n: u32,
    kp: u32,
    lda: u32,
    ldo: u32,
    scale: f32,
) {
    let (as_, ws) = tiles!();
    let (m0, n0) = (blockIdx_x() * T as u32, blockIdx_y() * T as u32);
    require(
        gemm_ok(a, w, bias, m, n, kp, lda)
            & (m0 < m)
            & (n0 < n)
            & (n <= ldo)
            & (m as usize * ldo as usize <= out.len()),
    );
    // SAFETY: as in sm_linear
    let acc = unsafe {
        gemm_tile(
            as_,
            ws,
            a.as_ptr().wrapping_add(m0 as usize * lda as usize),
            lda,
            (m - m0).min(T as u32),
            w.as_ptr().wrapping_add(n0 as usize * kp as usize),
            kp,
            (n - n0).min(T as u32),
        )
    };
    let op = out.as_mut_ptr();
    each_pair(&acc, m0, n0, m, n, bias, |row, col, y0, y1| {
        // SAFETY: as in sm_linear
        unsafe {
            let at = op.add(row as usize * ldo as usize + col as usize);
            *at = scale * y0;
            *at.add(1) = scale * y1;
        }
    });
}

/// The qkv projection (`w` `[384, 128]`) of `m = B * ntok` rows of width 128, split into heads: column `c` is q, k or
/// v (`c / 128`), head `(c % 128) / 32`, dim `c % 32`. q goes to `[B, 4, ntok, 32]`, k and v to rows `koff + token`
/// of `[B, 4, ldk, 32]`, all bf16 as `u32` pairs. Columns from `n0` on are computed (128: k and v only, and `q` may
/// be empty)
///
/// Grid `(ceil(m / 64), (384 - n0) / 64)`, block 128
#[kernel]
#[launch_contract(domain = 2, block = (128, 1, 1))]
#[allow(clippy::too_many_arguments)]
pub fn sm_linear_qkv(
    mut q: DisjointSlice<u32>,
    mut k: DisjointSlice<u32>,
    mut v: DisjointSlice<u32>,
    a: &[Bf16],
    w: &[Bf16],
    bias: &[f32],
    m: u32,
    ntok: u32,
    ldk: u32,
    koff: u32,
    n0: u32,
) {
    let (as_, ws) = tiles!();
    let (m0, nb) = (blockIdx_x() * T as u32, n0 + blockIdx_y() * T as u32);
    let decisions = (m / ntok.max(1)) as usize;
    require(
        gemm_ok(a, w, bias, m, 384, 128, 128)
            & (m0 < m)
            & (nb < 384)
            & (n0 % T as u32 == 0)
            & (ntok > 0)
            & (m % ntok.max(1) == 0)
            & (koff + ntok <= ldk)
            & ((n0 >= 128) | (decisions * 4 * ntok as usize * 16 <= q.len()))
            & (decisions * 4 * ldk as usize * 16 <= k.len())
            & (decisions * 4 * ldk as usize * 16 <= v.len()),
    );
    // SAFETY: as in sm_linear
    let acc = unsafe {
        gemm_tile(
            as_,
            ws,
            a.as_ptr().wrapping_add(m0 as usize * 128),
            128,
            (m - m0).min(T as u32),
            w.as_ptr().wrapping_add(nb as usize * 128),
            128,
            (384 - nb).min(T as u32),
        )
    };
    let (qp, kp, vp) = (q.as_mut_ptr(), k.as_mut_ptr(), v.as_mut_ptr());
    each_pair(&acc, m0, nb, m, 384, bias, |row, col, y0, y1| {
        let (b, tok) = (row / ntok, row % ntok);
        let (which, h, d) = (col / 128, (col % 128) / 32, col % 32);
        let p = pack_bf16(y0, y1);
        // SAFETY: b < decisions, h < 4, tok < ntok, koff + tok < ldk and d + 1 < 32, inside the checked buffers;
        // each pair is this thread's alone. which is 0 only when n0 < 128, when q was checked
        unsafe {
            if which == 0 {
                *qp.add((((b * 4 + h) * ntok + tok) * 16 + d / 2) as usize) = p;
            } else {
                let at = (((b * 4 + h) as usize * ldk as usize) + (koff + tok) as usize) * 16 + d as usize / 2;
                if which == 1 {
                    *kp.add(at) = p;
                } else {
                    *vp.add(at) = p;
                }
            }
        }
    });
}

/// Token epilogue modes of [`sm_linear_tok`]
pub const TOK_PLAIN: u32 = 0;
/// Add `tab` row `off + r % rpd` (bf16) and round, before `vec`: cell tokens plus their position embedding
pub const TOK_PER_TOKEN: u32 = 1;
/// Add `tab` row `r / rpd` (bf16) and round, before `vec`: strip tokens plus their decision's orientation embedding
pub const TOK_PER_DECISION: u32 = 2;

/// The last Linear of a token family, written into its rows of a `[B, orpd, 128]` bf16 sequence (as `u32` pairs):
/// row `r` of `m = B * rpd` lands at `(r / rpd) * orpd + off + r % rpd`. `y = bf16(a @ w^T + bias)`; zero where
/// `flag` (one byte per output row, read at `(r / rpd) * flag_ld + flag_off + r % rpd`; empty: none) is zero; plus
/// the table row `mode` names, rounded; plus `vec` (the token kind, fp32 or already bf16-rounded), rounded
///
/// Grid `(ceil(m / 64), 2)`, block 128
#[kernel]
#[launch_contract(domain = 2, block = (128, 1, 1))]
#[allow(clippy::too_many_arguments)]
pub fn sm_linear_tok(
    mut out: DisjointSlice<u32>,
    a: &[Bf16],
    w: &[Bf16],
    bias: &[f32],
    tab: &[u32],
    vec: &[f32],
    flag: &[u8],
    m: u32,
    kp: u32,
    lda: u32,
    rpd: u32,
    orpd: u32,
    off: u32,
    mode: u32,
    flag_ld: u32,
    flag_off: u32,
) {
    let (as_, ws) = tiles!();
    let (m0, n0) = (blockIdx_x() * T as u32, blockIdx_y() * T as u32);
    let decisions = m / rpd.max(1);
    require(
        gemm_ok(a, w, bias, m, 128, kp, lda)
            & (m0 < m)
            & (n0 < 128)
            & (rpd > 0)
            & (m % rpd.max(1) == 0)
            & (off + rpd <= orpd)
            & (decisions as usize * orpd as usize * 64 <= out.len())
            & (vec.len() >= 128)
            & ((mode == TOK_PLAIN)
                | ((mode == TOK_PER_TOKEN) & ((off + rpd) as usize * 64 <= tab.len()))
                | ((mode == TOK_PER_DECISION) & (decisions as usize * 64 <= tab.len())))
            & (flag.is_empty() | (decisions.saturating_sub(1) as usize * flag_ld as usize + (flag_off + rpd) as usize <= flag.len())),
    );
    // SAFETY: as in sm_linear
    let acc = unsafe {
        gemm_tile(
            as_,
            ws,
            a.as_ptr().wrapping_add(m0 as usize * lda as usize),
            lda,
            (m - m0).min(T as u32),
            w.as_ptr().wrapping_add(n0 as usize * kp as usize),
            kp,
            (128 - n0).min(T as u32),
        )
    };
    let op = out.as_mut_ptr();
    let flagged = !flag.is_empty();
    each_pair(&acc, m0, n0, m, 128, bias, |row, col, y0, y1| {
        let (b, r) = (row / rpd, row % rpd);
        // SAFETY: every index below was checked up front for b < decisions and r < rpd; each pair is this
        // thread's alone
        unsafe {
            let (mut y0, mut y1) = (y0, y1);
            if flagged && *flag.get_unchecked((b * flag_ld + flag_off + r) as usize) == 0 {
                (y0, y1) = (0.0, 0.0);
            }
            let trow = if mode == TOK_PER_TOKEN { off + r } else { b };
            if mode != TOK_PLAIN {
                let tv = *tab.get_unchecked((trow * 64 + col / 2) as usize);
                y0 = round_bf16(y0 + lo_f32(tv));
                y1 = round_bf16(y1 + hi_f32(tv));
            }
            let p = pack_bf16(y0 + *vec.get_unchecked(col as usize), y1 + *vec.get_unchecked(col as usize + 1));
            *op.add(((b * orpd + off + r) * 64 + col / 2) as usize) = p;
        }
    });
}
