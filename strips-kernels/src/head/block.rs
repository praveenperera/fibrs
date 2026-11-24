//! Fused transformer-layer kernels: a block takes 64 rows of the residual stream, LayerNorms them into shared
//! memory once, and runs every GEMM that reads them, so neither the normalized rows nor the FFN hidden ever reach
//! global memory
//!
//! - [`sm_ln_qkv`]: `norm1` then the qkv projection, split into heads (replaces `sm_layernorm` + `sm_linear_qkv`)
//! - [`sm_ffn`]: `norm2`, `ffn.0`, GELU, `ffn.2` and the residual add (replaces `sm_layernorm` + `sm_linear` +
//!   `sm_linear_res`/`sm_linear_resf`)
//!
//! Numerics equal the unfused kernels' bit for bit: the same LayerNorm ([`super::norm::ln4`]), the same bf16
//! rounding points, and each accumulator sums its depth in the same 16-wide steps in the same order

use core::ptr::addr_of_mut;

use cuda_device::thread::{__unroll_config, blockIdx_x, threadIdx_x};
use cuda_device::{DisjointSlice, SharedArray, kernel, launch_contract, thread};

use super::linear::{ACT_GELU, T, THREADS, activate};
use super::mma::{SharedTile, mma, sts4};
use super::norm::{C, ln4};
use super::util::{Bf16, hi_f32, ldg_u4, lo_f32, pack_bf16, require, round_bf16};

/// The normalized rows' tile: 64 x 128, rows padded by 8 bf16 (272 bytes, an odd count of 16-byte units, so
/// `ldmatrix` stays bank-conflict free)
const LDA: usize = C + 8;
/// A 64 x 64 tile's row length, likewise padded
const LDH: usize = T + 8;
/// The weight tile holds 64 rows of 128 (`LDA`) or 128 rows of 64 (`LDH`): the larger of the two
const W_ELEMS: usize = if T * LDA > 2 * T * LDH { T * LDA } else { 2 * T * LDH };
/// FFN width
const FFN: usize = 2 * C;

/// Rows `[m0, m0 + 64)` of the residual stream (`xb` bf16 pairs `[m, 64]`, or `xf` fp32 `[m, 128]` when
/// `from_f32`), LayerNormed with `gamma`, `beta` and rounded to bf16, into the `[64, LDA]` tile at `dst`; rows from
/// `m` on as zeros. Warp `w` normalizes rows `[16 w, 16 w + 16)`, the rows its own GEMM fragments read
///
/// # Safety
/// Every thread of the 128-thread block calls it; the stream holds `m` rows, `gamma` and `beta` 128 values, and
/// `dst` is this block's tile
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn ln_rows(
    dst: *mut u32,
    xb: *const u32,
    xf: *const f32,
    from_f32: bool,
    gamma: &[f32],
    beta: &[f32],
    m0: u32,
    m: u32,
) {
    let (warp, lane) = (threadIdx_x() / 32, (threadIdx_x() % 32) as usize);
    let c0 = 4 * lane;
    let mut i = 0;
    while i < 16 {
        let r = warp * 16 + i;
        let row = m0 + r;
        // warp-uniform: the whole warp takes the same row
        let y = if row < m {
            // SAFETY: row < m and c0 + 4 <= 128, inside the stream; gamma and beta hold 128 values
            unsafe {
                let x = if from_f32 {
                    let p = xf.add(row as usize * C + c0);
                    [*p, *p.add(1), *p.add(2), *p.add(3)]
                } else {
                    let p = xb.add((row as usize * C + c0) / 2);
                    let (a, b) = (*p, *p.add(1));
                    [lo_f32(a), hi_f32(a), lo_f32(b), hi_f32(b)]
                };
                ln4(x, gamma.as_ptr().add(c0), beta.as_ptr().add(c0))
            }
        } else {
            [0.0; 4]
        };
        // SAFETY: r < 64 and c0 + 4 <= 128 < LDA; these two words are this lane's alone
        unsafe {
            let p = dst.add((r as usize * LDA + c0) / 2);
            *p = pack_bf16(y[0], y[1]);
            *p.add(1) = pack_bf16(y[2], y[3]);
        }
        i += 1;
    }
}

/// Stages `ROWS` rows of `COLS` bf16 from a matrix whose row `r` starts at `src + r * ld` into the tile `dst` (row
/// length `LD`)
///
/// # Safety
/// Every thread of the 128-thread block calls it between barriers; the rows hold `COLS` elements, `src` and `ld`
/// are multiples of 8 elements from a 16-byte aligned buffer, and `COLS` a multiple of 8
#[inline(always)]
unsafe fn stage<const ROWS: usize, const COLS: usize, const LD: usize>(dst: SharedTile<LD>, src: *const Bf16, ld: u32) {
    let per_row = COLS / 8;
    let mut it = 0;
    while it < ROWS * per_row / THREADS {
        __unroll_config::<0>();
        let i = (it * THREADS) as u32 + threadIdx_x();
        let (r, c) = (i / per_row as u32, (i % per_row as u32) * 8);
        // SAFETY: r < ROWS, c + 8 <= COLS, inside the caller's rows and the tile; 16-byte aligned on both sides
        unsafe { sts4(dst.at(r as usize, c as usize), ldg_u4(src.add((r * ld + c) as usize))) };
        it += 1;
    }
}

/// The warp's 16 x 64 product of its 16 normalized rows (`a`, `[64, LDA]`) with the 64 weight rows staged in `w`
/// (`[64, LDA]`), over depth 128: `acc[j]` holds columns `[8 j, 8 j + 8)` in the m16n8 accumulator layout
///
/// # Safety
/// Both tiles are complete (after a barrier)
#[inline(always)]
unsafe fn product_128(a: SharedTile<LDA>, w: SharedTile<LDA>) -> [[f32; 4]; 8] {
    let wr = (threadIdx_x() as usize / 32) * 16;
    let mut acc = [[0.0f32; 4]; 8];
    let mut kk = 0;
    while kk < C / 16 {
        __unroll_config::<0>();
        // SAFETY: warp-uniform offsets inside the tiles
        unsafe {
            let fa = a.frag_a(wr, kk * 16);
            let mut j = 0;
            while j < 8 {
                __unroll_config::<0>();
                let fb = w.frag_b_pair_nk(j * 8, kk * 16);
                acc[j] = mma(acc[j], fa, fb[0]);
                acc[j + 1] = mma(acc[j + 1], fa, fb[1]);
                j += 2;
            }
        }
        kk += 1;
    }
    acc
}

/// Calls `f(r, col, y0, y1)` for each pair of columns `col, col + 1` (from `n0`) of the warp's `J * 8`-column
/// accumulators, `r` the row within the block tile and `y = bf16(acc + bias)` as f32
#[inline(always)]
fn each_pair<const J: usize>(acc: &[[f32; 4]; J], n0: u32, bias: &[f32], mut f: impl FnMut(u32, u32, f32, f32)) {
    let lane = threadIdx_x() % 32;
    let (g, t, wr) = (lane / 4, lane % 4, (threadIdx_x() / 32) * 16);
    let mut j = 0;
    while j < J {
        __unroll_config::<0>();
        let col = n0 + j as u32 * 8 + 2 * t;
        // SAFETY: col + 1 < the layer's width <= bias.len(), checked by every kernel up front
        let (b0, b1) = unsafe { (*bias.get_unchecked(col as usize), *bias.get_unchecked(col as usize + 1)) };
        let mut h = 0;
        while h < 2 {
            __unroll_config::<0>();
            f(wr + g + 8 * h as u32, col, round_bf16(acc[j][2 * h] + b0), round_bf16(acc[j][2 * h + 1] + b1));
            h += 1;
        }
        j += 1;
    }
}

/// Both fused kernels' tiles: the normalized rows and one weight tile
macro_rules! tiles {
    () => {{
        static mut AS: SharedArray<u16, { T * LDA }, 16> = SharedArray::UNINIT;
        static mut WS: SharedArray<u16, W_ELEMS, 16> = SharedArray::UNINIT;
        let a = addr_of_mut!(AS) as *mut u16;
        let w = addr_of_mut!(WS) as *mut u16;
        // SAFETY: each array is this kernel's own and 16-byte aligned; WS holds 64 rows of LDA and 128 of LDH
        unsafe { (a as *mut u32, SharedTile::<LDA>::new(a), SharedTile::<LDA>::new(w), SharedTile::<LDH>::new(w)) }
    }};
}

/// `norm1` of 64 rows of the residual stream (`xb` bf16 pairs or, when not empty, `xf` fp32; `m = B * ntok` rows of
/// 128), then the qkv projection (`w` `[384, 128]`) split into heads as [`super::linear::sm_linear_qkv`] writes it:
/// q to `[B, 4, ntok, 32]`, k and v to rows `koff + token` of `[B, 4, ldk, 32]`. Columns from `n0` on (128: k and
/// v only, and `q` may be empty)
///
/// Grid `ceil(m / 64)`, block 128
#[kernel]
#[launch_contract(domain = 1, block = (128, 1, 1))]
#[allow(clippy::too_many_arguments)]
pub fn sm_ln_qkv(
    mut q: DisjointSlice<u32>,
    mut k: DisjointSlice<u32>,
    mut v: DisjointSlice<u32>,
    xb: &[u32],
    xf: &[f32],
    gamma: &[f32],
    beta: &[f32],
    w: &[Bf16],
    bias: &[f32],
    m: u32,
    ntok: u32,
    ldk: u32,
    koff: u32,
    n0: u32,
) {
    let (a_words, as_, ws, _) = tiles!();
    let m0 = blockIdx_x() * T as u32;
    let from_f32 = !xf.is_empty();
    let decisions = (m / ntok.max(1)) as usize;
    require(
        (m0 < m)
            & (ntok > 0)
            & (m % ntok.max(1) == 0)
            & (koff + ntok <= ldk)
            & (n0 % T as u32 == 0)
            & (n0 < 384)
            & (from_f32 | (xb.len() >= m as usize * C / 2))
            & (!from_f32 | (xf.len() >= m as usize * C))
            & (gamma.len() >= C)
            & (beta.len() >= C)
            & (w.len() >= 384 * C)
            & (bias.len() >= 384)
            & ((n0 >= 128) | (decisions * 4 * ntok as usize * 16 <= q.len()))
            & (decisions * 4 * ldk as usize * 16 <= k.len())
            & (decisions * 4 * ldk as usize * 16 <= v.len()),
    );
    // SAFETY: the stream holds m rows (checked); a_words is this block's tile
    unsafe { ln_rows(a_words, xb.as_ptr(), xf.as_ptr(), from_f32, gamma, beta, m0, m) };

    let (qp, kp, vp) = (q.as_mut_ptr(), k.as_mut_ptr(), v.as_mut_ptr());
    let mut nb = n0;
    while nb < 384 {
        thread::sync_threads();
        // SAFETY: rows [nb, nb + 64) of w lie inside its 384 rows of 128
        unsafe { stage::<T, C, LDA>(ws, w.as_ptr().add(nb as usize * C), C as u32) };
        thread::sync_threads();
        // SAFETY: both tiles are complete after the barrier
        let acc = unsafe { product_128(as_, ws) };
        each_pair(&acc, nb, bias, |r, col, y0, y1| {
            let row = m0 + r;
            if row >= m {
                return;
            }
            let (b, tok) = (row / ntok, row % ntok);
            let (which, h, d) = (col / 128, (col % 128) / 32, col % 32);
            let p = pack_bf16(y0, y1);
            // SAFETY: as in sm_linear_qkv: b < decisions, h < 4, tok < ntok, koff + tok < ldk, d + 1 < 32; which
            // is 0 only when n0 < 128, when q was checked; each pair is this thread's alone
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
        nb += T as u32;
    }
}

/// The FFN sublayer on 64 rows of the residual stream in place: `x += ffn.2(gelu(ffn.0(norm2(x))))`, the stream
/// bf16 (`xb`, `res = bf16(res + y)`) or, when `xf` is not empty, fp32 (`res = res + y`), `y = bf16(... + b2)`.
/// The hidden (256) is formed and consumed 64 columns at a time in shared memory
///
/// Grid `ceil(m / 64)`, block 128
#[kernel]
#[launch_contract(domain = 1, block = (128, 1, 1))]
#[allow(clippy::too_many_arguments)]
pub fn sm_ffn(
    mut xb: DisjointSlice<u32>,
    mut xf: DisjointSlice<f32>,
    gamma: &[f32],
    beta: &[f32],
    w0: &[Bf16],
    b0: &[f32],
    w2: &[Bf16],
    b2: &[f32],
    m: u32,
) {
    static mut HS: SharedArray<u16, { T * LDH }, 16> = SharedArray::UNINIT;
    let (a_words, as_, w0s, w2s) = tiles!();
    let h_ptr = addr_of_mut!(HS) as *mut u16;
    // SAFETY: HS is this kernel's own, 16-byte aligned, 64 rows of LDH
    let hs = unsafe { SharedTile::<LDH>::new(h_ptr) };
    let m0 = blockIdx_x() * T as u32;
    let from_f32 = xf.len() != 0;
    require(
        (m0 < m)
            & (from_f32 | (xb.len() >= m as usize * C / 2))
            & (!from_f32 | (xf.len() >= m as usize * C))
            & (gamma.len() >= C)
            & (beta.len() >= C)
            & (w0.len() >= FFN * C)
            & (b0.len() >= FFN)
            & (w2.len() >= C * FFN)
            & (b2.len() >= C),
    );
    let (xbp, xfp) = (xb.as_mut_ptr(), xf.as_mut_ptr());
    // SAFETY: the stream holds m rows (checked); a_words is this block's tile. The block reads its rows here and
    // writes only those rows at the end
    unsafe { ln_rows(a_words, xbp, xfp, from_f32, gamma, beta, m0, m) };

    let lane = threadIdx_x() % 32;
    let (g, t, wr) = (lane / 4, lane % 4, (threadIdx_x() / 32) * 16);
    let mut out = [[0.0f32; 4]; 16];
    let mut c = 0;
    while c < FFN / T {
        // hidden columns [64 c, 64 c + 64): gelu(bf16(norm2(x) @ w0[rows]^T + b0)), into the warp's rows of HS
        thread::sync_threads();
        // SAFETY: rows [64 c, 64 c + 64) of w0 lie inside its 256 rows of 128
        unsafe { stage::<T, C, LDA>(w0s, w0.as_ptr().add(c * T * C), C as u32) };
        thread::sync_threads();
        // SAFETY: both tiles are complete after the barrier
        let acc = unsafe { product_128(as_, w0s) };
        let mut j = 0;
        while j < 8 {
            __unroll_config::<0>();
            let col = j as u32 * 8 + 2 * t;
            // SAFETY: 64 c + col + 1 < 256 <= b0.len()
            let (bb0, bb1) =
                unsafe { (*b0.get_unchecked(c * T + col as usize), *b0.get_unchecked(c * T + col as usize + 1)) };
            let mut h = 0;
            while h < 2 {
                __unroll_config::<0>();
                let y0 = activate(round_bf16(acc[j][2 * h] + bb0), ACT_GELU);
                let y1 = activate(round_bf16(acc[j][2 * h + 1] + bb1), ACT_GELU);
                // SAFETY: row wr + g + 8 h < 64 and col + 1 < 64 < LDH: this lane's pair of the warp's own rows
                unsafe {
                    *(h_ptr.add((wr + g + 8 * h as u32) as usize * LDH + col as usize) as *mut u32) = pack_bf16(y0, y1)
                };
                h += 1;
            }
            j += 1;
        }

        // out += hidden[:, 64 c..] @ w2[:, 64 c..]^T over depth 64; the weight tile is reused, so every warp must be
        // done with w0's rows first
        thread::sync_threads();
        // SAFETY: w2's 128 rows of 256 hold columns [64 c, 64 c + 64)
        unsafe { stage::<{ 2 * T }, T, LDH>(w2s, w2.as_ptr().add(c * T), FFN as u32) };
        thread::sync_threads();
        let mut kk = 0;
        while kk < T / 16 {
            __unroll_config::<0>();
            // SAFETY: the warp's own HS rows (written above, before the barrier) and the complete w2 tile
            unsafe {
                let fa = hs.frag_a(wr as usize, kk * 16);
                let mut j = 0;
                while j < 16 {
                    __unroll_config::<0>();
                    let fb = w2s.frag_b_pair_nk(j * 8, kk * 16);
                    out[j] = mma(out[j], fa, fb[0]);
                    out[j + 1] = mma(out[j + 1], fa, fb[1]);
                    j += 2;
                }
            }
            kk += 1;
        }
        c += 1;
    }

    each_pair(&out, 0, b2, |r, col, y0, y1| {
        let row = m0 + r;
        if row >= m {
            return;
        }
        // SAFETY: row < m and col + 1 < 128, inside the stream; the pair is this thread's alone, and only this
        // block reads or writes its rows
        unsafe {
            if from_f32 {
                let at = xfp.add(row as usize * C + col as usize);
                *at += y0;
                *at.add(1) += y1;
            } else {
                let at = xbp.add((row as usize * C + col as usize) / 2);
                let p = *at;
                *at = pack_bf16(lo_f32(p) + y0, hi_f32(p) + y1);
            }
        }
    });
}
