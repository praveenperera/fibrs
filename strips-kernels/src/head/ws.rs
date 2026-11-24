//! Weight-stationary persistent GEMMs for the transformer layers' Linears (`y = a @ w^T + bias`, depth 128 or 256)
//!
//! The 64x64-tile GEMMs (`linear.rs`) re-read every weight tile and every activation row from L2 for each output
//! tile, which keeps them memory-bound on these small depths (ncu at b64: L2 57-73% of peak, SM 32-50%). Here a
//! block stages one 128-column slice of `w` into shared memory once and walks row tiles of 64 (`blockIdx.x`,
//! stepping by `gridDim.x`), prefetching the next tile's rows into registers while it multiplies the current one.
//! `blockIdx.y` picks the slice. 8 warps: warp `w` takes rows `[16 (w % 4), +16)` and columns `[64 (w / 4), +64)`
//!
//! The 128-column Linears that add into the residual stream own whole rows, so their epilogue also LayerNorms the
//! new stream rows for the next Linear ([`sm_ws_res128`], [`sm_ws_res256`]), which replaces a LayerNorm pass. Its
//! statistics are two-pass fp32 like `norm.rs`, summed in a different order
//!
//! Every epilogue first forms `bf16(acc + bias)`, as PyTorch's autocast Linear does. Shared memory is dynamic:
//! `w`'s slice `[128, K + 8]`, then the row tile `[64, K + 8]` (the LayerNorm reuses the tile's bytes for its row
//! sums); [`smem_bytes`] gives the size

use cuda_device::thread::{__unroll_config, blockIdx_x, blockIdx_y, gridDim_x, threadIdx_x};
use cuda_device::{DisjointSlice, DynamicSharedArray, kernel, launch_contract, thread, warp};

use super::linear::activate;
use super::mma::{SharedTile, mma, sts4};
use super::util::{Bf16, hi_f32, ldg_u4, lo_f32, math, pack_bf16, require, round_bf16};

const THREADS: u32 = 256;
/// Rows per tile and columns per slice
const TM: u32 = 64;
const TN: u32 = 128;
/// Accumulators per warp: its 64 columns in groups of 8
const NJ: usize = 8;
/// LayerNorm epsilon (PyTorch's default) and width
const EPS: f32 = 1e-5;
const C: u32 = 128;

/// Dynamic shared memory of a depth-`k` kernel: the weight slice and the row tile, rows padded by 8 bf16
pub const fn smem_bytes(k: u32) -> u32 {
    (TN + TM) * (k + 8) * 2
}

/// The block's tiles in the dynamic pool, rows of `LD`
#[inline(always)]
fn tiles<const LD: usize>() -> (SharedTile<LD>, SharedTile<LD>, *mut f32) {
    let pool: *mut u16 = DynamicSharedArray::<u16, 16>::get();
    // SAFETY: the launch gives `smem_bytes(LD - 8)` bytes: 128 rows of the slice, then 64 of the tile
    unsafe {
        let a = pool.add(TN as usize * LD);
        (SharedTile::<LD>::new(pool), SharedTile::<LD>::new(a), a as *mut f32)
    }
}

/// Stages rows `[0, 128)` of `w` (row stride `K`) into the slice, rows from `valid` on as zeros
///
/// # Safety
/// Every thread of the block calls it; rows below `valid` hold `K` elements, 16-byte aligned
#[inline(always)]
unsafe fn stage_w<const K: usize, const LD: usize>(dst: SharedTile<LD>, w: *const Bf16, valid: u32) {
    let per_row = (K / 8) as u32;
    let mut i = threadIdx_x();
    while i < TN * per_row {
        let (r, c) = (i / per_row, (i % per_row) * 8);
        let v = if r < valid {
            // SAFETY: r < valid and c + 8 <= K
            unsafe { ldg_u4(w.add((r * K as u32 + c) as usize)) }
        } else {
            [0; 4]
        };
        // SAFETY: (r, c) is this thread's alone, inside the slice
        unsafe { sts4(dst.at(r as usize, c as usize), v) };
        i += THREADS;
    }
}

/// This thread's share of a row tile: rows `[m0, m0 + 64)` of `a` (stride `lda`), rows from `m` on as zeros
///
/// # Safety
/// Rows below `m` hold `K` elements from a 16-byte aligned `a`, `lda` a multiple of 8
#[inline(always)]
unsafe fn load_tile<const K: usize, const NA: usize>(a: *const Bf16, lda: u32, m0: u32, m: u32) -> [[u32; 4]; NA] {
    let per_row = (K / 8) as u32;
    let mut x = [[0u32; 4]; NA];
    let mut it = 0;
    while it < NA {
        __unroll_config::<0>();
        let i = it as u32 * THREADS + threadIdx_x();
        let (r, c) = (i / per_row, (i % per_row) * 8);
        if m0 + r < m {
            // SAFETY: row m0 + r < m, c + 8 <= K
            x[it] = unsafe { ldg_u4(a.add(((m0 + r) * lda + c) as usize)) };
        }
        it += 1;
    }
    x
}

/// Stores [`load_tile`]'s registers into the tile
///
/// # Safety
/// Every thread calls it between barriers
#[inline(always)]
unsafe fn store_tile<const K: usize, const LD: usize, const NA: usize>(dst: SharedTile<LD>, x: [[u32; 4]; NA]) {
    let per_row = (K / 8) as u32;
    let mut it = 0;
    while it < NA {
        __unroll_config::<0>();
        let i = it as u32 * THREADS + threadIdx_x();
        let (r, c) = (i / per_row, (i % per_row) * 8);
        // SAFETY: (r, c) is this thread's alone; r < 64 and c + 8 <= K
        unsafe { sts4(dst.at(r as usize, c as usize), x[it]) };
        it += 1;
    }
}

/// The warp's 16 x 64 product over depth `K`
///
/// # Safety
/// Both tiles are complete (after a barrier)
#[inline(always)]
unsafe fn product<const K: usize, const LD: usize>(a: SharedTile<LD>, w: SharedTile<LD>) -> [[f32; 4]; NJ] {
    let warp = threadIdx_x() as usize / 32;
    let (wr, wc) = ((warp % 4) * 16, (warp / 4) * 64);
    let mut acc = [[0.0f32; 4]; NJ];
    let mut kk = 0;
    while kk < K / 16 {
        __unroll_config::<0>();
        // SAFETY: warp-uniform offsets inside the tiles
        unsafe {
            let fa = a.frag_a(wr, kk * 16);
            let mut j = 0;
            while j < NJ {
                __unroll_config::<0>();
                let fb = w.frag_b_pair_nk(wc + j * 8, kk * 16);
                acc[j] = mma(acc[j], fa, fb[0]);
                acc[j + 1] = mma(acc[j + 1], fa, fb[1]);
                j += 2;
            }
        }
        kk += 1;
    }
    acc
}

/// The persistent loop: stages the slice (`wrows` valid rows of `w`), then for each of the block's row tiles calls
/// `epi(m0, acc)` with the warp's accumulators of rows from `m0`. `epi` may call barriers: every thread runs it
///
/// # Safety
/// Every thread of the block calls it; `a` holds `m` rows of stride `lda` and `w` `wrows` rows of `K`, both 16-byte
/// aligned; the dynamic pool holds `smem_bytes(K)`
#[inline(always)]
unsafe fn persist<const K: usize, const LD: usize, const NA: usize>(
    a: *const Bf16,
    lda: u32,
    m: u32,
    w: *const Bf16,
    wrows: u32,
    mut epi: impl FnMut(u32, [[f32; 4]; NJ]),
) {
    let (ws, as_, _) = tiles::<LD>();
    let tiles = m.div_ceil(TM);
    let mut t = blockIdx_x();
    if t >= tiles {
        return;
    }
    // SAFETY: the caller's contract
    unsafe { stage_w::<K, LD>(ws, w, wrows) };
    let mut x = unsafe { load_tile::<K, NA>(a, lda, t * TM, m) };
    while t < tiles {
        // the previous tile's readers are done with the tile (and the slice's stores are visible)
        thread::sync_threads();
        // SAFETY: between barriers
        unsafe { store_tile::<K, LD, NA>(as_, x) };
        thread::sync_threads();
        let next = t + gridDim_x();
        if next < tiles {
            // issued now, consumed next iteration: the loads overlap this tile's product
            x = unsafe { load_tile::<K, NA>(a, lda, next * TM, m) };
        }
        let acc = unsafe { product::<K, LD>(as_, ws) };
        epi(t * TM, acc);
        t = next;
    }
}

/// This thread's place in the warp's accumulators: `(row within the tile of r = 0, first column within the slice
/// of j = 0)`; accumulator `[j][2 r + e]` is row `+ 8 r`, column `+ 8 j + e`
#[inline(always)]
fn place() -> (u32, u32) {
    let (warp, lane) = (threadIdx_x() / 32, threadIdx_x() % 32);
    ((warp % 4) * 16 + lane / 4, (warp / 4) * 64 + 2 * (lane % 4))
}

/// The four lanes of a row (lanes `4 g..4 g + 4`) summed
#[inline(always)]
fn quad_sum(mut v: f32) -> f32 {
    v += warp::shuffle_xor_f32(v, 1);
    v += warp::shuffle_xor_f32(v, 2);
    v
}

/// The two rows' sums of this thread's values (`[j][2 r + e]`), over its quad
#[inline(always)]
fn row_sums(v: [[f32; 4]; NJ]) -> [f32; 2] {
    let mut t = [0.0f32; 2];
    let mut j = 0;
    while j < NJ {
        __unroll_config::<0>();
        t[0] += v[j][0] + v[j][1];
        t[1] += v[j][2] + v[j][3];
        j += 1;
    }
    [quad_sum(t[0]), quad_sum(t[1])]
}

/// Whole-row totals of the two column halves' sums `part` (rows `r0`, `r0 + 8`), through `red` round `slot`
///
/// # Safety
/// Every thread of the block calls it; `red` holds 256 floats no one else touches until the loop's next barrier
#[inline(always)]
unsafe fn exchange(red: *mut f32, slot: usize, half: usize, lane: u32, r0: u32, part: [f32; 2]) -> [f32; 2] {
    if lane % 4 == 0 {
        // SAFETY: inside red; each (slot, half, row) has one writer
        unsafe {
            *red.add(slot * 128 + half * 64 + r0 as usize) = part[0];
            *red.add(slot * 128 + half * 64 + r0 as usize + 8) = part[1];
        }
    }
    thread::sync_threads();
    let at = slot * 128 + r0 as usize;
    // SAFETY: written before the barrier
    unsafe { [*red.add(at) + *red.add(at + 64), *red.add(at + 8) + *red.add(at + 72)] }
}

/// The checks the kernels share
#[inline(always)]
fn ws_ok(a: &[Bf16], w: &[Bf16], bias: &[f32], m: u32, n: u32, k: u32, lda: u32) -> bool {
    (lda >= k)
        & (lda % 8 == 0)
        & (m as usize * lda as usize <= a.len())
        & (n as usize * k as usize <= w.len())
        & (n as usize <= bias.len())
        & (blockIdx_y() * TN < n)
}

/// `y = act(a @ w^T + bias)` in bf16, depth 128: `out` `[m, ldo]` bf16 pairs, columns `[0, n)`
///
/// Grid `(blocks per slice, n / 128)`, block 256, `smem_bytes(128)` dynamic shared memory
#[kernel]
#[launch_contract(domain = 2, block = (256, 1, 1), dynamic_shared = 52224)]
#[allow(clippy::too_many_arguments)]
pub fn sm_ws_linear(
    mut out: DisjointSlice<u32>,
    a: &[Bf16],
    w: &[Bf16],
    bias: &[f32],
    m: u32,
    n: u32,
    lda: u32,
    ldo: u32,
    act: u32,
) {
    require(
        ws_ok(a, w, bias, m, n, 128, lda)
            & (n % TN == 0)
            & (ldo % 2 == 0)
            & (n <= ldo)
            & (m as usize * ldo as usize / 2 <= out.len()),
    );
    let n0 = blockIdx_y() * TN;
    let (r0, c0) = place();
    let op = out.as_mut_ptr();
    // SAFETY: checked above; the slice's rows [n0, n0 + 128) lie inside w
    unsafe {
        persist::<128, 136, 4>(a.as_ptr(), lda, m, w.as_ptr().add(n0 as usize * 128), TN, |m0, acc| {
            let mut j = 0;
            while j < NJ {
                __unroll_config::<0>();
                let col = n0 + c0 + 8 * j as u32;
                let (b0, b1) = (*bias.get_unchecked(col as usize), *bias.get_unchecked(col as usize + 1));
                let mut r = 0;
                while r < 2 {
                    __unroll_config::<0>();
                    let row = m0 + r0 + 8 * r as u32;
                    if row < m {
                        let y0 = activate(round_bf16(acc[j][2 * r] + b0), act);
                        let y1 = activate(round_bf16(acc[j][2 * r + 1] + b1), act);
                        *op.add((row as usize * ldo as usize + col as usize) / 2) = pack_bf16(y0, y1);
                    }
                    r += 1;
                }
                j += 1;
            }
        });
    }
}

/// The attention projection, depth 128: slice `s` (from `s0 = n0 / 128`, the grid's y adding to it) is q, k or v
/// (`w`'s rows `[128 s, 128 s + 128)`), each split into 4 heads of 32: q into `[B, 4, ntok, 32]`, k and v into
/// `[B, 4, ldk, 32]` at key `koff + token`, bf16 pairs
///
/// Grid `(blocks per slice, 3 - s0)`, block 256, `smem_bytes(128)` dynamic shared memory
#[kernel]
#[launch_contract(domain = 2, block = (256, 1, 1), dynamic_shared = 52224)]
#[allow(clippy::too_many_arguments)]
pub fn sm_ws_qkv(
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
    s0: u32,
) {
    let s = s0 + blockIdx_y();
    let decisions = (m / ntok.max(1)) as usize;
    require(
        ws_ok(a, w, bias, m, 384, 128, 128)
            & (s < 3)
            & (ntok > 0)
            & (m % ntok.max(1) == 0)
            & (koff + ntok <= ldk)
            & ((s0 >= 1) | (decisions * 4 * ntok as usize * 16 <= q.len()))
            & (decisions * 4 * ldk as usize * 16 <= k.len())
            & (decisions * 4 * ldk as usize * 16 <= v.len()),
    );
    let n0 = s * TN;
    let (r0, c0) = place();
    let (qp, kp, vp) = (q.as_mut_ptr(), k.as_mut_ptr(), v.as_mut_ptr());
    // SAFETY: checked above; rows [n0, n0 + 128) lie inside w's 384
    unsafe {
        persist::<128, 136, 4>(a.as_ptr(), 128, m, w.as_ptr().add(n0 as usize * 128), TN, |m0, acc| {
            let mut j = 0;
            while j < NJ {
                __unroll_config::<0>();
                let c = c0 + 8 * j as u32;
                let (b0, b1) = (*bias.get_unchecked((n0 + c) as usize), *bias.get_unchecked((n0 + c) as usize + 1));
                let (h, d) = (c / 32, c % 32);
                let mut r = 0;
                while r < 2 {
                    __unroll_config::<0>();
                    let row = m0 + r0 + 8 * r as u32;
                    if row < m {
                        let (b, tok) = (row / ntok, row % ntok);
                        let p = pack_bf16(round_bf16(acc[j][2 * r] + b0), round_bf16(acc[j][2 * r + 1] + b1));
                        if s == 0 {
                            *qp.add((((b * 4 + h) * ntok + tok) * 16 + d / 2) as usize) = p;
                        } else {
                            let at = ((b * 4 + h) as usize * ldk as usize + (koff + tok) as usize) * 16 + d as usize / 2;
                            if s == 1 {
                                *kp.add(at) = p;
                            } else {
                                *vp.add(at) = p;
                            }
                        }
                    }
                    r += 1;
                }
                j += 1;
            }
        });
    }
}

/// The residual Linears' epilogue: adds `bf16(acc + bias)` into the stream rows (bf16 pairs `rb` `[m, 64]`, or fp32
/// `rf` `[m, 128]` plus `vec` when not empty, whichever `rb` is not), then, when `gamma` is not empty, LayerNorms
/// the new rows into `ln` (bf16 pairs `[m, 64]`)
///
/// # Safety
/// Every thread of the block calls it, with checked buffers; the row tile is free (its product is done) until the
/// next barrier of the loop
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn residual_ln(
    rb: *mut u32,
    rf: *mut f32,
    from_f32: bool,
    vec: &[f32],
    ln: *mut u32,
    gamma: &[f32],
    beta: &[f32],
    bias: &[f32],
    red: *mut f32,
    m: u32,
    m0: u32,
    acc: [[f32; 4]; NJ],
) {
    let (r0, c0) = place();
    let with_vec = !vec.is_empty();
    // the new stream values, as the next LayerNorm reads them
    let mut s = [[0.0f32; 4]; NJ];
    let mut j = 0;
    while j < NJ {
        __unroll_config::<0>();
        let col = c0 + 8 * j as u32;
        // SAFETY: col + 1 < 128 <= bias.len(), vec.len() when not empty
        let (b0, b1) = unsafe { (*bias.get_unchecked(col as usize), *bias.get_unchecked(col as usize + 1)) };
        let mut r = 0;
        while r < 2 {
            __unroll_config::<0>();
            let row = m0 + r0 + 8 * r as u32;
            let (y0, y1) = (round_bf16(acc[j][2 * r] + b0), round_bf16(acc[j][2 * r + 1] + b1));
            if row < m {
                // SAFETY: row < m, inside the checked stream; this pair is this thread's alone
                unsafe {
                    if from_f32 {
                        let at = rf.add(row as usize * C as usize + col as usize);
                        let (mut v0, mut v1) = (*at + y0, *at.add(1) + y1);
                        if with_vec {
                            v0 += *vec.get_unchecked(col as usize);
                            v1 += *vec.get_unchecked(col as usize + 1);
                        }
                        *at = v0;
                        *at.add(1) = v1;
                        (s[j][2 * r], s[j][2 * r + 1]) = (v0, v1);
                    } else {
                        let at = rb.add((row as usize * C as usize + col as usize) / 2);
                        let p = *at;
                        let v = pack_bf16(lo_f32(p) + y0, hi_f32(p) + y1);
                        *at = v;
                        (s[j][2 * r], s[j][2 * r + 1]) = (lo_f32(v), hi_f32(v));
                    }
                }
            }
            r += 1;
        }
        j += 1;
    }
    if gamma.is_empty() {
        return;
    }

    // row sums over this thread's 16 values per row, its quad, then the two column halves through shared memory
    // (the row tile's bytes: every warp's product is done after this barrier)
    thread::sync_threads();
    let half = (threadIdx_x() / 128) as usize;
    let lane = threadIdx_x() % 32;
    // SAFETY: after the barrier above; every thread calls it
    let sums = unsafe { exchange(red, 0, half, lane, r0, row_sums(s)) };
    let mean = [sums[0] / C as f32, sums[1] / C as f32];
    let mut d = [[0.0f32; 4]; NJ];
    let mut j = 0;
    while j < NJ {
        __unroll_config::<0>();
        let mut e = 0;
        while e < 4 {
            __unroll_config::<0>();
            d[j][e] = s[j][e] - mean[e / 2];
            e += 1;
        }
        j += 1;
    }
    let mut sq = [[0.0f32; 4]; NJ];
    let mut j = 0;
    while j < NJ {
        __unroll_config::<0>();
        let mut e = 0;
        while e < 4 {
            __unroll_config::<0>();
            sq[j][e] = d[j][e] * d[j][e];
            e += 1;
        }
        j += 1;
    }
    // SAFETY: as above
    let var = unsafe { exchange(red, 1, half, lane, r0, row_sums(sq)) };
    let rstd = [math::rsqrtf(var[0] / C as f32 + EPS), math::rsqrtf(var[1] / C as f32 + EPS)];
    let mut j = 0;
    while j < NJ {
        __unroll_config::<0>();
        let col = (c0 + 8 * j as u32) as usize;
        // SAFETY: col + 1 < 128 <= gamma.len(), beta.len()
        let (g0, g1, e0, e1) = unsafe {
            (*gamma.get_unchecked(col), *gamma.get_unchecked(col + 1), *beta.get_unchecked(col), *beta.get_unchecked(col + 1))
        };
        let mut r = 0;
        while r < 2 {
            __unroll_config::<0>();
            let row = m0 + r0 + 8 * r as u32;
            if row < m {
                let y0 = g0 * (rstd[r] * d[j][2 * r]) + e0;
                let y1 = g1 * (rstd[r] * d[j][2 * r + 1]) + e1;
                // SAFETY: row < m, inside the checked output; this pair is this thread's alone
                unsafe { *ln.add((row as usize * C as usize + col) / 2) = pack_bf16(y0, y1) };
            }
            r += 1;
        }
        j += 1;
    }
}

macro_rules! residual_kernel {
    ($name:ident, $k:literal, $ld:literal, $na:literal, $smem:literal) => {
        /// The residual Linear (128 columns) of depth
        #[doc = stringify!($k)]
        /// : the stream rows (`rb` bf16 pairs `[m, 64]`, or `rf` fp32 `[m, 128]` plus `vec` when not empty) gain
        /// `bf16(a @ w^T + bias)`, then, when `gamma` is not empty, the next LayerNorm of the new rows goes to `ln`
        ///
        /// Grid `(blocks, 1)`, block 256, `smem_bytes(K)` dynamic shared memory
        #[kernel]
        #[launch_contract(domain = 2, block = (256, 1, 1), dynamic_shared = $smem)]
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
            mut rb: DisjointSlice<u32>,
            mut rf: DisjointSlice<f32>,
            mut ln: DisjointSlice<u32>,
            a: &[Bf16],
            w: &[Bf16],
            bias: &[f32],
            vec: &[f32],
            gamma: &[f32],
            beta: &[f32],
            m: u32,
            lda: u32,
        ) {
            let from_f32 = rb.len() == 0;
            let normed = !gamma.is_empty();
            require(
                ws_ok(a, w, bias, m, 128, $k, lda)
                    & (from_f32 | (m as usize * 64 <= rb.len()))
                    & (!from_f32 | (m as usize * 128 <= rf.len()))
                    & (vec.is_empty() | (vec.len() >= 128))
                    & (!normed | ((gamma.len() >= 128) & (beta.len() >= 128) & (m as usize * 64 <= ln.len()))),
            );
            let (_, _, red) = tiles::<$ld>();
            let (rbp, rfp, lnp) = (rb.as_mut_ptr(), rf.as_mut_ptr(), ln.as_mut_ptr());
            // SAFETY: checked above
            unsafe {
                persist::<$k, $ld, $na>(a.as_ptr(), lda, m, w.as_ptr(), TN, |m0, acc| {
                    residual_ln(rbp, rfp, from_f32, vec, lnp, gamma, beta, bias, red, m, m0, acc);
                });
            }
        }
    };
}

// dynamic shared memory: smem_bytes(128), smem_bytes(256)
residual_kernel!(sm_ws_res128, 128, 136, 4, 52224);
residual_kernel!(sm_ws_res256, 256, 264, 8, 101376);
