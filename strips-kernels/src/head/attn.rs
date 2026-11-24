//! Fused multi-head attention for the head's transformer (FlashAttention-2 style): head dim 32, 4 heads, bf16
//! operands, fp32 softmax state, a per-decision key mask, no score matrix in global memory
//!
//! A block owns 64 query rows (4 warps of 16) of one (decision, head) and walks the keys in tiles of 64. QK^T runs on
//! bf16 `mma.sync` m16n8k16 with fp32 accumulation, the softmax in fp32 with an online max and `ex2` with the scale
//! folded into its FMA, and P @ V on bf16 `mma` with P rounded to nearest-even bf16 (as PyTorch's memory-efficient
//! SDPA kernel feeds its second GEMM) and fp32 accumulation. Ported from dinov2rs's `attn_bf16_d64` with the head
//! dim halved, q/k/v read from head-major buffers, and the key mask: a byte per key per decision, combined with
//! `key < nk`. A tile with no valid key is skipped whole; a tile of only valid keys takes the unmasked softmax

use core::ptr::addr_of_mut;

use cuda_device::float::{ex2_approx_ftz_f32, fma_rn_f32};
use cuda_device::thread::{__unroll_config, blockIdx_x, blockIdx_y, blockIdx_z, gridDim_z, threadIdx_x};
use cuda_device::{DisjointSlice, SharedArray, kernel, launch_contract, thread, warp};

use super::mma::{FragA, SharedTile, mma, sts4};
use super::util::{Bf16, ldg_u4, math, pack_bf16, require};

/// Head dim
const D: usize = 32;
/// Heads
const HEADS: u32 = 4;
const WARPS: usize = 4;
const THREADS: usize = 32 * WARPS;
/// Query rows per block
const BM: usize = 16 * WARPS;
/// Keys per tile
const BN: usize = 64;
/// Shared rows padded by 8 bf16 (16 bytes), which keeps the `ldmatrix` loads bank-conflict free
const LD: usize = D + 8;
/// A fragments per query row band: one per 16 of the head dim
const KK: usize = D / 16;
/// Output accumulators per row band: one per 8 of the head dim
const DJ: usize = D / 8;
/// Score fragments per key tile: one per 8 keys
const SJ: usize = BN / 8;
/// 16-byte chunks each thread stages per 64-row tile
const STAGE_ITERS: usize = BM * D / 8 / THREADS;

#[inline(always)]
fn ex2(x: f32) -> f32 {
    ex2_approx_ftz_f32(x)
}

/// Stages rows `[0, 64)` of a `[_, 32]` bf16 matrix starting at `src`, rows from `valid` on as zeros
///
/// # Safety
/// Every thread of the 128-thread block calls it between barriers; rows below `valid` lie in bounds, and `src` is
/// 16-byte aligned
#[inline(always)]
unsafe fn stage(dst: SharedTile<LD>, src: *const Bf16, valid: u32) {
    let mut x = [[0u32; 4]; STAGE_ITERS];
    for it in 0..STAGE_ITERS {
        __unroll_config::<0>();
        let i = (it * THREADS) as u32 + threadIdx_x();
        let (r, c) = (i / (D as u32 / 8), (i % (D as u32 / 8)) * 8);
        if r < valid {
            // SAFETY: r < valid and c + 8 <= 32; the offset is a multiple of 8 elements, so 16-byte aligned
            x[it] = unsafe { ldg_u4(src.add((r * D as u32 + c) as usize)) };
        }
    }
    for it in 0..STAGE_ITERS {
        __unroll_config::<0>();
        let i = (it * THREADS) as u32 + threadIdx_x();
        let (r, c) = (i / (D as u32 / 8), (i % (D as u32 / 8)) * 8);
        // SAFETY: (r, c) is this thread's alone and r < 64; LD and c are multiples of 8
        unsafe { sts4(dst.at(r as usize, c as usize), x[it]) };
    }
}

/// Multi-head attention of `nq` queries over the first `nk` keys, per decision and head
///
/// `q` is `[B, 4, nq, 32]`, `k` and `v` `[B, 4, ldk, 32]`, `keymask` `[B, ldk]` (nonzero: the key may be
/// attended), all bf16 but the mask. `o` is the merged-heads output `[B, nq, 128]` bf16 as `u32` pairs (`[B, nq,
/// 64]` slots). `scale_log2 = log2(e) / sqrt(32)`. Keys from `nk` on are masked whatever the mask says, but key rows
/// up to the end of the last tile are read, so `ldk` is a multiple of 64 and those rows hold finite values
///
/// Grid `(ceil(nq / 64), 4, B)`, block 128: block `(x, h, b)` computes query rows `[64x, 64x + 64)` of decision `b`,
/// head `h`. A warp whose rows all lie past `nq` stages tiles with the others but computes nothing. Every query row
/// needs at least one valid key, or its output is zero
#[kernel]
#[launch_contract(domain = 3, block = (128, 1, 1))]
pub fn sm_attn(
    mut o: DisjointSlice<u32>,
    q: &[Bf16],
    k: &[Bf16],
    v: &[Bf16],
    keymask: &[u8],
    nq: u32,
    nk: u32,
    ldk: u32,
    scale_log2: f32,
) {
    static mut QS: SharedArray<u16, { BM * LD }, 16> = SharedArray::UNINIT;
    static mut KS: SharedArray<u16, { BN * LD }, 16> = SharedArray::UNINIT;
    static mut VS: SharedArray<u16, { BN * LD }, 16> = SharedArray::UNINIT;
    // SAFETY: each array is this kernel's own, 16-byte aligned and 64 rows of LD
    let (qs, ks, vs) = unsafe {
        (
            SharedTile::<LD>::new(addr_of_mut!(QS) as *mut u16),
            SharedTile::<LD>::new(addr_of_mut!(KS) as *mut u16),
            SharedTile::<LD>::new(addr_of_mut!(VS) as *mut u16),
        )
    };

    let (b, h, q0) = (blockIdx_z(), blockIdx_y(), blockIdx_x() * BM as u32);
    let nb = gridDim_z() as usize;
    // everything the grid reads or writes, checked at once: one uniform branch to one trap
    require(
        (h < HEADS)
            & (q0 < nq)
            & (nk <= ldk)
            & (ldk % BN as u32 == 0)
            & (nb * HEADS as usize * nq as usize * D <= q.len())
            & (nb * HEADS as usize * ldk as usize * D <= k.len())
            & (nb * HEADS as usize * ldk as usize * D <= v.len())
            & (nb * ldk as usize <= keymask.len())
            & (nb * nq as usize * (HEADS as usize * D / 2) <= o.len()),
    );
    let nqv = (nq - q0).min(BM as u32);

    let (warp, lane) = (threadIdx_x() as usize / 32, threadIdx_x() as usize % 32);
    let (g, t, wr) = (lane / 4, lane % 4, warp * 16);
    let bh = (b * HEADS + h) as usize;
    let qp = q.as_ptr().wrapping_add((bh * nq as usize + q0 as usize) * D);
    let kp = k.as_ptr().wrapping_add(bh * ldk as usize * D);
    let vp = v.as_ptr().wrapping_add(bh * ldk as usize * D);
    let mp = keymask.as_ptr().wrapping_add(b as usize * ldk as usize);

    // SAFETY: rows [q0, q0 + nqv) of this (decision, head) are inside the checked rows; the barrier orders the
    // stores before the fragment loads
    unsafe { stage(qs, qp, nqv) };
    thread::sync_threads();
    let mut qa = [FragA::ZERO; KK];
    let mut kk = 0;
    while kk < KK {
        __unroll_config::<0>();
        // SAFETY: the whole warp runs this with warp-uniform offsets inside the tile
        qa[kk] = unsafe { qs.frag_a(wr, kk * 16) };
        kk += 1;
    }
    // a warp past the last query row skips the math (warp-uniform), not the barriers
    let active = (wr as u32) < nqv;

    let mut acc = [[0.0f32; 4]; DJ];
    let (mut m, mut l) = ([f32::NEG_INFINITY; 2], [0.0f32; 2]);
    let mut k0 = 0u32;
    while k0 < nk {
        // this lane's two keys, 2 lane and 2 lane + 1 of the tile; every warp reads the same 64 bytes, so the votes
        // below agree across the block
        let key = k0 + 2 * lane as u32;
        // SAFETY: key + 1 < k0 + 64 <= ldk (ldk is a multiple of 64 and nk <= ldk), inside this decision's row of
        // the mask; key is even, so the u16 read is aligned
        let pair = unsafe { *(mp.add(key as usize) as *const u16) };
        let bits = ((((pair & 0xff) != 0) & (key < nk)) as u32) | (((((pair >> 8) != 0) & (key + 1 < nk)) as u32) << 1);
        let any = warp::ballot(bits != 0) != 0;
        let all = warp::ballot(bits == 3) == u32::MAX;
        if !any {
            k0 += BN as u32;
            continue;
        }
        thread::sync_threads();
        // SAFETY: rows [k0, k0 + 64) lie below ldk, inside the checked rows; the barriers on either side keep the
        // previous tile's readers and this tile's writers apart
        unsafe {
            stage(ks, kp.wrapping_add(k0 as usize * D), BN as u32);
            stage(vs, vp.wrapping_add(k0 as usize * D), BN as u32);
        }
        thread::sync_threads();

        if active {
            let mut s = [[0.0f32; 4]; SJ];
            let mut j = 0;
            while j < SJ {
                __unroll_config::<0>();
                let mut kk = 0;
                while kk < KK {
                    __unroll_config::<0>();
                    // SAFETY: as for the q fragments
                    unsafe {
                        let kb = ks.frag_b_pair_nk(j * 8, kk * 16);
                        s[j] = mma(s[j], qa[kk], kb[0]);
                        s[j + 1] = mma(s[j + 1], qa[kk], kb[1]);
                    }
                    kk += 1;
                }
                j += 2;
            }

            let mut corr = [0.0f32; 2];
            if all {
                // only valid keys: no masking or infinity guards, and the scale folds into the exponent's FMA. A
                // row max of -inf only comes before the first valid tile, where corr = ex2(-inf) = 0 is right
                let mut mx = [f32::NEG_INFINITY; 2];
                let mut j = 0;
                while j < SJ {
                    __unroll_config::<0>();
                    mx[0] = math::fmaxf(mx[0], math::fmaxf(s[j][0], s[j][1]));
                    mx[1] = math::fmaxf(mx[1], math::fmaxf(s[j][2], s[j][3]));
                    j += 1;
                }
                let mut r = 0;
                while r < 2 {
                    __unroll_config::<0>();
                    mx[r] = math::fmaxf(mx[r], warp::shuffle_xor_f32(mx[r], 1));
                    mx[r] = math::fmaxf(mx[r], warp::shuffle_xor_f32(mx[r], 2));
                    mx[r] = math::fmaxf(m[r], mx[r] * scale_log2);
                    corr[r] = ex2(m[r] - mx[r]);
                    m[r] = mx[r];
                    l[r] *= corr[r];
                    r += 1;
                }
                let mut j = 0;
                while j < SJ {
                    __unroll_config::<0>();
                    let mut e = 0;
                    while e < 4 {
                        __unroll_config::<0>();
                        let p = ex2(fma_rn_f32(s[j][e], scale_log2, -m[e / 2]));
                        s[j][e] = p;
                        l[e / 2] += p;
                        e += 1;
                    }
                    j += 1;
                }
            } else {
                // a mixed tile: score column j * 8 + 2 t + (e & 1) is key 2 (4 j + t) + (e & 1), whose bit lane 4 j
                // + t holds
                let mut mx = m;
                let mut j = 0;
                while j < SJ {
                    __unroll_config::<0>();
                    let owner = warp::shuffle(bits, (j * 4 + t) as u32);
                    let mut e = 0;
                    while e < 4 {
                        __unroll_config::<0>();
                        let x = if (owner >> (e & 1)) & 1 != 0 { s[j][e] * scale_log2 } else { f32::NEG_INFINITY };
                        s[j][e] = x;
                        mx[e / 2] = math::fmaxf(mx[e / 2], x);
                        e += 1;
                    }
                    j += 1;
                }
                let mut r = 0;
                while r < 2 {
                    __unroll_config::<0>();
                    mx[r] = math::fmaxf(mx[r], warp::shuffle_xor_f32(mx[r], 1));
                    mx[r] = math::fmaxf(mx[r], warp::shuffle_xor_f32(mx[r], 2));
                    corr[r] = if mx[r] == f32::NEG_INFINITY { 1.0 } else { ex2(m[r] - mx[r]) };
                    m[r] = mx[r];
                    l[r] *= corr[r];
                    r += 1;
                }
                let mut j = 0;
                while j < SJ {
                    __unroll_config::<0>();
                    let mut e = 0;
                    while e < 4 {
                        __unroll_config::<0>();
                        let p = if m[e / 2] == f32::NEG_INFINITY { 0.0 } else { ex2(s[j][e] - m[e / 2]) };
                        s[j][e] = p;
                        l[e / 2] += p;
                        e += 1;
                    }
                    j += 1;
                }
            }
            let mut dj = 0;
            while dj < DJ {
                __unroll_config::<0>();
                acc[dj][0] *= corr[0];
                acc[dj][1] *= corr[0];
                acc[dj][2] *= corr[1];
                acc[dj][3] *= corr[1];
                dj += 1;
            }

            // P @ V: P rounded to bf16, accumulated in fp32 straight into the totals
            let mut kk = 0;
            while kk < BN / 16 {
                __unroll_config::<0>();
                let pa = FragA::from_acc_pair(s[2 * kk], s[2 * kk + 1]);
                let mut dj = 0;
                while dj < DJ {
                    __unroll_config::<0>();
                    // SAFETY: as for QK^T
                    unsafe {
                        let vb = vs.frag_b_pair_kn(kk * 16, dj * 8);
                        acc[dj] = mma(acc[dj], pa, vb[0]);
                        acc[dj + 1] = mma(acc[dj + 1], pa, vb[1]);
                    }
                    dj += 2;
                }
                kk += 1;
            }
        }
        k0 += BN as u32;
    }

    let mut r = 0;
    while r < 2 {
        __unroll_config::<0>();
        l[r] += warp::shuffle_xor_f32(l[r], 1);
        l[r] += warp::shuffle_xor_f32(l[r], 2);
        // a row with no valid key keeps l = 0 and writes zeros
        l[r] = if l[r] > 0.0 { 1.0 / l[r] } else { 0.0 };
        r += 1;
    }

    // each thread stores its own two rows at slots h * 16 + 4 dj + t, which no other thread of any block writes,
    // inside the rows checked up front
    let o_ld = HEADS as usize * D / 2;
    let op = o.as_mut_ptr().wrapping_add((b as usize * nq as usize + q0 as usize) * o_ld + h as usize * (D / 2));
    let mut r = 0;
    while r < 2 {
        __unroll_config::<0>();
        let row = (wr + g + r * 8) as u32;
        if row < nqv {
            let out = op.wrapping_add(row as usize * o_ld);
            let mut dj = 0;
            while dj < DJ {
                __unroll_config::<0>();
                let (a, c) = (acc[dj][2 * r] * l[r], acc[dj][2 * r + 1] * l[r]);
                // SAFETY: row < nqv keeps the row inside the checked rows, and slot 4 dj + t < 16 inside this
                // head's columns
                unsafe { *out.add(dj * 4 + t) = pack_bf16(a, c) };
                dj += 1;
            }
        }
        r += 1;
    }
}
