//! The sector tokens in one kernel: the fixed sampling operator's gather (host `gather.rs`) straight into the
//! shared A tile of the sector projection, so the projection input never goes through DRAM
//!
//! The projection's 520 inputs per token are 8 points of 64 channels plus a support each. The weight comes
//! re-laid-out as `[128, 528]`: columns `64 p..64 p + 64` are point `p`'s channels, `512 + p` its support, 520..528
//! zero. A block takes 64 tokens of one decision and all 128 outputs, in 9 depth chunks: one per point, each
//! gathered into the A tile while the chunk's weight slice is staged, then the 8 supports as a 16-deep tail. Each
//! point's channels are summed in fp32 in entry order and rounded to bf16, as the unfused gather wrote them, so
//! only the GEMM's summation order differs from the unfused path
//!
//! The gather is latency-bound on dependent loads (entry, then cell), so a warp walks 4 rows at once: 8 lanes per
//! row, each loading 8 channels (16 bytes) of a cell per entry, two entries in flight

use core::ptr::addr_of_mut;

use cuda_device::thread::{__unroll_config, blockIdx_x, blockIdx_y, threadIdx_x};
use cuda_device::{DisjointSlice, SharedArray, kernel, launch_contract, thread};

use super::mma::{SharedTile, mma, sts4};
use super::tokens::{CD, CELLS, CH, CP, CW, SECTORS};
use super::util::{Bf16, hi_f32, ldg_u4, lo_f32, pack_bf16, require, round_bf16};

/// Projection depth: 8 points of 64 channels, then 8 supports padded to 16
pub const K_SECTOR: u32 = 528;
/// Threads, token rows and outputs per block
const THREADS: usize = 256;
const TM: usize = 64;
const TN: usize = 128;
/// Points per sector token, channels per point
const POINTS: u32 = 8;
const CH_POINT: usize = 64;
/// Shared rows padded by 8 bf16, which keeps the `ldmatrix` loads bank-conflict free
const LD: usize = CH_POINT + 8;
/// Accumulators per warp: one per 8 of its 64 columns
const NJ: usize = 8;

/// Point `(token, p)`'s operator entries: `table[4 (8 token + p)..]` = first entry, entry count, support (f32 bits)
///
/// # Safety
/// `token < SECTORS`, `p < 8`, and `table` holds `SECTORS * 8` rows of 4
#[inline(always)]
unsafe fn point(table: &[u32], token: u32, p: u32) -> (u32, u32, f32) {
    // SAFETY: the caller's token < SECTORS and p < 8, inside the checked table
    unsafe {
        let d = table.as_ptr().add(((token * POINTS + p) * 4) as usize);
        (*d, *d.add(1), f32::from_bits(*d.add(2)))
    }
}

/// One entry's 8 channels `c` (bf16 pairs) times its weight `w`, added into `v` in the operator's fp32 order
#[inline(always)]
fn accumulate(v: [f32; 8], c: [u32; 4], w: f32) -> [f32; 8] {
    [
        lo_f32(c[0]) * w + v[0],
        hi_f32(c[0]) * w + v[1],
        lo_f32(c[1]) * w + v[2],
        hi_f32(c[1]) * w + v[3],
        lo_f32(c[2]) * w + v[4],
        hi_f32(c[2]) * w + v[5],
        lo_f32(c[3]) * w + v[6],
        hi_f32(c[3]) * w + v[7],
    ]
}

/// The `l8`-th 8 channels of `count` entries from `first`, summed: two entries' loads in flight per step
///
/// # Safety
/// `first + count` entries lie inside `entries`, and `cells` is one decision's lattice
#[inline(always)]
unsafe fn gather(cells: *const u32, entries: *const u32, first: u32, count: u32, l8: u32) -> [f32; 8] {
    let lattice = CD * CH * CW;
    let at = |cell: u32| (cell.min(lattice - 1) * CP + 4 * l8) as usize;
    let mut v = [0.0f32; 8];
    let mut e = 0;
    while e + 1 < count {
        // SAFETY: first + e + 1 < first + count; a cell index past the lattice reads its last cell
        unsafe {
            let q = entries.add(((first + e) * 2) as usize);
            let (c0, w0, c1, w1) = (*q, f32::from_bits(*q.add(1)), *q.add(2), f32::from_bits(*q.add(3)));
            let (p0, p1) = (ldg_u4(cells.add(at(c0))), ldg_u4(cells.add(at(c1))));
            v = accumulate(v, p0, w0);
            v = accumulate(v, p1, w1);
        }
        e += 2;
    }
    if e < count {
        // SAFETY: as above
        unsafe {
            let q = entries.add(((first + e) * 2) as usize);
            v = accumulate(v, ldg_u4(cells.add(at(*q))), f32::from_bits(*q.add(1)));
        }
    }
    v
}

/// Stages the weight's columns `[k0, k0 + width)` (`width` 64 or 16) for all 128 outputs
///
/// # Safety
/// Every thread of the block calls it between barriers; `w` holds 128 rows of `K_SECTOR`
#[inline(always)]
unsafe fn stage_w(ws: SharedTile<LD>, w: *const Bf16, k0: u32, width: u32) {
    let per_row = width / 8;
    let mut i = threadIdx_x();
    while i < TN as u32 * per_row {
        __unroll_config::<0>();
        let (n, c) = (i / per_row, (i % per_row) * 8);
        // SAFETY: n < 128 and k0 + c + 8 <= K_SECTOR, 16-byte aligned; (n, c) is this thread's alone
        unsafe { sts4(ws.at(n as usize, c as usize), ldg_u4(w.add((n * K_SECTOR + k0 + c) as usize))) };
        i += THREADS as u32;
    }
}

/// The sector tokens' rows of the context sequence: `bf16(bf16(bf16(gather @ w^T + bias) + tab[token]) + vec)`,
/// as `sm_linear_tok`'s per-token mode writes them, into rows `b * orpd + token` of `out` (`[B, orpd, 128]` bf16 as
/// `u32` pairs). `cells` `[B, 36, 32, 32, 32]` bf16 pairs; `table` `[1404 * 8, 4]` and `entries` `[n, 2]` the
/// operator; `w` `[128, 528]` re-laid-out (module doc); `tab` the cell position table, bf16 pairs; `vec` the cell
/// kind, f32
///
/// Grid `(ceil(1404 / 64), B)`, block 256: warp `w` computes rows `16 (w % 4)..`, columns `64 (w / 4)..`
#[kernel]
#[launch_contract(domain = 2, block = (256, 1, 1))]
#[allow(clippy::too_many_arguments)]
pub fn sm_sector_tokens(
    mut out: DisjointSlice<u32>,
    cells: &[u32],
    table: &[u32],
    entries: &[u32],
    w: &[Bf16],
    bias: &[f32],
    tab: &[u32],
    vec: &[f32],
    batch: u32,
    orpd: u32,
) {
    static mut AS: SharedArray<u16, { TM * LD }, 16> = SharedArray::UNINIT;
    static mut WS: SharedArray<u16, { TN * LD }, 16> = SharedArray::UNINIT;
    // SAFETY: each array is this kernel's own, 16-byte aligned, 64 and 128 rows of LD
    let (as_, ws) = unsafe {
        (SharedTile::<LD>::new(addr_of_mut!(AS) as *mut u16), SharedTile::<LD>::new(addr_of_mut!(WS) as *mut u16))
    };
    let (t0, b) = (blockIdx_x() * TM as u32, blockIdx_y());
    let nb = batch as usize;
    let nent = (entries.len() / 2) as u32;
    require(
        (t0 < SECTORS)
            & (b < batch)
            & (cells.len() >= nb * CELLS)
            & (table.len() >= (SECTORS * POINTS * 4) as usize)
            & (w.len() >= TN * K_SECTOR as usize)
            & (bias.len() >= TN)
            & (tab.len() >= SECTORS as usize * 64)
            & (vec.len() >= TN)
            & (SECTORS <= orpd)
            & (out.len() >= nb * orpd as usize * 64),
    );
    let tid = threadIdx_x();
    let (warp, lane) = (tid / 32, tid % 32);
    let (sub, l8) = (lane / 8, lane % 8);
    // SAFETY: b < batch, inside the checked cells
    let lattice = unsafe { cells.as_ptr().add(b as usize * CELLS) };

    let (wr, wc) = (16 * (warp as usize % 4), 64 * (warp as usize / 4));
    let mut acc = [[0.0f32; 4]; NJ];
    let mut p = 0;
    while p <= POINTS {
        thread::sync_threads();
        let width = if p < POINTS { CH_POINT as u32 } else { 16 };
        // SAFETY: every thread calls it between the barriers, inside the checked weight
        unsafe { stage_w(ws, w.as_ptr(), p * CH_POINT as u32, width) };
        if p < POINTS {
            let mut q = 0;
            while q < 2 {
                __unroll_config::<0>();
                let r = q * 32 + warp * 4 + sub;
                let token = t0 + r;
                let mut v = [0.0f32; 8];
                if token < SECTORS {
                    // SAFETY: token < SECTORS and p < 8; the entry range is clipped to the entries
                    unsafe {
                        let (first, count, _) = point(table, token, p);
                        let count = count.min(nent.saturating_sub(first));
                        v = gather(lattice, entries.as_ptr(), first, count, l8);
                    }
                }
                let c = [pack_bf16(v[0], v[1]), pack_bf16(v[2], v[3]), pack_bf16(v[4], v[5]), pack_bf16(v[6], v[7])];
                // SAFETY: row r < 64, columns 8 l8..8 l8 + 8 are this lane's alone
                unsafe { sts4(as_.at(r as usize, 8 * l8 as usize), c) };
                q += 1;
            }
        } else if tid < 2 * TM as u32 {
            // the supports: row tid / 2, columns 0..8 (one per point) or the zero pad 8..16
            let (r, half) = (tid / 2, tid % 2);
            let token = t0 + r;
            let mut c = [0u32; 4];
            if (half == 0) & (token < SECTORS) {
                let mut i = 0;
                while i < 4 {
                    __unroll_config::<0>();
                    // SAFETY: token < SECTORS and 2 i + 1 < 8
                    let (s0, s1) = unsafe { (point(table, token, 2 * i).2, point(table, token, 2 * i + 1).2) };
                    c[i as usize] = pack_bf16(s0, s1);
                    i += 1;
                }
            }
            // SAFETY: row r < 64, columns 8 half..8 half + 8 are this thread's alone
            unsafe { sts4(as_.at(r as usize, 8 * half as usize), c) };
        }
        thread::sync_threads();

        let mut kk = 0;
        while kk < width as usize / 16 {
            // SAFETY: warp-uniform offsets inside the tiles, after the barrier
            unsafe {
                let fa = as_.frag_a(wr, kk * 16);
                let mut j = 0;
                while j < NJ {
                    __unroll_config::<0>();
                    let fb = ws.frag_b_pair_nk(wc + j * 8, kk * 16);
                    acc[j] = mma(acc[j], fa, fb[0]);
                    acc[j + 1] = mma(acc[j + 1], fa, fb[1]);
                    j += 2;
                }
            }
            kk += 1;
        }
        p += 1;
    }

    let (g, t) = (lane / 4, lane % 4);
    let op = out.as_mut_ptr();
    let mut j = 0;
    while j < NJ {
        __unroll_config::<0>();
        let col = (wc + j * 8) as u32 + 2 * t;
        // SAFETY: col + 1 < 128, inside the checked bias and kind
        let (b0, b1, k0, k1) = unsafe {
            (
                *bias.get_unchecked(col as usize),
                *bias.get_unchecked(col as usize + 1),
                *vec.get_unchecked(col as usize),
                *vec.get_unchecked(col as usize + 1),
            )
        };
        let mut r = 0;
        while r < 2 {
            __unroll_config::<0>();
            let token = t0 + (wr + g as usize) as u32 + 8 * r as u32;
            if token < SECTORS {
                // SAFETY: token < SECTORS <= orpd and b < batch, inside the checked table and output; each pair is
                // this thread's alone
                unsafe {
                    let tv = *tab.get_unchecked((token * 64 + col / 2) as usize);
                    let y0 = round_bf16(round_bf16(acc[j][2 * r] + b0) + lo_f32(tv));
                    let y1 = round_bf16(round_bf16(acc[j][2 * r + 1] + b1) + hi_f32(tv));
                    *op.add(((b * orpd + token) * 64 + col / 2) as usize) = pack_bf16(y0 + k0, y1 + k1);
                }
            }
            r += 1;
        }
        j += 1;
    }
}
