//! The strip model's crop CNN (`sn_`): the glue around cuDNN's channels-last bf16 convolutions
//!
//! Activations are bf16 channels-last `[B, S, C]` (S = D*H*W), carried as `u32` pairs of bf16 values, low half
//! first, and read or written two or four pairs per thread (`U32x2`, `U32x4`). The kernels repeat the rounding
//! points of PyTorch's bf16 autocast forward:
//! - InstanceNorm takes float32 statistics (biased variance) per sample and channel of the bf16 conv output and
//!   writes `rbf(gamma * (x - mean) * invstd + beta)`;
//! - the residual add and LeakyReLU(0.01) each round to bf16 again;
//! - AvgPool3d and the trilinear upsample sum in float32 and round once;
//! - a conv's bias (stage 0's two convs and the context fuse) is added to its bf16 output and rounded again,
//!   `rbf(rbf(conv) + bias)`, as PyTorch's cuDNN convolution adds it; the fuse's sum is then added to the stage
//!   output and rounded again.
//!
//! The stage-0 biases feed InstanceNorms, which remove a per-channel constant in exact arithmetic, but not their
//! rounding: on the near-constant regions of a crop (the zero padding, uniform input) `rbf(c + bias)` lands on the
//! coarser grid of `c + bias` in the same direction at every voxel, a fixed offset per (position, channel) that
//! survives the norm and that the trained head reads. Statistics are deterministic: fixed-order partial sums per
//! chunk, merged in a fixed order.
//! Index math is 32-bit: the host keeps every buffer under 2^32 elements

use core::ptr::addr_of_mut;

use cuda_device::convert::cvt_bf16x2_f32;
use cuda_device::float::sqrt_rn_f32;
use cuda_device::shared::{DynamicSharedArray, cvta_generic_to_shared_u32};
use cuda_device::thread::{__unroll_config, blockDim_x, blockIdx_x, blockIdx_y, sync_threads, threadIdx_x};
use cuda_device::vector::{F32x2, F32x4, U32x2, U32x4};
use cuda_device::wmma::{ldmatrix_x4_shared_u32, mma_m16n8k16_f32_bf16};
use cuda_device::{DisjointSlice, SharedArray, kernel, launch_contract, thread, warp};

/// The low bf16 of a pair
#[inline(always)]
fn lo(p: u32) -> f32 {
    f32::from_bits(p << 16)
}

/// The high bf16 of a pair
#[inline(always)]
fn hi(p: u32) -> f32 {
    f32::from_bits(p & 0xffff_0000)
}

/// Two floats rounded to nearest-even bf16 and packed, `a` in the low half
#[inline(always)]
fn pack(a: f32, b: f32) -> u32 {
    cvt_bf16x2_f32(a, b)
}

/// `x` as bf16 holds it, round to nearest even
#[inline(always)]
fn rbf(x: f32) -> f32 {
    lo(pack(x, 0.0))
}

/// PyTorch's LeakyReLU(0.01) on a bf16 value: computed in float, rounded to bf16
#[inline(always)]
fn leaky(v: f32) -> f32 {
    if v > 0.0 { v } else { rbf(v * 0.01) }
}

/// The CNN input as the first convs read it: `xb = bf16(x)`, and `pool = bf16(avgpool2(x))` for the first block's
/// skip (PyTorch pools the float32 input, then autocast casts it for the 1x1 conv)
///
/// `x` is `[B, D, H, W]` of (CT, tube) float32 pairs read as `F32x4` (two voxels along W); `xb` is the same grid of
/// bf16 pairs as `U32x2`; `pool` is `[B, D/2, H/2, W/2]` bf16 pairs. One thread per pooled voxel: `n` = its count,
/// `w2` = W/2, `hw2` = (H/2)(W/2), `dhw2` = (D/2)(H/2)(W/2). The pooled sum runs over (z, y, x) in that order, as
/// ATen's avg_pool3d loops
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn sn_input(mut pool: DisjointSlice<u32>, mut xb: DisjointSlice<U32x2>, x: &[F32x4], w2: u32, hw2: u32, dhw2: u32) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let (b, r) = (i / dhw2, i % dhw2);
    let (z, r) = (r / hw2, r % hw2);
    let (y, xx) = (r / w2, r % w2);
    // in F32x4 (and U32x2) units the full grid has rows of W/2 = w2, planes of H * W/2 = 2 hw2 and samples of
    // D * H * W/2 = 4 dhw2
    let base = b * 4 * dhw2 + 2 * z * 2 * hw2 + 2 * y * w2 + xx;
    if xb.len() < x.len() {
        return;
    }
    let Some(mut out) = pool.element_thread32(t) else { return };
    let (mut s0, mut s1) = (0f32, 0f32);
    let mut dz = 0;
    while dz < 2 {
        let mut dy = 0;
        while dy < 2 {
            let k = (base + dz * 2 * hw2 + dy * w2) as usize;
            let [a0, a1, c0, c1] = x[k].0;
            s0 = s0 + a0 + c0;
            s1 = s1 + a1 + c1;
            // SAFETY: `k < x.len() == xb.len()` (the read above is checked), and only this thread covers the two
            // voxels of element `k`
            unsafe { *xb.get_unchecked_mut(k) = U32x2([pack(a0, a1), pack(c0, c1)]) };
            dy += 1;
        }
        dz += 1;
    }
    out.write(pack(s0 * 0.125, s1 * 0.125));
}

/// Partial shifted sums of one spatial chunk of every channel, for InstanceNorm statistics
///
/// `x` is `[B, rows, cols]` bf16 pairs, `cols` even. Grid `(chunks, B)`, block `cols / 2 * lanes` threads (at most
/// 256): thread `t` owns channel pair `t % (cols / 2)` and starts at row `t / (cols / 2)` of the chunk, stepping by
/// `lanes`. Writes `partial[b, chunk, c] = [sum(x - x0), sum((x - x0)^2)]` with the shift `x0 = x[b, 0, c]`, so the
/// sums do not cancel when a channel's mean is large against its spread. The lanes merge in a fixed order
#[kernel]
pub fn sn_stats(mut partial: DisjointSlice<[f32; 2]>, x: &[u32], rows: u32, cols: u32, lanes: u32, chunk_rows: u32) {
    static mut ACC: SharedArray<[f32; 4], 256> = SharedArray::UNINIT;
    let (chunk, b, tid) = (blockIdx_x(), blockIdx_y(), threadIdx_x());
    let pairs = cols / 2;
    let chunks = (rows + chunk_rows - 1) / chunk_rows.max(1);
    // block-uniform, so a block that fails it never reaches a barrier
    if (pairs == 0)
        | (blockDim_x() != pairs * lanes)
        | (blockDim_x() > 256)
        | (((b + 1) * rows * pairs) as usize > x.len())
        | ((((b * chunks + chunk + 1) * cols) as usize) > partial.len())
    {
        return;
    }
    let (cp, lane) = (tid % pairs, tid / pairs);
    let sample = b * rows * pairs;
    let first = x[(sample + cp) as usize];
    let (x0, x1) = (lo(first), hi(first));

    let (mut a0, mut a1, mut q0, mut q1) = (0f32, 0f32, 0f32, 0f32);
    let end = ((chunk + 1) * chunk_rows).min(rows);
    let mut r = chunk * chunk_rows + lane;
    while r < end {
        let p = x[(sample + r * pairs + cp) as usize];
        let (d0, d1) = (lo(p) - x0, hi(p) - x1);
        a0 += d0;
        a1 += d1;
        q0 += d0 * d0;
        q1 += d1 * d1;
        r += lanes;
    }
    let acc = addr_of_mut!(ACC) as *mut [f32; 4];
    // SAFETY: each thread writes its own slot, read only after the barrier
    unsafe { *acc.add(tid as usize) = [a0, a1, q0, q1] };
    sync_threads();
    if lane != 0 {
        return;
    }
    let (mut t0, mut t1, mut u0, mut u1) = (0f32, 0f32, 0f32, 0f32);
    let mut l = 0;
    while l < lanes {
        // SAFETY: `l * pairs + cp < blockDim <= 256`
        let [p0, p1, p2, p3] = unsafe { *acc.add((l * pairs + cp) as usize) };
        t0 += p0;
        t1 += p1;
        u0 += p2;
        u1 += p3;
        l += 1;
    }
    let o = ((b * chunks + chunk) * cols + 2 * cp) as usize;
    // SAFETY: `o + 1 < partial.len()` by the check above, and only thread `cp` of this block writes channels
    // `2cp, 2cp + 1` of this chunk
    unsafe {
        *partial.get_unchecked_mut(o) = [t0, u0];
        *partial.get_unchecked_mut(o + 1) = [t1, u1];
    }
}

/// Per (sample, channel) `[mean, invstd]` from [`sn_stats`]'s partial sums, one thread each, merged in float64
///
/// `stats` is `[B, cols]`; `invstd = 1 / sqrt(var + eps)` with the biased variance, in float32 as PyTorch's float32
/// statistics give it
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn sn_finalize(
    mut stats: DisjointSlice<[f32; 2]>,
    partial: &[[f32; 2]],
    x: &[u32],
    rows: u32,
    cols: u32,
    chunks: u32,
    eps: f32,
) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let (b, c) = (i / cols, i % cols);
    let Some(mut out) = stats.element_thread32(t) else { return };
    let first = x[((b * rows * cols + c) / 2) as usize];
    let shift = if c % 2 == 0 { lo(first) } else { hi(first) } as f64;
    let (mut s, mut q) = (0f64, 0f64);
    let mut k = 0;
    while k < chunks {
        let [p, pq] = partial[((b * chunks + k) * cols + c) as usize];
        s += p as f64;
        q += pq as f64;
        k += 1;
    }
    let n = rows as f64;
    let m = s / n;
    let var = q / n - m * m;
    let var = if var > 0.0 { var as f32 } else { 0.0 };
    out.write([(shift + m) as f32, 1.0 / sqrt_rn_f32(var + eps)]);
}

/// One InstanceNorm output as bf16: `rbf(gamma * (x - mean) * invstd + beta)`, the last multiply-add fused
#[inline(always)]
fn affine(v: f32, st: [f32; 2], gamma: f32, beta: f32) -> f32 {
    rbf((gamma * (v - st[0])) * st[1] + beta)
}

/// InstanceNorm apply in place, with an optional residual and LeakyReLU, four channels per thread:
/// `y = leaky?(rbf(affine(y) + residual?))`
///
/// - `y`: the conv output `[B, rows, cols]` bf16 pairs as `U32x2`, `cols % 4 == 0`; `stats` its [`sn_finalize`]
///   statistics (`[B, cols]`), `w` and `b` its float32 affine;
/// - residual, at most one of: `res`, a bf16 tensor shaped as `y`; or `rx` with `rstats`, `rw`, `rb`, a conv output
///   normalised on load (the skip's 1x1 conv and its InstanceNorm), `rbf(affine(rx))`;
/// - `act != 0` applies LeakyReLU
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn sn_apply(
    mut y: DisjointSlice<U32x2>,
    stats: &[[f32; 2]],
    w: &[f32],
    b: &[f32],
    res: &[U32x2],
    rx: &[U32x2],
    rstats: &[[f32; 2]],
    rw: &[f32],
    rb: &[f32],
    rows: u32,
    cols: u32,
    act: u32,
) {
    let t = thread::index_1d_u32(lc);
    let e = t.get();
    let quads = cols / 4;
    let (pix, q) = (e / quads, e % quads);
    let s = (pix / rows) * cols;
    let c = 4 * q;
    let Some(mut out) = y.element_thread32(t) else { return };
    let [p0, p1] = out.read().0;
    let mut v = [lo(p0), hi(p0), lo(p1), hi(p1)];
    let mut j = 0;
    while j < 4 {
        __unroll_config::<0>();
        let k = (c + j) as usize;
        v[j as usize] = affine(v[j as usize], stats[s as usize + k], w[k], b[k]);
        j += 1;
    }
    if !res.is_empty() {
        let [r0, r1] = res[e as usize].0;
        v = [rbf(v[0] + lo(r0)), rbf(v[1] + hi(r0)), rbf(v[2] + lo(r1)), rbf(v[3] + hi(r1))];
    } else if !rx.is_empty() {
        let [r0, r1] = rx[e as usize].0;
        let r = [lo(r0), hi(r0), lo(r1), hi(r1)];
        let mut j = 0;
        while j < 4 {
            __unroll_config::<0>();
            let k = (c + j) as usize;
            v[j as usize] = rbf(v[j as usize] + affine(r[j as usize], rstats[s as usize + k], rw[k], rb[k]));
            j += 1;
        }
    }
    if act != 0 {
        v = [leaky(v[0]), leaky(v[1]), leaky(v[2]), leaky(v[3])];
    }
    out.write(U32x2([pack(v[0], v[1]), pack(v[2], v[3])]));
}

/// A cuDNN conv's bias in place, as PyTorch adds it to the bf16 output: `y = rbf(y + bias)`, four channels per thread
///
/// `y` is `[B, rows, cols]` bf16 pairs as `U32x2`, `cols % 4 == 0`; `bias` the bf16-rounded bias `[cols]`
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn sn_bias(mut y: DisjointSlice<U32x2>, bias: &[f32], cols: u32) {
    let t = thread::index_1d_u32(lc);
    let c = (4 * (t.get() % (cols / 4))) as usize;
    let Some(mut out) = y.element_thread32(t) else { return };
    let [p0, p1] = out.read().0;
    let v = [lo(p0) + bias[c], hi(p0) + bias[c + 1], lo(p1) + bias[c + 2], hi(p1) + bias[c + 3]];
    out.write(U32x2([pack(v[0], v[1]), pack(v[2], v[3])]));
}

/// AvgPool3d(2, 2) of a bf16 tensor: `out[b, z, y, x] = rbf(sum / 8)` over the 2x2x2 block, summed in float32 in
/// (z, y, x) order as ATen's avg_pool3d loops
///
/// `x` is `[B, D, H, W, C]` bf16 pairs as `U32x2`, `C % 4 == 0`; `out` is `[B, D/2, H/2, W/2, C]`. One thread per
/// output `U32x2`: `quads` = C/4, `w2` = W/2, `hw2` = (H/2)(W/2), `dhw2` = (D/2)(H/2)(W/2)
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn sn_pool(mut out: DisjointSlice<U32x2>, x: &[U32x2], quads: u32, w2: u32, hw2: u32, dhw2: u32) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let (v, q) = (i / quads, i % quads);
    let (b, r) = (v / dhw2, v % dhw2);
    let (z, r) = (r / hw2, r % hw2);
    let (yy, xx) = (r / w2, r % w2);
    let Some(mut o) = out.element_thread32(t) else { return };
    // full-grid strides in U32x2 units: rows of W * quads, planes of H * W * quads, samples of 8 dhw2 quads
    let row = 2 * w2 * quads;
    let plane = 4 * hw2 * quads;
    let base = b * 8 * dhw2 * quads + 2 * z * plane + 2 * yy * row + 2 * xx * quads + q;
    let mut s = [0f32; 4];
    let mut dz = 0;
    while dz < 2 {
        let mut dy = 0;
        while dy < 2 {
            let mut dx = 0;
            while dx < 2 {
                let [p0, p1] = x[(base + dz * plane + dy * row + dx * quads) as usize].0;
                s = [s[0] + lo(p0), s[1] + hi(p0), s[2] + lo(p1), s[3] + hi(p1)];
                dx += 1;
            }
            dy += 1;
        }
        dz += 1;
    }
    o.write(U32x2([pack(s[0] * 0.125, s[1] * 0.125), pack(s[2] * 0.125, s[3] * 0.125)]));
}

/// The source index and weight of a 2x trilinear upsample along one axis (align_corners False), as ATen's
/// `area_pixel_compute_source_index` with scale 1/2: `src = max(0.5 (dst + 0.5) - 0.5, 0)`, `i0 = floor(src)`,
/// `i1 = i0 + 1` clamped to the input, weight `src - i0` on `i1`
#[inline(always)]
fn source(dst: u32, size: u32) -> (u32, u32, f32) {
    let src = 0.5 * (dst as f32 + 0.5) - 0.5;
    let src = if src > 0.0 { src } else { 0.0 };
    let i0 = src as u32;
    let i1 = if i0 + 1 < size { i0 + 1 } else { i0 };
    (i0, i1, src - i0 as f32)
}

/// The CNN output from the last stage's output `x` and the context fuse's 1x1 conv `y` of the context block's
/// output, at the context's size: `cells = rbf(x + rbf(rbf(up(y)) + bias))`, `up` the 2x trilinear upsample
/// (align_corners False) in float32, `bias` the fuse's bias as bf16 holds it
///
/// PyTorch upsamples first and convolves the upsampled tensor; both are linear and the upsample's weights sum to 1,
/// so the order only moves bf16 rounding, and the conv runs on an eighth of the voxels. The bias goes on as PyTorch's
/// cuDNN conv adds it: to the conv output already rounded to bf16, then rounded again. That second rounding snaps the
/// small bias onto the coarser bf16 grid of the conv's values, a per-channel offset of up to a few 1e-3 the trained
/// head expects; adding the bias in float32 before one rounding drops it and tilts the strip heights
///
/// `x` and `cells` are `[B, 2D, 2H, 2W, C]` and `y` `[B, D, H, W, C]`, bf16 pairs as `U32x2`, `C % 4 == 0`. One
/// thread per output `U32x2`: `quads` = C/4, `d`, `h`, `w` the context's size
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
#[allow(clippy::too_many_arguments)]
pub fn sn_upfuse(
    mut cells: DisjointSlice<U32x2>,
    x: &[U32x2],
    y: &[U32x2],
    bias: &[f32],
    quads: u32,
    d: u32,
    h: u32,
    w: u32,
) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let (v, q) = (i / quads, i % quads);
    let (w2, hw2) = (2 * w, 4 * h * w);
    let (b, r) = (v / (2 * d * hw2), v % (2 * d * hw2));
    let (oz, r) = (r / hw2, r % hw2);
    let (oy, ox) = (r / w2, r % w2);
    let Some(mut o) = cells.element_thread32(t) else { return };
    let (z0, z1, lz) = source(oz, d);
    let (y0, y1, ly) = source(oy, h);
    let (x0, x1, lx) = source(ox, w);
    let at = |z: u32, yy: u32, xx: u32| -> [f32; 4] {
        let [p0, p1] = y[((((b * d + z) * h + yy) * w + xx) * quads + q) as usize].0;
        [lo(p0), hi(p0), lo(p1), hi(p1)]
    };
    let (a000, a001, a010, a011) = (at(z0, y0, x0), at(z0, y0, x1), at(z0, y1, x0), at(z0, y1, x1));
    let (a100, a101, a110, a111) = (at(z1, y0, x0), at(z1, y0, x1), at(z1, y1, x0), at(z1, y1, x1));
    let [s0, s1] = x[i as usize].0;
    let skip = [lo(s0), hi(s0), lo(s1), hi(s1)];
    let (mz, my, mx) = (1.0 - lz, 1.0 - ly, 1.0 - lx);
    let mut r = [0f32; 4];
    let mut j = 0;
    while j < 4 {
        __unroll_config::<0>();
        let k = j as usize;
        let near = my * (mx * a000[k] + lx * a001[k]) + ly * (mx * a010[k] + lx * a011[k]);
        let far = my * (mx * a100[k] + lx * a101[k]) + ly * (mx * a110[k] + lx * a111[k]);
        r[k] = rbf(skip[k] + rbf(rbf(mz * near + lz * far) + bias[(4 * q) as usize + k]));
        j += 1;
    }
    o.write(U32x2([pack(r[0], r[1]), pack(r[2], r[3])]));
}

/// Sums `v` over the block (256 threads) in a fixed order through `red` (8 slots); every thread gets the total
#[inline(always)]
fn block_sum256(v: f32, red: *mut f32, tid: u32) -> f32 {
    let mut v = v;
    let mut m = 16;
    while m > 0 {
        v += warp::shuffle_xor_f32(v, m);
        m >>= 1;
    }
    // SAFETY: lane 0 of each warp writes its own slot, every thread reads after the barrier
    unsafe {
        if tid % 32 == 0 {
            *red.add((tid / 32) as usize) = v;
        }
        sync_threads();
        let mut s = 0f32;
        let mut w = 0;
        while w < 8 {
            s += *red.add(w);
            w += 1;
        }
        // the next reduction reuses `red`
        sync_threads();
        s
    }
}

/// Merges per-tile `[mean, M2]` partials of equal `count` into per (sample, channel) `[mean, invstd]`: the mean of
/// the tile means, then `M2 = sum(M2_t) + count * sum((mean_t - mean)^2)`, both as fixed-order block sums
///
/// `part` is `[B, tiles, pcols]`; `stats` `[B, cols]` takes partial columns `c0..c0 + cols`. Grid `(cols, B)`,
/// block 256 threads, each striding over the tiles
#[kernel]
pub fn sn_merge(
    mut stats: DisjointSlice<[f32; 2]>,
    part: &[[f32; 2]],
    tiles: u32,
    pcols: u32,
    c0: u32,
    cols: u32,
    count: f32,
    eps: f32,
) {
    static mut RED: SharedArray<f32, 8> = SharedArray::UNINIT;
    let (c, b, tid) = (blockIdx_x(), blockIdx_y(), threadIdx_x());
    // block-uniform, so a block that fails it never reaches a barrier
    if (blockDim_x() != 256)
        | (c >= cols)
        | (c0 + cols > pcols)
        | (tiles == 0)
        | ((((b + 1) * tiles * pcols) as usize) > part.len())
        | ((((b + 1) * cols) as usize) > stats.len())
    {
        return;
    }
    let red = addr_of_mut!(RED) as *mut f32;
    let base = b * tiles * pcols + c0 + c;

    let mut s = 0f32;
    let mut k = tid;
    while k < tiles {
        s += part[(base + k * pcols) as usize][0];
        k += 256;
    }
    let mean = block_sum256(s, red, tid) / tiles as f32;

    let mut q = 0f32;
    let mut k = tid;
    while k < tiles {
        let [m, m2] = part[(base + k * pcols) as usize];
        let d = m - mean;
        q += m2 + count * d * d;
        k += 256;
    }
    let q = block_sum256(q, red, tid);
    if tid != 0 {
        return;
    }
    let var = q / (count * tiles as f32);
    let var = if var > 0.0 { var } else { 0.0 };
    // SAFETY: `b * cols + c < stats.len()` by the check above, and only thread 0 of block (c, b) writes it
    unsafe { *stats.get_unchecked_mut((b * cols + c) as usize) = [mean, 1.0 / sqrt_rn_f32(var + eps)] };
}

/// Sums 64 per-lane values over the warp as a reduce-scatter: five butterfly steps, each sending half of what the
/// lane still holds, so lane `L` ends with the warp totals of slots `2L` and `2L + 1` (62 shuffles instead of 320)
#[inline(always)]
fn reduce_scatter64(v: &[f32; 64], lane: u32) -> [f32; 2] {
    let mut h32 = [0f32; 32];
    let up = lane & 16 != 0;
    let mut i = 0;
    while i < 32 {
        __unroll_config::<0>();
        let (send, keep) = if up { (v[i], v[32 + i]) } else { (v[32 + i], v[i]) };
        h32[i] = keep + warp::shuffle_xor_f32(send, 16);
        i += 1;
    }
    let mut h16 = [0f32; 16];
    let up = lane & 8 != 0;
    let mut i = 0;
    while i < 16 {
        __unroll_config::<0>();
        let (send, keep) = if up { (h32[i], h32[16 + i]) } else { (h32[16 + i], h32[i]) };
        h16[i] = keep + warp::shuffle_xor_f32(send, 8);
        i += 1;
    }
    let mut h8 = [0f32; 8];
    let up = lane & 4 != 0;
    let mut i = 0;
    while i < 8 {
        __unroll_config::<0>();
        let (send, keep) = if up { (h16[i], h16[8 + i]) } else { (h16[8 + i], h16[i]) };
        h8[i] = keep + warp::shuffle_xor_f32(send, 4);
        i += 1;
    }
    let mut h4 = [0f32; 4];
    let up = lane & 2 != 0;
    let mut i = 0;
    while i < 4 {
        __unroll_config::<0>();
        let (send, keep) = if up { (h8[i], h8[4 + i]) } else { (h8[4 + i], h8[i]) };
        h4[i] = keep + warp::shuffle_xor_f32(send, 2);
        i += 1;
    }
    let up = lane & 1 != 0;
    let (s0, k0) = if up { (h4[0], h4[2]) } else { (h4[2], h4[0]) };
    let (s1, k1) = if up { (h4[1], h4[3]) } else { (h4[3], h4[1]) };
    [k0 + warp::shuffle_xor_f32(s0, 1), k1 + warp::shuffle_xor_f32(s1, 1)]
}

/// The first block's stride-2 conv and its skip, straight from the float32 input crop
///
/// - `y = conv1(bf16(x))`: 3x3x3, stride 2, 2 -> 12 channels, bf16 inputs and weights with float32 accumulation,
///   plus its bias `rbf(rbf(acc) + bias)`, written as bf16 `[B, 72, 64, 64, 12]` (three `U32x2` per voxel);
/// - `p = bf16(avgpool2(x))`, `[B, 72, 64, 64]` bf16 pairs: the skip's pooled input, summed in float32 in (z, y, x)
///   order as ATen's avg_pool3d;
/// - `part[b, tile, 0..12]` the tile's `[mean, M2]` of `y`, `part[b, tile, 12..24]` of the skip's 1x1 conv
///   `rbf(wsk p)`, for [`sn_merge`] (count 256).
///
/// `x` is the float32 `[B, 144, 128, 128]` (CT, tube) pairs; `w` the bf16-rounded conv weights `[27 taps][2][12]`
/// as 162 `F32x4`; `wsk` the skip's `[2][12]`; `bias` conv1's bf16-rounded bias `[12]`. Grid `(1152, B)`: block `t` owns the output tile `2 x 8 x 16` at
/// `(t / 32, t / 4 % 8, t % 4)` tiles; 256 threads, one output voxel each
#[kernel]
pub fn sn_conv0(
    mut y: DisjointSlice<U32x2>,
    mut p: DisjointSlice<u32>,
    mut part: DisjointSlice<[f32; 2]>,
    x: &[F32x2],
    w: &[F32x4],
    wsk: &[f32],
    bias: &[f32],
) {
    static mut HALO: SharedArray<F32x2, 2805> = SharedArray::UNINIT;
    static mut WS: SharedArray<F32x4, 162> = SharedArray::UNINIT;
    static mut SK: SharedArray<f32, 24> = SharedArray::UNINIT;
    static mut SHIFT: SharedArray<f32, 24> = SharedArray::UNINIT;
    static mut RED: SharedArray<f32, 512> = SharedArray::UNINIT;
    const D: u32 = 144;
    const H: u32 = 128;
    const W: u32 = 128;
    const TILES: u32 = 1152;
    let (tile, b, tid) = (blockIdx_x(), blockIdx_y(), threadIdx_x());
    let out_voxels = (D / 2) * (H / 2) * (W / 2);
    // block-uniform, so a block that fails it never reaches a barrier
    if (blockDim_x() != 256)
        | (tile >= TILES)
        | (((b + 1) * D * H * W) as usize > x.len())
        | (w.len() < 162)
        | (wsk.len() < 24)
        | (bias.len() < 12)
        | (((b + 1) * out_voxels * 3) as usize > y.len())
        | (((b + 1) * out_voxels) as usize > p.len())
        | ((((b * TILES + tile) + 1) * 24) as usize > part.len())
    {
        return;
    }
    let halo = addr_of_mut!(HALO) as *mut F32x2;
    let ws = addr_of_mut!(WS) as *mut F32x4;
    let sk = addr_of_mut!(SK) as *mut f32;
    let shift = addr_of_mut!(SHIFT) as *mut f32;
    let red = addr_of_mut!(RED) as *mut f32;
    let (bz, by, bx) = (tile / 32, (tile / 4) % 8, tile % 4);
    // the halo's input origin: the tile's output origin times 2, minus the padding
    let (iz0, iy0, ix0) = ((bz * 4) as i32 - 1, (by * 16) as i32 - 1, (bx * 32) as i32 - 1);

    // stage the 5 x 17 x 33 input halo (zero outside the crop) and the weights; loads first, then stores, so each
    // thread's eleven loads are in flight together
    let mut vals = [F32x2([0.0; 2]); 11];
    let mut k = 0;
    while k < 11 {
        __unroll_config::<0>();
        let u = tid + 256 * k;
        let (hz, r) = (u / 561, u % 561);
        let (hy, hx) = (r / 33, r % 33);
        let (iz, iy, ix) = (iz0 + hz as i32, iy0 + hy as i32, ix0 + hx as i32);
        let inside =
            (u < 2805) & (iz >= 0) & (iz < D as i32) & (iy >= 0) & (iy < H as i32) & (ix >= 0) & (ix < W as i32);
        if inside {
            let o = ((b * D + iz as u32) * H + iy as u32) * W + ix as u32;
            // SAFETY: inside the sample, which the check above puts inside `x`
            vals[k as usize] = unsafe { *x.get_unchecked(o as usize) };
        }
        k += 1;
    }
    let mut k = 0;
    while k < 11 {
        __unroll_config::<0>();
        let u = tid + 256 * k;
        if u < 2805 {
            // SAFETY: u < 2805, the halo's size
            unsafe { *halo.add(u as usize) = vals[k as usize] };
        }
        k += 1;
    }
    if tid < 162 {
        // SAFETY: tid < 162 <= w.len()
        unsafe { *ws.add(tid as usize) = *w.get_unchecked(tid as usize) };
    }
    if tid < 24 {
        // SAFETY: tid < 24 <= wsk.len()
        unsafe { *sk.add(tid as usize) = *wsk.get_unchecked(tid as usize) };
    }
    sync_threads();

    let (lz, ly, lx) = (tid / 128, (tid / 16) % 8, tid % 16);
    let mut acc = [0f32; 12];
    let mut t = 0;
    while t < 27 {
        __unroll_config::<0>();
        let (dz, dy, dx) = (t / 9, (t / 3) % 3, t % 3);
        // SAFETY: (2 lz + dz, 2 ly + dy, 2 lx + dx) < (5, 17, 33)
        let [a0, a1] = unsafe { (*halo.add((((2 * lz + dz) * 17 + 2 * ly + dy) * 33 + 2 * lx + dx) as usize)).0 };
        let (a0, a1) = (rbf(a0), rbf(a1));
        let mut j = 0;
        while j < 3 {
            __unroll_config::<0>();
            // SAFETY: (t * 2 + c) * 3 + j < 162
            let [w0, w1, w2, w3] = unsafe { (*ws.add((t * 6 + j) as usize)).0 };
            let [v0, v1, v2, v3] = unsafe { (*ws.add((t * 6 + 3 + j) as usize)).0 };
            let o = (4 * j) as usize;
            acc[o] = v0 * a1 + (w0 * a0 + acc[o]);
            acc[o + 1] = v1 * a1 + (w1 * a0 + acc[o + 1]);
            acc[o + 2] = v2 * a1 + (w2 * a0 + acc[o + 2]);
            acc[o + 3] = v3 * a1 + (w3 * a0 + acc[o + 3]);
            j += 1;
        }
        t += 1;
    }

    // the skip's pooled input from the float32 halo, (z, y, x) order
    let (mut s0, mut s1) = (0f32, 0f32);
    let mut q = 0;
    while q < 8 {
        __unroll_config::<0>();
        let (dz, dy, dx) = (q / 4, (q / 2) % 2, q % 2);
        // SAFETY: (2 lz + 1 + dz, 2 ly + 1 + dy, 2 lx + 1 + dx) < (5, 17, 33)
        let [a0, a1] =
            unsafe { (*halo.add((((2 * lz + 1 + dz) * 17 + 2 * ly + 1 + dy) * 33 + 2 * lx + 1 + dx) as usize)).0 };
        s0 += a0;
        s1 += a1;
        q += 1;
    }
    let (p0, p1) = (rbf(s0 * 0.125), rbf(s1 * 0.125));

    let mut v = [0f32; 24];
    let mut o = 0;
    while o < 12 {
        __unroll_config::<0>();
        // SAFETY: o < 12 <= bias.len()
        v[o] = rbf(rbf(acc[o]) + unsafe { *bias.get_unchecked(o) });
        // SAFETY: o + 12 < 24
        let (k0, k1) = unsafe { (*sk.add(o), *sk.add(12 + o)) };
        v[12 + o] = rbf(k1 * p1 + k0 * p0);
        o += 1;
    }
    let (oz, oy, ox) = (bz * 2 + lz, by * 8 + ly, bx * 16 + lx);
    let voxel = b * out_voxels + (oz * (H / 2) + oy) * (W / 2) + ox;
    // SAFETY: `voxel` is inside the sample, inside `y` and `p` by the check above; one thread per voxel
    unsafe {
        *y.get_unchecked_mut((3 * voxel) as usize) = U32x2([pack(v[0], v[1]), pack(v[2], v[3])]);
        *y.get_unchecked_mut((3 * voxel + 1) as usize) = U32x2([pack(v[4], v[5]), pack(v[6], v[7])]);
        *y.get_unchecked_mut((3 * voxel + 2) as usize) = U32x2([pack(v[8], v[9]), pack(v[10], v[11])]);
        *p.get_unchecked_mut(voxel as usize) = pack(p0, p1);
    }

    // statistics of the tile, shifted by its first voxel's values
    if tid == 0 {
        let mut c = 0;
        while c < 24 {
            __unroll_config::<0>();
            // SAFETY: c < 24
            unsafe { *shift.add(c) = v[c] };
            c += 1;
        }
    }
    sync_threads();
    let mut slots = [0f32; 64];
    let mut c = 0;
    while c < 24 {
        __unroll_config::<0>();
        // SAFETY: c < 24
        let d = v[c] - unsafe { *shift.add(c) };
        slots[c] = d;
        slots[32 + c] = d * d;
        c += 1;
    }
    let (wid, lane) = (tid / 32, tid % 32);
    let [r0, r1] = reduce_scatter64(&slots, lane);
    // SAFETY: wid < 8, 2 lane + 1 < 64
    unsafe {
        *red.add((wid * 64 + 2 * lane) as usize) = r0;
        *red.add((wid * 64 + 2 * lane + 1) as usize) = r1;
    }
    sync_threads();
    if tid < 24 {
        let (mut a, mut q) = (0f32, 0f32);
        let mut k = 0;
        while k < 8 {
            // SAFETY: k < 8, tid + 32 < 64
            unsafe {
                a += *red.add((k * 64 + tid) as usize);
                q += *red.add((k * 64 + 32 + tid) as usize);
            }
            k += 1;
        }
        let m2 = q - a * a / 256.0;
        // SAFETY: tid < 24
        let mean = unsafe { *shift.add(tid as usize) } + a / 256.0;
        // SAFETY: inside `part` by the check above; one thread per (sample, tile, channel)
        unsafe {
            *part.get_unchecked_mut(((b * TILES + tile) * 24 + tid) as usize) = [mean, if m2 > 0.0 { m2 } else { 0.0 }]
        };
    }
}

/// The position of 16-byte chunk `k` of halo pixel `px` with `CH` chunks per pixel: with two chunks (32-byte pixels)
/// the pair is swapped on every other run of four pixels, so the eight rows of an `ldmatrix` (eight consecutive
/// pixels) fall in distinct banks; 48-byte pixels already do
#[inline(always)]
fn swz<const CH: u32>(px: u32, k: u32) -> u32 {
    if CH == 2 { px * 2 + (k ^ ((px >> 2) & 1)) } else { px * CH + k }
}

/// The implicit-GEMM 3x3x3 conv ("same" padding, stride `S`) of every `sn_conv_*` kernel: bf16 channels-last in,
/// float32 accumulation on the tensor cores (`mma.m16n8k16`), bf16 out with its InstanceNorm partials
///
/// M is the output voxels of a `TD x TH x TW` tile, N the `COUT` output channels padded to `8 NT`, K the 27 taps
/// times the input channels. The block stages the tile's input halo, all `4 CQ` channels, in shared memory as `CH`
/// 16-byte chunks per voxel (channels past `4 CQ` zero), applying the producer's InstanceNorm and LeakyReLU on load
/// when `PRO == 1` (`leaky(rbf(affine(x)))`, the post-norm zero padding included). `PRO == 2` loads the first block's
/// output instead of its conv2 output `x`, as [`sn_apply_s0`] computes it from `x`, the pooled input `p` and the skip's
/// norm (`skstats`, `skg`, `skb`, weights `wsk`), and also writes its 2x2x2 average (`S == 2`: each output voxel's
/// window is in the halo) to `pool` (`[B, Do, Ho, Wo, 4 CQ]` bf16 as `U32x2`). Then each warp runs `MT` 16-voxel
/// tiles against every (tap, chunk) pair as one k16 step, the A fragments loaded with `ldmatrix` from the halo at
/// the tap's offset (each 8-channel half may come from a different tap, so 24-channel inputs pack without padding),
/// the B fragments (`wf`, host-ordered `[step][n tile][lane]` `U32x2`) from global memory one step ahead.
///
/// The epilogue writes `rbf(acc)` to `out` (`[B, Do, Ho, Wo, COUT]` bf16 pairs), or with `BIAS` the conv's bias added
/// as PyTorch adds it, `rbf(rbf(acc) + bias[c])`, and the tile's `[mean, M2]` of those values per channel to
/// `part[b, tile, c]` (two passes over the registers, fixed reduction order)
///
/// # Safety
/// `smem` is the block's dynamic shared memory of [`conv_smem`] bytes; every thread of the block calls this together
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn conv3<
    const S: u32,
    const CQ: u32,
    const CH: u32,
    const NT: usize,
    const COUT: u32,
    const TD: u32,
    const TH: u32,
    const TW: u32,
    const MT: usize,
    const NW: u32,
    const PRO: u32,
    const BIAS: bool,
>(
    smem: *mut U32x4,
    out: &mut DisjointSlice<u32>,
    part: &mut DisjointSlice<[f32; 2]>,
    x: &[U32x2],
    stats: &[[f32; 2]],
    gamma: &[f32],
    beta: &[f32],
    p: &[u32],
    skstats: &[[f32; 2]],
    skg: &[f32],
    skb: &[f32],
    wsk: &[f32],
    pool: &mut DisjointSlice<U32x2>,
    wf: &[U32x2],
    bias: &[f32],
    d: u32,
    h: u32,
    w: u32,
) {
    let (hd, hh, hw) = ((TD - 1) * S + 3, (TH - 1) * S + 3, (TW - 1) * S + 3);
    let hpx = hd * hh * hw;
    let cin = 4 * CQ;
    let steps = (27 * CH + 1) / 2;
    let threads = NW * 32;
    let (od, oh, ow) = (d / S, h / S, w / S);
    let (ty, tx) = (oh / TH, ow / TW);
    let tiles = (od / TD) * ty * tx;
    let (tile, b, tid) = (blockIdx_x(), blockIdx_y(), threadIdx_x());
    // block-uniform, so no block leaves before a barrier, shuffle, ldmatrix or mma that others reach
    if (blockDim_x() != threads)
        | (NW * MT as u32 * 16 != TD * TH * TW)
        | (2 * CH < CQ)
        | ((8 * NT as u32) < COUT)
        | (od * S != d)
        | (oh * S != h)
        | (ow * S != w)
        | (od % TD != 0)
        | (oh % TH != 0)
        | (ow % TW != 0)
        | (tile >= tiles)
        | (((b + 1) * d * h * w * CQ) as usize > x.len())
        | ((steps * NT as u32 * 32) as usize > wf.len())
        | (((b + 1) * od * oh * ow * COUT / 2) as usize > out.len())
        | (((b * tiles + tile + 1) * COUT) as usize > part.len())
        | (BIAS & (bias.len() < COUT as usize))
        | ((PRO != 0)
            & ((((b + 1) * cin) as usize > stats.len()) | (gamma.len() < cin as usize) | (beta.len() < cin as usize)))
        | ((PRO == 2)
            & ((S != 2)
                | (((b + 1) * d * h * w) as usize > p.len())
                | (((b + 1) * cin) as usize > skstats.len())
                | (skg.len() < cin as usize)
                | (skb.len() < cin as usize)
                | (wsk.len() < 2 * cin as usize)
                | (((b + 1) * od * oh * ow * CQ) as usize > pool.len())))
    {
        return;
    }
    let halo = smem;
    // SAFETY: the layout `conv_smem` sizes: halo chunks, then 3 cin params (the norm, and for `PRO == 2` the skip's
    // norm and weights), then the reduction scratch
    let par = unsafe { halo.add((hpx * CH) as usize) } as *mut [f32; 4];
    let (skpar, wpar) = unsafe { (par.add(cin as usize), par.add(2 * cin as usize)) };
    let red = unsafe { par.add(3 * cin as usize) } as *mut f32;
    let means = unsafe { red.add(NW as usize * NT * 8) };
    let (bz, rest) = (tile / (ty * tx), tile % (ty * tx));
    let (by, bx) = (rest / tx, rest % tx);
    let (z0, y0, x0) = (bz * TD, by * TH, bx * TW);

    if PRO != 0 {
        let mut c = tid;
        while c < cin {
            // SAFETY: c < cin; the stats, gamma and beta lengths (and for `PRO == 2` the skip's) are checked above
            unsafe {
                let st = *stats.get_unchecked((b * cin + c) as usize);
                *par.add(c as usize) =
                    [st[0], st[1], *gamma.get_unchecked(c as usize), *beta.get_unchecked(c as usize)];
                if PRO == 2 {
                    let st = *skstats.get_unchecked((b * cin + c) as usize);
                    let (g, bt) = (*skg.get_unchecked(c as usize), *skb.get_unchecked(c as usize));
                    *skpar.add(c as usize) = [st[0], st[1], g, bt];
                    let (w0, w1) = (*wsk.get_unchecked(c as usize), *wsk.get_unchecked((cin + c) as usize));
                    *wpar.add(c as usize) = [w0, w1, 0.0, 0.0];
                }
            }
            c += threads;
        }
        sync_threads();
    }

    // stage the halo, one 4-channel quad per item: `2 CH` quads per voxel, those past `CQ` zero
    let qpp = 2 * CH;
    let (iz0, iy0, ix0) = ((z0 * S) as i32 - 1, (y0 * S) as i32 - 1, (x0 * S) as i32 - 1);
    let mut u = tid;
    while u < hpx * qpp {
        let (px, q) = (u / qpp, u % qpp);
        let (hz, r) = (px / (hh * hw), px % (hh * hw));
        let (hy, hx) = (r / hw, r % hw);
        let (iz, iy, ix) = (iz0 + hz as i32, iy0 + hy as i32, ix0 + hx as i32);
        let inside = (q < CQ) & (iz >= 0) & (iz < d as i32) & (iy >= 0) & (iy < h as i32) & (ix >= 0) & (ix < w as i32);
        let mut v = U32x2([0, 0]);
        if inside {
            let o = (((b * d + iz as u32) * h + iy as u32) * w + ix as u32) * CQ + q;
            // SAFETY: inside the sample, inside `x` by the check above
            let [p0, p1] = unsafe { *x.get_unchecked(o as usize) }.0;
            v = U32x2([p0, p1]);
            if PRO == 1 {
                let c = 4 * q;
                // SAFETY: c + 3 < cin
                let f = |val: f32, k: u32| {
                    let [m, is, g, bt] = unsafe { *par.add((c + k) as usize) };
                    leaky(affine(val, [m, is], g, bt))
                };
                v = U32x2([pack(f(lo(p0), 0), f(hi(p0), 1)), pack(f(lo(p1), 2), f(hi(p1), 3))]);
            }
            if PRO == 2 {
                let c = 4 * q;
                // SAFETY: the voxel is inside the sample, so inside `p` by the check above
                let pp = unsafe {
                    *p.get_unchecked(((b * d + iz as u32) * h + iy as u32) as usize * w as usize + ix as usize)
                };
                let (i0, i1) = (lo(pp), hi(pp));
                // SAFETY: c + 3 < cin
                let f = |val: f32, k: u32| {
                    let ([m, is, g, bt], [sm, sis, sg, sbt], [w0, w1, _, _]) = unsafe {
                        (*par.add((c + k) as usize), *skpar.add((c + k) as usize), *wpar.add((c + k) as usize))
                    };
                    let sk = rbf(w1 * i1 + w0 * i0);
                    leaky(rbf(affine(val, [m, is], g, bt) + affine(sk, [sm, sis], sg, sbt)))
                };
                v = U32x2([pack(f(lo(p0), 0), f(hi(p0), 1)), pack(f(lo(p1), 2), f(hi(p1), 3))]);
            }
        }
        // SAFETY: px < hpx and the chunk index stays inside the pixel's CH chunks
        unsafe { *(halo.add(swz::<CH>(px, q / 2) as usize) as *mut U32x2).add((q % 2) as usize) = v };
        u += threads;
    }
    sync_threads();

    if PRO == 2 {
        // the 2x2x2 average of the block output under each output voxel: halo voxels 2o + 1 and 2o + 2 per axis
        let mut u = tid;
        while u < TD * TH * TW * CQ {
            let (o, q) = (u / CQ, u % CQ);
            let (oz, oy, ox) = (o / (TH * TW), (o / TW) % TH, o % TW);
            let mut acc = [0f32; 4];
            let mut k = 0;
            while k < 8 {
                __unroll_config::<0>();
                let px = ((2 * oz + 1 + (k >> 2)) * hh + 2 * oy + 1 + ((k >> 1) & 1)) * hw + 2 * ox + 1 + (k & 1);
                // SAFETY: px < hpx, chunk inside the pixel
                let [a0, a1] =
                    unsafe { *(halo.add(swz::<CH>(px, q / 2) as usize) as *const U32x2).add((q % 2) as usize) }.0;
                acc = [acc[0] + lo(a0), acc[1] + hi(a0), acc[2] + lo(a1), acc[3] + hi(a1)];
                k += 1;
            }
            let ov = ((b * od + z0 + oz) * oh + y0 + oy) * ow + x0 + ox;
            let packed = U32x2([pack(acc[0] * 0.125, acc[1] * 0.125), pack(acc[2] * 0.125, acc[3] * 0.125)]);
            // SAFETY: inside `pool` by the check above; each (voxel, quad) belongs to one thread of one tile
            unsafe { *pool.get_unchecked_mut((ov * CQ + q) as usize) = packed };
            u += threads;
        }
    }

    // SAFETY: `halo` is block-shared
    let halo_at = unsafe { cvta_generic_to_shared_u32(halo as *const u8) };
    let (wid, lane) = (tid / 32, tid % 32);
    let (g, th) = (lane / 4, lane % 4);
    let row = (lane & 7) + 8 * ((lane >> 3) & 1);
    let khalf = lane >> 4;
    let mut hp0 = [0u32; MT];
    let mut mt = 0;
    while mt < MT {
        __unroll_config::<0>();
        let o = (wid * MT as u32 + mt as u32) * 16 + row;
        let (oz, oy, ox) = (o / (TH * TW), (o / TW) % TH, o % TW);
        hp0[mt] = ((oz * S) * hh + oy * S) * hw + ox * S;
        mt += 1;
    }

    let wbase = lane as usize;
    // SAFETY: step 0 is inside `wf` by the check above
    let mut bnext = [[0u32; 2]; NT];
    let mut nt = 0;
    while nt < NT {
        __unroll_config::<0>();
        bnext[nt] = unsafe { *wf.get_unchecked(wbase + nt * 32) }.0;
        nt += 1;
    }
    let mut acc = [[[0f32; 4]; NT]; MT];
    let mut st = 0;
    while st < steps {
        __unroll_config::<0>();
        let b = bnext;
        if st + 1 < steps {
            let mut nt = 0;
            while nt < NT {
                __unroll_config::<0>();
                // SAFETY: step st + 1 < steps is inside `wf`
                bnext[nt] = unsafe { *wf.get_unchecked(wbase + (((st + 1) * NT as u32) as usize + nt) * 32) }.0;
                nt += 1;
            }
        }
        // this lane's half of the step: (tap, chunk) pair 2 st + khalf, the padding pair (zero weights) reading
        // the last real one
        let flat = (2 * st + khalf).min(27 * CH - 1);
        let (t, k) = (flat / CH, flat % CH);
        let toff = ((t / 9) * hh + (t / 3) % 3) * hw + t % 3;
        let mut mt = 0;
        while mt < MT {
            __unroll_config::<0>();
            let hp = hp0[mt] + toff;
            // SAFETY: the halo voxel is inside the staged halo; the whole warp runs this
            let a = unsafe { ldmatrix_x4_shared_u32(halo_at + swz::<CH>(hp, k) * 16) };
            let mut nt = 0;
            while nt < NT {
                __unroll_config::<0>();
                // SAFETY: the whole warp runs this, with fragments in the documented layout
                acc[mt][nt] = unsafe { mma_m16n8k16_f32_bf16(acc[mt][nt], a, b[nt]) };
                nt += 1;
            }
            mt += 1;
        }
        st += 1;
    }

    // the bias on the rounded conv output, so the rounding below (outputs and statistics) completes PyTorch's
    // `rbf(rbf(conv) + bias)`; padded channels stay as they are
    if BIAS {
        let mut mt = 0;
        while mt < MT {
            __unroll_config::<0>();
            let mut nt = 0;
            while nt < NT {
                __unroll_config::<0>();
                let mut e = 0;
                while e < 4 {
                    __unroll_config::<0>();
                    let n = nt * 8 + 2 * th as usize + e % 2;
                    if n < COUT as usize {
                        // SAFETY: n < COUT <= bias.len() by the check above
                        acc[mt][nt][e] = rbf(acc[mt][nt][e]) + unsafe { *bias.get_unchecked(n) };
                    }
                    e += 1;
                }
                nt += 1;
            }
            mt += 1;
        }
    }

    // the bf16 outputs
    let mut mt = 0;
    while mt < MT {
        __unroll_config::<0>();
        let mut half = 0;
        while half < 2 {
            __unroll_config::<0>();
            let o = (wid * MT as u32 + mt as u32) * 16 + g + 8 * half;
            let (oz, oy, ox) = (o / (TH * TW), (o / TW) % TH, o % TW);
            let pix = ((b * od + z0 + oz) * oh + y0 + oy) * ow + x0 + ox;
            let mut nt = 0;
            while nt < NT {
                __unroll_config::<0>();
                let n = nt as u32 * 8 + 2 * th;
                if n < COUT {
                    let v = acc[mt][nt];
                    // SAFETY: inside `out` by the check above; one lane writes each (voxel, pair)
                    unsafe {
                        *out.get_unchecked_mut((pix * (COUT / 2) + n / 2) as usize) =
                            pack(v[2 * half as usize], v[2 * half as usize + 1])
                    };
                }
                nt += 1;
            }
            half += 1;
        }
        mt += 1;
    }

    // the tile's mean, then its squared deviations, per channel: lanes sum their rows, the warp's lanes with the
    // same `th` sum by shuffles, the warps sum in order
    let bn = (NT * 8) as u32;
    let count = (TD * TH * TW) as f32;
    let mut pass = 0;
    while pass < 2 {
        let mut q = [[0f32; 2]; NT];
        let mut mt = 0;
        while mt < MT {
            __unroll_config::<0>();
            let mut nt = 0;
            while nt < NT {
                __unroll_config::<0>();
                let v = acc[mt][nt];
                let mut e = 0;
                while e < 2 {
                    __unroll_config::<0>();
                    let (a, c) = (rbf(v[e]), rbf(v[2 + e]));
                    if pass == 0 {
                        q[nt][e] += a + c;
                    } else {
                        // SAFETY: the channel < bn, written before the last barrier
                        let m = unsafe { *means.add(nt * 8 + 2 * th as usize + e) };
                        q[nt][e] += (a - m) * (a - m) + (c - m) * (c - m);
                    }
                    e += 1;
                }
                nt += 1;
            }
            mt += 1;
        }
        let mut nt = 0;
        while nt < NT {
            __unroll_config::<0>();
            let mut e = 0;
            while e < 2 {
                __unroll_config::<0>();
                let mut v = q[nt][e];
                v += warp::shuffle_xor_f32(v, 4);
                v += warp::shuffle_xor_f32(v, 8);
                v += warp::shuffle_xor_f32(v, 16);
                if g == 0 {
                    // SAFETY: wid < NW, the channel < bn
                    unsafe { *red.add((wid * bn + nt as u32 * 8 + 2 * th + e as u32) as usize) = v };
                }
                e += 1;
            }
            nt += 1;
        }
        sync_threads();
        let mut i = tid;
        while i < bn {
            let mut t = 0f32;
            let mut k = 0;
            while k < NW {
                // SAFETY: k < NW, i < bn
                t += unsafe { *red.add((k * bn + i) as usize) };
                k += 1;
            }
            if pass == 0 {
                // SAFETY: i < bn
                unsafe { *means.add(i as usize) = t / count };
            } else if i < COUT {
                // SAFETY: inside `part` by the check above; one thread per (sample, tile, channel)
                unsafe {
                    *part.get_unchecked_mut(((b * tiles + tile) * COUT + i) as usize) = [*means.add(i as usize), t]
                };
            }
            i += threads;
        }
        sync_threads();
        pass += 1;
    }
}

/// Shared memory bytes [`conv3`] takes: the halo, the per-channel norm parameters, the reduction scratch and the
/// means
pub const fn conv_smem(s: u32, ch: u32, cq: u32, nt: u32, td: u32, th: u32, tw: u32, nw: u32) -> u32 {
    let hpx = ((td - 1) * s + 3) * ((th - 1) * s + 3) * ((tw - 1) * s + 3);
    hpx * ch * 16 + 3 * 4 * cq * 16 + (nw * nt * 8 + nt * 8) * 4
}

/// One `sn_conv_*` kernel: [`conv3`] at fixed parameters, `out` bf16 pairs, `part` the tile partials
macro_rules! conv_kernel {
    ($name:ident, $s:literal, $cq:literal, $ch:literal, $nt:literal, $cout:literal, $td:literal, $th:literal,
     $tw:literal, $mt:literal, $nw:literal, $pro:literal, $bias:literal) => {
        #[doc = concat!("[`conv3`] with stride ", $s, ", ", $cq, " input quads in ", $ch, " chunks, ", $cout,
                                " outputs, tile ", $td, "x", $th, "x", $tw, ", ", $nw, " warps, prologue ", $pro,
                                ", bias ", $bias)]
        #[kernel]
        pub fn $name(
            mut out: DisjointSlice<u32>,
            mut part: DisjointSlice<[f32; 2]>,
            x: &[U32x2],
            stats: &[[f32; 2]],
            gamma: &[f32],
            beta: &[f32],
            p: &[u32],
            skstats: &[[f32; 2]],
            skg: &[f32],
            skb: &[f32],
            wsk: &[f32],
            mut pool: DisjointSlice<U32x2>,
            wf: &[U32x2],
            bias: &[f32],
            d: u32,
            h: u32,
            w: u32,
        ) {
            let smem = DynamicSharedArray::<U32x4, 16>::get();
            // SAFETY: the host launches with `conv_smem` bytes of dynamic shared memory
            unsafe {
                conv3::<$s, $cq, $ch, $nt, $cout, $td, $th, $tw, $mt, $nw, $pro, $bias>(
                    smem, &mut out, &mut part, x, stats, gamma, beta, p, skstats, skg, skb, wsk, &mut pool, wf,
                    bias, d, h, w,
                )
            }
        }
    };
}

// s0b0 conv2: 12 -> 12 at 72x64x64, conv1's norm and LeakyReLU on load, plus its bias
conv_kernel!(sn_conv_12_12_n, 1, 3, 2, 2, 12, 4, 8, 16, 4, 8, 1, true);
// s1b0 conv1: 12 -> 24, stride 2, on the stage-0 output rebuilt from s0b0's conv2 output, writing its average for
// s1b0's skip
conv_kernel!(sn_conv_12_24_s2, 2, 3, 2, 3, 24, 2, 4, 16, 1, 8, 2, false);
// stage-1 conv2s: 24 -> 24, conv1's norm and LeakyReLU on load
conv_kernel!(sn_conv_24_24_n, 1, 6, 3, 3, 24, 2, 8, 16, 2, 8, 1, false);
// s1b1 conv1: 24 -> 24 on s1b0's output
conv_kernel!(sn_conv_24_24, 1, 6, 3, 3, 24, 2, 8, 16, 2, 8, 0, false);

/// The first block's output in place of its conv2 output: `y = leaky(rbf(rbf(affine(y)) + rbf(affine(sk))))`, the
/// skip `sk = rbf(wsk p)` recomputed from the pooled input `p` (as [`sn_conv0`] computed it for its statistics)
///
/// `y` is `[B, S, 12]` bf16 as `U32x2` (three per voxel), `p` `[B, S]` bf16 pairs, `stats` `[B, 12]` conv2's
/// statistics and `skstats` `[B, 12]` the skip's, `w`/`b` and `skw`/`skb` the affines, `wsk` the skip's weights
/// `[2][12]`. One thread per `U32x2`
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn sn_apply_s0(
    mut y: DisjointSlice<U32x2>,
    p: &[u32],
    stats: &[[f32; 2]],
    w: &[f32],
    b: &[f32],
    skstats: &[[f32; 2]],
    skw: &[f32],
    skb: &[f32],
    wsk: &[f32],
    rows: u32,
) {
    let t = thread::index_1d_u32(lc);
    let e = t.get();
    let (pix, q) = (e / 3, e % 3);
    let s = pix / rows;
    let Some(mut out) = y.element_thread32(t) else { return };
    let pp = p[pix as usize];
    let (p0, p1) = (lo(pp), hi(pp));
    let [y0, y1] = out.read().0;
    let v = [lo(y0), hi(y0), lo(y1), hi(y1)];
    let mut r = [0f32; 4];
    let mut j = 0;
    while j < 4 {
        __unroll_config::<0>();
        let c = (4 * q + j) as usize;
        let k = rbf(wsk[12 + c] * p1 + wsk[c] * p0);
        let main = affine(v[j as usize], stats[(s * 12) as usize + c], w[c], b[c]);
        let skip = affine(k, skstats[(s * 12) as usize + c], skw[c], skb[c]);
        r[j as usize] = leaky(rbf(main + skip));
        j += 1;
    }
    out.write(U32x2([pack(r[0], r[1]), pack(r[2], r[3])]));
}
