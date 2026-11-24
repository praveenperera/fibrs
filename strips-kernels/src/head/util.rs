//! Helpers the head's kernels share: bf16 rounding and packing, libdevice math, warp sums, wide global loads and
//! the up-front launch check

use cuda_device::convert::cvt_bf16x2_f32;
use cuda_device::{ptx_asm, warp};

// libdevice, as nvcc's math library calls it, so results match the CUDA C++ PyTorch runs
unsafe extern "C" {
    fn __nv_rsqrtf(x: f32) -> f32;
    fn __nv_fmaxf(x: f32, y: f32) -> f32;
    fn __nv_expf(x: f32) -> f32;
    fn __nv_erff(x: f32) -> f32;
    fn __nv_log1pf(x: f32) -> f32;
    fn __nv_sinpif(x: f32) -> f32;
    fn __nv_cospif(x: f32) -> f32;
    fn __nv_floorf(x: f32) -> f32;
}

/// Math functions under their CUDA C++ names
pub mod math {
    /// `rsqrtf`
    #[inline(always)]
    pub fn rsqrtf(x: f32) -> f32 {
        unsafe { super::__nv_rsqrtf(x) }
    }

    /// One `max.f32`
    #[inline(always)]
    pub fn fmaxf(x: f32, y: f32) -> f32 {
        unsafe { super::__nv_fmaxf(x, y) }
    }

    /// `expf` (full precision, as PyTorch's SiLU)
    #[inline(always)]
    pub fn expf(x: f32) -> f32 {
        unsafe { super::__nv_expf(x) }
    }

    /// `erff`, as PyTorch's exact GELU
    #[inline(always)]
    pub fn erff(x: f32) -> f32 {
        unsafe { super::__nv_erff(x) }
    }

    /// `log1pf`
    #[inline(always)]
    pub fn log1pf(x: f32) -> f32 {
        unsafe { super::__nv_log1pf(x) }
    }

    /// `sin(pi x)`: exact argument reduction, so no Payne-Hanek slow path (and no stack frame) as `sinf` has
    #[inline(always)]
    pub fn sinpif(x: f32) -> f32 {
        unsafe { super::__nv_sinpif(x) }
    }

    /// `cos(pi x)`
    #[inline(always)]
    pub fn cospif(x: f32) -> f32 {
        unsafe { super::__nv_cospif(x) }
    }

    /// `floorf`
    #[inline(always)]
    pub fn floorf(x: f32) -> f32 {
        unsafe { super::__nv_floorf(x) }
    }
}

/// Sum over the warp's 32 lanes in a fixed butterfly order, the same on every lane
#[inline(always)]
pub fn warp_sum(mut v: f32) -> f32 {
    let mut o = 16;
    while o > 0 {
        v += warp::shuffle_xor_f32(v, o);
        o >>= 1;
    }
    v
}

/// Two floats rounded to nearest-even bf16, `lo` in the low half
#[inline(always)]
pub fn pack_bf16(lo: f32, hi: f32) -> u32 {
    cvt_bf16x2_f32(lo, hi)
}

/// `v` rounded to nearest-even bf16 and back to f32: what a bf16 tensor holds
#[inline(always)]
pub fn round_bf16(v: f32) -> f32 {
    f32::from_bits(pack_bf16(v, 0.0) << 16)
}

/// The low bf16 of a pair as f32
#[inline(always)]
pub fn lo_f32(p: u32) -> f32 {
    f32::from_bits(p << 16)
}

/// The high bf16 of a pair as f32
#[inline(always)]
pub fn hi_f32(p: u32) -> f32 {
    f32::from_bits(p & 0xffff_0000)
}

/// One bf16 value's bits as f32
#[inline(always)]
pub fn bf16_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// The raw bits of a bf16 value
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Bf16(pub u16);

/// Sixteen bytes from a 16-byte aligned global address
///
/// # Safety
/// `p` is 16-byte aligned and valid for a 16-byte read
#[inline(always)]
pub unsafe fn ldg_u4<T>(p: *const T) -> [u32; 4] {
    let (a, b, c, d): (u32, u32, u32, u32);
    unsafe {
        ptx_asm!(
            "{ .reg .u64 g; cvta.to.global.u64 g, %4; ld.global.v4.u32 {%0, %1, %2, %3}, [g]; }",
            out("=r") a,
            out("=r") b,
            out("=r") c,
            out("=r") d,
            in("l") p as u64,
        );
    }
    [a, b, c, d]
}

/// Traps when the host's geometry doesn't fit the buffers it passed, as a failed slice index would. Call it once
/// with a block-uniform condition: a thread-dependent trap makes ptxas treat the warp as divergent from there on
#[inline(always)]
pub fn require(ok: bool) {
    if !ok {
        panic!();
    }
}
