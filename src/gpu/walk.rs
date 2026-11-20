//! The GPU fibre walk: a [`crate::lookahead::Device`] that traces chunks of a family's seeds on the GPU, each
//! seed's result the one `kernels::trace_seed` gives for the coverage values its checks read, or
//! [`crate::lookahead::Status::Cpu`] for a seed the CPU must trace itself

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub use super::cuda::walk::Walk;
#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
pub use unsupported::Walk;

/// Steps per half the device traces before it hands a seed back to the CPU
///
/// On a dense PHerc0358 crop fewer than 0.02% of traced seeds took more than 256 steps over both halves
pub const STEP_CAP: u32 = 256;
/// Most seeds in one chunk
pub const CHUNK: usize = 1 << 16;
/// Chunks launched and not yet collected at most
pub const IN_FLIGHT: usize = 3;

/// Room in one chunk slot's buffers; a seed whose results reach past one goes back to the CPU
#[derive(Clone, Copy, Debug)]
pub struct Caps {
    /// Seeds past the start traced at once
    pub survivors: u32,
    pub checks: u32,
    /// Accepted points (and tangents)
    pub pts: u32,
    pub curves: u32,
}

impl Default for Caps {
    /// Room for a dense chunk: on a dense PHerc0358 crop, 64K seeds had about 7K past the start, 230K checks and
    /// 60K accepted points
    fn default() -> Self {
        Self { survivors: 12 << 10, checks: 2 << 20, pts: 1 << 20, curves: 256 << 10 }
    }
}

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
mod unsupported {
    use std::convert::Infallible;
    use std::marker::PhantomData;
    use std::ops::Range;

    use color_eyre::eyre::Result;

    use crate::kernels::Change;
    use crate::lookahead::{ChunkOut, Device};

    /// No GPU walk on this platform; `Gpu::open` always fails, so no value exists
    pub struct Walk<'g>(Infallible, PhantomData<&'g ()>);

    impl Walk<'_> {
        pub fn device_bytes(&self) -> usize {
            match self.0 {}
        }
    }

    impl Device for Walk<'_> {
        fn update(&mut self, _changes: &[Change]) -> Result<()> {
            match self.0 {}
        }

        fn launch(&mut self, _seeds: Range<usize>) -> Result<()> {
            match self.0 {}
        }

        fn collect(&mut self) -> Result<Option<ChunkOut>> {
            match self.0 {}
        }

        fn in_flight(&self) -> usize {
            match self.0 {}
        }
    }
}
