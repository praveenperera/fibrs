//! `--device cuda`: the direction field, centrality and ridge seeds on an NVIDIA GPU, centrality and seeds as the
//! reference's `--device cuda` computes them
//!
//! The direction field is bit for bit the CPU's (`cuda/axes.rs`). For centrality, the reference's CUDA kernels
//! round some multiply-adds as one `fma` where its CPU kernel rounds the product and the sum apart (see
//! `stage1.rs`), so this path gives the reference's CUDA output, not its CPU output. The
//! kernels (`fibrs-kernels/`) are built to PTX ahead of time and embedded; at run time the NVIDIA driver loads
//! them, so neither building nor running fibrs needs a CUDA toolkit. Linux x86_64 only, like `fibrs predict`;
//! `--device auto`, the default, uses the GPU when [`Gpu::open`] succeeds and the CPU otherwise

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod cuda;

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub use cuda::Gpu;

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
pub use unsupported::Gpu;

/// How many voxels the GPU direction field found present, and how many of those the kernels could not certify,
/// so the host solved them with `dsyevd` as the CPU path does
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AxesCounts {
    pub present: usize,
    pub fallback: usize,
}

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
mod unsupported {
    use color_eyre::eyre::{Result, bail};

    use super::AxesCounts;
    use crate::axes::Lapack;
    use crate::fields::FamilyFields;
    use crate::vol::Vol;

    /// No GPU backend on this platform; [`Gpu::open`] always fails, so no value exists
    pub enum Gpu {}

    impl Gpu {
        pub fn open(_device: usize) -> Result<Self> {
            bail!("fibrs runs on the GPU on Linux x86_64 only")
        }

        pub fn name(&self) -> &str {
            match *self {}
        }

        pub fn centrality(
            &self,
            _tracing: &[Vol<u8>; 2],
            _family: usize,
            _nx: &Vol<u8>,
            _ny: &Vol<u8>,
        ) -> Result<FamilyFields> {
            match *self {}
        }

        pub fn direction_field(
            &self,
            _tracing: &[Vol<u8>; 2],
            _lapack: &Lapack,
        ) -> Result<(Vol<u8>, Vol<u8>, AxesCounts)> {
            match *self {}
        }
    }
}
