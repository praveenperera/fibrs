//! `--device cuda`: centrality and ridge seeds on an NVIDIA GPU, as the reference's `--device cuda` computes them
//!
//! The reference's CUDA kernels round some multiply-adds as one `fma` where its CPU kernel rounds the product and
//! the sum apart (see `stage1.rs`), so this path gives the reference's CUDA output, not its CPU output. The
//! kernels (`fibrs-kernels/`) are built to PTX ahead of time and embedded; at run time the NVIDIA driver loads
//! them, so neither building nor running fibrs needs a CUDA toolkit. Linux x86_64 only, like `fibrs predict`;
//! `--device auto`, the default, uses the GPU when [`Gpu::open`] succeeds and the CPU otherwise

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod cuda;

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub use cuda::Gpu;

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
pub use unsupported::Gpu;

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
mod unsupported {
    use color_eyre::eyre::{Result, bail};

    use crate::fields::FamilyFields;
    use crate::vol::Vol;

    /// No GPU backend on this platform; [`Gpu::open`] always fails, so no value exists
    pub enum Gpu {}

    impl Gpu {
        pub fn open(_device: Option<usize>) -> Result<Self> {
            bail!("fibrs runs centrality on the GPU on Linux x86_64 only")
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
    }
}
