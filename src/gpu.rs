//! `--device cuda`: the direction field, centrality, ridge seeds and trace finishing on an NVIDIA GPU, centrality
//! and seeds as the reference's `--device cuda` computes them
//!
//! The direction field and finishing are bit for bit the CPU's (`cuda/axes.rs`, `cuda/finish.rs`). For centrality, the reference's CUDA kernels
//! round some multiply-adds as one `fma` where its CPU kernel rounds the product and the sum apart (see
//! `stage1.rs`), so this path gives the reference's CUDA output, not its CPU output. The
//! kernels (`fibrs-kernels/`) are built to PTX ahead of time and embedded; at run time the NVIDIA driver loads
//! them, so neither building nor running fibrs needs a CUDA toolkit. Linux x86_64 only, like `fibrs predict`;
//! `--device auto`, the default, uses the GPU when [`Gpu::open`] succeeds and the CPU otherwise

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod cuda;

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub use cuda::{Gpu, GpuFinisher};

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
pub use unsupported::{Gpu, GpuFinisher};

/// The stages a run gives the GPU, which set how much free memory [`Gpu::open`] looks for when it picks a device
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stages {
    /// A run that keeps centrality on the CPU it started on
    DirectionField,
    DirectionFieldAndCentrality,
}

/// How many voxels the GPU direction field found present, and how many of those the kernels could not certify,
/// so the host solved them with `dsyevd` as the CPU path does
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AxesCounts {
    pub present: usize,
    pub fallback: usize,
}

/// One curve's finished pieces, in order
pub type Pieces = Vec<Vec<[f32; 3]>>;

/// Native slabs a [`GpuFinisher`] put in its slab cache, counting slabs again after eviction, by where they were
/// decoded
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SlabLoads {
    /// Decoded on the GPU by nvCOMP from their compressed chunks, or filled there for a chunk never stored
    pub on_gpu: u64,
    /// Decoded on the CPU through the `NativeCache` and uploaded
    pub on_cpu: u64,
}

impl SlabLoads {
    pub fn total(&self) -> u64 {
        self.on_gpu + self.on_cpu
    }
}

/// What [`GpuFinisher::finish`] made of a batch of raw curves
#[cfg_attr(not(all(target_os = "linux", target_arch = "x86_64")), allow(dead_code))]
pub enum Finished {
    /// Every curve's pieces, in batch order, equal to `FinishCtx::finish`'s
    Pieces(Vec<Pieces>),
    /// The GPU handed the batch back for the CPU to finish
    OnCpu(CpuReason),
}

/// Why the GPU handed a batch of raw curves back for the CPU to finish
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(not(all(target_os = "linux", target_arch = "x86_64")), allow(dead_code))]
pub enum CpuReason {
    /// The batch can read more native slabs than the GPU's slab cache holds
    Slabs,
    /// A sample read a native slab the host had not listed for the batch
    Unlisted,
    /// A fitted piece outgrew the room the host gave it
    Capacity,
    /// A native slab the batch could read failed to decode; the CPU reads only the slabs it needs, and reports the
    /// error if it needs that one
    Decode,
}

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
mod unsupported {
    use color_eyre::eyre::{Result, bail};

    use super::{AxesCounts, Finished, SlabLoads, Stages};
    use crate::axes::Lapack;
    use crate::fields::FamilyFields;
    use crate::finish::{NativeCache, SlabLayout};
    use crate::kernels::RawCurve;
    use crate::vol::Vol;

    /// No GPU backend on this platform; [`Gpu::open`] always fails, so no value exists
    pub enum Gpu {}

    /// No GPU backend on this platform, so no finisher exists either
    pub enum GpuFinisher {}

    impl GpuFinisher {
        pub fn slots(&self) -> usize {
            match *self {}
        }

        pub fn loaded(&self) -> SlabLoads {
            match *self {}
        }

        pub fn finish(
            &mut self,
            _pool: &rayon::ThreadPool,
            _cache: &NativeCache,
            _batch: &[RawCurve],
        ) -> Result<Finished> {
            match *self {}
        }
    }

    impl Gpu {
        pub fn open(_device: Option<usize>, _stages: Stages) -> Result<Self> {
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

        pub fn finisher(
            &self,
            _tracing: &[Vol<u8>; 2],
            _family: usize,
            _layout: SlabLayout,
            _batch_points: usize,
        ) -> Result<GpuFinisher> {
            match *self {}
        }
    }
}
