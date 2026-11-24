//! The GPU side of `fibrs strips`: the device (volkit's GPU with cuDNN and cuBLAS, and the strips PTX module) and
//! the checkpoint's weights
//!
//! Stages live in submodules: `crops` (the CT device cache, the strip and frame crops, the tube), `frame` (the
//! frame model), `cnn` (the crop CNN), `head` (cell tokens, transformer and strip heads)

pub mod cnn;
pub mod crops;
pub mod frame;
pub mod head;
#[cfg(test)]
mod input_tests;

use std::collections::HashMap;
use std::path::Path;

use color_eyre::eyre::{Result, WrapErr, bail, ensure, eyre};
use cudakit::PtxTier;
use volkit::dnn::{Blas, Dnn};
use volkit::gpu::{Gpu, Module};

/// The strips kernels, built from `strips-kernels` by `scripts/build-strips-ptx.sh`
pub const PTX: &[(PtxTier, &str)] = &[(PtxTier::Sm80, include_str!("../../ptx/strips.sm80.ptx"))];

/// The GPU, its stream, the strips kernels and the cuDNN and cuBLAS handles on that stream
pub struct Device {
    /// cuDNN, bound to the GPU's stream
    pub dnn: Dnn,
    /// cuBLAS, bound to the GPU's stream
    pub blas: Blas,
    /// The strips kernels
    pub kernels: Module,
    /// volkit's GPU: context, stream, buffers and volkit's own kernels
    pub gpu: Gpu,
}

impl Device {
    /// Opens GPU `ordinal` and loads the strips kernels
    pub fn open(ordinal: usize) -> Result<Self> {
        let gpu = Gpu::open_device(ordinal).wrap_err("open the GPU")?;
        let kernels = gpu.load_module(PTX).wrap_err("load the strips kernels")?;
        let dnn = Dnn::new(&gpu).wrap_err("create the cuDNN handle")?;
        let blas = Blas::new(&gpu).wrap_err("create the cuBLAS handle")?;
        Ok(Self { dnn, blas, kernels, gpu })
    }
}

/// One float32 tensor of the checkpoint
pub struct Tensor {
    /// Its shape, outermost first
    pub shape: Vec<usize>,
    /// Its values, row-major
    pub values: Vec<f32>,
}

/// The exported checkpoint (`ops/vesuvius/strips-native/export_weights.py`): the strip model's EMA state under
/// `strip.`, the frame model's under `frame.`, and the sources' hashes and configs as metadata
pub struct Weights {
    tensors: HashMap<String, Tensor>,
    /// The export's metadata: `source`, `strip_sha256`, `frame_sha256`, `strip_cfg`, `frame_cfg`, `sample_cfg`,
    /// `vol_spec`
    pub metadata: HashMap<String, String>,
}

/// The `last.pt` and `heading_model.pt` of `scrollprize/fiber_strip_small_v1@7bd163a2` the runner is built for
pub const STRIP_SHA256: &str = "0f6d64da50e47af10c86912a8e268c99fa55bf784823b7ca5b87f84d47f1d6e4";
/// The frame model `heading_model.pt` next to it
pub const FRAME_SHA256: &str = "c10e21e62d05228bdadeda65334d5c9eede91d942b80ca9d318616eda7cbf35f";

impl Weights {
    /// Reads an export, refusing one made from other checkpoint files than the runner's
    pub fn open(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).wrap_err_with(|| format!("read {}", path.display()))?;
        let (_, header) = safetensors::SafeTensors::read_metadata(&bytes).wrap_err("parse the weights header")?;
        let metadata: HashMap<String, String> = header.metadata().clone().unwrap_or_default();
        for (key, want) in [("strip_sha256", STRIP_SHA256), ("frame_sha256", FRAME_SHA256)] {
            let got = metadata.get(key).map(String::as_str).unwrap_or("missing");
            ensure!(got == want, "{}: {key} {got}, expected {want}", path.display());
        }

        let st = safetensors::SafeTensors::deserialize(&bytes).wrap_err("parse the weights")?;
        let mut tensors = HashMap::new();
        for (name, view) in st.tensors() {
            if view.dtype() != safetensors::Dtype::F32 {
                bail!("{name}: {:?}, expected F32", view.dtype());
            }
            let values = view.data().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
            tensors.insert(name, Tensor { shape: view.shape().to_vec(), values });
        }
        Ok(Self { tensors, metadata })
    }

    /// Tensor `name`, checked to have `shape`
    pub fn get(&self, name: &str, shape: &[usize]) -> Result<&Tensor> {
        let t = self.tensors.get(name).ok_or_else(|| eyre!("weights: no tensor {name}"))?;
        ensure!(t.shape == shape, "weights: {name} has shape {:?}, expected {shape:?}", t.shape);
        Ok(t)
    }

    /// The names of every tensor, sorted
    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.tensors.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }
}
