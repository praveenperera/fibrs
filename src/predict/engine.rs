//! The network: AFV v2 on volkit's `AfvV2`, which runs the normalisation, the network, the softmax, the blend
//! and the column's finalise on the GPU, in the reference's float32 arithmetic
//!
//! volkit is CUDA-only and is built in for Linux x86_64 only; elsewhere `fibrs predict` plans (`--dry-run`) but
//! refuses to run

use std::path::Path;

use color_eyre::eyre::Result;
#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
use color_eyre::eyre::bail;
use serde_json::{Map, Value};

use super::contract::Origin;

/// The arithmetic the network runs in
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// The reference contract: float32 tensors with TF32 convolution and GEMM math, as PyTorch's float32
    /// defaults run the model
    Fp32,
    /// Opt-in, changed output: the network and the seg head in bf16 as PyTorch's bf16 autocast rounds them
    Bf16,
}

impl Precision {
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "fp32" => Ok(Self::Fp32),
            "bf16" => Ok(Self::Bf16),
            "fp32-ieee" | "fp16" => Err(format!(
                "--precision {name} is a PyTorch mode the native engine does not run; it runs fp32 (the reference \
                 contract, the default) or bf16"
            )),
            _ => Err(format!("--precision {name:?}: the native engine runs fp32 (the default) or bf16")),
        }
    }

    pub fn short(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Bf16 => "bf16",
        }
    }

    /// What the provenance records for this precision
    pub fn label(self) -> &'static str {
        match self {
            Self::Fp32 => {
                "fp32 (float32 tensors, TF32 convolution and GEMM math, as torch's defaults; fibrs native engine)"
            }
            Self::Bf16 => {
                "bf16 (network and seg head in bf16 as torch's bf16 autocast rounds them; normalize_ct, softmax and \
                 blend fp32; fibrs native engine)"
            }
        }
    }
}

/// One column's outputs, `None` when no tile ran
// built only by the GPU engine, which other targets lack
#[cfg_attr(not(all(target_os = "linux", target_arch = "x86_64")), allow(dead_code))]
pub struct ColumnOut {
    pub semantic: Option<Vec<u8>>,
    pub tracing: Option<Vec<u8>>,
    pub uncovered: u64,
}

/// A column accumulator on the GPU: `begin` a column, `add` its non-air tiles in order, `finish` it
pub trait Engine {
    fn begin(&mut self, shape: [usize; 3], y0: usize, x0: usize) -> Result<()>;
    fn add(&mut self, tiles: &[u8], origins: &[Origin]) -> Result<()>;
    fn finish(&mut self, quantize: bool) -> Result<ColumnOut>;
}

/// What the run needs to know about the opened engine
// built only by the GPU engine, which other targets lack
#[cfg_attr(not(all(target_os = "linux", target_arch = "x86_64")), allow(dead_code))]
pub struct Opened {
    pub engine: Box<dyn Engine>,
    /// The GPU's name
    pub gpu: String,
    /// Device memory free once the engine has loaded
    pub free_bytes: u64,
    /// Device bytes one output voxel of a column takes at its peak
    pub column_bytes_per_voxel: f64,
    /// Provenance fields naming the engine, its library and the weights it loaded
    pub provenance: Map<String, Value>,
    /// The cuDNN and cuBLAS files the process mapped
    pub libraries: Vec<String>,
}

/// Most tiles one `add` takes
pub const MAX_BATCH: usize = 16;

/// The CUDA ordinal of `--device` (`cuda`, `cuda:N` or `N`)
pub fn device_index(device: &str) -> Result<usize, String> {
    let n = if device == "cuda" { "0" } else { device.strip_prefix("cuda:").unwrap_or(device) };
    n.parse().map_err(|_| format!("fibrs predict runs on a CUDA device (cuda, cuda:N), not {device:?}"))
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub fn open(model_dir: &Path, device: usize, precision: Precision) -> Result<Opened> {
    native::open(model_dir, device, precision)
}

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
pub fn open(_model_dir: &Path, _device: usize, _precision: Precision) -> Result<Opened> {
    bail!(
        "fibrs predict needs an NVIDIA GPU on Linux x86_64; this build has no GPU engine (--dry-run plans a run \
         without one)"
    )
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod native {
    use std::path::Path;

    use color_eyre::eyre::{Result, WrapErr};
    use serde_json::{Map, json};
    use volkit::dnn::Pick;
    use volkit::nnunet::afv2::{AfvPrecision, AfvV2, Options};

    use super::{ColumnOut, Engine, Opened, Origin, Precision};

    /// The engine library this build links, for the provenance
    const LIBRARY: &str = concat!(env!("FIBRS_VOLKIT"), ", AfvV2");

    struct Native(AfvV2);

    impl Engine for Native {
        fn begin(&mut self, shape: [usize; 3], y0: usize, x0: usize) -> Result<()> {
            self.0.begin(shape, y0, x0)
        }

        fn add(&mut self, tiles: &[u8], origins: &[Origin]) -> Result<()> {
            self.0.add(tiles, origins).map(drop)
        }

        fn finish(&mut self, quantize: bool) -> Result<ColumnOut> {
            let out = self.0.finish(quantize)?;
            Ok(ColumnOut { semantic: out.semantic, tracing: out.tracing, uncovered: out.uncovered })
        }
    }

    pub fn open(model_dir: &Path, device: usize, precision: Precision) -> Result<Opened> {
        super::libs::prefer_environment();
        let precision_ = match precision {
            Precision::Fp32 => AfvPrecision::Fp32,
            Precision::Bf16 => AfvPrecision::Bf16,
        };
        let opts = Options { device, pick: Pick::Find, precision: precision_ };
        let net =
            AfvV2::open(model_dir, &opts).wrap_err_with(|| format!("load AFV v2 from {}", model_dir.display()))?;
        let free = free_bytes(device)?;
        let load = &net.load;
        let mut provenance = Map::new();
        provenance.insert("network".into(), json!(LIBRARY));
        provenance.insert("weights".into(), json!("checkpoint_final.pth, read without torch"));
        provenance.insert("cudnn".into(), json!(load.cudnn));
        provenance.insert("ptx_tier".into(), json!(load.ptx_tier));
        provenance.insert("algo".into(), json!("find"));
        // acc (2 x float32), wsum (float32), semantic (2 x uint8) and tracing (2 x uint8 at 1/8)
        let column_bytes_per_voxel = 8.0 + 4.0 + 2.0 + 0.25;
        Ok(Opened {
            gpu: load.gpu.clone(),
            engine: Box::new(Native(net)),
            free_bytes: free,
            column_bytes_per_voxel,
            provenance,
            libraries: super::libs::mapped(),
        })
    }

    /// Free device memory, through the device's primary context (the one the engine runs in)
    fn free_bytes(device: usize) -> Result<u64> {
        let ctx = cudarc::driver::CudaContext::new(device).wrap_err("open the CUDA device")?;
        ctx.bind_to_thread()?;
        let (free, _total) = cudarc::driver::result::mem_get_info()?;
        Ok(free as u64)
    }
}

/// cuDNN 9 and cuBLAS 12 from the `predict` extra's pip packages
///
/// The engine (through cudarc) dlopens each library by name, trying a fixed list of names in order
/// (`libcudnn.so`, ..., `libcudnn.so.9`; `libcublas.so`, ..., `libcublas.so.12`); the first that opens wins, and
/// a name matches a library already loaded into the process by its soname before the loader's search path is
/// tried. So the pip copies in this binary's own environment (`<prefix>/lib/python3.*/site-packages/nvidia/*/lib`
/// beside `<prefix>/bin/fibrs`) are loaded here first, by absolute path with `RTLD_GLOBAL`, and the engine's
/// `libcudnn.so.9` and `libcublas.so.12` resolve to them. A system library under a name the engine tries
/// earlier (an unversioned `libcublas.so`, say) still wins; the summary records the files the process mapped.
/// `FIBRS_CUDA_LIBS=system` skips the pip copies
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod libs {
    use std::path::{Path, PathBuf};

    /// (pip files, dependencies first; their directory under site-packages)
    const PIP: [(&[&str], &str); 2] =
        [(&["libcublasLt.so.12", "libcublas.so.12"], "nvidia/cublas/lib"), (&["libcudnn.so.9"], "nvidia/cudnn/lib")];

    /// The site-packages directories of the environment this binary is installed in
    fn site_packages() -> Vec<PathBuf> {
        let mut prefixes = Vec::new();
        if let Some(prefix) = std::env::current_exe().ok().and_then(|e| e.parent()?.parent().map(Path::to_path_buf)) {
            prefixes.push(prefix);
        }
        if let Some(venv) = std::env::var_os("VIRTUAL_ENV") {
            prefixes.push(PathBuf::from(venv));
        }

        let mut out = Vec::new();
        for prefix in prefixes {
            for lib in ["lib", "lib64"] {
                let Ok(entries) = std::fs::read_dir(prefix.join(lib)) else { continue };
                let mut found: Vec<PathBuf> = entries
                    .filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("python3")))
                    .map(|p| p.join("site-packages"))
                    .filter(|p| p.is_dir())
                    .collect();
                found.sort();
                for p in found {
                    if !out.contains(&p) {
                        out.push(p);
                    }
                }
            }
        }
        out
    }

    /// Loads the pip copies of cuDNN and cuBLAS from this binary's environment, when it has them
    pub fn prefer_environment() {
        if std::env::var("FIBRS_CUDA_LIBS").is_ok_and(|v| v == "system") {
            return;
        }

        let sites = site_packages();
        for (files, sub) in PIP {
            let Some(dir) = sites.iter().map(|s| s.join(sub)).find(|d| files.iter().all(|f| d.join(f).is_file()))
            else {
                continue;
            };
            for f in files {
                let path = dir.join(f);
                // SAFETY: cuDNN's and cuBLAS's initialisers have no preconditions; the handle is kept for the
                // life of the process, so the engine's later dlopen by soname finds this copy
                match unsafe {
                    libloading::os::unix::Library::open(
                        Some(&path),
                        libloading::os::unix::RTLD_NOW | libloading::os::unix::RTLD_GLOBAL,
                    )
                } {
                    Ok(lib) => std::mem::forget(lib),
                    Err(e) => eprintln!("fibrs predict: could not load {}: {e}; using the system copy", path.display()),
                }
            }
        }
    }

    /// The cuDNN and cuBLAS files mapped into this process
    pub fn mapped() -> Vec<String> {
        let maps = std::fs::read_to_string("/proc/self/maps").unwrap_or_default();
        let mut out: Vec<String> = maps
            .lines()
            .filter_map(|l| l.split_whitespace().nth(5))
            .filter(|p| {
                let name = Path::new(p).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                name.starts_with("libcudnn") || name.starts_with("libcublas")
            })
            .map(str::to_owned)
            .collect();
        out.sort();
        out.dedup();
        out
    }
}
