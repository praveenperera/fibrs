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

/// Most voxels one column may have: the engine indexes its two-channel accumulators with 32-bit integers and
/// refuses a column of more than `u32::MAX / 2` voxels when it begins it
pub const MAX_COLUMN_VOXELS: usize = u32::MAX as usize / 2;

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

    use color_eyre::eyre::{Result, WrapErr, bail};
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
        // before the checkpoint is read: without a driver or the device there is nothing to load it for
        probe(device)?;
        let net =
            AfvV2::open(model_dir, &opts).wrap_err_with(|| format!("load AFV v2 from {}", model_dir.display()))?;
        let free = free_bytes(device)?;
        warn_cublas_major(net.load.cublas);
        let load = &net.load;
        let mut provenance = Map::new();
        provenance.insert("network".into(), json!(LIBRARY));
        provenance.insert("weights".into(), json!("checkpoint_final.pth, read without torch"));
        provenance.insert("cudnn".into(), json!(load.cudnn));
        provenance.insert("cublas".into(), json!(load.cublas));
        provenance.insert("ptx_tier".into(), json!(load.ptx_tier));
        provenance.insert("algo".into(), json!("find"));
        // the convolution algorithms cuDNN chose, one per distinct convolution in forward order: they set the
        // rounding, so a resume whose choices differ (another cache, cuDNN or driver) is refused
        provenance.insert("cudnn_algos".into(), json!(load.algos));
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

    /// Warns when the loaded cuBLAS is not 12.x: the engine binds cuBLAS 12 but asks for the unversioned
    /// `libcublas.so` first, so a newer toolkit's copy on the loader's path wins over the 12.x ones. The run goes
    /// on: the version is in the provenance, so the output says which cuBLAS made it and a resume under another
    /// is refused
    fn warn_cublas_major(version: i32) {
        // cublasGetVersion: major * 10000 + minor * 100 + patch
        let (major, minor, patch) = (version / 10000, version / 100 % 100, version % 100);
        if major != 12 {
            eprintln!(
                "fibrs predict: warning: loaded cuBLAS {major}.{minor}.{patch}, not 12.x, because the loader found \
                 an unversioned libcublas.so first; to use cuBLAS 12, put a directory holding libcublas.so -> \
                 libcublas.so.12 and libcudnn.so -> libcudnn.so.9 symlinks first on LD_LIBRARY_PATH"
            );
        }
    }

    /// Checks that the NVIDIA driver loads and has device `device`; cudarc would panic on a missing driver
    fn probe(device: usize) -> Result<()> {
        // SAFETY: the driver library's initialisers have no preconditions
        if !unsafe { cudarc::driver::sys::is_culib_present() } {
            bail!("no NVIDIA driver found (libcuda.so); fibrs predict needs an NVIDIA GPU and its driver");
        }
        cudarc::driver::CudaContext::new(device).wrap_err_with(|| format!("open CUDA device {device}"))?;
        Ok(())
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

    /// The site-packages directories of the environment this binary is installed in: the one whose
    /// `fibrs-*.dist-info` installed it first, when a prefix holds several Python versions, so a binary installed
    /// by one Python does not load another's libraries
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
                // the installing Python's site-packages first, then the rest in a fixed order
                found.sort_by_key(|p| (!installed_fibrs(p), p.clone()));
                for p in found {
                    if !out.contains(&p) {
                        out.push(p);
                    }
                }
            }
        }
        out
    }

    /// Whether `site` holds the `fibrs` wheel's install record
    fn installed_fibrs(site: &Path) -> bool {
        let Ok(entries) = std::fs::read_dir(site) else { return false };
        entries.filter_map(Result::ok).any(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            name.starts_with("fibrs-") && name.ends_with(".dist-info")
        })
    }

    /// Whether `site` holds all of a pip package's files
    fn holds(site: &Path, (files, sub): (&[&str], &str)) -> bool {
        files.iter().all(|f| site.join(sub).join(f).is_file())
    }

    /// Loads the pip copies of cuDNN and cuBLAS from this binary's environment, when it has them, all from one
    /// site-packages: the first that holds both, else the first that holds either (the one that installed fibrs
    /// comes first), whose missing library then comes from the system, never from another install
    pub fn prefer_environment() {
        if std::env::var("FIBRS_CUDA_LIBS").is_ok_and(|v| v == "system") {
            return;
        }

        let sites = site_packages();
        let site = sites
            .iter()
            .find(|s| PIP.iter().all(|&p| holds(s, p)))
            .or_else(|| sites.iter().find(|s| PIP.iter().any(|&p| holds(s, p))));
        let Some(site) = site else { return };
        for (files, sub) in PIP {
            if !holds(site, (files, sub)) {
                continue;
            }
            let dir = site.join(sub);
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
