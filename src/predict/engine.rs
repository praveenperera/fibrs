//! The network: AFV v2 on volkit's `AfvV2`, which runs the normalisation, the network, the softmax, the blend
//! and the column's finalise on the GPU, in the reference's float32 arithmetic
//!
//! The engine is CUDA on Linux and Windows x86_64 and Metal on macOS arm64 (`fibrs_engine`, set by build.rs);
//! elsewhere `fibrs predict` plans (`--dry-run`) but refuses to run. The Metal engine runs fp32 only, with float32
//! convolution operands where CUDA's fp32 contract runs TF32, so its output sits inside TF32's rounding of CUDA's

use std::path::Path;

use color_eyre::eyre::Result;
#[cfg(not(fibrs_engine))]
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
            "bf16" if cfg!(fibrs_engine = "metal") => {
                Err("--precision bf16 runs on CUDA only; the Metal engine runs fp32 (the default)".into())
            }
            "bf16" => Ok(Self::Bf16),
            "fp32-ieee" | "fp16" if cfg!(fibrs_engine = "metal") => Err(format!(
                "--precision {name} is a PyTorch mode the native engine does not run; the Metal engine runs fp32 \
                 only (the default)"
            )),
            "fp32-ieee" | "fp16" => Err(format!(
                "--precision {name} is a PyTorch mode the native engine does not run; it runs fp32 (the reference \
                 contract, the default) or bf16"
            )),
            _ if cfg!(fibrs_engine = "metal") => {
                Err(format!("--precision {name:?}: the Metal engine runs fp32 only (the default)"))
            }
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
            // Metal rounds differently from CUDA's TF32, so the label tells the two apart and a resume on the
            // other engine is refused
            Self::Fp32 if cfg!(fibrs_engine = "metal") => {
                "fp32 (float32 tensors, float32 convolution and GEMM math; fibrs native engine on Metal)"
            }
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

/// `--precision`'s help, for the engine this build runs
pub const PRECISION_HELP: &str = if cfg!(fibrs_engine = "metal") {
    "fp32, the default and the only precision on Metal: float32 tensors with float32 convolutions, within TF32's \
     rounding of the NVIDIA reference (measured in docs/predict.md)"
} else {
    "fp32 (default): float32 tensors with TF32 tensor-core convolutions, the precision PyTorch uses for the reference \
     by default. bf16 (opt-in): faster, small output differences (measured in docs/benchmarks.md)"
};

/// One column's outputs, `None` when no tile ran
// built only by the GPU engine, which other targets lack
#[cfg_attr(not(fibrs_engine), allow(dead_code))]
pub struct ColumnOut {
    pub semantic: Option<Vec<u8>>,
    pub tracing: Option<Vec<u8>>,
    pub uncovered: u64,
}

/// A written column's semantic and tracing buffers, for the engine to download a later column into
pub type Buffers = (Vec<u8>, Vec<u8>);

/// A column accumulator on the GPU: `begin` a column, `add` its non-air tiles in order, `finish` it
pub trait Engine {
    fn begin(&mut self, shape: [usize; 3], y0: usize, x0: usize) -> Result<()>;
    fn add(&mut self, tiles: &[u8], origins: &[Origin]) -> Result<()>;
    /// Ends the column, its outputs written into `reuse`, an earlier column's buffers, or fresh ones
    fn finish(&mut self, quantize: bool, reuse: Buffers) -> Result<ColumnOut>;
}

/// What the run needs to know about the opened engine
// built only by the GPU engine, which other targets lack
#[cfg_attr(not(fibrs_engine), allow(dead_code))]
pub struct Opened {
    pub engine: Box<dyn Engine>,
    /// The GPU's name
    pub gpu: String,
    /// What a column may take on the device
    pub column: ColumnMemory,
    /// Provenance fields naming the engine, its library and the weights it loaded
    pub provenance: Map<String, Value>,
    /// The cuDNN and cuBLAS files the process mapped
    pub libraries: Vec<String>,
}

/// What a column may take on the device once the engine has loaded
#[cfg_attr(not(fibrs_engine), allow(dead_code))]
pub struct ColumnMemory {
    /// Device memory free once the engine has loaded
    pub free_bytes: u64,
    /// Device bytes one output voxel of a column takes at its peak
    pub bytes_per_voxel: f64,
    /// The device's limit on one buffer, where it has one below its memory
    pub buffer_limit: Option<BufferLimit>,
}

/// A device's limit on one buffer (Metal's `maxBufferLength`), against the column's largest buffer
// only the Metal engine has one
#[cfg_attr(not(fibrs_engine = "metal"), allow(dead_code))]
pub struct BufferLimit {
    /// The most bytes one buffer holds
    pub max_bytes: u64,
    /// Bytes one output voxel takes in the column's largest buffer
    pub bytes_per_voxel: f64,
}

// acc (2 x float32), wsum (float32), semantic (2 x uint8) and tracing (2 x uint8 at 1/8)
#[cfg(fibrs_engine)]
const COLUMN_BYTES_PER_VOXEL: f64 = 8.0 + 4.0 + 2.0 + 0.25;

/// Most tiles one `add` takes
pub const MAX_BATCH: usize = 16;

/// Most voxels one column may have: the engine indexes its two-channel accumulators with 32-bit integers and
/// refuses a column of more than `u32::MAX / 2` voxels when it begins it
pub const MAX_COLUMN_VOXELS: usize = u32::MAX as usize / 2;

/// The kind of device the engine runs on, `cuda` or `metal`, for `--device` and the provenance
pub const DEVICE_KIND: &str = if cfg!(fibrs_engine = "metal") { "metal" } else { "cuda" };

/// The engine's device as messages name it, `CUDA` or `Metal`
const DEVICE_NAME: &str = if cfg!(fibrs_engine = "metal") { "Metal" } else { "CUDA" };

/// The device ordinal of `--device`: `cuda`, `cuda:N` or `N` on CUDA builds, `metal`, `metal:N` or `N` on Metal ones
pub fn device_index(device: &str) -> Result<usize, String> {
    let n = if device == DEVICE_KIND {
        "0"
    } else {
        device.strip_prefix(DEVICE_KIND).and_then(|d| d.strip_prefix(':')).unwrap_or(device)
    };
    n.parse().map_err(|_| {
        format!("fibrs predict runs on a {DEVICE_NAME} device ({DEVICE_KIND}, {DEVICE_KIND}:N), not {device:?}")
    })
}

#[cfg(fibrs_engine = "cuda")]
pub fn open(model_dir: &Path, device: usize, precision: Precision) -> Result<Opened> {
    native::open(model_dir, device, precision)
}

#[cfg(fibrs_engine = "metal")]
pub fn open(model_dir: &Path, device: usize, precision: Precision) -> Result<Opened> {
    metal::open(model_dir, device, precision)
}

#[cfg(not(fibrs_engine))]
pub fn open(_model_dir: &Path, _device: usize, _precision: Precision) -> Result<Opened> {
    bail!(
        "fibrs predict needs an NVIDIA GPU on Linux or Windows x86_64, or Apple silicon on macOS; this build has no GPU \
         engine (--dry-run plans a run without one)"
    )
}

#[cfg(fibrs_engine = "cuda")]
mod native {
    use std::path::Path;

    use color_eyre::eyre::{Result, WrapErr, bail};
    use serde_json::{Map, json};
    use volkit::dnn::Pick;
    use volkit::nnunet::afv2::{AfvPrecision, AfvV2, Options};

    use super::{Buffers, COLUMN_BYTES_PER_VOXEL, ColumnMemory, ColumnOut, Engine, Opened, Origin, Precision};

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

        fn finish(&mut self, quantize: bool, (semantic, tracing): Buffers) -> Result<ColumnOut> {
            let out = self.0.finish_into(quantize, semantic, tracing)?;
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
        // a CUDA allocation is bounded only by the free memory
        let column = ColumnMemory { free_bytes: free, bytes_per_voxel: COLUMN_BYTES_PER_VOXEL, buffer_limit: None };
        Ok(Opened {
            gpu: load.gpu.clone(),
            engine: Box::new(Native(net)),
            column,
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
        // only Linux can get here: Windows installs each cuBLAS under a name that carries its major
        // (`cublas64_13.dll`), which the 12.x names never match
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
            bail!("no NVIDIA driver found ({}); fibrs predict needs an NVIDIA GPU and its driver", crate::gpu::DRIVER);
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

#[cfg(fibrs_engine = "metal")]
mod metal {
    use std::path::Path;

    use color_eyre::eyre::{Result, WrapErr};
    use serde_json::{Map, json};
    use volkit::nnunet::afv2::{AfvPrecision, AfvV2, Options};

    use super::{
        BufferLimit, Buffers, COLUMN_BYTES_PER_VOXEL, ColumnMemory, ColumnOut, Engine, Opened, Origin, Precision,
    };

    /// The engine library this build links, for the provenance
    const LIBRARY: &str = concat!(env!("FIBRS_VOLKIT"), ", AfvV2 on Metal");

    struct Metal(AfvV2);

    impl Engine for Metal {
        fn begin(&mut self, shape: [usize; 3], y0: usize, x0: usize) -> Result<()> {
            self.0.begin(shape, y0, x0)
        }

        fn add(&mut self, tiles: &[u8], origins: &[Origin]) -> Result<()> {
            self.0.add(tiles, origins).map(drop)
        }

        fn finish(&mut self, quantize: bool, (semantic, tracing): Buffers) -> Result<ColumnOut> {
            let out = self.0.finish_into(quantize, semantic, tracing)?;
            Ok(ColumnOut { semantic: out.semantic, tracing: out.tracing, uncovered: out.uncovered })
        }
    }

    pub fn open(model_dir: &Path, device: usize, precision: Precision) -> Result<Opened> {
        // `Precision::parse` refuses bf16 on Metal builds
        debug_assert_eq!(precision, Precision::Fp32);
        let opts = Options { device, precision: AfvPrecision::Fp32 };
        let net =
            AfvV2::open(model_dir, &opts).wrap_err_with(|| format!("load AFV v2 from {}", model_dir.display()))?;
        let load = &net.load;
        let mut provenance = Map::new();
        provenance.insert("network".into(), json!(LIBRARY));
        provenance.insert("weights".into(), json!("checkpoint_final.pth, read without torch"));
        provenance.insert("metal_math".into(), json!("float32 matmul2d operands and accumulation, fp contraction off"));
        // Winograd rounds differently from a direct conv, so a resume under another level list would mix outputs
        provenance.insert("metal_winograd_levels".into(), json!(load.winograd_levels));
        // unified memory: the column's buffers come out of the working set Metal recommends for this process, less
        // what the model already holds
        let free = load.working_set.saturating_sub(load.model_bytes) as u64;
        // and each of its buffers within Metal's per-buffer limit, the accumulator (2 x float32) the largest
        let buffer_limit = Some(BufferLimit { max_bytes: load.max_buffer as u64, bytes_per_voxel: 8.0 });
        let column = ColumnMemory { free_bytes: free, bytes_per_voxel: COLUMN_BYTES_PER_VOXEL, buffer_limit };
        Ok(Opened { gpu: load.gpu.clone(), engine: Box::new(Metal(net)), column, provenance, libraries: Vec::new() })
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
///
/// On Windows the wheels keep the DLLs in `nvidia/*/bin` under `<prefix>\Lib\site-packages`, beside
/// `<prefix>\Scripts\fibrs.exe`. Each is loaded by full path, which resolves its imports from its own directory
/// (so `cublas64_12.dll` binds the `cublasLt64_12.dll` beside it), and Windows reuses a loaded DLL for a load by
/// the same name: volkit's `dll::pin` then takes these copies, cuDNN's sublibraries from beside `cudnn64_9.dll`
/// and NVRTC's builtins from beside the pip `nvrtc64_120_0.dll`, before cudarc opens anything
#[cfg(all(any(target_os = "linux", target_os = "windows"), target_arch = "x86_64"))]
mod libs {
    use std::cmp::Reverse;
    use std::path::{Path, PathBuf};
    use std::time::SystemTime;

    /// (pip files, dependencies first; their directory under site-packages)
    #[cfg(unix)]
    const PIP: [(&[&str], &str); 2] =
        [(&["libcublasLt.so.12", "libcublas.so.12"], "nvidia/cublas/lib"), (&["libcudnn.so.9"], "nvidia/cudnn/lib")];

    /// (pip files, dependencies first; their directory under site-packages)
    #[cfg(windows)]
    const PIP: [(&[&str], &str); 2] =
        [(&["cublasLt64_12.dll", "cublas64_12.dll"], "nvidia/cublas/bin"), (&["cudnn64_9.dll"], "nvidia/cudnn/bin")];

    /// Pip files loaded, before [`PIP`]'s, from the site chosen for cuDNN and cuBLAS when it has them, but never
    /// the reason a site is chosen
    #[cfg(unix)]
    const COMPANIONS: [(&[&str], &str); 0] = [];

    /// Pip files loaded, before [`PIP`]'s, from the site chosen for cuDNN and cuBLAS when it has them, but never
    /// the reason a site is chosen: NVRTC, which cuDNN's runtime-compiled engines use and the cuDNN and cuBLAS
    /// Windows wheels bring. volkit's `dll::pin` takes NVRTC only from a copy already loaded or from the system's
    /// trusted places, so without this the venv's copy would go unused
    #[cfg(windows)]
    const COMPANIONS: [(&[&str], &str); 1] = [(&["nvrtc64_120_0.dll"], "nvidia/cuda_nvrtc/bin")];

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
            for group in site_groups(&prefix) {
                let mut found: Vec<PathBuf> = group.into_iter().filter(|p| p.is_dir()).collect();
                // the installing Python's site-packages first: the newest fibrs install record, as the last
                // install is the one that wrote bin/fibrs; then the rest in a fixed order
                found.sort_by_key(|p| (Reverse(fibrs_record(p)), p.clone()));
                for p in found {
                    if !out.contains(&p) {
                        out.push(p);
                    }
                }
            }
        }
        out
    }

    /// The site-packages directories a prefix may hold, in groups ranked apart: `lib/python3.*/site-packages`,
    /// then the same under `lib64`
    #[cfg(unix)]
    fn site_groups(prefix: &Path) -> Vec<Vec<PathBuf>> {
        let group = |lib: &str| -> Vec<PathBuf> {
            let Ok(entries) = std::fs::read_dir(prefix.join(lib)) else { return Vec::new() };
            entries
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("python3")))
                .map(|p| p.join("site-packages"))
                .collect()
        };
        vec![group("lib"), group("lib64")]
    }

    /// The site-packages directories a prefix may hold: `Lib\site-packages` in a venv or an installed Python,
    /// then `site-packages` beside the `Scripts` of a per-user install (`%APPDATA%\Python\Python3*`)
    #[cfg(windows)]
    fn site_groups(prefix: &Path) -> Vec<Vec<PathBuf>> {
        vec![vec![prefix.join("Lib").join("site-packages")], vec![prefix.join("site-packages")]]
    }

    /// When the `fibrs` wheel's install record in `site` was written, the newest when there are several
    fn fibrs_record(site: &Path) -> Option<SystemTime> {
        let entries = std::fs::read_dir(site).ok()?;
        entries
            .filter_map(Result::ok)
            .filter(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                name.starts_with("fibrs-") && name.ends_with(".dist-info")
            })
            .filter_map(|e| e.metadata().and_then(|m| m.modified()).ok())
            .max()
    }

    /// Whether `site` holds all of a pip package's files
    fn holds(site: &Path, (files, sub): (&[&str], &str)) -> bool {
        files.iter().all(|f| site.join(sub).join(f).is_file())
    }

    /// Loads the pip copies of cuDNN and cuBLAS from this binary's environment, when it has them, all from one
    /// site-packages: the one that installed fibrs when it holds either, else the first that holds both, else
    /// the first that holds either. A library that site lacks comes from the system, never from another install
    pub fn prefer_environment() {
        if std::env::var("FIBRS_CUDA_LIBS").is_ok_and(|v| v == "system") {
            return;
        }

        let sites = site_packages();
        let any = |s: &&PathBuf| PIP.iter().any(|&p| holds(s, p));
        let site = sites
            .iter()
            .find(|s| fibrs_record(s).is_some() && any(s))
            .or_else(|| sites.iter().find(|s| PIP.iter().all(|&p| holds(s, p))))
            .or_else(|| sites.iter().find(any));
        let Some(site) = site else { return };
        for (files, sub) in COMPANIONS.into_iter().chain(PIP) {
            if !holds(site, (files, sub)) {
                continue;
            }
            let dir = site.join(sub);
            for f in files {
                let path = dir.join(f);
                // SAFETY: cuDNN's and cuBLAS's initialisers have no preconditions; the handle is kept for the
                // life of the process, so the engine's later load by name finds this copy
                match unsafe { open_global(&path) } {
                    Ok(lib) => std::mem::forget(lib),
                    Err(e) => eprintln!("fibrs predict: could not load {}: {e}; using the system copy", path.display()),
                }
            }
        }
    }

    /// Opens `path` so that a later load by its soname resolves to it
    ///
    /// # Safety
    ///
    /// Loading a library runs its initialisers
    #[cfg(unix)]
    unsafe fn open_global(path: &Path) -> Result<libloading::Library, libloading::Error> {
        use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_NOW};

        unsafe { Library::open(Some(path), RTLD_NOW | RTLD_GLOBAL) }.map(libloading::Library::from)
    }

    /// Opens `path`; Windows matches a later load by name against the DLLs already loaded
    ///
    /// # Safety
    ///
    /// Loading a library runs its initialisers
    #[cfg(windows)]
    unsafe fn open_global(path: &Path) -> Result<libloading::Library, libloading::Error> {
        unsafe { crate::pyenv::open_library(path) }
    }

    /// Whether a file is one of cuDNN's or cuBLAS's libraries, by its name
    #[cfg(unix)]
    fn cuda_library(path: &str) -> bool {
        let name = Path::new(path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        name.starts_with("libcudnn") || name.starts_with("libcublas")
    }

    /// Whether a file is one of cuDNN's or cuBLAS's DLLs, by its name in any case, as Windows matches names
    #[cfg(windows)]
    fn cuda_library(path: &str) -> bool {
        let name = Path::new(path).file_name().map(|n| n.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
        name.starts_with("cudnn") || name.starts_with("cublas")
    }

    /// The cuDNN and cuBLAS files mapped into this process
    #[cfg(target_os = "linux")]
    pub fn mapped() -> Vec<String> {
        let maps = std::fs::read_to_string("/proc/self/maps").unwrap_or_default();
        let mut out: Vec<String> = maps
            .lines()
            .filter_map(|l| l.split_whitespace().nth(5))
            .filter(|p| cuda_library(p))
            .map(str::to_owned)
            .collect();
        out.sort();
        out.dedup();
        out
    }

    /// The cuDNN and cuBLAS DLLs loaded into this process, from its module list
    #[cfg(windows)]
    pub fn mapped() -> Vec<String> {
        use windows_sys::Win32::Foundation::HMODULE;
        use windows_sys::Win32::System::LibraryLoader::GetModuleFileNameW;
        use windows_sys::Win32::System::ProcessStatus::K32EnumProcessModules;
        use windows_sys::Win32::System::Threading::GetCurrentProcess;

        // the list can grow between the two calls, so ask again until it fits
        let mut modules: Vec<HMODULE> = vec![std::ptr::null_mut(); 512];
        loop {
            let size = std::mem::size_of_val(modules.as_slice()) as u32;
            let mut needed = 0;
            // SAFETY: the buffer holds `size` bytes of module handles, and the call writes at most that many
            let ok = unsafe { K32EnumProcessModules(GetCurrentProcess(), modules.as_mut_ptr(), size, &mut needed) };
            if ok == 0 {
                return Vec::new();
            }
            let count = needed as usize / std::mem::size_of::<HMODULE>();
            if needed <= size {
                modules.truncate(count);
                break;
            }
            modules.resize(count, std::ptr::null_mut());
        }

        let mut out: Vec<String> = modules
            .iter()
            .filter_map(|&module| {
                let mut name = vec![0u16; 32768];
                // SAFETY: the buffer holds `name.len()` UTF-16 units, and the call writes at most that many
                let n = unsafe { GetModuleFileNameW(module, name.as_mut_ptr(), name.len() as u32) } as usize;
                (n > 0).then(|| String::from_utf16_lossy(&name[..n]))
            })
            .filter(|p| cuda_library(p))
            .collect();
        out.sort();
        out.dedup();
        out
    }
}
