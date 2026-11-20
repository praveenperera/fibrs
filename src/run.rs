//! The command line and the run (`cli.run`): validation, the work directory, the stages and the export
//!
//! Flags, defaults, choices, validation messages and exit codes are the reference's, so the executable
//! can replace `avf-spline-generator` in a venv. The work directory differs: fields live in memory, and
//! the checkpoint is one finished curve file per family instead of a part file and an fsync per 16
//! seeds. `run.json` carries a `native` key, so neither tool resumes the other's work directory

use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{SyncSender, sync_channel};
use std::time::Instant;

use clap::error::ErrorKind;
use clap::{Arg, ArgAction, Command, CommandFactory, Parser, ValueEnum};
use color_eyre::eyre::{Result, bail, eyre};
use rayon::prelude::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::afv::{Curve, ExportOptions, MinLength, Summary, export_afv, tracing_to_native};
use crate::axes::{Lapack, direction_field, find_lapack};
use crate::fields::{FamilyFields, Presence, RegionFrame, Seed, catalogue, centrality};
use crate::finish::{CachePools, FinishCtx, NativeCache, NativeSampler};
use crate::gpu::{AxesCounts, CpuReason, Finished, Gpu, GpuFinisher, Pieces, SlabLoads, Stages};
use crate::input::Predictions;
use crate::kernels::{Coverage, DirField, DirForm, Directions, Fields, RawCurve, SplitDirs};
use crate::log;
use crate::produce::{Flow, Producer, Speculation, Stats};
use crate::stage1::{Fused, Separate};
use crate::tiles::{Tile, Tiling, alignment, seam_curves};
use crate::vol::{EDGE, Shape, Vol};

/// The reference release whose interface and output this reproduces
pub const REFERENCE_VERSION: &str = "0.2.0";

/// Whether the binary runs as the drop-in `avf-spline-generator`, by the last component of the path it was run by,
/// taken as `Path::file_name` takes it: trailing `/` and `.` components skipped. It allocates nothing, since the
/// global allocator asks before `std::env::args` can be collected
pub fn dropin(arg0: &[u8]) -> bool {
    let name = arg0.split(|&b| b == b'/').rev().find(|c| !c.is_empty() && *c != b".");
    name == Some(b"avf-spline-generator".as_slice())
}

/// What `--version` prints: run as the drop-in, the version of the reference release it stands in for, so callers
/// that check it see the tool they expect; run as `fibrs`, this package's own version
pub fn version(arg0: &[u8]) -> &'static str {
    if dropin(arg0) { REFERENCE_VERSION } else { env!("CARGO_PKG_VERSION") }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum ChannelOrder {
    #[value(name = "VH")]
    Vh,
    #[value(name = "HV")]
    Hv,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Family {
    #[value(name = "both")]
    Both,
    #[value(name = "V")]
    V,
    #[value(name = "H")]
    H,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Device {
    /// The GPU when a usable one is found, else the CPU
    #[value(name = "auto")]
    Auto,
    #[value(name = "cpu")]
    Cpu,
    #[value(name = "cuda")]
    Cuda,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Storage {
    #[value(name = "dense")]
    Dense,
    #[value(name = "compressed")]
    Compressed,
}

/// Trace H/V probability volumes into AFV splines.
#[derive(Parser, Debug)]
#[command(
    name = "avf-spline-generator",
    disable_version_flag = true,
    infer_long_args = true,
    args_override_self = true,
    allow_negative_numbers = true
)]
pub struct Args {
    /// Native (2,Z,Y,X) NPY or Zarr
    #[arg(long)]
    pub predictions: PathBuf,
    /// Array key inside a native Zarr group
    #[arg(long)]
    pub array: Option<String>,
    #[arg(long, value_enum, default_value = "VH")]
    pub channel_order: ChannelOrder,
    /// Optional prepooled 2x array
    #[arg(long)]
    pub tracing_predictions: Option<PathBuf>,
    /// Array key inside the prepooled Zarr group
    #[arg(long)]
    pub tracing_array: Option<String>,
    /// Isotropic native voxel size
    #[arg(long)]
    pub voxel_size_um: f64,
    /// Destination .afv
    #[arg(long)]
    pub output: PathBuf,
    /// Persistent scratch directory
    #[arg(long)]
    pub work_dir: PathBuf,
    /// Resume the same immutable inputs
    #[arg(long)]
    pub resume: bool,
    #[arg(long, value_enum, default_value = "both")]
    pub family: Family,
    /// AFV export cutoff (inclusive)
    #[arg(long)]
    pub min_length_mm: Option<f64>,
    // Set, not clap's default Append: a repeated three-value option replaces the earlier one, as in argparse
    #[arg(long, num_args = 3, action = ArgAction::Set, value_names = ["X", "Y", "Z"], default_values_t = [0i64, 0, 0])]
    pub origin_xyz: Vec<i64>,
    /// Full native domain; required for a nonzero crop origin
    #[arg(long, num_args = 3, action = ArgAction::Set, value_names = ["Z", "Y", "X"])]
    pub volume_shape_zyx: Option<Vec<i64>>,
    /// Optional AFV scan identifier
    #[arg(long)]
    pub coordinate_space: Option<String>,
    /// Where the direction field and centrality run (tracing always runs on the CPU): auto uses an NVIDIA GPU when
    /// one is usable and the CPU otherwise, cpu never uses the GPU, and cuda fails without one. A GPU run writes the
    /// original's --device cuda output, whose centrality can differ from its cpu output by a rounding. A resumed
    /// run keeps the rounding it started with, so one started on the CPU keeps centrality there. A GPU run also
    /// finishes curves there; its prediction slabs decode on the CPU with c-blosc's checks unless GPU decoding
    /// with nvCOMP is opted in to (FIBRS_NVCOMP=on, with fibrs[gpu-zstd] installed, or FIBRS_NVCOMP set to a
    /// libnvcomp.so.6 path). On a corrupt chunk nvCOMP's behaviour is undefined: it may give wrong output or a
    /// CUDA fault
    #[arg(long, value_enum, default_value = "auto")]
    pub device: Device,
    /// Visible CUDA device index for the GPU; without it, the first usable visible GPU
    #[arg(long)]
    pub cuda_device: Option<i64>,
    #[arg(long)]
    pub threads: Option<i64>,
    #[arg(long, default_value_t = 128)]
    pub native_cache_mib: i64,
    /// Intermediate fields: memory-mapped NPY or sparse lossless Zarr
    #[arg(long, value_enum, default_value = "dense")]
    pub storage: Storage,
    /// Compressed-mode tracing cache; minimum 16 MiB (default: 1024)
    #[arg(long, default_value_t = 1024)]
    pub field_cache_mib: i64,
    /// Trace as NY x NX tiles in Y and X, one at a time, and join fibres across the seams (default: untiled)
    // Set, as --origin-xyz: a repeated option replaces the earlier one
    #[arg(long, num_args = 2, action = ArgAction::Set, value_names = ["NY", "NX"])]
    pub tiles_yx: Option<Vec<i64>>,
    /// Native voxels each tile is traced beyond its core, rounded up to the tile alignment (default: 256)
    #[arg(long, value_name = "VOXELS")]
    pub tile_halo: Option<i64>,
    /// Print the version and exit
    #[arg(long)]
    version: bool,
}

/// The outcome of argparse's left-to-right pass, replayed before clap parses
#[derive(Debug)]
pub enum Prescan {
    /// A help action came first
    Help,
    /// A version action came first
    Version,
    /// An option error came before any action
    Error(clap::Error),
    /// No action: the arguments for clap, single values written as `--option=value` so that clap reads a
    /// value such as `-.5` as argparse does, not as a flag
    Args(Vec<OsString>),
}

/// Replay argparse's left-to-right pass over `args` (without the program name)
///
/// Long options match by unique prefix, each option takes its value count, and its values are converted
/// as they are read, so an ambiguous prefix or a missing or bad value before a help or version action is
/// an error, as in argparse. Unknown options and stray values only fail at argparse's end, so they are
/// passed on for clap to reject
pub fn prescan(args: &[OsString]) -> Prescan {
    let mut command = Args::command();
    command.build();
    let longs: Vec<&str> = command.get_arguments().filter_map(Arg::get_long).collect();
    let error = |kind, message: String| Prescan::Error(Args::command().error(kind, message));
    let mut out: Vec<OsString> = Vec::with_capacity(args.len());
    let mut i = 0;
    while i < args.len() {
        let original = &args[i];
        let token = original.to_string_lossy();
        i += 1;
        if token == "--" {
            out.extend(args[i - 1..].iter().cloned());
            break;
        }
        if let Some(cluster) = token.strip_prefix("-h") {
            // a short-flag cluster: -h, -hh, -hx (the rest after a non-h is left over); an `=` attaches an
            // explicit value to -h, which argparse rejects
            match cluster.trim_start_matches('h').starts_with('=') {
                true => return error(ErrorKind::TooManyValues, "argument -h/--help: ignored explicit argument".into()),
                false => return Prescan::Help,
            }
        }
        let Some(body) = token.strip_prefix("--").filter(|b| !b.is_empty()) else {
            out.push(original.clone());
            continue;
        };
        let (name, explicit) = match body.split_once('=') {
            Some((name, value)) => (name, Some(value)),
            None => (body, None),
        };
        let candidates: Vec<&str> = match longs.iter().find(|&&l| l == name) {
            Some(&l) => vec![l],
            None => longs.iter().copied().filter(|l| l.starts_with(name)).collect(),
        };
        // a token that names no option is left for clap to reject, as argparse leaves it to its end; one with a
        // space would be a stray value to argparse, which ends the same way
        let long = match candidates[..] {
            [] => {
                out.push(original.clone());
                continue;
            }
            [long] => long,
            _ => {
                let options = candidates.iter().map(|c| format!("--{c}")).collect::<Vec<_>>().join(", ");
                return error(ErrorKind::UnknownArgument, format!("ambiguous option: --{name} could match {options}"));
            }
        };
        let arg = command.get_arguments().find(|a| a.get_long() == Some(long)).expect("a listed option");
        if !arg.get_action().takes_values() {
            if let Some(value) = explicit {
                return error(
                    ErrorKind::TooManyValues,
                    format!("argument --{long}: ignored explicit argument '{value}'"),
                );
            }
            match long {
                "help" => return Prescan::Help,
                "version" => return Prescan::Version,
                _ => out.push(format!("--{long}").into()),
            }
            continue;
        }
        let count = arg.get_num_args().map_or(1, |r| r.min_values());
        let values: Vec<OsString> = match explicit {
            Some(_) if count == 1 => {
                // keep the value's own bytes: only the option name is text
                let bytes = original.as_encoded_bytes();
                let at = bytes.iter().position(|&b| b == b'=').expect("an explicit value") + 1;
                // SAFETY: splitting just after an ASCII '=' keeps a valid encoded-bytes boundary
                vec![unsafe { OsString::from_encoded_bytes_unchecked(bytes[at..].to_vec()) }]
            }
            Some(_) => {
                return error(ErrorKind::WrongNumberOfValues, format!("argument --{long}: expected {count} arguments"));
            }
            None => {
                let values: Vec<OsString> =
                    args[i..].iter().take_while(|v| is_value(&v.to_string_lossy())).take(count).cloned().collect();
                if values.len() < count {
                    let message = format!("argument --{long}: expected {count} argument(s)");
                    return error(ErrorKind::WrongNumberOfValues, message);
                }
                i += count;
                values
            }
        };
        // Python's int() and float() accept surrounding whitespace and single underscores between digits
        let values: Vec<OsString> = match NUMERIC_OPTIONS.contains(&long) {
            true => values.into_iter().map(|v| py_number(&v.to_string_lossy()).map_or(v, OsString::from)).collect(),
            false => values,
        };
        let option = |value: &OsString| {
            let mut joined = OsString::from(format!("--{long}="));
            joined.push(value);
            joined
        };
        // the option's own value parser decides, through a command that holds only this option
        let probe = Command::new("avf-spline-generator").arg(arg.clone().required(false));
        let probe_args: Vec<OsString> = match &values[..] {
            [value] => vec![option(value)],
            _ => std::iter::once(OsString::from(format!("--{long}"))).chain(values.iter().cloned()).collect(),
        };
        let argv = std::iter::once(OsString::from("avf-spline-generator")).chain(probe_args.iter().cloned());
        if let Err(e) = probe.allow_negative_numbers(true).try_get_matches_from(argv) {
            return Prescan::Error(e);
        }
        out.extend(probe_args);
    }
    Prescan::Args(out)
}

/// The options argparse converts with `int` or `float`, and fibrs' own integer options, read the same way
const NUMERIC_OPTIONS: [&str; 10] = [
    "voxel-size-um",
    "min-length-mm",
    "origin-xyz",
    "volume-shape-zyx",
    "cuda-device",
    "threads",
    "native-cache-mib",
    "field-cache-mib",
    "tiles-yx",
    "tile-halo",
];

/// `--tile-halo`'s default: the Z halo v2-run windows carry, which met the crop bar in its tests
const TILE_HALO: i64 = 256;

/// A numeric argument in the form Rust parses, when Python's `int` or `float` would read it differently:
/// whitespace stripped and underscores between digits removed. `None` leaves the text as it is, for the
/// option's parser to accept or reject
fn py_number(text: &str) -> Option<String> {
    let trimmed = text.trim();
    let bytes = trimmed.as_bytes();
    let digit = |i: usize| bytes.get(i).is_some_and(u8::is_ascii_digit);
    let grouped = bytes.iter().enumerate().all(|(i, &b)| b != b'_' || (i > 0 && digit(i - 1) && digit(i + 1)));
    if !grouped {
        return None;
    }
    (trimmed.len() != text.len() || trimmed.contains('_')).then(|| trimmed.replace('_', ""))
}

/// Whether argparse reads `token` as a value rather than an option: no leading dash, a lone dash, a
/// negative number (the parser has no option that looks like one) or text with a space
fn is_value(token: &str) -> bool {
    let Some(rest) = token.strip_prefix('-') else { return true };
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    let negative = digits(rest)
        || rest.split_once('.').is_some_and(|(whole, frac)| whole.bytes().all(|b| b.is_ascii_digit()) && digits(frac));
    rest.is_empty() || negative || token.contains(' ')
}

/// Numba's thread ceiling: `NUMBA_NUM_THREADS`, else the CPUs this process may run on
fn numba_max_threads() -> i64 {
    if let Some(n) = std::env::var("NUMBA_NUM_THREADS").ok().and_then(|v| v.trim().parse().ok()) {
        return n;
    }
    std::thread::available_parallelism().map_or(1, |n| n.get() as i64)
}

/// `Path.resolve()` without `strict`: symlinks of the existing prefix resolved, the rest appended
fn resolve(path: &Path) -> Result<PathBuf> {
    let path = if path.is_absolute() { path.to_path_buf() } else { std::env::current_dir()?.join(path) };
    let mut rest = Vec::new();
    let mut base = path.as_path();
    loop {
        if let Ok(real) = fs::canonicalize(base) {
            let mut out = real;
            for part in rest.iter().rev() {
                out.push(part);
            }
            return Ok(normalize(&out));
        }
        let Some(name) = base.file_name() else { return Ok(normalize(&path)) };
        rest.push(name.to_os_string());
        base = base.parent().unwrap_or(Path::new("/"));
    }
}

fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            other => out.push(other),
        }
    }
    out
}

/// `storage.file_signature`: sha256 over each file's relative path, size and mtime in Python's
/// `sorted(Path)` order, which compares path components rather than whole strings
fn file_signature(path: &Path) -> Result<String> {
    let root = fs::canonicalize(path)?;
    let mut digest = Sha256::new();
    let mut entries: Vec<(Vec<String>, PathBuf)> = Vec::new();
    if root.is_file() {
        let name = root.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        entries.push((vec![name], root.clone()));
    } else {
        walk(&root, &mut Vec::new(), &mut entries)?;
        entries.sort_by(|a, b| a.0.cmp(&b.0));
    }
    for (parts, file) in entries {
        let meta = fs::metadata(&file)?;
        let mtime_ns = meta.mtime() as i128 * 1_000_000_000 + meta.mtime_nsec() as i128;
        digest.update(format!("{}\0{}\0{}\n", parts.join("/"), meta.len(), mtime_ns).as_bytes());
    }
    Ok(digest.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// `rglob("*")` keeping files: symlinks to files count, symlinked directories are not entered
fn walk(dir: &Path, prefix: &mut Vec<String>, out: &mut Vec<(Vec<String>, PathBuf)>) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let kind = entry.file_type()?;
        prefix.push(name);
        if kind.is_dir() {
            walk(&entry.path(), prefix, out)?;
        } else if entry.path().is_file() {
            out.push((prefix.clone(), entry.path()));
        }
        prefix.pop();
    }
    Ok(())
}

/// `save_json`: indented JSON, newline, fsync, rename
fn save_json(path: &Path, value: &Value) -> Result<()> {
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(".tmp");
    let temporary = PathBuf::from(temporary);
    let mut f = File::create(&temporary)?;
    f.write_all(crate::pyfmt::py_json_indent(value).as_bytes())?;
    f.write_all(b"\n")?;
    f.sync_all()?;
    fs::rename(&temporary, path)?;
    Ok(())
}

/// An exclusive, non-blocking lock on `work/.lock` held for the run (`FileLock(timeout=0)`)
struct WorkLock(#[allow(dead_code)] File);

impl WorkLock {
    fn acquire(path: &Path) -> Result<Self> {
        let f = OpenOptions::new().read(true).write(true).create(true).truncate(true).mode(0o644).open(path)?;
        if unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            bail!("The file lock '{}' could not be acquired.", path.display());
        }
        Ok(Self(f))
    }
}

/// Run the whole pipeline: the reference's `run(args)`
pub fn run(args: Args) -> Result<Summary> {
    let min_mm = args.min_length_mm.map_or(MinLength::Default, MinLength::Given);
    if !args.voxel_size_um.is_finite()
        || args.voxel_size_um <= 0.0
        || !min_mm.value().is_finite()
        || min_mm.value() < 0.0
    {
        bail!("Voxel size must be positive and minimum length nonnegative");
    }
    if args.origin_xyz.iter().min().copied().unwrap_or(0) < 0
        || args.native_cache_mib < 1
        || args.cuda_device.is_some_and(|d| d < 0)
    {
        bail!("Invalid origin, cache size or CUDA device");
    }
    fix_malloc_thresholds();
    let max_threads = numba_max_threads();
    let threads = args.threads.unwrap_or(8.min(max_threads));
    if threads < 1 || threads > max_threads {
        bail!("The number of threads must be between 1 and {max_threads}");
    }
    if args.field_cache_mib < 16 {
        bail!("--field-cache-mib must be at least 16");
    }
    if args.output.exists() {
        bail!("Output already exists: {}", args.output.display());
    }
    let order = match args.channel_order {
        ChannelOrder::Vh => "VH",
        ChannelOrder::Hv => "HV",
    };
    let native = Predictions::open(&args.predictions, args.array.as_deref(), order)?;
    if args.tracing_array.is_some() && args.tracing_predictions.is_none() {
        bail!("--tracing-array requires --tracing-predictions");
    }
    let supplied = match &args.tracing_predictions {
        Some(p) => {
            let s = Predictions::open(p, args.tracing_array.as_deref(), order)?;
            if s.shape != native.shape.map(|n| n / 2) {
                bail!("Prepooled tracing shape must equal native shape // 2");
            }
            Some(s)
        }
        None => {
            if !native.dtype.is_float() {
                log!(
                    "Pooling quantized uint8 inputs. Supply prepooled predictions to preserve \
                     pooling-before-quantization values."
                );
            }
            None
        }
    };
    if native.shape.iter().min().copied().unwrap_or(0) < 8 {
        bail!("Native shape must be at least (8,8,8)");
    }
    if native.shape.iter().any(|n| n % 2 == 1) {
        log!("The final unpaired voxel plane on odd-sized axes is excluded from tracing");
    }
    if args.origin_xyz.iter().any(|&v| v != 0) && args.volume_shape_zyx.is_none() {
        bail!("A nonzero origin requires --volume-shape-zyx");
    }
    let domain: Vec<i64> = args.volume_shape_zyx.clone().unwrap_or_else(|| native.shape.map(|n| n as i64).to_vec());
    let origin_zyx = [args.origin_xyz[2], args.origin_xyz[1], args.origin_xyz[0]];
    if (0..3).any(|a| native.shape[a] as i64 + origin_zyx[a] > domain[a]) {
        bail!("Prediction crop lies outside the declared native volume");
    }
    let tiling = plan_tiles(&args, &native, supplied.as_ref())?;
    let mut paths = vec![native.path.clone()];
    if let Some(s) = &supplied {
        paths.push(s.path.clone());
    }
    let work = resolve(&args.work_dir)?;
    let output = resolve(&args.output)?;
    for path in &paths {
        let inside = |a: &Path, b: &Path| a != b && a.starts_with(b);
        if *path == output || *path == work || inside(&work, path) || inside(path, &work) {
            bail!("Inputs must be separate from the work directory and output");
        }
        if path.is_dir() && inside(&output, path) {
            bail!("Output cannot be inside an input Zarr store");
        }
    }
    let families: Vec<usize> = match args.family {
        Family::Both => vec![1, 2],
        Family::V => vec![1],
        Family::H => vec![2],
    };
    fs::create_dir_all(&work)?;
    let _lock = WorkLock::acquire(&work.join(".lock"))?;
    // the work directory and a resumed run's settings are checked under the lock, so the saved rounding and the
    // finished families are the ones this run resumes, and before the GPU opens, so a run that cannot start says
    // why rather than why the GPU is missing
    let manifest = work.join("run.json");
    let saved = match (manifest.exists(), args.resume) {
        (true, false) => bail!("Work directory already contains a run; use --resume"),
        (false, true) => bail!("No run to resume in this work directory"),
        (true, true) => Some(saved_run(&manifest)?),
        (false, false) => {
            for entry in fs::read_dir(&work)? {
                if entry?.file_name() != ".lock" {
                    bail!("A new run requires an empty work directory");
                }
            }
            None
        }
    };
    let family_name = match args.family {
        Family::Both => "both",
        Family::V => "V",
        Family::H => "H",
    };
    let signatures = paths.iter().map(|p| file_signature(p)).collect::<Result<Vec<_>>>()?;
    let mut config = json!({
        "version": REFERENCE_VERSION,
        "predictions": native.path.to_string_lossy(),
        "array": args.array,
        "tracing_predictions": supplied.as_ref().map(|s| s.path.to_string_lossy().into_owned()),
        "tracing_array": args.tracing_array,
        "channel_order": order,
        "shape": native.shape,
        // filled in below: a resumed run's saved rounding, or the rounding a new run's device gives
        "device": null,
        "storage": match args.storage { Storage::Dense => "dense", Storage::Compressed => "compressed" },
        "family": family_name,
        "voxel_size_um": args.voxel_size_um,
        "origin_xyz": args.origin_xyz,
        "volume_shape_zyx": domain,
        "coordinate_space": args.coordinate_space,
        "input_signatures": signatures,
        "native": {"tool": "fibrs", "version": env!("CARGO_PKG_VERSION"), "checkpoint": 1},
    });
    // only a tiled run records its tiling, so an untiled run's manifest is the one earlier releases wrote
    if let Some(t) = &tiling {
        config["tiling"] = t.record();
    }
    let changed = || eyre!("Inputs or tracing settings changed; use a new work directory");
    let started_on = match &saved {
        Some(saved) => {
            let rounding = match saved["device"].as_str() {
                Some("cpu") => Rounding::Cpu,
                Some("cuda") => Rounding::Cuda,
                _ => return Err(changed()),
            };
            config["device"] = json!(rounding.name());
            if *saved != config {
                return Err(changed());
            }
            Some(rounding)
        }
        None => None,
    };

    // a tiled run traces in each tile's own work directory
    let tracing_needed = match &tiling {
        None => families.iter().any(|&f| !family_complete(&work, f)),
        Some(t) => t.tiles().iter().any(|tile| families.iter().any(|&f| !family_complete(&work.join(tile.name()), f))),
    };
    let cent = CentralityDevice::choose(args.device, args.cuda_device.map(|d| d as usize), started_on, tracing_needed)?;
    if saved.is_none() {
        config["device"] = json!(cent.rounding.name());
        save_json(&manifest, &config)?;
    }
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads as usize).build()?;
    let compute = Compute { threads: threads as usize, cent: &cent };
    let cache_bytes = (args.native_cache_mib + args.field_cache_mib) as usize * 1024 * 1024;
    let origin_xyz = [args.origin_xyz[0], args.origin_xyz[1], args.origin_xyz[2]];
    let curves: Box<dyn Iterator<Item = Result<Curve>> + Send + '_> = match &tiling {
        None => {
            let pending: Vec<usize> = families.iter().copied().filter(|&f| !family_complete(&work, f)).collect();
            if !pending.is_empty() {
                let frame = RegionFrame::whole(native.shape.map(|n| n / 2));
                let trace = || trace_families(&native, supplied.as_ref(), &work, &pending, cache_bytes, frame, compute);
                pool.install(trace)?;
            }
            let origin = origin_xyz.map(|v| v as f64);
            let curves = families.iter().flat_map(|&f| CurveReader::open(&work, f).unwrap_or_else(CurveReader::failed));
            Box::new(curves.map(move |c| c.map(|(family, points)| (family, tracing_to_native(&points, origin)))))
        }
        Some(tiling) => {
            for tile in &tiling.tiles() {
                let dir = work.join(tile.name());
                let pending: Vec<usize> = families.iter().copied().filter(|&f| !family_complete(&dir, f)).collect();
                if pending.is_empty() {
                    log!("Tile {}: already traced", describe(tile, tiling.grid()));
                    continue;
                }
                log!("Tile {}", describe(tile, tiling.grid()));
                fs::create_dir_all(&dir)?;
                let lo = [0, tile.traced[0].lo, tile.traced[1].lo];
                let hi = [native.shape[0], tile.traced[0].hi, tile.traced[1].hi];
                let crop = native.crop(lo, hi)?;
                // the tracing array is half the native one, and the alignment keeps the tile's half offset on its
                // chunks
                let tracing_crop =
                    supplied.as_ref().map(|s| s.crop(lo.map(|v| v / 2), [s.shape[0], hi[1] / 2, hi[2] / 2]));
                let tracing_crop = tracing_crop.transpose()?;
                let frame = RegionFrame { offset: lo.map(|v| v / 2), window: native.shape.map(|n| n / 2) };
                let trace =
                    || trace_families(&crop, tracing_crop.as_ref(), &dir, &pending, cache_bytes, frame, compute);
                pool.install(trace)?;
                drop((crop, tracing_crop));
                release_freed();
            }
            Box::new(seam_curves(&work, tiling, &families, origin_xyz))
        }
    };
    let started = Instant::now();
    let opt = ExportOptions {
        voxel_um: args.voxel_size_um,
        origin_xyz,
        volume_shape: [domain[0], domain[1], domain[2]],
        minimum_mm: min_mm,
        coordinate_space: args.coordinate_space.as_deref(),
    };
    let summary = pool.install(|| export_afv(curves, &args.output, &opt))?;
    log!("Export: {:.1} s", started.elapsed().as_secs_f64());
    log!("Wrote {} fibres ({} points) to {}", summary.fiber_count, summary.point_count, output.display());
    Ok(summary)
}

/// The tiling `--tiles-yx` and `--tile-halo` ask for, or `None` for an untiled run
fn plan_tiles(args: &Args, native: &Predictions, supplied: Option<&Predictions>) -> Result<Option<Tiling>> {
    let Some(grid) = &args.tiles_yx else {
        if args.tile_halo.is_some() {
            bail!("--tile-halo requires --tiles-yx");
        }
        return Ok(None);
    };
    if grid.iter().any(|&n| n < 1) {
        bail!("--tiles-yx needs at least one tile along each axis");
    }
    let halo = args.tile_halo.unwrap_or(TILE_HALO);
    if halo < 0 {
        bail!("--tile-halo must be nonnegative");
    }
    let align = alignment(native, supplied);
    let extent = [native.shape[1], native.shape[2]];
    Ok(Some(Tiling::new([grid[0] as usize, grid[1] as usize], halo as usize, align, extent)?))
}

/// A tile's place and boxes for the log: `tile-00-01 of 2 x 2: core y 0..256, x 256..512; traced y ...`
fn describe(tile: &Tile, grid: [usize; 2]) -> String {
    format!(
        "{} of {} x {}: core y {}..{}, x {}..{}; traced y {}..{}, x {}..{}",
        tile.name(),
        grid[0],
        grid[1],
        tile.core[0].lo,
        tile.core[0].hi,
        tile.core[1].lo,
        tile.core[1].hi,
        tile.traced[0].lo,
        tile.traced[0].hi,
        tile.traced[1].lo,
        tile.traced[1].hi
    )
}

pub(crate) fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn family_dir(work: &Path, family: usize) -> PathBuf {
    work.join(format!("family-{family}"))
}

fn family_complete(work: &Path, family: usize) -> bool {
    let progress = family_dir(work, family).join("progress.json");
    let Ok(text) = fs::read_to_string(progress) else { return false };
    serde_json::from_str::<Value>(&text).is_ok_and(|v| v["complete"] == json!(true))
}

/// The pooled tracing predictions, both channels (`prepare.prepare_tracing`)
fn load_tracing(native: &Predictions, supplied: Option<&Predictions>) -> Result<[Vol<u8>; 2]> {
    let shape = native.shape.map(|n| n / 2);
    // tiles are the supplied array's chunks so each is decoded once; the values do not depend on tiling
    let tile = supplied.and_then(Predictions::chunk_shape).unwrap_or([64, 64, 64]);
    let tgrid = [0, 1, 2].map(|a| shape[a].div_ceil(tile[a]));
    let mut out = [Vol::new(shape, 0u8), Vol::new(shape, 0u8)];
    let brick_rows = out[0].grid()[0];
    // a slab spans whole tiles where that is at most two tiles deep (lcm(tile, 16) for tiles such as 24 or 40
    // planes), so no tile is decoded for two slabs. Deeper alignments fall back to the brick rows within one tile,
    // which can decode a tile twice but never hold more than the two tiles a straddling slab reads
    let aligned = tile[0] / gcd(tile[0], EDGE) * EDGE;
    let slab = if aligned <= 2 * tile[0].max(EDGE) { aligned / EDGE } else { tile[0] / EDGE }.max(1);
    let mut bz = 0;
    while bz < brick_rows {
        let rows = bz..(bz + slab).min(brick_rows);
        let (z0, z1) = (rows.start * EDGE, (rows.end * EDGE).min(shape[0]));
        let tz = (z0 / tile[0])..z1.div_ceil(tile[0]);
        let jobs: Vec<Shape> =
            tz.flat_map(|z| (0..tgrid[1]).flat_map(move |y| (0..tgrid[2]).map(move |x| [z, y, x]))).collect();
        let tiles: Vec<Option<(Shape, Shape, Vec<u8>)>> = jobs
            .par_iter()
            .map(|&t| {
                let lo = [0, 1, 2].map(|a| t[a] * tile[a]);
                let hi = [0, 1, 2].map(|a| (lo[a] + tile[a]).min(shape[a]));
                let data = match supplied {
                    Some(s) => s.read_u8(lo, hi)?,
                    None => pool_block(native, lo, hi)?,
                };
                let dims = [0, 1, 2].map(|a| hi[a] - lo[a]);
                Ok(data.iter().any(|&v| v != 0).then_some((lo, dims, data)))
            })
            .collect::<Result<_>>()?;
        let index = |t: Shape| ((t[0] - z0 / tile[0]) * tgrid[1] + t[1]) * tgrid[2] + t[2];
        for (ch, vol) in out.iter_mut().enumerate() {
            vol.fill_brick_rows(rows.clone(), |lo, dims, dst| {
                let mut any = false;
                let t0 = [0, 1, 2].map(|a| lo[a] / tile[a]);
                let t1 = [0, 1, 2].map(|a| (lo[a] + dims[a] - 1) / tile[a]);
                for tz in t0[0]..=t1[0] {
                    for ty in t0[1]..=t1[1] {
                        for tx in t0[2]..=t1[2] {
                            let Some((tlo, tdims, data)) = &tiles[index([tz, ty, tx])] else { continue };
                            let plane = tdims.iter().product::<usize>();
                            let a0 = [0, 1, 2].map(|a| lo[a].max(tlo[a]));
                            let a1 = [0, 1, 2].map(|a| (lo[a] + dims[a]).min(tlo[a] + tdims[a]));
                            for z in a0[0]..a1[0] {
                                for y in a0[1]..a1[1] {
                                    let src = ch * plane + ((z - tlo[0]) * tdims[1] + y - tlo[1]) * tdims[2];
                                    let dst_at = ((z - lo[0]) * dims[1] + y - lo[1]) * dims[2];
                                    for x in a0[2]..a1[2] {
                                        let v = data[src + x - tlo[2]];
                                        dst[dst_at + x - lo[2]] = v;
                                        any |= v != 0;
                                    }
                                }
                            }
                        }
                    }
                }
                any
            });
        }
        bz = rows.end;
    }
    Ok(out)
}

/// 2×2×2 average pooling of the native predictions in float32, eight additions in corner order, then
/// `rint` and clip to uint8
fn pool_block(native: &Predictions, lo: Shape, hi: Shape) -> Result<Vec<u8>> {
    let dims = [0, 1, 2].map(|a| hi[a] - lo[a]);
    let nd = dims.map(|n| 2 * n);
    let block = native.read_f32(lo.map(|v| 2 * v), hi.map(|v| 2 * v))?;
    let nplane = nd.iter().product::<usize>();
    let plane = dims.iter().product::<usize>();
    let scale = native.dtype.is_float();
    let mut out = vec![0u8; 2 * plane];
    for ch in 0..2 {
        for z in 0..dims[0] {
            for y in 0..dims[1] {
                for x in 0..dims[2] {
                    let mut acc = 0.0f32;
                    for c in 0..8 {
                        let (dz, dy, dx) = (c >> 2 & 1, c >> 1 & 1, c & 1);
                        acc += block[ch * nplane + ((2 * z + dz) * nd[1] + 2 * y + dy) * nd[2] + 2 * x + dx];
                    }
                    let mut v = acc / 8.0;
                    if scale {
                        v *= 255.0;
                    }
                    out[ch * plane + (z * dims[1] + y) * dims[2] + x] = v.round_ties_even().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    Ok(out)
}

/// Raw curves handed from the committing thread to the finishing threads at a time
///
/// Each batch is finished in native-chunk order, so a larger batch reads and decodes each chunk for more curves:
/// on a dense PHerc0358 crop 8192 read 55% fewer chunks and decoded 40% fewer slabs than 2048, for 10% less CPU
/// time
const FINISH_BATCH: usize = 8192;
/// Raw curve points at which a batch is handed over even with fewer than [`FINISH_BATCH`] curves, so the batches in
/// flight stay in the tens of megabytes (12 bytes a point) however long the curves are; the GPU leaves room for a
/// batch of this many beside its slab cache
const FINISH_POINTS: usize = 2 << 20;

fn trace_families(
    native: &Predictions,
    supplied: Option<&Predictions>,
    work: &Path,
    families: &[usize],
    cache_bytes: usize,
    frame: RegionFrame,
    compute: Compute,
) -> Result<()> {
    let t = Instant::now();
    log!("Preparing tracing predictions");
    let tracing = load_tracing(native, supplied)?;
    let resident = tracing.iter().map(Vol::resident_bytes).sum::<usize>();
    log!("Tracing predictions: {:.1} s, {:.2} GiB resident", t.elapsed().as_secs_f64(), gib(resident));
    let lapack = Lapack::init(&find_lapack()?)?;
    let dirs = direction_field_on(compute.cent, &tracing, lapack)?;
    release_freed();
    let pools = CachePools::new();
    let cache = NativeCache::new(native, cache_bytes, &pools);
    let families = Families { tracing: &tracing, resident, cache: &cache, work, families, frame, compute };
    match &dirs {
        Directions::Packed(packed) => families.trace(packed),
        Directions::Split { nx, ny } => families.trace(&SplitDirs { nx, ny }),
    }
}

/// What tracing each pending family needs besides the direction field
struct Families<'a, 'n, 'p> {
    tracing: &'a [Vol<u8>; 2],
    resident: usize,
    cache: &'a NativeCache<'n, 'p>,
    work: &'a Path,
    families: &'a [usize],
    /// Where the traced volume lies in the window whose seed order it keeps
    frame: RegionFrame,
    compute: Compute<'a>,
}

impl Families<'_, '_, '_> {
    fn trace<D: DirField>(&self, dirs: &D) -> Result<()> {
        debug_dump(self.work, "nx", || dir_bytes(dirs, 8))?;
        debug_dump(self.work, "ny", || dir_bytes(dirs, 0))?;
        debug_mem("direction field", &[("tracing", self.resident), ("directions", dirs.resident_bytes())]);
        for &family in self.families {
            trace_family(self.tracing, dirs, self.cache, self.work, family, self.frame, self.compute)?;
            release_freed();
        }
        Ok(())
    }
}

/// One direction byte of every voxel as a dense C-order array, `nx` for `shift` 8 and `ny` for 0, read a voxel at a
/// time so a dump holds one byte a voxel in either form
fn dir_bytes<D: DirField>(dirs: &D, shift: u32) -> Vec<u8> {
    let [d, h, w] = dirs.shape().map(|n| n as i64);
    let mut out = Vec::with_capacity(dirs.shape().iter().product());
    let voxels = (0..d).flat_map(|z| (0..h).flat_map(move |y| (0..w).map(move |x| [z, y, x])));
    out.extend(voxels.map(|[z, y, x]| (dirs.at(z, y, x) >> shift) as u8));
    out
}

/// The direction field, on the GPU when there is one, in the form the run's centrality reads
///
/// Both devices give the same bytes, so when the GPU fails, such as out of memory, the CPU computes it instead,
/// unless `--device cuda` asked for the GPU only
fn direction_field_on(cent: &CentralityDevice, tracing: &[Vol<u8>; 2], lapack: &Lapack) -> Result<Directions> {
    let t = Instant::now();
    let on_cpu = || -> Result<Directions> {
        let field = direction_field(tracing, lapack, cent.dir_form())?;
        log!("Direction field: {:.1} s", t.elapsed().as_secs_f64());
        Ok(field)
    };
    let Some(gpu) = &cent.gpu else { return on_cpu() };
    match gpu.direction_field(tracing, lapack) {
        Ok((nx, ny, AxesCounts { present, fallback })) => {
            let secs = t.elapsed().as_secs_f64();
            log!("Direction field: {secs:.1} s, {fallback} of {present} voxels solved on the CPU");
            Ok(Directions::Split { nx, ny })
        }
        Err(e) if !cent.strict => {
            log!("Direction field on the CPU: the GPU failed ({e:#})");
            on_cpu()
        }
        Err(e) => Err(e),
    }
}

/// With `FIBRS_DEBUG_FIELDS` set, write a field densely as raw little-endian `NAME.raw` for comparison
/// with the reference's work arrays; only sensible for small crops
fn debug_dump<T: Copy + ToLe>(dir: &Path, name: &str, dense: impl FnOnce() -> Vec<T>) -> Result<()> {
    if std::env::var_os("FIBRS_DEBUG_FIELDS").is_none() {
        return Ok(());
    }
    let dense = dense();
    let bytes: Vec<u8> = dense.iter().flat_map(|v| v.to_le()).collect();
    fs::write(dir.join(format!("{name}.raw")), bytes)?;
    Ok(())
}

/// Little-endian bytes of a field element, for debug dumps
trait ToLe {
    fn to_le(self) -> Vec<u8>;
}

impl ToLe for u8 {
    fn to_le(self) -> Vec<u8> {
        vec![self]
    }
}

impl ToLe for u16 {
    fn to_le(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}

/// Hand the heap pages freed since the last call back to the system
///
/// Each stage frees what the next does not read (the direction field's and centrality's tile results, the seed
/// field, a finished family's centrality and coverage), but glibc keeps freed memory in the arena of
/// the thread that freed it and only trims the top of each heap. The next stage allocates from other threads'
/// arenas (coverage grows on the committing thread, the next family's fields on the pool's), so without this the
/// freed bytes stay resident beside the new ones: on a 0826 band the H family peaked 1.7 GiB above V with the
/// same fields
fn release_freed() {
    // mimalloc keeps a freed large block in its arena and purges it only after a delay, on a later allocation or
    // free, and a smaller block in a page of the thread that allocated it, which only that thread collects. So each
    // thread of the pool collects its own pages, which the stages do between stages, when the pool is idle, and one
    // more forced collect after them all purges what they handed back. This is best effort: a page freed while
    // another thread holds mimalloc's purge lock (a pool thread mid-collect, or a finishing pool's thread still
    // exiting) can stay until mimalloc's own delayed purge, which runs about 1 s later as the next stage allocates
    // SAFETY: mi_collect only returns free pages to the kernel; it touches no live allocation
    #[cfg(feature = "mimalloc")]
    unsafe {
        rayon::broadcast(|_| libmimalloc_sys::mi_collect(true));
        libmimalloc_sys::mi_collect(true);
    }

    // SAFETY: malloc_trim only returns free pages to the kernel; it touches no live allocation. Without mimalloc
    // it covers every allocation; with it, the ones C libraries such as OpenBLAS make through glibc
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    unsafe {
        libc::malloc_trim(0);
    }
}

/// Serve every allocation of 1 MiB or more with its own mapping, which goes back to the system when freed
///
/// glibc starts at 128 KiB but raises the threshold to the size of each mapped block freed, up to 32 MiB, so
/// after the first few the chunk cache's 4 MiB chunks and the stages' tile buffers come from the per-thread
/// arenas instead, where they fragment and stay resident as they are evicted and freed on other threads: on a
/// 512-plane 0826 band the peak varied from 8.9 to 10.8 GiB between runs without this and was 8.8 GiB in each
/// of three runs with it, at the same speed. Setting the threshold also stops glibc's adjustment, so the trim
/// threshold is set too (see below).
/// Either threshold set in the environment, by `MALLOC_MMAP_THRESHOLD_`, `MALLOC_TRIM_THRESHOLD_` or
/// `GLIBC_TUNABLES`, still takes precedence
fn fix_malloc_thresholds() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    if std::env::var_os("MALLOC_MMAP_THRESHOLD_").is_none()
        && !std::env::var("GLIBC_TUNABLES").is_ok_and(|t| t.contains("glibc.malloc.mmap_threshold"))
    {
        // SAFETY: mallopt only changes how later allocations are served; it touches no live allocation
        unsafe {
            libc::mallopt(libc::M_MMAP_THRESHOLD, crate::arena::OWN_MAPPING_BYTES as libc::c_int);
        }
    }
    // fixing the mmap threshold also fixes the trim threshold at 128 KiB, so each finishing thread's heap gave
    // back c-blosc's 384 KiB decode scratch on every slab and faulted it in again: on a dense crop that was 460,000
    // page faults a second and 15% of the CPU time in the kernel. 8 MiB stops it; `release_freed` still trims the
    // heaps between stages
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    if std::env::var_os("MALLOC_TRIM_THRESHOLD_").is_none()
        && !std::env::var("GLIBC_TUNABLES").is_ok_and(|t| t.contains("glibc.malloc.trim_threshold"))
    {
        // SAFETY: as above
        unsafe {
            libc::mallopt(libc::M_TRIM_THRESHOLD, 8 << 20);
        }
    }
}

/// With `FIBRS_DEBUG_MEM` set, log named resident sizes and the process RSS
fn debug_mem(stage: &str, parts: &[(&str, usize)]) {
    if std::env::var_os("FIBRS_DEBUG_MEM").is_none() {
        return;
    }
    let rss =
        fs::read_to_string("/proc/self/statm").ok().and_then(|s| s.split_whitespace().nth(1)?.parse::<usize>().ok());
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let parts: Vec<String> = parts.iter().map(|(k, v)| format!("{k} {:.2}", gib(*v))).collect();
    log!("mem {stage}: {} GiB; rss {:.2} GiB", parts.join(", "), gib(rss.unwrap_or(0) * page));
}

fn gib(bytes: usize) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

/// Where a family's stages run: tracing and finishing on `threads` CPU threads, centrality as `cent` says
#[derive(Clone, Copy)]
struct Compute<'a> {
    threads: usize,
    cent: &'a CentralityDevice,
}

/// Which of the original's devices a run's centrality reproduces: its CPU kernel rounds some multiply-adds twice
/// where its CUDA kernels round them once (`stage1.rs`). `run.json` records it as `device`, so one run never mixes
/// the two, while the hardware a resumed run computes it on may change
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rounding {
    Cpu,
    Cuda,
}

impl Rounding {
    fn name(self) -> &'static str {
        match self {
            Rounding::Cpu => "cpu",
            Rounding::Cuda => "cuda",
        }
    }
}

/// A resumable run's `run.json`: one this release of fibrs started
fn saved_run(manifest: &Path) -> Result<Value> {
    let saved: Value = serde_json::from_str(&fs::read_to_string(manifest)?)?;
    if saved.get("native").is_none() {
        bail!("Work directory holds a reference avf-spline-generator run; use a new work directory");
    }
    // the checkpoints carry no format of their own beyond the release that wrote them
    let started_by = &saved["native"]["version"];
    if started_by.as_str() != Some(env!("CARGO_PKG_VERSION")) {
        bail!(
            "Work directory was started by fibrs {}; resume it with that version, or use a new work directory",
            started_by.as_str().unwrap_or("of another version")
        );
    }
    Ok(saved)
}

/// How a run computes the direction field and centrality: the centrality rounding it reproduces, and the GPU when
/// one runs the direction field and, in the CUDA rounding, centrality (without one, the CPU computes either rounding,
/// the CUDA one more slowly)
struct CentralityDevice {
    rounding: Rounding,
    gpu: Option<Gpu>,
    /// Whether a GPU failure mid-run ends the run (`--device cuda`) instead of moving to the CPU
    strict: bool,
}

impl CentralityDevice {
    /// The rounding a new run takes from `device`, or a resumed run from its `run.json`, and the GPU to compute
    /// it on, opened only when a family still needs centrality
    ///
    /// `auto` takes the GPU when one opens, and otherwise the CPU, saying why. A run started on the CPU keeps
    /// centrality there, but its direction field and finishing, the same bytes on either device, still take the GPU
    fn choose(
        device: Device,
        index: Option<usize>,
        started_on: Option<Rounding>,
        tracing_needed: bool,
    ) -> Result<Self> {
        let strict = device == Device::Cuda;
        let cpu = |rounding| Self { rounding, gpu: None, strict };
        let on_gpu = |gpu: Gpu| {
            log!("Direction field and centrality on {} (--device cpu to use the CPU)", gpu.name());
            Self { rounding: Rounding::Cuda, gpu: Some(gpu), strict }
        };
        let open = || Gpu::open(index, Stages::DirectionFieldAndCentrality);
        let chosen = match (started_on, device) {
            (Some(Rounding::Cpu), Device::Cuda) => {
                bail!("This run started with centrality on the CPU and resumes only there; use --device cpu or auto")
            }
            (Some(Rounding::Cpu), Device::Auto) if tracing_needed => match Gpu::open(index, Stages::DirectionField) {
                Ok(gpu) => {
                    log!(
                        "Direction field on {}, centrality on the CPU this run started on (--device cpu to use the CPU)",
                        gpu.name()
                    );
                    Self { rounding: Rounding::Cpu, gpu: Some(gpu), strict }
                }
                Err(e) => {
                    log!("Direction field and centrality on the CPU: no usable GPU ({e:#})");
                    cpu(Rounding::Cpu)
                }
            },
            (Some(Rounding::Cpu), _) | (None, Device::Cpu) => cpu(Rounding::Cpu),
            (Some(Rounding::Cuda), _) if !tracing_needed => cpu(Rounding::Cuda),
            (Some(Rounding::Cuda), Device::Cpu) => {
                log!("Direction field and centrality on the CPU, centrality rounded as on the GPU this run started on");
                cpu(Rounding::Cuda)
            }
            (_, Device::Cuda) => on_gpu(open().map_err(|e| eyre!("--device cuda needs a usable NVIDIA GPU: {e:#}"))?),
            (Some(Rounding::Cuda), Device::Auto) => match open() {
                Ok(gpu) => on_gpu(gpu),
                Err(e) => {
                    log!(
                        "Direction field and centrality on the CPU, centrality rounded as on the GPU this run started \
                         on: no usable GPU ({e:#})"
                    );
                    cpu(Rounding::Cuda)
                }
            },
            (None, Device::Auto) => match open() {
                Ok(gpu) => on_gpu(gpu),
                Err(e) => {
                    log!("Direction field and centrality on the CPU: no usable GPU ({e:#})");
                    cpu(Rounding::Cpu)
                }
            },
        };
        Ok(chosen)
    }

    /// The form the direction field is written in: split when the GPU computes centrality, which reads `nx` and
    /// `ny` as two volumes, so a field the CPU computes after a GPU failure needs no copy
    fn dir_form(&self) -> DirForm {
        match (self.rounding, &self.gpu) {
            (Rounding::Cuda, Some(_)) => DirForm::Split,
            _ => DirForm::Packed,
        }
    }

    /// One family's centrality and seeds in the run's rounding
    ///
    /// When the GPU fails, such as out of memory, the CPU computes the same CUDA rounding unless `--device cuda`
    /// asked for the GPU only
    fn compute<D: DirField>(
        &self,
        tracing: &[Vol<u8>; 2],
        family: usize,
        pres: &Presence,
        dirs: &D,
    ) -> Result<FamilyFields> {
        let (gpu, (nx, ny)) = match (self.rounding, &self.gpu, dirs.split()) {
            (Rounding::Cpu, _, _) => return Ok(centrality::<Separate, _>(pres, dirs)),
            (Rounding::Cuda, Some(gpu), Some(split)) => (gpu, split),
            // no GPU, or a packed field, which [`Self::dir_form`] asks for only when there is no GPU
            (Rounding::Cuda, _, _) => return Ok(centrality::<Fused, _>(pres, dirs)),
        };
        match gpu.centrality(tracing, family, nx, ny) {
            Ok(fields) => Ok(fields),
            Err(e) if !self.strict => {
                log!("Centrality on the CPU, rounded as on the GPU: the GPU failed ({e:#})");
                Ok(centrality::<Fused, _>(pres, dirs))
            }
            Err(e) => Err(e),
        }
    }
}

fn trace_family<D: DirField>(
    tracing: &[Vol<u8>; 2],
    dirs: &D,
    cache: &NativeCache,
    work: &Path,
    family: usize,
    frame: RegionFrame,
    Compute { threads, cent: device }: Compute,
) -> Result<()> {
    let label = if family == 1 { "V" } else { "H" };
    let folder = family_dir(work, family);
    fs::create_dir_all(&folder)?;
    let t = Instant::now();
    let pres = Presence::new(tracing, family);
    let FamilyFields { cent, seeds } = device.compute(tracing, family, &pres, dirs)?;
    log!("{label} centrality: {:.1} s", t.elapsed().as_secs_f64());
    debug_mem(label, &[("cent", cent.resident_bytes()), ("seeds", seeds.len() * std::mem::size_of::<Seed>())]);
    let t = Instant::now();
    let cat = catalogue(&seeds, &cent, frame);
    debug_dump(&folder, "presence", || pres.read_block([0, 0, 0], pres.shape()))?;
    debug_dump(&folder, "centrality", || cent.read_block([0, 0, 0], cent.shape()))?;
    debug_dump(&folder, "seeds", || {
        let mut dense = Vol::new(cent.shape(), 0u8);
        for &seed in &seeds {
            let [z, y, x] = seed.coords(cent.shape());
            dense.set(z as usize, y as usize, x as usize, seed.priority());
        }
        dense.read_block([0, 0, 0], dense.shape())
    })?;
    if std::env::var_os("FIBRS_DEBUG_FIELDS").is_some() {
        let bytes: Vec<u8> = cat.coords.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
        fs::write(folder.join("coordinates.i4"), bytes)?;
    }
    drop(seeds);
    release_freed();
    log!("{label} seeds: {} in {:.1} s", cat.coords.len(), t.elapsed().as_secs_f64());
    let shape = pres.shape();
    let fields = Fields { pres: &pres, cent: &cent, dirs };
    let ctx = FinishCtx { tracing, pres: &pres, family };
    let mut finisher = BatchFinisher::new(device, tracing, family, label);
    // the pool's idle threads trace seeds ahead of the commit point, and it gets every thread: with one fewer, as
    // when one thread traced every seed, a dense crop at --threads 4 traced no faster than that thread did, since
    // finishing kept the pool busy. The committing thread traces the seeds no pool thread has started
    let finish_pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build()?;
    let path = folder.join("curves.bin");
    let mut writer = CurveWriter::create(&path)?;
    // FIBRS_DEBUG_RAW=1 also keeps the raw curves (family-N/raw.bin, same format) to compare with the
    // reference's part files stage by stage
    let mut raw_writer = match std::env::var_os("FIBRS_DEBUG_RAW") {
        Some(_) => Some(CurveWriter::create(&folder.join("raw.bin"))?),
        None => None,
    };
    let t = Instant::now();
    let total = cat.coords.len();
    let mut pieces = 0;
    let spec = Speculation::from_env()?;
    let cov = Coverage::new(shape);
    let producer = Producer::new(&fields, &cov, &cat, spec)?;
    let produce = |tx: &SyncSender<Vec<RawCurve>>| -> Result<usize> {
        let mut batch: Vec<RawCurve> = Vec::new();
        let mut points = 0;
        let mut count = 0;
        let mut last_log = Instant::now();
        let deliver = |i: usize, curves: &mut Vec<RawCurve>| -> Result<Flow> {
            points += curves.iter().map(|c| c.pts.len()).sum::<usize>();
            batch.append(curves);
            if last_log.elapsed().as_secs_f64() >= 10.0 {
                log!("{label} seeds: {}/{total}; {} raw curves", i + 1, count + batch.len());
                last_log = Instant::now();
            }
            if batch.len() < FINISH_BATCH && points < FINISH_POINTS {
                return Ok(Flow::Continue);
            }
            count += batch.len();
            points = 0;
            // a closed channel means finishing failed; its error is the one reported
            match tx.send(std::mem::take(&mut batch)) {
                Ok(()) => Ok(Flow::Continue),
                Err(_) => Ok(Flow::Stop),
            }
        };
        let stats = producer.run(&finish_pool, deliver)?;
        count += batch.len();
        if !batch.is_empty() {
            let _ = tx.send(batch);
        }
        let Stats { seeds, window, retraced, committer } = stats;
        let share = 100.0 * retraced as f64 / seeds.max(1) as f64;
        log!("{label} window {window}: {retraced} seeds traced again ({share:.3}%), {committer} on the committer");
        debug_mem(label, &[("coverage", cov.resident_bytes()), ("cache", cache.bytes()), ("catalogue", 12 * total)]);
        Ok(count)
    };
    let consume = |batch: Vec<RawCurve>| -> Result<()> {
        if let Some(w) = raw_writer.as_mut() {
            for curve in &batch {
                w.push(&curve.pts)?;
            }
        }
        for curve in finisher.finish(&finish_pool, &ctx, cache, &batch)?.iter().flatten() {
            writer.push(curve)?;
            pieces += 1;
        }
        Ok(())
    };
    let raw = pipeline(2, produce, consume)?;
    finisher.summary();
    writer.finish()?;
    if let Some(w) = raw_writer {
        w.finish()?;
    }
    log!(
        "{label} traced: {raw} raw curves, {pieces} finished pieces in {:.1} s ({} native slabs decoded from {} chunk reads)",
        t.elapsed().as_secs_f64(),
        cache.decoded(),
        cache.read()
    );
    save_json(&folder.join("progress.json"), &json!({"complete": true, "raw_curves": raw, "curves": pieces}))?;
    Ok(())
}

/// Where a family's batches of raw curves are finished: on the GPU when the run has one, and on the CPU for each
/// batch the GPU hands back, or for every batch once the GPU has failed. Both give the same pieces
struct BatchFinisher<'a> {
    gpu: OnGpu<'a>,
    /// Whether a GPU failure ends the run (`--device cuda`) instead of moving finishing to the CPU
    strict: bool,
    label: &'static str,
    counts: FinishCounts,
}

/// Where a family's finishing stands on the GPU
enum OnGpu<'a> {
    /// The run's GPU, set up for the family when its first batch of curves comes, so a family with none never
    /// needs it
    Unopened { gpu: &'a Gpu, tracing: &'a [Vol<u8>; 2], family: usize },
    /// Boxed, so the other states do not take the finisher's size
    Open(Box<GpuFinisher>),
    /// The run has no GPU, or setting it up or a batch on it failed
    Off,
}

/// What the GPU finished of a family, for the log
#[derive(Default)]
struct FinishCounts {
    /// Whether the GPU took part: the family's batches after it fails count as finished on the CPU
    used: bool,
    batches: usize,
    loaded: SlabLoads,
    /// Batches handed back, by [`CpuReason`], and the batch the GPU failed on and every one after it
    slabs: usize,
    unlisted: usize,
    capacity: usize,
    decode: usize,
    failed: usize,
}

impl<'a> BatchFinisher<'a> {
    /// Finishing on the run's GPU, if it has one, else on the CPU
    fn new(device: &'a CentralityDevice, tracing: &'a [Vol<u8>; 2], family: usize, label: &'static str) -> Self {
        let gpu = match &device.gpu {
            Some(gpu) => OnGpu::Unopened { gpu, tracing, family },
            None => OnGpu::Off,
        };
        Self { gpu, strict: device.strict, label, counts: FinishCounts::default() }
    }

    /// Set the GPU up for the family, if it is not yet, when the family's pooled predictions and a slab cache fit
    /// on it, else move finishing to the CPU; says which
    fn open(&mut self, cache: &NativeCache) -> Result<()> {
        let OnGpu::Unopened { gpu, tracing, family } = self.gpu else { return Ok(()) };
        self.gpu = OnGpu::Off;
        let (label, layout) = (self.label, cache.layout());
        match gpu.finisher(tracing, family, layout, FINISH_POINTS) {
            Ok(on_gpu) => {
                let gib = gib(on_gpu.slots() * layout.slab_bytes());
                log!(
                    "{label} finishing on {}, {} native slabs cached there ({gib:.2} GiB)",
                    gpu.name(),
                    on_gpu.slots()
                );
                self.gpu = OnGpu::Open(Box::new(on_gpu));
                self.counts.used = true;
            }
            Err(e) if !self.strict => log!("{label} finishing on the CPU: the GPU failed ({e:#})"),
            Err(e) => return Err(e),
        }
        Ok(())
    }

    /// One batch's pieces, as [`finish_batch`] gives them
    fn finish(
        &mut self,
        pool: &rayon::ThreadPool,
        ctx: &FinishCtx,
        cache: &NativeCache,
        batch: &[RawCurve],
    ) -> Result<Vec<Pieces>> {
        if !batch.is_empty() {
            self.open(cache)?;
        }
        let c = &mut self.counts;
        c.batches += c.used as usize;
        let OnGpu::Open(gpu) = &mut self.gpu else {
            c.failed += c.used as usize;
            return finish_batch(pool, ctx, cache, batch);
        };
        match gpu.finish(pool, cache, batch) {
            Ok(Finished::Pieces(pieces)) => return Ok(pieces),
            Ok(Finished::OnCpu(CpuReason::Slabs)) => c.slabs += 1,
            Ok(Finished::OnCpu(CpuReason::Unlisted)) => c.unlisted += 1,
            Ok(Finished::OnCpu(CpuReason::Capacity)) => c.capacity += 1,
            Ok(Finished::OnCpu(CpuReason::Decode)) => c.decode += 1,
            Err(e) if !self.strict => {
                log!("{} finishing on the CPU from here: the GPU failed ({e:#})", self.label);
                c.loaded = gpu.loaded();
                c.failed += 1;
                self.gpu = OnGpu::Off;
            }
            Err(e) => return Err(e),
        }
        finish_batch(pool, ctx, cache, batch)
    }

    /// Log the slabs the GPU cached, by where they were decoded, and the batches the CPU finished instead, if the
    /// GPU took part
    fn summary(&mut self) {
        let c = &mut self.counts;
        if !c.used {
            return;
        }
        if let OnGpu::Open(gpu) = &self.gpu {
            c.loaded = gpu.loaded();
        }
        let reasons = [
            (c.slabs, "read more slabs than the GPU holds"),
            (c.unlisted, "read a slab not uploaded"),
            (c.capacity, "fitted a piece beyond its room"),
            (c.decode, "had a slab that did not decode"),
            (c.failed, "came at or after a GPU failure"),
        ];
        let on_cpu = reasons.iter().map(|&(n, _)| n).sum::<usize>();
        let why: Vec<String> = reasons.iter().filter(|&&(n, _)| n > 0).map(|(n, why)| format!("{n} {why}")).collect();
        let why = if why.is_empty() { String::new() } else { format!(": {}", why.join(", ")) };
        log!(
            "{} finishing: {} native slabs cached on the GPU ({} decoded there, {} on the CPU), {on_cpu} of {} \
             batches finished on the CPU{why}",
            self.label,
            c.loaded.total(),
            c.loaded.on_gpu,
            c.loaded.on_cpu,
            c.batches
        );
    }
}

/// Run `produce` on its own thread, handing what it sends to `consume` through a channel of `capacity`
/// items; returns the producer's result, or the first error of either side
///
/// The receiver is dropped as soon as `consume` fails, so a producer blocked on a full channel gets a send
/// error and returns instead of holding the scope open. A producer error closes the channel from its side
fn pipeline<T: Send, R: Send>(
    capacity: usize,
    produce: impl FnOnce(&SyncSender<T>) -> Result<R> + Send,
    mut consume: impl FnMut(T) -> Result<()>,
) -> Result<R> {
    let (tx, rx) = sync_channel(capacity);
    std::thread::scope(|scope| {
        let producer = scope.spawn(move || produce(&tx));
        let consumed = rx.into_iter().try_for_each(&mut consume);
        let produced = producer.join().map_err(|_| eyre!("tracing thread panicked"))?;
        consumed?;
        produced
    })
}

/// Finish one batch on the finishing pool, results in input order; work is handed out in native-chunk
/// order of each curve's first point so threads share decoded chunks
fn finish_batch(
    pool: &rayon::ThreadPool,
    ctx: &FinishCtx,
    cache: &NativeCache,
    batch: &[RawCurve],
) -> Result<Vec<Pieces>> {
    let mut order: Vec<usize> = (0..batch.len()).collect();
    order.sort_by_key(|&i| {
        let p = batch[i].pts[0];
        let v = |c: f32| (c.max(0.0) * 2.0 + 0.5) as usize;
        cache.key(v(p[2]), v(p[1]), v(p[0]))
    });
    let done: Vec<(usize, Vec<Vec<[f32; 3]>>)> = pool.install(|| {
        order
            .par_iter()
            .map_init(|| NativeSampler::new(cache), |s, &i| ctx.finish(&batch[i].pts, s).map(|p| (i, p)))
            .collect::<Result<_>>()
    })?;
    let mut out = vec![Vec::new(); batch.len()];
    for (i, p) in done {
        out[i] = p;
    }
    Ok(out)
}

const MAGIC: &[u8; 8] = b"AFVTCRV1";

/// The per-family checkpoint: finished curves in export order, tracing-space float32 x, y, z
struct CurveWriter {
    path: PathBuf,
    temporary: PathBuf,
    out: BufWriter<File>,
}

impl CurveWriter {
    fn create(path: &Path) -> Result<Self> {
        let temporary = path.with_extension("bin.tmp");
        let mut out = BufWriter::with_capacity(1 << 20, File::create(&temporary)?);
        out.write_all(MAGIC)?;
        Ok(Self { path: path.to_path_buf(), temporary, out })
    }

    fn push(&mut self, curve: &[[f32; 3]]) -> Result<()> {
        self.out.write_all(&(curve.len() as u32).to_le_bytes())?;
        for p in curve {
            for v in p {
                self.out.write_all(&v.to_le_bytes())?;
            }
        }
        Ok(())
    }

    fn finish(self) -> Result<()> {
        let f = self.out.into_inner().map_err(|e| e.into_error())?;
        f.sync_all()?;
        fs::rename(&self.temporary, &self.path)?;
        Ok(())
    }
}

/// Reads one family's finished curves back for the export
pub(crate) enum CurveReader {
    Open { family: usize, input: BufReader<File> },
    Failed(Option<color_eyre::Report>),
}

impl CurveReader {
    /// The finished curves of `family` in the work directory `work` (a tile's, in a tiled run)
    pub(crate) fn open(work: &Path, family: usize) -> Result<Self> {
        if !family_complete(work, family) {
            bail!("Cannot export an unfinished tracing run");
        }
        let mut input = BufReader::with_capacity(1 << 20, File::open(family_dir(work, family).join("curves.bin"))?);
        let mut magic = [0u8; 8];
        input.read_exact(&mut magic)?;
        if &magic != MAGIC {
            bail!("Invalid curve file for family {family}");
        }
        Ok(Self::Open { family, input })
    }

    fn failed(e: color_eyre::Report) -> Self {
        Self::Failed(Some(e))
    }
}

impl Iterator for CurveReader {
    type Item = Result<(usize, Vec<[f32; 3]>)>;

    fn next(&mut self) -> Option<Self::Item> {
        let (family, input) = match self {
            Self::Failed(e) => return e.take().map(Err),
            Self::Open { family, input } => (*family, input),
        };
        let mut len = [0u8; 4];
        match input.read_exact(&mut len) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return None,
            Err(e) => return Some(Err(e.into())),
        }
        let n = u32::from_le_bytes(len) as usize;
        let mut bytes = vec![0u8; n * 12];
        if let Err(e) = input.read_exact(&mut bytes) {
            return Some(Err(e.into()));
        }
        let pts = bytes.as_chunks::<4>().0.as_chunks::<3>().0.iter().map(|p| p.map(f32::from_le_bytes)).collect();
        Some(Ok((family, pts)))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc::channel;
    use std::time::Duration;

    use super::*;

    #[test]
    fn pipeline_returns_a_consumer_error_while_the_producer_still_sends() {
        let (done_tx, done_rx) = channel();
        std::thread::spawn(move || {
            let produce = |tx: &SyncSender<usize>| -> Result<usize> {
                let mut sent = 0;
                for i in 0..1000 {
                    if tx.send(i).is_err() {
                        break;
                    }
                    sent += 1;
                }
                Ok(sent)
            };
            let consume = |_: usize| -> Result<()> { bail!("disk full") };
            let _ = done_tx.send(pipeline(2, produce, consume).map_err(|e| e.to_string()));
        });
        let result = done_rx.recv_timeout(Duration::from_secs(10)).expect("pipeline deadlocked");
        assert_eq!(result, Err("disk full".to_string()));
    }

    /// The prescan outcome for one argument line: "help", "version", "error" or "args"
    fn action(line: &str) -> &'static str {
        let args: Vec<OsString> = line.split_whitespace().map(OsString::from).collect();
        match prescan(&args) {
            Prescan::Help => "help",
            Prescan::Version => "version",
            Prescan::Error(_) => "error",
            Prescan::Args(_) => "args",
        }
    }

    /// Parse one argument line as main does: prescan, then clap
    fn parse(line: &str) -> Args {
        let args: Vec<OsString> = line.split_whitespace().map(OsString::from).collect();
        let Prescan::Args(args) = prescan(&args) else { panic!("{line}: no clap parse") };
        Args::try_parse_from(std::iter::once(OsString::from("x")).chain(args)).expect("parses")
    }

    #[test]
    fn early_action_follows_argparse_order_and_abbreviations() {
        assert_eq!(action("--vers"), "version");
        assert_eq!(action("--ve --help"), "version");
        assert_eq!(action("--help --version"), "help");
        assert_eq!(action("--threads 4 -h"), "help");
        assert_eq!(action("--origin-xyz -1 0 0 --version"), "version");
        assert_eq!(action("--voxel-size-um -.5 --version"), "version");
        // unknown options and stray values fail only at the end of argparse's pass
        assert_eq!(action("--bogus 3 --version"), "version");
        assert_eq!(action("-hh --threads x"), "help");
        assert_eq!(action("-hx --family X"), "help");
        assert_eq!(action("--threads 1_0 --help"), "help");
        assert_eq!(action("--voxel-size-um 1_0.5 --version"), "version");
        assert_eq!(action("-- --version"), "args");
        assert_eq!(action("--threads 4"), "args");
    }

    #[test]
    fn early_action_reports_option_errors_before_the_action() {
        for line in [
            "--threads x --help",
            "--threads --help",
            "--origin-xyz 1 --help",
            "--version=1 --help",
            "--v --version",
            "--family X --version",
            "--resume=1 --version",
            "--origin-xyz=1 --version",
            // argparse's negative-number pattern has no exponent form, so `-1e-3` reads as an option
            "--min-length-mm -1e-3 --version",
            "--family=X_Y --version",
            "--threads=x_y --help",
            "-h=bad --version",
            "-hh=x",
            "--threads _1 --help",
            "--threads 1__0 --help",
        ] {
            assert_eq!(action(line), "error", "{line}");
        }
    }

    #[test]
    fn values_with_spaces_are_checked_before_the_action() {
        for argv in [["--family=X Y", "--version"], ["--threads=x y", "--help"], ["--version=1 2", "--help"]] {
            let args: Vec<OsString> = argv.iter().map(OsString::from).collect();
            assert!(matches!(prescan(&args), Prescan::Error(_)), "{argv:?}");
        }
    }

    #[test]
    fn repeated_options_keep_the_last_value() {
        let args = parse(
            "--predictions p --voxel-size-um 1 --output o --work-dir w --threads 2 --threads 3 \
             --origin-xyz 1 2 3 --origin-xyz 4 5 6 --resume --resume --volume-shape-zyx 9 9 9 \
             --volume-shape-zyx 1 2 3",
        );
        assert_eq!(args.threads, Some(3));
        assert_eq!(args.origin_xyz, [4, 5, 6]);
        assert_eq!(args.volume_shape_zyx, Some(vec![1, 2, 3]));
        assert!(args.resume);
    }

    #[test]
    fn values_that_look_like_flags_parse_as_argparse_reads_them() {
        let args = parse("--pred p --voxel-size-um -.5 --output o --work-dir w --min-length-mm -2 --origin-xyz -1 0 0");
        assert_eq!(args.voxel_size_um, -0.5);
        assert_eq!(args.min_length_mm, Some(-2.0));
        assert_eq!(args.origin_xyz, [-1, 0, 0]);
        let mut args: Vec<OsString> = "--pred p --voxel-size-um 1_0.5 --output o --work-dir w --threads"
            .split_whitespace()
            .map(OsString::from)
            .collect();
        args.push(" 1_2 ".into());
        args.push("--family=V".into());
        let Prescan::Args(args) = prescan(&args) else { panic!("no clap parse") };
        let args = Args::try_parse_from(std::iter::once(OsString::from("x")).chain(args)).expect("parses");
        assert_eq!((args.voxel_size_um, args.threads, args.family), (10.5, Some(12), Family::V));
        assert_eq!(args.predictions, PathBuf::from("p"));
    }

    /// The tile options read their values as the reference's integer options do, a repeated one replaces the
    /// earlier, and their shared prefix is ambiguous as argparse finds it
    #[test]
    fn tile_options_parse_as_the_numeric_options() {
        let args = parse(
            "--pred p --voxel-size-um 1 --output o --work-dir w --tiles-yx 2 1_0 --tiles-yx 3 4 --tile-halo 1_28",
        );
        assert_eq!((args.tiles_yx, args.tile_halo), (Some(vec![3, 4]), Some(128)));
        let args = parse("--pred p --voxel-size-um 1 --output o --work-dir w --tiles 1 1");
        assert_eq!((args.tiles_yx, args.tile_halo), (Some(vec![1, 1]), None));
        assert_eq!(action("--tiles-yx 2 --help"), "error");
        assert_eq!(action("--tiles-yx 2 x --help"), "error");
        assert_eq!(action("--ti 2 --version"), "error");
        assert_eq!(action("--tile-h 64 --version"), "version");
    }

    #[test]
    fn pipeline_reports_a_producer_error_after_draining() {
        let mut seen = Vec::new();
        let produce = |tx: &SyncSender<usize>| -> Result<()> {
            tx.send(1)?;
            tx.send(2)?;
            bail!("trace failed")
        };
        let result = pipeline(2, produce, |v| {
            seen.push(v);
            Ok(())
        });
        assert_eq!(result.map_err(|e| e.to_string()), Err("trace failed".to_string()));
        assert_eq!(seen, [1, 2]);
    }

    /// Write a C-order, uncompressed `|u1` Zarr v2 array of shape (2, z, y, x) with chunks (2, depth, 16, 16);
    /// `value(c, z, y, x)` gives each voxel, and chunk files are written only when `value` is given
    fn write_zarr(dir: &Path, shape: Shape, depth: usize, value: Option<&dyn Fn(usize, usize, usize, usize) -> u8>) {
        let chunk = [depth, 16, 16];
        fs::create_dir_all(dir).unwrap();
        let meta = json!({
            "zarr_format": 2, "shape": [2, shape[0], shape[1], shape[2]], "chunks": [2, depth, 16, 16],
            "dtype": "|u1", "compressor": null, "fill_value": 0, "order": "C", "filters": null,
        });
        fs::write(dir.join(".zarray"), meta.to_string()).unwrap();
        let Some(value) = value else { return };
        let grid = [0, 1, 2].map(|a| shape[a].div_ceil(chunk[a]));
        for (gz, gy, gx) in
            (0..grid[0]).flat_map(|z| (0..grid[1]).flat_map(move |y| (0..grid[2]).map(move |x| (z, y, x))))
        {
            let mut bytes = Vec::with_capacity(2 * chunk.iter().product::<usize>());
            for c in 0..2 {
                for z in gz * chunk[0]..(gz + 1) * chunk[0] {
                    for y in gy * chunk[1]..(gy + 1) * chunk[1] {
                        for x in gx * chunk[2]..(gx + 1) * chunk[2] {
                            let inside = z < shape[0] && y < shape[1] && x < shape[2];
                            bytes.push(if inside { value(c, z, y, x) } else { 0 });
                        }
                    }
                }
            }
            fs::write(dir.join(format!("0.{gz}.{gy}.{gx}")), bytes).unwrap();
        }
    }

    /// The loaded tracing fields hold exactly the supplied array for chunk depths that take each slab path: whole
    /// chunks per slab (8, 24, 64 planes), brick rows within a chunk (17, 60), a partial final slab (150 planes is
    /// a multiple of none of them) and slabs whose chunks are all zero
    #[test]
    fn tracing_loads_the_supplied_array_at_any_chunk_depth() {
        let shape = [150, 40, 36];
        let root = std::env::temp_dir().join(format!("fibrs-load-tracing-{}", std::process::id()));
        let native_dir = root.join("native");
        write_zarr(&native_dir, shape.map(|n| 2 * n), 64, None);
        let native = Predictions::open(&native_dir, None, "VH").unwrap();
        let value = |c: usize, z: usize, y: usize, x: usize| -> u8 {
            if (40..80).contains(&z) || (z / 5 + y / 7 + x / 3).is_multiple_of(4) {
                return 0;
            }
            ((c * 31 + z * 7 + y * 3 + x) % 255 + 1) as u8
        };
        for depth in [8, 17, 24, 60, 64] {
            let dir = root.join(format!("tracing-{depth}"));
            write_zarr(&dir, shape, depth, Some(&value));
            let supplied = Predictions::open(&dir, None, "VH").unwrap();
            let fields = load_tracing(&native, Some(&supplied)).unwrap();
            for (c, field) in fields.iter().enumerate() {
                let got = field.read_block([0, 0, 0], shape);
                let mut i = 0;
                for z in 0..shape[0] {
                    for y in 0..shape[1] {
                        for x in 0..shape[2] {
                            assert_eq!(got[i], value(c, z, y, x), "depth {depth}, channel {c} at {z}, {y}, {x}");
                            i += 1;
                        }
                    }
                }
            }
        }
        fs::remove_dir_all(&root).unwrap();
    }
}
