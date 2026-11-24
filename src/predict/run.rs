//! The `fibrs predict` command: arguments, outputs and resume behaviour

use std::collections::HashSet;
use std::ffi::OsString;
use std::fs::{self, TryLockError};
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use clap::error::ErrorKind;
use clap::{CommandFactory, Parser, ValueEnum};
use color_eyre::eyre::{Result, bail, eyre};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use super::contract::{
    BORDER, CORE, Col, Columns, DROP_IN_FOR, MODEL_FILES, OVERLAP, Origin, REFERENCE, TILE, ct_identity, hex,
    positions, prediction_identity, sha256,
};
use super::ct::Ct;
use super::engine::{self, ColumnMemory, Engine, MAX_BATCH, MAX_COLUMN_VOXELS, Opened, Precision};
use super::feed::{FeedPlan, Item, Rule, TileFeed};
use super::out::{ColumnWriter, Finished, Output, now_unix};
use super::villa::{self, VillaSelection};
use crate::pyfmt::{JsonStyle, py_dumps};

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Mode {
    /// A z band [z0, z1) in batches of --batch, all-zero tiles skipped (the default)
    Band,
    /// The whole array as villa's lasagna.qualzz_inference computes it (batch 1, villa's storage-based tile skip)
    Villa,
}

/// AFV fibre model V2 probabilities (Qualzz20/afv_fiber_9um_v2) for a CT, on the GPU without PyTorch
///
/// Writes the probability zarr of villa lasagna/qualzz_inference.py: semantic (2, Z, Y, X) and tracing
/// (2, Z/2, Y/2, X/2) uint8, plus attempts.jsonl, progress.jsonl and summary-*.json records, so a capped run
/// resumes
#[derive(Parser, Debug)]
#[command(name = "fibrs predict", version = env!("CARGO_PKG_VERSION"))]
pub struct Args {
    /// Level-0 uint8 zarr v2 array (the /0 directory)
    #[arg(long)]
    ct: PathBuf,
    /// band: a z band [z0, z1); villa: the whole array as villa main computes it
    #[arg(long, value_enum, default_value = "band")]
    mode: Mode,
    /// First plane of the band (band mode; villa mode takes the whole array)
    #[arg(long, allow_negative_numbers = true)]
    z0: Option<i64>,
    /// End plane of the band, exclusive
    #[arg(long, allow_negative_numbers = true)]
    z1: Option<i64>,
    /// The model's Hugging Face snapshot (model.py, plans.json, dataset.json, checkpoint_final.pth)
    #[arg(long)]
    model_dir: PathBuf,
    /// The output zarr group
    #[arg(long)]
    out: PathBuf,
    /// Output column edge, an even multiple of 96 (band mode's output depends on it)
    #[arg(long, default_value_t = 480, allow_negative_numbers = true)]
    column: i64,
    /// Tiles sent to the GPU at a time (band mode, default 4, at most 16; villa mode: 1); the engine runs the
    /// network on one tile at a time, so this changes the speed, not the output
    #[arg(long, allow_negative_numbers = true)]
    batch: Option<i64>,
    /// Stop between columns after this long; 0 = no cap
    #[arg(long, default_value_t = 0.0)]
    max_minutes: f64,
    /// GPU device: cuda or cuda:N, or metal on macOS
    #[arg(long, default_value = engine::DEVICE_KIND)]
    device: String,
    /// Checkpoint sha256 the run must load
    #[arg(long)]
    expect_sha256: Option<String>,
    /// Tile reader threads (output unchanged)
    #[arg(long, default_value_t = 4, allow_negative_numbers = true)]
    readers: i64,
    // the help names the engine's precisions (fp32 and bf16 on CUDA, fp32 alone on Metal)
    #[arg(long, default_value = "fp32", help = engine::PRECISION_HELP)]
    precision: String,
    /// Comma list of column indices (in column order) to run; for timing subsets
    #[arg(long)]
    only_columns: Option<String>,
    /// Plan the run and read its tiles, print each column's batches and air count, and write nothing; needs no
    /// GPU
    #[arg(long)]
    dry_run: bool,
    /// The CT's voxel size in micrometres, written as the output's `voxel_size_um` attribute. The default is the
    /// PHerc0826 CT's, which the reference writes for every CT
    #[arg(long, default_value_t = 9.362, allow_negative_numbers = true)]
    voxel_size_um: f64,
}

/// The arguments after the checks argparse makes in the reference
struct Checked {
    a: Args,
    /// `--ct` canonicalized once; every read of the CT goes through it, so the store the overlap check passed is the
    /// one read even if a link on the given path changes during the run. The CT must exist by then, or a link made
    /// later would be read through but recorded unresolved, and a resume would not match it
    ct: PathBuf,
    /// `--out` resolved once; every write goes through it, since the given path may step through the CT on its way
    /// (`<ct>/0.0.0/../../pred.zarr`)
    out: PathBuf,
    batch: usize,
    readers: usize,
    column: usize,
    precision: Precision,
    device: usize,
    /// When the command started, before the paths were resolved, so `--max-minutes` and `this_run_wall_s` count it
    t_start: f64,
    started: Instant,
}

/// What each summary timing field measures here
fn timing() -> Value {
    json!({
        "forward_s": "host time in the native engine's add calls per batch: upload (which waits for the previous \
                      batch), normalise, network, seg head and blend; excludes the column's finalise",
        "wall_s": "from the column's first batch to its write finishing; columns overlap, so column_wall_s (their \
                   sum) exceeds the run's time and is not comparable with the band reference's",
        "this_run_wall_s": "end to end, the comparable figure",
    })
}

fn usage_error(kind: ErrorKind, msg: impl std::fmt::Display) -> ! {
    Args::command().error(kind, msg).exit()
}

fn parse(argv: &[OsString]) -> Checked {
    let (t_start, started) = (now_unix(), Instant::now());
    let a = Args::parse_from(std::iter::once(OsString::from("fibrs predict")).chain(argv.iter().cloned()));
    let batch = match (a.mode, a.batch) {
        (Mode::Band, b) => {
            if a.z0.is_none() || a.z1.is_none() {
                usage_error(ErrorKind::MissingRequiredArgument, "band mode needs --z0 and --z1");
            }
            b.unwrap_or(4)
        }
        // villa forwards one tile at a time
        (Mode::Villa, None | Some(1)) => 1,
        (Mode::Villa, Some(_)) => {
            usage_error(ErrorKind::ArgumentConflict, "villa mode runs batches of 1; leave --batch out or pass 1")
        }
    };
    let precision = Precision::parse(&a.precision).unwrap_or_else(|e| usage_error(ErrorKind::InvalidValue, e));
    let device = engine::device_index(&a.device).unwrap_or_else(|e| usage_error(ErrorKind::InvalidValue, e));
    if !(a.voxel_size_um.is_finite() && a.voxel_size_um > 0.0) {
        usage_error(ErrorKind::InvalidValue, "--voxel-size-um must be a positive number of micrometres");
    }
    if batch < 1 || a.readers < 1 {
        fail("--batch and --readers must be at least 1");
    }
    if batch as usize > MAX_BATCH {
        fail(format!("--batch is at most {MAX_BATCH}, the most tiles the engine runs at once"));
    }
    let column = usize::try_from(a.column).unwrap_or(0);
    let readers = a.readers as usize;
    // chunks are read by the path resolved here for the whole run, which assumes the CT's parent directories are not
    // renamed or swapped while it runs; holding a directory handle would cover that but is not worth it for a CLI
    let ct = fs::canonicalize(&a.ct).unwrap_or_else(|e| fail(format!("--ct {}: {e}", a.ct.display())));
    let out = resolved(&a.out).unwrap_or_else(|e| fail(format!("{e:#}")));
    Checked { a, ct, out, batch: batch as usize, readers, column, precision, device, t_start, started }
}

fn fail(msg: impl std::fmt::Display) -> ! {
    eprintln!("fibrs predict: {msg}");
    std::process::exit(1)
}

/// `fibrs predict ARGS`; returns the exit status
pub fn main(argv: &[OsString]) -> i32 {
    let c = parse(argv);
    match run(c) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("fibrs predict: {e:#}");
            1
        }
    }
}

/// The columns this run computes: `(index, column, tiles)`
type Todo = Vec<(usize, Col, Vec<Origin>)>;

/// The band and the columns to run, from the CT and the output's progress
struct Plan {
    ct: Arc<Ct>,
    z0: usize,
    z1: usize,
    nz: usize,
    ny: usize,
    nx: usize,
    cols: Columns,
    todo: Todo,
    selection: Option<Arc<VillaSelection>>,
    /// The length of progress.jsonl's complete records, when a torn last record follows them
    progress_complete: Option<u64>,
}

fn plan(c: &Checked, progress: &Path) -> Result<Plan> {
    let a = &c.a;
    let ct = Arc::new(Ct::open(&c.ct)?);
    let [d, ny, nx] = ct.shape;
    let (z0, z1) = match a.mode {
        Mode::Villa => {
            // the whole array is the volume, so tiles reflect at its own edges, as villa's do
            if !matches!((a.z0, a.z1), (None, None)) && (a.z0, a.z1) != (Some(0), Some(d as i64)) {
                bail!("villa mode runs the whole array: --z0/--z1 must be left out or be 0 {d}");
            }
            (0, d)
        }
        Mode::Band => {
            let (z0, z1) = (a.z0.unwrap_or_default(), a.z1.unwrap_or_default());
            if !(0 <= z0 && z0 < z1 && z1 <= d as i64) {
                bail!("the band [{z0}, {z1}) must lie within the CT's {d} planes");
            }
            if (z1 - z0) % 2 != 0 {
                bail!("band height must be even");
            }
            (z0 as usize, z1 as usize)
        }
    };

    let nz = z1 - z0;
    let mut done = HashSet::new();
    let mut progress_complete = None;
    if progress.exists() {
        let text = fs::read_to_string(progress)?;
        // every record is written with its newline, so text after the last newline is a record torn by a crash or
        // a full disk: its column has not been recorded and runs again
        let complete = text.rfind('\n').map_or(0, |i| i + 1);
        if complete < text.len() {
            progress_complete = Some(complete as u64);
        }
        for line in text[..complete].lines().filter(|l| !l.is_empty()) {
            let rec: Value = serde_json::from_str(line)?;
            let col: Vec<usize> = serde_json::from_value(rec["col"].clone())?;
            done.insert(col);
        }
    }
    let cols = Columns::new(ny, nx, c.column)?;
    let only = match &a.only_columns {
        None => None,
        Some(list) => {
            let only: HashSet<usize> = list
                .split(',')
                .map(str::trim)
                .filter(|v| !v.is_empty())
                .map(|v| v.parse::<i64>().map_err(|_| eyre!("--only-columns: {v:?} is not an integer")))
                .collect::<Result<Vec<i64>>>()?
                .into_iter()
                .map(|i| usize::try_from(i).ok().filter(|&i| i < cols.cols.len()))
                .collect::<Option<_>>()
                .ok_or_else(|| eyre!("--only-columns outside 0..{}", cols.cols.len() as i64 - 1))?;
            Some(only)
        }
    };
    let pz_all = positions(nz);
    let todo = cols
        .cols
        .iter()
        .enumerate()
        .filter(|(i, col)| !done.contains(col.as_slice()) && only.as_ref().is_none_or(|o| o.contains(i)))
        .map(|(i, &col)| (i, col, cols.tiles(col, &pz_all)))
        .collect();
    let selection = match a.mode {
        Mode::Villa => Some(Arc::new(VillaSelection::new(&c.ct, ct.shape)?)),
        Mode::Band => None,
    };
    Ok(Plan { ct, z0, z1, nz, ny, nx, cols, todo, selection, progress_complete })
}

fn start_feed(c: &Checked, p: &Plan) -> Result<TileFeed> {
    TileFeed::start(FeedPlan {
        ct: p.ct.clone(),
        ct_path: c.ct.clone(),
        z0: p.z0,
        z1: p.z1,
        columns: p.todo.iter().map(|(i, _, tiles)| (*i, tiles.clone())).collect(),
        batch: c.batch,
        readers: c.readers,
        rule: match &p.selection {
            Some(sel) => Rule::Villa(sel.clone()),
            None => Rule::Air,
        },
    })
}

/// `--dry-run`: each column's batches as the run would feed them, one JSON line per column
fn dry_run(c: &Checked) -> Result<()> {
    let p = plan(c, &c.out.join("progress.jsonl"))?;
    let feed = start_feed(c, &p)?;
    let (mut run, mut air) = (0, 0);
    let mut stdout = std::io::stdout().lock();
    for (i, col, tiles) in &p.todo {
        let mut batches = Vec::new();
        let tiles_air = loop {
            match feed.next()? {
                // the digest of the bytes the GPU would get pins down the reads and their reflection
                Item::Batch(b) => {
                    batches.push(json!({"origins": b.origins, "sha256": hex(&Sha256::digest(&b.data))}));
                    feed.recycle(b.data);
                }
                Item::ColumnEnd { air, .. } => break air,
            }
        };
        run += tiles.len() - tiles_air;
        air += tiles_air;
        let line = json!({"index": i, "col": col, "tiles": tiles.len(), "tiles_air": tiles_air, "batches": batches});
        writeln!(stdout, "{}", py_dumps(&line, JsonStyle::DEFAULT))?;
    }
    let total =
        json!({"columns_total": p.cols.cols.len(), "columns": p.todo.len(), "tiles_run": run, "tiles_air": air});
    writeln!(stdout, "{}", py_dumps(&total, JsonStyle::DEFAULT))?;
    Ok(())
}

/// Refuse, before the output is created or any column runs, pending columns the GPU can't hold
fn check_column_memory(c: &Checked, p: &Plan, e: &Opened) -> Result<()> {
    let Some([y0, y1, x0, x1]) = p.todo.iter().map(|t| t.1).max_by_key(|c| (c[1] - c[0]) * (c[3] - c[2])) else {
        return Ok(());
    };
    column_fits(&e.column, c.a.mode, &c.a.device, [p.nz, p.ny, p.nx], [y1 - y0, x1 - x0])
}

/// Refuses the widest pending column, `h` x `w` in a volume of `shape`, when the device can't hold it: past the
/// free memory, the engine's 32-bit indexing or the device's limit on one buffer. A column's accumulators span the
/// whole band height, so a tall volume needs a narrower column; the message names the largest size that fits. The
/// estimate leaves out the network's activations, which the engine has already allocated
fn column_fits(m: &ColumnMemory, mode: Mode, device: &str, shape: [usize; 3], [h, w]: [usize; 2]) -> Result<()> {
    let [nz, ny, nx] = shape;
    let voxels = |h: usize, w: usize| nz * h * w;
    let bytes = |h: usize, w: usize| m.bytes_per_voxel * voxels(h, w) as f64;
    let free = m.free_bytes as f64;
    // the column's largest buffer and the device's limit on one, where it has one
    let buffer = |h: usize, w: usize| {
        m.buffer_limit.as_ref().map(|l| (l.bytes_per_voxel * voxels(h, w) as f64, l.max_bytes as f64))
    };
    let fits = |h: usize, w: usize| {
        bytes(h, w) <= free && voxels(h, w) <= MAX_COLUMN_VOXELS && buffer(h, w).is_none_or(|(b, max)| b <= max)
    };
    if fits(h, w) {
        return Ok(());
    }

    // the largest column size whose widest column fits; sizes are multiples of the tile core, and a column is
    // cut short where the volume ends
    let top = ny.max(nx).div_ceil(CORE) * CORE;
    let fits = (1..=top / CORE).rev().map(|k| k * CORE).find(|&n| fits(n.min(ny), n.min(nx)));
    let hint = match (mode, fits) {
        (Mode::Villa, Some(n)) => format!("pass --column {n} or less, which leaves villa mode's output unchanged"),
        (Mode::Villa, None) => format!("no column of {nz} planes fits"),
        // in band mode the column size is part of the output, so a shorter band is the other way
        (Mode::Band, Some(n)) => {
            format!("run a shorter band, or pass --column {n} or less (band mode's output depends on it)")
        }
        (Mode::Band, None) => "run a shorter band".to_string(),
    };
    let gib = |b: f64| b / (1u64 << 30) as f64;
    if voxels(h, w) > MAX_COLUMN_VOXELS {
        bail!(
            "a {h} x {w} column of {nz} planes has {} voxels, more than the {MAX_COLUMN_VOXELS} the engine takes \
             in one column; {hint}",
            voxels(h, w)
        );
    }
    if let Some((b, max)) = buffer(h, w).filter(|(b, max)| b > max) {
        bail!(
            "a {h} x {w} column of {nz} planes needs a {:.2} GiB buffer on {device}, past its {:.2} GiB limit on one \
             buffer; {hint}",
            gib(b),
            gib(max)
        );
    }
    bail!(
        "a {h} x {w} column of {nz} planes needs about {:.1} GiB on {device}, which has {:.1} GiB free; {hint}",
        gib(bytes(h, w)),
        gib(free)
    )
}

/// `path` made absolute and resolved as the system would open it: the part that exists is canonicalized, links
/// and `..` included, and the rest, which a run creates, may hold only plain names, so the path given and the path
/// resolved always name the same place. On Windows, `std::path::absolute` already resolves `..` lexically, as the
/// system does before it opens a path
fn resolved(path: &Path) -> Result<PathBuf> {
    let path = std::path::absolute(path)?;
    let parts: Vec<Component> = path.components().filter(|c| *c != Component::CurDir).collect();
    // the longest prefix that exists, walked as the system walks it: a missing part ends it, and any other error
    // (a file followed by more of the path, say) refuses the path
    let mut prefix = PathBuf::new();
    let mut existing = 0;
    for part in &parts {
        prefix.push(part);
        match fs::symlink_metadata(&prefix) {
            Ok(_) => existing += 1,
            Err(e) if e.kind() == io::ErrorKind::NotFound => break,
            Err(e) => bail!("{} steps through {}: {e}", path.display(), prefix.display()),
        }
    }

    let head: PathBuf = parts[..existing].iter().collect();
    let Ok(mut out) = head.canonicalize() else {
        bail!("{} steps through {}, which does not resolve", path.display(), head.display());
    };
    for part in &parts[existing..] {
        let Component::Normal(name) = part else {
            let missing: PathBuf = parts[..=existing].iter().collect();
            bail!("{} uses .. after {}, which does not exist", path.display(), missing.display());
        };
        out.push(name);
    }

    Ok(out)
}

/// Locks the output for this process; the lock goes with the process, so a killed run leaves none behind
fn lock_output(attempts: &fs::File, out: &Path) -> Result<()> {
    match try_lock(attempts) {
        Ok(()) => Ok(()),
        Err(TryLockError::WouldBlock) => {
            bail!("{} is in use by another fibrs predict run; wait for it to end or use another --out", out.display())
        }
        // unguarded, two runs would lose each other's columns, so a filesystem without locks is refused
        Err(TryLockError::Error(e)) => {
            bail!("cannot lock {} ({e}); fibrs predict needs an output on a filesystem with file locks", out.display())
        }
    }
}

/// An exclusive lock on the whole file: flock(LOCK_EX | LOCK_NB), advisory, so the attempt log stays readable
#[cfg(not(windows))]
fn try_lock(file: &fs::File) -> Result<(), TryLockError> {
    file.try_lock()
}

/// An exclusive lock on one byte far past any attempt log's end. Windows' locks are mandatory, so the whole-file lock
/// std takes would keep anyone from reading the log while the run goes on; this byte excludes a second run just as
/// well, and a whole-file lock overlaps it
#[cfg(windows)]
fn try_lock(file: &fs::File) -> Result<(), TryLockError> {
    use std::os::windows::io::AsRawHandle;

    use windows_sys::Win32::Foundation::ERROR_LOCK_VIOLATION;
    use windows_sys::Win32::Storage::FileSystem::{LOCKFILE_EXCLUSIVE_LOCK, LOCKFILE_FAIL_IMMEDIATELY, LockFileEx};
    use windows_sys::Win32::System::IO::OVERLAPPED;

    // the byte at offset 2^62
    let mut at = OVERLAPPED::default();
    at.Anonymous.Anonymous.OffsetHigh = 1 << 30;
    let flags = LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY;
    // SAFETY: the handle is open for the call, and on a synchronous handle LockFileEx is done with `at` on return
    if unsafe { LockFileEx(file.as_raw_handle(), flags, 0, 1, 0, &mut at) } != 0 {
        return Ok(());
    }

    let e = io::Error::last_os_error();
    if e.raw_os_error() == Some(ERROR_LOCK_VIOLATION as i32) {
        Err(TryLockError::WouldBlock)
    } else {
        Err(TryLockError::Error(e))
    }
}

fn run(c: Checked) -> Result<()> {
    if c.a.dry_run {
        return dry_run(&c);
    }

    let a = &c.a;
    let (t_start, started) = (c.t_start, c.started);
    // nothing is written before the arguments and the model folder are checked: an output in, at or around the CT
    // would write into the input store
    let out = &c.out;
    if out.starts_with(&c.ct) || c.ct.starts_with(out) {
        bail!("--out {} overlaps --ct {}; write the output outside the input", a.out.display(), a.ct.display());
    }
    // the loader takes a fold_* folder's plans.json and dataset.json from its parent, so those hashed here
    // would not be the ones it runs with
    if a.model_dir.file_name().is_some_and(|n| n.to_string_lossy().starts_with("fold_")) {
        bail!(
            "{} is named like a fold folder, so its plans.json and dataset.json would be read from its parent; \
             pass the model folder that holds model.py",
            a.model_dir.display()
        );
    }
    let mut hashes = Map::new();
    for name in MODEL_FILES {
        hashes.insert(name.into(), json!(sha256(&a.model_dir.join(name))?));
    }
    let checkpoint = hashes["checkpoint_final.pth"].as_str().unwrap_or_default().to_string();
    if let Some(want) = &a.expect_sha256
        && *want != checkpoint
    {
        bail!("checkpoint sha256 {checkpoint} != {want}");
    }
    // the loader reads fold_0's checkpoint, when there is one, before the top-level one hashed above (villa's
    // `load_model` order), so a run with one would check and record other weights than it runs
    for name in ["fold_0/checkpoint_final.pth", "fold_0/checkpoint_best.pth"] {
        if a.model_dir.join(name).exists() {
            bail!(
                "{} holds {name}, which would be loaded in place of the checkpoint_final.pth beside model.py; \
                 pass a model folder without it",
                a.model_dir.display()
            );
        }
    }

    // every run that passes the checks above is recorded before the model loads, so one capped or killed after
    // that still shows it
    fs::create_dir_all(out)?;
    // read as well as append: Windows locks a file only through a handle with read or write access, and append
    // alone is neither
    let mut attempts = fs::OpenOptions::new().create(true).read(true).append(true).open(out.join("attempts.jsonl"))?;
    // held until the run ends: two runs on one output, on different --only-columns say, would each rewrite the
    // chunks their columns share from what they read, losing the other's columns while both record progress
    lock_output(&attempts, out)?;
    let attempt = py_dumps(&json!({"attempt": t_start, "start": t_start}), JsonStyle::DEFAULT);
    attempts.write_all(format!("{attempt}\n").as_bytes())?;

    let progress = out.join("progress.jsonl");
    let p = plan(&c, &progress)?;
    let mut e = engine::open(&a.model_dir, c.device, c.precision)?;
    check_column_memory(&c, &p, &e)?;

    let villa = p.selection.is_some();
    let mut prov = Map::new();
    prov.insert("model_sha256".into(), Value::Object(hashes));
    let exe = std::env::current_exe()?;
    prov.insert("code_sha256".into(), json!({format!("fibrs {}", env!("CARGO_PKG_VERSION")): sha256(&exe)?}));
    // the CT was opened through `c.ct`; if a directory on it was replaced by a link since it was resolved, the store
    // read is not the one recorded, and a resume could mix two stores whose metadata match
    let now = fs::canonicalize(&c.ct)?;
    if now != c.ct {
        bail!("--ct {} now resolves to {}; it changed while the run started", c.ct.display(), now.display());
    }
    prov.insert("ct".into(), json!(c.ct.to_string_lossy()));
    prov.insert("ct_identity".into(), json!(ct_identity(&c.ct, p.z0, p.z1)?));
    for (k, v) in [
        ("z0", json!(p.z0)),
        ("z1", json!(p.z1)),
        ("tile", json!(TILE)),
        ("overlap", json!(OVERLAP)),
        ("border", json!(BORDER)),
        ("column", json!(c.column)),
        ("batch", json!(c.batch)),
        ("precision", json!(c.precision.label())),
        ("normalization", json!("model.py:normalize_ct per tile")),
        ("air_tiles", json!(if villa { villa::SKIP_RULE } else { "skipped (all-zero input)" })),
    ] {
        prov.insert(k.into(), v);
    }
    prov.extend(e.provenance.clone());
    prov.insert("channel_order".into(), json!("VH"));
    prov.insert("reference".into(), json!(if villa { villa::REFERENCE } else { REFERENCE }));
    prov.insert("engine".into(), json!("fibrs-predict"));
    if !villa {
        prov.insert("drop_in_for".into(), json!(DROP_IN_FOR));
    }
    // GPUs round differently, so a resume must run where the output started
    prov.insert("device".into(), json!(engine::DEVICE_KIND));
    prov.insert("gpu".into(), json!(e.gpu));
    if villa {
        prov.insert("mode".into(), json!("villa"));
    }

    let mut attrs = Map::new();
    attrs.insert("channel_order".into(), json!("VH"));
    attrs.insert("axes".into(), json!(["c", "z", "y", "x"]));
    attrs.insert("voxel_size_um".into(), json!(a.voxel_size_um));
    attrs.insert("origin_zyx".into(), json!([p.z0, 0, 0]));
    attrs.insert("inference".into(), Value::Object(prov));
    let output = Output::open_or_create(out, p.nz, p.ny, p.nx, &attrs)?;

    let feed = start_feed(&c, &p)?;
    if let Some(len) = p.progress_complete {
        // the torn record goes before this run appends its own
        fs::OpenOptions::new().write(true).open(&progress)?.set_len(len)?;
    }
    let mut writer = ColumnWriter::start(output, progress.clone())?;
    let stopped = columns(&c, &p, e.engine.as_mut(), &feed, &writer, t_start, started);
    drop(feed);
    let drained = writer.drain();
    let stopped = stopped?;
    drained?;

    // a cap reached during setup stops before any column, so a fresh output may have no progress yet
    let recs: Vec<Value> = match fs::read_to_string(&progress) {
        Ok(text) => text.lines().filter(|l| !l.is_empty()).map(serde_json::from_str).collect::<Result<_, _>>()?,
        Err(_) => Vec::new(),
    };
    let sum_f = |k: &str| recs.iter().map(|r| r[k].as_f64().unwrap_or(0.0)).sum::<f64>();
    let sum_u = |k: &str| recs.iter().map(|r| r[k].as_u64().unwrap_or(0)).sum::<u64>();
    let mut summary = json!({
        "model": "AFV v2",
        "precision": c.precision.short(),
        "attempt": t_start,
        "end": now_unix(),
        "complete": recs.len() == p.cols.cols.len(),
        "identity": prediction_identity(out)?,
        "columns_total": p.cols.cols.len(),
        "columns_done": recs.len(),
        "stopped_early": stopped,
        "tiles_run": sum_u("tiles_run"),
        "tiles_air": sum_u("tiles_air"),
        "forward_s": sum_f("forward_s"),
        "column_wall_s": sum_f("wall_s"),
        "this_run_wall_s": started.elapsed().as_secs_f64(),
        "peak_vram_gib": null,
        "gpu": e.gpu,
        "engine": "fibrs-predict",
        "readers": c.readers,
        "timing": timing(),
        "cuda_libraries": e.libraries,
        "cudnn": e.provenance.get("cudnn"),
        "cublas": e.provenance.get("cublas"),
    });
    if villa {
        summary["mode"] = json!("villa");
    }
    let name = format!("summary-{}.json", t_start as i64);
    fs::write(out.join(name), py_dumps(&summary, JsonStyle::indented(1)))?;
    println!("{}", py_dumps(&summary, JsonStyle::DEFAULT));
    Ok(())
}

/// The column loop: each column's non-air tiles, in the feed's order and batches, into the engine's
/// accumulators, then its finalise and write; returns whether `--max-minutes` stopped the run
fn columns(
    c: &Checked,
    p: &Plan,
    engine: &mut dyn Engine,
    feed: &TileFeed,
    writer: &ColumnWriter,
    t_start: f64,
    started: Instant,
) -> Result<bool> {
    for (i, col, tiles) in &p.todo {
        if c.a.max_minutes > 0.0 && started.elapsed().as_secs_f64() > c.a.max_minutes * 60.0 {
            return Ok(true);
        }
        let t0 = Instant::now();
        let [y0, y1, x0, x1] = *col;
        engine.begin([p.nz, y1 - y0, x1 - x0], y0, x0)?;
        let (mut forward, mut run) = (0.0f64, 0usize);
        let air = loop {
            match feed.next()? {
                Item::Batch(b) if b.col == *i => {
                    let t = Instant::now();
                    engine.add(&b.data, &b.origins)?;
                    forward += t.elapsed().as_secs_f64();
                    run += b.origins.len();
                    feed.recycle(b.data);
                }
                Item::ColumnEnd { col: ci, air } if ci == *i => break air,
                Item::Batch(b) => bail!("tile feed out of order: column {} while running {i}", b.col),
                Item::ColumnEnd { col: ci, .. } => bail!("tile feed out of order: column {ci} while running {i}"),
            }
        };
        // an all-air column quantizes nothing and the engine drops whatever buffers it is given, so only a
        // column that ran tiles takes a written column's pair, which the next populated column would otherwise lose
        let quantize = run > 0;
        let reuse = if quantize { writer.spare() } else { Default::default() };
        let mut out = engine.finish(quantize, reuse)?;
        if let (Some(sel), Some(sem), Some(tra)) = (&p.selection, &mut out.semantic, &mut out.tracing) {
            sel.mask_column(*col, sem, tra);
        }
        let mut record = Map::new();
        record.insert("col".into(), json!(col));
        record.insert("tiles".into(), json!(tiles.len()));
        record.insert("attempt".into(), json!(t_start));
        record.insert("forward_s".into(), json!(forward));
        record.insert("tiles_run".into(), json!(run));
        record.insert("tiles_air".into(), json!(air));
        record.insert("voxels_uncovered".into(), json!(out.uncovered));
        writer.submit(Finished { col: *col, nz: p.nz, sem: out.semantic, tra: out.tracing, record, started: t0 })?;
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use std::fs::{self, TryLockError};
    use std::io::{Read, Write};

    use super::{Mode, column_fits, try_lock};
    use crate::predict::engine::{BufferLimit, ColumnMemory};

    /// The output lock turns away a second run, and on Windows, where locks are mandatory, still lets another
    /// handle read the attempt log while it is held
    #[test]
    fn the_output_lock_refuses_a_second_run_and_leaves_the_log_readable() {
        let dir = std::env::temp_dir().join(format!("fibrs-output-lock-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let log = dir.join("attempts.jsonl");
        let open = || fs::OpenOptions::new().create(true).read(true).append(true).open(&log).unwrap();

        let mut held = open();
        try_lock(&held).unwrap();
        held.write_all(b"{\"attempt\": 1}\n").unwrap();

        assert!(matches!(try_lock(&open()), Err(TryLockError::WouldBlock)));
        let mut read = String::new();
        fs::File::open(&log).unwrap().read_to_string(&mut read).unwrap();
        assert_eq!(read, "{\"attempt\": 1}\n");

        drop(held);
        try_lock(&open()).unwrap();
        fs::remove_dir_all(&dir).unwrap();
    }

    /// A column that fits the free memory but not one device buffer is refused, and the suggested column is the
    /// largest whose accumulator fits that buffer: at 512 planes and a 1 GiB limit on 8 bytes a voxel, 576 x 576
    /// needs 1.27 GiB and 480 x 480 0.88 GiB. Without a buffer limit (CUDA) the same column runs
    #[test]
    fn a_column_past_the_device_buffer_limit_is_refused_with_a_column_that_fits() {
        let limit = |max_gib: u64| Some(BufferLimit { max_bytes: max_gib << 30, bytes_per_voxel: 8.0 });
        let memory = |buffer_limit| ColumnMemory { free_bytes: 64 << 30, bytes_per_voxel: 14.25, buffer_limit };
        let (shape, col) = ([512, 1024, 1024], [576, 576]);

        let err = column_fits(&memory(limit(1)), Mode::Villa, "metal", shape, col).unwrap_err().to_string();
        assert!(err.contains("needs a 1.27 GiB buffer on metal, past its 1.00 GiB limit on one buffer"), "{err}");
        assert!(err.contains("pass --column 480 or less"), "{err}");
        let err = column_fits(&memory(limit(1)), Mode::Band, "metal", shape, col).unwrap_err().to_string();
        assert!(err.contains("or pass --column 480 or less"), "{err}");

        column_fits(&memory(None), Mode::Villa, "cuda", shape, col).unwrap();
        column_fits(&memory(limit(2)), Mode::Villa, "metal", shape, col).unwrap();
    }
}
