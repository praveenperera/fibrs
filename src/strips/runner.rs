//! `fibrs strips run`: the native strip runner, with the Python reference runner's command line and outputs
//!
//! Fibres are selected and sharded as the Python reference runner does (Morton order, contiguous shards,
//! `--exclude-ids-file`), and a shard resumes from every shard's index. CPU workers each take the next fibre: load
//! it, plan its decisions (CT seed headings included), hand the frame requests to the GPU engine, finish the
//! decisions from the frame model's output, hand the strip requests to the engine, then stitch and write the strip.
//! The engine thread batches both models' work across the fibres in flight. The main thread writes the index: one
//! line per fibre as it finishes, its `seconds` the time since the shard's previous line (fibres overlap, so the
//! column sums to the shard's wall time, as the Python runner's does)

use std::collections::{BTreeSet, HashSet};
use std::fs::{self, OpenOptions};
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{Sender, channel};
use std::time::Instant;

use clap::Parser;
use color_eyre::eyre::{Result, WrapErr, bail, ensure, eyre};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use super::ct::{Uncached, Volume};
use super::engine::{EngineConfig, LaunchError, Work, serve};
use super::fibre::{Afv, FibreRow};
use super::geom::{Geometry, Umbilicus};
use super::gpu::Weights;
use super::out::{Record, Settings, run_json};
use super::seed::{SeedHeadingError, Workspace};
use super::stitch::{stitched_strip, strip_f};
use super::types::{FrameRequest, StripRequest};

/// The checkpoint the weights export is pinned to (its sha256 is checked at load), as trace.json records it
const CHECKPOINT: &str = "hf://scrollprize/fiber_strip_small_v1@7bd163a22c839ff312879ae1916195d0c9729c0e/last.pt";
/// Its training step
const STEP: i64 = 15000;
/// Trace voxels per patch voxel (`ct_grid_scale / grid_scale`): the checkpoint takes one CT voxel per patch voxel,
/// and the runner supports two CT voxels per trace voxel, the scale every scroll's AFV is traced at
const TRACE_PER_PATCH: f64 = 0.5;

/// The native sheet-strip runner
#[derive(Parser, Debug)]
#[command(name = "fibrs strips run")]
pub struct Args {
    /// The AFV
    #[arg(long)]
    afv: PathBuf,
    /// What trace.json records as the fibre source (default: --afv)
    #[arg(long)]
    afv_ref: Option<String>,
    #[arg(long)]
    out: PathBuf,
    /// Fibre ids, in the order given
    #[arg(long, num_args = 1..)]
    ids: Vec<i64>,
    /// Fibre ids, whitespace-separated
    #[arg(long)]
    ids_file: Option<PathBuf>,
    /// Ids to skip, whitespace-separated (repeatable)
    #[arg(long)]
    exclude_ids_file: Vec<PathBuf>,
    #[arg(long, default_value_t = 2.5)]
    min_length_mm: f64,
    /// Scan voxel size: AFV lengths and area_cm2
    #[arg(long)]
    voxel_size_um: f64,
    /// The strip model's weights export (the checkpoint's tensors as float32 safetensors)
    #[arg(long)]
    weights: PathBuf,
    /// The CT zarr (`s3://...`), read from vc_reader's disk cache
    #[arg(long)]
    ct: String,
    #[arg(long, default_value_t = 0)]
    ct_level: u32,
    /// Base voxels per CT voxel
    #[arg(long, default_value_t = 1.0)]
    ct_grid_scale: f64,
    /// Base voxels per trace voxel
    #[arg(long, default_value_t = 2.0)]
    grid_scale: f64,
    /// vc_reader's disk cache holding the CT (prefetched)
    #[arg(long)]
    cache_dir: PathBuf,
    #[arg(long)]
    umbilicus: PathBuf,
    /// CUDA device ordinal
    #[arg(long, default_value_t = 0)]
    device: usize,
    /// CT chunks in the GPU cache (2 MiB each)
    #[arg(long, default_value_t = 4096)]
    gpu_chunks: usize,
    /// Decisions per GPU launch, across fibres
    #[arg(long, default_value_t = 64)]
    batch: usize,
    /// CPU workers (fibres in flight)
    #[arg(long, default_value_t = 16)]
    workers: usize,
    #[arg(long, default_value_t = 8.0)]
    step: f64,
    #[arg(long, default_value_t = 16.0)]
    blend: f64,
    /// End trim, patch voxels
    #[arg(long, default_value_t = 75.0)]
    trim: f64,
    /// This shard K of N
    #[arg(long, num_args = 2, value_names = ["K", "N"], default_values_t = [0, 1])]
    shard: Vec<usize>,
    /// At most this many of the shard's fibres
    #[arg(long)]
    limit: Option<usize>,
    /// Run and stitch, write no strips and no index lines (timing only)
    #[arg(long)]
    no_write: bool,
    /// Also write a per-fibre npz of every decision's raw results here
    #[arg(long)]
    npz: Option<PathBuf>,
    /// The perturbation the weights export carries (negative controls); must match its metadata
    #[arg(long, default_value_t = 0.0)]
    perturb: f64,
    /// JSON object recorded in run.json; its "runner" is the run's tag, on every index line
    #[arg(long, default_value = "{}")]
    provenance: String,
}

/// What a worker hands the index writer for one fibre
enum Outcome {
    /// Its record (None for a fibre too short to run) and its decisions
    Indexed(Option<Record>, usize),
    /// Deferred: it needs CT chunks the disk cache lacks (these, or those of a launch it shared). It is not indexed,
    /// so the run after their fetch takes it up again
    Uncached(Vec<[usize; 3]>),
}

type Done = Result<Outcome>;

/// The exit code of a run that finished every fibre it could but deferred some for uncached CT chunks, listed in
/// `<out>/logs/uncached_chunks.txt`: fetch them and run again
pub const EXIT_UNCACHED: i32 = 75;

/// A run that deferred fibres for uncached CT chunks; `fibrs strips` exits with [`EXIT_UNCACHED`]
#[derive(Debug)]
pub struct Deferred {
    pub fibres: usize,
    pub chunks: usize,
    pub list: PathBuf,
}

impl std::fmt::Display for Deferred {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} fibres deferred: {} CT chunks are not in the disk cache, listed in {}; fetch them and run again",
            self.fibres,
            self.chunks,
            self.list.display()
        )
    }
}

impl std::error::Error for Deferred {}

/// Refuses settings the runner can't run correctly, before anything is read or written
fn validate(args: &Args) -> Result<()> {
    ensure!(args.shard.len() == 2 && args.shard[1] > 0 && args.shard[0] < args.shard[1], "--shard K N with K < N");
    ensure!(args.workers > 0 && args.batch > 0, "--workers and --batch above 0");
    ensure!(args.step.is_finite() && args.step > 0.0, "--step: a positive number of patch voxels");
    // decisions land on whole strip rows only when they are a whole number of rows apart
    let f = strip_f();
    let rows = args.step / (f[1] - f[0]);
    ensure!(rows.fract() == 0.0, "--step {}: a multiple of the strip's row spacing, {}", args.step, f[1] - f[0]);
    ensure!(args.blend.is_finite() && args.blend >= 0.0, "--blend: zero or more patch voxels");
    ensure!(args.trim.is_finite() && args.trim >= 0.0, "--trim: zero or more patch voxels");
    ensure!(
        args.ct_grid_scale > 0.0 && args.grid_scale == 2.0 * args.ct_grid_scale,
        "--grid-scale {} with --ct-grid-scale {}: the runner supports two CT voxels per trace voxel only",
        args.grid_scale,
        args.ct_grid_scale
    );
    Ok(())
}

/// `--provenance`: a JSON object, which run.json records over the run's own fields, so it may not set a field that
/// decides whether an OUT can be resumed
fn provenance(text: &str) -> Result<Map<String, Value>> {
    let Ok(Value::Object(m)) = serde_json::from_str(text) else { bail!("--provenance: a JSON object") };
    let reserved: Vec<&str> = RESUME_FIELDS.into_iter().filter(|key| m.contains_key(*key)).collect();
    ensure!(reserved.is_empty(), "--provenance may not set {reserved:?}: the run itself records them");
    Ok(m)
}

/// Runs the shard; the progress lines go to stdout as the Python runner prints them
pub fn run(args: &Args) -> Result<()> {
    validate(args)?;
    let (k, n) = (args.shard[0], args.shard[1]);
    let input_scale = args.grid_scale / args.ct_grid_scale;
    let provenance = provenance(&args.provenance)?;
    let tag = provenance.get("runner").and_then(Value::as_str).map_or_else(default_tag, str::to_string);

    let t0 = Instant::now();
    let weights = Weights::open(&args.weights)?;
    let carried: f64 = weights.metadata.get("perturb").map_or(Ok(0.0), |p| p.parse()).wrap_err("weights perturb")?;
    ensure!(carried == args.perturb, "--perturb {} but the weights carry perturb {carried}", args.perturb);

    // every shard claims or checks run.json before it reads or repairs the index
    fs::create_dir_all(args.out.join("index")).wrap_err_with(|| format!("create {}", args.out.display()))?;
    let inputs = Inputs::read(args, &weights)?;
    let fields = run_fields(args, n, &tag, &inputs);
    claim_run_json(&args.out, &fields, &provenance, &tag)?;
    let todo = select(args, k, n)?;

    let vol = Arc::new(Volume::open(&args.cache_dir, &args.ct, args.ct_level)?);
    let umbilicus = Umbilicus::read(&args.umbilicus)?;
    let geometry = Geometry { ct: &vol, input_scale, trace_per_patch: TRACE_PER_PATCH, umbilicus: &umbilicus };
    let settings = Settings {
        out: (!args.no_write).then(|| args.out.clone()),
        npz: args.npz.clone(),
        checkpoint: CHECKPOINT.into(),
        step: Some(STEP),
        afv_ref: args.afv_ref.clone().unwrap_or_else(|| args.afv.display().to_string()),
        decision_step: args.step,
        blend: args.blend,
        trim: args.trim,
        voxel_size_um: args.voxel_size_um,
        trace_per_patch: TRACE_PER_PATCH,
    };

    // the engine thread owns the device; workers block on its replies
    let config =
        EngineConfig { device: args.device, batch: args.batch, chunk_slots: args.gpu_chunks, staging: 64, input_scale };
    let (work_tx, work_rx) = channel::<Work>();
    let (ready_tx, ready_rx) = channel();
    let engine_vol = vol.clone();
    let engine = std::thread::Builder::new()
        .name("strips-engine".into())
        .spawn(move || serve(&config, &weights, engine_vol, work_rx, ready_tx))?;
    match ready_rx.recv() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => bail!("the GPU engine: {e}"),
        Err(_) => bail!("the GPU engine stopped before it was ready"),
    }
    println!(
        "{}",
        json!({"load_s": round2(t0.elapsed().as_secs_f64()), "shard": [k, n], "fibres": todo.total,
               "todo": todo.rows.len(), "workers": args.workers, "batch": args.batch, "pid": std::process::id()})
    );

    let next = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let (done_tx, done_rx) = channel::<Done>();
    let started = Instant::now();
    let written = std::thread::scope(|scope| -> Result<Written> {
        for _ in 0..args.workers.min(todo.rows.len()) {
            let (work_tx, done_tx) = (work_tx.clone(), done_tx.clone());
            let (rows, next, stop, geometry, settings) = (&todo.rows, &next, &stop, &geometry, &settings);
            scope.spawn(move || worker(args, rows, next, stop, geometry, settings, &work_tx, &done_tx));
        }
        drop((work_tx, done_tx));
        let written = write_index(args, k, &tag, todo.rows.len(), done_rx, started);
        if written.is_err() {
            stop.store(true, Ordering::Relaxed);
        }
        written
    });
    let counts = engine.join().map_err(|_| eyre!("the GPU engine panicked"))??;
    let written = written?;
    let list = write_uncached(args, k, n, &written)?;
    let decisions = written.decisions;

    let wall = started.elapsed().as_secs_f64();
    println!(
        "{}",
        json!({"shard": k, "finished": true, "decisions": decisions, "wall_s": round2(wall),
               "dps": round2(decisions as f64 / wall.max(1e-9)), "gpu_busy_s": round2(counts.busy),
               "frame_launches": counts.frame_launches, "strip_launches": counts.strip_launches,
               "frame_deferrals": counts.frame_deferrals,
               "chunks": {"needed": counts.cache.needed, "uploads": counts.cache.uploads, "fills": counts.cache.fills,
                          "evictions": counts.cache.evictions, "most_per_launch": counts.cache.most,
                          "prefetched": counts.cache.prefetched},
               "deferred": written.deferred, "uncached_chunks": written.uncached.len()})
    );
    if written.deferred > 0 {
        return Err(Deferred { fibres: written.deferred, chunks: written.uncached.len(), list }.into());
    }
    Ok(())
}

/// The shard's fibres not yet in any shard's index, and the shard's size before that
struct Todo {
    rows: Vec<FibreRow>,
    total: usize,
}

fn select(args: &Args, k: usize, n: usize) -> Result<Todo> {
    let mut ids = args.ids.clone();
    if let Some(path) = &args.ids_file {
        ids = read_ids(path)?;
        // an empty list would otherwise select every fibre of the scroll
        ensure!(!ids.is_empty(), "{} holds no fibre ids", path.display());
    }
    let afv = Afv::open(&args.afv)?;
    let rows = afv.select(args.min_length_mm, args.voxel_size_um, (!ids.is_empty()).then_some(ids.as_slice()))?;
    let mut excluded = HashSet::new();
    for path in &args.exclude_ids_file {
        excluded.extend(read_ids(path)?);
    }
    let rows: Vec<FibreRow> = rows.into_iter().filter(|r| !excluded.contains(&r.id)).collect();

    let per = rows.len().div_ceil(n);
    let mut chosen: Vec<FibreRow> = rows.into_iter().skip(k * per).take(per).collect();
    if let Some(limit) = args.limit {
        chosen.truncate(limit);
    }
    let done = indexed(&args.out.join("index"), k)?;
    let total = chosen.len();
    Ok(Todo { rows: chosen.into_iter().filter(|r| !done.contains(&r.id)).collect(), total })
}

fn read_ids(path: &Path) -> Result<Vec<i64>> {
    let text = fs::read_to_string(path).wrap_err_with(|| format!("read {}", path.display()))?;
    text.split_whitespace().map(|t| t.parse().wrap_err_with(|| format!("{}: id {t:?}", path.display()))).collect()
}

/// Every fibre id in `index/shard_*.jsonl`. Shard `own`'s index loses a torn last line for good; other shards may
/// be writing theirs, so a partial last line there is only skipped
fn indexed(dir: &Path, own: usize) -> Result<HashSet<i64>> {
    let own = dir.join(format!("shard_{own}.jsonl"));
    let mut done = HashSet::new();
    let Ok(entries) = fs::read_dir(dir) else { return Ok(done) };
    for entry in entries {
        let path = entry?.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default();
        if !(name.starts_with("shard_") && name.ends_with(".jsonl")) {
            continue;
        }
        let text = if path == own {
            repair_index(&path)?
        } else {
            let text = fs::read_to_string(&path).wrap_err_with(|| format!("read {}", path.display()))?;
            whole_lines(text)
        };
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let record: Value = serde_json::from_str(line).wrap_err_with(|| format!("{}: {line}", path.display()))?;
            done.insert(record["fiber"].as_i64().ok_or_else(|| eyre!("{}: no fiber in {line}", path.display()))?);
        }
    }
    Ok(done)
}

/// The run.json fields that change a strip's content: a resumed OUT must match them, so one index never mixes
/// strips made from different inputs or settings
const RESUME_FIELDS: [&str; 16] = [
    "checkpoint",
    "afv",
    "afv_uuid",
    "afv_bytes",
    "umbilicus_sha256",
    "weights_sha256",
    "voxel_size_um",
    "decision_step",
    "blend",
    "trim_patch_voxels",
    "trace_per_patch",
    "ct",
    "ct_level",
    "ct_grid_scale",
    "grid_scale",
    "perturb",
];

/// Drops a shard index's torn last line (a write cut short, as by a full disk), so the fibre reruns and later
/// appends start on a line of their own; the file's text after the repair
fn repair_index(path: &Path) -> Result<String> {
    let full = fs::read_to_string(path).wrap_err_with(|| format!("read {}", path.display()))?;
    let len = full.len();
    let text = whole_lines(full);
    if text.len() == len {
        return Ok(text);
    }

    let file = OpenOptions::new().write(true).open(path).wrap_err_with(|| format!("open {}", path.display()))?;
    file.set_len(text.len() as u64).wrap_err_with(|| format!("truncate {}", path.display()))?;
    Ok(text)
}

/// `text` up to its last newline: without a partial last line
fn whole_lines(mut text: String) -> String {
    text.truncate(text.rfind('\n').map_or(0, |i| i + 1));
    text
}

/// Refuses an OUT begun under another runner tag (one scroll, one runner) or with other output-affecting settings
fn check_resume(out: &Path, tag: &str, fields: &Map<String, Value>) -> Result<()> {
    let path = out.join("run.json");
    let Ok(text) = fs::read_to_string(&path) else { return Ok(()) };
    let previous: Value = serde_json::from_str(&text).wrap_err_with(|| format!("parse {}", path.display()))?;
    if let Some(p) = previous.get("runner").and_then(Value::as_str)
        && p != tag
    {
        bail!("{} was started with runner {p:?}, not {tag:?}: one scroll, one runner", out.display());
    }

    for key in RESUME_FIELDS {
        let (Some(was), Some(now)) = (previous.get(key), fields.get(key)) else { continue };
        ensure!(was == now, "{} was started with {key} {was}, not {now}: use a new --out", out.display());
    }
    Ok(())
}

/// What the run reads, by content, so a resume can't mix strips made from other files under the same names
struct Inputs {
    afv_uuid: String,
    afv_bytes: u64,
    umbilicus_sha256: String,
    weights_sha256: String,
}

impl Inputs {
    fn read(args: &Args, weights: &Weights) -> Result<Self> {
        // an AFV runs to tens of GB, so its uuid and size stand for its content
        let afv_uuid = Afv::open(&args.afv)?.uuid()?;
        let afv_bytes = fs::metadata(&args.afv).wrap_err_with(|| format!("stat {}", args.afv.display()))?.len();
        let umbilicus = fs::read(&args.umbilicus).wrap_err_with(|| format!("read {}", args.umbilicus.display()))?;
        Ok(Self { afv_uuid, afv_bytes, umbilicus_sha256: sha256(&umbilicus), weights_sha256: weights.sha256.clone() })
    }
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// run.json's fields: the Python runner's, with this runner's values, then the inputs' identities
fn run_fields(args: &Args, shards: usize, tag: &str, inputs: &Inputs) -> Map<String, Value> {
    let exe = std::env::current_exe().map(|p| p.display().to_string()).unwrap_or_default();
    let fields = json!({
        "checkpoint": CHECKPOINT, "step": STEP,
        "afv": args.afv_ref.clone().unwrap_or_else(|| args.afv.display().to_string()),
        "min_length_mm": args.min_length_mm, "voxel_size_um": args.voxel_size_um, "decision_step": args.step,
        "blend": args.blend, "trim_patch_voxels": args.trim, "shards": shards, "units": "patch voxels",
        "trace_per_patch": TRACE_PER_PATCH, "crops": "gpu", "batch": args.batch, "across_fibers": true,
        "compile": Value::Null, "runner": tag, "runner_script": exe,
        "precision": "native: strip CNN bf16 (fp32 accumulate and norms), head bf16 with fp32 norms and residuals, frame model fp32, crops fp32",
        "runner_commit": format!("fibrs {}", env!("CARGO_PKG_VERSION")), "workers": args.workers, "ct": args.ct,
        "ct_level": args.ct_level, "ct_grid_scale": args.ct_grid_scale, "grid_scale": args.grid_scale,
        "perturb": args.perturb, "weights": args.weights.display().to_string(),
        "afv_uuid": inputs.afv_uuid, "afv_bytes": inputs.afv_bytes, "umbilicus_sha256": inputs.umbilicus_sha256,
        "weights_sha256": inputs.weights_sha256,
    });
    fields.as_object().cloned().unwrap_or_default()
}

/// Writes run.json (the fields, then the provenance, then the tag) when OUT has none, else refuses a run that
/// doesn't match the one it records. The file appears whole or not at all, so of shards starting together one writes
/// it and the rest check against it
fn claim_run_json(out: &Path, fields: &Map<String, Value>, provenance: &Map<String, Value>, tag: &str) -> Result<()> {
    let path = out.join("run.json");
    let tmp = out.join(format!("run.json.{}.tmp", std::process::id()));
    fs::write(&tmp, run_json(fields, provenance, tag)).wrap_err_with(|| format!("write {}", tmp.display()))?;
    let linked = fs::hard_link(&tmp, &path);
    fs::remove_file(&tmp).wrap_err_with(|| format!("remove {}", tmp.display()))?;
    match linked {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == ErrorKind::AlreadyExists => check_resume(out, tag, fields),
        Err(e) => Err(e).wrap_err_with(|| format!("create {}", path.display())),
    }
}

fn default_tag() -> String {
    format!("fibrs-strips-{}", env!("CARGO_PKG_VERSION"))
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

/// One CPU worker: takes fibres until none are left or the run stops
#[allow(clippy::too_many_arguments)]
fn worker(
    args: &Args,
    rows: &[FibreRow],
    next: &AtomicUsize,
    stop: &AtomicBool,
    geometry: &Geometry,
    settings: &Settings,
    work: &Sender<Work>,
    done: &Sender<Done>,
) {
    let afv = match Afv::open(&args.afv) {
        Ok(afv) => afv,
        Err(e) => {
            let _ = done.send(Err(e.wrap_err("open the AFV")));
            return;
        }
    };
    let mut space = Workspace::default();
    while !stop.load(Ordering::Relaxed) {
        let i = next.fetch_add(1, Ordering::Relaxed);
        let Some(row) = rows.get(i) else { return };
        let result =
            fibre(args, &afv, row, geometry, settings, &mut space, work).wrap_err_with(|| format!("fibre {}", row.id));
        if done.send(result).is_err() {
            return;
        }
    }
}

/// One fibre end to end
fn fibre(
    args: &Args,
    afv: &Afv,
    row: &FibreRow,
    geometry: &Geometry,
    settings: &Settings,
    space: &mut Workspace,
    work: &Sender<Work>,
) -> Result<Outcome> {
    let fibre = afv.fibre(row, args.grid_scale)?;
    let Some(arcs) = fibre.arcs(args.step) else { return Ok(Outcome::Indexed(None, 0)) };
    let plans = match geometry.plan(&fibre, &arcs, space) {
        Ok(plans) => plans,
        Err(e) => {
            if let Some(Uncached(chunks)) = e.downcast_ref::<Uncached>() {
                return Ok(Outcome::Uncached(chunks.clone()));
            }
            let Some(reason) = e.downcast_ref::<SeedHeadingError>() else { return Err(e) };
            return Ok(Outcome::Indexed(Some(Record::failed(row, settings, &reason.to_string())), 0));
        }
    };

    let requests: Vec<FrameRequest> = plans.iter().map(|p| p.request.clone()).collect();
    let (tx, rx) = channel();
    work.send(Work::Frames(requests, tx)).map_err(|_| eyre!("the GPU engine stopped"))?;
    let frames = match rx.recv().map_err(|_| eyre!("the GPU engine stopped"))? {
        Ok(frames) => frames,
        Err(LaunchError::Uncached(chunks)) => return Ok(Outcome::Uncached(chunks)),
        Err(LaunchError::Failed(e)) => bail!("frame model: {e}"),
    };
    ensure!(frames.len() == plans.len(), "{} frame results for {} decisions", frames.len(), plans.len());

    let requests: Vec<StripRequest> = plans.iter().zip(&frames).map(|(p, f)| p.finish(geometry, &fibre, f)).collect();
    let final_frames: Vec<_> = requests.iter().map(|r| r.frame).collect();
    let positions: Vec<_> = requests.iter().map(|r| r.pos).collect();
    let (tx, rx) = channel();
    work.send(Work::Strips(requests, tx)).map_err(|_| eyre!("the GPU engine stopped"))?;
    let results = match rx.recv().map_err(|_| eyre!("the GPU engine stopped"))? {
        Ok(results) => results,
        Err(LaunchError::Uncached(chunks)) => return Ok(Outcome::Uncached(chunks)),
        Err(LaunchError::Failed(e)) => bail!("strip model: {e}"),
    };
    ensure!(results.len() == plans.len(), "{} strip results for {} decisions", results.len(), plans.len());

    let heights: Vec<_> = results.iter().map(|r| r.heights).collect();
    let log_scale: Vec<_> = results.iter().map(|r| r.log_scale).collect();
    let strip = stitched_strip(
        &heights,
        &log_scale,
        &final_frames,
        &positions,
        &arcs,
        fibre.length(),
        args.blend,
        args.trim * TRACE_PER_PATCH,
        TRACE_PER_PATCH,
    );
    let record = settings.write_fibre(&fibre, &arcs, &strip, &heights, &log_scale, &final_frames, &positions)?;
    Ok(Outcome::Indexed(Some(record), arcs.len()))
}

/// What the index writer saw: the decisions run, and the fibres deferred with the uncached chunks they need
struct Written {
    decisions: usize,
    deferred: usize,
    uncached: BTreeSet<[usize; 3]>,
}

/// Writes each finished fibre's index line and the progress lines until every fibre is done or deferred
fn write_index(
    args: &Args,
    k: usize,
    tag: &str,
    total: usize,
    done: std::sync::mpsc::Receiver<Done>,
    started: Instant,
) -> Result<Written> {
    let path = args.out.join("index").join(format!("shard_{k}.jsonl"));
    let mut log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .wrap_err_with(|| format!("open {}", path.display()))?;
    let (mut count, mut decisions, mut deferred) = (0usize, 0usize, 0usize);
    let mut uncached = BTreeSet::new();
    let mut last_record = Instant::now();
    let mut last_print = Instant::now();
    for result in done {
        count += 1;
        let record = match result? {
            Outcome::Indexed(record, n) => {
                decisions += n;
                record
            }
            Outcome::Uncached(chunks) => {
                deferred += 1;
                uncached.extend(chunks);
                None
            }
        };
        // a --no-write run indexes nothing, so a later run in the same OUT still writes those strips
        if let Some(record) = record.filter(|_| !args.no_write) {
            let seconds = last_record.elapsed().as_secs_f64();
            if record.failed.is_none() {
                last_record = Instant::now();
            }
            let mut line = record.line(seconds, tag);
            line.push('\n');
            log.write_all(line.as_bytes())
                .and_then(|()| log.flush())
                .wrap_err_with(|| format!("write {}", path.display()))?;
        }
        if last_print.elapsed().as_secs_f64() >= 30.0 || count == total {
            last_print = Instant::now();
            let seconds = started.elapsed().as_secs_f64();
            println!(
                "{}",
                json!({"shard": k, "done": count, "todo": total, "decisions": decisions,
                       "dps": round2(decisions as f64 / seconds.max(1e-9))})
            );
        }
    }
    ensure!(count == total, "{count} of {total} fibres finished");
    Ok(Written { decisions, deferred, uncached })
}

/// Lists `uncached` (chunk-grid `z/y/x`, one per line, the cache's own layout) in `<out>/logs/`, where the next pass's
/// fetch reads them; a run that deferred nothing removes a stale list
fn write_uncached(args: &Args, k: usize, n: usize, written: &Written) -> Result<PathBuf> {
    let name = if n == 1 { "uncached_chunks.txt".to_string() } else { format!("uncached_chunks_shard{k}.txt") };
    let path = args.out.join("logs").join(name);
    if written.deferred == 0 {
        match fs::remove_file(&path) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                return Err(e).wrap_err_with(|| format!("remove {}", path.display()));
            }
            _ => return Ok(path),
        }
    }
    fs::create_dir_all(args.out.join("logs")).wrap_err_with(|| format!("create {}", args.out.display()))?;
    let text: String = written.uncached.iter().map(|[z, y, x]| format!("{z}/{y}/{x}\n")).collect();
    fs::write(&path, text).wrap_err_with(|| format!("write {}", path.display()))?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use clap::Parser;
    use serde_json::{Map, Value, json};

    use super::{Afv, Args, Umbilicus, claim_run_json, indexed, provenance, validate};

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("fibrs-strips-runner-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("index")).unwrap();
        dir
    }

    fn args(extra: &[&str]) -> Args {
        let base =
            ["run", "--afv", "a.afv", "--out", "o", "--voxel-size-um", "9.362", "--weights", "w", "--ct", "s3://c"];
        let rest = ["--cache-dir", "c", "--umbilicus", "u.json"];
        Args::try_parse_from(base.iter().chain(&rest).chain(extra)).unwrap()
    }

    /// A write cut short leaves a torn last line. The shard's own index drops it for good, so that fibre reruns and
    /// the next append starts on a line of its own; another shard's index, which may be mid-write, is left as is
    #[test]
    fn torn_index_line_is_dropped_from_own_shard_only() {
        let dir = scratch("torn");
        let own = dir.join("index").join("shard_0.jsonl");
        let other = dir.join("index").join("shard_1.jsonl");
        fs::write(&own, "{\"fiber\": 1}\n{\"fiber\": 2}\n{\"fib").unwrap();
        fs::write(&other, "{\"fiber\": 3}\n{\"fiber\": 4").unwrap();

        let done = indexed(&dir.join("index"), 0).unwrap();
        assert_eq!(done, [1, 2, 3].into_iter().collect());
        assert_eq!(fs::read_to_string(&own).unwrap(), "{\"fiber\": 1}\n{\"fiber\": 2}\n");
        assert_eq!(fs::read_to_string(&other).unwrap(), "{\"fiber\": 3}\n{\"fiber\": 4");
        let _ = fs::remove_dir_all(&dir);
    }

    /// The first shard to start writes run.json; every later one, any shard, resumes only with the same tag and
    /// the same output-affecting settings and inputs, and leaves the file as it was
    #[test]
    fn run_json_is_claimed_once_and_checked_by_every_shard() {
        let dir = scratch("claim");
        let fields = |level: u32, umbilicus: &str| -> Map<String, Value> {
            json!({"ct": "s3://scroll.zarr", "ct_level": level, "umbilicus_sha256": umbilicus, "workers": 16})
                .as_object()
                .cloned()
                .unwrap()
        };
        let run_json = || fs::read_to_string(dir.join("run.json")).unwrap();

        claim_run_json(&dir, &fields(0, "aa"), &Map::new(), "tag-a").unwrap();
        let first = run_json();
        let mut more_workers = fields(0, "aa");
        more_workers.insert("workers".into(), json!(8));
        claim_run_json(&dir, &more_workers, &Map::new(), "tag-a").unwrap();
        assert_eq!(run_json(), first);

        let level = claim_run_json(&dir, &fields(1, "aa"), &Map::new(), "tag-a").unwrap_err();
        assert!(format!("{level}").contains("ct_level 0, not 1"), "{level}");
        let umbilicus = claim_run_json(&dir, &fields(0, "bb"), &Map::new(), "tag-a").unwrap_err();
        assert!(format!("{umbilicus}").contains("umbilicus_sha256"), "{umbilicus}");
        let tag = claim_run_json(&dir, &fields(0, "aa"), &Map::new(), "tag-b").unwrap_err();
        assert!(format!("{tag}").contains("one scroll, one runner"), "{tag}");
        assert_eq!(run_json(), first);
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 2, "only index/ and run.json, no temporary file");
        let _ = fs::remove_dir_all(&dir);
    }

    /// Provenance may describe the run (production sets runner_commit, precision and weights) but not change what
    /// a resume compares
    #[test]
    fn provenance_cannot_set_resume_fields() {
        let ok = provenance(r#"{"runner": "r", "runner_commit": "c", "precision": "p", "weights": "w"}"#).unwrap();
        assert_eq!(ok.len(), 4);
        let err = provenance(r#"{"runner": "r", "decision_step": 4}"#).unwrap_err();
        assert!(format!("{err}").contains("decision_step"), "{err}");
        assert!(provenance("[1]").is_err());
    }

    /// The step must be whole strip rows, and the grid two CT voxels per trace voxel (PHerc0211's 1:2 and
    /// PHercParis4's 4:8)
    #[test]
    fn settings_the_runner_cannot_run_are_refused() {
        validate(&args(&[])).unwrap();
        validate(&args(&["--ct-grid-scale", "4", "--grid-scale", "8"])).unwrap();
        validate(&args(&["--step", "6"])).unwrap();

        let step = validate(&args(&["--step", "7"])).unwrap_err();
        assert!(format!("{step}").contains("row spacing, 2"), "{step}");
        let grid = validate(&args(&["--ct-grid-scale", "1", "--grid-scale", "4"])).unwrap_err();
        assert!(format!("{grid}").contains("two CT voxels per trace voxel"), "{grid}");
        assert!(validate(&args(&["--step", "0"])).is_err());
        assert!(validate(&args(&["--blend=-1"])).is_err());
    }

    fn afv(path: &Path) {
        let db = rusqlite::Connection::open(path).unwrap();
        db.execute_batch(
            "create table metadata(key text, value text);
             insert into metadata values ('uuid', '\"836fb26c-a576-4540-ab75-df137bbc0e32\"');
             create table fibers(id integer, family text, length real, min_x real, max_x real, min_y real, max_y real,
                                 min_z real, max_z real);
             insert into fibers values (1, 'H', 900, 0, 1, 0, 1, 0, 1), (2, 'V', 900, 0, 1, 0, 1, 0, 1);",
        )
        .unwrap();
    }

    /// The AFV's identity is its uuid; ids are taken in the order given, and a repeated id is refused
    #[test]
    fn afv_uuid_and_repeated_ids() {
        let dir = scratch("afv");
        let path = dir.join("fibers.afv");
        afv(&path);
        let afv = Afv::open(&path).unwrap();

        assert_eq!(afv.uuid().unwrap(), "836fb26c-a576-4540-ab75-df137bbc0e32");
        let rows = afv.select(2.5, 9.362, Some(&[2, 1])).unwrap();
        assert_eq!(rows.iter().map(|r| r.id).collect::<Vec<_>>(), [2, 1]);
        let err = afv.select(2.5, 9.362, Some(&[1, 2, 1])).unwrap_err();
        assert!(format!("{err}").contains("more than once: [1]"), "{err}");
        let _ = fs::remove_dir_all(&dir);
    }

    /// An umbilicus without control points is refused at load, not met as a panic in a worker
    #[test]
    fn empty_umbilicus_is_refused() {
        let dir = scratch("umbilicus");
        let path = dir.join("umbilicus.json");
        fs::write(&path, r#"{"control_points": []}"#).unwrap();
        let err = Umbilicus::read(&path).unwrap_err();
        assert!(format!("{err}").contains("control_points is empty"), "{err}");
        fs::write(&path, r#"{"control_points": [{"x": 1, "y": 2, "z": 3}]}"#).unwrap();
        Umbilicus::read(&path).unwrap();
        let _ = fs::remove_dir_all(&dir);
    }
}
