//! `fibrs strips run`: the native strip runner, with `strips_fast.py`'s command line and outputs
//!
//! Fibres are selected and sharded as `strips_run.py` does (Morton order, contiguous shards, `--exclude-ids-file`),
//! and a shard resumes from every shard's index. CPU workers each take the next fibre: load it, plan its decisions
//! (CT seed headings included), hand the frame requests to the GPU engine, finish the decisions from the frame
//! model's output, hand the strip requests to the engine, then stitch and write the strip. The engine thread batches
//! both models' work across the fibres in flight. The main thread writes the index: one line per fibre as it
//! finishes, its `seconds` the time since the shard's previous line (fibres overlap, so the column sums to the
//! shard's wall time, as `strips_fast.py`'s does)

use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{Sender, channel};
use std::time::Instant;

use clap::Parser;
use color_eyre::eyre::{Result, WrapErr, bail, ensure, eyre};
use serde_json::{Map, Value, json};

use super::ct::Volume;
use super::engine::{EngineConfig, Work, serve};
use super::fibre::{Afv, FibreRow};
use super::geom::{Geometry, Umbilicus};
use super::gpu::Weights;
use super::out::{Record, Settings, run_json};
use super::seed::{SeedHeadingError, Workspace};
use super::stitch::stitched_strip;
use super::types::{FrameRequest, StripRequest};

/// The checkpoint the weights export is pinned to (its sha256 is checked at load), as trace.json records it
const CHECKPOINT: &str = "hf://scrollprize/fiber_strip_small_v1@7bd163a22c839ff312879ae1916195d0c9729c0e/last.pt";
/// Its training step
const STEP: i64 = 15000;
/// Trace voxels per patch voxel of that checkpoint
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
    /// The strip model's weights export (`export_weights.py`)
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
    /// Run and stitch, write no strips
    #[arg(long)]
    no_write: bool,
    /// Also write Check A's per-fibre npz here
    #[arg(long)]
    npz: Option<PathBuf>,
    /// The perturbation the weights export carries (negative controls); must match its metadata
    #[arg(long, default_value_t = 0.0)]
    perturb: f64,
    /// JSON object recorded in run.json; its "runner" is the run's tag, on every index line
    #[arg(long, default_value = "{}")]
    provenance: String,
}

/// What a worker hands the index writer for one fibre: its record (None for a fibre too short to run) and its
/// decisions
type Done = Result<(Option<Record>, usize)>;

/// Runs the shard; the progress lines go to stdout as `strips_fast.py` prints them
pub fn run(args: &Args) -> Result<()> {
    ensure!(args.shard.len() == 2 && args.shard[1] > 0 && args.shard[0] < args.shard[1], "--shard K N with K < N");
    ensure!(args.workers > 0 && args.batch > 0, "--workers and --batch above 0");
    let (k, n) = (args.shard[0], args.shard[1]);
    let input_scale = args.grid_scale / args.ct_grid_scale;
    ensure!(input_scale.fract() == 0.0, "CT sampling requires a whole number of CT voxels per trace voxel");
    let provenance: Map<String, Value> = match serde_json::from_str(&args.provenance) {
        Ok(Value::Object(m)) => m,
        _ => bail!("--provenance: a JSON object"),
    };
    let tag = provenance.get("runner").and_then(Value::as_str).map_or_else(default_tag, str::to_string);

    let t0 = Instant::now();
    let weights = Weights::open(&args.weights)?;
    let carried: f64 = weights.metadata.get("perturb").map_or(Ok(0.0), |p| p.parse()).wrap_err("weights perturb")?;
    ensure!(carried == args.perturb, "--perturb {} but the weights carry perturb {carried}", args.perturb);

    let todo = select(args, k, n)?;
    fs::create_dir_all(args.out.join("index")).wrap_err_with(|| format!("create {}", args.out.display()))?;
    check_tag(&args.out, &tag)?;
    if k == 0 {
        write_run_json(args, n, &provenance, &tag)?;
    }

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
    let decisions = std::thread::scope(|scope| -> Result<usize> {
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
    let decisions = decisions?;

    let wall = started.elapsed().as_secs_f64();
    println!(
        "{}",
        json!({"shard": k, "finished": true, "decisions": decisions, "wall_s": round2(wall),
               "dps": round2(decisions as f64 / wall.max(1e-9)), "gpu_busy_s": round2(counts.busy),
               "frame_launches": counts.frame_launches, "strip_launches": counts.strip_launches,
               "chunks": {"needed": counts.cache.needed, "uploads": counts.cache.uploads, "fills": counts.cache.fills,
                          "evictions": counts.cache.evictions, "most_per_launch": counts.cache.most,
                          "prefetched": counts.cache.prefetched}})
    );
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
    let done = indexed(&args.out.join("index"))?;
    let total = chosen.len();
    Ok(Todo { rows: chosen.into_iter().filter(|r| !done.contains(&r.id)).collect(), total })
}

fn read_ids(path: &Path) -> Result<Vec<i64>> {
    let text = fs::read_to_string(path).wrap_err_with(|| format!("read {}", path.display()))?;
    text.split_whitespace().map(|t| t.parse().wrap_err_with(|| format!("{}: id {t:?}", path.display()))).collect()
}

/// Every fibre id in `index/shard_*.jsonl`
fn indexed(dir: &Path) -> Result<HashSet<i64>> {
    let mut done = HashSet::new();
    let Ok(entries) = fs::read_dir(dir) else { return Ok(done) };
    for entry in entries {
        let path = entry?.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default();
        if !(name.starts_with("shard_") && name.ends_with(".jsonl")) {
            continue;
        }
        for line in fs::read_to_string(&path)?.lines().filter(|l| !l.trim().is_empty()) {
            let record: Value = serde_json::from_str(line).wrap_err_with(|| format!("{}: {line}", path.display()))?;
            done.insert(record["fiber"].as_i64().ok_or_else(|| eyre!("{}: no fiber in {line}", path.display()))?);
        }
    }
    Ok(done)
}

/// Refuses an OUT begun under another runner tag: one scroll, one runner
fn check_tag(out: &Path, tag: &str) -> Result<()> {
    let path = out.join("run.json");
    let Ok(text) = fs::read_to_string(&path) else { return Ok(()) };
    let previous: Value = serde_json::from_str(&text).wrap_err_with(|| format!("parse {}", path.display()))?;
    match previous.get("runner").and_then(Value::as_str) {
        Some(p) if p != tag => {
            bail!("{} was started with runner {p:?}, not {tag:?}: one scroll, one runner", out.display())
        }
        _ => Ok(()),
    }
}

/// run.json, `strips_fast.py`'s fields with this runner's values, then the provenance, then the tag
fn write_run_json(args: &Args, shards: usize, provenance: &Map<String, Value>, tag: &str) -> Result<()> {
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
    });
    let fields = fields.as_object().cloned().unwrap_or_default();
    let path = args.out.join("run.json");
    fs::write(&path, run_json(&fields, provenance, tag)).wrap_err_with(|| format!("write {}", path.display()))
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
) -> Result<(Option<Record>, usize)> {
    let fibre = afv.fibre(row, args.grid_scale)?;
    let Some(arcs) = fibre.arcs(args.step) else { return Ok((None, 0)) };
    let plans = match geometry.plan(&fibre, &arcs, space) {
        Ok(plans) => plans,
        Err(e) => {
            let Some(reason) = e.downcast_ref::<SeedHeadingError>() else { return Err(e) };
            return Ok((Some(Record::failed(row, settings, &reason.to_string())), 0));
        }
    };

    let requests: Vec<FrameRequest> = plans.iter().map(|p| p.request.clone()).collect();
    let (tx, rx) = channel();
    work.send(Work::Frames(requests, tx)).map_err(|_| eyre!("the GPU engine stopped"))?;
    let frames = rx.recv().map_err(|_| eyre!("the GPU engine stopped"))?.map_err(|e| eyre!("frame model: {e}"))?;
    ensure!(frames.len() == plans.len(), "{} frame results for {} decisions", frames.len(), plans.len());

    let requests: Vec<StripRequest> = plans.iter().zip(&frames).map(|(p, f)| p.finish(geometry, &fibre, f)).collect();
    let final_frames: Vec<_> = requests.iter().map(|r| r.frame).collect();
    let positions: Vec<_> = requests.iter().map(|r| r.pos).collect();
    let (tx, rx) = channel();
    work.send(Work::Strips(requests, tx)).map_err(|_| eyre!("the GPU engine stopped"))?;
    let results = rx.recv().map_err(|_| eyre!("the GPU engine stopped"))?.map_err(|e| eyre!("strip model: {e}"))?;
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
    Ok((Some(record), arcs.len()))
}

/// Writes each finished fibre's index line and the progress lines until every fibre is done; the decisions run
fn write_index(
    args: &Args,
    k: usize,
    tag: &str,
    total: usize,
    done: std::sync::mpsc::Receiver<Done>,
    started: Instant,
) -> Result<usize> {
    let path = args.out.join("index").join(format!("shard_{k}.jsonl"));
    let mut log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .wrap_err_with(|| format!("open {}", path.display()))?;
    let (mut count, mut decisions) = (0usize, 0usize);
    let mut last_record = Instant::now();
    let mut last_print = Instant::now();
    for result in done {
        let (record, n) = result?;
        count += 1;
        decisions += n;
        if let Some(record) = record {
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
    Ok(decisions)
}
