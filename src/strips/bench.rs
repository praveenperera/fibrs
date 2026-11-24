//! `fibrs strips bench-cpu`: the CPU floor of the native runner. Every CPU stage of a fibre (load, plan with its CT
//! seed headings, finish, stitch and write) runs on fixed stand-in model outputs, one fibre per task on rayon pools
//! of the given sizes, and each stage's time is reported per fibre and per decision with the implied decisions per
//! second at 40 decisions per fibre. The structure tensor is also timed alone. `/proc/loadavg` is recorded with
//! every pool, since the host is shared

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use color_eyre::eyre::{Result, WrapErr, bail, ensure};
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use serde_json::{Value, json};

use super::ct::Volume;
use super::fibre::{Afv, FibreRow};
use super::geom::{Geometry, Umbilicus};
use super::np::{M3, V3};
use super::out::{Record, Settings, run_json};
use super::seed::{SIDE, SeedHeadingError, Workspace, oriented_seed_heading};
use super::stitch::stitched_strip;
use super::types::{FrameResult, STRIP_COLS, STRIP_ROWS, StripRequest};

/// `fibrs strips bench-cpu`
#[derive(Parser, Debug)]
#[command(name = "fibrs strips bench-cpu")]
pub struct Args {
    /// The AFV
    #[arg(long)]
    afv: PathBuf,
    /// Fibre ids, whitespace-separated
    #[arg(long)]
    ids: PathBuf,
    /// Every Nth id of --ids (a sample that keeps the file's length mix)
    #[arg(long, default_value_t = 1)]
    every: usize,
    /// At most this many fibres
    #[arg(long)]
    limit: Option<usize>,
    /// The CT zarr, as its disk cache names it
    #[arg(long)]
    ct: String,
    #[arg(long, default_value_t = 0)]
    ct_level: u32,
    /// vc_reader's disk cache holding the fibres' CT chunks
    #[arg(long)]
    cache_dir: PathBuf,
    /// Base voxels per CT voxel
    #[arg(long, default_value_t = 1.0)]
    ct_grid_scale: f64,
    /// Base voxels per trace voxel
    #[arg(long, default_value_t = 2.0)]
    grid_scale: f64,
    #[arg(long)]
    voxel_size_um: f64,
    #[arg(long)]
    umbilicus: PathBuf,
    /// Trace voxels per patch voxel (the strip checkpoint's)
    #[arg(long, default_value_t = 0.5)]
    trace_per_patch: f64,
    #[arg(long, default_value_t = 8.0)]
    step: f64,
    #[arg(long, default_value_t = 16.0)]
    blend: f64,
    /// Patch voxels
    #[arg(long, default_value_t = 75.0)]
    trim: f64,
    /// Strips are written under OUT/strips, as the runner writes them
    #[arg(long)]
    out: PathBuf,
    /// Also write Check A's npz (the runner writes none in production)
    #[arg(long)]
    npz: bool,
    /// Pool sizes, comma-separated
    #[arg(long, value_delimiter = ',', default_value = "1,4,8,16")]
    threads: Vec<usize>,
    /// Structure tensors to time alone
    #[arg(long, default_value_t = 200)]
    tensors: usize,
    /// Results as JSON
    #[arg(long)]
    json: Option<PathBuf>,
}

/// Stage times of one fibre, seconds
#[derive(Clone, Copy, Default)]
struct Times {
    load: f64,
    plan: f64,
    finish: f64,
    stitch: f64,
    write: f64,
}

/// One fibre's run
#[derive(Clone, Copy)]
struct Fibre {
    decisions: usize,
    times: Times,
    /// Failed with a SeedHeadingError: its plan time counts, no decisions
    failed: bool,
    kink_repairs: usize,
}

/// Deterministic stand-ins for the models' outputs: a heading along the request frame's f with a small tilt, a
/// normal along its u, and strip heights and log scales in their usual ranges
struct Outputs(u64);

impl Outputs {
    fn next(&mut self) -> f32 {
        // xorshift64*: uniform in [0, 1)
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32 / (1u64 << 24) as f32
    }

    fn frame(&mut self) -> FrameResult {
        let tilt = [0.3 * (self.next() - 0.5), 0.3 * (self.next() - 0.5)];
        let size = (tilt[0] * tilt[0] + tilt[1] * tilt[1] + 1.0).sqrt();
        FrameResult { heading: [tilt[0] / size, tilt[1] / size, 1.0 / size], normal: [1.0, 0.0, 0.0] }
    }

    fn grid(&mut self, lo: f32, hi: f32) -> [[f32; STRIP_COLS]; STRIP_ROWS] {
        std::array::from_fn(|_| std::array::from_fn(|_| lo + (hi - lo) * self.next()))
    }
}

/// A pool's run over the fibres
struct Pool {
    threads: usize,
    wall: f64,
    loadavg: String,
    fibres: Vec<Fibre>,
    /// Fibres that stopped on another error (a CT chunk the cache lacks): how many, and the first
    errors: (usize, Option<String>),
}

/// The 1, 5 and 15 minute load averages
fn loadavg() -> String {
    let text = fs::read_to_string("/proc/loadavg").unwrap_or_default();
    text.split_whitespace().take(3).collect::<Vec<_>>().join(" ")
}

/// Runs every CPU stage of one fibre
fn run_fibre(
    afv: &Afv,
    row: &FibreRow,
    args: &Args,
    geometry: &Geometry,
    settings: &Settings,
    work: &mut Workspace,
) -> Result<Option<Fibre>> {
    let mut times = Times::default();
    let t = Instant::now();
    let fibre = afv.fibre(row, args.grid_scale)?;
    times.load = t.elapsed().as_secs_f64();
    let Some(arcs) = fibre.arcs(args.step) else { return Ok(None) };

    let t = Instant::now();
    let plans = match geometry.plan(&fibre, &arcs, work) {
        Ok(plans) => plans,
        Err(e) => {
            let Some(reason) = e.downcast_ref::<SeedHeadingError>() else { return Err(e) };
            let _ = Record::failed(row, settings, &reason.to_string()).line(0.0, "bench");
            times.plan = t.elapsed().as_secs_f64();
            return Ok(Some(Fibre { decisions: 0, times, failed: true, kink_repairs: fibre.kink_repairs }));
        }
    };
    times.plan = t.elapsed().as_secs_f64();

    let mut outputs = Outputs(row.id as u64 | 1);
    let frames: Vec<FrameResult> = plans.iter().map(|_| outputs.frame()).collect();
    let heights: Vec<_> = plans.iter().map(|_| outputs.grid(-8.0, 8.0)).collect();
    let log_scale: Vec<_> = plans.iter().map(|_| outputs.grid(-0.7, 1.3)).collect();
    let t = Instant::now();
    let requests: Vec<StripRequest> = plans.iter().zip(&frames).map(|(p, f)| p.finish(geometry, &fibre, f)).collect();
    times.finish = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let frames: Vec<M3> = requests.iter().map(|r| r.frame).collect();
    let positions: Vec<V3> = requests.iter().map(|r| r.pos).collect();
    let strip = stitched_strip(
        &heights,
        &log_scale,
        &frames,
        &positions,
        &arcs,
        fibre.length(),
        args.blend,
        args.trim * args.trace_per_patch,
        args.trace_per_patch,
    );
    times.stitch = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let record = settings.write_fibre(&fibre, &arcs, &strip, &heights, &log_scale, &frames, &positions)?;
    let _ = record.line(times.load + times.plan + times.finish + times.stitch, "bench");
    times.write = t.elapsed().as_secs_f64();
    Ok(Some(Fibre { decisions: arcs.len(), times, failed: false, kink_repairs: fibre.kink_repairs }))
}

/// Least squares `a + b * decisions` of one stage's seconds over the fibres that ran
fn fit(fibres: &[Fibre], stage: impl Fn(&Times) -> f64) -> (f64, f64) {
    let ran: Vec<(f64, f64)> =
        fibres.iter().filter(|f| !f.failed).map(|f| (f.decisions as f64, stage(&f.times))).collect();
    let n = ran.len() as f64;
    let (mx, my) = (ran.iter().map(|p| p.0).sum::<f64>() / n, ran.iter().map(|p| p.1).sum::<f64>() / n);
    let sxx: f64 = ran.iter().map(|p| (p.0 - mx) * (p.0 - mx)).sum();
    let sxy: f64 = ran.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    let b = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    (my - b * mx, b)
}

type Stage = (&'static str, fn(&Times) -> f64);

const STAGES: [Stage; 6] = [
    ("load", |t| t.load),
    ("plan", |t| t.plan),
    ("finish", |t| t.finish),
    ("stitch", |t| t.stitch),
    ("write", |t| t.write),
    ("total", |t| t.load + t.plan + t.finish + t.stitch + t.write),
];

/// The structure tensor alone, ms per call, at the first `n` fibres' seed (their first vertex) and the CT read
/// with it
fn time_tensors(afv: &Afv, rows: &[FibreRow], args: &Args, ct: &Volume, input_scale: f64) -> Result<Value> {
    let mut work = Workspace::default();
    let (mut read, mut tensor, mut heading, mut count) = (0.0, 0.0, 0.0, 0usize);
    let shape = ct.shape();
    for row in rows.iter().take(args.tensors) {
        let fibre = afv.fibre(row, args.grid_scale)?;
        let pos = fibre.points[0];
        let centre = [pos[2] * input_scale, pos[1] * input_scale, pos[0] * input_scale];
        let start: [i64; 3] = std::array::from_fn(|a| centre[a].floor() as i64 - super::seed::RADIUS);
        if (0..3).any(|a| start[a] < 0 || start[a] + SIDE as i64 > shape[a] as i64) {
            continue;
        }
        let local = std::array::from_fn(|a| centre[a] - start[a] as f64);
        let t = Instant::now();
        ct.read_region(start.map(|v| v as usize), [SIDE; 3], &mut work.cube)?;
        read += t.elapsed().as_secs_f64();
        let t = Instant::now();
        std::hint::black_box(work.structure_tensor(local));
        tensor += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let direction = super::np::sub(fibre.points[1], fibre.points[0]);
        let _ = std::hint::black_box(oriented_seed_heading(ct, input_scale, pos, &row.family, direction, &mut work));
        heading += t.elapsed().as_secs_f64();
        count += 1;
    }
    let ms = |s: f64| 1e3 * s / count.max(1) as f64;
    Ok(json!({"calls": count, "read_ms": ms(read), "tensor_ms": ms(tensor), "seed_heading_ms": ms(heading)}))
}

pub fn main(argv: &[std::ffi::OsString]) -> Result<()> {
    let args = Args::try_parse_from(std::iter::once("bench-cpu".into()).chain(argv.iter().cloned()))?;
    let input_scale = args.grid_scale / args.ct_grid_scale;
    ensure!(input_scale.fract() == 0.0, "CT sampling requires a whole number of CT voxels per trace voxel");
    let text = fs::read_to_string(&args.ids).wrap_err_with(|| format!("read {}", args.ids.display()))?;
    let mut ids: Vec<i64> = text.split_whitespace().map(str::parse).collect::<Result<_, _>>()?;
    ids = ids.into_iter().step_by(args.every.max(1)).collect();
    ids.truncate(args.limit.unwrap_or(usize::MAX));
    let afv = Afv::open(&args.afv)?;
    let rows = afv.select(0.0, args.voxel_size_um, Some(&ids))?;
    let ct = Volume::open(&args.cache_dir, &args.ct, args.ct_level)?;
    let umbilicus = Umbilicus::read(&args.umbilicus)?;
    let geometry = Geometry { ct: &ct, input_scale, trace_per_patch: args.trace_per_patch, umbilicus: &umbilicus };
    let settings = Settings {
        out: Some(args.out.clone()),
        npz: args.npz.then(|| args.out.join("npz")),
        checkpoint: "bench".into(),
        step: None,
        afv_ref: args.afv.display().to_string(),
        decision_step: args.step,
        blend: args.blend,
        trim: args.trim,
        voxel_size_um: args.voxel_size_um,
        trace_per_patch: args.trace_per_patch,
    };
    eprintln!("bench-cpu: {} fibres, pools {:?}", rows.len(), args.threads);
    // the run's record, as the runner writes it once
    fs::create_dir_all(&args.out).wrap_err_with(|| format!("create {}", args.out.display()))?;
    let fields = json!({
        "afv": settings.afv_ref, "voxel_size_um": args.voxel_size_um, "decision_step": args.step, "blend": args.blend,
        "trim_patch_voxels": args.trim, "units": "patch voxels", "trace_per_patch": args.trace_per_patch,
        "ct": args.ct, "ct_level": args.ct_level, "ct_grid_scale": args.ct_grid_scale, "grid_scale": args.grid_scale,
    });
    let fields = fields.as_object().cloned().unwrap_or_default();
    fs::write(args.out.join("run.json"), run_json(&fields, &serde_json::Map::new(), "fibrs-bench-cpu"))?;

    let tensors = time_tensors(&afv, &rows, &args, &ct, input_scale)?;
    eprintln!("structure tensor: {tensors}");

    // one untimed pass at the largest pool warms the page cache for the CT and the AFV
    let mut pools = Vec::new();
    let largest = args.threads.iter().copied().max().unwrap_or(1);
    for (warm, threads) in std::iter::once((true, largest)).chain(args.threads.iter().map(|&t| (false, t))) {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build()?;
        let load_before = loadavg();
        let t = Instant::now();
        let results: Vec<Result<Option<Fibre>>> = pool.install(|| {
            rows.par_iter()
                .map_init(
                    || (Afv::open(&args.afv), Workspace::default()),
                    |(afv, work), row| {
                        let afv = afv.as_ref().map_err(|e| color_eyre::eyre::eyre!("open the AFV: {e}"))?;
                        run_fibre(afv, row, &args, &geometry, &settings, work)
                            .wrap_err_with(|| format!("fibre {}", row.id))
                    },
                )
                .collect()
        });
        let wall = t.elapsed().as_secs_f64();
        let mut fibres = Vec::with_capacity(results.len());
        let mut errors = (0, None);
        for r in results {
            match r {
                Ok(Some(f)) => fibres.push(f),
                Ok(None) => {}
                Err(e) => {
                    errors.0 += 1;
                    errors.1.get_or_insert_with(|| format!("{e:#}"));
                }
            }
        }
        if fibres.is_empty() {
            bail!("no fibre ran");
        }
        let loadavg = format!("{load_before} -> {}", loadavg());
        eprintln!(
            "{} pool {threads}: {wall:.2} s, load {loadavg}, {} errors",
            if warm { "warm-up" } else { "timed" },
            errors.0
        );
        if !warm {
            pools.push(Pool { threads, wall, loadavg, fibres, errors });
        }
    }

    let report = report(&pools, &tensors, rows.len());
    println!("{}", report.1);
    if let Some(path) = &args.json {
        fs::write(path, serde_json::to_string_pretty(&report.0)?)?;
    }
    Ok(())
}

/// The results as JSON and as a Markdown table
fn report(pools: &[Pool], tensors: &Value, selected: usize) -> (Value, String) {
    let mut md = format!("structure tensor: {tensors}\n\n");
    let first = &pools[0];
    let fibres = first.fibres.len();
    let failed = first.fibres.iter().filter(|f| f.failed).count();
    let decisions: usize = first.fibres.iter().map(|f| f.decisions).sum();
    let repairs: usize = first.fibres.iter().map(|f| f.kink_repairs).sum();
    let (errors, first_error) = &first.errors;
    md.push_str(&format!(
        "{selected} selected, {fibres} ran ({failed} SeedHeadingError), {} too short, {errors} stopped on an error \
         (first: {}), {decisions} decisions, {:.1} decisions/fibre, {repairs} kinks repaired\n\n",
        selected - fibres - errors,
        first_error.as_deref().unwrap_or("none"),
        decisions as f64 / (fibres - failed).max(1) as f64
    ));
    md.push_str("| threads | load avg (before -> after) | wall s | dec/s | stage | ms/fibre | ms/decision | fit ms/fibre at 40 dec | 40-dec ceiling dec/s |\n");
    md.push_str("|---:|---|---:|---:|---|---:|---:|---:|---:|\n");
    let mut out = Vec::new();
    let single = pools.iter().find(|p| p.threads == 1).map(|p| p.wall);
    for pool in pools {
        let n = pool.fibres.len() as f64;
        let decisions: usize = pool.fibres.iter().map(|f| f.decisions).sum();
        let dps = decisions as f64 / pool.wall;
        // the pool's speedup over one thread scales the one-thread ceiling; without a one-thread pool, its threads
        let speedup = single.map_or(pool.threads as f64, |s| s / pool.wall);
        let mut stages = serde_json::Map::new();
        for (name, stage) in STAGES {
            let sum: f64 = pool.fibres.iter().map(|f| stage(&f.times)).sum();
            let (a, b) = fit(&pool.fibres, stage);
            let at40 = a + 40.0 * b;
            // the ceiling is per-thread CPU time at 40 decisions, so it is set by `total`
            let ceiling = 40.0 / at40 * speedup;
            md.push_str(&format!(
                "| {} | {} | {:.2} | {:.0} | {name} | {:.3} | {:.4} | {:.3} | {} |\n",
                pool.threads,
                pool.loadavg,
                pool.wall,
                dps,
                1e3 * sum / n,
                1e3 * sum / decisions as f64,
                1e3 * at40,
                if name == "total" { format!("{ceiling:.0}") } else { String::new() },
            ));
            stages.insert(
                name.into(),
                json!({"ms_per_fibre": 1e3 * sum / n, "ms_per_decision": 1e3 * sum / decisions as f64,
                       "fit_ms_at_40": 1e3 * at40, "fit_ms_per_fibre": 1e3 * a, "fit_ms_per_decision": 1e3 * b}),
            );
        }
        let total = fit(&pool.fibres, STAGES[5].1);
        out.push(json!({
            "threads": pool.threads, "loadavg": pool.loadavg, "wall_s": pool.wall, "decisions": decisions,
            "fibres": pool.fibres.len(), "errors": pool.errors.0, "first_error": pool.errors.1,
            "dec_per_s": dps, "speedup": speedup,
            "ceiling_dec_per_s_at_40": 40.0 / (total.0 + 40.0 * total.1) * speedup, "stages": stages,
        }));
    }
    (json!({"selected": selected, "structure_tensor": tensors, "pools": out}), md)
}
