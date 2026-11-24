//! `fibrs strips bench-model`: the native strip model timed per decision and checked against PyTorch's reference
//! on recorded decisions (`refset`), for the milestone and the gate's model check
//!
//! Timing runs each batch size on the reference's first decisions (repeated to fill the batch), warm, after the
//! plans for that size are made: the CNN alone, then the whole model (CNN and head, including the head's record
//! upload and result download). Parity runs every recorded decision once in batches of the smallest size and
//! holds the final heights and log scales to PyTorch's own bf16 spread (track-c-native.md, 11:40 PM Nov 23)

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use color_eyre::eyre::{Result, WrapErr, bail, ensure};
use serde_json::{Value, json};

use super::gpu::cnn::CropCnn;
use super::gpu::head::Head;
use super::gpu::{Device, Weights};
use super::refset::{CELLS_LEN, CROP_LEN, GRID_LEN, RefSet, Spread, spread};
use super::types::StripRequest;

/// The parity bar against PyTorch's float32 forward, from PyTorch's own bf16-vs-fp32 spread on the reference
/// (heights mean 0.0104, p99 0.059; log scales mean 0.0035, p99 0.013) with ~15% headroom
const HEIGHTS_BAR: Bar = Bar { mean: 0.012, p99: 0.065 };
const LOGS_BAR: Bar = Bar { mean: 0.004, p99: 0.015 };

struct Bar {
    mean: f64,
    p99: f64,
}

impl Bar {
    fn passes(&self, s: &Spread) -> bool {
        s.mean <= self.mean && s.p99 <= self.p99
    }
}

/// Times the native strip model and checks it against PyTorch's reference
#[derive(Parser, Debug)]
#[command(name = "fibrs strips bench-model")]
pub struct Args {
    /// The reference directory (`ref_strip.py`)
    #[arg(long = "ref")]
    reference: PathBuf,
    /// The weights export
    #[arg(long)]
    weights: PathBuf,
    /// Batch sizes to time, comma-separated
    #[arg(long, value_delimiter = ',', default_values_t = [64, 128, 256])]
    batches: Vec<usize>,
    /// Timed forwards per batch size
    #[arg(long, default_value_t = 10)]
    reps: usize,
    /// Decisions whose cells are compared with PyTorch's (each 2.4 M values)
    #[arg(long, default_value_t = 16)]
    cell_decisions: usize,
    /// Where the JSON report goes
    #[arg(long)]
    out: PathBuf,
    /// CUDA device ordinal
    #[arg(long, default_value_t = 0)]
    device: usize,
}

/// Runs the bench; the report also goes to stdout, a line per result
pub fn run(args: &Args) -> Result<()> {
    ensure!(!args.batches.is_empty() && args.batches.iter().all(|&b| b > 0), "--batches: sizes above 0");
    let refs = RefSet::open(&args.reference)?;
    let weights = Weights::open(&args.weights)?;
    let dev = Device::open(args.device)?;
    let max = *args.batches.iter().max().unwrap();
    let mut cnn = CropCnn::new(&dev, &weights, max).wrap_err("build the CNN")?;
    let head = Head::new(&dev, &weights, max).wrap_err("build the head")?;
    let requests: Vec<StripRequest> = (0..refs.n).map(|i| refs.request(i)).collect();
    let input = dev.gpu.alloc(max * CROP_LEN * 4)?;
    let mut report = json!({"decisions": refs.n, "batches": args.batches, "reps": args.reps});

    // timing: the first decisions, repeated up to the largest batch
    for i in (0..max).step_by(refs.n) {
        let k = refs.n.min(max - i);
        dev.gpu.upload_into(&input, i * CROP_LEN * 4, refs.crop(0..k))?;
    }

    let mut timing = Vec::new();
    for &n in &args.batches {
        cnn.prepare(&dev, n)?;
        for _ in 0..3 {
            cnn.forward(&dev, &input, n)?;
        }
        dev.gpu.sync()?;

        let load = loadavg();
        let t = Instant::now();
        for _ in 0..args.reps {
            cnn.forward(&dev, &input, n)?;
        }
        dev.gpu.sync()?;
        let cnn_ms = t.elapsed().as_secs_f64() * 1e3 / (args.reps * n) as f64;

        let batch: Vec<StripRequest> = (0..n).map(|i| requests[i % refs.n].clone()).collect();
        let cells = cnn.forward(&dev, &input, n)?;
        head.forward(&dev, cells, &input, &batch)?;
        let t = Instant::now();
        for _ in 0..args.reps {
            let cells = cnn.forward(&dev, &input, n)?;
            head.forward(&dev, cells, &input, &batch)?;
        }
        let model_ms = t.elapsed().as_secs_f64() * 1e3 / (args.reps * n) as f64;
        let row = json!({"batch": n, "cnn_ms_per_decision": cnn_ms, "model_ms_per_decision": model_ms,
                         "head_ms_per_decision": model_ms - cnn_ms, "loadavg_before": load,
                         "loadavg_after": loadavg()});
        println!("{row}");
        timing.push(row);
    }
    report["timing"] = Value::from(timing);

    // parity: the first `cell_decisions` decisions' cells against the stock (bf16 autocast) cells
    let b = *args.batches.iter().min().unwrap();
    let k = args.cell_decisions.min(refs.n).min(b);
    dev.gpu.upload_into(&input, 0, refs.crop(0..k))?;
    let cells = cnn.forward(&dev, &input, k)?;
    let ours: Vec<u16> = dev.gpu.download_range(cells, 0, k * CELLS_LEN)?;
    let ours: Vec<f32> = ours.iter().map(|&v| bf16(v)).collect();
    let theirs: Vec<f32> = refs.cells(0..k).iter().map(|&v| bf16(v)).collect();
    let cells_vs_stock = spread(&ours, &theirs);
    println!("{}", json!({"cells_vs_stock": cells_vs_stock.json(), "decisions": k}));
    report["cells_vs_stock"] = json!({"decisions": k, "spread": cells_vs_stock.json()});

    // parity: every decision through the whole model, against each PyTorch variant
    let mut h1 = Vec::with_capacity(refs.n * GRID_LEN);
    let mut logs = Vec::with_capacity(refs.n * GRID_LEN);
    for a in (0..refs.n).step_by(b) {
        let e = (a + b).min(refs.n);
        dev.gpu.upload_into(&input, 0, refs.crop(a..e))?;
        let cells = cnn.forward(&dev, &input, e - a)?;
        for r in head.forward(&dev, cells, &input, &requests[a..e])? {
            h1.extend(r.heights.iter().flatten());
            logs.extend(r.log_scale.iter().flatten());
        }
    }
    ensure!(h1.len() == refs.n * GRID_LEN && logs.len() == h1.len(), "the model returned {} grid values", h1.len());
    // the raw heights beside the report, for signed (bias) comparisons the spreads cannot show
    let raw = args.out.with_extension("h1.f32");
    std::fs::write(&raw, h1.iter().flat_map(|v: &f32| v.to_le_bytes()).collect::<Vec<u8>>())
        .wrap_err_with(|| format!("write {}", raw.display()))?;

    // the head alone: the same decisions on the stock CNN's own cells, to split a bias between the CNN and the head
    let stock_cells = dev.gpu.alloc(b * CELLS_LEN * 2)?;
    let mut h1_head = Vec::with_capacity(refs.n * GRID_LEN);
    for a in (0..refs.n).step_by(b) {
        let e = (a + b).min(refs.n);
        dev.gpu.upload_into(&input, 0, refs.crop(a..e))?;
        dev.gpu.upload_into(&stock_cells, 0, refs.cells(a..e))?;
        for r in head.forward(&dev, &stock_cells, &input, &requests[a..e])? {
            h1_head.extend(r.heights.iter().flatten());
        }
    }
    let raw = args.out.with_extension("h1_stockcells.f32");
    std::fs::write(&raw, h1_head.iter().flat_map(|v: &f32| v.to_le_bytes()).collect::<Vec<u8>>())
        .wrap_err_with(|| format!("write {}", raw.display()))?;

    let mut parity = serde_json::Map::new();
    let mut failed = Vec::new();
    for (field, ours, bar) in [("h1", &h1, &HEIGHTS_BAR), ("logs", &logs, &LOGS_BAR)] {
        for variant in ["stock", "b64", "fp32"] {
            let s = spread(ours, refs.output(&format!("{variant}.{field}"))?);
            if variant == "fp32" && !bar.passes(&s) {
                failed.push(format!("{field} vs fp32 {s:?}"));
            }
            parity.insert(format!("{field}_vs_{variant}"), s.json());
        }
    }
    println!("{}", Value::from(parity.clone()));
    report["parity"] = Value::from(parity);
    report["parity_pass"] = Value::from(failed.is_empty());

    std::fs::write(&args.out, serde_json::to_vec_pretty(&report)?)
        .wrap_err_with(|| format!("write {}", args.out.display()))?;
    if !failed.is_empty() {
        bail!("parity outside PyTorch's bf16 spread: {}", failed.join("; "));
    }
    Ok(())
}

fn bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn loadavg() -> Value {
    let text = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    Value::from(text.split_whitespace().take(3).collect::<Vec<_>>().join(" "))
}
