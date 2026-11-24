//! `fibrs strips bench-model`: the native strip model timed per decision and checked against PyTorch's reference
//! on recorded decisions (`refset`), for the milestone and the gate's model check
//!
//! Timing runs each batch size on the reference's first decisions (repeated to fill the batch), warm, after the
//! plans for that size are made; parity runs every recorded decision once in batches of the smallest size

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use color_eyre::eyre::{Result, WrapErr, ensure};
use serde_json::{Value, json};

use super::gpu::cnn::CropCnn;
use super::gpu::{Device, Weights};
use super::refset::{CELLS_LEN, CROP_LEN, RefSet, spread};

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
        let ms = t.elapsed().as_secs_f64() * 1e3 / (args.reps * n) as f64;
        let row = json!({"batch": n, "cnn_ms_per_decision": ms, "loadavg_before": load, "loadavg_after": loadavg()});
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

    std::fs::write(&args.out, serde_json::to_vec_pretty(&report)?)
        .wrap_err_with(|| format!("write {}", args.out.display()))?;
    Ok(())
}

fn bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn loadavg() -> Value {
    let text = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    Value::from(text.split_whitespace().take(3).collect::<Vec<_>>().join(" "))
}
