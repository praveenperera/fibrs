//! `fibrs strips bench-model`: the native strip model timed per decision and checked against PyTorch's reference
//! on recorded decisions (`refset`), for the milestone and the gate's model check
//!
//! Timing runs each batch size on the reference's first decisions (repeated to fill the batch), warm, after the
//! plans for that size are made: the CNN alone, then the whole model (CNN and head, including the head's record
//! upload and result download). Parity runs every recorded decision once in batches of the smallest size and
//! holds the final heights and log scales to PyTorch's own bf16 spread (track-c-native.md, 11:40 PM Nov 23)
//!
//! `--invariance` instead runs every recorded decision under several batchings (the smallest batch size in order,
//! again, batches of 17, a fixed shuffle) and reports per stage (the CNN's cells, the head on those cells, the head
//! alone on the reference's stock cells) how many decisions' output bits differ from the first run, with the first
//! run's per-decision hashes for comparing separate processes

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use color_eyre::eyre::{Result, WrapErr, bail, ensure};
use serde_json::{Value, json};

use super::gpu::cnn::CropCnn;
use super::gpu::head::Head;
use super::gpu::{Device, Weights};
use super::refset::{CELLS_LEN, CROP_LEN, GRID_LEN, RefSet, Spread, spread};
use super::types::{STRIP_COLS, StripRequest};

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

/// The most the heights may tilt across the strip's columns against stock: the least-squares slope of the per-column
/// mean signed difference, times the strip's width. PyTorch b64 against stock tilts about 3e-4 on the reference; a
/// CNN convention bug that failed the end-to-end gate (Q1) tilted 5e-3 while every spread bar passed
const TILT_BAR: f64 = 1.5e-3;

/// The tilt of `ours - theirs` (row-major `[n, rows, cols]` grids) across the columns: the least-squares slope of the
/// per-column mean difference, times `cols - 1`
fn column_tilt(ours: &[f32], theirs: &[f32]) -> f64 {
    let mut sums = [0f64; STRIP_COLS];
    for (k, (a, b)) in ours.iter().zip(theirs).enumerate() {
        sums[k % STRIP_COLS] += f64::from(a - b);
    }

    let n = (ours.len() / STRIP_COLS) as f64;
    let centre = (STRIP_COLS - 1) as f64 / 2.0;
    let (mut num, mut den) = (0.0, 0.0);
    for (c, sum) in sums.iter().enumerate() {
        let x = c as f64 - centre;
        num += x * sum / n;
        den += x * x;
    }
    num / den * (STRIP_COLS - 1) as f64
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
    /// Also writes the native cells of every reference decision here (bf16 bits, the layout of the reference's
    /// `cells.bf16`)
    #[arg(long)]
    cells_out: Option<PathBuf>,
    /// Cells files (the layout of `cells.bf16`) to run the head alone on; each one's heights go to
    /// `<file>.h1.f32`, and its column tilt against stock to the report
    #[arg(long)]
    head_cells: Vec<PathBuf>,
    /// Checks batch invariance instead of timing and parity (module doc)
    #[arg(long)]
    invariance: bool,
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
    if args.invariance {
        let b = *args.batches.iter().min().unwrap();
        let probe = Probe { dev: &dev, head: &head, input: &input, refs: &refs, requests: &requests };
        report["invariance"] = probe.invariance(&mut cnn, b)?;
        return std::fs::write(&args.out, serde_json::to_vec_pretty(&report)?)
            .wrap_err_with(|| format!("write {}", args.out.display()));
    }

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
    let mut native_cells: Vec<u16> = Vec::new();
    for a in (0..refs.n).step_by(b) {
        let e = (a + b).min(refs.n);
        dev.gpu.upload_into(&input, 0, refs.crop(a..e))?;
        let cells = cnn.forward(&dev, &input, e - a)?;
        if args.cells_out.is_some() {
            native_cells.extend(dev.gpu.download_range::<u16>(cells, 0, (e - a) * CELLS_LEN)?);
        }
        for r in head.forward(&dev, cells, &input, &requests[a..e])? {
            h1.extend(r.heights.iter().flatten());
            logs.extend(r.log_scale.iter().flatten());
        }
    }
    if let Some(path) = &args.cells_out {
        std::fs::write(path, native_cells.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())
            .wrap_err_with(|| format!("write {}", path.display()))?;
    }
    ensure!(h1.len() == refs.n * GRID_LEN && logs.len() == h1.len(), "the model returned {} grid values", h1.len());
    // the raw heights beside the report, for signed (bias) comparisons the spreads cannot show
    write_f32(&args.out.with_extension("h1.f32"), &h1)?;

    // the head alone on the stock CNN's own cells, and on any cells files given, to split a bias between the CNN and
    // the head (and, with hybrid cells files, within the cells)
    let h1_head = head_on_cells(&dev, &head, &input, &refs, &requests, b, |r| Ok(refs.cells(r).to_vec()))?;
    write_f32(&args.out.with_extension("h1_stockcells.f32"), &h1_head)?;
    let mut head_tilts = serde_json::Map::new();
    head_tilts.insert("stock cells".into(), Value::from(column_tilt(&h1_head, refs.output("stock.h1")?)));
    for path in &args.head_cells {
        let bytes = std::fs::read(path).wrap_err_with(|| format!("read {}", path.display()))?;
        ensure!(
            bytes.len() == refs.n * CELLS_LEN * 2,
            "{}: {} bytes for {} decisions",
            path.display(),
            bytes.len(),
            refs.n
        );
        let values: Vec<u16> = bytes.as_chunks::<2>().0.iter().map(|c| u16::from_le_bytes(*c)).collect();
        let h = head_on_cells(&dev, &head, &input, &refs, &requests, b, |r| {
            Ok(values[r.start * CELLS_LEN..r.end * CELLS_LEN].to_vec())
        })?;
        write_f32(&path.with_extension("h1.f32"), &h)?;
        head_tilts.insert(path.display().to_string(), Value::from(column_tilt(&h, refs.output("stock.h1")?)));
    }
    println!("{}", json!({"head_column_tilt_vs_stock": head_tilts}));
    report["head_column_tilt_vs_stock"] = Value::from(head_tilts);

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
    // a signed check the spreads cannot make: no tilt across the strip against stock
    let tilt = column_tilt(&h1, refs.output("stock.h1")?);
    if tilt.abs() > TILT_BAR {
        failed.push(format!("h1 tilts {tilt:+.2e} across the columns vs stock (bar {TILT_BAR:.1e})"));
    }
    parity.insert("h1_column_tilt_vs_stock".into(), Value::from(tilt));

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

/// The head on every reference decision, batches of `b`, with each batch's cells from `cells` (bf16 bits); the
/// heights in decision order
fn head_on_cells(
    dev: &Device,
    head: &Head,
    input: &volkit::gpu::Buf,
    refs: &RefSet,
    requests: &[StripRequest],
    b: usize,
    cells: impl Fn(std::ops::Range<usize>) -> Result<Vec<u16>>,
) -> Result<Vec<f32>> {
    let buf = dev.gpu.alloc(b * CELLS_LEN * 2)?;
    let mut h1 = Vec::with_capacity(refs.n * GRID_LEN);
    for a in (0..refs.n).step_by(b) {
        let e = (a + b).min(refs.n);
        dev.gpu.upload_into(input, 0, refs.crop(a..e))?;
        dev.gpu.upload_into(&buf, 0, &cells(a..e)?)?;
        for r in head.forward(dev, &buf, input, &requests[a..e])? {
            h1.extend(r.heights.iter().flatten());
        }
    }
    Ok(h1)
}

/// Per decision, a hash of each stage's output bits under one batching
struct Bits {
    cells: Vec<u64>,
    model: Vec<u64>,
    head: Vec<u64>,
}

fn hash<T: std::hash::Hash>(v: &[T]) -> u64 {
    use std::hash::{DefaultHasher, Hash, Hasher};
    let mut h = DefaultHasher::new();
    v.hash(&mut h);
    h.finish()
}

fn hash_result(r: &super::types::StripResult) -> u64 {
    let bits: Vec<u32> = r.heights.iter().flatten().chain(r.log_scale.iter().flatten()).map(|v| v.to_bits()).collect();
    hash(&bits)
}

/// What every batching of the reference shares
#[derive(Clone, Copy)]
struct Probe<'a> {
    dev: &'a Device,
    head: &'a Head,
    input: &'a volkit::gpu::Buf,
    refs: &'a RefSet,
    requests: &'a [StripRequest],
}

impl Probe<'_> {
    /// One batching of the reference: decisions in `order`, launches of `b`
    fn batching(&self, cnn: &mut CropCnn, order: &[usize], b: usize) -> Result<Bits> {
        let Self { dev, head, input, refs, requests } = *self;
        let n = refs.n;
        let mut bits = Bits { cells: vec![0; n], model: vec![0; n], head: vec![0; n] };
        let stock = dev.gpu.alloc(b * CELLS_LEN * 2)?;
        for launch in order.chunks(b) {
            for (j, &i) in launch.iter().enumerate() {
                dev.gpu.upload_into(input, j * CROP_LEN * 4, refs.crop(i..i + 1))?;
                dev.gpu.upload_into(&stock, j * CELLS_LEN * 2, refs.cells(i..i + 1))?;
            }
            let batch: Vec<StripRequest> = launch.iter().map(|&i| requests[i].clone()).collect();
            let cells = cnn.forward(dev, input, launch.len())?;
            let native: Vec<u16> = dev.gpu.download_range(cells, 0, launch.len() * CELLS_LEN)?;
            let model = head.forward(dev, cells, input, &batch)?;
            let alone = head.forward(dev, &stock, input, &batch)?;
            for (j, &i) in launch.iter().enumerate() {
                bits.cells[i] = hash(&native[j * CELLS_LEN..(j + 1) * CELLS_LEN]);
                bits.model[i] = hash_result(&model[j]);
                bits.head[i] = hash_result(&alone[j]);
            }
        }
        Ok(bits)
    }

    /// Every decision under four batchings (`b` in order, again, 17 in order, `b` in a fixed shuffle), and per stage
    /// the decisions whose bits differ from the first, with the CNN's algorithm choices for each launch size
    fn invariance(&self, cnn: &mut CropCnn, b: usize) -> Result<Value> {
        let n = self.refs.n;
        let ordered: Vec<usize> = (0..n).collect();
        // a fixed shuffle: stride by a prime coprime to n
        let stride = [7919, 7907, 7901].into_iter().find(|p| !n.is_multiple_of(*p)).unwrap_or(1);
        let shuffled: Vec<usize> = (0..n).map(|i| i * stride % n).collect();
        let runs = [("b", &ordered, b), ("b again", &ordered, b), ("17", &ordered, 17), ("b shuffled", &shuffled, b)];

        let base = self.batching(cnn, runs[0].1, runs[0].2)?;
        let mut out = serde_json::Map::new();
        for (name, order, size) in &runs[1..] {
            let bits = self.batching(cnn, order, *size)?;
            let differ = |a: &[u64], b: &[u64]| -> Vec<usize> { (0..n).filter(|&i| a[i] != b[i]).collect() };
            let (cells, model, alone) =
                (differ(&base.cells, &bits.cells), differ(&base.model, &bits.model), differ(&base.head, &bits.head));
            let row = json!({
                "launch_size": size,
                "cells_differ": cells.len(), "model_differ": model.len(), "head_on_stock_cells_differ": alone.len(),
                "cells_differ_first": &cells[..cells.len().min(8)],
                "head_on_stock_cells_differ_first": &alone[..alone.len().min(8)],
            });
            println!("{}", json!({"invariance": name, "vs": format!("b={b} in order"), "result": row.clone()}));
            out.insert(format!("{name} (b={b})"), row);
        }
        let mut sizes: Vec<usize> = [16, b, 17, n % b, n % 17].into_iter().filter(|&k| k > 0).collect();
        sizes.sort_unstable();
        sizes.dedup();
        // only the sizes the CNN planned: with fixed sample groups that is the group alone
        let algos: serde_json::Map<String, Value> = sizes
            .iter()
            .map(|&k| (k, cnn.algorithms(k)))
            .filter(|(_, a)| !a.is_empty())
            .map(|(k, a)| (k.to_string(), Value::from(a)))
            .collect();
        println!("{}", json!({"cnn_algorithms": algos.clone()}));
        out.insert("cnn_algorithms".into(), Value::from(algos));
        // the first batching's bits, to compare runs in separate processes (each plans its own algorithms)
        let hex = |v: &[u64]| -> Vec<String> { v.iter().map(|h| format!("{h:016x}")).collect() };
        out.insert(
            "first_run_bits".into(),
            json!({"cells": hex(&base.cells), "model": hex(&base.model), "head_on_stock_cells": hex(&base.head)}),
        );
        Ok(Value::from(out))
    }
}

/// `values` as raw little-endian float32
fn write_f32(path: &std::path::Path, values: &[f32]) -> Result<()> {
    std::fs::write(path, values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())
        .wrap_err_with(|| format!("write {}", path.display()))
}

fn bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn loadavg() -> Value {
    let text = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    Value::from(text.split_whitespace().take(3).collect::<Vec<_>>().join(" "))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A difference rising 1e-4 per column tilts 24e-4 across the 25 columns; a constant offset or a symmetric bow
    /// tilts nothing
    #[test]
    fn column_tilt_reads_the_linear_part_across_columns() {
        let theirs = vec![0.5f32; 3 * STRIP_COLS];
        let shifted = |f: &dyn Fn(f64) -> f64| -> Vec<f32> {
            theirs.iter().enumerate().map(|(k, t)| t + f((k % STRIP_COLS) as f64) as f32).collect()
        };

        let rising = shifted(&|c| 1e-4 * c);
        assert!((column_tilt(&rising, &theirs) - 24e-4).abs() < 1e-7);

        let offset = shifted(&|_| 3e-3);
        assert!(column_tilt(&offset, &theirs).abs() < 1e-7);

        let bow = shifted(&|c| 1e-5 * (c - 12.0).powi(2));
        assert!(column_tilt(&bow, &theirs).abs() < 1e-7);
    }
}
