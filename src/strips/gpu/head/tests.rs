//! The head against PyTorch: per-stage parity with villa's own forward on recorded decisions
//! (dumped by a PyTorch script outside this repository), with the bf16-vs-fp32 and batch-4-vs-64 noise bands
//! beside each stage, and the head's speed per stage at several batch sizes
//!
//! Both need an NVIDIA GPU and the dump: `FIBRS_HEAD_REF` (default the b64 dump under
//! `/build/vesuvius/strips-scale/speed/native/head/`), `FIBRS_STRIP_WEIGHTS` (default the exported checkpoint)

use std::path::PathBuf;
use std::time::Instant;

use color_eyre::eyre::{Result, WrapErr, ensure, eyre};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};

use super::{CONTEXT, Head, LDK, STRIPS, bf16, bf16_f32, layout};
use crate::strips::gpu::{Device, Weights};
use crate::strips::types::{GEOMETRY_ROWS, HISTORY, StripRequest, StripResult};

fn dir() -> PathBuf {
    std::env::var_os("FIBRS_HEAD_REF")
        .map_or_else(|| PathBuf::from("/build/vesuvius/strips-scale/speed/native/head/ref_b64"), PathBuf::from)
}

fn weights() -> Result<Weights> {
    let path = std::env::var_os("FIBRS_STRIP_WEIGHTS").map_or_else(
        || PathBuf::from("/build/vesuvius/strips-scale/speed/native/weights/fiber_strip_small_v1.safetensors"),
        PathBuf::from,
    );
    Weights::open(&path)
}

/// A safetensors file, mapped
struct Dump(Mmap);

impl Dump {
    fn open(name: &str) -> Result<Self> {
        let path = dir().join(format!("{name}.safetensors"));
        let file = std::fs::File::open(&path).wrap_err_with(|| format!("open {}", path.display()))?;
        // SAFETY: the dump is written once by its GPU job and only read here
        Ok(Self(unsafe { Mmap::map(&file) }?))
    }

    fn tensors(&self) -> Result<SafeTensors<'_>> {
        SafeTensors::deserialize(&self.0).wrap_err("parse a dump")
    }
}

/// Tensor `name` of `st` as f32, with its shape
fn get(st: &SafeTensors, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
    let view = st.tensor(name).wrap_err_with(|| format!("dump tensor {name}"))?;
    let data = view.data();
    let values = match view.dtype() {
        Dtype::F32 => data.as_chunks::<4>().0.iter().map(|&b| f32::from_le_bytes(b)).collect(),
        Dtype::BF16 => data.as_chunks::<2>().0.iter().map(|&b| bf16_f32(u16::from_le_bytes(b))).collect(),
        Dtype::BOOL | Dtype::U8 => data.iter().map(|&b| f32::from(b)).collect(),
        Dtype::I64 => data.as_chunks::<8>().0.iter().map(|&b| i64::from_le_bytes(b) as f32).collect(),
        other => return Err(eyre!("{name}: dtype {other:?}")),
    };
    Ok((view.shape().to_vec(), values))
}

/// Mean, 99th percentile and max of `|got - want|` (NaN counts as infinite)
#[derive(Clone, Copy)]
struct Err3 {
    mean: f64,
    p99: f64,
    max: f64,
}

impl Err3 {
    fn of(got: &[f32], want: &[f32]) -> Self {
        assert_eq!(got.len(), want.len(), "compared lengths");
        let mut d: Vec<f64> = got
            .iter()
            .zip(want)
            .map(|(&g, &w)| {
                let e = (f64::from(g) - f64::from(w)).abs();
                if e.is_nan() { f64::INFINITY } else { e }
            })
            .collect();
        let mean = d.iter().sum::<f64>() / d.len() as f64;
        d.sort_unstable_by(f64::total_cmp);
        Self { mean, p99: d[(d.len() - 1) * 99 / 100], max: d[d.len() - 1] }
    }
}

impl std::fmt::Display for Err3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:>10.3e} {:>10.3e} {:>10.3e}", self.mean, self.p99, self.max)
    }
}

/// The recorded decisions' requests (the head reads no pose or tube)
fn requests(st: &SafeTensors, b: usize) -> Result<Vec<StripRequest>> {
    let (_, hist) = get(st, "hist")?;
    let (_, hmask) = get(st, "hmask")?;
    let (_, seed) = get(st, "seed")?;
    let (_, seed_mask) = get(st, "seed_mask")?;
    let (_, seed_age) = get(st, "seed_age")?;
    let (_, seed_tangent) = get(st, "seed_tangent")?;
    let (_, geometry) = get(st, "path_geometry")?;
    let (_, valid) = get(st, "path_geometry_valid")?;
    // the head dump names it as the model's input does, the recorded batches as the data loader does
    let (_, orientation) = get(st, "orientation").or_else(|_| get(st, "crop_orientation"))?;
    let n = hmask.len() / HISTORY;
    Ok((0..b)
        .map(|i| {
            let j = i % n;
            let mut q = StripRequest {
                pos: [0.0; 3],
                frame: [[0.0; 3]; 3],
                hist: [[0.0; 3]; HISTORY],
                hmask: hmask[j * HISTORY..(j + 1) * HISTORY].try_into().unwrap(),
                seed: seed[j * 3..j * 3 + 3].try_into().unwrap(),
                seed_mask: seed_mask[j],
                seed_age: seed_age[j],
                seed_tangent: seed_tangent[j * 3..j * 3 + 3].try_into().unwrap(),
                path_geometry: [[0.0; 8]; GEOMETRY_ROWS],
                path_geometry_valid: [false; GEOMETRY_ROWS],
                tube: Vec::new(),
                orientation: orientation[j * 5..j * 5 + 5].try_into().unwrap(),
            };
            for (d, s) in q.hist.iter_mut().zip(hist[j * HISTORY * 3..].as_chunks::<3>().0) {
                *d = *s;
            }
            for (d, s) in q.path_geometry.iter_mut().zip(geometry[j * GEOMETRY_ROWS * 8..].as_chunks::<8>().0) {
                *d = *s;
            }
            for (d, &s) in q.path_geometry_valid.iter_mut().zip(&valid[j * GEOMETRY_ROWS..]) {
                *d = s != 0.0;
            }
            q
        })
        .collect())
}

/// `cells` and `crop` of the recorded decisions on the device, `b` of them (repeating the recording past its end)
fn upload_inputs(dev: &Device, st: &SafeTensors, b: usize) -> Result<(volkit::gpu::Buf, volkit::gpu::Buf)> {
    let mut out = Vec::new();
    for name in ["cells", "crop"] {
        let view = st.tensor(name)?;
        let n = view.shape()[0];
        let per = view.data().len() / n;
        let buf = dev.gpu.alloc(per * b)?;
        for i in (0..b).step_by(n) {
            let count = n.min(b - i);
            dev.gpu.upload_into(&buf, i * per, &view.data()[..count * per])?;
        }
        out.push(buf);
    }
    let crop = out.pop().unwrap();
    Ok((out.pop().unwrap(), crop))
}

/// The bf16 buffer `buf`'s first `rows * ld` values, columns `..cols` of each row of `ld`
fn rows_bf16(dev: &Device, buf: &volkit::gpu::Buf, rows: usize, ld: usize, cols: usize) -> Result<Vec<f32>> {
    let raw: Vec<u16> = dev.gpu.download_range(buf, 0, rows * ld)?;
    Ok(raw.chunks_exact(ld).flat_map(|r| r[..cols].iter().map(|&v| bf16_f32(v))).collect())
}

fn rows_f32(dev: &Device, buf: &volkit::gpu::Buf, rows: usize, ld: usize, cols: usize) -> Result<Vec<f32>> {
    let raw: Vec<f32> = dev.gpu.download_range(buf, 0, rows * ld)?;
    Ok(raw.chunks_exact(ld).flat_map(|r| r[..cols].to_vec()).collect())
}

/// One layer's keys or values `[B, 4, LDK, 32]` cut to the context rows, as the dump's `[B, 4, 1580, 32]`
fn context_kv(dev: &Device, buf: &volkit::gpu::Buf, b: usize) -> Result<Vec<f32>> {
    let raw: Vec<u16> = dev.gpu.download_range(buf, 0, b * 4 * LDK * 32)?;
    Ok(raw.as_chunks::<{ LDK * 32 }>().0.iter().flat_map(|h| h[..CONTEXT * 32].iter().map(|&v| bf16_f32(v))).collect())
}

/// Compares stages against the dump and prints a row each: ours vs PyTorch bf16, then fp32 and batch 4 vs the
/// same PyTorch bf16 run (the noise bands)
struct Report<'a> {
    reference: SafeTensors<'a>,
    f32: SafeTensors<'a>,
    b4: SafeTensors<'a>,
    rows: Vec<(String, Err3)>,
}

impl Report<'_> {
    /// `got` against dump stage `stage` (times `scale`; rounded to bf16 first when `round`, for inputs our
    /// kernels hold as bf16 that PyTorch records in fp32)
    fn check(&mut self, label: &str, got: &[f32], stage: &str, scale: f32, round: bool) -> Result<Err3> {
        let fix = |v: Vec<f32>| -> Vec<f32> {
            v.into_iter().map(|x| x * scale).map(|x| if round { bf16_f32(bf16(x)) } else { x }).collect()
        };
        let (_, want) = get(&self.reference, stage)?;
        let want = fix(want);
        ensure!(got.len() == want.len(), "{label}: {} values, the dump's {stage} has {}", got.len(), want.len());
        let ours = Err3::of(got, &want);
        let band = |st: &SafeTensors| -> Result<String> {
            let Ok((_, other)) = get(st, stage) else { return Ok(format!("{:>21}", "-")) };
            let other = fix(other);
            let e = Err3::of(&other, &want[..other.len()]);
            Ok(format!("{:>10.3e} {:>10.3e}", e.p99, e.max))
        };
        let (f, b) = (band(&self.f32)?, band(&self.b4)?);
        println!("{label:<14} {ours} | {f} | {b}");
        self.rows.push((label.to_string(), ours));
        Ok(ours)
    }
}

#[test]
#[ignore = "needs an NVIDIA GPU and the head dump (FIBRS_HEAD_REF); run through the GPU queue"]
fn head_matches_pytorch() -> Result<()> {
    let w = weights()?;
    let dev = Device::open(0)?;
    let (inputs, reference, f32, b4, layout_dump) =
        (Dump::open("inputs")?, Dump::open("ref")?, Dump::open("f32")?, Dump::open("b4")?, Dump::open("layout")?);
    let inputs = inputs.tensors()?;
    let b = inputs.tensor("cells")?.shape()[0];

    // the fixed tables against PatchTokens' and the strip grid's buffers
    let lay = layout_dump.tensors()?;
    let cells = layout::cells();
    let strip = layout::strip();
    let flat = |v: &[[f32; 3]]| v.iter().flatten().copied().collect::<Vec<f32>>();
    for (name, ours) in [
        ("cell_xyz", flat(&cells.centres)),
        ("centers", flat(&cells.centres)),
        ("strip_anchor", flat(&strip.anchors)),
        ("strip_f", strip.f.clone()),
        ("strip_v", strip.v.clone()),
        ("strip_profile_offsets", strip.offsets.clone()),
    ] {
        let (_, want) = get(&lay, name)?;
        let e = Err3::of(&ours, &want);
        println!("layout {name:<22} max {:.3e}", e.max);
        ensure!(e.max <= 1e-6, "layout {name}: max error {}", e.max);
    }

    let head = Head::new(&dev, &w, b)?;
    let requests = requests(&inputs, b)?;
    let (cells_buf, crop_buf) = upload_inputs(&dev, &inputs, b)?;
    let mut report =
        Report { reference: reference.tensors()?, f32: f32.tensors()?, b4: b4.tensors()?, rows: Vec::new() };
    println!(
        "{:<14} {:>10} {:>10} {:>10} | {:>21} | {:>21}",
        "stage", "mean", "p99", "max", "fp32 p99, max", "b4 p99, max"
    );
    let position = rows_bf16(&dev, &head.params.position, layout::CELL_TOKENS, 128, 128)?;
    report.check("cell_position", &position, "cell_position", 1.0, false)?;

    let ms = b * STRIPS;
    let mut trace = |stage: &str, h: &Head| -> Result<()> {
        let x = &h.bufs;
        let r = &mut report;
        match stage {
            "token_inputs" => {
                // the sector tokens' inputs never leave their projection's kernel; seq0 checks its output
                r.check("tube_in", &rows_bf16(&dev, &x.xt, b * layout::TUBE_TOKENS, 960, 910)?, "tube_in", 1.0, true)?;
                r.check("ref_in", &rows_bf16(&dev, &x.xr, b * 129, 128, 72)?, "ref_in", 1.0, true)?;
                r.check("geom_in", &rows_bf16(&dev, &x.xg, b * 29, 128, 68)?, "geom_in", 1.0, true)?;
                r.check("squery_in", &rows_bf16(&dev, &x.xq, ms, 128, 67)?, "squery_in", 1.0, true)?;
                let mask: Vec<u8> = dev.gpu.download_range(&x.keymask, 0, b * LDK)?;
                let (_, padding) = get(&r.reference, "padding")?;
                let wrong = mask
                    .as_chunks::<LDK>()
                    .0
                    .iter()
                    .zip(padding.as_chunks::<CONTEXT>().0)
                    .map(|(m, p)| m[..CONTEXT].iter().zip(p).filter(|&(&m, &p)| (m != 0) == (p != 0.0)).count())
                    .sum::<usize>();
                println!("key mask: {wrong} of {} context keys disagree with PyTorch's padding", b * CONTEXT);
                ensure!(wrong == 0, "the key mask disagrees with PyTorch's padding at {wrong} keys");
            }
            "seq0" | "ctx0" | "ctx1" | "ctx2" => {
                r.check(stage, &rows_bf16(&dev, &x.seq, b * CONTEXT, 128, 128)?, stage, 1.0, false)?;
            }
            "context" => {
                for i in 0..4 {
                    r.check(&format!("k{i}"), &context_kv(&dev, &x.k[i], b)?, &format!("k{i}"), 1.0, false)?;
                    r.check(&format!("v{i}"), &context_kv(&dev, &x.v[i], b)?, &format!("v{i}"), 1.0, false)?;
                }
            }
            "tokens1" => {
                r.check("sorient", &rows_bf16(&dev, &x.orient, b, 128, 128)?, "sorient", 1.0, false)?;
                r.check(stage, &rows_bf16(&dev, &x.strip1, ms, 128, 128)?, stage, 1.0, false)?;
            }
            s if s.starts_with("p1l") => {
                r.check(stage, &rows_bf16(&dev, &x.strip1, ms, 128, 128)?, stage, 1.0, false)?;
            }
            s if s.starts_with("p2l") || s == "tokens2" => {
                if s == "tokens2" {
                    r.check("refine_in", &rows_bf16(&dev, &x.xf, ms, 192, 132)?, "refine_in", 1.0, true)?;
                    r.check("profile_in", &rows_bf16(&dev, &x.xp, ms, 128, 100)?, "profile_in", 1.0, true)?;
                }
                r.check(stage, &rows_f32(&dev, &x.strip2, ms, 128, 128)?, stage, 1.0, false)?;
            }
            "decoded1" => {
                r.check(stage, &rows_f32(&dev, &x.strip2, ms, 128, 128)?, stage, 1.0, false)?;
                r.check("height1", &rows_f32(&dev, &x.h0, ms, 2, 2)?, "height1", 8.0, false)?;
            }
            "decoded2" => {
                // pass 2's final norm is kept as bf16 only: its error includes that rounding
                r.check("decoded2 (bf16)", &rows_bf16(&dev, &x.ln, ms, 128, 128)?, stage, 1.0, false)?;
                r.check("height2", &rows_f32(&dev, &x.h1, ms, 2, 2)?, "height2", 8.0, false)?;
                r.check("logscale_raw", &rows_f32(&dev, &x.logs, ms, 2, 2)?, "logscale_raw", 1.0, false)?;
            }
            _ => {}
        }
        Ok(())
    };
    let results = head.forward_traced(&dev, &cells_buf, &crop_buf, &requests, &mut trace)?;

    // the result grid against the last pass of `heights` (B, 2, 36, 25) and `logs` (B, 36, 25)
    let (_, heights) = get(&report.reference, "heights")?;
    let last: Vec<f32> = heights.as_chunks::<1800>().0.iter().flat_map(|d| d[900..].to_vec()).collect();
    let grid = |f: fn(&StripResult) -> &[[f32; 25]; 36]| {
        results.iter().flat_map(|r| f(r).iter().flatten().copied()).collect::<Vec<f32>>()
    };
    let ours_h = grid(|r| &r.heights);
    let e = Err3::of(&ours_h, &last);
    println!("{:<14} {e}", "heights");
    let (_, logs) = get(&report.reference, "logs")?;
    println!("{:<14} {}", "log_scale", Err3::of(&grid(|r| &r.log_scale), &logs));
    // fast1's distance to stock: mean 0.0106, p99 0.0625, max 0.25
    ensure!(e.max <= 0.25 && e.p99 <= 0.0625, "heights: {e} against PyTorch, past fast1's distance to stock");
    Ok(())
}

/// Median of `v`
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_unstable_by(f64::total_cmp);
    v[v.len() / 2]
}

#[test]
#[ignore = "needs an NVIDIA GPU and the head dump (FIBRS_HEAD_REF); run through the GPU queue"]
fn head_speed() -> Result<()> {
    let batches: Vec<usize> = std::env::var("FIBRS_HEAD_BATCHES")
        .unwrap_or_else(|_| "32,64,128".into())
        .split(',')
        .map(|s| s.trim().parse())
        .collect::<Result<_, _>>()?;
    let reps: usize = std::env::var("FIBRS_HEAD_REPS").map_or(Ok(20), |s| s.parse())?;
    let w = weights()?;
    let dev = Device::open(0)?;
    let inputs = Dump::open("inputs")?;
    let inputs = inputs.tensors()?;
    let loadavg = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    println!("loadavg at start: {}", loadavg.trim());
    for &b in &batches {
        let head = Head::new(&dev, &w, b)?;
        let requests = requests(&inputs, b)?;
        let (cells, crop) = upload_inputs(&dev, &inputs, b)?;
        for _ in 0..3 {
            head.forward(&dev, &cells, &crop, &requests)?;
        }

        // untraced: the whole forward, from packing the requests to the downloaded results
        let mut totals = Vec::new();
        for _ in 0..reps {
            let start = Instant::now();
            head.forward(&dev, &cells, &crop, &requests)?;
            totals.push(start.elapsed().as_secs_f64() * 1e3);
        }

        // per stage: synchronized at each stage's end
        let mut stages: Vec<(String, Vec<f64>)> = Vec::new();
        for _ in 0..reps {
            let mut last = Instant::now();
            let mut at = 0;
            let mut trace = |stage: &str, _: &Head| -> Result<()> {
                dev.gpu.sync()?;
                let now = Instant::now();
                let ms = (now - last).as_secs_f64() * 1e3;
                last = now;
                if stages.len() <= at {
                    stages.push((stage.to_string(), Vec::new()));
                }
                stages[at].1.push(ms);
                at += 1;
                Ok(())
            };
            head.forward_traced(&dev, &cells, &crop, &requests, &mut trace)?;
        }
        let total = median(totals);
        println!("b{b}: {:.4} ms/decision ({total:.3} ms per batch, median of {reps})", total / b as f64);
        for (stage, times) in stages {
            println!("b{b}   {stage:<14} {:.4} ms/decision", median(times) / b as f64);
        }
    }
    let loadavg = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    println!("loadavg at end: {}", loadavg.trim());
    Ok(())
}

/// A raw little-endian file, mapped
fn map(path: &std::path::Path) -> Result<Mmap> {
    let file = std::fs::File::open(path).wrap_err_with(|| format!("open {}", path.display()))?;
    // SAFETY: the reference is written once by its GPU job and only read here
    Ok(unsafe { Mmap::map(&file) }?)
}

/// The head on the shared strip-model reference's decisions (see `refset`;
/// `FIBRS_STRIP_REF`, default its 340-decision run) in batches of 64 on PyTorch's stock cells, against the stock
/// (bf16, batch 4) and batch-64 PyTorch heights and log scales, with the reference's own variant distances beside;
/// with `FIBRS_HEAD_DUMP` set, also writes the native grids there
#[test]
#[ignore = "needs an NVIDIA GPU and the strip-model reference (FIBRS_STRIP_REF); run through the GPU queue"]
fn head_matches_strip_reference() -> Result<()> {
    let dir = std::env::var_os("FIBRS_STRIP_REF")
        .map_or_else(|| PathBuf::from("/build/vesuvius/strips-scale/speed/native/ref/strip340"), PathBuf::from);
    let records = map(&dir.join("records.safetensors"))?;
    let records = SafeTensors::deserialize(&records)?;
    let outputs = map(&dir.join("outputs.safetensors"))?;
    let outputs = SafeTensors::deserialize(&outputs)?;
    let (cells, crop) = (map(&dir.join("cells.bf16"))?, map(&dir.join("crop.f32"))?);
    let n = records.tensor("hmask")?.shape()[0];
    let (per_cells, per_crop) = (36 * 32 * 32 * 64 * 2, 144 * 128 * 128 * 2 * 4);
    ensure!(cells.len() == n * per_cells && crop.len() == n * per_crop, "cells or crop do not hold {n} decisions");

    let w = weights()?;
    let dev = Device::open(0)?;
    let batch = 64;
    let head = Head::new(&dev, &w, batch)?;
    let requests = requests(&records, n)?;
    let (cells_buf, crop_buf) = (dev.gpu.alloc(batch * per_cells)?, dev.gpu.alloc(batch * per_crop)?);
    let (mut heights, mut logs) = (Vec::with_capacity(n * 900), Vec::with_capacity(n * 900));
    for start in (0..n).step_by(batch) {
        let end = n.min(start + batch);
        dev.gpu.upload_into(&cells_buf, 0, &cells[start * per_cells..end * per_cells])?;
        dev.gpu.upload_into(&crop_buf, 0, &crop[start * per_crop..end * per_crop])?;
        for r in head.forward(&dev, &cells_buf, &crop_buf, &requests[start..end])? {
            heights.extend(r.heights.iter().flatten());
            logs.extend(r.log_scale.iter().flatten());
        }
    }

    // the raw grids for signed-bias and worst-item checks (tilt.py): `h1.f32`, `logs.f32`, `[n, 36, 25]` f32
    if let Some(dump) = std::env::var_os("FIBRS_HEAD_DUMP").map(PathBuf::from) {
        std::fs::create_dir_all(&dump)?;
        let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        std::fs::write(dump.join("h1.f32"), bytes(&heights))?;
        std::fs::write(dump.join("logs.f32"), bytes(&logs))?;
    }

    println!("{n} decisions, batches of {batch}: |native - torch| and torch's own variant distances");
    println!("{:<22} {:>10} {:>10} {:>10}", "", "mean", "p99", "max");
    for (field, ours) in [("h1", &heights), ("logs", &logs)] {
        for variant in ["stock", "b64", "fp32"] {
            let (_, want) = get(&outputs, &format!("{variant}.{field}"))?;
            println!("{:<22} {}", format!("{field} native vs {variant}"), Err3::of(ours, &want));
        }
        let (_, stock) = get(&outputs, &format!("stock.{field}"))?;
        for variant in ["b64", "fp32"] {
            let (_, other) = get(&outputs, &format!("{variant}.{field}"))?;
            println!("{:<22} {}", format!("{field} {variant} vs stock"), Err3::of(&other, &stock));
        }
    }

    // bench-model's bars against the fp32 forward (PyTorch's own bf16 spread with ~15% headroom), and the worst
    // single height no further from fp32 than PyTorch's own worst (stock or b64) plus one bf16 step at that
    // height: PyTorch's run-to-run rounding moves a single item by a step (0.125 at |h| >= 16), so a bar without
    // the step fails any change of summation order on a coin flip. Against stock alone the worst item compounds
    // two bf16 errors (0.375 when the two round opposite ways)
    let (_, fp32) = get(&outputs, "fp32.h1")?;
    let (_, stock) = get(&outputs, "stock.h1")?;
    let (_, b64) = get(&outputs, "b64.h1")?;
    let h = Err3::of(&heights, &fp32);
    let own = Err3::of(&stock, &fp32).max.max(Err3::of(&b64, &fp32).max);
    let (worst, _) = heights
        .iter()
        .zip(&fp32)
        .map(|(&a, &b)| (a, (a - b).abs()))
        .fold((0.0f32, -1.0f32), |w, x| if x.1 > w.1 { x } else { w });
    let bar = own + f64::from(bf16_step(worst));
    let (_, fp32_logs) = get(&outputs, "fp32.logs")?;
    let l = Err3::of(&logs, &fp32_logs);
    ensure!(h.mean <= 0.012 && h.p99 <= 0.065, "heights: {h} against fp32, past bench-model's bar");
    println!("h1 worst vs fp32 {:.4} at h {worst} (bar {bar:.4}: PyTorch's own worst {own:.4} + one bf16 step)", h.max);
    ensure!(h.max <= bar, "heights: worst {} against fp32 at h {worst}, past {bar}", h.max);
    ensure!(l.mean <= 0.004 && l.p99 <= 0.015, "log scales: {l} against fp32, past bench-model's bar");

    // bench-model's column-tilt bar against stock: a lateral convention bug tilted 5e-3 with every spread in bar
    let tilt = column_tilt(&heights, &stock);
    println!("h1 column tilt vs stock {tilt:+.3e} (bar 1.5e-3)");
    ensure!(tilt.abs() <= 1.5e-3, "heights tilt {tilt:+.3e} across the columns against stock");
    Ok(())
}

/// One bf16 step (the spacing of bf16 values) at `v`'s magnitude
fn bf16_step(v: f32) -> f32 {
    // the exponent alone is 2^e for |v| in [2^e, 2^(e+1)); bf16 keeps 7 fraction bits
    f32::from_bits(v.to_bits() & 0x7f80_0000) / 128.0
}

/// The tilt of `ours - theirs` (row-major `[n, 36, 25]` grids) across the strip's columns: the least-squares slope
/// of the per-column mean signed difference times 24, as bench-model computes it
fn column_tilt(ours: &[f32], theirs: &[f32]) -> f64 {
    let mut sums = [0f64; 25];
    for (k, (a, b)) in ours.iter().zip(theirs).enumerate() {
        sums[k % 25] += f64::from(a - b);
    }

    let n = (ours.len() / 25) as f64;
    let (mut num, mut den) = (0.0, 0.0);
    for (c, sum) in sums.iter().enumerate() {
        let x = c as f64 - 12.0;
        num += x * sum / n;
        den += x * x;
    }
    num / den * 24.0
}

/// One gpu-ab arm: the head at batch `FIBRS_HEAD_BATCH` (default
/// 64) on the dump's decisions, `FIBRS_HEAD_REPS` (default 50) untraced forwards after 5 warmups; writes the median
/// ms/decision and the host load to `FIBRS_HEAD_OUT/metrics.json`, the sha256 of the last forward's heights and
/// log scales to `fp.txt`, so arms that should match bit for bit are checked, and the values themselves (every
/// decision's heights, then every decision's log scales, little-endian f32) to `outputs.f32` for arms whose
/// arithmetic changed (`head_gate.py`)
#[test]
#[ignore = "needs an NVIDIA GPU and the head dump (FIBRS_HEAD_REF); run through the GPU queue"]
fn head_bench() -> Result<()> {
    use sha2::{Digest, Sha256};

    let out = PathBuf::from(std::env::var_os("FIBRS_HEAD_OUT").ok_or_else(|| eyre!("FIBRS_HEAD_OUT is not set"))?);
    let b: usize = std::env::var("FIBRS_HEAD_BATCH").map_or(Ok(64), |s| s.parse())?;
    let reps: usize = std::env::var("FIBRS_HEAD_REPS").map_or(Ok(50), |s| s.parse())?;
    let w = weights()?;
    let dev = Device::open(0)?;
    let inputs = Dump::open("inputs")?;
    let inputs = inputs.tensors()?;
    let head = Head::new(&dev, &w, b)?;
    let requests = requests(&inputs, b)?;
    let (cells, crop) = upload_inputs(&dev, &inputs, b)?;
    let load = || std::fs::read_to_string("/proc/loadavg").unwrap_or_default().trim().to_string();
    let load_start = load();
    for _ in 0..5 {
        head.forward(&dev, &cells, &crop, &requests)?;
    }

    let mut times = Vec::with_capacity(reps);
    let mut results = Vec::new();
    for _ in 0..reps {
        let start = Instant::now();
        results = head.forward(&dev, &cells, &crop, &requests)?;
        times.push(start.elapsed().as_secs_f64() * 1e3);
    }
    let ms = median(times) / b as f64;
    let mut hash = Sha256::new();
    for r in &results {
        for v in r.heights.iter().flatten().chain(r.log_scale.iter().flatten()) {
            hash.update(v.to_le_bytes());
        }
    }
    let fp = format!("{:x}", hash.finalize());
    let values: Vec<u8> = results
        .iter()
        .flat_map(|r| r.heights.iter().flatten())
        .chain(results.iter().flat_map(|r| r.log_scale.iter().flatten()))
        .flat_map(|v| v.to_le_bytes())
        .collect();
    std::fs::create_dir_all(&out)?;
    std::fs::write(out.join("fp.txt"), format!("{fp}\n"))?;
    std::fs::write(out.join("outputs.f32"), values)?;
    let metrics = serde_json::json!({
        "batch": b,
        "reps": reps,
        "ms_per_decision": ms,
        "loadavg_start": load_start,
        "loadavg_end": load(),
    });
    std::fs::write(out.join("metrics.json"), serde_json::to_string_pretty(&metrics)?)?;
    println!("b{b}: {ms:.5} ms/decision, fingerprint {fp}");
    Ok(())
}
