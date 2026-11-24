//! PyTorch's reference for the strip model on recorded decisions, as `ops/vesuvius/strips-native/ref_strip.py`
//! writes it, and the spread statistics the native outputs are judged by
//!
//! The directory holds `crop.f32` (N x 144 x 128 x 128 x 2 float32, NDHWC: the z-scored CT and the rendered tube),
//! `cells.bf16` (N x 36 x 32 x 32 x 64 bfloat16 bits, NDHWC: the stock CNN output), `records.safetensors` (the
//! head's per-decision inputs) and `outputs.safetensors` (`<variant>.{h0,h1,logs}`, N x 36 x 25 float32, for the
//! variants `stock`, `b64` and `fp32`). PyTorch's bf16 is chaotic on this model, so the native outputs are held to
//! the spread between those variants, not to bit parity

use std::collections::HashMap;
use std::fs;
use std::ops::Range;
use std::path::Path;

use color_eyre::eyre::{Result, WrapErr, ensure, eyre};
use memmap2::Mmap;
use serde_json::{Value, json};

use super::types::{GEOMETRY_ROWS, HISTORY, STRIP_COLS, STRIP_ROWS, StripRequest, TUBE_POINTS};

/// Strip crop voxels per decision, both channels
pub const CROP_LEN: usize = 144 * 128 * 128 * 2;
/// CNN cell values per decision
pub const CELLS_LEN: usize = 36 * 32 * 32 * 64;
/// Grid points per decision output
pub const GRID_LEN: usize = STRIP_ROWS * STRIP_COLS;

/// A tensor of a safetensors file: its shape and its values as float32
type Tensor = (Vec<usize>, Vec<f32>);

/// A reference directory, memory-mapped
pub struct RefSet {
    /// Decisions
    pub n: usize,
    crop: Mmap,
    cells: Mmap,
    records: HashMap<String, Tensor>,
    outputs: HashMap<String, Vec<f32>>,
}

/// |a - b| over every value: mean, 99th percentile and max
#[derive(Clone, Copy, Debug)]
pub struct Spread {
    pub mean: f64,
    pub p99: f64,
    pub max: f64,
}

impl Spread {
    /// As JSON, for reports
    pub fn json(&self) -> Value {
        json!({"mean": self.mean, "p99": self.p99, "max": self.max})
    }
}

impl RefSet {
    /// Opens `dir`, checking every file's size against the decision count
    pub fn open(dir: &Path) -> Result<Self> {
        let map = |name: &str| -> Result<Mmap> {
            let path = dir.join(name);
            let file = fs::File::open(&path).wrap_err_with(|| format!("open {}", path.display()))?;
            // SAFETY: the reference files are written once by ref_strip.py and only read here
            unsafe { Mmap::map(&file) }.wrap_err_with(|| format!("map {}", path.display()))
        };
        let crop = map("crop.f32")?;
        let cells = map("cells.bf16")?;
        ensure!(crop.len() % (CROP_LEN * 4) == 0, "crop.f32: {} bytes", crop.len());
        let n = crop.len() / (CROP_LEN * 4);
        ensure!(cells.len() == n * CELLS_LEN * 2, "cells.bf16: {} bytes for {n} decisions", cells.len());

        let records = read_f32_tensors(&dir.join("records.safetensors"))?;
        let outputs: HashMap<String, Vec<f32>> = read_f32_tensors(&dir.join("outputs.safetensors"))?
            .into_iter()
            .map(|(name, (shape, values))| {
                ensure!(shape == [n, STRIP_ROWS, STRIP_COLS], "outputs {name}: shape {shape:?}");
                Ok((name, values))
            })
            .collect::<Result<_>>()?;

        for (name, shape) in [
            ("hist", vec![n, HISTORY, 3]),
            ("hmask", vec![n, HISTORY]),
            ("seed", vec![n, 1, 3]),
            ("seed_mask", vec![n, 1]),
            ("seed_age", vec![n]),
            ("seed_tangent", vec![n, 3]),
            ("path_geometry", vec![n, GEOMETRY_ROWS, 8]),
            ("path_geometry_valid", vec![n, GEOMETRY_ROWS]),
            ("tube_points", vec![n, TUBE_POINTS, 3]),
            ("tube_valid", vec![n, TUBE_POINTS]),
            ("crop_orientation", vec![n, 5]),
        ] {
            let got = &records.get(name).ok_or_else(|| eyre!("records: no {name}"))?.0;
            ensure!(*got == shape, "records {name}: shape {got:?}, expected {shape:?}");
        }
        Ok(Self { n, crop, cells, records, outputs })
    }

    /// The model input of decisions `range`, NDHWC float32
    pub fn crop(&self, range: Range<usize>) -> &[f32] {
        let all = cast::<f32>(&self.crop);
        &all[range.start * CROP_LEN..range.end * CROP_LEN]
    }

    /// The stock CNN cells of decisions `range`, NDHWC bfloat16 bits
    pub fn cells(&self, range: Range<usize>) -> &[u16] {
        let all = cast::<u16>(&self.cells);
        &all[range.start * CELLS_LEN..range.end * CELLS_LEN]
    }

    /// Output `name` (`stock.h1`, `fp32.logs`, ...), N x 36 x 25
    pub fn output(&self, name: &str) -> Result<&[f32]> {
        self.outputs.get(name).map(Vec::as_slice).ok_or_else(|| eyre!("outputs: no {name}"))
    }

    /// Decision `i`'s head inputs as the CPU side hands them over; `pos` and `frame` are unset (the crop is given)
    pub fn request(&self, i: usize) -> StripRequest {
        let row = |name: &str, len: usize| &self.records[name].1[i * len..(i + 1) * len];
        let hist = row("hist", HISTORY * 3);
        let geometry = row("path_geometry", GEOMETRY_ROWS * 8);
        let tube_points = row("tube_points", TUBE_POINTS * 3);
        let tube_valid = row("tube_valid", TUBE_POINTS);
        StripRequest {
            pos: [0.0; 3],
            frame: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            hist: std::array::from_fn(|k| [hist[3 * k], hist[3 * k + 1], hist[3 * k + 2]]),
            hmask: row("hmask", HISTORY).try_into().unwrap(),
            seed: row("seed", 3).try_into().unwrap(),
            seed_mask: row("seed_mask", 1)[0],
            seed_age: row("seed_age", 1)[0],
            seed_tangent: row("seed_tangent", 3).try_into().unwrap(),
            path_geometry: std::array::from_fn(|r| geometry[8 * r..8 * r + 8].try_into().unwrap()),
            path_geometry_valid: std::array::from_fn(|r| row("path_geometry_valid", GEOMETRY_ROWS)[r] != 0.0),
            // the recorded tubes are valid-prefixed (batched.py fills rows from the start)
            tube: (0..TUBE_POINTS)
                .take_while(|&k| tube_valid[k] != 0.0)
                .map(|k| [tube_points[3 * k], tube_points[3 * k + 1], tube_points[3 * k + 2]])
                .collect(),
            orientation: row("crop_orientation", 5).try_into().unwrap(),
        }
    }
}

/// |a - b| statistics; NaN anywhere (a gap the native side never wrote) reads as an infinite max. Differences are
/// held in f32 (cell comparisons run to tens of millions of values), summed in f64
pub fn spread(a: &[f32], b: &[f32]) -> Spread {
    assert_eq!(a.len(), b.len(), "spread of unequal lengths");
    assert!(!a.is_empty(), "spread of nothing");
    let mut d: Vec<f32> = a.iter().zip(b).map(|(&x, &y)| (x - y).abs()).collect();
    if d.iter().any(|v| v.is_nan()) {
        return Spread { mean: f64::NAN, p99: f64::INFINITY, max: f64::INFINITY };
    }

    let mean = d.iter().map(|&v| f64::from(v)).sum::<f64>() / d.len() as f64;
    let max = d.iter().copied().fold(0.0f32, f32::max);
    let k = ((d.len() - 1) as f64 * 0.99).round() as usize;
    let (_, p99, _) = d.select_nth_unstable_by(k, f32::total_cmp);
    Spread { mean, p99: f64::from(*p99), max: f64::from(max) }
}

/// The page-aligned bytes of a map as `T`s
fn cast<T: Copy>(bytes: &[u8]) -> &[T] {
    // SAFETY: the maps are page-aligned and hold whole little-endian values; every bit pattern is a valid f32/u16
    let (head, body, tail) = unsafe { bytes.align_to::<T>() };
    assert!(head.is_empty() && tail.is_empty(), "misaligned reference map");
    body
}

/// Every tensor of a safetensors file as float32 (booleans and bytes widened), with its shape
fn read_f32_tensors(path: &Path) -> Result<HashMap<String, Tensor>> {
    let bytes = fs::read(path).wrap_err_with(|| format!("read {}", path.display()))?;
    let st = safetensors::SafeTensors::deserialize(&bytes).wrap_err_with(|| format!("parse {}", path.display()))?;
    st.tensors()
        .into_iter()
        .map(|(name, view)| {
            let values: Vec<f32> = match view.dtype() {
                safetensors::Dtype::F32 => {
                    view.data().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
                }
                safetensors::Dtype::U8 | safetensors::Dtype::BOOL => {
                    view.data().iter().map(|&b| f32::from(b)).collect()
                }
                other => return Err(eyre!("{}: {name} is {other:?}", path.display())),
            };
            Ok((name, (view.shape().to_vec(), values)))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spread over a known difference set, and a NaN that fails loudly
    #[test]
    fn spread_reads_mean_p99_max_and_flags_nan() {
        let a: Vec<f32> = (0..101).map(|i| i as f32 * 0.01).collect();
        let b = vec![0.0f32; 101];
        let s = spread(&a, &b);
        assert!((s.mean - 0.5).abs() < 1e-6, "{s:?}");
        assert!((s.p99 - 0.99).abs() < 1e-6, "{s:?}");
        assert!((s.max - 1.0).abs() < 1e-6, "{s:?}");

        let mut c = b.clone();
        c[7] = f32::NAN;
        assert!(spread(&a, &c).max.is_infinite());
    }
}
