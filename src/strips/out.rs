//! A fibre's outputs as `strips_fast.py` writes them (`AFVJob.write`): the strip as a tifxyz directory
//! (`write_surface(..., mmap_layout=True)`: x/y/z/mask TIFFs, meta.json with its area), trace.json,
//! expected_error.tif, the index line, and optionally Check A's npz; and the run's run.json
//!
//! TIFFs are byte for byte what vc_reader's libtiff writer gives for uncompressed strips; JSON follows Python's
//! `json.dumps` ([`crate::pyfmt`]). The surface uuid is a fresh uuid4 unless the directory already has one, as
//! villa keeps it

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use color_eyre::eyre::{Result, WrapErr};
use serde_json::{Map, Value, json};

use super::fibre::Fibre;
use super::np::{M3, V3, cross, norm, sub};
use super::stitch::{Display, Strip, display_grid, plane, strip_f, strip_v};
use super::types::{STRIP_COLS, STRIP_ROWS};
use crate::pyfmt::{JsonStyle, py_dumps};

/// A TIFF sample type: bits and the SampleFormat tag
#[derive(Clone, Copy)]
pub enum Sample {
    U8,
    F32,
}

/// vc_reader's strips of about 256 KiB
const STRIP_BYTES: usize = 256 * 1024;

/// An uncompressed single-page TIFF, as vc_reader's `writeTiff` lays it out through libtiff: the strips from
/// offset 8, the directory at the next even offset, then the StripByteCounts and StripOffsets arrays when there is
/// more than one strip
pub fn tiff_bytes(data: &[u8], rows: usize, cols: usize, sample: Sample) -> Vec<u8> {
    let (bits, format) = match sample {
        Sample::U8 => (8u16, 1u16),
        Sample::F32 => (32, 3),
    };
    let line = cols * usize::from(bits / 8);
    assert_eq!(data.len(), rows * line, "TIFF data does not match its shape");
    let per_strip = (STRIP_BYTES / line).clamp(1, rows);
    let strips = rows.div_ceil(per_strip);
    let mut out = Vec::with_capacity(data.len() + 200);
    out.extend_from_slice(b"II*\0");
    out.extend_from_slice(&[0; 4]);
    let mut offsets = Vec::with_capacity(strips);
    let mut counts = Vec::with_capacity(strips);
    for s in 0..strips {
        let r0 = s * per_strip;
        let r1 = (r0 + per_strip).min(rows);
        offsets.push(out.len() as u32);
        counts.push(((r1 - r0) * line) as u32);
        out.extend_from_slice(&data[r0 * line..r1 * line]);
    }
    if out.len() % 2 == 1 {
        out.push(0);
    }
    let ifd = out.len() as u32;
    out[4..8].copy_from_slice(&ifd.to_le_bytes());
    let entries: u16 = 11;
    let data_at = ifd + 2 + 12 * u32::from(entries) + 4;
    // out-of-line arrays follow the directory: byte counts, then offsets
    let (counts_at, offsets_at) = (data_at, data_at + 4 * strips as u32);
    let short = |tag: u16, v: u16| (tag, 3u16, 1u32, u32::from(v));
    // ImageWidth/ImageLength/RowsPerStrip are SHORT when they fit, as libtiff writes them
    let short_long = |tag: u16, v: usize| match u16::try_from(v) {
        Ok(v) => (tag, 3u16, 1u32, u32::from(v)),
        Err(_) => (tag, 4u16, 1u32, v as u32),
    };
    let array = |tag: u16, values: &[u32], at: u32| match values {
        [one] => (tag, 4u16, 1u32, *one),
        _ => (tag, 4u16, values.len() as u32, at),
    };
    let tags = [
        short_long(256, cols),
        short_long(257, rows),
        short(258, bits),
        short(259, 1),
        short(262, 1),
        array(273, &offsets, offsets_at),
        short(277, 1),
        short_long(278, per_strip),
        array(279, &counts, counts_at),
        short(284, 1),
        short(339, format),
    ];
    out.extend_from_slice(&entries.to_le_bytes());
    for (tag, kind, count, value) in tags {
        out.extend_from_slice(&tag.to_le_bytes());
        out.extend_from_slice(&kind.to_le_bytes());
        out.extend_from_slice(&count.to_le_bytes());
        // a SHORT value sits in the field's first two bytes
        out.extend_from_slice(&value.to_le_bytes());
    }
    out.extend_from_slice(&0u32.to_le_bytes());
    if strips > 1 {
        for v in counts.iter().chain(&offsets) {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

/// Writes `bytes` to `path` through a temporary file beside it
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_file_name(format!(
        ".{}.tmp.{}",
        path.file_name().unwrap_or_default().to_string_lossy(),
        std::process::id()
    ));
    fs::write(&tmp, bytes).wrap_err_with(|| format!("write {}", tmp.display()))?;
    fs::rename(&tmp, path).wrap_err_with(|| format!("rename to {}", path.display()))
}

fn f32_bytes(values: impl Iterator<Item = f32>) -> Vec<u8> {
    values.flat_map(f32::to_le_bytes).collect()
}

/// A fresh random uuid4, as Python's `str(uuid4())`
fn uuid4() -> Result<String> {
    let mut b = [0u8; 16];
    getrandom::fill(&mut b).map_err(|e| color_eyre::eyre::eyre!("random uuid: {e}"))?;
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    let h: String = b.iter().map(|x| format!("{x:02x}")).collect();
    Ok(format!("{}-{}-{}-{}-{}", &h[..8], &h[8..12], &h[12..16], &h[16..20], &h[20..]))
}

/// The area (voxels^2) of a tifxyz grid whose invalid nodes are -1 (`surface_area_vx2`): two triangles per quad
/// with four valid corners, summed by NumPy's pairwise sum in row-major order
fn surface_area(xyz: &[[f32; 3]], rows: usize, cols: usize) -> f64 {
    let p = |r: usize, c: usize| xyz[r * cols + c].map(f64::from);
    let ok = |r: usize, c: usize| {
        let q = p(r, c);
        q.iter().all(|&v| v != -1.0 && v.is_finite())
    };
    let tri = |u: V3, v: V3, w: V3| 0.5 * norm(cross(sub(v, u), sub(w, u)));
    let mut areas = Vec::new();
    for r in 0..rows.saturating_sub(1) {
        for c in 0..cols.saturating_sub(1) {
            if ok(r, c) && ok(r, c + 1) && ok(r + 1, c) && ok(r + 1, c + 1) {
                let (a, b, cc, d) = (p(r, c), p(r, c + 1), p(r + 1, c), p(r + 1, c + 1));
                areas.push(tri(a, b, d) + tri(a, d, cc));
            }
        }
    }
    super::np::sum(&areas)
}

/// `np.nanmean(np.where(keep, expected, nan))` of the strip as displayed with `(rows, columns)` flipped. NumPy keeps
/// the strip's memory layout through display_grid's views (`np.where` and nanmean's copy allocate in K order), so
/// the pairwise sum runs over the flipped strip in row-major order, never the transposed one
fn mean_expected(strip: &Strip, (flip_rows, flip_cols): (bool, bool)) -> f64 {
    let rows = strip.keep.len();
    let mut values = Vec::with_capacity(rows * STRIP_COLS);
    let mut count = 0usize;
    for r in 0..rows {
        let r = if flip_rows { rows - 1 - r } else { r };
        for c in 0..STRIP_COLS {
            let c = if flip_cols { STRIP_COLS - 1 - c } else { c };
            let e = strip.expected[r][c];
            let kept = strip.keep[r][c] && !e.is_nan();
            count += usize::from(kept);
            values.push(if kept { e } else { 0.0 });
        }
    }
    super::np::sum(&values) / count as f64
}

/// What every fibre's outputs record about the run (`AFVJob`'s settings)
#[derive(Clone, Debug)]
pub struct Settings {
    /// OUT: strips go to OUT/strips/BUCKET/fiber_ID; None writes no strips (`--no-write`)
    pub out: Option<PathBuf>,
    /// Check A npz directory (`--npz`)
    pub npz: Option<PathBuf>,
    /// The checkpoint source and its training step, as trace.json records them
    pub checkpoint: String,
    pub step: Option<i64>,
    /// What trace.json records as the fibre source (`--afv-ref`)
    pub afv_ref: String,
    /// Trace voxels between decisions
    pub decision_step: f64,
    /// Stitch blend, trace voxels
    pub blend: f64,
    /// End trim, patch voxels
    pub trim: f64,
    pub voxel_size_um: f64,
    /// Trace voxels per patch voxel
    pub trace_per_patch: f64,
}

/// A fibre's index record (`AFVJob.write`/`failed`), before the runner adds `seconds` and `runner`
#[derive(Clone, Debug)]
pub struct Record {
    pub fiber: i64,
    pub family: String,
    pub length_mm: f64,
    pub decisions: usize,
    pub nodes: usize,
    /// Mean expected |error| over kept nodes, patch voxels (with nodes)
    pub expected_error_vox: Option<f64>,
    /// `SeedHeadingError: <reason>` for a fibre that could not run
    pub failed: Option<String>,
}

impl Record {
    /// The record of a fibre whose seed heading failed
    pub fn failed(fibre: &super::fibre::FibreRow, settings: &Settings, reason: &str) -> Self {
        Self {
            fiber: fibre.id,
            family: fibre.family.clone(),
            length_mm: fibre.length * settings.voxel_size_um / 1000.0,
            decisions: 0,
            nodes: 0,
            expected_error_vox: None,
            failed: Some(format!("SeedHeadingError: {reason}")),
        }
    }

    /// The index line `strips_fast.py` writes (no newline): `seconds` rounded to 3 decimals, `runner` the run's tag
    pub fn line(&self, seconds: f64, runner: &str) -> String {
        let mut m = Map::new();
        m.insert("fiber".into(), json!(self.fiber));
        m.insert("family".into(), json!(self.family));
        m.insert("length_mm".into(), json!(self.length_mm));
        m.insert("decisions".into(), json!(self.decisions));
        m.insert("nodes".into(), json!(self.nodes));
        match &self.failed {
            Some(reason) => {
                m.insert("failed".into(), json!(reason));
            }
            None => {
                m.insert("seconds".into(), json!(py_round(seconds, 3)));
            }
        }
        m.insert("runner".into(), json!(runner));
        if let Some(e) = self.expected_error_vox {
            m.insert("expected_error_vox".into(), json!(e));
        }
        py_dumps(&Value::Object(m), JsonStyle::DEFAULT)
    }
}

/// Python's `round(x, n)` for a float: the nearest decimal with `n` digits, as its repr reads back
fn py_round(x: f64, n: usize) -> f64 {
    // Python rounds the exact binary value correctly; formatting to n digits does the same
    format!("{x:.n$}").parse().unwrap_or(x)
}

impl Settings {
    /// The fibre's strip directory
    pub fn strip_dir(&self, id: i64) -> Option<PathBuf> {
        self.out
            .as_ref()
            .map(|o| o.join("strips").join(format!("{:02}", id.rem_euclid(100))).join(format!("fiber_{id}")))
    }

    /// Writes a fibre's outputs from its stitched strip and returns its index record (`AFVJob.write`).
    /// `heights`, `log_scale`, `frames` and `positions` are the decisions' raw results (for the npz)
    #[allow(clippy::too_many_arguments)]
    pub fn write_fibre(
        &self,
        fibre: &Fibre,
        arcs: &[f64],
        strip: &Strip,
        heights: &[[[f32; STRIP_COLS]; STRIP_ROWS]],
        log_scale: &[[[f32; STRIP_COLS]; STRIP_ROWS]],
        frames: &[M3],
        positions: &[V3],
    ) -> Result<Record> {
        let row = &fibre.row;
        let nodes = strip.keep.iter().flatten().filter(|&&k| k).count();
        let mut expected_error_vox = None;
        if nodes > 0 {
            let flips = match self.strip_dir(row.id) {
                Some(dir) => {
                    let display = display_grid(strip, &row.family);
                    self.write_strip(&dir, &display, row.id, &row.family, arcs)?;
                    (display.fiber_reversed, display.across_reversed)
                }
                None => (false, false),
            };
            expected_error_vox = Some(mean_expected(strip, flips));
        }
        if let Some(npz) = &self.npz {
            fs::create_dir_all(npz).wrap_err_with(|| format!("create {}", npz.display()))?;
            self.write_npz(
                &npz.join(format!("fiber_{}.npz", row.id)),
                fibre,
                arcs,
                strip,
                heights,
                log_scale,
                frames,
                positions,
            )?;
        }
        Ok(Record {
            fiber: row.id,
            family: row.family.clone(),
            length_mm: row.length * self.voxel_size_um / 1000.0,
            decisions: arcs.len(),
            nodes,
            expected_error_vox,
            failed: None,
        })
    }

    /// The tifxyz (`write_surface(dir, rows / tpp, keep, *steps, provenance, overwrite=True, voxel_size_um,
    /// mmap_layout=True)`), trace.json and expected_error.tif of a displayed strip
    fn write_strip(&self, dir: &Path, d: &Display, id: i64, family: &str, arcs: &[f64]) -> Result<()> {
        let tpp = self.trace_per_patch;
        let (f, v) = (strip_f(), strip_v());
        let row_step = (f[1] - f[0]) / tpp;
        let column_step = (v[1] - v[0]) / tpp;
        let (a, b) = if d.along_columns { (column_step, row_step) } else { (row_step, column_step) };
        let xyz: Vec<[f32; 3]> = d
            .points
            .iter()
            .zip(&d.keep)
            .map(|(p, &k)| if k { p.map(|x| (x / tpp) as f32) } else { [-1.0; 3] })
            .collect();
        // a kept point is valid unless villa's -1 marker or a non-finite value says otherwise
        let valid: Vec<bool> =
            xyz.iter().map(|p| p.iter().all(|v| v.is_finite()) && p.iter().any(|&v| v != -1.0)).collect();
        let parent = dir.parent().expect("strip directories have a parent");
        fs::create_dir_all(parent).wrap_err_with(|| format!("create {}", parent.display()))?;
        let name = dir.file_name().unwrap_or_default().to_string_lossy().into_owned();
        let existing = fs::read(dir.join("meta.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
            .and_then(|m| m.get("uuid").and_then(Value::as_str).map(str::to_owned))
            .filter(|u| !u.is_empty() && *u != name);
        let uuid = match existing {
            Some(u) => u,
            None => uuid4()?,
        };
        let stage = parent.join(format!(".tifxyz_tmp_{name}.{}.{}", std::process::id(), stamp()));
        fs::create_dir(&stage).wrap_err_with(|| format!("create {}", stage.display()))?;
        let cells = d.rows * d.cols;
        for (k, file) in ["x.tif", "y.tif", "z.tif"].iter().enumerate() {
            let bytes = f32_bytes((0..cells).map(|i| if valid[i] { xyz[i][k] } else { -1.0 }));
            fs::write(stage.join(file), tiff_bytes(&bytes, d.rows, d.cols, Sample::F32))?;
        }
        let mask: Vec<u8> = valid.iter().map(|&v| if v { 255 } else { 0 }).collect();
        fs::write(stage.join("mask.tif"), tiff_bytes(&mask, d.rows, d.cols, Sample::U8))?;
        // meta.json: vc_reader's keys (nlohmann's sorted order), then add_area's, written by Python's json
        let (mut lo, mut hi) = ([0.0f32; 3], [0.0f32; 3]);
        let mut any = false;
        for (p, _) in xyz.iter().zip(&valid).filter(|(_, v)| **v) {
            for k in 0..3 {
                lo[k] = if any { lo[k].min(p[k]) } else { p[k] };
                hi[k] = if any { hi[k].max(p[k]) } else { p[k] };
            }
            any = true;
        }
        let area = surface_area(&xyz, d.rows, d.cols);
        let mut meta = Map::new();
        meta.insert("bbox".into(), json!([lo.map(f64::from), hi.map(f64::from)]));
        meta.insert("format".into(), json!("tifxyz"));
        meta.insert("scale".into(), json!([1.0 / b, 1.0 / a]));
        meta.insert("type".into(), json!("seg"));
        meta.insert("uuid".into(), json!(uuid));
        meta.insert("area_vx2".into(), json!(area));
        meta.insert("area_cm2".into(), json!(area * (self.voxel_size_um * self.voxel_size_um) / 1e8));
        meta.insert("voxel_size_um".into(), json!(self.voxel_size_um));
        fs::write(stage.join("meta.json"), py_dumps(&Value::Object(meta), JsonStyle::indented(2)))?;
        publish(&stage, dir)?;
        let mut trace = Map::new();
        trace.insert("checkpoint".into(), json!(self.checkpoint));
        trace.insert("step".into(), self.step.map_or(Value::Null, |s| json!(s)));
        trace.insert("afv".into(), json!(self.afv_ref));
        trace.insert("fiber".into(), json!(id));
        trace.insert("family".into(), json!(family));
        trace.insert("decision_step".into(), json!(self.decision_step));
        trace.insert("blend".into(), json!(self.blend));
        trace.insert("trim_patch_voxels".into(), json!(self.trim));
        trace.insert("units".into(), json!("patch voxels"));
        trace.insert("trace_per_patch".into(), json!(tpp));
        trace.insert("first_row_arc_patch_voxels".into(), json!((arcs[0] + f[0]) / tpp));
        trace.insert("fiber_along".into(), json!(if d.along_columns { "columns" } else { "rows" }));
        trace.insert("fiber_reversed".into(), json!(d.fiber_reversed));
        trace.insert("across_reversed".into(), json!(d.across_reversed));
        fs::write(dir.join("trace.json"), py_dumps(&Value::Object(trace), JsonStyle::indented(2)))?;
        let expected = f32_bytes(d.expected.iter().zip(&d.keep).map(|(&e, &k)| if k { e as f32 } else { f32::NAN }));
        write_atomic(&dir.join("expected_error.tif"), &tiff_bytes(&expected, d.rows, d.cols, Sample::F32))
    }

    /// Check A's npz (`AFVJob.write_npz`): the stitched strip untransposed and every decision's raw results
    #[allow(clippy::too_many_arguments)]
    fn write_npz(
        &self,
        dest: &Path,
        fibre: &Fibre,
        arcs: &[f64],
        strip: &Strip,
        heights: &[[[f32; STRIP_COLS]; STRIP_ROWS]],
        log_scale: &[[[f32; STRIP_COLS]; STRIP_ROWS]],
        frames: &[M3],
        positions: &[V3],
    ) -> Result<()> {
        let tpp = self.trace_per_patch;
        let f = strip_f();
        let r = strip.rows.len();
        let k = arcs.len();
        let f64s = |v: &mut dyn Iterator<Item = f64>| -> Vec<u8> { v.flat_map(f64::to_le_bytes).collect() };
        let plane = plane(frames, positions, arcs, self.blend);
        let mut z = Npz::default();
        z.add("rows", "<f8", &[r, STRIP_COLS, 3], f64s(&mut strip.rows.iter().flatten().flatten().map(|x| x / tpp)));
        z.add("plane", "<f8", &[r, STRIP_COLS, 3], f64s(&mut plane.iter().flatten().flatten().map(|x| x / tpp)));
        z.add("keep", "|b1", &[r, STRIP_COLS], strip.keep.iter().flatten().map(|&b| u8::from(b)).collect());
        z.add("fiber", "<f8", &[fibre.points.len(), 3], f64s(&mut fibre.points.iter().flatten().map(|x| x / tpp)));
        z.add("fiber_arc", "<f8", &[fibre.s.len()], f64s(&mut fibre.s.iter().map(|x| x / tpp)));
        z.add("arcs", "<f8", &[k], f64s(&mut arcs.iter().copied()));
        z.add("first_row_arc", "<f8", &[], f64s(&mut std::iter::once((arcs[0] + f[0]) / tpp)));
        z.add("row_step", "<f8", &[], f64s(&mut std::iter::once((f[1] - f[0]) / tpp)));
        z.add(
            "heights",
            "<f8",
            &[k, STRIP_ROWS, STRIP_COLS],
            f64s(&mut heights.iter().flatten().flatten().map(|&h| f64::from(h))),
        );
        z.add("frames", "<f8", &[k, 3, 3], f64s(&mut frames.iter().flatten().flatten().copied()));
        z.add("positions", "<f8", &[k, 3], f64s(&mut positions.iter().flatten().copied()));
        z.add("flipped", "|b1", &[k], strip.flipped.iter().map(|&b| u8::from(b)).collect());
        let expected = log_scale.iter().flatten().flatten().map(|&s| super::np::exp_f32(s) / tpp as f32);
        z.add("expected_error_vox", "<f4", &[k, STRIP_ROWS, STRIP_COLS], f32_bytes(expected));
        z.add("seconds", "<f8", &[], f64s(&mut std::iter::once(0.0)));
        z.add("decisions", "<i8", &[], (k as i64).to_le_bytes().to_vec());
        let family: Vec<u8> = fibre.row.family.chars().flat_map(|c| (c as u32).to_le_bytes()).collect();
        z.add("family", &format!("<U{}", fibre.row.family.chars().count()), &[], family);
        write_atomic(dest, &z.finish()?)
    }
}

/// Nanoseconds since the epoch, for unique staging names
fn stamp() -> u128 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos())
}

/// Moves a staged tifxyz directory into place, replacing any existing one (vc_reader's `publishDirectory`)
fn publish(stage: &Path, dir: &Path) -> Result<()> {
    if dir.exists() {
        let old = dir.with_file_name(format!(
            ".tifxyz_old_{}.{}.{}",
            dir.file_name().unwrap_or_default().to_string_lossy(),
            std::process::id(),
            stamp()
        ));
        fs::rename(dir, &old).wrap_err_with(|| format!("move aside {}", dir.display()))?;
        fs::rename(stage, dir).wrap_err_with(|| format!("publish {}", dir.display()))?;
        fs::remove_dir_all(&old).wrap_err_with(|| format!("remove {}", old.display()))?;
    } else {
        fs::rename(stage, dir).wrap_err_with(|| format!("publish {}", dir.display()))?;
    }
    Ok(())
}

/// A `.npz` as `np.savez_compressed` writes it: one deflated `.npy` per array, in order
#[derive(Default)]
struct Npz {
    entries: Vec<(String, Vec<u8>)>,
}

impl Npz {
    /// An array `name` of NumPy dtype `descr`, C order
    fn add(&mut self, name: &str, descr: &str, shape: &[usize], data: Vec<u8>) {
        let shape_repr = match shape {
            [] => "()".to_string(),
            [n] => format!("({n},)"),
            _ => format!("({})", shape.iter().map(usize::to_string).collect::<Vec<_>>().join(", ")),
        };
        let mut header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_repr}, }}");
        // NumPy's room for axis 0 to grow, then padding to 64 bytes with a final newline
        if let Some(&first) = shape.first() {
            header.push_str(&" ".repeat(21usize.saturating_sub(first.to_string().len())));
        }
        let total = 10 + header.len() + 1;
        header.push_str(&" ".repeat((64 - total % 64) % 64));
        header.push('\n');
        let mut npy = Vec::with_capacity(10 + header.len() + data.len());
        npy.extend_from_slice(b"\x93NUMPY\x01\x00");
        npy.extend_from_slice(&(header.len() as u16).to_le_bytes());
        npy.extend_from_slice(header.as_bytes());
        npy.extend_from_slice(&data);
        self.entries.push((format!("{name}.npy"), npy));
    }

    /// The zip archive
    fn finish(self) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        let mut central = Vec::new();
        // 1980-01-01 00:00, as reproducible archives use
        let (time, date) = (0u16, 0x21u16);
        for (name, data) in &self.entries {
            let mut crc = flate2::Crc::new();
            crc.update(data);
            let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
            enc.write_all(data)?;
            let packed = enc.finish()?;
            let offset = out.len() as u32;
            let head = |sig: u32, central: bool| -> Vec<u8> {
                let mut h = Vec::new();
                h.extend_from_slice(&sig.to_le_bytes());
                if central {
                    h.extend_from_slice(&20u16.to_le_bytes());
                }
                h.extend_from_slice(&20u16.to_le_bytes());
                h.extend_from_slice(&0u16.to_le_bytes());
                h.extend_from_slice(&8u16.to_le_bytes());
                h.extend_from_slice(&time.to_le_bytes());
                h.extend_from_slice(&date.to_le_bytes());
                h.extend_from_slice(&crc.sum().to_le_bytes());
                h.extend_from_slice(&(packed.len() as u32).to_le_bytes());
                h.extend_from_slice(&(data.len() as u32).to_le_bytes());
                h.extend_from_slice(&(name.len() as u16).to_le_bytes());
                h.extend_from_slice(&0u16.to_le_bytes());
                if central {
                    // comment length, disk number, internal attributes
                    h.extend_from_slice(&[0; 6]);
                    h.extend_from_slice(&0o644u32.wrapping_shl(16).to_le_bytes());
                    h.extend_from_slice(&offset.to_le_bytes());
                }
                h.extend_from_slice(name.as_bytes());
                h
            };
            out.extend_from_slice(&head(0x04034b50, false));
            out.extend_from_slice(&packed);
            central.extend_from_slice(&head(0x02014b50, true));
        }
        let cd_at = out.len() as u32;
        out.extend_from_slice(&central);
        out.extend_from_slice(&0x06054b50u32.to_le_bytes());
        out.extend_from_slice(&[0; 4]);
        let n = self.entries.len() as u16;
        out.extend_from_slice(&n.to_le_bytes());
        out.extend_from_slice(&n.to_le_bytes());
        out.extend_from_slice(&(central.len() as u32).to_le_bytes());
        out.extend_from_slice(&cd_at.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        Ok(out)
    }
}

/// run.json (`strips_fast.py`, shard 0): the fields in the runner's order, then `provenance` over them, then the
/// runner tag; `json.dumps(..., indent=1)` and a newline
pub fn run_json(fields: &Map<String, Value>, provenance: &Map<String, Value>, runner: &str) -> String {
    let mut m = fields.clone();
    for (k, v) in provenance {
        m.insert(k.clone(), v.clone());
    }
    m.insert("runner".into(), json!(runner));
    let mut s = py_dumps(&Value::Object(m), JsonStyle::indented(1));
    s.push('\n');
    s
}
