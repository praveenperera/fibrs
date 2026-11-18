//! `--mode villa`: which tiles run and which output chunks are written, as villa's `lasagna.qualzz_inference`
//! does it
//!
//! villa main (97fb9336c) runs AFV V2 over a whole CT array with `lasagna.tiled_predict3d.run_tiled_inference_3d`.
//! It decides what to compute from what the input stores, not from its values:
//!
//! - an output chunk (128^3 on the volume's own lattice) is *supported* when any input chunk inside its
//!   footprint, clipped to the volume, is stored (`_chunk_supported`, tiled_predict3d.py:3317-3341, through
//!   `_input_has_chunks` :841-853)
//! - a tile runs when its input footprint (the 128^3 tile at padded origin p covers volume [p - 16, p + 112),
//!   clipped) has a stored input chunk and some output chunk meeting that footprint is supported (`_tile_work`
//!   :3402-3431, `_needed_chunks` :3343-3369); an all-zero tile over stored chunks still runs
//! - tiles add only into supported chunks and unsupported chunks are never written, so they read as the fill
//!   value 0 (`_accumulate_group` :3433-3466, qualzz_inference.py `ProbabilityZarrOutputAdapter`)
//!
//! villa forwards one tile at a time (the serial and worker paths default `max_batch_size` to 1,
//! tiled_predict3d.py:2880, and qualzz_inference never sets it), so this mode runs batches of one

use std::collections::HashSet;
use std::fs;
use std::path::Path;

use color_eyre::eyre::{Result, eyre};
use serde_json::Value;

use super::contract::{BORDER, Col, Origin, TILE};

pub const CHUNK: usize = 128;
pub const REFERENCE: &str = "villa lasagna.qualzz_inference @ 97fb9336c (batch 1, villa's storage-based tile skip)";
pub const SKIP_RULE: &str = "villa: a tile runs when its input footprint holds a stored input chunk and meets a \
                             supported output chunk; unsupported output chunks are not written";

/// The input array's chunk shape and the indices of its stored chunks, from the chunk files on disk
fn stored_chunks(ct: &Path) -> Result<([usize; 3], HashSet<[usize; 3]>)> {
    let meta: Value = serde_json::from_slice(&fs::read(ct.join(".zarray"))?)?;
    let chunks: Vec<usize> = serde_json::from_value(meta["chunks"].clone())?;
    let chunks: [usize; 3] = chunks.try_into().map_err(|_| eyre!("{}: not a 3D array", ct.display()))?;
    let mut stored = HashSet::new();
    let mut add = |parts: &[&str]| {
        if let [a, b, c] = parts
            && let (Ok(a), Ok(b), Ok(c)) = (digits(a), digits(b), digits(c))
        {
            stored.insert([a, b, c]);
        }
    };
    if meta.get("dimension_separator").and_then(Value::as_str) == Some("/") {
        walk(ct, ct, &mut |rel| add(&rel.split('/').collect::<Vec<_>>()))?;
    } else {
        for entry in fs::read_dir(ct)? {
            let path = entry?.path();
            if path.is_file() {
                let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                add(&name.split('.').collect::<Vec<_>>());
            }
        }
    }
    Ok((chunks, stored))
}

/// A chunk index part: ASCII digits only, as Python's `str.isdigit` accepts for these names
fn digits(s: &str) -> Result<usize, ()> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return Err(());
    }
    s.parse().map_err(|_| ())
}

/// Calls `f` with the `/`-joined path below `root` of every file under `dir`
fn walk(root: &Path, dir: &Path, f: &mut dyn FnMut(&str)) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            walk(root, &path, f)?;
        } else if path.is_file() {
            let rel = path.strip_prefix(root)?;
            let parts: Vec<String> = rel.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect();
            f(&parts.join("/"));
        }
    }
    Ok(())
}

/// villa's supported output chunks and running tiles for one whole-volume input
pub struct VillaSelection {
    shape: [usize; 3],
    in_chunks: [usize; 3],
    stored: HashSet<[usize; 3]>,
    grid: [usize; 3],
    supported: Vec<bool>,
}

impl VillaSelection {
    pub fn new(ct: &Path, shape: [usize; 3]) -> Result<Self> {
        let (in_chunks, stored) = stored_chunks(ct)?;
        let grid = shape.map(|n| n.div_ceil(CHUNK));
        let mut sel = Self { shape, in_chunks, stored, grid, supported: Vec::new() };
        let mut supported = Vec::with_capacity(grid.iter().product());
        for cz in 0..grid[0] {
            for cy in 0..grid[1] {
                for cx in 0..grid[2] {
                    let lo = [cz, cy, cx].map(|c| c * CHUNK);
                    let hi = [0, 1, 2].map(|a| (lo[a] + CHUNK).min(shape[a]));
                    supported.push(sel.has_input(lo, hi));
                }
            }
        }
        sel.supported = supported;
        Ok(sel)
    }

    fn supported_at(&self, c: [usize; 3]) -> bool {
        self.supported[(c[0] * self.grid[1] + c[1]) * self.grid[2] + c[2]]
    }

    /// Whether any stored input chunk meets the half-open region `[lo, hi)` (`_input_has_chunks`)
    fn has_input(&self, lo: [usize; 3], hi: [usize; 3]) -> bool {
        let c = self.in_chunks;
        let r = [0, 1, 2].map(|a| lo[a] / c[a]..hi[a].div_ceil(c[a]));
        r[0].clone().any(|z| r[1].clone().any(|y| r[2].clone().any(|x| self.stored.contains(&[z, y, x]))))
    }

    /// Whether villa runs the tile at padded origin `origin`
    pub fn tile_runs(&self, origin: Origin) -> bool {
        let lo = origin.map(|p| p.saturating_sub(BORDER));
        let hi = [0, 1, 2].map(|a| (origin[a] as i64 - BORDER as i64 + TILE as i64).min(self.shape[a] as i64));
        if (0..3).any(|a| hi[a] <= lo[a] as i64) {
            return false;
        }

        let hi = hi.map(|h| h as usize);
        if !self.has_input(lo, hi) {
            return false;
        }
        let r = [0, 1, 2].map(|a| lo[a] / CHUNK..=(hi[a] - 1) / CHUNK);
        r[0].clone().any(|z| r[1].clone().any(|y| r[2].clone().any(|x| self.supported_at([z, y, x]))))
    }

    /// Zero, in place, a column's semantic `(2, Z, y1-y0, x1-x0)` and tracing `(2, Z/2, (y1-y0)/2, (x1-x0)/2)`
    /// values in unsupported chunks
    pub fn mask_column(&self, col: Col, sem: &mut [u8], tra: &mut [u8]) {
        let [y0, y1, x0, x1] = col;
        let (nz, h, w) = (self.shape[0], y1 - y0, x1 - x0);
        let (tz, th, tw) = (nz / 2, h / 2, w / 2);
        let half = CHUNK / 2;
        for cz in 0..self.grid[0] {
            for cy in 0..self.grid[1] {
                for cx in 0..self.grid[2] {
                    if self.supported_at([cz, cy, cx]) {
                        continue;
                    }
                    let (ys, ye) = (y0.max(cy * CHUNK), y1.min((cy + 1) * CHUNK));
                    let (xs, xe) = (x0.max(cx * CHUNK), x1.min((cx + 1) * CHUNK));
                    if ye <= ys || xe <= xs {
                        continue;
                    }
                    let zr = cz * CHUNK..((cz + 1) * CHUNK).min(nz);
                    zero_box(sem, [nz, h, w], zr, ys - y0..ye - y0, xs - x0..xe - x0);
                    // the tracing chunk of a native chunk is the same index at half size; columns start on even
                    // rows
                    let (tys, tye) = ((y0 / 2).max(cy * half), (y0 / 2 + th).min((cy + 1) * half));
                    let (txs, txe) = ((x0 / 2).max(cx * half), (x0 / 2 + tw).min((cx + 1) * half));
                    if tye > tys && txe > txs {
                        let zr = (cz * half).min(tz)..((cz + 1) * half).min(tz);
                        zero_box(tra, [tz, th, tw], zr, tys - y0 / 2..tye - y0 / 2, txs - x0 / 2..txe - x0 / 2);
                    }
                }
            }
        }
    }
}

/// Zero `[z, y, x]` ranges of both channels of a `(2, dims)` C-order array
fn zero_box(
    a: &mut [u8],
    dims: [usize; 3],
    z: std::ops::Range<usize>,
    y: std::ops::Range<usize>,
    x: std::ops::Range<usize>,
) {
    let plane = dims.iter().product::<usize>();
    for c in 0..2 {
        for zz in z.clone() {
            for yy in y.clone() {
                let row = c * plane + (zz * dims[1] + yy) * dims[2];
                a[row + x.start..row + x.end].fill(0);
            }
        }
    }
}
