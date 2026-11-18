//! The output contract: how a CT is tiled, read and recorded, as villa `lasagna/qualzz_inference.py` @ 7e6b82164
//! does it
//!
//! Tiles of 128 on a lattice of stride 96 over the input padded by a border of 16, reflect padding at its edges,
//! and a blend ramp that is 0 on the 16-voxel border and 1 on the 96-voxel core (the engine applies it). These
//! are the band reference's functions (fibrs 0.2's Python `fibrs.predict.contract`), ported so band mode cannot
//! drift from it in tiling, air skipping or provenance

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use color_eyre::eyre::{Result, bail, ensure, eyre};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::ct::Ct;
use crate::pyfmt::{JsonStyle, py_dumps};

pub const TILE: usize = 128;
pub const OVERLAP: usize = 32;
pub const BORDER: usize = 16;
pub const STRIDE: usize = TILE - OVERLAP;
/// Equal to the stride, so the cores tile each axis with only the last, shifted one overlapping
pub const CORE: usize = TILE - 2 * BORDER;
/// Voxels in one tile
pub const TILE_VOXELS: usize = TILE * TILE * TILE;
pub const MODEL_FILES: [&str; 4] = ["model.py", "plans.json", "dataset.json", "checkpoint_final.pth"];
pub const REFERENCE: &str = "villa lasagna/qualzz_inference.py @ 7e6b82164";
pub const DROP_IN_FOR: &str = "AFV v2 reference predict.py @ 498ca259 (sha256 12784d48d98e\u{2026}ce916)";

/// A tile's origin on the padded lattice, `(pz, py, px)`
pub type Origin = [usize; 3];

/// An xy column of the output, `[y0, y1, x0, x1]`
pub type Col = [usize; 4];

/// Tile origins in padded coordinates (the reference `_build_tile_positions` over `size + 2 * border`)
pub fn positions(size: usize) -> Vec<usize> {
    let padded = size + 2 * BORDER;
    if padded <= TILE {
        return vec![0];
    }

    let mut out: Vec<usize> = (0..=padded - TILE).step_by(STRIDE).collect();
    if out.last() != Some(&(padded - TILE)) {
        out.push(padded - TILE);
    }
    out
}

/// Volume coordinates covered by the nonzero blend weights of the tile at padded origin `p`
pub fn core_range(p: usize) -> (usize, usize) {
    (p, p + CORE)
}

/// The xy columns of a band, each a `[y0, y1) x [x0, x1)` output region with the tiles whose core meets it
#[derive(Debug)]
pub struct Columns {
    py: Vec<usize>,
    px: Vec<usize>,
    pub cols: Vec<Col>,
}

impl Columns {
    pub fn new(ny: usize, nx: usize, size: usize) -> Result<Self> {
        // a multiple of the core puts every lattice tile in exactly one column (only the shifted last one
        // straddles), so no tile runs twice; even keeps the tracing pooling aligned
        if size == 0 || !size.is_multiple_of(CORE) || !size.is_multiple_of(2) {
            bail!("column size must be an even multiple of {CORE}");
        }

        let cols = (0..ny)
            .step_by(size)
            .flat_map(|y0| (0..nx).step_by(size).map(move |x0| [y0, (y0 + size).min(ny), x0, (x0 + size).min(nx)]))
            .collect();
        Ok(Self { py: positions(ny), px: positions(nx), cols })
    }

    /// The tiles whose core meets `col`, z outermost, in the reference's order
    pub fn tiles(&self, col: Col, pz_all: &[usize]) -> Vec<Origin> {
        let [y0, y1, x0, x1] = col;
        let meets = |p: &&usize, lo: usize, hi: usize| core_range(**p).0 < hi && core_range(**p).1 > lo;
        let ys: Vec<usize> = self.py.iter().filter(|p| meets(p, y0, y1)).copied().collect();
        let xs: Vec<usize> = self.px.iter().filter(|p| meets(p, x0, x1)).copied().collect();
        let mut out = Vec::with_capacity(pz_all.len() * ys.len() * xs.len());
        for &pz in pz_all {
            for &py in &ys {
                for &px in &xs {
                    out.push([pz, py, px]);
                }
            }
        }
        out
    }
}

/// Where tile voxel `i` reads from on an axis read as `[a, a + len)`: numpy's `np.pad(mode="reflect")` of the
/// read, mirrored about its first and last voxel without repeating them, repeatedly when the pad is longer
fn reflect(i: i64, len: usize) -> usize {
    if len == 1 {
        return 0;
    }

    let period = 2 * (len as i64 - 1);
    let m = i.rem_euclid(period);
    (if m >= len as i64 { period - m } else { m }) as usize
}

/// The `TILE^3` uint8 input of the tile at padded origin `origin` into `out`: the band `[z0, z1)` is the
/// volume, and outside it the read is reflected (the reference `read_tile`)
pub fn read_tile(ct: &Ct, z0: usize, z1: usize, origin: Origin, out: &mut [u8]) -> Result<()> {
    let shape = [z1 - z0, ct.shape[1], ct.shape[2]];
    let lo = origin.map(|p| p as i64 - BORDER as i64);
    let a = [0, 1, 2].map(|i| lo[i].max(0) as usize);
    let b = [0, 1, 2].map(|i| ((lo[i] + TILE as i64) as usize).min(shape[i]));
    let raw = ct.read([a[0] + z0, a[1], a[2]], [b[0] + z0, b[1], b[2]])?;
    if (0..3).all(|i| lo[i] >= 0 && b[i] - a[i] == TILE) {
        out.copy_from_slice(&raw);
        return Ok(());
    }

    let len = [0, 1, 2].map(|i| b[i] - a[i]);
    // per axis, the read voxel each tile voxel takes
    let map = |axis: usize| -> Vec<usize> {
        (0..TILE as i64).map(|t| reflect(lo[axis] + t - a[axis] as i64, len[axis])).collect()
    };
    let (mz, my, mx) = (map(0), map(1), map(2));
    for (tz, &rz) in mz.iter().enumerate() {
        for (ty, &ry) in my.iter().enumerate() {
            let src = &raw[(rz * len[1] + ry) * len[2]..][..len[2]];
            let dst = &mut out[(tz * TILE + ty) * TILE..][..TILE];
            for (d, &rx) in dst.iter_mut().zip(&mx) {
                *d = src[rx];
            }
        }
    }
    Ok(())
}

/// sha256 of a file, hex
pub fn sha256(path: &Path) -> Result<String> {
    let mut f = fs::File::open(path).map_err(|e| eyre!("{}: {e}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex(&h.finalize()))
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Every file under `dir`, recursively, as its path components below `root`
fn files_below(root: &Path, dir: &Path, out: &mut Vec<(Vec<String>, PathBuf)>) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            files_below(root, &path, out)?;
        } else if path.is_file() {
            let rel = path.strip_prefix(root)?;
            let parts: Vec<String> = rel.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect();
            if chunk_row(&parts).is_some() {
                out.push((parts, path));
            }
        }
    }
    Ok(())
}

/// The z row of a 3D chunk key's parts (`["z.y.x"]` flat, `["z", "y", "x"]` nested); `None` for any other file
fn chunk_row(parts: &[String]) -> Option<usize> {
    let indices: Vec<&str> = match parts {
        [name] => name.split('.').collect(),
        _ => parts.iter().map(String::as_str).collect(),
    };
    let digits = |s: &&str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    if indices.len() != 3 || !indices.iter().all(digits) {
        return None;
    }
    indices[0].parse().ok()
}

/// sha256 of the CT array's `.zarray` and the names and sizes of its chunk files in the band's chunk rows; other
/// files in the array (a README, say) are not part of it. Chunk contents are not read, which would mean reading
/// the whole band before every run, so an in-place edit that keeps chunk sizes is not detected; a changed CT
/// needs a fresh output
pub fn ct_identity(ct: &Path, z0: usize, z1: usize) -> Result<String> {
    let zarray = fs::read(ct.join(".zarray"))?;
    let meta: Value = serde_json::from_slice(&zarray)?;
    let cz = meta["chunks"][0].as_u64().ok_or_else(|| eyre!("{}: no chunk shape", ct.display()))? as usize;
    ensure!(z0 < z1 && cz > 0, "{}: an empty band ({z0}..{z1}) has no identity", ct.display());
    let rows = z0 / cz..=(z1 - 1) / cz;
    let mut files = Vec::new();
    if meta.get("dimension_separator").and_then(Value::as_str) == Some("/") {
        // nested: one directory per chunk row, so only the band's rows are listed
        for r in rows {
            let d = ct.join(r.to_string());
            if d.exists() {
                files_below(ct, &d, &mut files)?;
            }
        }
    } else {
        for entry in fs::read_dir(ct)? {
            let path = entry?.path();
            let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            if !path.is_file() {
                continue;
            }
            let parts = vec![name];
            if chunk_row(&parts).is_some_and(|row| rows.contains(&row)) {
                files.push((parts, path));
            }
        }
    }

    // pathlib orders paths by their parts
    files.sort();
    let mut h = Sha256::new();
    h.update(&zarray);
    for (parts, path) in files {
        h.update(format!("{} {}\n", parts.join("/"), fs::metadata(&path)?.len()).as_bytes());
    }
    Ok(hex(&h.finalize()))
}

/// sha256 of a prediction's attrs and its progress file: changes whenever the prediction does
pub fn prediction_identity(out: &Path) -> Result<String> {
    let attrs: Value = serde_json::from_slice(&fs::read(out.join(".zattrs"))?)?;
    let mut h = Sha256::new();
    h.update(py_dumps(&attrs, JsonStyle::DEFAULT.sorted()).as_bytes());
    let prog = out.join("progress.jsonl");
    if prog.exists() {
        h.update(fs::read(prog)?);
    }
    Ok(hex(&h.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_chunk_keys_count_toward_the_ct_identity() {
        let parts = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(chunk_row(&parts(&["12.3.4"])), Some(12));
        assert_eq!(chunk_row(&parts(&["7", "0", "1"])), Some(7));
        for other in [&["README"][..], &["_SUCCESS"], &[".zarray"], &["1.2"], &["1.2.x"], &["0", "1", "2", "3"]] {
            assert_eq!(chunk_row(&parts(other)), None, "{other:?}");
        }
    }

    #[test]
    fn reflect_is_numpy_reflect() {
        // np.pad(np.arange(3), (5, 5), mode="reflect")
        let got: Vec<usize> = (-5..8).map(|i| reflect(i, 3)).collect();
        assert_eq!(got, [1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1]);
        assert_eq!(reflect(-3, 1), 0);
    }

    #[test]
    fn positions_cover_the_padded_axis() {
        assert_eq!(positions(1), [0]);
        assert_eq!(positions(96), [0]);
        assert_eq!(positions(97), [0, 1]);
        assert_eq!(positions(300), [0, 96, 192, 204]);
    }
}
