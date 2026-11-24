//! The CT as villa's strip runners read it: a uint8 Zarr v2 level in vc_reader's disk cache
//!
//! vc_reader stores every chunk it fetched decoded, at `<cache>/s3/<bucket>/<key>/<level>/<z>/<y>/<x>` (the
//! `dimension_separator` "/" layout), with an empty `<x>.missing` file for a chunk the source does not hold, which
//! reads as the fill value. STRIPS-SCALE's `prefetch_ct.py` fills the cache before a scroll's run. Chunks are
//! memory-mapped, so threads share the page cache and nothing is copied until a crop reads it

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use color_eyre::eyre::{Result, WrapErr, bail, ensure, eyre};
use memmap2::Mmap;
use serde_json::Value;

/// One chunk: its bytes, or `None` where the source holds none (every voxel the fill value)
pub type Chunk = Option<Arc<Mmap>>;

/// A uint8 CT level in a vc_reader disk cache
pub struct Volume {
    dir: PathBuf,
    shape: [usize; 3],
    chunks: [usize; 3],
    fill: u8,
    open: RwLock<HashMap<[usize; 3], Chunk>>,
}

impl Volume {
    /// Opens level `level` of `url` (`s3://bucket/key.zarr`) in the cache at `cache_dir`
    pub fn open(cache_dir: &Path, url: &str, level: u32) -> Result<Self> {
        let key = url.strip_prefix("s3://").ok_or_else(|| eyre!("CT {url}: only s3:// volumes are supported"))?;
        let dir = cache_dir.join("s3").join(key.trim_end_matches('/')).join(level.to_string());
        let zarray = dir.join(".zarray");
        let meta: Value =
            serde_json::from_slice(&fs::read(&zarray).wrap_err_with(|| format!("read {}", zarray.display()))?)
                .wrap_err_with(|| format!("parse {}", zarray.display()))?;
        let dims = |field: &str| -> Result<[usize; 3]> {
            let v: Vec<usize> = meta[field]
                .as_array()
                .ok_or_else(|| eyre!("{}: no {field}", zarray.display()))?
                .iter()
                .map(|x| x.as_u64().map(|x| x as usize).ok_or_else(|| eyre!("{}: bad {field}", zarray.display())))
                .collect::<Result<_>>()?;
            v.try_into().map_err(|_| eyre!("{}: {field} is not 3-D", zarray.display()))
        };
        ensure!(meta["dtype"] == "|u1", "{}: dtype {}, expected |u1", zarray.display(), meta["dtype"]);
        ensure!(meta["compressor"].is_null(), "{}: the cache holds decoded chunks only", zarray.display());
        ensure!(meta["order"] == "C" && meta["dimension_separator"] == "/", "{}: unexpected layout", zarray.display());
        let fill = meta["fill_value"].as_u64().unwrap_or(0) as u8;
        Ok(Self { shape: dims("shape")?, chunks: dims("chunks")?, fill, dir, open: RwLock::new(HashMap::new()) })
    }

    /// The level's shape, zyx
    pub fn shape(&self) -> [usize; 3] {
        self.shape
    }

    /// The chunk shape, zyx
    pub fn chunk_shape(&self) -> [usize; 3] {
        self.chunks
    }

    /// The fill value of absent chunks
    pub fn fill(&self) -> u8 {
        self.fill
    }

    /// Chunk `c` (chunk-grid zyx), mapped on first use; an error when the cache holds neither it nor its
    /// missing marker
    pub fn chunk(&self, c: [usize; 3]) -> Result<Chunk> {
        if let Some(hit) = self.open.read().unwrap().get(&c) {
            return Ok(hit.clone());
        }
        let path = self.dir.join(c[0].to_string()).join(c[1].to_string()).join(c[2].to_string());
        let chunk = match fs::File::open(&path) {
            Ok(file) => {
                // SAFETY: cache files are written once, by rename, and never modified in place
                let map = unsafe { Mmap::map(&file) }.wrap_err_with(|| format!("map {}", path.display()))?;
                let want = self.chunks.iter().product::<usize>();
                ensure!(map.len() == want, "{}: {} bytes, expected {want}", path.display(), map.len());
                Some(Arc::new(map))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let mut marker = path.clone().into_os_string();
                marker.push(".missing");
                if !Path::new(&marker).exists() {
                    bail!("CT chunk {c:?} is not in the disk cache {} (prefetch the scroll first)", self.dir.display());
                }
                None
            }
            Err(e) => return Err(e).wrap_err_with(|| format!("open {}", path.display())),
        };
        self.open.write().unwrap().insert(c, chunk.clone());
        Ok(chunk)
    }

    /// Reads chunk `c` (chunk-grid zyx) into `out`, which holds one chunk; false when the source holds none (the
    /// cache has its missing marker), an error when the cache holds neither
    ///
    /// Unlike [`Volume::chunk`] this keeps no mapping open, so a run's device cache can stream every chunk of a
    /// scroll through it without growing the process's map count
    pub fn read_chunk(&self, c: [usize; 3], out: &mut [u8]) -> Result<bool> {
        use std::io::Read;

        let want = self.chunks.iter().product::<usize>();
        ensure!(out.len() == want, "read_chunk: output of {} bytes for a {want}-byte chunk", out.len());
        let path = self.dir.join(c[0].to_string()).join(c[1].to_string()).join(c[2].to_string());
        let mut file = match fs::File::open(&path) {
            Ok(file) => file,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let mut marker = path.clone().into_os_string();
                marker.push(".missing");
                if !Path::new(&marker).exists() {
                    bail!("CT chunk {c:?} is not in the disk cache {} (prefetch the scroll first)", self.dir.display());
                }
                return Ok(false);
            }
            Err(e) => return Err(e).wrap_err_with(|| format!("open {}", path.display())),
        };
        let len = file.metadata().wrap_err_with(|| format!("stat {}", path.display()))?.len();
        ensure!(len == want as u64, "{}: {len} bytes, expected {want}", path.display());
        file.read_exact(out).wrap_err_with(|| format!("read {}", path.display()))?;
        Ok(true)
    }

    /// Copies the box `start .. start + size` (zyx, inside the level) into `out`, zyx C order
    pub fn read_region(&self, start: [usize; 3], size: [usize; 3], out: &mut [u8]) -> Result<()> {
        ensure!(out.len() == size.iter().product::<usize>(), "read_region: output of {} for {size:?}", out.len());
        for a in 0..3 {
            ensure!(start[a] + size[a] <= self.shape[a], "read_region {start:?}+{size:?} outside {:?}", self.shape);
        }
        let [cz, cy, cx] = self.chunks;
        let end = [start[0] + size[0], start[1] + size[1], start[2] + size[2]];
        for gz in start[0] / cz..end[0].div_ceil(cz) {
            for gy in start[1] / cy..end[1].div_ceil(cy) {
                for gx in start[2] / cx..end[2].div_ceil(cx) {
                    let chunk = self.chunk([gz, gy, gx])?;
                    let lo = [start[0].max(gz * cz), start[1].max(gy * cy), start[2].max(gx * cx)];
                    let hi = [end[0].min((gz + 1) * cz), end[1].min((gy + 1) * cy), end[2].min((gx + 1) * cx)];
                    for z in lo[0]..hi[0] {
                        for y in lo[1]..hi[1] {
                            let o = ((z - start[0]) * size[1] + (y - start[1])) * size[2] + (lo[2] - start[2]);
                            let dst = &mut out[o..o + hi[2] - lo[2]];
                            match &chunk {
                                Some(map) => {
                                    let s = ((z - gz * cz) * cy + (y - gy * cy)) * cx + (lo[2] - gx * cx);
                                    dst.copy_from_slice(&map[s..s + hi[2] - lo[2]]);
                                }
                                None => dst.fill(self.fill),
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A region across chunk boundaries reads the same bytes as the chunks it spans, and an absent chunk reads as
    /// the fill value
    #[test]
    fn region_reads_across_chunks_and_missing_markers() {
        let root = std::env::temp_dir().join(format!("strips-ct-{}", std::process::id()));
        let level = root.join("s3/bucket/v.zarr/0");
        fs::create_dir_all(&level).unwrap();
        fs::write(
            level.join(".zarray"),
            r#"{"chunks":[4,4,4],"compressor":null,"dimension_separator":"/","dtype":"|u1","fill_value":7,"filters":null,"order":"C","shape":[8,8,6],"zarr_format":2}"#,
        )
        .unwrap();
        let value = |z: usize, y: usize, x: usize| (z * 64 + y * 8 + x) as u8;
        for (gz, gy, gx) in [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1)] {
            let dir = level.join(gz.to_string()).join(gy.to_string());
            fs::create_dir_all(&dir).unwrap();
            let mut bytes = vec![0u8; 64];
            for z in 0..4 {
                for y in 0..4 {
                    for x in 0..4 {
                        bytes[(z * 4 + y) * 4 + x] = value(gz * 4 + z, gy * 4 + y, gx * 4 + x);
                    }
                }
            }
            fs::write(dir.join(gx.to_string()), bytes).unwrap();
        }
        fs::write(level.join("1/0/1.missing"), b"").unwrap();
        let vol = Volume::open(&root, "s3://bucket/v.zarr", 0).unwrap();
        let mut out = vec![0u8; 5 * 6 * 5];
        vol.read_region([2, 1, 1], [5, 6, 5], &mut out).unwrap();
        for z in 0..5 {
            for y in 0..6 {
                for x in 0..5 {
                    let (gz, gy, gx) = ((z + 2) / 4, (y + 1) / 4, (x + 1) / 4);
                    let want = if (gz, gy, gx) == (1, 0, 1) { 7 } else { value(z + 2, y + 1, x + 1) };
                    assert_eq!(out[(z * 6 + y) * 5 + x], want, "{z} {y} {x}");
                }
            }
        }
        fs::remove_dir_all(level.join("0/0/0")).ok();
        fs::remove_file(level.join("0/0/0")).ok();
        assert!(Volume::open(&root, "s3://bucket/v.zarr", 0).unwrap().chunk([0, 0, 0]).is_err());
        fs::remove_dir_all(&root).unwrap();
    }
}
