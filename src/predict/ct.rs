//! The CT input: a level-0 uint8 Zarr v2 array, read by box from many threads at once
//!
//! Each 128^3 tile meets up to eight chunks and each chunk feeds up to eight tiles, so decoded chunks are kept
//! in a small shared cache; tiles run in lattice order, so the chunks a tile needs were mostly decoded for its
//! neighbours

use std::collections::{HashMap, VecDeque};
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use color_eyre::eyre::{Result, bail, ensure, eyre};
use serde_json::Value;

use crate::input::{Compressor, decode};

/// Decoded chunks kept, by size: 1 GiB, which is 512 of the usual 128^3 chunks (a few columns' worth of tile rows)
/// and fewer of larger ones
const CACHE_BYTES: usize = 1 << 30;

type Chunk = Arc<Vec<u8>>;

#[derive(Default)]
struct Cache {
    chunks: HashMap<[usize; 3], Chunk>,
    order: VecDeque<[usize; 3]>,
    bytes: usize,
    /// Bytes of the chunks being decoded, which count against the bound before they land
    decoding: usize,
}

impl Cache {
    /// Evicts the oldest chunks until `extra` more bytes fit, or none are left
    fn make_room(&mut self, extra: usize) {
        while self.bytes + self.decoding + extra > CACHE_BYTES {
            let Some(old) = self.order.pop_front() else { return };
            if let Some(c) = self.chunks.remove(&old) {
                self.bytes -= c.len();
            }
        }
    }
}

/// A decode's share of the cache bound, given back when the decode ends, whether it succeeded or not
struct Decoding<'a>(&'a Mutex<Cache>, usize);

impl Drop for Decoding<'_> {
    fn drop(&mut self) {
        if let Ok(mut cache) = self.0.lock() {
            cache.decoding -= self.1;
        }
    }
}

/// A 3D uint8 Zarr v2 array
pub struct Ct {
    dir: PathBuf,
    pub shape: [usize; 3],
    pub chunks: [usize; 3],
    nested: bool,
    compressor: Compressor,
    fill: u8,
    fortran: bool,
    cache: Mutex<Cache>,
}

/// What `.zarray` says when it isn't a 3D uint8 array: numpy's dtype name and the shape as Python prints it
fn describe(meta: &Value) -> String {
    let dtype = match meta["dtype"].as_str() {
        Some(t) => match t.trim_start_matches(['<', '>', '|', '=']) {
            "u1" => "uint8".into(),
            "u2" => "uint16".into(),
            "u4" => "uint32".into(),
            "i1" => "int8".into(),
            "i2" => "int16".into(),
            "f2" => "float16".into(),
            "f4" => "float32".into(),
            "f8" => "float64".into(),
            other => other.to_string(),
        },
        None => "unknown".into(),
    };
    let dims: Vec<String> = meta["shape"].as_array().into_iter().flatten().map(Value::to_string).collect();
    let shape = if dims.len() == 1 { format!("({},)", dims[0]) } else { format!("({})", dims.join(", ")) };
    format!("{dtype} {shape}")
}

impl Ct {
    pub fn open(dir: &Path) -> Result<Self> {
        let text = fs::read_to_string(dir.join(".zarray"))
            .map_err(|e| eyre!("{}: not a Zarr v2 array ({e})", dir.display()))?;
        let meta: Value = serde_json::from_str(&text)?;
        ensure!(meta["zarr_format"] == 2, "{}: only Zarr v2 arrays are read", dir.display());
        let dims = |key: &str| -> Option<Vec<usize>> { serde_json::from_value(meta[key].clone()).ok() };
        let (shape, chunks) = (dims("shape").unwrap_or_default(), dims("chunks").unwrap_or_default());
        let uint8 = meta["dtype"].as_str().is_some_and(|t| matches!(t, "|u1" | "<u1" | ">u1" | "u1"));
        if !uint8 || shape.len() != 3 || chunks.len() != 3 {
            bail!("expected a 3D uint8 CT array, got {}", describe(&meta));
        }
        ensure!(chunks.iter().all(|&c| c > 0), "{}: Zarr chunks must be positive", dir.display());
        ensure!(meta["filters"].as_array().is_none_or(Vec::is_empty), "{}: Zarr filters are not read", dir.display());
        let fortran = match meta["order"].as_str() {
            Some("C") => false,
            Some("F") => true,
            other => bail!("{}: unsupported Zarr order {other:?}", dir.display()),
        };
        let fill = match &meta["fill_value"] {
            Value::Null => 0,
            v => v.as_u64().filter(|&f| f <= 255).ok_or_else(|| eyre!("bad uint8 fill value {v}"))? as u8,
        };
        Ok(Self {
            dir: dir.to_path_buf(),
            shape: [shape[0], shape[1], shape[2]],
            chunks: [chunks[0], chunks[1], chunks[2]],
            nested: meta.get("dimension_separator").and_then(Value::as_str) == Some("/"),
            compressor: Compressor::of(&meta)?,
            fill,
            fortran,
            cache: Mutex::default(),
        })
    }

    /// The decoded chunk `key`, C order, or the fill value when its file is absent
    fn chunk(&self, key: [usize; 3]) -> Result<Chunk> {
        if let Some(c) = self.cache.lock().map_err(|_| eyre!("CT cache poisoned"))?.chunks.get(&key) {
            return Ok(c.clone());
        }

        let sep = if self.nested { "/" } else { "." };
        let name = format!("{}{sep}{}{sep}{}", key[0], key[1], key[2]);
        let n: usize = self.chunks.iter().product();
        // room for this chunk, and for the others being decoded, before it decodes, so the cache and the
        // decodes in flight together stay within the bound. When they cannot (chunks larger than the bound, or
        // more decodes than fit) the cache empties, and each --readers thread holds only the chunk it decodes;
        // a chunk larger than the bound is then kept alone, as decoding it for every tile costs more than
        // holding it
        let decoding = {
            let mut cache = self.cache.lock().map_err(|_| eyre!("CT cache poisoned"))?;
            cache.make_room(n);
            cache.decoding += n;
            Decoding(&self.cache, n)
        };
        let data = match fs::read(self.dir.join(&name)) {
            Ok(bytes) => {
                let out = decode(self.compressor, bytes, n).map_err(|e| eyre!("CT chunk {name}: {e}"))?;
                ensure!(out.len() == n, "CT chunk {name} decoded to {} bytes, expected {n}", out.len());
                if self.fortran { fortran_to_c(&out, self.chunks) } else { out }
            }
            Err(e) if e.kind() == ErrorKind::NotFound => vec![self.fill; n],
            Err(e) => bail!("{}: {e}", self.dir.join(&name).display()),
        };
        drop(decoding);

        let chunk = Arc::new(data);
        let mut cache = self.cache.lock().map_err(|_| eyre!("CT cache poisoned"))?;
        if cache.chunks.insert(key, chunk.clone()).is_none() {
            cache.order.push_back(key);
            cache.bytes += chunk.len();
        }
        // another reader's chunk may have landed meanwhile; the newest stays
        while cache.bytes + cache.decoding > CACHE_BYTES && cache.order.len() > 1 {
            let old = cache.order.pop_front().expect("non-empty");
            if let Some(c) = cache.chunks.remove(&old) {
                cache.bytes -= c.len();
            }
        }
        Ok(chunk)
    }

    /// The box `[lo, hi)`, C order
    pub fn read(&self, lo: [usize; 3], hi: [usize; 3]) -> Result<Vec<u8>> {
        ensure!((0..3).all(|a| lo[a] < hi[a] && hi[a] <= self.shape[a]), "read {lo:?}..{hi:?} outside the CT");
        let dims = [0, 1, 2].map(|a| hi[a] - lo[a]);
        let mut out = vec![0u8; dims.iter().product()];
        let cs = self.chunks;
        for kz in lo[0] / cs[0]..=(hi[0] - 1) / cs[0] {
            for ky in lo[1] / cs[1]..=(hi[1] - 1) / cs[1] {
                for kx in lo[2] / cs[2]..=(hi[2] - 1) / cs[2] {
                    let key = [kz, ky, kx];
                    let data = self.chunk(key)?;
                    let origin = [0, 1, 2].map(|a| key[a] * cs[a]);
                    let a0 = [0, 1, 2].map(|a| lo[a].max(origin[a]));
                    let a1 = [0, 1, 2].map(|a| hi[a].min(origin[a] + cs[a]));
                    let w = a1[2] - a0[2];
                    for z in a0[0]..a1[0] {
                        for y in a0[1]..a1[1] {
                            let src = ((z - origin[0]) * cs[1] + (y - origin[1])) * cs[2] + (a0[2] - origin[2]);
                            let dst = ((z - lo[0]) * dims[1] + (y - lo[1])) * dims[2] + (a0[2] - lo[2]);
                            out[dst..dst + w].copy_from_slice(&data[src..src + w]);
                        }
                    }
                }
            }
        }
        Ok(out)
    }
}

/// A Fortran-order chunk of shape `dims` in C order
fn fortran_to_c(data: &[u8], dims: [usize; 3]) -> Vec<u8> {
    let mut out = vec![0u8; data.len()];
    for z in 0..dims[0] {
        for y in 0..dims[1] {
            for x in 0..dims[2] {
                out[(z * dims[1] + y) * dims[2] + x] = data[z + dims[0] * (y + dims[1] * x)];
            }
        }
    }
    out
}
