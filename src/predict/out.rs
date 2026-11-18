//! The output: villa's probability Zarr v2 group, written as zarr 2.18 writes it, and the column writer
//!
//! `semantic` `(2, Z, Y, X)` in chunks of `(2, 128, 128, 128)` and `tracing` `(2, Z/2, Y/2, X/2)` in chunks of
//! `(2, 64, 64, 64)`, uint8, Blosc zstd level 3 with byte shuffle, fill value 0 and `write_empty_chunks=False`:
//! a chunk left all zero is not stored. Columns of 480 do not align with the chunks, so neighbours share chunks;
//! the writer thread writes one column at a time, then its progress record, so every read-modify-write of a
//! shared chunk is serial, and a progress record never names a column whose data is not written yet

use std::fs;
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{SyncSender, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use color_eyre::eyre::{Result, bail, ensure, eyre};
use serde_json::{Map, Value, json};

use super::contract::Col;
use crate::input::blosc_decompress;
use crate::pyfmt::{JsonStyle, py_dumps};

/// zarr's metadata layout (`zarr.util.json_dumps`)
const ZARR_JSON: JsonStyle = JsonStyle::indented(4).sorted();

/// One `(2, ...)` uint8 array of the output group
#[derive(Clone, Debug)]
pub struct OutArray {
    dir: PathBuf,
    /// Spatial shape and chunk edges; the channel axis is 2 in both
    shape: [usize; 3],
    chunks: [usize; 3],
}

impl OutArray {
    fn create(dir: &Path, shape: [usize; 3], chunk: usize) -> Result<Self> {
        fs::create_dir_all(dir)?;
        let meta = json!({
            "chunks": [2, chunk, chunk, chunk],
            "compressor": {"blocksize": 0, "clevel": 3, "cname": "zstd", "id": "blosc", "shuffle": 1},
            "dtype": "|u1",
            "fill_value": 0,
            "filters": null,
            "order": "C",
            "shape": [2, shape[0], shape[1], shape[2]],
            "zarr_format": 2,
        });
        write_atomic(&dir.join(".zarray"), py_dumps(&meta, ZARR_JSON).as_bytes())?;
        Ok(Self { dir: dir.to_path_buf(), shape, chunks: [chunk; 3] })
    }

    fn open(dir: &Path) -> Result<Self> {
        let meta: Value = serde_json::from_slice(&fs::read(dir.join(".zarray"))?)?;
        let dims = |key: &str| -> Result<Vec<usize>> { Ok(serde_json::from_value(meta[key].clone())?) };
        let (shape, chunks) = (dims("shape")?, dims("chunks")?);
        ensure!(shape.len() == 4 && chunks.len() == 4 && chunks[0] == 2, "{}: not a (2, Z, Y, X) array", dir.display());
        Ok(Self {
            dir: dir.to_path_buf(),
            shape: [shape[1], shape[2], shape[3]],
            chunks: [chunks[1], chunks[2], chunks[3]],
        })
    }

    /// Writes `data`, `(2, dims)` C order, at spatial offset `lo`, as zarr's `array[:, region] = data`
    pub fn write(&self, lo: [usize; 3], dims: [usize; 3], data: &[u8]) -> Result<()> {
        let plane: usize = dims.iter().product();
        ensure!(data.len() == 2 * plane, "{} bytes for a (2, {dims:?}) region", data.len());
        ensure!((0..3).all(|a| lo[a] + dims[a] <= self.shape[a]), "region past the array's end");
        if plane == 0 {
            return Ok(());
        }

        let cs = self.chunks;
        let cplane: usize = cs.iter().product();
        let hi = [0, 1, 2].map(|a| lo[a] + dims[a]);
        for kz in lo[0] / cs[0]..=(hi[0] - 1) / cs[0] {
            for ky in lo[1] / cs[1]..=(hi[1] - 1) / cs[1] {
                for kx in lo[2] / cs[2]..=(hi[2] - 1) / cs[2] {
                    let key = [kz, ky, kx];
                    let origin = [0, 1, 2].map(|a| key[a] * cs[a]);
                    let a0 = [0, 1, 2].map(|a| lo[a].max(origin[a]));
                    let a1 = [0, 1, 2].map(|a| hi[a].min(origin[a] + cs[a]));
                    let name = format!("0.{kz}.{ky}.{kx}");
                    let path = self.dir.join(&name);
                    // a chunk the region covers whole is replaced; a part-covered one is read, updated and
                    // rewritten (`Array._chunk_setitem_nosync`)
                    let whole = (0..3).all(|a| a0[a] == origin[a] && a1[a] == origin[a] + cs[a]);
                    let mut chunk = match (whole, fs::read(&path)) {
                        (true, _) => vec![0u8; 2 * cplane],
                        (false, Ok(bytes)) => {
                            let c = blosc_decompress(&bytes).map_err(|e| eyre!("{}: {e}", path.display()))?;
                            ensure!(
                                c.len() == 2 * cplane,
                                "{}: {} bytes, expected {}",
                                path.display(),
                                c.len(),
                                2 * cplane
                            );
                            c
                        }
                        (false, Err(e)) if e.kind() == ErrorKind::NotFound => vec![0u8; 2 * cplane],
                        (false, Err(e)) => bail!("{}: {e}", path.display()),
                    };
                    let w = a1[2] - a0[2];
                    for c in 0..2 {
                        for z in a0[0]..a1[0] {
                            for y in a0[1]..a1[1] {
                                let dst = c * cplane + ((z - origin[0]) * cs[1] + (y - origin[1])) * cs[2] + a0[2]
                                    - origin[2];
                                let src = c * plane + ((z - lo[0]) * dims[1] + (y - lo[1])) * dims[2] + a0[2] - lo[2];
                                chunk[dst..dst + w].copy_from_slice(&data[src..src + w]);
                            }
                        }
                    }

                    // write_empty_chunks=False: a chunk equal to the fill value everywhere is removed, not stored
                    if chunk.iter().all(|&v| v == 0) {
                        match fs::remove_file(&path) {
                            Err(e) if e.kind() != ErrorKind::NotFound => bail!("{}: {e}", path.display()),
                            _ => {}
                        }
                    } else {
                        write_atomic(&path, &blosc_compress(&chunk)?)?;
                    }
                }
            }
        }
        Ok(())
    }
}

/// numcodecs' `Blosc(cname="zstd", clevel=3, shuffle=SHUFFLE)` encode of uint8 data
fn blosc_compress(src: &[u8]) -> Result<Vec<u8>> {
    let mut out = vec![0u8; src.len() + blosc_src::BLOSC_MAX_OVERHEAD as usize];
    // SAFETY: `out` holds `src.len()` plus Blosc's maximum overhead, the most c-blosc writes; the context call
    // keeps no global state
    let n = unsafe {
        blosc_src::blosc_compress_ctx(
            3,
            blosc_src::BLOSC_SHUFFLE as i32,
            1,
            src.len(),
            src.as_ptr().cast(),
            out.as_mut_ptr().cast(),
            out.len(),
            c"zstd".as_ptr(),
            0,
            1,
        )
    };
    ensure!(n > 0, "Blosc failed to compress a chunk ({n})");
    out.truncate(n as usize);
    Ok(out)
}

/// Writes `path` through a temporary file beside it and a rename, as zarr's `DirectoryStore` does
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let tmp = path.with_file_name(format!(".{name}.{}.partial", std::process::id()));
    fs::write(&tmp, bytes).map_err(|e| eyre!("{}: {e}", tmp.display()))?;
    fs::rename(&tmp, path).map_err(|e| eyre!("{}: {e}", path.display()))?;
    Ok(())
}

/// The output group's two arrays
pub struct Output {
    pub semantic: OutArray,
    pub tracing: OutArray,
}

impl Output {
    /// Opens the group at `out`, creating it with `attrs` when it has no `semantic` array yet; an existing
    /// output must carry the same `inference` provenance, or the run would mix two predictions
    pub fn open_or_create(out: &Path, nz: usize, ny: usize, nx: usize, attrs: &Map<String, Value>) -> Result<Self> {
        fs::create_dir_all(out)?;
        if !out.join(".zgroup").exists() {
            write_atomic(&out.join(".zgroup"), py_dumps(&json!({"zarr_format": 2}), ZARR_JSON).as_bytes())?;
        }
        if !out.join("semantic").join(".zarray").exists() {
            let semantic = OutArray::create(&out.join("semantic"), [nz, ny, nx], 128)?;
            let tracing = OutArray::create(&out.join("tracing"), [nz / 2, ny / 2, nx / 2], 64)?;
            // `attrs.update(...)` merges into what the group already holds
            let mut merged = match fs::read(out.join(".zattrs")) {
                Ok(b) => serde_json::from_slice::<Map<String, Value>>(&b)?,
                Err(_) => Map::new(),
            };
            merged.extend(attrs.clone());
            write_atomic(&out.join(".zattrs"), py_dumps(&Value::Object(merged), ZARR_JSON).as_bytes())?;
            return Ok(Self { semantic, tracing });
        }

        let saved: Value = serde_json::from_slice(&fs::read(out.join(".zattrs"))?)?;
        if !same_json(&saved["inference"], &attrs["inference"]) {
            bail!("output exists with different provenance; use a new output");
        }
        Ok(Self { semantic: OutArray::open(&out.join("semantic"))?, tracing: OutArray::open(&out.join("tracing"))? })
    }
}

/// JSON equality as Python's `==` on the parsed values, where `1 == 1.0`
fn same_json(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64() && (x.is_f64() || y.is_f64() || x == y),
        (Value::Array(x), Value::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same_json(p, q)),
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| same_json(v, w)))
        }
        _ => a == b,
    }
}

/// One finished column for the writer: its arrays (`None` for a column whose tiles were all air, which writes
/// nothing), and its progress record
pub struct Finished {
    pub col: Col,
    pub nz: usize,
    pub sem: Option<Vec<u8>>,
    pub tra: Option<Vec<u8>>,
    pub record: Map<String, Value>,
    pub started: Instant,
}

/// Writes finished columns and then their progress records, in submission order, on one thread
pub struct ColumnWriter {
    tx: Option<SyncSender<Finished>>,
    thread: Option<JoinHandle<()>>,
    err: Arc<Mutex<Option<color_eyre::Report>>>,
}

/// The band reference's record key order, so the records read the same
const RECORD_KEYS: [&str; 9] =
    ["col", "tiles", "wall_s", "attempt", "t_end", "forward_s", "tiles_run", "tiles_air", "voxels_uncovered"];

pub fn now_unix() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
}

impl ColumnWriter {
    pub fn start(output: Output, progress: PathBuf) -> Result<Self> {
        let (tx, rx) = sync_channel::<Finished>(2);
        let err = Arc::new(Mutex::new(None));
        let slot = err.clone();
        let thread = std::thread::Builder::new().name("fibrs-predict-writer".into()).spawn(move || {
            let mut failed = false;
            for job in rx {
                // after a failure the rest is discarded; the main thread raises it
                if failed {
                    continue;
                }
                if let Err(e) = write_column(&output, &progress, job) {
                    failed = true;
                    if let Ok(mut s) = slot.lock() {
                        *s = Some(e);
                    }
                }
            }
        })?;
        Ok(Self { tx: Some(tx), thread: Some(thread), err })
    }

    fn raise(&self) -> Result<()> {
        match self.err.lock().map_err(|_| eyre!("writer error slot poisoned"))?.take() {
            Some(e) => Err(e.wrap_err("fibrs predict column writer failed")),
            None => Ok(()),
        }
    }

    pub fn submit(&self, job: Finished) -> Result<()> {
        self.raise()?;
        let tx = self.tx.as_ref().ok_or_else(|| eyre!("column writer closed"))?;
        tx.send(job).map_err(|_| eyre!("column writer stopped"))?;
        Ok(())
    }

    /// Writes everything submitted, stops the thread and raises any write failure
    pub fn drain(&mut self) -> Result<()> {
        drop(self.tx.take());
        if let Some(t) = self.thread.take() {
            t.join().map_err(|_| eyre!("column writer panicked"))?;
        }
        self.raise()
    }
}

impl Drop for ColumnWriter {
    fn drop(&mut self) {
        let _ = self.drain();
    }
}

fn write_column(output: &Output, progress: &Path, job: Finished) -> Result<()> {
    let [y0, y1, x0, x1] = job.col;
    let (h, w) = (y1 - y0, x1 - x0);
    if let (Some(sem), Some(tra)) = (&job.sem, &job.tra) {
        output.semantic.write([0, y0, x0], [job.nz, h, w], sem)?;
        output.tracing.write([0, y0 / 2, x0 / 2], [job.nz / 2, h / 2, w / 2], tra)?;
    }

    let mut rec = job.record;
    rec.insert("wall_s".into(), json!(job.started.elapsed().as_secs_f64()));
    rec.insert("t_end".into(), json!(now_unix()));
    let ordered: Map<String, Value> =
        RECORD_KEYS.iter().map(|k| (k.to_string(), rec.get(*k).cloned().unwrap_or(Value::Null))).collect();
    let line = py_dumps(&Value::Object(ordered), JsonStyle::DEFAULT);
    let mut f = fs::OpenOptions::new().create(true).append(true).open(progress)?;
    f.write_all(format!("{line}\n").as_bytes())?;
    println!("{line}");
    Ok(())
}
