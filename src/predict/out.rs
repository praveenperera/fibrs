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

    /// Writes `data`, `(2, dims)` C order, at spatial offset `lo`, as zarr's `array[:, region] = data`, through
    /// `bufs`
    pub fn write(&self, lo: [usize; 3], dims: [usize; 3], data: &[u8], bufs: &mut ChunkBufs) -> Result<()> {
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
                    // whatever the last chunk left in the buffer
                    bufs.chunk.resize(2 * cplane, 0);
                    let chunk = bufs.chunk.as_mut_slice();
                    match (whole, fs::read(&path)) {
                        // the copy below overwrites every byte
                        (true, _) => {}
                        (false, Ok(bytes)) => {
                            blosc_decompress_into(&bytes, chunk).map_err(|e| eyre!("{}: {e}", path.display()))?
                        }
                        (false, Err(e)) if e.kind() == ErrorKind::NotFound => chunk.fill(0),
                        (false, Err(e)) => bail!("{}: {e}", path.display()),
                    }
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
                        write_atomic(&path, blosc_compress_into(chunk, &mut bufs.packed)?)?;
                    }
                }
            }
        }
        Ok(())
    }
}

/// The writer's chunk buffers, kept across chunks and columns: a fresh 4 MiB buffer per chunk meant zeroing it
/// and faulting its pages in, hundreds of times a column
#[derive(Default)]
pub struct ChunkBufs {
    chunk: Vec<u8>,
    packed: Vec<u8>,
}

/// Decodes one Blosc frame into `out`, which must be exactly its decoded size
fn blosc_decompress_into(src: &[u8], out: &mut [u8]) -> Result<()> {
    let mut nbytes = 0usize;
    // SAFETY: c-blosc reads at most `src.len()` bytes of the header to validate it
    let valid = unsafe { blosc_src::blosc_cbuffer_validate(src.as_ptr().cast(), src.len(), &mut nbytes) };
    ensure!(valid == 0, "not a valid Blosc frame");
    ensure!(nbytes == out.len(), "{nbytes} bytes, expected {}", out.len());
    // SAFETY: the frame was validated against `src.len()`, and c-blosc writes at most `nbytes` bytes, the length of
    // `out`; the context call keeps no global state
    let n = unsafe { blosc_src::blosc_decompress_ctx(src.as_ptr().cast(), out.as_mut_ptr().cast(), nbytes, 1) };
    ensure!(n >= 0 && n as usize == nbytes, "Blosc frame failed to decode ({n})");
    Ok(())
}

/// numcodecs' `Blosc(cname="zstd", clevel=3, shuffle=SHUFFLE)` encode of uint8 data, into `out`'s storage
fn blosc_compress_into<'a>(src: &[u8], out: &'a mut Vec<u8>) -> Result<&'a [u8]> {
    out.clear();
    out.reserve(src.len() + blosc_src::BLOSC_MAX_OVERHEAD as usize);
    // SAFETY: `out` has room for `src.len()` plus Blosc's maximum overhead, the most c-blosc writes; the context
    // call keeps no global state
    let n = unsafe {
        blosc_src::blosc_compress_ctx(
            3,
            blosc_src::BLOSC_SHUFFLE as i32,
            1,
            src.len(),
            src.as_ptr().cast(),
            out.as_mut_ptr().cast(),
            out.capacity(),
            c"zstd".as_ptr(),
            0,
            1,
        )
    };
    ensure!(n > 0, "Blosc failed to compress a chunk ({n})");
    // SAFETY: c-blosc reported writing `n` bytes
    unsafe { out.set_len(n as usize) };
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
    /// Opens the group at `out`, creating it with `attrs` when it is not yet created; an existing output must
    /// carry the same `inference` provenance, or the run would mix two predictions, and the same other `attrs`, or
    /// its metadata would describe another CT
    ///
    /// The group's `inference` attribute is written last, so it marks a finished creation. A group without it is
    /// created again only while it holds no prediction yet (no chunks and no progress), as a run stopped part way
    /// through creating it leaves; anything else would put this run's provenance on data it did not write
    pub fn open_or_create(out: &Path, nz: usize, ny: usize, nx: usize, attrs: &Map<String, Value>) -> Result<Self> {
        fs::create_dir_all(out)?;
        // `attrs.update(...)` merges into what the group already holds
        let mut saved = match fs::read(out.join(".zattrs")) {
            Ok(b) => serde_json::from_slice::<Map<String, Value>>(&b)?,
            Err(e) if e.kind() == ErrorKind::NotFound => Map::new(),
            Err(e) => bail!("{}: {e}", out.join(".zattrs").display()),
        };
        if !saved.contains_key("inference") {
            if holds_prediction(out)? {
                bail!("{} holds a prediction without fibrs predict's provenance; use a new output", out.display());
            }
            if !out.join(".zgroup").exists() {
                write_atomic(&out.join(".zgroup"), py_dumps(&json!({"zarr_format": 2}), ZARR_JSON).as_bytes())?;
            }
            let semantic = OutArray::create(&out.join("semantic"), [nz, ny, nx], 128)?;
            let tracing = OutArray::create(&out.join("tracing"), [nz / 2, ny / 2, nx / 2], 64)?;
            saved.extend(attrs.clone());
            write_atomic(&out.join(".zattrs"), py_dumps(&Value::Object(saved), ZARR_JSON).as_bytes())?;
            return Ok(Self { semantic, tracing });
        }

        let changed = changed_keys(&saved["inference"], &attrs["inference"]);
        if !changed.is_empty() {
            bail!("output exists with different provenance ({}); use a new output", changed.join(", "));
        }

        // attributes outside the provenance, such as `voxel_size_um`, are kept from the run that created the group
        let differ: Vec<String> = attrs
            .iter()
            .filter(|(k, v)| *k != "inference" && !saved.get(*k).is_some_and(|s| same_json(s, v)))
            .map(|(k, v)| format!("{k} {} there, {v} now", saved.get(k).unwrap_or(&Value::Null)))
            .collect();
        if !differ.is_empty() {
            bail!("output exists with different attributes ({}); resume with the same arguments", differ.join(", "));
        }
        Ok(Self { semantic: OutArray::open(&out.join("semantic"))?, tracing: OutArray::open(&out.join("tracing"))? })
    }
}

/// Whether `out` holds any prediction: progress records, or array files besides the arrays' own metadata (and
/// its temporary files)
fn holds_prediction(out: &Path) -> Result<bool> {
    let progress = out.join("progress.jsonl");
    match fs::metadata(&progress) {
        Ok(m) if m.len() > 0 => return Ok(true),
        Ok(_) => {}
        // absent, and not a link to something absent, which could hold records again later
        Err(e) if e.kind() == ErrorKind::NotFound && fs::symlink_metadata(&progress).is_err() => {}
        Err(e) => bail!("cannot check {}: {e}", progress.display()),
    }

    for array in ["semantic", "tracing"] {
        let entries = match fs::read_dir(out.join(array)) {
            Ok(entries) => entries,
            Err(e) if e.kind() == ErrorKind::NotFound => continue,
            Err(e) => bail!("{}: {e}", out.join(array).display()),
        };
        for entry in entries {
            let name = entry?.file_name().to_string_lossy().into_owned();
            if name != ".zarray" && !(name.starts_with('.') && name.ends_with(".partial")) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// JSON equality as Python's `==` on the parsed values, where `1 == 1.0`
/// The provenance keys whose values differ between `saved` and `now`, in order, so a refusal can say what changed
fn changed_keys(saved: &Value, now: &Value) -> Vec<String> {
    let (Value::Object(saved), Value::Object(now)) = (saved, now) else {
        return vec!["inference".into()];
    };

    let mut keys: Vec<String> = saved.keys().chain(now.keys()).cloned().collect();
    keys.sort();
    keys.dedup();
    keys.retain(|k| match (saved.get(k), now.get(k)) {
        (Some(a), Some(b)) => !same_json(a, b),
        _ => true,
    });
    keys
}

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
            let mut bufs = ChunkBufs::default();
            for job in rx {
                // after a failure the rest is discarded; the main thread raises it
                if failed {
                    continue;
                }
                if let Err(e) = write_column(&output, &progress, job, &mut bufs) {
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

fn write_column(output: &Output, progress: &Path, job: Finished, bufs: &mut ChunkBufs) -> Result<()> {
    let [y0, y1, x0, x1] = job.col;
    let (h, w) = (y1 - y0, x1 - x0);
    if let (Some(sem), Some(tra)) = (&job.sem, &job.tra) {
        output.semantic.write([0, y0, x0], [job.nz, h, w], sem, bufs)?;
        output.tracing.write([0, y0 / 2, x0 / 2], [job.nz / 2, h / 2, w / 2], tra, bufs)?;
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

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use serde_json::{Map, Value, json};

    use super::{ChunkBufs, OutArray, Output};
    use crate::input::blosc_decompress;

    fn attrs() -> Map<String, Value> {
        let mut a = Map::new();
        a.insert("inference".into(), json!({"engine": "fibrs-predict"}));
        a
    }

    fn fresh(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("fibrs-predict-out-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The `(2, chunk, chunk, chunk)` voxels of one stored chunk, or `None` when it is not stored
    fn chunk(array: &Path, name: &str) -> Option<Vec<u8>> {
        let bytes = fs::read(array.join(name)).ok()?;
        Some(blosc_decompress(&bytes).unwrap())
    }

    /// A `(2, dims)` region holding `v[0]` in channel 0 and `v[1]` in channel 1
    fn region(dims: [usize; 3], v: [u8; 2]) -> Vec<u8> {
        let plane: usize = dims.iter().product();
        [vec![v[0]; plane], vec![v[1]; plane]].concat()
    }

    #[test]
    fn neighbouring_columns_share_a_chunk_and_a_cleared_chunk_is_removed() {
        let out = fresh("write");
        let dir = out.join("semantic");
        // two 4-wide chunks along x; the columns meet inside the first, as 480-wide columns meet in 128 chunks
        let array = OutArray::create(&dir, [4, 4, 8], 4).unwrap();
        // one set of buffers throughout, as the writer thread keeps it, so each chunk starts from the last one's bytes
        let mut bufs = ChunkBufs::default();
        array.write([0, 0, 0], [4, 4, 3], &region([4, 4, 3], [1, 2]), &mut bufs).unwrap();
        array.write([0, 0, 3], [4, 4, 5], &region([4, 4, 5], [3, 4]), &mut bufs).unwrap();

        let first = chunk(&dir, "0.0.0.0").unwrap();
        assert_eq!(first.len(), 2 * 64);
        for (c, (left, right)) in [(1, 3), (2, 4)].into_iter().enumerate() {
            let rows: Vec<&[u8]> = first[c * 64..(c + 1) * 64].chunks(4).collect();
            assert!(rows.iter().all(|row| row == &[left, left, left, right]), "channel {c}: {rows:?}");
        }
        assert_eq!(chunk(&dir, "0.0.0.1").unwrap(), region([4, 4, 4], [3, 4]));

        // clearing the second column keeps the first column's voxels and drops the chunk it covered whole
        array.write([0, 0, 3], [4, 4, 5], &region([4, 4, 5], [0, 0]), &mut bufs).unwrap();
        assert!(chunk(&dir, "0.0.0.1").is_none());
        let first = chunk(&dir, "0.0.0.0").unwrap();
        for (c, v) in [1, 2].into_iter().enumerate() {
            assert!(first[c * 64..(c + 1) * 64].chunks(4).all(|row| row == [v, v, v, 0]), "channel {c}");
        }
        array.write([0, 0, 0], [4, 4, 3], &region([4, 4, 3], [0, 0]), &mut bufs).unwrap();
        assert!(chunk(&dir, "0.0.0.0").is_none());
        fs::remove_dir_all(out).unwrap();
    }

    #[test]
    fn an_output_whose_creation_stopped_is_created_again() {
        let out = fresh("stopped");
        OutArray::create(&out.join("semantic"), [4, 4, 4], 128).unwrap();
        Output::open_or_create(&out, 4, 4, 4, &attrs()).unwrap();
        let saved: Value = serde_json::from_slice(&fs::read(out.join(".zattrs")).unwrap()).unwrap();
        assert_eq!(saved["inference"]["engine"], "fibrs-predict");
        assert!(out.join("tracing/.zarray").exists());
        fs::remove_dir_all(out).unwrap();
    }

    #[test]
    fn a_prediction_without_provenance_is_refused() {
        let chunk = fresh("chunk");
        OutArray::create(&chunk.join("semantic"), [4, 4, 4], 128).unwrap();
        fs::write(chunk.join("semantic/0.0.0.0"), b"x").unwrap();
        let progress = fresh("progress");
        fs::write(progress.join("progress.jsonl"), "{}\n").unwrap();
        for out in [chunk, progress] {
            let err = Output::open_or_create(&out, 4, 4, 4, &attrs()).err().unwrap();
            assert!(err.to_string().contains("without fibrs predict's provenance"), "{err}");
            assert!(!out.join(".zattrs").exists());
            fs::remove_dir_all(out).unwrap();
        }
    }

    #[test]
    fn a_resume_with_other_cudnn_algorithms_is_refused_by_name() {
        let out = fresh("algos");
        let with = |algo: &str| {
            let mut a = Map::new();
            a.insert("inference".into(), json!({"engine": "fibrs-predict", "cudnn_algos": [algo], "gpu": "x"}));
            a
        };
        Output::open_or_create(&out, 4, 4, 4, &with("IMPLICIT_PRECOMP_GEMM")).unwrap();
        Output::open_or_create(&out, 4, 4, 4, &with("IMPLICIT_PRECOMP_GEMM")).unwrap();
        let err = Output::open_or_create(&out, 4, 4, 4, &with("WINOGRAD_NONFUSED")).err().unwrap();
        assert!(err.to_string().contains("different provenance (cudnn_algos)"), "{err}");
        fs::remove_dir_all(out).unwrap();
    }

    #[test]
    fn a_resume_with_another_voxel_size_is_refused() {
        let out = fresh("voxel");
        let with = |um: f64| {
            let mut a = attrs();
            a.insert("voxel_size_um".into(), json!(um));
            a
        };
        Output::open_or_create(&out, 4, 4, 4, &with(8.64)).unwrap();
        Output::open_or_create(&out, 4, 4, 4, &with(8.64)).unwrap();
        let err = Output::open_or_create(&out, 4, 4, 4, &with(9.362)).err().unwrap();
        assert!(err.to_string().contains("different attributes (voxel_size_um 8.64 there, 9.362 now)"), "{err}");
        let saved: Value = serde_json::from_slice(&fs::read(out.join(".zattrs")).unwrap()).unwrap();
        assert_eq!(saved["voxel_size_um"], json!(8.64));
        fs::remove_dir_all(out).unwrap();

        // a value whose shortest decimal has 16 significant digits reads back as the same f64, not as its 9.1 neighbour
        let out = fresh("voxel-digits");
        let precise = 9.100000000000001;
        Output::open_or_create(&out, 4, 4, 4, &with(precise)).unwrap();
        Output::open_or_create(&out, 4, 4, 4, &with(precise)).unwrap();
        let err = Output::open_or_create(&out, 4, 4, 4, &with(9.1)).err().unwrap();
        assert!(err.to_string().contains("voxel_size_um 9.100000000000001 there, 9.1 now"), "{err}");
        fs::remove_dir_all(out).unwrap();
    }

    #[test]
    fn progress_that_cannot_be_checked_is_not_taken_as_empty() {
        let out = fresh("dangling");
        std::os::unix::fs::symlink(out.join("elsewhere.jsonl"), out.join("progress.jsonl")).unwrap();
        let err = Output::open_or_create(&out, 4, 4, 4, &attrs()).err().unwrap();
        assert!(err.to_string().contains("cannot check"), "{err}");
        assert!(!out.join(".zattrs").exists());
        fs::remove_dir_all(out).unwrap();
    }
}
