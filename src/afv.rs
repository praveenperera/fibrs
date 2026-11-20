//! The AFV SQLite container (`afv.export_afv`), written row for row as the reference does
//!
//! Every stored value matches the reference: the schema text (SQLite keeps it verbatim in
//! `sqlite_master`), the metadata JSON with Python's float repr and key order, the fibre rows, the
//! little-endian float64 blocks and the R*Tree rows. Only the per-run uuid and SQLite's page layout
//! differ, so outputs are compared table by table rather than by file hash

use std::fs::{self, File};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

use color_eyre::eyre::{Result, bail, eyre};
use rayon::prelude::*;
use rusqlite::{Connection, params};
use serde_json::{Map, Value, json};

use crate::pyfmt::{np_sum_f64, py_json};

const SCHEMA: &str = "
PRAGMA application_id=0x56434643;
PRAGMA user_version=1;
CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE fibers(id INTEGER PRIMARY KEY,name TEXT NOT NULL,family TEXT NOT NULL,
 point_count INTEGER NOT NULL CHECK(point_count>=2),length REAL NOT NULL,
 min_x REAL NOT NULL,max_x REAL NOT NULL,min_y REAL NOT NULL,max_y REAL NOT NULL,
 min_z REAL NOT NULL,max_z REAL NOT NULL,annotation TEXT NOT NULL);
CREATE TABLE blocks(id INTEGER PRIMARY KEY,fiber_id INTEGER NOT NULL REFERENCES fibers(id),
 first_segment INTEGER NOT NULL,points BLOB NOT NULL,UNIQUE(fiber_id,first_segment));
CREATE VIRTUAL TABLE block_bounds USING rtree(id,min_x,max_x,min_y,max_y,min_z,max_z);
";

/// What the export needs beyond the curves
pub struct ExportOptions<'a> {
    pub voxel_um: f64,
    pub origin_xyz: [i64; 3],
    pub volume_shape: [i64; 3],
    /// `--min-length-mm`; the reference stores the untouched default as the integer 0
    pub minimum_mm: MinLength,
    pub coordinate_space: Option<&'a str>,
}

/// `--min-length-mm` as argparse leaves it: the integer default or a parsed float
#[derive(Clone, Copy, Debug)]
pub enum MinLength {
    Default,
    Given(f64),
}

impl MinLength {
    pub fn value(self) -> f64 {
        match self {
            MinLength::Default => 0.0,
            MinLength::Given(v) => v,
        }
    }

    fn json(self) -> Value {
        match self {
            MinLength::Default => json!(0),
            MinLength::Given(v) => json!(v),
        }
    }
}

/// Totals stored in the finished file's metadata
#[derive(Debug, Default)]
pub struct Summary {
    pub fiber_count: i64,
    pub point_count: i64,
    pub block_count: i64,
}

/// Fill `buf` (at most 256 bytes) from the kernel's random source
///
/// `getentropy` rather than `getrandom`, since macOS has only the former; glibc has had both since 2.25
fn fill_random(buf: &mut [u8]) -> Result<()> {
    // SAFETY: the pointer and length describe `buf`, which getentropy fills in full or not at all
    if unsafe { libc::getentropy(buf.as_mut_ptr().cast(), buf.len()) } != 0 {
        bail!("getentropy failed: {}", std::io::Error::last_os_error());
    }
    Ok(())
}

/// A random version 4 UUID in canonical form, from the kernel's random source
fn uuid4() -> Result<String> {
    let mut b = [0u8; 16];
    fill_random(&mut b)?;
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    let h: String = b.iter().map(|v| format!("{v:02x}")).collect();
    Ok(format!("{}-{}-{}-{}-{}", &h[..8], &h[8..12], &h[12..16], &h[16..20], &h[20..]))
}

/// `tempfile.mkstemp(prefix=".afv-", suffix=".partial", dir=parent)`: an exclusively created, empty file
fn mkstemp(parent: &Path) -> Result<PathBuf> {
    for _ in 0..100 {
        let mut b = [0u8; 6];
        fill_random(&mut b)?;
        const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789_";
        let name: String = b.iter().map(|v| ALPHABET[*v as usize % ALPHABET.len()] as char).collect();
        let path = parent.join(format!(".afv-{name}.partial"));
        match fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(&path) {
            Ok(_) => return Ok(path),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.into()),
        }
    }
    bail!("No usable temporary file name found in {}", parent.display())
}

/// `bounds(points)`: min x, max x, min y, max y, min z, max z
fn bounds(q: &[[f64; 3]]) -> [f64; 6] {
    let mut out =
        [f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY];
    for p in q {
        for a in 0..3 {
            out[2 * a] = out[2 * a].min(p[a]);
            out[2 * a + 1] = out[2 * a + 1].max(p[a]);
        }
    }
    out
}

/// Removes the temporary file however the export ends, as the reference's `finally` does
struct TempGuard(PathBuf);

impl Drop for TempGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

/// An export in progress: the temporary file, open in one transaction, with the fibres appended so far
///
/// Appending the curves in several calls writes the same rows as one call with all of them, so a family can be
/// written while the next one is still being traced
pub struct AfvExport {
    // declared before the guard, so the connection closes before the guard removes an unfinished file
    db: Connection,
    temporary: TempGuard,
    destination: PathBuf,
    geometry: Geometry,
    summary: Summary,
}

impl AfvExport {
    /// Create the temporary file next to `destination` and write the schema and the first metadata rows
    pub fn start(destination: &Path, opt: &ExportOptions) -> Result<Self> {
        if destination.exists() {
            bail!("Output already exists: {}", destination.display());
        }
        let parent = destination.parent().map_or_else(|| PathBuf::from("."), Path::to_path_buf);
        let parent = if parent.as_os_str().is_empty() { PathBuf::from(".") } else { parent };
        fs::create_dir_all(&parent)?;
        let temporary = TempGuard(mkstemp(&parent)?);
        let db = Connection::open(&temporary.0)?;
        db.execute_batch("PRAGMA foreign_keys=ON")?;
        // the file is this run's own temporary until it is complete, so no other connection takes a lock on it
        db.execute_batch("PRAGMA locking_mode=EXCLUSIVE")?;
        db.execute_batch(&format!("PRAGMA cache_size=-{EXPORT_CACHE_KIB}"))?;
        db.execute_batch(SCHEMA)?;
        // one transaction from here to `finish`; a connection closed before then rolls it back
        db.execute_batch("BEGIN")?;
        {
            let mut meta = db.prepare("INSERT INTO metadata VALUES(?,?)")?;
            for (k, v) in metadata(opt)? {
                meta.execute(params![k, py_json(&v)])?;
            }
        }
        let geometry = Geometry {
            origin: opt.origin_xyz.map(|v| v as f64),
            limit: [opt.volume_shape[2], opt.volume_shape[1], opt.volume_shape[0]].map(|v| v as f64),
            voxel_um: opt.voxel_um,
            minimum_mm: opt.minimum_mm.value(),
        };
        let destination = destination.to_path_buf();
        Ok(Self { db, temporary, destination, geometry, summary: Summary::default() })
    }

    /// Append `curves` (family 1 = V or 2 = H, tracing-space float32 x, y, z points) after the fibres already
    /// written, numbering on from them
    ///
    /// Fibres are prepared on the current rayon pool while the calling thread inserts them in order
    pub fn append<I>(&mut self, curves: I) -> Result<()>
    where
        I: IntoIterator<Item = Result<(usize, Vec<[f32; 3]>)>>,
        I::IntoIter: Send,
    {
        let Self { db, geometry, summary: s, .. } = self;
        let mut fiber = db.prepare_cached("INSERT INTO fibers VALUES(?,?,?,?,?,?,?,?,?,?,?,?)")?;
        let mut block = db.prepare_cached("INSERT INTO blocks VALUES(?,?,?,?)")?;
        let mut bound = db.prepare_cached("INSERT INTO block_bounds VALUES(?,?,?,?,?,?,?)")?;
        let mut insert = |rows: Vec<FiberRows>, s: &mut Summary| -> Result<()> {
            for r in rows {
                s.fiber_count += 1;
                let count = s.fiber_count;
                let b = r.bounds;
                let name = format!("fiber_{count}.json");
                fiber.execute(params![
                    count,
                    name,
                    r.label,
                    r.point_count as i64,
                    r.length,
                    b[0],
                    b[1],
                    b[2],
                    b[3],
                    b[4],
                    b[5],
                    r.annotation
                ])?;
                for (first, bytes, b) in r.blocks {
                    s.block_count += 1;
                    block.execute(params![s.block_count, count, first as i64, bytes])?;
                    bound.execute(params![s.block_count, b[0], b[1], b[2], b[3], b[4], b[5]])?;
                }
                s.point_count += r.point_count as i64;
            }
            Ok(())
        };
        // fused so that, as when the curves were written one by one, the input ends at its first `None`
        let mut curves = curves.into_iter().fuse();
        let mut next = geometry.next_rows(&mut curves, s.fiber_count + 1);
        while let Some(Rows { fibers, error }) = next {
            // no batch is read after one with an error, though that batch itself was read whole: an invalid curve
            // is reported only once the rest of its batch has been read, ahead of any error found reading it
            if let Some(e) = error {
                insert(fibers, s)?;
                return Err(e);
            }
            let first = s.fiber_count + fibers.len() as i64 + 1;
            let mut prepared = None;
            // the pool makes the next batch's rows while this thread inserts this batch's
            let inserted = rayon::in_place_scope(|scope| {
                scope.spawn(|_| prepared = Some(geometry.next_rows(&mut curves, first)));
                insert(fibers, s)
            });
            inserted?;
            next = prepared.expect("the spawned batch has finished");
        }
        Ok(())
    }

    /// Index the fibres, store the totals, commit, check the file and publish it at the destination
    pub fn finish(self) -> Result<Summary> {
        let Self { db, temporary, destination, summary: s, .. } = self;
        db.execute_batch("CREATE INDEX fibers_length ON fibers(length DESC,id)")?;
        {
            let mut meta = db.prepare("INSERT OR REPLACE INTO metadata VALUES(?,?)")?;
            let rows = [
                ("complete", json!(true)),
                ("fiber_count", json!(s.fiber_count)),
                ("point_count", json!(s.point_count)),
                ("block_count", json!(s.block_count)),
            ];
            for (k, v) in rows {
                meta.execute(params![k, py_json(&v)])?;
            }
        }
        db.execute_batch("COMMIT")?;
        check(&db)?;
        db.close().map_err(|(_, e)| e)?;
        File::open(&temporary.0)?.sync_all()?;
        // a hard link publishes the complete file without overwriting a path that another process may
        // have created while tracing was running
        fs::hard_link(&temporary.0, &destination).map_err(|e| eyre!("{}: {e}", destination.display()))?;
        Ok(s)
    }
}

fn metadata(opt: &ExportOptions) -> Result<Vec<(&'static str, Value)>> {
    let mut frame = Map::new();
    frame.insert("coordinate_base_shape_zyx".into(), json!(opt.volume_shape));
    frame.insert("vc_open_data_source_coordinate_level".into(), json!(0));
    frame.insert("vc_open_data_source_coordinate_scale_factor".into(), json!(1));
    frame.insert("vc_open_data_source_original_resolution".into(), json!(opt.voxel_um));
    if let Some(space) = opt.coordinate_space.filter(|s| !s.is_empty()) {
        frame.insert("vc_open_data_coordinate_space".into(), json!(space));
    }
    let mut root = Map::new();
    root.insert("type".into(), json!("vc3d_fiber_collection"));
    root.insert("version".into(), json!(1));
    root.insert("scale".into(), json!(1));
    root.extend(frame.clone());
    root.insert("coordinate_order".into(), json!("XYZ"));
    root.insert("coordinate_units".into(), json!("voxel"));
    root.insert("voxel_size_mm".into(), json!(opt.voxel_um / 1000.0));
    let mut tracing = Map::new();
    tracing.insert("scale".into(), json!(2));
    tracing.insert("center_offset".into(), json!(0.5));
    tracing.insert("origin_xyz".into(), json!(opt.origin_xyz));
    tracing.insert("minimum_length_mm".into(), opt.minimum_mm.json());
    tracing.insert("family_locked".into(), json!(true));
    tracing.insert("ct_exterior_filter".into(), json!(false));
    Ok(vec![
        ("complete", json!(false)),
        ("uuid", json!(uuid4()?)),
        ("root", Value::Object(root)),
        ("frame", Value::Object(frame)),
        ("geometry", json!("XYZ binary64 little-endian native L0")),
        ("block_segments", json!(256)),
        ("source", json!("AVF_SPLINE_GENERATOR")),
        ("tracing", Value::Object(tracing)),
    ])
}

/// Curves prepared at a time: the pool makes one batch's rows while the calling thread inserts the batch before
const EXPORT_BATCH: usize = 4096;

/// Points prepared at a time: a batch ends at the fibre that brings it to this many, even before [`EXPORT_BATCH`]
/// curves. A fibre is held whole, so each of the two batches in flight holds under about 50 MiB of input, native
/// points and block bytes, plus its last fibre
const EXPORT_BATCH_POINTS: usize = 1 << 20;

/// SQLite's page cache for the export, in KiB (a negative `cache_size`): with 32 MiB, SQLite read evicted pages
/// back from a dense window's file of over 1.5 GiB, and a cache of the whole file would add that much to the
/// export's memory
const EXPORT_CACHE_KIB: i64 = 256 << 10;

/// One fibre's rows, made before the fibre is inserted
struct FiberRows {
    label: &'static str,
    point_count: usize,
    length: f64,
    bounds: [f64; 6],
    annotation: String,
    /// Each block's first segment, its points as little-endian float64 and its bounds
    blocks: Vec<(usize, Vec<u8>, [f64; 6])>,
}

/// A batch of curves read from the input, and the input's error after them, if reading stopped on one
struct Batch {
    curves: Vec<(usize, Vec<[f32; 3]>)>,
    error: Option<color_eyre::Report>,
}

/// A batch's rows up to its first error in input order, and that error. The rows are inserted before the error is
/// reported, so a failed insert of an earlier fibre is the error, as when the curves were written one by one
struct Rows {
    fibers: Vec<FiberRows>,
    error: Option<color_eyre::Report>,
}

impl Batch {
    /// The next batch, or `None` once the input has nothing left
    fn read(curves: &mut impl Iterator<Item = Result<(usize, Vec<[f32; 3]>)>>) -> Option<Self> {
        let mut out = Vec::with_capacity(EXPORT_BATCH);
        let mut points = 0;
        for item in curves.by_ref() {
            match item {
                Ok(curve) => {
                    points += curve.1.len();
                    out.push(curve);
                }
                Err(e) => return Some(Self { curves: out, error: Some(e) }),
            }
            if out.len() == EXPORT_BATCH || points >= EXPORT_BATCH_POINTS {
                break;
            }
        }
        (!out.is_empty()).then_some(Self { curves: out, error: None })
    }
}

/// What the export computes from the options for every fibre
struct Geometry {
    origin: [f64; 3],
    limit: [f64; 3],
    voxel_um: f64,
    minimum_mm: f64,
}

/// A curve's native points, length and length in mm, or `None` for one shorter than the minimum
struct Kept {
    label: &'static str,
    q: Vec<[f64; 3]>,
    length: f64,
    mm: f64,
}

impl Geometry {
    fn kept(&self, family: usize, points: &[[f32; 3]]) -> Result<Option<Kept>> {
        // points.astype("<f8") * 2 + 0.5 + origin
        let q: Vec<[f64; 3]> =
            points.iter().map(|p| [0, 1, 2].map(|a| p[a] as f64 * 2.0 + 0.5 + self.origin[a])).collect();
        let bad = q.iter().any(|p| (0..3).any(|a| !p[a].is_finite() || p[a] < 0.0 || p[a] >= self.limit[a]));
        if q.len() < 2 || bad {
            bail!("Spline contains invalid native coordinates");
        }
        // np.linalg.norm(np.diff(q, axis=0), axis=1).sum()
        let seg: Vec<f64> = q
            .windows(2)
            .map(|w| {
                let d = [w[1][0] - w[0][0], w[1][1] - w[0][1], w[1][2] - w[0][2]];
                (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
            })
            .collect();
        let length = np_sum_f64(&seg);
        let mm = length * self.voxel_um / 1000.0;
        if length <= 0.0 || mm < self.minimum_mm {
            return Ok(None);
        }
        let label = match family {
            1 => "V",
            2 => "H",
            other => bail!("Invalid family {other}"),
        };
        Ok(Some(Kept { label, q, length, mm }))
    }

    /// The rows of the input's next batch, numbered from `first`, or `None` once the input has nothing left. A
    /// batch can have no rows when all its fibres are too short; that does not end the input
    fn next_rows(&self, curves: &mut impl Iterator<Item = Result<(usize, Vec<[f32; 3]>)>>, first: i64) -> Option<Rows> {
        Batch::read(curves).map(|batch| self.rows(batch, first))
    }

    /// The rows of a batch's fibres, numbered from `first`, up to the batch's first error in input order
    fn rows(&self, batch: Batch, first: i64) -> Rows {
        let kept: Vec<Result<Option<Kept>>> = batch.curves.par_iter().map(|(f, p)| self.kept(*f, p)).collect();
        let mut numbered = Vec::with_capacity(kept.len());
        let mut error = batch.error;
        for k in kept {
            match k {
                Ok(Some(k)) => numbered.push((first + numbered.len() as i64, k)),
                Ok(None) => {}
                Err(e) => {
                    // an invalid curve comes before the read error that ended the batch
                    error = Some(e);
                    break;
                }
            }
        }
        let fibers = numbered.into_par_iter().map(|(count, k)| fiber_rows(count, k)).collect();
        Rows { fibers, error }
    }
}

fn fiber_rows(count: i64, k: Kept) -> FiberRows {
    let Kept { label, q, length, mm } = k;
    let annotation = json!({
        "type": "vc3d_fiber",
        "version": 1,
        "sequence": count,
        "filename": format!("fiber_{count}.json"),
        "hv_classification": {"manual_tag": label},
        "classification_source": "model_prediction_family_locked_tracing",
        "control_points": [q[0], q[q.len() - 1]],
        "length_voxels": length,
        "length_mm": mm,
    });
    let blocks = (0..q.len() - 1)
        .step_by(256)
        .map(|first| {
            let pts = &q[first..(first + 257).min(q.len())];
            let bytes: Vec<u8> = pts.iter().flat_map(|p| p.iter().flat_map(|v| v.to_le_bytes())).collect();
            (first, bytes, bounds(pts))
        })
        .collect();
    FiberRows { label, point_count: q.len(), length, bounds: bounds(&q), annotation: py_json(&annotation), blocks }
}

fn check(db: &Connection) -> Result<()> {
    let integrity: Vec<String> =
        db.prepare("PRAGMA integrity_check")?.query_map([], |r| r.get(0))?.collect::<Result<_, _>>()?;
    if integrity != ["ok"] {
        bail!("AFV integrity check failed");
    }
    if db.prepare("PRAGMA foreign_key_check")?.query([])?.next()?.is_some() {
        bail!("AFV contains invalid block references");
    }
    let rtree: String = db.query_row("SELECT rtreecheck('block_bounds')", [], |r| r.get(0))?;
    if rtree != "ok" {
        bail!("AFV spatial index check failed");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // the whole export in one call, as a run with nothing traced ahead of it writes it
    fn export_afv<I>(curves: I, destination: &Path, opt: &ExportOptions) -> Result<Summary>
    where
        I: IntoIterator<Item = Result<(usize, Vec<[f32; 3]>)>>,
        I::IntoIter: Send,
    {
        let mut export = AfvExport::start(destination, opt)?;
        export.append(curves)?;
        export.finish()
    }

    fn options() -> ExportOptions<'static> {
        ExportOptions {
            voxel_um: 10.0,
            origin_xyz: [0, 0, 0],
            volume_shape: [64, 1 << 16, 1 << 16],
            minimum_mm: MinLength::Given(0.01),
            coordinate_space: None,
        }
    }

    /// Curve `i` of a family alternating V and H, two points `i` voxels apart along x at y = 1 + i; every
    /// seventh is too short for the minimum length
    fn curve(i: usize) -> (usize, Vec<[f32; 3]>) {
        let len = if i.is_multiple_of(7) { 0.1 } else { 1.0 + i as f32 / 1000.0 };
        (1 + i % 2, vec![[0.0, 1.0 + i as f32, 1.0], [len, 1.0 + i as f32, 1.0]])
    }

    fn export(curves: Vec<Result<(usize, Vec<[f32; 3]>)>>, name: &str) -> (PathBuf, Result<Summary>) {
        let dir = std::env::temp_dir().join(format!("fibrs-afv-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let out = dir.join("out.afv");
        let pool = rayon::ThreadPoolBuilder::new().num_threads(3).build().unwrap();
        let result = pool.install(|| export_afv(curves, &out, &options()));
        (out, result)
    }

    /// Fibres exported over several batches are numbered and stored in input order, with the short ones left out
    #[test]
    fn batches_keep_the_input_order() {
        let n = 2 * EXPORT_BATCH + 123;
        let (out, summary) = export((0..n).map(|i| Ok(curve(i))).collect(), "order");
        let summary = summary.unwrap();
        let kept: Vec<usize> = (0..n).filter(|i| !i.is_multiple_of(7)).collect();
        assert_eq!(summary.fiber_count, kept.len() as i64);
        assert_eq!(summary.block_count, kept.len() as i64);

        let db = Connection::open(&out).unwrap();
        let mut rows = db.prepare("SELECT id, name, family, min_y, annotation FROM fibers ORDER BY id").unwrap();
        let rows: Vec<(i64, String, String, f64, String)> = rows
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(rows.len(), kept.len());
        for ((id, name, family, min_y, annotation), (k, &i)) in rows.iter().zip(kept.iter().enumerate()) {
            assert_eq!(*id, k as i64 + 1);
            assert_eq!(name, &format!("fiber_{id}.json"));
            assert_eq!(family, if i.is_multiple_of(2) { "V" } else { "H" });
            // tracing-space y = 1 + i is native 2 (1 + i) + 0.5
            assert_eq!(*min_y, 2.0 * (1.0 + i as f64) + 0.5);
            assert!(annotation.contains(&format!("\"sequence\":{id},")), "{annotation}");
        }
        let blocks: i64 = db
            .query_row("SELECT count(*) FROM blocks WHERE fiber_id NOT IN (SELECT id FROM fibers)", [], |r| r.get(0))
            .unwrap();
        assert_eq!(blocks, 0);
        let _ = fs::remove_dir_all(out.parent().unwrap());
    }

    /// A curve too short for the minimum length, two points 0.1 voxel apart
    fn short(i: usize) -> (usize, Vec<[f32; 3]>) {
        (1, vec![[0.0, 1.0 + i as f32, 1.0], [0.1, 1.0 + i as f32, 1.0]])
    }

    /// A whole batch of fibres too short to keep, first or in the middle, is followed by the rest of the input, and
    /// an input error after it still fails the export
    #[test]
    fn a_batch_of_only_short_fibres_does_not_end_the_export() {
        let mut curves: Vec<Result<(usize, Vec<[f32; 3]>)>> = (0..EXPORT_BATCH).map(|i| Ok(short(i))).collect();
        curves.push(Ok(curve(EXPORT_BATCH + 1)));
        let (out, summary) = export(curves, "short");
        let summary = summary.unwrap();
        assert_eq!((summary.fiber_count, summary.block_count), (1, 1));
        let db = Connection::open(&out).unwrap();
        let min_y: f64 = db.query_row("SELECT min_y FROM fibers WHERE id = 1", [], |r| r.get(0)).unwrap();
        assert_eq!(min_y, 2.0 * (2.0 + EXPORT_BATCH as f64) + 0.5);
        let _ = fs::remove_dir_all(out.parent().unwrap());

        // a middle batch of only short fibres, between batches with fibres kept
        let mut curves: Vec<Result<(usize, Vec<[f32; 3]>)>> = (0..EXPORT_BATCH).map(|i| Ok(curve(i))).collect();
        curves.extend((0..EXPORT_BATCH).map(|i| Ok(short(i))));
        curves.push(Ok(curve(EXPORT_BATCH + 1)));
        let first: i64 = (0..EXPORT_BATCH).filter(|i| !i.is_multiple_of(7)).count() as i64;
        let (out, summary) = export(curves, "short-middle");
        assert_eq!(summary.unwrap().fiber_count, first + 1);
        let db = Connection::open(&out).unwrap();
        let last: f64 = db.query_row("SELECT min_y FROM fibers WHERE id = ?", [first + 1], |r| r.get(0)).unwrap();
        assert_eq!(last, 2.0 * (2.0 + EXPORT_BATCH as f64) + 0.5);
        let _ = fs::remove_dir_all(out.parent().unwrap());

        let mut curves: Vec<Result<(usize, Vec<[f32; 3]>)>> = (0..EXPORT_BATCH).map(|i| Ok(short(i))).collect();
        curves.push(Err(eyre!("reading failed")));
        let (out, result) = export(curves, "short-error");
        assert_eq!(result.unwrap_err().to_string(), "reading failed");
        assert!(!out.exists());
        let _ = fs::remove_dir_all(out.parent().unwrap());
    }

    /// An invalid curve in a later batch fails the export with its own error, ahead of an input error read later in
    /// its batch: the earliest error in input order is reported, as when the curves were written one at a time
    #[test]
    fn the_first_error_in_input_order_is_reported() {
        let n = EXPORT_BATCH + 10;
        let mut curves: Vec<Result<(usize, Vec<[f32; 3]>)>> = (0..n).map(|i| Ok(curve(i))).collect();
        curves[EXPORT_BATCH + 3] = Ok((1, vec![[-5.0, 1.0, 1.0], [1.0, 1.0, 1.0]]));
        curves[EXPORT_BATCH + 5] = Err(eyre!("reading failed"));
        let (out, result) = export(curves, "error");
        let e = result.unwrap_err().to_string();
        assert_eq!(e, "Spline contains invalid native coordinates");
        assert!(!out.exists());
        let _ = fs::remove_dir_all(out.parent().unwrap());
    }

    /// The input ends at its first `None`, even for an iterator that would yield more after it
    #[test]
    fn the_input_ends_at_its_first_none() {
        let mut i = 0;
        let curves = std::iter::from_fn(move || {
            i += 1;
            (i != 2).then(|| Ok(curve(i)))
        })
        .take(3);
        let dir = std::env::temp_dir().join(format!("fibrs-afv-none-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let out = dir.join("out.afv");
        let summary = export_afv(curves, &out, &options()).unwrap();
        assert_eq!(summary.fiber_count, 1);
        let _ = fs::remove_dir_all(&dir);
    }

    /// Every row of every table but the per-run uuid, in a stable order
    fn table_rows(path: &Path) -> Vec<String> {
        let db = Connection::open(path).unwrap();
        let queries = [
            "SELECT key || '=' || value FROM metadata WHERE key != 'uuid' ORDER BY key",
            "SELECT id || ' ' || name || ' ' || family || ' ' || point_count || ' ' || length || ' ' || min_x || ' ' \
             || max_x || ' ' || min_y || ' ' || max_y || ' ' || min_z || ' ' || max_z || ' ' || annotation \
             FROM fibers ORDER BY id",
            "SELECT id || ' ' || fiber_id || ' ' || first_segment || ' ' || hex(points) FROM blocks ORDER BY id",
            "SELECT id || ' ' || min_x || ' ' || max_x || ' ' || min_y || ' ' || max_y || ' ' || min_z || ' ' || max_z \
             FROM block_bounds ORDER BY id",
            "SELECT CAST(id AS TEXT) FROM fibers ORDER BY length DESC, id",
        ];
        let mut out = Vec::new();
        for q in queries {
            let mut stmt = db.prepare(q).unwrap();
            out.extend(stmt.query_map([], |r| r.get::<_, String>(0)).unwrap().map(Result::unwrap));
            out.push(String::new());
        }
        out
    }

    /// A family appended on one pool while another is still to come, then the next family on another pool, stores
    /// the rows one export of both writes, with the split falling inside a batch
    #[test]
    fn an_export_appended_family_by_family_matches_one_export() {
        let n = EXPORT_BATCH + 321;
        let all: Vec<(usize, Vec<[f32; 3]>)> = (0..n).map(curve).collect();
        let (v, h): (Vec<_>, Vec<_>) = all.into_iter().partition(|(family, _)| *family == 1);
        let whole: Vec<Result<_>> = v.iter().chain(h.iter()).cloned().map(Ok).collect();
        let (one, summary) = export(whole, "whole");
        let summary = summary.unwrap();

        let dir = std::env::temp_dir().join(format!("fibrs-afv-parts-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let out = dir.join("out.afv");
        let mut parts = AfvExport::start(&out, &options()).unwrap();
        let early = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        early.install(|| parts.append(v.into_iter().map(Ok))).unwrap();
        let late = rayon::ThreadPoolBuilder::new().num_threads(3).build().unwrap();
        let parted = late
            .install(|| {
                parts.append(h.into_iter().map(Ok))?;
                parts.finish()
            })
            .unwrap();

        assert_eq!(
            (parted.fiber_count, parted.point_count, parted.block_count),
            (summary.fiber_count, summary.point_count, summary.block_count)
        );
        assert!(summary.fiber_count > EXPORT_BATCH as i64 / 2, "{summary:?}");
        assert_eq!(table_rows(&out), table_rows(&one));
        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(one.parent().unwrap());
    }
}
