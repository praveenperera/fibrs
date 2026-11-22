//! `fibrs subset`: the fibres of a complete `.afv` that touch a box, or pass a length or family filter, written as a
//! smaller complete `.afv`
//!
//! A whole-scroll fibre set is too large for most machines to import whole into villa's fibre store (villa's
//! importer holds about its whole output in memory, 27 GiB for an 18.6 GB set), and the tools that read the store
//! (VC3D, the sheet-strip runner, wgraph) work on a region at a time. The subset is a valid input for villa's AFV
//! importers, with the same uuid and the same fibre and block ids as the input, so a fibre imported from any subset
//! has the UUID it has in a whole-scroll import (villa names a fibre `uuid5(afv uuid, "fiber:<id>")`)
//!
//! Rows are copied as stored: fibres are kept whole, never cut at the box, and each kept fibre's row, blocks and
//! R*Tree rows are the input's. The metadata is the input's, with the counts of the subset and a `subset` key that
//! says how it was chosen

use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::{Parser, ValueEnum};
use color_eyre::eyre::{Result, bail, eyre};
use rusqlite::types::Value as Sql;
use rusqlite::{Connection, OpenFlags, params};
use serde_json::{Value, json};

use crate::afv::read::AfvReader;
use crate::afv::{self, Access, TempGuard, mkstemp};
use crate::log;
use crate::pyfmt::py_json;

/// Bytes of one stored point: x, y, z as little-endian float64
const POINT_BYTES: usize = 24;

/// Metadata keys the subset recomputes; every other key is copied as the input stores it
const RECOMPUTED: [&str; 5] = ["complete", "fiber_count", "point_count", "block_count", "subset"];

/// SQLite's page cache for each connection, in KiB: enough for the R*Tree and the fibre table's upper pages
const CACHE_KIB: i64 = 256 << 10;

/// Write the fibres of a complete .afv that touch a box or pass a length or family filter as a new, smaller .afv
///
/// Fibres are kept whole and keep their ids, and the output keeps the input's uuid, so villa's importers give each
/// fibre the same UUID as a whole-scroll import does
#[derive(Parser, Debug)]
#[command(name = "fibrs subset", version = env!("CARGO_PKG_VERSION"))]
pub struct Args {
    /// The complete .afv to read
    #[arg(long)]
    input: PathBuf,
    /// The subset .afv; must not exist
    #[arg(long)]
    output: PathBuf,
    /// Keep fibres with a point in this box of native level-0 voxels, bounds included
    #[arg(long = "box", num_args = 6, value_names = ["X0", "X1", "Y0", "Y1", "Z0", "Z1"], allow_negative_numbers = true)]
    region: Option<Vec<f64>>,
    /// Keep fibres at least this long, in mm at the set's voxel size
    #[arg(long)]
    min_length_mm: Option<f64>,
    /// Keep one family only
    #[arg(long, value_enum)]
    family: Option<Family>,
}

/// A fibre family as the AFV labels it
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Family {
    #[value(name = "V")]
    V,
    #[value(name = "H")]
    H,
}

impl Family {
    fn label(self) -> &'static str {
        match self {
            Family::V => "V",
            Family::H => "H",
        }
    }
}

/// A box of native x, y, z voxels, each axis's bounds included
#[derive(Clone, Copy, Debug)]
struct Region {
    lo: [f64; 3],
    hi: [f64; 3],
}

impl Region {
    fn new(v: &[f64]) -> Result<Self> {
        let [x0, x1, y0, y1, z0, z1] = v else { bail!("--box takes six numbers: X0 X1 Y0 Y1 Z0 Z1") };
        let region = Self { lo: [*x0, *y0, *z0], hi: [*x1, *y1, *z1] };
        if (0..3).any(|a| !region.lo[a].is_finite() || !region.hi[a].is_finite() || region.lo[a] > region.hi[a]) {
            bail!("--box needs finite bounds with each axis's low bound at most its high one");
        }
        Ok(region)
    }

    fn contains(&self, p: [f64; 3]) -> bool {
        (0..3).all(|a| self.lo[a] <= p[a] && p[a] <= self.hi[a])
    }

    fn json(&self) -> Value {
        json!([[self.lo[0], self.hi[0]], [self.lo[1], self.hi[1]], [self.lo[2], self.hi[2]]])
    }
}

/// How the subset is chosen, checked
struct Filter {
    region: Option<Region>,
    min_length_mm: Option<f64>,
    family: Option<Family>,
}

impl Filter {
    fn new(args: &Args) -> Result<Self> {
        let region = args.region.as_deref().map(Region::new).transpose()?;
        if let Some(mm) = args.min_length_mm
            && !(mm.is_finite() && mm >= 0.0)
        {
            bail!("--min-length-mm must be a finite length of at least 0");
        }
        if region.is_none() && args.min_length_mm.is_none() && args.family.is_none() {
            bail!("give at least one of --box, --min-length-mm and --family; with none the subset is the input");
        }
        Ok(Self { region, min_length_mm: args.min_length_mm, family: args.family })
    }

    /// What the output's `subset` metadata records
    fn json(&self, source: &Totals) -> Value {
        json!({
            "box_xyz": self.region.map(|r| r.json()),
            "minimum_length_mm": self.min_length_mm,
            "family": self.family.map(Family::label),
            "fibres_kept_whole": true,
            "source_fiber_count": source.fibers,
            "source_point_count": source.points,
            "source_block_count": source.blocks,
        })
    }
}

/// A file's fibre, point and block counts
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct Totals {
    fibers: i64,
    points: i64,
    blocks: i64,
}

/// Whether `argv` asks for `fibrs subset`; the drop-in `avf-spline-generator` has no subcommands
pub fn requested(argv: &[OsString]) -> bool {
    let Some(arg0) = argv.first() else { return false };
    !crate::run::dropin(arg0.as_encoded_bytes()) && argv.get(1).is_some_and(|a| a == "subset")
}

/// Run `fibrs subset` with the arguments after `subset`; the process's exit code
pub fn main(argv: &[OsString]) -> i32 {
    let args = Args::parse_from(std::iter::once(OsString::from("fibrs subset")).chain(argv.iter().cloned()));
    match run(&args) {
        Ok(()) => 0,
        Err(e) => {
            log!("fibrs subset: {e:#}");
            1
        }
    }
}

/// The input, open read-only, with what the subset needs from its metadata
struct Source {
    db: Connection,
    /// Every metadata row as stored, key and JSON text
    metadata: Vec<(String, String)>,
    voxel_um: f64,
    totals: Totals,
}

impl Source {
    fn open(path: &Path) -> Result<Self> {
        // the shared reader refuses a file not marked complete and reads the frame's voxel size
        let voxel_um = AfvReader::open(path)?.frame()?.voxel_um;
        let db = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX)
            .map_err(|e| eyre!("{}: {e}", path.display()))?;
        db.execute_batch(&format!("PRAGMA query_only=ON; PRAGMA cache_size=-{CACHE_KIB}"))?;
        let app: i64 = db.query_row("PRAGMA application_id", [], |r| r.get(0))?;
        let version: i64 = db.query_row("PRAGMA user_version", [], |r| r.get(0))?;
        if app != 0x56434643 || version != 1 {
            bail!("{} is not an AFV version 1 file", path.display());
        }

        let metadata: Vec<(String, String)> = db
            .prepare("SELECT key, value FROM metadata ORDER BY rowid")?
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<Result<_, _>>()?;
        let count = |key: &str| -> Result<i64> {
            let text = metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str());
            text.and_then(|t| t.parse().ok()).ok_or_else(|| eyre!("{}: no {key} in the metadata", path.display()))
        };
        let totals =
            Totals { fibers: count("fiber_count")?, points: count("point_count")?, blocks: count("block_count")? };
        Ok(Self { db, metadata, voxel_um, totals })
    }

    /// The ids, ascending, of a superset of the fibres the filter keeps: the box's fibres, else those the length
    /// index finds at least about the minimum length, else one family's. [`write`] applies the filter exactly
    fn candidates(&self, filter: &Filter) -> Result<Vec<i64>> {
        if let Some(region) = &filter.region {
            return self.touching(region);
        }

        let mut ids: Vec<i64> = match (filter.min_length_mm, filter.family) {
            (Some(mm), _) => {
                // the file's fibers_length index finds them; the bound is lowered a little, since the exact test is
                // on the length in mm as the writer computes it, not on voxels
                let voxels = mm * 1000.0 / self.voxel_um * (1.0 - 1e-9);
                let mut s = self.db.prepare("SELECT id FROM fibers WHERE length >= ?")?;
                s.query_map([voxels], |r| r.get(0))?.collect::<Result<_, _>>()?
            }
            (None, Some(family)) => {
                let mut s = self.db.prepare("SELECT id FROM fibers WHERE family = ?")?;
                s.query_map([family.label()], |r| r.get(0))?.collect::<Result<_, _>>()?
            }
            (None, None) => unreachable!("Filter::new refuses a filter with nothing to choose by"),
        };
        ids.sort_unstable();
        Ok(ids)
    }

    /// The ids of the fibres with a point in `region`, ascending: the R*Tree finds the blocks whose bounds meet the
    /// box, and each block's points are tested, since bounds can meet a box no point lies in
    fn touching(&self, region: &Region) -> Result<Vec<i64>> {
        let mut rows = self.db.prepare(
            "SELECT b.fiber_id, b.points FROM block_bounds r JOIN blocks b ON b.id = r.id \
             WHERE r.max_x >= ? AND r.min_x <= ? AND r.max_y >= ? AND r.min_y <= ? AND r.max_z >= ? AND r.min_z <= ?",
        )?;
        let (lo, hi) = (region.lo, region.hi);
        let mut rows = rows.query(params![lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]])?;
        let mut ids = Vec::new();
        while let Some(row) = rows.next()? {
            let id: i64 = row.get(0)?;
            let blob = row.get_ref(1)?.as_blob()?;
            if blob.is_empty() || !blob.len().is_multiple_of(POINT_BYTES) {
                bail!("fibre {id} has a block of {} bytes", blob.len());
            }
            let inside =
                blob.as_chunks::<8>().0.as_chunks::<3>().0.iter().any(|p| region.contains(p.map(f64::from_le_bytes)));
            if inside {
                ids.push(id);
            }
        }
        ids.sort_unstable();
        ids.dedup();
        Ok(ids)
    }
}

fn run(args: &Args) -> Result<()> {
    let filter = Filter::new(args)?;
    if fs::symlink_metadata(&args.output).is_ok() {
        bail!("--output {} already exists", args.output.display());
    }

    let start = Instant::now();
    let source = Source::open(&args.input)?;
    let candidates = source.candidates(&filter)?;
    log!("fibrs subset: {} candidate fibres ({:.1} s)", candidates.len(), start.elapsed().as_secs_f64());

    let parent = args.output.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let (temporary, _) = mkstemp(parent, Access::Owner)?;
    let _guard = TempGuard(temporary.clone());
    let kept = write(&source, &filter, &candidates, &temporary)?;
    if kept.fibers == 0 {
        bail!("no fibre of {} is in the subset", args.input.display());
    }
    afv::publish(&temporary, &args.output)?;
    log!(
        "fibrs subset: kept {} of {} fibres, {} points, {} blocks in {:.1} s",
        kept.fibers,
        source.totals.fibers,
        kept.points,
        kept.blocks,
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Copy the chosen fibres' rows into a new AFV at `path`; the totals it holds
///
/// `candidates` is a superset of the fibres the filter keeps, ascending
fn write(source: &Source, filter: &Filter, candidates: &[i64], path: &Path) -> Result<Totals> {
    let mut db = Connection::open(path)?;
    db.execute_batch("PRAGMA foreign_keys=ON; PRAGMA locking_mode=EXCLUSIVE")?;
    db.execute_batch(&format!("PRAGMA cache_size=-{CACHE_KIB}"))?;
    db.execute_batch(afv::SCHEMA)?;

    let tx = db.transaction()?;
    let mut kept = Totals::default();
    {
        let mut fibre = source.db.prepare("SELECT * FROM fibers WHERE id = ?")?;
        let mut blocks = source.db.prepare(
            "SELECT b.id, b.fiber_id, b.first_segment, b.points, r.min_x, r.max_x, r.min_y, r.max_y, r.min_z, r.max_z \
             FROM blocks b JOIN block_bounds r ON r.id = b.id WHERE b.fiber_id = ? ORDER BY b.first_segment",
        )?;
        let mut put_fibre = tx.prepare("INSERT INTO fibers VALUES(?,?,?,?,?,?,?,?,?,?,?,?)")?;
        let mut put_block = tx.prepare("INSERT INTO blocks VALUES(?,?,?,?)")?;
        let mut put_bound = tx.prepare("INSERT INTO block_bounds VALUES(?,?,?,?,?,?,?)")?;

        let mut copy = |id: i64| -> Result<()> {
            let row: Vec<Sql> = fibre.query_row([id], |r| (0..12).map(|i| r.get::<_, Sql>(i)).collect())?;
            let Sql::Text(family) = &row[2] else { bail!("fibre {id} has no family") };
            let (Sql::Integer(points), Sql::Real(length)) = (&row[3], &row[4]) else {
                bail!("fibre {id} has no point count or length")
            };
            // the length in mm as the writer computes it, so a fibre the writer kept at a minimum is kept here at it
            let short = filter.min_length_mm.is_some_and(|mm| length * source.voxel_um / 1000.0 < mm);
            if filter.family.is_some_and(|f| f.label() != family) || short {
                return Ok(());
            }
            put_fibre.execute(rusqlite::params_from_iter(&row))?;

            let mut rows = blocks.query([id])?;
            while let Some(r) = rows.next()? {
                let block: Vec<Sql> = (0..4).map(|i| r.get::<_, Sql>(i)).collect::<Result<_, _>>()?;
                let bound: Vec<Sql> = (4..10).map(|i| r.get::<_, Sql>(i)).collect::<Result<_, _>>()?;
                put_block.execute(rusqlite::params_from_iter(&block))?;
                put_bound.execute(rusqlite::params_from_iter(std::iter::once(&block[0]).chain(&bound)))?;
                kept.blocks += 1;
            }
            kept.fibers += 1;
            kept.points += points;
            Ok(())
        };

        candidates.iter().try_for_each(|&id| copy(id))?;
    }

    tx.execute_batch("CREATE INDEX fibers_length ON fibers(length DESC,id)")?;
    {
        let mut meta = tx.prepare("INSERT INTO metadata VALUES(?,?)")?;
        for (key, value) in source.metadata.iter().filter(|(k, _)| !RECOMPUTED.contains(&k.as_str())) {
            meta.execute(params![key, value])?;
        }
        let rows = [
            ("complete", json!(true)),
            ("fiber_count", json!(kept.fibers)),
            ("point_count", json!(kept.points)),
            ("block_count", json!(kept.blocks)),
            ("subset", filter.json(&source.totals)),
        ];
        for (key, value) in rows {
            meta.execute(params![key, py_json(&value)])?;
        }
    }
    tx.commit()?;
    afv::check(&db)?;
    db.close().map_err(|(_, e)| e)?;
    Ok(kept)
}
