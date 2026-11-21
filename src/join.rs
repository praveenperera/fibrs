//! `fibrs join`: a scroll traced in overlapping Z windows, joined into one `.afv`
//!
//! A whole scroll is traced as Z bands: each band's window is its core planes plus a halo of the bands beside it.
//! The join cuts each band's fibres to its core and joins the cut ends across the planes between adjoining cores,
//! with the seam rule tiled tracing uses ([`crate::seams`]), the v2-run `join.py` and `halojoin.py` rule. It writes
//! what `join.py` writes: every V fibre, then every H fibre, each family's uncut fibres in band order and then its
//! joined chains, with the public sets' metadata. A fibre or chain of zero length is dropped and counted
//!
//! The bands come from a manifest, a JSON object:
//!
//! ```json
//! {
//!   "bands": [{"afv": "z08448-08960/fibers.afv", "core_z": [8448, 8960], "window_z": [8192, 9216]}],
//!   "coordinate_space": "PHerc1545/20250821151648",
//!   "source_path": "https://.../PHerc1545/volumes/....zarr",
//!   "provenance": {"operation": "halo_band_trace_join"}
//! }
//! ```
//!
//! `afv` is relative to the manifest's directory unless absolute. Every key but `bands` may be left out

use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, sync_channel};
use std::thread;
use std::time::Instant;

use clap::Parser;
use color_eyre::eyre::{Result, bail, eyre};
use serde_json::{Map, Value, json};

use crate::afv::read::{AfvReader, Frame};
use crate::afv::{Collection, ExportOptions, MinLength, TempGuard, export_afv, mkstemp};
use crate::log;
use crate::pyfmt::py_json_indent;
use crate::seams::{CoreBox, Face, FamilyReport, FamilySeams, Point, RADIUS, Reports};

/// The families in the order a trace writes them: label and the seams' number
const FAMILIES: [(&str, usize); 2] = [("V", 1), ("H", 2)];

/// Join a scroll's Z-band traces into one .afv
///
/// Each band's fibres are cut to its core planes, and the cut ends are joined across the planes between adjoining
/// cores: same family, at most --radius voxels apart, shortest first, each end once, never into a loop
#[derive(Parser, Debug)]
#[command(name = "fibrs join", version = env!("CARGO_PKG_VERSION"))]
pub struct Args {
    /// The bands' manifest (JSON): each band's .afv, core planes and window planes, and the output's coordinate
    /// space, source path and provenance
    #[arg(long)]
    manifest: PathBuf,
    /// The joined .afv; must not exist
    #[arg(long)]
    output: PathBuf,
    /// Write the join's counts here as JSON (join.py's per-family counts)
    #[arg(long)]
    stats: Option<PathBuf>,
    /// Native voxels within which two cut ends are joined
    #[arg(long, default_value_t = RADIUS)]
    radius: f64,
    /// Threads preparing the output's rows; 0 = one per core
    #[arg(long, default_value_t = 0)]
    threads: usize,
}

/// Whether `argv` asks for `fibrs join`; the drop-in `avf-spline-generator` has no subcommands
pub fn requested(argv: &[OsString]) -> bool {
    let Some(arg0) = argv.first() else { return false };
    !crate::run::dropin(arg0.as_encoded_bytes()) && argv.get(1).is_some_and(|a| a == "join")
}

/// Run `fibrs join` with the arguments after `join`; the process's exit code
pub fn main(argv: &[OsString]) -> i32 {
    let args = Args::parse_from(std::iter::once(OsString::from("fibrs join")).chain(argv.iter().cloned()));
    match run(&args) {
        Ok(()) => 0,
        Err(e) => {
            log!("fibrs join: {e:#}");
            1
        }
    }
}

/// One band: its trace and the planes it keeps and was traced from, z0..z1
#[derive(Clone, Debug)]
struct Band {
    afv: PathBuf,
    core: [i64; 2],
    window: [i64; 2],
}

/// The manifest, checked
struct Manifest {
    /// In core order
    bands: Vec<Band>,
    coordinate_space: Option<String>,
    source_path: Option<String>,
    provenance: Option<Map<String, Value>>,
}

/// A pair of planes `[z0, z1]` with z0 < z1
fn planes(v: &Value, what: &str) -> Result<[i64; 2]> {
    let pair: Option<[i64; 2]> =
        v.as_array().and_then(|a| a.iter().map(Value::as_i64).collect::<Option<Vec<_>>>()?.try_into().ok());
    match pair {
        Some([z0, z1]) if z0 < z1 => Ok([z0, z1]),
        _ => bail!("{what} must be two integer planes [z0, z1] with z0 < z1, not {v}"),
    }
}

impl Manifest {
    fn read(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path).map_err(|e| eyre!("{}: {e}", path.display()))?;
        let doc: Value = serde_json::from_str(&text).map_err(|e| eyre!("{}: {e}", path.display()))?;
        let base = path.parent().unwrap_or(Path::new(""));
        let text_key = |key: &str| -> Result<Option<String>> {
            match &doc[key] {
                Value::Null => Ok(None),
                Value::String(s) => Ok(Some(s.clone())),
                other => bail!("{}: {key} must be a string, not {other}", path.display()),
            }
        };
        let provenance = match &doc["provenance"] {
            Value::Null => None,
            Value::Object(m) => Some(m.clone()),
            other => bail!("{}: provenance must be an object, not {other}", path.display()),
        };
        let Some(items) = doc["bands"].as_array().filter(|b| !b.is_empty()) else {
            bail!("{}: bands must be a nonempty list", path.display());
        };

        let mut bands = Vec::with_capacity(items.len());
        for (i, b) in items.iter().enumerate() {
            let Some(afv) = b["afv"].as_str() else { bail!("{}: band {i} has no afv path", path.display()) };
            let core = planes(&b["core_z"], &format!("band {i}'s core_z"))?;
            let window = planes(&b["window_z"], &format!("band {i}'s window_z"))?;
            if window[0] > core[0] || core[1] > window[1] {
                bail!("band {i}'s core {core:?} is not inside its window {window:?}");
            }
            bands.push(Band { afv: base.join(afv), core, window });
        }
        bands.sort_by_key(|b| b.core[0]);
        for pair in bands.windows(2) {
            if pair[1].core[0] < pair[0].core[1] {
                bail!("the cores of {} and {} overlap", pair[0].afv.display(), pair[1].afv.display());
            }
        }

        let (coordinate_space, source_path) = (text_key("coordinate_space")?, text_key("source_path")?);
        Ok(Self { bands, coordinate_space, source_path, provenance })
    }

    /// That the output and the stats are two files, and neither is the manifest or a band's trace, which writing
    /// them would replace; a link at the stats path is safe, since the stats replace the path rather than write
    /// through it
    fn check_outputs(&self, args: &Args) -> Result<()> {
        if let Some(stats) = &args.stats
            && stats.is_dir()
        {
            bail!("--stats {} is a directory", stats.display());
        }

        // an existing file by its real path, so a link to an input is caught too; a file still to be written by its
        // directory's real path, which must exist, so a missing directory fails before the join rather than after
        let key = |p: &Path| -> Result<PathBuf> {
            if let Ok(real) = fs::canonicalize(p) {
                return Ok(real);
            }
            let name = p.file_name().ok_or_else(|| eyre!("{} is not a file path", p.display()))?;
            let dir = p.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
            Ok(fs::canonicalize(dir).map_err(|e| eyre!("{}: {e}", dir.display()))?.join(name))
        };
        let inputs = std::iter::once(&args.manifest).chain(self.bands.iter().map(|b| &b.afv));
        let inputs = inputs.map(|p| key(p)).collect::<Result<Vec<_>>>()?;

        let output = key(&args.output)?;
        let stats = args.stats.as_deref().map(key).transpose()?;
        for (flag, path) in [("--output", Some(&output)), ("--stats", stats.as_ref())] {
            if let Some(path) = path
                && inputs.contains(path)
            {
                bail!("{flag} {} is one of the join's inputs", path.display());
            }
        }
        if stats.as_ref() == Some(&output) {
            bail!("--stats and --output are the same file, {}", output.display());
        }

        Ok(())
    }

    /// That every band was traced in one volume: the same voxel size and shape, and the same scan wherever a band or
    /// the manifest names it; that scan, if any names it
    fn check_volume(&self, frames: &[Frame]) -> Result<Option<String>> {
        let frame = &frames[0];
        let bands = self.bands.iter().zip(frames);
        if let Some((b, f)) =
            bands.clone().find(|(_, f)| (f.voxel_um, f.volume_shape) != (frame.voxel_um, frame.volume_shape))
        {
            bail!("{} was traced in another volume: {f:?}, not {frame:?}", b.afv.display());
        }

        // bands traced without --coordinate-space name no scan, so only the named ones are compared
        let mut scan = self.coordinate_space.as_deref();
        for (b, space) in bands.filter_map(|(b, f)| f.coordinate_space.as_deref().map(|s| (b, s))) {
            match scan {
                Some(expected) if expected != space => {
                    bail!("{} was traced in {space}, not {expected}", b.afv.display())
                }
                _ => scan = Some(space),
            }
        }

        Ok(scan.map(str::to_owned))
    }

    /// The planes between adjoining cores, where cut ends are joined
    fn interior(&self) -> Vec<Face> {
        let shared = self.bands.windows(2).filter(|p| p[0].core[1] == p[1].core[0]);
        shared.map(|p| Face { axis: 2, at: p[0].core[1] as f64 }).collect()
    }
}

impl Band {
    /// The band's core: its planes on z, open on x and y
    fn core_box(&self) -> CoreBox {
        let (lo, hi) = (self.core[0] as f64, self.core[1] as f64);
        CoreBox { lo: [f64::NEG_INFINITY, f64::NEG_INFINITY, lo], hi: [f64::INFINITY, f64::INFINITY, hi] }
    }

    /// That the band's trace is complete and was traced from the band's window; its frame
    fn check(&self) -> Result<Frame> {
        let afv = AfvReader::open(&self.afv)?;
        let frame = afv.frame()?;
        if frame.origin_xyz[2] != self.window[0] {
            bail!(
                "{} was traced from z {}, not from its window's first plane {}",
                self.afv.display(),
                frame.origin_xyz[2],
                self.window[0]
            );
        }
        Ok(frame)
    }

    /// The band's fibres of one family, refused if any point lies outside the band's window
    fn fibres(&self, label: &str) -> Result<Vec<Vec<Point>>> {
        let fibres = AfvReader::open(&self.afv)?.fibres(label)?;
        // checked here on points already read rather than by scanning each band's fibre bounds up front, which
        // cost about two seconds a band on 1545
        let (lo, hi) = (self.window[0] as f64, self.window[1] as f64);
        if let Some(p) = fibres.iter().flatten().find(|p| p[2] < lo || p[2] >= hi) {
            bail!("{} has a point at z {}, outside its window {:?}", self.afv.display(), p[2], self.window);
        }

        Ok(fibres)
    }
}

/// A band's fibres of one family, read ahead of the export
type BandFibres = std::iter::Map<std::vec::IntoIter<Vec<Point>>, fn(Vec<Point>) -> Result<Vec<Point>>>;

/// The bands' fibres of one family, each band read on a thread of its own while the one before it is exported
struct ReadAhead {
    receive: Receiver<Result<(CoreBox, BandFibres)>>,
    /// Bands not yet received
    left: usize,
}

impl ReadAhead {
    fn new(bands: Vec<Band>, label: &'static str) -> Self {
        // one band read ahead, so the export holds at most three bands' fibres of a family: the one it is cutting,
        // the one waiting and the one being read
        let (send, receive) = sync_channel(1);
        let left = bands.len();
        thread::spawn(move || {
            for band in bands {
                let read = band.fibres(label);
                let piece = read.map(|fibres| (band.core_box(), fibres.into_iter().map(Ok as fn(_) -> _)));
                let failed = piece.is_err();
                // a failed send means the export has stopped, on an error of its own
                if send.send(piece).is_err() || failed {
                    return;
                }
            }
        });
        Self { receive, left }
    }
}

impl Iterator for ReadAhead {
    type Item = Result<(CoreBox, BandFibres)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.left == 0 {
            return None;
        }
        // the reader stops early only after sending its error, so a closed channel here means it panicked
        let Ok(piece) = self.receive.recv() else {
            self.left = 0;
            return Some(Err(eyre!("the band reader stopped before reading every band")));
        };
        self.left = if piece.is_err() { 0 } else { self.left - 1 };
        Some(piece)
    }
}

fn run(args: &Args) -> Result<()> {
    let started = Instant::now();
    if !(args.radius >= 0.0 && args.radius.is_finite()) {
        bail!("--radius must be a nonnegative number of voxels");
    }
    let manifest = Manifest::read(&args.manifest)?;
    manifest.check_outputs(args)?;
    let frames = manifest.bands.iter().map(Band::check).collect::<Result<Vec<_>>>()?;
    let scan = manifest.check_volume(&frames)?;
    let frame = &frames[0];
    let interior = manifest.interior();
    log!("Joining {} bands across {} planes into {}", manifest.bands.len(), interior.len(), args.output.display());

    let reports = Reports::default();
    let curves = FAMILIES.iter().flat_map(|&(label, family)| {
        let pieces = ReadAhead::new(manifest.bands.clone(), label);
        FamilySeams::new(family, pieces, interior.clone(), args.radius, reports.clone())
    });
    let opt = ExportOptions {
        voxel_um: frame.voxel_um,
        origin_xyz: [0, 0, 0],
        volume_shape: frame.volume_shape,
        minimum_mm: MinLength::Default,
        coordinate_space: scan.as_deref(),
        collection: Some(Collection {
            source_path: manifest.source_path.as_deref(),
            provenance: manifest.provenance.as_ref(),
        }),
    };
    let pool = rayon::ThreadPoolBuilder::new().num_threads(args.threads).build()?;
    let summary = pool.install(|| export_afv(curves, &args.output, &opt))?;
    let wall = started.elapsed().as_secs_f64();
    log!(
        "Wrote {} fibres ({} points) to {} in {wall:.1} s",
        summary.fiber_count,
        summary.point_count,
        args.output.display()
    );

    let Some(stats) = &args.stats else { return Ok(()) };
    let reports = reports.lock().expect("the export has finished").clone();
    let families: Map<String, Value> = reports.iter().map(|r| (label(r).to_string(), family_json(r))).collect();
    let mut doc = json!({
        "radius": args.radius,
        "interior_planes": interior.len(),
        "families": families,
        "complete": true,
        "fiber_count": summary.fiber_count,
        "point_count": summary.point_count,
        "block_count": summary.block_count,
        "wall_s": (wall * 10.0).round() / 10.0,
    });
    // join.py's sidecar gives the coordinate space after the radius, and the v2 run's finish.sh reads it
    if let Some(space) = &scan {
        doc.as_object_mut().expect("an object").shift_insert(1, "coordinate_space".into(), json!(space));
    }
    write_replacing(stats, &(py_json_indent(&doc) + "\n"))
}

/// Write `text` to a new file beside `path` and rename it over `path`, so a link at `path` is replaced rather than
/// written through to the file it names
fn write_replacing(path: &Path, text: &str) -> Result<()> {
    let fail = |e: std::io::Error| eyre!("{}: {e}", path.display());
    let dir = path.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
    // a short name of the writer's own, created exclusively, so a long stats name still fits and the cleanup only
    // ever removes a file this call made
    let temporary = mkstemp(dir)?;
    let _guard = TempGuard(temporary.clone());
    fs::write(&temporary, text).map_err(fail)?;
    fs::rename(&temporary, path).map_err(fail)
}

fn label(r: &FamilyReport) -> &'static str {
    FAMILIES.iter().find(|(_, f)| *f == r.family).map_or("?", |(l, _)| l)
}

/// A family's counts under join.py's keys
fn family_json(r: &FamilyReport) -> Value {
    json!({
        "uncut": r.uncut,
        "cut_runs": r.join.cut_runs,
        "cut_ends": r.join.cut_ends,
        "joins": r.join.joins,
        "chains": r.join.chains,
        "zero_length_dropped": {"uncut": r.zero_length_uncut, "chains": r.zero_length_chains},
        "fibres": r.fibres(),
    })
}
