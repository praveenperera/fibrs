//! Tiled tracing (`--tiles-yx`) end to end through the binary
//!
//! A 1 × 1 tiling runs the tiled path (crop, tile work directory, cut, join, export) and must write what the
//! untiled run writes, row for row. A 2 × 2 tiling of straight fibres that cross both seams must join every fibre
//! back whole. Tracing needs numpy's OpenBLAS: these tests run where `FIBRS_LAPACK` names it, are skipped
//! elsewhere, and fail on CI without it

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use rusqlite::types::Value;
use rusqlite::{Connection, OpenFlags};

/// A fresh directory for one test's outputs
fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("tile").join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn fibrs(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_fibrs")).args(args).output().unwrap()
}

fn text(out: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))
}

/// Whether tracing can run here; on CI it must
fn can_trace() -> bool {
    if std::env::var_os("FIBRS_LAPACK").is_some() {
        return true;
    }
    assert!(std::env::var_os("CI").is_none(), "CI must set FIBRS_LAPACK to numpy's OpenBLAS for the tracer tests");
    eprintln!("skipped: set FIBRS_LAPACK to numpy's OpenBLAS (libscipy_openblas64_*) to trace");
    false
}

/// Trace into `dir/NAME.afv` with `args` and the extra `options`
fn trace(dir: &Path, name: &str, args: &[&str], options: &[&str]) -> PathBuf {
    let afv = dir.join(format!("{name}.afv"));
    let work = dir.join(format!("{name}-work"));
    let mut all: Vec<&str> = args.to_vec();
    all.extend(["--threads", "2", "--output", afv.to_str().unwrap(), "--work-dir", work.to_str().unwrap()]);
    all.extend(options);
    let out = fibrs(&all);
    assert!(out.status.success(), "{name}: {}", text(&out));
    afv
}

/// Every table of an `.afv` in id order, and its metadata but the per-run uuid
fn tables(path: &Path) -> BTreeMap<&'static str, Vec<Vec<Value>>> {
    let db = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    let queries = [
        ("metadata", "SELECT key, value FROM metadata WHERE key != 'uuid' ORDER BY key"),
        ("fibers", "SELECT * FROM fibers ORDER BY id"),
        ("blocks", "SELECT * FROM blocks ORDER BY id"),
        ("block_bounds", "SELECT * FROM block_bounds ORDER BY id"),
    ];
    let mut out = BTreeMap::new();
    for (name, query) in queries {
        let mut stmt = db.prepare(query).unwrap();
        let n = stmt.column_count();
        let rows = stmt.query_map([], |r| (0..n).map(|i| r.get::<_, Value>(i)).collect()).unwrap();
        out.insert(name, rows.collect::<Result<Vec<Vec<Value>>, _>>().unwrap());
    }
    out
}

/// Each fibre's family and first and last points
fn ends(path: &Path) -> Vec<(String, [f64; 3], [f64; 3])> {
    let db = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    let mut stmt = db
        .prepare("SELECT f.id, f.family, b.points FROM fibers f JOIN blocks b ON b.fiber_id = f.id ORDER BY f.id, b.id")
        .unwrap();
    let rows = stmt
        .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?, r.get::<_, Vec<u8>>(2)?)))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let point = |b: &[u8], i: usize| -> [f64; 3] {
        [0, 1, 2].map(|a| f64::from_le_bytes(b[(3 * i + a) * 8..(3 * i + a + 1) * 8].try_into().unwrap()))
    };
    let mut out: Vec<(i64, String, [f64; 3], [f64; 3])> = Vec::new();
    for (id, family, blob) in rows {
        let last = point(&blob, blob.len() / 24 - 1);
        match out.last_mut() {
            Some(f) if f.0 == id => f.3 = last,
            _ => out.push((id, family, point(&blob, 0), last)),
        }
    }
    out.into_iter().map(|(_, family, a, b)| (family, a, b)).collect()
}

/// The synthetic fixture's arguments, as CI traces it against the original's output
fn synthetic() -> Vec<String> {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/synthetic/predictions.zarr");
    let fixture = fixture.to_str().unwrap().to_string();
    [
        "--predictions",
        &fixture,
        "--array",
        "semantic",
        "--channel-order",
        "VH",
        "--tracing-predictions",
        &fixture,
        "--tracing-array",
        "tracing",
        "--voxel-size-um",
        "9.362",
        "--storage",
        "compressed",
    ]
    .map(String::from)
    .to_vec()
}

/// The negative control: a 1 × 1 tiling writes the untiled run's `.afv`, every row and metadata value but the uuid
#[test]
fn one_tile_writes_the_untiled_output() {
    if !can_trace() {
        return;
    }
    let dir = scratch("one-tile");
    let args = synthetic();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let untiled = tables(&trace(&dir, "untiled", &args, &[]));
    let tiled = tables(&trace(&dir, "tiled", &args, &["--tiles-yx", "1", "1"]));
    assert!(untiled["fibers"].len() > 100, "the fixture traces {} fibres", untiled["fibers"].len());
    for (name, rows) in &untiled {
        assert_eq!(rows.len(), tiled[name].len(), "{name} row count");
        for (i, (a, b)) in rows.iter().zip(&tiled[name]).enumerate() {
            assert_eq!(a, b, "{name} row {i}");
        }
    }

    let manifest: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(dir.join("tiled-work/run.json")).unwrap()).unwrap();
    assert_eq!(manifest["tiling"]["grid_yx"], serde_json::json!([1, 1]));
    let manifest: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(dir.join("untiled-work/run.json")).unwrap()).unwrap();
    assert!(manifest.get("tiling").is_none(), "an untiled run records no tiling");
}

/// Native extent of the straight-fibre predictions: Z, Y and X
const SHAPE: [usize; 3] = [48, 640, 640];
/// Chunk edge in Y and X; Z is one chunk
const CHUNK: usize = 128;

/// Straight tubes across the whole extent: H tubes along x at z 14 and V tubes along y at z 34, none near the seams
/// a 2 × 2 tiling puts at 256, as uint8 `rint(255 × exp(-d² / 8))`
fn tube(c: usize, z: usize, y: usize, x: usize) -> u8 {
    let (zc, across) = if c == 1 { (14.0, y) } else { (34.0, x) };
    let centres = [96.0, 176.0, 400.0, 520.0];
    let d = |v: usize, w: f64| v as f64 - w;
    let near = centres.iter().map(|&w| d(across, w).powi(2) + d(z, zc).powi(2)).fold(f64::INFINITY, f64::min);
    (255.0 * (-near / 8.0).exp()).round_ties_even() as u8
}

/// Write the tubes as an uncompressed uint8 Zarr v2 array, only the chunks that hold any
fn write_tubes(dir: &Path) {
    fs::create_dir_all(dir).unwrap();
    let meta = serde_json::json!({
        "zarr_format": 2, "shape": [2, SHAPE[0], SHAPE[1], SHAPE[2]], "chunks": [2, SHAPE[0], CHUNK, CHUNK],
        "dtype": "|u1", "compressor": null, "fill_value": 0, "order": "C", "filters": null,
    });
    fs::write(dir.join(".zarray"), meta.to_string()).unwrap();
    let grid = SHAPE[1] / CHUNK;
    for (gy, gx) in (0..grid).flat_map(|y| (0..grid).map(move |x| (y, x))) {
        let mut bytes = Vec::with_capacity(2 * SHAPE[0] * CHUNK * CHUNK);
        for c in 0..2 {
            for z in 0..SHAPE[0] {
                for y in gy * CHUNK..(gy + 1) * CHUNK {
                    bytes.extend((gx * CHUNK..(gx + 1) * CHUNK).map(|x| tube(c, z, y, x)));
                }
            }
        }
        if bytes.iter().any(|&v| v != 0) {
            fs::write(dir.join(format!("0.0.{gy}.{gx}")), bytes).unwrap();
        }
    }
}

/// A 2 × 2 tiling joins every fibre that crosses a seam: the same fibres as untiled, their ends where the untiled
/// fibres end, and none ending at a seam
#[test]
fn two_by_two_tiles_join_fibres_across_both_seams() {
    if !can_trace() {
        return;
    }
    let dir = scratch("two-by-two");
    let zarr = dir.join("tubes.zarr");
    write_tubes(&zarr);
    let args = ["--predictions", zarr.to_str().unwrap(), "--voxel-size-um", "9.362", "--storage", "compressed"];
    let untiled = ends(&trace(&dir, "untiled", &args, &[]));
    let tiled = ends(&trace(&dir, "tiled", &args, &["--tiles-yx", "2", "2"]));

    assert!(untiled.len() >= 8, "untiled traced {} fibres: {untiled:?}", untiled.len());
    assert_eq!(tiled.len(), untiled.len(), "untiled {untiled:?}\ntiled {tiled:?}");
    let near = |a: [f64; 3], b: [f64; 3]| (0..3).map(|i| (a[i] - b[i]).powi(2)).sum::<f64>().sqrt() <= 2.0;
    for (family, a, b) in &untiled {
        let found = tiled
            .iter()
            .any(|(f, c, d)| f == family && ((near(*a, *c) && near(*b, *d)) || (near(*a, *d) && near(*b, *c))));
        assert!(found, "no tiled fibre matches the untiled {family} fibre {a:?}..{b:?}: {tiled:?}");
    }
    let seam = 256.0;
    for (family, a, b) in &tiled {
        for p in [a, b] {
            assert!(
                (p[0] - seam).abs() > 2.0 && (p[1] - seam).abs() > 2.0,
                "a {family} fibre ends at a seam: {a:?}..{b:?}"
            );
        }
    }
}

/// The tile options are checked before anything is traced
#[test]
fn tile_options_are_checked() {
    let dir = scratch("options");
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/synthetic/predictions.zarr");
    let base = |extra: &[&str]| {
        let mut args = vec!["--predictions", fixture.to_str().unwrap(), "--array", "semantic", "--voxel-size-um", "9"];
        let out = dir.join("out.afv");
        let work = dir.join("work");
        args.extend(["--output", out.to_str().unwrap(), "--work-dir", work.to_str().unwrap()]);
        args.extend(extra);
        let out = fibrs(&args);
        assert!(!out.status.success(), "{extra:?} ran");
        assert!(!dir.join("work").exists(), "{extra:?} made a work directory");
        text(&out)
    };
    assert!(base(&["--tile-halo", "128"]).contains("--tile-halo requires --tiles-yx"));
    assert!(base(&["--tiles-yx", "0", "1"]).contains("at least one tile along each axis"));
    assert!(base(&["--tiles-yx", "1", "1", "--tile-halo", "-1"]).contains("--tile-halo must be nonnegative"));
    // the fixture's 192-voxel extent holds no whole 256-voxel unit for a second tile
    assert!(base(&["--tiles-yx", "2", "1"]).contains("does not split into 2 tiles"));
}
