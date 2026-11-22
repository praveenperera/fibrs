//! `fibrs subset` end to end through the binary, on a complete AFV written here with its R*Tree
//!
//! The set has five fibres in a 64³ volume: one inside the box, one outside it, one whose only point in the box is in
//! its second block, one whose block bounds meet the box though no point lies in it, and a short H fibre in the box

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use rusqlite::types::Value as Sql;
use rusqlite::{Connection, OpenFlags, params};
use serde_json::{Value, json};

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

type Point = [f64; 3];

/// The box every test chooses by, x, y, z from 10 to 20
const BOX: [&str; 6] = ["10", "20", "10", "20", "10", "20"];

const UUID: &str = "0b0c3c5e-8a51-4a43-9d36-3b2f8f0e7a11";

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("subset").join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn fibrs(dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_fibrs")).current_dir(dir).arg("subset").args(args).output().unwrap()
}

fn text(out: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))
}

fn bounds(points: &[Point]) -> [f64; 6] {
    let mut b = [f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY];
    for p in points {
        for a in 0..3 {
            b[2 * a] = b[2 * a].min(p[a]);
            b[2 * a + 1] = b[2 * a + 1].max(p[a]);
        }
    }
    b
}

fn length(points: &[Point]) -> f64 {
    points.windows(2).map(|w| (0..3).map(|a| (w[1][a] - w[0][a]).powi(2)).sum::<f64>().sqrt()).sum()
}

/// A complete AFV at `path` with `fibres` as ids 1, 2, …; each fibre's blocks hold the points split at `split`
/// segments (sharing their end points, as the writer stores them), and block ids run on across fibres
fn set(path: &Path, fibres: &[(&str, Vec<Point>, usize)]) {
    let db = Connection::open(path).unwrap();
    db.execute_batch(SCHEMA).unwrap();
    let frame = json!({
        "coordinate_base_shape_zyx": [64, 64, 64],
        "vc_open_data_source_coordinate_level": 0,
        "vc_open_data_source_coordinate_scale_factor": 1,
        "vc_open_data_source_original_resolution": 10.0,
    });
    let root =
        json!({"type": "vc3d_fiber_collection", "scale": 1, "coordinate_order": "XYZ", "coordinate_units": "voxel"});
    let tracing = json!({"scale": 2, "center_offset": 0.5, "origin_xyz": [0, 0, 0]});
    let (mut block, mut points) = (0, 0);
    for (i, (family, q, split)) in fibres.iter().enumerate() {
        let id = i as i64 + 1;
        let b = bounds(q);
        let annotation = json!({"type": "vc3d_fiber", "sequence": id}).to_string();
        db.execute(
            "INSERT INTO fibers VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            params![
                id,
                format!("fiber_{id}.json"),
                family,
                q.len() as i64,
                length(q),
                b[0],
                b[1],
                b[2],
                b[3],
                b[4],
                b[5],
                annotation
            ],
        )
        .unwrap();
        for first in (0..q.len() - 1).step_by(*split) {
            block += 1;
            let pts = &q[first..(first + split + 1).min(q.len())];
            let blob: Vec<u8> = pts.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
            let b = bounds(pts);
            db.execute("INSERT INTO blocks VALUES(?,?,?,?)", params![block, id, first as i64, blob]).unwrap();
            db.execute(
                "INSERT INTO block_bounds VALUES(?,?,?,?,?,?,?)",
                params![block, b[0], b[1], b[2], b[3], b[4], b[5]],
            )
            .unwrap();
        }
        points += q.len() as i64;
    }
    let rows = [
        ("complete", json!(true)),
        ("uuid", json!(UUID)),
        ("root", root),
        ("frame", frame),
        ("tracing", tracing),
        ("fiber_count", json!(fibres.len())),
        ("point_count", json!(points)),
        ("block_count", json!(block)),
    ];
    for (key, value) in rows {
        db.execute("INSERT INTO metadata VALUES(?,?)", params![key, value.to_string()]).unwrap();
    }
}

/// The five fibres the tests choose from
fn five() -> Vec<(&'static str, Vec<Point>, usize)> {
    vec![
        // 1: inside the box, 10 voxels long (0.1 mm)
        ("V", vec![[12.0, 12.0, 11.0], [12.0, 12.0, 15.0], [12.0, 12.0, 19.0]], 256),
        // 2: outside the box
        ("V", vec![[40.0, 40.0, 40.0], [40.0, 40.0, 50.0]], 256),
        // 3: blocks of two segments; only the fourth point, in the second block, is in the box
        ("H", vec![[2.0, 15.0, 15.0], [5.0, 15.0, 15.0], [8.0, 15.0, 15.0], [15.0, 15.0, 15.0], [30.0, 15.0, 15.0]], 2),
        // 4: one diagonal segment whose bounds hold the box's corner, though the segment never enters it
        ("V", vec![[0.0, 25.0, 15.0], [25.0, 0.0, 15.0]], 256),
        // 5: inside the box and 1 voxel long (0.01 mm)
        ("H", vec![[15.0, 15.0, 15.0], [15.0, 15.0, 16.0]], 256),
    ]
}

/// Every row of a table in id order, every column as stored
fn table(path: &Path, sql: &str) -> Vec<Vec<Sql>> {
    let db = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    let mut s = db.prepare(sql).unwrap();
    let n = s.column_count();
    s.query_map([], |r| (0..n).map(|i| r.get::<_, Sql>(i)).collect()).unwrap().map(Result::unwrap).collect()
}

fn ids(path: &Path) -> Vec<i64> {
    table(path, "SELECT id FROM fibers ORDER BY id")
        .into_iter()
        .map(|r| match r[0] {
            Sql::Integer(i) => i,
            _ => unreachable!(),
        })
        .collect()
}

fn metadata(path: &Path, key: &str) -> Value {
    let db = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    let text: String = db.query_row("SELECT value FROM metadata WHERE key = ?", [key], |r| r.get(0)).unwrap();
    serde_json::from_str(&text).unwrap()
}

/// The rows of `sql` in `input` whose first column is one of `keep`
fn rows_of(input: &Path, sql: &str, keep: &[i64]) -> Vec<Vec<Sql>> {
    table(input, sql).into_iter().filter(|r| matches!(r[0], Sql::Integer(i) if keep.contains(&i))).collect()
}

#[test]
fn a_box_keeps_whole_fibres_with_a_point_in_it_and_their_rows_as_stored() {
    let dir = scratch("box");
    let input = dir.join("set.afv");
    set(&input, &five());

    let mut args = vec!["--input", "set.afv", "--output", "box.afv", "--box"];
    args.extend(BOX);
    let run = fibrs(&dir, &args);
    assert!(run.status.success(), "{}", text(&run));
    let out = dir.join("box.afv");

    // fibre 3 is kept for its one point in the box, whole; fibre 4's bounds meet the box but no point lies in it
    assert_eq!(ids(&out), [1, 3, 5]);
    let fibres = "SELECT * FROM fibers ORDER BY id";
    assert_eq!(table(&out, fibres), rows_of(&input, fibres, &[1, 3, 5]));
    // fibre 3's two blocks are ids 3 and 4, fibre 5's is 6
    let blocks = "SELECT * FROM blocks ORDER BY id";
    assert_eq!(table(&out, blocks), rows_of(&input, blocks, &[1, 3, 4, 6]));
    let rtree = "SELECT * FROM block_bounds ORDER BY id";
    assert_eq!(table(&out, rtree), rows_of(&input, rtree, &[1, 3, 4, 6]));

    assert_eq!(metadata(&out, "uuid"), json!(UUID));
    assert_eq!(metadata(&out, "complete"), json!(true));
    assert_eq!(metadata(&out, "frame"), metadata(&input, "frame"));
    let counts = ["fiber_count", "point_count", "block_count"].map(|k| metadata(&out, k));
    assert_eq!(counts, [json!(3), json!(10), json!(4)]);
    let subset = metadata(&out, "subset");
    assert_eq!(subset["box_xyz"], json!([[10.0, 20.0], [10.0, 20.0], [10.0, 20.0]]));
    assert_eq!(subset["source_fiber_count"], json!(5));

    let db = Connection::open_with_flags(&out, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    let check: String = db.query_row("PRAGMA integrity_check", [], |r| r.get(0)).unwrap();
    let rtree: String = db.query_row("SELECT rtreecheck('block_bounds')", [], |r| r.get(0)).unwrap();
    assert_eq!((check.as_str(), rtree.as_str()), ("ok", "ok"));
}

#[test]
fn length_and_family_filters_combine_with_the_box() {
    let dir = scratch("filters");
    set(&dir.join("set.afv"), &five());

    // at 10 µm a voxel, fibre 1 is 0.1 mm long and fibre 5 0.01 mm
    let run = fibrs(&dir, &["--input", "set.afv", "--output", "long.afv", "--min-length-mm", "0.05"]);
    assert!(run.status.success(), "{}", text(&run));
    assert_eq!(ids(&dir.join("long.afv")), [1, 2, 3, 4]);

    let mut args = vec!["--input", "set.afv", "--output", "h.afv", "--family", "H", "--box"];
    args.extend(BOX);
    let run = fibrs(&dir, &args);
    assert!(run.status.success(), "{}", text(&run));
    assert_eq!(ids(&dir.join("h.afv")), [3, 5]);
    assert_eq!(metadata(&dir.join("h.afv"), "subset")["family"], json!("H"));
}

#[test]
fn a_subset_with_no_filter_or_no_fibre_or_an_existing_output_is_refused() {
    let dir = scratch("refused");
    set(&dir.join("set.afv"), &five());

    let run = fibrs(&dir, &["--input", "set.afv", "--output", "all.afv"]);
    assert!(!run.status.success());
    assert!(text(&run).contains("give at least one of"), "{}", text(&run));

    let run = fibrs(&dir, &["--input", "set.afv", "--output", "none.afv", "--box", "50", "60", "0", "5", "0", "5"]);
    assert!(!run.status.success());
    assert!(text(&run).contains("no fibre"), "{}", text(&run));
    assert!(!dir.join("none.afv").exists());
    // the temporary file is removed as well
    let left: Vec<_> = fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name()).collect();
    assert_eq!(left, ["set.afv"]);

    fs::write(dir.join("taken.afv"), b"").unwrap();
    let run = fibrs(&dir, &["--input", "set.afv", "--output", "taken.afv", "--family", "V"]);
    assert!(!run.status.success());
    assert!(text(&run).contains("already exists"), "{}", text(&run));
}
