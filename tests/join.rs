//! `fibrs join` end to end through the binary, on two band traces written here
//!
//! The bands meet at z = 10. A fibre traced across the plane in both bands is cut on each side and joined back; a
//! fibre whose only point in the upper core lies on the plane leaves a run of one point three times, which is
//! dropped and counted; uncut fibres are written in band order before the joined ones, V before H

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

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

/// A fresh directory for one test's files
fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("join").join(name);
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

/// A complete band trace at `path`, traced from z = `origin_z` in a 20 × 64 × 64 volume, holding `fibres` in
/// order, each in one block (the reader does not need the R*Tree or the annotations)
fn band(path: &Path, origin_z: i64, fibres: &[(&str, Vec<Point>)]) {
    let db = Connection::open(path).unwrap();
    db.execute_batch(SCHEMA).unwrap();
    let frame = json!({
        "coordinate_base_shape_zyx": [20, 64, 64],
        "vc_open_data_source_coordinate_level": 0,
        "vc_open_data_source_coordinate_scale_factor": 1,
        "vc_open_data_source_original_resolution": 9.362,
    });
    let tracing = json!({"scale": 2, "center_offset": 0.5, "origin_xyz": [0, 0, origin_z]});
    for (key, value) in [("complete", json!(true)), ("frame", frame), ("tracing", tracing)] {
        db.execute("INSERT INTO metadata VALUES(?,?)", params![key, value.to_string()]).unwrap();
    }
    for (i, (family, points)) in fibres.iter().enumerate() {
        let id = i as i64 + 1;
        let z = points.iter().map(|p| p[2]);
        let (lo, hi) = (z.clone().fold(f64::INFINITY, f64::min), z.fold(f64::NEG_INFINITY, f64::max));
        db.execute(
            "INSERT INTO fibers VALUES(?,?,?,?,0,0,64,0,64,?,?,'{}')",
            params![id, format!("fiber_{id}.json"), family, points.len() as i64, lo, hi],
        )
        .unwrap();
        let blob: Vec<u8> = points.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
        db.execute("INSERT INTO blocks VALUES(?,?,0,?)", params![id, id, blob]).unwrap();
    }
}

/// The joined file's fibres in id order: family and points
fn fibres(path: &Path) -> Vec<(String, Vec<Point>)> {
    let db = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    let mut rows = db
        .prepare("SELECT f.family, b.points FROM fibers f JOIN blocks b ON b.fiber_id = f.id ORDER BY f.id, b.id")
        .unwrap();
    let rows = rows.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?))).unwrap();
    rows.map(|r| {
        let (family, blob) = r.unwrap();
        let points =
            blob.chunks(24).map(|p| [0, 1, 2].map(|a| f64::from_le_bytes(p[8 * a..8 * a + 8].try_into().unwrap())));
        (family, points.collect())
    })
    .collect()
}

fn metadata(path: &Path, key: &str) -> String {
    let db = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    db.query_row("SELECT value FROM metadata WHERE key = ?", [key], |r| r.get(0)).unwrap()
}

fn line(x: f64, zs: &[f64]) -> Vec<Point> {
    zs.iter().map(|&z| [x, 5.0, z]).collect()
}

#[test]
fn bands_are_cut_to_their_cores_and_joined_across_the_plane() {
    let dir = scratch("two-bands");
    // the lower band keeps z 0..10, the upper 10..20; both are traced from the whole 0..20
    band(
        &dir.join("low.afv"),
        0,
        &[
            ("V", line(5.0, &[2.0, 8.0, 12.0, 16.0])),
            ("V", line(20.0, &[1.0, 3.0])),
            // in the halo only: no point in the core
            ("H", line(30.0, &[12.0, 14.0])),
        ],
    );
    band(
        &dir.join("high.afv"),
        0,
        &[
            ("V", line(5.0, &[6.0, 12.0, 16.0])),
            // the only point in the upper core lies on the plane, and both crossings land on it
            ("V", line(30.0, &[8.0, 10.0, 9.0])),
            ("H", line(40.0, &[12.0, 13.0])),
        ],
    );
    let manifest = json!({
        "bands": [
            {"afv": "high.afv", "core_z": [10, 20], "window_z": [0, 20]},
            {"afv": "low.afv", "core_z": [0, 10], "window_z": [0, 20]},
        ],
        "coordinate_space": "PHerc0000/20250101000000",
        "provenance": {"operation": "halo_band_trace_join", "joinRadiusVoxels": 1.5},
    });
    fs::write(dir.join("bands.json"), manifest.to_string()).unwrap();

    let (out, stats) = (dir.join("scroll.afv"), dir.join("scroll.json"));
    let args = ["join", "--manifest", "bands.json", "--output", "scroll.afv", "--stats", "scroll.json"];
    let args: Vec<String> = args.iter().map(|a| a.to_string()).collect();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_fibrs"));
    let run = cmd.current_dir(&dir).args(&args).output().unwrap();
    assert!(run.status.success(), "{}", text(&run));

    let got = fibres(&out);
    let want = vec![
        ("V".to_string(), line(20.0, &[1.0, 3.0])),
        ("V".to_string(), line(5.0, &[2.0, 8.0, 10.0, 10.0, 12.0, 16.0])),
        ("H".to_string(), line(40.0, &[12.0, 13.0])),
    ];
    assert_eq!(got, want);

    let stats: Value = serde_json::from_str(&fs::read_to_string(&stats).unwrap()).unwrap();
    let v = &stats["families"]["V"];
    assert_eq!(v["uncut"], 1);
    assert_eq!((v["cut_runs"].clone(), v["cut_ends"].clone(), v["joins"].clone()), (json!(3), json!(4), json!(1)));
    assert_eq!(v["chains"], 2);
    assert_eq!(v["zero_length_dropped"], json!({"uncut": 0, "chains": 1}));
    assert_eq!(v["fibres"], 2);
    assert_eq!(stats["families"]["H"]["fibres"], 1);
    assert_eq!((stats["interior_planes"].clone(), stats["fiber_count"].clone()), (json!(1), json!(3)));
    assert_eq!(stats["coordinate_space"], "PHerc0000/20250101000000");

    // the public sets' layout: the coordinate space first in the frame, provenance under its own key
    assert!(metadata(&out, "frame").starts_with(r#"{"vc_open_data_coordinate_space":"PHerc0000/20250101000000","#));
    assert_eq!(metadata(&out, "provenance"), r#"{"operation":"halo_band_trace_join","joinRadiusVoxels":1.5}"#);
    assert_eq!(metadata(&out, "complete"), "true");
}

#[test]
fn a_band_traced_from_another_window_is_refused() {
    let dir = scratch("wrong-window");
    band(&dir.join("low.afv"), 4, &[("V", line(5.0, &[6.0, 8.0]))]);
    let manifest = json!({"bands": [{"afv": dir.join("low.afv"), "core_z": [4, 10], "window_z": [0, 20]}]});
    fs::write(dir.join("bands.json"), manifest.to_string()).unwrap();
    let (manifest, out) = (dir.join("bands.json"), dir.join("scroll.afv"));
    let run = fibrs(&["join", "--manifest", manifest.to_str().unwrap(), "--output", out.to_str().unwrap()]);
    assert!(!run.status.success());
    assert!(text(&run).contains("not from its window's first plane 0"), "{}", text(&run));
    assert!(!out.exists());
}

#[test]
fn a_band_with_points_outside_its_window_is_refused() {
    let dir = scratch("outside-window");
    band(&dir.join("low.afv"), 0, &[("V", line(5.0, &[6.0, 14.0]))]);
    // a window's last plane is excluded, as in the manifest's [z0, z1)
    let manifest = json!({"bands": [{"afv": "low.afv", "core_z": [0, 10], "window_z": [0, 14]}]});
    fs::write(dir.join("bands.json"), manifest.to_string()).unwrap();
    let (manifest, out) = (dir.join("bands.json"), dir.join("scroll.afv"));
    let run = fibrs(&["join", "--manifest", manifest.to_str().unwrap(), "--output", out.to_str().unwrap()]);
    assert!(!run.status.success());
    assert!(text(&run).contains("has a point at z 14, outside its window [0, 14]"), "{}", text(&run));
}

#[test]
fn stats_that_would_replace_the_output_or_an_input_are_refused() {
    let dir = scratch("stats-path");
    band(&dir.join("low.afv"), 0, &[("V", line(5.0, &[6.0, 8.0]))]);
    let manifest = json!({"bands": [{"afv": "low.afv", "core_z": [0, 10], "window_z": [0, 20]}]});
    fs::write(dir.join("bands.json"), manifest.to_string()).unwrap();
    let manifest = dir.join("bands.json");
    let join = |out: &Path, stats: &Path| {
        let (m, o, s) = (manifest.to_str().unwrap(), out.to_str().unwrap(), stats.to_str().unwrap());
        text(&fibrs(&["join", "--manifest", m, "--output", o, "--stats", s]))
    };

    let out = dir.join("scroll.afv");
    assert!(join(&out, &out).contains("--stats and --output are the same file"));
    assert!(join(&out, &dir.join("low.afv")).contains("is one of the join's inputs"));
    assert!(join(&manifest, &dir.join("scroll.json")).contains("is one of the join's inputs"));
    assert!(fs::read_to_string(&manifest).unwrap().contains("low.afv"));
    assert!(!out.exists());
}
