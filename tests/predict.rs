//! `fibrs predict` without a GPU: its plans against the Python band reference's, and its argument checks
//!
//! `tests/fixtures/predict/expected.json` holds what the reference `TileFeed` (fibrs 0.2's Python
//! `fibrs.predict`) feeds the network for each fixture CT: per column, the batches (tile origins and the sha256 of
//! their bytes) and the air count. `scripts/predict_fixtures.py` writes the fixtures and the expectations;
//! `--dry-run` must print the same, which pins down the tiling, the columns, the reads and their reflection, the
//! air and villa skip rules and the batching

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/predict")
}

/// A fresh directory for one test's outputs
fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("predict").join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn predict(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_fibrs")).arg("predict").args(args).output().unwrap()
}

fn text(out: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))
}

/// `--dry-run`'s per-column lines, without its closing totals
fn dry_run(ct: &Path, case: &Value, out: &Path) -> Vec<Value> {
    let s = |k: &str| case[k].to_string();
    let mut args = vec![
        "--dry-run".to_string(),
        "--ct".into(),
        ct.to_string_lossy().into_owned(),
        "--mode".into(),
        case["mode"].as_str().unwrap().into(),
        "--column".into(),
        s("column"),
        "--batch".into(),
        s("batch"),
        "--model-dir".into(),
        fixtures().to_string_lossy().into_owned(),
        "--out".into(),
        out.to_string_lossy().into_owned(),
    ];
    if case["mode"] == "band" {
        args.extend(["--z0".into(), s("z0"), "--z1".into(), s("z1")]);
    }
    if let Some(only) = case["only"].as_array() {
        let list: Vec<String> = only.iter().map(Value::to_string).collect();
        args.extend(["--only-columns".into(), list.join(",")]);
    }
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let run = predict(&args);
    assert!(run.status.success(), "{}", text(&run));
    let mut lines: Vec<Value> =
        String::from_utf8(run.stdout).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    let totals = lines.pop().unwrap();
    assert_eq!(totals["columns"], json!(lines.len()));
    lines
}

#[test]
fn dry_run_feeds_what_the_python_reference_feeds() {
    let expected: Value = serde_json::from_slice(&fs::read(fixtures().join("expected.json")).unwrap()).unwrap();
    let cases = expected.as_object().unwrap();
    assert!(cases.len() >= 7);
    for (name, case) in cases {
        let out = scratch(&name.replace(':', "_"));
        let got = dry_run(&fixtures().join(case["ct"].as_str().unwrap()), case, &out);
        assert_eq!(Value::Array(got), case["columns"], "{name}");
        // a dry run writes nothing
        assert_eq!(fs::read_dir(&out).unwrap().count(), 0, "{name}");
    }
}

#[test]
fn a_resumed_run_skips_the_columns_its_progress_names() {
    let expected: Value = serde_json::from_slice(&fs::read(fixtures().join("expected.json")).unwrap()).unwrap();
    let case = &expected["band:band:4-36:c192:b3"];
    let columns = case["columns"].as_array().unwrap();
    let out = scratch("resume");
    let done: Vec<String> = [0, 3].iter().map(|&i| json!({"col": columns[i]["col"]}).to_string()).collect();
    // a record torn part way through its write (a crash or a full disk) does not count, nor stop the resume
    let torn = format!("{{\"col\": {}", columns[5]["col"]);
    fs::write(out.join("progress.jsonl"), done.join("\n") + "\n" + &torn).unwrap();
    let got = dry_run(&fixtures().join("band.zarr"), case, &out);
    let want: Vec<Value> =
        columns.iter().enumerate().filter(|(i, _)| ![0, 3].contains(i)).map(|(_, c)| c.clone()).collect();
    assert_eq!(got, want);
}

/// Runs `fibrs predict` on the band fixture with `extra` arguments; returns the status and the message
fn check(extra: &[&str]) -> (i32, String) {
    let ct = fixtures().join("band.zarr");
    let out = scratch("checks");
    let mut args = vec!["--ct", ct.to_str().unwrap(), "--model-dir", "unused", "--out", out.to_str().unwrap()];
    args.extend(extra);
    let run = predict(&args);
    (run.status.code().unwrap(), text(&run))
}

#[test]
fn bad_arguments_are_refused_with_the_reference_messages() {
    let cases: &[(&[&str], i32, &str)] = &[
        (&["--z0", "0"], 2, "band mode needs --z0 and --z1"),
        (&["--mode", "villa", "--batch", "4"], 2, "villa mode runs batches of 1; leave --batch out or pass 1"),
        (&["--z0", "0", "--z1", "8", "--precision", "fp16"], 2, "fp32 (the reference contract, the default) or bf16"),
        (&["--z0", "0", "--z1", "8", "--device", "cpu"], 2, "runs on a CUDA device (cuda, cuda:N), not \"cpu\""),
        (&["--z0", "0", "--z1", "8", "--batch", "0"], 1, "--batch and --readers must be at least 1"),
        (&["--z0", "0", "--z1", "8", "--batch", "17"], 1, "--batch is at most 16"),
        (&["--z0", "0", "--z1", "7", "--dry-run"], 1, "band height must be even"),
        (&["--z0", "30", "--z1", "50", "--dry-run"], 1, "the band [30, 50) must lie within the CT's 40 planes"),
        (&["--z0", "0", "--z1", "8", "--column", "100", "--dry-run"], 1, "column size must be an even multiple of 96"),
        (
            &["--z0", "0", "--z1", "8", "--column", "192", "--only-columns", "1,12", "--dry-run"],
            1,
            "--only-columns outside 0..11",
        ),
        (
            &["--mode", "villa", "--z0", "4", "--dry-run"],
            1,
            "villa mode runs the whole array: --z0/--z1 must be left out or be 0 40",
        ),
    ];
    for (args, code, msg) in cases {
        let (got, text) = check(args);
        assert_eq!(got, *code, "{args:?}: {text}");
        assert!(text.contains(msg), "{args:?}: {text}");
    }
}

#[test]
fn a_ct_that_is_not_3d_uint8_is_refused() {
    let ct = scratch("f4-ct");
    fs::write(
        ct.join(".zarray"),
        r#"{"chunks": [8, 8], "compressor": null, "dtype": "<f4", "fill_value": 0, "filters": null, "order": "C",
            "shape": [8, 8], "zarr_format": 2}"#,
    )
    .unwrap();
    let run = predict(&[
        "--ct",
        ct.to_str().unwrap(),
        "--z0",
        "0",
        "--z1",
        "2",
        "--model-dir",
        "m",
        "--out",
        "o",
        "--dry-run",
    ]);
    assert_eq!(run.status.code(), Some(1));
    assert!(text(&run).contains("expected a 3D uint8 CT array, got float32 (8, 8)"), "{}", text(&run));
}

#[test]
fn every_attempt_is_recorded_before_the_checkpoint_is_checked() {
    let model = scratch("model");
    for name in ["model.py", "plans.json", "dataset.json", "checkpoint_final.pth"] {
        fs::write(model.join(name), name).unwrap();
    }
    let out = scratch("attempt");
    let ct = fixtures().join("band.zarr");
    let args = ["--ct", ct.to_str().unwrap(), "--z0", "0", "--z1", "8", "--model-dir", model.to_str().unwrap()];
    let run = predict(&[&args[..], &["--out", out.to_str().unwrap(), "--expect-sha256", "00"]].concat());
    assert_eq!(run.status.code(), Some(1));
    let digest: String = Sha256::digest(b"checkpoint_final.pth").iter().map(|b| format!("{b:02x}")).collect();
    assert!(text(&run).contains(&format!("checkpoint sha256 {digest} != 00")), "{}", text(&run));
    let attempts = fs::read_to_string(out.join("attempts.jsonl")).unwrap();
    let rec: Value = serde_json::from_str(attempts.lines().next().unwrap()).unwrap();
    assert_eq!(rec["attempt"], rec["start"]);
    // nothing else is written before the checks pass
    assert!(!out.join("semantic").exists() && !out.join(".zattrs").exists());
}

#[test]
fn an_output_in_use_by_another_run_is_refused() {
    let out = scratch("locked");
    let held = fs::File::create(out.join("attempts.jsonl")).unwrap();
    held.lock().unwrap();
    let model = scratch("locked-model");
    for name in ["model.py", "plans.json", "dataset.json", "checkpoint_final.pth"] {
        fs::write(model.join(name), name).unwrap();
    }
    let ct = fixtures().join("band.zarr");
    let args = ["--ct", ct.to_str().unwrap(), "--z0", "0", "--z1", "8", "--model-dir", model.to_str().unwrap()];
    let run = predict(&[&args[..], &["--out", out.to_str().unwrap()]].concat());
    assert_eq!(run.status.code(), Some(1));
    assert!(text(&run).contains("is in use by another fibrs predict run"), "{}", text(&run));
    assert_eq!(fs::read_to_string(out.join("attempts.jsonl")).unwrap(), "");
    drop(held);
}

#[test]
fn a_fold_folder_as_the_model_folder_is_refused() {
    let model = scratch("fold_1");
    for name in ["model.py", "plans.json", "dataset.json", "checkpoint_final.pth"] {
        fs::write(model.join(name), name).unwrap();
    }
    let out = scratch("fold-dir-out");
    let ct = fixtures().join("band.zarr");
    let args = ["--ct", ct.to_str().unwrap(), "--z0", "0", "--z1", "8", "--model-dir", model.to_str().unwrap()];
    let run = predict(&[&args[..], &["--out", out.to_str().unwrap()]].concat());
    assert_eq!(run.status.code(), Some(1));
    assert!(text(&run).contains("is named like a fold folder"), "{}", text(&run));
    assert!(!out.join("semantic").exists());
}

#[test]
fn a_model_folder_whose_loader_would_pick_another_checkpoint_is_refused() {
    let model = scratch("fold-model");
    fs::create_dir_all(model.join("fold_0")).unwrap();
    for name in ["model.py", "plans.json", "dataset.json", "checkpoint_final.pth", "fold_0/checkpoint_final.pth"] {
        fs::write(model.join(name), name).unwrap();
    }
    let out = scratch("fold-out");
    let ct = fixtures().join("band.zarr");
    let args = ["--ct", ct.to_str().unwrap(), "--z0", "0", "--z1", "8", "--model-dir", model.to_str().unwrap()];
    let run = predict(&[&args[..], &["--out", out.to_str().unwrap()]].concat());
    assert_eq!(run.status.code(), Some(1));
    assert!(text(&run).contains("holds fold_0/checkpoint_final.pth"), "{}", text(&run));
    assert!(!out.join("semantic").exists());
}
