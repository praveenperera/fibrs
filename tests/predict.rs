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
    let args = dry_run_args(ct, case, out);
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    column_lines(predict(&args))
}

/// The `fibrs predict` arguments of a dry run of an `expected.json` case
fn dry_run_args(ct: &Path, case: &Value, out: &Path) -> Vec<String> {
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
    args
}

/// A finished dry run's per-column lines, without its closing totals
fn column_lines(run: Output) -> Vec<Value> {
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

/// A `current.zarr`-style `--ct` link repointed to another store while a run reads: every tile still comes from
/// the store the link named when the arguments were checked, in both modes
#[cfg(unix)]
#[test]
fn a_ct_link_repointed_during_a_run_is_not_followed() {
    use std::io::Write;
    use std::os::unix::fs::symlink;
    use std::process::Stdio;
    use std::sync::mpsc;
    use std::time::Duration;

    let expected: Value = serde_json::from_slice(&fs::read(fixtures().join("expected.json")).unwrap()).unwrap();
    // the held chunk is one the run reads early, after startup. Its villa neighbours are stored too, so villa's
    // selection, which counts only regular files as stored chunks, is the same while it is a FIFO
    for (name, held) in [("band:band:0-40:c96:b4", "0/0/0"), ("villa:villa:0-150:c192:b1", "2.2.2")] {
        let case = &expected[name];
        let dir = scratch(&format!("relink-{}", case["mode"].as_str().unwrap()));
        let first = dir.join("first.zarr");
        copy_tree(&fixtures().join(case["ct"].as_str().unwrap()), &first);
        // the store the link moves to has the same metadata and no chunks, so any tile read from it is fill
        let next = dir.join("next.zarr");
        fs::create_dir(&next).unwrap();
        fs::copy(first.join(".zarray"), next.join(".zarray")).unwrap();
        let current = dir.join("current.zarr");
        symlink(&first, &current).unwrap();

        // the held chunk becomes a FIFO, so the run waits at its read until the link has moved
        let fifo = first.join(held);
        let bytes = fs::read(&fifo).unwrap();
        fs::remove_file(&fifo).unwrap();
        mkfifo(&fifo);

        let out = dir.join("out");
        fs::create_dir(&out).unwrap();
        let mut args = dry_run_args(&current, case, &out);
        // one reader reads the tiles in order, so every tile after the held one is read after the link moves
        args.extend(["--readers".into(), "1".into()]);
        let mut run = Command::new(env!("CARGO_BIN_EXE_fibrs"))
            .arg("predict")
            .args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();

        // opening the FIFO to write returns once the run has opened it to read
        let (tx, rx) = mpsc::channel();
        let path = fifo.clone();
        std::thread::spawn(move || tx.send(fs::OpenOptions::new().write(true).open(path)));
        let Ok(writer) = rx.recv_timeout(Duration::from_secs(120)) else {
            let _ = run.kill();
            panic!("{name}: the run never read {held}: {}", text(&run.wait_with_output().unwrap()));
        };
        let mut writer = writer.unwrap();

        // moved the way a current.zarr link is updated: a new link renamed over the old one
        let moved = dir.join("current.zarr.new");
        symlink(&next, &moved).unwrap();
        fs::rename(&moved, &current).unwrap();
        // the chunk is a regular file again before the held read is answered, so a later read of it (a villa
        // chunk serves several tiles) finds it instead of a FIFO nothing will write to again
        let restored = dir.join("held.chunk");
        fs::write(&restored, &bytes).unwrap();
        fs::rename(&restored, &fifo).unwrap();
        writer.write_all(&bytes).unwrap();
        drop(writer);

        assert_eq!(Value::Array(column_lines(run.wait_with_output().unwrap())), case["columns"], "{name}");
    }
}

/// Copies the directory `from` to `to`, recursively
#[cfg(unix)]
fn copy_tree(from: &Path, to: &Path) {
    fs::create_dir_all(to).unwrap();
    for entry in fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let dest = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_tree(&entry.path(), &dest);
        } else {
            fs::copy(entry.path(), &dest).unwrap();
        }
    }
}

/// Makes a FIFO at `path`
#[cfg(unix)]
fn mkfifo(path: &Path) {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let c = CString::new(path.as_os_str().as_bytes()).unwrap();
    // SAFETY: `c` is a NUL-terminated path that lives through the call
    let rc = unsafe { libc::mkfifo(c.as_ptr(), 0o600) };
    assert_eq!(rc, 0, "mkfifo {}: {}", path.display(), std::io::Error::last_os_error());
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

/// What `--precision fp16` is refused with, naming the precisions this build's engine runs
const FP16_REFUSED: &str = if cfg!(fibrs_engine = "metal") {
    "the Metal engine runs fp32 only (the default)"
} else {
    "fp32 (the reference contract, the default) or bf16"
};

/// What `--device cpu` is refused with, naming the engine this build runs
const DEVICE_REFUSED: &str = if cfg!(fibrs_engine = "metal") {
    "runs on a Metal device (metal, metal:N), not \"cpu\""
} else {
    "runs on a CUDA device (cuda, cuda:N), not \"cpu\""
};

/// `--help` describes the precisions this build's engine runs: Metal's fp32 is true float32 and it has no bf16
#[test]
fn precision_help_matches_the_engine() {
    let help = text(&predict(&["--help"]));
    let line = help.lines().find(|l| l.contains("--precision")).expect("--precision in the help");
    let precision = &help[help.find(line).unwrap()..];
    let precision = &precision[..precision.find("--only-columns").unwrap_or(precision.len())];
    if cfg!(fibrs_engine = "metal") {
        assert!(precision.contains("only precision on Metal"), "{precision}");
        assert!(!precision.contains("TF32 tensor-core") && !precision.contains("bf16"), "{precision}");
    } else {
        assert!(precision.contains("TF32 tensor-core") && precision.contains("bf16"), "{precision}");
    }
}

#[test]
fn bad_arguments_are_refused_with_the_reference_messages() {
    let cases: &[(&[&str], i32, &str)] = &[
        (&["--z0", "0"], 2, "band mode needs --z0 and --z1"),
        (&["--mode", "villa", "--batch", "4"], 2, "villa mode runs batches of 1; leave --batch out or pass 1"),
        (&["--z0", "0", "--z1", "8", "--precision", "fp16"], 2, FP16_REFUSED),
        (&["--z0", "0", "--z1", "8", "--device", "cpu"], 2, DEVICE_REFUSED),
        (&["--z0", "0", "--z1", "8", "--voxel-size-um", "0"], 2, "--voxel-size-um must be a positive number"),
        (&["--z0", "0", "--z1", "8", "--voxel-size-um", "nan"], 2, "--voxel-size-um must be a positive number"),
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
fn a_checkpoint_mismatch_writes_nothing() {
    let model = scratch("model");
    for name in ["model.py", "plans.json", "dataset.json", "checkpoint_final.pth"] {
        fs::write(model.join(name), name).unwrap();
    }
    let out = scratch("attempt").join("pred.zarr");
    let ct = fixtures().join("band.zarr");
    let args = ["--ct", ct.to_str().unwrap(), "--z0", "0", "--z1", "8", "--model-dir", model.to_str().unwrap()];
    let run = predict(&[&args[..], &["--out", out.to_str().unwrap(), "--expect-sha256", "00"]].concat());
    assert_eq!(run.status.code(), Some(1));
    let digest: String = Sha256::digest(b"checkpoint_final.pth").iter().map(|b| format!("{b:02x}")).collect();
    assert!(text(&run).contains(&format!("checkpoint sha256 {digest} != 00")), "{}", text(&run));
    // the model is checked before the output is created
    assert!(!out.exists());
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
    // Windows locks are mandatory, so the file is read once the lock is released
    drop(held);
    assert_eq!(fs::read_to_string(out.join("attempts.jsonl")).unwrap(), "");
}

#[test]
fn an_output_that_overlaps_the_ct_is_refused() {
    let model = scratch("overlap-model");
    for name in ["model.py", "plans.json", "dataset.json", "checkpoint_final.pth"] {
        fs::write(model.join(name), name).unwrap();
    }
    let ct = scratch("overlap-ct").join("band.zarr");
    fs::create_dir_all(&ct).unwrap();
    for entry in fs::read_dir(fixtures().join("band.zarr")).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_file() {
            fs::copy(entry.path(), ct.join(entry.file_name())).unwrap();
        }
    }
    let before: Vec<_> = fs::read_dir(&ct).unwrap().map(|e| e.unwrap().file_name()).collect();
    let dir = ct.parent().unwrap();
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(&ct, dir.join("link.zarr")).unwrap();
        std::os::unix::fs::symlink(dir.join("gone"), dir.join("broken")).unwrap();
    }
    fs::write(dir.join("done.txt"), "").unwrap();
    let args = ["--ct", ct.to_str().unwrap(), "--z0", "0", "--z1", "8", "--model-dir", model.to_str().unwrap()];
    let refused = |out: PathBuf, why: &str| {
        let run = predict(&[&args[..], &["--out", out.to_str().unwrap()]].concat());
        assert_eq!(run.status.code(), Some(1));
        assert!(text(&run).contains(why), "{}", text(&run));
    };

    // the part of a path that exists resolves as the system resolves it, links and `..` included
    for out in [ct.clone(), ct.join("pred.zarr"), dir.to_path_buf(), ct.join("../band.zarr")] {
        refused(out, "overlaps --ct");
    }
    // a path the system would not open is refused, whatever it would name lexically. Windows resolves `..`
    // lexically before it opens a path, so there these name what they read as
    #[cfg(unix)]
    {
        for out in [dir.join("link.zarr"), dir.join("link.zarr/pred.zarr")] {
            refused(out, "overlaps --ct");
        }
        refused(dir.join("broken/../pred.zarr"), "which does not resolve");
        refused(dir.join("done.txt/../pred.zarr"), "steps through");
        refused(dir.join("missing/../pred.zarr"), "uses .. after");
        refused(ct.join("0.0.0/../../pred.zarr"), "uses .. after");
    }

    let after: Vec<_> = fs::read_dir(&ct).unwrap().map(|e| e.unwrap().file_name()).collect();
    assert_eq!(before, after);
    assert!(!dir.join("missing").exists() && !dir.join("pred.zarr").exists());
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
    // the model folder is checked before anything is written
    assert!(!out.join("attempts.jsonl").exists());
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
