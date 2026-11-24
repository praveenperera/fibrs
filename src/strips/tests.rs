//! Parity of the CPU stages against villa's batched path on fixtures from
//! `private-workbench/ops/vesuvius/strips-native/dump_geom_fixtures.py`
//!
//! Each stage runs on the fixture's inputs for that stage, so a difference shows where it starts: load (AFV row,
//! points, densify, kink repair, arcs), CT seed headings (cube, structure tensor, eigenpairs, heading or error),
//! plan, finish with the fixture's frame-model outputs, stitch with the fixture's strip-model outputs, and write.
//! The written files go to `rust_out` beside the fixtures' `py_out`, for `compare_geom_outputs.py`
//!
//! `FIBRS_GEOM_FIXTURES=DIR cargo test --release strips::tests -- --ignored --nocapture`

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use serde_json::{Map, Value};

use super::ct::Volume;
use super::fibre::{Afv, Fibre, FibreRow};
use super::geom::{DecisionPlan, Geometry, Umbilicus, tube_points};
use super::np::{M3, V3, dot};
use super::out::{Record, Settings, run_json};
use super::seed::{RADIUS, SIDE, SeedHeadingError, Workspace, eigh, oriented_seed_heading};
use super::stitch::stitched_strip;
use super::types::{FrameResult, STRIP_COLS, STRIP_ROWS};

const AFV: &str = "/shared/allscrolls/publish/hf/PHerc0211/PHerc0211_afv_v2.afv";
const CT: &str = "s3://vesuvius-challenge-open-data/PHerc0211/volumes/20250821151803-9.362um-1.2m-113keV-masked.zarr";
const CT_CACHE: &str = "/build/vesuvius/strips-scale/ctcache";
const GRID_SCALE: f64 = 2.0;
const INPUT_SCALE: f64 = 2.0;
const STEP: f64 = 8.0;
const RUNNER: &str = "fixture-runner@0";

/// How a field must agree
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Bar {
    /// float64: relative difference
    F64,
    /// float32: within 1 unit in the last place, or 1e-12 of an f64 cancellation near zero (where an f64 input's
    /// last bit shows through the cast)
    F32,
    /// integers, bools, strings: bit for bit
    Exact,
}

/// One field's differences over every fixture
#[derive(Clone, Debug)]
struct Stat {
    bar: Bar,
    n: usize,
    max_abs: f64,
    max_rel: f64,
    max_ulp: u64,
    mismatches: usize,
    worst: String,
}

/// Every field's differences
#[derive(Default)]
struct Stats(BTreeMap<String, Stat>);

impl Stats {
    fn compare(&mut self, fixture: &str, field: &str, bar: Bar, rust: &[f64], py: &[f64]) {
        let stat = self.0.entry(field.to_string()).or_insert(Stat {
            bar,
            n: 0,
            max_abs: 0.0,
            max_rel: 0.0,
            max_ulp: 0,
            mismatches: 0,
            worst: String::new(),
        });
        if rust.len() != py.len() {
            stat.mismatches += 1;
            stat.worst = format!("{fixture}: length {} vs {}", rust.len(), py.len());
            return;
        }

        for (i, (&r, &p)) in rust.iter().zip(py).enumerate() {
            stat.n += 1;
            if r.to_bits() == p.to_bits() || (r.is_nan() && p.is_nan()) {
                continue;
            }

            let abs = (r - p).abs();
            let rel = abs / p.abs().max(r.abs()).max(f64::MIN_POSITIVE);
            let ulp = if bar == Bar::F32 { ulps32(r as f32, p as f32) } else { ulps64(r, p) };
            let bad = match bar {
                Bar::F64 => rel > 1e-9 || abs.is_nan(),
                Bar::F32 => ulp > 1 && abs > 1e-12,
                Bar::Exact => r != p,
            };
            if bad {
                stat.mismatches += 1;
            }
            if abs > stat.max_abs || abs.is_nan() {
                stat.worst = format!("{fixture}[{i}]: {r:?} vs {p:?}");
            }
            stat.max_abs = stat.max_abs.max(abs);
            stat.max_rel = stat.max_rel.max(rel);
            stat.max_ulp = stat.max_ulp.max(ulp);
        }
    }

    fn merge(&mut self, other: Stats) {
        for (k, o) in other.0 {
            match self.0.get_mut(&k) {
                None => {
                    self.0.insert(k, o);
                }
                Some(s) => {
                    if o.max_abs > s.max_abs || (s.worst.is_empty() && !o.worst.is_empty()) {
                        s.worst = o.worst;
                    }
                    s.n += o.n;
                    s.max_abs = s.max_abs.max(o.max_abs);
                    s.max_rel = s.max_rel.max(o.max_rel);
                    s.max_ulp = s.max_ulp.max(o.max_ulp);
                    s.mismatches += o.mismatches;
                }
            }
        }
    }

    fn table(&self) -> String {
        let mut s = String::from("| field | bar | values | max abs | max rel | max ulp | over bar | worst |\n");
        s.push_str("|---|---|---:|---:|---:|---:|---:|---|\n");
        for (k, v) in &self.0 {
            s.push_str(&format!(
                "| {k} | {:?} | {} | {:.3e} | {:.3e} | {} | {} | {} |\n",
                v.bar, v.n, v.max_abs, v.max_rel, v.max_ulp, v.mismatches, v.worst
            ));
        }
        s
    }
}

fn ulps64(a: f64, b: f64) -> u64 {
    let key = |x: f64| {
        let i = x.to_bits() as i64;
        if i < 0 { i64::MIN - i } else { i }
    };
    key(a).abs_diff(key(b))
}

fn ulps32(a: f32, b: f32) -> u64 {
    let key = |x: f32| {
        let i = i64::from(x.to_bits() as i32);
        if i < 0 { i64::from(i32::MIN) - i } else { i }
    };
    key(a).abs_diff(key(b))
}

/// One fibre's fixture directory: raw little-endian arrays named in manifest.json
struct Fixture {
    dir: PathBuf,
    name: String,
    manifest: Map<String, Value>,
}

impl Fixture {
    fn open(dir: &Path) -> Self {
        let manifest = serde_json::from_slice(&fs::read(dir.join("manifest.json")).unwrap()).unwrap();
        Self { dir: dir.to_path_buf(), name: dir.file_name().unwrap().to_string_lossy().into_owned(), manifest }
    }

    fn meta(&self, key: &str) -> &Value {
        &self.manifest["_meta"][key]
    }

    fn has(&self, name: &str) -> bool {
        self.manifest.contains_key(name)
    }

    /// The array as float64 (exact for every stored dtype)
    fn get(&self, name: &str) -> Vec<f64> {
        let bytes = fs::read(self.dir.join(format!("{name}.bin"))).unwrap();
        match self.manifest[name][0].as_str().unwrap() {
            "f8" => bytes.as_chunks::<8>().0.iter().map(|&b| f64::from_le_bytes(b)).collect(),
            "f4" => bytes.as_chunks::<4>().0.iter().map(|&b| f64::from(f32::from_le_bytes(b))).collect(),
            "i8" => bytes.as_chunks::<8>().0.iter().map(|&b| i64::from_le_bytes(b) as f64).collect(),
            "u1" => bytes.iter().map(|&b| f64::from(b)).collect(),
            other => panic!("{name}: dtype {other}"),
        }
    }

    fn v3s(&self, name: &str) -> Vec<V3> {
        self.get(name).as_chunks::<3>().0.to_vec()
    }

    fn m3s(&self, name: &str) -> Vec<M3> {
        self.get(name).as_chunks::<3>().0.as_chunks::<3>().0.to_vec()
    }

    fn f32s(&self, name: &str) -> Vec<f32> {
        self.get(name).into_iter().map(|v| v as f32).collect()
    }

    fn grids(&self, name: &str) -> Vec<[[f32; STRIP_COLS]; STRIP_ROWS]> {
        self.f32s(name).as_chunks::<STRIP_COLS>().0.as_chunks::<STRIP_ROWS>().0.to_vec()
    }

    /// A ragged array (rows of 3) split by its `_len`
    fn ragged(&self, name: &str) -> Vec<Vec<f64>> {
        let flat = self.get(name);
        let mut at = 0;
        self.get(&format!("{name}_len"))
            .into_iter()
            .map(|n| {
                let n = n as usize * 3;
                at += n;
                flat[at - n..at].to_vec()
            })
            .collect()
    }

    fn row(&self) -> FibreRow {
        let r = self.meta("row").as_array().unwrap();
        FibreRow {
            id: r[0].as_i64().unwrap(),
            family: r[2].as_str().unwrap().into(),
            length: r[3].as_f64().unwrap(),
            bbox: std::array::from_fn(|k| r[4 + k].as_f64().unwrap()),
        }
    }
}

fn flat3(v: &[V3]) -> Vec<f64> {
    v.iter().flatten().copied().collect()
}

fn flat33(v: &[M3]) -> Vec<f64> {
    v.iter().flatten().flatten().copied().collect()
}

fn f32s(v: impl IntoIterator<Item = f32>) -> Vec<f64> {
    v.into_iter().map(f64::from).collect()
}

/// What a fixture run shares: the CT, the umbilicus and the write settings
struct Env<'a> {
    root: &'a Path,
    afv: &'a Afv,
    ct: &'a Volume,
    umbilicus: &'a Umbilicus,
    settings: &'a Settings,
}

impl Env<'_> {
    fn geometry(&self) -> Geometry<'_> {
        Geometry {
            ct: self.ct,
            input_scale: INPUT_SCALE,
            trace_per_patch: self.settings.trace_per_patch,
            umbilicus: self.umbilicus,
        }
    }

    /// The settings a fixture writes with: a synthetic fibre writes to its own tree, as the dumper's
    fn settings(&self, synthetic: bool) -> Settings {
        let out = self.root.join(if synthetic { "rust_out_trunc" } else { "rust_out" });
        Settings { npz: Some(out.join("npz")), out: Some(out), ..self.settings.clone() }
    }
}

/// The fibre's load: the AFV row and points, densified and kink-repaired, against villa's
fn load(env: &Env, fx: &Fixture, stats: &mut Stats) -> Fibre {
    let name = &fx.name;
    let row = fx.row();
    let selected = env.afv.select(0.0, env.settings.voxel_size_um, Some(&[row.id])).unwrap();
    let r = &selected[0];
    let same_row = r.id == row.id
        && r.family == row.family
        && r.length.to_bits() == row.length.to_bits()
        && r.bbox.map(f64::to_bits) == row.bbox.map(f64::to_bits);
    stats.compare(name, "load.row", Bar::Exact, &[f64::from(u8::from(same_row))], &[1.0]);
    let raw = env.afv.points(row.id).unwrap();
    stats.compare(name, "load.afv_points", Bar::Exact, &flat3(&raw), &fx.get("afv_raw"));
    if fx.meta("synthetic").as_bool().unwrap() {
        // the truncated fibre is villa's: its points are the fixture's
        return Fibre {
            row,
            points: fx.v3s("points"),
            s: fx.get("s"),
            kink_repairs: fx.meta("kink_repairs").as_u64().unwrap() as usize,
        };
    }

    let fibre = Fibre::from_base(row, &raw, GRID_SCALE).unwrap();
    stats.compare(name, "load.points", Bar::F64, &flat3(&fibre.points), &fx.get("points"));
    stats.compare(name, "load.s", Bar::F64, &fibre.s, &fx.get("s"));
    let repairs = fx.meta("kink_repairs").as_u64().unwrap() as f64;
    stats.compare(name, "load.kink_repairs", Bar::Exact, &[fibre.kink_repairs as f64], &[repairs]);
    fibre
}

/// Every CT seed heading villa computed for the fibre, recomputed from its inputs
fn seeds(env: &Env, fx: &Fixture, work: &mut Workspace, stats: &mut Stats) {
    let name = &fx.name;
    let count = fx.meta("seed_count").as_u64().unwrap_or(0) as usize;
    if count == 0 {
        return;
    }

    let (pos, direction) = (fx.v3s("seed_call_pos"), fx.v3s("seed_call_direction"));
    let (start, centre) = (fx.v3s("seed_call_start"), fx.v3s("seed_call_centre"));
    let (tensors, values, vectors) =
        (fx.m3s("seed_call_tensor"), fx.v3s("seed_call_values"), fx.m3s("seed_call_vectors"));
    let headings = fx.v3s("seed_call_heading");
    for i in 0..count {
        let family = fx.meta("seed_families")[i].as_str().unwrap();
        let error = fx.meta("seed_errors")[i].as_str();
        let result = oriented_seed_heading(env.ct, INPUT_SCALE, pos[i], family, direction[i], work);
        let message = result.as_ref().err().map(|e| match e.downcast_ref::<SeedHeadingError>() {
            Some(s) => format!("SeedHeadingError: {s}"),
            None => format!("other: {e}"),
        });
        let same_outcome = message.as_deref() == error;
        stats.compare(name, "seed.outcome", Bar::Exact, &[f64::from(u8::from(same_outcome))], &[1.0]);
        if !same_outcome {
            eprintln!("{name} seed {i}: {message:?} vs {error:?}");
        }
        if let Ok(h) = result {
            stats.compare(name, "seed.heading", Bar::F64, &h, &headings[i]);
        }
        if error.is_some_and(|e| e.contains("boundary")) {
            continue;
        }

        // the tensor and eigenpairs of the same cube
        let c = [pos[i][2] * INPUT_SCALE, pos[i][1] * INPUT_SCALE, pos[i][0] * INPUT_SCALE];
        let s: [i64; 3] = std::array::from_fn(|a| c[a].floor() as i64 - RADIUS);
        let local: V3 = std::array::from_fn(|a| c[a] - s[a] as f64);
        stats.compare(name, "seed.start", Bar::Exact, &s.map(|v| v as f64), &start[i]);
        stats.compare(name, "seed.centre", Bar::F64, &local, &centre[i]);
        env.ct.read_region(s.map(|v| v as usize), [SIDE; 3], &mut work.cube).unwrap();
        let t = work.structure_tensor(local);
        stats.compare(name, "seed.tensor", Bar::F64, &flat33(&[t]), &flat33(&[tensors[i]]));
        // villa's eigh, like sheet_normal's, runs on the tensor reversed to zyx
        let zyx: M3 = std::array::from_fn(|a| std::array::from_fn(|b| t[2 - a][2 - b]));
        let (w, v) = eigh(&zyx);
        stats.compare(name, "seed.eigenvalues", Bar::F64, &w, &values[i]);
        // eigenvectors up to sign: LAPACK and Jacobi pick their own
        let aligned: M3 = std::array::from_fn(|r| {
            std::array::from_fn(|k| {
                let ours = [v[0][k], v[1][k], v[2][k]];
                let theirs = [vectors[i][0][k], vectors[i][1][k], vectors[i][2][k]];
                if dot(ours, theirs) < 0.0 { -v[r][k] } else { v[r][k] }
            })
        });
        let normal = |m: &M3| [m[0][2], m[1][2], m[2][2]];
        stats.compare(name, "seed.normal", Bar::F64, &normal(&aligned), &normal(&vectors[i]));
        stats.compare(name, "seed.eigenvectors", Bar::F64, &flat33(&[aligned]), &flat33(&[vectors[i]]));
    }
}

fn compare_plan(fx: &Fixture, plans: &[DecisionPlan], stats: &mut Stats) {
    let name = &fx.name;
    let pick = |f: &dyn Fn(&DecisionPlan) -> Vec<f64>| plans.iter().flat_map(f).collect::<Vec<f64>>();
    stats.compare(name, "plan.decisions", Bar::Exact, &[plans.len() as f64], &[fx.meta("decisions").as_f64().unwrap()]);
    stats.compare(name, "plan.pos", Bar::F64, &pick(&|p| p.request.pos.to_vec()), &fx.get("pos"));
    stats.compare(name, "plan.frame", Bar::F64, &pick(&|p| flat33(&[p.frame])), &fx.get("frame"));
    stats.compare(name, "plan.hist_local", Bar::F64, &pick(&|p| flat3(&p.hist_local)), &fx.get("hist_local"));
    stats.compare(name, "plan.hmask", Bar::Exact, &pick(&|p| f32s(p.hmask)), &fx.get("hmask"));
    stats.compare(name, "plan.seed_pos", Bar::F64, &pick(&|p| p.seed_pos.to_vec()), &fx.get("seed_pos"));
    stats.compare(name, "plan.seed_tangent", Bar::F64, &pick(&|p| p.seed_tangent.to_vec()), &fx.get("seed_tangent"));
    stats.compare(name, "plan.seed_age", Bar::F64, &pick(&|p| vec![p.seed_age]), &fx.get("seed_age"));
    stats.compare(
        name,
        "plan.prev_prediction",
        Bar::F64,
        &pick(&|p| flat3(&p.prev_prediction)),
        &fx.get("prev_prediction"),
    );
    let observed = fx.ragged("observed_path");
    stats.compare(
        name,
        "plan.observed_len",
        Bar::Exact,
        &pick(&|p| vec![p.observed.len() as f64]),
        &fx.get("observed_path_len"),
    );
    stats.compare(name, "plan.observed_path", Bar::F64, &pick(&|p| flat3(&p.observed)), &observed.concat());
    stats.compare(name, "request.positions", Bar::F64, &pick(&|p| p.request.pos.to_vec()), &fx.get("req_positions"));
    stats.compare(name, "request.frames", Bar::F64, &pick(&|p| flat33(&[p.request.frame])), &fx.get("req_frames"));
    stats.compare(name, "request.paths", Bar::F32, &pick(&|p| f32s(p.request.paths)), &fx.get("req_paths"));
    stats.compare(
        name,
        "request.families",
        Bar::Exact,
        &pick(&|p| vec![p.request.family as i64 as f64]),
        &fx.get("req_families"),
    );
}

fn finish(env: &Env, fx: &Fixture, fibre: &Fibre, plans: &[DecisionPlan], stats: &mut Stats) {
    let name = &fx.name;
    let geometry = env.geometry();
    let (heading, normal) = (fx.f32s("frame_heading"), fx.f32s("frame_normal"));
    let requests: Vec<_> = plans
        .iter()
        .enumerate()
        .map(|(k, p)| {
            let out = FrameResult {
                heading: [heading[3 * k], heading[3 * k + 1], heading[3 * k + 2]],
                normal: [normal[3 * k], normal[3 * k + 1], normal[3 * k + 2]],
            };
            p.finish(&geometry, fibre, &out)
        })
        .collect();
    let pick = |f: &dyn Fn(&super::types::StripRequest) -> Vec<f64>| requests.iter().flat_map(f).collect::<Vec<f64>>();
    stats.compare(name, "finish.positions", Bar::F64, &pick(&|r| r.pos.to_vec()), &fx.get("fin_positions"));
    stats.compare(name, "finish.frames", Bar::F64, &pick(&|r| flat33(&[r.frame])), &fx.get("fin_frames"));
    stats.compare(
        name,
        "finish.hist",
        Bar::F32,
        &pick(&|r| f32s(r.hist.iter().flatten().copied())),
        &fx.get("fin_hist"),
    );
    stats.compare(name, "finish.hmask", Bar::Exact, &pick(&|r| f32s(r.hmask)), &fx.get("fin_hmask"));
    stats.compare(name, "finish.seed", Bar::F32, &pick(&|r| f32s(r.seed)), &fx.get("x_seed"));
    stats.compare(
        name,
        "finish.seed_mask",
        Bar::Exact,
        &pick(&|r| vec![f64::from(r.seed_mask)]),
        &fx.get("x_seed_mask"),
    );
    stats.compare(name, "finish.seed_age", Bar::F32, &pick(&|r| vec![f64::from(r.seed_age)]), &fx.get("x_seed_age"));
    stats.compare(name, "finish.seed_tangent", Bar::F32, &pick(&|r| f32s(r.seed_tangent)), &fx.get("x_seed_tangent"));
    stats.compare(
        name,
        "finish.path_geometry",
        Bar::F32,
        &pick(&|r| f32s(r.path_geometry.iter().flatten().copied())),
        &fx.get("x_path_geometry"),
    );
    stats.compare(
        name,
        "finish.path_geometry_valid",
        Bar::Exact,
        &pick(&|r| r.path_geometry_valid.iter().map(|&b| f64::from(u8::from(b))).collect()),
        &fx.get("x_path_geometry_valid"),
    );
    stats.compare(name, "finish.tube_len", Bar::Exact, &pick(&|r| vec![r.tube.len() as f64]), &fx.get("tube_len"));
    stats.compare(name, "finish.tube", Bar::F32, &pick(&|r| f32s(r.tube.iter().flatten().copied())), &fx.get("tube"));
    // the tube from villa's own final frames: near-zero coordinates are cancellations that carry the frames' last-bit
    // differences, so the tube's own arithmetic is checked apart from them
    let isolated: Vec<f64> = fx
        .m3s("fin_frames")
        .iter()
        .zip(fx.v3s("fin_positions"))
        .flat_map(|(frame, pos)| f32s(tube_points(&fibre.points, pos, frame).into_iter().flatten()))
        .collect();
    stats.compare(name, "finish.tube_from_villa_frames", Bar::F32, &isolated, &fx.get("tube"));
    stats.compare(name, "finish.orientation", Bar::F32, &pick(&|r| f32s(r.orientation)), &fx.get("orientation"));
}

/// Stitch and write from the fixture's final frames and strip-model outputs
fn stitch_and_write(env: &Env, fx: &Fixture, fibre: &Fibre, arcs: &[f64], stats: &mut Stats) {
    let name = &fx.name;
    let s = &env.settings;
    let (heights, log_scale) = (fx.grids("heights"), fx.grids("log_scale"));
    let (frames, positions) = (fx.m3s("fin_frames"), fx.v3s("fin_positions"));
    let strip = stitched_strip(
        &heights,
        &log_scale,
        &frames,
        &positions,
        arcs,
        fibre.length(),
        s.blend,
        s.trim * s.trace_per_patch,
        s.trace_per_patch,
    );
    let bools = |b: &mut dyn Iterator<Item = bool>| b.map(|x| f64::from(u8::from(x))).collect::<Vec<_>>();
    stats.compare(name, "stitch.flipped", Bar::Exact, &bools(&mut strip.flipped.iter().copied()), &fx.get("flipped"));
    stats.compare(
        name,
        "stitch.rows",
        Bar::F64,
        &strip.rows.iter().flat_map(|r| flat3(r)).collect::<Vec<_>>(),
        &fx.get("rows"),
    );
    stats.compare(name, "stitch.keep", Bar::Exact, &bools(&mut strip.keep.iter().flatten().copied()), &fx.get("keep"));
    stats.compare(name, "stitch.expected", Bar::F64, &strip.expected.concat(), &fx.get("expected"));
    let settings = env.settings(fx.meta("synthetic").as_bool().unwrap());
    let record = settings.write_fibre(fibre, arcs, &strip, &heights, &log_scale, &frames, &positions).unwrap();
    let line = record.line(0.125, RUNNER);
    let same = line == fx.meta("record").as_str().unwrap();
    stats.compare(name, "write.index_line", Bar::Exact, &[f64::from(u8::from(same))], &[1.0]);
    if !same {
        eprintln!("{name}:\n  {line}\n  {}", fx.meta("record").as_str().unwrap());
    }
}

fn check(env: &Env, fx: &Fixture, work: &mut Workspace) -> Stats {
    let mut stats = Stats::default();
    let name = fx.name.clone();
    let fibre = load(env, fx, &mut stats);
    seeds(env, fx, work, &mut stats);
    let arcs = fibre.arcs(STEP);
    let skipped = fx.meta("skipped").as_bool().unwrap_or(false);
    stats.compare(
        &name,
        "load.skipped",
        Bar::Exact,
        &[f64::from(u8::from(arcs.is_none()))],
        &[f64::from(u8::from(skipped))],
    );
    let Some(arcs) = arcs else { return stats };
    stats.compare(&name, "load.arcs", Bar::F64, &arcs, &fx.get("arcs"));
    let arcs = fx.get("arcs");

    let plans = env.geometry().plan(&fibre, &arcs, work);
    let plans = match plans {
        Ok(plans) => plans,
        Err(e) => {
            let reason = e.downcast_ref::<SeedHeadingError>().map(ToString::to_string);
            let line = reason.map(|r| Record::failed(&fibre.row, env.settings, &r).line(0.0, RUNNER));
            let same = line.as_deref() == fx.meta("record").as_str();
            stats.compare(&name, "plan.failed_line", Bar::Exact, &[f64::from(u8::from(same))], &[1.0]);
            if !same {
                eprintln!("{name}: {e}\n  {line:?}\n  {:?}", fx.meta("record"));
            }
            return stats;
        }
    };
    if fx.has("pos") {
        compare_plan(fx, &plans, &mut stats);
        finish(env, fx, &fibre, &plans, &mut stats);
        stitch_and_write(env, fx, &fibre, &arcs, &mut stats);
    } else {
        stats.compare(&name, "plan.failed_line", Bar::Exact, &[0.0], &[1.0]);
    }
    stats
}

#[test]
#[ignore = "needs the fixtures, the AFV and the cached CT"]
fn fixture_parity() {
    let root = PathBuf::from(
        std::env::var("FIBRS_GEOM_FIXTURES")
            .unwrap_or("/home/praveen/code/research/_scratch/strips-native/geom/fx".into()),
    );
    let run: Map<String, Value> = serde_json::from_slice(&fs::read(root.join("py_out/run.json")).unwrap()).unwrap();
    let umbilicus = std::env::var("FIBRS_UMBILICUS").unwrap_or(
        "/home/praveen/code/private-workbench/ops/vesuvius/strips-scale/umbilici/PHerc0211_umbilicus.json".into(),
    );
    let settings = Settings {
        out: None,
        npz: None,
        checkpoint: run["checkpoint"].as_str().unwrap().into(),
        step: run["step"].as_i64(),
        afv_ref: run["afv"].as_str().unwrap().into(),
        decision_step: run["decision_step"].as_f64().unwrap(),
        blend: run["blend"].as_f64().unwrap(),
        trim: run["trim_patch_voxels"].as_f64().unwrap(),
        voxel_size_um: run["voxel_size_um"].as_f64().unwrap(),
        trace_per_patch: run["trace_per_patch"].as_f64().unwrap(),
    };
    let ct = Volume::open(Path::new(CT_CACHE), CT, 0).unwrap();
    let umbilicus = Umbilicus::read(Path::new(&umbilicus)).unwrap();
    let mut dirs: Vec<PathBuf> = fs::read_dir(root.join("fixtures")).unwrap().map(|e| e.unwrap().path()).collect();
    dirs.sort();
    if let Ok(only) = std::env::var("FIBRS_GEOM_ONLY") {
        dirs.retain(|d| d.file_name().unwrap().to_string_lossy() == only);
    }

    // an AFV connection is not shared across threads: one per worker
    let total = Mutex::new(Stats::default());
    let threads = std::env::var("FIBRS_GEOM_THREADS").ok().and_then(|t| t.parse().ok()).unwrap_or(8);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
    pool.install(|| {
        dirs.par_iter().for_each_init(
            || (Workspace::default(), Afv::open(Path::new(AFV)).unwrap()),
            |(work, afv), dir| {
                let fx = Fixture::open(dir);
                let env = Env { root: &root, afv, ct: &ct, umbilicus: &umbilicus, settings: &settings };
                let stats = check(&env, &fx, work);
                total.lock().unwrap().merge(stats);
            },
        );
    });

    // run.json from the same fields
    let mut fields = run.clone();
    let note = fields.shift_remove("note").unwrap();
    let mut provenance = Map::new();
    provenance.insert("note".into(), note);
    let ours = run_json(&fields, &provenance, RUNNER);
    let theirs = fs::read_to_string(root.join("py_out/run.json")).unwrap();
    let mut total = total.into_inner().unwrap();
    total.compare("run.json", "write.run_json", Bar::Exact, &[f64::from(u8::from(ours == theirs))], &[1.0]);

    let table = total.table();
    println!("{table}");
    fs::write(root.join("parity.md"), &table).unwrap();
    let bad: Vec<&String> = total.0.iter().filter(|(_, s)| s.mismatches > 0).map(|(k, _)| k).collect();
    assert!(bad.is_empty(), "fields over the bar: {bad:?}");
}

/// Selection without ids: every fibre over 2.5 mm in villa's Morton order (`select_fibres`), against the order
/// dumped from it (fixtures' selection_2.5mm.txt)
#[test]
#[ignore = "needs the fixtures and the AFV"]
fn selection_order() {
    let root = PathBuf::from(
        std::env::var("FIBRS_GEOM_FIXTURES")
            .unwrap_or("/home/praveen/code/research/_scratch/strips-native/geom/fx".into()),
    );
    let theirs: Vec<i64> = fs::read_to_string(root.join("selection_2.5mm.txt"))
        .unwrap()
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect();
    let ours: Vec<i64> =
        Afv::open(Path::new(AFV)).unwrap().select(2.5, 9.362, None).unwrap().iter().map(|r| r.id).collect();
    assert_eq!(ours.len(), theirs.len());
    let first = ours.iter().zip(&theirs).position(|(a, b)| a != b);
    assert_eq!(first, None, "orders part at {first:?}");
}
