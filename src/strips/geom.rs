//! Each decision's geometry around the two models (villa `fibers/sheet_strip/batched.py` `StripGeometry`)
//!
//! [`Geometry::plan`] builds a fibre's decisions as `predict` does without training labels (`decision_item`,
//! `decision_on_path(labels=False)`), resolves their CT seed headings (`skip_remote_seed_heading`,
//! `reuse_seed_heading`, `resolve_trace_seed`) and lays out the frame model's request (`prior_frames`,
//! `path_features`). [`DecisionPlan::finish`] turns the frame model's output into the decision's final frame
//! (`_world_normals`, `_world_frames`, `apply_prediction_axis`) and the strip model's inputs (`reference_layout`,
//! `path_geometry_samples`, `polyline_points`, `crop_orientation`). Every quantity follows villa's float64
//! arithmetic in NumPy's and PyTorch's operation order ([`super::np`]); model inputs are rounded to float32 where
//! villa rounds them

use std::collections::HashMap;
use std::path::Path;

use color_eyre::eyre::{Result, WrapErr, ensure, eyre};

use super::ct::Volume;
use super::fibre::{Fibre, family_of};
use super::np::{
    Curve, M3, V3, arange, arclength, col, cross, dense_line, div, dot, from_cols, linspace, linspace_open, mat_t_mat,
    mat_vec, neg, norm, normalize, rows_mat, rows_mat_in_place, scale, sub, sum, vec_mat,
};
use super::seed::{SeedHeadingError, Workspace, oriented_seed_heading};
use super::types::{FrameRequest, FrameResult, GEOMETRY_ROWS, HISTORY, StripRequest, TUBE_POINTS};

/// The frame model's path window (`HeadingConfig.window`)
const FRAME_WINDOW: usize = 32;
/// Decisions' history step and the trace heading's fit span (`history_step`, `HEADING_SPAN`)
const HEADING_SPAN: f64 = 12.0;
/// The previous-prediction horizon (`prediction_window`: gate_horizon x future_step)
const WINDOW: usize = 16;
/// The strip crop (model_cfg.fine): depth 144, width 128, behind 72, spacing 0.5
const CROP_DEPTH: f64 = 144.0;
const CROP_WIDTH: f64 = 128.0;
const CROP_BEHIND: f64 = 72.0;
const CROP_SPACING: f64 = 0.5;
/// The tube's margin, `3 * tube_sigma * spacing`, and its sample spacing, `spacing / 2`
const TUBE_MARGIN: f64 = 1.5;
const TUBE_SPACING: f64 = 0.25;

/// The crop's farthest point from its head, plus a sample (`StripPredictor.crop_reach`)
fn crop_reach() -> f64 {
    let far = CROP_BEHIND.max(CROP_DEPTH - CROP_BEHIND) * CROP_SPACING;
    let half = (CROP_WIDTH - 1.0) / 2.0 * CROP_SPACING;
    (far * far + 2.0 * (half * half)).sqrt() + CROP_SPACING
}

/// villa's `frame_from_heading(f, u_hint)`: columns (u, v, f), u transported from the hint when it is not along f
pub fn frame_from_heading(f: V3, hint: Option<V3>) -> M3 {
    let f = normalize(f);
    let transported = hint.map(|h| sub(h, scale(f, dot(h, f)))).filter(|u| norm(*u) >= 1e-3);
    let u = transported.unwrap_or_else(|| {
        let r = if f[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
        cross(r, f)
    });
    let u = normalize(u);
    from_cols(u, cross(f, u), f)
}

/// A scroll's umbilicus: control points (patch voxels) sorted by z (`load_umbilicus`)
#[derive(Clone, Debug)]
pub struct Umbilicus {
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
}

impl Umbilicus {
    /// Reads an `umbilicus.json`'s `control_points`
    pub fn read(path: &Path) -> Result<Self> {
        Self::parse(&std::fs::read(path).wrap_err_with(|| format!("read {}", path.display()))?, path)
    }

    /// Parses an `umbilicus.json`'s bytes, read from `path`
    pub fn parse(bytes: &[u8], path: &Path) -> Result<Self> {
        let doc: serde_json::Value =
            serde_json::from_slice(bytes).wrap_err_with(|| format!("parse {}", path.display()))?;
        let pts = doc["control_points"].as_array().ok_or_else(|| eyre!("{}: no control_points", path.display()))?;
        ensure!(!pts.is_empty(), "{}: control_points is empty", path.display());
        let mut p: Vec<V3> = pts
            .iter()
            .map(|c| {
                let g = |k: &str| c[k].as_f64().ok_or_else(|| eyre!("{}: control point without {k}", path.display()));
                Ok([g("x")?, g("y")?, g("z")?])
            })
            .collect::<Result<_>>()?;
        // villa's argsort is not stable; equal heights would make the order matter, and no umbilicus has them
        p.sort_by(|a, b| a[2].total_cmp(&b[2]));
        Ok(Self::from_points(&p))
    }

    /// From control points already sorted by z
    pub fn from_points(p: &[V3]) -> Self {
        Self {
            x: p.iter().map(|q| q[0]).collect(),
            y: p.iter().map(|q| q[1]).collect(),
            z: p.iter().map(|q| q[2]).collect(),
        }
    }
}

/// `np.interp(x, xp, fp)` for one abscissa
fn interp1(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let last = xp.len() - 1;
    if x >= xp[last] {
        return fp[last];
    }
    if x < xp[0] {
        return fp[0];
    }
    let j = xp.partition_point(|&v| v <= x) - 1;
    if xp[j] == x {
        return fp[j];
    }
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    slope * (x - xp[j]) + fp[j]
}

/// The CPU geometry of a run: the CT for seed headings and the scroll's umbilicus (`StripGeometry`)
pub struct Geometry<'a> {
    pub ct: &'a Volume,
    /// CT voxels per trace voxel (`spec.input_scale`)
    pub input_scale: f64,
    /// Trace voxels per patch voxel
    pub trace_per_patch: f64,
    pub umbilicus: &'a Umbilicus,
}

/// One decision between [`Geometry::plan`] and [`DecisionPlan::finish`]: the frame model's request and the decision
/// item's state
#[derive(Clone, Debug)]
pub struct DecisionPlan {
    /// The frame model's inputs
    pub request: FrameRequest,
    /// The item's frame before the learned frame (the trace heading's, or the CT seed heading's)
    pub(super) frame: M3,
    pub(super) hist_local: Vec<V3>,
    pub(super) hmask: [f32; HISTORY],
    pub(super) seed_pos: V3,
    pub(super) seed_tangent: V3,
    pub(super) seed_age: f64,
    /// The fibre ahead of the head (`prev_prediction`)
    pub(super) prev_prediction: [V3; WINDOW],
    /// The committed path up to the head without repeated points (`observed_path`)
    pub(super) observed: Vec<V3>,
}

/// One seed-heading cache key: the seed's position and tangent, bit for bit (a fibre has one family)
type SeedKey = ([u64; 3], [u64; 3]);

/// The trusted 12-voxel heading fit of a path, or None (`linear12_heading(path, 0)`)
fn linear12_heading(path: &[V3]) -> Option<V3> {
    if path.len() < 2 {
        return None;
    }
    let mut distance = 0.0;
    let mut first = path.len() - 1;
    while first > 0 && distance < HEADING_SPAN {
        distance += norm(sub(path[first], path[first - 1]));
        first -= 1;
    }
    if distance < HEADING_SPAN - 1e-8 {
        return None;
    }
    let points = &path[first..];
    let arc = arclength(points);
    let curve = Curve::new(points, &arc);
    let end = points[points.len() - 1];
    let total = arc[arc.len() - 1];
    // (times - times.mean()) @ samples, times 0..12
    let mut direction = [0.0; 3];
    for k in 0..13 {
        let sample = sub(curve.at((total - HEADING_SPAN + k as f64).max(0.0)), end);
        let w = k as f64 - 6.0;
        for a in 0..3 {
            direction[a] += w * sample[a];
        }
    }
    let size = norm(direction);
    (size > 1e-8).then(|| div(direction, size))
}

/// `trace_history(path, 128)`: the path at unit arclength steps behind its head, newest first, and validity
fn trace_history(path: &[V3]) -> (Vec<V3>, [f32; HISTORY]) {
    let past = &path[path.len().saturating_sub(2 * HISTORY + 2)..];
    let arc = arclength(past);
    let curve = Curve::new(past, &arc);
    let total = arc[arc.len() - 1];
    let mut mask = [0.0f32; HISTORY];
    let hist = (0..HISTORY)
        .map(|k| {
            let target = total - (k + 1) as f64;
            mask[k] = if target >= 0.0 { 1.0 } else { 0.0 };
            curve.at(target.max(0.0))
        })
        .collect();
    (hist, mask)
}

/// `observed_path`: the path without consecutive repeats (gaps <= 1e-9)
fn observed_path(path: &[V3]) -> Vec<V3> {
    let mut out = Vec::with_capacity(path.len());
    out.push(path[0]);
    for w in path.windows(2) {
        if norm(sub(w[1], w[0])) > 1e-9 {
            out.push(w[1]);
        }
    }
    out
}

/// `reframe_item`: local quantities follow a new frame, world geometry unchanged
fn reframe(frame: &mut M3, hist_local: &mut [V3], new: M3) {
    let rotation = mat_t_mat(frame, &new);
    rows_mat_in_place(hist_local, &rotation);
    *frame = new;
}

/// `path_features(path, pos, frame, 32)`: the last 32 voxels behind the head, newest first, in `frame`
fn path_features(path: &[V3], pos: V3, frame: &M3) -> [f32; 4 * FRAME_WINDOW] {
    let mut out = [0.0f32; 4 * FRAME_WINDOW];
    if path.len() <= 1 {
        return out;
    }
    let arc = arclength(path);
    let curve = Curve::new(path, &arc);
    let total = arc[arc.len() - 1];
    let valid: Vec<usize> = (0..FRAME_WINDOW).filter(|&k| total - (k + 1) as f64 >= 0.0).collect();
    let rows: Vec<V3> = valid.iter().map(|&k| sub(curve.at(total - (k + 1) as f64), pos)).collect();
    for (&k, local) in valid.iter().zip(rows_mat(&rows, frame)) {
        out[4 * k..4 * k + 3].copy_from_slice(&local.map(|v| v as f32));
        out[4 * k + 3] = 1.0;
    }
    out
}

impl Geometry<'_> {
    /// The decisions at `arcs` and the frame model's request for each (`StripGeometry.plan`). A
    /// [`SeedHeadingError`] in the report fails the fibre (no decisions run); any other error stops the run
    pub fn plan(&self, fibre: &Fibre, arcs: &[f64], work: &mut Workspace) -> Result<Vec<DecisionPlan>> {
        let curve = fibre.curve();
        let tag = fibre.row.family.as_str();
        let reach = crop_reach();
        let mut seeds: HashMap<SeedKey, V3> = HashMap::new();
        let mut plans = Vec::with_capacity(arcs.len());
        for &t in arcs {
            // decision_item: the fibre traced exactly up to its head at unit steps
            let count = (t.floor() as usize).min(HISTORY);
            let path: Vec<V3> = (0..=count).map(|i| curve.at(t - (count - i) as f64)).collect();
            let a0 = t - count as f64;
            let pos = path[count];
            let seed_direction =
                normalize(sub(curve.at((a0 + 3.0).min(curve.length())), curve.at((a0 - 3.0).max(0.0))));
            let fit = linear12_heading(&path);
            let mut frame = frame_from_heading(fit.unwrap_or(seed_direction), None);
            let (hist, hmask) = trace_history(&path);
            let hist_rows: Vec<V3> = hist.iter().map(|&h| sub(h, pos)).collect();
            let mut hist_local = rows_mat(&hist_rows, &frame);
            let seed_pos = path[0];
            let seed_age = arclength(&path)[count];
            let total = curve.length();
            let prev_prediction: [V3; WINDOW] = std::array::from_fn(|k| curve.at((t + (k + 1) as f64).min(total)));
            let mut seed_tangent = seed_direction;
            let mut seed = |pos: V3, direction: V3| -> Result<V3> {
                oriented_seed_heading(self.ct, self.input_scale, pos, tag, direction, work)
            };
            if fit.is_none() {
                // resolve_trace_seed: the seed's CT heading is the crop heading too; not cached
                let heading = seed(seed_pos, seed_tangent)?;
                seed_tangent = heading;
                reframe(&mut frame, &mut hist_local, frame_from_heading(heading, None));
            } else if norm(sub(seed_pos, pos)) <= reach {
                // reuse_seed_heading; a seed beyond the crop's reach is never read (skip_remote_seed_heading)
                let key = (seed_pos.map(f64::to_bits), seed_tangent.map(f64::to_bits));
                seed_tangent = match seeds.get(&key) {
                    Some(&h) => h,
                    None => {
                        let h = seed(seed_pos, seed_tangent)?;
                        seeds.insert(key, h);
                        h
                    }
                };
            }
            let family =
                family_of(tag).map_err(|_| SeedHeadingError("CT seed headings require an H or V fiber family"))?;
            let observed = observed_path(&path);
            let prior = frame_from_heading(col(&frame, 2), None);
            let request = FrameRequest { pos, frame: prior, paths: path_features(&observed, pos, &prior), family };
            plans.push(DecisionPlan {
                request,
                frame,
                hist_local,
                hmask,
                seed_pos,
                seed_tangent,
                seed_age,
                prev_prediction,
                observed,
            });
        }
        Ok(plans)
    }

    /// `crop_orientation` of one decision: its axes against the scroll
    fn orientation(&self, pos: V3, frame: &M3) -> [f32; 5] {
        let p = div(pos, self.trace_per_patch);
        let u = self.umbilicus;
        let centre = [interp1(p[2], &u.z, &u.x), interp1(p[2], &u.z, &u.y)];
        let r = [p[0] - centre[0], p[1] - centre[1]];
        let size = (r[0] * r[0] + r[1] * r[1]).sqrt().max(1e-9);
        let r = [r[0] / size, r[1] / size];
        let outward = |j: usize| r[0] * frame[0][j] + r[1] * frame[1][j];
        [outward(0), outward(1), frame[2][0], frame[2][1], frame[2][2]].map(|v| v as f32)
    }
}

/// PyTorch's `norm(dim=-1)` of a float64 row: fused multiply-adds
fn torch_norm(a: V3) -> f64 {
    a[2].mul_add(a[2], a[1].mul_add(a[1], a[0] * a[0])).sqrt()
}

/// PyTorch's `torch.linalg.cross` of float64 rows
fn torch_cross(p: V3, q: V3) -> V3 {
    [p[1].mul_add(q[2], -(p[2] * q[1])), p[2].mul_add(q[0], -(p[0] * q[2])), p[0].mul_add(q[1], -(p[1] * q[0]))]
}

/// The index of the first smallest or largest value (PyTorch's and NumPy's argmin/argmax ties)
fn arg_first(a: V3, better: impl Fn(f64, f64) -> bool) -> usize {
    let mut best = 0;
    for k in 1..3 {
        if better(a[k], a[best]) {
            best = k;
        }
    }
    best
}

/// `orthonormal_frame(heading, normal, zeros)` (heading_model/frames.py) on float64 rows: right-handed, the u axis
/// signed so its largest component is positive
fn orthonormal_frame(heading: V3, normal: V3) -> M3 {
    let length = torch_norm(heading);
    let h = if length > 1e-6 { div(heading, length.max(1e-6)) } else { [0.0, 0.0, 1.0] };
    let mut basis = [0.0; 3];
    basis[arg_first(h.map(f64::abs), |a, b| a < b)] = 1.0;
    let b = sub(basis, scale(h, dot(basis, h)));
    let fallback = div(b, torch_norm(b).max(1e-12));
    let u = sub(normal, scale(h, dot(normal, h)));
    let size = torch_norm(u);
    let u = if size > 1e-6 { div(u, size.max(1e-6)) } else { fallback };
    // reference_u is zero: no agreement, so the canonical sign
    let canonical = u[arg_first(u.map(f64::abs), |a, b| a > b)];
    let u = if canonical < 0.0 { neg(u) } else { u };
    from_cols(u, torch_cross(h, u), h)
}

/// `_cone_coefficients(max_angle, step)`: per direction of every ring (centre excluded) cos a, sin a, cos b, sin b
struct Cone {
    ca: Vec<f64>,
    sa: Vec<f64>,
    cb: Vec<f64>,
    sb: Vec<f64>,
}

impl Cone {
    fn new(max_angle: f64, step: f64) -> Self {
        let mut c = Self { ca: Vec::new(), sa: Vec::new(), cb: Vec::new(), sb: Vec::new() };
        for deg in arange(step, max_angle + 1e-9, step) {
            let a = deg * (std::f64::consts::PI / 180.0);
            let n = ((360.0 * a.sin() / step).ceil() as usize).max(6);
            for b in linspace_open(0.0, 2.0 * std::f64::consts::PI, n) {
                c.ca.push(a.cos());
                c.sa.push(a.sin());
                c.cb.push(b.cos());
                c.sb.push(b.sin());
            }
        }
        c
    }

    /// `_cone(centre, ...)`: the unit directions, centre first
    fn directions(&self, centre: V3) -> Vec<V3> {
        let mut eye = [0.0; 3];
        eye[arg_first(centre.map(f64::abs), |a, b| a < b)] = 1.0;
        let e1 = cross(centre, eye);
        let e1 = div(e1, norm(e1));
        let e2 = cross(centre, e1);
        let mut dirs = Vec::with_capacity(self.ca.len() + 1);
        dirs.push(centre);
        for k in 0..self.ca.len() {
            let ring: V3 = std::array::from_fn(|a| self.cb[k] * e1[a] + self.sb[k] * e2[a]);
            dirs.push(std::array::from_fn(|a: usize| self.ca[k] * centre[a] + self.sa[k] * ring[a]));
        }
        dirs
    }
}

fn cones() -> &'static (Cone, Cone) {
    static CONES: std::sync::OnceLock<(Cone, Cone)> = std::sync::OnceLock::new();
    CONES.get_or_init(|| (Cone::new(75.0, 2.5), Cone::new(3.0, 0.25)))
}

/// `best(dirs)`: the direction whose axis through the head keeps `rel` (curve points less the head) closest, by
/// their largest lateral distance; the first on ties
fn best_direction(rel: &[V3], sq: &[f64], dirs: &[V3]) -> V3 {
    let mut best = (f64::INFINITY, 0);
    for (k, &d) in dirs.iter().enumerate() {
        let mut lateral = f64::NEG_INFINITY;
        for (r, &s) in rel.iter().zip(sq) {
            // rel @ dirs.T through dgemm
            let along = r[2].mul_add(d[2], r[1].mul_add(d[1], r[0] * d[0]));
            lateral = lateral.max((s - along * along).max(0.0).sqrt());
        }
        // np.argmin: the first minimum (NaN never arises here)
        if lateral < best.0 {
            best = (lateral, k);
        }
    }
    dirs[best.1]
}

/// `predicted_axis(future, head, 16)`: the straight axis through the head that keeps the fibre ahead closest over
/// the window, or None when the fibre ahead has no length
fn predicted_axis(future: &[V3; WINDOW], head: V3) -> Option<V3> {
    let mut pts = Vec::with_capacity(WINDOW + 1);
    pts.push(head);
    pts.extend_from_slice(future);
    let arc = arclength(&pts);
    let total = arc[arc.len() - 1];
    if total < 1e-6 {
        return None;
    }
    let window = WINDOW as f64;
    let along = Curve::new(&pts, &arc);
    let curve: Vec<V3> = if total >= window {
        arange(1.0, window + 1e-9, 1.0).into_iter().map(|t| along.at(t)).collect()
    } else {
        let last = pts[pts.len() - 1];
        let e = sub(last, pts[pts.len().saturating_sub(4)]);
        let end = div(e, norm(e).max(1e-9));
        let mut c: Vec<V3> = arange(1.0, total + 1e-9, 1.0).into_iter().map(|t| along.at(t)).collect();
        c.push(last);
        c.extend(arange(1.0, window - total + 1e-9, 1.0).into_iter().map(|k| super::np::add(last, scale(end, k))));
        c
    };
    let rel: Vec<V3> = curve.iter().map(|&c| sub(c, head)).collect();
    let sq: Vec<f64> = rel.iter().map(|&r| dot(r, r)).collect();
    let last = rel[rel.len() - 1];
    let chord = div(last, norm(last).max(1e-9));
    let (coarse, fine) = cones();
    let axis = best_direction(&rel, &sq, &coarse.directions(chord));
    Some(best_direction(&rel, &sq, &fine.directions(axis)))
}

/// The path-geometry arclengths behind the head (`OFFSETS`), then the path's first point
const OFFSETS: [f64; GEOMETRY_ROWS - 1] = [
    1., 2., 3., 4., 6., 8., 12., 16., 20., 24., 28., 32., 40., 48., 56., 64., 80., 96., 112., 128., 160., 192., 224.,
    256., 320., 384., 448., 512.,
];

/// `path_geometry_samples`: 29 x (u, v, f, tangent u, v, f, arclength back, first-point role) and validity
fn path_geometry(path: &[V3], pos: V3, frame: &M3) -> ([[f32; 8]; GEOMETRY_ROWS], [bool; GEOMETRY_ROWS]) {
    let arc = arclength(path);
    let curve = Curve::new(path, &arc);
    let total = arc[arc.len() - 1];
    let mut back = [0.0; GEOMETRY_ROWS];
    back[..GEOMETRY_ROWS - 1].copy_from_slice(&OFFSETS);
    back[GEOMETRY_ROWS - 1] = total;
    let mut valid = [false; GEOMETRY_ROWS];
    let mut points = [[0.0; 3]; GEOMETRY_ROWS];
    let mut tangents = [[0.0; 3]; GEOMETRY_ROWS];
    for r in 0..GEOMETRY_ROWS {
        let at = total - back[r];
        valid[r] = at >= -1e-8;
        let at = at.clamp(0.0, total);
        points[r] = sub(curve.at(at), pos);
        let tangent = sub(curve.at((at + 0.5).clamp(0.0, total)), curve.at((at - 0.5).clamp(0.0, total)));
        let size = norm(tangent);
        tangents[r] = div(tangent, size.max(1e-8));
        if r == GEOMETRY_ROWS - 1 {
            valid[r] = total >= 1.0;
        }
        valid[r] &= size > 1e-8;
    }
    let local = rows_mat(&points, frame);
    let dirs = rows_mat(&tangents, frame);
    let mut features = [[0.0f32; 8]; GEOMETRY_ROWS];
    for r in 0..GEOMETRY_ROWS {
        if !valid[r] {
            continue;
        }
        let f = &mut features[r];
        f[..3].copy_from_slice(&local[r].map(|v| v as f32));
        f[3..6].copy_from_slice(&dirs[r].map(|v| v as f32));
        f[6] = back[r].min(1024.0) as f32;
        f[7] = if r == GEOMETRY_ROWS - 1 { 1.0 } else { 0.0 };
    }
    (features, valid)
}

/// `polyline_points(fibre, pos, frame, crop, 0.25, 1.5, 2048)`: crop-local points of the fibre densified to a
/// quarter voxel, inside the crop grown by the tube's margin, evenly thinned to at most 2048
pub(super) fn tube_points(fibre: &[V3], pos: V3, frame: &M3) -> Vec<[f32; 3]> {
    let half = (CROP_WIDTH - 1.0) * CROP_SPACING / 2.0 + TUBE_MARGIN;
    let far = CROP_BEHIND.max(CROP_DEPTH - CROP_BEHIND) * CROP_SPACING + TUBE_MARGIN;
    let reach = (half * half * 2.0 + far * far).sqrt();
    let near = |p: &V3| norm(sub(*p, pos)) <= reach + 2.0;
    let Some(first) = fibre.iter().position(near) else { return Vec::new() };
    let last = fibre.iter().rposition(near).unwrap();
    let span = &fibre[first.saturating_sub(1)..(last + 2).min(fibre.len())];
    let dense;
    let world: &[V3] = if span.len() > 1 && {
        let lengths: Vec<f64> = span.windows(2).map(|w| norm(sub(w[1], w[0]))).collect();
        sum(&lengths) > 1e-9
    } {
        dense = dense_line(span, TUBE_SPACING).expect("a polyline with length");
        &dense
    } else {
        span
    };
    let rows: Vec<V3> = world.iter().map(|&w| sub(w, pos)).collect();
    let lo = -CROP_BEHIND * CROP_SPACING - TUBE_MARGIN;
    let hi = (CROP_DEPTH - 1.0 - CROP_BEHIND) * CROP_SPACING + TUBE_MARGIN;
    let inside: Vec<V3> = rows_mat(&rows, frame)
        .into_iter()
        .filter(|l| l[0].abs() <= half && l[1].abs() <= half && l[2] >= lo && l[2] <= hi)
        .collect();
    let f32s = |l: V3| l.map(|v| v as f32);
    if inside.len() <= TUBE_POINTS {
        return inside.into_iter().map(f32s).collect();
    }
    let at = linspace(0.0, (inside.len() - 1) as f64, TUBE_POINTS);
    at.into_iter().map(|x| f32s(inside[x.round_ties_even() as usize])).collect()
}

impl DecisionPlan {
    /// The strip model's inputs given the frame model's output (`StripGeometry.finish`, one decision): the
    /// learned frame (S6), the previous-prediction axis (S7), references (S8), path geometry (S9), tube (S10) and
    /// crop orientation (S11)
    pub fn finish(&self, geometry: &Geometry, fibre: &Fibre, out: &FrameResult) -> StripRequest {
        let prior = &self.request.frame;
        let h = mat_vec(prior, out.heading.map(f64::from));
        let h = if dot(h, col(&self.frame, 2)) >= 0.0 { h } else { neg(h) };
        let n = mat_vec(prior, out.normal.map(f64::from));
        let learned = orthonormal_frame(h, n);
        let mut frame = self.frame;
        let mut hist_local = self.hist_local.clone();
        reframe(&mut frame, &mut hist_local, learned);
        if let Some(axis) = predicted_axis(&self.prev_prediction, self.request.pos) {
            let next = frame_from_heading(axis, Some(col(&frame, 0)));
            reframe(&mut frame, &mut hist_local, next);
        }
        let pos = self.request.pos;
        // reference_layout's seed slot; the strip model reads the history unmasked (batched.py passes hist_local and
        // hmask, not the reference points)
        let hist: [[f32; 3]; HISTORY] = std::array::from_fn(|k| hist_local[k].map(|v| v as f32));
        let seed = vec_mat(sub(self.seed_pos, pos), &frame).map(|v| v as f32);
        let seed_visible = visible(seed);
        let (seed, seed_tangent, seed_age) = if seed_visible {
            (seed, vec_mat(self.seed_tangent, &frame).map(|v| v as f32), self.seed_age as f32)
        } else {
            ([0.0; 3], [0.0; 3], 0.0)
        };
        let (path_geometry, path_geometry_valid) = path_geometry(&self.observed, pos, &frame);
        let tube = tube_points(&fibre.points, pos, &frame);
        let orientation = geometry.orientation(pos, &frame);
        StripRequest {
            pos,
            frame,
            hist,
            hmask: self.hmask,
            seed,
            seed_mask: if seed_visible { 1.0 } else { 0.0 },
            seed_age,
            seed_tangent,
            path_geometry,
            path_geometry_valid,
            tube,
            orientation,
        }
    }
}

/// `visible_points` of the strip crop on a float32 point: finite and inside the inclusive box
fn visible(p: [f32; 3]) -> bool {
    let half = (CROP_WIDTH - 1.0) * CROP_SPACING / 2.0;
    let lo = [-half, -half, -CROP_BEHIND * CROP_SPACING];
    let hi = [half, half, (CROP_DEPTH - 1.0 - CROP_BEHIND) * CROP_SPACING];
    (0..3).all(|a| p[a].is_finite() && f64::from(p[a]) >= lo[a] && f64::from(p[a]) <= hi[a])
}
