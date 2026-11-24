//! Fibres from an AFV as the strip runners read them: the selection and order of the Python reference runner's
//! `select_fibres`, and villa's FiberStore geometry (`load_fiber`): blocks joined, base voxels to trace voxels,
//! densified to one trace voxel, kinks repaired (`shared/annotations/repair.py`)

use std::path::Path;

use color_eyre::eyre::{Result, WrapErr, bail, ensure, eyre};
use rusqlite::{Connection, OpenFlags};

use super::np::{Curve, V3, add, arange, arclength, dense_line, dot, linspace, norm, scale, sub};

/// One `fibers` row: what selection, the index record and the outputs read
#[derive(Clone, Debug)]
pub struct FibreRow {
    pub id: i64,
    /// The AFV family, `H` or `V` (anything else fails at the seed heading, as villa)
    pub family: String,
    /// Length in base voxels
    pub length: f64,
    /// min_x, max_x, min_y, max_y, min_z, max_z, base voxels
    pub bbox: [f64; 6],
}

/// A fibre in trace voxels: villa's `TracedFiber` (points, arclength, tag)
#[derive(Clone, Debug)]
pub struct Fibre {
    pub row: FibreRow,
    /// Densified, kink-repaired vertices, trace voxels
    pub points: Vec<V3>,
    /// Arclength at each vertex
    pub s: Vec<f64>,
    /// Kinks repaired at load
    pub kink_repairs: usize,
}

impl Fibre {
    /// The fibre's arclength, trace voxels
    pub fn length(&self) -> f64 {
        self.s[self.s.len() - 1]
    }

    pub fn curve(&self) -> Curve<'_> {
        Curve::new(&self.points, &self.s)
    }

    /// The decision arclengths `np.arange(12, L + 1e-9, step)`, or None for a fibre shorter than `12 + step`, which
    /// the runners skip without an index record
    pub fn arcs(&self, step: f64) -> Option<Vec<f64>> {
        (self.length() >= 12.0 + step).then(|| arange(12.0, self.length() + 1e-9, step))
    }
}

/// An AFV opened read-only
pub struct Afv {
    db: Connection,
}

const COLUMNS: &str = "id, family, length, min_x, max_x, min_y, max_y, min_z, max_z";

impl Afv {
    pub fn open(path: &Path) -> Result<Self> {
        let db = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX)
            .wrap_err_with(|| format!("open {}", path.display()))?;
        Ok(Self { db })
    }

    fn rows(&self, sql: &str, params: impl rusqlite::Params) -> Result<Vec<FibreRow>> {
        let mut stmt = self.db.prepare(sql)?;
        let rows = stmt.query_map(params, |r| {
            Ok(FibreRow {
                id: r.get(0)?,
                family: r.get(1)?,
                length: r.get(2)?,
                bbox: [r.get(3)?, r.get(4)?, r.get(5)?, r.get(6)?, r.get(7)?, r.get(8)?],
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// The fibres to run, as `select_fibres`: `ids` in their order (an error naming up to 10 missing ids), else
    /// every fibre longer than `min_length_mm` (AFV lengths are base voxels of `voxel_um`) in Morton order of its
    /// bbox centre, ties by id
    pub fn select(&self, min_length_mm: f64, voxel_um: f64, ids: Option<&[i64]>) -> Result<Vec<FibreRow>> {
        if let Some(ids) = ids {
            let list = ids.iter().map(i64::to_string).collect::<Vec<_>>().join(",");
            let found = self.rows(&format!("select {COLUMNS} from fibers where id in ({list})"), [])?;
            let by_id: std::collections::HashMap<i64, FibreRow> = found.into_iter().map(|r| (r.id, r)).collect();
            let missing: Vec<i64> = ids.iter().copied().filter(|i| !by_id.contains_key(i)).take(10).collect();
            ensure!(missing.is_empty(), "fibres not in the AFV: {missing:?}");
            return Ok(ids.iter().map(|i| by_id[i].clone()).collect());
        }
        let mut rows =
            self.rows(&format!("select {COLUMNS} from fibers where length > ?1"), [min_length_mm * 1000.0 / voxel_um])?;
        rows.sort_by_key(|r| r.id);
        let centres: Vec<V3> = rows
            .iter()
            .map(|r| [(r.bbox[0] + r.bbox[1]) / 2.0, (r.bbox[2] + r.bbox[3]) / 2.0, (r.bbox[4] + r.bbox[5]) / 2.0])
            .collect();
        let keys = morton_keys(&centres, 10);
        let mut order: Vec<usize> = (0..rows.len()).collect();
        order.sort_by_key(|&i| keys[i]);
        Ok(order.into_iter().map(|i| rows[i].clone()).collect())
    }

    /// The fibre's polyline as stored (base voxels): its blocks in segment order, a block's first point dropped
    /// when it repeats the previous block's last (`np.allclose`)
    pub fn points(&self, id: i64) -> Result<Vec<V3>> {
        let mut stmt = self.db.prepare_cached("select points from blocks where fiber_id=?1 order by first_segment")?;
        let blobs = stmt.query_map([id], |r| r.get::<_, Vec<u8>>(0))?;
        let mut out: Vec<V3> = Vec::new();
        for blob in blobs {
            let blob = blob?;
            ensure!(blob.len() % 24 == 0, "fibre {id}: a block of {} bytes", blob.len());
            let pts: Vec<V3> =
                blob.as_chunks::<8>().0.as_chunks::<3>().0.iter().map(|p| p.map(f64::from_le_bytes)).collect();
            let skip = match (out.last(), pts.first()) {
                (Some(&a), Some(&b)) => (0..3).all(|k| (a[k] - b[k]).abs() <= 1e-8 + 1e-5 * b[k].abs()),
                _ => false,
            };
            out.extend_from_slice(&pts[usize::from(skip)..]);
        }
        Ok(out)
    }

    /// The fibre of `row` in trace voxels (villa `load_fiber` with store scale 1)
    pub fn fibre(&self, row: &FibreRow, grid_scale: f64) -> Result<Fibre> {
        let raw = self.points(row.id)?;
        Fibre::from_base(row.clone(), &raw, grid_scale)
    }
}

impl Fibre {
    /// The fibre whose stored polyline is `raw` (base voxels): densified to one trace voxel and kink-repaired
    pub fn from_base(row: FibreRow, raw: &[V3], grid_scale: f64) -> Result<Self> {
        ensure!(
            raw.len() >= 2 && raw.iter().flatten().all(|v| v.is_finite()),
            "fibre {}: not a finite polyline",
            row.id
        );
        let trace: Vec<V3> = raw.iter().map(|p| [p[0] / grid_scale, p[1] / grid_scale, p[2] / grid_scale]).collect();
        let points = dense_line(&trace, 1.0).ok_or_else(|| eyre!("fibre {}: polyline has zero length", row.id))?;
        let s = arclength(&points);
        let (points, repairs) = repair_kinks(points, s);
        let s = arclength(&points);
        Ok(Self { row, points, s, kink_repairs: repairs })
    }
}

/// Morton (z-order) keys of points, each axis quantized to `bits` over the points' range (`morton_key`)
pub fn morton_keys(centres: &[V3], bits: u32) -> Vec<u64> {
    if centres.is_empty() {
        return Vec::new();
    }
    let mut lo = centres[0];
    let mut hi = centres[0];
    for c in centres {
        for a in 0..3 {
            lo[a] = lo[a].min(c[a]);
            hi[a] = hi[a].max(c[a]);
        }
    }
    let top = ((1u64 << bits) - 1) as f64;
    centres
        .iter()
        .map(|c| {
            let q: [u64; 3] = std::array::from_fn(|a| ((c[a] - lo[a]) / (hi[a] - lo[a]).max(1e-9) * top) as u64);
            let mut key = 0u64;
            for b in 0..bits {
                for (a, &qa) in q.iter().enumerate() {
                    key |= ((qa >> b) & 1) << (3 * b + a as u32);
                }
            }
            key
        })
        .collect()
}

// shared/annotations/repair.py's constants
const SHARP_TURN: f64 = 30.0;
const CLUSTER: usize = 8;
const NET_TURN: f64 = 40.0;
const NET_REACH: (usize, usize) = (2, 8);
const SMOOTH_TURN: f64 = 15.0;
const SURROUNDING_TURN: f64 = 1.25;
const BRIDGE_HALF: (usize, usize) = (2, 16);
const REPAIR_PASSES: usize = 4;

/// `np.degrees(np.arccos(np.clip(cos, -1, 1)))`
fn degrees_acos(cos: f64) -> f64 {
    cos.clamp(-1.0, 1.0).acos() * (180.0 / std::f64::consts::PI)
}

/// The curve at unit arclength steps (`unit_curve`)
fn unit_curve(points: &[V3], s: &[f64]) -> (Vec<f64>, Vec<V3>) {
    let c = Curve::new(points, s);
    let arcs = arange(0.0, c.length() + 1e-9, 1.0);
    let curve = arcs.iter().map(|&t| c.at(t)).collect();
    (arcs, curve)
}

/// The angle between successive chords at each point, degrees, 0 at the ends (`turning`)
fn turning(curve: &[V3]) -> Vec<f64> {
    let mut out = vec![0.0; curve.len()];
    if curve.len() >= 3 {
        for i in 1..curve.len() - 1 {
            let a = sub(curve[i], curve[i - 1]);
            let b = sub(curve[i + 1], curve[i]);
            let cos = dot(a, b) / (norm(a) * norm(b)).max(1e-9);
            out[i] = degrees_acos(cos);
        }
    }
    out
}

/// The larger turning of each unit-arc point and the half-voxel point after it (`sampled_turning`)
fn sampled_turning(points: &[V3], s: &[f64], arcs: &[f64], curve: &[V3]) -> Vec<f64> {
    let c = Curve::new(points, s);
    let total = c.length();
    let mut shifted_curve = Vec::with_capacity(curve.len() + 1);
    shifted_curve.push(curve[0]);
    shifted_curve.extend(arcs.iter().map(|&t| c.at((t + 0.5).clamp(0.0, total))));
    let mut shifted = turning(&shifted_curve).split_off(1);
    let last = shifted.len() - 1;
    shifted[last] = 0.0;
    turning(curve).iter().zip(&shifted).map(|(&a, &b)| a.max(b)).collect()
}

/// `clusters`: indices closer than `gap` joined into [first, last] runs
fn clusters(indices: impl Iterator<Item = usize>) -> Vec<(usize, usize)> {
    let mut groups: Vec<(usize, usize)> = Vec::new();
    for i in indices {
        match groups.last_mut() {
            Some(g) if i - g.1 <= CLUSTER => g.1 = i,
            _ => groups.push((i, i)),
        }
    }
    groups
}

fn net_turn(curve: &[V3], c0: usize, c1: usize) -> f64 {
    let (near, far) = NET_REACH;
    if c0 < far || c1 + far >= curve.len() {
        return 180.0;
    }
    let before = sub(curve[c0 - near], curve[c0 - far]);
    let after = sub(curve[c1 + far], curve[c1 + near]);
    degrees_acos(dot(before, after) / (norm(before) * norm(after)).max(1e-9))
}

/// The unit direction just outside index `i` on `side` (-1 before, +1 after), over 5 points
fn direction(curve: &[V3], i: usize, side: i64) -> V3 {
    let j = (i as i64 + side * 5).clamp(0, curve.len() as i64 - 1) as usize;
    let d = if side > 0 { sub(curve[j], curve[i]) } else { sub(curve[i], curve[j]) };
    super::np::div(d, norm(d).max(1e-9))
}

/// A cubic Hermite bridge: its ends and end tangents
type Ends = (V3, V3, V3, V3);

fn bridge(curve: &[V3], a: usize, b: usize) -> Ends {
    let chord = norm(sub(curve[b], curve[a]));
    (curve[a], curve[b], scale(direction(curve, a, -1), chord), scale(direction(curve, b, 1), chord))
}

/// The bridge at parameter `u` (`hermite`, its cubes through `pow` as NumPy's `u**3`)
fn hermite(ends: &Ends, u: f64) -> V3 {
    let (p0, p1, m0, m1) = ends;
    let u3 = u.powf(3.0);
    let u2 = u * u;
    let h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
    let h10 = u3 - 2.0 * u2 + u;
    let h01 = -2.0 * u3 + 3.0 * u2;
    let h11 = u3 - u2;
    add(add(add(scale(*p0, h00), scale(*m0, h10)), scale(*p1, h01)), scale(*m1, h11))
}

/// Kink repair (`repair_kinks`): passes until one changes nothing, at most four. Returns the vertices and the
/// number of repaired runs
pub fn repair_kinks(points: Vec<V3>, s: Vec<f64>) -> (Vec<V3>, usize) {
    let mut out = points;
    let mut s = s;
    let mut runs = 0;
    for _ in 0..REPAIR_PASSES {
        let (next, found) = repair_pass(&out, &s);
        out = next;
        if found == 0 {
            break;
        }
        s = arclength(&out);
        runs += found;
    }
    (out, runs)
}

/// One repair pass (`repair_pass`)
fn repair_pass(points: &[V3], s: &[f64]) -> (Vec<V3>, usize) {
    let (arcs, curve) = unit_curve(points, s);
    let turn = sampled_turning(points, s, &arcs, &curve);
    let n = curve.len();
    let mut fixed = curve.clone();
    let mut bridges: Vec<(f64, f64, Ends)> = Vec::new();
    let fast = (0..n).filter(|&i| turn[i] > SMOOTH_TURN);
    let kinks = clusters(fast).into_iter().filter(|&(c0, c1)| turn[c0..=c1].iter().any(|&t| t > SHARP_TURN));
    for (c0, c1) in kinks {
        if net_turn(&curve, c0, c1) > NET_TURN {
            continue;
        }
        for half in BRIDGE_HALF.0..=BRIDGE_HALF.1 {
            if c0 < half + 1 || c1 + half > n - 2 {
                break;
            }
            let (a, b) = (c0 - half, c1 + half);
            let ends = bridge(&fixed, a, b);
            let mut trial = fixed.clone();
            for (k, u) in linspace(0.0, 1.0, b - a + 1).into_iter().enumerate() {
                trial[a + k] = hermite(&ends, u);
            }
            let around: Vec<f64> = turn[a.saturating_sub(6).max(1)..a]
                .iter()
                .chain(&turn[b + 1..(b + 7).min(n - 1).max(b + 1)])
                .copied()
                .collect();
            let limit = match around.iter().copied().reduce(f64::max) {
                Some(m) => SMOOTH_TURN.max(SURROUNDING_TURN * m),
                None => SMOOTH_TURN,
            };
            let worst = turning(&trial[a - 1..b + 2]).into_iter().fold(f64::NEG_INFINITY, f64::max);
            if worst < limit {
                fixed = trial;
                bridges.push((arcs[a], arcs[b], ends));
                break;
            }
        }
    }
    let mut out = points.to_vec();
    for (start, end, ends) in &bridges {
        for (p, &si) in out.iter_mut().zip(s) {
            if si >= *start && si <= *end {
                *p = hermite(ends, (si - start) / (end - start));
            }
        }
    }
    (out, bridges.len())
}

/// A fibre family as villa's `fiber_family` reads it, or why it cannot orient a seed
pub fn family_of(tag: &str) -> Result<super::types::Family> {
    let up = tag.trim().to_uppercase();
    match up.as_str() {
        "H" | "HORIZONTAL" => Ok(super::types::Family::H),
        "V" | "VERTICAL" => Ok(super::types::Family::V),
        _ => bail!("CT seed headings require an H or V fiber family"),
    }
}
