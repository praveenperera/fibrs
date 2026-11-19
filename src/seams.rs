//! Cut each tile's fibres to the tile's core and join the cut ends across the faces between tiles
//!
//! The rule is halojoin's (FIBSETS v2-run `halojoin.py`, which joins Z bands), for faces across any axis. A fibre
//! is cut to its tile's core box: runs of its points inside the box are kept, and where a run leaves through an
//! interior face (a core face shared with another tile), the point where its segment crosses the face is added, so
//! the run ends on it. Where the point outside is outside on more than one axis, the face the segment crosses
//! first is the one it leaves through. At each interior face, a cut end from the core below the face and one from
//! the core above it, of the same family and at most [`RADIUS`] voxels apart, are joined: shortest first (ties by
//! run and end), each end once, and a join that would close a loop is skipped. Runs of fewer than two points are
//! dropped. Nothing depends on hash order, so the same runs always give the same chains

use std::cmp::Ordering;
use std::collections::BTreeMap;

/// Native voxels within which two cut ends are joined: halojoin's radius for the HF PHerc sets and v2-run
pub const RADIUS: f64 = 1.5;

/// A native x, y, z coordinate, the export's point
pub type Point = [f64; 3];

/// The plane across axis `axis` of the points (0 = x, 1 = y, 2 = z) at native coordinate `at`
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Face {
    pub axis: usize,
    pub at: f64,
}

impl Face {
    /// Faces by axis, then by coordinate
    fn order(&self, other: &Self) -> Ordering {
        self.axis.cmp(&other.axis).then(self.at.total_cmp(&other.at))
    }
}

/// A tile's core in native coordinates: the points with `lo <= p < hi` on every axis. An infinite bound is open: no
/// other tile lies beyond it
#[derive(Clone, Copy, Debug)]
pub struct CoreBox {
    pub lo: Point,
    pub hi: Point,
}

impl CoreBox {
    /// Whether `p` lies in the box; a NaN coordinate counts as inside, so the export refuses it as it would untiled
    fn holds(&self, p: &Point) -> bool {
        !(0..3).any(|a| p[a] < self.lo[a] || p[a] >= self.hi[a])
    }

    /// The face the segment from `inside` to `outside` leaves the box through: the first it crosses, the lower axis
    /// on a tie
    fn exit(&self, inside: &Point, outside: &Point) -> Face {
        let mut first: Option<(f64, Face)> = None;
        for axis in 0..3 {
            let at = match outside[axis] {
                v if v < self.lo[axis] => self.lo[axis],
                v if v >= self.hi[axis] => self.hi[axis],
                _ => continue,
            };
            let t = (at - inside[axis]) / (outside[axis] - inside[axis]);
            if first.is_none_or(|(best, _)| t < best) {
                first = Some((t, Face { axis, at }));
            }
        }

        first.expect("a point outside the box is outside on some axis").1
    }
}

/// A fibre's run inside one tile's core
#[derive(Clone, Debug)]
pub struct Run {
    pub points: Vec<Point>,
    /// 1 = V, 2 = H
    pub family: usize,
    /// The interior face the first point was cut onto, or `None` where the run starts with the fibre or at a face
    /// that is not interior
    pub head: Option<Face>,
    /// As `head`, for the last point
    pub tail: Option<Face>,
    /// The core's lower bounds, which tell on which side of a face the run lies
    pub core_lo: Point,
}

impl Run {
    /// Whether either end was cut onto an interior face, so it may be joined
    pub fn is_cut(&self) -> bool {
        self.head.is_some() || self.tail.is_some()
    }
}

/// What cutting one fibre to a core leaves
#[derive(Debug)]
pub enum Cut {
    /// Every point lies in the core: the fibre as it was, its points not copied
    Whole(Vec<Point>),
    /// The runs inside the core, in fibre order; none when the fibre only passes outside
    Runs(Vec<Run>),
}

/// `halojoin.crossing`: the point where the segment from `p` to `q` crosses `face`
fn crossing(p: &Point, q: &Point, face: Face) -> Point {
    let t = (face.at - p[face.axis]) / (q[face.axis] - p[face.axis]);
    [0, 1, 2].map(|a| p[a] + t * (q[a] - p[a]))
}

/// The runs of a fibre's `points` inside `core` (`halojoin.cut_fibre`); `interior` lists the core's faces shared
/// with another tile
pub fn cut(points: Vec<Point>, family: usize, core: &CoreBox, interior: &[Face]) -> Cut {
    let inside: Vec<bool> = points.iter().map(|p| core.holds(p)).collect();
    if inside.iter().all(|&v| v) {
        return Cut::Whole(points);
    }

    // the face a run leaves through between an inside and an outside point, when it is interior
    let leaves = |inside: &Point, outside: &Point| Some(core.exit(inside, outside)).filter(|f| interior.contains(f));
    let n = points.len();
    let mut runs = Vec::new();
    let mut i = 0;
    while i < n {
        if !inside[i] {
            i += 1;
            continue;
        }
        let mut j = i;
        while j + 1 < n && inside[j + 1] {
            j += 1;
        }

        let head = if i > 0 { leaves(&points[i], &points[i - 1]) } else { None };
        let tail = if j + 1 < n { leaves(&points[j], &points[j + 1]) } else { None };
        let mut run = Vec::with_capacity(j + 3 - i);
        run.extend(head.map(|f| crossing(&points[i - 1], &points[i], f)));
        run.extend_from_slice(&points[i..=j]);
        run.extend(tail.map(|f| crossing(&points[j], &points[j + 1], f)));
        if run.len() >= 2 {
            runs.push(Run { points: run, family, head, tail, core_lo: core.lo });
        }
        i = j + 1;
    }

    Cut::Runs(runs)
}

/// Counts of one join, as join.py reports them
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct JoinStats {
    pub cut_runs: usize,
    pub cut_ends: usize,
    pub joins: usize,
    pub chains: usize,
}

/// A candidate join: distance, then the run and end (0 = first point, 1 = last) below the face and above it
type Candidate = (f64, usize, usize, usize, usize);

/// The ends of `runs` cut onto `face` that lie within `radius` of an end on the face's other side, of the same
/// family, appended to `out`
///
/// Ends are paired through a grid on the face's plane with cells of `radius`, so only the 3 × 3 cells around an end
/// are searched; a pair is kept at a distance of at most `radius`, as cKDTree's `query_ball_tree` keeps it
fn face_candidates(runs: &[Run], face: Face, radius: f64, out: &mut Vec<Candidate>) {
    let (mut below, mut above) = (Vec::new(), Vec::new());
    for (i, run) in runs.iter().enumerate() {
        let ends = [(0, run.head, run.points[0]), (1, run.tail, run.points[run.points.len() - 1])];
        for (end, at, q) in ends {
            if at != Some(face) {
                continue;
            }
            let side = if run.core_lo[face.axis] < face.at { &mut below } else { &mut above };
            side.push((i, end, q));
        }
    }
    if below.is_empty() || above.is_empty() {
        return;
    }

    let plane = [(face.axis + 1) % 3, (face.axis + 2) % 3];
    let edge = if radius > 0.0 { radius } else { 1.0 };
    let cell = |q: &Point| plane.map(|a| (q[a] / edge).floor() as i64);
    let mut grid: BTreeMap<[i64; 2], Vec<usize>> = BTreeMap::new();
    for (b, (_, _, q)) in above.iter().enumerate() {
        grid.entry(cell(q)).or_default().push(b);
    }
    for &(r1, e1, p) in &below {
        let [u, v] = cell(&p);
        let near = (-1..=1).flat_map(|du| (-1..=1).map(move |dv| [u + du, v + dv]));
        for &b in near.filter_map(|c| grid.get(&c)).flatten() {
            let (r2, e2, q) = above[b];
            if runs[r1].family != runs[r2].family {
                continue;
            }
            let squared: f64 = (0..3).map(|a| (p[a] - q[a]) * (p[a] - q[a])).sum();
            if squared <= radius * radius {
                out.push((squared.sqrt(), r1, e1, r2, e2));
            }
        }
    }
}

/// The root of `x`'s set, halving the path on the way
fn root(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// The chains cut runs make once joined across their faces (`halojoin.join`), as (family, points), with counts
///
/// Chains come in the order of their first run in `runs`; each starts at the free end that run's chain reaches
/// first from the run's first point
pub fn join(runs: Vec<Run>, radius: f64) -> (Vec<(usize, Vec<Point>)>, JoinStats) {
    let mut faces: Vec<Face> = runs.iter().flat_map(|r| [r.head, r.tail]).flatten().collect();
    faces.sort_by(Face::order);
    faces.dedup();
    let mut candidates = Vec::new();
    for &face in &faces {
        face_candidates(&runs, face, radius, &mut candidates);
    }
    candidates.sort_by(|a, b| a.0.total_cmp(&b.0).then((a.1, a.2, a.3, a.4).cmp(&(b.1, b.2, b.3, b.4))));

    let n = runs.len();
    let mut parent: Vec<usize> = (0..n).collect();
    // the end each end is joined to, by 2 × run + end
    let mut link: Vec<Option<(usize, usize)>> = vec![None; 2 * n];
    let mut joins = 0;
    for (_, r1, e1, r2, e2) in candidates {
        if link[2 * r1 + e1].is_some() || link[2 * r2 + e2].is_some() {
            continue;
        }
        let (a, b) = (root(&mut parent, r1), root(&mut parent, r2));
        if a == b {
            continue;
        }
        parent[a] = b;
        link[2 * r1 + e1] = Some((r2, e2));
        link[2 * r2 + e2] = Some((r1, e1));
        joins += 1;
    }

    let mut done = vec![false; n];
    let mut chains = Vec::new();
    for r0 in 0..n {
        if done[r0] {
            continue;
        }
        // walk to one end of r0's chain, then along it
        let (mut run, mut entry) = (r0, 0);
        while let Some((next, end)) = link[2 * run + entry] {
            (run, entry) = (next, 1 - end);
        }
        let family = runs[run].family;
        let mut points = Vec::new();
        loop {
            let part = &runs[run].points;
            if entry == 1 {
                points.extend(part.iter().rev());
            } else {
                points.extend_from_slice(part);
            }
            done[run] = true;
            let Some(next) = link[2 * run + 1 - entry] else { break };
            (run, entry) = next;
        }
        chains.push((family, points));
    }

    let cut_ends = runs.iter().map(|r| r.head.is_some() as usize + r.tail.is_some() as usize).sum();
    let stats = JoinStats { cut_runs: n, cut_ends, joins, chains: chains.len() };
    (chains, stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    const OPEN: f64 = f64::INFINITY;

    /// Two cores side by side across x = 10 (open on y and z, the outer x faces at 0 and 20 not interior)
    fn cores() -> ([CoreBox; 2], Vec<Face>) {
        let left = CoreBox { lo: [0.0, -OPEN, -OPEN], hi: [10.0, OPEN, OPEN] };
        let right = CoreBox { lo: [10.0, -OPEN, -OPEN], hi: [20.0, OPEN, OPEN] };
        ([left, right], vec![Face { axis: 0, at: 10.0 }])
    }

    fn runs(cut: Cut) -> Vec<Run> {
        match cut {
            Cut::Runs(runs) => runs,
            Cut::Whole(_) => panic!("expected the fibre to be cut"),
        }
    }

    fn line(xs: &[f64], y: f64) -> Vec<Point> {
        xs.iter().map(|&x| [x, y, 5.0]).collect()
    }

    #[test]
    fn a_fibre_inside_the_core_is_whole() {
        let ([left, _], interior) = cores();
        let points = line(&[1.0, 2.0, 9.5], 0.0);
        assert!(matches!(cut(points.clone(), 1, &left, &interior), Cut::Whole(p) if p == points));
    }

    /// A fibre across the seam is cut on each side with the crossing point added, and the two runs join back into
    /// the fibre with the crossing point twice
    #[test]
    fn a_fibre_across_a_seam_is_cut_and_joined() {
        let ([left, right], interior) = cores();
        let points = line(&[6.0, 8.0, 12.0, 14.0], 3.0);
        let a = runs(cut(points.clone(), 2, &left, &interior));
        let b = runs(cut(points.clone(), 2, &right, &interior));
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].points, line(&[6.0, 8.0, 10.0], 3.0));
        assert_eq!((a[0].head, a[0].tail), (None, Some(interior[0])));
        assert_eq!(b[0].points, line(&[10.0, 12.0, 14.0], 3.0));
        assert_eq!((b[0].head, b[0].tail), (Some(interior[0]), None));

        let (chains, stats) = join([a, b].concat(), RADIUS);
        assert_eq!(chains, vec![(2, line(&[6.0, 8.0, 10.0, 10.0, 12.0, 14.0], 3.0))]);
        assert_eq!(stats, JoinStats { cut_runs: 2, cut_ends: 2, joins: 1, chains: 1 });
    }

    /// A segment from the corner of a core to a point beyond two faces leaves through the face it crosses first,
    /// and the crossing lies on that face
    #[test]
    fn a_run_leaving_past_two_faces_leaves_through_the_first() {
        let core = CoreBox { lo: [0.0, 0.0, -OPEN], hi: [10.0, 10.0, OPEN] };
        let interior = [Face { axis: 0, at: 10.0 }, Face { axis: 1, at: 10.0 }];
        // from (9, 8) to (13, 14): x = 10 at t = 0.25, y = 10 at t = 1/3, so the x face comes first
        let got = runs(cut(vec![[9.0, 8.0, 0.0], [13.0, 14.0, 0.0]], 1, &core, &interior));
        assert_eq!(got[0].tail, Some(interior[0]));
        assert_eq!(got[0].points[1], [10.0, 9.5, 0.0]);
        // from (9, 8) to (11, 14): y = 10 at t = 1/3 comes before x = 10 at t = 0.5
        let got = runs(cut(vec![[9.0, 8.0, 0.0], [11.0, 14.0, 0.0]], 1, &core, &interior));
        assert_eq!(got[0].tail, Some(interior[1]));
        let p = got[0].points[1];
        assert!((p[0] - (9.0 + 2.0 / 3.0)).abs() < 1e-12 && (p[1] - 10.0).abs() < 1e-12, "{p:?}");
    }

    /// A fibre that leaves the core and comes back gives two runs, each cut onto the face; one that leaves through
    /// a face no other tile shares just ends there, without a crossing point, and is not a cut run
    #[test]
    fn runs_that_leave_and_come_back_or_leave_the_volume() {
        let ([left, _], interior) = cores();
        let points = line(&[7.0, 9.0, 11.0, 12.0, 8.0, 7.0], 0.0);
        let got = runs(cut(points, 1, &left, &interior));
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].points, line(&[7.0, 9.0, 10.0], 0.0));
        assert_eq!((got[0].head, got[0].tail), (None, Some(interior[0])));
        assert_eq!(got[1].points, line(&[10.0, 8.0, 7.0], 0.0));
        assert_eq!((got[1].head, got[1].tail), (Some(interior[0]), None));

        let got = runs(cut(line(&[-2.0, 1.0, 3.0], 0.0), 1, &left, &interior));
        assert_eq!(got[0].points, line(&[1.0, 3.0], 0.0));
        assert!(!got[0].is_cut());
        // a run of one point inside is dropped
        assert!(runs(cut(line(&[-2.0, 1.0, -3.0], 0.0), 1, &left, &interior)).is_empty());
    }

    fn end_run(points: Vec<Point>, family: usize, head: Option<Face>, tail: Option<Face>, core_lo: f64) -> Run {
        Run { points, family, head, tail, core_lo: [core_lo, -OPEN, -OPEN] }
    }

    /// Of two ends within the radius of one end, the nearer is joined; ends of different families are not joined
    /// however near, and the radius is inclusive
    #[test]
    fn joins_go_shortest_first_within_the_family_and_radius() {
        let face = Some(Face { axis: 0, at: 10.0 });
        let left = end_run(line(&[8.0, 10.0], 0.0), 1, None, face, 0.0);
        let far = end_run(line(&[10.0, 12.0], 1.2), 1, face, None, 10.0);
        let near = end_run(line(&[10.0, 12.0], -0.5), 1, face, None, 10.0);
        let (chains, stats) = join(vec![left.clone(), far.clone(), near.clone()], RADIUS);
        assert_eq!(stats.joins, 1);
        assert_eq!(chains[0].1, [left.points.clone(), near.points.clone()].concat());
        assert_eq!(chains[1].1, far.points);

        let other = end_run(line(&[10.0, 12.0], 0.0), 2, face, None, 10.0);
        assert_eq!(join(vec![left.clone(), other], RADIUS).1.joins, 0);
        let edge = end_run(line(&[10.0, 12.0], 1.5), 1, face, None, 10.0);
        assert_eq!(join(vec![left.clone(), edge], RADIUS).1.joins, 1);
        let beyond = end_run(line(&[10.0, 12.0], 1.5 + 1e-9), 1, face, None, 10.0);
        assert_eq!(join(vec![left, beyond], RADIUS).1.joins, 0);
    }

    /// Two runs whose ends meet across two faces join once: the second join would close a loop
    #[test]
    fn a_join_that_closes_a_loop_is_skipped() {
        let (a, b) = (Some(Face { axis: 0, at: 10.0 }), Some(Face { axis: 0, at: 20.0 }));
        // a ring through three cores along x: the middle core [10, 20) holds two runs between the faces
        let middle_low = end_run(line(&[10.0, 15.0, 20.0], 0.0), 1, a, b, 10.0);
        let middle_high = end_run(line(&[20.0, 15.0, 10.0], 1.0), 1, b, a, 10.0);
        let left = end_run(vec![[10.0, 0.0, 5.0], [8.0, 0.5, 5.0], [10.0, 1.0, 5.0]], 1, a, a, 0.0);
        let right = end_run(vec![[20.0, 0.0, 5.0], [22.0, 0.5, 5.0], [20.0, 1.0, 5.0]], 1, b, b, 20.0);
        let (chains, stats) = join(vec![middle_low, middle_high, left, right], RADIUS);
        assert_eq!((stats.cut_ends, stats.joins, stats.chains), (8, 3, 1));
        assert_eq!(chains[0].1.len(), 12);
    }
}
