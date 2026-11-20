//! The GPU fibre walk's per-seed tracer (`fibrs-kernels/src/walk.rs`, which the `fb_walk_*` kernels run) on the
//! host, where the tests hold it equal to `kernels::trace_seed` seed by seed, with the coverage evolving in commit
//! order
//!
//! The core is the kernels' own source, included here as `eigen.rs` is in `axes.rs`, and runs in plain Rust
//! arithmetic, which never contracts, so a seed it traces like the CPU here is the device's computation too. The
//! host side is composed as the kernels compose it: the start, both halves into one path slot and the spans. A
//! result's coverage marks are the CPU's own `mark_list` over its accepted points, so they need no parity

// the core's unroll hint is the device's; the host needs none
macro_rules! unroll_full {
    () => {};
}

#[path = "../fibrs-kernels/src/walk.rs"]
#[allow(dead_code)]
mod shared;

use crate::kernels::{self, Coverage, Grid, test_support};
use crate::lookahead::RawCheck;
use crate::vol::{EDGE, SHIFT};
use shared::{Curve, Fields, HALF, Ops, Path, Reads};

/// The core's arithmetic in plain Rust, which rounds each operation to nearest and never contracts
struct Plain;

impl Ops for Plain {
    fn mul(a: f64, b: f64) -> f64 {
        a * b
    }

    fn add(a: f64, b: f64) -> f64 {
        a + b
    }

    fn div(a: f64, b: f64) -> f64 {
        a / b
    }

    fn sqrt(a: f64) -> f64 {
        a.sqrt()
    }

    fn mul32(a: f32, b: f32) -> f32 {
        a * b
    }

    fn add32(a: f32, b: f32) -> f32 {
        a + b
    }

    fn div32(a: f32, b: f32) -> f32 {
        a / b
    }

    fn sqrt32(a: f32) -> f32 {
        a.sqrt()
    }
}

/// The CPU tracer's fields as the core reads them
struct HostFields<'a, P, C, D> {
    f: &'a kernels::Fields<'a, P, C, D>,
}

impl<P: Grid<u8>, C: Grid<u16>, D: Grid<u16>> Fields for HostFields<'_, P, C, D> {
    fn dims(&self) -> [i32; 3] {
        self.f.pres.dims().map(|n| n as i32)
    }

    fn pres_corners(&self, z: i32, y: i32, x: i32) -> [u8; 8] {
        self.f.pres.corners(z as i64, y as i64, x as i64)
    }

    fn cent_corners(&self, z: i32, y: i32, x: i32) -> [u16; 8] {
        self.f.cent.corners(z as i64, y as i64, x as i64)
    }

    fn dir(&self, z: i32, y: i32, x: i32) -> u16 {
        self.f.dirs.at(z as i64, y as i64, x as i64)
    }

    fn dir_corners(&self, z: i32, y: i32, x: i32) -> [u16; 8] {
        self.f.dirs.corners(z as i64, y as i64, x as i64)
    }

    fn decode(&self, d: u16) -> [f64; 3] {
        kernels::decode_packed(d)
    }

    fn ring(&self, i: usize) -> f64 {
        kernels::ring()[i]
    }
}

/// The CPU's coverage read through the core, recording each read as the kernels do
struct HostReads<'a> {
    cov: &'a Coverage,
    dims: [i32; 3],
    checks: Vec<RawCheck>,
}

impl Reads for HostReads<'_> {
    fn read(&mut self, z: i32, y: i32, x: i32, kind: u8, t: [f64; 3]) -> u16 {
        let (slot, offset) = place(self.dims, z, y, x);
        let at = self.cov.place_at(slot, offset).expect("the core reads in-bounds voxels");
        let seen = test_support::get(self.cov, at);
        let same_dir = (kind == shared::SAME_DIR).then_some(t);
        self.checks.push(RawCheck { slot: slot as u32, offset, seen, same_dir });
        seen
    }
}

/// One path slot laid out as the kernels lay it out: `2 * cap + 1` points with the start at `cap`, the forward
/// half after it and the back half before it, reversed, its tangents negated
struct HostSlot {
    cap: u32,
    pts: Vec<[f32; 3]>,
    tans: Vec<[f32; 3]>,
}

impl HostSlot {
    fn new(cap: u32) -> Self {
        let n = 2 * cap as usize + 1;
        Self { cap, pts: vec![[f32::NAN; 3]; n], tans: vec![[f32::NAN; 3]; n] }
    }

    fn at(&self, half: u32, i: u32) -> usize {
        (if half == 0 { self.cap + 1 + i } else { self.cap - 1 - i }) as usize
    }
}

impl Path for HostSlot {
    fn pt(&self, half: u32, i: u32) -> [f32; 3] {
        self.pts[self.at(half, i)]
    }

    fn push(&mut self, half: u32, i: u32, p: [f32; 3], t: [f32; 3]) {
        let at = self.at(half, i);
        self.pts[at] = p;
        self.tans[at] = if half == 0 { t } else { t.map(|v| -v) };
    }
}

/// A seed's joined path (a window of its slot) with the spans' scratch
struct HostCurve<'a> {
    pts: &'a [[f32; 3]],
    stack: Vec<(u32, u32)>,
    cum: Vec<f32>,
    spans: Vec<(u32, u32)>,
}

impl Curve for HostCurve<'_> {
    fn pt(&self, i: u32) -> [f32; 3] {
        self.pts[i as usize]
    }

    fn push(&mut self, a: u32, b: u32) -> bool {
        self.stack.push((a, b));
        true
    }

    fn pop(&mut self) -> (u32, u32) {
        self.stack.pop().unwrap_or((u32::MAX, u32::MAX))
    }

    fn cum(&self, i: u32) -> f32 {
        self.cum[i as usize]
    }

    fn set_cum(&mut self, i: u32, v: f32) {
        self.cum[i as usize] = v;
    }

    fn accept(&mut self, a: u32, b: u32) -> bool {
        self.spans.push((a, b));
        true
    }
}

/// How a seed ends on the core
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Status {
    /// Traced: its curves (possibly none) are the CPU's
    Done,
    /// Back to the CPU: a half ran out of steps, or the spans failed
    Cpu,
}

/// One seed traced by the core, in the form the CPU's `Traced` has
#[derive(Debug)]
struct Outcome {
    status: Status,
    /// Whether the seed passed the start and was traced
    survived: bool,
    checks: Vec<RawCheck>,
    curves: Vec<Vec<[f32; 3]>>,
    tans: Vec<[f32; 3]>,
}

/// The brick slot and offset of voxel `z, y, x` in a volume of `dims`, as `vol::Place` has them
fn place(dims: [i32; 3], z: i32, y: i32, x: i32) -> (usize, u16) {
    let grid = dims.map(|n| (n as usize).div_ceil(EDGE));
    let (z, y, x) = (z as usize, y as usize, x as usize);
    let slot = ((z >> SHIFT) * grid[1] + (y >> SHIFT)) * grid[2] + (x >> SHIFT);
    let mask = EDGE - 1;
    (slot, ((z & mask) << (2 * SHIFT) | (y & mask) << SHIFT | (x & mask)) as u16)
}

/// One seed through the core as the kernels run it, with `cap` steps per half at most
fn trace<P: Grid<u8>, C: Grid<u16>, D: Grid<u16>>(
    f: &kernels::Fields<P, C, D>,
    cov: &Coverage,
    seed: [i32; 3],
    primary: bool,
    cap: u32,
) -> Outcome {
    let hf = HostFields { f };
    let mut r = HostReads { cov, dims: hf.dims(), checks: Vec::new() };
    let mut out =
        Outcome { status: Status::Done, survived: false, checks: Vec::new(), curves: Vec::new(), tans: Vec::new() };
    let Some((c, u)) = shared::prefix::<Plain, _, _>(&hf, &mut r, seed, primary) else {
        out.checks = r.checks;
        return out;
    };
    out.survived = true;
    let mut slot = HostSlot::new(cap);
    let halves = shared::trace_halves::<Plain, _, _, _>(&hf, &mut r, &mut slot, c, u, cap);
    out.checks = r.checks;
    let Some((nf, nb)) = halves else {
        out.status = Status::Cpu;
        return out;
    };
    let center = cap as usize;
    slot.pts[center] = c.map(|v| v as f32);
    slot.tans[center] = u.map(|v| v as f32);
    let ntot = nb + 1 + nf;
    if ntot < 8 {
        return out;
    }
    let lo = center - nb as usize;
    let range = lo..lo + ntot as usize;
    let mut cv = HostCurve {
        pts: &slot.pts[range.clone()],
        stack: Vec::new(),
        cum: vec![f32::NAN; ntot as usize],
        spans: Vec::new(),
    };
    if shared::spans::<Plain, _>(&mut cv, ntot, nb, primary).is_err() {
        out.status = Status::Cpu;
        return out;
    }
    let (pts, tans) = (&slot.pts[range.clone()], &slot.tans[range]);
    for &(a, b) in &cv.spans {
        let (a, b) = (a as usize, b as usize);
        out.tans.extend_from_slice(&tans[a..b]);
        out.curves.push(pts[a..b].to_vec());
    }
    out
}

#[cfg(test)]
mod tests {
    use std::path::Path as FsPath;

    use color_eyre::eyre::Result;

    use super::*;
    use crate::fields::{Catalogue, Presence, Seed, catalogue, centrality};
    use crate::kernels::test_support::{self, raw_check};
    use crate::kernels::{Scratch, Traced, encode_dir, pack_dir, trace_seed};
    use crate::stage1::Separate;
    use crate::vol::Vol;

    /// The step cap per half the kernels trace with
    const DEVICE_CAP: u32 = 256;

    /// What a parity run compared
    #[derive(Debug, Default)]
    struct Tally {
        /// Seeds the core at [`DEVICE_CAP`] sends back to the CPU
        cpu: usize,
        survivors: usize,
        curves: usize,
        /// Accepted points
        tans: usize,
        checks: usize,
    }

    /// A check with its tangent as bits, so a signed zero or NaN compares exactly
    fn check_bits(c: &RawCheck) -> (u32, u16, u16, Option<[u64; 3]>) {
        (c.slot, c.offset, c.seen, c.same_dir.map(|t| t.map(f64::to_bits)))
    }

    fn checks_bits(checks: &[RawCheck]) -> Vec<(u32, u16, u16, Option<[u64; 3]>)> {
        checks.iter().map(check_bits).collect()
    }

    /// A tracer that records every coverage read
    fn recording() -> Traced {
        Traced { reads: test_support::recording(), ..Traced::default() }
    }

    /// Every field of the core's outcome against `trace_seed`'s, for seed `i`
    fn compare(i: usize, got: &Outcome, cpu: &Result<()>, want: &Traced) {
        if cpu.is_err() {
            assert_eq!(got.status, Status::Cpu, "seed {i}: the CPU bails, the core gives {:?}", got.status);
            return;
        }
        assert_eq!(got.status, Status::Done, "seed {i}: the core sends back a seed the CPU traces");
        let cpu_checks: Vec<RawCheck> =
            test_support::checks(&want.reads).expect("a recording read set").iter().map(raw_check).collect();
        assert_eq!(checks_bits(&got.checks), checks_bits(&cpu_checks), "seed {i}: coverage reads");
        assert_eq!(got.curves.len(), want.curves.len(), "seed {i}: curve count");
        for (k, (a, b)) in got.curves.iter().zip(&want.curves).enumerate() {
            assert_eq!(a.len(), b.pts.len(), "seed {i}: curve {k} length");
            let bits = |p: &[[f32; 3]]| p.iter().flat_map(|q| q.map(f32::to_bits)).collect::<Vec<_>>();
            assert_eq!(bits(a), bits(&b.pts), "seed {i}: curve {k} points");
        }
        assert_eq!(got.tans.len(), want.tans.len(), "seed {i}: tangent count");
        let tan_bits = |t: &[[f32; 3]]| t.iter().flat_map(|q| q.map(f32::to_bits)).collect::<Vec<_>>();
        assert_eq!(tan_bits(&got.tans), tan_bits(&want.tans), "seed {i}: tangents");
    }

    /// Whether seed `i` of `cat` is primary: the first `primary[r]` seeds of its region
    fn primary(cat: &Catalogue, i: usize) -> bool {
        let r = cat.starts.partition_point(|&s| s <= i) - 1;
        i < cat.starts[r] + cat.primary[r]
    }

    /// Every seed of `cat` traced by `trace_seed` and by the core, committed in order: the core at [`HALF`] must
    /// equal the CPU in every field, and at [`DEVICE_CAP`] either equal it or send the seed back to the CPU
    fn parity<P: Grid<u8>, C: Grid<u16>, D: Grid<u16>>(f: &kernels::Fields<P, C, D>, cat: &Catalogue) -> Tally {
        let shape = f.pres.dims().map(|n| n as usize);
        let cov = Coverage::new(shape);
        let mut writer = cov.writer();
        let (mut scratch, mut traced) = (Scratch::default(), recording());
        let mut tally = Tally::default();
        for (i, &seed) in cat.coords.iter().enumerate() {
            let primary = primary(cat, i);
            let cpu = trace_seed(f, &cov, &mut scratch, seed, primary, &mut traced);
            let full = trace(f, &cov, seed, primary, HALF);
            compare(i, &full, &cpu, &traced);
            let capped = trace(f, &cov, seed, primary, DEVICE_CAP);
            if capped.status == Status::Cpu {
                tally.cpu += 1;
            } else {
                compare(i, &capped, &cpu, &traced);
                assert_eq!(
                    checks_bits(&capped.checks),
                    checks_bits(&full.checks),
                    "seed {i}: checks at the device cap"
                );
            }
            tally.survivors += full.survived as usize;
            tally.curves += full.curves.len();
            tally.tans += full.tans.len();
            tally.checks += full.checks.len();
            if cpu.is_ok() {
                writer.apply(i as u32, &traced);
            }
        }
        tally
    }

    /// The `produce.rs` test fixture: tilted fibres a few voxels apart, which come close enough for seeds to read
    /// bricks that other seeds' curves mark
    fn fibres() -> ([Vol<u8>; 2], Vol<u16>) {
        const SHAPE: [usize; 3] = [40, 96, 96];
        let mut own = Vol::new(SHAPE, 0u8);
        let mut dirs = Vol::new(SHAPE, pack_dir(128, 128));
        let mut lines = Vec::new();
        for (j, y0) in (4..92).step_by(6).enumerate() {
            for (k, x0) in (4..92).step_by(6).enumerate() {
                let slope = [((j * 7 + k * 3) % 9) as f64 * 0.04 - 0.16, ((j * 5 + k * 11) % 9) as f64 * 0.04 - 0.16];
                lines.push(([y0 as f64, x0 as f64], slope));
            }
        }
        for z in 0..SHAPE[0] {
            for y in 0..SHAPE[1] {
                for x in 0..SHAPE[2] {
                    let near = lines.iter().map(|&([y0, x0], [sy, sx])| {
                        let (dy, dx) = (y as f64 - (y0 + sy * z as f64), x as f64 - (x0 + sx * z as f64));
                        (dy * dy + dx * dx, [sy, sx])
                    });
                    let (d2, [sy, sx]) = near.min_by(|a, b| a.0.total_cmp(&b.0)).expect("fibres");
                    let p = 235.0 * (-d2 / 3.0).exp();
                    own.set(z, y, x, p as u8);
                    let n = (1.0 + sy * sy + sx * sx).sqrt();
                    let (nx, ny) = encode_dir((sx / n) as f32, (sy / n) as f32, (1.0 / n) as f32);
                    dirs.set(z, y, x, pack_dir(nx, ny));
                }
            }
        }
        ([own, Vol::new(SHAPE, 0u8)], dirs)
    }

    #[test]
    fn core_traces_the_fixture_as_the_cpu_does() {
        let (tracing, dirs) = fibres();
        let pres = Presence::new(&tracing, 1);
        let fam = centrality::<Separate, _>(&pres, &dirs);
        let cat = catalogue(&fam.seeds, &fam.cent);
        let fields = kernels::Fields { pres: &pres, cent: &fam.cent, dirs: &dirs };
        let tally = parity(&fields, &cat);
        eprintln!("fixture, {} seeds: {tally:?}", cat.coords.len());
        assert!(tally.curves >= 100, "the fixture should accept many curves: {tally:?}");
        assert!(tally.tans > 1000, "too few accepted points to test: {tally:?}");
    }

    /// One family's fields and catalogue from a run's `FIBRS_DEBUG_FIELDS` dumps
    struct Dumps {
        pres: Vol<u8>,
        cent: Vol<u16>,
        dirs: Vol<u16>,
        cat: Catalogue,
    }

    /// Read the dumps of `family` from a work dir: `nx.raw`, `ny.raw` and `family-N/{presence,centrality,seeds}.raw`
    /// as dense arrays of `shape`, checked against the catalogue's `coordinates.i4`
    fn load(work: &FsPath, family: usize, shape: [usize; 3]) -> Dumps {
        let read = |p: &FsPath| std::fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
        let fam = work.join(format!("family-{family}"));
        let n = shape.iter().product::<usize>();
        let (nx, ny) = (read(&work.join("nx.raw")), read(&work.join("ny.raw")));
        let pres_raw = read(&fam.join("presence.raw"));
        let cent_raw = read(&fam.join("centrality.raw"));
        let seeds_raw = read(&fam.join("seeds.raw"));
        assert_eq!([nx.len(), ny.len(), pres_raw.len(), cent_raw.len() / 2, seeds_raw.len()], [n; 5], "dump sizes");
        let mut pres = Vol::new(shape, 0u8);
        pres.write_block([0; 3], shape, &pres_raw);
        let cent_vals: Vec<u16> = cent_raw.as_chunks::<2>().0.iter().map(|&b| u16::from_le_bytes(b)).collect();
        let mut cent = Vol::new(shape, 0u16);
        cent.write_block([0; 3], shape, &cent_vals);
        let packed: Vec<u16> = nx.iter().zip(&ny).map(|(&a, &b)| pack_dir(a, b)).collect();
        let mut dirs = Vol::new(shape, pack_dir(128, 128));
        dirs.write_block([0; 3], shape, &packed);
        let mut seeds = Vec::new();
        for (i, &p) in seeds_raw.iter().enumerate() {
            if p != 0 {
                let at = [i / (shape[1] * shape[2]), i / shape[2] % shape[1], i % shape[2]];
                seeds.push(Seed::new(at, shape, p));
            }
        }
        let cat = catalogue(&seeds, &cent);
        let coords = read(&fam.join("coordinates.i4"));
        let want: Vec<i32> = coords.as_chunks::<4>().0.iter().map(|&b| i32::from_le_bytes(b)).collect();
        assert_eq!(cat.coords.concat(), want, "the rebuilt catalogue differs from the run's");
        Dumps { pres, cent, dirs, cat }
    }

    /// The tracing shape of a crop dump directory: its `shape.txt`, three numbers z y x
    fn crop_shape(work: &FsPath) -> [usize; 3] {
        let text = std::fs::read_to_string(work.join("shape.txt")).expect("shape.txt beside the dumps");
        let v: Vec<usize> = text.split_whitespace().map(|t| t.parse().expect("a shape number")).collect();
        [v[0], v[1], v[2]]
    }

    fn crop_parity(dir: &str) {
        let work = FsPath::new(dir).join("work");
        let shape = crop_shape(FsPath::new(dir));
        for family in [1, 2] {
            let d = load(&work, family, shape);
            let fields = kernels::Fields { pres: &d.pres, cent: &d.cent, dirs: &d.dirs };
            let tally = parity(&fields, &d.cat);
            eprintln!("{dir} family {family}, {} seeds: {tally:?}", d.cat.coords.len());
            assert!(tally.curves > 0, "family {family}: no curves to test");
        }
    }

    /// The core against the CPU on every seed of a real crop's two families (`FIBRS_WALK_MICRO`, default
    /// `/build/fibrs-prof/walk-dumps/crop-micro`, made by `dump.sh` there)
    #[test]
    #[ignore = "needs a crop's field dumps"]
    fn core_traces_crop_micro_as_the_cpu_does() {
        let dir = std::env::var("FIBRS_WALK_MICRO").unwrap_or("/build/fibrs-prof/walk-dumps/crop-micro".into());
        crop_parity(&dir);
    }

    /// As [`core_traces_crop_micro_as_the_cpu_does`] on the 128 × 512 × 512 crop (`FIBRS_WALK_CROP`)
    #[test]
    #[ignore = "needs a crop's field dumps"]
    fn core_traces_crop_256_as_the_cpu_does() {
        let dir = std::env::var("FIBRS_WALK_CROP").unwrap_or("/build/fibrs-prof/walk-dumps/crop-z2304-256".into());
        crop_parity(&dir);
    }

    /// The kernels on a CUDA device against the core and `trace_seed`
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    mod gpu {
        use std::ops::Range;
        use std::time::Instant;

        use super::*;
        use crate::gpu::walk::{self as gw, CHUNK, Caps, IN_FLIGHT, STEP_CAP};
        use crate::gpu::{Gpu, Stages};
        use crate::kernels::SplitDirs;
        use crate::lookahead::{self, ChunkOut, Device};

        /// What a GPU parity run compared
        #[derive(Debug, Default)]
        struct GpuTally {
            seeds: usize,
            /// Seeds the device traced, each equal to the core and to `trace_seed`
            done: usize,
            /// Seeds the device sent back to the CPU
            cpu: usize,
            /// Of those, seeds the core traces at the device's step cap, so a chunk's buffers ran out
            overflow: usize,
            /// Accepted points of the seeds the device traced
            pts: usize,
        }

        fn span(r: &Range<u32>) -> Range<usize> {
            r.start as usize..r.end as usize
        }

        /// Seed `k` of a chunk in the core's form
        fn outcome(out: &ChunkOut, k: usize) -> Outcome {
            let r = &out.results[k];
            let status = match r.status {
                lookahead::Status::Done => Status::Done,
                lookahead::Status::Cpu => Status::Cpu,
            };
            Outcome {
                status,
                survived: false,
                checks: out.checks[span(&r.checks)].to_vec(),
                curves: out.curves[span(&r.curves)].iter().map(|c| out.pts[span(c)].to_vec()).collect(),
                tans: out.tans[span(&r.tans)].to_vec(),
            }
        }

        /// The direction field as the two byte volumes the GPU walk takes
        fn split(dirs: &Vol<u16>) -> (Vol<u8>, Vol<u8>) {
            let shape = dirs.shape();
            let all = dirs.read_block([0; 3], shape);
            let half = |f: fn(u16) -> u8| {
                let mut v = Vol::new(shape, 128u8);
                v.write_block([0; 3], shape, &all.iter().map(|&d| f(d)).collect::<Vec<_>>());
                v
            };
            (half(|d| (d >> 8) as u8), half(|d| d as u8))
        }

        /// Every seed of `cat` through the device in chunks of `chunk`, [`IN_FLIGHT`] at a time against one
        /// coverage, each result held to the core and `trace_seed` on that coverage; then the group's seeds are
        /// committed in order on the CPU, and the device's mirror updated with what they marked
        ///
        /// `tracing` is the pair `Presence::new(tracing, 1)` reads as `pres`; the device gets the direction field
        /// split in two (as GPU runs hold it) when `split_dirs`, packed otherwise
        #[allow(clippy::too_many_arguments)]
        fn gpu_parity<P: Grid<u8>>(
            gpu: &Gpu,
            tracing: &[Vol<u8>; 2],
            pres: &P,
            cent: &Vol<u16>,
            dirs: &Vol<u16>,
            cat: &Catalogue,
            chunk: usize,
            caps: Caps,
            split_dirs: bool,
        ) -> Result<GpuTally> {
            let f = kernels::Fields { pres, cent, dirs };
            let (nx, ny) = split(dirs);
            let mut walk = if split_dirs {
                gpu.walk_with(tracing, 1, cent, &SplitDirs { nx: &nx, ny: &ny }, cat, kernels::ring(), caps)?
            } else {
                gpu.walk_with(tracing, 1, cent, dirs, cat, kernels::ring(), caps)?
            };
            eprintln!("walk device bytes: {:.2} GiB", walk.device_bytes() as f64 / (1u64 << 30) as f64);
            let cov = Coverage::new(cent.shape());
            let mut writer = cov.writer();
            writer.log_changes();
            let (mut scratch, mut traced) = (Scratch::default(), recording());
            let n = cat.coords.len();
            let mut tally = GpuTally { seeds: n, ..GpuTally::default() };
            let mut next = 0;
            while next < n {
                let group = next..(next + IN_FLIGHT * chunk).min(n);
                while next < group.end {
                    let end = (next + chunk).min(group.end);
                    walk.launch(next..end)?;
                    next = end;
                }
                let mut seen = group.start;
                while let Some(out) = walk.collect()? {
                    assert_eq!(out.seeds.start, seen, "chunks come back in launch order");
                    assert_eq!(out.results.len(), out.seeds.len(), "one result per seed");
                    seen = out.seeds.end;
                    for (k, i) in out.seeds.clone().enumerate() {
                        let (seed, primary) = (cat.coords[i], primary(cat, i));
                        let core = trace(&f, &cov, seed, primary, STEP_CAP);
                        let got = outcome(&out, k);
                        if got.status == Status::Cpu {
                            tally.cpu += 1;
                            tally.overflow += (core.status == Status::Done) as usize;
                            continue;
                        }
                        assert_eq!(core.status, Status::Done, "seed {i}: the device traced a seed the core sends back");
                        assert_eq!(
                            checks_bits(&got.checks),
                            checks_bits(&core.checks),
                            "seed {i}: checks, against the core"
                        );
                        let cpu = trace_seed(&f, &cov, &mut scratch, seed, primary, &mut traced);
                        compare(i, &got, &cpu, &traced);
                        tally.done += 1;
                        tally.pts += got.tans.len();
                    }
                }
                assert_eq!(seen, group.end, "every launched chunk came back");

                let mut changes = Vec::new();
                for i in group {
                    let cpu = trace_seed(&f, &cov, &mut scratch, cat.coords[i], primary(cat, i), &mut traced);
                    if cpu.is_err() {
                        continue;
                    }
                    writer.apply(i as u32, &traced);
                }
                writer.take_changes(&mut changes);
                walk.update(&changes)?;
            }
            assert_eq!(tally.done + tally.cpu, tally.seeds, "a result per seed");
            Ok(tally)
        }

        fn open() -> Gpu {
            Gpu::open(None, Stages::DirectionField).expect("a CUDA device")
        }

        /// Room small enough that the fixture's chunks run out of it
        const TIGHT: Caps = Caps { survivors: 48, checks: 2048, pts: 1024, curves: 32 };

        #[test]
        #[ignore = "needs a CUDA device"]
        fn gpu_traces_the_fixture_as_the_cpu_does() {
            let gpu = open();
            let (tracing, dirs) = fibres();
            let pres = Presence::new(&tracing, 1);
            let fam = centrality::<Separate, _>(&pres, &dirs);
            let cat = catalogue(&fam.seeds, &fam.cent);
            let t = gpu_parity(&gpu, &tracing, &pres, &fam.cent, &dirs, &cat, 512, Caps::default(), false).unwrap();
            eprintln!("fixture: {t:?}");
            assert_eq!(t.overflow, 0, "the default caps hold the fixture's chunks: {t:?}");
            assert!(t.done > 8000 && t.pts > 1000, "too little traced on the device: {t:?}");
            let t = gpu_parity(&gpu, &tracing, &pres, &fam.cent, &dirs, &cat, 512, TIGHT, true).unwrap();
            eprintln!("fixture, tight caps: {t:?}");
            assert!(t.overflow > 0, "tight caps should overflow: {t:?}");
            assert!(t.done > 0, "tight caps still trace some seeds: {t:?}");
        }

        /// Both families of a crop's dumps (`load`), the presence given to the device as an own prediction over an
        /// empty other, which `Presence` combines back to the same values
        /// `full_caps` asserts that no chunk ran out of room: true of a small crop, while a dense crop's chunks
        /// traced against coverage a few chunks behind can fill a buffer, which only sends seeds back to the CPU
        fn gpu_crop(dir: &str, chunk: usize, full_caps: bool) {
            let gpu = open();
            let work = FsPath::new(dir).join("work");
            let shape = crop_shape(FsPath::new(dir));
            for family in [1, 2] {
                let Dumps { pres, cent, dirs, cat } = load(&work, family, shape);
                let tracing = [pres, Vol::new(shape, 0u8)];
                let t =
                    gpu_parity(&gpu, &tracing, &tracing[0], &cent, &dirs, &cat, chunk, Caps::default(), family == 2)
                        .unwrap();
                eprintln!("{dir} family {family}: {t:?}");
                assert!(
                    !full_caps || t.overflow == 0,
                    "family {family}: the default caps hold the crop's chunks: {t:?}"
                );
                assert!(t.done > t.seeds / 4, "family {family}: too few seeds traced on the device: {t:?}");
            }
        }

        #[test]
        #[ignore = "needs a CUDA device and a crop's field dumps"]
        fn gpu_traces_crop_micro_as_the_cpu_does() {
            let dir = std::env::var("FIBRS_WALK_MICRO").unwrap_or("/build/fibrs-prof/walk-dumps/crop-micro".into());
            gpu_crop(&dir, 2048, true);
        }

        #[test]
        #[ignore = "needs a CUDA device and a crop's field dumps"]
        fn gpu_traces_crop_256_as_the_cpu_does() {
            let dir = std::env::var("FIBRS_WALK_CROP").unwrap_or("/build/fibrs-prof/walk-dumps/crop-z2304-256".into());
            gpu_crop(&dir, CHUNK / 2, false);
        }

        /// The larger crop's V family in chunks of 16K, 32K and 64K seeds, each launched with the coverage the
        /// CPU has committed up to its first seed, as the look-ahead launches just past the commit point: device
        /// time per chunk (update, launch and collect), seeds per second and the share sent back to the CPU
        #[test]
        #[ignore = "needs a CUDA device and a crop's field dumps"]
        fn gpu_walk_speed() {
            let dir = std::env::var("FIBRS_WALK_CROP").unwrap_or("/build/fibrs-prof/walk-dumps/crop-z2304-256".into());
            let gpu = open();
            let shape = crop_shape(FsPath::new(&dir));
            let Dumps { pres, cent, dirs, cat } = load(&FsPath::new(&dir).join("work"), 1, shape);
            let tracing = [pres, Vol::new(shape, 0u8)];
            let f = kernels::Fields { pres: &tracing[0], cent: &cent, dirs: &dirs };
            let n = cat.coords.len();
            for size in [CHUNK / 4, CHUNK / 2, CHUNK] {
                let mut walk: gw::Walk = gpu.walk(&tracing, 1, &cent, &dirs, &cat, kernels::ring()).unwrap();
                let bytes = walk.device_bytes() as f64 / (1u64 << 30) as f64;
                // one warm-up chunk for the driver's JIT and first-touch costs
                walk.launch(0..size.min(n)).unwrap();
                walk.collect().unwrap();
                let cov = Coverage::new(shape);
                let mut writer = cov.writer();
                writer.log_changes();
                let (mut scratch, mut traced, mut changes) = (Scratch::default(), Traced::default(), Vec::new());
                let (mut committed, mut cpu, mut total, mut worst) = (0, 0, 0.0f64, 0.0f64);
                for start in (0..n).step_by(size) {
                    while committed < start {
                        let seed = cat.coords[committed];
                        if trace_seed(&f, &cov, &mut scratch, seed, primary(&cat, committed), &mut traced).is_ok() {
                            writer.apply(committed as u32, &traced);
                        }
                        committed += 1;
                    }
                    changes.clear();
                    writer.take_changes(&mut changes);
                    let t = Instant::now();
                    walk.update(&changes).unwrap();
                    walk.launch(start..(start + size).min(n)).unwrap();
                    let out = walk.collect().unwrap().unwrap();
                    let ms = t.elapsed().as_secs_f64() * 1e3;
                    let back = out.results.iter().filter(|r| r.status == lookahead::Status::Cpu).count();
                    let pts = out.pts.len();
                    eprintln!(
                        "chunk {size} at {start}: {ms:.1} ms, {back} to the CPU, {pts} accepted points, {} changes",
                        changes.len()
                    );
                    (cpu, total, worst) = (cpu + back, total + ms, worst.max(ms));
                }
                let chunks = n.div_ceil(size);
                eprintln!(
                    "chunk {size}: {n} seeds, {:.0} seeds/s, {:.1} ms a chunk (slowest {worst:.1}), {:.2}% to the CPU, \
                     {bytes:.2} GiB on the device",
                    n as f64 / (total / 1e3),
                    total / chunks as f64,
                    100.0 * cpu as f64 / n as f64
                );
            }
        }
    }
}
