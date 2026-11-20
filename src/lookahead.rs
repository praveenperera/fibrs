//! Seeds traced far ahead of the commit point by another device, the GPU, and checked against the live coverage
//! before they are used
//!
//! The device traces chunks of seeds against its own copy of the coverage, which lags the committed coverage by the
//! changes still on their way to it. Each result lists every coverage read its tracing made, with the value it read
//! ([`RawCheck`]). Coverage only ever grows (uncovered, then one direction, then covered twice), so each read's
//! outcome changes at most once, from false to true. A result whose reads still give the same outcomes on the live
//! coverage is therefore the result tracing the seed now would give, and was at every commit since the device read
//! the coverage. A tracing thread that finds one uses it in place of tracing the seed ([`Ahead::fill`]); the commit
//! then treats it like any seed traced ahead, checking the bricks its reads touched for later changes. A seed whose
//! result no longer holds, that the device has not reached, or that it gave back ([`Status::Cpu`]) is traced on the
//! CPU as before, so the output is the CPU's whatever the device does or how far behind it is
//!
//! One thread drives the device ([`Ahead::drive`]): it sends the committed coverage changes, launches the next chunk
//! a little beyond the commit point and publishes chunks as they finish. When the commit passes the device, the next
//! chunk starts beyond the commit point again, so the CPU and the device split the seeds between them

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use color_eyre::eyre::Result;

use crate::kernels::{Change, Check, Coverage, Fields, Grid, Probe, RawCurve, Traced};

/// One coverage read of a seed traced on the device, as the device reports it
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RawCheck {
    /// Brick slot (`vol.rs`'s C order over the brick grid) and offset in the brick of the voxel read
    pub slot: u32,
    pub offset: u16,
    /// The value read
    pub seen: u16,
    /// `None` for `is_covered`, the tangent for `is_covered_same_dir`
    pub same_dir: Option<[f64; 3]>,
}

/// How the device left a seed
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// Traced: its result is in the chunk
    #[cfg_attr(not(all(target_os = "linux", target_arch = "x86_64")), allow(dead_code))]
    Done,
    /// Given back, for a reason the CPU's tracer handles itself (a path longer than the device holds, an error
    /// the CPU reports): the CPU traces it
    Cpu,
}

/// One seed of a [`ChunkOut`]: ranges into the chunk's arrays
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeedOut {
    pub status: Status,
    pub checks: Range<u32>,
    /// Into [`ChunkOut::curves`]
    pub curves: Range<u32>,
    /// The tangents of the accepted curves' points, in curve order, as [`Traced::tans`]
    pub tans: Range<u32>,
}

/// A chunk of consecutive seeds traced on the device
#[derive(Debug, Default)]
pub struct ChunkOut {
    pub seeds: Range<usize>,
    /// One per seed of `seeds`, in order
    pub results: Vec<SeedOut>,
    pub checks: Vec<RawCheck>,
    pub pts: Vec<[f32; 3]>,
    pub tans: Vec<[f32; 3]>,
    /// Each accepted curve's points, a range of `pts`
    pub curves: Vec<Range<u32>>,
}

/// A device that traces chunks of a family's seeds against its own copy of the coverage
pub trait Device: Send {
    /// Apply committed coverage changes to the device's copy; chunks launched later read them, and chunks running
    /// may read either value of a voxel that changes
    fn update(&mut self, changes: &[Change]) -> Result<()>;
    /// Start tracing `seeds`
    fn launch(&mut self, seeds: Range<usize>) -> Result<()>;
    /// The oldest launched chunk, waiting for it; `None` when none is in flight
    fn collect(&mut self) -> Result<Option<ChunkOut>>;
    fn in_flight(&self) -> usize;
}

/// How the driving thread paces the device
#[derive(Clone, Copy, Debug)]
pub struct Pace {
    /// Seeds per chunk
    pub chunk: usize,
    /// Chunks in flight on the device at once
    pub depth: usize,
    /// How far beyond the commit point the device may run, in seeds: the further, the older the coverage it reads
    pub reach: usize,
}

impl Default for Pace {
    fn default() -> Self {
        Self { chunk: 1 << 15, depth: 2, reach: 1 << 17 }
    }
}

/// What the device contributed, for the log
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AheadStats {
    /// Seeds the device traced
    pub traced: usize,
    /// Results a tracing thread used in place of tracing the seed
    pub used: usize,
    /// Results whose reads no longer held when a thread looked, so it traced the seed
    pub stale: usize,
    /// Seeds the device gave back
    pub given_back: usize,
}

/// Results traced ahead on a device, shared by the thread driving it and the threads tracing seeds
#[derive(Default)]
pub struct Ahead {
    /// Published chunks by first seed; none overlap
    chunks: RwLock<BTreeMap<usize, Arc<ChunkOut>>>,
    /// Committed coverage changes not yet sent to the device
    changes: Mutex<Vec<Change>>,
    traced: AtomicUsize,
    used: AtomicUsize,
    stale: AtomicUsize,
    given_back: AtomicUsize,
}

impl Ahead {
    /// Coverage changes the committer has applied, for the device
    pub fn send_changes(&self, writer_changes: impl FnOnce(&mut Vec<Change>)) {
        writer_changes(&mut self.changes.lock().expect("changes lock"));
    }

    /// Fill `out` with the device's result for seed `i` when it has one whose reads hold on `cov` now
    ///
    /// The marks come from [`Coverage::mark_list`] over the accepted curves, as [`crate::kernels::trace_seed`]
    /// makes them: the device returns only the curves, since a seed in fresh territory marks thousands of voxels
    /// and copying those for every seed of a chunk cost more host memory and time than marking the few used
    pub fn fill<P: Grid<u8>, C: Grid<u16>, D: Grid<u16>>(
        &self,
        f: &Fields<P, C, D>,
        cov: &Coverage,
        i: usize,
        out: &mut Traced,
    ) -> bool {
        let Some(chunk) = self.chunk_of(i) else { return false };
        let seed = &chunk.results[i - chunk.seeds.start];
        if seed.status == Status::Cpu {
            return false;
        }
        let checks = &chunk.checks[range(&seed.checks)];
        if !holds(cov, checks) {
            self.stale.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        out.discard();
        for c in checks {
            out.reads.push(c.slot as usize);
        }
        // each curve's tangents follow the last's in the seed's range, one per point
        let mut tans = &chunk.tans[range(&seed.tans)];
        for c in &chunk.curves[range(&seed.curves)] {
            let pts = &chunk.pts[range(c)];
            let (own, rest) = tans.split_at(pts.len());
            cov.mark_list(f, pts, own, out.tans.len(), &mut out.marks);
            out.tans.extend_from_slice(own);
            out.curves.push(RawCurve { pts: pts.to_vec() });
            tans = rest;
        }
        self.used.fetch_add(1, Ordering::Relaxed);
        true
    }

    fn chunk_of(&self, i: usize) -> Option<Arc<ChunkOut>> {
        let chunks = self.chunks.read().expect("chunks lock");
        let (_, chunk) = chunks.range(..=i).next_back()?;
        chunk.seeds.contains(&i).then(|| chunk.clone())
    }

    pub fn stats(&self) -> AheadStats {
        AheadStats {
            traced: self.traced.load(Ordering::Relaxed),
            used: self.used.load(Ordering::Relaxed),
            stale: self.stale.load(Ordering::Relaxed),
            given_back: self.given_back.load(Ordering::Relaxed),
        }
    }

    /// Drive `device` over seeds `0..total` until every seed is committed or `stop` is set
    ///
    /// `committed` is the producer's commit counter. Each chunk starts at the commit point plus the seeds the
    /// commit passed while the last chunk ran, so it lands just ahead of the commit, and no chunk starts more than
    /// `pace.reach` seeds beyond it
    pub fn drive(
        &self,
        device: &mut dyn Device,
        total: usize,
        committed: &AtomicUsize,
        stop: &AtomicBool,
        pace: Pace,
    ) -> Result<()> {
        let mut next = 0;
        let mut lead = 0;
        let mut sent = Vec::new();
        loop {
            let c = committed.load(Ordering::Acquire);
            if c >= total || stop.load(Ordering::Relaxed) {
                break;
            }
            let start = next.max(c + lead);
            let room = device.in_flight() < pace.depth && start < total;
            if !room && device.in_flight() > 0 {
                if let Some(chunk) = device.collect()? {
                    self.publish(chunk);
                }
                // the commit's progress while the device worked: start the next chunk that far ahead of it
                let now = committed.load(Ordering::Acquire);
                lead = now.saturating_sub(c).max(lead / 2);
                self.prune(now);
                continue;
            }
            if !room {
                break;
            }
            if start >= c + pace.reach {
                std::thread::sleep(Duration::from_micros(200));
                continue;
            }
            sent.clear();
            std::mem::swap(&mut sent, &mut self.changes.lock().expect("changes lock"));
            device.update(&sent)?;
            let end = (start + pace.chunk).min(total);
            device.launch(start..end)?;
            next = end;
        }
        // drain what is in flight so the device is idle when the family ends
        while device.collect()?.is_some() {}
        Ok(())
    }

    fn publish(&self, chunk: ChunkOut) {
        let given_back = chunk.results.iter().filter(|s| s.status == Status::Cpu).count();
        self.traced.fetch_add(chunk.results.len() - given_back, Ordering::Relaxed);
        self.given_back.fetch_add(given_back, Ordering::Relaxed);
        self.chunks.write().expect("chunks lock").insert(chunk.seeds.start, Arc::new(chunk));
    }

    /// Drop the chunks every seed of which is committed
    fn prune(&self, committed: usize) {
        let mut chunks = self.chunks.write().expect("chunks lock");
        chunks.retain(|_, c| c.seeds.end > committed);
    }
}

fn range(r: &Range<u32>) -> Range<usize> {
    r.start as usize..r.end as usize
}

/// Whether every read still gives the outcome it gave, as [`Coverage::holds`] for device reads
fn holds(cov: &Coverage, checks: &[RawCheck]) -> bool {
    checks.iter().all(|c| {
        let Some(at) = cov.place_at(c.slot as usize, c.offset) else { return false };
        let probe = c.same_dir.map_or(Probe::Covered, Probe::SameDir);
        cov.holds(&[Check { at, seen: c.seen, probe }])
    })
}
