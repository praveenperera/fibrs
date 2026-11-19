//! The tracing producer: seeds traced ahead on several threads and committed one at a time, in seed order
//!
//! Each seed's result depends on the coverage that the seeds before it marked, so the reference traces them one
//! after another, and so did this port: one thread traced every seed while the others finished curves, and on
//! dense bands that thread was the limit. Here the finishing pool's idle threads also trace the next `window`
//! seeds ahead of the commit point against the live coverage, each recording which coverage bricks it read and how
//! many seeds had been committed when it started. One thread, the committer, then takes the results in seed
//! order. A result whose bricks no seed committed since then has changed read exactly the values the serial
//! tracer would have, so it is committed as it is: its marks are applied and its curves handed on. Any other is
//! traced again by the committer, which is then the serial tracer itself. Either way every committed result is
//! the serial one, and the `.afv` is the same
//!
//! `FIBRS_DEBUG_WINDOW=N` sets the window (1 traces only the seed at the commit point) and `FIBRS_DEBUG_RETRACE=1`
//! makes the committer trace every seed traced ahead again; both are for testing

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::thread::Thread;

use color_eyre::eyre::{Result, bail};

use crate::fields::Catalogue;
use crate::kernels::{Coverage, Fields, Grid, RawCurve, Scratch, Traced, trace_seed};

/// Seeds traced ahead of the commit point
///
/// A seed traced ahead is traced again when a seed committed meanwhile changed a brick it read: on a dense
/// PHerc0358 crop 0.01% of seeds were at 64, 0.04% at 256 and 0.15% at 1024. A larger window gave the idle threads
/// no more to do there: 256 traced as fast as 64 and 1024 was slower, with more traced again and each seed's
/// result held longer
pub const WINDOW: usize = 64;

/// How the producer traces ahead
#[derive(Clone, Copy, Debug)]
pub struct Speculation {
    /// Seeds traced ahead of the commit point, at least 1
    pub window: usize,
    /// Trace every seed traced ahead again on the committer, as if each had read a changed brick
    pub retrace_all: bool,
}

impl Speculation {
    /// The default window, or the one `FIBRS_DEBUG_WINDOW` and `FIBRS_DEBUG_RETRACE` set
    pub fn from_env() -> Result<Self> {
        let window = match std::env::var("FIBRS_DEBUG_WINDOW") {
            Ok(v) => match v.parse::<usize>() {
                Ok(n) if n >= 1 => n,
                _ => bail!("FIBRS_DEBUG_WINDOW must be a whole number of seeds of at least 1, not {v:?}"),
            },
            Err(_) => WINDOW,
        };
        let retrace_all = std::env::var_os("FIBRS_DEBUG_RETRACE").is_some_and(|v| v != "0");
        Ok(Self { window, retrace_all })
    }
}

/// What the producer did, for the log
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub seeds: usize,
    /// Seeds traced ahead whose bricks changed before their commit, so the committer traced them again
    pub retraced: usize,
    /// Seeds no other thread had started when they reached the commit point, so the committer traced them
    pub committer: usize,
}

/// Whether the producer goes on after a seed's curves are handed over
pub enum Flow {
    Continue,
    /// The receiver is gone; stop without an error of the producer's own
    Stop,
}

impl<'a, P: Grid<u8>, C: Grid<u16>, D: Grid<u16>> Producer<'a, P, C, D> {
    /// A producer for the seeds of `cat`, which marks `cov`
    pub fn new(
        fields: &'a Fields<'a, P, C, D>,
        cov: &'a Coverage,
        cat: &'a Catalogue,
        spec: Speculation,
    ) -> Result<Self> {
        let total = cat.coords.len();
        // commit stamps are seed indices plus one
        if u32::try_from(total).map_or(true, |n| n == u32::MAX) {
            bail!("{total} seeds is more than one family can trace");
        }
        let window = spec.window.max(1);
        Ok(Self {
            fields,
            cov,
            cat,
            window,
            retrace_all: spec.retrace_all,
            slots: (0..window).map(|_| Slot::default()).collect(),
            next: AtomicUsize::new(0),
            committed: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            helpers: AtomicUsize::new(0),
            parked_on: AtomicUsize::new(0),
            panicked: AtomicBool::new(false),
            committer: OnceLock::new(),
            scratch: Mutex::new(Vec::new()),
        })
    }

    /// Commit every seed in order on the calling thread, handing each seed's accepted curves to `deliver`
    ///
    /// Idle threads of `pool` trace ahead. It should be the pool that finishes the curves: while finishing keeps
    /// it busy, the committer traces the seeds itself, as one thread traced every seed before
    pub fn run(
        &self,
        pool: &rayon::ThreadPool,
        mut deliver: impl FnMut(usize, &mut Vec<RawCurve>) -> Result<Flow>,
    ) -> Result<Stats> {
        if self.committer.set(std::thread::current()).is_err() {
            bail!("a producer runs once");
        }
        let max_helpers = pool.current_num_threads();
        pool.in_place_scope(|scope| {
            let result = self.commit_all(&mut deliver, || {
                if self.helpers.load(Ordering::Relaxed) < max_helpers && self.claimable() {
                    self.helpers.fetch_add(1, Ordering::Relaxed);
                    scope.spawn(|_| self.help());
                }
            });
            // helpers still tracing ahead finish their seed and see no more to claim
            self.stop.store(true, Ordering::Relaxed);
            result
        })
    }
}

/// Seeds traced ahead of the commit point and committed in order, shared by the committer and the threads that
/// trace ahead
pub struct Producer<'a, P, C, D> {
    fields: &'a Fields<'a, P, C, D>,
    cov: &'a Coverage,
    cat: &'a Catalogue,
    window: usize,
    retrace_all: bool,
    /// Seed `i`'s result, while it is traced ahead, is in `slots[i % window]`
    slots: Box<[Slot]>,
    /// The next seed to claim; seeds are claimed in order, by helpers or by the committer
    next: AtomicUsize,
    /// Seeds committed so far. Its release store after a commit publishes the commit's coverage writes and frees
    /// the committed seed's slot
    committed: AtomicUsize,
    stop: AtomicBool,
    helpers: AtomicUsize,
    /// One more than the seed whose result the committer is parked waiting for, or 0
    parked_on: AtomicUsize,
    panicked: AtomicBool,
    /// The thread in [`Producer::run`]
    committer: OnceLock<Thread>,
    /// Tracing buffers of helpers that have returned, a few MiB each, for the next helpers
    scratch: Mutex<Vec<Scratch>>,
}

/// One seed traced ahead
#[derive(Default)]
struct Slot {
    /// One more than the seed whose result `ahead` holds, stored with release once it is complete
    done: AtomicUsize,
    ahead: UnsafeCell<TracedAhead>,
}

// SAFETY: a slot's result is written only by the thread that claimed its seed, before `done` publishes it, and
// read only by the committer after it sees `done`; the slot is claimed again only after the committer's release
// of `committed` shows it is finished with it
unsafe impl Sync for Slot {}

struct TracedAhead {
    /// Seeds committed when tracing started
    committed: usize,
    traced: Traced,
    result: Result<()>,
}

impl Default for TracedAhead {
    fn default() -> Self {
        Self { committed: 0, traced: Traced::default(), result: Ok(()) }
    }
}

/// What the committer has for the seed at the commit point
enum Ready {
    /// Traced ahead, its result in the seed's slot
    Ahead,
    /// Claimed by the committer, which traces it against the coverage as it is
    Claimed,
}

/// Sets `panicked` and wakes the committer if a helper unwinds, so the committer stops waiting for its seed
struct PanicGuard<'g>(&'g AtomicBool, &'g Thread);

impl Drop for PanicGuard<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.store(true, Ordering::SeqCst);
            self.1.unpark();
        }
    }
}

/// Spins before a committer parks or a helper gives its thread back: most seeds take well under a microsecond
const SPINS: usize = 1 << 12;

impl<P: Grid<u8>, C: Grid<u16>, D: Grid<u16>> Producer<'_, P, C, D> {
    fn primary(&self, i: usize) -> bool {
        let r = self.cat.starts.partition_point(|&s| s <= i) - 1;
        i < self.cat.starts[r] + self.cat.primary[r]
    }

    fn trace(&self, i: usize, scratch: &mut Scratch, out: &mut Traced) -> Result<()> {
        trace_seed(self.fields, self.cov, scratch, self.cat.coords[i], self.primary(i), out)
    }

    /// Whether a helper could claim a seed now
    fn claimable(&self) -> bool {
        let next = self.next.load(Ordering::Relaxed);
        next < self.cat.coords.len() && next < self.committed.load(Ordering::Relaxed) + self.window
    }

    /// Claim the next seed if it lies in the window
    fn claim(&self) -> Option<usize> {
        let mut i = self.next.load(Ordering::Relaxed);
        loop {
            // acquire: the slot's previous seed has been committed and the committer is done with the slot
            let limit = self.committed.load(Ordering::Acquire) + self.window;
            if self.stop.load(Ordering::Relaxed) || i >= self.cat.coords.len() || i >= limit {
                return None;
            }
            match self.next.compare_exchange_weak(i, i + 1, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => return Some(i),
                Err(now) => i = now,
            }
        }
    }

    /// A helper on an idle pool thread: trace seeds ahead until none is left to claim for a while
    ///
    /// Finishing threads could also trace between curves, but on a PHerc0826 band at `--threads 4` that delayed
    /// their chunk reads enough to make tracing slower than one thread had been
    fn help(&self) {
        let mut scratch = self.scratch.lock().expect("scratch lock").pop().unwrap_or_default();
        'claims: loop {
            if let Some(i) = self.claim() {
                self.trace_ahead(i, &mut scratch);
                continue;
            }
            // the window is full: wait a little for the committer before handing the thread back to finishing
            for _ in 0..SPINS {
                if self.stop.load(Ordering::Relaxed) || self.next.load(Ordering::Relaxed) >= self.cat.coords.len() {
                    break 'claims;
                }
                if self.claimable() {
                    continue 'claims;
                }
                std::hint::spin_loop();
            }
            break;
        }
        self.scratch.lock().expect("scratch lock").push(scratch);
        self.helpers.fetch_sub(1, Ordering::Relaxed);
    }

    fn trace_ahead(&self, i: usize, scratch: &mut Scratch) {
        let committer = self.committer.get().expect("helpers start once the committer runs");
        let _guard = PanicGuard(&self.panicked, committer);
        let slot = &self.slots[i % self.window];
        // SAFETY: this thread claimed seed `i`, so it alone writes the slot until it publishes `done`
        let ahead = unsafe { &mut *slot.ahead.get() };
        // acquire: every coverage write of the first `committed` seeds is visible to the reads below
        ahead.committed = self.committed.load(Ordering::Acquire);
        ahead.result = self.trace(i, scratch, &mut ahead.traced);
        // sequentially consistent with the committer's `parked_on` store and `done` load, so either it sees this
        // store or this thread sees it parked
        slot.done.store(i + 1, Ordering::SeqCst);
        if self.parked_on.load(Ordering::SeqCst) == i + 1 {
            committer.unpark();
        }
    }

    /// The committer's view of seed `i`, the next to commit: its result traced ahead, or the seed claimed
    fn ready(&self, i: usize) -> Result<Ready> {
        let slot = &self.slots[i % self.window];
        let mut spins = 0;
        loop {
            if slot.done.load(Ordering::Acquire) == i + 1 {
                return Ok(Ready::Ahead);
            }
            if self.next.load(Ordering::Relaxed) == i
                && self.next.compare_exchange(i, i + 1, Ordering::Relaxed, Ordering::Relaxed).is_ok()
            {
                return Ok(Ready::Claimed);
            }
            if self.panicked.load(Ordering::SeqCst) {
                bail!("a tracing thread panicked");
            }
            if spins < SPINS {
                spins += 1;
                std::hint::spin_loop();
                continue;
            }
            self.parked_on.store(i + 1, Ordering::SeqCst);
            if slot.done.load(Ordering::SeqCst) != i + 1 && !self.panicked.load(Ordering::SeqCst) {
                std::thread::park();
            }
            self.parked_on.store(0, Ordering::Relaxed);
        }
    }

    /// The committer: commit every seed in order. `top_up` is called after each commit to start a helper when
    /// fewer are running than the pool has threads
    fn commit_all(
        &self,
        deliver: &mut impl FnMut(usize, &mut Vec<RawCurve>) -> Result<Flow>,
        mut top_up: impl FnMut(),
    ) -> Result<Stats> {
        let total = self.cat.coords.len();
        let mut writer = self.cov.writer();
        let mut scratch = Scratch::default();
        let mut own = Traced::default();
        let mut stats = Stats { seeds: total, ..Stats::default() };
        top_up();
        for i in 0..total {
            let traced = match self.ready(i)? {
                Ready::Claimed => {
                    stats.committer += 1;
                    self.trace(i, &mut scratch, &mut own)?;
                    &mut own
                }
                Ready::Ahead => {
                    // SAFETY: the helper published the slot's result, and the slot is claimed again only after
                    // the release of `committed` below
                    let ahead = unsafe { &mut *self.slots[i % self.window].ahead.get() };
                    if self.retrace_all || writer.stale(&ahead.traced.reads, ahead.committed as u32) {
                        stats.retraced += 1;
                        // the slot is traced into again only after its next claim; until then it would hold this
                        // seed's curves and room for marks beyond what a kept result may
                        ahead.traced.discard();
                        self.trace(i, &mut scratch, &mut own)?;
                        &mut own
                    } else {
                        // nothing it read has changed since, so this is the serial result, an error included
                        std::mem::replace(&mut ahead.result, Ok(()))?;
                        &mut ahead.traced
                    }
                }
            };
            writer.apply(i as u32, traced);
            let flow = deliver(i, &mut traced.curves)?;
            traced.trim();
            // release: publishes the marks just applied and hands the slot back
            self.committed.store(i + 1, Ordering::Release);
            if let Flow::Stop = flow {
                return Ok(stats);
            }
            top_up();
        }
        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fields::{Presence, catalogue, centrality};
    use crate::kernels::{encode_dir, pack_dir};
    use crate::stage1::Separate;
    use crate::vol::Vol;

    const SHAPE: [usize; 3] = [40, 96, 96];

    /// Pooled predictions and a direction field of tilted fibres a few voxels apart, which come close enough
    /// for seeds to read bricks that other seeds' curves mark
    fn fibres() -> ([Vol<u8>; 2], Vol<u16>) {
        let mut own = Vol::new(SHAPE, 0u8);
        let mut dirs = Vol::new(SHAPE, pack_dir(128, 128));
        // each fibre runs from (y0, x0) at z = 0 with slope (dy/dz, dx/dz)
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

    /// Every seed traced and committed in order on one thread: the serial tracer
    fn serial<P: Grid<u8>, C: Grid<u16>, D: Grid<u16>>(f: &Fields<P, C, D>, cat: &Catalogue) -> Vec<Vec<[f32; 3]>> {
        let cov = Coverage::new(SHAPE);
        let mut writer = cov.writer();
        let (mut scratch, mut traced, mut out) = (Scratch::default(), Traced::default(), Vec::new());
        for r in 0..cat.starts.len() - 1 {
            let primary_end = cat.starts[r] + cat.primary[r];
            for (i, &seed) in cat.coords.iter().enumerate().take(cat.starts[r + 1]).skip(cat.starts[r]) {
                trace_seed(f, &cov, &mut scratch, seed, i < primary_end, &mut traced).unwrap();
                writer.apply(i as u32, &traced);
                out.extend(traced.curves.drain(..).map(|c| c.pts));
            }
        }
        out
    }

    fn speculative<P: Grid<u8>, C: Grid<u16>, D: Grid<u16>>(
        f: &Fields<P, C, D>,
        cat: &Catalogue,
        spec: Speculation,
    ) -> (Vec<Vec<[f32; 3]>>, Stats) {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        let cov = Coverage::new(SHAPE);
        let producer = Producer::new(f, &cov, cat, spec).unwrap();
        let mut out = Vec::new();
        let stats = producer
            .run(&pool, |_, curves| {
                out.extend(curves.drain(..).map(|c| c.pts));
                Ok(Flow::Continue)
            })
            .unwrap();
        (out, stats)
    }

    #[test]
    fn tracing_ahead_commits_the_serial_curves() {
        let (tracing, dirs) = fibres();
        let pres = Presence::new(&tracing, 1);
        let fam = centrality::<Separate>(&pres, &dirs);
        let cat = catalogue(&fam.seeds, &fam.cent);
        let fields = Fields { pres: &pres, cent: &fam.cent, dirs: &dirs };
        let expected = serial(&fields, &cat);
        assert!(expected.len() >= 100, "the fixture should accept many curves, not {}", expected.len());
        let seeds = cat.coords.len();

        let (curves, stats) = speculative(&fields, &cat, Speculation { window: WINDOW, retrace_all: false });
        assert!(curves == expected, "window {WINDOW}: {} curves, {} expected", curves.len(), expected.len());
        assert_eq!(stats.seeds, seeds);
        // the fibres are close enough that some seeds traced ahead read a brick an earlier commit then marked
        assert!(stats.retraced > 0, "no seed was traced again, so nothing was checked: {stats:?}");

        // a window of one only ever traces the seed at the commit point, so nothing is ever stale
        let (curves, stats) = speculative(&fields, &cat, Speculation { window: 1, retrace_all: false });
        assert!(curves == expected, "window 1: {} curves, {} expected", curves.len(), expected.len());
        assert_eq!(stats.retraced, 0);

        // every seed traced ahead is traced again on the committer, so every seed is traced serially
        let (curves, stats) = speculative(&fields, &cat, Speculation { window: WINDOW, retrace_all: true });
        assert!(curves == expected, "all retraced: {} curves, {} expected", curves.len(), expected.len());
        assert_eq!(stats.retraced + stats.committer, seeds);
    }
}
