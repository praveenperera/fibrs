//! The tile feed: every column's tiles read in the reference order on `--readers` threads, running ahead of the
//! GPU across columns, and grouped into the reference batches
//!
//! The feed drops the tiles that do not run and groups the rest `batch` at a time within a column, with the
//! column's remainder as its last, smaller batch. The batches are the same tile groups at the same sizes as the
//! band reference's, so the network sees exactly the inputs it sees there. Reads finish in any order; the feed
//! takes them in tile order, so the batches never depend on thread timing

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use color_eyre::eyre::{Result, eyre};

use super::contract::{Origin, TILE_VOXELS, read_tile};
use super::ct::Ct;
use super::villa::VillaSelection;

/// Non-air tiles of one column: their origins and their inputs back to back
pub struct Batch {
    pub col: usize,
    pub origins: Vec<Origin>,
    pub data: Vec<u8>,
}

/// One item of the feed, in order: a column's batches, then its end
pub enum Item {
    Batch(Batch),
    /// Every tile of column `col` has been read; `air` of them did not run
    ColumnEnd {
        col: usize,
        air: usize,
    },
}

/// Which tiles run
pub enum Rule {
    /// Band mode: a tile whose input is all zero is skipped as air, as the band reference skips it
    Air,
    /// Villa mode: exactly the tiles villa runs, whatever their values; the others are skipped unread
    Villa(Arc<VillaSelection>),
}

pub struct FeedPlan {
    pub ct: Arc<Ct>,
    pub ct_path: PathBuf,
    pub z0: usize,
    pub z1: usize,
    /// `(column index, its tiles)` in run order
    pub columns: Vec<(usize, Vec<Origin>)>,
    pub batch: usize,
    pub readers: usize,
    pub rule: Rule,
}

/// The running feed; dropping it stops the reads
pub struct TileFeed {
    rx: Option<Receiver<Result<Item>>>,
    thread: Option<JoinHandle<()>>,
    /// Batch buffers handed back by the consumer
    batches: Arc<Spare>,
}

/// Buffers handed back for reuse. A tile is 2 MiB and a batch several, and a fresh one each time meant zeroing
/// it or growing it by copies, and faulting its pages in; the buffers in flight are bounded, so the list is too
#[derive(Default)]
struct Spare(Mutex<Vec<Vec<u8>>>);

impl Spare {
    fn take(&self) -> Option<Vec<u8>> {
        self.0.lock().ok()?.pop()
    }

    fn give(&self, buf: Vec<u8>) {
        if let Ok(mut spare) = self.0.lock() {
            spare.push(buf);
        }
    }
}

/// Batches and column ends queued ahead of the GPU
const DEPTH: usize = 6;

impl TileFeed {
    pub fn start(plan: FeedPlan) -> Result<Self> {
        if plan.columns.iter().any(|(_, tiles)| tiles.is_empty()) {
            return Err(eyre!("every column has at least one tile"));
        }

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(plan.readers)
            .thread_name(|i| format!("fibrs-predict-read-{i}"))
            .build()?;
        let (tx, rx) = sync_channel(DEPTH);
        let batches = Arc::new(Spare::default());
        let thread = std::thread::Builder::new().name("fibrs-predict-feed".into()).spawn({
            let batches = batches.clone();
            move || {
                let result = feed(&plan, &pool, &tx, &batches);
                if let Err(e) = result {
                    let _ = tx.send(Err(e));
                }
            }
        })?;
        Ok(Self { rx: Some(rx), thread: Some(thread), batches })
    }

    /// Hands a batch's buffer back once its tiles are used, for a later batch to fill
    pub fn recycle(&self, data: Vec<u8>) {
        self.batches.give(data);
    }

    /// The next item, in order
    pub fn next(&self) -> Result<Item> {
        let rx = self.rx.as_ref().ok_or_else(|| eyre!("tile feed closed"))?;
        match rx.recv() {
            Ok(Ok(item)) => Ok(item),
            Ok(Err(e)) => Err(e.wrap_err("tile feed failed")),
            Err(_) => Err(eyre!("tile feed ended before the column did")),
        }
    }
}

impl Drop for TileFeed {
    fn drop(&mut self) {
        // closing the receiver fails the feed's next send, which ends it
        drop(self.rx.take());
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// A tile's read in flight, or `None` for a tile skipped unread
type Pending = (usize, Origin, Option<Receiver<Result<Vec<u8>>>>);

/// The feed thread: returns when every column is fed or the consumer has gone. Batch buffers come back through
/// `batches`, and tile buffers to the readers once their tile is copied into a batch
fn feed(plan: &FeedPlan, pool: &rayon::ThreadPool, tx: &SyncSender<Result<Item>>, batches: &Spare) -> Result<()> {
    let tiles = Arc::new(Spare::default());
    let fresh_batch = || {
        let mut data = batches.take().unwrap_or_default();
        data.clear();
        data.reserve(plan.batch * TILE_VOXELS);
        data
    };
    // tiles of every column in one ordered stream, `lookahead` reads in flight
    let lookahead = 4 * plan.readers;
    let mut order = plan.columns.iter().flat_map(|(ci, tiles)| tiles.iter().map(move |t| (*ci, *t))).peekable();
    let mut window: VecDeque<Pending> = VecDeque::new();
    let mut seen = 0usize;
    let mut air = 0usize;
    let mut origins = Vec::new();
    let mut data = fresh_batch();
    let mut column = plan.columns.iter().map(|(ci, tiles)| (*ci, tiles.len()));
    let (mut ci, mut ends) = column.next().ok_or_else(|| eyre!("no columns"))?;

    loop {
        while window.len() < lookahead {
            let Some((c, t)) = order.next() else { break };
            window.push_back((c, t, read(plan, pool, t, &tiles)));
        }
        let Some((c, t, rx)) = window.pop_front() else { break };
        debug_assert_eq!(c, ci);
        let raw = rx.map(|r| r.recv().map_err(|_| eyre!("tile reader stopped"))?).transpose()?;
        seen += 1;
        match raw {
            Some(raw) if matches!(plan.rule, Rule::Villa(_)) || raw.iter().any(|&v| v != 0) => {
                origins.push(t);
                data.extend_from_slice(&raw);
                tiles.give(raw);
            }
            Some(raw) => {
                air += 1;
                tiles.give(raw);
            }
            None => air += 1,
        }

        if origins.len() == plan.batch || (seen == ends && !origins.is_empty()) {
            let batch = Batch {
                col: ci,
                origins: std::mem::take(&mut origins),
                data: std::mem::replace(&mut data, fresh_batch()),
            };
            if tx.send(Ok(Item::Batch(batch))).is_err() {
                return Ok(());
            }
        }
        if seen == ends {
            if tx.send(Ok(Item::ColumnEnd { col: ci, air })).is_err() {
                return Ok(());
            }
            (seen, air) = (0, 0);
            match column.next() {
                Some(next) => (ci, ends) = next,
                None => break,
            }
        }
    }
    Ok(())
}

/// Starts reading tile `t` on the pool into a spare buffer, unless the rule skips it unread
fn read(plan: &FeedPlan, pool: &rayon::ThreadPool, t: Origin, tiles: &Arc<Spare>) -> Option<Receiver<Result<Vec<u8>>>> {
    if let Rule::Villa(sel) = &plan.rule
        && !sel.tile_runs(t)
    {
        return None;
    }

    let (tx, rx) = sync_channel(1);
    let (ct, z0, z1) = (plan.ct.clone(), plan.z0, plan.z1);
    let path = plan.ct_path.clone();
    let tiles = tiles.clone();
    pool.spawn(move || {
        // read_tile writes every byte, so a spare buffer's old tile does not matter
        let mut out = tiles.take().unwrap_or_else(|| vec![0u8; TILE_VOXELS]);
        let r =
            read_tile(&ct, z0, z1, t, &mut out).map(|()| out).map_err(|e| e.wrap_err(format!("{}", path.display())));
        let _ = tx.send(r);
    });
    Some(rx)
}
