//! Reads ahead for the device CT cache: CT chunks the next launch will likely need, read on background threads
//! into pinned buffers while the current launch's kernels run, so [`super::ChunkCache::prepare`] uploads them
//! instead of waiting on the disk
//!
//! A read-ahead is only a guess: a chunk read for a launch that does not come stays held until a later read-ahead
//! needs its buffer, and a failed read is forgotten, so the launch that needs the chunk reads it again itself and
//! fails on its own error. The bytes are the chunk's either way, so a launch's crops do not depend on what was read
//! ahead

use std::collections::HashMap;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use color_eyre::eyre::{Result, eyre};
use cudarc::driver::sys::{CUevent, CUresult};
use cudarc::driver::{CudaContext, DriverError, result};

use super::{CHUNK_BYTES, Pinned};
use crate::strips::ct::Volume;

/// A pinned buffer's address and the event of its last queued upload, handed to the reader that owns the buffer
/// until it reports back
struct BufRef {
    ptr: *mut u8,
    done: CUevent,
}

// SAFETY: a buffer is reached by one side at a time: the reader from the job's send until its result is received;
// the event outlives the reader's use of it (the prefetcher waits for every read before freeing its buffers) and is
// not recorded again until the buffer comes back
unsafe impl Send for BufRef {}

impl BufRef {
    /// Waits for the buffer's last upload: it can still be queued behind the copy stream's wait on the previous
    /// launch's crops, and a reader waiting here keeps that wait off the engine thread. Polls with short sleeps,
    /// since a default event's synchronize spins a core
    fn idle(&self) -> Result<(), DriverError> {
        loop {
            // SAFETY: a live event, see the Send impl
            match unsafe { result::event::query(self.done) } {
                Ok(()) => return Ok(()),
                Err(DriverError(CUresult::CUDA_ERROR_NOT_READY)) => std::thread::sleep(Duration::from_micros(20)),
                Err(e) => return Err(e),
            }
        }
    }
}

/// One chunk to read: chunk-grid zyx `c` into buffer `buf`
struct Job {
    buf: usize,
    c: [usize; 3],
    to: BufRef,
}

/// A finished read: whether the source holds the chunk, or the read's error
struct Done {
    buf: usize,
    result: Result<bool, String>,
}

/// What a buffer holds; `generation` is the read-ahead that asked for it
#[derive(Clone, Copy)]
enum State {
    Free,
    Reading { idx: u32, generation: u64 },
    Ready { idx: u32, generation: u64, present: bool },
}

/// Background chunk readers and the pinned buffers they read into
pub(super) struct Prefetcher {
    jobs: Option<Sender<Job>>,
    results: Receiver<Done>,
    readers: Vec<JoinHandle<()>>,
    bufs: Vec<Pinned<u8>>,
    state: Vec<State>,
    /// Chunk table index to the buffer reading or holding it
    held: HashMap<u32, usize>,
    generation: u64,
}

impl Prefetcher {
    /// `buffers` pinned chunk buffers and `readers` threads reading `vol`
    pub(super) fn new(ctx: &Arc<CudaContext>, vol: Arc<Volume>, buffers: usize, readers: usize) -> Result<Self> {
        let bufs = (0..buffers).map(|_| Pinned::new(ctx, CHUNK_BYTES)).collect::<Result<Vec<_>>>()?;
        let (jobs, queue) = channel::<Job>();
        let (done, results) = channel::<Done>();
        let queue = Arc::new(Mutex::new(queue));
        let readers = (0..readers)
            .map(|_| {
                let (queue, done, vol, ctx) = (queue.clone(), done.clone(), vol.clone(), ctx.clone());
                std::thread::spawn(move || {
                    let bound = ctx.bind_to_thread().map_err(|e| format!("bind the CUDA context: {e:?}"));
                    // a closed queue (the prefetcher dropped) ends the reader
                    while let Ok(job) = queue.lock().map_err(drop).and_then(|q| q.recv().map_err(drop)) {
                        let result = bound.clone().and_then(|()| {
                            job.to.idle().map_err(|e| format!("wait for the buffer's last upload: {e:?}"))?;
                            // SAFETY: the buffer is CHUNK_BYTES long, idle, and this reader's until its result is
                            // received
                            let buf = unsafe { std::slice::from_raw_parts_mut(job.to.ptr, CHUNK_BYTES) };
                            vol.read_chunk(job.c, buf).map_err(|e| format!("{e:#}"))
                        });
                        if done.send(Done { buf: job.buf, result }).is_err() {
                            return;
                        }
                    }
                })
            })
            .collect();
        Ok(Self {
            jobs: Some(jobs),
            results,
            readers,
            state: vec![State::Free; bufs.len()],
            bufs,
            held: HashMap::new(),
            generation: 0,
        })
    }

    /// Starts reading the chunks `missing` (table index, chunk-grid zyx) not already read or being read, while
    /// buffers last: free ones first, then those holding chunks of earlier read-aheads
    pub(super) fn start(&mut self, missing: &[(u32, [usize; 3])]) -> Result<()> {
        self.generation += 1;
        self.collect();
        for &(idx, c) in missing {
            if self.held.contains_key(&idx) {
                continue;
            }
            let Some(buf) = self.claim() else {
                break;
            };
            // the reader, not this thread, waits for the upload last queued from this buffer
            let to = BufRef { ptr: self.bufs[buf].ptr, done: self.bufs[buf].done.cu_event() };
            let jobs = self.jobs.as_ref().ok_or_else(|| eyre!("the chunk readers have stopped"))?;
            jobs.send(Job { buf, c, to }).map_err(|_| eyre!("the chunk readers have stopped"))?;
            self.state[buf] = State::Reading { idx, generation: self.generation };
            self.held.insert(idx, buf);
        }
        Ok(())
    }

    /// The buffer read ahead for chunk `idx` and whether the source holds it, waiting for its read; `None` when it
    /// was not read ahead or its read failed. The buffer is free again once its upload completes
    pub(super) fn take(&mut self, idx: u32) -> Option<(usize, bool)> {
        let &buf = self.held.get(&idx)?;
        while matches!(self.state[buf], State::Reading { .. }) {
            // the readers outlive every buffer in Reading, so the channel stays open while one is
            let Ok(done) = self.results.recv() else {
                break;
            };
            self.finish(done);
        }
        // a failed read has already left `held`
        let State::Ready { present, .. } = self.state[buf] else {
            return None;
        };
        self.held.remove(&idx);
        self.state[buf] = State::Free;
        Some((buf, present))
    }

    /// Records every read finished so far, without waiting
    pub(super) fn collect(&mut self) {
        while let Ok(done) = self.results.try_recv() {
            self.finish(done);
        }
    }

    /// Whether chunk `idx` was read ahead and its read has finished, as of the last [`Prefetcher::collect`]
    pub(super) fn ready(&self, idx: u32) -> bool {
        self.held.get(&idx).is_some_and(|&buf| matches!(self.state[buf], State::Ready { .. }))
    }

    /// Buffer `buf`, for the upload of a chunk [`Prefetcher::take`] returned
    pub(super) fn buf_mut(&mut self, buf: usize) -> &mut Pinned<u8> {
        &mut self.bufs[buf]
    }

    /// Records a finished read; a failed one frees its buffer and forgets the chunk
    fn finish(&mut self, done: Done) {
        let State::Reading { idx, generation } = self.state[done.buf] else {
            return;
        };
        match done.result {
            Ok(present) => self.state[done.buf] = State::Ready { idx, generation, present },
            Err(_) => {
                self.state[done.buf] = State::Free;
                self.held.remove(&idx);
            }
        }
    }

    /// A free buffer, or the one holding the oldest finished read of an earlier read-ahead
    fn claim(&mut self) -> Option<usize> {
        if let Some(buf) = self.state.iter().position(|s| matches!(s, State::Free)) {
            return Some(buf);
        }
        let (buf, idx) = self
            .state
            .iter()
            .enumerate()
            .filter_map(|(buf, s)| match *s {
                State::Ready { idx, generation, .. } if generation < self.generation => Some((generation, buf, idx)),
                _ => None,
            })
            .min()
            .map(|(_, buf, idx)| (buf, idx))?;
        self.held.remove(&idx);
        self.state[buf] = State::Free;
        Some(buf)
    }
}

impl Drop for Prefetcher {
    fn drop(&mut self) {
        // every read in flight writes into a buffer freed below: wait for them, then stop the readers
        while self.state.iter().any(|s| matches!(s, State::Reading { .. })) {
            let Ok(done) = self.results.recv() else {
                break;
            };
            self.finish(done);
        }
        self.jobs = None;
        for reader in self.readers.drain(..) {
            let _ = reader.join();
        }
    }
}
