//! The GPU half of `fibrs strips`: one thread owns the device and runs the frame model and the strip model on
//! decisions batched across fibres
//!
//! CPU workers hand the engine a fibre's frame requests (and later its strip requests) as one [`Work`] item with a
//! reply channel, and block on the reply. The engine thread drains its queue, prefers frame work (cheap, and it
//! unblocks workers whose strip requests then fill the next strip batch), and runs strip work in batches of up to
//! the engine's batch size, splitting items across batches and sending each item's results once all of its
//! decisions are done
//!
//! Frame requests mostly arrive while a launch runs, too late for its read-ahead, so a frame launch would often wait
//! on the disk for its CT chunks with the GPU idle. When strips are queued, the engine starts those reads and runs
//! one strip launch first; the frames follow it with their chunks read. The model's outputs do not depend on how
//! decisions are grouped into launches, so the order changes no result

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::time::Instant;

use color_eyre::eyre::{Result, WrapErr, ensure};

use super::ct::{Uncached, Volume};
use super::gpu::cnn::CropCnn;
use super::gpu::crops::{CacheCounts, ChunkCache, CropAt, CropSpec, Crops, FRAME_CROP, STRIP_CROP};
use super::gpu::frame::FrameModel;
use super::gpu::head::Head;
use super::gpu::{Device, Weights};
use super::refset::CROP_LEN;
use super::types::{FrameRequest, FrameResult, StripRequest, StripResult};

/// The engine's configuration
#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// CUDA device ordinal
    pub device: usize,
    /// Decisions per GPU launch
    pub batch: usize,
    /// CT chunk slots on the device (2 MiB each)
    pub chunk_slots: usize,
    /// Pinned staging buffers for chunk uploads
    pub staging: usize,
    /// CT voxels per trace voxel (villa `input_scale`)
    pub input_scale: f64,
}

/// A fibre's requests for one model and the channel its results go back on, in request order
pub enum Work {
    Frames(Vec<FrameRequest>, Sender<Result<Vec<FrameResult>, LaunchError>>),
    Strips(Vec<StripRequest>, Sender<Result<Vec<StripResult>, LaunchError>>),
}

/// Why a launch failed, as each item it touched hears it
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LaunchError {
    /// Its crops need CT chunks the disk cache does not hold; nothing else went wrong, so its fibres can run once
    /// they are fetched
    Uncached(Vec<[usize; 3]>),
    /// Anything else, with its error chain
    Failed(String),
}

impl LaunchError {
    fn new(e: &color_eyre::eyre::Report) -> Self {
        match e.downcast_ref::<Uncached>() {
            Some(Uncached(chunks)) => Self::Uncached(chunks.clone()),
            None => Self::Failed(format!("{e:#}")),
        }
    }
}

impl std::fmt::Display for LaunchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Uncached(chunks) => Uncached(chunks.clone()).fmt(f),
            Self::Failed(message) => f.write_str(message),
        }
    }
}

/// What the engine did, for the run's report
#[derive(Clone, Copy, Debug, Default)]
pub struct EngineCounts {
    pub frame_decisions: usize,
    pub frame_launches: usize,
    pub strip_decisions: usize,
    pub strip_launches: usize,
    /// Strip launches run ahead of queued frames whose CT chunks were still being read
    pub frame_deferrals: usize,
    /// Seconds the engine spent running work (the rest it waited for work)
    pub busy: f64,
    /// The device CT cache's uploads, evictions and fills
    pub cache: CacheCounts,
}

/// The device and every model and buffer on it
struct Engine {
    dev: Device,
    cache: ChunkCache,
    crops: Crops,
    frame: FrameModel,
    cnn: CropCnn,
    head: Head,
    frame_crops: volkit::gpu::Buf,
    strip_input: volkit::gpu::Buf,
    batch: usize,
}

impl Engine {
    fn new(config: &EngineConfig, weights: &Weights, vol: Arc<Volume>) -> Result<Self> {
        ensure!(config.batch > 0, "engine batch 0");
        let dev = Device::open(config.device)?;
        let b = config.batch;
        let cache = ChunkCache::new(&dev, vol, config.chunk_slots, config.staging).wrap_err("the CT cache")?;
        let crops = Crops::new(&dev, b, config.input_scale)?;
        let frame = FrameModel::new(&dev, weights, b).wrap_err("the frame model")?;
        let mut cnn = CropCnn::new(&dev, weights, b).wrap_err("the CNN")?;
        cnn.prepare(&dev, b)?;
        let head = Head::new(&dev, weights, b).wrap_err("the head")?;
        let frame_crops = dev.gpu.alloc(b * FRAME_CROP.samples() * 4)?;
        let strip_input = dev.gpu.alloc(b * CROP_LEN * 4)?;
        Ok(Self { dev, cache, crops, frame, cnn, head, frame_crops, strip_input, batch: b })
    }

    /// The frame model on up to `batch` requests, reading `ahead`'s chunks while its kernels run
    fn frames(&mut self, requests: &[FrameRequest], ahead: &Ahead) -> Result<Vec<FrameResult>> {
        let at: Vec<CropAt> = requests.iter().map(|r| CropAt { pos: r.pos, frame: r.frame }).collect();
        self.crops.frame(&self.dev, &mut self.cache, &at, &self.frame_crops)?;
        self.crops.prefetch(&mut self.cache, &ahead.reads())?;
        self.frame.run(&self.dev, &self.frame_crops, requests)
    }

    /// The strip model on up to `batch` requests, reading `ahead`'s chunks while its kernels run
    fn strips(&mut self, requests: &[StripRequest], ahead: &Ahead) -> Result<Vec<StripResult>> {
        let at: Vec<CropAt> = requests.iter().map(|r| CropAt { pos: r.pos, frame: r.frame }).collect();
        let tubes: Vec<&[[f32; 3]]> = requests.iter().map(|r| r.tube.as_slice()).collect();
        self.crops.strip(&self.dev, &mut self.cache, &at, &tubes, &self.strip_input)?;
        self.crops.prefetch(&mut self.cache, &ahead.reads())?;
        let input = self.crops.strip_input(&self.strip_input);
        let cells = self.cnn.forward(&self.dev, input, requests.len())?;
        self.head.forward(&self.dev, cells, input, requests)
    }
}

/// The crops the launches after this one will likely sample, the next launch's first: a guess from the queues as
/// they stand, whose CT chunks are read while this launch runs
struct Ahead(Vec<(&'static CropSpec, Vec<CropAt>)>);

impl Ahead {
    fn one(spec: &'static CropSpec, at: Vec<CropAt>) -> Self {
        Self(vec![(spec, at)])
    }

    fn reads(&self) -> Vec<(&CropSpec, &[CropAt])> {
        self.0.iter().map(|(spec, at)| (*spec, at.as_slice())).collect()
    }
}

/// An item partly run: its requests, the results so far, where its reply goes
struct Pending<Q, R> {
    requests: Vec<Q>,
    done: Vec<R>,
    reply: Sender<Result<Vec<R>, LaunchError>>,
}

/// A queue of pending items of one model, cut into launches of up to `batch` decisions across items
struct Queue<Q, R> {
    items: VecDeque<Pending<Q, R>>,
}

impl<Q: Clone, R> Queue<Q, R> {
    fn new() -> Self {
        Self { items: VecDeque::new() }
    }

    /// Decisions not yet run
    fn decisions(&self) -> usize {
        self.items.iter().map(|item| item.requests.len() - item.done.len()).sum()
    }

    /// `f` of up to `n` decisions not yet run, after the first `skip` of them, in launch order
    fn peek<T>(&self, skip: usize, n: usize, f: impl Fn(&Q) -> T) -> Vec<T> {
        self.items.iter().flat_map(|item| &item.requests[item.done.len()..]).skip(skip).take(n).map(f).collect()
    }

    fn push(&mut self, requests: Vec<Q>, reply: Sender<Result<Vec<R>, LaunchError>>) {
        if requests.is_empty() {
            let _ = reply.send(Ok(Vec::new()));
            return;
        }
        self.items.push_back(Pending { done: Vec::with_capacity(requests.len()), requests, reply });
    }

    /// Runs one launch of up to `batch` decisions from the front items with `run` and replies to every item it
    /// completes; a failed launch fails (and drops) every item it touched. The decisions launched
    fn launch(&mut self, batch: usize, run: impl FnOnce(&[Q]) -> Result<Vec<R>>) -> usize {
        // (items touched, decisions from the last one): every item but the last is taken whole
        let mut requests = Vec::with_capacity(batch);
        let mut touched = 0;
        let mut last = 0;
        for item in &self.items {
            let start = item.done.len();
            last = (item.requests.len() - start).min(batch - requests.len());
            requests.extend_from_slice(&item.requests[start..start + last]);
            touched += 1;
            if requests.len() == batch {
                break;
            }
        }

        let n = requests.len();
        match run(&requests) {
            Ok(results) => {
                let mut results = results.into_iter();
                for (k, item) in self.items.iter_mut().take(touched).enumerate() {
                    let count = if k + 1 == touched { last } else { item.requests.len() - item.done.len() };
                    item.done.extend(results.by_ref().take(count));
                }
                while self.items.front().is_some_and(|item| item.done.len() == item.requests.len()) {
                    let item = self.items.pop_front().unwrap();
                    let _ = item.reply.send(Ok(item.done));
                }
            }
            Err(e) => {
                let error = LaunchError::new(&e);
                for item in self.items.drain(..touched) {
                    let _ = item.reply.send(Err(error.clone()));
                }
            }
        }
        n
    }
}

/// Runs the engine until every sender of `work` has dropped and the queues are empty; `ready` gets the outcome of
/// building it (the device, the models, the cache) before any work runs
pub fn serve(
    config: &EngineConfig,
    weights: &Weights,
    vol: Arc<Volume>,
    work: Receiver<Work>,
    ready: Sender<Result<(), String>>,
) -> Result<EngineCounts> {
    let mut engine = match Engine::new(config, weights, vol) {
        Ok(engine) => {
            let _ = ready.send(Ok(()));
            engine
        }
        Err(e) => {
            let _ = ready.send(Err(format!("{e:#}")));
            return Err(e);
        }
    };

    let mut frames: Queue<FrameRequest, FrameResult> = Queue::new();
    let mut strips: Queue<StripRequest, StripResult> = Queue::new();
    let mut counts = EngineCounts::default();
    let mut open = true;
    // the frames were put off for one strip launch: they run next whatever their reads
    let mut deferred = false;
    loop {
        // take everything queued; block only when there is nothing to run
        loop {
            let next = if frames.decisions() + strips.decisions() == 0 && open {
                work.recv().map_err(|_| TryRecvError::Disconnected)
            } else {
                work.try_recv()
            };
            match next {
                Ok(Work::Frames(requests, reply)) => frames.push(requests, reply),
                Ok(Work::Strips(requests, reply)) => strips.push(requests, reply),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    open = false;
                    break;
                }
            }
        }
        if frames.decisions() + strips.decisions() == 0 {
            if open {
                continue;
            }
            counts.cache = engine.cache.counts;
            return Ok(counts);
        }

        let t = Instant::now();
        let batch = engine.batch;
        let frame_at = |r: &FrameRequest| CropAt { pos: r.pos, frame: r.frame };
        let strip_at = |r: &StripRequest| CropAt { pos: r.pos, frame: r.frame };
        // frames whose chunks are still unread wait for one strip launch, which reads them meanwhile; a failed read
        // counts as read, and the frame launch reads the chunk itself and fails on its own error
        let front = frames.peek(0, batch, frame_at);
        let defer = !deferred
            && !front.is_empty()
            && strips.decisions() > 0
            && engine.crops.unread(&mut engine.cache, &FRAME_CROP, &front).unwrap_or(0) > 0;
        deferred = defer;
        if defer {
            // the frames are the next launch, so they are read first; the strips after them come next, since the
            // launch after the frames is often a strip launch, which would otherwise wait on its own reads
            let ahead = Ahead(vec![(&FRAME_CROP, front), (&STRIP_CROP, strips.peek(batch, batch, strip_at))]);
            counts.strip_decisions += strips.launch(batch, |q| engine.strips(q, &ahead));
            counts.strip_launches += 1;
            counts.frame_deferrals += 1;
        } else if frames.decisions() > 0 {
            // the next launch runs frames while any are queued, so it takes the frames beyond this launch, else the
            // front strips; work that arrives meanwhile can change it, which costs only the reads
            let ahead = if frames.decisions() > batch {
                Ahead::one(&FRAME_CROP, frames.peek(batch, batch, frame_at))
            } else {
                Ahead::one(&STRIP_CROP, strips.peek(0, batch, strip_at))
            };
            counts.frame_decisions += frames.launch(batch, |q| engine.frames(q, &ahead));
            counts.frame_launches += 1;
        } else {
            let ahead = Ahead::one(&STRIP_CROP, strips.peek(batch, batch, strip_at));
            counts.strip_decisions += strips.launch(batch, |q| engine.strips(q, &ahead));
            counts.strip_launches += 1;
        }
        counts.busy += t.elapsed().as_secs_f64();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc::channel;

    use color_eyre::eyre::eyre;

    use super::*;

    /// Items of 3, 5 and 2 decisions through launches of 4: every item gets its own results back in order, once,
    /// after the launch that finishes it; a failed launch fails only the items it touched
    #[test]
    fn queue_splits_items_across_launches_and_replies_in_order() {
        let mut q: Queue<u32, u32> = Queue::new();
        let replies: Vec<_> = [3u32, 5, 2]
            .iter()
            .scan(0u32, |next, &len| {
                let (tx, rx) = channel();
                q.push((*next..*next + len).collect(), tx);
                *next += len;
                Some(rx)
            })
            .collect();
        assert_eq!(q.decisions(), 10);
        // the launch after a launch of 4 starts at decision 4, across the item boundary
        assert_eq!(q.peek(4, 4, |v| v * 10), vec![40, 50, 60, 70]);
        assert_eq!(q.peek(8, 4, |&v| v), vec![8, 9]);

        let mut launched = Vec::new();
        while q.decisions() > 0 {
            q.launch(4, |batch| {
                launched.push(batch.to_vec());
                Ok(batch.iter().map(|v| v * 10).collect())
            });
        }
        assert_eq!(launched, vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![8, 9]]);
        assert!(q.peek(0, 4, |&v| v).is_empty());
        assert_eq!(replies[0].try_recv().unwrap().unwrap(), vec![0, 10, 20]);
        assert_eq!(replies[1].try_recv().unwrap().unwrap(), vec![30, 40, 50, 60, 70]);
        assert_eq!(replies[2].try_recv().unwrap().unwrap(), vec![80, 90]);
        assert!(replies.iter().all(|r| r.try_recv().is_err()));

        // a failure in the second launch fails items 1 and 2 (both touched), and item 0 completed in the first
        let mut q: Queue<u32, u32> = Queue::new();
        let rx: Vec<_> = [(0..3).collect::<Vec<u32>>(), (3..6).collect(), (6..9).collect()]
            .into_iter()
            .map(|items| {
                let (tx, rx) = channel();
                q.push(items, tx);
                rx
            })
            .collect();
        q.launch(4, |b| Ok(b.to_vec()));
        q.launch(4, |_| Err(eyre!("device lost")));
        assert_eq!(rx[0].try_recv().unwrap().unwrap(), vec![0, 1, 2]);
        assert_eq!(rx[1].try_recv().unwrap().unwrap_err(), LaunchError::Failed("device lost".into()));
        assert_eq!(rx[2].try_recv().unwrap().unwrap_err(), LaunchError::Failed("device lost".into()));
        assert_eq!(q.decisions(), 0);

        // a launch that needs uncached chunks tells each item it touched which, even under added context
        let (tx, rx) = channel();
        q.push(vec![1, 2], tx);
        q.launch(4, |_| Err(color_eyre::eyre::Report::new(Uncached(vec![[1, 2, 3]])).wrap_err("crops")));
        assert_eq!(rx.try_recv().unwrap().unwrap_err(), LaunchError::Uncached(vec![[1, 2, 3]]));
    }
}
