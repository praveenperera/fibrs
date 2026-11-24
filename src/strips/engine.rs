//! The GPU half of `fibrs strips`: one thread owns the device and runs the frame model and the strip model on
//! decisions batched across fibres
//!
//! CPU workers hand the engine a fibre's frame requests (and later its strip requests) as one [`Work`] item with a
//! reply channel, and block on the reply. The engine thread drains its queue, prefers frame work (cheap, and it
//! unblocks workers whose strip requests then fill the next strip batch), and runs strip work in batches of up to
//! the engine's batch size, splitting items across batches and sending each item's results once all of its
//! decisions are done

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::time::Instant;

use color_eyre::eyre::{Result, WrapErr, ensure};

use super::ct::Volume;
use super::gpu::cnn::CropCnn;
use super::gpu::crops::{ChunkCache, CropAt, Crops, FRAME_CROP};
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
    /// The weights export
    pub weights: PathBuf,
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
    Frames(Vec<FrameRequest>, Sender<Result<Vec<FrameResult>, String>>),
    Strips(Vec<StripRequest>, Sender<Result<Vec<StripResult>, String>>),
}

/// What the engine did, for the run's report
#[derive(Clone, Copy, Debug, Default)]
pub struct EngineCounts {
    pub frame_decisions: usize,
    pub frame_launches: usize,
    pub strip_decisions: usize,
    pub strip_launches: usize,
    /// Seconds the engine spent running work (the rest it waited for work)
    pub busy: f64,
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
    fn new(config: &EngineConfig, vol: Arc<Volume>) -> Result<Self> {
        ensure!(config.batch > 0, "engine batch 0");
        let weights = Weights::open(&config.weights)?;
        let dev = Device::open(config.device)?;
        let b = config.batch;
        let cache = ChunkCache::new(&dev, vol, config.chunk_slots, config.staging).wrap_err("the CT cache")?;
        let crops = Crops::new(&dev, b, config.input_scale)?;
        let frame = FrameModel::new(&dev, &weights, b).wrap_err("the frame model")?;
        let mut cnn = CropCnn::new(&dev, &weights, b).wrap_err("the CNN")?;
        cnn.prepare(&dev, b)?;
        let head = Head::new(&dev, &weights, b).wrap_err("the head")?;
        let frame_crops = dev.gpu.alloc(b * FRAME_CROP.samples() * 4)?;
        let strip_input = dev.gpu.alloc(b * CROP_LEN * 4)?;
        Ok(Self { dev, cache, crops, frame, cnn, head, frame_crops, strip_input, batch: b })
    }

    /// The frame model on up to `batch` requests
    fn frames(&mut self, requests: &[FrameRequest]) -> Result<Vec<FrameResult>> {
        let at: Vec<CropAt> = requests.iter().map(|r| CropAt { pos: r.pos, frame: r.frame }).collect();
        self.crops.frame(&self.dev, &mut self.cache, &at, &self.frame_crops)?;
        self.frame.run(&self.dev, &self.frame_crops, requests)
    }

    /// The strip model on up to `batch` requests
    fn strips(&mut self, requests: &[StripRequest]) -> Result<Vec<StripResult>> {
        let at: Vec<CropAt> = requests.iter().map(|r| CropAt { pos: r.pos, frame: r.frame }).collect();
        let tubes: Vec<&[[f32; 3]]> = requests.iter().map(|r| r.tube.as_slice()).collect();
        self.crops.strip(&self.dev, &mut self.cache, &at, &tubes, &self.strip_input)?;
        let cells = self.cnn.forward(&self.dev, &self.strip_input, requests.len())?;
        self.head.forward(&self.dev, cells, &self.strip_input, requests)
    }
}

/// An item partly run: its requests, the results so far, where its reply goes
struct Pending<Q, R> {
    requests: Vec<Q>,
    done: Vec<R>,
    reply: Sender<Result<Vec<R>, String>>,
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

    fn push(&mut self, requests: Vec<Q>, reply: Sender<Result<Vec<R>, String>>) {
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
                let message = format!("{e:#}");
                for item in self.items.drain(..touched) {
                    let _ = item.reply.send(Err(message.clone()));
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
    vol: Arc<Volume>,
    work: Receiver<Work>,
    ready: Sender<Result<(), String>>,
) -> Result<EngineCounts> {
    let mut engine = match Engine::new(config, vol) {
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
            return Ok(counts);
        }

        let t = Instant::now();
        let batch = engine.batch;
        if frames.decisions() > 0 {
            counts.frame_decisions += frames.launch(batch, |q| engine.frames(q));
            counts.frame_launches += 1;
        } else {
            counts.strip_decisions += strips.launch(batch, |q| engine.strips(q));
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

        let mut launched = Vec::new();
        while q.decisions() > 0 {
            q.launch(4, |batch| {
                launched.push(batch.to_vec());
                Ok(batch.iter().map(|v| v * 10).collect())
            });
        }
        assert_eq!(launched, vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![8, 9]]);
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
        assert_eq!(rx[1].try_recv().unwrap().unwrap_err(), "device lost");
        assert_eq!(rx[2].try_recv().unwrap().unwrap_err(), "device lost");
        assert_eq!(q.decisions(), 0);
    }
}
