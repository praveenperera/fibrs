//! The frame model on the GPU: villa's HeadingNet (`crop_frame_ct_path_family_v4`) in float32, as fast1 runs it
//! with cuDNN TF32 off (`strips-kernels/src/frame.rs`)
//!
//! [`FrameModel::run`] takes the `(B, 32, 32, 32)` crops [`super::crops::Crops::frame`] wrote, each decision's path
//! features and family, and returns each decision's [`FrameResult`]: four conv launches, one head launch and one
//! small download

use color_eyre::eyre::{Result, ensure};
use cudarc::driver::result as cur;
use volkit::gpu::{Arg, Buf};

use super::crops::Pinned;
use super::{Device, Weights};
use crate::strips::types::{FrameRequest, FrameResult};

/// Output channels and positions per conv block, and its threads (strips-kernels `CONV_CO_TILE`,
/// `CONV_POSITIONS`, `CONV_THREADS`)
const CO_TILE: u32 = 16;
const CONV_POSITIONS: u32 = 32;
const CONV_THREADS: u32 = 128;

/// One of layers 2-4: weights `[(tap · ci + c) · co + o]`, bias, and the layer's shape
struct Conv {
    w: Buf,
    b: Buf,
    ci: usize,
    co: usize,
    /// log2 of the input edge
    s_shift: u32,
}

/// The frame model's weights on the device and its activations for up to `capacity` decisions
pub struct FrameModel {
    capacity: usize,
    w1: Buf,
    b1: Buf,
    convs: [Conv; 3],
    wp: Buf,
    bp: Buf,
    emb: Buf,
    head: [Buf; 4],
    normal: [Buf; 4],
    acts: [Buf; 4],
    paths: Buf,
    families: Buf,
    out: Buf,
    host: Pinned<u32>,
    results: Vec<f32>,
}

impl FrameModel {
    /// Uploads the `frame.` tensors of `weights`, with activations for up to `capacity` decisions per run
    pub fn new(dev: &Device, weights: &Weights, capacity: usize) -> Result<Self> {
        let gpu = &dev.gpu;
        let get = |name: &str, shape: &[usize]| -> Result<Buf> {
            gpu.upload(&weights.get(&format!("frame.{name}"), shape)?.values)
        };
        let conv = |idx: usize, ci: usize, co: usize, s_shift: u32| -> Result<Conv> {
            // sf_conv's input copies split k into taps and channels with a shift and a mask
            ensure!(ci.is_power_of_two() && ci >= 16, "frame conv {idx}: {ci} input channels");
            let t = weights.get(&format!("frame.features.{idx}.weight"), &[co, ci, 4, 4, 4])?;
            let mut w = vec![0f32; 64 * ci * co];
            for o in 0..co {
                for c in 0..ci {
                    for tap in 0..64 {
                        w[(tap * ci + c) * co + o] = t.values[(o * ci + c) * 64 + tap];
                    }
                }
            }
            Ok(Conv { w: gpu.upload(&w)?, b: get(&format!("features.{idx}.bias"), &[co])?, ci, co, s_shift })
        };
        let mlp = |name: &str| -> Result<[Buf; 4]> {
            Ok([
                get(&format!("{name}.0.weight"), &[64, 576])?,
                get(&format!("{name}.0.bias"), &[64])?,
                get(&format!("{name}.2.weight"), &[3, 64])?,
                get(&format!("{name}.2.bias"), &[3])?,
            ])
        };
        Ok(Self {
            capacity,
            w1: get("features.0.weight", &[16, 1, 4, 4, 4])?,
            b1: get("features.0.bias", &[16])?,
            convs: [conv(2, 16, 32, 4)?, conv(4, 32, 64, 3)?, conv(6, 64, 64, 2)?],
            wp: get("path.0.weight", &[64, 128])?,
            bp: get("path.0.bias", &[64])?,
            emb: get("family_embedding.weight", &[2, 64])?,
            head: mlp("head")?,
            normal: mlp("normal_head")?,
            acts: [
                gpu.alloc(capacity * 4096 * 16 * 4)?,
                gpu.alloc(capacity * 512 * 32 * 4)?,
                gpu.alloc(capacity * 64 * 64 * 4)?,
                gpu.alloc(capacity * 8 * 64 * 4)?,
            ],
            paths: gpu.alloc(capacity * 128 * 4)?,
            families: gpu.alloc(capacity * 4)?,
            out: gpu.alloc_fill(capacity * 6 * 4, 0xFF)?,
            host: Pinned::new(gpu.inner.context(), capacity * 129)?,
            results: Vec::new(),
        })
    }

    /// Queues the model over the `(B, 32, 32, 32)` `crops` of `requests` (their path features and families),
    /// without waiting; [`FrameModel::results`] reads the outputs
    pub fn launch(&mut self, dev: &Device, crops: &Buf, requests: &[FrameRequest]) -> Result<()> {
        let b = requests.len();
        ensure!(b > 0 && b <= self.capacity, "{b} decisions for a frame model of {}", self.capacity);
        ensure!(crops.bytes() >= b * 32768 * 4, "frame crops of {} bytes for {b} decisions", crops.bytes());
        let gpu = &dev.gpu;
        let stream = gpu.inner.stream().clone();
        let host = self.host.slice_mut()?;
        for (k, r) in requests.iter().enumerate() {
            for (i, v) in r.paths.iter().enumerate() {
                host[k * 128 + i] = v.to_bits();
            }
            host[b * 128 + k] = r.family as u32;
        }
        let ptr = host.as_ptr();
        // SAFETY: the staging buffer stays untouched until `done`, recorded after the copies, completes
        unsafe {
            cur::memcpy_htod_async(self.paths.ptr(), std::slice::from_raw_parts(ptr, b * 128), stream.cu_stream())?;
            cur::memcpy_htod_async(
                self.families.ptr(),
                std::slice::from_raw_parts(ptr.add(b * 128), b),
                stream.cu_stream(),
            )?;
        }
        self.host.done.record(&stream)?;
        // NaN, so a decision the kernels skip fails the finiteness check in `results`
        gpu.fill(&self.out, 0xFF)?;

        let k = &dev.kernels;
        let n1 = (b * 4096) as u32;
        k.launch(
            gpu,
            "sf_conv1",
            (n1.div_ceil(256), 1, 1),
            (256, 1, 1),
            0,
            &[
                Arg::Ptr(crops.ptr()),
                Arg::Ptr(self.w1.ptr()),
                Arg::Ptr(self.b1.ptr()),
                Arg::U32(n1),
                Arg::Ptr(self.acts[0].ptr()),
            ],
        )?;
        for (i, c) in self.convs.iter().enumerate() {
            let out_edge = 1usize << (c.s_shift - 1);
            let n = (b * out_edge * out_edge * out_edge) as u32;
            k.launch(
                gpu,
                "sf_conv",
                (n.div_ceil(CONV_POSITIONS), c.co as u32 / CO_TILE, 1),
                (CONV_THREADS, 1, 1),
                0,
                &[
                    Arg::Ptr(self.acts[i].ptr()),
                    Arg::Ptr(c.w.ptr()),
                    Arg::Ptr(c.b.ptr()),
                    Arg::U32(c.s_shift),
                    Arg::U32(c.ci as u32),
                    Arg::U32(c.co as u32),
                    Arg::U32(n),
                    Arg::Ptr(self.acts[i + 1].ptr()),
                ],
            )?;
        }
        let [w0, b0, w2, b2] = &self.head;
        let [v0, c0, v2, c2] = &self.normal;
        k.launch(
            gpu,
            "sf_head",
            (b as u32, 1, 1),
            (64, 1, 1),
            0,
            &[
                Arg::Ptr(self.acts[3].ptr()),
                Arg::Ptr(self.paths.ptr()),
                Arg::Ptr(self.families.ptr()),
                Arg::Ptr(self.wp.ptr()),
                Arg::Ptr(self.bp.ptr()),
                Arg::Ptr(self.emb.ptr()),
                Arg::Ptr(w0.ptr()),
                Arg::Ptr(b0.ptr()),
                Arg::Ptr(w2.ptr()),
                Arg::Ptr(b2.ptr()),
                Arg::Ptr(v0.ptr()),
                Arg::Ptr(c0.ptr()),
                Arg::Ptr(v2.ptr()),
                Arg::Ptr(c2.ptr()),
                Arg::Ptr(self.out.ptr()),
            ],
        )?;
        Ok(())
    }

    /// Waits for the last launch over `b` decisions and returns their outputs
    pub fn results(&mut self, dev: &Device, b: usize) -> Result<Vec<FrameResult>> {
        ensure!(b <= self.capacity, "{b} results from a frame model of {}", self.capacity);
        dev.gpu.download_range_into(&self.out, 0, 6 * b, &mut self.results)?;
        let out: Vec<FrameResult> = self
            .results
            .as_chunks::<6>()
            .0
            .iter()
            .map(|r| FrameResult { heading: [r[0], r[1], r[2]], normal: [r[3], r[4], r[5]] })
            .collect();
        for (k, r) in out.iter().enumerate() {
            ensure!(
                r.heading.iter().chain(&r.normal).all(|v| v.is_finite()),
                "decision {k}: non-finite frame output {r:?}"
            );
        }
        Ok(out)
    }

    /// [`FrameModel::launch`] then [`FrameModel::results`]
    pub fn run(&mut self, dev: &Device, crops: &Buf, requests: &[FrameRequest]) -> Result<Vec<FrameResult>> {
        self.launch(dev, crops, requests)?;
        self.results(dev, requests.len())
    }
}
