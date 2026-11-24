//! The strip model's crop CNN: villa's `CropCNN` (`models/blocks.py`) as `fiber_strip_small_v1` configures it,
//! from the model input crop to the cell features, under PyTorch's bf16 autocast numerics
//!
//! Three stages of ResNet-D basic blocks (12, 24, 64 channels; strides 2, 2, 1; 1, 2, 2 blocks), then a stride-2
//! context block on the last stage's map, upsampled (trilinear) back to it and added through a 1x1 conv. Every conv
//! is a channels-last bf16 cuDNN convolution with float32 accumulation; InstanceNorm (float32 statistics per sample
//! and channel), LeakyReLU, the residual adds, the skips' average pools, the upsample and the fuse run in the
//! strips kernels (`strips-kernels/src/cnn.rs`), with PyTorch's bf16 rounding points. The stage-0 conv biases are
//! dropped: each feeds an InstanceNorm, which removes them in exact arithmetic

use std::collections::HashMap;

use color_eyre::eyre::{Result, WrapErr, ensure};
use volkit::dnn::{ConvPlan, ConvShape, Dtype, Pick};
use volkit::gpu::{Arg, Buf, Gpu};

use super::{Device, Weights};

/// The input crop's depth, height and width (villa `cfg.fine`: 144 x 128 x 128 samples)
pub const CROP: [usize; 3] = [144, 128, 128];
/// Input channels: z-scored CT, then the tube
pub const IN_CHANNELS: usize = 2;
/// The cell grid: the crop at a quarter of its resolution
pub const CELLS: [usize; 3] = [36, 32, 32];
/// Channels of a cell feature
pub const CELL_CHANNELS: usize = 64;

/// InstanceNorm3d's epsilon
const EPS: f32 = 1e-5;
/// The most workspace a cuDNN algorithm may take
const MAX_WORKSPACE: usize = 1 << 30;
/// Blocks the statistics kernel aims for per launch (a few waves of the GPU's SMs)
const STATS_BLOCKS: usize = 2048;

fn voxels(s: [usize; 3]) -> usize {
    s[0] * s[1] * s[2]
}

/// A Conv3d's weights as cuDNN's channels-last filter `[K, kd, kh, kw, C]` in bf16, and its geometry
struct Conv {
    w: Buf,
    cin: usize,
    cout: usize,
    k: usize,
    stride: usize,
    /// Input spatial size
    input: [usize; 3],
}

impl Conv {
    /// Conv `name` (`.weight` of shape `[cout, cin, k, k, k]`), rounded to bf16 as autocast casts it
    #[allow(clippy::too_many_arguments)]
    fn load(
        gpu: &Gpu,
        weights: &Weights,
        name: &str,
        cin: usize,
        cout: usize,
        k: usize,
        stride: usize,
        input: [usize; 3],
    ) -> Result<Self> {
        let t = weights.get(&format!("{name}.weight"), &[cout, cin, k, k, k])?;
        let taps = k * k * k;
        let mut w = vec![0u16; cout * taps * cin];
        for o in 0..cout {
            for c in 0..cin {
                for tap in 0..taps {
                    w[(o * taps + tap) * cin + c] =
                        half::bf16::from_f32(t.values[(o * cin + c) * taps + tap]).to_bits();
                }
            }
        }

        let w = gpu.upload(&w).wrap_err_with(|| format!("upload {name}"))?;
        Ok(Self { w, cin, cout, k, stride, input })
    }

    fn output(&self) -> [usize; 3] {
        self.input.map(|d| d / self.stride)
    }

    fn shape(&self, n: usize) -> ConvShape {
        let spatial: Vec<i32> = self.input.iter().map(|&d| d as i32).collect();
        ConvShape::same(n as i32, self.cin as i32, self.cout as i32, &spatial, self.k as i32, self.stride as i32)
    }

    fn plan(&self, dev: &Device, n: usize) -> Result<ConvPlan> {
        let shape = self.shape(n);
        ConvPlan::typed(&dev.gpu, &dev.dnn, shape.clone(), Dtype::Bf16, Pick::Find, MAX_WORKSPACE)
            .wrap_err_with(|| format!("plan {shape}"))
    }

    /// Queues `y = conv(x)` for batch `n`
    fn run(&self, dev: &Device, plan: &ConvPlan, x: &Buf, y: &Buf, ws: Option<&Buf>, n: usize) -> Result<()> {
        let xb = n * voxels(self.input) * self.cin * 2;
        let yb = n * voxels(self.output()) * self.cout * 2;
        ensure!(x.bytes() >= xb && y.bytes() >= yb, "conv buffers too small for batch {n}");
        plan.run(&dev.dnn, (x.ptr(), xb), (self.w.ptr(), self.w.bytes()), (y.ptr(), yb), ws)
    }
}

/// An affine InstanceNorm3d: its float32 weight and bias, and its `[B, C]` statistics buffer
struct Norm {
    w: Buf,
    b: Buf,
    c: usize,
    stats: Buf,
}

impl Norm {
    fn load(gpu: &Gpu, weights: &Weights, name: &str, c: usize, max_batch: usize) -> Result<Self> {
        let w = gpu.upload(&weights.get(&format!("{name}.weight"), &[c])?.values)?;
        let b = gpu.upload(&weights.get(&format!("{name}.bias"), &[c])?.values)?;
        let stats = gpu.alloc(max_batch * c * 8)?;
        Ok(Self { w, b, c, stats })
    }
}

/// A BasicBlockD's skip: an optional AvgPool3d(2) when the block strides, then a 1x1 conv and InstanceNorm when
/// the channels change, else the identity
struct Skip {
    pool: bool,
    proj: Option<(Conv, Norm)>,
}

/// One BasicBlockD: `leaky(norm2(conv2(leaky(norm1(conv1(x))))) + skip(x))`
struct Block {
    name: String,
    conv1: Conv,
    norm1: Norm,
    conv2: Conv,
    norm2: Norm,
    skip: Skip,
}

impl Block {
    /// Block `prefix` (`strip.cnn.stages.0.blocks.0`): `cin -> cout` channels at `input`, the first conv strided
    #[allow(clippy::too_many_arguments)]
    fn load(
        gpu: &Gpu,
        weights: &Weights,
        name: &str,
        prefix: &str,
        cin: usize,
        cout: usize,
        stride: usize,
        input: [usize; 3],
        max_batch: usize,
    ) -> Result<Self> {
        let conv1 = Conv::load(gpu, weights, &format!("{prefix}.conv1.conv"), cin, cout, 3, stride, input)?;
        let norm1 = Norm::load(gpu, weights, &format!("{prefix}.conv1.norm"), cout, max_batch)?;
        let conv2 = Conv::load(gpu, weights, &format!("{prefix}.conv2.conv"), cout, cout, 3, 1, conv1.output())?;
        let norm2 = Norm::load(gpu, weights, &format!("{prefix}.conv2.norm"), cout, max_batch)?;
        let pool = stride == 2;
        // villa's skip Sequential numbers the projection after the pool
        let proj = if cin != cout {
            let p = format!("{prefix}.skip.{}", usize::from(pool));
            let conv = Conv::load(gpu, weights, &format!("{p}.conv"), cin, cout, 1, 1, conv1.output())?;
            Some((conv, Norm::load(gpu, weights, &format!("{p}.norm"), cout, max_batch)?))
        } else {
            None
        };
        Ok(Self { name: name.to_string(), conv1, norm1, conv2, norm2, skip: Skip { pool, proj } })
    }
}

/// cuDNN plans for one batch size, in [`CropCnn::convs`] order
struct Plans {
    convs: Vec<ConvPlan>,
}

/// What a forward's caller sees of each block's output (parity tests): the block name, the buffer, and its shape
/// `[B, D, H, W, C]`
pub type Tap<'a> = &'a mut dyn FnMut(&str, &Buf, [usize; 5]) -> Result<()>;

/// The crop CNN on the GPU: weights, cuDNN plans per batch size, and activation buffers for up to `max_batch`
/// crops
pub struct CropCnn {
    blocks: Vec<Block>,
    fuse: Conv,
    /// The context fuse's bias, as bf16 holds it
    fuse_bias: Buf,
    plans: HashMap<usize, Plans>,
    max_batch: usize,
    /// The input crop in bf16 `[B, 144, 128, 128, 2]`
    xb: Buf,
    /// The input's 2x2x2 average in bf16 `[B, 72, 64, 64, 2]`
    pooled: Buf,
    /// Four activation buffers, each the largest activation (stage 0's `[B, 72, 64, 64, 12]`)
    t: [Buf; 4],
    /// The statistics kernel's partial sums, shared by every norm
    partial: Buf,
    workspace: Option<Buf>,
    cells: Buf,
}

impl CropCnn {
    /// Loads the strip model's CNN weights (`strip.cnn.*`) and allocates buffers for batches up to `max_batch`
    pub fn new(dev: &Device, weights: &Weights, max_batch: usize) -> Result<Self> {
        ensure!(max_batch > 0, "max_batch 0");
        let gpu = &dev.gpu;
        let s0 = CROP;
        let s1 = s0.map(|d| d / 2);
        let s2 = s1.map(|d| d / 2);
        let p = "strip.cnn";
        let blocks = vec![
            Block::load(gpu, weights, "stage0.block0", &format!("{p}.stages.0.blocks.0"), 2, 12, 2, s0, max_batch)?,
            Block::load(gpu, weights, "stage1.block0", &format!("{p}.stages.1.blocks.0"), 12, 24, 2, s1, max_batch)?,
            Block::load(gpu, weights, "stage1.block1", &format!("{p}.stages.1.blocks.1"), 24, 24, 1, s2, max_batch)?,
            Block::load(gpu, weights, "stage2.block0", &format!("{p}.stages.2.blocks.0"), 24, 64, 1, s2, max_batch)?,
            Block::load(gpu, weights, "stage2.block1", &format!("{p}.stages.2.blocks.1"), 64, 64, 1, s2, max_batch)?,
            Block::load(gpu, weights, "context", &format!("{p}.context_stage.blocks.0"), 64, 64, 2, s2, max_batch)?,
        ];
        let fuse = Conv::load(gpu, weights, &format!("{p}.context_fuse"), 64, 64, 1, 1, s2)?;
        let bias: Vec<f32> = weights
            .get(&format!("{p}.context_fuse.bias"), &[64])?
            .values
            .iter()
            .map(|&v| half::bf16::from_f32(v).to_f32())
            .collect();
        let fuse_bias = gpu.upload(&bias)?;

        let n = max_batch;
        let largest = voxels(s1) * 12;
        // `n * chunks` stays under STATS_BLOCKS + n (see `stats`)
        let partial = (STATS_BLOCKS + n) * 64 * 8;
        Ok(Self {
            blocks,
            fuse,
            fuse_bias,
            plans: HashMap::new(),
            max_batch,
            xb: gpu.alloc(n * voxels(s0) * IN_CHANNELS * 2)?,
            pooled: gpu.alloc(n * voxels(s1) * IN_CHANNELS * 2)?,
            t: [
                gpu.alloc(n * largest * 2)?,
                gpu.alloc(n * largest * 2)?,
                gpu.alloc(n * largest * 2)?,
                gpu.alloc(n * largest * 2)?,
            ],
            partial: gpu.alloc(partial)?,
            workspace: None,
            cells: gpu.alloc(n * voxels(CELLS) * CELL_CHANNELS * 2)?,
        })
    }

    /// The most crops one forward takes
    pub fn max_batch(&self) -> usize {
        self.max_batch
    }

    /// Every conv, in plan order: each block's conv1, conv2 and skip projection, then the context fuse
    fn convs(&self) -> Vec<&Conv> {
        let mut convs = Vec::new();
        for b in &self.blocks {
            convs.push(&b.conv1);
            convs.push(&b.conv2);
            if let Some((c, _)) = &b.skip.proj {
                convs.push(c);
            }
        }
        convs.push(&self.fuse);
        convs
    }

    /// Plans every conv for batch `n` (cuDNN's timed pick, replayed from volkit's algorithm cache when it has one),
    /// and grows the shared workspace to fit
    pub fn prepare(&mut self, dev: &Device, n: usize) -> Result<()> {
        ensure!(n > 0 && n <= self.max_batch, "batch {n} outside 1..={}", self.max_batch);
        if self.plans.contains_key(&n) {
            return Ok(());
        }
        let convs = self.convs().iter().map(|c| c.plan(dev, n)).collect::<Result<Vec<_>>>()?;
        let need = convs.iter().map(ConvPlan::workspace).max().unwrap_or(0);
        if self.workspace.as_ref().map_or(need > 0, |w| w.bytes() < need) {
            self.workspace = None;
            self.workspace = Some(dev.gpu.alloc(need)?);
        }
        self.plans.insert(n, Plans { convs });
        Ok(())
    }

    /// The chosen cuDNN algorithm of every conv for batch `n`, as `shape: choice`, once [`CropCnn::prepare`]d
    pub fn algorithms(&self, n: usize) -> Vec<String> {
        self.plans
            .get(&n)
            .map(|p| {
                p.convs
                    .iter()
                    .map(|c| format!("{}: {}{}", c.shape, c.algo.choice(), if c.cached { " (cached)" } else { "" }))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Queues the CNN on `n` crops: `input` holds float32 `[n, 144, 128, 128, 2]` (channel 0 the z-scored CT,
    /// 1 the tube); returns the cells, bf16 `[n, 36, 32, 32, 64]`, valid until the next forward
    pub fn forward(&mut self, dev: &Device, input: &Buf, n: usize) -> Result<&Buf> {
        self.run(dev, input, n, None)?;
        Ok(&self.cells)
    }

    /// [`CropCnn::forward`], also handing each block's output to `tap` (after a stream sync) as it is written
    pub fn forward_tapped(&mut self, dev: &Device, input: &Buf, n: usize, tap: Tap) -> Result<&Buf> {
        self.run(dev, input, n, Some(tap))?;
        Ok(&self.cells)
    }

    fn run(&mut self, dev: &Device, input: &Buf, n: usize, mut tap: Option<Tap>) -> Result<()> {
        self.prepare(dev, n)?;
        let need = n * voxels(CROP) * IN_CHANNELS * 4;
        ensure!(input.bytes() >= need, "CNN input: {} bytes, {n} crops need {need}", input.bytes());
        let plans = &self.plans[&n].convs;
        let ws = self.workspace.as_ref();
        let k = &dev.kernels;
        let gpu = &dev.gpu;

        // the bf16 input and its pooled copy for the first skip
        let s0 = CROP;
        let pooled = n * voxels(s0) / 8;
        let [d, h, w] = s0;
        let args = [
            self.pooled.view::<u32>(0, pooled),
            self.xb.view::<[u32; 2]>(0, pooled * 4),
            input.view::<[f32; 4]>(0, pooled * 4),
            Arg::U32((w / 2) as u32),
            Arg::U32((h / 2 * w / 2) as u32),
            Arg::U32((d / 2 * h / 2 * w / 2) as u32),
        ];
        k.launch(gpu, "sn_input", grid(pooled), (256, 1, 1), 0, &args)?;

        // block i reads `x` and writes its output into one of the other activation buffers
        let mut plan = plans.iter();
        let mut x: Option<usize> = None;
        let mut outputs = Vec::with_capacity(self.blocks.len());
        for blk in &self.blocks {
            let free: Vec<usize> = (0..4).filter(|&i| Some(i) != x).collect();
            let (a, b, c) = (&self.t[free[0]], &self.t[free[1]], &self.t[free[2]]);
            let xin = x.map_or(&self.xb, |i| &self.t[i]);
            let p1 = plan.next().expect("a plan per conv");
            let p2 = plan.next().expect("a plan per conv");
            let pp = blk.skip.proj.as_ref().map(|_| plan.next().expect("a plan per conv"));
            let s_in = blk.conv1.input;
            let s_out = blk.conv1.output();
            let rows = voxels(s_out);
            let cout = blk.conv1.cout;

            blk.conv1.run(dev, p1, xin, a, ws, n)?;
            self.stats(dev, a, &blk.norm1, n, rows)?;
            self.apply(dev, a, &blk.norm1, Residual::None, n, rows, true)?;
            blk.conv2.run(dev, p2, a, b, ws, n)?;
            self.stats(dev, b, &blk.norm2, n, rows)?;
            match (&blk.skip.proj, pp) {
                (Some((conv, norm)), Some(pp)) => {
                    // the projection's input: the pooled input crop (stage 0), the pooled block input, or the input
                    let src = match (blk.skip.pool, x) {
                        (true, None) => &self.pooled,
                        (true, Some(_)) => {
                            self.pool(dev, xin, c, n, s_in, blk.conv1.cin)?;
                            c
                        }
                        (false, _) => xin,
                    };
                    conv.run(dev, pp, src, a, ws, n)?;
                    self.stats(dev, a, norm, n, rows)?;
                    self.apply(dev, b, &blk.norm2, Residual::Norm(a, norm), n, rows, true)?;
                }
                _ if blk.skip.pool => {
                    self.pool(dev, xin, c, n, s_in, cout)?;
                    self.apply(dev, b, &blk.norm2, Residual::Plain(c), n, rows, true)?;
                }
                _ => self.apply(dev, b, &blk.norm2, Residual::Plain(xin), n, rows, true)?,
            }
            if let Some(tap) = tap.as_mut() {
                gpu.sync()?;
                tap(&blk.name, b, [n, s_out[0], s_out[1], s_out[2], cout])?;
            }
            x = Some(free[1]);
            outputs.push(free[1]);
        }

        // cells = stage output + fuse(upsample(context)), the context block's output in `x` and the last stage's
        // output in the buffer the context block read
        let (ctx, last) = (outputs[outputs.len() - 1], outputs[outputs.len() - 2]);
        let stage = self.blocks.len() - 2;
        let free: Vec<usize> = (0..4).filter(|&i| i != ctx && i != last).collect();
        let (up, fused) = (&self.t[free[0]], &self.t[free[1]]);
        let [cd, ch, cw] = self.blocks[stage + 1].conv1.output();
        let quads = (CELL_CHANNELS / 4) as u32;
        let cells = n * voxels(CELLS) * CELL_CHANNELS / 4;
        let args = [
            up.view::<[u32; 2]>(0, cells),
            self.t[ctx].view::<[u32; 2]>(0, cells / 8),
            Arg::U32(quads),
            Arg::U32(cd as u32),
            Arg::U32(ch as u32),
            Arg::U32(cw as u32),
        ];
        k.launch(gpu, "sn_upsample", grid(cells), (256, 1, 1), 0, &args)?;
        self.fuse.run(dev, plan.next().expect("a plan per conv"), up, fused, ws, n)?;
        let args = [
            self.cells.view::<[u32; 2]>(0, cells),
            self.t[last].view::<[u32; 2]>(0, cells),
            fused.view::<[u32; 2]>(0, cells),
            self.fuse_bias.arg::<f32>(),
            Arg::U32(quads),
        ];
        k.launch(gpu, "sn_fuse", grid(cells), (256, 1, 1), 0, &args)?;
        if let Some(tap) = tap.as_mut() {
            gpu.sync()?;
            tap("cells", &self.cells, [n, CELLS[0], CELLS[1], CELLS[2], CELL_CHANNELS])?;
        }
        Ok(())
    }

    /// InstanceNorm statistics of `y` (`[n, rows, norm.c]` bf16) into `norm.stats`
    fn stats(&self, dev: &Device, y: &Buf, norm: &Norm, n: usize, rows: usize) -> Result<()> {
        let c = norm.c;
        let pairs = c / 2;
        let lanes = 256 / pairs;
        let want = STATS_BLOCKS.div_ceil(n).max(1);
        let chunk_rows = rows.div_ceil(want).max(lanes * 8);
        let chunks = rows.div_ceil(chunk_rows);
        ensure!(n * chunks * c * 8 <= self.partial.bytes(), "statistics partials overflow");
        let args = [
            self.partial.view::<[f32; 2]>(0, n * chunks * c),
            y.view::<u32>(0, n * rows * pairs),
            Arg::U32(rows as u32),
            Arg::U32(c as u32),
            Arg::U32(lanes as u32),
            Arg::U32(chunk_rows as u32),
        ];
        dev.kernels.launch(
            &dev.gpu,
            "sn_stats",
            (chunks as u32, n as u32, 1),
            ((pairs * lanes) as u32, 1, 1),
            0,
            &args,
        )?;
        let args = [
            norm.stats.view::<[f32; 2]>(0, n * c),
            self.partial.view::<[f32; 2]>(0, n * chunks * c),
            y.view::<u32>(0, n * rows * pairs),
            Arg::U32(rows as u32),
            Arg::U32(c as u32),
            Arg::U32(chunks as u32),
            Arg::F32(EPS),
        ];
        dev.kernels.launch(&dev.gpu, "sn_finalize", grid(n * c), (256, 1, 1), 0, &args)
    }

    /// `y = leaky?(rbf(norm(y) + residual))` in place
    #[allow(clippy::too_many_arguments)]
    fn apply(&self, dev: &Device, y: &Buf, norm: &Norm, res: Residual, n: usize, rows: usize, act: bool) -> Result<()> {
        let c = norm.c;
        let quads = n * rows * c / 4;
        let (plain, rx, rnorm) = match res {
            Residual::None => (Arg::EMPTY, Arg::EMPTY, None),
            Residual::Plain(r) => (r.view::<[u32; 2]>(0, quads), Arg::EMPTY, None),
            Residual::Norm(r, rn) => (Arg::EMPTY, r.view::<[u32; 2]>(0, quads), Some(rn)),
        };
        let (rs, rw, rb) = match rnorm {
            Some(rn) => (rn.stats.view::<[f32; 2]>(0, n * c), rn.w.arg::<f32>(), rn.b.arg::<f32>()),
            None => (Arg::EMPTY, Arg::EMPTY, Arg::EMPTY),
        };
        let args = [
            y.view::<[u32; 2]>(0, quads),
            norm.stats.view::<[f32; 2]>(0, n * c),
            norm.w.arg::<f32>(),
            norm.b.arg::<f32>(),
            plain,
            rx,
            rs,
            rw,
            rb,
            Arg::U32(rows as u32),
            Arg::U32(c as u32),
            Arg::U32(u32::from(act)),
        ];
        dev.kernels.launch(&dev.gpu, "sn_apply", grid(quads), (256, 1, 1), 0, &args)
    }

    /// AvgPool3d(2) of `x` (`[n, s, c]` bf16) into `out`
    fn pool(&self, dev: &Device, x: &Buf, out: &Buf, n: usize, s: [usize; 3], c: usize) -> Result<()> {
        let [d, h, w] = s.map(|v| v / 2);
        let quads = c / 4;
        let total = n * d * h * w * quads;
        let args = [
            out.view::<[u32; 2]>(0, total),
            x.view::<[u32; 2]>(0, total * 8),
            Arg::U32(quads as u32),
            Arg::U32(w as u32),
            Arg::U32((h * w) as u32),
            Arg::U32((d * h * w) as u32),
        ];
        dev.kernels.launch(&dev.gpu, "sn_pool", grid(total), (256, 1, 1), 0, &args)
    }
}

/// What an InstanceNorm apply adds before its LeakyReLU
enum Residual<'a> {
    None,
    /// A bf16 tensor shaped as the output
    Plain(&'a Buf),
    /// A bias-free conv output and the InstanceNorm to apply to it on load
    Norm(&'a Buf, &'a Norm),
}

/// A 1D grid of 256-thread blocks covering `n` threads
fn grid(n: usize) -> (u32, u32, u32) {
    (n.div_ceil(256) as u32, 1, 1)
}

#[cfg(test)]
mod tests {
    //! GPU checks, run through the GPU queue from a prebuilt test binary:
    //! - `parity`: every block's output and the cells against PyTorch's bf16-autocast and float32 forwards on
    //!   recorded inputs (`ops/vesuvius/strips-native/cnn_ref.py`), gated on PyTorch's own noise band
    //! - `bench`: ms per decision at batch 32, 64 and 128 (`STRIPS_CNN_BATCHES`), untraced
    //!
    //! Both read `STRIPS_CNN_REF` (the reference directory) and `STRIPS_WEIGHTS` (the weights export), and write
    //! JSON to `STRIPS_CNN_OUT` when it is set

    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use color_eyre::eyre::{Result, WrapErr, ensure};
    use serde_json::{Value, json};

    use super::{CROP, CropCnn, IN_CHANNELS, voxels};
    use crate::strips::gpu::{Device, Weights};

    const WEIGHTS: &str = "/build/vesuvius/strips-scale/speed/native/weights/fiber_strip_small_v1.safetensors";
    /// The p99 sample: every k-th element, at most this many, as the reference script takes it
    const SAMPLE: usize = 1 << 24;

    fn env_path(name: &str, default: Option<&str>) -> PathBuf {
        std::env::var_os(name).map(PathBuf::from).or(default.map(PathBuf::from)).unwrap_or_else(|| panic!("set {name}"))
    }

    fn map(path: &Path) -> Result<memmap2::Mmap> {
        let file = std::fs::File::open(path).wrap_err_with(|| format!("open {}", path.display()))?;
        // SAFETY: the reference files are written once by the reference script and not modified while mapped
        Ok(unsafe { memmap2::Mmap::map(&file) }?)
    }

    /// The mapped file as `T`s, checked to hold exactly `len` of them
    fn typed<'a, T: Copy>(m: &'a memmap2::Mmap, len: usize, what: &str) -> Result<&'a [T]> {
        ensure!(m.len() == len * size_of::<T>(), "{what}: {} bytes, expected {len} x {}", m.len(), size_of::<T>());
        ensure!((m.as_ptr() as usize).is_multiple_of(align_of::<T>()), "{what}: misaligned map");
        // SAFETY: length and alignment checked above; f32 and u16 accept every bit pattern
        Ok(unsafe { std::slice::from_raw_parts(m.as_ptr().cast::<T>(), len) })
    }

    fn bf16(bits: u16) -> f32 {
        f32::from_bits(u32::from(bits) << 16)
    }

    /// Mean, p99 (linear interpolation over the strided sample) and max of `|a - r|`, the reference's mean and max
    /// magnitude, and the count of non-finite values in `a`
    fn compare(a: &[u16], r: &dyn Fn(usize) -> f32) -> Value {
        let n = a.len();
        let (mut sum, mut rsum, mut max, mut rmax, mut bad) = (0f64, 0f64, 0f32, 0f32, 0usize);
        let step = (n / SAMPLE).max(1);
        let mut sample = Vec::with_capacity(SAMPLE.min(n));
        for (i, &bits) in a.iter().enumerate() {
            let v = bf16(bits);
            let rv = r(i);
            if !v.is_finite() {
                bad += 1;
                continue;
            }
            let d = (v - rv).abs();
            sum += f64::from(d);
            rsum += f64::from(rv.abs());
            max = max.max(d);
            rmax = rmax.max(rv.abs());
            if i % step == 0 && sample.len() < SAMPLE {
                sample.push(d);
            }
        }

        sample.sort_unstable_by(f32::total_cmp);
        let pos = 0.99 * (sample.len() - 1) as f64;
        let (lo, frac) = (pos.floor() as usize, pos - pos.floor());
        let hi = (lo + 1).min(sample.len() - 1);
        let p99 = f64::from(sample[lo]) + frac * f64::from(sample[hi] - sample[lo]);
        json!({"mean_abs": sum / n as f64, "p99_abs": p99, "max_abs": max, "ref_mean_abs": rsum / n as f64,
               "ref_max_abs": rmax, "nonfinite": bad})
    }

    fn loadavg() -> String {
        std::fs::read_to_string("/proc/loadavg").unwrap_or_default().trim().to_string()
    }

    fn write_out(name: &str, value: &Value) -> Result<()> {
        if let Some(dir) = std::env::var_os("STRIPS_CNN_OUT") {
            let dir = PathBuf::from(dir);
            std::fs::create_dir_all(&dir)?;
            std::fs::write(dir.join(name), serde_json::to_string_pretty(value)?)?;
        }
        Ok(())
    }

    struct Setup {
        dev: Device,
        weights: Weights,
        manifest: Value,
        reference: PathBuf,
        input: memmap2::Mmap,
        batch: usize,
    }

    fn setup() -> Result<Setup> {
        let reference = env_path("STRIPS_CNN_REF", None);
        let weights = Weights::open(&env_path("STRIPS_WEIGHTS", Some(WEIGHTS)))?;
        let manifest: Value = serde_json::from_slice(&std::fs::read(reference.join("manifest.json"))?)?;
        let batch = manifest["batch"].as_u64().expect("manifest batch") as usize;
        let input = map(&reference.join("input.f32"))?;
        typed::<f32>(&input, batch * voxels(CROP) * IN_CHANNELS, "input.f32")?;
        let dev = Device::open(0)?;
        Ok(Setup { dev, weights, manifest, reference, input, batch })
    }

    #[test]
    #[ignore = "needs the GPU and the reference dump"]
    fn parity() -> Result<()> {
        let s = setup()?;
        let n = s.batch;
        let mut cnn = CropCnn::new(&s.dev, &s.weights, n)?;
        let input = s.dev.gpu.upload(typed::<f32>(&s.input, n * voxels(CROP) * IN_CHANNELS, "input")?)?;
        // NaN-prefilled activations and output, so a region a kernel skips cannot pass
        for t in &cnn.t {
            s.dev.gpu.fill(t, 0xff)?;
        }
        s.dev.gpu.fill(&cnn.cells, 0xff)?;

        let blocks = s.manifest["blocks"].as_array().expect("manifest blocks").clone();
        let mut rows = Vec::new();
        let mut seen = Vec::new();
        let mut tap = |name: &str, buf: &volkit::gpu::Buf, shape: [usize; 5]| -> Result<()> {
            let len = shape.iter().product::<usize>();
            let entry = blocks.iter().find(|b| b["name"] == name).unwrap_or_else(|| panic!("no reference for {name}"));
            let want: Vec<usize> =
                entry["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
            ensure!(want == shape, "{name}: shape {shape:?}, reference {want:?}");
            let got: Vec<u16> = s.dev.gpu.download_range(buf, 0, len)?;
            let rb = map(&s.reference.join(entry["bf16"].as_str().unwrap()))?;
            let rf = map(&s.reference.join(entry["f32"].as_str().unwrap()))?;
            let (rb, rf) = (typed::<u16>(&rb, len, name)?, typed::<f32>(&rf, len, name)?);
            let vs_bf16 = compare(&got, &|i| bf16(rb[i]));
            let vs_f32 = compare(&got, &|i| rf[i]);
            let band = &s.manifest["band"][name];
            println!("{name}: vs bf16 {vs_bf16}\n  vs f32 {vs_f32}\n  band {band}");
            rows.push(json!({"name": name, "vs_bf16": vs_bf16, "vs_f32": vs_f32, "band": band}));
            seen.push(name.to_string());
            Ok(())
        };
        cnn.forward_tapped(&s.dev, &input, n, &mut tap)?;
        let first: Vec<u16> = s.dev.gpu.download(&cnn.cells)?;
        cnn.forward(&s.dev, &input, n)?;
        let second: Vec<u16> = s.dev.gpu.download(&cnn.cells)?;
        let deterministic = first == second;

        let out = json!({"batch": n, "blocks": rows, "deterministic": deterministic, "algorithms": cnn.algorithms(n),
                         "loadavg": loadavg()});
        write_out("parity.json", &out)?;
        ensure!(seen.len() == blocks.len(), "checked {seen:?}, the reference has {} outputs", blocks.len());
        ensure!(deterministic, "two forwards of the same input differ");
        // gate: against the float32 forward, our error stays within 1.5x PyTorch bf16's own (mean and p99)
        for r in &rows {
            let (ours, band) = (&r["vs_f32"], &r["band"]["bf16_vs_f32"]);
            ensure!(ours["nonfinite"] == 0, "{}: non-finite outputs", r["name"]);
            for k in ["mean_abs", "p99_abs"] {
                let (o, b) = (ours[k].as_f64().unwrap(), band[k].as_f64().unwrap());
                ensure!(o <= 1.5 * b, "{}: {k} {o:.3e} vs float32, PyTorch bf16's is {b:.3e}", r["name"]);
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "needs the GPU and the reference dump"]
    fn bench() -> Result<()> {
        let s = setup()?;
        let batches: Vec<usize> = std::env::var("STRIPS_CNN_BATCHES")
            .unwrap_or_else(|_| "32,64,128".into())
            .split(',')
            .map(|v| v.trim().parse().expect("STRIPS_CNN_BATCHES: comma-separated sizes"))
            .collect();
        let reps: usize = std::env::var("STRIPS_CNN_REPS").ok().map_or(20, |v| v.parse().expect("STRIPS_CNN_REPS"));
        let max = *batches.iter().max().expect("a batch size");
        let mut cnn = CropCnn::new(&s.dev, &s.weights, max)?;

        // the recorded crops, repeated up to the largest batch
        let per = voxels(CROP) * IN_CHANNELS;
        let recorded = typed::<f32>(&s.input, s.batch * per, "input")?;
        let input = s.dev.gpu.alloc(max * per * 4)?;
        for i in (0..max).step_by(s.batch) {
            let k = s.batch.min(max - i);
            s.dev.gpu.upload_into(&input, i * per * 4, &recorded[..k * per])?;
        }

        let mut results = Vec::new();
        for &n in &batches {
            cnn.prepare(&s.dev, n)?;
            for _ in 0..3 {
                cnn.forward(&s.dev, &input, n)?;
            }
            s.dev.gpu.sync()?;
            let load = loadavg();
            let t = Instant::now();
            for _ in 0..reps {
                cnn.forward(&s.dev, &input, n)?;
            }
            s.dev.gpu.sync()?;
            let ms = t.elapsed().as_secs_f64() * 1e3 / (reps * n) as f64;
            let row = json!({"batch": n, "reps": reps, "ms_per_decision": ms, "loadavg_before": load,
                             "loadavg_after": loadavg(), "algorithms": cnn.algorithms(n)});
            println!("{row}");
            results.push(row);
        }
        write_out("bench.json", &json!(results))
    }
}
