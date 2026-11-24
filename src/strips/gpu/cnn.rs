//! The strip model's crop CNN: villa's `CropCNN` (`models/blocks.py`) as `fiber_strip_small_v1` configures it,
//! from the model input crop to the cell features, under PyTorch's bf16 autocast numerics
//!
//! Three stages of ResNet-D basic blocks (12, 24, 64 channels; strides 2, 2, 1; 1, 2, 2 blocks), then a stride-2
//! context block on the last stage's map, upsampled (trilinear) back to it and added through a 1x1 conv.
//!
//! Stages 0 and 1 run on the strips kernels' own convs ([`Stages01::Native`]): the first conv straight from the
//! float32 crop, then bf16 tensor-core implicit-GEMM convs that apply the previous InstanceNorm and LeakyReLU on load
//! and write per-tile statistics partials from their epilogues. Stage 2, the context block, the skips' 1x1
//! projections and the fuse are channels-last bf16 cuDNN convolutions with float32 accumulation, with InstanceNorm
//! (float32 statistics per sample and channel), LeakyReLU, the residual adds and the skips' average pools in the
//! strips kernels (`strips-kernels/src/cnn.rs`), at PyTorch's bf16 rounding points. [`Stages01::Cudnn`] runs stages
//! 0 and 1 the same way, for comparison.
//!
//! Conv biases (stage 0's two convs and the context fuse) go on as PyTorch's cuDNN path adds them, to the bf16 conv
//! output with a second rounding: the stage-0 ones feed InstanceNorms, which remove them in exact arithmetic but not
//! their rounding, a fixed per-(position, channel) offset the trained head reads (it tilts the strip heights).
//!
//! One departure from PyTorch's order, exact in real arithmetic: the context fuse convolves the context output before
//! the upsample rather than after (both are linear and the upsample's weights sum to 1), on an eighth of the voxels

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

/// The first block of stage 2, where sample groups start (see [`CropCnn::set_group`])
const STAGE2: usize = 3;
/// Stage 2's default sample group
const DEFAULT_GROUP: usize = 16;
/// InstanceNorm3d's epsilon
const EPS: f32 = 1e-5;
/// The most workspace a cuDNN algorithm may take
const MAX_WORKSPACE: usize = 1 << 30;
/// Blocks the statistics kernel aims for per launch (a few waves of the GPU's SMs)
const STATS_BLOCKS: usize = 2048;

fn voxels(s: [usize; 3]) -> usize {
    s[0] * s[1] * s[2]
}

/// A Conv3d's weights as cuDNN's channels-last filter `[K, kd, kh, kw, C]` in bf16, its bias, and its geometry
struct Conv {
    w: Buf,
    /// The bias as bf16 holds it (float32 values), when the conv has one; cuDNN runs the conv without it
    bias: Option<Buf>,
    cin: usize,
    cout: usize,
    k: usize,
    stride: usize,
    /// Input spatial size
    input: [usize; 3],
}

impl Conv {
    /// Conv `name` (`.weight` of shape `[cout, cin, k, k, k]`, and `.bias` when the export has one), rounded to bf16
    /// as autocast casts it
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
        let bias = bias(weights, name, cout)?.map(|b| gpu.upload(&b)).transpose()?;
        Ok(Self { w, bias, cin, cout, k, stride, input })
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

    /// Queues `y = conv(x)` for samples `s` (planned for `s.n`)
    fn run(&self, dev: &Device, plan: &ConvPlan, x: &Buf, y: &Buf, ws: Option<&Buf>, s: Samples) -> Result<()> {
        let (xs, ys) = (voxels(self.input) * self.cin * 2, voxels(self.output()) * self.cout * 2);
        let end = s.first + s.n;
        ensure!(x.bytes() >= end * xs && y.bytes() >= end * ys, "conv buffers too small for samples {s:?}");
        let (xp, yp) = (x.ptr() + (s.first * xs) as u64, y.ptr() + (s.first * ys) as u64);
        plan.run(&dev.dnn, (xp, s.n * xs), (self.w.ptr(), self.w.bytes()), (yp, s.n * ys), ws)
    }
}

/// Conv `name`'s `.bias` rounded to bf16, as float32 values, when the export has one
fn bias(weights: &Weights, name: &str, cout: usize) -> Result<Option<Vec<f32>>> {
    let key = format!("{name}.bias");
    if !weights.has(&key) {
        return Ok(None);
    }
    let t = weights.get(&key, &[cout])?;
    Ok(Some(t.values.iter().map(|&v| half::bf16::from_f32(v).to_f32()).collect()))
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

/// How stages 0 and 1 (the 12- and 24-channel blocks) run
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stages01 {
    /// Every conv through cuDNN, each InstanceNorm in its own statistics and apply passes
    Cudnn,
    /// The strips kernels' own convs: the first conv straight from the float32 input, the 3x3x3 convs on the
    /// tensor cores with the producer's InstanceNorm and LeakyReLU applied on load and their own statistics
    /// written by their epilogues
    Native,
}

/// cuDNN plans of one block
struct BlockPlans {
    c1: ConvPlan,
    c2: ConvPlan,
    proj: Option<ConvPlan>,
}

/// cuDNN plans for one batch size
struct Plans {
    blocks: Vec<BlockPlans>,
    fuse: ConvPlan,
}

/// One of the strips kernels' tensor-core 3x3x3 convs (`sn_conv_*`): its kernel, its B fragments and the tile
/// geometry the kernel was instantiated with
struct MmaConv {
    kernel: &'static str,
    /// B fragments `[step][n tile][lane]` as bf16 pairs
    wf: Buf,
    /// The bf16-rounded bias, for a kernel built with one
    bias: Option<Buf>,
    cin: usize,
    cout: usize,
    stride: usize,
    input: [usize; 3],
    tile: [usize; 3],
    warps: usize,
    smem: u32,
}

impl MmaConv {
    /// Conv `name`'s weights as the kernel's B fragments: input channels in `ch` 8-channel chunks per voxel,
    /// output channels in `nt` 8-wide tiles, every (tap, chunk) pair a k16 half-step; and its bias when it has one,
    /// which `kernel` must have been built to add (`conv_kernel!`'s last parameter)
    #[allow(clippy::too_many_arguments)]
    fn load(
        dev: &Device,
        weights: &Weights,
        name: &str,
        kernel: &'static str,
        (cin, cout, stride): (usize, usize, usize),
        input: [usize; 3],
        tile: [usize; 3],
        (ch, nt, warps): (usize, usize, usize),
    ) -> Result<Self> {
        let t = weights.get(&format!("{name}.weight"), &[cout, cin, 3, 3, 3])?;
        let pairs = 27 * ch;
        let steps = pairs.div_ceil(2);
        let at = |step: usize, k: usize, n: usize| -> u16 {
            let flat = 2 * step + k / 8;
            let c = (flat % ch) * 8 + k % 8;
            let tap = flat / ch;
            if flat >= pairs || c >= cin || n >= cout {
                return 0;
            }
            half::bf16::from_f32(t.values[(n * cin + c) * 27 + tap]).to_bits()
        };
        let mut wf = Vec::with_capacity(steps * nt * 32);
        for step in 0..steps {
            for j in 0..nt {
                for lane in 0..32 {
                    let (g, th) = (lane / 4, lane % 4);
                    let n = j * 8 + g;
                    let pair = |k: usize| u32::from(at(step, k, n)) | (u32::from(at(step, k + 1, n)) << 16);
                    wf.push([pair(2 * th), pair(2 * th + 8)]);
                }
            }
        }

        let hpx: usize = [0, 1, 2].iter().map(|&a| (tile[a] - 1) * stride + 3).product();
        // as `conv_smem` in the kernels: the halo, three cin param slots, the reduction scratch
        let smem = hpx * ch * 16 + 3 * cin.next_multiple_of(4) * 16 + (warps * nt * 8 + nt * 8) * 4;
        let smem = u32::try_from(smem)?;
        dev.kernels.set_max_dynamic_smem(kernel, smem)?;
        let wf = dev.gpu.upload(&wf)?;
        let bias = bias(weights, name, cout)?.map(|b| dev.gpu.upload(&b)).transpose()?;
        Ok(Self { kernel, wf, bias, cin, cout, stride, input, tile, warps, smem })
    }

    fn output(&self) -> [usize; 3] {
        self.input.map(|d| d / self.stride)
    }

    /// Output tiles per sample
    fn tiles(&self) -> usize {
        let o = self.output();
        (0..3).map(|a| o[a] / self.tile[a]).product()
    }

    /// Voxels per tile, the count of each statistics partial
    fn count(&self) -> usize {
        voxels(self.tile)
    }

    /// Queues `out = conv(prologue(x))` and the tile partials of `out` into `part`; the kernel must have been
    /// built for `prologue`'s kind
    fn run(&self, dev: &Device, x: &Buf, prologue: Prologue, out: &Buf, part: &Buf, n: usize) -> Result<()> {
        let tiles = self.tiles();
        let (cin, o) = (self.cin, voxels(self.output()));
        let norm_args =
            |norm: &Norm| (norm.stats.view::<[f32; 2]>(0, n * cin), norm.w.arg::<f32>(), norm.b.arg::<f32>());
        let empty = (Arg::EMPTY, Arg::EMPTY, Arg::EMPTY);
        let (norm, skip, rest) = match prologue {
            Prologue::None => (empty, empty, [Arg::EMPTY, Arg::EMPTY, Arg::EMPTY]),
            Prologue::Norm(norm) => (norm_args(norm), empty, [Arg::EMPTY, Arg::EMPTY, Arg::EMPTY]),
            Prologue::Stage0 { norm2, skip, wsk, pooled_input, pool } => (
                norm_args(norm2),
                norm_args(skip),
                [
                    pooled_input.view::<u32>(0, n * voxels(self.input)),
                    wsk.arg::<f32>(),
                    pool.view::<[u32; 2]>(0, n * o * cin / 4),
                ],
            ),
        };
        let [p, wsk, pool] = rest;
        let [d, h, w] = self.input;
        let args = [
            out.view::<u32>(0, n * o * self.cout / 2),
            part.view::<[f32; 2]>(0, n * tiles * self.cout),
            x.view::<[u32; 2]>(0, n * voxels(self.input) * cin / 4),
            norm.0,
            norm.1,
            norm.2,
            p,
            skip.0,
            skip.1,
            skip.2,
            wsk,
            pool,
            self.wf.arg::<[u32; 2]>(),
            self.bias.as_ref().map_or(Arg::EMPTY, |b| b.arg::<f32>()),
            Arg::U32(d as u32),
            Arg::U32(h as u32),
            Arg::U32(w as u32),
        ];
        let block = (self.warps * 32) as u32;
        dev.kernels.launch(&dev.gpu, self.kernel, (tiles as u32, n as u32, 1), (block, 1, 1), self.smem, &args)
    }
}

/// What a tensor-core conv applies to its input as it loads it
#[derive(Clone, Copy)]
enum Prologue<'a> {
    /// The input as stored
    None,
    /// The producer's InstanceNorm, then LeakyReLU
    Norm(&'a Norm),
    /// The first block's output, from its conv2 output (the input), as `sn_apply_s0` computes it; the conv also writes
    /// that output's 2x2x2 average to `pool`
    Stage0 { norm2: &'a Norm, skip: &'a Norm, wsk: &'a Buf, pooled_input: &'a Buf, pool: &'a Buf },
}

/// The native stages 0 and 1: the first conv's weights for `sn_conv0` and the tensor-core convs
struct Native {
    /// Stage 0's conv1, bf16-rounded, `[27 taps][2][12]`
    conv0: Buf,
    /// Stage 0's skip projection, bf16-rounded, `[2][12]`
    skip0: Buf,
    /// Stage 0's conv2
    s0c2: MmaConv,
    /// Stage 1 block 0's conv1 (stride 2)
    s1c1: MmaConv,
    /// Stage 1's three 24 -> 24 convs: block 0's conv2, block 1's conv1 and conv2
    s1: [MmaConv; 3],
}

/// `sn_conv0`'s output tiles per sample (2 x 8 x 16 voxels of the 72 x 64 x 64 grid)
const CONV0_TILES: usize = 1152;

impl Native {
    fn load(dev: &Device, weights: &Weights) -> Result<Self> {
        let p = "strip.cnn.stages";
        let w = weights.get(&format!("{p}.0.blocks.0.conv1.conv.weight"), &[12, 2, 3, 3, 3])?;
        let mut conv0 = vec![0f32; 27 * 2 * 12];
        for o in 0..12 {
            for c in 0..2 {
                for tap in 0..27 {
                    conv0[(tap * 2 + c) * 12 + o] = half::bf16::from_f32(w.values[(o * 2 + c) * 27 + tap]).to_f32();
                }
            }
        }
        let w = weights.get(&format!("{p}.0.blocks.0.skip.1.conv.weight"), &[12, 2, 1, 1, 1])?;
        let mut skip0 = vec![0f32; 24];
        for o in 0..12 {
            for c in 0..2 {
                skip0[c * 12 + o] = half::bf16::from_f32(w.values[o * 2 + c]).to_f32();
            }
        }

        let s1 = CROP.map(|d| d / 2);
        let s2 = s1.map(|d| d / 2);
        let c24 =
            |name: &str, kernel| MmaConv::load(dev, weights, name, kernel, (24, 24, 1), s2, [2, 8, 16], (3, 3, 8));
        Ok(Self {
            conv0: dev.gpu.upload(&conv0)?,
            skip0: dev.gpu.upload(&skip0)?,
            s0c2: MmaConv::load(
                dev,
                weights,
                &format!("{p}.0.blocks.0.conv2.conv"),
                "sn_conv_12_12_n",
                (12, 12, 1),
                s1,
                [4, 8, 16],
                (2, 2, 8),
            )?,
            s1c1: MmaConv::load(
                dev,
                weights,
                &format!("{p}.1.blocks.0.conv1.conv"),
                "sn_conv_12_24_s2",
                (12, 24, 2),
                s1,
                [2, 4, 16],
                (2, 3, 8),
            )?,
            s1: [
                c24(&format!("{p}.1.blocks.0.conv2.conv"), "sn_conv_24_24_n")?,
                c24(&format!("{p}.1.blocks.1.conv1.conv"), "sn_conv_24_24")?,
                c24(&format!("{p}.1.blocks.1.conv2.conv"), "sn_conv_24_24_n")?,
            ],
        })
    }
}

/// What a forward's caller sees of each block's output (parity tests): the block name, the buffer, and its shape
/// `[B, D, H, W, C]`
pub type Tap<'a> = &'a mut dyn FnMut(&str, &Buf, [usize; 5]) -> Result<()>;

/// The crop CNN on the GPU: weights, cuDNN plans per batch size, and activation buffers for up to `max_batch`
/// crops
pub struct CropCnn {
    blocks: Vec<Block>,
    fuse: Conv,
    native: Native,
    stages01: Stages01,
    /// The sample group stage 2 onward runs in, `None` for the whole batch
    group: Option<usize>,
    plans: HashMap<usize, Plans>,
    max_batch: usize,
    /// The input crop in bf16 `[B, 144, 128, 128, 2]` (cuDNN stage 0 only)
    xb: Buf,
    /// The input's 2x2x2 average in bf16 `[B, 72, 64, 64, 2]`
    pooled: Buf,
    /// Four activation buffers, each the largest activation (stage 0's `[B, 72, 64, 64, 12]`)
    t: [Buf; 4],
    /// Statistics partials, shared by every norm
    partial: Buf,
    workspace: Option<Buf>,
    cells: Buf,
}

impl CropCnn {
    /// Loads the strip model's CNN weights (`strip.cnn.*`) and allocates buffers for batches up to `max_batch`;
    /// stages 0 and 1 run [`Stages01::Native`]
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
        // the fuse runs on the context block's output, before the upsample (see `sn_upfuse`)
        let fuse = Conv::load(gpu, weights, &format!("{p}.context_fuse"), 64, 64, 1, 1, s2.map(|d| d / 2))?;
        ensure!(fuse.bias.is_some(), "the context fuse has no bias");
        ensure!(blocks[0].conv1.bias.is_some() && blocks[0].conv2.bias.is_some(), "stage 0's convs have no bias");
        let native = Native::load(dev, weights)?;

        let n = max_batch;
        let largest = voxels(s1) * 12;
        // the larger of sn_conv0's tile partials and the statistics kernel's (`n * chunks` stays under
        // STATS_BLOCKS + n, see `stats`)
        let partial = (n * CONV0_TILES * 24).max((STATS_BLOCKS + n) * 64) * 8;
        Ok(Self {
            blocks,
            fuse,
            native,
            stages01: Stages01::Native,
            group: Some(DEFAULT_GROUP),
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

    /// Chooses how stages 0 and 1 run from the next forward on
    pub fn set_stages01(&mut self, mode: Stages01) {
        self.stages01 = mode;
    }

    /// Runs stage 2, the context block and the cells `group` samples at a time from the next forward on (`None`:
    /// the whole batch at once). A group's 64-channel activations (4.7 MB per sample) then stay in L2 between the
    /// cuDNN convs and the norm passes that read them back, at the cost of smaller conv launches
    pub fn set_group(&mut self, group: Option<usize>) {
        self.group = group;
    }

    /// Plans every conv for batch `n` and for the sample groups it splits into (cuDNN's timed pick, replayed from
    /// volkit's algorithm cache when it has one), and grows the shared workspace to fit
    pub fn prepare(&mut self, dev: &Device, n: usize) -> Result<()> {
        ensure!(n > 0 && n <= self.max_batch, "batch {n} outside 1..={}", self.max_batch);
        let group = self.group(n);
        for k in [n, group, n % group] {
            self.plan(dev, k)?;
        }
        Ok(())
    }

    fn plan(&mut self, dev: &Device, n: usize) -> Result<()> {
        if n == 0 || self.plans.contains_key(&n) {
            return Ok(());
        }
        let mut blocks = Vec::with_capacity(self.blocks.len());
        for b in &self.blocks {
            let proj = b.skip.proj.as_ref().map(|(c, _)| c.plan(dev, n)).transpose()?;
            blocks.push(BlockPlans { c1: b.conv1.plan(dev, n)?, c2: b.conv2.plan(dev, n)?, proj });
        }
        let plans = Plans { blocks, fuse: self.fuse.plan(dev, n)? };
        let all = plans.blocks.iter().flat_map(|b| [Some(&b.c1), Some(&b.c2), b.proj.as_ref()]).flatten();
        let need = all.chain([&plans.fuse]).map(ConvPlan::workspace).max().unwrap_or(0);
        if self.workspace.as_ref().map_or(need > 0, |w| w.bytes() < need) {
            self.workspace = None;
            self.workspace = Some(dev.gpu.alloc(need)?);
        }
        self.plans.insert(n, plans);
        Ok(())
    }

    /// The chosen cuDNN algorithm of every conv for batch `n`, as `shape: choice`, once [`CropCnn::prepare`]d
    pub fn algorithms(&self, n: usize) -> Vec<String> {
        let Some(plans) = self.plans.get(&n) else { return Vec::new() };
        let all = plans.blocks.iter().flat_map(|b| [Some(&b.c1), Some(&b.c2), b.proj.as_ref()]).flatten();
        all.chain([&plans.fuse])
            .map(|c| format!("{}: {}{}", c.shape, c.algo.choice(), if c.cached { " (cached)" } else { "" }))
            .collect()
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
        let plans = &self.plans[&n];
        let gpu = &dev.gpu;

        // stages 0 and 1 leave their output in activation buffer `x`; then each block reads `x` and writes its
        // output into one of the other activation buffers
        let (first, mut x) = match self.stages01 {
            Stages01::Native => (STAGE2, self.native01(dev, input, n, plans, &mut tap)?),
            Stages01::Cudnn => {
                self.cast_input(dev, input, n)?;
                (0, None)
            }
        };
        for i in first..STAGE2 {
            let blk = &self.blocks[i];
            x = Some(self.block(dev, blk, &plans.blocks[i], x, None, Samples::all(n))?);
            self.tap_block(dev, blk, x, n, &mut tap)?;
        }

        // stage 2, the context block and the cells, a sample group at a time (see `CropCnn::set_group`); a tapped
        // forward runs them on the whole batch, so each block's whole output exists when it is tapped. Stage 1's
        // output, in `x`, is packed at its own per-sample size, so no group writes its buffer: a group's 64-channel
        // tensors there would cover samples of later groups
        let keep = x;
        let group = if tap.is_some() { n } else { self.group(n) };
        for g in (0..n).step_by(group) {
            let s = Samples { first: g, n: group.min(n - g) };
            let plans = &self.plans[&s.n];
            let mut xg = x;
            let mut outputs = Vec::with_capacity(self.blocks.len() - STAGE2);
            for i in STAGE2..self.blocks.len() {
                let blk = &self.blocks[i];
                let out = self.block(dev, blk, &plans.blocks[i], xg, keep, s)?;
                self.tap_block(dev, blk, Some(out), n, &mut tap)?;
                xg = Some(out);
                outputs.push(out);
            }
            self.cells(dev, plans, outputs[outputs.len() - 1], outputs[outputs.len() - 2], keep, s)?;
        }
        if let Some(tap) = tap.as_mut() {
            gpu.sync()?;
            tap("cells", &self.cells, [n, CELLS[0], CELLS[1], CELLS[2], CELL_CHANNELS])?;
        }
        Ok(())
    }

    /// The sample group stage 2 onward runs in, for batch `n`
    fn group(&self, n: usize) -> usize {
        self.group.map_or(n, |g| g.clamp(1, n))
    }

    /// Hands block `blk`'s output, in activation buffer `out`, to `tap` after a stream sync
    fn tap_block(&self, dev: &Device, blk: &Block, out: Option<usize>, n: usize, tap: &mut Option<Tap>) -> Result<()> {
        let (Some(tap), Some(out)) = (tap.as_mut(), out) else { return Ok(()) };
        dev.gpu.sync()?;
        let s = blk.conv1.output();
        tap(&blk.name, &self.t[out], [n, s[0], s[1], s[2], blk.conv1.cout])
    }

    /// `cells = last + upsample(fuse(ctx))` for samples `s`, from the context block's output in activation buffer
    /// `ctx` and the last stage's in `last`, writing no activation buffer but a fourth one besides `keep`
    #[allow(clippy::too_many_arguments)]
    fn cells(
        &self,
        dev: &Device,
        plans: &Plans,
        ctx: usize,
        last: usize,
        keep: Option<usize>,
        s: Samples,
    ) -> Result<()> {
        let free = (0..4).find(|&i| i != ctx && i != last && Some(i) != keep);
        let fused = &self.t[free.expect("a free activation buffer for the fuse")];
        self.fuse.run(dev, &plans.fuse, &self.t[ctx], fused, self.workspace.as_ref(), s)?;
        let [cd, ch, cw] = self.fuse.input;
        let per = voxels(CELLS) * CELL_CHANNELS / 4;
        let (at, cells) = (s.first * per, s.n * per);
        let args = [
            self.cells.view::<[u32; 2]>(at, cells),
            self.t[last].view::<[u32; 2]>(at, cells),
            fused.view::<[u32; 2]>(at / 8, cells / 8),
            self.fuse.bias.as_ref().expect("the fuse's bias, checked at load").arg::<f32>(),
            Arg::U32((CELL_CHANNELS / 4) as u32),
            Arg::U32(cd as u32),
            Arg::U32(ch as u32),
            Arg::U32(cw as u32),
        ];
        dev.kernels.launch(&dev.gpu, "sn_upfuse", grid(cells), (256, 1, 1), 0, &args)
    }

    /// The bf16 input and its pooled copy for the first skip, for cuDNN's stage 0
    fn cast_input(&self, dev: &Device, input: &Buf, n: usize) -> Result<()> {
        let pooled = n * voxels(CROP) / 8;
        let [d, h, w] = CROP;
        let args = [
            self.pooled.view::<u32>(0, pooled),
            self.xb.view::<[u32; 2]>(0, pooled * 4),
            input.view::<[f32; 4]>(0, pooled * 4),
            Arg::U32((w / 2) as u32),
            Arg::U32((h / 2 * w / 2) as u32),
            Arg::U32((d / 2 * h / 2 * w / 2) as u32),
        ];
        dev.kernels.launch(&dev.gpu, "sn_input", grid(pooled), (256, 1, 1), 0, &args)
    }

    /// One block on cuDNN convs, reading activation buffer `x` (the bf16 input crop when `None`); returns the
    /// activation buffer holding its output
    ///
    /// The block writes only activation buffers other than `x` and `keep`: two of them, or three for a block that
    /// both pools and projects its skip
    #[allow(clippy::too_many_arguments)]
    fn block(
        &self,
        dev: &Device,
        blk: &Block,
        bp: &BlockPlans,
        x: Option<usize>,
        keep: Option<usize>,
        n: Samples,
    ) -> Result<usize> {
        let ws = self.workspace.as_ref();
        let free: Vec<usize> = (0..4).filter(|&i| Some(i) != x && Some(i) != keep).collect();
        let (a, b) = (&self.t[free[0]], &self.t[free[1]]);
        let xin = x.map_or(&self.xb, |i| &self.t[i]);
        let s_in = blk.conv1.input;
        let rows = voxels(blk.conv1.output());
        let cout = blk.conv1.cout;

        blk.conv1.run(dev, &bp.c1, xin, a, ws, n)?;
        self.bias(dev, a, &blk.conv1, n, rows)?;
        self.stats(dev, a, &blk.norm1, n, rows)?;
        self.apply(dev, a, &blk.norm1, Residual::None, n, rows, true)?;
        blk.conv2.run(dev, &bp.c2, a, b, ws, n)?;
        self.bias(dev, b, &blk.conv2, n, rows)?;
        self.stats(dev, b, &blk.norm2, n, rows)?;
        match (&blk.skip.proj, &bp.proj) {
            (Some((conv, norm)), Some(pp)) => {
                // the projection's input: the pooled input crop (stage 0), the pooled block input, or the input
                let src = match (blk.skip.pool, x) {
                    (true, None) => &self.pooled,
                    (true, Some(_)) => {
                        let c = &self.t[*free.get(2).expect("a pooling projection needs a third activation buffer")];
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
                // conv1's buffer is free once conv2 has read it
                self.pool(dev, xin, a, n, s_in, cout)?;
                self.apply(dev, b, &blk.norm2, Residual::Plain(a), n, rows, true)?;
            }
            _ => self.apply(dev, b, &blk.norm2, Residual::Plain(xin), n, rows, true)?,
        }
        Ok(free[1])
    }

    /// Stages 0 and 1 on the strips kernels' convs (see [`Stages01::Native`]); returns the activation buffer
    /// holding stage 1's output
    fn native01(
        &self,
        dev: &Device,
        input: &Buf,
        n: usize,
        plans: &Plans,
        tap: &mut Option<Tap>,
    ) -> Result<Option<usize>> {
        let nat = &self.native;
        let [t0, t1, t2, t3] = &self.t;
        let [b0, b1, b2] = [&self.blocks[0], &self.blocks[1], &self.blocks[2]];
        let (sk0, sk1) = match (&b0.skip.proj, &b1.skip.proj) {
            (Some((_, a)), Some((c, b))) => (a, (c, b)),
            _ => unreachable!("stages 0 and 1 open with projecting blocks"),
        };
        let s1 = CROP.map(|d| d / 2);
        let s2 = s1.map(|d| d / 2);
        let (v1, v2) = (voxels(s1), voxels(s2));
        let all = Samples::all(n);

        // stage 0: conv1 and the skip's pooled input from the float32 crop, then conv2 with conv1's norm on load. The
        // block's output is never stored: stage 1's first conv rebuilds it on load
        let args = [
            t0.view::<[u32; 2]>(0, n * v1 * 3),
            self.pooled.view::<u32>(0, n * v1),
            self.partial.view::<[f32; 2]>(0, n * CONV0_TILES * 24),
            input.view::<[f32; 2]>(0, n * voxels(CROP)),
            nat.conv0.view::<[f32; 4]>(0, 162),
            nat.skip0.arg::<f32>(),
            b0.conv1.bias.as_ref().expect("stage 0's conv1 bias, checked at load").arg::<f32>(),
        ];
        dev.kernels.launch(&dev.gpu, "sn_conv0", (CONV0_TILES as u32, n as u32, 1), (256, 1, 1), 0, &args)?;
        self.merge(dev, &b0.norm1.stats, CONV0_TILES, 24, 0, 12, 256, n)?;
        self.merge(dev, &sk0.stats, CONV0_TILES, 24, 12, 12, 256, n)?;
        self.mma(dev, &nat.s0c2, t0, Prologue::Norm(&b0.norm1), t1, &b0.norm2, n)?;

        // stage 1 block 0: the strided conv1 on stage 0's output (rebuilt from conv2's in t1), which also writes
        // that output's average for the skip; the skip's 1x1 conv and statistics; conv2 with conv1's norm on load;
        // the block output in place of conv2's
        let stage0 =
            Prologue::Stage0 { norm2: &b0.norm2, skip: sk0, wsk: &nat.skip0, pooled_input: &self.pooled, pool: t2 };
        self.mma(dev, &nat.s1c1, t1, stage0, t0, &b1.norm1, n)?;
        if let Some(tap) = tap.as_mut() {
            // only a tapped forward stores stage 0's output, in place of conv2's, which nothing reads any more
            self.apply_s0(dev, t1, b0, sk0, n)?;
            dev.gpu.sync()?;
            tap(&b0.name, t1, [n, s1[0], s1[1], s1[2], 12])?;
        }
        let (proj, proj_norm) = sk1;
        let pp = plans.blocks[1].proj.as_ref().expect("stage 1 block 0 projects");
        proj.run(dev, pp, t2, t3, self.workspace.as_ref(), all)?;
        self.stats(dev, t3, proj_norm, all, v2)?;
        self.mma(dev, &nat.s1[0], t0, Prologue::Norm(&b1.norm1), t2, &b1.norm2, n)?;
        self.apply(dev, t2, &b1.norm2, Residual::Norm(t3, proj_norm), all, v2, true)?;
        if let Some(tap) = tap.as_mut() {
            dev.gpu.sync()?;
            tap(&b1.name, t2, [n, s2[0], s2[1], s2[2], 24])?;
        }

        // stage 1 block 1: identity skip
        self.mma(dev, &nat.s1[1], t2, Prologue::None, t0, &b2.norm1, n)?;
        self.mma(dev, &nat.s1[2], t0, Prologue::Norm(&b2.norm1), t1, &b2.norm2, n)?;
        self.apply(dev, t1, &b2.norm2, Residual::Plain(t2), all, v2, true)?;
        if let Some(tap) = tap.as_mut() {
            dev.gpu.sync()?;
            tap(&b2.name, t1, [n, s2[0], s2[1], s2[2], 24])?;
        }
        Ok(Some(1))
    }

    /// Stage 0's output in place of its conv2 output `y` (`sn_apply_s0`)
    fn apply_s0(&self, dev: &Device, y: &Buf, b0: &Block, sk0: &Norm, n: usize) -> Result<()> {
        let v1 = voxels(b0.conv1.output());
        let args = [
            y.view::<[u32; 2]>(0, n * v1 * 3),
            self.pooled.view::<u32>(0, n * v1),
            b0.norm2.stats.view::<[f32; 2]>(0, n * 12),
            b0.norm2.w.arg::<f32>(),
            b0.norm2.b.arg::<f32>(),
            sk0.stats.view::<[f32; 2]>(0, n * 12),
            sk0.w.arg::<f32>(),
            sk0.b.arg::<f32>(),
            self.native.skip0.arg::<f32>(),
            Arg::U32(v1 as u32),
        ];
        dev.kernels.launch(&dev.gpu, "sn_apply_s0", grid(n * v1 * 3), (256, 1, 1), 0, &args)
    }

    /// One tensor-core conv and the merge of its tile partials into `norm`'s statistics
    #[allow(clippy::too_many_arguments)]
    fn mma(
        &self,
        dev: &Device,
        conv: &MmaConv,
        x: &Buf,
        prologue: Prologue,
        out: &Buf,
        norm: &Norm,
        n: usize,
    ) -> Result<()> {
        conv.run(dev, x, prologue, out, &self.partial, n)?;
        self.merge(dev, &norm.stats, conv.tiles(), conv.cout, 0, conv.cout, conv.count(), n)
    }

    /// `stats[b, c] = [mean, invstd]` from tile partials `[b, tile, pcols]` columns `c0..c0 + cols`
    #[allow(clippy::too_many_arguments)]
    fn merge(
        &self,
        dev: &Device,
        stats: &Buf,
        tiles: usize,
        pcols: usize,
        c0: usize,
        cols: usize,
        count: usize,
        n: usize,
    ) -> Result<()> {
        let args = [
            stats.view::<[f32; 2]>(0, n * cols),
            self.partial.view::<[f32; 2]>(0, n * tiles * pcols),
            Arg::U32(tiles as u32),
            Arg::U32(pcols as u32),
            Arg::U32(c0 as u32),
            Arg::U32(cols as u32),
            Arg::F32(count as f32),
            Arg::F32(EPS),
        ];
        dev.kernels.launch(&dev.gpu, "sn_merge", (cols as u32, n as u32, 1), (256, 1, 1), 0, &args)
    }

    /// Conv `conv`'s bias, if it has one, on its cuDNN output `y` (`[B, rows, conv.cout]` bf16) in place, samples `s`
    fn bias(&self, dev: &Device, y: &Buf, conv: &Conv, s: Samples, rows: usize) -> Result<()> {
        let Some(bias) = &conv.bias else { return Ok(()) };
        let c = conv.cout;
        let (at, quads) = (s.first * rows * c / 4, s.n * rows * c / 4);
        let args = [y.view::<[u32; 2]>(at, quads), bias.arg::<f32>(), Arg::U32(c as u32)];
        dev.kernels.launch(&dev.gpu, "sn_bias", grid(quads), (256, 1, 1), 0, &args)
    }

    /// InstanceNorm statistics of samples `s` of `y` (`[B, rows, norm.c]` bf16) into `norm.stats`
    fn stats(&self, dev: &Device, y: &Buf, norm: &Norm, s: Samples, rows: usize) -> Result<()> {
        let (n, c) = (s.n, norm.c);
        let pairs = c / 2;
        let lanes = 256 / pairs;
        let want = STATS_BLOCKS.div_ceil(n).max(1);
        let chunk_rows = rows.div_ceil(want).max(lanes * 8);
        let chunks = rows.div_ceil(chunk_rows);
        ensure!(n * chunks * c * 8 <= self.partial.bytes(), "statistics partials overflow");
        let args = [
            self.partial.view::<[f32; 2]>(0, n * chunks * c),
            y.view::<u32>(s.first * rows * pairs, n * rows * pairs),
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
            norm.stats.view::<[f32; 2]>(s.first * c, n * c),
            self.partial.view::<[f32; 2]>(0, n * chunks * c),
            y.view::<u32>(s.first * rows * pairs, n * rows * pairs),
            Arg::U32(rows as u32),
            Arg::U32(c as u32),
            Arg::U32(chunks as u32),
            Arg::F32(EPS),
        ];
        dev.kernels.launch(&dev.gpu, "sn_finalize", grid(n * c), (256, 1, 1), 0, &args)
    }

    /// `y = leaky?(rbf(norm(y) + residual))` in place, on samples `s`
    #[allow(clippy::too_many_arguments)]
    fn apply(
        &self,
        dev: &Device,
        y: &Buf,
        norm: &Norm,
        res: Residual,
        s: Samples,
        rows: usize,
        act: bool,
    ) -> Result<()> {
        let c = norm.c;
        let (at, quads) = (s.first * rows * c / 4, s.n * rows * c / 4);
        let (plain, rx, rnorm) = match res {
            Residual::None => (Arg::EMPTY, Arg::EMPTY, None),
            Residual::Plain(r) => (r.view::<[u32; 2]>(at, quads), Arg::EMPTY, None),
            Residual::Norm(r, rn) => (Arg::EMPTY, r.view::<[u32; 2]>(at, quads), Some(rn)),
        };
        let (rs, rw, rb) = match rnorm {
            Some(rn) => (rn.stats.view::<[f32; 2]>(s.first * c, s.n * c), rn.w.arg::<f32>(), rn.b.arg::<f32>()),
            None => (Arg::EMPTY, Arg::EMPTY, Arg::EMPTY),
        };
        let args = [
            y.view::<[u32; 2]>(at, quads),
            norm.stats.view::<[f32; 2]>(s.first * c, s.n * c),
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

    /// AvgPool3d(2) of samples `n` of `x` (`[B, s, c]` bf16) into `out`
    fn pool(&self, dev: &Device, x: &Buf, out: &Buf, n: Samples, s: [usize; 3], c: usize) -> Result<()> {
        let [d, h, w] = s.map(|v| v / 2);
        let quads = c / 4;
        let (at, total) = (n.first * d * h * w * quads, n.n * d * h * w * quads);
        let args = [
            out.view::<[u32; 2]>(at, total),
            x.view::<[u32; 2]>(at * 8, total * 8),
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

/// Samples `first..first + n` of the batch: every buffer is batch-major, so a kernel or conv acts on a run of samples
/// through offsets into the whole batch's buffers
#[derive(Clone, Copy, Debug)]
struct Samples {
    first: usize,
    n: usize,
}

impl Samples {
    fn all(n: usize) -> Self {
        Self { first: 0, n }
    }
}

#[cfg(test)]
mod tests {
    //! GPU checks, run through the GPU queue from a prebuilt test binary:
    //! - `parity`: every block's output and the cells against PyTorch's bf16-autocast and float32 forwards on
    //!   recorded inputs (`ops/vesuvius/strips-native/cnn_ref.py`), gated on PyTorch's own noise band
    //! - `bench`: ms per decision at batch 32, 64 and 128 (`STRIPS_CNN_BATCHES`), untraced
    //!
    //! Both read `STRIPS_CNN_REF` (the reference directory) and `STRIPS_WEIGHTS` (the weights export), run each
    //! stage 0-1 mode in `STRIPS_CNN_MODES` (default `native,cudnn`), and write JSON per mode to `STRIPS_CNN_OUT`
    //! when it is set

    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use color_eyre::eyre::{Result, WrapErr, ensure};
    use serde_json::{Value, json};

    use super::{CELL_CHANNELS, CELLS, CROP, CropCnn, IN_CHANNELS, Stages01, voxels};
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

    /// `got` (bf16 bits, shaped `shape`) against reference output `reference` from the manifest's `blocks`, as a
    /// parity row named `name`
    fn check(
        s: &Setup,
        blocks: &[Value],
        name: &str,
        reference: &str,
        got: &[u16],
        shape: [usize; 5],
    ) -> Result<Value> {
        let len = shape.iter().product::<usize>();
        ensure!(got.len() == len, "{name}: {} values, shape {shape:?}", got.len());
        let entry =
            blocks.iter().find(|b| b["name"] == reference).unwrap_or_else(|| panic!("no reference for {reference}"));
        let want: Vec<usize> =
            entry["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        ensure!(want == shape, "{name}: shape {shape:?}, reference {want:?}");
        let rb = map(&s.reference.join(entry["bf16"].as_str().unwrap()))?;
        let rf = map(&s.reference.join(entry["f32"].as_str().unwrap()))?;
        let (rb, rf) = (typed::<u16>(&rb, len, reference)?, typed::<f32>(&rf, len, reference)?);
        let vs_bf16 = compare(got, &|i| bf16(rb[i]));
        let vs_f32 = compare(got, &|i| rf[i]);
        let band = &s.manifest["band"][reference];
        println!("{name}: vs bf16 {vs_bf16}\n  vs f32 {vs_f32}\n  band {band}");
        Ok(json!({"name": name, "vs_bf16": vs_bf16, "vs_f32": vs_f32, "band": band}))
    }

    /// Under `STRIPS_CNN_DUMP`, block `name`'s output (bf16 bits, NDHWC) as `<STRIPS_CNN_OUT>/<mode>_<name>.bf16`, for
    /// offline signed-bias analysis (`ops/vesuvius/strips-native/cells_bias.py`; the parity rows hold only magnitudes)
    fn dump(mode: &str, name: &str, got: &[u16]) -> Result<()> {
        let (Some(_), Some(dir)) = (std::env::var_os("STRIPS_CNN_DUMP"), std::env::var_os("STRIPS_CNN_OUT")) else {
            return Ok(());
        };
        let dir = PathBuf::from(dir);
        std::fs::create_dir_all(&dir)?;
        let bytes: Vec<u8> = got.iter().flat_map(|v| v.to_le_bytes()).collect();
        Ok(std::fs::write(dir.join(format!("{mode}_{name}.bf16")), bytes)?)
    }

    fn modes() -> Vec<(&'static str, Stages01)> {
        let names = std::env::var("STRIPS_CNN_MODES").unwrap_or_else(|_| "native,cudnn".into());
        let modes = names.split(',').map(|m| match m.trim() {
            "native" => ("native", Stages01::Native),
            "cudnn" => ("cudnn", Stages01::Cudnn),
            other => panic!("STRIPS_CNN_MODES: unknown mode {other}"),
        });
        modes.collect()
    }

    #[test]
    #[ignore = "needs the GPU and the reference dump"]
    fn parity() -> Result<()> {
        let s = setup()?;
        // every mode runs before any gate fails, so one report covers both
        let results: Vec<Result<()>> = modes().into_iter().map(|(name, mode)| parity_mode(&s, name, mode)).collect();
        results.into_iter().collect()
    }

    fn parity_mode(s: &Setup, mode_name: &str, mode: Stages01) -> Result<()> {
        println!("== stages 0-1: {mode_name}");
        let n = s.batch;
        let mut cnn = CropCnn::new(&s.dev, &s.weights, n)?;
        cnn.set_stages01(mode);
        let input = s.dev.gpu.upload(typed::<f32>(&s.input, n * voxels(CROP) * IN_CHANNELS, "input")?)?;
        // NaN-prefilled activations and output, so a region a kernel skips cannot pass
        for t in &cnn.t {
            s.dev.gpu.fill(t, 0xff)?;
        }
        s.dev.gpu.fill(&cnn.cells, 0xff)?;

        let blocks = s.manifest["blocks"].as_array().expect("manifest blocks").clone();
        let mut rows = Vec::new();
        let mut seen = Vec::new();
        // the tapped forward runs stage 2 on the whole batch, so every block is checked ungrouped
        let mut tap = |name: &str, buf: &volkit::gpu::Buf, shape: [usize; 5]| -> Result<()> {
            let got: Vec<u16> = s.dev.gpu.download_range(buf, 0, shape.iter().product())?;
            dump(mode_name, name, &got)?;
            rows.push(check(s, &blocks, name, name, &got, shape)?);
            seen.push(name.to_string());
            Ok(())
        };
        cnn.forward_tapped(&s.dev, &input, n, &mut tap)?;

        // then the cells of the forward as it runs untapped (stage 2 in sample groups), twice
        let group = cnn.group(n);
        let shape = [n, CELLS[0], CELLS[1], CELLS[2], CELL_CHANNELS];
        s.dev.gpu.fill(&cnn.cells, 0xff)?;
        cnn.forward(&s.dev, &input, n)?;
        let first: Vec<u16> = s.dev.gpu.download(&cnn.cells)?;
        rows.push(check(s, &blocks, &format!("cells (group {group})"), "cells", &first, shape)?);
        s.dev.gpu.fill(&cnn.cells, 0xff)?;
        cnn.forward(&s.dev, &input, n)?;
        let second: Vec<u16> = s.dev.gpu.download(&cnn.cells)?;
        let deterministic = first == second;

        let out = json!({"batch": n, "mode": mode_name, "group": group, "blocks": rows,
                         "deterministic": deterministic, "algorithms": cnn.algorithms(n), "loadavg": loadavg()});
        write_out(&format!("parity_{mode_name}.json"), &out)?;
        ensure!(
            seen.len() == blocks.len(),
            "{mode_name}: checked {seen:?}, the reference has {} outputs",
            blocks.len()
        );
        ensure!(deterministic, "{mode_name}: two grouped forwards of the same input differ");
        // gate: against the float32 forward, our error stays within 1.5x PyTorch bf16's own (mean and p99)
        for r in &rows {
            let (ours, band) = (&r["vs_f32"], &r["band"]["bf16_vs_f32"]);
            ensure!(ours["nonfinite"] == 0, "{mode_name} {}: non-finite outputs", r["name"]);
            for k in ["mean_abs", "p99_abs"] {
                let (o, b) = (ours[k].as_f64().unwrap(), band[k].as_f64().unwrap());
                ensure!(o <= 1.5 * b, "{mode_name} {}: {k} {o:.3e} vs float32, PyTorch bf16's is {b:.3e}", r["name"]);
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
        // stage 2's sample groups to time, 0 for the whole batch
        let groups: Vec<usize> = std::env::var("STRIPS_CNN_GROUPS")
            .unwrap_or_else(|_| "0,8,16".into())
            .split(',')
            .map(|v| v.trim().parse().expect("STRIPS_CNN_GROUPS: comma-separated sizes"))
            .collect();
        let mut runs = Vec::new();
        for m in modes() {
            for &g in &groups {
                runs.extend(batches.iter().map(|&n| (m, g, n)));
            }
        }
        for ((mode_name, mode), group, n) in runs {
            cnn.set_stages01(mode);
            cnn.set_group((group > 0).then_some(group));
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
            let cells = cells_print(&s, &cnn, n)?;
            let row = json!({"mode": mode_name, "group": group, "batch": n, "reps": reps, "ms_per_decision": ms,
                             "loadavg_before": load, "loadavg_after": loadavg(), "cells_fnv": cells,
                             "algorithms": cnn.algorithms(n)});
            println!("{row}");
            results.push(row);
        }
        write_out("bench.json", &json!(results))
    }

    /// The last forward's cells as an output fingerprint: their FNV-1a 64 hash (returned), and, under
    /// `STRIPS_CNN_OUT`, every 64th value in `cells.sample` (bf16 bits) for a gate that allows arithmetic changes
    fn cells_print(s: &Setup, cnn: &CropCnn, n: usize) -> Result<String> {
        let cells: Vec<u16> = s.dev.gpu.download_range(&cnn.cells, 0, n * voxels(CELLS) * CELL_CHANNELS)?;
        let hash = cells
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3));
        if let Some(dir) = std::env::var_os("STRIPS_CNN_OUT") {
            let sample: Vec<u8> = cells.iter().step_by(64).flat_map(|v| v.to_le_bytes()).collect();
            std::fs::create_dir_all(&dir)?;
            std::fs::write(PathBuf::from(dir).join("cells.sample"), sample)?;
        }
        Ok(format!("{hash:016x}"))
    }
}
