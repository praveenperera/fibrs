//! The strip model after the CNN: cell, reference and path-geometry tokens, the four-layer context transformer,
//! the strip queries, both strip passes, and the heights and log error scales of the last (villa
//! `CropTransformer.context` minus the CNN, then `SheetStripFollower.strip_outputs`)
//!
//! Everything runs on the GPU for the whole batch: one upload of the decision records, the kernels of
//! `strips-kernels/src/head` (prefix `sm_`), one download of the heights and log scales. Numerics are PyTorch's
//! under bf16 autocast: Linear layers round input, weight and bias to bf16 and accumulate in fp32; LayerNorm reads
//! bf16 or fp32 and writes fp32; the context and first strip pass keep a bf16 residual stream, the refinement pass
//! an fp32 one; feature sampling runs in fp32 on the bf16 cells

mod layout;
#[cfg(test)]
mod tests;

use std::sync::Mutex;

use color_eyre::eyre::{Result, WrapErr, ensure, eyre};
use volkit::gpu::{Arg, Buf};

use super::{Device, Weights};
use crate::strips::types::{STRIP_COLS, STRIP_ROWS, StripRequest, StripResult};

/// Width of every token
const WIDTH: usize = 128;
/// Context tokens: cells, references (128 history + seed), path geometry
const CELLS: usize = layout::CELL_TOKENS;
const REFS: usize = 129;
const GEOMS: usize = 29;
const CONTEXT: usize = CELLS + REFS + GEOMS;
/// Strip tokens per decision, two height rows each
const STRIPS: usize = 450;
/// Key rows per decision and head: the context, then the strip tokens, padded to a tile of 64
const LDK: usize = 2048;
const LAYERS: usize = 4;
/// Floats per decision record (`strips-kernels` `head::tokens::REQ`) and its field offsets
const REQ: usize = 800;
const R_HIST: usize = 0;
const R_HMASK: usize = 384;
const R_SEED: usize = 512;
const R_SEED_MASK: usize = 515;
const R_SEED_AGE: usize = 516;
const R_SEED_TANGENT: usize = 517;
const R_GEOM: usize = 520;
const R_GEOM_VALID: usize = 752;
const R_ORIENT: usize = 781;
/// Padded input widths of the projections (`strips-kernels` `head::tokens::K_*`)
const K_SECTOR: usize = 576;
const K_TUBE: usize = 960;
const K_REF: usize = 128;
const K_GEOM: usize = 128;
const K_QUERY: usize = 128;
const K_ORIENT: usize = 64;
const K_REFINE: usize = 192;
const K_PROFILE: usize = 128;
/// Epilogue activations and token modes of `sm_linear` and `sm_linear_tok`
const ACT_NONE: u32 = 0;
const ACT_SILU: u32 = 2;
const TOK_PLAIN: u32 = 0;
const TOK_PER_TOKEN: u32 = 1;
const TOK_PER_DECISION: u32 = 2;
/// Strip heights are the readout times this
const STRIP_SCALE: f32 = 8.0;

/// `v` rounded to nearest-even bf16
pub(crate) fn bf16(v: f32) -> u16 {
    let b = v.to_bits();
    if v.is_nan() {
        return ((b >> 16) | 0x40) as u16;
    }
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// A bf16 value as f32
pub(crate) fn bf16_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// One Linear on the device: weight `[n, kp]` bf16 (input columns past the layer's zero), bias `[n]` rounded to
/// bf16 and held as f32
struct Linear {
    w: Buf,
    b: Buf,
    n: usize,
    kp: usize,
}

/// A LayerNorm's weight and bias, f32
struct Norm {
    g: Buf,
    b: Buf,
}

/// A two-Linear MLP (Linear, SiLU, Linear)
struct Mlp {
    l0: Linear,
    l2: Linear,
}

struct Layer {
    norm1: Norm,
    qkv: Linear,
    out: Linear,
    norm2: Norm,
    ffn0: Linear,
    ffn2: Linear,
}

/// The weights on the device
struct Params {
    sector: Linear,
    tube: Linear,
    reference: Mlp,
    geometry: Mlp,
    query: Mlp,
    orient: Mlp,
    refinement: Mlp,
    profile: Mlp,
    height: Linear,
    log_scale: Mlp,
    layers: Vec<Layer>,
    norm: Norm,
    /// `cell_position(cell_xyz / 16)`, bf16 `[1422, 128]`
    position: Buf,
    /// `kind[0]` and `strip_kind` rounded to bf16, `kind[1]`, `kind[2]` and `strip_stage[0]` as is, f32 `[128]`
    kind_cell: Buf,
    kind_reference: Buf,
    kind_geometry: Buf,
    kind_strip: Buf,
    stage: Buf,
}

/// The fixed sampling tables
struct Tables {
    /// Per cell-token point: first sub-point, sub-point count, token, point within the token
    points: Buf,
    npoints: usize,
    /// Sub-points, crop-local xyz
    subs: Buf,
    /// Strip anchors `[450, 3]`
    anchors: Buf,
    /// Strip columns v (25), rows f (36), profile offsets (25)
    grid: Buf,
}

/// Host memory one forward reuses: the packed decision records and the downloaded heights and log scales
struct Staging {
    records: Vec<f32>,
    heights: Vec<f32>,
    logs: Vec<f32>,
}

/// Activations, sized for the largest batch. The projection inputs' pad columns stay zero from allocation
pub(crate) struct Buffers {
    pub(crate) req: Buf,
    /// One byte per key and decision `[B, 2048]`: the cells and strip tokens valid and the padding invalid from
    /// allocation on, the reference and geometry keys rewritten by their token kernels every batch
    pub(crate) keymask: Buf,
    pub(crate) xs: Buf,
    pub(crate) xt: Buf,
    pub(crate) xr: Buf,
    pub(crate) xg: Buf,
    pub(crate) xq: Buf,
    pub(crate) xo: Buf,
    pub(crate) xf: Buf,
    pub(crate) xp: Buf,
    /// MLP hidden rows, bf16 `[max(B * 450, 1422), 128]` (the cell positions use it once at load)
    pub(crate) hid: Buf,
    /// The orientation embedding, bf16 `[B, 128]`
    pub(crate) orient: Buf,
    /// The context sequence, bf16 `[B, 1580, 128]`
    pub(crate) seq: Buf,
    /// The strip passes' final LayerNorm output, bf16 `[B, 450, 128]`
    pub(crate) ln: Buf,
    pub(crate) q: Buf,
    pub(crate) k: Vec<Buf>,
    pub(crate) v: Vec<Buf>,
    /// Attention output, merged heads, bf16 `[B, 1580, 128]`
    pub(crate) att: Buf,
    /// The first strip pass's residual stream, bf16 `[B, 450, 128]`
    pub(crate) strip1: Buf,
    /// The refinement pass's residual stream, f32 `[B, 450, 128]`
    pub(crate) strip2: Buf,
    /// Heights of both passes, f32 `[B, 450, 2]`
    pub(crate) h0: Buf,
    pub(crate) h1: Buf,
    /// The log-scale MLP's hidden, bf16 `[B, 450, 64]` (two columns used), and its output f32 `[B, 450, 2]`
    pub(crate) ls_hid: Buf,
    pub(crate) logs: Buf,
}

/// The strip model's head on one device, for batches of up to `max_batch` decisions
pub struct Head {
    params: Params,
    tables: Tables,
    pub(crate) bufs: Buffers,
    /// Held for a whole forward, which also keeps two forwards from sharing the device buffers at once
    staging: Mutex<Staging>,
    max_batch: usize,
}

fn round_bf16(values: &[f32]) -> Vec<f32> {
    values.iter().map(|&v| bf16_f32(bf16(v))).collect()
}

impl Linear {
    /// Linear `name` (`[n, k]` weight), its input padded to `kp` columns
    fn load(dev: &Device, w: &Weights, name: &str, n: usize, k: usize, kp: usize) -> Result<Self> {
        let weight = w.get(&format!("{name}.weight"), &[n, k])?;
        let bias = w.get(&format!("{name}.bias"), &[n])?;
        let mut padded = vec![0u16; n * kp];
        for (row, src) in padded.chunks_exact_mut(kp).zip(weight.values.chunks_exact(k)) {
            for (d, &s) in row.iter_mut().zip(src) {
                *d = bf16(s);
            }
        }
        Ok(Self { w: dev.gpu.upload(&padded)?, b: dev.gpu.upload(&round_bf16(&bias.values))?, n, kp })
    }
}

impl Norm {
    fn load(dev: &Device, w: &Weights, name: &str) -> Result<Self> {
        let g = w.get(&format!("{name}.weight"), &[WIDTH])?;
        let b = w.get(&format!("{name}.bias"), &[WIDTH])?;
        Ok(Self { g: dev.gpu.upload(&g.values)?, b: dev.gpu.upload(&b.values)? })
    }
}

impl Mlp {
    fn load(dev: &Device, w: &Weights, name: &str, (k, kp): (usize, usize), hidden: usize, out: usize) -> Result<Self> {
        let l0 = Linear::load(dev, w, &format!("{name}.0"), hidden, k, kp)?;
        let l2 = Linear::load(dev, w, &format!("{name}.2"), out, hidden, hidden.next_multiple_of(64))?;
        Ok(Self { l0, l2 })
    }
}

fn vector(dev: &Device, w: &Weights, name: &str, shape: &[usize], row: usize, bf: bool) -> Result<Buf> {
    let t = w.get(name, shape)?;
    let v = &t.values[row * WIDTH..(row + 1) * WIDTH];
    dev.gpu.upload(&if bf { round_bf16(v) } else { v.to_vec() })
}

impl Head {
    /// Uploads the weights and tables and allocates every buffer for batches up to `max_batch`; runs the
    /// batch-independent cell-position MLP once
    pub fn new(dev: &Device, w: &Weights, max_batch: usize) -> Result<Self> {
        ensure!(max_batch > 0, "a head needs a batch of at least one decision");
        let p = |name: &str| format!("strip.{name}");
        let layers = (0..LAYERS)
            .map(|i| -> Result<Layer> {
                let l = |name: &str| p(&format!("layers.{i}.{name}"));
                Ok(Layer {
                    norm1: Norm::load(dev, w, &l("norm1"))?,
                    qkv: Linear::load(dev, w, &l("qkv"), 3 * WIDTH, WIDTH, WIDTH)?,
                    out: Linear::load(dev, w, &l("out"), WIDTH, WIDTH, WIDTH)?,
                    norm2: Norm::load(dev, w, &l("norm2"))?,
                    ffn0: Linear::load(dev, w, &l("ffn.0"), 2 * WIDTH, WIDTH, WIDTH)?,
                    ffn2: Linear::load(dev, w, &l("ffn.2"), WIDTH, 2 * WIDTH, 2 * WIDTH)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let params = Params {
            sector: Linear::load(dev, w, &p("cell_token.sector_projection"), WIDTH, 520, K_SECTOR)?,
            tube: Linear::load(dev, w, &p("cell_token.tube_projection"), WIDTH, 910, K_TUBE)?,
            reference: Mlp::load(dev, w, &p("reference_token"), (72, K_REF), WIDTH, WIDTH)?,
            geometry: Mlp::load(dev, w, &p("path_geometry.embed"), (68, K_GEOM), WIDTH, WIDTH)?,
            query: Mlp::load(dev, w, &p("strip_query"), (67, K_QUERY), WIDTH, WIDTH)?,
            orient: Mlp::load(dev, w, &p("strip_orient"), (5, K_ORIENT), WIDTH, WIDTH)?,
            refinement: Mlp::load(dev, w, &p("strip_refinement"), (132, K_REFINE), WIDTH, WIDTH)?,
            profile: Mlp::load(dev, w, &p("strip_profile"), (100, K_PROFILE), WIDTH, WIDTH)?,
            height: Linear::load(dev, w, &p("strip_height"), 2, WIDTH, WIDTH)?,
            log_scale: Mlp::load(dev, w, &p("strip_log_scale"), (WIDTH, WIDTH), 2, 2)?,
            layers,
            norm: Norm::load(dev, w, &p("norm"))?,
            position: dev.gpu.alloc_fill(CELLS * WIDTH * 2, 0)?,
            kind_cell: vector(dev, w, &p("kind"), &[5, WIDTH], 0, true)?,
            kind_reference: vector(dev, w, &p("kind"), &[5, WIDTH], 1, false)?,
            kind_geometry: vector(dev, w, &p("kind"), &[5, WIDTH], 2, false)?,
            kind_strip: vector(dev, w, &p("strip_kind"), &[WIDTH], 0, true)?,
            stage: vector(dev, w, &p("strip_stage.weight"), &[1, WIDTH], 0, false)?,
        };

        let cells = layout::cells();
        let mut points = Vec::with_capacity(cells.points.len() * 4);
        let mut subs: Vec<f32> = Vec::new();
        for pt in &cells.points {
            points.extend([(subs.len() / 3) as u32, pt.subs.len() as u32, pt.token, pt.index]);
            subs.extend(pt.subs.iter().flatten());
        }
        let strip = layout::strip();
        let grid: Vec<f32> = strip.v.iter().chain(&strip.f).chain(&strip.offsets).copied().collect();
        ensure!(grid.len() == 86 && strip.anchors.len() == STRIPS, "the strip grid's shape");
        let mut keymask = vec![0u8; LDK];
        keymask[..CELLS].fill(1);
        keymask[CONTEXT..CONTEXT + STRIPS].fill(1);
        let tables = Tables {
            points: dev.gpu.upload(&points)?,
            npoints: cells.points.len(),
            subs: dev.gpu.upload(&subs)?,
            anchors: dev.gpu.upload(&strip.anchors.concat())?,
            grid: dev.gpu.upload(&grid)?,
        };

        let b = max_batch;
        let zeros = |bytes: usize| dev.gpu.alloc_fill(bytes, 0);
        let kv = || (0..LAYERS).map(|_| zeros(b * 4 * LDK * 32 * 2)).collect::<Result<Vec<_>>>();
        let bufs = Buffers {
            req: zeros(b * REQ * 4)?,
            keymask: dev.gpu.upload(&keymask.repeat(b))?,
            xs: zeros(b * layout::SECTOR_TOKENS * K_SECTOR * 2)?,
            xt: zeros(b * layout::TUBE_TOKENS * K_TUBE * 2)?,
            xr: zeros(b * REFS * K_REF * 2)?,
            xg: zeros(b * GEOMS * K_GEOM * 2)?,
            xq: zeros(b * STRIPS * K_QUERY * 2)?,
            xo: zeros(b * K_ORIENT * 2)?,
            xf: zeros(b * STRIPS * K_REFINE * 2)?,
            xp: zeros(b * STRIPS * K_PROFILE * 2)?,
            hid: zeros((b * STRIPS).max(CELLS) * WIDTH * 2)?,
            orient: zeros(b * WIDTH * 2)?,
            seq: zeros(b * CONTEXT * WIDTH * 2)?,
            ln: zeros(b * STRIPS * WIDTH * 2)?,
            q: zeros(b * 4 * CONTEXT * 32 * 2)?,
            k: kv()?,
            v: kv()?,
            att: zeros(b * CONTEXT * WIDTH * 2)?,
            strip1: zeros(b * STRIPS * WIDTH * 2)?,
            strip2: zeros(b * STRIPS * WIDTH * 4)?,
            h0: zeros(b * STRIPS * 2 * 4)?,
            h1: zeros(b * STRIPS * 2 * 4)?,
            ls_hid: zeros(b * STRIPS * 64 * 2)?,
            logs: zeros(b * STRIPS * 2 * 4)?,
        };
        let staging = Staging {
            records: vec![0.0; b * REQ],
            heights: Vec::with_capacity(b * STRIPS * 2),
            logs: Vec::with_capacity(b * STRIPS * 2),
        };
        let head = Self { params, tables, bufs, staging: Mutex::new(staging), max_batch };

        // cell_position(cell_xyz / 16) through the same kernels: the centres / 16 as bf16 rows of 64
        let mut xyz = vec![0u16; CELLS * 64];
        for (row, c) in xyz.as_chunks_mut::<64>().0.iter_mut().zip(&cells.centres) {
            for (d, &s) in row.iter_mut().zip(c) {
                *d = bf16(s / 16.0);
            }
        }
        let xyz = dev.gpu.upload(&xyz)?;
        let mlp = Mlp::load(dev, w, &p("cell_position"), (3, 64), WIDTH, WIDTH)?;
        head.linear(dev, (&xyz, 64), CELLS, &mlp.l0, (&head.bufs.hid, WIDTH), ACT_SILU)?;
        head.linear(dev, (&head.bufs.hid, WIDTH), CELLS, &mlp.l2, (&head.params.position, WIDTH), ACT_NONE)?;
        dev.gpu.sync().wrap_err("compute the cell positions")?;
        Ok(head)
    }

    /// The largest batch the buffers hold
    pub fn max_batch(&self) -> usize {
        self.max_batch
    }

    /// Heights and log error scales of the last strip pass for each request: `cells` bf16 NDHWC `(B, 36, 32, 32,
    /// 64)`, `crop` f32 NDHWC `(B, 144, 128, 128, 2)` (CT, tube), B = `requests.len()`
    pub fn forward(
        &self,
        dev: &Device,
        cells: &Buf,
        crop: &Buf,
        requests: &[StripRequest],
    ) -> Result<Vec<StripResult>> {
        self.forward_traced(dev, cells, crop, requests, &mut |_, _| Ok(()))
    }

    /// [`Head::forward`], calling `trace(stage, self)` after each stage once its kernels are queued (the parity
    /// tests download the stage's buffers there)
    pub(crate) fn forward_traced(
        &self,
        dev: &Device,
        cells: &Buf,
        crop: &Buf,
        requests: &[StripRequest],
        trace: &mut dyn FnMut(&str, &Self) -> Result<()>,
    ) -> Result<Vec<StripResult>> {
        let b = requests.len();
        ensure!(b > 0 && b <= self.max_batch, "a batch of {b} decisions, the head holds 1 to {}", self.max_batch);
        ensure!(cells.bytes() >= b * 36 * 32 * 32 * 64 * 2, "cells hold fewer than {b} decisions");
        ensure!(crop.bytes() >= b * 144 * 128 * 128 * 2 * 4, "the crop holds fewer than {b} decisions");
        let (p, t, x) = (&self.params, &self.tables, &self.bufs);
        let mut staging = self.staging.lock().map_err(|_| eyre!("a forward on this head panicked"))?;
        let Staging { records, heights, logs } = &mut *staging;
        for (r, q) in records.as_chunks_mut::<REQ>().0.iter_mut().zip(requests) {
            pack(r, q);
        }
        dev.gpu.upload_into(&x.req, 0, &records[..b * REQ])?;
        let bu = b as u32;

        // tokens
        let warps = |n: usize| (n.div_ceil(4) as u32, bu, 1);
        let k = &dev.kernels;
        k.launch(
            &dev.gpu,
            "sm_cell_tokens",
            warps(t.npoints),
            (128, 1, 1),
            0,
            &[
                x.xs.arg::<u16>(),
                x.xt.arg::<u16>(),
                cells.arg::<u32>(),
                t.points.arg::<u32>(),
                t.subs.arg::<f32>(),
                Arg::U32(t.npoints as u32),
            ],
        )?;
        let inv_log2049 = 1.0f32 / (2049f64.ln() as f32);
        k.launch(
            &dev.gpu,
            "sm_ref_tokens",
            warps(REFS),
            (128, 1, 1),
            0,
            &[x.xr.arg::<u16>(), x.keymask.arg::<u8>(), cells.arg::<u32>(), x.req.arg::<f32>(), Arg::F32(inv_log2049)],
        )?;
        let inv_log1p1024 = 1.0f32 / (1024f64.ln_1p() as f32);
        k.launch(
            &dev.gpu,
            "sm_geom_tokens",
            warps(GEOMS),
            (128, 1, 1),
            0,
            &[x.xg.arg::<u16>(), x.keymask.arg::<u8>(), x.req.arg::<f32>(), Arg::F32(inv_log1p1024)],
        )?;
        k.launch(
            &dev.gpu,
            "sm_query_tokens",
            warps(STRIPS),
            (128, 1, 1),
            0,
            &[x.xq.arg::<u16>(), x.xo.arg::<u16>(), cells.arg::<u32>(), x.req.arg::<f32>(), t.anchors.arg::<f32>()],
        )?;
        trace("token_inputs", self)?;

        let sectors = b * layout::SECTOR_TOKENS;
        let tok = |a: (&Buf, usize), m: usize, lin: &Linear, tab: Arg, vec: &Buf, (rpd, off): (usize, usize), mode| {
            self.tokens(dev, a, m, lin, tab, vec, None, (rpd, off), mode)
        };
        let position = p.position.arg::<u32>();
        tok((&x.xs, K_SECTOR), sectors, &p.sector, position, &p.kind_cell, (layout::SECTOR_TOKENS, 0), TOK_PER_TOKEN)?;
        let tubes = b * layout::TUBE_TOKENS;
        let tube_rows = (layout::TUBE_TOKENS, layout::SECTOR_TOKENS);
        tok((&x.xt, K_TUBE), tubes, &p.tube, position, &p.kind_cell, tube_rows, TOK_PER_TOKEN)?;
        trace("cell_tokens", self)?;
        self.linear(dev, (&x.xr, K_REF), b * REFS, &p.reference.l0, (&x.hid, WIDTH), ACT_SILU)?;
        let l2 = &p.reference.l2;
        tok((&x.hid, WIDTH), b * REFS, l2, Arg::EMPTY, &p.kind_reference, (REFS, CELLS), TOK_PLAIN)?;
        self.linear(dev, (&x.xg, K_GEOM), b * GEOMS, &p.geometry.l0, (&x.hid, WIDTH), ACT_SILU)?;
        let flag = Some((&x.keymask, CELLS + REFS));
        let rows = (GEOMS, CELLS + REFS);
        let l2 = &p.geometry.l2;
        self.tokens(dev, (&x.hid, WIDTH), b * GEOMS, l2, Arg::EMPTY, &p.kind_geometry, flag, rows, TOK_PLAIN)?;
        trace("seq0", self)?;

        // the context pass: layers 0-2 in full, layer 3 only its keys and values
        let m = b * CONTEXT;
        for (i, layer) in p.layers.iter().enumerate() {
            let n0 = if i == LAYERS - 1 { WIDTH } else { 0 };
            self.ln_qkv(dev, Stream::Bf16(&x.seq), m, (CONTEXT, 0, n0), layer, i)?;
            if i == LAYERS - 1 {
                break;
            }
            self.attention(dev, b, CONTEXT, CONTEXT, i)?;
            self.residual(dev, Stream::Bf16(&x.seq), (&x.att, WIDTH), m, &layer.out, None)?;
            self.ffn(dev, Stream::Bf16(&x.seq), m, layer)?;
            trace(["ctx0", "ctx1", "ctx2"][i], self)?;
        }
        trace("context", self)?;

        // strip queries: strip_query(evidence, anchor) + strip_orient(orientation) + strip_kind
        self.linear(dev, (&x.xo, K_ORIENT), b, &p.orient.l0, (&x.hid, WIDTH), ACT_SILU)?;
        self.linear(dev, (&x.hid, WIDTH), b, &p.orient.l2, (&x.orient, WIDTH), ACT_NONE)?;
        let ms = b * STRIPS;
        self.linear(dev, (&x.xq, K_QUERY), ms, &p.query.l0, (&x.hid, WIDTH), ACT_SILU)?;
        self.tokens(
            dev,
            (&x.hid, WIDTH),
            ms,
            &p.query.l2,
            x.orient.arg::<u32>(),
            &p.kind_strip,
            None,
            (STRIPS, 0),
            TOK_PER_DECISION,
        )
        .wrap_err("strip tokens")?;
        trace("tokens1", self)?;

        // first strip pass, bf16 residual; the final norm writes the refinement pass's f32 stream
        self.strip_pass(dev, b, Stream::Bf16(&x.strip1), &mut *trace, 1)?;
        self.layernorm(dev, Stream::Bf16(&x.strip1), ms, &p.norm, Some(&x.ln), Some(&x.strip2))?;
        self.heights(dev, ms, &x.h0)?;
        trace("decoded1", self)?;

        // refinement: evidence at the first heights and the crop's profiles along u
        k.launch(
            &dev.gpu,
            "sm_refine_tokens",
            warps(STRIPS),
            (128, 1, 1),
            0,
            &[
                x.xf.arg::<u16>(),
                x.xp.arg::<u16>(),
                cells.arg::<u32>(),
                crop.arg::<f32>(),
                x.h0.arg::<f32>(),
                t.grid.arg::<f32>(),
            ],
        )?;
        self.linear(dev, (&x.xf, K_REFINE), ms, &p.refinement.l0, (&x.hid, WIDTH), ACT_SILU)?;
        self.residual(dev, Stream::F32(&x.strip2), (&x.hid, WIDTH), ms, &p.refinement.l2, Some(&p.stage))?;
        self.linear(dev, (&x.xp, K_PROFILE), ms, &p.profile.l0, (&x.hid, WIDTH), ACT_SILU)?;
        self.residual(dev, Stream::F32(&x.strip2), (&x.hid, WIDTH), ms, &p.profile.l2, None)?;
        trace("tokens2", self)?;

        self.strip_pass(dev, b, Stream::F32(&x.strip2), &mut *trace, 2)?;
        self.layernorm(dev, Stream::F32(&x.strip2), ms, &p.norm, Some(&x.ln), None)?;
        self.heights(dev, ms, &x.h1)?;
        self.linear(dev, (&x.ln, WIDTH), ms, &p.log_scale.l0, (&x.ls_hid, 64), ACT_SILU)?;
        self.linear_f32(dev, (&x.ls_hid, 64), ms, &p.log_scale.l2, &x.logs, 1.0)?;
        trace("decoded2", self)?;

        dev.gpu.download_range_into(&x.h1, 0, ms * 2, heights)?;
        dev.gpu.download_range_into(&x.logs, 0, ms * 2, logs)?;
        Ok(heights
            .as_chunks::<{ STRIPS * 2 }>()
            .0
            .iter()
            .zip(logs.as_chunks::<{ STRIPS * 2 }>().0)
            .map(|(h, l)| result(h, l))
            .collect())
    }

    /// One strip pass's four layers over the strip tokens in `stream`, each reading its context keys and values
    /// plus the strip tokens' own
    fn strip_pass(
        &self,
        dev: &Device,
        b: usize,
        stream: Stream,
        trace: &mut dyn FnMut(&str, &Self) -> Result<()>,
        pass: usize,
    ) -> Result<()> {
        let x = &self.bufs;
        let ms = b * STRIPS;
        for (i, layer) in self.params.layers.iter().enumerate() {
            self.ln_qkv(dev, stream, ms, (STRIPS, CONTEXT, 0), layer, i)?;
            self.attention(dev, b, STRIPS, CONTEXT + STRIPS, i)?;
            self.residual(dev, stream, (&x.att, WIDTH), ms, &layer.out, None)?;
            self.ffn(dev, stream, ms, layer)?;
            trace(["p1l0", "p1l1", "p1l2", "p1l3", "p2l0", "p2l1", "p2l2", "p2l3"][(pass - 1) * 4 + i], self)?;
        }
        Ok(())
    }

    /// The strip heights of the normalized states in `ln`: `8 * bf16(strip_height(x))` into `out` `[m, 2]`
    fn heights(&self, dev: &Device, m: usize, out: &Buf) -> Result<()> {
        self.linear_f32(dev, (&self.bufs.ln, WIDTH), m, &self.params.height, out, STRIP_SCALE)
    }

    /// `out = act(bf16(a @ w^T + b))`, bf16, `m` rows of `a` (row stride `lda`) into rows of `ldo`
    fn linear(
        &self,
        dev: &Device,
        (a, lda): (&Buf, usize),
        m: usize,
        l: &Linear,
        (out, ldo): (&Buf, usize),
        act: u32,
    ) -> Result<()> {
        dev.kernels.launch(
            &dev.gpu,
            "sm_linear",
            grid(m, l.n),
            (128, 1, 1),
            0,
            &[
                out.arg::<u32>(),
                a.arg::<u16>(),
                l.w.arg::<u16>(),
                l.b.arg::<f32>(),
                Arg::U32(m as u32),
                Arg::U32(l.n as u32),
                Arg::U32(l.kp as u32),
                Arg::U32(lda as u32),
                Arg::U32(ldo as u32),
                Arg::U32(act),
            ],
        )
    }

    /// `out = scale * bf16(a @ w^T + b)`, f32 `[m, n]`
    fn linear_f32(
        &self,
        dev: &Device,
        (a, lda): (&Buf, usize),
        m: usize,
        l: &Linear,
        out: &Buf,
        scale: f32,
    ) -> Result<()> {
        dev.kernels.launch(
            &dev.gpu,
            "sm_linear_f32",
            grid(m, l.n),
            (128, 1, 1),
            0,
            &[
                out.arg::<f32>(),
                a.arg::<u16>(),
                l.w.arg::<u16>(),
                l.b.arg::<f32>(),
                Arg::U32(m as u32),
                Arg::U32(l.n as u32),
                Arg::U32(l.kp as u32),
                Arg::U32(lda as u32),
                Arg::U32(l.n as u32),
                Arg::F32(scale),
            ],
        )
    }

    /// A residual branch's projection added to `stream` in place (then `vec` per column, f32 streams only)
    fn residual(
        &self,
        dev: &Device,
        stream: Stream,
        (a, lda): (&Buf, usize),
        m: usize,
        l: &Linear,
        vec: Option<&Buf>,
    ) -> Result<()> {
        let common = [a.arg::<u16>(), l.w.arg::<u16>(), l.b.arg::<f32>()];
        let sizes = [
            Arg::U32(m as u32),
            Arg::U32(l.n as u32),
            Arg::U32(l.kp as u32),
            Arg::U32(lda as u32),
            Arg::U32(WIDTH as u32),
        ];
        match stream {
            Stream::Bf16(res) => {
                ensure!(vec.is_none(), "a bf16 residual takes no vector");
                let args: Vec<Arg> = [res.arg::<u32>()].into_iter().chain(common).chain(sizes).collect();
                dev.kernels.launch(&dev.gpu, "sm_linear_res", grid(m, l.n), (128, 1, 1), 0, &args)
            }
            Stream::F32(res) => {
                let vec = vec.map_or(Arg::EMPTY, Buf::arg::<f32>);
                let args: Vec<Arg> = [res.arg::<f32>()].into_iter().chain(common).chain([vec]).chain(sizes).collect();
                dev.kernels.launch(&dev.gpu, "sm_linear_resf", grid(m, l.n), (128, 1, 1), 0, &args)
            }
        }
    }

    /// A token family's last Linear into its rows of the context sequence (or the strip tokens)
    #[allow(clippy::too_many_arguments)]
    fn tokens(
        &self,
        dev: &Device,
        (a, lda): (&Buf, usize),
        m: usize,
        l: &Linear,
        tab: Arg,
        vec: &Buf,
        flag: Option<(&Buf, usize)>,
        (rpd, off): (usize, usize),
        mode: u32,
    ) -> Result<()> {
        let (out, orpd) =
            if mode == TOK_PER_DECISION { (&self.bufs.strip1, STRIPS) } else { (&self.bufs.seq, CONTEXT) };
        let (flag, flag_off) = flag.map_or((Arg::EMPTY, 0), |(f, o)| (f.arg::<u8>(), o));
        dev.kernels.launch(
            &dev.gpu,
            "sm_linear_tok",
            grid(m, WIDTH),
            (128, 1, 1),
            0,
            &[
                out.arg::<u32>(),
                a.arg::<u16>(),
                l.w.arg::<u16>(),
                l.b.arg::<f32>(),
                tab,
                vec.arg::<f32>(),
                flag,
                Arg::U32(m as u32),
                Arg::U32(l.kp as u32),
                Arg::U32(lda as u32),
                Arg::U32(rpd as u32),
                Arg::U32(orpd as u32),
                Arg::U32(off as u32),
                Arg::U32(mode),
                Arg::U32(LDK as u32),
                Arg::U32(flag_off as u32),
            ],
        )
    }

    /// `norm1` of `m` rows of `stream` (`ntok` per decision), then the qkv projection into q and rows `koff..` of
    /// layer `index`'s keys and values; columns from `n0` (128: keys and values only)
    fn ln_qkv(
        &self,
        dev: &Device,
        stream: Stream,
        m: usize,
        (ntok, koff, n0): (usize, usize, usize),
        layer: &Layer,
        index: usize,
    ) -> Result<()> {
        let x = &self.bufs;
        let (xb, xf) = stream.args();
        dev.kernels.launch(
            &dev.gpu,
            "sm_ln_qkv",
            (m.div_ceil(64) as u32, 1, 1),
            (128, 1, 1),
            0,
            &[
                x.q.arg::<u32>(),
                x.k[index].arg::<u32>(),
                x.v[index].arg::<u32>(),
                xb,
                xf,
                layer.norm1.g.arg::<f32>(),
                layer.norm1.b.arg::<f32>(),
                layer.qkv.w.arg::<u16>(),
                layer.qkv.b.arg::<f32>(),
                Arg::U32(m as u32),
                Arg::U32(ntok as u32),
                Arg::U32(LDK as u32),
                Arg::U32(koff as u32),
                Arg::U32(n0 as u32),
            ],
        )
    }

    /// The FFN sublayer on `m` rows of `stream` in place: `x += ffn.2(gelu(ffn.0(norm2(x))))`
    fn ffn(&self, dev: &Device, stream: Stream, m: usize, layer: &Layer) -> Result<()> {
        let (xb, xf) = stream.args();
        dev.kernels.launch(
            &dev.gpu,
            "sm_ffn",
            (m.div_ceil(64) as u32, 1, 1),
            (128, 1, 1),
            0,
            &[
                xb,
                xf,
                layer.norm2.g.arg::<f32>(),
                layer.norm2.b.arg::<f32>(),
                layer.ffn0.w.arg::<u16>(),
                layer.ffn0.b.arg::<f32>(),
                layer.ffn2.w.arg::<u16>(),
                layer.ffn2.b.arg::<f32>(),
                Arg::U32(m as u32),
            ],
        )
    }

    /// Attention of `nq` queries per decision over the first `nk` keys of layer `layer` into `att`
    fn attention(&self, dev: &Device, b: usize, nq: usize, nk: usize, layer: usize) -> Result<()> {
        let x = &self.bufs;
        let scale_log2 = std::f32::consts::LOG2_E / (32f32).sqrt();
        dev.kernels.launch(
            &dev.gpu,
            "sm_attn",
            (nq.div_ceil(64) as u32, 4, b as u32),
            (128, 1, 1),
            0,
            &[
                x.att.arg::<u32>(),
                x.q.arg::<u16>(),
                x.k[layer].arg::<u16>(),
                x.v[layer].arg::<u16>(),
                x.keymask.arg::<u8>(),
                Arg::U32(nq as u32),
                Arg::U32(nk as u32),
                Arg::U32(LDK as u32),
                Arg::F32(scale_log2),
            ],
        )
    }

    /// LayerNorm of `m` rows of `stream` into `bf` (bf16) and/or `f32` (f32)
    fn layernorm(
        &self,
        dev: &Device,
        stream: Stream,
        m: usize,
        n: &Norm,
        bf: Option<&Buf>,
        f32: Option<&Buf>,
    ) -> Result<()> {
        let (xb, xf) = stream.args();
        dev.kernels.launch(
            &dev.gpu,
            "sm_layernorm",
            (m.div_ceil(4) as u32, 1, 1),
            (128, 1, 1),
            0,
            &[
                bf.map_or(Arg::EMPTY, Buf::arg::<u32>),
                f32.map_or(Arg::EMPTY, Buf::arg::<f32>),
                xb,
                xf,
                n.g.arg::<f32>(),
                n.b.arg::<f32>(),
                Arg::U32(m as u32),
            ],
        )
    }
}

/// A residual stream: bf16 (the context and the first strip pass) or f32 (the refinement pass)
#[derive(Clone, Copy)]
enum Stream<'a> {
    Bf16(&'a Buf),
    F32(&'a Buf),
}

impl Stream<'_> {
    /// The stream as a kernel's `(bf16 pairs, fp32)` argument pair, the other one empty
    fn args(self) -> (Arg, Arg) {
        match self {
            Stream::Bf16(s) => (s.arg::<u32>(), Arg::EMPTY),
            Stream::F32(s) => (Arg::EMPTY, s.arg::<f32>()),
        }
    }
}

fn grid(m: usize, n: usize) -> (u32, u32, u32) {
    (m.div_ceil(64) as u32, n.div_ceil(64) as u32, 1)
}

/// One decision's record for the token kernels
fn pack(r: &mut [f32], q: &StripRequest) {
    r[R_HIST..R_HIST + 384].copy_from_slice(q.hist.as_flattened());
    r[R_HMASK..R_HMASK + 128].copy_from_slice(&q.hmask);
    r[R_SEED..R_SEED + 3].copy_from_slice(&q.seed);
    r[R_SEED_MASK] = q.seed_mask;
    r[R_SEED_AGE] = q.seed_age;
    r[R_SEED_TANGENT..R_SEED_TANGENT + 3].copy_from_slice(&q.seed_tangent);
    r[R_GEOM..R_GEOM + 232].copy_from_slice(q.path_geometry.as_flattened());
    for (d, &v) in r[R_GEOM_VALID..R_GEOM_VALID + GEOMS].iter_mut().zip(&q.path_geometry_valid) {
        *d = if v { 1.0 } else { 0.0 };
    }
    r[R_ORIENT..R_ORIENT + 5].copy_from_slice(&q.orientation);
}

/// Token-major `[450, 2]` heights and log scales to the strip grid: row `2 slab + n`, column `col` of token `slab
/// * 25 + col`
fn result(h: &[f32], l: &[f32]) -> StripResult {
    let mut out = StripResult { heights: [[0.0; STRIP_COLS]; STRIP_ROWS], log_scale: [[0.0; STRIP_COLS]; STRIP_ROWS] };
    for t in 0..STRIPS {
        let (slab, col) = (t / STRIP_COLS, t % STRIP_COLS);
        for n in 0..2 {
            out.heights[2 * slab + n][col] = h[t * 2 + n];
            out.log_scale[2 * slab + n][col] = l[t * 2 + n];
        }
    }
    out
}
