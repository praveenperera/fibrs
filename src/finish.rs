//! `tracing.finish`: split raw curves where predictions disagree, smooth, resample and re-validate
//!
//! `finish` treats every raw curve on its own and reads only immutable fields (pooled predictions,
//! presence, native predictions), so curves are finished in parallel once tracing is done. Sampling
//! reproduces `scipy.ndimage.map_coordinates(order=1, mode="nearest")` (`compressed.sample_tracing`)
//! and `storage.NativeSampler.sample` exactly

use std::collections::{BTreeMap, HashMap};
use std::hash::Hash;
use std::sync::{Arc, Mutex};

use color_eyre::eyre::Result;

use crate::arena::BufferPool;
use crate::fields::Presence;
use crate::input::{NativeChunk, PackedChunk, Predictions};
use crate::kernels::{evaluate_curve, fit_curve};
use crate::vol::{Shape, Vol};

/// Planes of a chunk decoded together: in prediction zarrs one Blosc block of each channel holds 8 planes of
/// 128 × 128, so a slab is the least c-blosc can decode
const SLAB_PLANES: usize = 8;

/// The cache's unit: a native chunk's index z, y, x and the index of the slab of planes within it
type SlabKey = [usize; 4];

/// Decoded native predictions for a box of voxels, both channels, in a buffer of the cache's pool
struct Slab<'p> {
    /// The box's first voxel, z, y, x
    lo: Shape,
    /// Elements between consecutive planes and rows
    stride: [usize; 2],
    /// Offset of the H channel
    half: usize,
    data: Vec<u8, &'p BufferPool>,
}

impl<'p> Slab<'p> {
    fn from_chunk(chunk: NativeChunk<&'p BufferPool>) -> Self {
        let [_, dy, dx] = chunk.dims;
        Self { lo: chunk.lo, stride: [dy * dx, dx], half: chunk.dims.iter().product(), data: chunk.data }
    }

    /// V and H at a voxel inside the box
    #[inline]
    fn value(&self, z: usize, y: usize, x: usize) -> [u8; 2] {
        let at = (z - self.lo[0]) * self.stride[0] + (y - self.lo[1]) * self.stride[1] + (x - self.lo[2]);
        [self.data[at], self.data[self.half + at]]
    }
}

/// Bytes charged for an entry's own bookkeeping (its `Arc`, map and order slots), so entries that hold no data,
/// such as chunks missing from a sparse array, still count against the budget
const ENTRY_BYTES: usize = 128;

/// A byte-bounded map that drops its least recently used entries first; one entry is always kept
struct Lru<K, V> {
    map: HashMap<K, (V, u64, usize)>,
    /// Entries by the clock of their last use, oldest first
    order: BTreeMap<u64, K>,
    budget: usize,
    bytes: usize,
    clock: u64,
}

impl<K: Copy + Eq + Hash, V: Clone> Lru<K, V> {
    fn new(budget: usize) -> Self {
        Self { map: HashMap::new(), order: BTreeMap::new(), budget, bytes: 0, clock: 0 }
    }

    fn get(&mut self, key: &K) -> Option<V> {
        self.clock += 1;
        let (value, used, _) = self.map.get_mut(key)?;
        self.order.remove(used);
        *used = self.clock;
        self.order.insert(self.clock, *key);
        Some(value.clone())
    }

    fn insert(&mut self, key: K, value: V, bytes: usize) {
        let bytes = bytes + ENTRY_BYTES;
        self.clock += 1;
        if let Some((_, used, old)) = self.map.insert(key, (value, self.clock, bytes)) {
            self.order.remove(&used);
            self.bytes -= old;
        }
        self.order.insert(self.clock, key);
        self.bytes += bytes;
        while self.bytes > self.budget && self.map.len() > 1 {
            let (_, oldest) = self.order.pop_first().expect("an entry per map entry");
            let (_, _, gone) = self.map.remove(&oldest).expect("present");
            self.bytes -= gone;
        }
    }
}

/// A bounded cache of decoded native predictions shared by the finishing threads, least recently used first out
///
/// A point's value needs 8 voxels, so when the array allows ([`Predictions::slab_layout`]) the cache decodes
/// slabs of [`SLAB_PLANES`] planes of a chunk instead of whole chunks, from the chunk's compressed bytes, which it
/// keeps in a smaller cache of their own. A curve across a chunk needs only the slabs it passes through: on a
/// dense PHerc0358 crop whole-chunk decoding took about 87% of the finishing threads' time. Other arrays are
/// decoded a whole chunk at a time.
///
/// Dense windows evict and decode hundreds of times a second, so every buffer comes from `pool`: an evicted
/// slab's buffer goes back to it once the last finishing thread lets go of the slab, and the next decode takes it
pub struct NativeCache<'a, 'p> {
    pred: &'a Predictions,
    edge: Shape,
    /// Planes per slab; a whole chunk when the array cannot be decoded by planes
    planes: usize,
    pool: &'p BufferPool,
    slabs: Mutex<Lru<SlabKey, Arc<Slab<'p>>>>,
    packed: Mutex<Lru<Shape, Arc<PackedChunk<&'p BufferPool>>>>,
    counts: Mutex<Counts>,
}

#[derive(Default)]
struct Counts {
    decoded: u64,
    read: u64,
}

impl<'a, 'p> NativeCache<'a, 'p> {
    /// `budget_bytes` bounds the decoded slabs and compressed chunks held at once, counted as the pool buffers they
    /// occupy; one of each is always kept. Freed buffers the pool keeps for reuse are outside the budget
    pub fn new(pred: &'a Predictions, budget_bytes: usize, pool: &'p BufferPool) -> Self {
        // whole Zarr chunks are the unit of reading; NPY input uses the reference's 64³ blocks
        let edge = pred.chunk_shape().unwrap_or([64, 64, 64]);
        let sliced = pred.slab_layout().is_some_and(|e| e[0] % SLAB_PLANES == 0) && pred.blocks_fit_planes(SLAB_PLANES);
        let planes = if sliced { SLAB_PLANES } else { edge[0] };
        // compressed chunks are about a fifth of their decoded size, so a quarter of the budget keeps the chunks
        // of several times as many slabs as the rest holds
        let packed = if sliced { budget_bytes / 4 } else { 0 };
        Self {
            pred,
            edge,
            planes,
            pool,
            slabs: Mutex::new(Lru::new(budget_bytes - packed)),
            packed: Mutex::new(Lru::new(packed)),
            counts: Mutex::default(),
        }
    }

    /// The chunk grid key of a native voxel
    pub fn key(&self, z: usize, y: usize, x: usize) -> Shape {
        [z / self.edge[0], y / self.edge[1], x / self.edge[2]]
    }

    fn slab_key(&self, z: usize, y: usize, x: usize) -> SlabKey {
        let [cz, cy, cx] = self.key(z, y, x);
        [cz, cy, cx, (z % self.edge[0]) / self.planes]
    }

    /// Bytes of the pool buffers held by decoded slabs and compressed chunks now
    pub fn bytes(&self) -> usize {
        self.slabs.lock().expect("cache lock").bytes + self.packed.lock().expect("cache lock").bytes
    }

    /// Slabs decoded so far, counting decodes again after eviction
    pub fn decoded(&self) -> u64 {
        self.counts.lock().expect("cache lock").decoded
    }

    /// Chunk files read so far, counting reads again after eviction
    pub fn read(&self) -> u64 {
        self.counts.lock().expect("cache lock").read
    }

    fn slab(&self, key: SlabKey) -> Result<Arc<Slab<'p>>> {
        if let Some(slab) = self.slabs.lock().expect("cache lock").get(&key) {
            return Ok(slab);
        }
        // decode outside the lock; two threads may decode the same slab once each, which only costs time
        let slab = Arc::new(self.decode(key)?);
        self.counts.lock().expect("cache lock").decoded += 1;
        self.slabs.lock().expect("cache lock").insert(key, slab.clone(), BufferPool::footprint(slab.data.capacity()));
        Ok(slab)
    }

    fn decode(&self, key: SlabKey) -> Result<Slab<'p>> {
        let [cz, cy, cx, s] = key;
        if self.planes == self.edge[0] {
            self.counts.lock().expect("cache lock").read += 1;
            return Ok(Slab::from_chunk(self.pred.read_chunk_in([cz, cy, cx], self.edge, self.pool)?));
        }
        let packed = self.packed([cz, cy, cx])?;
        let [_, ey, ex] = self.edge;
        let half = self.planes * ey * ex;
        let mut data = Vec::with_capacity_in(2 * half, self.pool);
        data.resize(2 * half, 0);
        self.pred.decode_planes(&packed, s * self.planes..(s + 1) * self.planes, &mut data)?;
        let lo = [cz * self.edge[0] + s * self.planes, cy * ey, cx * ex];
        Ok(Slab { lo, stride: [ey * ex, ex], half, data })
    }

    fn packed(&self, key: Shape) -> Result<Arc<PackedChunk<&'p BufferPool>>> {
        if let Some(packed) = self.packed.lock().expect("cache lock").get(&key) {
            return Ok(packed);
        }
        let packed = Arc::new(self.pred.packed_chunk_in(key, self.pool)?);
        let bytes = match &*packed {
            PackedChunk::Blosc(b) => BufferPool::footprint(b.capacity()),
            PackedChunk::Fill(_) => 0,
        };
        self.counts.lock().expect("cache lock").read += 1;
        self.packed.lock().expect("cache lock").insert(key, packed.clone(), bytes);
        Ok(packed)
    }
}

/// Recently used slabs each finishing thread keeps without touching the shared lock; a point's eight corners
/// span at most eight slabs
const RECENT: usize = 8;

/// Per-thread view of the native cache
pub struct NativeSampler<'c, 'a, 'p> {
    cache: &'c NativeCache<'a, 'p>,
    recent: Vec<(SlabKey, Arc<Slab<'p>>)>,
}

impl<'c, 'a, 'p> NativeSampler<'c, 'a, 'p> {
    pub fn new(cache: &'c NativeCache<'a, 'p>) -> Self {
        Self { cache, recent: Vec::with_capacity(RECENT) }
    }

    fn value(&mut self, z: usize, y: usize, x: usize) -> Result<[u8; 2]> {
        let key = self.cache.slab_key(z, y, x);
        let i = match self.recent.iter().position(|(k, _)| *k == key) {
            Some(i) => i,
            None => {
                let slab = self.cache.slab(key)?;
                if self.recent.len() == RECENT {
                    self.recent.remove(0);
                }
                self.recent.push((key, slab));
                self.recent.len() - 1
            }
        };
        Ok(self.recent[i].1.value(z, y, x))
    }

    /// `NativeSampler.sample` of one point given as float32 x, y, z in native voxels
    fn sample(&mut self, p: [f32; 3]) -> Result<[f32; 2]> {
        let shape = self.cache.pred.shape;
        // np.clip(points[:, ::-1], 0, shape - 1) is float64
        let q = [p[2], p[1], p[0]];
        let q: [f64; 3] = [0, 1, 2].map(|a| {
            let v = q[a].max(0.0);
            let hi = (shape[a] - 1) as f64;
            if hi < v as f64 { hi } else { v as f64 }
        });
        let lo = q.map(|v| v.floor() as i64);
        let frac = [0, 1, 2].map(|a| q[a] - lo[a] as f64);
        let mut values = [0.0f32; 2];
        for corner in 0..8 {
            let c = [(corner >> 2) & 1, (corner >> 1) & 1, corner & 1];
            let ij = [0, 1, 2].map(|a| ((lo[a] + c[a] as i64) as usize).min(shape[a] - 1));
            let w = [0, 1, 2].map(|a| if c[a] == 1 { frac[a] } else { 1.0 - frac[a] });
            let weight = w[0] * w[1] * w[2];
            let v = self.value(ij[0], ij[1], ij[2])?;
            for ch in 0..2 {
                values[ch] = (values[ch] as f64 + v[ch] as f64 * weight) as f32;
            }
        }
        Ok(values)
    }
}

/// `map_coordinates(field, q, order=1, mode="nearest", output="f4")` for both channels at a point
/// given as float32 x, y, z: SciPy keeps the unclamped coordinate for the weights and clamps only the
/// indices, and its second weight is `1 - (1 - t)`
fn sample_tracing(tracing: &[Vol<u8>; 2], p: [f32; 3]) -> [f32; 2] {
    let shape = tracing[0].shape();
    let cc = [p[2] as f64, p[1] as f64, p[0] as f64];
    let mut idx = [[0usize; 2]; 3];
    let mut wts = [[0.0f64; 2]; 3];
    for a in 0..3 {
        let start = cc[a].floor();
        let t = cc[a] - start;
        let w0 = 1.0 - t;
        wts[a] = [w0, 1.0 - w0];
        let s = start as i64;
        let len = shape[a] as i64;
        idx[a] = [s, s + 1].map(|i| i.clamp(0, len - 1) as usize);
    }
    let mut out = [0.0f32; 2];
    for (ch, field) in tracing.iter().enumerate() {
        let mut acc = 0.0f64;
        for kz in 0..2 {
            for ky in 0..2 {
                for kx in 0..2 {
                    let mut coeff = field.get(idx[0][kz], idx[1][ky], idx[2][kx]) as f64;
                    coeff *= wts[0][kz];
                    coeff *= wts[1][ky];
                    coeff *= wts[2][kx];
                    acc += coeff;
                }
            }
        }
        out[ch] = acc as f32;
    }
    out
}

/// What finishing reads: the pooled predictions, the family's presence and the native predictions
pub struct FinishCtx<'a> {
    pub tracing: &'a [Vol<u8>; 2],
    pub pres: &'a Presence<'a>,
    /// 1 = V, 2 = H
    pub family: usize,
}

impl FinishCtx<'_> {
    /// `valid_segments` (tracing.py:18): whether every sample along each segment favours the family in
    /// both the pooled and the native predictions; stops at the first failure when `all` is set
    fn valid_segments(&self, pts: &[[f32; 3]], native: &mut NativeSampler, all: bool) -> Result<Vec<bool>> {
        let (own, other) = (self.family - 1, 2 - self.family);
        let mut valid = vec![true; pts.len().saturating_sub(1)];
        for (i, seg) in pts.windows(2).enumerate() {
            let d = [seg[1][0] - seg[0][0], seg[1][1] - seg[0][1], seg[1][2] - seg[0][2]];
            // np.linalg.norm of float32 rows, then / 0.25 and ceil, all float32
            let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let n = ((len / 0.25f32).ceil() as i64).max(1);
            let step = 1.0 / n as f64;
            for j in 0..=n {
                // np.linspace(0, 1, n + 1, dtype="f4"): j * step in float64, the last exactly 1
                let t = if j == n { 1.0f32 } else { (j as f64 * step) as f32 };
                let q = [0, 1, 2].map(|k| seg[0][k] + d[k] * t);
                let tv = sample_tracing(self.tracing, q);
                let mut good = tv[own] + 1e-4f32 >= tv[other];
                if good {
                    let nv = native.sample(q.map(|v| v * 2.0f32 + 0.5f32))?;
                    good = nv[own] + 1e-4f32 >= nv[other];
                }
                if !good {
                    valid[i] = false;
                    break;
                }
            }
            if all && !valid[i] {
                break;
            }
        }
        Ok(valid)
    }

    /// `finish` for one raw curve: the finished pieces in order
    pub fn finish(&self, raw: &[[f32; 3]], native: &mut NativeSampler) -> Result<Vec<Vec<[f32; 3]>>> {
        let valid = self.valid_segments(raw, native, false)?;
        let mut pieces = Vec::new();
        let mut i = 0;
        while i < valid.len() {
            if !valid[i] {
                i += 1;
                continue;
            }
            let start = i;
            while i < valid.len() && valid[i] {
                i += 1;
            }
            pieces.push(raw[start..=i].to_vec());
        }
        let shape = self.pres.shape();
        for piece in pieces.iter_mut() {
            let (smoothed, count) = fit_curve(piece, self.pres);
            if count < 2 {
                continue;
            }
            let fitted = evaluate_curve(&smoothed, count, shape);
            if self.valid_segments(&fitted, native, true)?.iter().all(|&v| v) {
                *piece = fitted;
            }
        }
        Ok(pieces)
    }
}
