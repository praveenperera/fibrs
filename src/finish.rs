//! `tracing.finish`: split raw curves where predictions disagree, smooth, resample and re-validate
//!
//! `finish` treats every raw curve on its own and reads only immutable fields (pooled predictions,
//! presence, native predictions), so curves are finished in parallel once tracing is done. Sampling
//! reproduces `scipy.ndimage.map_coordinates(order=1, mode="nearest")` (`compressed.sample_tracing`)
//! and `storage.NativeSampler.sample` exactly

use std::collections::{BTreeMap, HashMap};
use std::hash::Hash;
use std::ops::RangeInclusive;
use std::sync::{Arc, Mutex};

use color_eyre::eyre::{Result, ensure, eyre};

use crate::arena::BufferPool;
use crate::fields::Presence;
use crate::input::{NativeChunk, PackedChunk, Predictions};
use crate::kernels::{MIN_FIT_POINTS, evaluate_curve, fit_curve};
use crate::vol::{Shape, Vol};

/// Planes of a chunk decoded together: in prediction zarrs one Blosc block of each channel holds 8 planes of
/// 128 × 128, so a slab is the least c-blosc can decode
const SLAB_PLANES: usize = 8;

/// The cache's unit: a native chunk's index z, y, x and the index of the slab of planes within it
pub(crate) type SlabKey = [usize; 4];

/// Where the native predictions' slabs lie: chunks of `edge` over `shape`, each cut into slabs of `planes` z planes
///
/// The GPU's slab cache keeps a table entry per slab in this order and each slab's voxels in
/// [`NativeCache::copy_slab`]'s layout, so its kernels read what [`NativeSampler`] reads
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SlabLayout {
    /// The native shape, z, y, x
    pub shape: Shape,
    /// The chunk edge, z, y, x
    pub edge: Shape,
    /// Planes per slab, which divides `edge[0]`
    pub planes: usize,
}

#[cfg_attr(not(all(any(target_os = "linux", target_os = "windows"), target_arch = "x86_64")), allow(dead_code))]
impl SlabLayout {
    /// The chunk grid, z, y, x
    pub fn chunks(&self) -> Shape {
        [0, 1, 2].map(|a| self.shape[a].div_ceil(self.edge[a]))
    }

    /// Slabs per chunk
    pub fn per_chunk(&self) -> usize {
        self.edge[0] / self.planes
    }

    /// Slabs in the whole grid: the length of a table with an entry per slab
    pub fn len(&self) -> usize {
        self.chunks().iter().product::<usize>() * self.per_chunk()
    }

    /// Bytes of one slab in [`NativeCache::copy_slab`]'s layout
    pub fn slab_bytes(&self) -> usize {
        2 * self.planes * self.edge[1] * self.edge[2]
    }

    /// The table index of the slab with key `[cz, cy, cx, s]`: `((cz · ncy + cy) · ncx + cx) · spc + s`
    pub fn index(&self, [cz, cy, cx, s]: SlabKey) -> usize {
        let [_, ncy, ncx] = self.chunks();
        ((cz * ncy + cy) * ncx + cx) * self.per_chunk() + s
    }

    /// The key of the slab at table index `i`
    pub fn key(&self, i: usize) -> SlabKey {
        let [_, ncy, ncx] = self.chunks();
        let spc = self.per_chunk();
        let chunk = i / spc;
        [chunk / ncx / ncy, chunk / ncx % ncy, chunk % ncx, i % spc]
    }

    /// The table index of every slab holding a voxel of the box from `lo` to `hi`, both inclusive
    pub fn indices_in(&self, lo: Shape, hi: Shape) -> impl Iterator<Item = usize> + '_ {
        let [_, ey, ex] = self.edge;
        let spc = self.per_chunk();
        // slab row `r` along z is slab `r % spc` of chunk `r / spc`, as `cz · ez + s · planes = r · planes`
        (lo[0] / self.planes..=hi[0] / self.planes).flat_map(move |r| {
            (lo[1] / ey..=hi[1] / ey)
                .flat_map(move |cy| (lo[2] / ex..=hi[2] / ex).map(move |cx| self.index([r / spc, cy, cx, r % spc])))
        })
    }
}

/// Decoded native predictions for a box of voxels, both channels, in a buffer of the cache's pool
struct Slab<'p> {
    /// The box's first voxel, z, y, x
    lo: Shape,
    /// The box's extent, z, y, x: a chunk clipped to the array for a whole chunk read, else the slab's planes of
    /// the chunk's full extent
    dims: Shape,
    /// Elements between consecutive planes and rows
    stride: [usize; 2],
    /// Offset of the H channel
    half: usize,
    data: Vec<u8, &'p BufferPool>,
}

impl<'p> Slab<'p> {
    fn from_chunk(chunk: NativeChunk<&'p BufferPool>) -> Self {
        let [_, dy, dx] = chunk.dims;
        let (lo, dims) = (chunk.lo, chunk.dims);
        Self { lo, dims, stride: [dy * dx, dx], half: dims.iter().product(), data: chunk.data }
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

    /// Entries of `bytes` each that the whole budget holds
    fn capacity(&self, bytes: usize) -> usize {
        self.budget / (bytes + ENTRY_BYTES)
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

/// Freed decoded slabs (or whole chunks) kept for reuse: each eviction frees one and the decode that caused it
/// takes one, so few wait at a time; this bounds a burst of evictions, at 512 slabs of the usual 2 × 8 × 128²
const DECODED_RETAIN: usize = 128 << 20;
/// Freed compressed chunks kept for reuse. A chunk's compressed size varies from chunk to chunk, so the chunk an
/// eviction frees is seldom of the size class the next read asks for; a kept buffer of the asked class is found
/// only when the pool holds many of each
const PACKED_RETAIN: usize = 128 << 20;

/// The buffer pools behind a [`NativeCache`]
///
/// Decoded slabs and compressed chunks are kept apart: a slab an eviction frees is the size the next decode asks
/// for, and in one pool compressed chunks of many sizes filled the room kept slabs need, so freed slabs were let
/// go and the next decodes mapped fresh memory
pub struct CachePools {
    decoded: BufferPool,
    packed: BufferPool,
}

impl CachePools {
    pub fn new() -> Self {
        Self {
            decoded: BufferPool::new(DECODED_RETAIN),
            // the same 64 KiB classes as decoded buffers: coarser classes found a kept buffer more often (2% of
            // reads missed in a replay of z02304's chunk sizes, against 9% here) but charged some chunk sizes more
            // of the cache's budget, so a small cache held fewer chunks and read them again
            packed: BufferPool::new(PACKED_RETAIN),
        }
    }
}

impl Default for CachePools {
    fn default() -> Self {
        Self::new()
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
/// Dense windows evict and decode hundreds of times a second, so every buffer comes from `pools`: an evicted
/// slab's buffer goes back to them once the last finishing thread lets go of the slab, and the next decode takes it
pub struct NativeCache<'a, 'p> {
    pred: &'a Predictions,
    edge: Shape,
    /// Planes per slab; a whole chunk when the array cannot be decoded by planes
    planes: usize,
    pools: &'p CachePools,
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
    pub fn new(pred: &'a Predictions, budget_bytes: usize, pools: &'p CachePools) -> Self {
        // whole Zarr chunks are the unit of reading; NPY input uses the reference's 64³ blocks
        let edge = pred.read_unit();
        // a chunk of one slab is decoded whole either way, and needs no cache of compressed chunks
        let sliced = pred.slab_layout().is_some_and(|e| e[0] > SLAB_PLANES && e[0].is_multiple_of(SLAB_PLANES))
            && pred.blocks_fit_planes(SLAB_PLANES);
        let planes = if sliced { SLAB_PLANES } else { edge[0] };
        // compressed chunks are about a fifth of their decoded size, so a quarter of the budget keeps the chunks
        // of several times as many slabs as the rest holds
        let packed = if sliced { budget_bytes / 4 } else { 0 };
        Self {
            pred,
            edge,
            planes,
            pools,
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

    /// Where the cache's slabs lie
    #[cfg_attr(not(all(any(target_os = "linux", target_os = "windows"), target_arch = "x86_64")), allow(dead_code))]
    pub(crate) fn layout(&self) -> SlabLayout {
        SlabLayout { shape: self.pred.shape, edge: self.edge, planes: self.planes }
    }

    /// Write slab `key`, decoded through the cache, into `out` as [`SlabLayout`] lays a slab out: V then H, each
    /// `planes × ey × ex` at the chunk's full extent, voxel z, y, x at `(z - lo_z) · ey · ex + (y - lo_y) · ex +
    /// (x - lo_x)`; voxels past the array's edge, which no sample reads, are zero
    #[cfg_attr(not(all(any(target_os = "linux", target_os = "windows"), target_arch = "x86_64")), allow(dead_code))]
    pub(crate) fn copy_slab(&self, key: SlabKey, out: &mut [u8]) -> Result<()> {
        let slab = self.slab(key)?;
        let [_, ey, ex] = self.edge;
        let half = self.planes * ey * ex;
        ensure!(out.len() == 2 * half, "a slab of {half} voxels a channel copied into {} bytes", out.len());
        let channels = [&slab.data[..slab.half], &slab.data[slab.half..2 * slab.half]];
        // a slab decoded by planes, or a whole chunk inside the array, is already in this layout
        if slab.dims == [self.planes, ey, ex] {
            for (to, from) in out.chunks_exact_mut(half).zip(channels) {
                to.copy_from_slice(from);
            }
            return Ok(());
        }

        out.fill(0);
        let [dz, dy, dx] = slab.dims;
        for (to, from) in out.chunks_exact_mut(half).zip(channels) {
            for z in 0..dz {
                for y in 0..dy {
                    let row = z * slab.stride[0] + y * slab.stride[1];
                    to[(z * ey + y) * ex..][..dx].copy_from_slice(&from[row..row + dx]);
                }
            }
        }
        Ok(())
    }

    /// The predictions the cache reads
    #[cfg_attr(not(all(any(target_os = "linux", target_os = "windows"), target_arch = "x86_64")), allow(dead_code))]
    pub(crate) fn predictions(&self) -> &'a Predictions {
        self.pred
    }

    /// Chunk `key`'s compressed bytes through the cache of compressed chunks, for a decoder outside the cache (the
    /// GPU's); `None` when the cache decodes whole chunks, and so keeps no compressed ones
    #[cfg_attr(not(all(any(target_os = "linux", target_os = "windows"), target_arch = "x86_64")), allow(dead_code))]
    pub(crate) fn packed_chunk(&self, key: Shape) -> Option<Result<Arc<PackedChunk<&'p BufferPool>>>> {
        (self.planes != self.edge[0]).then(|| self.packed(key))
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
            return Ok(Slab::from_chunk(self.pred.read_chunk_in(
                [cz, cy, cx],
                self.edge,
                &self.pools.decoded,
                &self.pools.packed,
            )?));
        }
        let packed = self.packed([cz, cy, cx])?;
        if !self.pred.planes_fit(&packed, self.planes) {
            return self.decode_whole([cz, cy, cx], s, &packed);
        }
        let [_, ey, ex] = self.edge;
        let half = self.planes * ey * ex;
        let mut data = Vec::with_capacity_in(2 * half, &self.pools.decoded);
        let planes = s * self.planes..(s + 1) * self.planes;
        self.pred.decode_planes(&packed, planes, &mut data).map_err(|e| eyre!("chunk {:?}: {e}", [cz, cy, cx]))?;
        Ok(self.slab_at([cz, cy, cx], s, data))
    }

    /// Decode a chunk whose Blosc blocks are larger than a slab once, and return its slab `s`; decoding it slab by
    /// slab would decode each block once for every slab it holds
    ///
    /// The chunk's other slabs are cached too, as many as the cache holds beside slab `s`: the ones after `s` first,
    /// since curves are mostly read forward, then the ones before it, nearest first in each direction. They are
    /// inserted in reverse so the most wanted is dropped last, and `slab` caches slab `s` after them. Caching every
    /// slab of a chunk larger than the cache would drop the first ones cached before they are read, and decode the
    /// chunk again for each
    fn decode_whole(&self, chunk: Shape, s: usize, packed: &PackedChunk<&'p BufferPool>) -> Result<Slab<'p>> {
        let [ez, ey, ex] = self.edge;
        let channel = ez * ey * ex;
        let mut all = Vec::with_capacity_in(2 * channel, &self.pools.decoded);
        self.pred.decode_planes(packed, 0..ez, &mut all).map_err(|e| eyre!("chunk {chunk:?}: {e}"))?;
        let half = self.planes * ey * ex;
        let slab = |i: usize| {
            let mut data = Vec::with_capacity_in(2 * half, &self.pools.decoded);
            data.extend_from_slice(&all[i * half..(i + 1) * half]);
            data.extend_from_slice(&all[channel + i * half..channel + (i + 1) * half]);
            self.slab_at(chunk, i, data)
        };
        let n = ez / self.planes;
        let bytes = BufferPool::footprint(2 * half);
        let others = self.slabs.lock().expect("cache lock").capacity(bytes).saturating_sub(1);
        let wanted: Vec<usize> = (s + 1..n).chain((0..s).rev()).take(others).collect();
        // each copy is made before the lock is taken, so threads reading cached slabs do not wait on it, and is
        // inserted before the next is made, so a thread holds one copy outside the cache at a time instead of a
        // chunk's worth
        for &i in wanted.iter().rev() {
            let cached = Arc::new(slab(i));
            self.slabs.lock().expect("cache lock").insert([chunk[0], chunk[1], chunk[2], i], cached, bytes);
        }

        Ok(slab(s))
    }

    /// Slab `s` of `chunk`, holding `data`: V then H, each `planes × cy × cx`
    fn slab_at(&self, chunk: Shape, s: usize, data: Vec<u8, &'p BufferPool>) -> Slab<'p> {
        let [ez, ey, ex] = self.edge;
        let lo = [chunk[0] * ez + s * self.planes, chunk[1] * ey, chunk[2] * ex];
        Slab { lo, dims: [self.planes, ey, ex], stride: [ey * ex, ex], half: self.planes * ey * ex, data }
    }

    fn packed(&self, key: Shape) -> Result<Arc<PackedChunk<&'p BufferPool>>> {
        if let Some(packed) = self.packed.lock().expect("cache lock").get(&key) {
            return Ok(packed);
        }
        let packed = Arc::new(self.pred.packed_chunk_in(key, &self.pools.packed)?);
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

/// The pieces `finish` cuts a curve of `segments + 1` points into: each maximal run of segments `valid` accepts,
/// as the range of its points, the run's segments and the point after the last
///
/// The GPU path forms its pieces from the kernels' flags with this too, so both cut a curve alike
pub(crate) fn valid_runs(
    segments: usize,
    valid: impl Fn(usize) -> bool,
) -> impl Iterator<Item = RangeInclusive<usize>> {
    let mut i = 0;
    std::iter::from_fn(move || {
        while i < segments && !valid(i) {
            i += 1;
        }
        if i == segments {
            return None;
        }
        let start = i;
        while i < segments && valid(i) {
            i += 1;
        }
        Some(start..=i)
    })
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
        let mut pieces: Vec<Vec<[f32; 3]>> = valid_runs(valid.len(), |i| valid[i]).map(|r| raw[r].to_vec()).collect();
        let shape = self.pres.shape();
        for piece in pieces.iter_mut() {
            if piece.len() < MIN_FIT_POINTS {
                continue;
            }
            let (smoothed, count) = fit_curve(piece, self.pres);
            let fitted = evaluate_curve(&smoothed, count, shape);
            if self.valid_segments(&fitted, native, true)?.iter().all(|&v| v) {
                *piece = fitted;
            }
        }
        Ok(pieces)
    }
}

/// Synthetic prediction stores for tests
#[cfg(test)]
pub(crate) mod test_support {
    use std::fs;
    use std::path::PathBuf;

    use crate::input::Predictions;
    use crate::vol::Shape;

    /// A zstd Blosc frame of uint8 data with blocks of `blocksize` bytes
    pub fn frame(data: &[u8], blocksize: usize) -> Vec<u8> {
        let mut out = vec![0u8; data.len() + 64];
        // SAFETY: `out` has room for the frame and its 16-byte header; the context call keeps no global state
        let n = unsafe {
            blosc_src::blosc_compress_ctx(
                3,
                1,
                1,
                data.len(),
                data.as_ptr().cast(),
                out.as_mut_ptr().cast(),
                out.len(),
                c"zstd".as_ptr(),
                blocksize,
                1,
            )
        };
        assert!(n > 0, "blosc_compress_ctx failed ({n})");
        out.truncate(n as usize);
        out
    }

    /// A uint8 prediction store in `name` under the temp directory over `shape` in chunks of `edge`³, chunk
    /// z, y, x compressed in blocks of `blocksize([cz, cy, cx])` bytes and voxel z, y, x of channel `c` (0 = V)
    /// holding `value(c, z, y, x)`
    ///
    /// Chunks at the array's edge are stored at their full extent, as Zarr stores them, zero past the edge
    pub fn store_with(
        name: &str,
        shape: Shape,
        edge: usize,
        blocksize: impl Fn(Shape) -> usize,
        value: impl Fn(usize, usize, usize, usize) -> u8,
    ) -> (PathBuf, Predictions) {
        let dir = std::env::temp_dir().join(format!("fibrs-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let n = edge * edge * edge;
        let grid = shape.map(|s| s.div_ceil(edge));
        for cz in 0..grid[0] {
            for cy in 0..grid[1] {
                for cx in 0..grid[2] {
                    let mut data = vec![0u8; 2 * n];
                    for (i, v) in data.iter_mut().enumerate() {
                        let (c, at) = (i / n, i % n);
                        let [z, y, x] =
                            [cz * edge + at / (edge * edge), cy * edge + at / edge % edge, cx * edge + at % edge];
                        if z < shape[0] && y < shape[1] && x < shape[2] {
                            *v = value(c, z, y, x);
                        }
                    }
                    let path = dir.join(format!("0.{cz}.{cy}.{cx}"));
                    fs::write(path, frame(&data, blocksize([cz, cy, cx]))).unwrap();
                }
            }
        }
        let meta = serde_json::json!({
            "zarr_format": 2, "shape": [2, shape[0], shape[1], shape[2]], "chunks": [2, edge, edge, edge],
            "dtype": "|u1", "fill_value": 0, "order": "C", "filters": null, "dimension_separator": ".",
            "compressor": {"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 1, "blocksize": 0},
        });
        fs::write(dir.join(".zarray"), meta.to_string()).unwrap();
        let pred = Predictions::open(&dir, None, "VH").unwrap();
        (dir, pred)
    }

    /// A uint8 prediction store in `name` under the temp directory with one row of chunks of `edge`³ along x,
    /// chunk x compressed in blocks of `blocks[x]` bytes
    pub fn store(name: &str, edge: usize, blocks: &[usize]) -> (PathBuf, Predictions) {
        let n = edge * edge * edge;
        let shape = [edge, edge, blocks.len() * edge];
        let value = |c: usize, z: usize, y: usize, x: usize| {
            let i = c * n + (z * edge + y) * edge + x % edge;
            (i * 7 + i / 977 + 101 * (x / edge)) as u8
        };
        store_with(name, shape, edge, |[_, _, cx]| blocks[cx], value)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::test_support::{store, store_with};
    use super::*;

    /// Every slab, in both cache modes, copies out in the GPU's layout with every voxel's values where the layout
    /// puts them, including the chunks clipped at the array's edge, which a whole-chunk read returns at their
    /// clipped extent; and the layout's table index and key agree with the cache's own slab keys
    #[test]
    fn copied_slabs_hold_each_voxel_where_the_layout_says() {
        let edge = 2 * SLAB_PLANES;
        let shape = [40, 37, 35];
        let value = |c: usize, z: usize, y: usize, x: usize| ((z * 31 + y * 7 + x * 3 + c * 101) % 251) as u8;
        let one_block = 2 * edge * edge * edge;
        for (name, blocksize, planes) in
            [("copy-sliced", SLAB_PLANES * edge * edge, SLAB_PLANES), ("copy-whole", one_block, edge)]
        {
            let (dir, pred) = store_with(name, shape, edge, |_| blocksize, value);
            let pools = CachePools::new();
            let cache = NativeCache::new(&pred, 64 << 20, &pools);
            let layout = cache.layout();
            assert_eq!(layout, SlabLayout { shape, edge: [edge; 3], planes }, "{name}");
            let half = layout.slab_bytes() / 2;
            let mut out = vec![0u8; 2 * half];
            for i in 0..layout.len() {
                let key = layout.key(i);
                assert_eq!(layout.index(key), i);
                cache.copy_slab(key, &mut out).unwrap();
                let [cz, cy, cx, s] = key;
                let lo = [cz * edge + s * planes, cy * edge, cx * edge];
                for z in lo[0]..(lo[0] + planes).min(shape[0]) {
                    for y in lo[1]..(lo[1] + edge).min(shape[1]) {
                        for x in lo[2]..(lo[2] + edge).min(shape[2]) {
                            assert_eq!(cache.slab_key(z, y, x), key);
                            let at = ((z - lo[0]) * edge + y - lo[1]) * edge + x - lo[2];
                            let want = [value(0, z, y, x), value(1, z, y, x)];
                            assert_eq!([out[at], out[half + at]], want, "{name}: voxel {z} {y} {x}");
                        }
                    }
                }
            }
            let _ = fs::remove_dir_all(&dir);
        }
    }

    /// A cache in slab mode with `slab_budget` bytes for slabs, as the probe chooses it when the chunk it lists
    /// first fits; which chunk it lists first depends on the file system
    fn slab_cache<'a, 'p>(pred: &'a Predictions, pools: &'p CachePools, slab_budget: usize) -> NativeCache<'a, 'p> {
        NativeCache {
            pred,
            edge: pred.slab_layout().unwrap(),
            planes: SLAB_PLANES,
            pools,
            slabs: Mutex::new(Lru::new(slab_budget)),
            packed: Mutex::new(Lru::new(16 << 20)),
            counts: Mutex::default(),
        }
    }

    /// Through the slab cache, a store that mixes chunks of slab-sized Blosc blocks with a chunk compressed as one
    /// block reads the reference window reader's values, and the one-block chunk is decoded once for both its slabs
    #[test]
    fn mixed_block_sizes_read_the_window_values() {
        let edge = 2 * SLAB_PLANES;
        let n = edge * edge * edge;
        let (dir, pred) = store("mixed-blocks", edge, &[SLAB_PLANES * edge * edge, 2 * n]);
        let pools = CachePools::new();
        let cache = slab_cache(&pred, &pools, 64 << 20);
        let mut sampler = NativeSampler::new(&cache);
        let shape = [edge, edge, 2 * edge];
        let window = pred.read_u8([0; 3], shape).unwrap();
        let plane = shape.iter().product::<usize>();
        for z in 0..shape[0] {
            for y in 0..shape[1] {
                for x in 0..shape[2] {
                    let at = (z * shape[1] + y) * shape[2] + x;
                    assert_eq!(sampler.value(z, y, x).unwrap(), [window[at], window[plane + at]], "{z} {y} {x}");
                }
            }
        }
        // two slab decodes of the first chunk, one whole decode of the second
        assert_eq!(cache.decoded(), 3);
        let _ = fs::remove_dir_all(&dir);
    }

    /// The cache over a crop, keyed by the crop's own chunks, reads the whole array's values at the crop's offset,
    /// by slabs and by whole decodes alike
    #[test]
    fn a_cropped_cache_reads_the_array_at_the_crop_offset() {
        let edge = 2 * SLAB_PLANES;
        let n = edge * edge * edge;
        let (dir, pred) = store("crop-cache", edge, &[2 * n, SLAB_PLANES * edge * edge, 2 * n]);
        let crop = pred.crop([0, 0, edge], [edge, edge, 3 * edge]).unwrap();
        let pools = CachePools::new();
        let cache = slab_cache(&crop, &pools, 64 << 20);
        let mut sampler = NativeSampler::new(&cache);
        let shape = [edge, edge, 2 * edge];
        let window = pred.read_u8([0, 0, edge], [edge, edge, 3 * edge]).unwrap();
        let plane = shape.iter().product::<usize>();
        for z in 0..shape[0] {
            for y in 0..shape[1] {
                for x in 0..shape[2] {
                    let at = (z * shape[1] + y) * shape[2] + x;
                    assert_eq!(sampler.value(z, y, x).unwrap(), [window[at], window[plane + at]], "{z} {y} {x}");
                }
            }
        }
        let _ = fs::remove_dir_all(&dir);
    }

    /// A one-block chunk of 8 slabs read slab by slab through a cache of 3 slabs is decoded again only once the
    /// slabs cached with the last decode are used up. Read forward it is decoded at slabs 0, 3 and 6, where caching
    /// every slab of each decode dropped the next ones before they were read and decoded it 6 times, and caching
    /// the nearest on both sides decoded it 4 times
    #[test]
    fn a_one_block_chunk_in_a_small_cache_is_decoded_once_per_cached_run() {
        let edge = 8 * SLAB_PLANES;
        let (dir, pred) = store("small-cache", edge, &[2 * edge * edge * edge]);
        let pools = CachePools::new();
        let slab = BufferPool::footprint(2 * SLAB_PLANES * edge * edge) + ENTRY_BYTES;
        let decodes = |reads: &[usize]| {
            let cache = slab_cache(&pred, &pools, 3 * slab);
            for &s in reads {
                cache.slab([0, 0, 0, s]).unwrap();
            }

            cache.decoded()
        };

        // each decode caches the two slabs after the one read
        assert_eq!(decodes(&[0, 1, 2, 3, 4, 5, 6, 7]), 3);
        // the last slabs have none after them, so the one before is kept instead
        assert_eq!(decodes(&[5, 6, 7]), 1);
        assert_eq!(decodes(&[7, 6, 5]), 1);
        let _ = fs::remove_dir_all(&dir);
    }
}
