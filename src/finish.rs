//! `tracing.finish`: split raw curves where predictions disagree, smooth, resample and re-validate
//!
//! `finish` treats every raw curve on its own and reads only immutable fields (pooled predictions,
//! presence, native predictions), so curves are finished in parallel once tracing is done. Sampling
//! reproduces `scipy.ndimage.map_coordinates(order=1, mode="nearest")` (`compressed.sample_tracing`)
//! and `storage.NativeSampler.sample` exactly

use std::collections::{BTreeMap, HashMap};
use std::hash::Hash;
use std::sync::{Arc, Mutex};

use color_eyre::eyre::{Result, eyre};

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
        if !self.pred.planes_fit(&packed, self.planes) {
            return self.decode_whole([cz, cy, cx], s, &packed);
        }
        let [_, ey, ex] = self.edge;
        let half = self.planes * ey * ex;
        let mut data = Vec::with_capacity_in(2 * half, self.pool);
        data.resize(2 * half, 0);
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
        let mut all = Vec::with_capacity_in(2 * channel, self.pool);
        all.resize(2 * channel, 0);
        self.pred.decode_planes(packed, 0..ez, &mut all).map_err(|e| eyre!("chunk {chunk:?}: {e}"))?;
        let half = self.planes * ey * ex;
        let slab = |i: usize| {
            let mut data = Vec::with_capacity_in(2 * half, self.pool);
            data.extend_from_slice(&all[i * half..(i + 1) * half]);
            data.extend_from_slice(&all[channel + i * half..channel + (i + 1) * half]);
            self.slab_at(chunk, i, data)
        };
        let n = ez / self.planes;
        let bytes = BufferPool::footprint(2 * half);
        let others = self.slabs.lock().expect("cache lock").capacity(bytes).saturating_sub(1);
        let wanted: Vec<(SlabKey, Arc<Slab<'p>>)> = (s + 1..n)
            .chain((0..s).rev())
            .take(others)
            .map(|i| ([chunk[0], chunk[1], chunk[2], i], Arc::new(slab(i))))
            .collect();

        // the copies above run outside the lock, so threads reading cached slabs do not wait on them
        let mut slabs = self.slabs.lock().expect("cache lock");
        for (key, cached) in wanted.into_iter().rev() {
            slabs.insert(key, cached, bytes);
        }

        drop(slabs);
        Ok(slab(s))
    }

    /// Slab `s` of `chunk`, holding `data`: V then H, each `planes × cy × cx`
    fn slab_at(&self, chunk: Shape, s: usize, data: Vec<u8, &'p BufferPool>) -> Slab<'p> {
        let [ez, ey, ex] = self.edge;
        let lo = [chunk[0] * ez + s * self.planes, chunk[1] * ey, chunk[2] * ex];
        Slab { lo, stride: [ey * ex, ex], half: self.planes * ey * ex, data }
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

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    /// A zstd Blosc frame of uint8 data with blocks of `blocksize` bytes
    fn frame(data: &[u8], blocksize: usize) -> Vec<u8> {
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

    /// A uint8 prediction store in `name` under the temp directory with one row of chunks of `edge`³ along x,
    /// chunk x compressed in blocks of `blocks[x]` bytes
    fn store(name: &str, edge: usize, blocks: &[usize]) -> (std::path::PathBuf, Predictions) {
        let n = edge * edge * edge;
        let dir = std::env::temp_dir().join(format!("fibrs-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        for (x, &blocksize) in blocks.iter().enumerate() {
            let data: Vec<u8> = (0..2 * n).map(|i| (i * 7 + i / 977 + 101 * x) as u8).collect();
            fs::write(dir.join(format!("0.0.0.{x}")), frame(&data, blocksize)).unwrap();
        }
        let meta = serde_json::json!({
            "zarr_format": 2, "shape": [2, edge, edge, blocks.len() * edge], "chunks": [2, edge, edge, edge],
            "dtype": "|u1", "fill_value": 0, "order": "C", "filters": null, "dimension_separator": ".",
            "compressor": {"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 1, "blocksize": 0},
        });
        fs::write(dir.join(".zarray"), meta.to_string()).unwrap();
        let pred = Predictions::open(&dir, None, "VH").unwrap();
        (dir, pred)
    }

    /// A cache in slab mode with `slab_budget` bytes for slabs, as the probe chooses it when the chunk it lists
    /// first fits; which chunk it lists first depends on the file system
    fn slab_cache<'a, 'p>(pred: &'a Predictions, pool: &'p BufferPool, slab_budget: usize) -> NativeCache<'a, 'p> {
        NativeCache {
            pred,
            edge: pred.slab_layout().unwrap(),
            planes: SLAB_PLANES,
            pool,
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
        let pool = BufferPool::new(1 << 20);
        let cache = slab_cache(&pred, &pool, 64 << 20);
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

    /// A one-block chunk of 8 slabs read slab by slab through a cache of 3 slabs is decoded again only once the
    /// slabs cached with the last decode are used up. Read forward it is decoded at slabs 0, 3 and 6, where caching
    /// every slab of each decode dropped the next ones before they were read and decoded it 6 times, and caching
    /// the nearest on both sides decoded it 4 times
    #[test]
    fn a_one_block_chunk_in_a_small_cache_is_decoded_once_per_cached_run() {
        let edge = 8 * SLAB_PLANES;
        let (dir, pred) = store("small-cache", edge, &[2 * edge * edge * edge]);
        let pool = BufferPool::new(1 << 20);
        let slab = BufferPool::footprint(2 * SLAB_PLANES * edge * edge) + ENTRY_BYTES;
        let decodes = |reads: &[usize]| {
            let cache = slab_cache(&pred, &pool, 3 * slab);
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
