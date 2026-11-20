//! Prediction readers: Zarr v2 and NPY, as `Predictions` in the reference
//!
//! Mirrors `storage.Predictions`: a `(2, Z, Y, X)` array of uint8 or floats in [0, 1], returned in V, H
//! channel order, with floats quantized by `rint(x * 255)` in float32. Both array orders and both byte
//! orders are read. Zarr chunks are decoded by the c-blosc release numcodecs 0.15 bundles, or by
//! numcodecs' Zstd, Zlib, GZip and LZ4 formats

use std::alloc::{Allocator, Global};
use std::fs;
use std::io::{self, Read};
use std::mem::MaybeUninit;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use color_eyre::eyre::{Result, bail, ensure, eyre};
use flate2::read::{MultiGzDecoder, ZlibDecoder};
use half::f16;
use memmap2::Mmap;
use serde_json::Value;

const DTYPE_ERROR: &str = "Predictions must be uint8 [0,255] or floating point [0,1]";
const VALUE_ERROR: &str = "Predictions contain NaN, infinity or values outside [0,1]";

/// The unit NPY input is read in by the native cache, in place of Zarr chunks: the reference's 64³ blocks
pub const NPY_BLOCK: [usize; 3] = [64, 64, 64];

/// Element kind of a prediction array
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    U8,
    F16,
    F32,
    F64,
}

/// Element type of a prediction array: kind and byte order
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dtype {
    kind: Kind,
    big_endian: bool,
}

impl Dtype {
    fn size(self) -> usize {
        match self.kind {
            Kind::U8 => 1,
            Kind::F16 => 2,
            Kind::F32 => 4,
            Kind::F64 => 8,
        }
    }

    pub fn is_float(self) -> bool {
        self.kind != Kind::U8
    }

    /// Parse a numpy type string such as `|u1`, `<f4` or `>f8`
    fn parse(text: &str) -> Result<Self> {
        let (big_endian, code) = match text.as_bytes().first() {
            Some(b'>') => (true, &text[1..]),
            Some(b'<' | b'|' | b'=') => (false, &text[1..]),
            _ => (false, text),
        };
        let kind = match code {
            "u1" => Kind::U8,
            "f2" => Kind::F16,
            "f4" => Kind::F32,
            "f8" => Kind::F64,
            // numpy's long double: a float the reference accepts but no input pipeline writes
            c if c.starts_with('f') => bail!("extended-precision float predictions ({text}) are not supported"),
            _ => bail!("{DTYPE_ERROR}"),
        };
        Ok(Self { kind, big_endian })
    }

    /// One element as float64; floats only
    fn value(self, b: &[u8]) -> f64 {
        let big = self.big_endian;
        match self.kind {
            Kind::F16 => {
                let bits = [b[0], b[1]];
                f16::from_bits(if big { u16::from_be_bytes(bits) } else { u16::from_le_bytes(bits) }).to_f64()
            }
            Kind::F32 => {
                let bits = [b[0], b[1], b[2], b[3]];
                (if big { f32::from_be_bytes(bits) } else { f32::from_le_bytes(bits) }) as f64
            }
            Kind::F64 => {
                let bits = b[..8].try_into().expect("8 bytes");
                if big { f64::from_be_bytes(bits) } else { f64::from_le_bytes(bits) }
            }
            Kind::U8 => b[0] as f64,
        }
    }

    /// `v` as one element's bytes in this type and byte order
    fn encode(self, v: f64) -> Vec<u8> {
        let big = self.big_endian;
        match self.kind {
            Kind::U8 => vec![v as u8],
            Kind::F16 => {
                let bits = f16::from_f64(v).to_bits();
                (if big { bits.to_be_bytes() } else { bits.to_le_bytes() }).to_vec()
            }
            Kind::F32 => (if big { (v as f32).to_be_bytes() } else { (v as f32).to_le_bytes() }).to_vec(),
            Kind::F64 => (if big { v.to_be_bytes() } else { v.to_le_bytes() }).to_vec(),
        }
    }
}

/// A Zarr v2 compressor this reader decodes
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Compressor {
    Raw,
    Blosc,
    Zstd,
    Zlib,
    Gzip,
    Lz4,
}

enum Store {
    Zarr { dir: PathBuf, chunks: [usize; 4], sep: char, fill: Vec<u8>, compressor: Compressor, fortran: bool },
    Npy { map: Mmap, offset: usize, fortran: bool },
}

/// A local two-channel prediction array, or a chunk-aligned crop of one ([`Predictions::crop`])
///
/// Every coordinate and chunk key a caller passes is local to the crop; reads translate them by the crop's offset,
/// so a crop reads exactly the values of the whole array at `offset + local`. A whole array is its own crop at
/// offset zero
pub struct Predictions {
    /// Resolved input path, as recorded in `run.json`
    pub path: PathBuf,
    /// Spatial shape of the crop, z, y, x
    pub shape: [usize; 3],
    pub dtype: Dtype,
    reverse: bool,
    store: Arc<Store>,
    /// The crop's first voxel in the stored array, z, y, x; on a chunk edge of every axis
    offset: [usize; 3],
    /// Spatial shape of the stored array
    full: [usize; 3],
}

impl Predictions {
    /// Open `path` (a `.npy` file or a Zarr array or group, with `key` naming the array in a group)
    pub fn open(path: &Path, key: Option<&str>, order: &str) -> Result<Self> {
        let path = fs::canonicalize(path).map_err(|e| eyre!("{}: {e}", path.display()))?;
        let (store, full_shape, dtype) = if path.extension().is_some_and(|e| e == "npy") {
            if key.is_some_and(|k| !k.is_empty()) {
                bail!("--array is only valid for a Zarr group");
            }
            open_npy(&path)?
        } else {
            open_zarr(&path, key)?
        };
        if full_shape.len() != 4 {
            bail!("Predictions must be an array with shape (2, Z, Y, X)");
        }
        if full_shape[0] != 2 || full_shape[1..].iter().min().copied().unwrap_or(0) < 2 {
            bail!("Expected two channels and at least 2 voxels along each spatial axis");
        }
        let shape = [full_shape[1], full_shape[2], full_shape[3]];
        let store = Arc::new(store);
        Ok(Self { path, shape, dtype, reverse: order == "HV", store, offset: [0; 3], full: shape })
    }

    /// The box `[lo, hi)` of these predictions (local coordinates) as predictions of its own, sharing the store
    ///
    /// The box must lie on whole chunks of the stored array (NPY input: [`NPY_BLOCK`]s): its start on a chunk edge
    /// and its end on one or at the array's end. The crop's chunk grid is then the array's shifted by whole chunks,
    /// so its chunk keys translate to the array's and whole-chunk, packed and plane reads work unchanged. An
    /// unaligned box is refused rather than read across the chunk grid
    pub fn crop(&self, lo: [usize; 3], hi: [usize; 3]) -> Result<Self> {
        let edge = self.read_unit();
        for a in 0..3 {
            ensure!(
                lo[a] < hi[a] && hi[a] <= self.shape[a],
                "crop {lo:?}..{hi:?} lies outside predictions of shape {:?}",
                self.shape
            );
            let (start, end) = (self.offset[a] + lo[a], self.offset[a] + hi[a]);
            ensure!(
                start.is_multiple_of(edge[a]) && (end.is_multiple_of(edge[a]) || end == self.full[a]),
                "crop {lo:?}..{hi:?} of {} is not aligned to its {edge:?} chunks",
                self.path.display()
            );
        }
        Ok(Self {
            path: self.path.clone(),
            shape: [0, 1, 2].map(|a| hi[a] - lo[a]),
            dtype: self.dtype,
            reverse: self.reverse,
            store: self.store.clone(),
            offset: [0, 1, 2].map(|a| self.offset[a] + lo[a]),
            full: self.full,
        })
    }

    /// The box whole reads are made of: the Zarr chunk, or [`NPY_BLOCK`] for NPY input
    pub fn read_unit(&self) -> [usize; 3] {
        self.chunk_shape().unwrap_or(NPY_BLOCK)
    }

    /// A chunk key of the crop as the stored array's key; the crop's offset is whole chunks
    fn stored_key(&self, key: [usize; 3]) -> Result<[usize; 3]> {
        let Store::Zarr { chunks, .. } = &*self.store else { bail!("only Zarr arrays have chunk keys") };
        Ok([0, 1, 2].map(|a| key[a] + self.offset[a] / chunks[a + 1]))
    }

    /// Read `(2, hi - lo)` as uint8 in V, H order (`Predictions.read(lo, hi)`)
    pub fn read_u8(&self, lo: [usize; 3], hi: [usize; 3]) -> Result<Vec<u8>> {
        self.read_u8_in(lo, hi, Global)
    }

    /// [`Predictions::read_u8`] into a buffer from `alloc`; uint8 arrays are read straight into it
    fn read_u8_in<A: Allocator + Clone>(&self, lo: [usize; 3], hi: [usize; 3], alloc: A) -> Result<Vec<u8, A>> {
        let raw = self.read_raw_in(lo, hi, alloc.clone())?;
        if self.dtype.kind == Kind::U8 {
            return Ok(raw);
        }
        let floats = self.to_f32(&raw)?;
        // np.rint(block * 255).astype("u1") in float32; values are already checked to lie in [0, 1]
        let mut out = Vec::with_capacity_in(floats.len(), alloc);
        out.extend(floats.iter().map(|&v| (v * 255.0f32).round_ties_even() as u8));
        Ok(out)
    }

    /// Read `(2, hi - lo)` as float32 without quantization (`read(..., quantized=False)`)
    pub fn read_f32(&self, lo: [usize; 3], hi: [usize; 3]) -> Result<Vec<f32>> {
        let raw = self.read_raw_in(lo, hi, Global)?;
        if self.dtype.kind == Kind::U8 {
            return Ok(raw.iter().map(|&v| v as f32).collect());
        }
        self.to_f32(&raw)
    }

    /// The reference's finite and [0, 1] check in the input type, then `astype("f4")`
    fn to_f32(&self, raw: &[u8]) -> Result<Vec<f32>> {
        let dtype = self.dtype;
        raw.chunks_exact(dtype.size())
            .map(|b| {
                let v = dtype.value(b);
                (v.is_finite() && (0.0..=1.0).contains(&v)).then_some(v as f32)
            })
            .collect::<Option<Vec<f32>>>()
            .ok_or_else(|| eyre!("{VALUE_ERROR}"))
    }

    /// Raw element bytes of `(2, hi - lo)`, C order, channels already in V, H order
    fn read_raw_in<A: Allocator>(&self, lo: [usize; 3], hi: [usize; 3], alloc: A) -> Result<Vec<u8, A>> {
        let dims = [0, 1, 2].map(|a| hi[a] - lo[a]);
        // from here on, coordinates are the stored array's
        let lo = [0, 1, 2].map(|a| lo[a] + self.offset[a]);
        let es = self.dtype.size();
        let plane = dims.iter().product::<usize>() * es;
        let mut out = Vec::with_capacity_in(2 * plane, alloc);
        out.resize(2 * plane, 0u8);
        match &*self.store {
            Store::Npy { map, offset, fortran } => {
                let s = self.full;
                // element strides of (c, z, y, x)
                let strides = match fortran {
                    false => [s[0] * s[1] * s[2], s[1] * s[2], s[2], 1],
                    true => [1, 2, 2 * s[0], 2 * s[0] * s[1]],
                };
                let data = &map[*offset..];
                for c in 0..2 {
                    for z in 0..dims[0] {
                        for y in 0..dims[1] {
                            let row = c * strides[0] + (lo[0] + z) * strides[1] + (lo[1] + y) * strides[2];
                            let dst = c * plane + ((z * dims[1] + y) * dims[2]) * es;
                            copy_strided(data, row + lo[2] * strides[3], strides[3], dims[2], es, &mut out[dst..]);
                        }
                    }
                }
            }
            Store::Zarr { chunks, .. } => {
                let first = [0, 1, 2].map(|a| lo[a] / chunks[a + 1]);
                let last = [0, 1, 2].map(|a| (lo[a] + dims[a] - 1) / chunks[a + 1]);
                for kc in 0..2usize.div_ceil(chunks[0]) {
                    for kz in first[0]..=last[0] {
                        for ky in first[1]..=last[1] {
                            for kx in first[2]..=last[2] {
                                self.copy_chunk([kc, kz, ky, kx], lo, dims, &mut out)?;
                            }
                        }
                    }
                }
            }
        }
        if self.reverse {
            let (a, b) = out.split_at_mut(plane);
            a.swap_with_slice(b);
        }
        Ok(out)
    }

    /// Copy the part of stored chunk `key` (channel chunk, z, y, x) that falls in the window at `lo`, in the stored
    /// array's coordinates
    fn copy_chunk(&self, key: [usize; 4], lo: [usize; 3], dims: [usize; 3], out: &mut [u8]) -> Result<()> {
        let Store::Zarr { chunks, .. } = &*self.store else { unreachable!("zarr only") };
        let es = self.dtype.size();
        let data = self.chunk_bytes(key)?;
        let cs = [chunks[1], chunks[2], chunks[3]];
        let origin = [0, 1, 2].map(|a| key[a + 1] * cs[a]);
        let a0 = [0, 1, 2].map(|a| lo[a].max(origin[a]));
        let a1 = [0, 1, 2].map(|a| (lo[a] + dims[a]).min(origin[a] + cs[a]).min(self.full[a]));
        let plane = dims.iter().product::<usize>() * es;
        let cplane = cs.iter().product::<usize>() * es;
        let n = (a1[2] - a0[2]) * es;
        let c0 = key[0] * chunks[0];
        for local in 0..chunks[0].min(2 - c0) {
            let c = c0 + local;
            for z in a0[0]..a1[0] {
                for y in a0[1]..a1[1] {
                    let src =
                        local * cplane + (((z - origin[0]) * cs[1] + y - origin[1]) * cs[2] + a0[2] - origin[2]) * es;
                    let dst = c * plane + (((z - lo[0]) * dims[1] + y - lo[1]) * dims[2] + a0[2] - lo[2]) * es;
                    out[dst..dst + n].copy_from_slice(&data[src..src + n]);
                }
            }
        }
        Ok(())
    }

    /// Decoded bytes of one stored chunk in C order, or the fill value when the chunk file is absent
    fn chunk_bytes(&self, key: [usize; 4]) -> Result<Vec<u8>> {
        let Store::Zarr { dir, chunks, sep, fill, compressor, fortran } = &*self.store else {
            unreachable!("zarr only")
        };
        let n = chunks.iter().product::<usize>() * self.dtype.size();
        let name = format!("{}{sep}{}{sep}{}{sep}{}", key[0], key[1], key[2], key[3]);
        let bytes = match fs::read(dir.join(&name)) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(fill.iter().copied().cycle().take(n).collect());
            }
            Err(e) => return Err(eyre!("{}: {e}", dir.join(&name).display())),
        };
        let out = decode(*compressor, bytes, n).map_err(|e| eyre!("chunk {name}: {e}"))?;
        ensure!(out.len() == n, "chunk {name} decoded to {} bytes, expected {n}", out.len());
        Ok(if *fortran { fortran_to_c(&out, *chunks, self.dtype.size()) } else { out })
    }

    /// Zarr chunk edge along each spatial axis, or `None` for NPY input
    pub fn chunk_shape(&self) -> Option<[usize; 3]> {
        match &*self.store {
            Store::Zarr { chunks, .. } => Some([chunks[1], chunks[2], chunks[3]]),
            Store::Npy { .. } => None,
        }
    }

    /// One whole spatial chunk as uint8 `(2, cz, cy, cx)` in V, H order, clipped to the array shape, in a buffer
    /// from `alloc`; returns the chunk's origin and extent too. NPY input is served in blocks of the same size
    ///
    /// A whole uint8 Blosc chunk holding both channels, which is what prediction writers store, is decoded
    /// straight into the returned buffer, its compressed bytes read into a buffer from `packed`; anything
    /// else is read through [`Predictions::read_u8`]'s path into a buffer from `alloc`
    pub fn read_chunk_in<A: Allocator + Copy, P: Allocator>(
        &self,
        key: [usize; 3],
        edge: [usize; 3],
        alloc: A,
        packed: P,
    ) -> Result<NativeChunk<A>> {
        let lo = [0, 1, 2].map(|a| key[a] * edge[a]);
        let hi = [0, 1, 2].map(|a| (lo[a] + edge[a]).min(self.shape[a]));
        let dims = [0, 1, 2].map(|a| hi[a] - lo[a]);
        if let Some(data) = self.whole_chunk_in(key, edge, dims, alloc, packed)? {
            return Ok(NativeChunk { lo, dims, data });
        }
        Ok(NativeChunk { lo, dims, data: self.read_u8_in(lo, hi, alloc)? })
    }

    /// The fast path of [`Predictions::read_chunk_in`]: the chunk's decoded bytes are already `(2, dims)` in C order
    /// when its file holds both channels of an unclipped uint8 C-order chunk, so `read_raw`'s copy is the identity;
    /// `None` when the chunk is not of that kind
    fn whole_chunk_in<A: Allocator + Copy, P: Allocator>(
        &self,
        key: [usize; 3],
        edge: [usize; 3],
        dims: [usize; 3],
        alloc: A,
        packed: P,
    ) -> Result<Option<Vec<u8, A>>> {
        if self.slab_layout() != Some(edge) || dims != edge {
            return Ok(None);
        }
        let n = 2 * dims.iter().product::<usize>();
        // the compressed bytes come from `packed`, so they never take the room `alloc` keeps for decoded chunks
        let mut out = match self.packed_chunk_in(key, packed)? {
            PackedChunk::Fill(fill) => {
                let mut out = Vec::with_capacity_in(n, alloc);
                out.resize(n, fill);
                return Ok(Some(out));
            }
            PackedChunk::Blosc(bytes) => blosc_decompress_in(&bytes, alloc).map_err(|e| eyre!("chunk {key:?}: {e}"))?,
        };
        ensure!(out.len() == n, "chunk {key:?} decoded to {} bytes, expected {n}", out.len());
        if self.reverse {
            let (a, b) = out.split_at_mut(n / 2);
            a.swap_with_slice(b);
        }
        Ok(Some(out))
    }

    /// The stored z, y, x extent of each channel of a chunk when the array's chunks can be decoded a few planes at
    /// a time: uint8 Blosc chunks in C order holding both channels. `None` for any other layout
    ///
    /// Zarr stores the chunks at the volume's edge at this full extent too, so the same strides address every chunk
    pub fn slab_layout(&self) -> Option<[usize; 3]> {
        let Store::Zarr { chunks, compressor, fortran, .. } = &*self.store else { return None };
        let sliced = chunks[0] == 2 && !*fortran && *compressor == Compressor::Blosc && self.dtype.kind == Kind::U8;
        sliced.then_some([chunks[1], chunks[2], chunks[3]])
    }

    /// The still-compressed bytes of chunk `key` of the crop, read into a buffer from `alloc`, for an array with a
    /// [`Predictions::slab_layout`]
    pub fn packed_chunk_in<A: Allocator>(&self, key: [usize; 3], alloc: A) -> Result<PackedChunk<A>> {
        self.stored_packed_in(self.stored_key(key)?, alloc)
    }

    /// [`Predictions::packed_chunk_in`] by the stored array's chunk key
    fn stored_packed_in<A: Allocator>(&self, key: [usize; 3], alloc: A) -> Result<PackedChunk<A>> {
        let Store::Zarr { dir, sep, fill, .. } = &*self.store else { bail!("only Zarr chunks are packed") };
        let name = format!("0{sep}{}{sep}{}{sep}{}", key[0], key[1], key[2]);
        let path = dir.join(&name);
        let file = match fs::File::open(&path) {
            Ok(f) => f,
            // a uint8 fill value is one byte, the same in both channels, so channel order does not matter
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(PackedChunk::Fill(fill[0])),
            Err(e) => return Err(eyre!("{}: {e}", path.display())),
        };
        let size = file.metadata().map_err(|e| eyre!("{}: {e}", path.display()))?.len() as usize;
        let mut bytes = Vec::with_capacity_in(size, alloc);
        read_exact_uninit(&file, &mut bytes.spare_capacity_mut()[..size])
            .map_err(|e| eyre!("{}: {e}", path.display()))?;
        // SAFETY: the read filled all `size` bytes
        unsafe { bytes.set_len(size) };
        Ok(PackedChunk::Blosc(bytes))
    }

    /// Decode z planes `planes` of both channels of a packed chunk into `out`, which must be empty, V then H, each
    /// `(planes, cy, cx)` at the stored extent of [`Predictions::slab_layout`]
    ///
    /// Blosc compresses a chunk in independent blocks (128 KiB, 8 planes of 128 × 128, in prediction zarrs), and
    /// only the blocks the planes fall in are decoded. They are decoded into `out`'s spare capacity, which is
    /// reserved as needed: zeroing a buffer that the decode overwrites whole took longer than reading the slab's
    /// compressed bytes
    pub fn decode_planes<A: Allocator, B: Allocator>(
        &self,
        packed: &PackedChunk<A>,
        planes: std::ops::Range<usize>,
        out: &mut Vec<u8, B>,
    ) -> Result<()> {
        let edge = self.slab_layout().ok_or_else(|| eyre!("the array's chunks cannot be decoded by planes"))?;
        let plane = edge[1] * edge[2];
        let channel = edge[0] * plane;
        let len = planes.len() * plane;
        ensure!(planes.end <= edge[0] && out.is_empty(), "planes {planes:?} do not fit a {edge:?} chunk");
        out.reserve_exact(2 * len);
        let spare = &mut out.spare_capacity_mut()[..2 * len];
        match packed {
            PackedChunk::Fill(fill) => spare.fill(MaybeUninit::new(*fill)),
            PackedChunk::Blosc(bytes) => {
                let (v, h) = spare.split_at_mut(len);
                // stored channel order is V, H, or H, V when `reverse`
                let parts = if self.reverse { [h, v] } else { [v, h] };
                frame_channel_ranges(bytes, channel, planes.start * plane, parts)?;
            }
        }
        // SAFETY: both channels' `len` bytes were written, by the fill or by a decode that returned success
        unsafe { out.set_len(2 * len) };
        Ok(())
    }

    /// Where planes `planes` of both channels of a packed chunk are, V then H, for a decoder outside c-blosc (the
    /// GPU's zstd), when each channel's planes are one block it can reproduce on its own; `None` when they are not,
    /// and [`Predictions::decode_planes`] decodes them instead
    ///
    /// Each channel's planes must be exactly one Blosc block: a single zstd frame or a block stored as is, with no
    /// shuffle to undo (byte shuffle at one-byte items moves nothing) and no split into streams per item byte
    // only GPU finishing reads slabs this way
    #[cfg_attr(not(all(target_os = "linux", target_arch = "x86_64")), allow(dead_code))]
    pub fn plane_blocks<A: Allocator>(
        &self,
        packed: &PackedChunk<A>,
        planes: std::ops::Range<usize>,
    ) -> Option<[BlockSource; 2]> {
        let edge = self.slab_layout()?;
        let plane = edge[1] * edge[2];
        if planes.is_empty() || planes.end > edge[0] {
            return None;
        }
        let bytes = match packed {
            PackedChunk::Fill(fill) => return Some([BlockSource::Fill(*fill), BlockSource::Fill(*fill)]),
            PackedChunk::Blosc(bytes) => bytes,
        };
        let [first, second] = frame_channel_blocks(bytes, edge[0] * plane, planes.start * plane, planes.len() * plane)?;
        // stored channel order is V, H, or H, V when `reverse`
        Some(if self.reverse { [second, first] } else { [first, second] })
    }

    /// Whether the array's Blosc blocks each fall within runs of `planes` planes, so decoding those planes decodes
    /// no more than they hold
    ///
    /// Judged by one stored chunk, since a writer compresses every chunk of an array alike; it only decides whether
    /// reading by planes is worth its cache of compressed chunks. Each chunk is checked again as it is read
    /// ([`Predictions::planes_fit`]), so a store that mixes block sizes is still decoded once per chunk
    pub fn blocks_fit_planes(&self, planes: usize) -> bool {
        let Some(edge) = self.slab_layout() else { return false };
        let run = planes * edge[1] * edge[2];
        match self.stored_chunk().map(|key| self.stored_packed_in(key, Global)) {
            Some(Ok(PackedChunk::Blosc(bytes))) => blocks_fit(&bytes, run),
            // an array with no chunk stored holds only its fill value, which no decode reads
            None => true,
            Some(Ok(PackedChunk::Fill(_)) | Err(_)) => false,
        }
    }

    /// Whether decoding `planes` planes of `packed` at a time decodes no more than they hold; a chunk the writer
    /// never stored decodes nothing
    pub fn planes_fit<A: Allocator>(&self, packed: &PackedChunk<A>, planes: usize) -> bool {
        let Some(edge) = self.slab_layout() else { return false };
        match packed {
            PackedChunk::Fill(_) => true,
            PackedChunk::Blosc(bytes) => blocks_fit(bytes, planes * edge[1] * edge[2]),
        }
    }

    /// The stored array's key of a chunk it stores, found by listing its directory, or `None` when it stores none
    ///
    /// Listing instead of opening keys in grid order finds a chunk however sparse the array is
    fn stored_chunk(&self) -> Option<[usize; 3]> {
        let Store::Zarr { dir, sep, chunks, .. } = &*self.store else { return None };
        let grid = [0, 1, 2].map(|a| self.full[a].div_ceil(chunks[a + 1]));
        let key = |parts: &[&str]| -> Option<[usize; 3]> {
            let [c, z, y, x] = parts else { return None };
            let k = [z.parse().ok()?, y.parse().ok()?, x.parse().ok()?];
            (*c == "0" && (0..3).all(|a| k[a] < grid[a])).then_some(k)
        };
        // listed lazily: a whole-scroll array can store a million chunks, and the first will do
        let names =
            |d: &Path| fs::read_dir(d).into_iter().flatten().flatten().filter_map(|e| e.file_name().into_string().ok());
        if *sep != '/' {
            return names(dir).find_map(|n| key(&n.split(*sep).collect::<Vec<_>>()));
        }

        // nested keys: 0/z/y/x
        for z in names(&dir.join("0")) {
            for y in names(&dir.join("0").join(&z)) {
                for x in names(&dir.join("0").join(&z).join(&y)) {
                    if let Some(k) = key(&["0", &z, &y, &x]) {
                        return Some(k);
                    }
                }
            }
        }
        None
    }
}

/// A chunk of a [`Predictions::slab_layout`] array as stored, for [`Predictions::decode_planes`]
pub enum PackedChunk<A: Allocator> {
    /// The chunk's Blosc frame
    Blosc(Vec<u8, A>),
    /// A chunk the writer never stored: every voxel holds the fill value
    Fill(u8),
}

/// Where one channel's run of planes is in a packed chunk, as [`Predictions::plane_blocks`] finds it
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BlockSource {
    /// One zstd frame of the chunk's bytes that decodes to the run, its header checked by [`zstd_frame_fits`]
    Zstd(std::ops::Range<usize>),
    /// The run's bytes as stored: a block zstd could not shrink, or a chunk Blosc stored uncompressed
    Raw(std::ops::Range<usize>),
    /// A chunk the writer never stored: every byte of the run is this value
    Fill(u8),
}

/// One spatial chunk of predictions read by [`Predictions::read_chunk_in`]: uint8 `(2, dims)` in V, H order, C order
pub struct NativeChunk<A: Allocator> {
    /// Origin in voxels, z, y, x
    pub lo: [usize; 3],
    /// Extent in voxels, clipped to the array
    pub dims: [usize; 3],
    pub data: Vec<u8, A>,
}

/// Copy `count` elements of `es` bytes, `stride` elements apart from element `first`, contiguously into `out`
fn copy_strided(data: &[u8], first: usize, stride: usize, count: usize, es: usize, out: &mut [u8]) {
    if stride == 1 {
        out[..count * es].copy_from_slice(&data[first * es..(first + count) * es]);
        return;
    }
    for i in 0..count {
        let at = (first + i * stride) * es;
        out[i * es..(i + 1) * es].copy_from_slice(&data[at..at + es]);
    }
}

/// A 4-D chunk of `dims` in Fortran order, reordered to C order
fn fortran_to_c(data: &[u8], dims: [usize; 4], es: usize) -> Vec<u8> {
    let mut out = vec![0u8; data.len()];
    let mut at = 0;
    for i0 in 0..dims[0] {
        for i1 in 0..dims[1] {
            for i2 in 0..dims[2] {
                for i3 in 0..dims[3] {
                    let src = (i0 + dims[0] * (i1 + dims[1] * (i2 + dims[2] * i3))) * es;
                    out[at..at + es].copy_from_slice(&data[src..src + es]);
                    at += es;
                }
            }
        }
    }
    out
}

impl Compressor {
    /// The compressor a `.zarray` names
    pub(crate) fn of(meta: &Value) -> Result<Self> {
        Ok(match meta["compressor"]["id"].as_str() {
            None => Self::Raw,
            Some("blosc") => Self::Blosc,
            Some("zstd") => Self::Zstd,
            Some("zlib") => Self::Zlib,
            Some("gzip") => Self::Gzip,
            Some("lz4") => Self::Lz4,
            Some(other) => bail!("unsupported Zarr compressor {other}"),
        })
    }
}

/// One chunk file through its compressor; `n` is the decoded size
pub(crate) fn decode(compressor: Compressor, bytes: Vec<u8>, n: usize) -> Result<Vec<u8>> {
    Ok(match compressor {
        Compressor::Raw => bytes,
        Compressor::Blosc => blosc_decompress(&bytes)?,
        Compressor::Zstd => zstd::stream::decode_all(&bytes[..])?,
        Compressor::Zlib => {
            let mut out = Vec::with_capacity(n);
            ZlibDecoder::new(&bytes[..]).read_to_end(&mut out)?;
            out
        }
        Compressor::Gzip => {
            let mut out = Vec::with_capacity(n);
            MultiGzDecoder::new(&bytes[..]).read_to_end(&mut out)?;
            out
        }
        Compressor::Lz4 => {
            // numcodecs' LZ4: the decoded size as a little-endian int32, then one LZ4 block
            ensure!(bytes.len() >= 4, "LZ4 chunk too short");
            let size = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            ensure!(size >= 0, "bad LZ4 chunk size {size}");
            lz4_flex::block::decompress(&bytes[4..], size as usize).map_err(|e| eyre!("LZ4: {e}"))?
        }
    })
}

/// Decompress one Blosc 1 frame with c-blosc
pub(crate) fn blosc_decompress(src: &[u8]) -> Result<Vec<u8>> {
    blosc_decompress_in(src, Global)
}

/// Decompress one Blosc 1 frame with c-blosc into a buffer from `alloc`
pub(crate) fn blosc_decompress_in<A: Allocator>(src: &[u8], alloc: A) -> Result<Vec<u8, A>> {
    let mut nbytes = 0usize;
    // SAFETY: c-blosc reads at most `src.len()` bytes of the header to validate it
    let valid = unsafe { blosc_src::blosc_cbuffer_validate(src.as_ptr().cast(), src.len(), &mut nbytes) };
    ensure!(valid == 0, "not a valid Blosc frame");
    let mut out = Vec::with_capacity_in(nbytes, alloc);
    if nbytes == 0 {
        return Ok(out);
    }
    // SAFETY: the frame was validated against `src.len()`, and c-blosc writes at most `nbytes` bytes into the
    // buffer's capacity; the context call keeps no global state, so rayon threads may decode at once
    let n = unsafe { blosc_src::blosc_decompress_ctx(src.as_ptr().cast(), out.as_mut_ptr().cast(), nbytes, 1) };
    ensure!(n >= 0 && n as usize == nbytes, "Blosc frame failed to decode ({n})");
    // SAFETY: c-blosc reported writing all `nbytes` bytes
    unsafe { out.set_len(nbytes) };
    Ok(out)
}

/// The header fields of a Blosc 1 frame that decoding part of it depends on
struct FrameInfo {
    /// Decoded size
    nbytes: usize,
    typesize: usize,
    /// Bytes per independently compressed block
    blocksize: usize,
}

/// c-blosc's `BLOSC_MAX_BLOCKSIZE`: the largest block it decodes, so its scratch of three blocks and a size per
/// item byte fits an `int`
const BLOSC_MAX_BLOCKSIZE: usize = (i32::MAX as usize - 255 * 4) / 3;

/// Validate a Blosc 1 frame and read its header: version, version, flags, typesize, then nbytes, blocksize and
/// cbytes as little-endian u32
///
/// Refuses the block and item sizes c-blosc's decoders refuse, so a frame they would not decode gives no block
/// sources for a decoder outside c-blosc either
fn blosc_frame_info(src: &[u8]) -> Result<FrameInfo> {
    let mut nbytes = 0usize;
    // SAFETY: c-blosc reads at most `src.len()` bytes of the header to validate it
    let valid = unsafe { blosc_src::blosc_cbuffer_validate(src.as_ptr().cast(), src.len(), &mut nbytes) };
    ensure!(valid == 0 && src.len() >= 16, "not a valid Blosc frame");
    let blocksize = u32::from_le_bytes(src[8..12].try_into().expect("4 bytes")) as usize;
    let typesize = src[3] as usize;
    // c-blosc decodes an empty frame without reading its block size, which it writes as 0
    let sized = blocksize > 0 && blocksize <= nbytes.min(BLOSC_MAX_BLOCKSIZE) && typesize > 0;
    ensure!(nbytes == 0 || sized, "Blosc frame has {blocksize}-byte blocks of {typesize}-byte items");
    Ok(FrameInfo { nbytes, typesize, blocksize })
}

/// Decode the same byte range of both channels of a frame holding two channels of `channel` bytes each: bytes
/// `start..` of each channel into the matching part, decoding only the blocks they fall in where it can
fn frame_channel_ranges(bytes: &[u8], channel: usize, start: usize, parts: [&mut [MaybeUninit<u8>]; 2]) -> Result<()> {
    // a frame of another size would put one channel's bytes where the other's are read
    let frame = blosc_frame_info(bytes)?;
    ensure!(frame.nbytes == 2 * channel, "chunk frame holds {} bytes, expected {}", frame.nbytes, 2 * channel);
    let len = parts[0].len();
    ensure!(parts[1].len() == len && start + len <= channel, "bytes {start}.. lie outside a channel");
    let starts = [start, channel + start];
    let ts = frame.typesize;
    let ranged = item_ranges_decode_whole_blocks(&frame) && frame.blocksize <= len;
    if ranged && len.is_multiple_of(ts) && starts.iter().all(|s| s.is_multiple_of(ts)) {
        for (at, part) in starts.into_iter().zip(parts) {
            blosc_getitem(bytes, at, part)?;
        }
        return Ok(());
    }

    // c-blosc reads a range in whole items of the frame's typesize, so a frame whose items straddle the range's
    // bounds is decoded whole, as is one whose blocks a range read could take short, and one whose blocks are larger
    // than the range: reading each channel's range would decode a block holding both twice
    let all = blosc_decompress(bytes)?;
    for (at, part) in starts.into_iter().zip(parts) {
        part.write_copy_of_slice(&all[at..at + len]);
    }
    Ok(())
}

/// Whether decoding `run` aligned bytes of a frame decodes no more than they hold: its blocks are no larger, and
/// its items divide the run
fn blocks_fit(bytes: &[u8], run: usize) -> bool {
    blosc_frame_info(bytes)
        .is_ok_and(|f| f.blocksize <= run && item_ranges_decode_whole_blocks(&f) && run.is_multiple_of(f.typesize))
}

/// Whether c-blosc's `blosc_getitem` decodes every block of the frame in full
///
/// A block is compressed as `typesize` streams of `blocksize / typesize` bytes each. When the type size does not
/// divide the block size, which c-blosc never writes but a damaged header can claim, the streams cover less than
/// the block; `blosc_getitem` does not check how much it decoded and copies the rest from its scratch buffer,
/// where decoding the whole frame fails
fn item_ranges_decode_whole_blocks(frame: &FrameInfo) -> bool {
    frame.typesize > 0 && frame.blocksize > 0 && frame.blocksize.is_multiple_of(frame.typesize)
}

/// Decode bytes `start..start + out.len()` of one Blosc 1 frame into `out`, decoding only the blocks they fall in;
/// on success every byte of `out` is written
fn blosc_getitem(src: &[u8], start: usize, out: &mut [MaybeUninit<u8>]) -> Result<()> {
    let frame = blosc_frame_info(src)?;
    let FrameInfo { nbytes, typesize, blocksize } = frame;
    ensure!(
        item_ranges_decode_whole_blocks(&frame),
        "Blosc frame has {blocksize}-byte blocks of {typesize}-byte items"
    );
    // c-blosc counts `start` and the length in items of the frame's typesize
    ensure!(start.is_multiple_of(typesize) && out.len().is_multiple_of(typesize), "bytes {start}.. split a Blosc item");
    ensure!(start + out.len() <= nbytes && nbytes <= i32::MAX as usize, "bytes {start}.. lie outside the frame");
    let (first, items) = ((start / typesize) as i32, (out.len() / typesize) as i32);
    // SAFETY: the frame was validated against `src.len()` and the range lies inside it, so c-blosc writes exactly
    // `out.len()` bytes; it decodes with a context of its own, so rayon threads may call it at once
    let n = unsafe { blosc_src::blosc_getitem(src.as_ptr().cast(), first, items, out.as_mut_ptr().cast()) };
    ensure!(n >= 0 && n as usize == out.len(), "Blosc frame failed to decode bytes {start}.. ({n})");
    Ok(())
}

/// Blosc 1 header flags (c-blosc's `blosc.h`, and `blosc_d` in `blosc.c` for how decoding reads them)
const BLOSC_SHUFFLE: u8 = 0x01;
const BLOSC_MEMCPYED: u8 = 0x02;
const BLOSC_BITSHUFFLE: u8 = 0x04;
/// Unused by Blosc 1, which refuses a frame that sets it (Blosc 2 marks delta coding with it)
const BLOSC_FUTURE: u8 = 0x08;
const BLOSC_DONT_SPLIT: u8 = 0x10;
/// The compressor format in the flags' top three bits
const BLOSC_ZSTD_FORMAT: u8 = 4;
/// c-blosc's `BLOSC_ZSTD_VERSION_FORMAT`, the one zstd codec version it decodes
const BLOSC_ZSTD_VERSION_FORMAT: u8 = 1;

/// Where bytes `start..start + len` of both channels of a frame holding two channels of `channel` bytes each are,
/// in stored order, when each is a whole block a zstd decoder or a copy reproduces with nothing left to undo
///
/// Mirrors c-blosc's `blosc_d`: a block is split into one stream per item byte unless the frame says not to or
/// its items are too large or too few, and a stream whose compressed size equals its decoded size is stored as
/// is. `None` for anything else: a frame c-blosc would refuse, another codec, bit shuffle, byte shuffle of wider
/// items, split blocks, a range that is not one block per channel, or a zstd frame [`zstd_frame_fits`] refuses
fn frame_channel_blocks(bytes: &[u8], channel: usize, start: usize, len: usize) -> Option<[BlockSource; 2]> {
    // the header checks c-blosc makes, so a frame [`Predictions::decode_planes`] refuses, such as one with bytes
    // past its recorded length or a block size it does not decode, is refused here too, and the header's
    // compressed size is the frame's length
    let FrameInfo { nbytes, typesize, blocksize } = blosc_frame_info(bytes).ok()?;
    let (version, flags, cbytes) = (bytes[0], bytes[2], bytes.len());
    let shuffled = flags & BLOSC_SHUFFLE != 0 && typesize > 1;
    if version != 2 || flags & (BLOSC_FUTURE | BLOSC_BITSHUFFLE) != 0 || shuffled {
        return None;
    }
    // a frame of another size would put one channel's bytes where the other's are read
    if nbytes != 2 * channel || len == 0 || start + len > channel {
        return None;
    }
    let frame = bytes;
    let starts = [start, channel + start];
    if flags & BLOSC_MEMCPYED != 0 {
        // the decoded bytes follow the header as they are
        return (cbytes == nbytes + 16).then(|| starts.map(|at| BlockSource::Raw(16 + at..16 + at + len)));
    }

    // one block per channel, so the channels' blocks are blocks `start / len` and `(channel + start) / len`, and
    // the last block is never a short one
    // c-blosc's `blosc_d` refuses a zstd frame of another codec version (byte 1) before decoding a block
    let zstd = flags >> 5 == BLOSC_ZSTD_FORMAT && bytes[1] == BLOSC_ZSTD_VERSION_FORMAT;
    if !zstd || blocksize != len || !start.is_multiple_of(len) || !channel.is_multiple_of(len) {
        return None;
    }
    let split = flags & BLOSC_DONT_SPLIT == 0 && typesize <= 16 && blocksize / typesize >= 128;
    if split && typesize > 1 {
        return None;
    }
    let nblocks = nbytes / blocksize;
    let word = |at: usize| Some(i32::from_le_bytes(frame.get(at..at + 4)?.try_into().ok()?));
    let block = |at: usize| -> Option<BlockSource> {
        let begin = usize::try_from(word(16 + 4 * (at / blocksize))?).ok()?;
        let csize = usize::try_from(word(begin)?).ok()?;
        let range = begin + 4..(begin + 4).checked_add(csize).filter(|&end| end <= cbytes)?;
        if csize == blocksize {
            return Some(BlockSource::Raw(range));
        }
        zstd_frame_fits(&frame[range.clone()], blocksize).then_some(BlockSource::Zstd(range))
    };
    // the block offsets table must lie inside the frame, as c-blosc checks before decoding
    if 16 + 4 * nblocks > cbytes {
        return None;
    }
    Some([block(starts[0])?, block(starts[1])?])
}

/// Whether `frame` is exactly one zstd frame that decodes to `size` bytes without a dictionary, as far as its
/// header and block headers tell: the zstd magic, a content size of `size` when the header records one, no
/// dictionary id, and blocks that end where `frame` does
///
/// nvCOMP treats a malformed frame as undefined behaviour, so these are checked on the CPU before a frame goes
/// to the GPU; the decoded size and status are still checked after
pub fn zstd_frame_fits(frame: &[u8], size: usize) -> bool {
    const MAGIC: u32 = 0xFD2F_B528;
    let Some((magic, rest)) = frame.split_first_chunk::<4>() else { return false };
    let Some(&descriptor) = rest.first() else { return false };
    // the descriptor's dictionary id flag (bits 0-1) and reserved bit (3)
    if u32::from_le_bytes(*magic) != MAGIC || descriptor & 0x0b != 0 {
        return false;
    }
    let content = zstd::zstd_safe::get_frame_content_size(frame);
    let sized = matches!(content, Ok(Some(n)) if n == size as u64) || matches!(content, Ok(None));
    sized && zstd::zstd_safe::find_frame_compressed_size(frame) == Ok(frame.len())
}

/// `Read::read_exact` of `file` from its start into `buf` without initialising `buf` first; on success every byte is
/// written
fn read_exact_uninit(file: &fs::File, buf: &mut [MaybeUninit<u8>]) -> io::Result<()> {
    let mut done = 0;
    while done < buf.len() {
        let rest = &mut buf[done..];
        // SAFETY: `rest` is valid for writes of `rest.len()` bytes, and pread writes at most that many
        let n = unsafe { libc::pread(file.as_raw_fd(), rest.as_mut_ptr().cast(), rest.len(), done as libc::off_t) };
        match n {
            0 => return Err(io::ErrorKind::UnexpectedEof.into()),
            n if n > 0 => done += n as usize,
            _ => {
                let e = io::Error::last_os_error();
                if e.kind() != io::ErrorKind::Interrupted {
                    return Err(e);
                }
            }
        }
    }
    Ok(())
}

fn open_npy(path: &Path) -> Result<(Store, Vec<usize>, Dtype)> {
    let file = fs::File::open(path)?;
    // SAFETY: the input is read-only for the whole run; the reference memory-maps it the same way
    let map = unsafe { Mmap::map(&file)? };
    ensure!(map.len() >= 10 && &map[..6] == b"\x93NUMPY", "{}: not an NPY file", path.display());
    let (hlen, start) = match map[6] {
        1 => (u16::from_le_bytes([map[8], map[9]]) as usize, 10),
        _ => (u32::from_le_bytes([map[8], map[9], map[10], map[11]]) as usize, 12),
    };
    let header = std::str::from_utf8(&map[start..start + hlen])?;
    let field = |name: &str| -> Result<&str> {
        let at = header.find(&format!("'{name}'")).ok_or_else(|| eyre!("NPY header lacks {name}"))?;
        let rest = header[at + name.len() + 2..].trim_start().trim_start_matches(':').trim_start();
        Ok(rest)
    };
    let descr = field("descr")?;
    let descr = descr.trim_start_matches('\'').split('\'').next().unwrap_or_default();
    let dtype = Dtype::parse(descr)?;
    let fortran = field("fortran_order")?.starts_with("True");
    let shape_text = field("shape")?;
    let shape_text = &shape_text[1..shape_text.find(')').ok_or_else(|| eyre!("bad NPY shape"))?];
    let shape: Vec<usize> =
        shape_text.split(',').map(str::trim).filter(|s| !s.is_empty()).map(str::parse).collect::<Result<_, _>>()?;
    let offset = start + hlen;
    ensure!(
        map.len() >= offset + shape.iter().product::<usize>() * dtype.size(),
        "{}: NPY file is truncated",
        path.display()
    );
    Ok((Store::Npy { map, offset, fortran }, shape, dtype))
}

fn open_zarr(path: &Path, key: Option<&str>) -> Result<(Store, Vec<usize>, Dtype)> {
    let dir = match key {
        Some(k) if !k.is_empty() => {
            let d = path.join(k);
            if !d.join(".zarray").is_file() {
                bail!("Zarr array not found: {k}");
            }
            d
        }
        _ => path.to_path_buf(),
    };
    let text = fs::read_to_string(dir.join(".zarray"))
        .map_err(|_| eyre!("Predictions must be an array with shape (2, Z, Y, X)"))?;
    let meta: Value = serde_json::from_str(&text)?;
    ensure!(meta["zarr_format"] == 2, "only Zarr v2 is supported");
    let fortran = match meta["order"].as_str() {
        Some("C") => false,
        Some("F") => true,
        other => bail!("unsupported Zarr order {other:?}"),
    };
    // numcodecs filters (delta, scale-offset, quantize and so on) are not used by any prediction writer
    ensure!(meta["filters"].as_array().is_none_or(Vec::is_empty), "Zarr filters are not supported");
    let shape: Vec<usize> = serde_json::from_value(meta["shape"].clone())?;
    let chunks: Vec<usize> = serde_json::from_value(meta["chunks"].clone())?;
    ensure!(shape.len() == 4 && chunks.len() == 4, "Predictions must be an array with shape (2, Z, Y, X)");
    ensure!(chunks.iter().all(|&c| c > 0), "Zarr chunks must be positive");
    let dtype = Dtype::parse(meta["dtype"].as_str().ok_or_else(|| eyre!("{DTYPE_ERROR}"))?)?;
    let sep = match meta.get("dimension_separator").and_then(Value::as_str) {
        Some("/") => '/',
        _ => '.',
    };
    let compressor = Compressor::of(&meta)?;
    let fill = dtype.encode(fill_value(&meta["fill_value"])?);
    let chunks = [chunks[0], chunks[1], chunks[2], chunks[3]];
    Ok((Store::Zarr { dir, chunks, sep, fill, compressor, fortran }, shape, dtype))
}

/// A Zarr v2 `fill_value`: a number, null (zero here; zarr leaves such chunks uninitialized), or one of
/// the strings the spec uses for non-finite floats
fn fill_value(v: &Value) -> Result<f64> {
    Ok(match v {
        Value::Null => 0.0,
        Value::Number(n) => n.as_f64().ok_or_else(|| eyre!("bad Zarr fill value {n}"))?,
        Value::String(s) => match s.as_str() {
            "NaN" => f64::NAN,
            "Infinity" => f64::INFINITY,
            "-Infinity" => f64::NEG_INFINITY,
            other => bail!("unsupported Zarr fill value {other:?}"),
        },
        other => bail!("unsupported Zarr fill value {other}"),
    })
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::mem::MaybeUninit;
    use std::path::Path;

    use serde_json::Value;

    use super::{
        BlockSource, NativeChunk, PackedChunk, Predictions, blocks_fit, blosc_decompress, blosc_getitem,
        frame_channel_blocks, frame_channel_ranges, zstd_frame_fits,
    };
    use crate::arena::BufferPool;

    /// Every case in `tests/fixtures/readers` (the expected values were written by the reference's own
    /// reader): the same bytes as `Predictions.read`, quantized and not, over the whole
    /// array and a window that cuts chunks, or the same error
    #[test]
    fn readers_match_the_reference_on_every_fixture() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/readers");
        let manifest: Vec<Value> =
            serde_json::from_str(&std::fs::read_to_string(root.join("manifest.json")).expect("manifest"))
                .expect("manifest json");
        assert!(manifest.len() >= 30, "fixtures missing");
        let expected =
            |case: &Value, field: &str| std::fs::read(root.join("expected").join(case[field].as_str().unwrap()));
        let corner = |v: &Value| -> [usize; 3] { serde_json::from_value(v.clone()).unwrap() };
        for case in &manifest {
            let name = case["name"].as_str().unwrap();
            let read = || -> color_eyre::Result<Vec<(String, Vec<u8>)>> {
                let p = Predictions::open(
                    &root.join(case["path"].as_str().unwrap()),
                    case["key"].as_str(),
                    case["order"].as_str().unwrap(),
                )?;
                let (lo, hi) = (corner(&case["window"][0]), corner(&case["window"][1]));
                let le = |v: Vec<f32>| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
                Ok(vec![
                    ("full_u8".into(), p.read_u8([0; 3], p.shape)?),
                    ("full_f32".into(), le(p.read_f32([0; 3], p.shape)?)),
                    ("window_u8".into(), p.read_u8(lo, hi)?),
                    ("window_f32".into(), le(p.read_f32(lo, hi)?)),
                ])
            };
            match (read(), case["error"].as_str()) {
                (Ok(reads), None) => {
                    for (field, got) in reads {
                        assert!(got == expected(case, &field).unwrap(), "{name}: {field} differs from the reference");
                    }
                }
                (Err(e), Some(message)) => assert_eq!(e.to_string(), message, "{name}"),
                (Ok(_), Some(message)) => panic!("{name}: read succeeded, the reference raised {message:?}"),
                (Err(e), None) => panic!("{name}: {e}"),
            }
        }
    }

    /// Every chunk of every readable fixture, read through a pool as the native cache does, holds the same
    /// bytes as the reference reader's window over it, whether it takes the direct Blosc path or not
    #[test]
    fn pooled_chunk_reads_match_window_reads() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/readers");
        let manifest: Vec<Value> =
            serde_json::from_str(&std::fs::read_to_string(root.join("manifest.json")).expect("manifest"))
                .expect("manifest json");
        let pool = BufferPool::new(1 << 20);
        let (mut checked, mut direct) = (0, 0);
        for case in manifest.iter().filter(|c| c["error"].is_null()) {
            let p = Predictions::open(
                &root.join(case["path"].as_str().unwrap()),
                case["key"].as_str(),
                case["order"].as_str().unwrap(),
            )
            .unwrap();
            let edge = p.chunk_shape().unwrap_or([64, 64, 64]);
            let grid = [0, 1, 2].map(|a| p.shape[a].div_ceil(edge[a]));
            for z in 0..grid[0] {
                for y in 0..grid[1] {
                    for x in 0..grid[2] {
                        let NativeChunk { lo, dims, data } = p.read_chunk_in([z, y, x], edge, &pool, &pool).unwrap();
                        let hi = [0, 1, 2].map(|a| lo[a] + dims[a]);
                        assert_eq!(&data[..], &p.read_u8(lo, hi).unwrap()[..], "{}", case["name"]);
                        checked += 1;
                        direct += p.whole_chunk_in([z, y, x], edge, dims, &pool, &pool).unwrap().is_some() as usize;
                    }
                }
            }
        }
        assert!(checked > 30 && direct > 10, "too few chunks checked: {checked}, {direct} directly");
    }

    /// Every run of planes of every chunk of every fixture that can be decoded by planes holds, in both channels,
    /// the same bytes as the reference reader's window over it; the planes past the array's edge are not compared
    #[test]
    fn plane_decodes_match_window_reads() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/readers");
        let manifest: Vec<Value> =
            serde_json::from_str(&std::fs::read_to_string(root.join("manifest.json")).expect("manifest"))
                .expect("manifest json");
        let pool = BufferPool::new(1 << 20);
        let (mut checked, mut reversed, mut sourced) = (0, 0, 0);
        for case in manifest.iter().filter(|c| c["error"].is_null()) {
            let order = case["order"].as_str().unwrap();
            let p = Predictions::open(&root.join(case["path"].as_str().unwrap()), case["key"].as_str(), order).unwrap();
            let Some(edge) = p.slab_layout() else { continue };
            let grid = [0, 1, 2].map(|a| p.shape[a].div_ceil(edge[a]));
            // runs of one plane and of half a chunk, so runs inside one Blosc block and across blocks both occur
            for width in [1, edge[0].div_ceil(2)] {
                for key in
                    (0..grid[0]).flat_map(|z| (0..grid[1]).flat_map(move |y| (0..grid[2]).map(move |x| [z, y, x])))
                {
                    let packed = p.packed_chunk_in(key, &pool).unwrap();
                    for first in (0..edge[0]).step_by(width) {
                        let planes = first..(first + width).min(edge[0]);
                        let lo = [key[0] * edge[0] + first, key[1] * edge[1], key[2] * edge[2]];
                        if lo[0] >= p.shape[0] {
                            continue;
                        }
                        let len = planes.len() * edge[1] * edge[2];
                        let mut out = Vec::new();
                        p.decode_planes(&packed, planes.clone(), &mut out).unwrap();
                        assert_eq!(out.len(), 2 * len);
                        if let Some(sources) = p.plane_blocks(&packed, planes.clone()) {
                            let bytes = match &packed {
                                PackedChunk::Blosc(bytes) => &bytes[..],
                                PackedChunk::Fill(_) => &[],
                            };
                            let got: Vec<u8> = sources.iter().flat_map(|s| decode_source(bytes, s, len)).collect();
                            assert!(got == out, "{} chunk {key:?} planes {planes:?}: block sources", case["name"]);
                            sourced += 1;
                        }
                        let hi = [0, 1, 2].map(|a| (lo[a] + [planes.len(), edge[1], edge[2]][a]).min(p.shape[a]));
                        let dims = [0, 1, 2].map(|a| hi[a] - lo[a]);
                        let window = p.read_u8(lo, hi).unwrap();
                        let half = dims.iter().product::<usize>();
                        for c in 0..2 {
                            for z in 0..dims[0] {
                                for y in 0..dims[1] {
                                    for x in 0..dims[2] {
                                        let got = out[c * len + (z * edge[1] + y) * edge[2] + x];
                                        let want = window[c * half + (z * dims[1] + y) * dims[2] + x];
                                        assert_eq!(got, want, "{} chunk {key:?} planes {planes:?}", case["name"]);
                                    }
                                }
                            }
                        }
                        checked += 1;
                        reversed += (order == "HV") as usize;
                    }
                }
            }
        }
        assert!(checked > 30 && sourced > 10, "too few plane runs checked: {checked}, {sourced} by block sources");
        println!("{checked} plane runs checked, {reversed} in H, V order, {sourced} by block sources");
    }

    /// Compress `data` as one Blosc 1 frame with zstd and shuffle, as zarr writes prediction chunks
    fn blosc_frame(data: &[u8], typesize: usize, blocksize: usize) -> Vec<u8> {
        blosc_frame_with(c"zstd", data, typesize, blocksize)
    }

    fn blosc_frame_with(name: &std::ffi::CStr, data: &[u8], typesize: usize, blocksize: usize) -> Vec<u8> {
        blosc_compress(3, 1, name, data, typesize, blocksize)
    }

    /// Compress `data` as one Blosc 1 frame at level `clevel` (0 stores it uncompressed) with shuffle mode `shuffle`
    /// (0 none, 1 byte, 2 bit)
    fn blosc_compress(
        clevel: i32,
        shuffle: i32,
        name: &std::ffi::CStr,
        data: &[u8],
        typesize: usize,
        blocksize: usize,
    ) -> Vec<u8> {
        let mut out = vec![0u8; data.len() + 64];
        // SAFETY: `out` has room for the frame and its 16-byte header; the context call keeps no global state
        let n = unsafe {
            blosc_src::blosc_compress_ctx(
                clevel,
                shuffle,
                typesize,
                data.len(),
                data.as_ptr().cast(),
                out.as_mut_ptr().cast(),
                out.len(),
                name.as_ptr(),
                blocksize,
                1,
            )
        };
        assert!(n > 0, "blosc_compress_ctx failed ({n})");
        out.truncate(n as usize);
        out
    }

    fn uninit(n: usize) -> Vec<MaybeUninit<u8>> {
        vec![MaybeUninit::uninit(); n]
    }

    /// Ranges of both channels decode to the stored bytes whatever the frame's item size and block size, and a
    /// frame of the wrong size is refused rather than read across channels
    #[test]
    fn channel_ranges_decode_any_frame() {
        let channel = 8 * 64 * 64;
        let data: Vec<u8> = (0..2 * channel).map(|i| (i * 7 + i / 977) as u8).collect();
        let run = 2 * 64 * 64;
        for (typesize, blocksize, fits) in
            [(1, run, true), (3, run, false), (1, 2 * channel, false), (4, run / 2, true)]
        {
            let frame = blosc_frame(&data, typesize, blocksize);
            assert_eq!(blocks_fit(&frame, run), fits, "typesize {typesize} blocksize {blocksize}");
            for start in [0, run, channel - run] {
                let (mut a, mut b) = (uninit(run), uninit(run));
                frame_channel_ranges(&frame, channel, start, [&mut a, &mut b]).unwrap();
                // SAFETY: a successful decode writes every byte of both parts
                let (a, b) = unsafe { (a.assume_init_ref(), b.assume_init_ref()) };
                assert_eq!(a, &data[start..start + run], "typesize {typesize} start {start}");
                assert_eq!(b, &data[channel + start..channel + start + run], "typesize {typesize} start {start}");
            }
        }
        let frame = blosc_frame(&data[..channel], 1, run);
        let (mut a, mut b) = (uninit(run), uninit(run));
        assert!(frame_channel_ranges(&frame, channel / 2 + 1, 0, [&mut a, &mut b]).is_err());
    }

    /// A header whose block size its type size does not divide, which c-blosc never writes, is refused as decoding
    /// the whole frame refuses it, where reading item ranges would fill part of each block from scratch memory
    ///
    /// LZ4 splits each block into one stream per byte of the type. Claiming one byte more per block than the four
    /// streams hold leaves every block but the last a byte short, and only the last fails to decode, so ranges
    /// that stop before it would read back as decoded
    #[test]
    fn a_damaged_block_size_is_refused() {
        let channel = 256 << 10;
        let data: Vec<u8> = (0..2 * channel).map(|i| (i * 7 + i / 977) as u8).collect();
        let mut frame = blosc_frame_with(c"lz4", &data, 4, 16 << 10);
        let block = u32::from_le_bytes(frame[8..12].try_into().unwrap()) as usize;
        assert_eq!((frame[2] & 0x10, block), (0, 64 << 10), "an LZ4 frame split into 64 KiB blocks");
        frame[8..12].copy_from_slice(&(block as u32 + 1).to_le_bytes());
        let len = 2 * block;
        assert!(!blocks_fit(&frame, len));
        assert!(blosc_getitem(&frame, 0, &mut uninit(block)).is_err());
        let (mut a, mut b) = (uninit(len), uninit(len));
        assert!(frame_channel_ranges(&frame, channel, 0, [&mut a, &mut b]).is_err());
    }

    /// A crop reads what the whole array holds at the crop's offset: whole and in part, quantized and not, chunk by
    /// chunk through the native cache's read, and plane by plane from packed chunks where the array has a slab layout
    fn assert_crop_reads(full: &Predictions, lo: [usize; 3], hi: [usize; 3]) {
        let crop = full.crop(lo, hi).unwrap();
        let at = |p: [usize; 3]| [0, 1, 2].map(|a| p[a] + lo[a]);
        assert_eq!(crop.shape, [0, 1, 2].map(|a| hi[a] - lo[a]));
        assert_eq!(crop.read_u8([0; 3], crop.shape).unwrap(), full.read_u8(lo, hi).unwrap());
        assert_eq!(crop.read_f32([0; 3], crop.shape).unwrap(), full.read_f32(lo, hi).unwrap());
        let (a, b) = ([1, 1, 1], crop.shape.map(|n| n - 1));
        assert_eq!(crop.read_u8(a, b).unwrap(), full.read_u8(at(a), at(b)).unwrap());

        let pool = BufferPool::new(1 << 20);
        let edge = crop.read_unit();
        let grid = [0, 1, 2].map(|a| crop.shape[a].div_ceil(edge[a]));
        for key in (0..grid[0]).flat_map(|z| (0..grid[1]).flat_map(move |y| (0..grid[2]).map(move |x| [z, y, x]))) {
            let NativeChunk { lo: clo, dims, data } = crop.read_chunk_in(key, edge, &pool, &pool).unwrap();
            let chi = [0, 1, 2].map(|a| clo[a] + dims[a]);
            assert_eq!(&data[..], &full.read_u8(at(clo), at(chi)).unwrap()[..], "chunk {key:?}");
            let Some(layout) = crop.slab_layout() else { continue };
            let (mut got, mut want) = (Vec::new(), Vec::new());
            crop.decode_planes(&crop.packed_chunk_in(key, &pool).unwrap(), 0..layout[0], &mut got).unwrap();
            let stored = [0, 1, 2].map(|a| key[a] + lo[a] / layout[a]);
            full.decode_planes(&full.packed_chunk_in(stored, &pool).unwrap(), 0..layout[0], &mut want).unwrap();
            assert_eq!(got, want, "packed chunk {key:?}");
        }
    }

    /// Write a C-order uint8 Zarr v2 array of (2, `shape`) in `chunks` of (channels, z, y, x), Blosc or raw
    fn write_zarr(dir: &Path, shape: [usize; 3], chunks: [usize; 4], blosc: bool) {
        let _ = fs::remove_dir_all(dir);
        fs::create_dir_all(dir).unwrap();
        let value = |c: usize, z: usize, y: usize, x: usize| (c * 89 + z * 31 + y * 7 + x * 3 + z * y * x / 5) as u8;
        let grid = [0, 1, 2, 3].map(|a| [2, shape[0], shape[1], shape[2]][a].div_ceil(chunks[a]));
        for k in (0..grid[0]).flat_map(|c| {
            (0..grid[1]).flat_map(move |z| (0..grid[2]).flat_map(move |y| (0..grid[3]).map(move |x| [c, z, y, x])))
        }) {
            let mut bytes = Vec::new();
            for c in k[0] * chunks[0]..(k[0] + 1) * chunks[0] {
                for z in k[1] * chunks[1]..(k[1] + 1) * chunks[1] {
                    for y in k[2] * chunks[2]..(k[2] + 1) * chunks[2] {
                        for x in k[3] * chunks[3]..(k[3] + 1) * chunks[3] {
                            let inside = c < 2 && z < shape[0] && y < shape[1] && x < shape[2];
                            bytes.push(if inside { value(c, z, y, x) } else { 0 });
                        }
                    }
                }
            }
            let bytes = if blosc { blosc_frame(&bytes, 1, 4 * chunks[2] * chunks[3]) } else { bytes };
            fs::write(dir.join(format!("{}.{}.{}.{}", k[0], k[1], k[2], k[3])), bytes).unwrap();
        }
        let compressor = match blosc {
            true => serde_json::json!({"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 1, "blocksize": 0}),
            false => Value::Null,
        };
        let meta = serde_json::json!({
            "zarr_format": 2, "shape": [2, shape[0], shape[1], shape[2]], "chunks": chunks,
            "dtype": "|u1", "fill_value": 0, "order": "C", "filters": null, "compressor": compressor,
        });
        fs::write(dir.join(".zarray"), meta.to_string()).unwrap();
    }

    /// Write an NPY file of float32 values in [0, 1], shape (2, `shape`), in Fortran or C order
    fn write_npy(path: &Path, shape: [usize; 3], fortran: bool) {
        let dims = [2, shape[0], shape[1], shape[2]];
        let mut header = format!(
            "{{'descr': '<f4', 'fortran_order': {}, 'shape': ({}, {}, {}, {}), }}",
            if fortran { "True" } else { "False" },
            dims[0],
            dims[1],
            dims[2],
            dims[3]
        );
        while (10 + header.len() + 1) % 64 != 0 {
            header.push(' ');
        }
        header.push('\n');
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend((header.len() as u16).to_le_bytes());
        bytes.extend(header.as_bytes());
        let value = |i: [usize; 4]| ((i[0] * 89 + i[1] * 31 + i[2] * 7 + i[3] * 3) % 256) as f32 / 255.0;
        let mut index = [0usize; 4];
        for _ in 0..dims.iter().product::<usize>() {
            bytes.extend(value(index).to_le_bytes());
            // the last axis varies fastest in C order, the first in Fortran order
            let order: [usize; 4] = if fortran { [0, 1, 2, 3] } else { [3, 2, 1, 0] };
            for a in order {
                index[a] += 1;
                if index[a] < dims[a] {
                    break;
                }
                index[a] = 0;
            }
        }
        fs::write(path, bytes).unwrap();
    }

    /// Crops of Zarr arrays chunked as native predictions are (Blosc chunks holding both channels, read by planes)
    /// and as tracing ones may be (raw chunks of one channel), and of NPY input in both orders, read the values the
    /// whole array holds at their offset; a crop off the chunk grid is refused
    #[test]
    fn crops_read_the_whole_array_at_their_offset() {
        let root = std::env::temp_dir().join(format!("fibrs-crop-{}", std::process::id()));
        let native = root.join("native.zarr");
        write_zarr(&native, [24, 48, 40], [2, 16, 16, 16], true);
        let full = Predictions::open(&native, None, "HV").unwrap();
        assert!(full.slab_layout().is_some());
        assert_crop_reads(&full, [0, 16, 16], [24, 48, 40]);
        assert_crop_reads(&full, [16, 0, 32], [24, 32, 40]);
        // a crop of a crop is a crop of the array
        let inner = full.crop([0, 16, 0], [24, 48, 40]).unwrap();
        assert_eq!(inner.crop([0, 16, 16], [24, 32, 32]).unwrap().read_u8([0; 3], [24, 16, 16]).unwrap(), {
            full.read_u8([0, 32, 16], [24, 48, 32]).unwrap()
        });
        assert!(full.crop([0, 8, 0], [24, 48, 40]).is_err());
        assert!(full.crop([0, 0, 0], [24, 48, 36]).is_err());
        assert!(full.crop([0, 0, 0], [24, 64, 40]).is_err());

        let tracing = root.join("tracing.zarr");
        write_zarr(&tracing, [12, 24, 20], [1, 8, 8, 8], false);
        let full = Predictions::open(&tracing, None, "VH").unwrap();
        assert!(full.slab_layout().is_none());
        assert_crop_reads(&full, [0, 8, 16], [12, 24, 20]);

        for fortran in [false, true] {
            let path = root.join(format!("pred-{fortran}.npy"));
            write_npy(&path, [4, 130, 70], fortran);
            let full = Predictions::open(&path, None, "VH").unwrap();
            assert_crop_reads(&full, [0, 64, 0], [4, 130, 70]);
            assert_crop_reads(&full, [0, 0, 64], [4, 128, 70]);
            assert!(full.crop([0, 32, 0], [4, 130, 70]).is_err());
        }
        let _ = fs::remove_dir_all(&root);
    }

    /// The `len` bytes a block source stands for, decoded on the CPU as a GPU decoder would: zstd by the zstd
    /// crate, raw bytes copied
    fn decode_source(bytes: &[u8], source: &BlockSource, len: usize) -> Vec<u8> {
        match source {
            BlockSource::Zstd(range) => {
                let out = zstd::bulk::decompress(&bytes[range.clone()], len).unwrap();
                assert_eq!(out.len(), len, "a zstd block decodes to {} bytes", out.len());
                out
            }
            BlockSource::Raw(range) => bytes[range.clone()].to_vec(),
            BlockSource::Fill(fill) => vec![*fill; len],
        }
    }

    /// Two channels of `blocks` blocks of `block` bytes, compressible except the first block of the second
    /// channel, which is noise zstd cannot shrink, so Blosc stores it as is
    fn two_channels(block: usize, blocks: usize) -> Vec<u8> {
        let channel = block * blocks;
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        (0..2 * channel)
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                if (channel..channel + block).contains(&i) { (state >> 56) as u8 } else { (i / 97 % 7 * 31) as u8 }
            })
            .collect()
    }

    /// Every block's source, zstd or stored as is, decodes on its own to the bytes c-blosc decodes for that run of
    /// both channels, from a frame written as zarr writes prediction chunks
    #[test]
    fn block_sources_reproduce_channel_ranges() {
        let (block, blocks) = (8 * 32 * 32, 4);
        let channel = block * blocks;
        let data = two_channels(block, blocks);
        let frame = blosc_frame(&data, 1, block);
        assert_eq!(frame[2], 0x91, "shuffle, don't split, zstd, as prediction chunks are written");
        let (mut zstd, mut raw) = (0, 0);
        for s in 0..blocks {
            let start = s * block;
            let sources = frame_channel_blocks(&frame, channel, start, block).expect("one zstd block per channel");
            let (mut a, mut b) = (uninit(block), uninit(block));
            frame_channel_ranges(&frame, channel, start, [&mut a, &mut b]).unwrap();
            // SAFETY: a successful decode writes every byte of both parts
            let (a, b) = unsafe { (a.assume_init_ref(), b.assume_init_ref()) };
            assert_eq!(decode_source(&frame, &sources[0], block), a, "block {s} of the first channel");
            assert_eq!(decode_source(&frame, &sources[1], block), b, "block {s} of the second channel");
            for source in &sources {
                zstd += matches!(source, BlockSource::Zstd(_)) as usize;
                raw += matches!(source, BlockSource::Raw(_)) as usize;
            }
        }
        assert_eq!((zstd, raw), (2 * blocks - 1, 1), "the noise block is stored as is, the rest as zstd");
        assert!(frame_channel_blocks(&frame, channel, channel, block).is_none());
    }

    /// A frame Blosc stored uncompressed gives each channel's run as a slice of the frame, at any run of planes
    #[test]
    fn a_stored_frame_gives_raw_ranges() {
        let (block, blocks) = (8 * 32 * 32, 4);
        let channel = block * blocks;
        let data = two_channels(block, blocks);
        let frame = blosc_compress(0, 1, c"zstd", &data, 1, block);
        assert_ne!(frame[2] & 0x02, 0, "level 0 stores the frame as is");
        for (start, len) in [(0, block), (block, 3 * block), (5, 77)] {
            let sources = frame_channel_blocks(&frame, channel, start, len).unwrap();
            assert!(sources.iter().all(|s| matches!(s, BlockSource::Raw(_))));
            assert_eq!(decode_source(&frame, &sources[0], len), data[start..start + len]);
            assert_eq!(decode_source(&frame, &sources[1], len), data[channel + start..channel + start + len]);
        }
    }

    /// Frames a lone zstd decoder or a copy cannot reproduce block by block are left to c-blosc: other codecs, bit
    /// shuffle, byte shuffle of wider items, blocks split per item byte, runs that are not one block, a frame of
    /// the wrong size, and a zstd frame whose header names a dictionary
    #[test]
    fn frames_a_zstd_decoder_cannot_reproduce_are_refused() {
        let (block, blocks) = (8 * 32 * 32, 4);
        let channel = block * blocks;
        let data = two_channels(block, blocks);
        let refused =
            |frame: &[u8], start: usize, len: usize| frame_channel_blocks(frame, channel, start, len).is_none();
        assert!(refused(&blosc_frame_with(c"lz4", &data, 1, block), 0, block));
        assert!(refused(&blosc_compress(3, 2, c"zstd", &data, 1, block), 0, block));
        assert!(refused(&blosc_frame(&data, 2, block), 0, block));
        assert!(refused(&blosc_frame(&data, 1, 2 * block), 0, block));
        assert!(refused(&blosc_frame(&data, 1, block), 0, 2 * block));
        assert!(refused(&blosc_frame(&data, 1, block), block / 2, block));
        assert!(frame_channel_blocks(&blosc_frame(&data, 1, block), channel / 2, 0, block).is_none());

        // without shuffle a wider item is fine, but clearing "don't split" makes c-blosc read one stream per item
        // byte; at one-byte items there is one stream either way
        let mut wide = blosc_compress(3, 0, c"zstd", &data, 2, block);
        assert!(!refused(&wide, 0, block));
        wide[2] &= !0x10;
        assert!(refused(&wide, 0, block));
        let mut narrow = blosc_frame(&data, 1, block);
        narrow[2] &= !0x10;
        assert!(!refused(&narrow, 0, block));

        let mut dictionary = blosc_frame(&data, 1, block);
        let Some([BlockSource::Zstd(range), _]) = frame_channel_blocks(&dictionary, channel, 0, block) else {
            panic!("a zstd block")
        };
        dictionary[range.start + 4] |= 0x01;
        assert!(refused(&dictionary, 0, block));
    }

    /// A chunk file with a byte past its Blosc frame, or a stored frame whose header claims a block size c-blosc
    /// does not decode, is refused by both decoders: c-blosc's header checks refuse it for the CPU, and the GPU's
    /// block sources check the same header, so neither decodes a file the other refuses
    #[test]
    fn a_frame_with_trailing_bytes_is_refused_by_both_decoders() {
        let edge = 16;
        let data = two_channels(8 * edge * edge, 2);
        let frame = blosc_frame(&data, 1, 8 * edge * edge);
        let stored = blosc_compress(0, 1, c"zstd", &data, 1, 8 * edge * edge);
        assert_ne!(stored[2] & 0x02, 0, "level 0 stores the frame as is");
        let with_blocksize = |blocksize: u32| {
            let mut bad = stored.clone();
            bad[8..12].copy_from_slice(&blocksize.to_le_bytes());
            bad
        };
        // c-blosc itself refuses these, not only the header check this crate makes before it
        for blocksize in [0, data.len() as u32 + 1] {
            assert!(blosc_decompress(&with_blocksize(blocksize)).is_err(), "c-blosc decodes {blocksize}-byte blocks");
        }
        let dir = std::env::temp_dir().join(format!("fibrs-trailing-byte-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("0.0.0.0"), &frame).unwrap();
        fs::write(dir.join("0.0.0.1"), [&frame[..], &[0]].concat()).unwrap();
        fs::write(dir.join("0.0.0.2"), &stored).unwrap();
        fs::write(dir.join("0.0.0.3"), with_blocksize(0)).unwrap();
        fs::write(dir.join("0.0.0.4"), with_blocksize(data.len() as u32 + 1)).unwrap();
        let mut codec = frame.clone();
        codec[1] = 2;
        fs::write(dir.join("0.0.0.5"), &codec).unwrap();
        let meta = serde_json::json!({
            "zarr_format": 2, "shape": [2, edge, edge, 6 * edge], "chunks": [2, edge, edge, edge],
            "dtype": "|u1", "fill_value": 0, "order": "C", "filters": null, "dimension_separator": ".",
            "compressor": {"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 1, "blocksize": 0},
        });
        fs::write(dir.join(".zarray"), meta.to_string()).unwrap();
        let p = Predictions::open(&dir, None, "VH").unwrap();
        let pool = BufferPool::new(1 << 20);

        let whole = p.packed_chunk_in([0, 0, 0], &pool).unwrap();
        p.decode_planes(&whole, 0..8, &mut Vec::new()).unwrap();
        assert!(p.plane_blocks(&whole, 0..8).is_some(), "the frame as Blosc wrote it gives block sources");

        let as_stored = p.packed_chunk_in([0, 0, 2], &pool).unwrap();
        p.decode_planes(&as_stored, 0..8, &mut Vec::new()).unwrap();
        assert!(p.plane_blocks(&as_stored, 0..8).is_some(), "the stored frame as Blosc wrote it gives block sources");

        for (x, what) in [
            (1, "trailing byte"),
            (3, "0-byte blocks"),
            (4, "blocks larger than the frame"),
            (5, "zstd codec version 2"),
        ] {
            let bad = p.packed_chunk_in([0, 0, x], &pool).unwrap();
            assert!(p.decode_planes(&bad, 0..8, &mut Vec::new()).is_err(), "the CPU decodes a frame with {what}");
            assert!(p.plane_blocks(&bad, 0..8).is_none(), "the GPU accepts a frame with {what} the CPU refuses");
        }
        let _ = fs::remove_dir_all(&dir);
    }

    /// The header check takes exactly one dictionary-free frame of the expected size, and nothing shorter or longer
    #[test]
    fn zstd_frames_are_checked_before_the_gpu_sees_them() {
        let data: Vec<u8> = (0..4096u32).map(|i| (i / 13) as u8).collect();
        let frame = zstd::bulk::compress(&data, 3).unwrap();
        assert!(zstd_frame_fits(&frame, data.len()));
        assert!(!zstd_frame_fits(&frame, data.len() + 1));
        assert!(!zstd_frame_fits(&frame[..frame.len() - 1], data.len()));
        assert!(!zstd_frame_fits(&[&frame[..], &[0]].concat(), data.len()));
        assert!(!zstd_frame_fits(&[&frame[..], &frame[..]].concat(), data.len()));
        assert!(!zstd_frame_fits(&frame[..3], data.len()));
        let mut magic = frame.clone();
        magic[0] ^= 1;
        assert!(!zstd_frame_fits(&magic, data.len()));
    }

    /// The probe judges an array by a chunk it stores even when every chunk before it in grid order is missing,
    /// with either key separator
    #[test]
    fn a_sparse_array_is_judged_by_a_stored_chunk() {
        let edge = 16;
        let data: Vec<u8> = (0..2 * edge * edge * edge).map(|i| (i * 7 + i / 977) as u8).collect();
        // one Blosc block for the whole chunk: decoding 8 planes would decode all 16
        let frame = blosc_frame(&data, 1, data.len());
        let root = std::env::temp_dir().join(format!("fibrs-sparse-probe-{}", std::process::id()));
        for sep in [".", "/"] {
            let dir = root.join(if sep == "." { "flat" } else { "nested" });
            let _ = fs::remove_dir_all(&dir);
            // 64 × 65 chunk positions: the stored one is past the first 4,096 in grid order
            let last = "0/0/63/64".replace('/', sep);
            fs::create_dir_all(dir.join(&last).parent().unwrap()).unwrap();
            fs::write(dir.join(&last), &frame).unwrap();
            let meta = serde_json::json!({
                "zarr_format": 2, "shape": [2, edge, 64 * edge, 65 * edge], "chunks": [2, edge, edge, edge],
                "dtype": "|u1", "fill_value": 0, "order": "C", "filters": null, "dimension_separator": sep,
                "compressor": {"id": "blosc", "cname": "zstd", "clevel": 5, "shuffle": 0, "blocksize": 0},
            });
            fs::write(dir.join(".zarray"), meta.to_string()).unwrap();
            let p = Predictions::open(&dir, None, "VH").unwrap();
            assert_eq!(p.stored_chunk(), Some([0, 63, 64]), "separator {sep}");
            assert!(!p.blocks_fit_planes(8), "separator {sep}");
        }
        let _ = fs::remove_dir_all(&root);
    }
}
