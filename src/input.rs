//! Prediction readers: Zarr v2 and NPY, as `Predictions` in the reference
//!
//! Mirrors `storage.Predictions`: a `(2, Z, Y, X)` array of uint8 or floats in [0, 1], returned in V, H
//! channel order, with floats quantized by `rint(x * 255)` in float32. Both array orders and both byte
//! orders are read. Zarr chunks are decoded by the c-blosc release numcodecs 0.15 bundles, or by
//! numcodecs' Zstd, Zlib, GZip and LZ4 formats

use std::alloc::{Allocator, Global};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use color_eyre::eyre::{Result, bail, ensure, eyre};
use flate2::read::{MultiGzDecoder, ZlibDecoder};
use half::f16;
use memmap2::Mmap;
use serde_json::Value;

const DTYPE_ERROR: &str = "Predictions must be uint8 [0,255] or floating point [0,1]";
const VALUE_ERROR: &str = "Predictions contain NaN, infinity or values outside [0,1]";

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

/// A local two-channel prediction array
pub struct Predictions {
    /// Resolved input path, as recorded in `run.json`
    pub path: PathBuf,
    /// Spatial shape, z, y, x
    pub shape: [usize; 3],
    pub dtype: Dtype,
    reverse: bool,
    store: Store,
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
        Ok(Self { path, shape, dtype, reverse: order == "HV", store })
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
        let es = self.dtype.size();
        let plane = dims.iter().product::<usize>() * es;
        let mut out = Vec::with_capacity_in(2 * plane, alloc);
        out.resize(2 * plane, 0u8);
        match &self.store {
            Store::Npy { map, offset, fortran } => {
                let s = self.shape;
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
                let last = [0, 1, 2].map(|a| (hi[a] - 1) / chunks[a + 1]);
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

    /// Copy the part of chunk `key` (channel chunk, z, y, x) that falls in the window at `lo`
    fn copy_chunk(&self, key: [usize; 4], lo: [usize; 3], dims: [usize; 3], out: &mut [u8]) -> Result<()> {
        let Store::Zarr { chunks, .. } = &self.store else { unreachable!("zarr only") };
        let es = self.dtype.size();
        let data = self.chunk_bytes(key)?;
        let cs = [chunks[1], chunks[2], chunks[3]];
        let origin = [0, 1, 2].map(|a| key[a + 1] * cs[a]);
        let a0 = [0, 1, 2].map(|a| lo[a].max(origin[a]));
        let a1 = [0, 1, 2].map(|a| (lo[a] + dims[a]).min(origin[a] + cs[a]).min(self.shape[a]));
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

    /// Decoded bytes of one chunk in C order, or the fill value when the chunk file is absent
    fn chunk_bytes(&self, key: [usize; 4]) -> Result<Vec<u8>> {
        let Store::Zarr { dir, chunks, sep, fill, compressor, fortran } = &self.store else {
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
        match &self.store {
            Store::Zarr { chunks, .. } => Some([chunks[1], chunks[2], chunks[3]]),
            Store::Npy { .. } => None,
        }
    }

    /// One whole spatial chunk as uint8 `(2, cz, cy, cx)` in V, H order, clipped to the array shape, in a buffer
    /// from `alloc`; returns the chunk's origin and extent too. NPY input is served in blocks of the same size
    ///
    /// A whole uint8 Blosc chunk holding both channels, which is what prediction writers store, is decoded
    /// straight into the returned buffer, its compressed bytes read into another buffer from `alloc`; anything
    /// else is read through [`Predictions::read_u8`]'s path into a buffer from `alloc`
    pub fn read_chunk_in<A: Allocator + Copy>(
        &self,
        key: [usize; 3],
        edge: [usize; 3],
        alloc: A,
    ) -> Result<NativeChunk<A>> {
        let lo = [0, 1, 2].map(|a| key[a] * edge[a]);
        let hi = [0, 1, 2].map(|a| (lo[a] + edge[a]).min(self.shape[a]));
        let dims = [0, 1, 2].map(|a| hi[a] - lo[a]);
        if let Some(data) = self.whole_chunk_in(key, edge, dims, alloc)? {
            return Ok(NativeChunk { lo, dims, data });
        }
        Ok(NativeChunk { lo, dims, data: self.read_u8_in(lo, hi, alloc)? })
    }

    /// The fast path of [`Predictions::read_chunk_in`]: the chunk's decoded bytes are already `(2, dims)` in C order
    /// when its file holds both channels of an unclipped uint8 C-order chunk, so `read_raw`'s copy is the identity;
    /// `None` when the chunk is not of that kind
    fn whole_chunk_in<A: Allocator + Copy>(
        &self,
        key: [usize; 3],
        edge: [usize; 3],
        dims: [usize; 3],
        alloc: A,
    ) -> Result<Option<Vec<u8, A>>> {
        if self.slab_layout() != Some(edge) || dims != edge {
            return Ok(None);
        }
        let n = 2 * dims.iter().product::<usize>();
        let mut out = match self.packed_chunk_in(key, alloc)? {
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
        let Store::Zarr { chunks, compressor, fortran, .. } = &self.store else { return None };
        let sliced = chunks[0] == 2 && !*fortran && *compressor == Compressor::Blosc && self.dtype.kind == Kind::U8;
        sliced.then_some([chunks[1], chunks[2], chunks[3]])
    }

    /// The still-compressed bytes of chunk `key`, read into a buffer from `alloc`, for an array with a
    /// [`Predictions::slab_layout`]
    pub fn packed_chunk_in<A: Allocator>(&self, key: [usize; 3], alloc: A) -> Result<PackedChunk<A>> {
        let Store::Zarr { dir, sep, fill, .. } = &self.store else { bail!("only Zarr chunks are packed") };
        let name = format!("0{sep}{}{sep}{}{sep}{}", key[0], key[1], key[2]);
        let path = dir.join(&name);
        let mut file = match fs::File::open(&path) {
            Ok(f) => f,
            // a uint8 fill value is one byte, the same in both channels, so channel order does not matter
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(PackedChunk::Fill(fill[0])),
            Err(e) => return Err(eyre!("{}: {e}", path.display())),
        };
        let size = file.metadata().map_err(|e| eyre!("{}: {e}", path.display()))?.len() as usize;
        let mut bytes = Vec::with_capacity_in(size, alloc);
        bytes.resize(size, 0);
        file.read_exact(&mut bytes).map_err(|e| eyre!("{}: {e}", path.display()))?;
        Ok(PackedChunk::Blosc(bytes))
    }

    /// Decode z planes `planes` of both channels of a packed chunk into `out`, V then H, each `(planes, cy, cx)`
    /// at the stored extent of [`Predictions::slab_layout`]
    ///
    /// Blosc compresses a chunk in independent blocks (128 KiB, 8 planes of 128 × 128, in prediction zarrs), and
    /// only the blocks the planes fall in are decoded
    pub fn decode_planes<A: Allocator>(
        &self,
        packed: &PackedChunk<A>,
        planes: std::ops::Range<usize>,
        out: &mut [u8],
    ) -> Result<()> {
        let edge = self.slab_layout().ok_or_else(|| eyre!("the array's chunks cannot be decoded by planes"))?;
        let plane = edge[1] * edge[2];
        let channel = edge[0] * plane;
        let len = planes.len() * plane;
        ensure!(planes.end <= edge[0] && out.len() == 2 * len, "planes {planes:?} do not fit a {edge:?} chunk");
        let bytes = match packed {
            PackedChunk::Fill(fill) => {
                out.fill(*fill);
                return Ok(());
            }
            PackedChunk::Blosc(bytes) => bytes,
        };
        let (v, h) = out.split_at_mut(len);
        // stored channel order is V, H, or H, V when `reverse`
        let parts = if self.reverse { [h, v] } else { [v, h] };
        frame_channel_ranges(bytes, channel, planes.start * plane, parts)
    }

    /// Whether the array's Blosc blocks each fall within runs of `planes` planes, so decoding those planes decodes
    /// no more than they hold
    ///
    /// Judged by one stored chunk, since a writer compresses every chunk of an array alike; a frame compressed as
    /// one block, for example, would be decoded whole for every run of planes
    pub fn blocks_fit_planes(&self, planes: usize) -> bool {
        let Some(edge) = self.slab_layout() else { return false };
        let run = planes * edge[1] * edge[2];
        match self.stored_chunk().map(|key| self.packed_chunk_in(key, Global)) {
            Some(Ok(PackedChunk::Blosc(bytes))) => blocks_fit(&bytes, run),
            // an array with no chunk stored holds only its fill value, which no decode reads
            None => true,
            Some(Ok(PackedChunk::Fill(_)) | Err(_)) => false,
        }
    }

    /// The key of a chunk the array stores, found by listing its directory, or `None` when it stores none
    ///
    /// Listing instead of opening keys in grid order finds a chunk however sparse the array is
    fn stored_chunk(&self) -> Option<[usize; 3]> {
        let Store::Zarr { dir, sep, chunks, .. } = &self.store else { return None };
        let grid = [0, 1, 2].map(|a| self.shape[a].div_ceil(chunks[a + 1]));
        let key = |parts: &[&str]| -> Option<[usize; 3]> {
            let [c, z, y, x] = parts else { return None };
            let k = [z.parse().ok()?, y.parse().ok()?, x.parse().ok()?];
            (*c == "0" && (0..3).all(|a| k[a] < grid[a])).then_some(k)
        };
        let names = |d: &Path| -> Vec<String> {
            fs::read_dir(d).into_iter().flatten().flatten().filter_map(|e| e.file_name().into_string().ok()).collect()
        };
        if *sep != '/' {
            return names(dir).iter().find_map(|n| key(&n.split(*sep).collect::<Vec<_>>()));
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

/// Validate a Blosc 1 frame and read its header: version, version, flags, typesize, then nbytes, blocksize and
/// cbytes as little-endian u32
fn blosc_frame_info(src: &[u8]) -> Result<FrameInfo> {
    let mut nbytes = 0usize;
    // SAFETY: c-blosc reads at most `src.len()` bytes of the header to validate it
    let valid = unsafe { blosc_src::blosc_cbuffer_validate(src.as_ptr().cast(), src.len(), &mut nbytes) };
    ensure!(valid == 0 && src.len() >= 16, "not a valid Blosc frame");
    let blocksize = u32::from_le_bytes(src[8..12].try_into().expect("4 bytes")) as usize;
    Ok(FrameInfo { nbytes, typesize: src[3] as usize, blocksize })
}

/// Decode the same byte range of both channels of a frame holding two channels of `channel` bytes each: bytes
/// `start..` of each channel into the matching part, decoding only the blocks they fall in where it can
fn frame_channel_ranges(bytes: &[u8], channel: usize, start: usize, parts: [&mut [u8]; 2]) -> Result<()> {
    // a frame of another size would put one channel's bytes where the other's are read
    let frame = blosc_frame_info(bytes)?;
    ensure!(frame.nbytes == 2 * channel, "chunk frame holds {} bytes, expected {}", frame.nbytes, 2 * channel);
    let len = parts[0].len();
    ensure!(parts[1].len() == len && start + len <= channel, "bytes {start}.. lie outside a channel");
    let starts = [start, channel + start];
    let ts = frame.typesize;
    if item_ranges_decode_whole_blocks(&frame) && len.is_multiple_of(ts) && starts.iter().all(|s| s.is_multiple_of(ts))
    {
        for (at, part) in starts.into_iter().zip(parts) {
            blosc_getitem(bytes, at, part)?;
        }
        return Ok(());
    }

    // c-blosc reads a range in whole items of the frame's typesize, so a frame whose items straddle the range's
    // bounds is decoded whole, as is one whose blocks a range read could take short
    let all = blosc_decompress(bytes)?;
    for (at, part) in starts.into_iter().zip(parts) {
        part.copy_from_slice(&all[at..at + len]);
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

/// Decode bytes `start..start + out.len()` of one Blosc 1 frame into `out`, decoding only the blocks they fall in
fn blosc_getitem(src: &[u8], start: usize, out: &mut [u8]) -> Result<()> {
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
    use std::path::Path;

    use serde_json::Value;

    use super::{NativeChunk, Predictions, blocks_fit, blosc_getitem, frame_channel_ranges};
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
                        let NativeChunk { lo, dims, data } = p.read_chunk_in([z, y, x], edge, &pool).unwrap();
                        let hi = [0, 1, 2].map(|a| lo[a] + dims[a]);
                        assert_eq!(&data[..], &p.read_u8(lo, hi).unwrap()[..], "{}", case["name"]);
                        checked += 1;
                        direct += p.whole_chunk_in([z, y, x], edge, dims, &pool).unwrap().is_some() as usize;
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
        let (mut checked, mut reversed) = (0, 0);
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
                        let mut out = vec![0u8; 2 * len];
                        p.decode_planes(&packed, planes.clone(), &mut out).unwrap();
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
        assert!(checked > 30, "too few plane runs checked: {checked}");
        println!("{checked} plane runs checked, {reversed} in H, V order");
    }

    /// Compress `data` as one Blosc 1 frame with zstd and shuffle, as zarr writes prediction chunks
    fn blosc_frame(data: &[u8], typesize: usize, blocksize: usize) -> Vec<u8> {
        let mut out = vec![0u8; data.len() + 64];
        let name = c"zstd";
        // SAFETY: `out` has room for the frame and its 16-byte header; the context call keeps no global state
        let n = unsafe {
            blosc_src::blosc_compress_ctx(
                3,
                1,
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
                let (mut a, mut b) = (vec![0u8; run], vec![0u8; run]);
                frame_channel_ranges(&frame, channel, start, [&mut a, &mut b]).unwrap();
                assert_eq!(a, data[start..start + run], "typesize {typesize} start {start}");
                assert_eq!(b, data[channel + start..channel + start + run], "typesize {typesize} start {start}");
            }
        }
        let frame = blosc_frame(&data[..channel], 1, run);
        let (mut a, mut b) = (vec![0u8; run], vec![0u8; run]);
        assert!(frame_channel_ranges(&frame, channel / 2 + 1, 0, [&mut a, &mut b]).is_err());
    }

    /// A header whose block size its type size does not divide, which c-blosc never writes, is refused as decoding
    /// the whole frame refuses it, where reading item ranges would fill part of each block from scratch memory
    #[test]
    fn a_damaged_block_size_is_refused() {
        let channel = 8 * 64 * 64;
        let data: Vec<u8> = (0..2 * channel).map(|i| (i * 7 + i / 977) as u8).collect();
        let run = 2 * 64 * 64;
        let mut frame = blosc_frame(&data, 4, run / 2);
        frame[8..12].copy_from_slice(&(run as u32 / 2 - 2).to_le_bytes());
        assert!(!blocks_fit(&frame, run));
        assert!(blosc_getitem(&frame, 0, &mut vec![0u8; run]).is_err());
        let (mut a, mut b) = (vec![0u8; run], vec![0u8; run]);
        assert!(frame_channel_ranges(&frame, channel, 0, [&mut a, &mut b]).is_err());
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
