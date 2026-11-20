//! zstd decoding on the GPU with NVIDIA's nvCOMP, loaded at run time
//!
//! nvCOMP is under NVIDIA's own licence, so fibrs neither links nor ships it: [`NvComp::open`] loads the copy in
//! NVIDIA's `nvidia-libnvcomp-cu12` wheel, which links the CUDA runtime statically and needs only the driver, and a
//! caller that cannot open it decodes on the CPU. Bound against nvCOMP 6's C API (`nvcomp/zstd.h`), whose
//! decompression options and argument order differ from earlier majors, so only a major 6 library is accepted
//!
//! nvCOMP treats a malformed frame as undefined behaviour, so frames reach [`NvComp::decompress`] only after
//! [`zstd_frame_fits`](crate::input::zstd_frame_fits) passed their headers on the CPU, and every frame's status and
//! decoded size are checked after

use std::ffi::{CStr, c_char, c_void};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use color_eyre::eyre::{Result, WrapErr, bail, ensure, eyre};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut, sys};
use libloading::Library;

use super::Pending;
use crate::pyenv::site_packages;

/// The library's soname, as the wheel and NVIDIA's other packages install it
const SONAME: &str = "libnvcomp.so.6";
/// The major version these bindings follow
const MAJOR: u32 = 6;
/// Where the wheel puts the library inside `site-packages`
const WHEEL_DIR: &str = "nvidia/libnvcomp/lib64";
/// Largest frame and output nvCOMP's zstd decoder takes (`nvcompZstdDecompressionMaxAllowedChunkSize`)
const MAX_CHUNK: usize = (1 << 31) - 1;

/// `nvcompDecompressBackend_t`: let nvCOMP choose the hardware decompression engine or a CUDA kernel
const BACKEND_DEFAULT: i32 = 0;

/// `nvcompBatchedZstdDecompressOpts_t`: the backend, then reserved bytes that must be zero
#[repr(C)]
#[derive(Clone, Copy)]
struct DecompressOpts {
    backend: i32,
    reserved: [u8; 60],
}

const OPTS: DecompressOpts = DecompressOpts { backend: BACKEND_DEFAULT, reserved: [0; 60] };

/// `nvcompProperties_t`
#[repr(C)]
#[derive(Default)]
struct Properties {
    /// 1000 × major + 100 × minor + patch
    version: u32,
    /// The CUDA runtime nvCOMP was built with, 1000 × major + 10 × minor
    cudart_version: u32,
}

/// The minimum alignment in bytes of each buffer nvCOMP's zstd decoder is given (`nvcompAlignmentRequirements_t`)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Alignments {
    pub input: usize,
    pub output: usize,
    pub temp: usize,
}

/// An `nvcompStatus_t`, kept as the integer the library wrote, since a device array may hold any value
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Status(pub i32);

impl Status {
    /// `nvcompSuccess`
    pub const SUCCESS: Status = Status(0);

    pub fn is_success(self) -> bool {
        self == Self::SUCCESS
    }
}

type GetProperties = unsafe extern "C" fn(properties: *mut Properties) -> Status;
type GetStatusString = unsafe extern "C" fn(status: Status) -> *const c_char;
type GetRequiredAlignments = unsafe extern "C" fn(opts: DecompressOpts, alignments: *mut Alignments) -> Status;
type GetTempSize = unsafe extern "C" fn(
    num_chunks: usize,
    max_uncompressed_chunk_bytes: usize,
    opts: DecompressOpts,
    temp_bytes: *mut usize,
    max_total_uncompressed_bytes: usize,
    stream: sys::CUstream,
) -> Status;
type DecompressAsync = unsafe extern "C" fn(
    device_compressed_chunk_ptrs: *const *const c_void,
    device_compressed_chunk_bytes: *const usize,
    device_uncompressed_buffer_bytes: *const usize,
    device_uncompressed_chunk_bytes: *mut usize,
    num_chunks: usize,
    device_temp_ptr: *mut c_void,
    temp_bytes: usize,
    device_uncompressed_chunk_ptrs: *const *mut c_void,
    opts: DecompressOpts,
    device_statuses: *mut Status,
    stream: sys::CUstream,
) -> Status;

/// One frame of a batch: where its compressed bytes are in the input buffer, and the room for its decoded bytes
/// in the output buffer, which must be exactly the size it decodes to
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    pub input: Range<usize>,
    pub output: Range<usize>,
}

/// What nvCOMP reported for one frame
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Decoded {
    pub status: Status,
    /// Bytes it decoded
    pub size: usize,
}

/// The device memory [`NvComp::decompress`] works in, set aside once by [`NvComp::workspace`] for every batch up to
/// a size, so a batch allocates none: memory another process takes after setup cannot fail a decode
pub struct Workspace {
    /// Most frames a batch may hold
    frames: usize,
    /// nvCOMP's scratch space, sized for the largest batch
    temp: CudaSlice<u8>,
    /// The four per-frame arrays nvCOMP reads, `frames` 64-bit words each: input pointers, output pointers,
    /// input sizes and output sizes; a batch of `n` frames uses the first `n` of each
    table: CudaSlice<u64>,
    statuses: CudaSlice<i32>,
    sizes: CudaSlice<usize>,
}

/// nvCOMP's batched zstd decoder, loaded from a shared library
pub struct NvComp {
    /// Keeps the functions below mapped
    _lib: Library,
    path: PathBuf,
    properties: Properties,
    alignments: Alignments,
    status_string: GetStatusString,
    temp_size: GetTempSize,
    decompress: DecompressAsync,
}

impl NvComp {
    /// Load nvCOMP from the first place that has it: `FIBRS_NVCOMP` (a path to `libnvcomp.so.6`, and then the only
    /// place tried), the `nvidia-libnvcomp-cu12` wheel in the venv this executable was started from, the one in
    /// `$VIRTUAL_ENV`, then the dynamic loader's search path. `FIBRS_NVCOMP=off`, in any case, loads none
    pub fn open() -> Result<Self> {
        if let Some(path) = std::env::var_os("FIBRS_NVCOMP") {
            // nvCOMP trusts its input, so this is how a run that doubts its prediction files decodes them on the CPU
            ensure!(!path.eq_ignore_ascii_case("off"), "FIBRS_NVCOMP is off");
            return Self::load(Path::new(&path));
        }
        let wheels = site_packages().into_iter().map(|site| site.join(WHEEL_DIR).join(SONAME));
        let mut reasons = Vec::new();
        for path in wheels.filter(|p| p.is_file()).chain([PathBuf::from(SONAME)]) {
            match Self::load(&path) {
                Ok(nvcomp) => return Ok(nvcomp),
                Err(e) => reasons.push(format!("{e:#}")),
            }
        }
        bail!(
            "cannot load nvCOMP ({}); install the nvidia-libnvcomp-cu12 wheel into this executable's venv or set \
             FIBRS_NVCOMP",
            reasons.join("; ")
        )
    }

    /// Load nvCOMP from `path`, a path or a bare soname for the dynamic loader to find, and check it is major 6
    pub fn load(path: &Path) -> Result<Self> {
        let at = path.display();
        // SAFETY: loading the library runs its initializers, which set up its statically linked CUDA runtime
        // without touching a device; nvCOMP is built to be loaded into any process
        let lib = unsafe { Library::new(path) }.wrap_err_with(|| format!("load {at}"))?;
        let missing = |name: &str| format!("{at} has no {name}");
        // SAFETY: each symbol is declared in nvCOMP 6's `nvcomp.h` and `nvcomp/zstd.h` with the signature of its
        // type here; the version is checked before any but `nvcompGetProperties`, which every major has, is called
        let (properties_fn, status_string, alignments_fn, temp_size, decompress) = unsafe {
            (
                *lib.get::<GetProperties>(b"nvcompGetProperties\0").wrap_err_with(|| missing("nvcompGetProperties"))?,
                *lib.get::<GetStatusString>(b"nvcompGetStatusString\0")
                    .wrap_err_with(|| missing("nvcompGetStatusString"))?,
                *lib.get::<GetRequiredAlignments>(b"nvcompBatchedZstdDecompressGetRequiredAlignments\0")
                    .wrap_err_with(|| missing("nvcompBatchedZstdDecompressGetRequiredAlignments"))?,
                *lib.get::<GetTempSize>(b"nvcompBatchedZstdDecompressGetTempSize\0")
                    .wrap_err_with(|| missing("nvcompBatchedZstdDecompressGetTempSize"))?,
                *lib.get::<DecompressAsync>(b"nvcompBatchedZstdDecompressAsync\0")
                    .wrap_err_with(|| missing("nvcompBatchedZstdDecompressAsync"))?,
            )
        };
        let mut properties = Properties::default();
        // SAFETY: the function only writes the struct through the pointer
        let status = unsafe { properties_fn(&mut properties) };
        ensure!(status.is_success(), "{at}: nvcompGetProperties failed ({})", status.0);
        let major = properties.version / 1000;
        ensure!(major == MAJOR, "{at} is nvCOMP {major}; fibrs needs nvCOMP {MAJOR}");
        let mut alignments = Alignments::default();
        // SAFETY: the function only writes the struct through the pointer; the options are the defaults, with the
        // reserved bytes zero
        let status = unsafe { alignments_fn(OPTS, &mut alignments) };
        ensure!(status.is_success(), "{at}: nvCOMP's zstd alignment query failed ({})", status.0);
        Ok(Self { _lib: lib, path: path.to_path_buf(), properties, alignments, status_string, temp_size, decompress })
    }

    /// Where the library was loaded from, and its version and CUDA runtime, for the log
    pub fn name(&self) -> String {
        let (v, rt) = (self.properties.version, self.properties.cudart_version);
        format!(
            "nvCOMP {}.{}.{} (CUDA runtime {}.{}) from {}",
            v / 1000,
            v % 1000 / 100,
            v % 100,
            rt / 1000,
            rt % 1000 / 10,
            self.path.display()
        )
    }

    pub fn alignments(&self) -> Alignments {
        self.alignments
    }

    /// nvCOMP's description of a status
    pub fn status_string(&self, status: Status) -> String {
        // SAFETY: the function returns a pointer to a static string, or null for a value it does not know
        let text = unsafe { (self.status_string)(status) };
        if text.is_null() {
            return format!("nvCOMP status {}", status.0);
        }
        // SAFETY: a non-null result is a NUL-terminated static string
        unsafe { CStr::from_ptr(text) }.to_string_lossy().into_owned()
    }

    /// Set aside on `stream` the device memory [`NvComp::decompress`] needs for any batch of up to `frames`
    /// frames, none decoding to more than `max_out` bytes
    pub fn workspace(&self, stream: &Arc<CudaStream>, frames: usize, max_out: usize) -> Result<Workspace> {
        ensure!(frames > 0, "an nvCOMP workspace holds at least one frame");
        let total_out = frames.checked_mul(max_out).ok_or_else(|| eyre!("{frames} frames of {max_out} bytes"))?;
        let temp_bytes = self.temp_bytes(stream, frames, max_out, total_out)?;
        // SAFETY: nvCOMP only uses the scratch space as scratch, so it needs no initial value
        let mut temp = unsafe { stream.alloc::<u8>(temp_bytes.max(1)) }.wrap_err("allocate nvCOMP's scratch space")?;
        let (temp_ptr, temp_guard) = temp.device_ptr_mut(stream);
        let aligned = (temp_ptr as usize).is_multiple_of(self.alignments.temp.max(1));
        drop(temp_guard);
        ensure!(aligned, "nvCOMP's scratch space is misaligned");
        let table = stream.alloc_zeros::<u64>(4 * frames).wrap_err("allocate nvCOMP's frame tables")?;
        let statuses = stream.alloc_zeros::<i32>(frames).wrap_err("allocate nvCOMP's frame tables")?;
        let sizes = stream.alloc_zeros::<usize>(frames).wrap_err("allocate nvCOMP's frame tables")?;
        Ok(Workspace { frames, temp, table, statuses, sizes })
    }

    /// Device scratch space in bytes nvCOMP needs for a batch of `frames` frames, none decoding to more than
    /// `max_out` bytes and all to at most `total_out`
    fn temp_bytes(&self, stream: &Arc<CudaStream>, frames: usize, max_out: usize, total_out: usize) -> Result<usize> {
        // nvCOMP's statically linked CUDA runtime works in the context current on the calling thread
        stream.context().bind_to_thread()?;
        let mut temp_bytes = 0usize;
        // SAFETY: the function only computes the size and writes it through the pointer; it enqueues nothing
        let status = unsafe { (self.temp_size)(frames, max_out, OPTS, &mut temp_bytes, total_out, stream.cu_stream()) };
        ensure!(status.is_success(), "nvCOMP's zstd temp size query failed: {}", self.status_string(status));
        Ok(temp_bytes)
    }

    /// Decode a batch of zstd frames held in `input` into `output` on `stream`, working in `workspace`, waiting
    /// for it to finish, and report each frame's status and decoded size, in the order of `frames`
    ///
    /// The batch must fit the workspace: no more frames than it holds, and scratch space no larger than it set
    /// aside. The ranges are checked against the buffers and nvCOMP's alignments, and the outputs must not overlap. A
    /// frame that decodes to more than its output is reported with an error status, not written past it; one
    /// that decodes to less reports the smaller size, so callers compare `size` with the output's length
    ///
    /// # Safety
    ///
    /// Every input range must hold one zstd frame that [`zstd_frame_fits`](crate::input::zstd_frame_fits)
    /// accepted for its output's length, copied unchanged to the device: nvCOMP does not validate frames and
    /// may read or write out of bounds on a malformed one
    pub unsafe fn decompress<I, O>(
        &self,
        stream: &Arc<CudaStream>,
        workspace: &mut Workspace,
        input: &I,
        output: &mut O,
        frames: &[Frame],
    ) -> Result<Vec<Decoded>>
    where
        I: DevicePtr<u8>,
        O: DevicePtrMut<u8>,
    {
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        let ctx = stream.context();
        let same = |s: &Arc<CudaStream>| s.context().cu_ctx() == ctx.cu_ctx();
        ensure!(
            same(input.stream()) && same(output.stream()) && same(workspace.temp.stream()),
            "nvCOMP's buffers and stream belong to different CUDA contexts"
        );
        check_frames(frames, input.len(), output.len())?;
        let n = frames.len();
        let max_out = frames.iter().map(|f| f.output.len()).max().unwrap_or(0);
        let total_out: usize = frames.iter().map(|f| f.output.len()).sum();
        ensure!(n <= workspace.frames, "{n} frames exceed the {} nvCOMP's workspace holds", workspace.frames);
        // the workspace was sized for its largest batch; asked again, since nvCOMP does not document that a smaller
        // batch needs no more
        let needed = self.temp_bytes(stream, n, max_out, total_out)?;
        let Workspace { temp, table: table_dev, statuses, sizes, .. } = workspace;
        ensure!(
            needed <= temp.len(),
            "{n} frames need {needed} bytes of nvCOMP scratch, {} were set aside",
            temp.len()
        );

        let (in_base, in_guard) = input.device_ptr(stream);
        let (out_base, out_guard) = output.device_ptr_mut(stream);
        let align = self.alignments;
        let aligned = frames.iter().all(|f| {
            (in_base as usize + f.input.start).is_multiple_of(align.input.max(1))
                && (out_base as usize + f.output.start).is_multiple_of(align.output.max(1))
        });
        ensure!(
            aligned,
            "nvCOMP's zstd decoder needs inputs aligned to {} and outputs to {}",
            align.input,
            align.output
        );

        // the four per-frame arrays nvCOMP reads, in one upload: input pointers, output pointers, input sizes and
        // output sizes, each a 64-bit word per frame
        let mut table = Vec::with_capacity(4 * n);
        table.extend(frames.iter().map(|f| in_base + f.input.start as u64));
        table.extend(frames.iter().map(|f| out_base + f.output.start as u64));
        table.extend(frames.iter().map(|f| f.input.len() as u64));
        table.extend(frames.iter().map(|f| f.output.len() as u64));
        let mut status_host = vec![0i32; n];
        let mut size_host = vec![0usize; n];
        // after the host memory the copies read and write, so an early return waits for them before it is freed;
        // the device buffers are the caller's and the workspace's, which outlive the call, and cudarc orders their
        // freeing after the work queued on them
        let pending = Pending(stream);
        stream.memcpy_htod(&table, &mut table_dev.slice_mut(..4 * n))?;
        // a frame nvCOMP leaves unwritten reports 0 bytes, never the size a previous batch's frame decoded to
        stream.memset_zeros(&mut statuses.slice_mut(..n))?;
        stream.memset_zeros(&mut sizes.slice_mut(..n))?;
        let temp_bytes = temp.len();
        let (temp_ptr, temp_guard) = temp.device_ptr_mut(stream);
        let (table_ptr, table_guard) = table_dev.device_ptr(stream);
        let word = |k: u64| (table_ptr + k * 8 * n as u64) as *const c_void;
        let (status_ptr, status_guard) = statuses.device_ptr_mut(stream);
        let (sizes_ptr, sizes_guard) = sizes.device_ptr_mut(stream);
        // nvCOMP's statically linked CUDA runtime works in the context current on the calling thread
        ctx.bind_to_thread()?;
        // SAFETY: the table holds `n` words of each kind at a stride of `n`, the status and size arrays at least `n`
        // entries of the type nvCOMP writes there, and the scratch space at least what nvCOMP asked for this batch,
        // aligned as it asked when the workspace was set aside; the table's pointers address ranges `check_frames`
        // held inside the input and the disjoint outputs, aligned as nvCOMP asked, and the frames are well formed
        // by the caller's contract. The guards above order these buffers' other work on `stream` around the decode,
        // and `pending` waits for the stream before the host memory the copies use is freed
        let status = unsafe {
            (self.decompress)(
                word(0).cast(),
                word(2).cast(),
                word(3).cast(),
                sizes_ptr as *mut usize,
                n,
                temp_ptr as *mut c_void,
                temp_bytes,
                word(1).cast(),
                OPTS,
                status_ptr as *mut Status,
                stream.cu_stream(),
            )
        };
        drop((in_guard, out_guard, temp_guard, table_guard, status_guard, sizes_guard));
        // `pending` waits, on this early return, for whatever the call managed to enqueue
        ensure!(status.is_success(), "nvCOMP's zstd decoder failed to launch: {}", self.status_string(status));
        stream.memcpy_dtoh(&statuses.slice(..n), &mut status_host)?;
        stream.memcpy_dtoh(&sizes.slice(..n), &mut size_host)?;
        pending.wait("decode with nvCOMP")?;
        Ok(status_host
            .into_iter()
            .zip(size_host)
            .map(|(status, size)| Decoded { status: Status(status), size })
            .collect())
    }
}

/// Hold each frame's ranges inside the buffers, nonempty, within nvCOMP's size limit, and the outputs disjoint
fn check_frames(frames: &[Frame], input: usize, output: usize) -> Result<()> {
    for f in frames {
        let (i, o) = (&f.input, &f.output);
        ensure!(
            !i.is_empty() && i.end <= input && i.len() <= MAX_CHUNK,
            "frame bytes {i:?} lie outside the {input}-byte input or past nvCOMP's limit"
        );
        ensure!(
            !o.is_empty() && o.end <= output && o.len() <= MAX_CHUNK,
            "output bytes {o:?} lie outside the {output}-byte output or past nvCOMP's limit"
        );
    }
    let mut outputs: Vec<&Range<usize>> = frames.iter().map(|f| &f.output).collect();
    outputs.sort_by_key(|r| r.start);
    if let Some(w) = outputs.windows(2).find(|w| w[0].end > w[1].start) {
        bail!("outputs {:?} and {:?} overlap", w[0], w[1]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use cudarc::driver::CudaContext;

    use super::{Alignments, Frame, NvComp, Status, check_frames};
    use crate::input::{BlockSource, PackedChunk, Predictions, zstd_frame_fits};

    #[test]
    fn frame_ranges_are_checked() {
        let frame = |input: std::ops::Range<usize>, output: std::ops::Range<usize>| Frame { input, output };
        assert!(check_frames(&[frame(0..10, 0..64), frame(10..20, 64..128)], 20, 128).is_ok());
        assert!(check_frames(&[frame(0..21, 0..64)], 20, 128).is_err());
        assert!(check_frames(&[frame(0..10, 0..129)], 20, 128).is_err());
        assert!(check_frames(&[frame(0..0, 0..64)], 20, 128).is_err());
        assert!(check_frames(&[frame(0..10, 0..64), frame(10..20, 63..127)], 20, 128).is_err());
    }

    /// The library loads, is major 6, and states the alignments these bindings were checked against; loading
    /// and these queries touch no device
    #[test]
    #[ignore = "needs nvCOMP: FIBRS_NVCOMP=.../nvidia/libnvcomp/lib64/libnvcomp.so.6 cargo test nvcomp_loads -- --ignored"]
    fn nvcomp_loads() {
        let nvcomp = NvComp::open().unwrap();
        eprintln!("{}", nvcomp.name());
        assert_eq!(nvcomp.alignments(), Alignments { input: 1, output: 1, temp: 8 });
        assert_eq!(nvcomp.status_string(Status::SUCCESS), "The operation completed successfully");
    }

    /// Every slab of a real prediction chunk (or `FIBRS_NVCOMP_CHUNK`, a chunk file of a `[2, 128, 128, 128]`
    /// uint8 Blosc zstd array) decoded through nvCOMP holds the bytes c-blosc decodes on the CPU: all 32 blocks at
    /// once, the whole chunk uploaded as stored
    ///
    /// nvCOMP comes from `FIBRS_NVCOMP`, for example the wheel installed for this test at
    /// `/build/fibrs-gpu/venv-nvcomp/lib/python3.14/site-packages/nvidia/libnvcomp/lib64/libnvcomp.so.6`
    #[test]
    #[ignore = "needs an NVIDIA GPU and nvCOMP (FIBRS_NVCOMP); run through the GPU queue"]
    fn nvcomp_decodes_a_chunk_as_the_cpu_does() {
        let chunk = std::env::var_os("FIBRS_NVCOMP_CHUNK").map_or_else(
            || "/build/fibrs-alloc/bands/0358-z02304-02816.zarr/semantic/0.3.39.45".into(),
            std::path::PathBuf::from,
        );
        let array = chunk.parent().unwrap();
        let p = Predictions::open(array, None, "VH").unwrap();
        let key: Vec<usize> =
            chunk.file_name().unwrap().to_str().unwrap().split('.').map(|k| k.parse().unwrap()).collect();
        let packed = p.packed_chunk_in([key[1], key[2], key[3]], std::alloc::Global).unwrap();
        let edge = p.slab_layout().unwrap();
        let plane = edge[1] * edge[2];
        let channel = edge[0] * plane;
        let slab = 8;
        let block = slab * plane;

        let mut want = vec![0u8; 2 * channel];
        p.decode_planes(&packed, 0..edge[0], &mut want).unwrap();
        let PackedChunk::Blosc(bytes) = &packed else { panic!("the chunk is stored") };
        // slab s of channel c lands at c * channel + s * block, as `decode_planes` lays both channels out
        let mut frames = Vec::new();
        for s in 0..edge[0] / slab {
            let sources = p.plane_blocks(&packed, s * slab..(s + 1) * slab).expect("zstd blocks");
            for (c, source) in sources.into_iter().enumerate() {
                let BlockSource::Zstd(input) = source else { panic!("slab {s} channel {c}: {source:?}") };
                assert!(zstd_frame_fits(&bytes[input.clone()], block));
                // the parser's ranges, decoded on the CPU, already give c-blosc's bytes
                let at = c * channel + s * block;
                let cpu = zstd::bulk::decompress(&bytes[input.clone()], block).unwrap();
                assert!(cpu == want[at..at + block], "slab {s} channel {c}: the CPU decode of the frame differs");
                frames.push(Frame { input, output: at..at + block });
            }
        }
        assert_eq!(frames.len(), 32);

        let nvcomp = NvComp::open().unwrap();
        eprintln!("{}, alignments {:?}", nvcomp.name(), nvcomp.alignments());
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let input = stream.clone_htod(&bytes[..]).unwrap();
        let mut output = stream.alloc_zeros::<u8>(2 * channel).unwrap();
        let mut workspace = nvcomp.workspace(&stream, frames.len(), block).unwrap();
        // SAFETY: each input range is a frame `zstd_frame_fits` accepted, uploaded unchanged
        let decoded = unsafe { nvcomp.decompress(&stream, &mut workspace, &input, &mut output, &frames) }.unwrap();
        for (f, d) in frames.iter().zip(&decoded) {
            assert!(d.status.is_success(), "{:?}: {}", f, nvcomp.status_string(d.status));
            assert_eq!(d.size, block, "{f:?}");
        }
        let got = stream.clone_dtoh(&output).unwrap();
        stream.synchronize().unwrap();
        assert!(got == want, "nvCOMP's decode differs from c-blosc's");
    }
}
