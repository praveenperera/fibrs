# Tracer reference

How `fibrs` runs, beyond the [README](../README.md): devices, threads, the drop-in, tiled tracing, numerics, inputs and
where it differs from the original.

## Devices and threads

The direction field and centrality run on the GPU when a usable one is found: Linux x86-64, an NVIDIA GPU of compute
capability 8.0 or later, and a driver for CUDA 12.0 or newer (no CUDA toolkit). Otherwise they run on the CPU, and
the log says why. `--device cpu` keeps them on the CPU, `--device cuda` fails instead of falling back, and
`--cuda-device N` picks the GPU (by default the first usable one). The GPU direction field is the CPU's bit for bit:
the GPU certifies each voxel's eigenvector against an error bound, and any voxels it cannot certify, usually few, are
solved on the CPU with the same `dsyevd` (the log line counts them). Centrality rounds some steps differently on the
GPU (see [Differences from the original](#differences-from-the-original)), and a resumed run keeps the rounding it started with: one started on the
CPU computes centrality there, and with `--device auto` only its direction field and finishing (below) take the GPU
(`--device cuda` refuses to resume it). A run started on the GPU can resume without one: the CPU then computes centrality as the GPU
does, more slowly. `--device auto` also moves the direction field or centrality to the CPU if the GPU fails mid-run,
for example when it runs out of memory.

When a run uses the GPU, finishing (cutting each traced curve where the predictions disagree, then smoothing,
resampling and re-checking each piece) runs there too, with the same result as on the CPU, whatever the rounding the
run started with. Both pooled predictions and a cache of native prediction slabs, up to 8 GiB, stay on the GPU while
a family is finished. The GPU finishes the curves in batches, and a batch it cannot finish, for example one that
reads more slabs than its cache holds, is finished on the CPU. At the end of each family the log counts the slabs
cached, by where they were decoded, and the batches the CPU finished. If the GPU fails while finishing,
`--device auto` finishes the rest of the family on the CPU and `--device cuda` stops the run.

By default the slabs are decoded on the CPU, with c-blosc's checks. Decoding them on the GPU with NVIDIA's nvCOMP is
opt-in, with the same output on sound files: run `pip install "fibrs[gpu-zstd]"` and set `FIBRS_NVCOMP=on`, or
set `FIBRS_NVCOMP` to the path of a `libnvcomp.so.6`. GPU slab decoding trusts the prediction files: on a corrupt
chunk nvCOMP's behaviour is undefined, so it may give wrong output or a CUDA fault where the CPU decoder would
report the corruption.

nvCOMP is not part of fibrs: it is NVIDIA's own PyPI package, `nvidia-libnvcomp-cu12` (version 6, under NVIDIA's
licence), which the `gpu-zstd` extra installs. With `FIBRS_NVCOMP=on` (in any case), fibrs loads that wheel from
the environment fibrs is installed into or the active one (`$VIRTUAL_ENV`), then tries the system's library path.
Run from Python (`python -m fibrs` or `fibrs.run`), it loads only the wheel in that Python's environment, venv or
not, when there is one, so a wheel whose library fails to load leaves slabs on the CPU. Unset or `off`,
nvCOMP is never loaded, whether or not the wheel is installed. The log says once whether slabs decode on the GPU, or
why they decode on the CPU. Even with nvCOMP, a slab whose chunk it cannot decode as Blosc stored it (its Blosc
blocks are not one slab of one channel each, or a zstd frame fails fibrs's checks) is decoded on the CPU, as is
every slab of an array decoded a whole chunk at a time, and when nvCOMP's buffers would leave too little GPU memory
for the slab cache, the family's slabs are decoded on the CPU and the log says so.

`fibrs.run([...])` runs it from Python and returns the exit status.

`--threads N` sets the threads that trace and finish curves. Threads that finishing leaves idle trace seeds ahead of
the one being committed, and one more thread commits the seeds in order, so tracing can run one thread beyond
`--threads`. On a dense 512 × 2048 × 2048 crop of PHerc0358, tracing kept 4.2–4.5 cores busy at `--threads 4` and
7.7 at `--threads 8`. A seed traced ahead is traced again if an earlier seed marked the region it read before it was
committed, so the output is the same at every `--threads`. The export to `.afv`, including the one a `--resume`
of a finished run writes, prepares each batch of fibres' rows on the same `--threads` pool while one thread inserts
the previous batch's rows in order.

## The drop-in

To make an environment's `avf-spline-generator` run the port, so scripts that call it need no change:

```sh
scripts/install-dropin.sh VENV [BINARY]   # VENV/bin/avf-spline-generator -> fibrs
scripts/install-dropin.sh --restore VENV  # put the original back
```

BINARY defaults to the `fibrs` in the archive the script came from, then to the one on `PATH`.
The original entry is kept as `avf-spline-generator.reference`, and `NOTICE` and `licenses/` are copied to
`VENV/share/fibrs`. Callers that run `Path(sys.executable).parent / "avf-spline-generator"`
need no other change. Run as `avf-spline-generator`, `--version` prints `0.2.0`, the release it stands in for; run
as `fibrs` or `python -m fibrs`, it prints fibrs' own version.

## Tiled tracing

A dense window can need more memory than the machine has: the tracer holds the fields of every voxel that holds
fibres, so its peak grows with how much of the window is data. `--tiles-yx NY NX` traces the window instead as
NY × NX tiles in Y and X, one after another in the same process, so the run's peak is about its densest tile's.
This mode is fibrs's own; the original has no such option, and without `--tiles-yx` fibrs runs as before.

```sh
fibrs --predictions P --array semantic --tracing-predictions P --tracing-array tracing \
  --voxel-size-um 9.362 --storage compressed --threads 8 --output fibers.afv --work-dir work \
  --tiles-yx 2 2 --tile-halo 256
```

- **Tiles.** The Y and X extent is split into NY × NX tile cores. Each tile is traced from its core plus up to
  `--tile-halo` native voxels (default 256) of its neighbours on each side, clipped to the input. Z is not tiled.
  With more than one tile the halo must be at least 1, since a tile traced from its core alone stops short of its
  seams and no fibre would be joined across them.
- **Alignment.** Core edges and the halo are whole units of A native voxels: the least common multiple of the
  native array's chunk edge, twice the tracing array's chunk edge, and 256 (twice the centrality pass's 128-voxel
  tile, whose edges change centrality). NPY input counts as 64-voxel chunks. For prediction zarrs with 128³ native
  and 64³ tracing chunks, A is 256. Core edges fall at `floor(k × extent / N / A) × A`, and the halo is rounded up to
  a multiple of A. Each core must hold at least one unit, so a window narrower than `N × A` voxels cannot be split
  into N tiles along that axis. Alignment keeps each tile's reads on whole chunks, and keeps each tile on the window's
  pooling and centrality grids, so that away from the seams a tile can trace as the whole window does. Seeds are
  ordered by the window's 1024-voxel regions, not the tile's.
- **Seams.** Each tile's fibres are cut to its core. Where a fibre leaves the core through a face shared with another
  tile, the point where it crosses the face is added. The cut ends on the two sides of each face, from tiles that
  share that face (not tiles meeting only at a corner), are joined when they are of the same family and at most 1.5
  native voxels apart: shortest first, each end once, and never into a loop. Pieces under two points are dropped. This is the rule the v2-run pipeline uses to join its Z bands. Near a
  seam the output can differ from an untiled run: a fibre traced from each side whose cut ends land more than 1.5
  voxels apart stays split, and a tile sees its neighbours' data only as far as the halo reaches. A step of a fibre
  that cuts across a tile corner, with no point in the tile it passes through, gives that tile the piece between
  its two crossings, so the fibre joins through the corner.
- **Output.** The `.afv` has the untiled run's metadata. Each family is written in the untiled order: every tile's
  uncut fibres in tile order (row by row), then the fibres joined across seams. A `--tiles-yx 1 1` run takes the
  tiled path and writes the same `.afv` as an untiled run, every row and metadata value but the uuid.
- **Memory.** Each tile allocates its own fields and native chunk cache (`--native-cache-mib` plus
  `--field-cache-mib`) and frees them before the next tile starts. Besides the batches of fibres the export
  prepares as in an untiled run, it holds only the current family's pieces that end on a seam.
- **Work directory.** Each tile has its own directory, `work/tile-YY-XX/family-N/`. `--resume` skips tiles whose
  families are complete. `run.json` records the grid, the halo, the alignment and each tile's boxes; an untiled
  run's `run.json` has no tiling key, as before.

## Numerics

On the CPU, the direction field's eigenvectors come from `dsyevd` in numpy's own OpenBLAS, on float64 copies as
`numpy.linalg.eigh` computes them, so they match `eigh` bit for bit. On the GPU, each voxel's eigenvector is
certified against an error bound to equal that result, and voxels that can't be certified go to the same `dsyevd`. The binary finds that library from its
own path (`bin/../lib/python*/site-packages/numpy.libs/libscipy_openblas64_*.so`), then from
`$VIRTUAL_ENV`, looking in each environment in numpy's copy first and then in the `scipy-openblas64`
package (`site-packages/scipy_openblas64/lib`). `FIBRS_LAPACK` overrides both.

macOS numpy wheels link Accelerate and bundle no OpenBLAS, so on macOS the wheel depends on
`scipy-openblas64==0.3.31.188.0`, the OpenBLAS build that numpy 2.4.6's Linux wheel bundles. With it, a
Mac writes the same `.afv` as Linux.

### Outside a pip install

The archive's binary, and one installed with cargo, bundles no OpenBLAS: it loads numpy's at run time, from the environment it is installed
into (`scripts/install-dropin.sh` puts it there) or else from the active one (`$VIRTUAL_ENV`). On macOS that
environment also needs `scipy-openblas64==0.3.31.188.0`. Run from anywhere else, it
stops with "cannot find numpy's OpenBLAS" once the tracing predictions are loaded, after it has written the work
directory's `run.json`. Activate the numpy environment, or set `FIBRS_LAPACK` to the library's full path (on
Linux, the `libscipy_openblas64_*.so` file in its `site-packages/numpy.libs`; fibrs does not expand a `*`), and run
the same command again with `--resume`, through the same binary: an activated environment's `fibrs` can come first
on `PATH`, and a run resumes only with the version that started it. `pip install fibrs` into an environment
that holds its own numpy wheel needs none of this. fibrs looks only inside that environment, so one that inherits
numpy (`--system-site-packages`) or has a distribution's numpy, such as Debian's `python3-numpy`, needs
`scipy-openblas64==0.3.31.188.0` installed into it, as on macOS.

## Inputs

The same arrays the original reads through numpy and zarr 2: `.npy` files and Zarr v2 arrays or groups of
uint8, float16, float32 or float64, in either byte order and either array order. Zarr chunks may use any
channel chunking, either dimension separator and the Blosc, Zstd, Zlib, GZip or LZ4 compressor; Blosc
frames are decoded by c-blosc 1.21.6, the release numcodecs 0.15 bundles. Missing chunks read as the fill
value, including `NaN` and infinities, which the value check then rejects as the original does. Not
supported: Zarr filters, other numcodecs compressors (BZ2, LZMA and so on) and extended-precision floats.
`tests/fixtures/readers` holds one small array in each supported encoding with what the original reads
from it; `cargo test` checks every case.

## Differences from the original

- Fields live in memory as sparse 16³ bricks instead of a Zarr work store, so peak memory is higher and grows
  with how much of the band holds fibres: 9.0 GiB against the original's 4.6 GiB on a 512 × 8169 × 8169 band of
  PHerc0826, and 13.3 GiB on 1024 × 8169 × 8169 of the same scroll before predictions were decoded in slabs. With
  `--field-cache-mib 4096`, 4.1 GiB of that is the chunk cache; a smaller cache lowers the peak by the
  difference but decodes again more often (measured while chunks were decoded whole: at a 1 GiB cache the
  1024-plane window peaked at 9.8 GiB and traced about 19% slower; at 16 MiB tracing took about 60% longer).
  Memory each stage frees goes back to the system before the next stage allocates.
- A point's value needs 8 voxels of the native predictions, so they are decoded 8 planes at a time (one 128 KiB
  Blosc block of each channel in a prediction zarr) instead of a whole chunk at a time. The cache holds these
  slabs, with the chunks' compressed bytes in a quarter of its budget, and a chunk whose Blosc blocks are larger
  than 8 planes is decoded once and cut into slabs. Arrays read a whole chunk at a time (chunks of 8 planes or
  fewer, or other layouts and compressors) give the whole budget to decoded chunks. Decoded slabs, compressed chunks and the
  direction field's tile buffers come from pools that each keep at most 128 MiB of freed buffers for reuse,
  beside the cache, and each field's bricks are cut from 64 MiB mappings the field owns. Dense bands evict and decode
  hundreds of times a second, and on Linux mapping and unmapping memory takes a lock that every page fault in
  the process waits for, so a mapping per decode and per brick stalled the tracing threads. Other allocations
  of 1 MiB or more have their own mapping on Linux, so freed memory does not stay resident. `FIBRS_DEBUG_MEM=1`
  logs each stage's field sizes and resident memory.
- The work directory holds `run.json` (with a `native` key, so neither tool resumes the other's
  directory, and only the fibrs version that started it resumes it) and one finished curve file per
  family. An interrupted family is traced again from its first
  seed on `--resume`, instead of from the last 16-seed batch.
- `--storage` and `--field-cache-mib` are validated, and `--storage` is recorded in `run.json` as the
  original records it, but neither selects anything. The chunk cache is bounded by `--native-cache-mib`
  plus `--field-cache-mib`.
- `--device` defaults to `auto` (the GPU when one is usable) where the original defaults to `cpu`. On the GPU
  the port writes what the original writes with `--device cuda`, whose centrality kernels round some multiply-adds
  as one fused operation, so its output can differ from the original's `--device cpu` output; pass `--device cpu`
  to match that. The original always computes the direction field on the CPU; the port computes it on the GPU
  too, with the same result.
- The command line replays argparse's pass, so exit status and selected values match, but help text and
  usage-error wording differ. Some argparse readings are not reproduced: numbers written with non-ASCII
  digits; non-UTF-8 bytes in the text options (`--array`, `--tracing-array`, `--coordinate-space`);
  negative numbers with an underscore or trailing whitespace inside a three-value option or after a
  numeric option (`-0_0`, `"-1\n"`), which this tool rejects and argparse accepts; and a short help
  cluster followed by a dash (`-h-`, `-h--`), which this tool treats as help and argparse rejects.
- The original's batch-buffer capacity checks are not reproduced. Its smoothing-snapshot limit
  (`--field-cache-mib`) is not either, because no snapshot is taken.
