# fibrs

A native, multi-threaded drop-in for `avf-spline-generator` 0.2.0, Qual's AFV fibre tracer
([Qualzz/AVF_SPLINE_GENERATOR](https://github.com/Qualzz/AVF_SPLINE_GENERATOR) @ `bee798f`). It takes the
same flags and writes the same `.afv`: every metadata value except the per-run uuid, and every fibre row,
point block and spatial-index row. The tracing algorithm, the AFV format and the fibre model whose
predictions it reads are Qual's; this port changes how fast the tracer runs, not what it writes.

`fibrs predict` runs the step before: Qual's AFV fibre model V2 over a CT volume on an NVIDIA GPU, without
PyTorch, writing the probability zarr the tracer reads, as villa's inference writes it (see
[Predictions](#predictions)).

From CT to `.afv`, fibrs 0.4.0 runs three 512-plane bands of PHerc1545, PHerc0826 and PHerc0358 in 800 s in total
on one RTX 5090, with prediction at fp32. villa's public prediction at fp32 followed by the original tracer takes
about 35,000 s on the same machine, so fibrs is about 44× faster; villa's part and the original's PHerc0358 time are
estimated, and without villa's write of its accumulators to disk fibrs is about 32× faster (see [Speed](#speed)). A
whole scroll end to end would take fibrs about 2.6–3 hours and villa with the original tracer roughly 80–140 hours,
both estimated, at fp32. Tracing runs on the CPU; the direction field, centrality, the seed tests and the finishing
of traced curves run on an NVIDIA GPU when one is found, and on the CPU otherwise.

## Install

```sh
pip install fibrs
```

Install it into the environment that has numpy (normally the one that runs `avf-spline-generator`). It
adds a `fibrs` command and a Python module, `fibrs`. `fibrs predict` is built into the same binary on Linux
and Windows x86-64; `pip install "fibrs[predict]"` adds the cuDNN and cuBLAS it needs when the system has none (see
[Predictions](#predictions)). Prebuilt CLI archives for Linux
x86-64 (glibc 2.28+) and Apple silicon (macOS 11+) are named `fibrs-<version>-<target>.tar.gz`. The Windows x86-64
archive is `fibrs-<version>-x86_64-pc-windows-msvc.zip`, holding `fibrs.exe`, which needs no Visual C++
redistributable.

The archive's binary bundles no OpenBLAS: it loads numpy's at run time, from the environment it is installed
into (`scripts/install-dropin.sh` puts it there) or else from the active one (`$VIRTUAL_ENV`). On macOS that
environment also needs `scipy-openblas64==0.3.31.188.0` (see [Numerics](#numerics)). Run from anywhere else, it
stops with "cannot find numpy's OpenBLAS" once the tracing predictions are loaded, after it has written the work
directory's `run.json`. Activate the numpy environment, or set `FIBRS_LAPACK` to the library's full path (on
Linux, the `libscipy_openblas64_*.so` file in its `site-packages/numpy.libs`, and on Windows the
`libscipy_openblas64_*.dll` in `Lib\site-packages\numpy.libs`; fibrs does not expand a `*`), and run
the same command again with `--resume`, through the same binary: an activated environment's `fibrs` can come first
on `PATH`, and a run resumes only with the version that started it. `pip install fibrs` into an environment
that holds its own numpy wheel needs none of this. fibrs looks only inside that environment, so one that inherits
numpy (`--system-site-packages`) or has a distribution's numpy, such as Debian's `python3-numpy`, needs
`scipy-openblas64==0.3.31.188.0` installed into it, as on macOS.

### With cargo

```sh
cargo binstall fibrs        # the release's prebuilt CLI binary
cargo install fibrs --locked  # a build from source
```

[cargo-binstall](https://github.com/cargo-bins/cargo-binstall) downloads the prebuilt binary from the GitHub release
on Linux x86-64, Apple silicon and Windows x86-64. Elsewhere it tries
[cargo-quickinstall](https://github.com/cargo-bins/cargo-quickinstall)'s third-party builds and then builds from
source. `cargo install` builds the crate from crates.io, with `--locked` at the dependency versions the release was
tested with; it needs Rust 1.100 or later, and on Windows the MSVC toolchain (see [Build](#build)). Either way only
the binary is installed, into `~/.cargo/bin` (`%USERPROFILE%\.cargo\bin` on Windows). Like the CLI archive's, it
loads numpy's OpenBLAS at run time, so activate the numpy environment or set `FIBRS_LAPACK`, as above.
`scripts/install-dropin.sh`, which makes it the drop-in `avf-spline-generator`, is in the CLI archive and the
repository.

## Use

```sh
fibrs --predictions P --array semantic --channel-order VH --tracing-predictions P \
  --tracing-array tracing --voxel-size-um 9.362 --storage compressed --threads 4 \
  --output fibers.afv --work-dir work
python -m fibrs ARGS      # the same, from Python
```

The direction field and centrality run on the GPU when a usable one is found: Linux or Windows x86-64, an NVIDIA GPU of compute
capability 8.0 or later, and a driver for CUDA 12.0 or newer (no CUDA toolkit). Otherwise they run on the CPU, and
the log says why. `--device cpu` keeps them on the CPU, `--device cuda` fails instead of falling back, and
`--cuda-device N` picks the GPU (by default the first usable one). The GPU direction field is the CPU's bit for bit:
the GPU certifies each voxel's eigenvector against an error bound, and any voxels it cannot certify, usually few, are
solved on the CPU with the same `dsyevd` (the log line counts them). Centrality rounds some steps differently on the
GPU (see "Differences from the original"), and a resumed run keeps the rounding it started with: one started on the
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
set `FIBRS_NVCOMP` to the path of a `libnvcomp.so.6` (`nvcomp64_6.dll` on Windows, where a relative path or bare
file name is taken from the current directory). GPU slab decoding trusts the prediction files: on a corrupt chunk nvCOMP's behaviour is undefined, so it may give wrong output or a CUDA fault where the CPU decoder would
report the corruption.

nvCOMP is not part of fibrs: it is NVIDIA's own PyPI package, `nvidia-libnvcomp-cu12` (version 6, under NVIDIA's
licence), which the `gpu-zstd` extra installs. With `FIBRS_NVCOMP=on` (in any case), fibrs loads that wheel from
the environment fibrs is installed into or the active one (`$VIRTUAL_ENV`), then tries the system's library path.
On Windows that last search covers only the executable's directory and System32, since fibrs takes the current
directory and `PATH` out of the DLL search, so point `FIBRS_NVCOMP` at a standalone nvCOMP's DLL. NVIDIA's Windows
nvCOMP needs the Visual C++ runtime (`vcruntime140.dll`, `msvcp140.dll`), which `fibrs.exe` itself does not; without
it, slabs stay on the CPU and the log says nvCOMP did not load.
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

To make an environment's `avf-spline-generator` run the port, so scripts that call it need no change:

```sh
scripts/install-dropin.sh VENV [BINARY]   # VENV/bin/avf-spline-generator -> fibrs
scripts/install-dropin.sh --restore VENV  # put the original back
```

BINARY defaults to the `fibrs` in the archive the script came from, then to the one on `PATH`. On Linux and macOS
the original entry is kept as `VENV/bin/avf-spline-generator.reference`, and callers that run
`Path(sys.executable).parent / "avf-spline-generator"` need no other change. On Windows, run it from Git Bash:
`VENV\Scripts\avf-spline-generator.exe` becomes a copy of `fibrs.exe`, the original is kept as
`VENV\Scripts\avf-spline-generator.reference.exe`, and a caller that builds the path itself names it with `.exe`
(`Path(sys.executable).parent / "avf-spline-generator.exe"`). Either way `NOTICE` and `licenses/` are copied to
`VENV/share/fibrs`. Run as `avf-spline-generator`, `--version` prints `0.2.0`, the release it stands in for; run
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
  native voxels apart: shortest first, each end once, and never into a loop. Pieces under two points are dropped,
  and so are fibres of zero length, as when a fibre's only point in a core lies on a seam and both its crossings land
  on that point; the log counts them. This is the rule the v2-run pipeline uses to join its Z bands, and the one
  [`fibrs join`](#joining-z-bands) applies to them. Near a
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

## Joining Z bands

A whole scroll is usually traced as Z bands, each from a window of its core planes plus a halo of the bands beside
it. `fibrs join` cuts each band's `.afv` to its core and joins the cut ends across the planes between adjoining
cores into one `.afv`, by the seam rule of [tiled tracing](#tiled-tracing) on Z. This command is fibrs's own; the
original has none.

```sh
fibrs join --manifest bands.json --output fibers-scroll.afv --stats fibers-scroll.json
```

The manifest lists the bands, each with its trace, the core planes it keeps and the window planes it was traced
from, `[z0, z1)`:

```json
{
  "bands": [
    {"afv": "trace/z08448-08960/fibers.afv", "core_z": [8448, 8960], "window_z": [8192, 9216]},
    {"afv": "trace/z08960-09472/fibers.afv", "core_z": [8960, 9472], "window_z": [8704, 9728]}
  ],
  "coordinate_space": "PHerc1545/20250821151648",
  "source_path": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1545/volumes/....zarr",
  "provenance": {"operation": "halo_band_trace_join", "haloPlanes": 256}
}
```

- **Bands.** A relative `afv` path is taken from the manifest's directory. Bands may be listed in any order; their
  cores must not overlap. Each trace must be complete, traced from its window's first plane (`tracing.origin_xyz`),
  with every point inside its window, `z0 <= z < z1`, and every band must be traced in the same volume (voxel size
  and shape), which the output takes. Cut ends are joined across a plane only where one core ends and the next begins.
- **Cut and join.** A fibre is cut to its band's core `z0 <= z < z1`; where it leaves through a plane shared with the
  next core, the point where it crosses the plane is added. Cut ends on the two sides of a plane are joined when they
  are of the same family and at most `--radius` native voxels apart (default 1.5): shortest first, each end once,
  never into a loop. Pieces under two points are dropped. A fibre or joined chain of zero length (every point one
  point, as when a fibre's only point in a core lies on the plane and both crossings land on it) is dropped and
  counted.
- **Output.** Every V fibre, then every H fibre; each family's uncut fibres in band order, then its joined fibres.
  The metadata is the public sets' for a whole volume (origin 0): the coordinate space first in the frame, the
  source path in the root and the manifest's `provenance` under its own key. `--stats` writes the counts per family
  (`uncut`, `cut_runs`, `cut_ends`, `joins`, `chains`, `zero_length_dropped`, `fibres`), the manifest's coordinate
  space and the file's totals. The output and the stats must be two files, neither of them an input.
- **Memory.** The join holds one family's pieces that end on a plane, plus up to three bands' fibres of that family:
  the band being cut, the next one, read ahead on its own thread, and the one after it being read.
- **Closing checks.** `--skip-closing-checks` skips the SQLite checks the output gets before it is closed
  (`integrity_check`, `foreign_key_check` and `rtreecheck` on `block_bounds`); the caller must then run them itself.
- **Speed.** On PHerc1545's 42 bands (34.6 GiB), `fibrs join` writes the same fibres, blocks and metadata as
  `join.py` in 13 minutes at 5.6 GiB of memory, against 26 minutes for `join.py` on the same machine. About six of
  its minutes are SQLite's closing `integrity_check`, which `join.py` also runs.
- **The v2-run pipeline.** This is `join.py`'s join (with `halojoin.py`), in the same order and with the same
  metadata. `join.py` can write the manifest from its plan (each band's `core_z` from the plan, `window_z` from
  its `trace.json`, and the coordinate space, source path and provenance it already builds), run `fibrs join`, and
  add its own keys (`plan_sha256`, `halo`, `afv_sha256`) to the `--stats` counts for its sidecar.

## Build

```sh
cargo build --release
FIBRS_LAPACK=/path/to/libscipy_openblas64_.so cargo test --release
```

The tests that trace (`tests/tile.rs`) need numpy's OpenBLAS, named by `FIBRS_LAPACK`; without it they are
skipped, except on CI, where they fail.

It builds on Linux, macOS and Windows (the MSVC toolchain) with Rust 1.100 or later, the first stable release with
the allocator API it uses. The tracer allocates through mimalloc. On Linux with glibc, on macOS and on Windows,
`fibrs predict` allocates through the system allocator; elsewhere it keeps mimalloc. `cargo build --release --no-default-features` builds a binary that
uses the system allocator throughout.
The wheel is built with maturin from
`pyproject.toml` (`maturin build --release`; the published Linux wheel uses `--zig --compatibility
manylinux_2_28`, the macOS wheel `MACOSX_DEPLOYMENT_TARGET=11.0`, and the Windows wheel
`RUSTFLAGS="-C target-feature=+crt-static"`, which links the C runtime into the binary).

## Numerics

On the CPU, the direction field's eigenvectors come from `dsyevd` in numpy's own OpenBLAS, on float64 copies as
`numpy.linalg.eigh` computes them, so they match `eigh` bit for bit. On the GPU, each voxel's eigenvector is
certified against an error bound to equal that result, and voxels that can't be certified go to the same `dsyevd`. The binary finds that library from its
own path (`bin/../lib/python*/site-packages/numpy.libs/libscipy_openblas64_*.so`, or on Windows
`Scripts\..\Lib\site-packages\numpy.libs\libscipy_openblas64_*.dll`), then from
`$VIRTUAL_ENV`, looking in each environment in numpy's copy first and then in the `scipy-openblas64`
package (`site-packages/scipy_openblas64/lib`). `FIBRS_LAPACK` overrides both.

macOS numpy wheels link Accelerate and bundle no OpenBLAS, so on macOS the wheel depends on
`scipy-openblas64==0.3.31.188.0`, the OpenBLAS build that numpy 2.4.6's Linux wheel bundles. With it, a
Mac writes the same `.afv` as Linux.

## Inputs

The same arrays the original reads through numpy and zarr 2: `.npy` files and Zarr v2 arrays or groups of
uint8, float16, float32 or float64, in either byte order and either array order. Zarr chunks may use any
channel chunking, either dimension separator and the Blosc, Zstd, Zlib, GZip or LZ4 compressor; Blosc
frames are decoded by c-blosc 1.21.6, the release numcodecs 0.15 bundles. Missing chunks read as the fill
value, including `NaN` and infinities, which the value check then rejects as the original does. Not
supported: Zarr filters, other numcodecs compressors (BZ2, LZMA and so on) and extended-precision floats.
`tests/fixtures/readers` holds one small array in each supported encoding with what the original reads
from it; `cargo test` checks every case.

## Same output

The published 0.4.0 binary, the one in the release archive and the PyPI wheel, writes the original's `.afv` with
`--device cuda --threads 8` on a 512-plane band of PHerc1545 (z 1280–1792: 82,646 fibres, 83,741 point blocks) and
on a 512-plane band of PHerc0826 (z 12032–12544: 149,680 fibres, 161,787 point blocks). The two files were compared
table by table: every fibre row, point block and spatial-index row, and every metadata value except the per-run
uuid. With `--device cpu --threads 4`, 0.4.0 also wrote the original's `.afv` on both bands, in four runs.

Earlier releases were checked the same way. On the PHerc0826 band, the port's `.afv` is identical to the original's:
149,680 fibres, 161,787 point blocks and 161,787 spatial-index rows, plus the metadata and schema. The 0.1.0 build
was also checked on a 512³ crop of the band: the wheel and the drop-in were installed into a fresh environment
beside the original, and the crop was traced through `avf-spline-generator` and through `python -m fibrs`. Both
outputs were identical to the original's `.afv`.

Centrality on the GPU gives the same `.afv`. On the PHerc0826 band and on a 512-plane band of PHerc1545
(z 1280–1792), with `--threads 4` and `--threads 8`, the port's output was identical with `--device cuda` and
`--device cpu`, and identical to the original's with `--device cpu` and with `--device cuda`. The original's CUDA
kernels round some multiply-adds as one operation where its CPU code rounds twice; the port reproduces each, and on
these bands both roundings give the same `.afv`.

CI traces a small synthetic input (`tests/fixtures/synthetic`) with every build's wheel on Linux, macOS and Windows,
through the drop-in and through `python -m fibrs`. Both must write what the original wrote for it, compared by
`scripts/compare_afv.py`: every metadata value except the per-run uuid, the schema and every table row. The drop-in
must also write it with no seed traced ahead of the one being committed (`FIBRS_DEBUG_WINDOW=1`) and with every seed
traced ahead traced again (`FIBRS_DEBUG_RETRACE=1`).

## Speed

### End to end, fibrs 0.4.0

From CT to `.afv` on three 512-plane bands, each across the whole plane, on one RTX 5090 and an Intel Core Ultra 7
270K Plus (24 cores). All prediction is fp32: float32 tensors with TF32 tensor-core convolutions, PyTorch's default
and the precision villa's runner uses. fibrs 0.4.0, the published release binary, runs `fibrs predict` at fp32 in
band mode, then traces with `--device cuda --threads 8`. Both fibrs times are measured; each is the faster of two
runs on its band. The public pipeline is villa's `lasagna.qualzz_inference` at fp32 (villa main `0bddf2757`), run
as published, followed by `avf-spline-generator` 0.2.0 on the CPU with `--threads 4`, its fastest setting here
(`--threads 8` was slower on both bands tried). The original tracer's times are measured; villa's are estimated.

| Band | Tiles with data | villa prediction, fp32 (estimated) | Original tracer, `--threads 4` | Public pipeline | fibrs 0.4.0 prediction, fp32 | fibrs 0.4.0 tracing | fibrs 0.4.0 total | Speed-up, fp32 |
|---|---|---|---|---|---|---|---|---|
| PHerc1545, z 1280–1792 | 8.4% | about 5,000 s | 1,613.8 s | about 6,614 s | 105.8 s | 11.0 s | 116.9 s | 56.6× |
| PHerc0826, z 12032–12544 | 15.2% | about 5,850 s | 4,615.7 s | about 10,466 s | 215.2 s | 25.0 s | 240.2 s | 43.6× |
| PHerc0358, z 2304–2816 | 33.2% | about 5,400 s | about 12,540 s (estimated) | about 17,940 s | 399.6 s | 43.2 s | 442.9 s | 40.5× |
| All three | | | | about 35,020 s | | | 800.0 s | 43.8× |

- **Tiles with data** is the share of the band's 128³ tiles with CT data inside the mask, the tiles band mode runs.
  villa runs every tile.
- **The machine's load.** The original tracer's runs had a one-minute load average of 10.9 (PHerc1545) and 14.9
  (PHerc0826), against about 3–4 during fibrs's runs. It kept about 2.6 cores busy, so the load probably cost it
  little, but it is not corrected for. Its PHerc0358 run took 15,631.6 s while the machine carried heavy other work
  (load average 26.4, up to 79.9). The table uses an estimate for a lighter load instead: its 32,739 s of CPU time
  over the 2.61 cores its other two runs got. Against the measured time, that band is 47.5× faster.
- **villa's estimate.** On PHerc1545 villa ran all 37,446 tiles in 4,266 s and was stopped during its final
  accumulator write, estimated at about 800 s from the 790 s write it made mid-band. PHerc0826 (44,376 tiles) and
  PHerc0358 (about 40,300) are scaled from that rate, 114 ms per tile, plus the same final write. That PHerc1545 run
  had a one-minute load average of 13–24, which may have slowed it, and it is not corrected for either.
- Both tracers traced fibrs's predictions of each band. A trace of villa's predictions could take somewhat longer
  or shorter, as the tracer's time depends on what it reads.

By stage, prediction is 47.3×, 27.2× and 13.5× faster (fp32 for both tools; villa's side estimated), and tracing
146×, 185× and 290× faster (362× against the measured PHerc0358 time; fibrs on the GPU at `--threads 8`, the original
on the CPU at `--threads 4`).

villa writes its tile accumulators to disk as float32 arrays as it runs, and on this machine that write took a large
part of its time; its runner has no setting for a smaller accumulator type. Without the write, estimated from villa
runs on 1,536-tile boxes of the same bands with the accumulators in memory (53–54 ms per tile), the public pipeline
would take about 3,600 s, 7,000 s and 14,700 s on the three bands, and fibrs 0.4.0 would still be 30.9×, 29.1× and
33.2× faster (31.6× over all three, fp32). The older villa figures under [Predictions](#predictions) and
[Why the speed-up varies](#why-the-speed-up-varies) come from a crop and a box small enough to keep the accumulators
in memory, so they leave this write out; they agree with these figures (2,231 s for villa's PHerc0358 band
prediction there, about 2,150 s here).

A whole scroll, end to end on one RTX 5090, estimated (not measured), with prediction at fp32:

| Scroll | Planes | fibrs 0.4.0 | villa and the original tracer |
|---|---|---|---|
| PHerc1545 | 20,961 | about 176–181 min | about 93–137 h |
| PHerc0826 | 16,920 | about 156–163 min | about 87–127 h |
| PHerc0358 | 14,744 | about 173–175 min | about 81–123 h |

That is roughly 28–49× faster per scroll. For fibrs, prediction was run over the whole of PHerc1545 and PHerc0826
with fibrs 0.2.0, whose prediction runs at about 0.4.0's speed (2.7 h and 2.4 h, fp32, measured). PHerc0358's
prediction and all three traces are scaled from the bands above by each scroll's count of tiles with data, with each
scroll traced as its 512-plane bands. For villa and the original tracer, villa's rate per tile above (114–134 ms, the
upper figure including its final write) is multiplied by the scroll's tile count, and the original tracer's time
per tile with data on the three bands (0.51–0.94 s) by the scroll's tiles with data; the range spans those rates.
Neither counts the download or joining the bands' outputs into one file, and neither reference tool was run on a
whole scroll. Tracing in overlapping 1024-plane windows instead of single bands would roughly double the tracing for
both tools, adding about 15–20 minutes for fibrs and about 40–80 hours for the original.

### Tracer alone, fibrs 0.1.0 to 0.3.1

These runs, with earlier releases, time the tracer alone on the CPU.

Five runs on an Intel Core Ultra 7 270K Plus (24 cores), with `--threads 4` for both tools and other work
running on the machine each time. Runs 1 and 2 trace the same 512 × 8169 × 8169 band of PHerc0826, run 3 a
512-plane band of PHerc1545 (z 1280–1792), run 4 a 512 × 7783 × 7783 band of PHerc0358 (z 2304–2816) and run 5
a 512-plane band of PHerc0813 (z 2304–2816):

| Run | Tool | Wall | CPU time | Peak memory |
|---|---|---|---|---|
| 1 | `avf-spline-generator` 0.2.0 | 11,840 s (3.3 h) | about 11,940 s | 4.6 GiB |
| 1 | `fibrs`, built before 0.1.0 | 191 s | 516 s | 12.4 GiB |
| 2 | `avf-spline-generator` 0.2.0 | 4,686 s (1.3 h) | 12,296 s | 4.6 GiB |
| 2 | `fibrs` 0.1.0 | 172 s | 472 s | 10.8 GiB |
| 3 | `avf-spline-generator` 0.2.0 | 1,621 s (27 min) | 4,149 s | 4.6 GiB |
| 3 | `fibrs` 0.1.0 | 73.4 s | 191 s | 7.0 GiB |
| 4 | `avf-spline-generator` 0.2.0 | 15,632 s (4.3 h) | 32,739 s | 4.6 GiB |
| 4 | `fibrs` `42c4710` | 332 s | 902 s | 14.0 GiB |
| 5 | `avf-spline-generator` 0.2.0 | about 9,176 s (2.5 h) | about 27,460 s | 4.7 GiB |
| 5 | `fibrs` 0.2.0 | 202 s | 564 s | 11.8 GiB |
| Speedup, runs 2 to 4 | | **22×–47×** | **about 22×–38×** (38× with the 853 s repeat of run 4) | |

- Run 1: the one-minute load average was 17.5–18.8. The original's run was interrupted by a reboot and
  finished with `--resume`. Its wall time is the sum of the two parts, and the first part's CPU time is
  estimated from the second part's CPU-to-wall ratio (1.01). An earlier uninterrupted run of the original on
  the same band took 12,728 s. Run 1's total is estimated, so it sets neither end of the range.
- Run 2: the one-minute load average averaged 13.9 during the original's run and 8.8 during fibrs's. The two
  `.afv` files held the same 149,680 fibres (compared logically, as each holds its own run uuid).
- Run 3: the one-minute load average was 10.0 during the original's run and 7.4 during fibrs's. This band is
  sparser. The two `.afv` files held the same 82,646 fibres and 83,741 point blocks.
- Run 4: the two runs were not at the same time. This is the densest band measured. The two `.afv` files
  held the same 448,286 fibres and 460,560 point blocks.
- Run 5: the two runs were not at the same time. The original's first attempt logged progress for 5,074 s before it
  was stopped and finished with `--resume` in 4,102 s. Its wall time is the sum of the two parts, and the first
  part's CPU time is estimated from the second part's CPU-to-wall ratio (2.99). The resumed part loaded the
  direction field and the V centrality instead of recomputing them; it restored 103,340 batches 105 s after it
  started and was tracing again after 126 s. The ratios are about 45× on wall time and about 49× on CPU time; as
  the total is estimated, the run sets neither end of the range. The two `.afv` files held the same 337,735 fibres
  and 344,279 point blocks. Run 5's memory is the peak RSS of both tools.

The original kept about 2.1–3.0 cores busy in runs 2 to 5 and about one in run 1; the CPU times of runs 1 and 5 are
partly estimated. The port kept about 2.6–2.8. The ratio varies with the run and with the machine's load: the
ranges are the lowest and highest of the uninterrupted runs, 2 to 4. The CPU range also covers a second run of the
same build on the run 4 band, which used 853 s of CPU time (38×).

On dense bands, fibrs's speed also depends on how it reads the predictions and allocates memory. Take a
512 × 2048 × 2048 crop of PHerc0358's densest area (z 1792–2304, y 2176–4224, x 2944–4992) with `--threads 8` and
`--field-cache-mib 224`, each pair of builds run back to back on the same busy machine. A pre-release build of
0.2.0 that decodes the predictions 8 planes at a time took 50–52 s with a 2.6 GiB peak and wrote the same `.afv` as
0.1.0, which took 104 s with a 3.5 GiB peak. One that decoded whole chunks took 69–72 s, and one that also gave
each decoded chunk its own mapping spent 40% of its CPU time in the kernel.

Tracing seeds ahead on idle threads (0.3.1, measured with glibc malloc, before mimalloc became the allocator) takes
that crop at `--threads 8` from 0.2.0's 47–55 s to 41–43 s, in two pairs of runs pinned to the same eight cores on a busy machine: tracing V from 12.0–15.4 s to 8.5–9.9 s and H
from 8.7–10.2 s to 6.2–6.4 s. The `.afv` is the same, and the peak is 2.6 GiB for both.

On whole bands with `--device cpu` and glibc malloc, each pair of builds run back to back on the same busy machine, it took
PHerc1545 from 0.3.0's 77 s to 58 s at `--threads 4` and from 46 s to 43 s at `--threads 8`, and PHerc0826 from
174 s to 153 s and from 119 s to 106 s. The `.afv` is the same, and the peak is at most 0.3 GiB higher (5.6 GiB
for PHerc1545, 9.1–9.2 GiB for PHerc0826).

### Centrality on the GPU

Measured before the direction field moved to the GPU, with `--threads 8` on an RTX 5090 host, each band ran four
times back to back, CPU, GPU, GPU, CPU, with other work running on the machine:

| Band | Centrality and seeds, CPU | Same, GPU | Whole run, CPU | Whole run, GPU |
|---|---|---|---|---|
| PHerc1545, z 1280–1792 | 8.1 s | 1.8 s | 44.2 s | 38.0 s |
| PHerc0826, z 12032–12544 | 20.4 s | 5.0 s | 120 s | 105 s |

Each figure is the mean of its two runs, and all eight `.afv` files were identical. The direction field (20–51 s
here) and tracing ran on the CPU in these runs, so the whole run gains less, and its time moves with the machine's load
(a mean one-minute load of 8–13 across these runs).

The original's `--device cuda` doesn't make centrality faster. On the same bands with `--threads 8` its centrality
took 52.6 s and 65.9 s with `--device cuda`, against 35.3 s and 59.3 s with `--device cpu`. Its kernels ran for
2.2 s and 4.7 s of that; the rest went to host work and to copying each 128³ tile and its halo to the GPU and back.

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
  logs each stage's field sizes and resident memory (the working set on Windows; zero on macOS, which the log
  cannot read it from).
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

## Predictions

`fibrs predict` runs Qual's [AFV fibre model V2](https://huggingface.co/Qualzz20/afv_fiber_9um_v2) over a CT
volume and writes the probability zarr that villa's `python -m lasagna.qualzz_inference` writes: `semantic`
(2 × Z × Y × X, uint8 V and H fibre probabilities) and `tracing` (the same at half resolution), the input that
`fibrs` and [AVF_SPLINE_GENERATOR](https://github.com/Qualzz/AVF_SPLINE_GENERATOR) trace. Chunks that are all
zero are not written (fill value 0), so a reader must treat a missing chunk as zero, as zarr does.

The model, its weights and its output are Qual's, and the tiling, blending and quantisation are villa's lasagna
inference. The network runs natively on the GPU through [volkit](https://github.com/praveenperera/volkit)'s
`AfvV2`, with no PyTorch or Python: per tile, the model's own normalisation, the network in float32 with TF32
convolutions (PyTorch's defaults), the softmax and the blend, then each column's finalise, in the reference's
float32 order. Tiles are read on several threads ahead of the GPU, and one thread writes the output.

### Requirements

- Linux or Windows x86-64 and an NVIDIA GPU of compute capability 8.0 or later (RTX 30 series, A100 and newer;
  not the T4 or the RTX 20 series), with a driver for CUDA 12 or later. The binary needs only the driver
  (`libcuda.so.1`, or `nvcuda.dll` in System32 on Windows), not a CUDA toolkit.
- cuDNN 9 (`libcudnn.so.9`, `cudnn64_9.dll`) and cuBLAS 12 (`libcublas.so.12`, `cublas64_12.dll`).
  `pip install "fibrs[predict]"` installs them from NVIDIA's pip packages (`nvidia-cudnn-cu12`,
  `nvidia-cublas-cu12`); a system install works too.
- On Windows, the GPU paths (`fibrs predict` and `--device cuda`) build and pass CI, which has no GPU, but have not
  yet run on a Windows machine with an NVIDIA GPU.
- No PyTorch, and no Python at run time beyond what installs the wheel.

On macOS and other platforms the binary has no GPU engine: `fibrs predict --dry-run` plans a run there, and a real
run stops with an error naming what it needs. The tracer is unaffected.

### Install and use

```sh
pip install "fibrs[predict]"   # or: pip install fibrs, with cuDNN 9 and cuBLAS 12 installed system-wide
```

Download the model snapshot (`model.py`, `plans.json`, `dataset.json`, `checkpoint_final.pth`) from
[Qualzz20/afv_fiber_9um_v2](https://huggingface.co/Qualzz20/afv_fiber_9um_v2); `--model-dir` is that directory, which
must not hold a `fold_0` checkpoint or be named `fold_*`, as the loader would then read other files.

```sh
# villa's output, for a whole CT array
fibrs predict --mode villa --ct scroll.zarr/0 --model-dir afv_fiber_9um_v2 --out pred.zarr

# one z band, resumable
fibrs predict --ct scroll.zarr/0 --z0 12032 --z1 12544 --model-dir afv_fiber_9um_v2 --out band.zarr --max-minutes 60

python -m fibrs predict ARGS   # the same, from Python
```

`--ct` is a level-0 uint8 zarr v2 array (the `/0` directory of an OME-Zarr). `--device` is `cuda` or `cuda:N`.
`--voxel-size-um` is the CT's voxel size, written as the output's `voxel_size_um` attribute (default 9.362).
Run the same command again on the same `--out` to resume a capped (`--max-minutes`) or interrupted run at the
first unfinished column; the resume must use the same settings, binary and GPU model, or it refuses the output.
One run at a time may use an output: a run locks it until it ends, and a second run on the same `--out` is
refused.
It recognises the CT by its metadata and the names and sizes of its chunk files, not their contents, so a chunk
rewritten in place at the same size is not detected: after changing the CT, use a fresh `--out`.
`--expect-sha256` refuses a checkpoint with another hash. `--dry-run` reads the CT and prints, per column, the
tile batches a run would send to the GPU and how many tiles it would skip, without a GPU and without writing.

The binary finds cuDNN and cuBLAS in the Python environment it is installed in (`site-packages/nvidia/*/lib`
beside its `bin/fibrs`) before the system's, by loading those copies before the engine asks for the libraries by
name; `FIBRS_CUDA_LIBS=system` skips the pip copies. The engine asks for the unversioned names (`libcublas.so`,
`libcudnn.so`) first, so a system copy under one of those on the loader's path, such as a CUDA 13 toolkit's
`libcublas.so`, is used instead, whatever its version. The run still works; it warns when the cuBLAS it loaded is
not 12.x, and its provenance records the cuBLAS and cuDNN versions. To use cuBLAS 12 on such a machine, put a
directory holding `libcublas.so -> libcublas.so.12` and `libcudnn.so -> libcudnn.so.9` symlinks first on
`LD_LIBRARY_PATH`, or keep the CUDA 13 library directories off the loader's path. Each run's `summary-*.json`
lists the cuDNN and cuBLAS files it loaded.

On Windows the pip copies are the DLLs in `Lib\site-packages\nvidia\*\bin` beside `Scripts\fibrs.exe`. A system
cuDNN or cuBLAS is found in the executable's directory, System32, each absolute `PATH` entry, then
`%CUDA_PATH%\bin\x64` and `%CUDA_PATH%\bin`, never in the current directory: the binary takes the current
directory and `PATH` out of Windows' default DLL search when it starts, and loads each library by full path (see
volkit's `dll` module). A newer cuBLAS is never picked up by mistake there, as each CUDA release installs its DLLs
under names that carry the major version (`cublas64_13.dll`). The driver is looked up by name before it is pinned to
System32, so a stray `nvcuda.dll` or `cuda.dll` beside `fibrs.exe` is opened first: keep none there.

cuDNN's convolution algorithms are timed on the first run and cached under `~/.cache` (`%LOCALAPPDATA%` on Windows;
see volkit's documentation for the cache variables). They set the rounding, so the provenance
lists them and a resume that picks others is refused: keep the cache between a run and its resume.

### Modes

| Mode | What it computes |
|---|---|
| `--mode villa` | the whole array, as villa's `lasagna.qualzz_inference` computes it, with villa `97fb9336c`'s rule for absent input chunks: a 128³ output chunk that no stored input chunk overlaps is not written, so it reads as zero; villa main writes those chunks too, so the outputs differ there |
| band mode (the default) | a z band `[z0, z1)`, reflected at the band's own edges, tiles whose input is all zero skipped |

Use `--mode villa` when you want villa's output; on an input with absent chunks, see the table. Band mode is for
splitting a scroll into z bands that run separately and resume: it does not reproduce villa's output, because villa
runs the whole volume, so its tiles near a band's edges see the planes beyond it, and villa main runs every tile,
even where the input is all zero.

The network runs on one tile at a time in both modes. `--batch` (band mode, default 4, at most 16) is how many
tiles go to the GPU together, and `--readers` (tile reader threads, default 4) how many read ahead; both change
the speed, not the output. `--column` (the size of the y–x columns a run is split into, default 480) leaves villa
mode's output unchanged; in band mode it is part of the run's settings, as in the Python version.

### Precision

By default `fibrs predict` runs at the precision of villa's `lasagna.qualzz_inference` on PyTorch, to keep parity
with it; `--mode villa` also computes the same tiles over the whole array when every input chunk is stored, while
band mode differs at the band's edges and in skipping all-zero tiles (see Modes). bf16 is opt-in, never the default
and will never be used for published fibre sets.

- `--precision fp32` (default): float32 tensors with TF32 tensor-core convolutions, the precision PyTorch uses for
  the reference by default.
- `--precision bf16` (opt-in): faster, small output differences. The network and its seg head run in bf16, with
  the normalisation, softmax and blend still in float32. Against fp32, on 512-plane bands of PHerc0826 and
  PHerc0813 traced with `fibrs`, the traced fibre length agreed to about 99% (99.2–99.4%, V and H fibres),
  97.0–97.8% of fibres matched, and the fibre counts were within 0.3%; the 99.9th percentile of the probability
  differences over the foreground was 11 (`semantic`) and 9 (`tracing`) of 255 on PHerc0826, and 8 and 7 on
  PHerc0813.

PyTorch's `fp32-ieee` and `fp16` modes are not offered.

### Speed

Against villa's `lasagna.qualzz_inference` on PyTorch, at its default precision (float32 with TF32 tensor-core
convolutions). Both ran the same 726-tile crop of PHerc0826 (z 12032–12544, y 4096–5120, x 1536–2560) on one RTX
5090 (driver 595.112.07). The runs alternated (ABBA), with 6 timed pairs per precision after one warm-up run of
each. Each time is the whole command's wall time: both read the same CT crop and write arrays of the same shapes
and chunk sizes, and the time includes those reads and writes, so it measures more than the network. On this crop
villa ran 726 tiles and fibrs 864, as `--mode villa` runs a tile once for each of its columns that the tile's core
overlaps. The speedup is the mean of the per-pair ratios:

| `fibrs predict --mode villa` precision | villa, PyTorch fp32 (median) | fibrs (median) | Speedup, mean (95% CI) |
|---|---|---|---|
| fp32 (default) | 45.37 s | 26.77 s | 1.72× (1.66–1.78×) |
| bf16 (opt-in) | 45.35 s | 14.06 s | 3.19× (3.10–3.27×) |

On a 512 × 2048 × 2048 box of PHerc0358 (z 2304–2816, y 2176–4224, x 2944–4992) with no all-zero tile, one run of
each at fp32 on the same GPU, not at the same time, took 165.6 s for villa, 85.3 s for fibrs `42c4710` in band mode
(1.94×) and 84.5 s with `--mode villa` (1.96×). At fp32 the measured speedup is 1.68×–1.96×; the lowest is the
single PHerc0826 run in band mode behind the end-to-end row under
[Why the speed-up varies](#why-the-speed-up-varies) (45.4 s and 27.0 s). `42c4710` has the same prediction code and
dependencies as the 0.2.0 release, which adds one tracer change.

On the PHerc0826 crop, fibrs was a pre-release build of 0.2.0 (`e9adcc9`); both fibrs builds ran with cuDNN 9.27.0
and cuBLAS 13.4.1.3. villa ran PyTorch 2.14.1 (CUDA 13.0) with cuDNN 9.24, at `0bddf2757`.

### Parity

- **fp32 (default)**, `fibrs predict --mode villa` against villa main's `lasagna.qualzz_inference` (`0bddf2757`)
  on the PHerc0826 crop above:
  - the 99.9th percentile of the probability differences over the foreground is 1 of 255 in each channel;
  - the largest difference is 3 of 255;
  - the most likely class agrees at 99.97% of voxels in `semantic` and 99.973% in `tracing`;
  - both cover all 256 output chunks, and leave nothing uncovered.

  The outputs aren't identical, as expected from a different implementation. The differences are of the same
  order as two runs of villa itself with `cudnn.benchmark` on (99.9th percentile 1).
  The PHerc0826 timed runs' outputs were scored the same way. At fp32 they meet the bar: each of fibrs's seven runs
  differs from villa's at 1.16% of voxel positions in `semantic` and 0.99% in `tracing`, by at most 3 of 255,
  with the same counts in every run. At bf16 (opt-in) they differ from villa's fp32 at 10.7% and 11.5% of
  positions, by at most 33 and 20 of 255; the 99.9th percentile over the foreground is 6 and 4, and the most
  likely class agrees at 99.55% and 99.62% of voxels.
- **Band mode** matches the reference engine value for value on a full PHerc0826 band, with 0 differences at
  fp32 and at bf16. fibrs doesn't store chunks that are entirely zero, so its output is smaller on disk; zarr
  reads a missing chunk as zero.
- **Libraries:** running the release build with the cuDNN 9.28.1 and cuBLAS 12.9 that `fibrs[predict]` installs,
  instead of a system cuDNN 9.27.0 and cuBLAS 13.4, left its outputs unchanged on three band columns at fp32 and
  bf16. With the pip cuDNN alone, the villa-mode PHerc0826 crop at fp32 was unchanged too.
- **bf16 (opt-in)** against fp32: the figures under Precision.

### Limits

- The output's `voxel_size_um` attribute is 9.362, the PHerc0826 CT's voxel size, unless `--voxel-size-um` gives
  another; the reference writes 9.362 for every CT. fibrs does not read it from the CT's OME-Zarr metadata. The
  tracer doesn't read the attribute either: pass the CT's voxel size to it as `--voxel-size-um` too.
- A column's accumulators span the whole band height on the GPU, about 14 bytes per voxel (a 480 × 480 column of
  8,000 planes needs about 24 GiB), and the engine takes at most 2^31 − 1 voxels in one column (a 480 × 480 column
  of up to 9,320 planes). A run whose pending columns don't fit, in the free GPU memory or in that limit, stops
  before it starts and says what would fit: a smaller `--column` when one does (in villa mode the column size
  leaves the output unchanged), else a shorter band.
- Villa mode reads which chunks are stored from the input's directory listing, so the input must be a zarr v2
  directory store (`.` or `/` chunk separator).
- Outputs from different GPU models, or with different cuDNN versions, can differ slightly, as the model's own
  outputs do under PyTorch.

### Tests

`cargo test` checks, without a GPU, that `fibrs predict --dry-run` feeds the network exactly the tiles, batches and
bytes that fibrs's Python band reference (`fibrs.predict` at fibrs PR #3) feeds it, on small synthetic CTs in band
and villa mode (`tests/predict.rs`, `tests/fixtures/predict`, written by `scripts/predict_fixtures.py`), and that
bad arguments are refused with the reference's messages.

## Why the speed-up varies

Each speed-up above depends on the data, so each is quoted as the range of the measured points where there is
more than one:

- **Density.** The share of a region that is papyrus rather than air changes from scroll to scroll and along
  each scroll. The table gives it as the share of 128³ tiles that `fibrs predict --dry-run` would run (band mode,
  tiles whose input isn't all zero), measured the same way for every row.
- **Prediction skips air.** villa main runs every tile; band mode skips tiles whose input is all zero, so on a
  sparse band it runs a fraction of villa's tiles and on a dense one about as many. `--mode villa` doesn't skip
  all-zero tiles, and on the PHerc0826 crop it ran 864 tiles where villa ran 726, since it runs a tile once for
  each of its columns that the tile's core overlaps. Every measured prediction time is the whole command's wall
  time, reading the same CT crop and writing arrays of the same shapes and chunk sizes. villa's prediction was not
  run on the whole PHerc0358 band; its time there is estimated, so that end-to-end ratio is an estimate too.
- **Tracing varies with the run.** The tracer ratio varies with the run and with the machine's load, so it is given
  as the lowest to highest of the uninterrupted runs; the two runs whose original was interrupted and resumed have
  estimated totals and set neither end. The original's time per fibre also differs between bands and between runs:
  about 0.035 s on the dense PHerc0358 band (run 4) and about 0.027 s on PHerc0813 (run 5, estimated), and its two
  uninterrupted runs on the same PHerc0826 band took 4,686 s and 12,728 s. fibrs took about 0.6–0.75 ms per fibre
  on runs 4 and 5. Pre-release builds of 0.2.0 traced dense data slower than 0.1.0; a later pre-release build is
  faster than 0.1.0 there (the last row).

| Stage | Scroll | Region (z / y / x) | Tiles run | Original | fibrs | Ratio | Precision | Measured as |
|---|---|---|---|---|---|---|---|---|
| Tracing | PHerc0826 | z 12032–12544, whole plane | 15.2% | 11,840 s | 191 s | about 62× | CPU | one run each; the original was interrupted and resumed; total estimated (run 1; fibrs built before 0.1.0) |
| Tracing | PHerc0826 | z 12032–12544, whole plane | 15.2% | 4,686 s | 172 s | 27× | CPU | one run each (run 2; fibrs 0.1.0) |
| Tracing | PHerc1545 | z 1280–1792, whole plane | 8.4% | 1,621 s | 73.4 s | 22× | CPU | one run each (run 3; fibrs 0.1.0) |
| Tracing | PHerc0358 | z 2304–2816, whole plane | 33.2% | 15,632 s | 332 s | 47× | CPU | one run each, not at the same time (run 4; fibrs `42c4710`, before 0.2.0's chunk-cache change) |
| Tracing | PHerc0813 | z 2304–2816, whole plane | 25.4% | 9,176 s | 202 s | about 45× | CPU | one run each, not at the same time; the original was interrupted and resumed; total estimated (run 5; fibrs 0.2.0) |
| Prediction, `--mode villa` | PHerc0826 | z 12032–12544 / y 4096–5120 / x 1536–2560 | 100% | 45.37 s | 26.77 s | 1.72× (1.66–1.78×) | fp32 | ABBA, 6 pairs: medians, mean ratio with 95% CI |
| Prediction, `--mode villa` | PHerc0826 | same crop | 100% | 45.35 s | 14.06 s | 3.19× (3.10–3.27×) | bf16 (opt-in) | ABBA, 6 pairs: medians, mean ratio with 95% CI |
| Prediction, band mode | PHerc0358 | z 2304–2816 / y 2176–4224 / x 2944–4992 | 100% | 165.6 s | 85.3 s | 1.94× | fp32 | one run each, not at the same time; fibrs `42c4710` |
| Prediction, `--mode villa` | PHerc0358 | same box | 100% | 165.6 s | 84.5 s | 1.96× | fp32 | one run each, not at the same time, against the same villa run; fibrs `42c4710` |
| End to end, prediction in band mode | PHerc0826 | z 12032–12544 / y 4096–5120 / x 1536–2560 | 100% | 387.5 s | 41.7 s | 9.3× | fp32 | one run of each stage |
| End to end, prediction in band mode | PHerc0358 | z 2304–2816, whole plane | 33.2% | 17,863 s, prediction estimated | 739 s | 24×, estimated | fp32 | villa's prediction time extrapolated from the box above; the rest one run of each stage, fibrs `e9adcc9` predicting and `42c4710` tracing; both tracers read fibrs's prediction |
| Tracing, fibrs 0.1.0 → a pre-release build of 0.2.0 | PHerc0358 | z 1792–2304 / y 2176–4224 / x 2944–4992 | 100% | 104 s (0.1.0) | 50–52 s (pre-release 0.2.0) | 2.0–2.1× | CPU | back to back on one machine, `--threads 8`; not in the tracing range |

"Original" is `avf-spline-generator` 0.2.0 for tracing and villa's `lasagna.qualzz_inference` (main `0bddf2757`,
PyTorch fp32) for prediction; end to end is the two stages added, on the PHerc0826 crop from a single run of each
(prediction 45.4 s and 27.0 s, `fibrs predict` in band mode, not the ABBA medians). On the PHerc0358 band villa's
prediction was not run: its 2,231 s is the box run's 5.4 s outside inference plus its 160.2 s of inference scaled
from the box's 2,904 tiles to the band's 40,344, as villa main runs every tile. fibrs's prediction (407.2 s) and
both tracers (15,632 s and 332 s) were measured. Both tracers read the same saved fibrs prediction of the band, and
each stage was timed in its own run. Tracing ran with `--threads 4` for both tools unless the row says otherwise.

## Licence

Apache-2.0, like the original; the licence text is `licenses/apache-2.0.txt`. `NOTICE` lists the ported code
(AVF_SPLINE_GENERATOR, one NumPy sort routine), the statically linked C libraries and what `fibrs predict` takes
from villa (MIT: its tile rule in villa mode); their licence texts are in `licenses/`. The AFV model and its
weights are Qual's, and cuDNN, cuBLAS and nvCOMP are NVIDIA's; none of them is distributed here.
