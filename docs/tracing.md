# Tracing details

## GPU

When a usable GPU is found, it computes the direction field, centrality, the seed tests and the finishing of
traced curves. Tracing itself always runs on the CPU.

- **Direction field.** The GPU's result matches the CPU's bit for bit. The GPU checks each voxel's eigenvector
  against an error bound. Any voxel that fails the check, usually only a few, is solved on the CPU with the same
  `dsyevd`, and the log counts them.
- **Centrality.** The GPU rounds some steps differently, the way the original's `--device cuda` does (see
  [Differences from the original](#differences-from-the-original)). A resumed run keeps the rounding it started
  with:
  - a run started on the CPU keeps centrality on the CPU. With `--device auto` it can still use the GPU for the
    direction field and finishing. `--device cuda` refuses to resume it.
  - a run started on the GPU can resume without one. The CPU then reproduces the GPU's rounding, more slowly.
- **Finishing** cuts each traced curve where the predictions disagree, then smooths, resamples and re-checks each
  piece. It gives the same result on the GPU and the CPU. The GPU keeps a cache of up to 8 GiB of prediction slabs
  per family, and any batch of curves that needs more than the cache holds is finished on the CPU. The log counts
  both after each family.
- **Failures.** If the GPU fails mid-run, for example by running out of memory, `--device auto` carries on on the
  CPU and `--device cuda` stops.

### GPU slab decoding

Prediction slabs are decoded on the CPU by default. To decode them on the GPU with NVIDIA's nvCOMP, run
`pip install "fibrs[gpu-zstd]"` and set `FIBRS_NVCOMP=on`. You can also set `FIBRS_NVCOMP` to the path of a
`libnvcomp.so.6` (`nvcomp64_6.dll` on Windows). The log says whether slabs are decoded on the GPU, and if not, why.

The output is the same on sound files, but GPU decoding does not check the files. Where the CPU decoder would
report a corrupt chunk, nvCOMP can give wrong output or a CUDA fault. Use it only on prediction files you trust.

## Threads

`--threads N` sets how many threads trace and finish curves. Idle threads trace seeds ahead, and one more thread
commits them in order, so up to N + 1 threads can be busy. On a dense crop of PHerc0358, tracing kept 4.2–4.5 cores
busy at `--threads 4` and 7.7 at `--threads 8`.

A seed traced ahead is traced again if an earlier seed changes the region it read. That keeps the output the same
at every thread count. The export to `.afv` uses the same threads.

## Tiled tracing

`--tiles-yx NY NX` traces a window as NY × NX tiles in Y and X, one after another in the same process. fibrs
holds the fields of every voxel with fibres, so peak memory grows with how much of the window holds data. With
tiles, the peak is roughly that of the densest tile. The original has no such option.

- **Tiles.** Each tile is traced from its core plus a halo of up to `--tile-halo` voxels (default 256) on each
  side. With more than one tile the halo must be at least 1. Z is not tiled.
- **Alignment.** Tile edges and the halo are multiples of A voxels. A is the least common multiple of the native
  chunk edge, twice the tracing chunk edge, and 256. For the usual 128³ native and 64³ tracing chunks, A is 256.
  NPY input counts as 64-voxel chunks. Each core must be at least A wide, so a window narrower than N × A can't be
  split into N tiles along that axis. Alignment keeps reads on whole chunks and keeps every tile on the window's
  own grids, so away from the seams a tile traces just as the whole window would.
- **Seams.** Each tile's fibres are cut to its core, adding the point where they cross the tile edge. Cut ends from
  tiles that share an edge are joined when they belong to the same family and are at most 1.5 voxels apart. Joins
  go shortest first, use each end once and never make a loop. Pieces with fewer than two points and fibres of zero
  length are dropped, and the log counts them. A fibre that cuts across a tile's corner still joins through it.
  [`fibrs join`](#joining-z-bands) and the v2-run pipeline join Z bands by the same rule.
- **Differences from an untiled run.** These happen only near seams. Two cut ends more than 1.5 voxels apart stay
  split, and a tile sees its neighbours' data only as far as its halo reaches.
- **Output.** The `.afv` has the untiled run's metadata. Each family lists every tile's uncut fibres, tile by tile
  and row by row, then the fibres joined across seams. `--tiles-yx 1 1` writes the same `.afv` as an untiled run.
- **Memory.** Each tile allocates its own fields and chunk cache, and frees them before the next tile starts.
- **Work directory.** Each tile has its own directory, `work/tile-YY-XX/family-N/`. `--resume` skips finished
  tiles. `run.json` records the tile grid.

## Joining Z bands

`fibrs join` applies the tiled-tracing seam rule along Z. The original has no such command.

- **Bands.** List them in any order. Their cores must not overlap. Each trace must be complete, traced from its
  window's first plane, with every point inside its window. All bands must come from the same volume (voxel size,
  shape and scan).
- **Cut and join.** Each fibre is cut to its band's core. Cut ends on either side of a band boundary are joined
  when they belong to the same family and are at most `--radius` voxels apart (default 1.5).
- **Output.** First every V fibre, then every H fibre. Within a family, the uncut fibres come in band order,
  followed by the joined ones. The metadata describes the whole volume (origin 0), with the manifest's coordinate
  space, source path and `provenance`.
- **Stats.** `--stats` writes per-family counts (`uncut`, `cut_runs`, `cut_ends`, `joins`, `chains`,
  `zero_length_dropped`, `fibres`) and totals. The output and stats files must not exist yet, but their
  directories must.
- **Memory.** The join holds at most three bands' fibres of one family at a time, plus the pieces waiting to be
  joined.
- **Speed.** On PHerc1545's 42 bands (34.6 GiB), `fibrs join` writes the same output as `join.py` in 13 minutes,
  using 5.6 GiB of memory, where `join.py` takes 26 minutes. About six of those 13 minutes are SQLite's closing
  `integrity_check`, which `join.py` also runs.
- **The v2-run pipeline.** `fibrs join` does the same join as `join.py` (with `halojoin.py`), in the same order and
  with the same metadata. `join.py` can write the manifest from its plan, run `fibrs join`, then add its own keys
  (`plan_sha256`, `halo`, `afv_sha256`) to the stats.

## Numerics

On the CPU, the direction field's eigenvectors come from `dsyevd` in numpy's OpenBLAS, computed on float64 copies
as `numpy.linalg.eigh` computes them, so they match `eigh` bit for bit. The GPU checks each of its eigenvectors
against that result, as described [above](#gpu).

On macOS fibrs uses `scipy-openblas64==0.3.31.188.0`, the same OpenBLAS build that numpy 2.4.6's Linux wheel
bundles, so a Mac writes the same `.afv` as Linux.

## Inputs

fibrs reads the same arrays as the original:

- `.npy` files and Zarr v2 arrays or groups
- uint8, float16, float32 or float64, in either byte order and either array order
- Zarr chunks with any channel chunking, either dimension separator, and Blosc, Zstd, Zlib, GZip or LZ4
  compression. Blosc is decoded by c-blosc 1.21.6, the version numcodecs 0.15 bundles.

Missing chunks read as the fill value. A fill value of `NaN` or infinity is rejected, as the original rejects it.
Zarr filters, other compressors (BZ2, LZMA and so on) and extended-precision floats are not supported.
`tests/fixtures/readers` holds a small array in each supported encoding, and `cargo test` checks every one.

## Memory allocator

The tracer allocates through mimalloc. `fibrs predict` uses the system allocator on Linux with glibc, macOS and
Windows. `cargo build --release --no-default-features` uses the system allocator everywhere.

On Linux with glibc, fibrs turns off transparent huge pages (THP) for the tracer, as `MIMALLOC_ALLOW_THP=0` would,
and asks Linux to disable them for its process with `prctl(PR_SET_THP_DISABLE)`. Child processes inherit this.
With THP in `madvise` mode, most of mimalloc's huge-page requests fall back to small pages after the kernel has
tried to compact memory. On a dense 1024-plane window of PHerc0358 (`--device cuda --threads 8`), turning THP off
took tracing from 114–134 s to about 103 s, with the same `.afv` and less than half the kernel time.
`MIMALLOC_ALLOW_THP=1` restores mimalloc's default. Hosts with THP set to `always` have not been measured.

`FIBRS_DEBUG_MEM=1` logs each stage's field sizes and resident memory.

## Differences from the original

- **Memory.** Fields are kept in memory as sparse 16³ bricks instead of a Zarr work store. Peak memory is higher
  and grows with how much of the band holds fibres: 9.0 GiB against the original's 4.6 GiB on a
  512 × 8169 × 8169 band of PHerc0826. The chunk cache is bounded by `--native-cache-mib` plus
  `--field-cache-mib`. A smaller cache lowers the peak but decodes more often.
- **Decoding.** Predictions are decoded 8 planes at a time instead of a whole chunk at a time. Arrays whose layout
  or compressor doesn't allow that are still decoded whole.
- **Work directory.** `run.json` has a `native` key, so neither tool resumes the other's work directory, and only
  the fibrs version that started a run resumes it. An interrupted family is traced again from its first seed,
  where the original restarts from its last 16-seed batch.
- **Ignored options.** `--storage` and `--field-cache-mib` are validated, and `--storage` is recorded in
  `run.json`, but neither changes anything.
- **`--device` defaults to `auto`**, where the original defaults to `cpu`. On the GPU, fibrs writes what the
  original writes with `--device cuda`. Its centrality kernels fuse some multiply-adds, so the output can differ
  from the original's `--device cpu`. Pass `--device cpu` to match that. On the test bands both give the same
  `.afv`.
- **Command line.** fibrs parses arguments as argparse does, so exit status and values match, but help and error
  text differ. A few rare inputs are read differently: numbers in non-ASCII digits, non-UTF-8 bytes in text
  options, negative numbers like `-0_0` or `"-1\n"` in some positions (rejected here, accepted by argparse), and
  `-h-` or `-h--` (help here, an error in argparse).
- **Not reproduced.** The original's batch-buffer capacity checks and its smoothing-snapshot limit
  (`--field-cache-mib`). fibrs takes no snapshots.
