# Changelog

## [Unreleased]

- **Tiled tracing.** `--tiles-yx NY NX` traces a window as NY × NX tiles in Y and X, one at a time, so the run's
  peak memory is about its densest tile's instead of the whole window's. Each tile is traced from its core plus
  `--tile-halo` native voxels (default 256) of its neighbours. Core edges and the halo are whole units of 256 native
  voxels for prediction zarrs. Each tile's fibres are cut to its core and joined across the seams by the rule the
  v2-run pipeline uses for its Z bands: cut ends of the same family at most 1.5 voxels apart, shortest first, never
  into a loop. A `--tiles-yx 1 1` run writes the untiled run's `.afv`. Without `--tiles-yx`, runs, their output and
  their `run.json` are unchanged. See [Tiled tracing](README.md#tiled-tracing).

## [0.4.0] - 2026-11-20

fibrs 0.4.0 does more of a GPU run on the GPU and spends less CPU time around tracing. Traced curves are finished
on the GPU, and GPU centrality takes less than half as long on dense data. The seed catalogue and the export's
rows are built on the `--threads` pool, idle threads trace further ahead, and finishing does less memory work.
`fibrs predict` allocates through the system allocator again, which brings its speed back to 0.3.0's. Finishing
can also decode prediction slabs on the GPU with nvCOMP, off by default and opted into with `FIBRS_NVCOMP`. None
of these changes alters the `.afv` or the predictions; nvCOMP decoding gives the same output on sound files.

### What's new

- **Fewer GPU stalls in centrality.** The GPU waits less on the host between units of tiles:
  - each stream's page-locked buffers are sized once per family, for the largest unit it runs, instead of being
    regrown mid-stage, which made the GPU finish the other stream's work first. A sparse or faint unit pins room
    only for the output bricks where its family's predictions are strong enough to give centrality;
  - each unit's seeds come back as a list of their own, joined once at the end, instead of being appended to one
    list that kept growing;
  - a unit's inputs are built while the GPU still runs the units before it, and its centrality is installed on
    the `--threads` pool once the next unit on its stream is queued.

  The centrality and seeds are unchanged. On a dense PHerc0358 window on an RTX 5090, V and H centrality take
  19.1 s together instead of 20.9 s (mean of two runs each), about 1% of the whole run.

- **Faster GPU centrality and seed tests.** Before the float64 work, the GPU now lists the voxels that need it:
  those with presence enough for centrality, then those that pass the seed test's cheap checks. The float64 work
  then runs on full warps instead of mostly idle ones. The centrality and seeds are unchanged. On a dense PHerc0358
  window on an RTX 5090, the V centrality kernels take 1.3 s instead of 8.3 s, and V and H centrality take 7.0 s
  together instead of 15.7 s (mean of two runs each).

- **Trace finishing on the GPU.** When a run uses the GPU, finishing (cutting each traced curve where the
  predictions disagree, then smoothing, resampling and re-checking each piece) runs there too. It gives the CPU's
  pieces bit for bit, so the `.afv` is unchanged, whatever rounding the run started with. Both pooled predictions
  and a cache of native prediction slabs, up to 8 GiB, stay on the GPU while a family is finished. A batch of
  curves the GPU cannot finish, for example one that reads more slabs than its cache holds, is finished on the
  CPU, and the log counts those batches and the slabs cached, by where they were decoded, at the end of each
  family. If the GPU fails while finishing, `--device auto` finishes the rest of the family on the CPU and
  `--device cuda` stops the run, as in the other GPU stages. `--device cpu` finishes on the CPU as before.

- **Opt-in GPU decoding of prediction slabs with nvCOMP.** With NVIDIA's nvCOMP 6, the native slabs GPU
  finishing reads go to the GPU still compressed and are decoded there, with the same output on sound files. It is
  off by default, so slabs decode on the CPU with c-blosc's checks; opt in with `FIBRS_NVCOMP=on`, which loads
  the nvCOMP the new `fibrs[gpu-zstd]` extra installs (or the system's), or with `FIBRS_NVCOMP` set to a
  `libnvcomp.so.6` path. GPU slab decoding trusts the prediction files: on a corrupt chunk nvCOMP's behaviour is
  undefined, so it may give wrong output or a CUDA fault where the CPU decoder would report the corruption.
  Installing the extra alone does not turn it on. nvCOMP is never bundled: it is NVIDIA's own PyPI
  package, `nvidia-libnvcomp-cu12`, under NVIDIA's licence, and the log says once whether slabs decode on the GPU
  or why they decode on the CPU. A slab nvCOMP cannot decode as Blosc stored it is decoded on the CPU, as are the
  slabs of arrays decoded a whole chunk at a time, and when the decoder's buffers leave too little GPU memory for
  the slab cache, finishing stays on the GPU and decodes slabs on the CPU.

- **Faster seed catalogue.** Each family's seed catalogue (priority split, descending-centrality argsort, sort by
  1024³ region) is built on the `--threads` pool:
  - the split and gathers keep their order;
  - the partitions NumPy's quicksort sets aside are sorted in parallel, which gives its exact order, equal keys
    included;
  - the stable sort by region is now a counting sort.

  On the dense PHerc0358 window z02304–02816, the catalogues of its 22 million V and 21 million H seeds took
  about 10 CPU-s on one thread, most of the 12 s the seed stage took. The tracing order, and so the `.afv`, is
  unchanged.

- **`fibrs predict` allocates through the system allocator again.** Under mimalloc, which 0.3.1 made the
  allocator, the predictor zeroed the large buffers it reused, copied growing vectors that glibc malloc remaps, and
  faulted fresh pages in. That doubled its kernel time, and on nine columns of a PHerc0826 band a run took 42–43 s
  instead of 0.3.0's 37–39 s, with the same forward pass. With the system allocator the same run takes 37.1–37.2 s
  again. The tracer keeps mimalloc. The predictions are unchanged.

- **Less memory work while finishing curves.** Native prediction slabs are decoded straight into their buffers
  instead of into zero-filled ones, and chunk files are read the same way. Compressed chunks get a buffer pool of
  their own. In the one pool they shared with decoded slabs, a chunk read found no buffer to reuse one time in
  five, and freed slabs were dropped instead of being kept for the next decode. On
  the dense PHerc0358 window z02304–02816 at `--threads 8`, the zero-filling took about 34 CPU-s and the page
  faults of fresh buffers about 35 CPU-s, 14% of the finishing threads' time. The `.afv` is unchanged.

- **Faster tracing on dense data.** Idle threads trace up to 256 seeds ahead of the commit point instead of 64. On
  the dense PHerc0358 window z02304–02816 at `--threads 8`, two runs of each put both families' tracing at 97 and
  103 s against 109 and 114 s, with 0.13% of seeds traced again instead of 0.04%. Peak RSS was 23.0–23.4 GiB against
  23.0–23.1 GiB, and the `.afv` is identical.

- **Faster export.** The `.afv` export makes each batch of fibres' rows (annotations with Python's float repr, point
  blocks, bounds) on the `--threads` pool while the batch before it is inserted, in order, on one thread. SQLite
  also gets a 256 MiB page cache instead of 32 MiB and holds its lock for the whole export. A batch ends at 4,096
  fibres, or at the fibre that brings it to about a million points, so only its last fibre can take it further. On
  the dense PHerc0358 window z02304–02816 the export took 16 s on one thread. About a third of that was formatting
  rows, and a fifth was SQLite reading evicted pages back from the file of over 1.5 GiB. Fibre numbering, every
  table and the first error in input order are unchanged.

### Upgrading

- 0.4.0 doesn't resume a tracer work directory that 0.3.1 started, finished or not, so a `--resume` that only
  writes the `.afv` again needs 0.3.1 too. Finish a run with the binary that started it, or start a new work
  directory. A `fibrs predict` output that 0.3.1 started, in band or villa mode, is refused on resume as well, as
  it is by any other binary; finish it with 0.3.1 or write a new `--out`.

## [0.3.1] - 2026-11-19

fibrs 0.3.1 traces dense data faster: threads that finishing leaves idle trace seeds ahead, and the binary
allocates through mimalloc instead of glibc malloc. The `.afv` is unchanged. On a PHerc1545 and a PHerc0826 band
it was identical to the original's with `--device cpu` and to 0.3.0's with `--device cuda`, but for the per-run
uuid. `fibrs predict` gave 0.3.0's predictions voxel for voxel on nine columns of a PHerc0826 band.

### What's new

- **Faster tracing on dense data.** Threads that finishing leaves idle now trace seeds ahead of the one being
  committed. A seed traced ahead is traced again if an earlier seed marked the region it read before it was
  committed, so the `.afv` is the same as before at every `--threads`. Measured with glibc malloc, before mimalloc
  became the allocator: on a dense 512 × 2048 × 2048 crop of PHerc0358 at `--threads 8`, a run took 41–43 s instead of 47–55 s, with the same 2.6 GiB peak.
  On whole bands, also with glibc malloc, PHerc1545 took 58 s at `--threads 4` and 43 s at `--threads 8` instead of 0.3.0's 77 s and 46 s,
  and PHerc0826 took 153 s and 106 s instead of 174 s and 119 s, with peaks at most 0.3 GiB higher.
- **Tracing can run one thread beyond `--threads`.** One more thread commits the seeds in order. On that crop,
  tracing kept 4.2–4.5 cores busy at `--threads 4` and 7.7 at `--threads 8`, where 0.3.0 kept 2.9 and 4.0 busy.
  Size the cores you give a run with that in mind.
- **mimalloc is the allocator.** The binary now allocates through mimalloc 3.3.2 instead of the system allocator.
  On dense windows, glibc malloc maps and unmaps large blocks while holding the kernel's `mmap_lock`, which stalls
  the page faults of every tracing thread. On the dense PHerc0358 window z02304–02816 at `--threads 8`, 0.3.0 traced
  both families in 682 s with glibc malloc and in 200 s and 274 s with mimalloc, and the whole run took 500–640 s
  instead of 1,084 s. Peak memory is unchanged: 9.0 GiB on a PHerc0826 band and 5.6 GiB on a PHerc1545 band at
  `--threads 4`. Output is unchanged. `cargo build --release --no-default-features` builds with the system
  allocator.

### Upgrading

- 0.3.1 doesn't resume a tracer work directory that 0.3.0 started, finished or not, so a `--resume` that only
  writes the `.afv` again needs 0.3.0 too. Finish a run with the binary that started it, or start a new work
  directory. A `fibrs predict` output that 0.3.0 started, in band or villa mode, is refused on resume as well, as
  it is by any other binary; finish it with 0.3.0 or write a new `--out`.

### Documentation

- **Running the CLI archive's binary.** The README's Install section says how the archive's binary finds numpy's
  OpenBLAS, and how to point it there with `FIBRS_LAPACK` when it runs outside a numpy environment.

## [0.3.0] - 2026-11-19

fibrs 0.3.0 runs the tracer's direction field and centrality on an NVIDIA GPU when one is usable, and on the CPU
otherwise. The GPU direction field is the CPU's bit for bit. With both on the GPU, the `.afv` of a PHerc1545 band
matched the CPU's in every fibre, point block, spatial-index row and metadata value but the per-run uuid. With
centrality alone on the GPU, a PHerc0826 band matched too. `fibrs predict` can now record the CT's voxel size.

### What's new

- **Centrality on the GPU.** `--device` now defaults to `auto`: centrality and the ridge seeds run on an NVIDIA GPU
  of compute capability 8.0 or later when one is usable (Linux x86-64; a driver for CUDA 12.0 or newer is enough, no
  CUDA toolkit), and on the CPU otherwise, with the reason in the log. `--device cpu` keeps them on the CPU,
  `--device cuda` fails instead of falling back, and `--cuda-device N` picks the GPU (by default the first usable
  one); `--device cuda` used to be rejected. A run keeps the rounding it started with when resumed, and one started
  on the GPU can resume or finish on the CPU, which then computes centrality as the GPU does. The kernels are
  written in Rust with cuda-oxide and ship inside the binary as PTX. On a PHerc1545 and a PHerc0826 band the `.afv`
  was identical on the GPU and the CPU, and identical to the original's with either of its devices. With
  `--threads 8`, centrality took 1.8 s instead of 8.1 s on the PHerc1545 band and 5.0 s instead of 20.4 s on the
  PHerc0826 band. Tracing still runs on the CPU (see the README's "Speed").
- **Direction field on the GPU.** The direction field now runs on the GPU as well, under the same `--device` choice,
  and its bytes are the CPU's: the GPU certifies each voxel's eigenvector against an error bound, and the CPU solves
  the voxels it cannot certify with the same `dsyevd`, which the log line counts. If the GPU fails mid-run, the CPU
  computes the direction field instead, unless `--device cuda` was given.
- **`fibrs predict --voxel-size-um`** sets the output's `voxel_size_um` attribute, which was always 9.362, the
  PHerc0826 CT's voxel size, as in the reference. The default is still 9.362, so pass it for a CT with another voxel
  size (8.64 for example). Predictions are unchanged. A resume with a different value is refused.

### Upgrading

- 0.3.0 doesn't resume a tracer work directory that 0.2.0 started, finished or not, so a `--resume` that only
  writes the `.afv` again needs 0.2.0 too; otherwise start a new work directory. A `fibrs predict` band output that
  0.2.0 started is refused on resume as well, as it is by any other binary. Predictions that 0.2.0 finished are
  traced by 0.3.0 as before: the tracer takes the voxel size from `--voxel-size-um`, not from the prediction's
  attributes.
- `--device` now defaults to `auto`. Pass `--device cpu` to keep the direction field and centrality on the CPU, or to
  match the original's default `--device cpu` output, whose centrality can round differently from the GPU's.

### Fixes

- `fibrs predict` reads the CT through the path its overlap check resolved. It used to open the CT again by the path
  as given, so a link on that path changed during a run could make it read a different store from the one checked. A
  test repoints such a link mid-run and checks that every tile still reads the CT that was checked.

### Documentation

- The README's "Same output" section now gives the PHerc0826 band's counts as measured: 149,680 fibres and 161,787
  point blocks and spatial-index rows. It said 149,780 and 161,887.
- The README's tracer speed range is now 22×–47× on wall time and about 22×–38× on CPU time, from the three
  uninterrupted runs on bands of PHerc0826, PHerc1545 and PHerc0358. Two runs whose original was interrupted and
  resumed, one of them on a band of PHerc0813, are listed with estimated totals. The original's time per fibre
  differs between bands and between runs; on runs 4 and 5 fibrs's differed less.
- The README's dense PHerc0358 box prediction rows are now from fibrs `42c4710`, which has the 0.2.0 release's
  prediction code: 85.3 s in band mode (1.94×) and 84.5 s with `--mode villa` (1.96×), with prediction values
  identical to those of the pre-release build timed before.
- Correction to the 0.2.0 notes: two tracer results there were not measured with the tagged release. The 50–52 s
  run on the 512 × 2048 × 2048 PHerc0358 crop used a pre-release build of 0.2.0, and the 332 s run on the PHerc0358
  band, which is also the tracing part of the 739 s end-to-end row, used `42c4710`, which lacks 0.2.0's chunk-cache
  change.

## [0.2.0] - 2026-11-19

fibrs 0.2.0 adds `fibrs predict`. It runs Qual's [AFV fibre model V2](https://huggingface.co/Qualzz20/afv_fiber_9um_v2)
over a CT volume on an NVIDIA GPU, natively in the `fibrs` binary, with no PyTorch. It writes the probability zarr that
`fibrs` and AVF_SPLINE_GENERATOR trace. The tracer also needs less memory.

### What's new

- **`fibrs predict`** writes the same zarr v2 group as villa's `python -m lasagna.qualzz_inference`:
  - `semantic` is 2 × Z × Y × X uint8, holding the V and H fibre probabilities.
  - `tracing` is the same at half resolution.

  The network, softmax, blend and finalise run on the GPU through [volkit](https://github.com/praveenperera/volkit).
  The weights are read from the model's `checkpoint_final.pth` directly.
- **Two modes:**
  - `--mode villa` computes villa's tiles over the whole array. A 128³ output chunk that no stored input chunk
    overlaps is not written, as in villa `97fb9336c`; villa main writes every chunk.
  - Band mode, the default, computes a resumable z band (`--z0`, `--z1`) column by column.

  Band mode can be capped with `--max-minutes` and resumed by running the same command again.
- **`--dry-run`** plans a run and reads its tiles without a GPU. It works on macOS too.
- **Provenance.** Every output records how it was made: the settings, the checkpoint's sha256, the GPU model, and
  the cuDNN and cuBLAS versions. A resume with different settings is refused and names what changed.
- **Tracing holds less in memory**, with the same `.afv`. Presence is computed from the two prediction fields as
  it is read instead of being stored, centrality is written tile batch by tile batch, coverage takes two bytes a
  voxel instead of three, seeds are kept as a packed list, and fields are stored in 16³ bricks instead of 32³. On
  Linux, allocations of 1 MiB or more get their own mapping, so the chunk cache's evicted chunks go back to the
  system. On a 1024 × 8169 × 8169 window of PHerc0826 the peak fell from 17.4 GiB to 13.3 GiB, measured before
  predictions were decoded 8 planes at a time. On a 512-plane band it is 9.0 GiB, where 0.1.0 peaked at
  8.9–10.8 GiB.
- **Tracing is faster on dense bands**, with the same `.afv`:
  - The native predictions are decoded 8 planes at a time instead of a whole chunk at a time when the array's
    chunks allow it. If one chunk of such an array still has to be decoded whole, its 8-plane slabs are cached one
    at a time as they are copied, instead of all being copied before any is cached.
  - Decoded predictions and the direction field's tile buffers are reused from a pool.
  - Each field's bricks are cut from large mappings it owns.
  - The direction field's Gaussian filters sum a whole row of voxels at a time, in the same order as before.

  Dense bands no longer stall on the kernel's memory-map lock. On a 512 × 2048 × 2048 crop of PHerc0358's densest
  area at `--threads 8`, a run takes 50–52 s with a 2.6 GiB peak, where 0.1.0 took 104 s with a 3.5 GiB peak on the
  same machine. Pre-release builds of 0.2.0 stalled on dense data; 0.1.0 was not affected.

### Install

```sh
pip install "fibrs[predict]"
```

The `predict` extra adds NVIDIA's cuDNN 9 and cuBLAS 12 pip packages, and the binary finds them in its own Python
environment. If cuDNN 9 and cuBLAS 12 are installed system-wide, plain `pip install fibrs` is enough. The binary needs
only the NVIDIA driver: no CUDA toolkit and no PyTorch.

Download the model snapshot (`model.py`, `plans.json`, `dataset.json`, `checkpoint_final.pth`) from
[Qualzz20/afv_fiber_9um_v2](https://huggingface.co/Qualzz20/afv_fiber_9um_v2), then:

```sh
fibrs predict --mode villa --ct scroll.zarr/0 --model-dir afv_fiber_9um_v2 --out pred.zarr
```

### Precision

By default `fibrs predict` keeps parity with villa's `lasagna.qualzz_inference` on PyTorch.

- **`--precision fp32` (default):** float32 tensors with TF32 tensor-core convolutions. This is the precision
  PyTorch uses for the reference by default.
- **`--precision bf16` (opt-in):** faster, with small output differences. It is never the default and will never
  be used for published fibre sets. It was measured against fp32 on 512-plane bands of PHerc0826 and PHerc0813,
  traced with `fibrs`:
  - the traced fibre length agreed to about 99% (99.2–99.4%, V and H fibres);
  - 97.0–97.8% of fibres matched;
  - fibre counts were within 0.3%;
  - the 99.9th percentile of the foreground probability differences was 11 (`semantic`) and 9 (`tracing`) of
    255 on PHerc0826, and 8 and 7 on PHerc0813.

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

On a 512 × 2048 × 2048 box of PHerc0358 (z 2304–2816, y 2176–4224, x 2944–4992) with no all-zero tile, one run
of each at fp32 on the same GPU took 165.6 s for villa, 85.4 s for fibrs in band mode (1.94×) and 84.8 s with
`--mode villa` (1.95×). At fp32 the measured speedup is 1.68×–1.95×; the lowest is the single PHerc0826 run in
band mode behind the end-to-end row under "Why the speed-up varies" (45.4 s and 27.0 s).

fibrs, a pre-release build of 0.2.0 (`e9adcc9`), ran with cuDNN 9.27.0 and cuBLAS 13.4.1.3. villa ran PyTorch
2.14.1 (CUDA 13.0) with cuDNN 9.24, at `0bddf2757`.

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

### Why the speed-up varies

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
- **Tracing varies with the run.** The tracer ratio varies with the run and with the machine's load, so it is
  given as the lowest to highest measured. Pre-release builds of 0.2.0 traced dense data slower than 0.1.0;
  0.2.0 is faster than 0.1.0 there (the last row).

| Stage | Scroll | Region (z / y / x) | Tiles run | Original | fibrs | Ratio | Precision | Measured as |
|---|---|---|---|---|---|---|---|---|
| Tracing | PHerc0826 | z 12032–12544, whole plane | 15.2% | 11,840 s | 191 s | 62× | CPU | one run each (run 1; fibrs built before 0.1.0) |
| Tracing | PHerc0826 | z 12032–12544, whole plane | 15.2% | 4,686 s | 172 s | 27× | CPU | one run each (run 2; fibrs 0.1.0) |
| Tracing | PHerc1545 | z 1280–1792, whole plane | 8.4% | 1,621 s | 73.4 s | 22× | CPU | one run each (run 3; fibrs 0.1.0) |
| Tracing | PHerc0358 | z 2304–2816, whole plane | 33.2% | 15,632 s | 332 s | 47× | CPU | one run each, not at the same time (fibrs 0.2.0) |
| Prediction, `--mode villa` | PHerc0826 | z 12032–12544 / y 4096–5120 / x 1536–2560 | 100% | 45.37 s | 26.77 s | 1.72× (1.66–1.78×) | fp32 | ABBA, 6 pairs: medians, mean ratio with 95% CI |
| Prediction, `--mode villa` | PHerc0826 | same crop | 100% | 45.35 s | 14.06 s | 3.19× (3.10–3.27×) | bf16 (opt-in) | ABBA, 6 pairs: medians, mean ratio with 95% CI |
| Prediction, band mode | PHerc0358 | z 2304–2816 / y 2176–4224 / x 2944–4992 | 100% | 165.6 s | 85.4 s | 1.94× | fp32 | one run each; fibrs `e9adcc9` |
| Prediction, `--mode villa` | PHerc0358 | same box | 100% | 165.6 s | 84.8 s | 1.95× | fp32 | one run each, against the same villa run; fibrs `e9adcc9` |
| End to end, prediction in band mode | PHerc0826 | z 12032–12544 / y 4096–5120 / x 1536–2560 | 100% | 387.5 s | 41.7 s | 9.3× | fp32 | one run of each stage |
| End to end, prediction in band mode | PHerc0358 | z 2304–2816, whole plane | 33.2% | 17,863 s, prediction estimated | 739 s | 24×, estimated | fp32 | villa's prediction time extrapolated from the box above; the rest one run of each stage, fibrs `e9adcc9` predicting; both tracers read fibrs's prediction |
| Tracing, fibrs 0.1.0 → 0.2.0 | PHerc0358 | z 1792–2304 / y 2176–4224 / x 2944–4992 | 100% | 104 s (0.1.0) | 50–52 s (0.2.0) | 2.0–2.1× | CPU | back to back on one machine, `--threads 8`; not in the tracing range |

"Original" is `avf-spline-generator` 0.2.0 for tracing and villa's `lasagna.qualzz_inference` (main `0bddf2757`,
PyTorch fp32) for prediction; end to end is the two stages added, on the PHerc0826 crop from a single run of each
(prediction 45.4 s and 27.0 s, `fibrs predict` in band mode, not the ABBA medians). On the PHerc0358 band villa's
prediction was not run: its 2,231 s is the box run's 5.4 s outside inference plus its 160.2 s of inference scaled
from the box's 2,904 tiles to the band's 40,344, as villa main runs every tile. fibrs's prediction (407.2 s) and
both tracers (15,632 s and 332 s) were measured. Both tracers read the same saved fibrs prediction of the band, and
each stage was timed in its own run. Tracing ran with `--threads 4` for both tools unless the row says otherwise.

### Known limitations

- **Platform:** `fibrs predict` runs on Linux x86-64 with an NVIDIA GPU of compute capability 8.0 or later (RTX 30
  series, A100 and newer). On macOS and other platforms it plans runs (`--dry-run`) but doesn't run the network.
- **cuBLAS version:** if a CUDA 13 toolkit's unversioned `libcublas.so` is on the loader's path, the engine loads
  it instead of cuBLAS 12. The run still works and warns. To use cuBLAS 12, put a directory of
  `libcublas.so -> libcublas.so.12` and `libcudnn.so -> libcudnn.so.9` symlinks first on `LD_LIBRARY_PATH`.
- **Resume:** a resume recognises the CT by its metadata and the names and sizes of its chunk files, not their
  contents. After changing the CT, use a fresh `--out`. Only one run at a time may use an output.
- **Column memory:** a column's accumulators take about 14 bytes per voxel on the GPU, and at most 2^31 − 1 voxels
  per column. A run that won't fit stops before it starts and says what would: a smaller `--column`, or a
  shorter band.
- **GPUs and cuDNN:** outputs from different GPU models or cuDNN versions can differ slightly, as the model's own
  outputs do under PyTorch.
- **Precision modes:** PyTorch's `fp32-ieee` and `fp16` modes aren't offered.
- **Tracer resume across versions:** a tracer work directory started by fibrs 0.1.x resumes only with 0.1.x.
  Finish it with that version, or start again in a new work directory.

## [0.1.1] - 2026-11-18

- `scripts/install-dropin.sh --restore` puts back the `fibrs` that the wheel installed. It used to remove it, which
  broke the `fibrs` command and `python -m fibrs` in that environment until the wheel was reinstalled. A drop-in
  binary that differs from the wheel's now keeps the wheel's as `VENV/bin/fibrs.wheel` until the restore.
- CI traces a synthetic input with each build's wheel, through the drop-in and `python -m fibrs`, and requires
  the same `.afv` as the original wrote for it (`tests/fixtures/synthetic`, `scripts/compare_afv.py`).

## [0.1.0] - 2026-11-18

The first release.

- `fibrs`, a native, multi-threaded drop-in for Qual's `avf-spline-generator` 0.2.0
  ([AVF_SPLINE_GENERATOR](https://github.com/Qualzz/AVF_SPLINE_GENERATOR) @ `bee798f`): the same flags and the
  same `.afv`, 22×–62× faster across three measured runs on 512-plane bands of PHerc0826 and PHerc1545 at
  `--threads 4` (one with a build from before 0.1.0, two with 0.1.0)
- wheels for Linux x86-64 (glibc 2.28+) and Apple silicon (macOS 11+), with the `fibrs` command and the `fibrs`
  Python module, and a CLI archive for each with `scripts/install-dropin.sh`
- `--version` prints `0.2.0`, the original's version, as a drop-in has to

[0.4.0]: https://github.com/praveenperera/fibrs/releases/tag/v0.4.0
[0.3.1]: https://github.com/praveenperera/fibrs/releases/tag/v0.3.1
[0.3.0]: https://github.com/praveenperera/fibrs/releases/tag/v0.3.0
[0.2.0]: https://github.com/praveenperera/fibrs/releases/tag/v0.2.0
[0.1.1]: https://github.com/praveenperera/fibrs/releases/tag/v0.1.1
[0.1.0]: https://github.com/praveenperera/fibrs/releases/tag/v0.1.0
