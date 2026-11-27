# `fibrs predict` reference

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

## Requirements

- Linux x86-64 and an NVIDIA GPU of compute capability 8.0 or later (RTX 30 series, A100 and newer; not the
  T4 or the RTX 20 series), with a driver for CUDA 12 or later. The binary needs only the driver
  (`libcuda.so.1`), not a CUDA toolkit.
- cuDNN 9 (`libcudnn.so.9`) and cuBLAS 12 (`libcublas.so.12`). `pip install "fibrs[predict]"` installs them
  from NVIDIA's pip packages (`nvidia-cudnn-cu12`, `nvidia-cublas-cu12`); a system install works too.
- No PyTorch, and no Python at run time beyond what installs the wheel.

On macOS and other platforms the binary has no GPU engine: `fibrs predict --dry-run` plans a run there, and a real
run stops with an error naming what it needs. The tracer is unaffected.

## Install and use

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
lists the cuDNN and cuBLAS files it loaded. cuDNN's convolution algorithms are timed on the first run and cached
under `~/.cache` (see volkit's documentation for the cache variables). They set the rounding, so the provenance
lists them and a resume that picks others is refused: keep the cache between a run and its resume.

## Modes

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

## Precision

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

## Speed

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
[Why the speed-up varies](benchmarks.md#why-the-speed-up-varies) (45.4 s and 27.0 s). `42c4710` has the same prediction code and
dependencies as the 0.2.0 release, which adds one tracer change.

On the PHerc0826 crop, fibrs was a pre-release build of 0.2.0 (`e9adcc9`); both fibrs builds ran with cuDNN 9.27.0
and cuBLAS 13.4.1.3. villa ran PyTorch 2.14.1 (CUDA 13.0) with cuDNN 9.24, at `0bddf2757`.

## Parity

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

## Limits

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

## Tests

`cargo test` checks, without a GPU, that `fibrs predict --dry-run` feeds the network exactly the tiles, batches and
bytes that fibrs's Python band reference (`fibrs.predict` at fibrs PR #3) feeds it, on small synthetic CTs in band
and villa mode (`tests/predict.rs`, `tests/fixtures/predict`, written by `scripts/predict_fixtures.py`), and that
bad arguments are refused with the reference's messages.
