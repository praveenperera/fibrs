# fibrs predict details

`fibrs predict` writes the same probability zarr as villa's `python -m lasagna.qualzz_inference`:

- `semantic`: 2 × Z × Y × X, uint8 V and H fibre probabilities
- `tracing`: the same at half resolution

All-zero chunks are not written (fill value 0), so readers must treat a missing chunk as zero, as zarr does.

The model, its weights and its output are Qual's. The tiling, blending and quantisation follow villa's lasagna
inference. The network runs natively on the GPU through [volkit](https://github.com/praveenperera/volkit)'s
`AfvV2`, in float32 with TF32 convolutions (PyTorch's defaults), in the reference's float32 order. Several threads
read tiles ahead of the GPU, and one thread writes the output.

## Requirements

- Linux or Windows x86-64
- an NVIDIA GPU of compute capability 8.0 or later (RTX 30 series, A100 and newer, but not the T4 or RTX 20 series)
- a driver for CUDA 12 or later. Only the driver is needed (`libcuda.so.1`, or `nvcuda.dll` in System32), not a
  CUDA toolkit.
- cuDNN 9 and cuBLAS 12. `pip install "fibrs[predict]"` installs them from NVIDIA's pip packages, or use a system
  install.

No PyTorch is needed, and no Python beyond what installs the wheel.

On Windows, the GPU code builds and passes CI, but CI has no GPU, and it has not yet run on a Windows machine with
an NVIDIA GPU. On macOS and other platforms only `--dry-run` works. The tracer is unaffected.

## The model

Download the snapshot (`model.py`, `plans.json`, `dataset.json`, `checkpoint_final.pth`) from
[Qualzz20/afv_fiber_9um_v2](https://huggingface.co/Qualzz20/afv_fiber_9um_v2) and pass its directory as
`--model-dir`. The directory must not contain a `fold_0` checkpoint or be named `fold_*`, or the loader reads the
wrong files. `--expect-sha256` refuses a checkpoint with a different hash.

## Options

| Option | Meaning |
|---|---|
| `--device` | `cuda` or `cuda:N` |
| `--voxel-size-um` | the CT's voxel size, written to the output (default 9.362) |
| `--batch` | tiles sent to the GPU together, band mode only (default 4, at most 16) |
| `--readers` | tile reader threads (default 4) |
| `--column` | size of the y–x columns a run is split into (default 480) |
| `--max-minutes` | stop after this long, to resume later |
| `--dry-run` | print the tile batches a run would send to the GPU, without a GPU and without writing |

`--batch` and `--readers` change only the speed. `--column` doesn't change villa mode's output. In band mode it is
part of the run's settings.

fibrs doesn't read the voxel size from the CT's OME-Zarr metadata. The tracer doesn't read it from the predictions
either, so pass it to both.

## Resuming

Run the same command again on the same `--out` to resume a capped or interrupted run. A resume must use the same
settings, binary, GPU model and cuDNN algorithms, or it refuses the output. Only one run at a time can use an
output directory.

fibrs recognises the CT by its metadata and the names and sizes of its chunk files, not their contents. If you
change the CT, use a fresh `--out`.

cuDNN picks its convolution algorithms by timing them on the first run, and caches them under `~/.cache`
(`%LOCALAPPDATA%` on Windows). They affect rounding, so keep the cache between a run and its resume.

## Modes

- **`--mode villa`** computes the whole array as villa's `lasagna.qualzz_inference` does. When the input has
  absent chunks, it follows villa `97fb9336c`: an output chunk that no stored input chunk overlaps is not written.
  villa main writes those chunks, so the outputs differ there. The input must be a zarr v2 directory store.
- **Band mode** (the default) computes one z band `[z0, z1)` and skips tiles whose input is all zero. It is for
  splitting a scroll into bands that run and resume separately. It doesn't reproduce villa's output: villa runs the
  whole volume, so its tiles near a band's edges see planes beyond the band.

## Precision

- **`--precision fp32`** (default): float32 tensors with TF32 tensor-core convolutions, PyTorch's default and
  villa's precision.
- **`--precision bf16`** (opt-in): faster, with small differences. The network runs in bf16, while normalisation,
  softmax and blending stay in float32. Traced fibre length agrees with fp32 to about 99%. It is never the default
  and will never be used for published fibre sets.

PyTorch's `fp32-ieee` and `fp16` modes are not offered.

## Finding cuDNN and cuBLAS

**Linux.** fibrs loads the pip copies in its own environment (`site-packages/nvidia/*/lib`) before the system's.
`FIBRS_CUDA_LIBS=system` skips the pip copies.

The engine looks for the unversioned names (`libcublas.so`, `libcudnn.so`) first, so one on the loader's path wins
whatever its version. For example, a CUDA 13 toolkit's `libcublas.so` would be used. The run still works, but warns
when cuBLAS is not 12.x. To force cuBLAS 12, put a directory with `libcublas.so -> libcublas.so.12` and
`libcudnn.so -> libcudnn.so.9` symlinks first on `LD_LIBRARY_PATH`, or keep CUDA 13 off the loader's path.

**Windows.** The pip copies are in `Lib\site-packages\nvidia\*\bin`. After those, fibrs searches the executable's
directory, System32, each absolute `PATH` entry, `%CUDA_PATH%\bin\x64` and `%CUDA_PATH%\bin`, but never the current
directory. Windows DLL names carry the CUDA major version, so a newer cuBLAS is never picked up by mistake. Don't
put an `nvcuda.dll` or `cuda.dll` next to `fibrs.exe`, as it would load before the real driver.

Each run's `summary-*.json` lists the cuDNN and cuBLAS files it loaded.

## Limits

- A column's accumulators take about 14 bytes per voxel of GPU memory. A 480 × 480 column of 8,000 planes needs
  about 24 GiB. One column can hold at most 2^31 − 1 voxels (480 × 480 × 9,320 planes). A run that doesn't fit
  stops before it starts and suggests a smaller `--column` or a shorter band.
- Outputs can differ slightly between GPU models and cuDNN versions, as the model's outputs do under PyTorch.

## Tests

`cargo test` checks, without a GPU, that `--dry-run` sends the network exactly the tiles, batches and bytes that
fibrs's Python reference does, on small synthetic CTs in both modes (`tests/predict.rs`). It also checks that bad
arguments are refused with the reference's messages.

Speed and parity against villa: [benchmarks](benchmarks.md#prediction-speed).
