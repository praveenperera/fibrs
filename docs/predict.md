# fibrs predict

Runs Qual's AFV fibre model V2 over a CT volume and writes the same prediction zarr as villa's
`lasagna.qualzz_inference`. No PyTorch needed.

The output has two arrays:

- `semantic`: V and H fibre probabilities, uint8
- `tracing`: the same at half resolution

All-zero chunks aren't written. They read as zero.

## Requirements

- Linux or Windows x86-64
- NVIDIA GPU, compute capability 8.0+ (RTX 30 series, A100 or newer)
- CUDA 12+ driver
- cuDNN 9 and cuBLAS 12: `pip install "fibrs[predict]"`, or a system install
- the model from [Qualzz20/afv_fiber_9um_v2](https://huggingface.co/Qualzz20/afv_fiber_9um_v2), passed as
  `--model-dir`

Windows hasn't been tested on a real GPU yet. On macOS, only `--dry-run` works.

## Options

| Option | Meaning |
|---|---|
| `--ct` | level-0 uint8 zarr v2 array (the `/0` of an OME-Zarr) |
| `--mode villa` | whole array, matching villa |
| `--z0`, `--z1` | band to run, in band mode (the default) |
| `--precision` | `fp32` (default) or `bf16` |
| `--device` | `cuda` or `cuda:N` |
| `--voxel-size-um` | voxel size written to the output (default 9.362) |
| `--max-minutes` | stop after this long |
| `--dry-run` | show what would run, without a GPU |
| `--batch` | tiles per GPU batch (default 4) |
| `--readers` | reader threads (default 4) |
| `--column` | column size (default 480) |

## Modes

**`--mode villa`** matches villa's output for the whole array.

**Band mode** (the default) runs one z band and skips empty tiles. Use it to split a scroll into bands. It differs
from villa near band edges.

## Precision

**fp32** is the default and matches villa.

**bf16** is about twice as fast. Traced fibre length agrees with fp32 to about 99%. It's never used for published
fibre sets.

## Resuming

Run the same command again on the same `--out`. It must use the same settings, binary and GPU model.

If you change the CT, use a new `--out`.

## Library problems

`FIBRS_CUDA_LIBS=system` ignores the pip copies of cuDNN and cuBLAS.

If a CUDA 13 toolkit is on the loader path, fibrs may pick up its cuBLAS. It still works, with a warning. Each
run's `summary-*.json` lists the libraries it loaded.

## Limits

- Very tall bands may not fit in GPU memory. fibrs checks before starting and suggests a smaller `--column`.
- Output can differ slightly between GPU models and cuDNN versions.

Speed and accuracy against villa: [benchmarks](benchmarks.md#prediction).
