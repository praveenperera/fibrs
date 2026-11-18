# fibrs

A native, multi-threaded drop-in for `avf-spline-generator` 0.2.0, Qual's AFV fibre tracer
([Qualzz/AVF_SPLINE_GENERATOR](https://github.com/Qualzz/AVF_SPLINE_GENERATOR) @ `bee798f`). It takes the
same flags and writes the same `.afv`: every metadata value except the per-run uuid, and every fibre row,
point block and spatial-index row. The tracing algorithm, the AFV format and the fibre model whose
predictions it reads are Qual's; this port changes how fast the tracer runs, not what it writes.

On a 512 × 8169 × 8169 band of PHerc0826, with `--threads 4` for both tools, it took 191 s where the
original took 11,840 s: 62 times faster, with about 23 times less CPU time, and the same `.afv`. Tracing
runs on the CPU and needs no GPU.

`fibrs predict` runs the step before: Qual's AFV fibre model V2 over a CT volume on an NVIDIA GPU, without
PyTorch, writing the probability zarr the tracer reads, as villa's inference writes it (see
[Predictions](#predictions)).

## Install

```sh
pip install fibrs
```

Install it into the environment that has numpy (normally the one that runs `avf-spline-generator`). It
adds a `fibrs` command and a Python module, `fibrs`. `fibrs predict` is built into the same binary on Linux
x86-64; `pip install "fibrs[predict]"` adds the cuDNN and cuBLAS it needs when the system has none (see
[Predictions](#predictions)). Prebuilt CLI archives for Linux
x86-64 (glibc 2.28+) and Apple silicon (macOS 11+) are named `fibrs-<version>-<target>.tar.gz`.

## Use

```sh
fibrs --predictions P --array semantic --channel-order VH --tracing-predictions P \
  --tracing-array tracing --voxel-size-um 9.362 --storage compressed --device cpu --threads 4 \
  --output fibers.afv --work-dir work
python -m fibrs ARGS      # the same, from Python
```

`fibrs.run([...])` runs it from Python and returns the exit status.

To make an environment's `avf-spline-generator` run the port, so scripts that call it need no change:

```sh
scripts/install-dropin.sh VENV [BINARY]   # VENV/bin/avf-spline-generator -> fibrs
scripts/install-dropin.sh --restore VENV  # put the original back
```

BINARY defaults to the `fibrs` in the archive the script came from, then to the one on `PATH`.
The original entry is kept as `avf-spline-generator.reference`, and `NOTICE` and `licenses/` are copied to
`VENV/share/fibrs`. Callers that run `Path(sys.executable).parent / "avf-spline-generator"`
need no other change.

## Build

```sh
cargo build --release
cargo test --release
```

It builds on Linux and macOS. The wheel is built with maturin from
`pyproject.toml` (`maturin build --release`; the published Linux wheel uses `--zig --compatibility
manylinux_2_28`, and the macOS wheel `MACOSX_DEPLOYMENT_TARGET=11.0`).

## Numerics

The direction field's eigenvectors come from `dsyevd` in numpy's own OpenBLAS, on float64 copies as
`numpy.linalg.eigh` computes them, so they match `eigh` bit for bit. The binary finds that library from its
own path (`bin/../lib/python*/site-packages/numpy.libs/libscipy_openblas64_*.so`), then from
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

On that PHerc0826 band (z 12032–12544), the port's `.afv` is identical to the original's: 149,780 fibres,
161,887 point blocks and 161,887 spatial-index rows, plus the metadata and schema. The 0.1.0 build was also
checked on a 512³ crop of the band: the wheel and the drop-in were installed into a fresh environment beside
the original, and the crop was traced through `avf-spline-generator` and through `python -m fibrs`. Both
outputs were identical to the original's `.afv`.

CI traces a small synthetic input (`tests/fixtures/synthetic`) with every build's wheel on Linux and macOS,
through the drop-in and through `python -m fibrs`. Both must write what the original wrote for it, compared by
`scripts/compare_afv.py`: every metadata value except the per-run uuid, the schema and every table row.

## Speed

The same band, on an Intel Core Ultra 7 270K Plus (24 cores), with `--threads 4` for both tools and other
work running on the machine (one-minute load average 17.5–18.8):

| Run | Wall | CPU time | Peak memory |
|---|---|---|---|
| `avf-spline-generator` 0.2.0 | 11,840 s (3.3 h) | about 11,940 s | 4.6 GiB |
| `fibrs` | 191 s | 516 s | 12.4 GiB |
| Speedup | **62×** | **about 23×** | |

The original uses about one core whatever `--threads` says, and the port keeps about 2.7 busy. The
original's run was interrupted by a reboot and finished with `--resume`. Its wall time is the sum of the two
parts, and the first part's CPU time is estimated from the second part's CPU-to-wall ratio (1.01). An
earlier uninterrupted run of the original on the same band took 12,728 s.

## Differences from the original

- Fields live in memory as sparse 32³ bricks instead of a Zarr work store, so peak memory is higher and
  grows with how much of the band holds fibres: 11–12 GiB against the original's 4.6 GiB on a 512 × 8169 ×
  8169 band of PHerc0826, and 18.4 GiB on a denser 512 × 7947 × 7947 band of PHerc0813. The peak varies by
  about a gigabyte between runs of the same band, with thread timing. With
  `--field-cache-mib 4096`, 4.1 GiB of that is the chunk cache; a smaller cache lowers the peak by the
  difference but decodes chunks again (at 16 MiB the PHerc0826 band took about 45% longer). Memory each stage
  frees goes back to the system before the next stage allocates. `FIBRS_DEBUG_MEM=1` logs each stage's
  field sizes and resident memory.
- The work directory holds `run.json` (with a `native` key, so neither tool resumes the other's
  directory) and one finished curve file per family. An interrupted family is traced again from its first
  seed on `--resume`, instead of from the last 16-seed batch.
- `--storage` and `--field-cache-mib` are validated, and `--storage` is recorded in `run.json` as the
  original records it, but neither selects anything. The chunk cache is bounded by `--native-cache-mib`
  plus `--field-cache-mib`.
- `--device cuda` is rejected: the original uses CUDA only for centrality, and this port computes
  centrality on the CPU.
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
`fibrs` and [AVF_SPLINE_GENERATOR](https://github.com/Qualzz/AVF_SPLINE_GENERATOR) trace.

The model, its weights and its output are Qual's, and the tiling, blending and quantisation are villa's lasagna
inference. The network runs natively on the GPU through [volkit](https://github.com/praveenperera/volkit)'s
`AfvV2`, with no PyTorch or Python: per tile, the model's own normalisation, the network in float32 with TF32
convolutions (PyTorch's defaults), the softmax and the blend, then each column's finalise, in the reference's
float32 order. Tiles are read on several threads ahead of the GPU, and one thread writes the output.

### Requirements

- Linux x86-64 and an NVIDIA GPU with a driver for CUDA 12 or later. The binary needs only the driver
  (`libcuda.so.1`), not a CUDA toolkit.
- cuDNN 9 (`libcudnn.so.9`) and cuBLAS 12 (`libcublas.so.12`). `pip install "fibrs[predict]"` installs them
  from NVIDIA's pip packages (`nvidia-cudnn-cu12`, `nvidia-cublas-cu12`); a system install works too.
- No PyTorch, and no Python at run time beyond what installs the wheel.

On macOS and other platforms the binary has no GPU engine: `fibrs predict --dry-run` plans a run there, and a real
run stops with an error naming what it needs. The tracer is unaffected.

### Install and use

```sh
pip install "fibrs[predict]"   # or: pip install fibrs, with cuDNN 9 and cuBLAS 12 installed system-wide
```

Download the model snapshot (`model.py`, `plans.json`, `checkpoint_final.pth`) from
[Qualzz20/afv_fiber_9um_v2](https://huggingface.co/Qualzz20/afv_fiber_9um_v2); `--model-dir` is that directory.

```sh
# villa's output, for a whole CT array
fibrs predict --mode villa --ct scroll.zarr/0 --model-dir afv_fiber_9um_v2 --out pred.zarr

# one z band, resumable
fibrs predict --ct scroll.zarr/0 --z0 12032 --z1 12544 --model-dir afv_fiber_9um_v2 --out band.zarr --max-minutes 60

python -m fibrs predict ARGS   # the same, from Python
```

`--ct` is a level-0 uint8 zarr v2 array (the `/0` directory of an OME-Zarr). `--device` is `cuda` or `cuda:N`.
Run the same command again on the same `--out` to resume a capped (`--max-minutes`) or interrupted run at the
first unfinished column; the resume must use the same settings, binary and GPU model, or it refuses the output.
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
lists the cuDNN and cuBLAS files it loaded. cuDNN algorithm choices are timed on the first run and cached under `~/.cache` (see volkit's
documentation for the cache variables).

### Modes

| Mode | What it computes |
|---|---|
| `--mode villa` | the whole array, as villa main's `lasagna.qualzz_inference` computes it, with villa's rule for which tiles run (from the input's stored chunks) |
| band mode (the default) | a z band `[z0, z1)`, reflected at the band's own edges, tiles whose input is all zero skipped |

Use `--mode villa` when you want villa's output. Band mode is for splitting a scroll into z bands that run
separately and resume: it does not reproduce villa's output, because villa runs the whole volume, so its tiles near
a band's edges see the planes beyond it, and it also runs tiles of stored zeros.

The network runs on one tile at a time in both modes. `--batch` (band mode, default 4, at most 16) is how many
tiles go to the GPU together, and `--readers` (tile reader threads, default 4) how many read ahead; both change
the speed, not the output. `--column` (the size of the y–x columns a run is split into, default 480) leaves villa
mode's output unchanged; in band mode it is part of the run's settings, as in the Python version.

### Precision

By default `fibrs predict` keeps parity with the current best implementation, villa's
`lasagna.qualzz_inference` on PyTorch: it runs at that reference's own precision, and bf16 is only ever opt-in.

- `--precision fp32` (default): float32 tensors with TF32 tensor-core convolutions, the precision PyTorch uses for
  the reference by default.
- `--precision bf16` (opt-in): faster, small output differences. The network and its seg head run in bf16, with
  the normalisation, softmax and blend still in float32. Against fp32, on 512-plane bands of PHerc0826 and
  PHerc0813 traced with `fibrs`, the traced fibre length agreed to about 99% (99.2–99.4%, V and H fibres),
  97.0–97.8% of fibres matched, and the fibre counts were within 0.3%; the 99.9th percentile of the probability
  differences over the foreground was 11 and 9 of 255 on PHerc0826 and 8 and 7 on PHerc0813.

PyTorch's `fp32-ieee` and `fp16` modes are not offered.

### Limits

- The output's `voxel_size_um` attribute is always 9.362, the PHerc0826 CT's voxel size. If your CT has another
  voxel size, pass the real one to the tracer (`--voxel-size-um`), which reads its own flag, not the attribute.
- A column's accumulators span the whole band height on the GPU, about 14 bytes per voxel (a 480 × 480 column of
  8,000 planes needs about 24 GiB), and the engine takes at most 2^31 − 1 voxels in one column (a 480 × 480 column
  of up to 9,320 planes). A run whose pending columns don't fit, in the free GPU memory or in that limit, stops
  before it starts and names a `--column` that does; in villa mode the column size leaves the output unchanged.
- Villa mode reads which chunks are stored from the input's directory listing, so the input must be a zarr v2
  directory store (`.` or `/` chunk separator).
- Outputs from different GPU models, or with different cuDNN versions, can differ slightly, as the model's own
  outputs do under PyTorch.

### Tests

`cargo test` checks, without a GPU, that `fibrs predict --dry-run` feeds the network exactly the tiles, batches and
bytes that fibrs's Python band reference (`fibrs.predict` at fibrs PR #3) feeds it, on small synthetic CTs in band
and villa mode (`tests/predict.rs`, `tests/fixtures/predict`, written by `scripts/predict_fixtures.py`), and that
bad arguments are refused with the reference's messages.

## Licence

Apache-2.0, like the original; the licence text is `licenses/apache-2.0.txt`. `NOTICE` lists the ported code
(AVF_SPLINE_GENERATOR, one NumPy sort routine), the statically linked C libraries and what `fibrs predict` takes
from villa (MIT: its tile rule in villa mode); their licence texts are in `licenses/`. The AFV model and its
weights are Qual's, and cuDNN and cuBLAS are NVIDIA's; none of them is distributed here.
