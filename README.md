# fibrs

A fast, native drop-in for `avf-spline-generator` 0.2.0, Qual's AFV fibre tracer
([Qualzz/AVF_SPLINE_GENERATOR](https://github.com/Qualzz/AVF_SPLINE_GENERATOR) @ `bee798f`). It takes the same
flags and writes the same `.afv`. Only the per-run uuid differs.

It also includes:

- **`fibrs predict`**: runs Qual's AFV fibre model V2 over a CT volume on an NVIDIA GPU, without PyTorch, and writes
  the prediction zarr the tracer reads.
- **`fibrs join`**: joins a scroll's Z bands into one `.afv`.

The tracing algorithm, the AFV format and the model are Qual's. fibrs makes them faster without changing what they
write.

## Fibre sets

Whole-scroll fibre sets traced with fibrs are published at
[praveenperera/afv-v2-fibers](https://huggingface.co/datasets/praveenperera/afv-v2-fibers), one `.afv` per
scroll (CC BY-NC 4.0).

## Speed

From CT to `.afv` on three 512-plane bands (PHerc1545, PHerc0826 and PHerc0358), on one RTX 5090 at fp32:

| Pipeline | Time |
|---|---|
| villa's prediction, then the original tracer | about 35,000 s |
| fibrs 0.4.0 | 800 s |

That is about 44× faster, or about 32× if you leave out the time villa spends writing its accumulators to disk.
Some of the villa and original-tracer times are estimated. A whole scroll would take fibrs about 2.6–3 hours,
against roughly 80–140 hours.

Methodology and every measurement: [docs/benchmarks.md](https://github.com/praveenperera/fibrs/blob/master/docs/benchmarks.md).

## Install

```sh
pip install fibrs              # the tracer, fibrs join and fibrs predict
pip install "fibrs[predict]"   # also installs the cuDNN and cuBLAS that fibrs predict needs
```

Install it into the environment that has numpy, usually the one that runs `avf-spline-generator`. fibrs loads
numpy's OpenBLAS at run time. You get a `fibrs` command and a `fibrs` Python module.

Standalone binaries for Linux x86-64, Apple silicon and Windows x86-64 are on the GitHub releases page. You can
also install with cargo:

```sh
cargo binstall fibrs           # prebuilt binary
cargo install fibrs --locked   # build from source (Rust 1.100+)
```

A standalone binary still needs numpy's OpenBLAS. Activate the numpy environment, or point `FIBRS_LAPACK` at the
library. See [docs/install.md](https://github.com/praveenperera/fibrs/blob/master/docs/install.md).

## Trace

```sh
fibrs --predictions P --array semantic --channel-order VH --tracing-predictions P \
  --tracing-array tracing --voxel-size-um 9.362 --storage compressed --threads 4 \
  --output fibers.afv --work-dir work
```

From Python, run `python -m fibrs ARGS`, or call `fibrs.run([...])`, which returns the exit status.

- **GPU.** fibrs uses an NVIDIA GPU when it finds one it can use: Linux or Windows x86-64, compute capability 8.0+
  and a CUDA 12 driver (no CUDA toolkit needed). Otherwise it runs on the CPU, and the log says why. Use
  `--device cpu` to stay on the CPU, `--device cuda` to fail rather than fall back, and `--cuda-device N` to pick a
  GPU.
- **Threads.** `--threads N` sets how many threads trace. The output is the same at any thread count.
- **Resume.** Run the same command again with `--resume` to continue an interrupted run.

More on the GPU, threads and memory: [docs/tracing.md](https://github.com/praveenperera/fibrs/blob/master/docs/tracing.md).

### Replace `avf-spline-generator`

To make an environment's `avf-spline-generator` run fibrs, so existing scripts work unchanged:

```sh
scripts/install-dropin.sh VENV [BINARY]   # VENV/bin/avf-spline-generator -> fibrs
scripts/install-dropin.sh --restore VENV  # put the original back
```

On Windows, run it from Git Bash.

### Tiled tracing

A dense window can need more memory than the machine has. `--tiles-yx NY NX` traces it as a grid of tiles in Y
and X, one tile at a time, so peak memory is roughly that of the densest tile:

```sh
fibrs ... --tiles-yx 2 2 --tile-halo 256
```

Fibres are joined across tile seams. Near a seam the output can differ slightly from an untiled run. See
[docs/tracing.md](https://github.com/praveenperera/fibrs/blob/master/docs/tracing.md#tiled-tracing).

### Join Z bands

A whole scroll is usually traced as overlapping Z bands. `fibrs join` trims each band's `.afv` to its core planes
and joins the fibres across band boundaries into one `.afv`:

```sh
fibrs join --manifest bands.json --output fibers-scroll.afv --stats fibers-scroll.json
```

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

`core_z` is the range of planes a band keeps and `window_z` the range it was traced from, both `[z0, z1)`.
Relative `afv` paths are resolved from the manifest's directory. See
[docs/tracing.md](https://github.com/praveenperera/fibrs/blob/master/docs/tracing.md#joining-z-bands).

## Predict

```sh
# the whole CT array, as villa's lasagna.qualzz_inference computes it
fibrs predict --mode villa --ct scroll.zarr/0 --model-dir afv_fiber_9um_v2 --out pred.zarr

# one z band, stopping after an hour; run it again to resume
fibrs predict --ct scroll.zarr/0 --z0 12032 --z1 12544 --model-dir afv_fiber_9um_v2 --out band.zarr --max-minutes 60
```

You need:

- Linux or Windows x86-64
- an NVIDIA GPU of compute capability 8.0+ (RTX 30 series, A100 or newer) and a CUDA 12+ driver
- cuDNN 9 and cuBLAS 12, from `pip install "fibrs[predict]"` or a system install
- the model snapshot from [Qualzz20/afv_fiber_9um_v2](https://huggingface.co/Qualzz20/afv_fiber_9um_v2), passed as
  `--model-dir`

`--ct` is a level-0 uint8 zarr v2 array (the `/0` directory of an OME-Zarr).

- **Modes.** `--mode villa` matches villa's output. Band mode, the default, is for splitting a scroll into bands
  that run and resume separately. It skips all-zero tiles and differs from villa near the band's edges.
- **Precision.** `--precision fp32` is the default and matches villa. `--precision bf16` is faster, with small
  differences in the output. It is never the default and is never used for published fibre sets.
- **Dry run.** `--dry-run` shows what a run would do, without a GPU.

At fp32, `fibrs predict` is 1.7–2× faster than villa on the same GPU. More in [docs/predict.md](https://github.com/praveenperera/fibrs/blob/master/docs/predict.md).

## Build

```sh
cargo build --release
FIBRS_LAPACK=/path/to/libscipy_openblas64_.so cargo test --release
```

Building needs Rust 1.100 or later, and the MSVC toolchain on Windows. The tests that trace need `FIBRS_LAPACK`.
Without it they are skipped locally and fail on CI. The wheel is built with `maturin build --release`.

## Licence

Apache-2.0, like the original. See `licenses/apache-2.0.txt`. `NOTICE` lists the ported code, the statically
linked C libraries and what `fibrs predict` takes from villa, with their licence texts in `licenses/`. The AFV
model and its weights are Qual's, and cuDNN, cuBLAS and nvCOMP are NVIDIA's. None of them is distributed here.
