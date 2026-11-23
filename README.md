# fibrs

A fast drop-in for `avf-spline-generator` 0.2.0, Qual's AFV fibre tracer
([Qualzz/AVF_SPLINE_GENERATOR](https://github.com/Qualzz/AVF_SPLINE_GENERATOR) @ `bee798f`). Same flags, same
`.afv` output.

- **`fibrs predict`** runs Qual's AFV fibre model V2 on an NVIDIA GPU or an Apple silicon Mac, without PyTorch.
- **`fibrs join`** joins a scroll's Z bands into one `.afv`.

Fibre sets for whole scrolls, traced with fibrs:
[praveenperera/afv-v2-fibers](https://huggingface.co/datasets/praveenperera/afv-v2-fibers).

## Speed

CT to `.afv` on three 512-plane bands, one RTX 5090, fp32:

| Pipeline | Time |
|---|---|
| villa's prediction + original tracer | about 35,000 s |
| fibrs 0.4.0 | 800 s |

About 44× faster. A whole scroll takes fibrs an estimated 2.6–3 hours, against 80–140.
[Benchmarks](https://github.com/praveenperera/fibrs/blob/master/docs/benchmarks.md).

## Install

```sh
pip install fibrs
pip install "fibrs[predict]"   # adds cuDNN and cuBLAS for fibrs predict
```

Install into the environment that has numpy. fibrs uses numpy's OpenBLAS.

Standalone binaries are on the releases page, or:

```sh
cargo binstall fibrs
cargo install fibrs --locked
```

A standalone binary needs the numpy environment active, or `FIBRS_LAPACK` set.
[Install details](https://github.com/praveenperera/fibrs/blob/master/docs/install.md).

## Trace

```sh
fibrs --predictions P --array semantic --channel-order VH --tracing-predictions P \
  --tracing-array tracing --voxel-size-um 9.362 --storage compressed --threads 4 \
  --output fibers.afv --work-dir work
```

Or from Python: `python -m fibrs ARGS`.

- Uses an NVIDIA GPU when available, otherwise the CPU. Force one with `--device cpu` or `--device cuda`.
- Output is the same at any `--threads`.
- Resume an interrupted run with `--resume`.
- `--tiles-yx NY NX` traces a dense window as tiles, to use less memory.

To make an existing `avf-spline-generator` run fibrs:

```sh
scripts/install-dropin.sh VENV            # install
scripts/install-dropin.sh --restore VENV  # undo
```

[Tracing details](https://github.com/praveenperera/fibrs/blob/master/docs/tracing.md).

## Join

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

`core_z` is the planes a band keeps, `window_z` the planes it was traced from.

## Subset

```sh
# the fibres with a point in a box of native level-0 voxels: X0 X1 Y0 Y1 Z0 Z1
fibrs subset --input fibers-scroll.afv --output region.afv --box 2500 3524 3600 4624 8000 9024

# every fibre of 2.5 mm or more, the sheet-strip runner's default minimum
fibrs subset --input fibers-scroll.afv --output long.afv --min-length-mm 2.5
```

Writes a smaller `.afv` that villa's AFV importers read like the whole set. Fibres are kept whole and keep their
ids, and the file keeps the set's uuid, so an imported fibre has the same UUID as in a whole-scroll import.
`--box`, `--min-length-mm` and `--family V|H` combine: a fibre is kept only when it passes each one given. A box
of a 27.6 GB scroll set takes about 20 s and under 1 GB of memory.

## Predict

```sh
# whole array, matching villa's output
fibrs predict --mode villa --ct scroll.zarr/0 --model-dir afv_fiber_9um_v2 --out pred.zarr

# one z band, resumable
fibrs predict --ct scroll.zarr/0 --z0 12032 --z1 12544 --model-dir afv_fiber_9um_v2 --out band.zarr
```

Needs the model from [Qualzz20/afv_fiber_9um_v2](https://huggingface.co/Qualzz20/afv_fiber_9um_v2) and either
Linux or Windows x86-64 with an RTX 30-series or newer GPU, cuDNN 9 and cuBLAS 12, or an Apple silicon Mac on
macOS 26 or newer.
[Predict details](https://github.com/praveenperera/fibrs/blob/master/docs/predict.md).

## Build

```sh
cargo build --release
FIBRS_LAPACK=/path/to/libscipy_openblas64_.so cargo test --release
```

Rust 1.100+.

## Licence

Apache-2.0. See `NOTICE` and `licenses/`.
