# fibrs

fibrs runs Qual's AFV fibre pipeline, from CT to traced fibres, as one native Rust and CUDA tool.

- `fibrs` is a drop-in for Qual's tracer, [`avf-spline-generator`](https://github.com/Qualzz/AVF_SPLINE_GENERATOR)
  0.2.0. It takes the same flags and writes the same `.afv`.
- `fibrs predict` runs Qual's [AFV fibre model V2](https://huggingface.co/Qualzz20/afv_fiber_9um_v2) on an NVIDIA GPU
  without PyTorch. It writes the probability zarr that villa's `lasagna.qualzz_inference` writes.

On three 512-plane bands, fibrs goes from CT to `.afv` in 800 s on one RTX 5090. villa's prediction followed by
Qual's tracer takes about 35,000 s, so fibrs is about 44× faster. A whole scroll takes fibrs about 3 hours, against
roughly 80–140 hours (estimated). See [Speed](#speed).

The model, the tracing algorithm and the AFV format are Qual's. fibrs changes how fast they run, not what they
produce.

## Install

```sh
pip install fibrs              # the tracer, plus fibrs predict on Linux x86-64
pip install "fibrs[predict]"   # also installs the cuDNN and cuBLAS that fibrs predict needs
cargo binstall fibrs           # prebuilt binary
cargo install fibrs --locked   # from source
```

Prebuilt binaries cover Linux x86-64 (glibc 2.28+) and Apple silicon (macOS 11+). `fibrs predict` runs only on Linux
x86-64.

The tracer loads numpy's own OpenBLAS at run time, so its direction field matches numpy bit for bit. Installed with
pip into an environment that has numpy's PyPI wheel, it needs nothing else. Otherwise:

- Run it with that environment activated, or set `FIBRS_LAPACK` to the full path of numpy's
  `libscipy_openblas64_*.so`.
- On macOS, or with a numpy that isn't the PyPI wheel, install `scipy-openblas64==0.3.31.188.0` into the environment.

`fibrs predict` needs cuDNN 9 and cuBLAS 12, from the `predict` extra or a system install. A binary installed with
cargo finds the pip copies in the active virtual environment.

## Trace

```sh
fibrs --predictions P --array semantic --channel-order VH --tracing-predictions P \
  --tracing-array tracing --voxel-size-um 9.362 --storage compressed --threads 8 \
  --output fibers.afv --work-dir work
python -m fibrs ARGS   # the same, from Python
```

- `--device auto`, the default, uses an NVIDIA GPU when one is usable and the CPU otherwise. `--device cuda` fails
  instead of falling back. On the GPU, fibrs writes what the original writes with `--device cuda`; pass
  `--device cpu` to match the original's default.
- `--threads N` sets the tracing threads. The output is the same at every thread count.
- `--resume` continues an interrupted run from its work directory.
- `--tiles-yx NY NX` traces a dense window as tiles, one at a time, to lower peak memory. The original has no such
  option.

To make an environment's `avf-spline-generator` run fibrs, so scripts that call it need no change:

```sh
scripts/install-dropin.sh VENV [BINARY]   # VENV/bin/avf-spline-generator -> fibrs
scripts/install-dropin.sh --restore VENV  # put the original back
```

More in the [tracer reference](https://github.com/praveenperera/fibrs/blob/master/docs/tracer.md).

## Predict

Download the model snapshot (`model.py`, `plans.json`, `dataset.json`, `checkpoint_final.pth`) from
[Qualzz20/afv_fiber_9um_v2](https://huggingface.co/Qualzz20/afv_fiber_9um_v2).

```sh
# villa's output, for a whole CT array
fibrs predict --mode villa --ct scroll.zarr/0 --model-dir afv_fiber_9um_v2 --out pred.zarr

# one z band, resumable
fibrs predict --ct scroll.zarr/0 --z0 12032 --z1 12544 --model-dir afv_fiber_9um_v2 --out band.zarr --max-minutes 60
```

- `--mode villa` computes villa's output over the whole array.
- Band mode, the default, runs one z band and skips tiles with no CT data. Run the same command again to resume a
  capped or interrupted run.
- `fibrs predict` needs an NVIDIA GPU of compute capability 8.0 or later (RTX 30 series, A100 and newer) and a driver
  for CUDA 12 or later. It needs no CUDA toolkit and no PyTorch.
- `--precision fp32`, the default, is the precision villa runs at. `--precision bf16` is faster, with small
  differences in the output.

More in the [`fibrs predict` reference](https://github.com/praveenperera/fibrs/blob/master/docs/predict.md).

## Same output

- The tracer writes the same `.afv` as Qual's on 512-plane bands of PHerc1545 and PHerc0826, on the GPU and on the
  CPU: every fibre row, point block and spatial-index row, and every metadata value except the per-run uuid. CI
  checks a synthetic input on Linux and macOS for every build.
- `fibrs predict --mode villa` matches villa's output within 1 of 255 at the 99.9th percentile on a crop of
  PHerc0826. Two runs of villa itself differ by the same amount.
- Band mode matches the reference engine value for value on a full PHerc0826 band.

## Speed

CT to `.afv` on 512-plane bands, on one RTX 5090 with prediction at fp32:

| Band | villa + Qual's tracer | fibrs | Speed-up |
|---|---|---|---|
| PHerc1545 | about 6,614 s | 116.9 s | 56.6× |
| PHerc0826 | about 10,466 s | 240.2 s | 43.6× |
| PHerc0358 | about 17,940 s | 442.9 s | 40.5× |
| All three | about 35,020 s | 800.0 s | 43.8× |

fibrs's times are measured. villa's times and Qual's PHerc0358 time are estimated; Qual's other two are measured.
Tracing alone is 146–290× faster. A whole scroll takes fibrs an estimated 156–181 minutes, against 81–137 hours.

The method and the earlier figures are in
[benchmarks](https://github.com/praveenperera/fibrs/blob/master/docs/benchmarks.md).

## Differences from the original

- The tracer uses more memory: 6.6–14.5 GiB on the bands above, against 4.5–4.6 GiB.
- `--device` defaults to `auto`, where the original defaults to `cpu`.
- Neither tool resumes the other's work directory.
- Exit status matches the original's, but help text and usage-error wording differ.

## Build

```sh
cargo build --release
FIBRS_LAPACK=/path/to/libscipy_openblas64_.so cargo test --release
```

It builds on Linux and macOS with Rust 1.100 or later.

## Licence

Apache-2.0, like the original. `NOTICE` lists the ported code and the bundled libraries, and their licences are in
`licenses/`. The AFV model and its weights are Qual's, and cuDNN, cuBLAS and nvCOMP are NVIDIA's; none of them is
distributed here.
