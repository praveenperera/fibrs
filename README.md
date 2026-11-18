# fibrs

A native, multi-threaded drop-in for `avf-spline-generator` 0.2.0, Qual's AFV fibre tracer
([Qualzz/AVF_SPLINE_GENERATOR](https://github.com/Qualzz/AVF_SPLINE_GENERATOR) @ `bee798f`). It takes the
same flags and writes the same `.afv`: every metadata value except the per-run uuid, and every fibre row,
point block and spatial-index row. The tracing algorithm, the AFV format and the fibre model whose
predictions it reads are Qual's; this port changes how fast the tracer runs, not what it writes.

On a 512 × 8169 × 8169 band of PHerc0826, with `--threads 4` for both tools, it took 191 s where the
original took 11,840 s: 62 times faster, with about 23 times less CPU time, and the same `.afv`. Tracing
runs on the CPU and needs no GPU.

`fibrs predict` runs the step before: Qual's AFV fibre model V2 over a CT volume, writing the probability zarr
the tracer reads, as villa's inference writes it but faster (see [Predictions](#predictions)).

## Install

```sh
pip install fibrs
```

Install it into the environment that has numpy (normally the one that runs `avf-spline-generator`). It
adds a `fibrs` command and a Python module, `fibrs`. `pip install "fibrs[predict]"` also installs what
`fibrs predict` needs (torch, zarr 2); the tracer needs none of it. Prebuilt CLI archives for Linux
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

The model, its weights and its output are Qual's, and the tiling, blending and quantisation are villa's
lasagna inference. `fibrs predict` changes how fast that runs, not what it computes. It keeps the GPU busy by
reading tiles on several threads ahead of the network, and by blending and finalising on the GPU instead of the
host. It is Python (`fibrs.predict`), not part of the native binary.

### Modes

| Mode | What it computes | Checked against |
|---|---|---|
| `--mode villa` | the whole array, as villa main's `lasagna.qualzz_inference` computes it: one tile per forward, and villa's rule for which tiles run (from the input's stored chunks) | villa main `97fb9336c`, voxel for voxel |
| band mode (the default) | a z band `[z0, z1)`, reflected at the band's own edges, 4 tiles per forward, tiles whose input is all zero skipped | batch-4 tiling of the same band, voxel for voxel |

Use `--mode villa` when you want villa's output. Band mode is for splitting a scroll into z bands that run
separately and resume: it does not reproduce villa's output, because villa runs the whole volume, forwards one tile
at a time (convolutions round slightly differently at other batch sizes) and also runs tiles of stored zeros.

### Install and use

```sh
pip install "fibrs[predict]"
```

The extra brings torch 2.14.1 to 2.15 (install a CUDA build of torch for GPU runs), zarr 2 and
`dynamic-network-architectures` 0.4.4, the version the model's repository pins; torch needs Python 3.10 or
later. Download the model snapshot (`model.py`, `plans.json`, `checkpoint_final.pth`) from
[Qualzz20/afv_fiber_9um_v2](https://huggingface.co/Qualzz20/afv_fiber_9um_v2); `--model-dir` is that directory.

```sh
# villa's output, for a whole CT array
fibrs predict --mode villa --ct scroll.zarr/0 --model-dir afv_fiber_9um_v2 --out pred.zarr

# one z band, resumable
fibrs predict --ct scroll.zarr/0 --z0 12032 --z1 12544 --model-dir afv_fiber_9um_v2 --out band.zarr --max-minutes 60

python -m fibrs.predict ARGS   # the same, from Python (or python -m fibrs predict ARGS)
```

`fibrs predict` runs `python -m fibrs.predict` with the Python beside the `fibrs` binary, or the one
`FIBRS_PYTHON` names. `--ct` is a level-0 uint8 zarr v2 array (the `/0` directory of an OME-Zarr). `--device`
defaults to `cuda`; `cpu` and, on Apple silicon, `mps` also run. In band mode, run the same command again on the
same `--out` to resume a capped (`--max-minutes`) or interrupted run at the first unfinished column.
`--expect-sha256` refuses a checkpoint with another hash.

`--readers` (tile reader threads, default 4) changes the speed, not the output. `--column` (the size of the y–x
columns a run is split into, default 480) does the same in villa mode; in band mode the 4-tile batches are formed
within each column, so the column size is part of what a band-mode output is.

### Same output

In villa mode, `fibrs predict` and villa main's `python -m lasagna.qualzz_inference` (97fb9336c, `--device cpu`)
wrote the same `semantic` and `tracing` arrays, every value, on:

- a 160 × 300 × 300 crop of the PHerc0826 CT, every chunk stored;
- a 159 × 301 × 299 crop of the same CT copied chunk for chunk, with 9 of its 18 chunks absent, so that villa's
  skip rule decides which tiles run;
- a synthetic volume whose shape is odd on every axis, with absent chunks, stored all-zero regions and input
  chunks that are not the output chunks (`tests/predict`).

The two crop checks ran on this code before it moved into fibrs, under its earlier name; the move renamed its
module and command and changed no computation, and the synthetic check passes on the fibrs wheel. On CUDA, villa mode's output has not been compared with villa main's,
because villa main's GPU run did not finish on our test host (see Speed). The check was planned to compare CUDA
output as well; it was narrowed to the CPU after those GPU runs failed, so this is an amended check.

Band mode writes the same arrays as a batch-4 tiling of the band that runs the network on every non-air tile in
order and blends on the host, on the CPU, CUDA and Apple MPS.

### Speed

On PHerc0826, z 12032–12544, y 5120–5632, all x (512 × 512 × 8169, with 561 of its 1,024 input chunks stored),
`--mode villa --device cuda` took 89–106 s on an NVIDIA RTX 5090, over six runs (89–93 s in five; the 106 s run
started with the host's load average at 26). That is the wall time of the whole command: loading the model,
reading the CT, the network and writing the zarr.

Villa main's GPU run did not finish on our test host: on the same slab and GPU, its GPU manager stopped answering
partway through each of the four attempts in which its GPU run started, so we give our time alone.

In the last run (93 s), both tools ran pinned to the same 4 CPU cores at nice 10, with `NUMPY_MADVISE_HUGEPAGE=0`.
Villa main ran with `--flush-workers 2 --accumulator-workers 2`, because with its default of 16 workers on 4 cores
its GPU helper stalled; its GPU manager ran outside the pin at nice 0. Earlier runs had only some of these
settings. In the last run, memory "full" pressure averaged 3.4% (at most 9.2%) on the host while `fibrs predict`
ran, and 4.2% (at most 8.5%) while villa main ran until it stopped.

### Limits

- The output's `voxel_size_um` attribute is always 9.362, the PHerc0826 CT's voxel size. If your CT has another
  voxel size, pass the real one to the tracer (`--voxel-size-um`), which reads its own flag, not the attribute.
- `--precision bf16` and `fp16` (autocast for the network) are faster on some GPUs but change the output;
  `fp32` (torch's defaults, which run convolutions in TF32 on CUDA) is the default and the checked mode.
- Outputs from different devices (CPU, CUDA, MPS) differ from each other slightly, as the model's own outputs do.
- Villa mode reads which chunks are stored from the input's directory listing, so the input must be a zarr v2
  directory store (`.` or `/` chunk separator).

### Tests

```sh
pip install "dist/fibrs-<version>-<platform>.whl[predict]" pytest
python -m pytest tests/predict    # CPU only, a stub network, under a minute
```

CI runs them on Linux and macOS with the CPU build of torch. `tests/predict/test_villa.py` also runs villa main's
`python -m lasagna.qualzz_inference` on the synthetic volume above and compares the arrays when
`FIBRS_PREDICT_VILLA_PYTHON` (a Python with villa's requirements) and `FIBRS_PREDICT_VILLA_SRC` (a villa checkout)
are set; otherwise it skips that test.

## Licence

Apache-2.0, like the original; the licence text is `licenses/apache-2.0.txt`. `NOTICE` lists the ported code
(AVF_SPLINE_GENERATOR, one NumPy sort routine), the statically linked C libraries and what `fibrs predict` takes
from villa (MIT: its blend ramp, and its tile rule in villa mode); their licence texts are in `licenses/`. The
AFV model and its weights are Qual's and are not distributed here.
