# Changelog

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
  - `--mode villa` computes villa's tiles over the whole array. A 128³ output chunk with no stored input chunk
    inside it is not computed or written, as in villa `97fb9336c`; villa main computes every chunk.
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
  - The native predictions are decoded 8 planes at a time instead of a whole chunk at a time.
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
each. Each time is the whole command's wall time on the same input and output, reading the CT and writing the zarr
included, so it measures more than the network. On this crop villa ran 726 tiles and fibrs 864, as `--mode villa`
runs a tile that straddles two of its columns once for each. The speedup is the mean of the per-pair ratios:

| `fibrs predict --mode villa` precision | villa, PyTorch fp32 (median) | fibrs (median) | Speedup, mean (95% CI) |
|---|---|---|---|
| fp32 (default) | 45.37 s | 26.77 s | 1.72× (1.66–1.78×) |
| bf16 (opt-in) | 45.35 s | 14.06 s | 3.19× (3.10–3.27×) |

fibrs, a pre-release build of 0.2.0 (`e9adcc9`), ran with cuDNN 9.27.0 and cuBLAS 13.4.1.3. villa ran PyTorch
2.14.1 (CUDA 13.0) with cuDNN 9.24, at `0bddf2757`.

### Parity

- **fp32 (default)**, `fibrs predict --mode villa` against villa main's `lasagna.qualzz_inference` (`0bddf2757`)
  on the same crop:
  - the 99.9th percentile of the probability differences over the foreground is 1 of 255 in each channel;
  - the largest difference is 3 of 255;
  - the most likely class agrees at 99.97% of voxels in `semantic` and 99.973% in `tracing`;
  - both cover all 256 output chunks, and leave nothing uncovered.

  The outputs aren't identical, as expected from a different implementation. The differences are of the same
  order as two runs of villa itself with `cudnn.benchmark` on (99.9th percentile 1).
  The timed runs' outputs were scored the same way. At fp32 they meet the bar: each of fibrs's seven runs
  differs from villa's at 1.16% of voxel positions in `semantic` and 0.99% in `tracing`, by at most 3 of 255,
  with the same counts in every run. At bf16 (opt-in) they differ from villa's fp32 at 10.7% and 11.5% of
  positions, by at most 33 and 20 of 255; the 99.9th percentile over the foreground is 6 and 4, and the most
  likely class agrees at 99.55% and 99.62% of voxels.
- **Band mode** matches the reference engine value for value on a full PHerc0826 band, with 0 differences at
  fp32 and at bf16. fibrs doesn't store chunks that are entirely zero, so its output is smaller on disk; zarr
  reads a missing chunk as zero.
- **Libraries:** running the release build with the cuDNN 9.28.1 and cuBLAS 12.9 that `fibrs[predict]` installs,
  instead of a system cuDNN 9.27.0 and cuBLAS 13.4, left its outputs unchanged on three band columns at fp32 and
  bf16. With the pip cuDNN alone, the villa-mode crop at fp32 was unchanged too.
- **bf16 (opt-in)** against fp32: the figures under Precision.

### Why the speed-up varies

Each speed-up above depends on the data, so each is quoted as the range of the measured points where there is
more than one:

- **Density.** The share of a region that is papyrus rather than air changes from scroll to scroll and along
  each scroll. The table gives it as the share of 128³ tiles that `fibrs predict --dry-run` would run (band mode,
  tiles whose input isn't all zero), measured the same way for every row.
- **Prediction skips air.** villa main runs every tile; band mode skips tiles whose input is all zero, so on a
  sparse band it runs a fraction of villa's tiles and on a dense one about as many. `--mode villa` doesn't skip
  all-zero tiles, and on the PHerc0826 crop it ran 864 tiles where villa ran 726, since it runs a tile that
  straddles two of its columns once for each. Every prediction time is the whole command's wall time on the same
  input and output.
- **Tracing varies with the run.** The tracer ratio varies with the run and with the machine's load, so it is
  given as the lowest to highest measured. Pre-release builds of 0.2.0 traced dense data slower than 0.1.0;
  0.2.0 is faster than 0.1.0 there (the last row).

| Stage | Scroll | Region (z / y / x) | Tiles run | Original | fibrs | Ratio | Precision | Measured as |
|---|---|---|---|---|---|---|---|---|
| Tracing | PHerc0826 | z 12032–12544, whole plane | 15.2% | 11,840 s | 191 s | 62× | CPU | one run each (run 1; fibrs built before 0.1.0) |
| Tracing | PHerc0826 | z 12032–12544, whole plane | 15.2% | 4,686 s | 172 s | 27× | CPU | one run each (run 2; fibrs 0.1.0) |
| Tracing | PHerc1545 | z 1280–1792, whole plane | 8.4% | 1,621 s | 73.4 s | 22× | CPU | one run each (run 3; fibrs 0.1.0) |
| Prediction, `--mode villa` | PHerc0826 | z 12032–12544 / y 4096–5120 / x 1536–2560 | 100% | 45.37 s | 26.77 s | 1.72× (1.66–1.78×) | fp32 | ABBA, 6 pairs: medians, mean ratio with 95% CI |
| Prediction, `--mode villa` | PHerc0826 | same crop | 100% | 45.35 s | 14.06 s | 3.19× (3.10–3.27×) | bf16 (opt-in) | ABBA, 6 pairs: medians, mean ratio with 95% CI |
| End to end, prediction in band mode | PHerc0826 | z 12032–12544 / y 4096–5120 / x 1536–2560 | 100% | 387.5 s | 41.7 s | 9.3× | fp32 | one run of each stage |
| Tracing, fibrs 0.1.0 → 0.2.0 | PHerc0358 | z 1792–2304 / y 2176–4224 / x 2944–4992 | 100% | 104 s (0.1.0) | 50–52 s (0.2.0) | 2.0–2.1× | CPU | back to back on one machine, `--threads 8`; not in the tracing range |

"Original" is `avf-spline-generator` 0.2.0 for tracing and villa's `lasagna.qualzz_inference` (main `0bddf2757`,
PyTorch fp32) for prediction; end to end is the two stages added, from a single run of each (prediction 45.4 s and
27.0 s, `fibrs predict` in band mode, not the ABBA medians). Tracing ran with `--threads 4` for both tools unless
the row says otherwise.

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

[0.2.0]: https://github.com/praveenperera/fibrs/releases/tag/v0.2.0
[0.1.1]: https://github.com/praveenperera/fibrs/releases/tag/v0.1.1
[0.1.0]: https://github.com/praveenperera/fibrs/releases/tag/v0.1.0
