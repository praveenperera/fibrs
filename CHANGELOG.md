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
  - `--mode villa` computes villa's tiles over the whole array.
  - Band mode, the default, computes a resumable z band (`--z0`, `--z1`) column by column.

  Band mode can be capped with `--max-minutes` and resumed by running the same command again.
- **`--dry-run`** plans a run and reads its tiles without a GPU. It works on macOS too.
- **Provenance.** Every output records how it was made: the settings, the checkpoint's sha256, the GPU model, and
  the cuDNN and cuBLAS versions. A resume with different settings is refused and names what changed.
- **Tracing holds less in memory**, with the same `.afv`. Presence is computed from the two prediction fields as
  it is read instead of being stored, centrality is written tile batch by tile batch, coverage takes two bytes a
  voxel instead of three, seeds are kept as a packed list, and fields are stored in 16³ bricks instead of 32³.
  On Linux, allocations of 1 MiB or more get their own mapping, so the chunk cache's evicted chunks go back to
  the system. On a 1024 × 8169 × 8169 window of PHerc0826 the peak fell from 17.4 GiB to 13.3 GiB, and on a
  512-plane band from 8.9–10.8 GiB to 8.8 GiB.

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

By default `fibrs predict` keeps parity with the current best implementation, villa's `lasagna.qualzz_inference` on
PyTorch.

- **`--precision fp32` (default):** float32 tensors with TF32 tensor-core convolutions. This is the precision
  PyTorch uses for the reference by default.
- **`--precision bf16` (opt-in):** faster, with small output differences. It is never the default and is never
  used for the published fibre sets. It was measured against fp32 on 512-plane bands of PHerc0826 and PHerc0813,
  traced with `fibrs`:
  - the traced fibre length agreed to about 99% (99.2–99.4%, V and H fibres);
  - 97.0–97.8% of fibres matched;
  - fibre counts were within 0.3%;
  - the 99.9th percentile of the foreground probability differences was 11 (`semantic`) and 9 (`tracing`) of
    255 on PHerc0826, and 8 and 7 on PHerc0813.

### Speed

⟦SPEED⟧ The figures come from interleaved (ABBA) runs on the same 726-tile crop of PHerc0826, end to end, on one
RTX 5090:

| Precision | villa `lasagna.qualzz_inference` | `fibrs predict --mode villa` | Speedup |
|---|---|---|---|
| fp32 | ⟦⟧ | ⟦⟧ | ⟦⟧ |
| bf16 | ⟦⟧ | ⟦⟧ | ⟦⟧ |

### Parity

⟦PARITY⟧

### Known limitations

- **Platform:** `fibrs predict` runs on Linux x86-64 with an NVIDIA GPU of compute capability 8.0 or later (RTX 30
  series, A100 and newer). On macOS and other platforms it plans runs (`--dry-run`) but doesn't run the network.
- **cuBLAS version:** if a CUDA 13 toolkit's unversioned `libcublas.so` is on the loader's path, the engine loads
  it instead of cuBLAS 12. The run still works and warns. To use cuBLAS 12, put a directory of
  `libcublas.so -> libcublas.so.12` and `libcudnn.so -> libcudnn.so.9` symlinks first on `LD_LIBRARY_PATH`.
- **Resume:** a resume recognises the CT by its metadata and the names and sizes of its chunk files, not their
  contents. After changing the CT, use a fresh `--out`. Only one run at a time may use an output.
- **Column memory:** a column's accumulators take about 14 bytes per voxel on the GPU, and at most 2^31 − 1 voxels
  per column. A run that won't fit stops before it starts and suggests a `--column` that fits.
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
  same `.afv`, 62 times faster on a 512 × 8169 × 8169 band of PHerc0826 at `--threads 4`
- wheels for Linux x86-64 (glibc 2.28+) and Apple silicon (macOS 11+), with the `fibrs` command and the `fibrs`
  Python module, and a CLI archive for each with `scripts/install-dropin.sh`
- `--version` prints `0.2.0`, the original's version, as a drop-in has to

[0.2.0]: https://github.com/praveenperera/fibrs/releases/tag/v0.2.0
[0.1.1]: https://github.com/praveenperera/fibrs/releases/tag/v0.1.1
[0.1.0]: https://github.com/praveenperera/fibrs/releases/tag/v0.1.0
