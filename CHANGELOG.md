# Changelog

## [0.2.0] - 2026-11-19

- `fibrs predict` runs Qual's AFV fibre model V2 over a CT on an NVIDIA GPU and writes the probability zarr that
  villa's `lasagna.qualzz_inference` writes (`--mode villa`), or a resumable z band of it (band mode). It is native,
  in the `fibrs` binary on Linux x86-64, with no PyTorch: the network runs on volkit's `AfvV2`.
  `pip install "fibrs[predict]"` adds NVIDIA's cuDNN 9 and cuBLAS 12 pip packages, which the binary finds in its
  own environment; a system cuDNN and cuBLAS work too. On other platforms `fibrs predict` only plans a run
  (`--dry-run`).
- `fibrs predict --precision fp32`, the default, keeps parity with that PyTorch reference: float32 tensors with
  TF32 tensor-core convolutions. `--precision bf16` is an opt-in speed mode with small output differences,
  measured in the README.

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
