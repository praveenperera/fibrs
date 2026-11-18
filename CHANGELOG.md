# Changelog

## [Unreleased]

- `scripts/install-dropin.sh --restore` keeps the `fibrs` that the wheel installed. It used to remove it, which
  broke the `fibrs` command and `python -m fibrs` in that environment until the wheel was reinstalled.
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

[0.1.0]: https://github.com/praveenperera/fibrs/releases/tag/v0.1.0
