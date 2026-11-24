# Install

## numpy's OpenBLAS

fibrs uses numpy's OpenBLAS so its results match numpy bit for bit. It looks for it in:

1. the Python environment fibrs is installed in
2. the active environment (`$VIRTUAL_ENV`)

Or set `FIBRS_LAPACK` to the library's full path:

- Linux: `site-packages/numpy.libs/libscipy_openblas64_*.so`
- Windows: `Lib\site-packages\numpy.libs\libscipy_openblas64_*.dll`

You also need `pip install scipy-openblas64==0.3.31.188.0` if:

- you're running a standalone binary on macOS
- the environment uses `--system-site-packages`
- numpy comes from a system package, like Debian's `python3-numpy`

If fibrs can't find it, it stops with "cannot find numpy's OpenBLAS". Fix the environment and rerun with
`--resume`.

## Standalone binaries

| Platform | Archive |
|---|---|
| Linux x86-64 (glibc 2.28+) | `fibrs-<version>-x86_64-unknown-linux-gnu.tar.gz` |
| Apple silicon (macOS 11+) | `fibrs-<version>-aarch64-apple-darwin.tar.gz` |
| Windows x86-64 | `fibrs-<version>-x86_64-pc-windows-msvc.zip` |

## cargo

```sh
cargo binstall fibrs           # downloads the release binary
cargo install fibrs --locked   # builds from source, needs Rust 1.100+
```

## Drop-in script

```sh
scripts/install-dropin.sh VENV [BINARY]
scripts/install-dropin.sh --restore VENV
```

This replaces `VENV`'s `avf-spline-generator` with fibrs and keeps the original as
`avf-spline-generator.reference`. On Windows, run it from Git Bash.

Run as `avf-spline-generator`, `--version` prints `0.2.0`, the version it replaces.
