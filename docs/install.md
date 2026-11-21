# Install details

## numpy's OpenBLAS

fibrs bundles no OpenBLAS. It loads numpy's at run time, so its eigenvectors match numpy's bit for bit.

It looks in two places, in this order:

1. the Python environment the binary is installed in
2. the active environment (`$VIRTUAL_ENV`)

In each environment it tries numpy's own copy first, then the `scipy-openblas64` package. `FIBRS_LAPACK`
overrides the search. Set it to the library's full path, with no `*`:

- Linux: `site-packages/numpy.libs/libscipy_openblas64_*.so`
- Windows: `Lib\site-packages\numpy.libs\libscipy_openblas64_*.dll`

`pip install fibrs` into an environment with its own numpy wheel needs nothing more. Otherwise, install
`scipy-openblas64==0.3.31.188.0` into the environment when:

- you run a standalone binary on macOS, where numpy's wheels use Accelerate instead of OpenBLAS (the macOS wheel
  already depends on it)
- the environment inherits numpy (`--system-site-packages`)
- numpy comes from a system package, such as Debian's `python3-numpy`

If fibrs can't find the library, it stops with "cannot find numpy's OpenBLAS". Fix the environment, then run the
same command again with `--resume` and the same fibrs binary. A run only resumes with the version that started it.

## Standalone binaries

| Platform | Archive |
|---|---|
| Linux x86-64 (glibc 2.28+) | `fibrs-<version>-x86_64-unknown-linux-gnu.tar.gz` |
| Apple silicon (macOS 11+) | `fibrs-<version>-aarch64-apple-darwin.tar.gz` |
| Windows x86-64 | `fibrs-<version>-x86_64-pc-windows-msvc.zip` |

`fibrs.exe` needs no Visual C++ redistributable. Each archive also holds `scripts/install-dropin.sh`.

## cargo

`cargo binstall fibrs` downloads the release binary on the three platforms above. On other platforms it tries
[cargo-quickinstall](https://github.com/cargo-bins/cargo-quickinstall)'s third-party builds, then builds from
source.

`cargo install fibrs --locked` builds from crates.io with the dependency versions the release was tested with. It
needs Rust 1.100 or later, and the MSVC toolchain on Windows.

Both install only the binary, into `~/.cargo/bin` (`%USERPROFILE%\.cargo\bin` on Windows).

## The drop-in script

`scripts/install-dropin.sh VENV [BINARY]` makes `VENV`'s `avf-spline-generator` run fibrs. `BINARY` defaults to
the `fibrs` in the script's archive, then the one on `PATH`.

- **Linux and macOS.** The original entry point is kept as `VENV/bin/avf-spline-generator.reference`.
- **Windows** (run from Git Bash). `VENV\Scripts\avf-spline-generator.exe` becomes a copy of `fibrs.exe`, and the
  original is kept as `avf-spline-generator.reference.exe`. Callers that build the path themselves need the `.exe`.

The script also copies `NOTICE` and `licenses/` to `VENV/share/fibrs`. Run as `avf-spline-generator`,
`--version` prints `0.2.0`, the release it replaces. Run as `fibrs`, it prints fibrs's own version.
