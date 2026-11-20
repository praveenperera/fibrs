"""fibrs from Python: the native drop-in for ``avf-spline-generator`` 0.2.0

    python -m fibrs --predictions P --output fibers.afv --work-dir work ...
    from fibrs import run; run(["--predictions", P, ...])  # returns the exit status

Takes the reference CLI's arguments unchanged (``avf_spline_generator.cli.parser``) and writes the same ``.afv``.
The native binary is found through ``FIBRS_BIN``, then ``bin/fibrs`` next to this package,
then the environment's scripts directory (where the wheel installs it) and the per-user one
(``pip install --user``), then ``PATH``; on Windows each name ends in
``.exe``. When ``FIBRS_LAPACK`` is unset, the OpenBLAS that supplies the binary's ``dsyevd`` is numpy's bundled
copy (Linux and Windows), else the ``scipy-openblas64`` package's (macOS, whose numpy wheels link Accelerate). GPU slab decoding with nvCOMP is
opt-in: with ``FIBRS_NVCOMP=on``, the binary is pointed at the ``nvidia-libnvcomp-cu12`` wheel the
``fibrs[gpu-zstd]`` extra put in this Python's environment, when there is one; unset, slabs decode on the CPU
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import site
import subprocess
import sys
import sysconfig
from pathlib import Path

__all__ = ["binary", "main", "run"]

_WINDOWS = sys.platform == "win32"
_EXE = "fibrs.exe" if _WINDOWS else "fibrs"


def binary() -> str:
    """Path of the native executable"""
    explicit = os.environ.get("FIBRS_BIN")
    if explicit:
        return explicit
    bundled = Path(__file__).resolve().parent / "bin" / _EXE
    if bundled.is_file():
        return str(bundled)
    # the wheel installs the binary as a script; a venv's python run without activating the venv has no PATH to it,
    # and neither has one whose per-user scripts directory (pip install --user) is off PATH, as it is by default
    # on Windows
    for scripts in (d / _EXE for d in _script_dirs()):
        if scripts.is_file():
            return str(scripts)
    found = shutil.which("fibrs")
    if found:
        return found
    raise FileNotFoundError("fibrs not found; set FIBRS_BIN or put it on PATH")


def _script_dirs() -> list[Path]:
    """The scripts directories that may hold the wheel's binary: this interpreter's default one, then its per-user
    one when user site-packages are enabled (never in a venv)"""
    dirs = [Path(sysconfig.get_path("scripts"))]
    if site.ENABLE_USER_SITE:
        # get_preferred_scheme came in 3.10; before it, the user scheme is named after os.name
        preferred = getattr(sysconfig, "get_preferred_scheme", None)
        user = preferred("user") if preferred else f"{os.name}_user"
        if user in sysconfig.get_scheme_names():
            dirs.append(Path(sysconfig.get_path("scripts", user)))
    return dirs


def _environment() -> dict[str, str]:
    env = dict(os.environ)
    # a library not found here the binary looks for next to its own path and in $VIRTUAL_ENV, which miss an
    # environment that is not a venv
    if "FIBRS_LAPACK" not in env:
        lib = _openblas()
        if lib:
            env["FIBRS_LAPACK"] = lib
    # only FIBRS_NVCOMP=on opts in to GPU slab decoding, since the wheel can be another package's dependency; this
    # environment's wheel then stands in for the binary's own search
    if env.get("FIBRS_NVCOMP", "").lower() == "on":
        lib = _nvcomp()
        if lib:
            env["FIBRS_NVCOMP"] = lib
    return env


def _nvcomp() -> str | None:
    """The nvidia-libnvcomp-cu12 wheel's libnvcomp.so.6 (bin/nvcomp64_6.dll on Windows), found through the import
    system without loading it, so nothing CUDA starts in this process"""
    try:
        spec = importlib.util.find_spec("nvidia.libnvcomp")
    except (ImportError, ValueError):
        return None
    for root in (spec and spec.submodule_search_locations) or []:
        lib = Path(root) / "bin" / "nvcomp64_6.dll" if _WINDOWS else Path(root) / "lib64" / "libnvcomp.so.6"
        if lib.is_file():
            return str(lib)
    return None


def _openblas() -> str | None:
    """numpy's bundled OpenBLAS (Linux and Windows wheels), else the scipy-openblas64 package's (macOS, where numpy
    links Accelerate)"""
    try:
        import numpy
    except ImportError:
        return None
    libs = Path(numpy.__file__).resolve().parent.parent / "numpy.libs"
    candidates = sorted(libs.glob("libscipy_openblas64_*.dll" if _WINDOWS else "libscipy_openblas64_*.so"))
    if candidates:
        return str(candidates[0])
    try:
        import scipy_openblas64
    except ImportError:
        return None
    lib_dir = Path(scipy_openblas64.get_lib_dir())
    patterns = ("libscipy_openblas64_*.so", "libscipy_openblas64_*.dylib", "libscipy_openblas64_*.dll")
    candidates = sorted(lib for pattern in patterns for lib in lib_dir.glob(pattern))
    return str(candidates[0]) if candidates else None


def run(argv: list[str]) -> int:
    """Run the tracer with ``argv`` (no program name) and return its exit status"""
    return subprocess.run([binary(), *map(str, argv)], env=_environment(), check=False).returncode


def main() -> None:
    """Console entry: the process exits with the tracer's status"""
    raise SystemExit(run(sys.argv[1:]))
