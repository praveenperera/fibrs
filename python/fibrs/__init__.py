"""fibrs from Python: the native drop-in for ``avf-spline-generator`` 0.2.0

    python -m fibrs --predictions P --output fibers.afv --work-dir work ...
    from fibrs import run; run(["--predictions", P, ...])  # returns the exit status
    python -m fibrs predict --ct CT --model-dir SNAPSHOT --out OUT.zarr ...  # fibrs.predict, the predict extra

Takes the reference CLI's arguments unchanged (``avf_spline_generator.cli.parser``) and writes the same ``.afv``.
The native binary is found through ``FIBRS_BIN``, then ``bin/fibrs`` next to this package,
then the environment's scripts directory (where the wheel installs it), then ``PATH``. When ``FIBRS_LAPACK`` is
unset, the OpenBLAS that supplies the binary's ``dsyevd`` is numpy's bundled copy (Linux), else the
``scipy-openblas64`` package's (macOS, whose numpy wheels link Accelerate)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

__all__ = ["binary", "main", "run"]


def binary() -> str:
    """Path of the native executable"""
    explicit = os.environ.get("FIBRS_BIN")
    if explicit:
        return explicit
    bundled = Path(__file__).resolve().parent / "bin" / "fibrs"
    if bundled.is_file():
        return str(bundled)
    # the wheel installs the binary as a script; a venv's python run without activating the venv has no PATH to it
    scripts = Path(sysconfig.get_path("scripts")) / "fibrs"
    if scripts.is_file():
        return str(scripts)
    found = shutil.which("fibrs")
    if found:
        return found
    raise FileNotFoundError("fibrs not found; set FIBRS_BIN or put it on PATH")


def _environment() -> dict[str, str]:
    env = dict(os.environ)
    if "FIBRS_LAPACK" in env:
        return env
    lib = _openblas()
    if lib:
        env["FIBRS_LAPACK"] = lib
    # otherwise the binary looks next to its own path and in $VIRTUAL_ENV
    return env


def _openblas() -> str | None:
    """numpy's bundled OpenBLAS (Linux wheels), else the scipy-openblas64 package's (macOS, where numpy links
    Accelerate)"""
    try:
        import numpy
    except ImportError:
        return None
    libs = Path(numpy.__file__).resolve().parent.parent / "numpy.libs"
    candidates = sorted(libs.glob("libscipy_openblas64_*.so"))
    if candidates:
        return str(candidates[0])
    try:
        import scipy_openblas64
    except ImportError:
        return None
    lib_dir = Path(scipy_openblas64.get_lib_dir())
    candidates = sorted([*lib_dir.glob("libscipy_openblas64_*.so"), *lib_dir.glob("libscipy_openblas64_*.dylib")])
    return str(candidates[0]) if candidates else None


def run(argv: list[str]) -> int:
    """Run the tracer with ``argv`` (no program name) and return its exit status"""
    return subprocess.run([binary(), *map(str, argv)], env=_environment(), check=False).returncode


def main() -> None:
    """Console entry: ``predict`` runs fibrs.predict in this interpreter, anything else the tracer; the process
    exits with its status"""
    argv = sys.argv[1:]
    if argv[:1] == ["predict"]:
        from .predict import main as predict

        raise SystemExit(predict(argv[1:]))
    raise SystemExit(run(argv))
