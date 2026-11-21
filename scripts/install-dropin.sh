#!/usr/bin/env bash
# Make VENV's avf-spline-generator the native tracer, keeping the reference script beside it
#
# usage: install-dropin.sh VENV [BINARY]
#        install-dropin.sh --restore VENV
# BINARY defaults to the fibrs beside this script's directory (the release archive's layout), then to
# the one on PATH. The reference entry script moves to VENV/bin/avf-spline-generator.reference; the native
# binary is copied to VENV/bin/fibrs and avf-spline-generator becomes a symlink to it; NOTICE and licenses/ go
# to VENV/share/fibrs. When the fibrs wheel installed VENV/bin/fibrs, a different BINARY replaces it and the wheel's
# moves to VENV/bin/fibrs.wheel. --restore puts the reference entry and the wheel's fibrs back. The binary finds numpy's OpenBLAS through its own path (VENV/bin/.. /lib/python*/site-packages/numpy.libs, or the
# scipy-openblas64 package's lib/ on macOS, where numpy links Accelerate instead), so
# callers that run Path(sys.executable).parent / "avf-spline-generator" need no other change
#
# A Windows venv (run from Git Bash) keeps its executables in VENV\Scripts, each named with .exe, and its
# packages in VENV\Lib\site-packages: there the reference entry moves to avf-spline-generator.reference.exe, and
# avf-spline-generator.exe becomes a copy of fibrs.exe, since a link needs a privilege there; the binary finds
# numpy's OpenBLAS beside its own copy
set -euo pipefail

# where this venv keeps its executables and packages, and the suffix its executables carry
layout() {
  if [[ -d $venv/Scripts && ! -d $venv/bin ]]; then
    bin=$venv/Scripts exe=.exe sites=("$venv"/Lib/site-packages)
  else
    bin=$venv/bin exe='' sites=("$venv"/lib/python*/site-packages)
  fi
}

# pip lists the files it installed in the wheel's RECORD: bin/fibrs as ../../../bin/fibrs, or Scripts\fibrs.exe as
# ../../Scripts/fibrs.exe
wheel_owns_fibrs() {
  local records=()
  for site in "${sites[@]}"; do records+=("$site"/fibrs-*.dist-info/RECORD); done
  grep -qs -E "^(\.\./)+(bin|Scripts)/fibrs${exe//./\\.}," "${records[@]}"
}

# whether the entry is already the native binary: the symlink this script makes, or on Windows its copy
entry_is_fibrs() {
  if [[ -n $exe ]]; then
    [[ -e $bin/fibrs$exe ]] && cmp -s "$entry" "$bin/fibrs$exe"
  else
    [[ -L $entry && $(readlink "$entry") == fibrs ]]
  fi
}

if [[ ${1:-} == --restore ]]; then
  venv=${2:?usage: install-dropin.sh --restore VENV}
  layout
  ref=$bin/avf-spline-generator.reference$exe
  [[ -e $ref || -L $ref ]] || { echo "no $ref to restore" >&2; exit 1; }
  rm -f "$bin/avf-spline-generator$exe"
  mv "$ref" "$bin/avf-spline-generator$exe"
  # put back the wheel's binary if a different one replaced it, and remove only a copy this script installed
  if [[ -e $bin/fibrs.wheel$exe ]]; then
    mv -f "$bin/fibrs.wheel$exe" "$bin/fibrs$exe"
  elif ! wheel_owns_fibrs; then
    rm -f "$bin/fibrs$exe"
  fi
  rm -rf "$venv/share/fibrs"
  echo "restored the reference avf-spline-generator in $venv"
  exit 0
fi

venv=${1:?usage: install-dropin.sh VENV [BINARY]}
layout
crate=$(cd "$(dirname "$0")/.." && pwd)
binary=${2:-}
if [[ -z $binary ]]; then
  binary=$crate/fibrs$exe
  [[ -x $binary ]] || binary=$(command -v fibrs || true)
  [[ -n $binary ]] || { echo "no fibrs beside $0 or on PATH; pass BINARY" >&2; exit 1; }
fi
entry=$bin/avf-spline-generator$exe
reference=$bin/avf-spline-generator.reference$exe
[[ -x $binary ]] || { echo "$binary is not an executable" >&2; exit 1; }
[[ -d $bin ]] || { echo "$venv is not a venv" >&2; exit 1; }
# any one match will do; numpy.libs holds it on Linux and Windows, scipy_openblas64/lib on macOS
openblas=''
for site in "${sites[@]}"; do
  openblas=$(compgen -G "$site/numpy.libs/libscipy_openblas64_*.so" ||
    compgen -G "$site/numpy.libs/libscipy_openblas64_*.dll" ||
    compgen -G "$site/scipy_openblas64/lib/libscipy_openblas64_*.dylib" ||
    compgen -G "$site/scipy_openblas64/lib/libscipy_openblas64_*.so" ||
    compgen -G "$site/scipy_openblas64/lib/libscipy_openblas64_*.dll" || true)
  [[ -z $openblas ]] || break
done
[[ -n $openblas ]] ||
  { echo "$venv has no numpy OpenBLAS (numpy.libs/libscipy_openblas64_*); on macOS, install it with" \
      "pip install scipy-openblas64==0.3.31.188.0" >&2; exit 1; }

# keep the reference entry once, whether a script or a symlink (some installers link it); a second install
# finds the entry already pointing at the native binary and only replaces the binary
if [[ -e $reference || -L $reference ]]; then
  :
elif entry_is_fibrs; then
  echo "$entry already points at fibrs but $reference is missing" >&2
  exit 1
elif [[ -e $entry || -L $entry ]]; then
  mv "$entry" "$reference"
fi
# the wheel may already install the binary there; copying a file onto itself would fail, and a different binary
# must not lose the wheel's, which a second install finds already kept
if ! [[ -e $bin/fibrs$exe && $binary -ef $bin/fibrs$exe ]]; then
  if [[ -e $bin/fibrs$exe && ! -e $bin/fibrs.wheel$exe ]] && wheel_owns_fibrs; then
    mv "$bin/fibrs$exe" "$bin/fibrs.wheel$exe"
  fi
  install -m 0755 "$binary" "$bin/fibrs$exe"
fi
# the notices travel with the binary: the licences of the ported code and the statically linked libraries
install -d "$venv/share/fibrs/licenses"
install -m 0644 "$crate/NOTICE" "$venv/share/fibrs/"
install -m 0644 "$crate"/licenses/*.txt "$venv/share/fibrs/licenses/"
if [[ -n $exe ]]; then
  install -m 0755 "$bin/fibrs$exe" "$entry"
else
  ln -sfn fibrs "$entry"
fi
echo "$entry -> fibrs ($("$entry" --version)); reference kept as $reference"
