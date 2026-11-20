#!/usr/bin/env bash
# Regenerates the committed PTX in ptx/ from the cuda-oxide kernel crate in fibrs-kernels, then its source
# manifest (scripts/check-ptx.sh --write). One tier, sm_80 (PTX ISA 7.0), which the driver JITs for every GPU
# from RTX 30 / A100 up: every float64 operation in it carries an explicit rounding mode, so its output is the
# same whatever architecture it is compiled for. Adapted from volkit's scripts/build-ptx.sh.
# Needs the CUDA toolkit (libdevice, ptxas) and the pinned toolchain; fibrs itself builds from the committed PTX
# without either
#
# cargo-oxide and its codegen backend come from the cuda-rust commit fibrs-kernels/Cargo.lock pins, installed
# under a CARGO_HOME of their own (the backend cache has one slot). Codegen uses the toolchain's own llc, opt
# and llvm-link and stops when the toolchain's llc is missing; their versions and libdevice's hash go to
# ptx/tools.txt, which the manifest covers. A missing ptxas, or PTX it refuses, stops the build
#
#   scripts/build-ptx.sh
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
crate="$root/fibrs-kernels"
ptxdir="$root/ptx"
toolchain="$(sed -n 's/^channel = "\(.*\)"/\1/p' "$crate/rust-toolchain.toml")"
# the compiler is picked from Cargo.lock, so a lockfile behind Cargo.toml would build with one cuda-rust revision
# and compile with another
if ! cargo "+$toolchain" metadata --locked --format-version 1 --manifest-path "$crate/Cargo.toml" > /dev/null; then
  echo "$crate/Cargo.lock does not match Cargo.toml; update it (cargo update -p cuda-device) and rerun" >&2
  exit 1
fi
lock_hash="$(sha256sum < "$crate/Cargo.lock")"
source="$(sed -n 's|^source = "git+\(https://github.com/[^/]*/cuda-rust.git\)?rev=[^#]*#\([0-9a-f]*\)"$|\1 \2|p' "$crate/Cargo.lock" | sort -u)"
repo="${source% *}"
rev="${source#* }"
if [[ ! "$rev" =~ ^[0-9a-f]{40}$ || "$repo" == "$source" ]]; then
  echo "expected one cuda-rust repository and commit in $crate/Cargo.lock, found: $source" >&2
  exit 1
fi

# the cache volkit's and inkrs's build-ptx.sh fill: one cargo-oxide build per cuda-rust commit
export CARGO_HOME="${FIBRS_OXIDE_HOME:-$HOME/.cache/inkrs/cargo-oxide-${rev:0:12}}"
export PATH="$CARGO_HOME/bin:$PATH"
export LIBCLANG_PATH="${LIBCLANG_PATH:-/usr/lib/llvm-21/lib}"
export CUDA_OXIDE_LIBDEVICE="${CUDA_OXIDE_LIBDEVICE:-${CUDA_HOME:-/usr/local/cuda}/nvvm/libdevice/libdevice.10.bc}"
unset RUSTUP_TOOLCHAIN CARGO RUSTC

ptxas="$(realpath -m -- "${PTXAS:-$(command -v ptxas || echo "${CUDA_HOME:-/usr/local/cuda}/bin/ptxas")}")"
if [[ ! -x "$ptxas" ]]; then
  echo "no executable ptxas at $ptxas; install the CUDA toolkit or set PTXAS" >&2
  exit 1
fi

# the toolchain's own llc, never one from PATH: another llc changes the codegen
host="$(rustc "+$toolchain" -vV | sed -n 's/^host: //p')"
llc="$(rustc "+$toolchain" --print sysroot)/lib/rustlib/$host/bin/llc"
if [[ ! -x "$llc" ]]; then
  echo "no llc at $llc; run: rustup component add llvm-tools --toolchain $toolchain" >&2
  exit 1
fi
export CUDA_OXIDE_LLC="$llc"
export CUDA_OXIDE_OPT="${llc%llc}opt"
export CUDA_OXIDE_LLVM_LINK="${llc%llc}llvm-link"

stamp="$CARGO_HOME/cargo-oxide.rev"
if [[ "$(cat "$stamp" 2>/dev/null)" != "$repo $rev" ]]; then
  cargo "+$toolchain" install -j 8 --locked --force --git "$repo" --rev "$rev" cargo-oxide
  echo "$repo $rev" > "$stamp"
fi

llvm_version() {
  local version
  version="$("$1" --version | sed -n 's/^ *\(.*LLVM version .*\)$/\1/p')"
  if [[ -z "$version" ]]; then
    echo "no LLVM version from $1 --version" >&2
    return 1
  fi
  echo "$version"
}
libdevice_hash="$(sha256sum < "$CUDA_OXIDE_LIBDEVICE")"
printf 'llc: %s\nopt: %s\nllvm-link: %s\nlibdevice: %s\nptxas: %s\n' \
  "$(llvm_version "$CUDA_OXIDE_LLC")" "$(llvm_version "$CUDA_OXIDE_OPT")" "$(llvm_version "$CUDA_OXIDE_LLVM_LINK")" \
  "${libdevice_hash%% *}" "$("$ptxas" --version | tail -1)" > "$ptxdir/tools.txt"

cd "$crate"
arch=80 max_isa=7.0
out="${FIBRS_PTX_BUILD:-$root/target/ptx-build}/fibrs-kernels/sm$arch"
mkdir -p "$out/ptx"
rm -f "$out/ptx/fibrs_kernels.ptx"
# cargo-oxide only writes PTX when the crate itself recompiles
cargo "+$toolchain" clean -q --target-dir "$out/cargo" -p fibrs-kernels
CARGO_TARGET_DIR="$out/cargo" CUDA_OXIDE_PTX_DIR="$out/ptx" cargo "+$toolchain" oxide build --arch "sm_$arch"
ptx="$out/ptx/fibrs_kernels.ptx"
# cargo-oxide's own build takes no --locked (cargo flags after -- switch it to another build mode), so a build
# that rewrote the lockfile is caught here
if [[ "$(sha256sum < "$crate/Cargo.lock")" != "$lock_hash" ]]; then
  echo "the build changed $crate/Cargo.lock; check it and rerun" >&2
  exit 1
fi
if [[ ! -s "$ptx" ]]; then
  echo "sm_$arch: cargo-oxide wrote no PTX at $ptx" >&2
  exit 1
fi
if ! grep -qx "\.target sm_$arch" "$ptx"; then
  echo "sm_$arch PTX has target '$(sed -n 's/^\.target //p' "$ptx")'" >&2
  exit 1
fi
version="$(sed -n 's/^\.version \([0-9]*\.[0-9]*\)$/\1/p' "$ptx")"
if ! awk -v v="$version" -v m="$max_isa" 'BEGIN { exit !(v != "" && v + 0 <= m + 0) }'; then
  echo "sm_$arch PTX has ISA version '$version', above the $max_isa this tier ships at" >&2
  exit 1
fi
# a call left in the PTX is a body that wasn't inlined
if grep -q 'call\.uni\|^\s*call\b' "$ptx"; then
  echo "sm_$arch PTX has calls; make the callee a plain #[inline(always)] fn:" >&2
  grep -o 'call\.uni [A-Za-z0-9_]*' "$ptx" | sort -u >&2
  exit 1
fi
# the parity contract: no float64 multiply or add without an explicit rounding mode, which ptxas could fuse
if grep -qE '^\s*(mul|add|sub|fma|mad)\.f64' "$ptx"; then
  echo "sm_$arch PTX has float64 arithmetic without a rounding mode, which ptxas may contract:" >&2
  grep -nE '^\s*(mul|add|sub|fma|mad)\.f64' "$ptx" | head >&2
  exit 1
fi
# the walk's kernels hold to the CPU tracer's float32 rounding too (its float32 path lengths and point steps), so
# none of their float32 multiplies or adds may be left for ptxas to fuse
walk_f32="$(awk '/\.entry fb_walk_/ { entry = $3; sub(/\(.*/, "", entry) } /^}/ { entry = "" }
  entry != "" && /^[ \t]*(mul|add|sub|fma|mad)(\.ftz)?(\.sat)?\.f32/ { print entry ": " $0 }' "$ptx")"
if [[ -n "$walk_f32" ]]; then
  echo "sm_$arch walk kernels have float32 arithmetic without a rounding mode, which ptxas may contract:" >&2
  head <<< "$walk_f32" >&2
  exit 1
fi
# libdevice: these kernels link none, so THIRD_PARTY notices need no NVIDIA libdevice section
if grep -q '__nv_' "$ptx"; then
  echo "sm_$arch PTX links libdevice functions; add their notice before shipping:" >&2
  grep -o '__nv_[A-Za-z0-9_]*' "$ptx" | sort -u >&2
  exit 1
fi

cp "$ptx" "$ptxdir/fibrs.sm$arch.ptx"
# registers, stack and spills per kernel, for the report; PTX ptxas refuses stops the build
if ! "$ptxas" -arch=sm_$arch -v "$ptx" -o /dev/null 2> "$ptxdir/ptxas-sm$arch.txt"; then
  cat "$ptxdir/ptxas-sm$arch.txt" >&2
  echo "ptxas refuses the sm_$arch PTX" >&2
  exit 1
fi
# a stack frame or a spill is a register array or a call that stayed local; none is expected
if grep -qE '[1-9][0-9]* bytes (stack frame|spill)' "$ptxdir/ptxas-sm$arch.txt"; then
  cat "$ptxdir/ptxas-sm$arch.txt" >&2
  echo "sm_$arch kernels have a stack frame or spills" >&2
  exit 1
fi
"$root/scripts/check-ptx.sh" --write
