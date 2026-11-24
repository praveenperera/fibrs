#!/usr/bin/env bash
# Regenerates ptx/strips.sm80.ptx from the cuda-oxide kernel crate in strips-kernels (fibrs strips), then its source
# manifest (scripts/check-strips-ptx.sh --write). scripts/build-ptx.sh with the strips crate and outputs, minus
# fibrs-kernels' float64 rounding and no-libdevice rules: the strip model's kernels call libdevice's expf and erff.
# Needs the CUDA toolkit (libdevice, ptxas) and the pinned toolchain; fibrs builds from the committed PTX
#
#   scripts/build-strips-ptx.sh
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
crate="$root/strips-kernels"
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
  "${libdevice_hash%% *}" "$("$ptxas" --version | tail -1)" > "$ptxdir/strips-tools.txt"

cd "$crate"
arch=80 max_isa=7.0
out="${FIBRS_PTX_BUILD:-$root/target/ptx-build}/strips-kernels/sm$arch"
mkdir -p "$out/ptx"
rm -f "$out/ptx/strips_kernels.ptx"
# cargo-oxide only writes PTX when the crate itself recompiles
cargo "+$toolchain" clean -q --target-dir "$out/cargo" -p strips-kernels
CARGO_TARGET_DIR="$out/cargo" CUDA_OXIDE_PTX_DIR="$out/ptx" cargo "+$toolchain" oxide build --arch "sm_$arch"
ptx="$out/ptx/strips_kernels.ptx"
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
cp "$ptx" "$ptxdir/strips.sm$arch.ptx"
# registers, stack and spills per kernel, for the report; PTX ptxas refuses stops the build
if ! "$ptxas" -arch=sm_$arch -v "$ptx" -o /dev/null 2> "$ptxdir/strips-ptxas-sm$arch.txt"; then
  cat "$ptxdir/strips-ptxas-sm$arch.txt" >&2
  echo "ptxas refuses the sm_$arch PTX" >&2
  exit 1
fi
# a stack frame or a spill is a register array or a call that stayed local; none is expected
if grep -qE '[1-9][0-9]* bytes (stack frame|spill)' "$ptxdir/strips-ptxas-sm$arch.txt"; then
  cat "$ptxdir/strips-ptxas-sm$arch.txt" >&2
  echo "sm_$arch kernels have a stack frame or spills" >&2
  exit 1
fi
"$root/scripts/check-strips-ptx.sh" --write
