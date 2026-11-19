#!/usr/bin/env bash
# Fails when the committed PTX is stale: ptx/sources.sha256 records the hashes of the kernel crate's sources,
# manifest, lockfile and toolchain pin, the build script, the tool versions it built with (ptx/tools.txt), the
# PTX it produced and ptxas's report on it. A source is hashed without its comments and blank lines, so editing
# a comment leaves the PTX current. Needs no CUDA toolkit or cuda-oxide
#
#   scripts/check-ptx.sh            # check
#   scripts/check-ptx.sh --write    # record the current files (build-ptx.sh does this)
set -euo pipefail

cd "$(dirname "$0")/.."
manifest=ptx/sources.sha256
crate=fibrs-kernels
mapfile -t sources < <(find "$crate/src" -name '*.rs' | LC_ALL=C sort)
files=("${sources[@]}" "$crate/Cargo.toml" "$crate/Cargo.lock" "$crate/rust-toolchain.toml" scripts/build-ptx.sh
  ptx/tools.txt ptx/*.ptx ptx/ptxas-*.txt)

# the PTX files on disk are exactly the tiers src/gpu/cuda.rs embeds
if ! diff <(printf '%s\n' ptx/*.ptx | LC_ALL=C sort) \
  <(grep -o 'include_str!("\.\./\.\./ptx/[^"]*")' src/gpu/cuda.rs | sed 's|include_str!("\.\./\.\./\(.*\)")|\1|' | LC_ALL=C sort); then
  echo "ptx/*.ptx and the PTX src/gpu/cuda.rs embeds differ" >&2
  exit 1
fi

# one `HASH  PATH` line per file; kernel sources hashed with `//` comments and blank lines removed (the kernels
# hold no string literal with `//` in it)
hashes() {
  local f
  for f in "${files[@]}"; do
    if [[ "$f" == *.rs ]]; then
      printf '%s  %s\n' "$(sed -E 's|[[:space:]]*//.*$||; /^[[:space:]]*$/d' "$f" | sha256sum | cut -d' ' -f1)" "$f"
    else
      sha256sum "$f"
    fi
  done
}

if [[ "${1:-}" == "--write" ]]; then
  hashes > "$manifest"
  exit 0
fi

if ! diff <(hashes) "$manifest"; then
  echo "ptx is stale: run scripts/build-ptx.sh" >&2
  exit 1
fi
echo "ptx matches its sources"
