#!/usr/bin/env bash
# scripts/check-ptx.sh for the strips kernels (ptx/strips.*). Fails when the committed PTX is stale: ptx/sources.sha256 records the hashes of the kernel crate's sources,
# manifest, lockfile and toolchain pin, the build script, the tool versions it built with (ptx/tools.txt), the
# PTX it produced and ptxas's report on it. A source is hashed without its comments and blank lines, so editing
# a comment leaves the PTX current. Needs no CUDA toolkit or cuda-oxide
#
#   scripts/check-ptx.sh            # check
#   scripts/check-ptx.sh --write    # record the current files (build-ptx.sh does this)
set -euo pipefail

cd "$(dirname "$0")/.."
manifest=ptx/strips-sources.sha256
crate=strips-kernels
mapfile -t sources < <(find "$crate/src" -name '*.rs' | LC_ALL=C sort)
files=("${sources[@]}" "$crate/Cargo.toml" "$crate/Cargo.lock" "$crate/rust-toolchain.toml" scripts/build-strips-ptx.sh
  ptx/strips-tools.txt ptx/strips.*.ptx ptx/strips-ptxas-*.txt)

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
  echo "strips ptx is stale: run scripts/build-strips-ptx.sh" >&2
  exit 1
fi
echo "strips ptx matches its sources"
