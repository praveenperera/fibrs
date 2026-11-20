#!/usr/bin/env bash
# Builds the CLI archive fibrs-<version>-<target>.tar.gz from a built wheel's binary, so the wheel and the archive
# ship the same build, and writes SHA256SUMS-<target> for the wheel and the archive
#
#   scripts/release-archive.sh DIST_DIR TARGET
#
# DIST_DIR holds exactly one fibrs wheel of this version. The archive holds the binary, README.md, NOTICE, LICENSE,
# licenses/ and scripts/install-dropin.sh under fibrs-<version>-<target>/
set -euo pipefail

dist=${1:?usage: release-archive.sh DIST_DIR TARGET}
target=${2:?usage: release-archive.sh DIST_DIR TARGET}
root="$(cd "$(dirname "$0")/.." && pwd)"
dist="$(cd "$dist" && pwd)"
version="$(sed -n 's/^version = "\(.*\)"/\1/p' "$root/Cargo.toml" | head -1)"
wheels=("$dist"/fibrs-"$version"-*.whl)
[[ ${#wheels[@]} -eq 1 && -f ${wheels[0]} ]] || { echo "expected one fibrs $version wheel in $dist" >&2; exit 1; }
wheel=${wheels[0]}

name="fibrs-$version-$target"
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/$name/scripts" "$stage/$name/licenses"
unzip -p "$wheel" "fibrs-$version.data/scripts/fibrs" > "$stage/$name/fibrs"
chmod 0755 "$stage/$name/fibrs"
cp "$root/README.md" "$root/NOTICE" "$root/LICENSE" "$stage/$name/"
cp "$root"/licenses/*.txt "$stage/$name/licenses/"
cp "$root/scripts/install-dropin.sh" "$stage/$name/scripts/"
tar -C "$stage" -czf "$dist/$name.tar.gz" "$name"

# run as fibrs, the binary prints this package's version; run as the drop-in, the original's
test "$("$stage/$name/fibrs" --version)" = "$version"
ln -s "$stage/$name/fibrs" "$stage/avf-spline-generator"
test "$("$stage/avf-spline-generator" --version)" = 0.2.0

sum=(sha256sum)
command -v sha256sum > /dev/null || sum=(shasum -a 256)
(cd "$dist" && "${sum[@]}" "$(basename "$wheel")" "$name.tar.gz" | tee "SHA256SUMS-$target")
