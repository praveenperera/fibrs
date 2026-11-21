#!/usr/bin/env bash
# Builds the CLI archive fibrs-<version>-<target>.tar.gz from a built wheel's binary, so the wheel and the archive
# ship the same build, and writes SHA256SUMS-<target> for the wheel and the archive
#
#   scripts/release-archive.sh DIST_DIR TARGET
#
# DIST_DIR holds exactly one fibrs wheel of this version. The archive holds the binary, README.md, NOTICE, LICENSE,
# licenses/ and scripts/install-dropin.sh under fibrs-<version>-<target>/. For a Windows target the archive is
# fibrs-<version>-<target>.zip and the binary fibrs.exe; Python's zipfile reads the wheel and writes the zip there,
# since Git Bash has no zip
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
exe='' archive="$name.tar.gz"
if [[ $target == *-windows-* ]]; then exe=.exe archive="$name.zip"; fi
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/$name/scripts" "$stage/$name/licenses"
member="fibrs-$version.data/scripts/fibrs$exe"
if [[ -n $exe ]]; then
  python -I -c 'import sys, zipfile; sys.stdout.buffer.write(zipfile.ZipFile(sys.argv[1]).read(sys.argv[2]))' \
    "$wheel" "$member" > "$stage/$name/fibrs$exe"
else
  unzip -p "$wheel" "$member" > "$stage/$name/fibrs"
fi
chmod 0755 "$stage/$name/fibrs$exe"
cp "$root/README.md" "$root/NOTICE" "$root/LICENSE" "$stage/$name/"
cp "$root"/licenses/*.txt "$stage/$name/licenses/"
cp "$root/scripts/install-dropin.sh" "$stage/$name/scripts/"
if [[ -n $exe ]]; then
  # members named with / and stored in sorted order, as tar lists them
  (cd "$stage" && python -I -c '
import pathlib, sys, zipfile
root = pathlib.Path(sys.argv[2])
with zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_DEFLATED) as z:
    z.write(root, root.as_posix())
    for path in sorted(root.rglob("*")):
        z.write(path, path.as_posix())
' "$dist/$archive" "$name")
else
  tar -C "$stage" -czf "$dist/$archive" "$name"
fi

# run as fibrs, the binary prints this package's version; run as the drop-in, the original's. On Windows the
# drop-in is a copy, as install-dropin.sh makes it
test "$("$stage/$name/fibrs$exe" --version)" = "$version"
if [[ -n $exe ]]; then
  cp "$stage/$name/fibrs$exe" "$stage/avf-spline-generator$exe"
else
  ln -s "$stage/$name/fibrs" "$stage/avf-spline-generator"
fi
test "$("$stage/avf-spline-generator$exe" --version)" = 0.2.0

sum=(sha256sum)
command -v sha256sum > /dev/null || sum=(shasum -a 256)
(cd "$dist" && "${sum[@]}" "$(basename "$wheel")" "$archive" | tee "SHA256SUMS-$target")
