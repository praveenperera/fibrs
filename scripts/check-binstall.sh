#!/usr/bin/env bash
# Renders Cargo.toml's [package.metadata.binstall] for each target a release attaches a CLI archive for, filling in
# the template as cargo-binstall does, and prints each target's download URL and binary path. Each URL must be the
# v<version> GitHub release's asset, since package.yml's release-check requires the tag v<version> and the release
# carries its archives. Given a directory of the CLI archives package.yml builds (scripts/release-archive.sh), it
# also checks that each target's archive is there under the rendered name and holds the binary at the rendered
# path, and that the directory holds no other CLI archive. With --download, it downloads each rendered URL instead,
# as cargo-binstall will, and checks what it gets the same way
#
#   scripts/check-binstall.sh [ARCHIVE_DIR | --download]
#
# Needs cargo, jq, tar and unzip, and curl for --download
set -euo pipefail

# the targets package.yml builds; cargo-binstall falls back to a source build anywhere else
targets=(x86_64-unknown-linux-gnu aarch64-apple-darwin x86_64-pc-windows-msvc)

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dir="${1:-}"
download=false
if [ "$dir" = --download ]; then
  download=true
  dir="$(mktemp -d)"
  trap 'rm -rf "$dir"' EXIT
fi

package="$(cargo metadata --no-deps --format-version 1 --locked --manifest-path "$root/Cargo.toml" \
  | jq -c '.packages[] | select(.name == "fibrs")')"
name="$(jq -r .name <<< "$package")"
version="$(jq -r .version <<< "$package")"
repo="$(jq -r '.repository | sub("\\.git$"; "")' <<< "$package")"
bin="$(jq -r '[.targets[] | select(.kind[] == "bin") | .name] | if length == 1 then .[0] else error("expected one binary") end' \
  <<< "$package")"

# a target's override of a field, else the field itself
field() {
  jq -r --arg target "$1" --arg key "$2" \
    '.metadata.binstall | (.overrides[$target][$key] // .[$key]) // error("no \($key) for \($target)")' <<< "$package"
}

# cargo-binstall's template variables; its templates accept spaces inside the braces or none
render() {
  local out
  out="$(sed -E \
    -e "s|\{ *name *\}|$name|g" \
    -e "s|\{ *version *\}|$version|g" \
    -e "s|\{ *target *\}|$target|g" \
    -e "s|\{ *archive-suffix *\}|$suffix|g" \
    -e "s|\{ *binary-ext *\}|$binary_ext|g" \
    -e "s|\{ *bin *\}|$bin|g" \
    -e "s|\{ *repo *\}|$repo|g" <<< "$1")"
  if [[ "$out" == *"{"* ]]; then
    echo "$out uses a variable this script doesn't fill in" >&2
    exit 1
  fi
  printf '%s\n' "$out"
}

expected=()
for target in "${targets[@]}"; do
  # release-archive.sh builds a zip for Windows and a tgz for the rest, and cargo-binstall gives a Windows
  # binary its .exe
  fmt="$(field "$target" pkg-fmt)"
  case "$target:$fmt" in
    *-windows-*:zip) suffix=.zip binary_ext=.exe ;;
    *-windows-*:*) echo "$target: release-archive.sh builds a zip for Windows, not $fmt" >&2; exit 1 ;;
    *:tgz) suffix=.tar.gz binary_ext='' ;;
    *) echo "$target: release-archive.sh builds only tgz archives here, not $fmt" >&2; exit 1 ;;
  esac

  url="$(render "$(field "$target" pkg-url)")"
  path="$(render "$(field "$target" bin-dir)")"
  archive="${url##*/}"
  echo "$target: $url -> $path"
  if [ "$url" != "$repo/releases/download/v$version/$archive" ]; then
    echo "$url is not an asset of the v$version release" >&2
    exit 1
  fi
  expected+=("$archive")

  if [ -z "$dir" ]; then continue; fi

  if $download; then
    curl -fsSL --retry 3 -o "$dir/$archive" "$url"
  fi

  if ! [ -f "$dir/$archive" ]; then
    echo "$dir has no $archive" >&2
    exit 1
  fi

  if [ "$suffix" = .zip ]; then
    entries="$(unzip -Z1 "$dir/$archive")"
  else
    entries="$(tar -tzf "$dir/$archive")"
  fi
  if ! grep -Fxq "$path" <<< "$entries"; then
    echo "$archive has no $path; it holds:" >&2
    echo "$entries" >&2
    exit 1
  fi
done

if $download; then
  echo "every released CLI archive matches the binstall metadata"
elif [ -n "$dir" ]; then
  found="$(find "$dir" -type f \( -name '*.tar.gz' -o -name '*.zip' \) -exec basename {} \; | sort)"
  want="$(printf '%s\n' "${expected[@]}" | sort)"
  if [ "$found" != "$want" ]; then
    echo "$dir holds"$'\n'"$found"$'\n'"but the binstall metadata names"$'\n'"$want" >&2
    exit 1
  fi
  echo "every CLI archive matches the binstall metadata"
fi
