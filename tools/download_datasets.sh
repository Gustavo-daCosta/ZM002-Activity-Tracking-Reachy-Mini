#!/usr/bin/env bash
# Download the public skeleton datasets used to train the action classifier (~1.8 GB total).
# Idempotent: a file whose size already matches is left alone.
#
#   tools/download_datasets.sh            # all of them
#   tools/download_datasets.sh ntu60      # just one
#
# Provenance, licenses and the classes we use: datasets/SOURCES.md
set -euo pipefail

DIR="$(cd "$(dirname "$0")/.." && pwd)/datasets"
mkdir -p "$DIR"

# name|expected bytes|url
FILES=(
"ntu60_hrnet.pkl|705401771|https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_hrnet.pkl"
"ucf101_hrnet.pkl|1070780736|https://download.openmmlab.com/mmaction/pyskl/data/ucf101/ucf101_hrnet.pkl"
"hmdb51_2d.pkl|212759219|https://download.openmmlab.com/mmaction/v1.0/skeleton/data/hmdb51_2d.pkl"
)

for entry in "${FILES[@]}"; do
  IFS='|' read -r name expected url <<<"$entry"
  [ $# -eq 0 ] || [[ "$name" == *"$1"* ]] || continue
  target="$DIR/$name"
  actual=$(stat -f%z "$target" 2>/dev/null || echo 0)
  if [ "$actual" = "$expected" ]; then
    echo "$name: already complete ($actual bytes)"
  else
    echo "$name: downloading from $url"
    curl -fL --retry 3 -C - -o "$target" "$url"
    actual=$(stat -f%z "$target")
    if [ "$actual" != "$expected" ]; then
      echo "$name: expected $expected bytes, got $actual - the download is incomplete" >&2
      exit 1
    fi
  fi
  echo "  bytes: $actual"
  echo "  sha256: $(shasum -a 256 "$target" | cut -d' ' -f1)"
done
