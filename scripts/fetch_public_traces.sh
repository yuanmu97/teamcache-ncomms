#!/usr/bin/env bash
# Download the public multi-agent traces used by the public-trace experiment
# (Figs. 1b and 4d) into $TEAMCACHE_PUBLIC (default ./data/public_raw).
#
#   MAST      Cemri et al. 2025, huggingface.co/datasets/mcemri/MAST-Data, CC BY 4.0
#   Who&When  Zhang et al. ICML 2025, huggingface.co/datasets/Kevin355/Who_and_When, MIT
#
# Revisions are pinned; the checksums are those of the files the paper used.
set -euo pipefail
cd "$(dirname "$0")/.."

DEST="${TEAMCACHE_PUBLIC:-${TEAMCACHE_DATA:-data}/public_raw}"
MAST_REV=95118ac951421753cf1deb87ddea3b01e693c41b
WW_REV=59b9fcba1aaed7bbf206b5f4d3c68b8face2f49c
mkdir -p "$DEST"

fetch() {  # url  file  sha256
  if [[ ! -f "$DEST/$2" ]]; then
    curl -fL --retry 3 -o "$DEST/$2" "$1"
  fi
  got="$(shasum -a 256 "$DEST/$2" | cut -d' ' -f1)"
  if [[ "$got" == "$3" ]]; then
    echo "ok        $2"
  else
    echo "WARNING   $2 differs from the file used in the paper (sha256 $got)" >&2
  fi
}

fetch "https://huggingface.co/datasets/mcemri/MAST-Data/resolve/$MAST_REV/MAD_full_dataset.json" \
  MAD_full_dataset.json d636ac63dfc1c6af2d312e862f4b7d383b62d3898d431ccd0e7a79d21d85406f
fetch "https://huggingface.co/datasets/Kevin355/Who_and_When/resolve/$WW_REV/Algorithm-Generated.parquet" \
  ww_Algorithm-Generated.parquet 1d1720f4a8416ee0ed64bc13fa3f75655891e760a917924a071deac443133cf8
fetch "https://huggingface.co/datasets/Kevin355/Who_and_When/resolve/$WW_REV/Hand-Crafted.parquet" \
  ww_Hand-Crafted.parquet d28ffe10b124b2ba46a662be3b438fbe0de7c4c4fcba4f46302c0a4b2575f62f
