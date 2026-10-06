#!/usr/bin/env bash
# Download the Zenodo data deposit and unpack it into ./data (or $TEAMCACHE_DATA).
#
#   scripts/fetch_data.sh                      # from Zenodo
#   TEAMCACHE_DATA_ZIP=path/to.zip scripts/fetch_data.sh   # from a local copy
set -euo pipefail
cd "$(dirname "$0")/.."

RECORD="${ZENODO_RECORD:-23187430}"
NAME="teamcache-ncomms-data-v1.0.0.zip"
DEST="${TEAMCACHE_DATA:-data}"
ZIP="${TEAMCACHE_DATA_ZIP:-}"

if [[ -f "$DEST/MANIFEST.csv" ]]; then
  echo "$DEST already contains the deposit"
  exit 0
fi
if [[ -z "$ZIP" ]]; then
  ZIP="$(mktemp -d)/$NAME"
  curl -fL --retry 3 -o "$ZIP" "https://zenodo.org/records/$RECORD/files/$NAME?download=1"
fi
mkdir -p "$DEST"
unzip -q "$ZIP" -d "$DEST.tmp"
inner="$(find "$DEST.tmp" -name MANIFEST.csv -maxdepth 2 | head -1 | xargs dirname)"
cp -R "$inner"/. "$DEST"/
rm -rf "$DEST.tmp"
python3 - "$DEST" <<'EOF'
import csv, hashlib, sys
from pathlib import Path
root = Path(sys.argv[1]); bad = 0
rows = list(csv.DictReader(open(root / "MANIFEST.csv")))
for r in rows:
    h = hashlib.sha256((root / r["path"]).read_bytes()).hexdigest()
    if h != r["sha256"]:
        bad += 1
        print("checksum mismatch", r["path"])
print(f"{len(rows) - bad} of {len(rows)} files verified")
sys.exit(1 if bad else 0)
EOF
