#!/usr/bin/env python3
"""Compare regenerated JSON results with the deposited ones.

    python -m teamcache_ncomms.compare cache-level   # out/cache_level vs data/cache_level
    python -m teamcache_ncomms.compare figure-data   # out/figure_data vs data/gpu/figure_data

Numbers must agree to a relative tolerance of 1e-9; keys present on only one
side and provenance strings are ignored.  Exit status 1 on any difference.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from . import paths

IGNORED = {"source"}


def diff(a, b, where="", out=None, limit=20):
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) & set(b), key=str):
            if k not in IGNORED:
                diff(a[k], b[k], f"{where}.{k}", out, limit)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{where}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{where}[{i}]", out, limit)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) \
            and not isinstance(a, bool) and not isinstance(b, bool):
        if not (math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
                or (math.isnan(a) and math.isnan(b))):
            out.append(f"{where}: {a!r} != {b!r}")
    elif a != b:
        out.append(f"{where}: {a!r} != {b!r}")
    return out


def compare_dirs(new: Path, old: Path) -> int:
    files = sorted(new.glob("*.json"))
    if not files:
        print(f"nothing to compare in {new}")
        return 1
    bad = 0
    for f in files:
        ref = old / f.name
        if not ref.exists():
            print(f"skip {f.name}: not in deposit")
            continue
        d = diff(json.loads(f.read_text()), json.loads(ref.read_text()), f.stem)
        print(f"{'identical' if not d else 'DIFFERENT':<10} {f.name}")
        for line in d:
            print("   ", line)
        bad += bool(d)
    return 1 if bad else 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    what = argv[0] if argv else "cache-level"
    if what == "cache-level":
        return compare_dirs(paths.CACHE_LEVEL_OUT, paths.CACHE_LEVEL_DEPOSIT)
    if what == "figure-data":
        return compare_dirs(paths.FIGDATA_OUT, paths.FIGDATA)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
