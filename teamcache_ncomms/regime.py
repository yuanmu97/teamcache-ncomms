#!/usr/bin/env python3
"""S10: how many recorded agent teams fit in a serving node's prefix pool."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .paths import CACHE_LEVEL_OUT, RES, ensure
POOLS = {"Kimi K3, GPU pool": 420_800, "Kimi K3, GPU and host pool": 1_260_000,
         "GLM 5.3, GPU pool": 2_700_000}


def main():
    rows = [r for r in json.load(open(RES / "collections-summary.json"))
            if r.get("k") and r.get("prompt_median")]
    ws = np.array([r["k"] * r["prompt_median"] for r in rows], float)
    out = {"teams": len(rows), "team_working_set_median": float(np.median(ws)),
           "team_working_set_q25": float(np.percentile(ws, 25)),
           "team_working_set_q75": float(np.percentile(ws, 75)), "pools": {}}
    for name, pool in POOLS.items():
        n = pool / ws
        out["pools"][name] = {"pool_tokens": pool, "teams_median": float(np.median(n)),
                              "teams_min": float(n.min()), "teams_max": float(n.max())}
    json.dump(out, open(ensure(CACHE_LEVEL_OUT) / "s10_regime.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
