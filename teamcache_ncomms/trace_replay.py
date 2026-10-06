#!/usr/bin/env python3
"""S7: every policy on the request order recorded on the GPUs.

The twelve-team mixed workload (six rings, six AutoGen selector teams) holds
60 histories of up to 10,240 tokens in a 420,800-token pool, i.e. 41 histories.  Each recorded LRU
run gives one request order; all policies are replayed on that same order and
scored after each team's ten warm-up requests, as on the GPUs.  Simulated LRU
and TeamCache are checked against the hit rates measured on the GPUs.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import classic as C
from . import optimal as O
from .paths import CACHE_LEVEL_OUT, MIXED as GPU

OUT = CACHE_LEVEL_OUT
SEEDS = {11: "", 12: "_s12", 13: "_s13", 14: "_s14"}
MEMBERS, WARMUP = 5, 10
CAP = 420_800 // 10_240
GPU_KEY = {"LRU": "lru", "TC": "teamcache"}


def load(policy, suffix):
    rows = [json.loads(l) for l in open(GPU / f"mixed_{policy}_12t{suffix}.requests.jsonl")]
    return sorted(rows, key=lambda r: r["t_send"])


def measured_mask(rows):
    seen, mask = defaultdict(int), []
    for r in rows:
        mask.append(seen[r["team"]] >= WARMUP)
        seen[r["team"]] += 1
    return mask


def score(flags, rows, mask):
    per = defaultdict(lambda: [0, 0])
    for f, r, m in zip(flags, rows, mask):
        if m:
            per[r["team"]][0] += 1
            per[r["team"]][1] += not f
    hit = sum(h for _, h in per.values()) / sum(n for n, _ in per.values())
    rates = {t: h / n for t, (n, h) in per.items()}
    return hit, min(rates.values()), sum(v <= 0.10 for v in rates.values()), rates


def policies(refs):
    flat = C.flatten(refs, MEMBERS)
    return {
        "LRU": O.run_lru(refs, CAP),
        "ARC": C.run_arc(flat, CAP),
        "LIRS": C.run_lirs(flat, CAP),
        "2Q": C.run_2q(flat, CAP),
        "S3-FIFO": C.run_s3fifo(flat, CAP),
        "W-TinyLFU": C.run_wtinylfu(flat, CAP),
        "TC-global": C.run_teamcache(flat, CAP),
        "TC": C.run_teamcache(refs, CAP),
        "OPT": O.run_optb(flat, CAP),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"config": {"members": MEMBERS, "capacity": CAP, "warmup_per_team": WARMUP,
                      "order_source": "recorded LRU runs"}, "runs": [], "gpu": []}
    for seed, suf in SEEDS.items():
        rows = load("lru", suf)
        refs = [(r["team"], r["member"]) for r in rows]
        mask = measured_mask(rows)
        for pol, flags in policies(refs).items():
            hit, worst, low, rates = score(flags, rows, mask)
            out["runs"].append({"seed": seed, "policy": pol, "request_hit": hit,
                                "worst_team_hit": worst, "low_reuse_teams": low,
                                "ring_hit": float(np.mean([rates[t] for t in range(6)])),
                                "selector_hit": float(np.mean([rates[t] for t in range(6, 12)]))})
        for pol, key in GPU_KEY.items():
            g = load(key, suf)
            m = measured_mask(g)
            flags = [(r.get("cached") or 0) < 0.5 for r in g]
            hit, worst, low, _ = score(flags, g, m)
            out["gpu"].append({"seed": seed, "policy": pol, "request_hit": hit,
                               "worst_team_hit": worst, "low_reuse_teams": low})
    summary = {}
    for pol in policies([(0, 0)]):
        rs = [r for r in out["runs"] if r["policy"] == pol]
        summary[pol] = {k: float(np.mean([r[k] for r in rs]))
                        for k in ("request_hit", "worst_team_hit", "low_reuse_teams",
                                  "ring_hit", "selector_hit")}
    for pol in GPU_KEY:
        rs = [r for r in out["gpu"] if r["policy"] == pol]
        summary["GPU " + pol] = {k: float(np.mean([r[k] for r in rs]))
                                 for k in ("request_hit", "worst_team_hit", "low_reuse_teams")}
    out["summary"] = summary
    json.dump(out, open(OUT / "s7_trace_replay.json", "w"), indent=1)
    for k, v in summary.items():
        print(f"{k:12s}", {a: round(b, 3) for a, b in v.items()})


if __name__ == "__main__":
    main()
