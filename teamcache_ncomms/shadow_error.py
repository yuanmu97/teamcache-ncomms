#!/usr/bin/env python3
"""S12: how well the shadow stack predicts what the real engine retained.

TeamCache never asks the engine what it holds.  It keeps a shadow LRU stack
of member histories, in tokens, and for each returning member records a
would-hold bit: whether the tokens used since that member's last turn, plus
its own history, fit in the pool.  This script replays the request logs of
the LRU arms of the GPU experiments through the deployed ``observe`` code and
compares every would-hold bit with what the engine actually served: a
reference counts as held when at least half of the member's previous prompt
came back as cached tokens.

Reported per run: per-request agreement, the rate of each error direction,
and per-team mean absolute error and bias of the would-hold rate, the
quantity the rule prices.  Only references taken under pressure (the stack
larger than the pool) are scored, because only those enter the rule.

    python -m teamcache_ncomms.shadow_error [--extra label=path ...]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

from .engine_rule import EngineRule
from .paths import CACHE_LEVEL_OUT, DATA

POOL = 420_800
HELD_FRACTION = 0.5


def _b300_sources():
    tr, mx = DATA / "traces", DATA / "gpu" / "mixed"
    src = [("B300 ring (live AutoGen, 20 teams)", tr / "e2e-autogen-design-lru-20t-p3000r8.jsonl"),
           ("B300 selector (live AutoGen, 32 teams)", tr / "e2e-selector-debug-lru-32t-p3000r8.jsonl")]
    for f in sorted(mx.glob("mixed_lru_12t*.requests.jsonl")):
        seed = f.name.split("_s")[-1].split(".")[0] if "_s1" in f.name else "11"
        src.append((f"B300 mixed fleet (replay, seed {seed})", f))
    return src


def load(path):
    rows = []
    for line in open(path):
        r = json.loads(line)
        rid = str(r.get("rid") or "")
        parts = rid.split(":")
        if len(parts) < 2 or not parts[1].isdigit():
            continue                       # chooser calls carry no team id
        t = r.get("t_recv", r.get("t_send"))
        if t is None or not r.get("prompt_tokens"):
            continue
        rows.append((float(t), parts[0], int(parts[1]), int(r["prompt_tokens"]),
                     int(r.get("cached_tokens") or 0)))
    rows.sort()
    return rows


def score(rows, pool=POOL):
    rule = EngineRule(pool)
    st = rule.state
    prev_prompt = {}
    pairs = []                             # (team, predicted, observed)
    for _, team, member, prompt, cached in rows:
        key = (team, member)
        hist = st.held.get(key)
        n0 = len(hist) if hist else 0
        st.observe(team, member, prompt, pool, 10 ** 9)
        hist = st.held.get(key) or []
        if len(hist) > n0 and key in prev_prompt:
            bit, under = hist[-1]
            if under:
                observed = int(cached >= HELD_FRACTION * prev_prompt[key])
                pairs.append((team, bit, observed))
        prev_prompt[key] = prompt
    if not pairs:
        return None
    pred = np.array([p for _, p, _ in pairs])
    obs = np.array([o for _, _, o in pairs])
    per = defaultdict(lambda: [0, 0, 0])
    for team, p, o in pairs:
        per[team][0] += p
        per[team][1] += o
        per[team][2] += 1
    err = [(p - o) / n for p, o, n in per.values() if n >= 3]
    return {"references": len(pairs), "teams": len(per),
            "agreement": float((pred == obs).mean()),
            "false_hold": float(((pred == 1) & (obs == 0)).mean()),
            "false_miss": float(((pred == 0) & (obs == 1)).mean()),
            "observed_held": float(obs.mean()), "predicted_held": float(pred.mean()),
            "team_mae_pp": 100 * float(np.mean(np.abs(err))) if err else None,
            "team_bias_pp": 100 * float(np.mean(err)) if err else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--extra", action="append", default=[],
                    help="label=glob of further LRU request logs (e.g. the A100 runs)")
    a = ap.parse_args(argv)
    sources = [("B300", lbl, p) for lbl, p in _b300_sources() if os.path.exists(p)]
    for spec in a.extra:
        lbl, pat = spec.split("=", 1)
        for f in sorted(glob.glob(pat)):
            sources.append((lbl.split()[0], f"{lbl} ({os.path.basename(f)})", f))
    runs = []
    for group, lbl, p in sources:
        res = score(load(p))
        if res:
            runs.append({"group": group, "run": lbl, "file": os.path.basename(str(p)), **res})
            print(f"{lbl:60s} n={res['references']:4d} agree={res['agreement']:.3f} "
                  f"false_hold={res['false_hold']:.3f} false_miss={res['false_miss']:.3f} "
                  f"team MAE={res['team_mae_pp']:.1f}pp bias={res['team_bias_pp']:+.1f}pp")

    def summarize(rs):
        n = sum(r["references"] for r in rs)
        return {"references": n, "runs": len(rs),
                "agreement": sum(r["agreement"] * r["references"] for r in rs) / n,
                "agreement_min": float(min(r["agreement"] for r in rs)),
                "agreement_max": float(max(r["agreement"] for r in rs)),
                "team_mae_pp_median": float(np.median([r["team_mae_pp"] for r in rs])),
                "team_mae_pp_max": float(np.max([r["team_mae_pp"] for r in rs]))}

    groups = {g: summarize([r for r in runs if r["group"] == g])
              for g in dict.fromkeys(r["group"] for r in runs)}
    CACHE_LEVEL_OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"pool_tokens": POOL, "held_fraction": HELD_FRACTION,
               "summary": groups["B300"], "groups": groups, "runs": runs},
              open(CACHE_LEVEL_OUT / "s12_shadow_error.json", "w"), indent=1)
    print(json.dumps(groups, indent=1))


if __name__ == "__main__":
    sys.exit(main())
