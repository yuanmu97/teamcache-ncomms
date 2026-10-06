#!/usr/bin/env python3
"""Per-team policy values over time on a mixed ring/free-form workload (Fig. 3).

TeamCache's two per-team values are functions of the request sequence alone:
the shadow recency order and the per-member reference counts are advanced for
every request, admitted or not.  This script therefore recomputes them exactly
as optimal.run_concentrate does and records them alongside the hit/miss flags
of LRU and TeamCache on the same sequence.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from . import optimal as O
from .paths import CACHE_LEVEL_OUT, ensure

TEAMS, MEMBERS, ROUNDS, CAP, SEED = 8, 7, 40, 24, 3
MARGIN, WINDOW, MIN_SAMPLES = 1.02, 200, 2
KINDS = ["ring", "empirical"]


def reference():
    rng = random.Random(SEED)
    gens = {t: O._intra_order(KINDS[t % 2], MEMBERS, SEED ^ (t << 8)) for t in range(TEAMS)}
    return [(t, next(gens[t])) for t in (rng.randrange(TEAMS)
                                         for _ in range(ROUNDS * TEAMS * MEMBERS))]


def main():
    refs = reference()
    share = max(int(CAP / TEAMS), 1)
    dist, seen, last = defaultdict(list), defaultdict(int), {}
    values, snapshot = [], None
    for clock, key in enumerate(refs):
        team, member = key
        reuse = None
        if key in last:
            since = last[key]
            reuse = sum(1 for v in last.values() if v > since)
            dist[key].append((reuse, len(last) > CAP))
            if len(dist[key]) > WINDOW:
                del dist[key][:-WINDOW]
        if snapshot is None and clock > 600 and reuse is not None and KINDS[team % 2] == "ring":
            order = sorted(last, key=lambda k: -last[k])
            snapshot = {"clock": clock, "returning": list(key), "distance": reuse,
                        "order": [list(k) for k in order]}
        last[key] = clock
        seen[key] += 1
        p_hit, ok = {}, True
        for m in range(MEMBERS):
            bits = [1 if x < CAP else 0 for x, under in dist[(team, m)] if under]
            if len(bits) < MIN_SAMPLES:
                ok = False
                break
            p_hit[m] = sum(bits) / len(bits)
        if ok:
            rate = {m: seen[(team, m)] for m in range(MEMBERS)}
            mass = sum(rate.values()) or 1
            top = sorted(range(MEMBERS), key=lambda m: -rate[m])[:share]
            conc = sum(rate[m] for m in top) / mass
            spread = sum(rate[m] * p_hit[m] for m in range(MEMBERS)) / mass
            values.append({"clock": clock, "team": team, "kind": KINDS[team % 2],
                           "conc": conc, "lru": spread,
                           "concentrate": conc > spread * MARGIN})
    flags = {"LRU": O.run_lru(refs, CAP),
             "TeamCache": O.run_concentrate(refs, CAP, MEMBERS, margin=MARGIN)}
    out = {"config": {"teams": TEAMS, "members": MEMBERS, "rounds": ROUNDS, "cap": CAP,
                      "share": share, "seed": SEED, "margin": MARGIN, "kinds": KINDS},
           "refs": [list(r) for r in refs], "values": values, "snapshot": snapshot,
           "miss": {k: [bool(x) for x in v] for k, v in flags.items()}}
    (ensure(CACHE_LEVEL_OUT) / "s5_policy_trace.json").write_text(json.dumps(out))
    for kind in KINDS:
        for pol, f in flags.items():
            idx = [i for i, (t, _) in enumerate(refs) if KINDS[t % 2] == kind and i > 3 * TEAMS * MEMBERS]
            print(kind, pol, round(1 - sum(f[i] for i in idx) / len(idx), 3))
    last_vals = {}
    for v in values:
        last_vals[v["team"]] = v
    for t, v in sorted(last_vals.items()):
        print(t, v["kind"], round(v["conc"], 3), round(v["lru"], 3), v["concentrate"])
    print("snapshot", snapshot["returning"], snapshot["distance"])


if __name__ == "__main__":
    main()
