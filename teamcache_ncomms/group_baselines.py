#!/usr/bin/env python3
"""S14: group-aware baselines.

Classic replacement policies are team-agnostic.  This experiment gives them
team identity in the two standard ways and asks whether that is enough.

* Partitioned: the pool is split equally between teams and each team runs
  its own LRU, ARC or W-TinyLFU over its own members.
* Utility-based cache partitioning (UCP; Qureshi and Patt, MICRO 2006):
  each team's LRU miss curve is measured over the whole trace in hindsight
  and the pool is allocated by lookahead to minimise total misses.
* Max-min partitioning: the same hindsight curves, allocated to maximise the
  worst team's hit rate.

Static Partitioning and TeamCache are the same as in S1.  The grid is the S1
grid: six topologies, seven capacities, ``SEEDS`` seeds.

    python -m teamcache_ncomms.group_baselines     # writes s14_group_baselines.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import classic as C
from . import optimal as O
from .paths import CACHE_LEVEL_OUT
from .run_sims import CAPS, MEMBERS, SESSIONS, TEAMS, TOPOS, team_rates

SEEDS = 10
POLICIES = ["LRU", "Partitioned LRU", "Partitioned ARC", "Partitioned W-TinyLFU",
            "UCP (hindsight)", "Max-min partitioning (hindsight)",
            "Static Partitioning", "TeamCache"]


def _per_team(refs, alloc, run):
    """Run `run(keys, cap)` on each team's own subsequence with its allocation."""
    idx = defaultdict(list)
    for i, (t, _) in enumerate(refs):
        idx[t].append(i)
    flags = [True] * len(refs)
    for t, ids in idx.items():
        cap = alloc.get(t, 0)
        if cap <= 0:
            continue
        keys = [refs[i][1] for i in ids]
        for i, f in zip(ids, run(keys, cap)):
            flags[i] = f
    return flags


def _lru(keys, cap):
    return O.run_lru([(0, k) for k in keys], cap)


def _wtinylfu(keys, cap):
    # With one slot there is no main region to admit into; it is an LRU slot.
    return C.run_wtinylfu(keys, cap) if cap >= 2 else _lru(keys, cap)


def _equal(refs, cap):
    teams = sorted({t for t, _ in refs})
    base, extra = divmod(cap, len(teams))
    return {t: base + (1 if i < extra else 0) for i, t in enumerate(teams)}


def _curves(refs):
    """hits[t][c]: team t's LRU hits with c slots, c = 0..MEMBERS."""
    idx = defaultdict(list)
    for t, m in refs:
        idx[t].append(m)
    out = {}
    for t, keys in idx.items():
        h = [0]
        for c in range(1, MEMBERS + 1):
            h.append(sum(not f for f in _lru(keys, c)))
        out[t] = (h, len(keys))
    return out


def _ucp(curves, cap):
    """Lookahead allocation (Qureshi and Patt): best marginal utility per slot."""
    alloc = {t: 0 for t in curves}
    left = cap
    while left > 0:
        best = None
        for t, (h, _) in curves.items():
            a = alloc[t]
            for extra in range(1, min(left, MEMBERS - a) + 1):
                mu = (h[a + extra] - h[a]) / extra
                if best is None or mu > best[0]:
                    best = (mu, t, extra)
        if best is None or best[0] <= 0:
            break
        _, t, extra = best
        alloc[t] += extra
        left -= extra
    return alloc


def _maxmin(curves, cap):
    """Raise the worst team first; lookahead over convex stretches of its curve."""
    alloc = {t: 0 for t in curves}
    left = cap
    while left > 0:
        order = sorted(curves, key=lambda t: curves[t][0][alloc[t]] / curves[t][1])
        moved = False
        for t in order:
            h, n = curves[t]
            a = alloc[t]
            step = next((e for e in range(1, min(left, MEMBERS - a) + 1)
                         if h[a + e] > h[a]), None)
            if step is not None:
                alloc[t] += step
                left -= step
                moved = True
                break
        if not moved:
            break
    return alloc


def run_all(refs, cap):
    eq = _equal(refs, cap)
    curves = _curves(refs)
    return {
        "LRU": O.run_lru(refs, cap),
        "Partitioned LRU": _per_team(refs, eq, _lru),
        "Partitioned ARC": _per_team(refs, eq, C.run_arc),
        "Partitioned W-TinyLFU": _per_team(refs, eq, _wtinylfu),
        "UCP (hindsight)": _per_team(refs, _ucp(curves, cap), _lru),
        "Max-min partitioning (hindsight)": _per_team(refs, _maxmin(curves, cap), _lru),
        "Static Partitioning": O.run_cohort(refs, cap, MEMBERS),
        "TeamCache": O.run_concentrate(refs, cap, MEMBERS, margin=1.02),
    }


def _job(args):
    topo, seed = args
    refs = O.topo_reference(topo, TEAMS, MEMBERS, 40, seed=seed)
    tof = [t for t, _ in refs]
    rows = []
    for cap in CAPS[1:]:
        for pol, f in run_all(refs, cap).items():
            hit, worst = team_rates(f, tof)
            rows.append({"policy": pol, "topo": topo, "seed": seed, "cap": cap,
                         "load": SESSIONS / cap, "hit": hit, "worst": worst})
    return rows


def main():
    CACHE_LEVEL_OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor() as pool:
        rows = [r for ch in pool.map(_job, [(t, s) for t in TOPOS for s in range(SEEDS)])
                for r in ch]
    cells = defaultdict(list)
    for r in rows:
        cells[(r["policy"], r["topo"], r["cap"])].append(r)
    grid = [{"policy": p, "topo": t, "cap": c, "load": SESSIONS / c, "n": len(rs),
             "hit_mean": float(np.mean([r["hit"] for r in rs])),
             "worst_mean": float(np.mean([r["worst"] for r in rs]))}
            for (p, t, c), rs in sorted(cells.items())]
    head = {}
    for p in POLICIES:
        cs = [g for g in grid if g["policy"] == p]
        head[p] = {"mean_hit": float(np.mean([g["hit_mean"] for g in cs])),
                   "mean_worst": float(np.mean([g["worst_mean"] for g in cs])),
                   "min_worst": float(min(g["worst_mean"] for g in cs)),
                   "cells_worst_below_5pct": sum(g["worst_mean"] < 0.05 for g in cs),
                   "cells": len(cs),
                   "by_topology_mean_worst": {
                       t: float(np.mean([g["worst_mean"] for g in cs if g["topo"] == t]))
                       for t in TOPOS}}
    json.dump({"policies": POLICIES, "seeds": SEEDS, "caps": CAPS[1:],
               "headline": head, "grid": grid},
              open(CACHE_LEVEL_OUT / "s14_group_baselines.json", "w"), indent=1)
    for p in POLICIES:
        h = head[p]
        print(f"{p:34s} hit={h['mean_hit']:.3f} worst={h['mean_worst']:.3f} "
              f"min_worst={h['min_worst']:.3f} starved={h['cells_worst_below_5pct']}/{h['cells']} "
              + " ".join(f"{t[:4]}={v:.2f}" for t, v in h["by_topology_mean_worst"].items()))


if __name__ == "__main__":
    main()
