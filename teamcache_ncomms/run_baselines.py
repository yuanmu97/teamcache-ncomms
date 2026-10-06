#!/usr/bin/env python3
"""S6: TeamCache against team-agnostic policies from the caching literature.

Grid  six topologies x seven capacities x 24 seeds (the S1 grid).
Phase rotation regularity p x six oversubscribed capacities x 10 seeds (the S3 grid).

TC-global runs TeamCache's own rule with team identity erased, so that the
difference to TC isolates the value of grouping histories by team.
"""

from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import classic as C
from . import optimal as O
from .paths import CACHE_LEVEL_OUT
from .run_sims import (CAPS, MEMBERS, S1_SEEDS, SESSIONS, TOPOS, boot_ci,
                       regular_reference, team_rates)

OUT = CACHE_LEVEL_OUT
POLICIES = ["LRU", "ARC", "LIRS", "2Q", "S3-FIFO", "W-TinyLFU", "TC-global", "TC", "OPT"]


def run_all(refs, cap):
    flat = C.flatten(refs, MEMBERS)
    return {
        "LRU": O.run_lru(refs, cap),
        "ARC": C.run_arc(flat, cap),
        "LIRS": C.run_lirs(flat, cap),
        "2Q": C.run_2q(flat, cap),
        "S3-FIFO": C.run_s3fifo(flat, cap),
        "W-TinyLFU": C.run_wtinylfu(flat, cap),
        "TC-global": O.run_concentrate(flat, cap, SESSIONS, margin=1.02),
        "TC": O.run_concentrate(refs, cap, MEMBERS, margin=1.02),
        "OPT": O.run_optb(flat, cap),
    }


def _grid_job(args):
    topo, seed = args
    refs = O.topo_reference(topo, 9, MEMBERS, 40, seed=seed)
    tof = [t for t, _ in refs]
    rows = []
    for cap in CAPS:
        for pol, f in run_all(refs, cap).items():
            hit, worst = team_rates(f, tof)
            rows.append({"policy": pol, "topo": topo, "seed": seed, "cap": cap,
                         "load": SESSIONS / cap, "hit": hit, "worst": worst})
    return rows


def _phase_job(args):
    p, cap, seed = args
    refs = regular_reference(p, seed)
    tof = [t for t, _ in refs]
    out = []
    for pol, f in run_all(refs, cap).items():
        hit, worst = team_rates(f, tof)
        out.append({"policy": pol, "p": p, "cap": cap, "seed": seed,
                    "load": SESSIONS / cap, "hit": hit, "worst": worst})
    return out


def summarise(rows, keys):
    rng = np.random.default_rng(0)
    cells = defaultdict(list)
    for r in rows:
        cells[tuple(r[k] for k in keys)].append(r)
    out = []
    for key, rs in sorted(cells.items()):
        hit = [r["hit"] for r in rs]
        worst = [r["worst"] for r in rs]
        lo, hi = boot_ci(worst, rng)
        out.append({**dict(zip(keys, key)), "load": rs[0]["load"], "n": len(rs),
                    "hit_mean": float(np.mean(hit)), "worst_mean": float(np.mean(worst)),
                    "worst_lo": lo, "worst_hi": hi})
    return out


def headline(grid, phase):
    over = [c for c in grid if c["load"] > 1.01]
    head = {}
    for pol in POLICIES:
        cs = [c for c in over if c["policy"] == pol]
        ph = [c for c in phase if c["policy"] == pol]
        head[pol] = {
            "grid_mean_hit": float(np.mean([c["hit_mean"] for c in cs])),
            "grid_cells_worst_below_5pct": sum(c["worst_mean"] < 0.05 for c in cs),
            "grid_min_worst": float(min(c["worst_mean"] for c in cs)),
            "phase_mean_hit": float(np.mean([c["hit_mean"] for c in ph])),
        }
    best_online = {}
    for topo in TOPOS:
        for cap in CAPS[1:]:
            cs = {c["policy"]: c for c in over if c["topo"] == topo and c["cap"] == cap}
            best_online[f"{topo}:{cap}"] = max(
                (p for p in POLICIES if p not in ("OPT",)), key=lambda p: cs[p]["hit_mean"])
    head["best_online_by_cell"] = best_online
    return head


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor() as pool:
        grid_rows = [r for ch in pool.map(_grid_job, [(t, s) for t in TOPOS
                                                     for s in range(S1_SEEDS)]) for r in ch]
        ps = [round(x, 2) for x in np.linspace(0, 1, 11)]
        phase_rows = [r for ch in pool.map(_phase_job, [(p, c, s) for p in ps
                                                       for c in CAPS[1:] for s in range(10)])
                      for r in ch]
    grid = summarise(grid_rows, ("policy", "topo", "cap"))
    phase = summarise(phase_rows, ("policy", "p", "cap"))
    head = headline(grid, phase)
    json.dump({"policies": POLICIES, "headline": head, "grid": grid, "phase": phase,
               "grid_runs": grid_rows, "phase_runs": phase_rows},
              open(OUT / "s6_baselines.json", "w"), indent=1)
    for pol in POLICIES:
        print(pol, head[pol])


if __name__ == "__main__":
    main()
