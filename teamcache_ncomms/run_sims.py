#!/usr/bin/env python3
"""CPU simulations for the Nature Communications version of TeamCache.

S1  42-cell topology x load grid with 24 seeds and bootstrap confidence intervals.
S2  LRU collapse on interleaved rings: simulation versus a characteristic-time model.
S3  Ring-regularity x load phase diagram for LRU, Static Partitioning and TeamCache.
S4  Sensitivity of the opening-figure speedup estimate to the cold/hit cost ratio.

The policies are imported unchanged from ../../optimal.py; seeds 0-2 of S1
reproduce results/pareto-126.json exactly.
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import optimal as O
from .paths import CACHE_LEVEL_OUT, SWEEP

OUT = CACHE_LEVEL_OUT
TOPOS = ["ring", "empirical", "star", "debate", "pairs", "mixed"]
TEAMS, MEMBERS, ROUNDS = 9, 7, 40
SESSIONS = TEAMS * MEMBERS
CAPS = [63, 38, 27, 21, 17, 15, 13]
S1_SEEDS = 24
BOOT = 4000


def team_rates(flags, tof, skip=0):
    per = defaultdict(lambda: [0, 0])
    for f, t in zip(flags[skip:], tof[skip:]):
        per[t][0] += 1
        per[t][1] += f
    hits = sum(n - m for n, m in per.values())
    total = sum(n for n, _ in per.values())
    return hits / total, min(1 - m / n for n, m in per.values())


def boot_ci(vals, rng):
    vals = np.asarray(vals, dtype=float)
    idx = rng.integers(0, len(vals), size=(BOOT, len(vals)))
    means = vals[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------------------------------------------------------- S1

def _s1_job(args):
    topo, seed = args
    refs = O.topo_reference(topo, TEAMS, MEMBERS, ROUNDS, seed=seed)
    tof = [t for t, _ in refs]
    rows = []
    for cap in CAPS:
        flags = {
            "LRU": O.run_lru(refs, cap),
            "Blind": O.run_lru_bypass(refs, cap),
            "Static": O.run_cohort(refs, cap, MEMBERS),
            "TC": O.run_concentrate(refs, cap, MEMBERS, margin=1.02),
        }
        for pol, f in flags.items():
            hit, worst = team_rates(f, tof)
            rows.append({"policy": pol, "topo": topo, "seed": seed, "cap": cap,
                         "load": SESSIONS / cap, "hit": hit, "worst": worst})
    return rows


def s1(pool):
    jobs = [(t, s) for t in TOPOS for s in range(S1_SEEDS)]
    rows = [r for chunk in pool.map(_s1_job, jobs) for r in chunk]
    rng = np.random.default_rng(0)
    cells = defaultdict(list)
    for r in rows:
        cells[(r["policy"], r["topo"], r["cap"])].append(r)
    summary = []
    for (pol, topo, cap), rs in sorted(cells.items()):
        worst = [r["worst"] for r in rs]
        hit = [r["hit"] for r in rs]
        lo, hi = boot_ci(worst, rng)
        hlo, hhi = boot_ci(hit, rng)
        summary.append({
            "policy": pol, "topo": topo, "cap": cap, "load": SESSIONS / cap,
            "n": len(rs), "worst_mean": float(np.mean(worst)),
            "worst_lo": lo, "worst_hi": hi,
            "hit_mean": float(np.mean(hit)), "hit_lo": hlo, "hit_hi": hhi,
        })
    headline = {}
    for pol in ("LRU", "Blind", "Static", "TC"):
        over = [c for c in summary if c["policy"] == pol and c["load"] > 1.01]
        headline[pol] = {
            "cells_below_5pct": sum(c["worst_mean"] < 0.05 for c in over),
            "min_cell_worst": min(c["worst_mean"] for c in over),
            "min_cell": min(over, key=lambda c: c["worst_mean"])["topo"],
            "n_oversubscribed_cells": len(over),
        }
    json.dump({"config": {"teams": TEAMS, "members": MEMBERS, "rounds": ROUNDS,
                          "seeds": S1_SEEDS, "caps": CAPS, "bootstrap": BOOT},
               "headline": headline, "cells": summary, "runs": rows},
              open(OUT / "s1_grid.json", "w"), indent=1)
    print("S1", json.dumps(headline, indent=1))


# ---------------------------------------------------------------- S2

def che_hit(samples_by_item, rate_by_item, capacity):
    """Characteristic-time approximation of LRU from inter-reference samples.

    Each item i has an empirical inter-reference distribution tau_i (in global
    references) and reference rate lambda_i.  The characteristic time t_C
    solves sum_i P(age_i < t_C) = C with P(age_i < t) = lambda_i *
    sum_{s<t} P(tau_i > s); an item hits when tau_i <= t_C.
    """
    items = [i for i in samples_by_item if len(samples_by_item[i]) > 0]
    if capacity >= len(items):
        return 1.0
    sorted_s = {i: np.sort(np.asarray(samples_by_item[i])) for i in items}
    tmax = max(int(s[-1]) for s in sorted_s.values()) + 1
    grid = np.arange(tmax + 1)
    occ = np.zeros_like(grid, dtype=float)
    cdf = {}
    for i in items:
        s = sorted_s[i]
        c = np.searchsorted(s, grid, side="right") / len(s)
        cdf[i] = c
        surv = 1.0 - c
        age = np.concatenate([[0.0], np.cumsum(surv[:-1])]) * rate_by_item[i]
        occ += np.minimum(age, 1.0)
    k = int(np.searchsorted(occ, capacity, side="left"))
    k = min(max(k, 0), tmax)
    if 0 < k <= tmax:
        lo, hi = occ[k - 1], occ[k]
        frac = 0.0 if hi == lo else (capacity - lo) / (hi - lo)
    else:
        frac = 0.0
    total = sum(rate_by_item[i] for i in items)
    hit = 0.0
    for i in items:
        c = cdf[i]
        p_prev = c[k - 1] if k >= 1 else 0.0
        p = p_prev + frac * (c[k] - p_prev)
        # tau <= t_C counts samples at or below the continuous characteristic time
        hit += rate_by_item[i] * p
    return hit / total


def interref(refs, skip):
    last, samples, count = {}, defaultdict(list), defaultdict(int)
    for idx, key in enumerate(refs):
        if key in last and idx >= skip:
            samples[key].append(idx - last[key])
        last[key] = idx
        if idx >= skip:
            count[key] += 1
    n = max(len(refs) - skip, 1)
    return samples, {k: count[k] / n for k in count}


def ring_negbin_che(teams, members, capacity):
    """Closed-form inter-reference law for a k-member ring among T random teams.

    A member returns after its own team takes k steps; each global reference
    belongs to that team with probability 1/T, so tau ~ k + NegBin(k, 1/T).
    """
    p = 1.0 / teams
    tmax = int(members * teams * 8 + 50)
    pmf = np.zeros(tmax + 1)
    for t in range(members, tmax + 1):
        x = t - members
        pmf[t] = math.comb(x + members - 1, x) * p ** members * (1 - p) ** x
    cdf = np.cumsum(pmf)
    surv = 1.0 - cdf
    rate = 1.0 / (members * teams)
    age = np.concatenate([[0.0], np.cumsum(surv[:-1])]) * rate
    occ = teams * members * np.minimum(age, 1.0)
    if capacity >= teams * members:
        return 1.0
    k = int(np.searchsorted(occ, capacity, side="left"))
    lo, hi = occ[k - 1], occ[k]
    frac = 0.0 if hi == lo else (capacity - lo) / (hi - lo)
    return float(cdf[k - 1] + frac * (cdf[k] - cdf[k - 1]))


def _s2_job(args):
    members, cap, seed = args
    refs = O.topo_reference("ring", TEAMS, members, ROUNDS, seed=seed)
    skip = 3 * TEAMS * members
    tof = [t for t, _ in refs]
    hit_lru, worst = team_rates(O.run_lru(refs, cap), tof, skip)
    hit_tc, _ = team_rates(O.run_concentrate(refs, cap, members, margin=1.02), tof, skip)
    hit_st, _ = team_rates(O.run_cohort(refs, cap, members), tof, skip)
    samples, rates = interref(refs, skip)
    che_emp = che_hit(samples, rates, cap)
    return {"members": members, "cap": cap, "seed": seed,
            "load": TEAMS * members / cap, "lru": hit_lru, "lru_worst": worst,
            "tc": hit_tc, "static": hit_st, "che_empirical": che_emp,
            "che_closed": ring_negbin_che(TEAMS, members, cap),
            "lockstep": 1.0 if cap >= TEAMS * members else 0.0}


def _s2_topo_job(args):
    topo, cap, seed = args
    refs = O.topo_reference(topo, TEAMS, MEMBERS, ROUNDS, seed=seed)
    skip = 3 * SESSIONS
    tof = [t for t, _ in refs]
    hit, _ = team_rates(O.run_lru(refs, cap), tof, skip)
    samples, rates = interref(refs, skip)
    return {"topo": topo, "cap": cap, "seed": seed, "load": SESSIONS / cap,
            "lru": hit, "che": che_hit(samples, rates, cap)}


def s2(pool):
    jobs = []
    for members in range(3, 9):
        n = TEAMS * members
        caps = sorted({max(1, int(round(n / L)))
                       for L in np.linspace(0.6, 1.8, 25)})
        jobs += [(members, c, s) for c in caps for s in range(5)]
    rows = list(pool.map(_s2_job, jobs))
    topo_jobs = [(t, c, s) for t in TOPOS
                 for c in sorted({max(1, int(round(SESSIONS / L)))
                                  for L in np.linspace(0.6, 5.0, 23)})
                 for s in range(3)]
    topo_rows = list(pool.map(_s2_topo_job, topo_jobs))
    err = [abs(r["lru"] - r["che"]) for r in topo_rows]
    ring_err = [abs(r["lru"] - r["che_closed"]) for r in rows]
    ss_res = sum((r["lru"] - r["che"]) ** 2 for r in topo_rows)
    mean = np.mean([r["lru"] for r in topo_rows])
    ss_tot = sum((r["lru"] - mean) ** 2 for r in topo_rows)
    summary = {
        "ring_closed_mae": float(np.mean(ring_err)),
        "topology_che_mae": float(np.mean(err)),
        "topology_che_r2": float(1 - ss_res / ss_tot),
        "topology_points": len(topo_rows),
    }
    gpu = []
    root = SWEEP
    for t in (6, 8, 10, 12, 14, 16):
        d = json.load(open(root / f"lru_{t}t_ring.json"))
        ws = d["teams"] * d["members"] * (d["context"] + 512 + d["gen_tokens"])
        gpu.append({"teams": t, "members": d["members"], "load": ws / 420_800,
                    "hit": 1 - d["turn_miss_rate"]})
    json.dump({"summary": summary, "ring": rows, "topology": topo_rows, "gpu": gpu},
              open(OUT / "s2_cliff.json", "w"), indent=1)
    print("S2", summary)


# ---------------------------------------------------------------- S3

def regular_order(members, p, seed):
    """Ring successor with probability p, else the fitted recency-biased draw."""
    rng = random.Random(seed)
    weights = [math.exp(-O.EMPIRICAL_LAMBDA * r) for r in range(members)]
    stack = list(range(members))
    last = members - 1
    for m in range(members):
        yield m
    while True:
        if rng.random() < p:
            m = (last + 1) % members
        else:
            m = stack[rng.choices(range(members), weights=weights)[0]]
        stack.remove(m)
        stack.insert(0, m)
        last = m
        yield m


def regular_reference(p, seed):
    rng = random.Random(seed)
    gens = {t: regular_order(MEMBERS, p, seed ^ (t << 8)) for t in range(TEAMS)}
    return [(t, next(gens[t])) for t in (rng.randrange(TEAMS)
                                         for _ in range(ROUNDS * SESSIONS))]


def _s3_job(args):
    p, cap, seed = args
    refs = regular_reference(p, seed)
    tof = [t for t, _ in refs]
    out = {"p": p, "cap": cap, "seed": seed, "load": SESSIONS / cap}
    for name, f in (("LRU", O.run_lru(refs, cap)),
                    ("Static", O.run_cohort(refs, cap, MEMBERS)),
                    ("TC", O.run_concentrate(refs, cap, MEMBERS, margin=1.02))):
        out[name], out[name + "_worst"] = team_rates(f, tof)
    skip = 3 * SESSIONS
    samples, rates = interref(refs, skip)
    out["che_lru"] = che_hit(samples, rates, cap)
    # Static value from the frequency lemma: top-quota mass per team.
    quota = O.cohort_quota(refs, cap, MEMBERS)
    count = defaultdict(int)
    for k in refs[skip:]:
        count[k] += 1
    tot = sum(count.values())
    out["lemma_static"] = sum(
        sum(sorted((count[(t, m)] for m in range(MEMBERS)), reverse=True)[:quota[t]])
        for t in range(TEAMS)) / tot
    return out


def s3(pool):
    ps = [round(x, 2) for x in np.linspace(0, 1, 11)]
    jobs = [(p, c, s) for p in ps for c in CAPS[1:] for s in range(10)]
    rows = list(pool.map(_s3_job, jobs))
    cell = defaultdict(list)
    for r in rows:
        cell[(r["p"], r["cap"])].append(r)
    summary = []
    for (p, cap), rs in sorted(cell.items()):
        m = {k: float(np.mean([r[k] for r in rs]))
             for k in ("LRU", "Static", "TC", "LRU_worst", "Static_worst",
                       "TC_worst", "che_lru", "lemma_static")}
        best = max(m["LRU"], m["Static"])
        m.update({"p": p, "cap": cap, "load": SESSIONS / cap, "n": len(rs),
                  "winner": "LRU" if m["LRU"] >= m["Static"] else "Static",
                  "predicted_winner": "LRU" if m["che_lru"] >= m["lemma_static"] else "Static",
                  "tc_minus_best": m["TC"] - best,
                  "tc_minus_worse": m["TC"] - min(m["LRU"], m["Static"])})
        summary.append(m)
    gaps = [c["tc_minus_best"] for c in summary]
    agree = sum(c["winner"] == c["predicted_winner"] for c in summary)
    head = {"cells": len(summary),
            "tc_gap_to_best_median_pp": 100 * float(np.median(gaps)),
            "tc_gap_to_best_p10_pp": 100 * float(np.percentile(gaps, 10)),
            "tc_gap_to_best_min_pp": 100 * float(min(gaps)),
            "cells_within_2pp_of_best": sum(g >= -0.02 for g in gaps),
            "cells_tc_beats_both": sum(g > 0 for g in gaps),
            "winner_prediction_agreement": agree}
    json.dump({"headline": head, "cells": summary, "runs": rows},
              open(OUT / "s3_phase.json", "w"), indent=1)
    print("S3", head)


# ---------------------------------------------------------------- S4

def s4():
    data = json.load(open(OUT / "s1_grid.json"))
    runs = data["runs"]
    key = defaultdict(dict)
    for r in runs:
        key[(r["topo"], r["seed"], r["cap"])][r["policy"]] = r["worst"]
    ratios = [2, 3, 4.875, 7, 10, 15, 21.5, 30]
    out = []
    for ratio in ratios:
        def wall(h):
            return (1 - h) * ratio + h * 1.0
        for topo in TOPOS:
            for pol in ("Blind", "Static", "TC"):
                vals = [wall(v["LRU"]) / wall(v[pol])
                        for (t, s, c), v in key.items()
                        if t == topo and SESSIONS / c >= 2.2]
                out.append({"ratio": ratio, "topo": topo, "policy": pol,
                            "speedup": float(np.mean(vals))})
    stable = {}
    for topo in TOPOS:
        ranks = []
        for ratio in ratios:
            sp = {r["policy"]: r["speedup"] for r in out
                  if r["ratio"] == ratio and r["topo"] == topo}
            ranks.append(tuple(sorted(sp, key=lambda k: -sp[k])))
        stable[topo] = {"rank_stable": len(set(ranks)) == 1,
                        "tc_above_1": all(r["speedup"] >= 0.999 for r in out
                                          if r["topo"] == topo and r["policy"] == "TC")}
    json.dump({"ratios": ratios, "rows": out, "stability": stable},
              open(OUT / "s4_cost.json", "w"), indent=1)
    print("S4", stable)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or ["s1", "s2", "s3", "s4"]
    with ProcessPoolExecutor() as pool:
        if "s1" in which:
            s1(pool)
        if "s2" in which:
            s2(pool)
        if "s3" in which:
            s3(pool)
    if "s4" in which:
        s4()


if __name__ == "__main__":
    main()
