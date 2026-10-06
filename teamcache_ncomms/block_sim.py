#!/usr/bin/env python3
"""S11: variable-size, block-level prefix-cache experiment.

The session-level experiments treat every member history as one equal-sized,
indivisible object.  This experiment replaces that abstraction with the
structure of a radix prefix cache:

* histories have different sizes (log-normal, coefficient of variation CV),
* every history grows by a fixed number of tokens each time its member speaks,
* a system template of fraction F of the mean history is shared by all
  requests and stored once,
* the cache holds 16-token blocks and evicts the least recently used leaf
  first, so a partially evicted history keeps its head and a returning
  request reuses whatever prefix is still resident,
* every team has one request in flight, whose working KV occupies the pool.

TeamCache is the deployed predicate (``engine_rule``), fed the same request
identifiers and prompt lengths as the engine.  The baselines are LRU,
Unpartitioned Admission, Static Partitioning and Offline LFU with
token-based quotas.  The headline metric is prefix-token reuse (cached prompt
tokens over prompt tokens) of the worst team and of all teams.

    python -m teamcache_ncomms.block_sim            # writes s11_block_level.json
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import optimal as O
from .engine_rule import EngineRule
from .paths import CACHE_LEVEL_OUT

OUT = CACHE_LEVEL_OUT
TOPOS = ["ring", "empirical", "star", "debate", "pairs", "mixed"]
TEAMS, MEMBERS, ROUNDS = 9, 7, 30
BLOCK = 16
MEAN_TOKENS = 10_000
LOADS = [1.25, 1.66, 2.33, 3.0]
SEEDS = 5
DT = 0.25                      # seconds of engine time per request
WARMUP = 0.25                  # fraction of requests excluded from the metrics
POLICIES = ["LRU", "Unpartitioned Admission", "Static Partitioning",
            "Offline LFU", "TeamCache", "TeamCache + keep-set protection"]
BASELINES = POLICIES[:4]
BASE = {"cv": 0.5, "shared": 0.1, "growth": 0.01}
BYPASS = ("bypass",)
SWEEPS = {"cv": [0.0, 0.25, 0.5, 1.0],
          "shared": [0.0, 0.1, 0.3, 0.5],
          "growth": [0.0, 0.01, 0.05]}


def _blocks(tokens):
    return (int(tokens) // BLOCK) * BLOCK


def workload(topo, seed, cv, shared, growth):
    refs = O.topo_reference(topo, TEAMS, MEMBERS, ROUNDS, seed=seed)
    rng = random.Random(seed * 7919 + 17)
    if cv > 0:
        sigma = math.sqrt(math.log(1 + cv * cv))
        mu = math.log(MEAN_TOKENS) - sigma * sigma / 2
        init = {(t, m): max(BLOCK, rng.lognormvariate(mu, sigma))
                for t in range(TEAMS) for m in range(MEMBERS)}
    else:
        init = {(t, m): float(MEAN_TOKENS) for t in range(TEAMS) for m in range(MEMBERS)}
    template = _blocks(shared * MEAN_TOKENS)
    step = growth * MEAN_TOKENS
    # Prompt length of every request, and the mid-run working set that defines load.
    size, prompts = dict(init), []
    mid_ws = mid_prompt = None
    for i, key in enumerate(refs):
        prompts.append(_blocks(size[key]))
        size[key] += step
        if i == len(refs) // 2:
            mid_ws = template + sum(_blocks(v) for v in size.values())
            mid_prompt = template + float(np.mean([_blocks(v) for v in size.values()]))
    return refs, prompts, template, mid_ws, mid_prompt


def static_quota(capacity, mid_prompt):
    share = max(capacity - TEAMS * mid_prompt, mid_prompt) / TEAMS
    return max(1, int(share // mid_prompt))


def run(policy, refs, prompts, template, capacity, mid_prompt, freq_rank,
        rule_kw=None, size_noise=0.0, pool_scale=1.0, seed=0, labels=None,
        inflight=TEAMS):
    """Replay one request sequence; returns per-request (team, reused, prompt).

    `rule_kw`, `size_noise` (log-normal sigma on the prompt length TeamCache
    is told) and `pool_scale` (misreported pool) exist for the sensitivity
    analysis; the defaults are the deployed configuration.  `labels`, one
    (team, member) pair or None per request, is what TeamCache is told instead
    of the true identity: None is a request without an id, which the engine
    serves on the upstream path, and BYPASS one that is never inserted.
    """
    resident = defaultdict(int)          # session -> resident private tokens
    access = {}                          # session -> last access index
    tpl_resident = False
    recent = []                          # prompts in flight (one per team)
    out = []
    q = static_quota(capacity, mid_prompt)
    admitted_global, global_cap = set(), TEAMS * q
    rule = (EngineRule(int(capacity * pool_scale), **(rule_kw or {}))
            if policy.startswith("TeamCache") else None)
    noise = np.random.default_rng(seed + 1009)
    protect = policy.endswith("protection")
    pinned = {}                          # team -> keep-set sessions
    turns = Counter()
    for i, ((t, m), p) in enumerate(zip(refs, prompts)):
        key = (t, m)
        total = template + p
        reused = (template if tpl_resident else 0) + min(resident[key], p)
        out.append((t, reused, total))
        lab = key if labels is None else labels[i]
        if lab is not None:
            turn = turns[lab]
            turns[lab] += 1

        if policy == "LRU":
            skip = False
        elif policy == "Unpartitioned Admission":
            if key in admitted_global:
                skip = False
            elif len(admitted_global) < global_cap:
                admitted_global.add(key)
                skip = False
            else:
                skip = True
        elif policy == "Static Partitioning":
            skip = m >= q
        elif policy == "Offline LFU":
            skip = freq_rank[t][m] >= q
        else:
            told = total * float(np.exp(noise.normal(0, size_noise))) if size_noise else total
            if lab is None:
                skip = False
            elif lab == BYPASS:
                skip = True
            else:
                skip = rule.skip(lab[0], lab[1], turn, told, i * DT)
            want = rule.want.get(f"t{t}")
            if protect and want is not None:
                pinned[t] = {(t, w) for w in want}

        if resident[key] > 0:
            access[key] = i
        if not skip:
            resident[key] = p
            access[key] = i
            tpl_resident = tpl_resident or template > 0

        recent.append(total)
        if len(recent) > inflight:
            recent.pop(0)
        limit = capacity - sum(recent)
        used = (template if tpl_resident else 0) + sum(resident.values())
        while used > limit:
            live = [k for k, v in resident.items() if v > 0]
            if not live:
                if tpl_resident:
                    tpl_resident = False
                    used -= template
                    continue
                break
            free = [k for k in live if k not in pinned.get(k[0], ())]
            victim = min(free or live, key=lambda k: access.get(k, -1))
            cut = min(resident[victim], _blocks(used - limit) + BLOCK)
            resident[victim] -= cut
            used -= cut
    return out


def metrics(rows):
    skip = int(len(rows) * WARMUP)
    per = defaultdict(lambda: [0, 0, 0, 0])
    for t, reused, total in rows[skip:]:
        r = per[t]
        r[0] += reused
        r[1] += total
        r[2] += reused >= 0.5 * total
        r[3] += 1
    tok = {t: r[0] / r[1] for t, r in per.items()}
    hit = {t: r[2] / r[3] for t, r in per.items()}
    agg_tok = sum(r[0] for r in per.values()) / sum(r[1] for r in per.values())
    agg_hit = sum(r[2] for r in per.values()) / sum(r[3] for r in per.values())
    return {"tok": agg_tok, "tok_worst": min(tok.values()),
            "hit": agg_hit, "hit_worst": min(hit.values())}


def _job(args):
    topo, seed, cv, shared, growth = args
    refs, prompts, template, mid_ws, mid_prompt = workload(topo, seed, cv, shared, growth)
    counts = Counter(refs)
    freq_rank = {t: {m: r for r, m in enumerate(sorted(range(MEMBERS),
                                                       key=lambda m: (-counts[(t, m)], m)))}
                 for t in range(TEAMS)}
    rows = []
    for load in LOADS:
        capacity = int(mid_ws / load + TEAMS * mid_prompt)
        for pol in POLICIES:
            m = metrics(run(pol, refs, prompts, template, capacity, mid_prompt, freq_rank))
            rows.append({"topo": topo, "seed": seed, "cv": cv, "shared": shared,
                         "growth": growth, "load": load, "capacity_tokens": capacity,
                         "policy": pol, **m})
    return rows


def configs():
    seen = {(0.0, 0.0, 0.0)}
    yield (0.0, 0.0, 0.0)      # equal sizes, no template, no growth
    for factor, values in SWEEPS.items():
        for v in values:
            cfg = dict(BASE, **{factor: v})
            key = (cfg["cv"], cfg["shared"], cfg["growth"])
            if key not in seen:
                seen.add(key)
                yield key


def summarise(rows):
    cells = defaultdict(list)
    for r in rows:
        cells[(r["cv"], r["shared"], r["growth"], r["topo"], r["load"], r["policy"])].append(r)
    out = []
    for (cv, sh, gr, topo, load, pol), rs in sorted(cells.items()):
        out.append({"cv": cv, "shared": sh, "growth": gr, "topo": topo, "load": load,
                    "policy": pol, "n": len(rs),
                    **{k: float(np.mean([r[k] for r in rs]))
                       for k in ("tok", "tok_worst", "hit", "hit_worst")}})
    return out


def headline(cells):
    """Per configuration: cells where each policy's worst team falls below 10%
    token reuse, and TeamCache's median gap to the best baseline per cell."""
    by_cfg = defaultdict(list)
    for c in cells:
        by_cfg[(c["cv"], c["shared"], c["growth"])].append(c)
    res = []
    for (cv, sh, gr), cs in sorted(by_cfg.items()):
        grid = defaultdict(dict)
        for c in cs:
            grid[(c["topo"], c["load"])][c["policy"]] = c
        gaps, wins, agg = defaultdict(list), Counter(), defaultdict(list)
        for cell in grid.values():
            best = max(cell[p]["tok_worst"] for p in BASELINES)
            for tc in ("TeamCache", "TeamCache + keep-set protection"):
                gap = cell[tc]["tok_worst"] - best
                gaps[tc].append(gap)
                wins[tc] += gap >= -0.02
            agg_best = max(cell[p]["tok"] for p in BASELINES)
            for tc in ("TeamCache", "TeamCache + keep-set protection"):
                agg[tc].append(cell[tc]["tok"] - agg_best)
        res.append({
            "cv": cv, "shared": sh, "growth": gr, "cells": len(grid),
            "starved": {p: sum(cell[p]["tok_worst"] < 0.10 for cell in grid.values())
                        for p in POLICIES},
            "min_worst": {p: min(cell[p]["tok_worst"] for cell in grid.values())
                          for p in POLICIES},
            **{f"{tag}_worst_gap_median_pp": 100 * float(np.median(gaps[tc]))
               for tag, tc in (("tc", "TeamCache"), ("tcp", "TeamCache + keep-set protection"))},
            **{f"{tag}_worst_gap_min_pp": 100 * float(np.min(gaps[tc]))
               for tag, tc in (("tc", "TeamCache"), ("tcp", "TeamCache + keep-set protection"))},
            **{f"{tag}_within_2pp": wins[tc]
               for tag, tc in (("tc", "TeamCache"), ("tcp", "TeamCache + keep-set protection"))},
            **{f"{tag}_agg_gap_median_pp": 100 * float(np.median(agg[tc]))
               for tag, tc in (("tc", "TeamCache"), ("tcp", "TeamCache + keep-set protection"))},
        })
    return res


def overall_summary(cells):
    """Counts over every (configuration, topology, load) cell."""
    grid = defaultdict(dict)
    for c in cells:
        grid[(c["cv"], c["shared"], c["growth"], c["topo"], c["load"])][c["policy"]] = c
    out = {"cells": len(grid)}
    for tc in ("TeamCache", "TeamCache + keep-set protection"):
        agg_ok = worst_ok = 0
        per_topo = defaultdict(list)
        for (_, _, _, topo, _), cell in grid.items():
            agg_best = max(cell[p]["tok"] for p in BASELINES)
            worst_best = max(cell[p]["tok_worst"] for p in BASELINES)
            agg_ok += cell[tc]["tok"] >= agg_best - 0.02
            worst_ok += cell[tc]["tok_worst"] >= worst_best - 0.02
            per_topo[topo].append(100 * (cell[tc]["tok_worst"] - worst_best))
        out[tc] = {"aggregate_within_2pp": agg_ok, "worst_within_2pp": worst_ok,
                   "worst_gap_median_pp_by_topology":
                       {t: float(np.median(v)) for t, v in sorted(per_topo.items())}}
    for p in POLICIES:
        out.setdefault("starved_cells", {})[p] = sum(
            cell[p]["tok_worst"] < 0.10 for cell in grid.values())
    lru_gain = [100 * (cell["TeamCache"]["tok"] - cell["LRU"]["tok"]) for cell in grid.values()]
    out["teamcache_minus_lru_aggregate_pp"] = {
        "median": float(np.median(lru_gain)), "min": float(np.min(lru_gain)),
        "max": float(np.max(lru_gain))}
    return out


SENS = {"window_s": [10.0, 30.0, 60.0, 120.0],
        "adapt_window": [10, 20, 40, 80],
        "margin": [0.95, 1.0, 1.02, 1.1, 1.25],
        "min_samples": [1, 2, 4, 8],
        "size_noise": [0.0, 0.1, 0.25, 0.5],
        "pool_scale": [0.8, 0.9, 1.0, 1.1, 1.2]}
SENS_DEFAULT = {"window_s": 30.0, "adapt_window": 40, "margin": 1.02, "min_samples": 2,
                "size_noise": 0.0, "pool_scale": 1.0}
SENS_SEEDS = 3


def _sens_job(args):
    topo, seed, param, value = args
    cfg = dict(BASE)
    refs, prompts, template, mid_ws, mid_prompt = workload(topo, seed, cfg["cv"],
                                                           cfg["shared"], cfg["growth"])
    rule_kw = {k: SENS_DEFAULT[k] for k in ("window_s", "adapt_window", "margin", "min_samples")}
    extra = {"size_noise": SENS_DEFAULT["size_noise"], "pool_scale": SENS_DEFAULT["pool_scale"]}
    if param in rule_kw:
        rule_kw[param] = value
    else:
        extra[param] = value
    rows = []
    for load in LOADS:
        capacity = int(mid_ws / load + TEAMS * mid_prompt)
        m = metrics(run("TeamCache", refs, prompts, template, capacity, mid_prompt, None,
                        rule_kw=rule_kw, seed=seed, **extra))
        rows.append({"topo": topo, "seed": seed, "load": load, "param": param,
                     "value": value, **m})
    return rows


def sensitivity():
    """S13: the deployed rule's constants and inputs, one at a time."""
    jobs = [(topo, seed, param, v) for param, vals in SENS.items() for v in vals
            for topo in TOPOS for seed in range(SENS_SEEDS)]
    with ProcessPoolExecutor() as pool:
        rows = [r for chunk in pool.map(_sens_job, jobs, chunksize=4) for r in chunk]
    base = {(r["topo"], r["seed"], r["load"]): r for r in rows
            if r["param"] == "margin" and r["value"] == SENS_DEFAULT["margin"]}
    out = []
    for param, vals in SENS.items():
        for v in vals:
            rs = [r for r in rows if r["param"] == param and r["value"] == v]
            dw = [r["tok_worst"] - base[(r["topo"], r["seed"], r["load"])]["tok_worst"] for r in rs]
            da = [r["tok"] - base[(r["topo"], r["seed"], r["load"])]["tok"] for r in rs]
            out.append({"param": param, "value": v, "default": v == SENS_DEFAULT[param],
                        "n": len(rs), "worst_mean": float(np.mean([r["tok_worst"] for r in rs])),
                        "aggregate_mean": float(np.mean([r["tok"] for r in rs])),
                        "worst_delta_pp_mean": 100 * float(np.mean(dw)),
                        "worst_delta_pp_min": 100 * float(np.min(dw)),
                        "aggregate_delta_pp_mean": 100 * float(np.mean(da)),
                        "starved_cells": sum(r["tok_worst"] < 0.10 for r in rs)})
    doc = {"config": {"base_workload": BASE, "loads": LOADS, "seeds": SENS_SEEDS,
                      "topologies": TOPOS, "default": SENS_DEFAULT, "sweeps": SENS},
           "summary": out, "runs": rows}
    json.dump(doc, open(OUT / "s13_sensitivity.json", "w"), indent=1)
    for o in out:
        print(f"{o['param']:12s} {o['value']!s:6} worst={o['worst_mean']:.3f} "
              f"agg={o['aggregate_mean']:.3f} dworst={o['worst_delta_pp_mean']:+.1f}pp "
              f"(min {o['worst_delta_pp_min']:+.1f}) dagg={o['aggregate_delta_pp_mean']:+.1f}pp "
              f"starved={o['starved_cells']}/{o['n']}")


SENS = {"window_s": [10.0, 30.0, 60.0, 120.0],
        "adapt_window": [10, 20, 40, 80],
        "margin": [0.95, 1.0, 1.02, 1.1, 1.25],
        "min_samples": [1, 2, 4, 8],
        "size_noise": [0.0, 0.1, 0.25, 0.5],
        "pool_scale": [0.8, 0.9, 1.0, 1.1, 1.2]}
SENS_DEFAULT = {"window_s": 30.0, "adapt_window": 40, "margin": 1.02, "min_samples": 2,
                "size_noise": 0.0, "pool_scale": 1.0}
SENS_SEEDS = 3


def _sens_job(args):
    topo, seed, param, value = args
    cfg = dict(BASE)
    refs, prompts, template, mid_ws, mid_prompt = workload(topo, seed, cfg["cv"],
                                                           cfg["shared"], cfg["growth"])
    rule_kw = {k: SENS_DEFAULT[k] for k in ("window_s", "adapt_window", "margin", "min_samples")}
    extra = {"size_noise": SENS_DEFAULT["size_noise"], "pool_scale": SENS_DEFAULT["pool_scale"]}
    if param in rule_kw:
        rule_kw[param] = value
    else:
        extra[param] = value
    rows = []
    for load in LOADS:
        capacity = int(mid_ws / load + TEAMS * mid_prompt)
        m = metrics(run("TeamCache", refs, prompts, template, capacity, mid_prompt, None,
                        rule_kw=rule_kw, seed=seed, **extra))
        rows.append({"topo": topo, "seed": seed, "load": load, "param": param,
                     "value": value, **m})
    return rows


def sensitivity():
    """S13: the deployed rule's constants and inputs, one at a time."""
    jobs = [(topo, seed, param, v) for param, vals in SENS.items() for v in vals
            for topo in TOPOS for seed in range(SENS_SEEDS)]
    with ProcessPoolExecutor() as pool:
        rows = [r for chunk in pool.map(_sens_job, jobs, chunksize=4) for r in chunk]
    base = {(r["topo"], r["seed"], r["load"]): r for r in rows
            if r["param"] == "margin" and r["value"] == SENS_DEFAULT["margin"]}
    out = []
    for param, vals in SENS.items():
        for v in vals:
            rs = [r for r in rows if r["param"] == param and r["value"] == v]
            dw = [r["tok_worst"] - base[(r["topo"], r["seed"], r["load"])]["tok_worst"] for r in rs]
            da = [r["tok"] - base[(r["topo"], r["seed"], r["load"])]["tok"] for r in rs]
            out.append({"param": param, "value": v, "default": v == SENS_DEFAULT[param],
                        "n": len(rs), "worst_mean": float(np.mean([r["tok_worst"] for r in rs])),
                        "aggregate_mean": float(np.mean([r["tok"] for r in rs])),
                        "worst_delta_pp_mean": 100 * float(np.mean(dw)),
                        "worst_delta_pp_min": 100 * float(np.min(dw)),
                        "aggregate_delta_pp_mean": 100 * float(np.mean(da)),
                        "starved_cells": sum(r["tok_worst"] < 0.10 for r in rs)})
    doc = {"config": {"base_workload": BASE, "loads": LOADS, "seeds": SENS_SEEDS,
                      "topologies": TOPOS, "default": SENS_DEFAULT, "sweeps": SENS},
           "summary": out, "runs": rows}
    json.dump(doc, open(OUT / "s13_sensitivity.json", "w"), indent=1)
    for o in out:
        print(f"{o['param']:12s} {o['value']!s:6} worst={o['worst_mean']:.3f} "
              f"agg={o['aggregate_mean']:.3f} dworst={o['worst_delta_pp_mean']:+.1f}pp "
              f"(min {o['worst_delta_pp_min']:+.1f}) dagg={o['aggregate_delta_pp_mean']:+.1f}pp "
              f"starved={o['starved_cells']}/{o['n']}")


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if "--sensitivity" in argv:
        OUT.mkdir(parents=True, exist_ok=True)
        return sensitivity()
    if argv is None:
        argv = sys.argv[1:]
    if "--sensitivity" in argv:
        OUT.mkdir(parents=True, exist_ok=True)
        return sensitivity()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(topo, seed, cv, sh, gr) for cv, sh, gr in configs()
            for topo in TOPOS for seed in range(SEEDS)]
    with ProcessPoolExecutor() as pool:
        rows = [r for chunk in pool.map(_job, jobs, chunksize=2) for r in chunk]
    cells = summarise(rows)
    head = headline(cells)
    overall = overall_summary(cells)
    doc = {"config": {"teams": TEAMS, "members": MEMBERS, "rounds": ROUNDS,
                      "block": BLOCK, "mean_tokens": MEAN_TOKENS, "loads": LOADS,
                      "seeds": SEEDS, "base": BASE, "sweeps": SWEEPS,
                      "warmup_fraction": WARMUP, "dt_s": DT},
           "overall": overall, "headline": head, "cells": cells}
    json.dump(doc, open(OUT / "s11_block_level.json", "w"), indent=1)
    print(json.dumps(overall, indent=1))
    for h in head:
        print(f"cv={h['cv']:<4} shared={h['shared']:<4} growth={h['growth']:<5} "
              f"starved={list(h['starved'].values())} "
              f"worst gap med/min: TC {h['tc_worst_gap_median_pp']:+.1f}/{h['tc_worst_gap_min_pp']:+.1f} "
              f"TC+P {h['tcp_worst_gap_median_pp']:+.1f}/{h['tcp_worst_gap_min_pp']:+.1f} "
              f"agg gap TC {h['tc_agg_gap_median_pp']:+.1f} TC+P {h['tcp_agg_gap_median_pp']:+.1f}")


if __name__ == "__main__":
    sys.exit(main())
