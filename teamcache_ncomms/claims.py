#!/usr/bin/env python3
"""Recompute every number quoted in the paper from the data and compare.

    python -m teamcache_ncomms.claims            # check, exit 1 on mismatch
    python -m teamcache_ncomms.claims --csv F    # also write the claim table

Each claim names the paper location, the value as printed, the tolerance
implied by its rounding, the files it is computed from and how.  Claims
reproduced by a separate command are listed with status "by command" (CPU)
or "GPU probe" (a probe script run on the GPU server).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import paths
from .gpu_tables import (MIXED_KEY, SEEDS, extract_mixed, extract_reset, extract_robustness,
                         extract_star, extract_sweep, largest_n, load_jsonl, loadj, mixed_paths,
                         split_meas)

T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
CLAIMS = []


def claim(cid, where, text, expected, tol, sources, how=""):
    def deco(fn):
        CLAIMS.append(dict(id=cid, where=where, text=text, expected=expected, tol=tol,
                           sources=sources, how=how or fn.__doc__ or "", fn=fn))
        return fn
    return deco


def missing(cid, where, text, expected, note):
    status = "by command" if note.startswith("reproduced by") else "GPU probe"
    CLAIMS.append(dict(id=cid, where=where, text=text, expected=expected, tol=None,
                       sources="", how=note, fn=None, status=status))


def sim(name):
    return loadj(paths.cache_level() / f"{name}.json")


def sweep(policy_arm, teams):
    return loadj(paths.SWEEP / f"{policy_arm}_{teams}t_ring.json")


def e2e(name):
    return loadj(paths.E2E / f"{name}.json")


def paired(a, b):
    """Mean paired difference a - b and two-sided 95% t interval."""
    d = np.asarray(a, float) - np.asarray(b, float)
    h = T975[len(d)] * d.std(ddof=1) / math.sqrt(len(d))
    return d.mean(), d.mean() - h, d.mean() + h


_MIXED = None


def mixed():
    global _MIXED
    if _MIXED is None:
        _MIXED = extract_mixed()
    return _MIXED


def mixed_vec(policy, field):
    return [next(r for r in mixed() if r["seed"] == s and r["policy"] == policy)[field]
            for s in SEEDS]


def ring_runs(arm):
    hits, spans = [], []
    for s in ("", "-s1", "-s2"):
        d = e2e(f"e2e-autogen-design-{arm}-20t-p3000r8{s}")
        hits.append(100 * d["steady"]["prefix_hit_rate"])
        spans.append(d["steady"]["span_s"])
    return hits, spans


# ------------------------------------------------------------------ capacity cliff

@claim("cliff_kimi_8t", "Intro; Results 'Rings create a capacity cliff'; Fig. 2a",
       "eight teams: 82% request hit rate (LRU, Kimi K3)", 82, 0.5,
       "gpu/results/team-sweep/lru_8t_ring.json")
def _():
    """1 - turn_miss_rate"""
    return 100 * (1 - sweep("lru", 8)["turn_miss_rate"])


@claim("cliff_kimi_10t", "Intro; Results; Fig. 2a",
       "ten teams: 0.4% request hit rate (LRU, Kimi K3)", 0.4, 0.05,
       "gpu/results/team-sweep/lru_10t_ring.json")
def _():
    """1 - turn_miss_rate"""
    return 100 * (1 - sweep("lru", 10)["turn_miss_rate"])


def _turn(d):
    """median completion time of a team divided by its sequential turns"""
    return d["ttta_p50"] / (d["steps"] / d["teams"])


for cid, text, f, val, tol in (
        ("turn_lru_10t", "ten teams: LRU turn 6.1 s", lambda: _turn(sweep("lru", 10)), 6.1, 0.05),
        ("turn_tc_10t", "ten teams: TeamCache turn 4.0 s", lambda: _turn(sweep("unified", 10)), 4.0, 0.05),
        ("ttft_lru_10t", "ten teams: LRU median time to first token 3.2 s",
         lambda: sweep("lru", 10)["ttft_p50"], 3.2, 0.05),
        ("ttft_tc_10t", "ten teams: TeamCache median time to first token 0.5 s",
         lambda: sweep("unified", 10)["ttft_p50"], 0.5, 0.05),
        ):
    claim(cid, "Discussion", text, val, tol, "gpu/results/team-sweep/{lru,unified}_10t_ring.json")(f)


def _ws(d):
    return d["teams"] * d["members"] * (d["context"] + 512 + d["gen_tokens"])


@claim("cliff_ws_8t", "Intro", "working set 0.42 million tokens at eight teams", 0.42, 0.005,
       "gpu/results/team-sweep/lru_8t_ring.json")
def _():
    """teams x members x (context + 512 + generated tokens)"""
    return _ws(sweep("lru", 8)) / 1e6


@claim("cliff_ws_10t", "Intro", "working set 0.52 million tokens at ten teams", 0.52, 0.005,
       "gpu/results/team-sweep/lru_10t_ring.json")
def _():
    """teams x members x (context + 512 + generated tokens)"""
    return _ws(sweep("lru", 10)) / 1e6


@claim("cliff_excess", "Intro; Results; Fig. 2a", "ten-team working set exceeds the pool by 23%",
       23, 0.5, "gpu/results/team-sweep/lru_10t_ring.json")
def _():
    """working set / 420,800-token pool - 1"""
    return 100 * (_ws(sweep("lru", 10)) / 420_800 - 1)


def _glm_rows():
    rows = [e2e(f"e2e-ring-lru-{t}t-p56k-glm") for t in (6, 8, 10, 12, 16)]
    return rows


def _glm_ws(d):
    return d["teams"] * d["members"] * (d["context"] + 512 + d["gen_tokens"])


@claim("cliff_glm_hi", "Results; Fig. 2a", "GLM 5.3: 100% at 2.27 million tokens", 100, 0.5,
       "gpu/results/e2e/e2e-ring-lru-{6,8,10,12,16}t-p56k-glm.json")
def _():
    """highest hit rate among working sets <= pool"""
    rows = _glm_rows()
    return max(100 * (1 - d["turn_miss_rate"]) for d in rows if _glm_ws(d) < 2.3e6)


@claim("cliff_glm_ws_lo", "Results", "GLM 5.3 working set 2.27 million tokens (eight teams)",
       2.27, 0.005, "gpu/results/e2e/e2e-ring-lru-8t-p56k-glm.json")
def _():
    """teams x members x (context + 512 + generated tokens)"""
    return _glm_ws(e2e("e2e-ring-lru-8t-p56k-glm")) / 1e6


@claim("cliff_glm_lo", "Results; Fig. 2a", "GLM 5.3: 11% at 4.53 million tokens", 11, 0.5,
       "gpu/results/e2e/e2e-ring-lru-16t-p56k-glm.json")
def _():
    """1 - turn_miss_rate at sixteen teams"""
    return 100 * (1 - e2e("e2e-ring-lru-16t-p56k-glm")["turn_miss_rate"])


@claim("cliff_glm_ws_hi", "Results", "GLM 5.3 working set 4.53 million tokens (sixteen teams)",
       4.53, 0.005, "gpu/results/e2e/e2e-ring-lru-16t-p56k-glm.json")
def _():
    """teams x members x (context + 512 + generated tokens)"""
    return _glm_ws(e2e("e2e-ring-lru-16t-p56k-glm")) / 1e6


@claim("ring_closed_form_mae", "Results; Supplementary Fig. 1",
       "interleaved rings match the closed form within 0.9 points (MAE)", 0.9, 0.05,
       "cache_level/s2_cliff.json")
def _():
    """summary.ring_closed_mae"""
    return 100 * sim("s2_cliff")["summary"]["ring_closed_mae"]


@claim("che_r2", "Results; Fig. 2d", "characteristic-time model R^2 = 0.9997", 0.9997, 0.00005,
       "cache_level/s2_cliff.json")
def _():
    """summary.topology_che_r2"""
    return sim("s2_cliff")["summary"]["topology_che_r2"]


@claim("che_points", "Results; Fig. 2d", "378 configurations", 378, 0, "cache_level/s2_cliff.json")
def _():
    """summary.topology_points"""
    return sim("s2_cliff")["summary"]["topology_points"]


# ------------------------------------------------------------------ dilemma

@claim("phase_agreement", "Results; Fig. 2e", "predicted winner agrees in 60 of 66 conditions",
       60, 0, "cache_level/s3_phase.json")
def _():
    """headline.winner_prediction_agreement"""
    return sim("s3_phase")["headline"]["winner_prediction_agreement"]


def _phase_grid():
    cells = sim("s3_phase")["cells"]
    lru = np.array([c["LRU"] for c in cells])
    st = np.array([c["Static"] for c in cells])
    tc = np.array([c["TC"] for c in cells])
    return cells, lru, st, tc


@claim("phase_gap_cells", "Results; Supplementary Fig. 2d",
       "62 conditions in which the fixed policies differ by more than three points", 62, 0,
       "cache_level/s3_phase.json")
def _():
    """count of cells with |LRU - Static| > 0.03"""
    _, lru, st, _ = _phase_grid()
    return int((np.abs(lru - st) > 0.03).sum())


@claim("phase_gap_recovered", "Results; Supplementary Fig. 2d", "TeamCache recovers a median 93% of the gap",
       93, 0.5, "cache_level/s3_phase.json")
def _():
    """median of (TC - worse) / (better - worse) over cells with a gap > 3 points"""
    _, lru, st, tc = _phase_grid()
    span = np.abs(lru - st)
    m = span > 0.03
    return 100 * float(np.median((tc[m] - np.minimum(lru, st)[m]) / span[m]))


@claim("phase_below_worse", "Results", "never more than 0.5 points below the worse fixed policy",
       0.5, 0.05, "cache_level/s3_phase.json")
def _():
    """max over all 66 cells of (worse fixed policy - TC), in points"""
    _, lru, st, tc = _phase_grid()
    return 100 * float(np.max(np.minimum(lru, st) - tc))


@claim("phase_max_shortfall", "Results", "largest shortfall 10.7 points", 10.7, 0.05,
       "cache_level/s3_phase.json")
def _():
    """-headline.tc_gap_to_best_min_pp"""
    return -sim("s3_phase")["headline"]["tc_gap_to_best_min_pp"]


for pol, key, n in (("LRU", "LRU", 10), ("Unpartitioned Admission", "Blind", 8),
                    ("Static Partitioning", "Static", 5), ("TeamCache", "TC", 0)):
    claim(f"grid_below5_{key}", "Results; Fig. 4c",
          f"{pol} leaves the worst team below 5% in {n} of 36 configurations", n, 0,
          "cache_level/s1_grid.json", "headline.<policy>.cells_below_5pct")(
        lambda key=key: sim("s1_grid")["headline"][key]["cells_below_5pct"])


@claim("grid_tc_floor", "Intro; Results; Fig. 4c", "TeamCache keeps every team above 12.8%",
       12.8, 0.05, "cache_level/s1_grid.json")
def _():
    """headline.TC.min_cell_worst"""
    return 100 * sim("s1_grid")["headline"]["TC"]["min_cell_worst"]


# ------------------------------------------------------------------ classical policies

for pol, n in (("ARC", 7), ("LIRS", 8), ("2Q", 8), ("S3-FIFO", 12), ("W-TinyLFU", 5)):
    claim(f"classic_below5_{pol}", "Results 'TeamCache tracks the better policy and protects the worst-served team'; Fig. 4c",
          f"{pol} leaves some team below 5% in {n} (>= 5) of 36 configurations", n, 0,
          "cache_level/s6_baselines.json", "headline.<policy>.grid_cells_worst_below_5pct")(
        lambda pol=pol: sim("s6_baselines")["headline"][pol]["grid_cells_worst_below_5pct"])


def _classic_topo(policy, topo):
    s6 = sim("s6_baselines")
    caps = sorted({c["cap"] for c in s6["grid"]})
    over = [c for c in s6["grid"] if c["policy"] == policy and c["topo"] == topo
            and 63 / c["cap"] > 1.0]
    assert over and caps
    return 100 * float(np.mean([c["hit_mean"] for c in over]))


for cid, pol, topo, val in (("classic_ring_LRU", "LRU", "ring", 3),
                            ("classic_ring_WTinyLFU", "W-TinyLFU", "ring", 28),
                            ("classic_ring_ARC", "ARC", "ring", 8),
                            ("classic_ring_TC", "TC", "ring", 32),
                            ("classic_free_LRU", "LRU", "empirical", 57),
                            ("classic_free_TC", "TC", "empirical", 57)):
    claim(cid, "Results; Fig. 4b", f"{pol} mean hit rate on {topo} teams {val}%", val, 0.5,
          "cache_level/s6_baselines.json",
          "mean hit rate over the six oversubscribed loads")(
        lambda pol=pol, topo=topo: _classic_topo(pol, topo))


@claim("classic_free_WTinyLFU_loss", "Results; Fig. 4b",
       "W-TinyLFU loses 10 points on free-form teams", 10, 0.5, "cache_level/s6_baselines.json")
def _():
    """LRU minus W-TinyLFU on free-form teams, oversubscribed loads"""
    return _classic_topo("LRU", "empirical") - _classic_topo("W-TinyLFU", "empirical")


@claim("public_traces", "Results; Fig. 1b, 5c", "548 public traces", 548, 0,
       "cache_level/s8_public_traces.json")
def _():
    """n_kept"""
    return sim("s8_public_traces")["n_kept"]


def _public_delta():
    rows = sim("s8_public_traces")["replay"]
    lru = {(r["system"], r["load"]): r["hit_mean"] for r in rows if r["policy"] == "LRU"}
    out = defaultdict(list)
    for r in rows:
        out[r["policy"]].append(100 * (r["hit_mean"] - lru[(r["system"], r["load"])]))
    return out


@claim("public_freq_loss", "Results; Fig. 4d", "frequency-aware policies lose up to 23 points",
       23, 0.5, "cache_level/s8_public_traces.json")
def _():
    """max loss vs LRU over ARC, LIRS, S3-FIFO and W-TinyLFU"""
    d = _public_delta()
    return -min(min(d[p]) for p in ("ARC", "LIRS", "S3-FIFO", "W-TinyLFU"))


@claim("public_static_loss", "Results; Fig. 4d", "Static Partitioning loses up to 31 points", 31,
       0.5, "cache_level/s8_public_traces.json")
def _():
    """max loss vs LRU"""
    return -min(_public_delta()["Static"])


@claim("public_tc_loss", "Intro; Results; Fig. 4d", "TeamCache never more than 0.1 points below LRU",
       0.1, 0.05, "cache_level/s8_public_traces.json")
def _():
    """max loss vs LRU"""
    return max(0.0, -min(_public_delta()["TeamCache"]))


@claim("public_tc_gain", "Results; Fig. 4d", "TeamCache gains up to 3.6 points (Magentic-One)",
       3.6, 0.05, "cache_level/s8_public_traces.json")
def _():
    """max gain vs LRU"""
    return max(_public_delta()["TeamCache"])


for pol, val in (("LIRS", 78), ("W-TinyLFU", 76), ("TC", 70)):
    claim(f"replay_{pol}", "Results; Methods", f"recorded GPU order replayed: {pol} {val}%", val,
          0.5, "cache_level/s7_trace_replay.json", "summary.<policy>.request_hit")(
        lambda pol=pol: 100 * sim("s7_trace_replay")["summary"][pol]["request_hit"])


@claim("replay_LRU", "Methods", "LRU reaches 39.9% in the replay", 39.9, 0.05,
       "cache_level/s7_trace_replay.json")
def _():
    """summary.LRU.request_hit"""
    return 100 * sim("s7_trace_replay")["summary"]["LRU"]["request_hit"]


@claim("replay_gpu_LRU", "Methods", "41.4% measured on the GPUs", 41.4, 0.05,
       "cache_level/s7_trace_replay.json; gpu/mixed/mixed_lru_12t*.requests.jsonl")
def _():
    """summary['GPU LRU'].request_hit"""
    return 100 * sim("s7_trace_replay")["summary"]["GPU LRU"]["request_hit"]


# ------------------------------------------------------------------ GPU: ring and selector

RING = "gpu/results/e2e/e2e-autogen-design-{lru,cohort,unified}-20t-p3000r8{,-s1,-s2}.json"


@claim("ring_lru_reuse", "Intro; Results; Fig. 5a", "LRU reuses 1.2% on the twenty-team ring",
       1.2, 0.05, RING)
def _():
    """mean steady-state prefix hit rate over three runs"""
    return float(np.mean(ring_runs("lru")[0]))


@claim("ring_tc_reuse", "Intro; Results; Fig. 5a", "TeamCache raises reuse to 35.5%", 35.5, 0.05,
       RING)
def _():
    """mean steady-state prefix hit rate over three runs"""
    return float(np.mean(ring_runs("unified")[0]))


@claim("ring_lru_time", "Results; Fig. 5a", "LRU completes in 185 s", 185, 0.5, RING)
def _():
    """mean steady-state span"""
    return float(np.mean(ring_runs("lru")[1]))


@claim("ring_tc_time", "Results; Fig. 5a", "TeamCache completes in 150 s", 150, 0.5, RING)
def _():
    """mean steady-state span"""
    return float(np.mean(ring_runs("unified")[1]))


@claim("ring_speedup", "Abstract-level; Intro; Results; Fig. 5a", "1.24-fold speedup", 1.24,
       0.005, RING)
def _():
    """mean LRU span / mean TeamCache span"""
    return float(np.mean(ring_runs("lru")[1]) / np.mean(ring_runs("unified")[1]))


@claim("ring_paired_diff", "Results", "paired difference 35.3 s", 35.3, 0.05, RING)
def _():
    """mean of per-seed LRU - TeamCache span"""
    return paired(ring_runs("lru")[1], ring_runs("unified")[1])[0]


@claim("ring_paired_lo", "Results", "95% CI lower bound 20.8 s", 20.8, 0.05, RING)
def _():
    """t interval, n = 3"""
    return paired(ring_runs("lru")[1], ring_runs("unified")[1])[1]


@claim("ring_paired_hi", "Results", "95% CI upper bound 49.8 s", 49.8, 0.05, RING)
def _():
    """t interval, n = 3"""
    return paired(ring_runs("lru")[1], ring_runs("unified")[1])[2]


SEL = "gpu/results/e2e/e2e-selector-debug-{lru,cohort,unified}-20t-p3000r8.json"
for cid, arm, field, val, tol in (("sel_static_reuse", "cohort", "hit", 30.7, 0.05),
                                  ("sel_static_time", "cohort", "wall", 546, 0.5),
                                  ("sel_tc_reuse", "unified", "hit", 59.0, 0.05),
                                  ("sel_tc_time", "unified", "wall", 416, 0.5),
                                  ("sel_lru_reuse", "lru", "hit", 58.4, 0.05),
                                  ("sel_lru_time", "lru", "wall", 414, 0.5)):
    claim(cid, "Results; Fig. 5b", f"selector workflow, {arm}: {val}", val, tol, SEL,
          "steady prefix hit rate / wall time")(
        lambda arm=arm, field=field: (
            100 * e2e(f"e2e-selector-debug-{arm}-20t-p3000r8")["steady"]["prefix_hit_rate"]
            if field == "hit" else e2e(f"e2e-selector-debug-{arm}-20t-p3000r8")["wall_s"]))


# ------------------------------------------------------------------ GPU: mixed workload

MIX = "gpu/mixed/mixed_{lru,group_blind,static_rate,teamcache}_12t{,_s12,_s13,_s14}.{json,requests.jsonl}"
for pol, reuse, time in (("LRU", 38.8, 148.6), ("Unpartitioned Admission", 33.3, 155.3),
                         ("Offline LFU", 48.5, 135.4), ("TeamCache", 53.8, 128.7)):
    claim(f"mixed_reuse_{MIXED_KEY[pol]}", "Intro; Results; Fig. 5c",
          f"mixed workload, {pol}: {reuse}% prefix-token reuse", reuse, 0.05, MIX,
          "mean over seeds 11-14 of measured-phase token reuse")(
        lambda pol=pol: 100 * float(np.mean(mixed_vec(pol, "token_reuse"))))
    claim(f"mixed_time_{MIXED_KEY[pol]}", "Results; Fig. 5c",
          f"mixed workload, {pol}: {time} s", time, 0.05, MIX, "mean wall time over seeds 11-14")(
        lambda pol=pol: float(np.mean(mixed_vec(pol, "makespan_s"))))


@claim("mixed_cut", "Intro; Results; Fig. 5c,d", "13.4% shorter completion time than LRU", 13.4,
       0.05, MIX)
def _():
    """1 - mean TeamCache time / mean LRU time"""
    return 100 * (1 - np.mean(mixed_vec("TeamCache", "makespan_s"))
                  / np.mean(mixed_vec("LRU", "makespan_s")))


for cid, base, idx, val in (("mixed_vs_lru_diff", "LRU", 0, 19.9),
                            ("mixed_vs_lru_lo", "LRU", 1, 12.4),
                            ("mixed_vs_lru_hi", "LRU", 2, 27.4),
                            ("mixed_vs_lfu_diff", "Offline LFU", 0, 6.7),
                            ("mixed_vs_lfu_lo", "Offline LFU", 1, 1.8),
                            ("mixed_vs_lfu_hi", "Offline LFU", 2, 11.6)):
    claim(cid, "Results; Fig. 5d", f"paired time saved vs {base}: {val} s", val, 0.05, MIX,
          "paired difference, two-sided 95% t interval, n = 4")(
        lambda base=base, idx=idx: paired(mixed_vec(base, "makespan_s"),
                                          mixed_vec("TeamCache", "makespan_s"))[idx])


@claim("mixed_every_run_lru", "Results", "faster than LRU in every paired run", 4, 0, MIX)
def _():
    """number of seeds with TeamCache time < LRU time"""
    return sum(t < l for t, l in zip(mixed_vec("TeamCache", "makespan_s"),
                                     mixed_vec("LRU", "makespan_s")))


@claim("mixed_every_run_lfu", "Intro; Results", "faster than Offline LFU in every run", 4, 0, MIX)
def _():
    """number of seeds with TeamCache time < Offline LFU time"""
    return sum(t < l for t, l in zip(mixed_vec("TeamCache", "makespan_s"),
                                     mixed_vec("Offline LFU", "makespan_s")))


def _per_team(policy, seed):
    jp, _ = mixed_paths(MIXED_KEY[policy], seed)
    return sorted(loadj(jp)["per_team"], key=lambda t: t["team"])


@claim("fair_low_reuse_UA", "Results; Fig. 6b", "Unpartitioned Admission leaves five of twelve "
       "teams with a miss rate >= 90% (maximum over runs)", 5, 0, MIX)
def _():
    """max over seeds of teams with miss_rate >= 0.9"""
    return max(sum(t["miss_rate"] >= 0.9 for t in _per_team("Unpartitioned Admission", s))
               for s in SEEDS)


@claim("fair_low_reuse_LRU", "Results; Fig. 6b", "LRU leaves two", 2, 0, MIX)
def _():
    """max over seeds of teams with miss_rate >= 0.9"""
    return max(sum(t["miss_rate"] >= 0.9 for t in _per_team("LRU", s)) for s in SEEDS)


@claim("fair_low_reuse_TC", "Results; Fig. 6b", "TeamCache at most one", 1, 0, MIX)
def _():
    """max over seeds of teams with miss_rate >= 0.9"""
    return max(sum(t["miss_rate"] >= 0.9 for t in _per_team("TeamCache", s)) for s in SEEDS)


def _slots(policy):
    """Team-member slots with pooled prefix-token reuse of at least 0.1."""
    cached, prompt = defaultdict(int), defaultdict(int)
    for s in SEEDS:
        _, lp = mixed_paths(MIXED_KEY[policy], s)
        _, meas = split_meas(load_jsonl(lp))
        for r in meas:
            cached[(r["team"], r["member"])] += r.get("cached_tokens") or 0
            prompt[(r["team"], r["member"])] += r.get("prompt_tokens") or 0
    return sum(cached[k] / prompt[k] >= 0.1 for k in prompt if prompt[k])


for pol, val in (("TeamCache", 48), ("LRU", 30), ("Unpartitioned Admission", 26),
                 ("Offline LFU", 37)):
    claim(f"fair_slots_{MIXED_KEY[pol]}", "Results; Fig. 6c",
          f"{pol}: {val} of 60 team-member slots retain >= 10% reuse", val, 0, MIX,
          "pooled over seeds, measured phase")(lambda pol=pol: _slots(pol))


# ------------------------------------------------------------------ capacity sweep

SW = "gpu/results/team-sweep/{lru,bypass,cohort,unified}_{6..16}t_ring.json"


@claim("cap_tc_10t", "Results; Fig. 7a", "TeamCache retains about 60% at ten teams", 60, 2, SW)
def _():
    """1 - turn_miss_rate"""
    return 100 * (1 - sweep("unified", 10)["turn_miss_rate"])


@claim("cap_tc_12t", "Results; Fig. 7a", "TeamCache retains 41.7% at twelve teams", 41.7, 0.05, SW)
def _():
    """1 - turn_miss_rate"""
    return 100 * (1 - sweep("unified", 12)["turn_miss_rate"])


for cid, pol, budget, val in (("cap_120_lru", "LRU", 120, 8), ("cap_120_tc", "TeamCache", 120, 10),
                              ("cap_150_lru", "LRU", 150, 8), ("cap_150_tc", "TeamCache", 150, 12)):
    claim(cid, "Intro; Results; Fig. 7c", f"{pol} supports {val} teams under {budget} s", val, 0,
          SW, "largest team count whose wall time meets the budget")(
        lambda pol=pol, budget=budget: largest_n(extract_sweep(), pol, budget))


for t, val in ((12, 3), (14, 8), (16, 10)):
    claim(f"cap_ua_low_{t}t", "Results", f"Unpartitioned Admission: {val} low-reuse teams at {t}",
          val, 0, SW, "starved_teams")(lambda t=t: sweep("bypass", t)["starved_teams"])
    claim(f"cap_tc_low_{t}t", "Results", f"TeamCache: no low-reuse team at {t}", 0, 0, SW,
          "starved_teams")(lambda t=t: sweep("unified", t)["starved_teams"])


# ------------------------------------------------------------------ hierarchy, models, engines

_ROB = None


def rob():
    global _ROB
    if _ROB is None:
        _ROB = extract_robustness()
    return _ROB


HI = "gpu/results/e2e/e2e-*-20t-p3000r8-hicache.json; e2e-fleet-mixed-*-10t-hicache-joint.json"
for cid, f, val, tol in (
        ("hicache_short_lru", lambda: 100 * rob()["short_lru_steady"], 90.3, 0.05),
        ("hicache_short_gpu", lambda: 100 * rob()["short_gpu_steady"], 1.1, 0.05),
        ("hicache_long_lru_reuse", lambda: 100 * rob()["hicache"]["lru_cached"], 25.2, 0.05),
        ("hicache_long_tc_reuse", lambda: 100 * rob()["hicache"]["tc_cached"], 38.1, 0.05),
        ("hicache_long_lru_time", lambda: rob()["hicache"]["lru_wall"], 452, 0.5),
        ("hicache_long_tc_time", lambda: rob()["hicache"]["tc_wall"], 390, 0.5),
        ("hicache_long_speedup", lambda: rob()["hicache"]["speedup"], 1.16, 0.005),
        ("hicache_long_lru_low", lambda: rob()["hicache"]["lru_low"], 4, 0)):
    claim(cid, "Intro; Results; Fig. 8a,b", cid.replace("_", " "), val, tol, HI)(f)

GLM = "gpu/results/e2e/e2e-fleet-mixed-{lru,unified}-32t-glm-scale.json"
for cid, f, val, tol in (
        ("glm_member_lru", lambda: 100 * max(rob()["glm"]["member_cached_lru"]), 47, 0.5),
        ("glm_member_tc", lambda: 100 * max(rob()["glm"]["member_cached_tc"]), 64, 0.5),
        ("glm_better_teams", lambda: rob()["glm"]["n_better"], 25, 0),
        ("glm_no_reuse_lru", lambda: rob()["glm"]["n_full_lru"], 9, 0),
        ("glm_no_reuse_tc", lambda: rob()["glm"]["n_full_tc"], 4, 0),
        ("glm_lru_time", lambda: rob()["glm"]["lru_wall"], 1107, 0.5),
        ("glm_tc_time", lambda: rob()["glm"]["tc_wall"], 1008, 0.5),
        ("glm_speedup", lambda: rob()["glm"]["speedup"], 1.10, 0.005)):
    claim(cid, "Intro; Results; Fig. 8c,d", cid.replace("_", " "), val, tol, GLM)(f)

VL = "gpu/results/vllm/e2e-autogen-design-*-8t-p3000r{8,16}-vllm-mask-280.metrics.json"
for cid, key, val in (("vllm_r8", "r8_hits", 1.26), ("vllm_r16", "r16_hits", 2.20)):
    claim(cid, "Intro; Results; Fig. 8e; Supplementary Note 8; Supplementary Table 7",
          f"vLLM block hits {val}-fold", val, 0.005, VL, "delta_hits TeamCache / LRU")(
        lambda key=key: rob()["vllm"][key][1] / rob()["vllm"][key][0])


@claim("overhead_us_50", "Results", "77 us per admission decision at 50 sessions", 77, 0.5,
       "cpu/phase0_cpu.json")
def _():
    """microbench.sizes['50'].us_per_req_p50"""
    return loadj(paths.CPU / "phase0_cpu.json")["microbench"]["sizes"]["50"]["us_per_req_p50"]


@claim("overhead_us_200", "Results", "101 us per admission decision at 200 sessions", 101, 0.5,
       "cpu/phase0_cpu.json")
def _():
    """microbench.sizes['200'].us_per_req_p50"""
    return loadj(paths.CPU / "phase0_cpu.json")["microbench"]["sizes"]["200"]["us_per_req_p50"]


missing("cold_prefill", "Intro; Supplementary Note 10",
        "61,440-token cold prefill 2.79 s vs 0.13 s from cache (about 21-fold)", "2.79 / 0.13",
        "measured on the GPU server by gpu_experiments/probes/measure_cost_model.py")
missing("waits_ablation", "Supplementary Note 15",
        "a wait-graph alternative selected no different members (three speeds, four loads)", "",
        "reproduced by `python -m teamcache_ncomms.optimal waits-ablation` (about 5 s)")
missing("throughput_nonregression", "Results",
        "independent-request throughput within +-0.2% of unmodified LRU", "0.2",
        "measured on the GPU server by gpu_experiments/probes/probe_nonregress.py, policy off and on")

# ------------------------------------------------------------------ Supplementary Information


@claim("si_discovery", "Supplementary Table 1", "member ARI, team precision and recall all 1.000",
       1.0, 0, "gpu/results/team_discovery_accuracy.json")
def _():
    """minimum of member ARI and team pair precision/recall over the six table settings"""
    rows = [r for r in loadj(paths.RES / "team_discovery_accuracy.json")
            if r["case"] in DISCOVERY_TABLE]
    assert len(rows) == len(DISCOVERY_TABLE)
    return min(min(r["member_ari"], r["team_pair_prf"][0], r["team_pair_prf"][1]) for r in rows)


DISCOVERY_TABLE = ("autogen-ring", "autogen-design-glm", "autogen-debug", "autogen-review",
                   "autogen-design-k6", "autogen-design-k8", "mix-autogen-4-kimi",
                   "selector-debug-kimi-agents", "selector-design-kimi-agents",
                   "selector-debug-glm-agents", "crewai-ring", "crewai-debug", "mix-crewai")


@claim("si_discovery_requests", "Supplementary Table 1",
       "272 + 184 + 120 + 20 + 40 + 60 requests in the six settings", 696, 0,
       "gpu/results/team_discovery_accuracy.json")
def _():
    """sum of request counts over the table settings"""
    rows = loadj(paths.RES / "team_discovery_accuracy.json")
    return sum(r["n"] for r in rows if r["case"] in DISCOVERY_TABLE)


LC = "gpu/results/e2e/e2e-longctx-ring-{lru,unified}-{3t-p61k-c64k,4t-p61k}.json"
for cid, f, val, tol in (
        ("si_longctx_lru_hit", lambda: 100 * (1 - e2e("e2e-longctx-ring-lru-3t-p61k-c64k")["turn_miss_rate"]), 0, 0.5),
        ("si_longctx_lru_time", lambda: e2e("e2e-longctx-ring-lru-3t-p61k-c64k")["wall_s"], 281, 0.5),
        ("si_longctx_lru_low", lambda: e2e("e2e-longctx-ring-lru-3t-p61k-c64k")["starved_teams"], 3, 0),
        ("si_longctx_tc_hit", lambda: 100 * (1 - e2e("e2e-longctx-ring-unified-3t-p61k-c64k")["turn_miss_rate"]), 20, 0.5),
        ("si_longctx_tc_time", lambda: e2e("e2e-longctx-ring-unified-3t-p61k-c64k")["wall_s"], 237, 0.5),
        ("si_longctx_tc_low", lambda: e2e("e2e-longctx-ring-unified-3t-p61k-c64k")["starved_teams"], 0, 0),
        ("si_longctx_4t_lru_hit", lambda: 100 * (1 - e2e("e2e-longctx-ring-lru-4t-p61k")["turn_miss_rate"]), 0, 0.5),
        ("si_longctx_4t_tc_hit", lambda: 100 * (1 - e2e("e2e-longctx-ring-unified-4t-p61k")["turn_miss_rate"]), 0, 0.5)):
    claim(cid, "Supplementary Table 9", cid.replace("_", " "), val, tol, LC)(f)


@claim("si_longctx_speedup", "Supplementary Note 10", "TeamCache 1.19 times faster on the long-context ring", 1.19, 0.005, LC)
def _():
    """LRU wall time / TeamCache wall time, three teams"""
    return e2e("e2e-longctx-ring-lru-3t-p61k-c64k")["wall_s"] / e2e("e2e-longctx-ring-unified-3t-p61k-c64k")["wall_s"]

AB = "gpu/results/team-ablate/{cohort,rotate,invdist}_9t_ring.json"
for cid, arm, field, val, tol in (
        ("si_ablate_rate_hit", "cohort", "hit", 39.1, 0.05),
        ("si_ablate_rate_time", "cohort", "wall", 189, 0.5),
        ("si_ablate_rate_low", "cohort", "low", 0, 0),
        ("si_ablate_swap_hit", "rotate", "hit", 21.3, 0.05),
        ("si_ablate_swap_time", "rotate", "wall", 203, 0.5),
        ("si_ablate_dist_hit", "invdist", "hit", 7.1, 0.05),
        ("si_ablate_dist_time", "invdist", "wall", 231, 0.5),
        ("si_ablate_dist_low", "invdist", "low", 8, 0)):
    def _ab(arm=arm, field=field):
        d = loadj(paths.ABLATE / f"{arm}_9t_ring.json")
        return {"hit": 100 * (1 - d["turn_miss_rate"]), "wall": d["wall_s"],
                "low": d["starved_teams"]}[field]
    claim(cid, "Supplementary Table 12", cid.replace("_", " "), val, tol, AB)(_ab)

for cid, idx, val, tol in (("si_star_lfu_time", ("Offline LFU", "makespan_s"), 82.97, 0.005),
                           ("si_star_lfu_hit", ("Offline LFU", "request_hit"), 71.6, 0.05),
                           ("si_star_tc_time", ("TeamCache", "makespan_s"), 96.01, 0.005),
                           ("si_star_tc_hit", ("TeamCache", "request_hit"), 56.0, 0.05)):
    def _star(idx=idx):
        r = next(r for r in extract_star() if r["policy"] == idx[0])
        return r[idx[1]] * (100 if idx[1] == "request_hit" else 1)
    claim(cid, "Supplementary Note 11", cid.replace("_", " "), val, tol,
          "gpu/mixed/star_{static_rate,teamcache}_9t.json")(_star)


@claim("si_reset_order", "Supplementary Note 14",
       "reset validation keeps TeamCache < Offline LFU < LRU < Unpartitioned Admission", 1, 0,
       "gpu/mixed/isolation_s11/*.json")
def _():
    """1 if the completion-time ordering holds"""
    t = {r["policy"]: r["makespan_s"] for r in extract_reset()}
    return int(t["TeamCache"] < t["Offline LFU"] < t["LRU"] < t["Unpartitioned Admission"])


@claim("si_bypass_min", "Supplementary Note 15",
       "forced insertion lowers TeamCache's hit rate by 2-9 points (lower end)", 2.2, 0.05,
       "gpu/results/sensitivity-margin-bypass.txt",
       "smallest delta among star, debate, ring, mixed; regenerate with "
       "`python -m teamcache_ncomms.optimal sensitivity`")
def _():
    """smallest positive bypass-minus-admit delta among star, debate, ring, mixed"""
    return min(_bypass_deltas().values())


@claim("si_bypass_max", "Supplementary Note 15",
       "forced insertion lowers TeamCache's hit rate by 2-9 points (upper end)", 8.7, 0.05,
       "gpu/results/sensitivity-margin-bypass.txt")
def _():
    """largest bypass-minus-admit delta among star, debate, ring, mixed"""
    return max(_bypass_deltas().values())


def _bypass_deltas():
    text = (paths.RES / "sensitivity-margin-bypass.txt").read_text()
    out = {}
    for line in text.split("=== bypass ablation")[1].splitlines():
        parts = line.split()
        if parts and parts[0] in ("star", "debate", "ring", "mixed"):
            out[parts[0]] = float(parts[3].rstrip("p").rstrip("p"))
    return out


# ------------------------------------------------------------------ figure-derived numbers

def source_data(fig, panel):
    """A Source Data sheet; regenerated copies in $TEAMCACHE_OUT take precedence."""
    import pandas as pd
    name = f"SourceData_Fig{fig}.xlsx"
    path = paths.SOURCE_DATA_OUT / name
    if not path.exists():
        path = paths.DATA / "source_data" / name
    return pd.read_excel(path, sheet_name=f"Fig{fig}{panel}", header=2)


for topo, val in (("Ring", 50), ("Free-form", 91)):
    claim(f"rt_hit_{topo}", "Results; Fig. 2c",
          f"with room for 36 of 45 histories a {topo.lower()} team keeps {val}% of its hits", val,
          0.5, "source_data/SourceData_Fig2.xlsx (Fig2c)", "predicted_hit_rate_pct")(
        lambda topo=topo: float(source_data(2, "c").groupby("topology")
                                .predicted_hit_rate_pct.first()[topo]))


def _fig3d(kind, policy):
    d = source_data(3, "d")
    return float(d[(d.team_kind == kind) & (d.policy == policy)].request_hit_rate_pct.mean())


for kind, pol, val in (("ring", "LRU", 6), ("ring", "TeamCache", 43),
                       ("free-form", "LRU", 62), ("free-form", "TeamCache", 70)):
    claim(f"design_{kind}_{pol}", "Results 'TeamCache chooses each team's policy online'; Fig. 3d",
          f"{kind} teams under {pol}: {val}%", val, 0.5,
          "cache_level/s5_policy_trace.json via source_data/SourceData_Fig3.xlsx (Fig3d)",
          "mean request hit rate over teams of this kind")(
        lambda kind=kind, pol=pol: _fig3d(kind, pol))


def _fig6a(policy):
    a = source_data(6, "a")
    g = a[a.policy == policy].groupby("team").prefix_token_reuse_pct.mean()
    return [float(g[t]) for t in range(12)]


@claim("fair_ua_first_ring", "Results; Fig. 6a", "Unpartitioned Admission: 95% on the first ring "
       "team", 95, 0.5, MIX)
def _():
    """mean over seeds of team 0 prefix-token reuse"""
    return _fig6a("Unpartitioned Admission")[0]


@claim("fair_ua_last_ring", "Results; Fig. 6a", "... 38% on the last ring team", 38, 0.5, MIX)
def _():
    """mean over seeds of team 5 prefix-token reuse"""
    return _fig6a("Unpartitioned Admission")[5]


@claim("fair_ua_zero_selector", "Results; Fig. 6a",
       "zero reuse on four of the six selector teams", 4, 0, MIX)
def _():
    """selector teams (6-11) with zero mean reuse"""
    return sum(v == 0 for v in _fig6a("Unpartitioned Admission")[6:])


@claim("fair_tc_late_ring", "Results; Fig. 6a", "TeamCache: about 49% on the later ring teams",
       49, 1, MIX)
def _():
    """mean over ring teams 3-5"""
    return float(np.mean(_fig6a("TeamCache")[3:6]))


@claim("fair_tc_selector_min", "Results; Fig. 6a", "selector teams between 68% ...", 68, 0.5, MIX)
def _():
    """minimum over selector teams"""
    return min(_fig6a("TeamCache")[6:])


@claim("fair_tc_selector_max", "Results; Fig. 6a", "... and 74%", 74, 0.5, MIX)
def _():
    """maximum over selector teams"""
    return max(_fig6a("TeamCache")[6:])


def _s8(system, key):
    return sim("s8_public_traces")["systems"][system][key]


@claim("public_median_lo", "Results; Fig. 1b", "median ranks between 0.10 ...", 0.10, 0.005,
       "cache_level/s8_public_traces.json")
def _():
    """HyperAgent median normalised recency rank"""
    return _s8("HyperAgent", "rank_median")


@claim("public_median_hi", "Results; Fig. 1b", "... and 0.31", 0.31, 0.005,
       "cache_level/s8_public_traces.json")
def _():
    """AppWorld median normalised recency rank"""
    return _s8("AppWorld", "rank_median")


@claim("public_magentic_third", "Results; Fig. 1b",
       "a third of Magentic-One traces have ranks of at least 0.6", 1 / 3, 0.02,
       "cache_level/s8_public_traces.json")
def _():
    """fraction of Magentic-One traces with rank >= 0.6"""
    return _s8("Magentic-One", "frac_rank_ge_0.6")


# ------------------------------------------------------------------ block level, shadow, groups

S11, S12, S14 = "s11_block_level.json", "s12_shadow_error.json", "s14_group_baselines.json"


@claim("block_within2", "Results 'TeamCache tracks the better policy and protects the worst-served team'; Supplementary Note 5",
       "aggregate reuse within 2 points of the best baseline in 205 of 240", 205, 0, S11)
def _():
    return sim("s11_block_level")["overall"]["TeamCache"]["aggregate_within_2pp"]


@claim("block_cells", "Results 'TeamCache tracks the better policy and protects the worst-served team'", "240 topology-load-size configurations",
       240, 0, S11)
def _():
    return sim("s11_block_level")["overall"]["cells"]


@claim("block_over_lru", "Results 'TeamCache tracks the better policy and protects the worst-served team'",
       "exceeded LRU's aggregate reuse by a median of 10.3 points", 10.3, 0.05, S11)
def _():
    return sim("s11_block_level")["overall"]["teamcache_minus_lru_aggregate_pp"]["median"]


@claim("shadow_agree", "Methods; Supplementary Note 3",
       "agreed with the engine's retention for 93% of references", 93, 0.5, S12)
def _():
    return 100 * sim("s12_shadow_error")["summary"]["agreement"]


@claim("shadow_refs", "Methods; Supplementary Note 3", "3,199 references", 3199, 0, S12)
def _():
    return sim("s12_shadow_error")["summary"]["references"]


def _group_starved():
    h = sim("s14_group_baselines")["headline"]
    return [v["cells_worst_below_5pct"] for k, v in h.items()
            if k.startswith(("Partitioned", "UCP", "Max-min"))]


@claim("group_starved_lo", "Results 'classical policies'; Supplementary Note 4",
       "some team below 5% reuse in 15 ...", 15, 0, S14)
def _():
    return min(_group_starved())


@claim("group_starved_hi", "Results 'classical policies'; Supplementary Note 4",
       "... to 21 of the 36 configurations", 21, 0, S14)
def _():
    return max(_group_starved())


for _cid, _pol, _key, _val in (("group_ucp_hit", "UCP (hindsight)", "mean_hit", 49.7),
                               ("group_tc_hit", "TeamCache", "mean_hit", 51.0),
                               ("group_tc_worst", "TeamCache", "mean_worst", 42.3)):
    claim(_cid, "Results 'classical policies'", f"{_pol} {_key} {_val}", _val, 0.05, S14)(
        lambda p=_pol, k=_key: 100 * sim("s14_group_baselines")["headline"][p][k])


@claim("group_tc_starved", "Supplementary Note 4", "TeamCache left no team below 5%", 0, 0, S14)
def _():
    return sim("s14_group_baselines")["headline"]["TeamCache"]["cells_worst_below_5pct"]


for _cid, _pol, _val in (("block_tc_starved", "TeamCache", 9), ("block_lru_starved", "LRU", 34)):
    claim(_cid, "Intro; Results 'TeamCache tracks the better policy and protects the worst-served team'; Supplementary Note 5",
          f"{_pol} left a team below 10% reuse in {_val} of 240", _val, 0, S11)(
        lambda p=_pol: sim("s11_block_level")["overall"]["starved_cells"][p])


S13 = "s13_sensitivity.json"


def _sens_min(param, value):
    return next(r["worst_delta_pp_min"] for r in sim("s13_sensitivity")["summary"]
                if r["param"] == param and r["value"] == value)


for _cid, _param, _value, _val in (("sens_w20_min", "adapt_window", 20, -21.0),
                                   ("sens_eta125_min", "margin", 1.25, -24.0),
                                   ("sens_pool11_min", "pool_scale", 1.1, -22.1),
                                   ("sens_pool12_min", "pool_scale", 1.2, -26.9)):
    claim(_cid, "Supplementary Note 6", f"single-run worst-team drop {_val} ({_param}={_value})",
          _val, 0.05, S13)(lambda p=_param, v=_value: _sens_min(p, v))


# ------------------------------------------------------------------ selector speaker-choice calls

S15 = "s15_discovery.json"


def _chooser_share(hw=None, arm=None):
    return [r for r in sim("s15_discovery")["chooser"]["gpu_share"]
            if (hw is None or r["hardware"] == hw) and (arm is None or r["arm"] == arm)]


def _chooser_cf(hw, arm_prefix):
    return next(r for r in sim("s15_discovery")["chooser"]["summary"]
                if r["hardware"] == hw and r["arm"].startswith(arm_prefix))


for _cid, _fn, _val, _tol in (
        ("chooser_req_lo", lambda: 100 * min(r["chooser_request_share"] for r in _chooser_share()),
         50.2, 0.05),
        ("chooser_req_hi", lambda: 100 * max(r["chooser_request_share"] for r in _chooser_share()),
         53.7, 0.05),
        ("chooser_tok_lo", lambda: 100 * min(r["chooser_token_share"] for r in _chooser_share()),
         33.0, 0.05),
        ("chooser_tok_hi", lambda: 100 * max(r["chooser_token_share"] for r in _chooser_share()),
         34.6, 0.05),
        ("chooser_members_a100", lambda: max(r["chooser_default_team_members"]
                                             for r in _chooser_share("A100")), 781, 0),
        ("chooser_members_b300", lambda: _chooser_share("B300")[0]["chooser_default_team_members"],
         1249, 0),
        ("chooser_a100_tc", lambda: 100 * min(r["chooser_reuse"]
                                              for r in _chooser_share("A100", "TeamCache")), 94.1, 0.05),
        ("chooser_a100_lru_lo", lambda: 100 * min(r["chooser_reuse"]
                                                  for r in _chooser_share("A100", "LRU")), 93.3, 0.05),
        ("chooser_a100_lru_hi", lambda: 100 * max(r["chooser_reuse"]
                                                  for r in _chooser_share("A100", "LRU")), 94.1, 0.05),
        ("chooser_b300_tc", lambda: 100 * _chooser_share("B300", "TeamCache")[0]["chooser_reuse"],
         87.1, 0.05),
        ("chooser_b300_lru", lambda: 100 * _chooser_share("B300", "LRU")[0]["chooser_reuse"],
         82.2, 0.05),
        ("chooser_static_lo", lambda: 100 * min(r["chooser_reuse"] for r in
                                                _chooser_share("A100", "Static Partitioning")), 7.2, 0.05),
        ("chooser_static_hi", lambda: 100 * max(r["chooser_reuse"] for r in
                                                _chooser_share("A100", "Static Partitioning")), 7.4, 0.05),
        ("chooser_cf_a100_default", lambda: 100 * _chooser_cf("A100", "TeamCache, chooser calls in one")
         ["all_tok_mean"], 73.9, 0.05),
        ("chooser_cf_b300_default", lambda: 100 * _chooser_cf("B300", "TeamCache, chooser calls in one")
         ["all_tok_mean"], 74.8, 0.05),
        ("chooser_cf_a100_own", lambda: 100 * _chooser_cf("A100", "TeamCache, chooser calls as")
         ["all_tok_mean"], 72.0, 0.05),
        ("chooser_cf_b300_own", lambda: 100 * _chooser_cf("B300", "TeamCache, chooser calls as")
         ["all_tok_mean"], 70.8, 0.05),
        ("chooser_cf_a100_lru", lambda: 100 * _chooser_cf("A100", "LRU")["all_tok_mean"], 71.6, 0.05),
        ("chooser_cf_b300_lru", lambda: 100 * _chooser_cf("B300", "LRU")["all_tok_mean"], 69.6, 0.05),
        ("chooser_cf_a100_bypass", lambda: 100 * _chooser_cf("A100", "TeamCache, chooser calls bypassed")
         ["all_tok_mean"], 44.6, 0.05),
        ("chooser_cf_b300_bypass", lambda: 100 * _chooser_cf("B300", "TeamCache, chooser calls bypassed")
         ["all_tok_mean"], 42.6, 0.05),
        ("chooser_cf_worst_gap", lambda: max(
            100 * (_chooser_cf(h, "LRU")["member_worst_mean"]
                   - _chooser_cf(h, "TeamCache, chooser calls in one")["member_worst_mean"])
            for h in ("A100", "B300")), 2.5, 0.05)):
    claim(_cid, "Methods; Supplementary Note 12", _cid.replace("_", " "), _val, _tol, S15)(_fn)


# ------------------------------------------------------------------ A100 repetitions

_A100 = None
A1 = "gpu_a100/{e2e,mixed,star,team-sweep,hicache,vllm,nonregress}"


def a100():
    global _A100
    if _A100 is None:
        from . import a100 as _m
        _A100 = _m.summary()
    return _A100


def _a(*keys):
    def get():
        v = a100()
        for k in keys:
            v = v[k]
        return v
    return get


for _cid, _where, _text, _val, _tol, _fn in (
        ("a100_ring_lru_reuse", "Intro; Results", "A100 ring LRU reuse 0.6%", 0.6, 0.05,
         _a("ring", "LRU", "reuse_pct", "mean")),
        ("a100_ring_tc_reuse", "Results", "A100 ring TeamCache reuse 29.8%", 29.8, 0.05,
         _a("ring", "TeamCache", "reuse_pct", "mean")),
        ("a100_ring_speedup", "Intro; Results", "A100 ring 1.24-fold", 1.24, 0.005,
         _a("ring", "speedup")),
        ("a100_ring_diff", "Results", "paired difference 95.3 s", 95.3, 0.05,
         _a("ring", "paired_time_saving_s", "mean")),
        ("a100_ring_lo", "Results", "95% CI 91.7 ...", 91.7, 0.05,
         _a("ring", "paired_time_saving_s", "lo")),
        ("a100_ring_hi", "Results", "... 99.0 s", 99.0, 0.05,
         _a("ring", "paired_time_saving_s", "hi")),
        ("a100_ring_vs_static", "Results", "within 4.1% of Static Partitioning", 4.1, 0.05,
         lambda: 100 * (a100()["ring"]["TeamCache"]["time_s"]["mean"]
                        / a100()["ring"]["Static Partitioning"]["time_s"]["mean"] - 1)),
        ("a100_sel_tc_reuse", "Results", "A100 selector TeamCache 57.5%", 57.5, 0.05,
         _a("selector", "TeamCache", "reuse_pct", "mean")),
        ("a100_sel_lru_reuse", "Results", "A100 selector LRU 57.0%", 57.0, 0.05,
         _a("selector", "LRU", "reuse_pct", "mean")),
        ("a100_sel_diff", "Results", "paired difference 9.9 s", 9.9, 0.05,
         _a("selector", "paired_time_saving_s", "mean")),
        ("a100_sel_lo", "Results", "-24.4 ...", -24.4, 0.05,
         _a("selector", "paired_time_saving_s", "lo")),
        ("a100_sel_hi", "Results", "... 44.3 s", 44.3, 0.05,
         _a("selector", "paired_time_saving_s", "hi")),
        ("a100_sel_static_reuse", "Results", "Static Partitioning fell to 48.0%", 48.0, 0.05,
         _a("selector", "Static Partitioning", "reuse_pct", "mean")),
        ("a100_sel_static_ratio", "Results", "took 1.48 times as long", 1.48, 0.005,
         lambda: a100()["selector"]["Static Partitioning"]["time_s"]["mean"]
         / a100()["selector"]["TeamCache"]["time_s"]["mean"]),
        ("a100_mixed_lru_reuse", "Results", "A100 mixed LRU 37.9%", 37.9, 0.05,
         _a("mixed", "LRU", "reuse_pct", "mean")),
        ("a100_mixed_tc_reuse", "Results", "A100 mixed TeamCache 58.2%", 58.2, 0.05,
         _a("mixed", "TeamCache", "reuse_pct", "mean")),
        ("a100_mixed_cut", "Intro; Results", "16.5% shorter than LRU", 16.5, 0.05,
         _a("mixed", "LRU", "teamcache_saving_pct")),
        ("a100_mixed_diff", "Results", "48.1 s", 48.1, 0.05,
         _a("mixed", "LRU", "teamcache_saving_s", "mean")),
        ("a100_mixed_lo", "Results", "36.0 ...", 36.0, 0.05,
         _a("mixed", "LRU", "teamcache_saving_s", "lo")),
        ("a100_mixed_hi", "Results", "... 60.2 s", 60.2, 0.05,
         _a("mixed", "LRU", "teamcache_saving_s", "hi")),
        ("a100_mixed_olfu_cut", "Results", "4.6% vs Offline LFU", 4.6, 0.05,
         _a("mixed", "Offline LFU", "teamcache_saving_pct")),
        ("a100_mixed_olfu_diff", "Results", "11.8 s", 11.8, 0.05,
         _a("mixed", "Offline LFU", "teamcache_saving_s", "mean")),
        ("a100_mixed_olfu_lo", "Results", "4.9 ...", 4.9, 0.05,
         _a("mixed", "Offline LFU", "teamcache_saving_s", "lo")),
        ("a100_mixed_olfu_hi", "Results", "... 18.7 s", 18.7, 0.05,
         _a("mixed", "Offline LFU", "teamcache_saving_s", "hi")),
        ("a100_mixed_every_run", "Results", "in every paired run (1 = true)", 1, 0,
         lambda: float(a100()["mixed"]["LRU"]["teamcache_saving_s"]["all_positive"]
                       and a100()["mixed"]["Offline LFU"]["teamcache_saving_s"]["all_positive"])),
        ("a100_tput_max_gap", "Results 'Across memory hierarchies'",
         "throughput within 1.2% on A100", 1.2, 0.05,
         lambda: max(abs(v["mean_pct"]) for v in a100()["nonregress"].values())),
        ("a100_tput_rep_spread", "Results 'Across memory hierarchies'",
         "repetitions varied by up to 8.4%", 8.4, 0.05,
         lambda: max(v["lru_rep_range_pct"] for v in a100()["nonregress"].values())),
        ("a100_star_cut", "Supplementary Note 7", "star 13.3% faster than LRU", 13.3, 0.05,
         _a("star", "LRU", "teamcache_saving_pct")),
        ("a100_star_every_run", "Supplementary Note 7", "star: in every run (1 = true)", 1, 0,
         lambda: float(a100()["star"]["LRU"]["teamcache_saving_s"]["all_positive"])),
        ("a100_sweep_lru_10", "Results; Supplementary Note 7", "LRU 3.1% at ten teams", 3.1, 0.05,
         lambda: _sweep_mean(10, "LRU")),
        ("a100_sweep_tc_10", "Results; Supplementary Note 7", "TeamCache 58.4% at ten teams",
         58.4, 0.05, lambda: _sweep_mean(10, "TeamCache")),
        ("a100_sweep_tc_12", "Supplementary Note 7", "TeamCache 44.0% at twelve teams",
         44.0, 0.05, lambda: _sweep_mean(12, "TeamCache")),
        ("a100_sweep_210_lru", "Results; Supplementary Note 7", "210 s budget: LRU 8 teams", 8, 0,
         lambda: max(a100()["sweep"]["largest_n"]["210"]["LRU"])),
        ("a100_sweep_210_tc", "Results; Supplementary Note 7", "210 s budget: TeamCache 10 teams",
         10, 0, lambda: min(a100()["sweep"]["largest_n"]["210"]["TeamCache"])),
        ("a100_sweep_tc_starved", "Supplementary Note 7",
         "TeamCache never left a low-reuse team in the sweep", 0, 0,
         lambda: max(r["low_reuse_teams"] for r in a100()["sweep"]["rows"]
                     if r["policy"] == "TeamCache")),
        ("a100_hicache_lru", "Results; Supplementary Note 7", "HiCache ring LRU 86.9%", 86.9, 0.05,
         lambda: _mean(r["reuse_pct"] for r in a100()["hicache"]["LRU"]["ring"])),
        ("a100_hicache_tc", "Results; Supplementary Note 7", "HiCache ring TeamCache 87.4%",
         87.4, 0.05,
         lambda: _mean(r["reuse_pct"] for r in a100()["hicache"]["TeamCache"]["ring"])),
        ("a100_fleet_lru", "Supplementary Note 7", "fleet LRU 29.2%", 29.2, 0.05,
         lambda: a100()["hicache"]["LRU"]["fleet"][0]["reuse_pct"]),
        ("a100_fleet_tc", "Supplementary Note 7", "fleet TeamCache 31.2%", 31.2, 0.05,
         lambda: a100()["hicache"]["TeamCache"]["fleet"][0]["reuse_pct"]),
        ("a100_vllm_ratio_mean", "Results; Supplementary Note 7",
         "vLLM 1.35 times LRU's cached tokens after sixteen rounds", 1.35, 0.005,
         lambda: _mean(a100()["vllm"]["hit_ratio_16"])),
        ("a100_vllm_ratio_lo4", "Results; Supplementary Note 7", "1.40 ... in four runs", 1.40, 0.005,
         lambda: sorted(a100()["vllm"]["hit_ratio_16"])[1]),
        ("a100_vllm_ratio_hi", "Results; Supplementary Note 7", "... 1.50 times", 1.50, 0.005,
         lambda: max(a100()["vllm"]["hit_ratio_16"])),
        ("a100_vllm_ratio_min", "Results; Supplementary Note 7", "unchanged in the fifth", 1.00, 0.005,
         lambda: min(a100()["vllm"]["hit_ratio_16"])),
        ("a100_vllm_runs", "Results; Supplementary Note 7", "five paired vLLM runs", 5, 0,
         lambda: len(a100()["vllm"]["hit_ratio_16"])),
        ("a100_vllm_r8_lo", "Supplementary Note 7", "0.96 ... after eight rounds", 0.96, 0.005,
         lambda: min(a100()["vllm"]["hit_ratio_8"])),
        ("a100_vllm_r8_hi", "Supplementary Note 7", "... 1.05 times", 1.05, 0.005,
         lambda: max(a100()["vllm"]["hit_ratio_8"])),
        ("a100_vllm_lru_lo", "Supplementary Note 7", "LRU 48.5% from cache", 48.5, 0.05,
         lambda: min(100 * r["hits_16"] / r["queries_16"] for r in a100()["vllm"]["LRU"])),
        ("a100_vllm_lru_hi", "Supplementary Note 7", "LRU 54.3% from cache", 54.3, 0.05,
         lambda: max(100 * r["hits_16"] / r["queries_16"] for r in a100()["vllm"]["LRU"])),
        ("a100_vllm_tc_lo4", "Supplementary Note 7", "TeamCache 70.8% in four runs", 70.8, 0.05,
         lambda: sorted(100 * r["hits_16"] / r["queries_16"] for r in a100()["vllm"]["TeamCache"])[1]),
        ("a100_vllm_tc_hi", "Supplementary Note 7", "TeamCache 76.6%", 76.6, 0.05,
         lambda: max(100 * r["hits_16"] / r["queries_16"] for r in a100()["vllm"]["TeamCache"])),
        ("a100_vllm_time_spread", "Supplementary Note 7", "times varied up to 4.0-fold", 4.0, 0.05,
         lambda: max(max(r["wall_s"] for r in a100()["vllm"][p]) / min(r["wall_s"] for r in a100()["vllm"][p])
                     for p in ("LRU", "TeamCache"))),
        ("a100_fleet_lru_starved", "Results; Supplementary Note 7", "fleet: four teams without reuse",
         4, 0, lambda: a100()["hicache"]["LRU"]["fleet"][0]["low_reuse_teams"]),
):
    claim(_cid, _where, _text, _val, _tol, A1)(_fn)


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def _sweep_mean(n, policy):
    return _mean(r["request_hit_pct"] for r in a100()["sweep"]["rows"]
                 if r["n_teams"] == n and r["policy"] == policy)


@claim("shadow_a100_ring_lo", "Main text; Supplementary Note 3", "97% of A100 ring references",
       97.3, 0.05, S12)
def _():
    return 100 * min(r["agreement"] for r in sim("s12_shadow_error")["runs"]
                     if r["run"].startswith("A100 ring"))


@claim("shadow_a100_ring_hi", "Main text; Supplementary Note 3", "98% of A100 ring references",
       98.4, 0.05, S12)
def _():
    return 100 * max(r["agreement"] for r in sim("s12_shadow_error")["runs"]
                     if r["run"].startswith("A100 ring"))


# ------------------------------------------------------------------ runner

def evaluate():
    rows = []
    for c in CLAIMS:
        row = {k: c[k] for k in ("id", "where", "text", "expected", "tol", "sources", "how")}
        if c["fn"] is None:
            row.update(value="", status=c["status"])
        else:
            try:
                v = c["fn"]()
                ok = abs(float(v) - float(c["expected"])) <= c["tol"] + 1e-9
                row.update(value=f"{float(v):.6g}", status="ok" if ok else "MISMATCH")
            except Exception as exc:  # noqa: BLE001
                row.update(value="", status=f"ERROR {type(exc).__name__}: {exc}")
        rows.append(row)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", type=Path, help="write the claim table to this CSV file")
    args = ap.parse_args(argv)
    rows = evaluate()
    bad = [r for r in rows if r["status"] not in ("ok", "GPU probe", "by command")]
    for r in rows:
        mark = {"ok": "ok  ", "GPU probe": "gpu ", "by command": "cmd "}.get(r["status"], "FAIL")
        print(f"{mark} {r['id']:<28} expected {r['expected']!s:<10} got {r['value']:<10} {r['status'] if mark == 'FAIL' else ''}")
    n_ok = sum(r["status"] == "ok" for r in rows)
    print(f"\n{n_ok} of {len(rows)} claims reproduced; "
          f"{sum(r['status'] == 'by command' for r in rows)} reproduced by command; "
          f"{sum(r['status'] == 'GPU probe' for r in rows)} by GPU probe; {len(bad)} failed")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("wrote", args.csv)
    return 1 if bad else 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    sys.exit(main())
