#!/usr/bin/env python3
"""A100 repetitions: per-seed tables and paired statistics.

Reads the raw results of the 4xA100-80GB campaign (GLM-4.7-Flash, SGLang
0.5.18 and vLLM 0.30.0) from $TEAMCACHE_DATA/gpu_a100 and writes
$TEAMCACHE_OUT/gpu_a100_summary.json.  Every comparison has three seeds with
the policy order rotated across seeds; differences are paired by seed.

    python -m teamcache_ncomms.a100
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .gpu_tables import load_jsonl, split_meas
from .paths import DATA, OUT

A100 = DATA / "gpu_a100"
T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
LIVE_SEEDS = ("", "-s1", "-s2")
VLLM_SEEDS = ("", "-s1", "-s2", "-s3", "-s4")
REPLAY_SEEDS = (11, 12, 13)
REPLAY_ARM = {"LRU": "lru", "Unpartitioned Admission": "group_blind",
              "Offline LFU": "static_rate", "TeamCache": "teamcache"}
SWEEP_ARM = {"LRU": "lru", "Unpartitioned Admission": "bypass",
             "Static Partitioning": "cohort", "TeamCache": "unified"}
LIVE_ARM = {"LRU": "lru", "Static Partitioning": "cohort", "TeamCache": "unified"}
SWEEP_TEAMS = (6, 8, 10, 12, 14, 16)
BUDGETS = (120, 150, 180, 210, 240, 300)


def loadj(p):
    return json.loads(Path(p).read_text())


def paired(a, b):
    """Mean of a - b with a two-sided 95% t interval."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if len(d) < 2:
        return {"mean": float(d.mean()), "lo": None, "hi": None, "n": len(d)}
    h = T975[len(d)] * d.std(ddof=1) / math.sqrt(len(d))
    return {"mean": float(d.mean()), "lo": float(d.mean() - h), "hi": float(d.mean() + h),
            "n": len(d), "all_positive": bool((d > 0).all())}


def stats(xs):
    xs = np.asarray(xs, float)
    return {"mean": float(xs.mean()), "sd": float(xs.std(ddof=1)) if len(xs) > 1 else 0.0,
            "runs": [float(x) for x in xs]}


def live(harness, task):
    out = {}
    for paper, arm in LIVE_ARM.items():
        hit, span, tok = [], [], []
        for s in LIVE_SEEDS:
            p = A100 / "e2e" / f"e2e-{harness}-{task}-{arm}-20t-p3000r8-a100{s}.json"
            if not p.exists():
                break
            d = loadj(p)
            hit.append(100 * d["steady"]["prefix_hit_rate"])
            span.append(d["steady"]["span_s"])
            tok.append(d["prompt_tokens_total"])
        if hit:
            out[paper] = {"reuse_pct": stats(hit), "time_s": stats(span), "prompt_tokens": tok}
    if "LRU" in out and "TeamCache" in out:
        lru, tc = out["LRU"], out["TeamCache"]
        n = min(len(lru["time_s"]["runs"]), len(tc["time_s"]["runs"]))
        out["paired_time_saving_s"] = paired(lru["time_s"]["runs"][:n], tc["time_s"]["runs"][:n])
        out["speedup"] = lru["time_s"]["mean"] / tc["time_s"]["mean"]
    toks = [r for v in out.values() if isinstance(v, dict) and "prompt_tokens" in v
            for r in v["prompt_tokens"]]
    if toks:
        by_seed = list(zip(*[v["prompt_tokens"] for v in out.values()
                             if isinstance(v, dict) and "prompt_tokens" in v
                             and len(v["prompt_tokens"]) == 3]))
        out["max_prompt_token_spread_pct"] = max(
            (100 * (max(s) - min(s)) / min(s) for s in by_seed), default=None)
    return out


def replay(kind, teams):
    out = {}
    for paper, arm in REPLAY_ARM.items():
        rows = []
        for s in REPLAY_SEEDS:
            p = A100 / kind / f"{kind}_{arm}_{teams}t_s{s}.json"
            if not p.exists():
                continue
            d = loadj(p)
            reuse = d.get("prefix_tokens_cached", 0) / max(d.get("prefix_tokens_prompt", 1), 1)
            lp = p.with_name(p.stem + ".requests.jsonl")
            if lp.exists():
                _, meas = split_meas(load_jsonl(lp))
                reuse = (sum(r.get("cached_tokens") or 0 for r in meas)
                         / sum(r.get("prompt_tokens") or 0 for r in meas))
            hit = d.get("request_hit")
            if hit is None:
                hit = 1.0 - d["turn_miss_rate"]
            low = d.get("low_reuse_teams", d.get("starved_teams"))
            rows.append({"seed": s, "time_s": d["wall_s"], "reuse_pct": 100 * reuse,
                         "request_hit_pct": 100 * hit, "low_reuse_teams": low})
        if rows:
            out[paper] = {"runs": rows,
                          "time_s": stats([r["time_s"] for r in rows]),
                          "reuse_pct": stats([r["reuse_pct"] for r in rows]),
                          "request_hit_pct": stats([r["request_hit_pct"] for r in rows])}
    if "TeamCache" in out:
        tc = {r["seed"]: r["time_s"] for r in out["TeamCache"]["runs"]}
        for paper in out:
            if paper == "TeamCache" or "runs" not in out[paper]:
                continue
            base = {r["seed"]: r["time_s"] for r in out[paper]["runs"]}
            seeds = sorted(set(tc) & set(base))
            if len(seeds) >= 2:
                out[paper]["teamcache_saving_s"] = paired([base[s] for s in seeds],
                                                          [tc[s] for s in seeds])
                out[paper]["teamcache_saving_pct"] = 100 * out[paper]["teamcache_saving_s"]["mean"] \
                    / float(np.mean([base[s] for s in seeds]))
    return out


def sweep():
    rows = []
    for n in SWEEP_TEAMS:
        for paper, arm in SWEEP_ARM.items():
            for s in REPLAY_SEEDS:
                p = A100 / "team-sweep" / f"{arm}_{n}t_ring_s{s}.json"
                if not p.exists():
                    continue
                d = loadj(p)
                rows.append({"n_teams": n, "policy": paper, "seed": s, "wall_s": d["wall_s"],
                             "request_hit_pct": 100 * (1 - d["turn_miss_rate"]),
                             "low_reuse_teams": d["starved_teams"]})
    out = {"rows": rows, "largest_n": {}}
    for budget in BUDGETS:
        for paper in SWEEP_ARM:
            for s in REPLAY_SEEDS:
                ok = [r["n_teams"] for r in rows
                      if r["policy"] == paper and r["seed"] == s and r["wall_s"] <= budget]
                out["largest_n"].setdefault(str(budget), {}).setdefault(paper, []).append(
                    max(ok) if ok else None)
    return out


def nonregress():
    out = {}
    for lv in ("1", "8", "16"):
        a = [loadj(A100 / "nonregress" / f"quota0_r{r}.json")["levels"][lv]["out_tok_s"]
             for r in (1, 2, 3)]
        b = [loadj(A100 / "nonregress" / f"unified_r{r}.json")["levels"][lv]["out_tok_s"]
             for r in (1, 2, 3)]
        out[lv] = {"lru_tok_s": a, "teamcache_tok_s": b,
                   "paired_pct": [100 * (y / x - 1) for x, y in zip(a, b)],
                   "mean_pct": 100 * (np.mean(b) / np.mean(a) - 1),
                   "lru_rep_range_pct": 100 * (max(a) - min(a)) / min(a)}
    return out


def hicache():
    out = {}
    for arm, paper in (("lru", "LRU"), ("unified", "TeamCache")):
        ring, fleet = [], []
        for s in LIVE_SEEDS:
            p = A100 / "hicache" / f"e2e-autogen-design-{arm}-20t-p3000r8-a100-hicache{s}.json"
            if p.exists():
                d = loadj(p)
                ring.append({"reuse_pct": 100 * d["steady"]["prefix_hit_rate"],
                             "time_s": d["steady"]["span_s"]})
        for s in REPLAY_SEEDS:
            p = A100 / "hicache" / f"fleet_{arm}_10t_s{s}.json"
            if p.exists():
                d = loadj(p)
                fleet.append({"seed": s, "time_s": d["wall_s"],
                              "reuse_pct": 100 * d["mean_cached"],
                              "miss_pct": 100 * d["turn_miss_rate"],
                              "low_reuse_teams": d["starved_teams"],
                              "team_miss_pct": [100 * t["miss_rate"] for t in d["per_team"]]})
        out[paper] = {"ring": ring, "fleet": fleet}
    if out["LRU"]["fleet"] and out["TeamCache"]["fleet"]:
        n = min(len(out["LRU"]["fleet"]), len(out["TeamCache"]["fleet"]))
        out["fleet_saving_s"] = paired([r["time_s"] for r in out["LRU"]["fleet"][:n]],
                                       [r["time_s"] for r in out["TeamCache"]["fleet"][:n]])
    return out


def vllm():
    out = {}
    for arm, paper in (("lru", "LRU"), ("unified", "TeamCache")):
        rows = []
        for s in VLLM_SEEDS:
            stem = f"e2e-autogen-design-{arm}-8t-p3000r16-a100-vllm{s}"
            mp = A100 / "vllm" / f"{stem}.metrics.json"
            sp = A100 / "vllm" / f"{stem}.samples.jsonl"
            if not mp.exists():
                continue
            m = loadj(mp)
            samples = [json.loads(l) for l in sp.read_text().splitlines() if l.strip()]
            half = 8 * 5 * 8
            at8 = next((x for x in samples if x["requests"] >= half), samples[-1])
            rows.append({"wall_s": m["wall_s"], "hits_16": m["delta_hits"],
                         "queries_16": m["delta_queries"], "hits_8": at8["hits"],
                         "queries_8": at8["queries"]})
        out[paper] = rows
    if out["LRU"] and out["TeamCache"]:
        n = min(len(out["LRU"]), len(out["TeamCache"]))
        for k in ("8", "16"):
            out[f"hit_ratio_{k}"] = [t[f"hits_{k}"] / max(l[f"hits_{k}"], 1)
                                     for l, t in zip(out["LRU"][:n], out["TeamCache"][:n])]
        out["speedup"] = [l["wall_s"] / t["wall_s"]
                          for l, t in zip(out["LRU"][:n], out["TeamCache"][:n])]
    return out


def summary():
    doc = {}
    for name, fn in (("nonregress", nonregress), ("ring", lambda: live("autogen", "design")),
                     ("selector", lambda: live("selector", "debug")),
                     ("mixed", lambda: replay("mixed", 12)), ("star", lambda: replay("star", 9)),
                     ("sweep", sweep), ("hicache", hicache), ("vllm", vllm)):
        try:
            doc[name] = fn()
        except (FileNotFoundError, KeyError, ValueError) as exc:
            doc[name] = {"missing": f"{type(exc).__name__}: {exc}"}
    return doc


def main():
    doc = summary()
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "gpu_a100_summary.json"
    p.write_text(json.dumps(doc, indent=1) + "\n")
    print("wrote", p)


if __name__ == "__main__":
    main()
