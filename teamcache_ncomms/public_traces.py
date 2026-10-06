#!/usr/bin/env python3
"""S8: speaking orders of public multi-agent traces.

Sources (downloaded into ./public):
  Who&When (Zhang et al., ICML 2025), Kevin355/Who_and_When on Hugging Face:
    AG2/CaptainAgent group chats and Magentic-One runs with per-message speakers.
  MAST (Cemri et al., 2025), mcemri/MAST-Data on Hugging Face:
    raw logs of ChatDev, MetaGPT, HyperAgent, AppWorld and Magentic-One.

Only members backed by a language model count as members; users, system
messages and code executors are dropped.  For every trace with at least two
members and 16 member turns we report the normalised recency rank (1 for strict
rotation) and the coefficient of variation of return times, and we replay
nine concurrent teams of each system through every cache policy.
"""

from __future__ import annotations

import ast
import json
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import classic as C
from . import optimal as O
from .paths import CACHE_LEVEL_OUT, PUBLIC_RAW as PUB

OUT = CACHE_LEVEL_OUT
NON_LLM = {"human", "user", "system", "computer_terminal", "computerterminal",
           "mathproxyagent", "content"}
MIN_TURNS = 16


def clean(seq, collapse=False):
    out = []
    for s in seq:
        s = re.sub(r"\s*\(.*\)$", "", str(s)).strip()
        if not s or s.lower() in NON_LLM:
            continue
        if collapse and out and out[-1] == s:
            continue
        out.append(s)
    return out


def who_and_when():
    import pandas as pd
    out = []
    for fname, system in (("ww_Algorithm-Generated.parquet", "AG2 CaptainAgent"),
                          ("ww_Hand-Crafted.parquet", "Magentic-One")):
        df = pd.read_parquet(PUB / fname)
        for i, row in df.iterrows():
            seq = [m.get("name") or m.get("role") for m in row["history"]]
            out.append({"source": "Who&When", "system": system, "id": f"{fname}:{i}",
                        "order": clean(seq)})
    return out


MAST_PATTERNS = {
    "ChatDev": (r"INFO\] ([A-Z][\w ]+?): \*\*", True),
    "MetaGPT": (r"NEW MESSAGES:\s*\n\s*\n(\w[\w ]*?):", False),
    "HyperAgent": (r"(Navigator|Editor|Executor|Planner)", False),
    "AppWorld": (r"Response from (\w+) Agent", False),
    "Magentic": (r"---------- (\S+) ----------", False),
}
MAST_NAME = {"Magentic": "Magentic-One"}


def mast():
    data = json.load(open(PUB / "MAD_full_dataset.json"))
    out = []
    for r in data:
        if r["mas_name"] not in MAST_PATTERNS:
            continue
        t = r["trace"]
        if isinstance(t, str):
            try:
                t = ast.literal_eval(t)
            except (ValueError, SyntaxError):
                t = {"trajectory": t}
        text = str(t.get("trajectory", ""))
        pat, collapse = MAST_PATTERNS[r["mas_name"]]
        out.append({"source": "MAST", "system": MAST_NAME.get(r["mas_name"], r["mas_name"]),
                    "id": f"{r['mas_name']}:{r['benchmark_name']}:{r['trace_id']}",
                    "order": clean(re.findall(pat, text), collapse)})
    return out


def stats(order):
    members = sorted(set(order))
    k = len(members)
    last, ranks, rts = {}, [], []
    for i, m in enumerate(order):
        if m in last:
            ranks.append(len(set(order[last[m] + 1:i])) / (k - 1))
            rts.append(i - last[m])
        last[m] = i
    rts = np.asarray(rts, float)
    return {"members": k, "turns": len(order),
            "recency_rank": float(np.mean(ranks)) if ranks else None,
            "return_cv": float(rts.std() / rts.mean()) if len(rts) > 1 else None,
            "immediate_return": float(np.mean([r == 0 for r in ranks])) if ranks else None}


# ---------------------------------------------------------------- multi-team replay

TEAMS = 9
LOADS = (1.5, 2.5)
REFS = 3000


def team_stream(orders, rng, team):
    """Concatenate randomly drawn traces; each new trace is a new set of histories."""
    ep = 0
    while True:
        order = orders[rng.randrange(len(orders))]
        idx = {m: j for j, m in enumerate(dict.fromkeys(order))}
        for m in order:
            yield (team, ep * 64 + idx[m])
        ep += 1


def interleave(orders, seed):
    rng = random.Random(seed)
    gens = [team_stream(orders, random.Random(seed * 101 + t), t) for t in range(TEAMS)]
    return [next(gens[rng.randrange(TEAMS)]) for _ in range(REFS)]


def live_histories(refs, horizon=200):
    """Mean number of distinct histories referenced within a sliding horizon."""
    return np.mean([len(set(refs[i:i + horizon])) for i in range(0, len(refs) - horizon, 50)])


def static_flags(refs, cap):
    """Fixed per-team subset: the most frequent member indices of each team's episodes."""
    count = defaultdict(Counter)
    for t, key in refs:
        count[t][key % 64] += 1
    quota = max(cap // TEAMS, 1)
    keep = {t: {m for m, _ in c.most_common(quota)} for t, c in count.items()}
    resident, flags = set(), []
    for t, key in refs:
        flags.append((t, key) not in resident)
        if key % 64 in keep[t]:
            resident.add((t, key))
    return flags


def _replay_job(args):
    system, orders, load, seed = args
    refs = interleave(orders, seed)
    cap = max(int(round(live_histories(refs) / load)), 2)
    skip = REFS // 5
    tof = [t for t, _ in refs]
    flat = [(0, (t, k)) for t, k in refs]
    keys = [k for _, k in flat]
    res = {"LRU": O.run_lru(refs, cap), "Static": static_flags(refs, cap),
           "LIRS": C.run_lirs(keys, cap), "W-TinyLFU": C.run_wtinylfu(keys, cap),
           "ARC": C.run_arc(keys, cap), "S3-FIFO": C.run_s3fifo(keys, cap),
           "TeamCache": C.run_teamcache(refs, cap),
           "OPT": O.run_optb(keys, cap)}
    out = []
    for pol, f in res.items():
        per = defaultdict(lambda: [0, 0])
        for fl, t in zip(f[skip:], tof[skip:]):
            per[t][0] += 1
            per[t][1] += not fl
        hit = sum(h for _, h in per.values()) / sum(n for n, _ in per.values())
        worst = min(h / n for n, h in per.values())
        out.append({"system": system, "load": load, "seed": seed, "cap": cap,
                    "policy": pol, "hit": hit, "worst": worst})
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    traces = who_and_when() + mast()
    for t in traces:
        t.update(stats(t["order"]) if len(set(t["order"])) >= 2 else
                 {"members": len(set(t["order"])), "turns": len(t["order"]),
                  "recency_rank": None, "return_cv": None, "immediate_return": None})
    keep = [t for t in traces if t["members"] >= 2 and t["turns"] >= MIN_TURNS]
    by_sys = defaultdict(list)
    for t in keep:
        by_sys[t["system"]].append(t)
    summary = {}
    for s, ts in sorted(by_sys.items()):
        r = np.array([t["recency_rank"] for t in ts])
        summary[s] = {"traces": len(ts), "median_members": float(np.median([t["members"] for t in ts])),
                      "median_turns": float(np.median([t["turns"] for t in ts])),
                      "rank_median": float(np.median(r)),
                      "rank_q25": float(np.percentile(r, 25)), "rank_q75": float(np.percentile(r, 75)),
                      "frac_rank_ge_0.6": float(np.mean(r >= 0.6)),
                      "immediate_return_median": float(np.median([t["immediate_return"] for t in ts]))}
        print(s, summary[s])
    jobs = [(s, [t["order"] for t in ts], L, seed) for s, ts in by_sys.items()
            for L in LOADS for seed in range(6) if len(ts) >= 5]
    with ProcessPoolExecutor() as pool:
        rows = [r for ch in pool.map(_replay_job, jobs) for r in ch]
    agg = defaultdict(list)
    for r in rows:
        agg[(r["system"], r["load"], r["policy"])].append((r["hit"], r["worst"]))
    replay = [{"system": s, "load": L, "policy": p, "n": len(v),
               "hit_mean": float(np.mean([a for a, _ in v])),
               "worst_mean": float(np.mean([b for _, b in v]))}
              for (s, L, p), v in sorted(agg.items())]
    json.dump({"min_turns": MIN_TURNS, "excluded_speakers": sorted(NON_LLM),
               "n_parsed": len(traces), "n_kept": len(keep), "systems": summary,
               "traces": [{k: v for k, v in t.items() if k != "order"} | {"order": t["order"]}
                          for t in keep],
               "replay": replay, "replay_runs": rows},
              open(OUT / "s8_public_traces.json", "w"), indent=1)
    pols = ["LRU", "Static", "ARC", "LIRS", "S3-FIFO", "W-TinyLFU", "TeamCache", "OPT"]
    print("parsed", len(traces), "kept", len(keep))
    print("system".ljust(18) + "load " + " ".join(p[:9].rjust(9) for p in pols))
    for s in sorted(by_sys):
        for L in LOADS:
            cs = {r["policy"]: r for r in replay if r["system"] == s and r["load"] == L}
            if cs:
                print(s.ljust(18) + f"{L:4.1f} " + " ".join(
                    f"{100*cs[p]['hit_mean']:4.0f}/{100*cs[p]['worst_mean']:3.0f}".rjust(9) for p in pols))


if __name__ == "__main__":
    main()
