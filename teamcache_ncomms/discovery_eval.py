#!/usr/bin/env python3
"""S15: Discovery robustness.

Three questions about serving without exported team ids.

A. Coverage and abstention.  Over every collected trace, which fraction of
   requests Discovery labels (a member lineage seen more than once, inside an
   inferred team of two or more members) and how pure those labels are.
   Everything else is abstained on and takes the engine's upstream path.

B. Inferred-id replay.  The live GPU traces (20-team ring, 32-team selector
   with its chooser calls) replayed through the block-level cache of S11,
   with TeamCache told (i) the ids the gateway exported, (ii) the ids
   Discovery infers online from prompt fingerprints alone, and (iii) nothing.

C. Error injection.  The S11 workloads with the labels TeamCache sees
   corrupted at controlled rates: abstention on whole members, members
   attached to the wrong team, and lineages split mid-session.

D. Selector chooser calls.  Their share of the live selector runs, and the
   selector traces replayed with the chooser calls kept in the gateway's
   default team, given a member of their own team, or bypassed.

    python -m teamcache_ncomms.discovery_eval     # writes s15_discovery.json
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import block_sim as B
from .paths import CACHE_LEVEL_OUT, DATA, REPO

TD_PATH = REPO / "gpu_experiments" / "analysis" / "team_discovery.py"


def _td():
    spec = importlib.util.spec_from_file_location("_team_discovery", TD_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_team_discovery"] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- A
def coverage():
    td = _td()
    traces = DATA / "traces"
    out = []
    for name, files in td.CASES:
        if any(not (traces / f).exists() for f in files):
            continue
        loaded = [(f.rsplit(".", 1)[0], td.load_jsonl(traces / f)) for f in files]
        filt = td._agent_rows if name.endswith("-agents") else None
        rows = td.overlay_rows(loaded, filter_fn=filt)
        if not rows:
            continue
        gt_for = td.attach_ground_truth(loaded)
        disc = td.TeamDiscovery()
        lids = [disc.observe(r).member_id for r in rows]
        teams = [disc.team_of(l) for l in lids]
        n_obs = Counter(lids)
        team_size = defaultdict(set)
        for l, t in zip(lids, teams):
            team_size[t].add(l)
        member_cov = [n_obs[l] > 1 for l in lids]
        covered = [n_obs[l] > 1 and len(team_size[t]) > 1 for l, t in zip(lids, teams)]
        gt = [gt_for(r) for r in rows]
        idx = [i for i, c in enumerate(covered) if c]
        if idx:
            tp, mp = td.pair_prf([teams[i] for i in idx], [gt[i][0] for i in idx]), \
                td.pair_prf([lids[i] for i in idx], [gt[i][1] for i in idx])
        else:
            tp = mp = (1.0, 1.0, 1.0)
        out.append({"case": name, "requests": len(rows),
                    "member_coverage": sum(member_cov) / len(rows),
                    "coverage": len(idx) / len(rows),
                    "abstained": 1 - len(idx) / len(rows),
                    "team_pair_precision_covered": tp[0],
                    "team_pair_recall_covered": tp[1],
                    "member_pair_precision_covered": mp[0],
                    "member_pair_recall_covered": mp[1]})
    return out


# --------------------------------------------------------------------------- B
TRACE_CASES = [("ring, 20 teams (live AutoGen)", "e2e-autogen-design-lru-20t-p3000r8.jsonl"),
               ("selector, 32 teams (live AutoGen, chooser calls included)",
                "e2e-selector-debug-lru-32t-p3000r8.jsonl")]


def _trace(name):
    td = _td()
    rows = td.load_jsonl(DATA / "traces" / name)
    rows = [r for r in rows if r.get("prefix_sha") and r.get("prompt_tokens")]
    rows.sort(key=lambda r: r["t_recv"])
    refs, prompts, gt_labels = [], [], []
    for r in rows:
        team, member = str(r["rid"]).split(":")[:2]
        refs.append((team, int(member)))
        prompts.append(B._blocks(int(r["prompt_tokens"])))
        gt_labels.append((team, int(member)))
    disc = td.TeamDiscovery()
    lids = [disc.observe(dict(r, _t=r["t_recv"])).member_id for r in rows]
    teams = [disc.team_of(l) for l in lids]
    n_obs = Counter(lids)
    members = defaultdict(dict)
    for l, t in zip(lids, teams):
        members[t].setdefault(l, len(members[t]))
    inferred = [(t, members[t][l]) if n_obs[l] > 1 and len(members[t]) > 1 else None
                for l, t in zip(lids, teams)]
    by_tenant = defaultdict(dict)
    tenant = []
    for l, (team, _) in zip(lids, gt_labels):
        by_tenant[team].setdefault(l, len(by_tenant[team]))
        tenant.append((team, by_tenant[team][l]) if n_obs[l] > 1 else None)
    scored = [str(r["rid"]).startswith("team") for r in rows]
    return refs, prompts, gt_labels, inferred, tenant, scored


def _trace_job(args):
    label, name, load = args
    refs, prompts, gt, inferred, tenant, scored = _trace(name)
    teams = len({t for (t, _), s in zip(refs, scored) if s})
    last = {}
    for k, p in zip(refs, prompts):
        last[k] = p
    ws = sum(last.values())
    mean_prompt = float(np.mean(prompts))
    capacity = int(ws / load + teams * mean_prompt)
    res = []
    for arm, policy, labels in (("LRU", "LRU", None),
                                ("TeamCache, exported ids", "TeamCache", gt),
                                ("TeamCache, Discovery ids", "TeamCache", inferred),
                                ("TeamCache, tenant team + Discovery member", "TeamCache",
                                 tenant)):
        rows = B.run(policy, refs, prompts, 0, capacity, mean_prompt, None,
                     labels=labels, inflight=teams)
        rows = [r for r, s in zip(rows, scored) if s]
        res.append({"trace": label, "load": load, "arm": arm, **B.metrics(rows)})
    cov = sum(l is not None for l, s in zip(inferred, scored) if s) / sum(scored)
    for r in res:
        r["discovery_coverage_agent_calls"] = cov
    return res


# --------------------------------------------------------------------------- D
# The selector's speaker-choice ("chooser") calls carry no x-team header, so
# the gateway files all of them under its default team and mints a member per
# distinct prompt.  D measures how much of the live workload they are and
# replays the selector traces with three identity treatments for them.
SELECTOR_GPU = [("B300", "Kimi K3", "traces/e2e-selector-debug-{arm}-32t-p3000r8.jsonl",
                 ("lru", "unified")),
                ("A100", "GLM-4.7-Flash",
                 "gpu_a100/traces/e2e-selector-debug-{arm}-20t-p3000r8-a100{seed}.jsonl",
                 ("lru", "unified", "cohort"))]
ARM_NAME = {"lru": "LRU", "unified": "TeamCache", "cohort": "Static Partitioning"}
A100_SEEDS = ("", "-s1", "-s2")
CHOOSER_TOL = 0.25          # seconds between a chooser call and its team's turns
CHOOSER_MEMBER = 10_000     # member index of the dedicated chooser identity


def _is_chooser(r):
    return str(r.get("rid", "")).startswith("t0:")


def chooser_share():
    out = []
    for hw, model, pattern, arms in SELECTOR_GPU:
        for arm in arms:
            for seed in (A100_SEEDS if hw == "A100" else ("",)):
                path = DATA / pattern.format(arm=arm, seed=seed)
                if not path.exists():
                    continue
                rows = [json.loads(l) for l in open(path)]
                rows = [r for r in rows if r.get("status") == 200 and r.get("prompt_tokens")]
                ch = [r for r in rows if _is_chooser(r)]
                tok = sum(r["prompt_tokens"] for r in rows)
                ch_tok = sum(r["prompt_tokens"] for r in ch)
                out.append({"hardware": hw, "model": model, "arm": ARM_NAME[arm],
                            "seed": seed or "-s0", "requests": len(rows),
                            "chooser_requests": len(ch),
                            "chooser_request_share": len(ch) / len(rows),
                            "chooser_token_share": ch_tok / tok,
                            "chooser_default_team_members": len({r["rid"].split(":")[1]
                                                                 for r in ch}),
                            "chooser_reuse": sum(r.get("cached_tokens") or 0 for r in ch)
                            / max(ch_tok, 1)})
    return out


def _attribute_choosers(rows):
    """Team of each chooser call from timing: AutoGen calls the selector right
    after a member of its team finishes and starts the chosen member right
    after the selector returns.  Returns {row index: team} and match counts."""
    by_team = defaultdict(list)
    for r in rows:
        if not _is_chooser(r):
            by_team[r["team"]].append(r)
    team_of, counts = {}, Counter()
    for i, c in enumerate(rows):
        if not _is_chooser(c):
            continue
        cands = []
        for t, rs in by_team.items():
            before = [c["t_recv"] - r["t_done"] for r in rs
                      if 0 <= c["t_recv"] - r["t_done"] <= CHOOSER_TOL]
            after = [r["t_recv"] - c["t_done"] for r in rs
                     if 0 <= r["t_recv"] - c["t_done"] <= CHOOSER_TOL]
            if before and after:
                cands.append((0, min(before) + min(after), t))
            elif before or after:
                cands.append((1, min(before or after), t))
        cands.sort()
        if not cands:
            counts["unattributed"] += 1
            continue
        both = [x for x in cands if x[0] == 0]
        counts["unique_both_sides" if len(both) == 1 else
               "ambiguous_both_sides" if both else "one_side"] += 1
        team_of[i] = cands[0][2]
    return team_of, counts


def _chooser_job(args):
    hw, path, load = args
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if r.get("status") == 200 and r.get("prefix_sha")
            and r.get("prompt_tokens")]
    rows.sort(key=lambda r: r["t_recv"])
    team_of, counts = _attribute_choosers(rows)
    refs, prompts, pooled, own, bypass, chooser = [], [], [], [], [], []
    for i, r in enumerate(rows):
        team, member = str(r["rid"]).split(":")[:2]
        prompts.append(B._blocks(int(r["prompt_tokens"])))
        if not _is_chooser(r):
            key = (team, int(member))
            refs.append(key)
            pooled.append(key), own.append(key), bypass.append(key), chooser.append(False)
            continue
        chooser.append(True)
        pooled.append((team, int(member)))
        bypass.append(B.BYPASS)
        if i in team_of:
            # The selector prompt of a team extends the previous one, so the
            # team's chooser calls form one physical lineage in the cache.
            refs.append((team_of[i], CHOOSER_MEMBER))
            own.append((team_of[i], CHOOSER_MEMBER))
        else:
            refs.append(("unattributed", i))
            own.append((team, int(member)))
    teams = len({k[0] for k, c in zip(refs, chooser) if not c})
    last = {}
    for k, p in zip(refs, prompts):
        last[k] = p
    capacity = int(sum(last.values()) / load + teams * float(np.mean(prompts)))
    res = []
    for arm, policy, labels in (("LRU", "LRU", None),
                                ("TeamCache, chooser calls in one default team (deployed)",
                                 "TeamCache", pooled),
                                ("TeamCache, chooser calls as a member of their own team",
                                 "TeamCache", own),
                                ("TeamCache, chooser calls bypassed", "TeamCache", bypass)):
        out = B.run(policy, refs, prompts, 0, capacity, float(np.mean(prompts)), None,
                    labels=labels, inflight=teams)
        skip = int(len(out) * B.WARMUP)
        ch = [o for j, (o, c) in enumerate(zip(out, chooser)) if c and j >= skip]
        every = out[skip:]
        res.append({"hardware": hw, "trace": path.name, "load": load, "arm": arm,
                    **B.metrics([o for o, c in zip(out, chooser) if not c]),
                    "chooser_tok": sum(o[1] for o in ch) / max(sum(o[2] for o in ch), 1),
                    "all_tok": sum(o[1] for o in every) / sum(o[2] for o in every),
                    "attribution": dict(counts)})
    return res


def chooser_counterfactual():
    jobs = []
    for hw, _, pattern, _ in SELECTOR_GPU:
        for seed in (A100_SEEDS if hw == "A100" else ("",)):
            path = DATA / pattern.format(arm="lru", seed=seed)
            if path.exists():
                jobs += [(hw, path, load) for load in B.LOADS]
    with ProcessPoolExecutor() as pool:
        rows = [r for ch in pool.map(_chooser_job, jobs) for r in ch]
    summary = []
    for hw in sorted({r["hardware"] for r in rows}):
        for arm in dict.fromkeys(r["arm"] for r in rows):
            rs = [r for r in rows if r["hardware"] == hw and r["arm"] == arm]
            summary.append({"hardware": hw, "arm": arm, "n": len(rs),
                            "member_tok_mean": float(np.mean([r["tok"] for r in rs])),
                            "member_worst_mean": float(np.mean([r["tok_worst"] for r in rs])),
                            "chooser_tok_mean": float(np.mean([r["chooser_tok"] for r in rs])),
                            "all_tok_mean": float(np.mean([r["all_tok"] for r in rs]))})
    return {"runs": rows, "summary": summary}


# --------------------------------------------------------------------------- C
NOISE = {"abstain_member": [0.0, 0.1, 0.25, 0.5],
         "wrong_team_member": [0.0, 0.05, 0.1, 0.2],
         "split_request": [0.0, 0.05, 0.1, 0.2]}
NOISE_SEEDS = 3


def corrupt(refs, kind, rate, seed):
    rng = random.Random(seed * 101 + int(rate * 1000))
    sessions = sorted(set(refs))
    if kind == "abstain_member":
        gone = {s for s in sessions if rng.random() < rate}
        return [None if k in gone else k for k in refs]
    if kind == "wrong_team_member":
        moved = {}
        for s in sessions:
            if rng.random() < rate:
                other = rng.choice([t for t in range(B.TEAMS) if t != s[0]])
                moved[s] = (other, B.MEMBERS + s[0] * B.MEMBERS + s[1])
        return [moved.get(k, k) for k in refs]
    if kind == "split_request":
        fresh, alias = B.MEMBERS, {}
        out = []
        for k in refs:
            if rng.random() < rate:
                alias[k] = (k[0], fresh)
                fresh += 1
            out.append(alias.get(k, k))
        return out
    raise ValueError(kind)


def _noise_job(args):
    topo, seed, kind, rate = args
    cfg = B.BASE
    refs, prompts, template, mid_ws, mid_prompt = B.workload(topo, seed, cfg["cv"],
                                                             cfg["shared"], cfg["growth"])
    labels = corrupt(refs, kind, rate, seed)
    rows = []
    for load in B.LOADS:
        capacity = int(mid_ws / load + B.TEAMS * mid_prompt)
        tc = B.metrics(B.run("TeamCache", refs, prompts, template, capacity, mid_prompt,
                             None, labels=labels))
        lru = B.metrics(B.run("LRU", refs, prompts, template, capacity, mid_prompt, None))
        rows.append({"topo": topo, "seed": seed, "kind": kind, "rate": rate, "load": load,
                     **{f"tc_{k}": v for k, v in tc.items()},
                     **{f"lru_{k}": v for k, v in lru.items()}})
    return rows


def main():
    CACHE_LEVEL_OUT.mkdir(parents=True, exist_ok=True)
    cov = coverage()
    for c in cov:
        print(f"A {c['case']:30s} n={c['requests']:5d} member_cov={c['member_coverage']:.3f} "
              f"coverage={c['coverage']:.3f} "
              f"teamP={c['team_pair_precision_covered']:.3f} "
              f"memberP={c['member_pair_precision_covered']:.3f}")
    with ProcessPoolExecutor() as pool:
        trace_rows = [r for ch in pool.map(_trace_job, [(l, n, ld) for l, n in TRACE_CASES
                                                        for ld in B.LOADS]) for r in ch]
        noise_rows = [r for ch in pool.map(_noise_job, [
            (topo, seed, kind, rate) for kind, rates in NOISE.items() for rate in rates
            for topo in B.TOPOS for seed in range(NOISE_SEEDS)], chunksize=2) for r in ch]
    for r in trace_rows:
        print(f"B {r['trace'][:28]:28s} load={r['load']:<5} {r['arm']:42s} "
              f"agg={r['tok']:.3f} worst={r['tok_worst']:.3f} cov={r['discovery_coverage_agent_calls']:.3f}")
    noise = []
    clean = {(r["topo"], r["seed"], r["load"]): r for r in noise_rows if r["rate"] == 0.0
             and r["kind"] == "abstain_member"}
    for kind, rates in NOISE.items():
        for rate in rates:
            rs = [r for r in noise_rows if r["kind"] == kind and r["rate"] == rate]
            d = [r["tc_tok_worst"] - clean[(r["topo"], r["seed"], r["load"])]["tc_tok_worst"]
                 for r in rs]
            above_lru = [r["tc_tok_worst"] >= r["lru_tok_worst"] - 0.02 for r in rs]
            noise.append({"kind": kind, "rate": rate, "n": len(rs),
                          "worst_mean": float(np.mean([r["tc_tok_worst"] for r in rs])),
                          "aggregate_mean": float(np.mean([r["tc_tok"] for r in rs])),
                          "lru_worst_mean": float(np.mean([r["lru_tok_worst"] for r in rs])),
                          "worst_delta_pp_mean": 100 * float(np.mean(d)),
                          "cells_within_2pp_of_lru_or_better": int(sum(above_lru)),
                          "starved_cells": sum(r["tc_tok_worst"] < 0.10 for r in rs)})
            n = noise[-1]
            print(f"C {kind:18s} {rate:<5} worst={n['worst_mean']:.3f} "
                  f"(lru {n['lru_worst_mean']:.3f}) d={n['worst_delta_pp_mean']:+.1f}pp "
                  f">=LRU-2pp {n['cells_within_2pp_of_lru_or_better']}/{n['n']} "
                  f"starved={n['starved_cells']}")
    share = chooser_share()
    for r in share:
        print(f"D {r['hardware']} {r['arm']:20s} {r['seed']} chooser req={r['chooser_request_share']:.3f} "
              f"tok={r['chooser_token_share']:.3f} members={r['chooser_default_team_members']} "
              f"reuse={r['chooser_reuse']:.3f}")
    cf = chooser_counterfactual()
    for r in cf["summary"]:
        print(f"D {r['hardware']} {r['arm']:58s} member={r['member_tok_mean']:.3f} "
              f"worst={r['member_worst_mean']:.3f} chooser={r['chooser_tok_mean']:.3f} "
              f"all={r['all_tok_mean']:.3f}")
    json.dump({"coverage": cov, "inferred_replay": trace_rows,
               "chooser": {"gpu_share": share, **cf},
               "error_injection": {"rates": NOISE, "seeds": NOISE_SEEDS,
                                   "summary": noise, "runs": noise_rows}},
              open(CACHE_LEVEL_OUT / "s15_discovery.json", "w"), indent=1)


if __name__ == "__main__":
    main()
