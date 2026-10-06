#!/usr/bin/env python3
"""Trace-informed Static-rate q* from HELPER + BlockLRU.

Selects q on a closed-loop-shaped star (or mixed) reference string.
Never scores unified. Never looks at TeamCache numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from heapq import heappop, heappush
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from replay_team import speaker_order  # noqa: E402
from replay_trace import BlockLRU, _Req, make_policy  # noqa: E402

PROMPT = 9728 + 512  # closed-loop ring/star session prompt
POOL = 420800
MEMBERS = 5


def rank_map(seq):
    counts = Counter(seq)
    ranked = sorted(set(seq), key=lambda m: (-counts[m], m))
    return {m: i for i, m in enumerate(ranked)}, dict(counts)


def closed_loop_events(topology: str, teams: int, members: int,
                       warmup: int, rounds: int, seed: int, stagger: float):
    """One sequential speaker per team; teams overlap in wall time.

    Service time is 1.0 (rank-order of events). Stagger matches the GPU client.
    """
    heap = []
    states = []
    for tid in range(teams):
        order = speaker_order(topology, members, seed=(seed << 8) ^ tid)
        seq = []
        turns = defaultdict(int)
        n = members * (warmup + rounds)
        for _ in range(n):
            m = next(order)
            seq.append((m, turns[m], turns[m] < warmup * 1))  # dummy
            turns[m] += 1
        # warmup flag: first members*warmup steps
        steps = []
        turn = defaultdict(int)
        order = speaker_order(topology, members, seed=(seed << 8) ^ tid)
        for i in range(n):
            m = next(order)
            steps.append({
                "member": m,
                "turn": turn[m],
                "warmup": i < members * warmup,
            })
            turn[m] += 1
        states.append(steps)
        heappush(heap, (tid * stagger, tid, 0))
    events = []
    while heap:
        t, tid, idx = heappop(heap)
        step = states[tid][idx]
        events.append({
            "t": t,
            "team": tid,
            "member": step["member"],
            "turn": step["turn"],
            "warmup": step["warmup"],
        })
        nxt = idx + 1
        if nxt < len(states[tid]):
            heappush(heap, (t + 1.0, tid, nxt))
    return events


def replay_q(events, q, ranks, prompt=PROMPT):
    extra = {"quota": q, "window_s": 1e9, "pool_tokens": POOL}
    skip, _cfg = make_policy("group", POOL, MEMBERS, extra)
    cache = BlockLRU(POOL)
    recomputed = 0
    tot = 0
    team_hit = defaultdict(lambda: [0, 0])
    for e in events:
        rank = ranks[e["member"]]
        rid = f"{e['team']}:{rank}:{e['turn']}"
        sess = f"{e['team']}:{e['member']}"  # physical KV still keyed by member
        h, nblk = cache.lookup(sess, prompt)
        tot += prompt
        recomputed += prompt - h
        if not e["warmup"]:
            team_hit[e["team"]][0] += h
            team_hit[e["team"]][1] += prompt
        if skip(_Req(rid, prompt)):
            pass
        else:
            cache.insert(sess, nblk)
    worst = min((a / b for a, b in team_hit.values() if b), default=0.0)
    return {
        "q": q,
        "recomputed_prefix_tokens": recomputed,
        "hit": 1.0 - recomputed / tot if tot else 0.0,
        "worst_team_hit": worst,
        "n": len(events),
    }


def pick_q(events, members=MEMBERS, prompt=PROMPT):
    seq = [e["member"] for e in events]
    ranks, counts = rank_map(seq)
    scored = [replay_q(events, q, ranks, prompt=prompt) for q in range(1, members + 1)]
    # Minimize recomputed prefix; tie worst-team hit (higher), then smaller q.
    scored.sort(key=lambda r: (r["recomputed_prefix_tokens"], -r["worst_team_hit"], r["q"]))
    return {
        "member_call_counts": {str(k): v for k, v in sorted(counts.items())},
        "rank_map_member_to_freq_rank": {str(k): v for k, v in sorted(ranks.items())},
        "candidates": scored,
        "q_star": scored[0]["q"],
        "equivalent_to_lru": scored[0]["q"] >= members,
        "unified_was_consulted": False,
    }


def selector_member_steps(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        d = json.loads(line)
        hdr = d.get("headers") or {}
        if "x-team" not in hdr:
            continue
        if d.get("status") != 200:
            continue
        rows.append(d)
    members = sorted({r["member"] for r in rows})
    remap = {m: i for i, m in enumerate(members)}
    steps = []
    turns = defaultdict(int)
    for r in rows:
        m = remap[r["member"]]
        steps.append({
            "member": m,
            "turn": turns[m],
            "prompt": r["prompt_tokens"],
            "orig_member": r["member"],
        })
        turns[m] += 1
    return steps, remap


def mixed_events(n_teams: int, seed: int, stagger: float,
                 warmup: int, rounds: int, trace: Path):
    """Half ring (9728 model), half selector speaking order + prompt_tokens."""
    assert n_teams % 2 == 0
    n_half = n_teams // 2
    sel_steps, remap = selector_member_steps(trace)
    heap = []
    states = []
    kinds = []
    for tid in range(n_teams):
        if tid < n_half:
            kinds.append("ring")
            order = speaker_order("ring", MEMBERS, seed=(seed << 8) ^ tid)
            n = MEMBERS * (warmup + rounds)
            steps = []
            turn = defaultdict(int)
            for i in range(n):
                m = next(order)
                steps.append({
                    "member": m, "turn": turn[m],
                    "warmup": i < MEMBERS * warmup,
                    "prompt": PROMPT, "kind": "ring",
                })
                turn[m] += 1
        else:
            kinds.append("selector")
            steps = []
            for i, s in enumerate(sel_steps):
                rec = dict(s)
                rec["warmup"] = i < MEMBERS * warmup
                rec["kind"] = "selector"
                steps.append(rec)
        states.append(steps)
        heappush(heap, (tid * stagger, tid, 0))
    events = []
    while heap:
        t, tid, idx = heappop(heap)
        step = dict(states[tid][idx])
        step.update({"t": t, "team": tid})
        events.append(step)
        nxt = idx + 1
        if nxt < len(states[tid]):
            heappush(heap, (t + 1.0, tid, nxt))
    return events, kinds, remap


def replay_q_mixed(events, q, per_team_ranks):
    extra = {"quota": q, "window_s": 1e9, "pool_tokens": POOL}
    skip, _cfg = make_policy("group", POOL, MEMBERS, extra)
    cache = BlockLRU(POOL)
    recomputed = tot = 0
    team_hit = defaultdict(lambda: [0, 0])
    for e in events:
        ranks = per_team_ranks[e["team"]]
        rank = ranks[e["member"]]
        rid = f"{e['team']}:{rank}:{e['turn']}"
        sess = f"{e['team']}:{e['member']}"
        prompt = e["prompt"]
        h, nblk = cache.lookup(sess, prompt)
        tot += prompt
        recomputed += prompt - h
        if not e["warmup"]:
            team_hit[e["team"]][0] += h
            team_hit[e["team"]][1] += prompt
        if not skip(_Req(rid, prompt)):
            cache.insert(sess, nblk)
    worst = min((a / b for a, b in team_hit.values() if b), default=0.0)
    return {
        "q": q,
        "recomputed_prefix_tokens": recomputed,
        "hit": 1.0 - recomputed / tot if tot else 0.0,
        "worst_team_hit": worst,
        "n": len(events),
    }


def pick_q_mixed(events, n_teams):
    per_team_seq = defaultdict(list)
    for e in events:
        per_team_seq[e["team"]].append(e["member"])
    per_team_ranks = {t: rank_map(seq)[0] for t, seq in per_team_seq.items()}
    scored = [replay_q_mixed(events, q, per_team_ranks) for q in range(1, MEMBERS + 1)]
    scored.sort(key=lambda r: (r["recomputed_prefix_tokens"], -r["worst_team_hit"], r["q"]))
    return {
        "q_star": scored[0]["q"],
        "equivalent_to_lru": scored[0]["q"] >= MEMBERS,
        "candidates": scored,
        "per_team_rank_map": {str(t): {str(k): v for k, v in r.items()}
                              for t, r in per_team_ranks.items()},
        "one_global_q": True,
        "unified_was_consulted": False,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", choices=("star", "mixed"), default="star")
    ap.add_argument("--teams", type=int, default=9)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    CPU = HERE / "cpu"
    CPU.mkdir(exist_ok=True)
    if args.workload == "star":
        events = closed_loop_events("star", args.teams, MEMBERS, 2, 5, args.seed, 2.0)
        result = pick_q(events)
        result["workload"] = "star"
        result["teams"] = args.teams
        result["seed"] = args.seed
    else:
        trace = ROOT / "traces" / "selector-debug-k5-Kimi-K3.jsonl"
        events, kinds, remap = mixed_events(
            args.teams, args.seed, 2.0, 2, 5, trace)
        result = pick_q_mixed(events, args.teams)
        result["workload"] = "mixed"
        result["teams"] = args.teams
        result["kinds"] = kinds
        result["selector_remap"] = {str(k): v for k, v in remap.items()}
        result["seed"] = args.seed
    dest = args.out or (CPU / f"static_rate_{args.workload}.json")
    dest.write_text(json.dumps(result, indent=2))
    print(f"q*={result['q_star']} equiv_lru={result['equivalent_to_lru']} -> {dest}")


if __name__ == "__main__":
    main()
