#!/usr/bin/env python3
"""Offline bound for the exact reference string the replay issues.

The closed-form (k-c)/k assumes a strict cycle. The replay interleaves teams at
random, so the distance between consecutive references to a session has real
variance and LRU sometimes gets lucky -- measured 59.0% misses where a strict
cycle would give 100%. Comparing the mechanism against the closed form would
therefore be comparing against the wrong number. This computes Belady with
bypass on the *same* string, which for uniform item sizes is exactly optimal,
so the gap to optimal is a fact about this workload rather than an idealisation.

  python3 bound_sglang.py --teams 8 --members 4 --rounds 8 --capacity 24 \
      --quota 3 --cold 0.77 --warm 0.11
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict

from replay_sglang import reference_string


def run_lru(refs, capacity):
    """Upstream behaviour: one global recency order, everything admitted."""
    order, misses = [], []
    for team, member, _ in refs:
        key = (team, member)
        hit = key in order
        if hit:
            order.remove(key)
        else:
            if len(order) >= capacity:
                order.pop(0)
        order.append(key)
        misses.append(not hit)
    return misses


def run_bypass_fifo(refs, capacity):
    """Admission control with no notion of a group: whatever arrives first
    keeps the pool. Reaches a good aggregate and starves whole teams, which is
    why aggregate cost alone cannot justify the group part of the mechanism."""
    resident, misses = set(), []
    for team, member, _ in refs:
        key = (team, member)
        hit = key in resident
        if not hit and len(resident) < capacity:
            resident.add(key)
        misses.append(not hit)
    return misses


def run_cohort(refs, capacity, quota):
    """Each team keeps `quota` members resident; the rest are served and never
    admitted, so they cannot evict their own teammates. LRU within the quota."""
    order, misses = [], []
    for team, member, _ in refs:
        key = (team, member)
        if member >= quota:            # bypassed: served, never retained
            misses.append(True)
            continue
        hit = key in order
        if hit:
            order.remove(key)
        elif len(order) >= capacity:
            order.pop(0)
        order.append(key)
        misses.append(not hit)
    return misses


def run_belady(refs, capacity):
    """Furthest-next-use with bypass: of the cached set plus the arriving item,
    keep the `capacity` whose next use is soonest. Optimal for uniform sizes."""
    nxt = defaultdict(list)
    for i, (team, member, _) in enumerate(refs):
        nxt[(team, member)].append(i)
    cursor = defaultdict(int)

    def next_use(key, after):
        uses = nxt[key]
        i = cursor[key]
        while i < len(uses) and uses[i] <= after:
            i += 1
        cursor[key] = i
        return uses[i] if i < len(uses) else float("inf")

    resident, misses = set(), []
    for t, (team, member, _) in enumerate(refs):
        key = (team, member)
        hit = key in resident
        misses.append(not hit)
        if hit:
            continue
        if len(resident) < capacity:
            resident.add(key)
            continue
        worst = max(resident, key=lambda k: next_use(k, t))
        if next_use(worst, t) > next_use(key, t):
            resident.discard(worst)
            resident.add(key)
        # else: bypass -- the arriving item is the furthest-used of all
    return misses


def score(name, refs, misses, cold, warm):
    rate = sum(misses) / len(misses)
    wall = sum(cold if m else warm for m in misses)
    per_team = defaultdict(lambda: [0, 0])
    for (team, _, _), m in zip(refs, misses):
        per_team[team][0] += 1
        per_team[team][1] += m
    rates = sorted(m / n for n, m in per_team.values())
    return {
        "name": name, "miss": rate, "wall": wall,
        "team_p50": statistics.median(rates), "team_max": max(rates),
        "starved": sum(1 for r in rates if r >= 0.9),
        "teams": len(rates),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teams", type=int, default=8)
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--capacity", type=int, default=24)
    ap.add_argument("--quota", type=int, default=3)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--cold", type=float, default=0.77,
                    help="measured miss latency, seconds")
    ap.add_argument("--warm", type=float, default=0.11,
                    help="measured hit latency, seconds")
    cfg = ap.parse_args()

    refs = reference_string(cfg.teams, cfg.members, cfg.rounds, cfg.seed,
                            cfg.jitter)
    sessions = cfg.teams * cfg.members
    print(f"teams={cfg.teams}  members={cfg.members}  sessions={sessions}  "
          f"capacity={cfg.capacity}  load={sessions / cfg.capacity:.2f}x")
    print(f"references={len(refs)}  cold={cfg.cold}s  warm={cfg.warm}s")
    print(f"strict-cycle closed form (k-c)/k = "
          f"{(sessions - cfg.capacity) / sessions * 100:.1f}% misses\n")

    rows = [
        score("LRU (upstream)", refs, run_lru(refs, cfg.capacity),
              cfg.cold, cfg.warm),
        score("bypass, no groups", refs,
              run_bypass_fifo(refs, cfg.capacity), cfg.cold, cfg.warm),
        score(f"Cohort quota={cfg.quota}", refs,
              run_cohort(refs, cfg.capacity, cfg.quota), cfg.cold, cfg.warm),
        score("OPT (Belady+bypass)", refs, run_belady(refs, cfg.capacity),
              cfg.cold, cfg.warm),
    ]

    print(f"{'policy':<22} {'miss':>7} {'wall':>9} {'vs OPT':>7} "
          f"{'team p50':>9} {'team max':>9} {'starved':>8}")
    opt = rows[-1]
    for r in rows:
        print(f"{r['name']:<22} {r['miss'] * 100:>6.1f}% {r['wall']:>8.1f}s "
              f"{r['wall'] / opt['wall']:>6.2f}x {r['team_p50'] * 100:>8.1f}% "
              f"{r['team_max'] * 100:>8.1f}% "
              f"{r['starved']:>4}/{r['teams']}")

    coh = rows[2]
    print(f"\nCohort reaches {opt['wall'] / coh['wall'] * 100:.1f}% of OPT's "
          f"wall clock and is {rows[0]['wall'] / coh['wall']:.2f}x faster "
          f"than upstream LRU on this string")


if __name__ == "__main__":
    main()
