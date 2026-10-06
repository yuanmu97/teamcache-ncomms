#!/usr/bin/env python3
"""Pick an e2e team count from one collected team trace.

Selector (and any free-form harness) must not inherit the ring's 20-team
point: intra-team locality is better, so the same fleet may still fit.
Replicate the recorded (member, size) stream across T teams, interleave
them the way concurrent teams actually arrive, and sweep T until LRU and
the best Cohort quota diverge under a full pool.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_trace import make_policy  # noqa: E402
from plan_workload import run  # noqa: E402


def load_stream(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        if d.get("status") != 200 or ":" not in (d.get("rid") or ""):
            continue
        team, member, turn = d["rid"].split(":")
        rows.append((int(member), int(turn), int(d["prompt_tokens"])))
    rows.sort(key=lambda r: r[1])
    return rows


def multiplex(stream, teams):
    refs = []
    # Interleave by recorded turn index, then by team, matching concurrent
    # teams that advance together.
    by_turn = {}
    for member, turn, size in stream:
        by_turn.setdefault(turn, []).append((member, size))
    for turn in sorted(by_turn):
        for t in range(teams):
            for member, size in by_turn[turn]:
                refs.append((f"team{t}", member, turn, size))
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--pool", type=int, default=420800)
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--teams", default="8,12,16,20,24,32")
    cfg = ap.parse_args()
    stream = load_stream(cfg.trace)
    if not stream:
        raise SystemExit("empty stream")
    first = min(s for _, _, s in stream)
    last = max(s for _, _, s in stream)
    print(f"stream {len(stream)} req  first={first} last={last} pool={cfg.pool}")
    print(f"{'T':>4}{'sess':>6}{'over':>8}{'LRU':>8}{'best q':>8}{'Cohort':>8}"
          f"{'unified':>9}{'gap':>8}")
    pick = None
    for t in [int(x) for x in cfg.teams.split(",")]:
        refs = multiplex(stream, t)
        sess = t * cfg.members
        over = sess * last / cfg.pool
        lru, _ = run(refs, cfg.pool, None)
        bq, bh = 0, -1.0
        for q in range(1, cfg.members):
            h, _ = run(refs, cfg.pool,
                       make_policy("group", cfg.pool, cfg.members,
                                   {"quota": q})[0])
            if h > bh:
                bq, bh = q, h
        uh, _ = run(refs, cfg.pool,
                    make_policy("unified", cfg.pool, cfg.members, {})[0])
        gap = (bh - lru) * 100
        print(f"{t:4d}{sess:6d}{over:7.2f}x{lru*100:7.2f}%{bq:8d}"
              f"{bh*100:7.2f}%{uh*100:8.2f}%{gap:+7.1f}pp")
        if pick is None and over > 1.0 and gap >= 5.0:
            pick = (t, bq, over, gap)
    if pick:
        print(f"\nTEAMS={pick[0]} QUOTA={pick[1]}  "
              f"peak oversub {pick[2]:.2f}x  Cohort-LRU {pick[3]:+.1f}pp")
    else:
        print("\nno cell both oversubscribed and ≥5pp apart — raise --teams")


if __name__ == "__main__":
    main()
