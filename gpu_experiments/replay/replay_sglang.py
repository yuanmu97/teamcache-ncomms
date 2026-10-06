#!/usr/bin/env python3
"""Peer-team replay against SGLang, with Cohort implemented inside the engine.

This is the clean version of the experiment. The gateway attempt (`replay.py`)
had to buy residency with extra traffic, and at 21 teams that traffic cost more
than the misses it prevented: 2600.8 s against 1604.9 s for untouched LRU, with
median touch latency 5.43 s because the refreshes were themselves prefills.
With retention decided where eviction is decided, the two arms issue *byte
identical* request streams and differ only in the engine's `COHORT_QUOTA`. There
is no overhead to attribute and no keeper to tune.

Two other things improve by moving to SGLang's native endpoint. `input_ids`
gives exact context control with no chat template, and `meta_info.cached_tokens`
reports each request's cached prefix directly, so hit measurement is exact at
any concurrency instead of being inferred from a global counter delta or from
latency.

Request ids carry the group: "<team>:<member>:<turn>". That is the whole
protocol change.

  # baseline arm: engine started with COHORT_QUOTA unset
  python3 replay_sglang.py --label baseline --teams 21 --rounds 5 --capacity 100
  # cohort arm: engine restarted with COHORT_QUOTA=4, same command otherwise
  python3 replay_sglang.py --label cohort-q4 --teams 21 --rounds 5 --capacity 100
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import urllib.request
from collections import defaultdict


def post(url, payload, timeout=1800.0):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    return time.monotonic() - t0, body


def reference_string(teams, members, rounds, seed, jitter):
    """Interleaved peer teams; the same generator optimal.py simulates."""
    rng = random.Random(seed)
    turn = {t: 0 for t in range(teams)}
    weight = [1.0 + jitter * rng.random() for _ in range(teams)]
    live = list(range(teams))
    refs = []
    for _ in range(rounds * teams * members):
        t = rng.choices(live, weights=[weight[x] for x in live])[0]
        refs.append((t, turn[t] % members, turn[t]))
        turn[t] += 1
    return refs


class Replay:
    def __init__(self, cfg):
        self.cfg = cfg
        self.url = f"http://{cfg.host}:{cfg.port}/generate"
        # Fixed-length private histories keep item sizes uniform, which is the
        # regime where Belady is exactly optimal and optimal.py's OPT is a true
        # bound rather than an estimate.
        self.context = {
            (t, m): [random.Random((t << 16) ^ m ^ cfg.seed)
                     .randrange(16, cfg.vocab) for _ in range(cfg.context)]
            for t in range(cfg.teams) for m in range(cfg.members)
        }
        # A turn must strictly extend what the engine stored, or it is not
        # served from cache at all: measured, a byte-identical re-request
        # recomputes the whole prompt (0.694 s against 0.704 s cold, zero
        # cached tokens) while the same prefix plus 512 tokens reads in 0.107 s.
        # So each turn appends an arriving peer message. Drawing that message
        # fresh every turn would instead grow an unbounded fan of sibling
        # branches under the shared prefix -- 31% of the pool went to dead
        # branches in one run -- so messages cycle through a fixed set. Each
        # session then occupies context + variants x message tokens, bounded
        # and equal across sessions, which is what makes Belady exactly optimal.
        self.messages = {
            (t, m): [[random.Random((t << 24) ^ (m << 8) ^ v ^ cfg.seed)
                      .randrange(16, cfg.vocab) for _ in range(cfg.message)]
                     for v in range(cfg.variants)]
            for t in range(cfg.teams) for m in range(cfg.members)
        }
        self.rng = random.Random(cfg.seed ^ 0x5EED)
        self.stats = []

    def send(self, team, member, turn):
        msg = self.messages[(team, member)][turn % self.cfg.variants]
        dt, body = post(self.url, {
            "input_ids": self.context[(team, member)] + msg,
            "sampling_params": {"max_new_tokens": self.cfg.gen_tokens,
                                "temperature": 0},
            "rid": f"{team}:{member}:{turn}",
        })
        meta = body.get("meta_info") or {}
        prompt = meta.get("prompt_tokens") or (self.cfg.context + self.cfg.message)
        cached = meta.get("cached_tokens")
        row = {"team": team, "member": member, "latency": dt,
               "cached_tokens": cached,
               "cached": (cached / prompt) if cached is not None else None}
        self.stats.append(row)
        return row

    def reset_cache(self):
        """Flush the pool with content unique to this label, so an arm never
        inherits the previous arm's warm cache. Sharing the flush content would
        make the second reset a run of cache hits that evict nothing."""
        n = int(self.cfg.capacity * self.cfg.reset_multiple)
        print(f"resetting cache with {n} unique {self.cfg.context:,}-token "
              f"contexts...", flush=True)
        t0 = time.monotonic()
        rng = random.Random(0xC0FFEE + sum(ord(c) for c in self.cfg.label))
        for _ in range(n):
            post(self.url, {
                "input_ids": [rng.randrange(16, self.cfg.vocab)
                              for _ in range(self.cfg.context)],
                "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            })
        print(f"reset done in {time.monotonic() - t0:.0f}s\n", flush=True)

    def run(self):
        cfg = self.cfg
        refs = reference_string(cfg.teams, cfg.members, cfg.rounds, cfg.seed,
                                cfg.jitter)
        print(f"label={cfg.label}  teams={cfg.teams}  members={cfg.members}  "
              f"context={cfg.context:,} tok  rounds={cfg.rounds}")
        print(f"sessions={cfg.teams * cfg.members}  references={len(refs)}")
        print(f"capacity={cfg.capacity} sessions  "
              f"load={cfg.teams * cfg.members / cfg.capacity:.2f}x")
        print("the engine decides retention; this client is identical in both "
              "arms\n", flush=True)

        if cfg.reset_multiple > 0:
            self.reset_cache()

        t0 = time.monotonic()
        step = max(len(refs) // 12, 1)
        for i, (team, member, turn) in enumerate(refs):
            self.send(team, member, turn)
            if (i + 1) % step == 0:
                recent = [r["cached"] for r in self.stats[-50:]
                          if r["cached"] is not None]
                hit = statistics.mean(recent) if recent else float("nan")
                print(f"  {i + 1:>5}/{len(refs)} turns  "
                      f"{time.monotonic() - t0:>6.0f}s  "
                      f"cached(last 50) {hit * 100:>5.1f}%", flush=True)
        self.report(time.monotonic() - t0)

    def report(self, wall):
        cfg = self.cfg
        rows = self.stats
        lat = sorted(r["latency"] for r in rows)
        cached = [r["cached"] for r in rows if r["cached"] is not None]
        cut = cfg.miss_threshold
        misses = sum(1 for r in rows if r["latency"] > cut)

        per_team = defaultdict(lambda: [0, 0])
        for r in rows:
            per_team[r["team"]][0] += 1
            per_team[r["team"]][1] += (r["latency"] > cut)
        rates = sorted(m / n for n, m in per_team.values())

        print(f"\n--- {cfg.label} ---")
        print(f"wall clock                    {wall:>10.1f} s")
        print(f"turns                         {len(rows):>10}")
        print(f"turns per second              {len(rows) / wall:>10.2f}")
        if cached:
            print(f"mean cached fraction          "
                  f"{statistics.mean(cached) * 100:>9.1f}%")
        print(f"turn miss rate (>{cut:.1f}s)         "
              f"{misses / len(rows) * 100:>9.1f}%")
        print(f"turn latency p50 / p95        {lat[len(lat) // 2]:>10.2f} / "
              f"{lat[int(len(lat) * 0.95)]:.2f} s")
        print(f"per-team miss rate p50 / max  "
              f"{statistics.median(rates) * 100:>9.1f}% / "
              f"{max(rates) * 100:.1f}%")
        print(f"starved teams (>=90% miss)    "
              f"{sum(1 for r in rates if r >= 0.9):>10} / {len(rates)}")

        out = cfg.out or f"replay_sgl_{cfg.label}_{cfg.teams}t.json"
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({
                "label": cfg.label, "teams": cfg.teams,
                "members": cfg.members, "context": cfg.context,
                "rounds": cfg.rounds, "capacity": cfg.capacity,
                "wall_s": wall, "turns": len(rows),
                "mean_cached": statistics.mean(cached) if cached else None,
                "turn_miss_rate": misses / len(rows),
                "lat_p50": lat[len(lat) // 2],
                "lat_p95": lat[int(len(lat) * 0.95)],
                "team_miss_p50": statistics.median(rates),
                "team_miss_max": max(rates),
                "starved_teams": sum(1 for r in rates if r >= 0.9),
            }, fh, indent=2)
        print(f"\nwrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True,
                    help="names the arm; also seeds the cache-reset content")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--teams", type=int, default=21)
    ap.add_argument("--members", type=int, default=7)
    ap.add_argument("--context", type=int, default=15360)
    ap.add_argument("--message", type=int, default=512,
                    help="arriving peer message appended each turn")
    ap.add_argument("--variants", type=int, default=4,
                    help="distinct peer messages a session cycles through;\n1 would make turns byte-identical, which does not hit")
    ap.add_argument("--gen-tokens", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--capacity", type=int, required=True)
    ap.add_argument("--miss-threshold", type=float, default=1.2)
    ap.add_argument("--reset-multiple", type=float, default=1.2)
    ap.add_argument("--vocab", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=None)
    Replay(ap.parse_args()).run()


if __name__ == "__main__":
    main()
