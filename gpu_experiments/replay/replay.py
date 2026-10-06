#!/usr/bin/env python3
"""Peer-team replay against a real engine: LRU as shipped, versus Cohort.

Runs the same reference string twice against the same unmodified vLLM and
measures what each policy costs end to end.

  baseline   Turns are issued and nothing else. The engine's LRU decides what
             stays resident, which is what any deployment gets today.

  cohort     Identical turns, plus a keeper that re-references the members a
             group needs resident. probe_eviction.py established that this is
             enough to pin a session in an unmodified engine: after a full pool
             turnover a touched session still read from cache in 0.128 s while
             its untouched twin paid a 2.902 s prefill. The keeper always
             refreshes whichever resident member has gone longest without a
             reference, so a member whose own turn just ran costs nothing extra.

Both arms share the worker pool size, so the keeper's work shows up in the
wall-clock result rather than being hidden.

Two instruments, each used only where it is valid. The arm-level hit rate comes
from vLLM's token-denominated `prefix_cache_queries_total` and
`prefix_cache_hits_total`, which is sound at any concurrency. Per-turn hit/miss
is classified by latency, because a per-request counter delta is not
attributable once anything else is in flight -- a missing turn racing a hitting
touch reads out at 49.6% cached, right on top of any sensible threshold.
Latency has no such problem here: cold and warm differ by 21.7x, so any cut
between them is safe.

  python3 replay.py --arm baseline --teams 12 --capacity 49 --rounds 3
  python3 replay.py --arm cohort   --teams 12 --capacity 49 --rounds 3
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import threading
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

METRIC = re.compile(r"^vllm:prefix_cache_(queries|hits)_total\{[^}]*\}\s+(\S+)",
                    re.M)


def post(url, payload, timeout=1800.0):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    return time.monotonic() - t0, body


def counters(base):
    with urllib.request.urlopen(base + "/metrics", timeout=30) as resp:
        got = {k: float(v) for k, v in METRIC.findall(resp.read().decode())}
    return got.get("queries", 0.0), got.get("hits", 0.0)


def reference_string(teams, members, rounds, seed, jitter):
    """Interleaved peer teams, identical to the generator in optimal.py."""
    rng = random.Random(seed)
    turn = {t: 0 for t in range(teams)}
    weight = [1.0 + jitter * rng.random() for _ in range(teams)]
    live = list(range(teams))
    refs = []
    for _ in range(rounds * teams * members):
        t = rng.choices(live, weights=[weight[x] for x in live])[0]
        refs.append((t, turn[t] % members))
        turn[t] += 1
    return refs


class Replay:
    def __init__(self, cfg):
        self.cfg = cfg
        self.base = f"http://{cfg.host}:{cfg.port}"
        self.url = self.base + "/v1/completions"
        self.lock = threading.Lock()
        self.rng = random.Random(cfg.seed ^ 0x5EED)
        # Each member's private history is fixed-length, so item sizes are
        # uniform -- the regime where Belady is exactly optimal and the
        # simulator's OPT is a true bound rather than an estimate.
        self.context = {
            (t, m): [random.Random((t << 16) ^ m ^ cfg.seed)
                     .randrange(16, cfg.vocab) for _ in range(cfg.context)]
            for t in range(cfg.teams) for m in range(cfg.members)
        }
        self.last_ref = {}
        self.inserted = 0          # fresh contexts admitted so far
        self.inserted_at = {}      # per session, the count at its last reference
        self.turn_stats = []
        self.touch_stats = []
        self.stop = threading.Event()

        groups = cfg.teams
        usable = int(cfg.capacity * cfg.quota_margin)
        self.quota = max(min(usable // max(groups, 1), cfg.members), 0)
        self.resident = [(t, m) for t in range(groups)
                         for m in range(self.quota)]

    def tail(self, n):
        with self.lock:
            return [self.rng.randrange(16, self.cfg.vocab) for _ in range(n)]

    def send(self, key, kind):
        prompt = self.context[key] + self.tail(self.cfg.tail)
        q0, h0 = counters(self.base)
        dt, _ = post(self.url, {"model": self.cfg.model, "prompt": prompt,
                                "max_tokens": self.cfg.gen_tokens,
                                "temperature": 0})
        q1, h1 = counters(self.base)
        dq = q1 - q0
        cached = (h1 - h0) / dq if dq > 0 else float("nan")
        row = {"key": key, "kind": kind, "latency": dt, "cached": cached,
               "t": time.monotonic()}
        with self.lock:
            self.last_ref[key] = row["t"]
            # Only a miss admits a fresh context, so this counts the eviction
            # pressure the keeper paces itself against.
            if dt > self.cfg.miss_threshold:
                self.inserted += 1
            self.inserted_at[key] = self.inserted
            (self.turn_stats if kind == "turn" else self.touch_stats).append(row)
        return row

    def keeper(self):
        """Refresh residents, paced by eviction pressure rather than by time.

        Timed refreshing does not work here. A resident reads back in 0.13 s, so
        a keeper that touches anything idle for a second cycles 75 residents in
        ten seconds and consumes the whole GPU by itself, starving the turns it
        exists to help.

        What actually threatens a resident is fresh content being admitted. Only
        a miss admits anything, so counting misses counts eviction pressure
        directly: a resident is at risk once roughly `capacity - |resident|`
        new contexts have landed since it was last referenced. Touching at that
        point, and not before, makes the overhead self-tuning -- and it falls to
        zero when the workload is not inserting anything.
        """
        headroom = max(self.cfg.capacity - len(self.resident), 1)
        trigger = max(int(headroom * self.cfg.touch_safety), 1)
        print(f"keeper: headroom={headroom} sessions, "
              f"refresh after {trigger} admissions", flush=True)
        while not self.stop.is_set():
            with self.lock:
                key = min(self.resident,
                          key=lambda k: self.inserted_at.get(k, -1))
                behind = self.inserted - self.inserted_at.get(key, -1)
            if behind < trigger:
                time.sleep(0.1)
                continue
            try:
                self.send(key, "touch")
            except Exception:
                time.sleep(0.2)

    def reset_cache(self):
        """Turn the pool over with unique content so neither arm inherits the
        other's warm cache. Without this the arm that runs second starts with
        its sessions already resident and wins for the wrong reason."""
        cfg = self.cfg
        n = int(cfg.capacity * cfg.reset_multiple)
        print(f"resetting cache with {n} unique {cfg.context:,}-token "
              f"contexts...", flush=True)
        t0 = time.monotonic()
        # Distinct flush content per arm. Reusing it would make the second
        # arm's reset a string of cache hits that evict nothing, leaving the
        # first arm's sessions resident and silently rigging the comparison.
        rng = random.Random(0xC0FFEE + sum(ord(c) for c in cfg.arm))
        for _ in range(n):
            post(self.url, {"model": cfg.model,
                            "prompt": [rng.randrange(16, cfg.vocab)
                                       for _ in range(cfg.context)],
                            "max_tokens": 1, "temperature": 0})
        print(f"reset done in {time.monotonic() - t0:.0f}s\n", flush=True)

    def run(self):
        cfg = self.cfg
        refs = reference_string(cfg.teams, cfg.members, cfg.rounds, cfg.seed,
                                cfg.jitter)
        if cfg.reset_multiple > 0:
            self.reset_cache()
        print(f"arm={cfg.arm}  teams={cfg.teams}  members={cfg.members}  "
              f"context={cfg.context:,} tok  rounds={cfg.rounds}")
        print(f"sessions={cfg.teams * cfg.members}  references={len(refs)}  "
              f"workers={cfg.workers}")
        print(f"engine capacity={cfg.capacity} sessions  "
              f"load={cfg.teams * cfg.members / cfg.capacity:.2f}x")
        if cfg.arm == "cohort":
            print(f"per-team quota={self.quota}/{cfg.members} resident  "
                  f"({len(self.resident)} sessions pinned, "
                  f"margin={cfg.quota_margin})")
        print(flush=True)

        q_start, h_start = counters(self.base)
        keeper = None
        if cfg.arm == "cohort" and self.resident:
            keeper = threading.Thread(target=self.keeper, daemon=True)
            keeper.start()

        t0 = time.monotonic()
        progress = {"n": 0}

        def one(i):
            self.send(refs[i], "turn")
            with self.lock:
                progress["n"] += 1
                n = progress["n"]
            if n % max(len(refs) // 12, 1) == 0:
                el = time.monotonic() - t0
                recent = self.turn_stats[-50:]
                hit = statistics.mean(r["cached"] for r in recent)
                print(f"  {n:>5}/{len(refs)} turns  {el:>6.0f}s  "
                      f"cached(last 50) {hit * 100:>5.1f}%  "
                      f"touches={len(self.touch_stats)}", flush=True)

        with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
            list(pool.map(one, range(len(refs))))
        wall = time.monotonic() - t0
        self.stop.set()
        if keeper:
            keeper.join(timeout=30)
        q_end, h_end = counters(self.base)

        self.report(refs, wall, q_end - q_start, h_end - h_start)

    def report(self, refs, wall, dq, dh):
        cfg = self.cfg
        turns = self.turn_stats
        lat = sorted(r["latency"] for r in turns)
        cached = [r["cached"] for r in turns if r["cached"] == r["cached"]]
        cut = cfg.miss_threshold
        miss = [r for r in turns if r["latency"] > cut]

        per_team = defaultdict(lambda: [0, 0])
        for r in turns:
            per_team[r["key"][0]][0] += 1
            per_team[r["key"][0]][1] += (r["latency"] > cut)
        rates = sorted(m / n for n, m in per_team.values())

        print(f"\n--- arm={cfg.arm} ---")
        print(f"wall clock                    {wall:>10.1f} s")
        print(f"turns                         {len(turns):>10}")
        print(f"turns per second              {len(turns) / wall:>10.2f}")
        print(f"mean cached fraction          {statistics.mean(cached) * 100:>9.1f}%")
        print(f"turn miss rate (>{cut:.1f}s)         "
              f"{len(miss) / len(turns) * 100:>9.1f}%")
        print(f"turn latency p50 / p95        {lat[len(lat) // 2]:>10.2f} / "
              f"{lat[int(len(lat) * 0.95)]:.2f} s")
        print(f"engine prefix hit rate        {dh / dq * 100:>9.1f}%  "
              f"({dh:,.0f} of {dq:,.0f} tok)")
        print(f"touches issued                {len(self.touch_stats):>10}")
        if self.touch_stats:
            tl = sorted(r["latency"] for r in self.touch_stats)
            print(f"touch latency p50             {tl[len(tl) // 2]:>10.2f} s")
        print(f"per-team miss rate p50 / max  "
              f"{statistics.median(rates) * 100:>9.1f}% / {max(rates) * 100:.1f}%")
        print(f"starved teams (>=90% miss)    "
              f"{sum(1 for r in rates if r >= 0.9):>10} / {len(rates)}")

        with open(cfg.out, "w", encoding="utf-8") as fh:
            json.dump({
                "arm": cfg.arm, "teams": cfg.teams, "members": cfg.members,
                "context": cfg.context, "rounds": cfg.rounds,
                "capacity": cfg.capacity, "quota": self.quota,
                "wall_s": wall, "turns": len(turns),
                "mean_cached": statistics.mean(cached),
                "turn_miss_rate": len(miss) / len(turns),
                "lat_p50": lat[len(lat) // 2],
                "lat_p95": lat[int(len(lat) * 0.95)],
                "engine_hit_rate": dh / dq if dq else None,
                "touches": len(self.touch_stats),
                "team_miss_p50": statistics.median(rates),
                "team_miss_max": max(rates),
                "starved_teams": sum(1 for r in rates if r >= 0.9),
            }, fh, indent=2)
        print(f"\nwrote {cfg.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("baseline", "cohort"), required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument("--model", default="moonshotai/Kimi-K3")
    ap.add_argument("--teams", type=int, default=12)
    ap.add_argument("--members", type=int, default=7)
    ap.add_argument("--context", type=int, default=61440)
    ap.add_argument("--tail", type=int, default=512)
    ap.add_argument("--gen-tokens", type=int, default=1,
                    help="1 isolates prefill, which is what this workload is "
                         "bound by at a 74:1 input:output ratio")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=1,
                    help="one turn at a time keeps latency attributable; "
                         "prefill is compute bound, so concurrency buys only "
                         "about 7%% throughput here anyway")
    ap.add_argument("--miss-threshold", type=float, default=1.2,
                    help="seconds above which a turn is counted as a miss; "
                         "cold and warm are 2.80 s and 0.13 s")
    ap.add_argument("--capacity", type=int, required=True,
                    help="resident sessions the engine holds, from "
                         "probe_capacity.py")
    ap.add_argument("--quota-margin", type=float, default=0.8,
                    help="fraction of capacity to pin, leaving room for the "
                         "transient members that are deliberately not resident")
    ap.add_argument("--touch-safety", type=float, default=0.6,
                    help="fraction of headroom to spend before refreshing a "
                         "resident; lower is safer and costs more touches")
    ap.add_argument("--reset-multiple", type=float, default=1.2,
                    help="unique contexts to flush before the arm, as a "
                         "multiple of capacity; 0 skips the reset")
    ap.add_argument("--vocab", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=None)
    cfg = ap.parse_args()
    if cfg.out is None:
        cfg.out = f"replay_{cfg.arm}_{cfg.teams}t.json"
    Replay(cfg).run()


if __name__ == "__main__":
    main()
