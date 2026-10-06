#!/usr/bin/env python3
"""Marginal-value admission: does one rule hold across topology *and* load?

The two-layer rule in `optimal.py` (`run_unified`) compares a session's
measured LRU hit rate against `share / members`, the hit rate a team gets by
concentrating its share on a fixed subset. That comparison assumes members are
referenced at equal rates, and the load sweep is where the assumption shows:
it holds inside ~1.1pp at 1.0-1.6x load and then breaks, worst of all on
`debate` at 2.5x (+36.8pp), where two principals carry 81% of a team's
references. Concentrating on those two is worth 0.8, not `share / members` =
0.4, so the rule sends them to the shared pool and they thrash there.

The fix is to stop comparing against a topology-derived constant and compare
the two things that are actually being traded:

  pinning session m gains   rate(m) * (1 - P_hit(m))
  and costs one pool slot, worth  d(hits)/d(slot)  to everyone else

Both are readable off the reuse-distance histogram: growing an LRU pool from s
to s+1 slots converts exactly the references whose reuse distance equals s,
which is Mattson's stack-distance result and the basis of every miss-ratio
curve. So the policy needs no notion of a topology, a member index, or a
threshold -- it needs a histogram it already has the samples for.
"""

from __future__ import annotations

import argparse
import importlib.util
from collections import OrderedDict, defaultdict

# The rule this file arrived at lives in optimal.py; what stays here is the
# two formulations it had to refute to get there, kept runnable because the
# measurements that killed them are the argument for its shape.

_spec = importlib.util.spec_from_file_location("optimal", "optimal.py")
opt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(opt)


def run_mrc(refs, capacity, members, window=200, min_samples=2, band=0.1):
    """Pin a session when it is worth more than the pool slot it costs.

    `band` smooths the histogram: the exact-distance count is too sparse to
    read a derivative off, so the marginal value of a slot is averaged over
    distances within `band` of the current pool size.
    """
    teams = max(len({t for t, _ in refs}), 1)
    share = max(int(capacity / teams), 1)

    cache = OrderedDict()
    pinned, protect = set(), defaultdict(set)
    dist = defaultdict(list)
    hist = []                      # recent reuse distances, fleet-wide
    seen = defaultdict(int)        # session -> references in the window
    last, clock, flags = {}, 0, []
    total = 0

    def stats(key):
        d = dist[key]
        if len(d) < min_samples:
            return None
        p_hit = sum(1 for x in d if x < capacity) / len(d)
        rate = seen[key] / max(total, 1)
        return rate * (1.0 - p_hit)

    def slot_value(pool):
        """d(hits)/d(slot) for the shared pool, from the distance histogram."""
        if not hist or pool <= 0:
            return 0.0
        lo, hi = pool * (1 - band), pool * (1 + band)
        width = max(hi - lo, 1.0)
        return sum(1 for x in hist if lo <= x < hi) / width / len(hist)

    for team, member in refs:
        key = (team, member)
        hit = key in cache
        flags.append(not hit)
        if hit:
            cache.move_to_end(key)

        if key in last:
            since = last[key]
            d = sum(1 for v in last.values() if v > since)
            dist[key].append(d)
            if len(dist[key]) > window:
                del dist[key][:-window]
            hist.append(d)
            if len(hist) > window * members:
                del hist[:-window * members]
        last[key] = clock
        clock += 1
        seen[key] += 1
        total += 1

        if hit:
            continue

        gain = stats(key)
        if gain is None:
            admit, pin = True, False            # still measuring: act as LRU
        else:
            cost = slot_value(capacity - len(pinned))
            want = gain > cost
            # The per-team cap is the only group-aware term left, and it is
            # here for fairness rather than for aggregate cost: without it a
            # team whose sessions all score well takes slots until the pool is
            # gone and the quiet teams starve.
            room = (member in protect[team]
                    or (len(protect[team]) < share and len(pinned) < capacity))
            if not room and protect[team] and want:
                weakest = min(protect[team],
                              key=lambda m: stats((team, m)) or 0.0)
                if (stats((team, weakest)) or 0.0) < gain * 0.8:
                    protect[team].discard(weakest)
                    pinned.discard((team, weakest))
                    room = True
            pin = want and room
            # A session not worth pinning still competes in the pool, unless
            # the pool is too small to hold it -- which is what bypass is for,
            # and is decided by the same histogram rather than by a mode flag.
            admit = pin or (gain > 0 and slot_value(capacity - len(pinned)) > 0)

        if not admit:
            continue
        cache[key] = True
        if pin:
            pinned.add(key)
            protect[team].add(member)
        while len(cache) > capacity:
            victim = next((k for k in cache if k not in pinned), None)
            if victim is None or victim == key:
                break
            del cache[victim]
    return flags


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--teams", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--capacities", type=lambda s: [int(x) for x in s.split(",")],
                    default=[50, 40, 31, 25, 20, 15, 10])
    ap.add_argument("--topologies", type=lambda s: s.split(","),
                    default=["ring", "empirical", "star", "debate", "pairs", "mixed"])
    cfg = ap.parse_args()

    K, T, R = cfg.members, cfg.teams, cfg.rounds
    print("gap = policy miss - best fixed policy miss, per topology\n")
    for name, fn in (("two-layer", opt.run_unified), ("marginal", run_mrc),
                     ("concentrate", opt.run_concentrate)):
        print(f"--- {name} ---")
        print(f"{'cap':>5}{'load':>7}" + "".join(f"{t:>11}" for t in cfg.topologies)
              + f"{'worst':>9}")
        for C in cfg.capacities:
            row = []
            for t in cfg.topologies:
                refs = opt.topo_reference(t, T, K, R, seed=0)
                best = min(sum(opt.run_lru(refs, C)) / len(refs),
                           sum(opt.run_lru_bypass(refs, C)) / len(refs),
                           sum(opt.run_cohort(refs, C, K)) / len(refs))
                row.append(sum(fn(refs, C, K)) / len(refs) - best)
            print(f"{C:>5}{T * K / C:>6.2f}x"
                  + "".join(f"{g * 100:>+10.1f}pp" for g in row)
                  + f"{max(row) * 100:>+8.1f}pp")
        print()


if __name__ == "__main__":
    main()
