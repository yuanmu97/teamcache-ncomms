#!/usr/bin/env python3
"""Calibrate SGLang before the Cohort A/B: cost model, and capacity.

Three things have to be re-measured because the engine changed. SGLang's KV
pool for Kimi K3 is 420,800 tokens where vLLM's was 22.2M -- 27.0 KB per token
against 1.86 KB -- so the working point that created 1.41x pressure on vLLM is
far off scale here, and the context length has to be chosen so that capacity
lands where several teams can be run at once.

  python3 probe_sglang.py cost     --lengths 4096,8192,16384,32768
  python3 probe_sglang.py capacity --context 8192 --max-sessions 120
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request

URL = "http://127.0.0.1:30000/generate"


def post(input_ids, rid=None, timeout=1800.0):
    body = {"input_ids": input_ids,
            "sampling_params": {"max_new_tokens": 1, "temperature": 0}}
    if rid:
        body["rid"] = rid
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read())
    meta = payload.get("meta_info") or {}
    return time.monotonic() - t0, meta


def fresh(rng, n, vocab=100000):
    return [rng.randrange(16, vocab) for _ in range(n)]


def mode_cost(cfg):
    rng = random.Random(cfg.seed)
    print(f"{'context':>9}  {'cold':>8}  {'wake':>8}  {'ratio':>7}  "
          f"{'cached_tok':>11}")
    rows = []
    for length in cfg.lengths:
        p = fresh(rng, length)
        cold, m_cold = post(p)
        # wake: the history is already resident, a short peer message arrives
        wake, m_wake = post(p + fresh(rng, cfg.tail))
        ratio = cold / wake if wake else float("nan")
        rows.append((length, cold, wake, ratio))
        print(f"{length:>9,}  {cold:>7.3f}s  {wake:>7.3f}s  {ratio:>6.1f}x  "
              f"{m_wake.get('cached_tokens', -1):>11,}  "
              f"(cold cached={m_cold.get('cached_tokens', -1):,})")

    # Fit cold(L) = a + L/r over the measured points; the intercept is the
    # per-request floor and the slope is effective prefill throughput.
    if len(rows) >= 2:
        (l0, c0, _, _), (l1, c1, _, _) = rows[0], rows[-1]
        slope = (c1 - c0) / (l1 - l0)
        print(f"\ncold(L) = {c0 - slope * l0:.3f} + L/{1 / slope:,.0f}")
    print("\npick a context where capacity lands at 25-50 sessions and the "
          "cold/wake ratio is still large")


def mode_capacity(cfg):
    """Fill, then scan every session newest-to-oldest exactly once.

    Two probe conventions matter here, both learned the hard way. A hit
    refreshes recency, so scanning more than once inverts the order being
    measured; one fill and one scan is the only safe sequence. And residency is
    *not* necessarily a suffix of fill order -- filling far past the pool drives
    the tail into thrash where requests are served without being retained, so a
    scan that stops at the first miss reports zero. Scan all of them and report
    where the hits actually land.
    """
    rng = random.Random(cfg.seed)
    sessions = [fresh(rng, cfg.context) for _ in range(cfg.max_sessions)]
    # Fill the way the replay references: the shared history plus each of the
    # peer messages the session cycles through, so a filled session occupies
    # the subtree it will actually occupy rather than a single chain.
    msgs = [[fresh(rng, cfg.tail) for _ in range(cfg.variants)]
            for _ in range(cfg.max_sessions)]
    per_session = cfg.context + cfg.variants * cfg.tail
    print(f"filling {cfg.max_sessions} sessions, {per_session:,} tokens each "
          f"({cfg.context:,} history + {cfg.variants} x {cfg.tail} message)"
          f"...", flush=True)
    t0 = time.monotonic()
    for i, s in enumerate(sessions):
        for m in msgs[i]:
            post(s + m)
        if (i + 1) % 10 == 0:
            print(f"  filled {i + 1}/{cfg.max_sessions}  "
                  f"{time.monotonic() - t0:.0f}s", flush=True)

    print("\nscanning all sessions, newest to oldest, once:", flush=True)
    marks, fracs = [], []
    for back, (s, ms) in enumerate(zip(reversed(sessions), reversed(msgs))):
        _, meta = post(s + ms[0])
        prompt = meta.get("prompt_tokens") or (cfg.context + cfg.tail)
        frac = (meta.get("cached_tokens") or 0) / prompt
        fracs.append(frac)
        marks.append("#" if frac > 0.95 else ("+" if frac > 0.05 else "."))

    print("  newest -> oldest  ('#' resident, '+' partial, '.' gone)")
    for i in range(0, len(marks), 50):
        print(f"    -{i:<3} {''.join(marks[i:i + 50])}")

    full = sum(1 for f in fracs if f > 0.95)
    partial = sum(1 for f in fracs if 0.05 < f <= 0.95)
    per_session = cfg.context + cfg.variants * cfg.tail
    print(f"\nfully resident sessions    {full}")
    print(f"partially resident         {partial}")
    print(f"tokens held by them        {full * per_session:,}")
    print(f"engine max_total_num_tokens {cfg.pool:,} -> ratio "
          f"{full * per_session / cfg.pool:.3f}x")
    print(f"\ncapacity for the replay    {full} sessions")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    c = sub.add_parser("cost")
    c.add_argument("--lengths", default="4096,8192,16384,32768")
    c.add_argument("--tail", type=int, default=512)
    c.add_argument("--seed", type=int, default=3)

    p = sub.add_parser("capacity")
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--max-sessions", type=int, default=120)
    p.add_argument("--tail", type=int, default=512)
    p.add_argument("--variants", type=int, default=4)
    p.add_argument("--pool", type=int, default=420800,
                   help="engine max_total_num_tokens, for the ratio only")
    p.add_argument("--seed", type=int, default=5)

    cfg = ap.parse_args()
    if cfg.mode == "cost":
        cfg.lengths = [int(x) for x in cfg.lengths.split(",")]
        mode_cost(cfg)
    else:
        mode_capacity(cfg)


if __name__ == "__main__":
    main()
