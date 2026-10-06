#!/usr/bin/env python3
"""How many peer-team sessions actually stay resident in the prefix cache?

Every capacity claim in the analysis divides a pool size by a context length,
and the engine reports two numbers that disagree by 7.4x about what the pool is:
`kv_cache_size_tokens=22202457` against `num_gpu_blocks=1948` at
`block_size=1536`, which multiply out to 2.99M tokens. One of those gives 361
resident 60K sessions and the other 49 -- the difference between the contention
regime starting at 52 teams or at 7. That has to be measured, not derived.

The instrument is vLLM's own accounting rather than latency. `prefix_cache_
queries_total` and `prefix_cache_hits_total` are counted in tokens, so the delta
across a single request gives that prompt's exact cached fraction, with no
interference from queueing.

Sessions are probed newest-first, and the pool is filled exactly once before
that single scan. An earlier version of this probe filled and scanned in
repeated batches, which produced a wrong answer worth recording: a hit admits
nothing, but it does refresh recency, so scanning newest-to-oldest *inverts* the
eviction order. The next batch of fills then evicted the sessions that had just
been scanned first, and because the scan stops at the first miss it reported a
capacity far below the truth -- 80 sessions in one run and 40 in another, while
an intermediate checkpoint in the second run had 120 sessions simultaneously
resident. One fill and one scan avoids the interaction entirely.

  python3 probe_capacity.py --context 61440 --max-sessions 200
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.request

METRIC = re.compile(r"^vllm:prefix_cache_(queries|hits)_total\{[^}]*\}\s+(\S+)",
                    re.M)


def post(url, payload, timeout=900.0):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    return time.monotonic() - t0, body


def counters(base):
    with urllib.request.urlopen(base + "/metrics", timeout=30) as resp:
        text = resp.read().decode()
    got = {k: float(v) for k, v in METRIC.findall(text)}
    return got.get("queries", 0.0), got.get("hits", 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument("--model", default="moonshotai/Kimi-K3")
    ap.add_argument("--context", type=int, default=61440)
    ap.add_argument("--tail", type=int, default=512)
    ap.add_argument("--vocab", type=int, default=100000)
    ap.add_argument("--max-sessions", type=int, default=200,
                    help="sessions to fill before the single scan; must exceed "
                         "capacity or the scan finds no boundary")
    ap.add_argument("--out", default="capacity_probe.json")
    cfg = ap.parse_args()

    base = f"http://{cfg.host}:{cfg.port}"
    url = base + "/v1/completions"
    rng = random.Random(4242)
    L, T = cfg.context, cfg.tail

    def fresh(n):
        return [rng.randrange(16, cfg.vocab) for _ in range(n)]

    def send(prompt):
        """Returns (latency, cached fraction of this prompt)."""
        q0, h0 = counters(base)
        dt, _ = post(url, {"model": cfg.model, "prompt": prompt,
                           "max_tokens": 1, "temperature": 0})
        q1, h1 = counters(base)
        dq = q1 - q0
        return dt, ((h1 - h0) / dq if dq > 0 else float("nan"))

    print(f"context={L:,} tok  filling {cfg.max_sessions} sessions "
          f"({cfg.max_sessions * L:,} tok) then scanning once\n")

    sessions = []
    t0 = time.monotonic()
    for i in range(cfg.max_sessions):
        s = fresh(L)
        post(url, {"model": cfg.model, "prompt": s, "max_tokens": 1,
                   "temperature": 0})
        sessions.append(s)
        if (i + 1) % 20 == 0:
            print(f"  filled {i + 1:>4}/{cfg.max_sessions} "
                  f"({(i + 1) * L:>11,} tok)  "
                  f"{time.monotonic() - t0:>5.0f}s", flush=True)

    print(f"\nfill done in {time.monotonic() - t0:.0f}s; scanning newest first")
    resident, boundary, profile = 0, None, []
    for i in range(len(sessions) - 1, -1, -1):
        dt, frac = send(sessions[i] + fresh(T))
        profile.append({"session": i, "cached": frac, "latency": dt})
        if frac > 0.95:
            resident += 1
            continue
        boundary = i
        print(f"  session {i} of {len(sessions)} is the first evicted: "
              f"cached {frac * 100:.1f}%, {dt:.3f}s")
        break

    result = None
    if boundary is not None:
        result = {"resident_sessions": resident,
                  "resident_tokens": resident * L,
                  "filled_sessions": len(sessions)}
    print(f"  resident from the top: {resident} sessions "
          f"= {resident * L:,} tok\n", flush=True)
    if result is None:
        print(f"no eviction up to {len(sessions)} sessions "
              f"({len(sessions) * L:,} tok); capacity is larger than probed")
    else:
        tok = result["resident_tokens"]
        print(f"measured capacity  {result['resident_sessions']} sessions of "
              f"{L:,} tok = {tok:,} tok")
        print(f"engine reports     kv_cache_size_tokens=22,202,457  "
              f"num_gpu_blocks=1948 x block_size=1536 = 2,992,128")
        print(f"ratio to each      {tok / 22_202_457:.3f}x and "
              f"{tok / 2_992_128:.3f}x")
        print(f"implied teams of 7 fully resident: "
              f"{result['resident_sessions'] / 7:.1f}")

    with open(cfg.out, "w", encoding="utf-8") as fh:
        json.dump({"context": L, "result": result,
                   "filled": len(sessions), "scan_profile": profile}, fh,
                  indent=2)
    print(f"wrote {cfg.out}")


if __name__ == "__main__":
    main()
