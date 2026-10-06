#!/usr/bin/env python3
"""Can a gateway pin a session in an unmodified engine's prefix cache?

Cohort needs to keep a chosen subset of a team's members resident. vLLM and
SGLang expose no pinning API, but both evict by recency -- and recency is
something a gateway can drive from outside. Re-referencing a session's prefix
refreshes it, so the very property that makes LRU pathological for peer teams
is also the handle for fixing it. A "touch" is a 1-token request carrying the
session's prefix, which costs a cached read rather than a prefill.

This probe answers two questions in a single pool flush, because flushing a
22.2M-token pool costs ~18 minutes of prefill no matter how it is sliced:

  1. Does eviction actually happen? Session B is established and never touched.
     If it is slow afterwards, the pool really did turn over.
  2. Does touching prevent it? Session A is established and touched periodically
     during the flush. If it stays fast, a gateway can pin.

The two sessions are symmetric and differ only in whether they were touched, so
one flush isolates the mechanism.

  python3 probe_eviction.py --flush-multiple 1.3 --concurrency 4
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

KV_POOL_TOKENS = 22_202_457


def post(url, payload, timeout=900.0):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    return time.monotonic() - t0, body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument("--model", default="moonshotai/Kimi-K3")
    ap.add_argument("--context", type=int, default=61440)
    ap.add_argument("--tail", type=int, default=512)
    ap.add_argument("--vocab", type=int, default=100000)
    ap.add_argument("--pool", type=int, default=KV_POOL_TOKENS)
    ap.add_argument("--flush-multiple", type=float, default=1.3,
                    help="unique tokens to push through, as a multiple of pool")
    ap.add_argument("--touches", type=int, default=24,
                    help="how many times to refresh session A during the flush")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default="eviction_probe.json")
    cfg = ap.parse_args()

    url = f"http://{cfg.host}:{cfg.port}/v1/completions"
    rng = random.Random(31337)
    lock = threading.Lock()
    L, T = cfg.context, cfg.tail

    def fresh(n):
        with lock:
            return [rng.randrange(16, cfg.vocab) for _ in range(n)]

    def send(prompt):
        return post(url, {"model": cfg.model, "prompt": prompt,
                          "max_tokens": 1, "temperature": 0})[0]

    print(f"context={L:,}  pool={cfg.pool:,}  "
          f"flush={cfg.flush_multiple}x pool  concurrency={cfg.concurrency}\n")

    # --- establish the two sessions
    A, B = fresh(L), fresh(L)
    a_cold = send(A)
    b_cold = send(B)
    a_warm = send(A + fresh(T))
    b_warm = send(B + fresh(T))
    print(f"session A established: cold {a_cold:.3f}s, then warm {a_warm:.3f}s")
    print(f"session B established: cold {b_cold:.3f}s, then warm {b_warm:.3f}s")
    baseline_cold = (a_cold + b_cold) / 2
    baseline_warm = (a_warm + b_warm) / 2

    # --- flush the pool with unique content, touching A on the way
    target = int(cfg.pool * cfg.flush_multiple)
    chunks = max(target // L, 1)
    touch_every = max(chunks // max(cfg.touches, 1), 1)
    print(f"\nflushing {target:,} unique tokens as {chunks} x {L:,} "
          f"(touch A every {touch_every} chunks)\n")

    done = {"chunks": 0, "tokens": 0}
    touch_times = []
    t_start = time.monotonic()

    def flush_one(i):
        send(fresh(L))
        with lock:
            done["chunks"] += 1
            done["tokens"] += L
            n = done["chunks"]
        if n % touch_every == 0:
            dt = send(A + fresh(T))
            with lock:
                touch_times.append(dt)
            el = time.monotonic() - t_start
            print(f"  {n:>4}/{chunks} chunks  {done['tokens']:>12,} tok  "
                  f"{el:>6.0f}s elapsed  touch(A) {dt:.3f}s", flush=True)

    with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
        list(pool.map(flush_one, range(chunks)))
    flush_s = time.monotonic() - t_start
    print(f"\nflush done in {flush_s:.0f}s "
          f"({done['tokens'] / flush_s:,.0f} tok/s)")

    # --- the measurement: A was pinned by touching, B was not
    a_after = send(A + fresh(T))
    b_after = send(B + fresh(T))

    print("\n--- after a full pool turnover ---")
    print(f"cold prefill baseline          {baseline_cold:7.3f}s")
    print(f"cached-read baseline           {baseline_warm:7.3f}s")
    print(f"A (touched during flush)       {a_after:7.3f}s")
    print(f"B (never touched)              {b_after:7.3f}s")
    print()
    print(f"B/A ratio                      {b_after / a_after:7.2f}x")
    print(f"A vs cached-read baseline      {a_after / baseline_warm:7.2f}x")
    print(f"B vs cold baseline             {b_after / baseline_cold:7.2f}x")
    if touch_times:
        print(f"touch cost: median {sorted(touch_times)[len(touch_times)//2]:.3f}s "
              f"over {len(touch_times)} touches")

    verdict = ("PINNING WORKS: B was evicted, A survived"
               if b_after > 2 * a_after else
               "INCONCLUSIVE: A and B behaved alike")
    print(f"\n{verdict}")

    with open(cfg.out, "w", encoding="utf-8") as fh:
        json.dump({"context": L, "pool": cfg.pool,
                   "flush_multiple": cfg.flush_multiple,
                   "flushed_tokens": done["tokens"], "flush_s": flush_s,
                   "baseline_cold": baseline_cold,
                   "baseline_warm": baseline_warm,
                   "a_touched_after": a_after, "b_untouched_after": b_after,
                   "touch_times": touch_times, "verdict": verdict}, fh, indent=2)
    print(f"wrote {cfg.out}")


if __name__ == "__main__":
    main()
