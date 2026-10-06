#!/usr/bin/env python3
"""Isolate how many prior exposures a prefix needs before reuse pays off.

The Cohort headroom claim is entirely (cold - wake): what a peer wake costs
when the member's private history is still cached, versus when it is not. A
first pass produced a result that would undermine that claim if taken at face
value -- re-sending an *identical* prompt looked an order of magnitude slower
than sending the same prompt plus a fresh tail. That ordering is suspicious,
because the identical prompt was always the second request and the appended one
always the third.

This probe separates the two confounds: number of prior exposures, and whether
the request appends anything new. Each arm uses a freshly generated prefix so
no arm inherits another's cache state.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request


def post(url, payload, timeout=600.0):
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
    ap.add_argument("--length", type=int, default=61440)
    ap.add_argument("--tail", type=int, default=512)
    ap.add_argument("--vocab", type=int, default=100000)
    ap.add_argument("--gap", type=float, default=1.0,
                    help="seconds between requests, to let blocks settle")
    cfg = ap.parse_args()

    url = f"http://{cfg.host}:{cfg.port}/v1/completions"
    rng = random.Random(7717)
    L, T = cfg.length, cfg.tail

    def fresh(n):
        return [rng.randrange(16, cfg.vocab) for _ in range(n)]

    def send(prompt, label):
        dt, body = post(url, {"model": cfg.model, "prompt": prompt,
                              "max_tokens": 1, "temperature": 0})
        print(f"  {label:<34} {dt:7.3f}s  prompt_tokens="
              f"{body.get('usage', {}).get('prompt_tokens')}", flush=True)
        time.sleep(cfg.gap)
        return dt

    print(f"L={L}, tail={T}, gap={cfg.gap}s\n")

    print("arm A: append after ONE exposure (the real wake pattern)")
    p = fresh(L)
    a_cold = send(p, "1st: cold, L tokens")
    a_wake = send(p + fresh(T), "2nd: L cached + T new")

    print("\narm B: identical repeat, then append")
    p = fresh(L)
    b_cold = send(p, "1st: cold, L tokens")
    b_same = send(p, "2nd: identical L tokens")
    b_wake = send(p + fresh(T), "3rd: L cached + T new")

    print("\narm C: three identical repeats")
    p = fresh(L)
    c1 = send(p, "1st: cold, L tokens")
    c2 = send(p, "2nd: identical")
    c3 = send(p, "3rd: identical")

    print("\narm D: two successive appends (a two-message wake chain)")
    p = fresh(L)
    d_cold = send(p, "1st: cold, L tokens")
    t1 = fresh(T)
    d_w1 = send(p + t1, "2nd: L cached + T new")
    d_w2 = send(p + t1 + fresh(T), "3rd: L+T cached + T new")

    print("\n--- summary ---")
    print(f"cold prefill of {L} tokens        {a_cold:7.3f}s  "
          f"({L / a_cold:,.0f} tok/s)")
    print(f"wake after 1 exposure            {a_wake:7.3f}s  "
          f"speedup {a_cold / a_wake:5.1f}x")
    print(f"wake after 2 exposures           {b_wake:7.3f}s  "
          f"speedup {b_cold / b_wake:5.1f}x")
    print(f"identical repeat (2nd)           {b_same:7.3f}s  "
          f"speedup {b_cold / b_same:5.1f}x")
    print(f"identical repeat (3rd)           {c3:7.3f}s  "
          f"speedup {c1 / c3:5.1f}x")
    print(f"second link of a wake chain      {d_w2:7.3f}s  "
          f"speedup {d_cold / d_w2:5.1f}x")


if __name__ == "__main__":
    main()
