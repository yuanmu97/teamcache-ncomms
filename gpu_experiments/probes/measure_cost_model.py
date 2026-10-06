#!/usr/bin/env python3
"""Calibrate the prefill cost model that the Cohort analysis depends on.

Everything in the peer-team argument reduces to one comparison: what does it
cost to bring a dormant member's private history back, depending on how much of
it survived in the prefix cache. We need three points on that curve, measured
on the real engine rather than derived from published throughput numbers:

  cold    - nothing cached; the whole context is recomputed
  warm    - the whole context is cached; only a 1-token tail is new
  wake    - the context is cached and a short peer message is appended,
            which is what a real wake looks like

Run this on the inference host so the loopback path, not the SSH tunnel,
carries the prompt bytes. /v1/completions is used because it accepts raw token
ids, giving exact control over context length with no chat template in the way.

  python3 measure_cost_model.py --port 8888 --model moonshotai/Kimi-K3
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import urllib.request


def post(url: str, payload: dict, timeout: float = 600.0):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"content-type": "application/json"}
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return time.monotonic() - t0, data


def cached_tokens(usage: dict) -> int:
    """vLLM reports prefix-cache hits under a few different shapes."""
    if not usage:
        return -1
    details = usage.get("prompt_tokens_details") or {}
    for key in ("cached_tokens", "cached_prompt_tokens"):
        if key in details:
            return details[key]
        if key in usage:
            return usage[key]
    return -1


def make_prompt(rng: random.Random, length: int, vocab: int) -> list:
    return [rng.randrange(16, vocab) for _ in range(length)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument("--model", default="moonshotai/Kimi-K3")
    ap.add_argument("--vocab", type=int, default=100000)
    ap.add_argument(
        "--lengths", default="8192,16384,32768,61440,122880",
        help="context lengths to probe",
    )
    ap.add_argument("--wake-tail", type=int, default=512,
                    help="tokens appended by an incoming peer message")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="cost_model.json")
    cfg = ap.parse_args()

    url = f"http://{cfg.host}:{cfg.port}/v1/completions"
    lengths = [int(x) for x in cfg.lengths.split(",")]
    rng = random.Random(20260913)

    results = []
    for length in lengths:
        row = {"length": length, "cold": [], "warm": [], "wake": [],
               "cold_cached": [], "warm_cached": [], "wake_cached": []}
        for trial in range(cfg.repeats):
            prompt = make_prompt(rng, length, cfg.vocab)

            # cold: this exact prefix has never been seen
            dt, data = post(url, {"model": cfg.model, "prompt": prompt,
                                  "max_tokens": 1, "temperature": 0})
            row["cold"].append(dt)
            row["cold_cached"].append(cached_tokens(data.get("usage")))

            # warm: identical prompt, so the whole thing should hit
            dt, data = post(url, {"model": cfg.model, "prompt": prompt,
                                  "max_tokens": 1, "temperature": 0})
            row["warm"].append(dt)
            row["warm_cached"].append(cached_tokens(data.get("usage")))

            # wake: cached history plus a fresh peer message at the tail
            tail = make_prompt(rng, cfg.wake_tail, cfg.vocab)
            dt, data = post(url, {"model": cfg.model, "prompt": prompt + tail,
                                  "max_tokens": 1, "temperature": 0})
            row["wake"].append(dt)
            row["wake_cached"].append(cached_tokens(data.get("usage")))

            print(f"  L={length:>7} trial {trial}: "
                  f"cold {row['cold'][-1]:6.3f}s  "
                  f"warm {row['warm'][-1]:6.3f}s  "
                  f"wake {row['wake'][-1]:6.3f}s  "
                  f"cached(warm)={row['warm_cached'][-1]}", flush=True)

        for key in ("cold", "warm", "wake"):
            row[key + "_med"] = statistics.median(row[key])
        results.append(row)

    print()
    print(f"{'L':>8} {'cold_s':>8} {'warm_s':>8} {'wake_s':>8} "
          f"{'cold_tok/s':>11} {'speedup':>8} {'wake/cold':>10}")
    for row in results:
        cold, warm, wake = row["cold_med"], row["warm_med"], row["wake_med"]
        print(f"{row['length']:>8} {cold:>8.3f} {warm:>8.3f} {wake:>8.3f} "
              f"{row['length'] / cold:>11,.0f} {cold / warm:>8.2f}x "
              f"{wake / cold:>10.3f}")

    with open(cfg.out, "w", encoding="utf-8") as fh:
        json.dump({"model": cfg.model, "wake_tail": cfg.wake_tail,
                   "repeats": cfg.repeats, "rows": results}, fh, indent=2)
    print(f"\nwrote {cfg.out}")


if __name__ == "__main__":
    main()
