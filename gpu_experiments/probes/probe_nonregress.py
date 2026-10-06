#!/usr/bin/env python3
"""Do ordinary independent requests get slower when Cohort is enabled?

The claim to test is narrow. The predicate reads the request id and returns
False for anything that is not "<int>:<int>:...", so a normal client -- whose
request ids are engine-generated and contain no colons -- takes the same code
path as upstream. What a measurement adds is confirmation that the extra
`getattr` and string split cost nothing observable, and that nothing about the
patched file perturbs the ordinary serving path.

Run the identical command against an engine started with the mechanism off and
one with it on, and compare. No request here carries a group id.

  python3 probe_nonregress.py --label quota0
  python3 probe_nonregress.py --label quota2
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import threading
import time
import urllib.request

URL = "http://127.0.0.1:30000/generate"


def post(input_ids, max_new):
    body = {"input_ids": input_ids,
            "sampling_params": {"max_new_tokens": max_new, "temperature": 0}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        payload = json.loads(resp.read())
    return time.monotonic() - t0, (payload.get("meta_info") or {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--input-len", type=int, default=8192)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--concurrency", default="1,8,16")
    ap.add_argument("--prompts-per-level", type=int, default=16)
    ap.add_argument("--vocab", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default=None)
    cfg = ap.parse_args()
    levels = [int(x) for x in cfg.concurrency.split(",")]

    print(f"label={cfg.label}  {cfg.input_len:,} in / {cfg.output_len} out  "
          f"no request carries a group id\n")
    print(f"{'conc':>5} {'prompts':>8} {'wall':>8} {'out tok/s':>10} "
          f"{'lat p50':>9} {'lat p95':>9}")

    rows = {}
    for conc in levels:
        rng = random.Random(cfg.seed ^ conc)
        # Unique prompts so no level is served from another level's cache.
        prompts = [[rng.randrange(16, cfg.vocab) for _ in range(cfg.input_len)]
                   for _ in range(cfg.prompts_per_level)]
        lats, lock, idx = [], threading.Lock(), [0]

        def worker():
            while True:
                with lock:
                    if idx[0] >= len(prompts):
                        return
                    p = prompts[idx[0]]
                    idx[0] += 1
                dt, _ = post(p, cfg.output_len)
                with lock:
                    lats.append(dt)

        t0 = time.monotonic()
        threads = [threading.Thread(target=worker) for _ in range(conc)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.monotonic() - t0

        s = sorted(lats)
        tps = len(prompts) * cfg.output_len / wall
        rows[conc] = {"wall_s": wall, "out_tok_s": tps,
                      "lat_p50": s[len(s) // 2],
                      "lat_p95": s[int(len(s) * 0.95)]}
        print(f"{conc:>5} {len(prompts):>8} {wall:>7.1f}s {tps:>10.1f} "
              f"{s[len(s) // 2]:>8.2f}s {s[int(len(s) * 0.95)]:>8.2f}s",
              flush=True)

    out = cfg.out or f"nonregress_{cfg.label}.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"label": cfg.label, "input_len": cfg.input_len,
                   "output_len": cfg.output_len, "levels": rows}, fh, indent=2)
    print(f"\nwrote {out}")
    print("compare out tok/s across arms; the mechanism should be invisible "
          "here because no request carries a group id")


if __name__ == "__main__":
    main()
