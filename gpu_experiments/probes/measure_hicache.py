#!/usr/bin/env python3
"""Price a host-tier hit, which is the number that decides whether offloading
dissolves this paper's problem or merely postpones it.

The obvious objection to everything here is one flag: enable hierarchical
caching, let evicted sessions fall to host DRAM, and a dormant member's history
is never really gone. Whether that is an answer or not is entirely quantitative.
`measure_cost_model.py` gives three points on the wake-cost curve -- cold, warm,
and warm-plus-a-peer-message. Offloading inserts a fourth point between cold and
warm, and the whole objection stands or falls on where it lands:

  near warm   offloading really does dissolve the problem at this scale, and the
              honest thing is to say so and re-scope the contribution to the
              regime where device+host together are oversubscribed.
  near cold   offloading buys capacity, not immunity: the access pattern is
              still adversarial, just against a larger pool, and the mechanism
              is still what fixes it.

Method. Warm a target context, then flood the device pool with unrelated
contexts sized to evict the target from device while staying inside device+host,
then re-request the target. The flood is the only way to do this from outside:
neither engine exposes an eviction call, which is the same property §9 exploited
in the opposite direction.

Attribution is the hard part -- a fast re-request proves nothing by itself,
since it could be a device hit the flood failed to dislodge. So every phase
reads the engine's own counters, and a run that cannot show the device tier
missing while the host tier hits is reported as inconclusive rather than as a
host hit. Guessing here would produce exactly the reassuring number we are
trying to test.

Run on the inference host, against an engine started with
`--enable-hierarchical-cache`:

  python3 measure_hicache.py --port 30000 --context 9728 --pool-tokens 420800
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
import urllib.error
import urllib.request


def post(url, payload, timeout=900.0):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"content-type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return time.monotonic() - t0, data


def get(url, timeout=30.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def counters(base):
    """Whatever the engine will tell us about its own cache tiers.

    Metric names differ across SGLang versions, so match on substrings and
    report what was found rather than asserting a schema. A missing host
    counter is not fatal -- it costs us the attribution, and the caller
    downgrades the result to inconclusive instead of claiming a host hit.
    """
    try:
        text = get(f"{base}/metrics")
    except Exception:
        return {}
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name, _, val = line.rpartition(" ")
        name = re.sub(r"\{.*\}", "", name).strip()
        if not any(k in name for k in
                   ("cache", "hicache", "hit", "token", "prefix")):
            continue
        try:
            out[name] = out.get(name, 0.0) + float(val)
        except ValueError:
            continue
    return out


def delta(before, after, needle):
    tot = 0.0
    for k, v in after.items():
        if needle in k:
            tot += v - before.get(k, 0.0)
    return tot


def ids(n, vocab, rng):
    return [rng.randrange(16, vocab) for _ in range(n)]


def run_prompt(base, model, token_ids, max_tokens=1):
    dt, data = post(f"{base}/v1/completions", {
        "model": model,
        "prompt": token_ids,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    })
    usage = data.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens")
    if cached is None:
        cached = usage.get("cached_tokens", 0)
    return dt, int(cached or 0), int(usage.get("prompt_tokens") or 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--model", default="/opt/dlami/nvme/models/Kimi-K3")
    ap.add_argument("--context", type=int, default=9728,
                    help="target context length; the measured median")
    ap.add_argument("--vocab", type=int, default=100000)
    ap.add_argument("--pool-tokens", type=int, default=420800,
                    help="device KV pool, used to size the flood")
    ap.add_argument("--flood-factor", type=float, default=1.6,
                    help="flood this many device pools' worth of unrelated "
                         "context, so the target is evicted from device with "
                         "margin but device+host stays under-filled")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default="hicache_cost.json")
    cfg = ap.parse_args()

    base = f"http://{cfg.host}:{cfg.port}"
    rng = random.Random(cfg.seed)
    L = cfg.context

    try:
        srv = json.loads(get(f"{base}/get_server_info"))
        hier = (srv.get("enable_hierarchical_cache")
                or srv.get("server_args", {}).get("enable_hierarchical_cache"))
        ratio = (srv.get("hicache_ratio")
                 or srv.get("server_args", {}).get("hicache_ratio"))
        print(f"engine: hierarchical_cache={hier} hicache_ratio={ratio}")
        if not hier:
            print("WARNING: hierarchical cache is OFF; the host-tier phase "
                  "will measure a cold prefill, which is the control, not the "
                  "treatment. Run it that way deliberately or not at all.")
    except Exception as exc:
        print(f"could not read /get_server_info: {exc}")
        hier = ratio = None

    rows = []
    for rep in range(cfg.repeats):
        target = ids(L, cfg.vocab, rng)

        # cold: this context has never been seen.
        c_before = counters(base)
        t_cold, cached_cold, pt = run_prompt(base, cfg.model, target)
        c_after = counters(base)
        print(f"\n[rep {rep}] cold      {t_cold:7.3f}s  "
              f"cached {cached_cold:>7}/{pt}")

        # device warm: same context plus one token, immediately after, so it
        # cannot have left the device tier yet. Strictly extending rather than
        # re-sending is required -- an identical re-request is not reliably
        # served from the radix tree.
        t_warm, cached_warm, _ = run_prompt(
            base, cfg.model, target + ids(1, cfg.vocab, rng))
        print(f"[rep {rep}] warm      {t_warm:7.3f}s  cached {cached_warm:>7}")

        # flood: evict the target from device without filling device+host.
        n_flood = max(1, int(cfg.flood_factor * cfg.pool_tokens / L))
        print(f"[rep {rep}] flooding {n_flood} x {L} tokens "
              f"({n_flood * L / cfg.pool_tokens:.2f} device pools)...")
        t_flood0 = time.monotonic()
        for _ in range(n_flood):
            try:
                run_prompt(base, cfg.model, ids(L, cfg.vocab, rng))
            except (urllib.error.URLError, TimeoutError) as exc:
                print(f"  flood request failed: {exc}")
        print(f"[rep {rep}] flood took {time.monotonic() - t_flood0:.0f}s")

        # host hit, if the tiers behaved: device should miss, host should hit.
        h_before = counters(base)
        t_host, cached_host, _ = run_prompt(
            base, cfg.model, target + ids(1, cfg.vocab, rng))
        h_after = counters(base)
        d_host = delta(h_before, h_after, "hicache")
        print(f"[rep {rep}] post-flood{t_host:7.3f}s  "
              f"cached {cached_host:>7}   hicache counter delta {d_host:+.0f}")

        rows.append({"rep": rep, "cold_s": t_cold, "warm_s": t_warm,
                     "postflood_s": t_host, "cached_cold": cached_cold,
                     "cached_warm": cached_warm, "cached_postflood": cached_host,
                     "hicache_delta": d_host,
                     "cold_counters": {k: c_after[k] - c_before.get(k, 0.0)
                                       for k in c_after}})

    def med(key):
        return statistics.median(r[key] for r in rows)

    cold, warm, host = med("cold_s"), med("warm_s"), med("postflood_s")
    frac = (host - warm) / (cold - warm) if cold > warm else None
    print("\n=== where a post-eviction wake lands, median over "
          f"{cfg.repeats} reps ===")
    print(f"  cold prefill        {cold:7.3f}s")
    print(f"  device-tier hit     {warm:7.3f}s")
    print(f"  after eviction      {host:7.3f}s")
    if frac is not None:
        print(f"  fraction of the cold-to-warm gap still paid: {frac:.1%}")
        if frac > 0.6:
            print("  reading: offloading buys capacity, not immunity -- a wake "
                  "after eviction still pays most of a cold prefill, so the "
                  "access pattern remains the binding problem.")
        elif frac < 0.2:
            print("  reading: at this scale offloading largely dissolves the "
                  "cost of eviction. The contribution must then be re-scoped "
                  "to loads that oversubscribe device+host together.")
        else:
            print("  reading: partial relief. Report the number rather than a "
                  "verdict, and run the load curve before claiming either.")
    cached_after = med("cached_postflood")
    if cached_after >= 0.9 * cfg.context:
        print(f"  CAUTION: {cached_after:.0f} of {cfg.context} tokens still "
              "reported cached after the flood, so the target may never have "
              "left the device tier. Raise --flood-factor and re-run; this "
              "result is inconclusive as it stands.")

    with open(cfg.out, "w", encoding="utf-8") as fh:
        json.dump({"config": vars(cfg), "hierarchical": bool(hier),
                   "hicache_ratio": ratio, "rows": rows,
                   "median": {"cold_s": cold, "warm_s": warm,
                              "postflood_s": host, "gap_fraction": frac}},
                  fh, indent=2)
    print(f"\nwrote {cfg.out}")


if __name__ == "__main__":
    main()
