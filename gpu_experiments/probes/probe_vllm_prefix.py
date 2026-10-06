#!/usr/bin/env python3
"""Can this vLLM recipe reuse a growing chat prefix?

Phase-5 arms reported 0.0% prefix hits on /v1/chat/completions while a
completions extend after the run was warm. API `cached_tokens` is null on
this recipe, so the decision is the /metrics delta on the *extend* call.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request


def post(url, body, timeout=180):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return time.time() - t0, data


def cached(data):
    usage = data.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return (details.get("cached_tokens")
            or usage.get("cached_tokens")
            or 0), usage.get("prompt_tokens")


def metrics(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics",
                                    timeout=5) as resp:
            text = resp.read().decode()
    except Exception as exc:
        return {"error": repr(exc)}
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or "prefix_cache" not in line:
            continue
        if line.startswith("vllm:prefix_cache_"):
            name = line.split("{", 1)[0].split(" ", 1)[0]
            try:
                out[name] = float(line.rsplit(" ", 1)[-1])
            except ValueError:
                pass
    return out


def hits(m):
    if not isinstance(m, dict):
        return 0.0
    return float(m.get("vllm:prefix_cache_hits_total") or 0.0)


def queries(m):
    if not isinstance(m, dict):
        return 0.0
    return float(m.get("vllm:prefix_cache_queries_total") or 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument("--model", default="moonshotai/Kimi-K3")
    ap.add_argument("--out", default="results/vllm_prefix_probe.json")
    a = ap.parse_args()
    chat = f"http://127.0.0.1:{a.port}/v1/chat/completions"
    comp = f"http://127.0.0.1:{a.port}/v1/completions"
    pad = ("alpha bravo charlie delta echo foxtrot golf hotel " * 80)
    user1 = f"Repeat the word ping. Context: {pad}"

    m0 = metrics(a.port)
    t1, d1 = post(chat, {
        "model": a.model, "messages": [{"role": "user", "content": user1}],
        "max_tokens": 8, "temperature": 0,
    })
    c1, p1 = cached(d1)
    m1 = metrics(a.port)
    assistant = ((d1.get("choices") or [{}])[0].get("message") or {}).get("content") or "ping"
    t2, d2 = post(chat, {
        "model": a.model,
        "messages": [
            {"role": "user", "content": user1},
            {"role": "assistant", "content": assistant},
            {"role": "user", "content": "Now say pong."},
        ],
        "max_tokens": 8, "temperature": 0,
    })
    c2, p2 = cached(d2)
    m2 = metrics(a.port)

    prompt = "The secret token is PREFIXPROBE42. " + pad
    t3, d3 = post(comp, {
        "model": a.model, "prompt": prompt, "max_tokens": 4, "temperature": 0,
    })
    c3, p3 = cached(d3)
    m3 = metrics(a.port)
    t4, d4 = post(comp, {
        "model": a.model, "prompt": prompt + " The next word is",
        "max_tokens": 4, "temperature": 0,
    })
    c4, p4 = cached(d4)
    m4 = metrics(a.port)

    chat_hits = hits(m2) - hits(m1)
    chat_q = queries(m2) - queries(m1)
    comp_hits = hits(m4) - hits(m3)
    comp_q = queries(m4) - queries(m3)
    # Extend is a hit if the second call added prefix-cache hits.
    # Require a real increment, not just queries moving.
    chat_ok = chat_hits > 0
    comp_ok = comp_hits > 0
    out = {
        "chat_turn1": {"s": round(t1, 3), "cached": c1, "prompt": p1},
        "chat_turn2_extend": {"s": round(t2, 3), "cached": c2, "prompt": p2},
        "comp_cold": {"s": round(t3, 3), "cached": c3, "prompt": p3},
        "comp_extend": {"s": round(t4, 3), "cached": c4, "prompt": p4},
        "metrics": {
            "before": m0, "after_chat1": m1, "after_chat2": m2,
            "after_comp1": m3, "after_comp2": m4,
        },
        "chat_extend_metrics": {"hits": chat_hits, "queries": chat_q},
        "comp_extend_metrics": {"hits": comp_hits, "queries": comp_q},
        "chat_extend_hits": chat_ok,
        "comp_extend_hits": comp_ok,
        "decision_instrument": "vllm:prefix_cache_hits_total",
    }
    pathlib = __import__("pathlib")
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    if chat_ok:
        print("decision=chat_ok")
    elif comp_ok:
        print("decision=chat_broken_comp_ok")
    else:
        print("decision=cache_dead")


if __name__ == "__main__":
    main()
