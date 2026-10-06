#!/usr/bin/env python3
"""全程回放，再按窗口计分。

热启动不是把后半段单独从头回放——那样规则会重新测量、缓存也是空的。
生产稳态是：同一条连续运行里，丢掉装得下的前缀，只计过订之后的命中。
"""
import sys
from collections import defaultdict
from replay_trace import load, BlockLRU, _Req, make_policy


def run(rows, mode, extra=None, pool=420800, members=5):
    cache = BlockLRU(pool)
    if mode == "upstream":
        skip = None
    else:
        skip, _ = make_policy(mode, pool, members, extra)
    rec = []
    for r in rows:
        sess = f"{r['team']}:{r['member']}"
        h, nblk = cache.lookup(sess, r["prompt"])
        skipped = False
        if skip is None:
            cache.insert(sess, nblk)
        else:
            skipped = skip(_Req(r["rid"], r["prompt"]))
            if not skipped:
                cache.insert(sess, nblk)
        rec.append((r, h, skipped))
    return rec


def score(rec, t0):
    hit = tot = 0
    for r, h, _ in rec:
        if r["turn"] >= t0:
            hit += h
            tot += r["prompt"]
    return hit / tot if tot else 0.0


def ceiling(rows, t0):
    prev, reuse, tot = {}, 0, 0
    for r in sorted(rows, key=lambda x: (x["team"], x["member"], x["turn"])):
        if r["turn"] < t0:
            prev[(r["team"], r["member"])] = r["prompt"]
            continue
        k = (r["team"], r["member"])
        tot += r["prompt"]
        if k in prev:
            reuse += min(prev[k], r["prompt"])
        prev[k] = r["prompt"]
    return reuse / tot if tot else 0.0


def main():
    rows = load(sys.argv[1])
    by = defaultdict(list)
    for r in rows:
        by[r["turn"]].append(r["prompt"])
    n_sess = len({(r["team"], r["member"]) for r in rows})
    contended = None
    for t in sorted(by):
        med = sorted(by[t])[len(by[t]) // 2]
        if med * n_sess > 420800:
            contended = t
            break
    if contended is None:
        contended = 0
    engaged = contended + 2
    cuts = [("全程", 0), ("热启动", contended), ("稳态", engaged)]
    arms = [
        ("LRU", "upstream", None),
        ("q=2", "group", {"quota": 2}),
        ("q=3", "group", {"quota": 3}),
        ("unified", "unified", {}),
    ]
    print(f"contended_from={contended}  steady_from={engaged}  "
          f"n={len(rows)}")
    print(f"{'窗口':<8}", end="")
    for n, _, _ in arms:
        print(f"{n:>10}", end="")
    print(f"{'上限':>10}")
    scored = {}
    for name, mode, extra in arms:
        scored[name] = run(rows, mode, extra)
    for tag, t0 in cuts:
        print(f"{tag:<8}", end="")
        for name, _, _ in arms:
            print(f"{score(scored[name], t0)*100:>9.2f}%", end="")
        print(f"{ceiling(rows, t0)*100:>9.1f}%")


if __name__ == "__main__":
    main()
