#!/usr/bin/env python3
"""从 e2e trace 出全程 / 热启动 / 稳态三套数。

三窗口不是三种实验，是同一条轨迹切不同前缀。生产节点的池子始终是满的，
第一轮就带着工具定义和私有仓库切片，不存在"从空上下文爬到过订"这一段。
主报稳态（过订且规则已介入）；热启动是过订后、规则尚未定价的过渡；
全程只衡量规则在装得下时会不会让路，不进摘要。
"""

from __future__ import annotations

import argparse
import json
import re


POOL_DEFAULT = 420800


def turn_of(r):
    rid = r.get("rid") or ""
    if rid.count(":") >= 2:
        try:
            return int(rid.split(":")[2])
        except ValueError:
            return None
    return None


def sess_of(r):
    rid = r.get("rid") or ""
    if rid.count(":") >= 2:
        a, b, _ = rid.split(":", 2)
        return f"{a}:{b}"
    return r.get("team")


def ttta(rows):
    by = {}
    for r in rows:
        by.setdefault(r.get("team"), []).append(r)
    spans = []
    for v in by.values():
        if not v:
            continue
        spans.append(max(x["t_done"] for x in v) - min(x["t_recv"] for x in v))
    spans.sort()
    if not spans:
        return None, None, None
    p95 = spans[min(max(int(round(0.95 * len(spans))) - 1, 0), len(spans) - 1)]
    return spans[len(spans) // 2], p95, spans[-1]


def ceiling(rows):
    """无限缓存上限：同会话上一轮 prompt 即本轮可复用前缀。"""
    prev, reuse, tot = {}, 0, 0
    for r in sorted(rows, key=lambda x: (sess_of(x) or "", turn_of(x) or 0)):
        tot += r.get("prompt_tokens") or 0
        k = sess_of(r)
        if k in prev:
            reuse += min(prev[k], r.get("prompt_tokens") or 0)
        prev[k] = r.get("prompt_tokens") or 0
    return (reuse / tot if tot else 0.0), reuse, tot


def pack(rows):
    cached = sum(r.get("cached_tokens") or 0 for r in rows)
    prompt = sum(r.get("prompt_tokens") or 0 for r in rows)
    p50, p95, mx = ttta(rows)
    ceil, creuse, ctot = ceiling(rows)
    stamps = [(r["t_recv"], r["t_done"]) for r in rows
              if r.get("t_recv") and r.get("t_done")]
    wall = (max(e for _, e in stamps) - min(s for s, _ in stamps)
            if stamps else None)
    return {
        "requests": len(rows),
        "prefix_hit_rate": cached / prompt if prompt else 0.0,
        "prompt_tokens_total": prompt,
        "cached_tokens_total": cached,
        "infinite_hit_rate": ceil,
        "infinite_tokens": creuse,
        "headroom_pp": (ceil - (cached / prompt if prompt else 0.0)) * 100,
        "ttta_p50": p50,
        "ttta_p95": p95,
        "ttta_max": mx,
        "span_s": wall,
    }


def per_turn(ok, n_sess, pool):
    by = {}
    for r in ok:
        t = turn_of(r)
        if t is None:
            continue
        by.setdefault(t, []).append(r)
    out, contended_from = [], None
    for t in sorted(by):
        g = by[t]
        prompts = sorted(r.get("prompt_tokens") or 0 for r in g)
        med = prompts[len(prompts) // 2] if prompts else 0
        live = med * n_sess
        fits = live <= pool
        if contended_from is None and not fits:
            contended_from = t
        p = pack(g)
        p.update({"turn": t, "median_prompt": med, "fleet_tokens": live,
                  "fits": fits})
        out.append(p)
    return out, contended_from


def summarize(rows, arm, teams, members, rounds, wall=None, pool=POOL_DEFAULT):
    ok = [r for r in rows if r.get("status") == 200]
    n_sess = len({sess_of(r) for r in ok}) or (teams * members)
    turns, contended_from = per_turn(ok, n_sess, pool)
    if contended_from is None:
        contended_from = max((turn_of(r) or 0) for r in ok) + 1 if ok else 0
    # 规则要两次过订样本才定价；过订后第 2 轮起才是它已经介入的稳态。
    engaged_from = contended_from + 2

    def after(t0):
        return [r for r in ok if (turn_of(r) or 0) >= t0]

    full = pack(ok)
    hot = pack(after(contended_from))
    engaged = pack(after(engaged_from))
    res = {
        "arm": arm, "teams": teams, "members": members, "rounds": rounds,
        "pool_tokens": pool, "sessions": n_sess,
        "wall_s": wall if wall is not None else full["span_s"],
        "failed_requests": len(rows) - len(ok),
        "teams_completed": len({r.get("team") for r in ok}),
        "contended_from_turn": contended_from,
        "engaged_from_turn": engaged_from,
        "full": full,
        "hot": hot,
        "steady": engaged,
        "by_turn": turns,
        # 兼容旧字段 = 全程
        "requests": full["requests"],
        "prefix_hit_rate": full["prefix_hit_rate"],
        "prompt_tokens_total": full["prompt_tokens_total"],
        "cached_tokens_total": full["cached_tokens_total"],
        "ttta_p50": full["ttta_p50"],
        "ttta_p95": full["ttta_p95"],
        "ttta_max": full["ttta_max"],
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--teams", type=int, required=True)
    ap.add_argument("--members", type=int, required=True)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--wall", type=int)
    ap.add_argument("--pool", type=int, default=POOL_DEFAULT)
    ap.add_argument("--out")
    ap.add_argument("--rid-re", default="",
                    help="keep only rows whose rid matches this regex "
                         "(Selector e2e: '^team[0-9]+:' drops selector calls)")
    cfg = ap.parse_args()

    rows = [json.loads(l) for l in open(cfg.trace, encoding="utf-8")
            if l.strip()]
    if cfg.rid_re:
        pat = re.compile(cfg.rid_re)
        rows = [r for r in rows if pat.match(r.get("rid") or "")]
    res = summarize(rows, cfg.arm, cfg.teams, cfg.members, cfg.rounds,
                    wall=cfg.wall, pool=cfg.pool)
    if cfg.wall is None:
        res["wall_source"] = ("recovered from trace span "
                              "(excludes harness startup/teardown)")
    print(json.dumps(res, indent=2))
    if cfg.out:
        json.dump(res, open(cfg.out, "w", encoding="utf-8"), indent=2)
        print(f"wrote {cfg.out}")


if __name__ == "__main__":
    main()
