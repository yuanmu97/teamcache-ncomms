#!/usr/bin/env python3
"""同一条真机轨迹上的缓存上界：无限缓存、有限池 OPTb、公平 OPTb。

对比原则（2026-09-15）：能估 optimal 上界的都估，不要只对 LRU。
主参照是同池 OPTb（策略界），公平 OPTb 是不饿死界，窗前仍在的无限缓存是容量界。

策略在全程上跑（缓存有记忆），数字按稳态窗（turn>=4）报。
墙钟外推来自 LRU / unified 两臂实测的漏算 token 线性拟合，不是新上机。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

POOL_DEFAULT = 420800


def load(path):
    rows = []
    for line in open(path):
        d = json.loads(line)
        if d.get("status") != 200 or (d.get("rid") or "").count(":") < 2:
            continue
        team, member, turn = d["rid"].split(":", 2)
        rows.append({
            "team": team, "member": int(member), "turn": int(turn),
            "prompt": d["prompt_tokens"],
            "cached": d.get("cached_tokens") or 0,
            "t": d["t_recv"],
        })
    rows.sort(key=lambda r: r["t"])
    return rows


def sess(r):
    return f"{r['team']}:{r['member']}"


def score_window(rows, cached, t0=4):
    prompt = hit = 0
    for r, c in zip(rows, cached):
        if r["turn"] < t0:
            continue
        prompt += r["prompt"]
        hit += min(c, r["prompt"])
    return hit / prompt if prompt else 0.0, hit, prompt


def infinite_warm(rows):
    """窗前一轮也算在：生产节点的无限缓存。"""
    prev, out = {}, []
    for r in rows:
        k = sess(r)
        out.append(min(prev[k], r["prompt"]) if k in prev else 0)
        prev[k] = r["prompt"]
    return out


def infinite_window_only(rows, t0=4):
    """只在窗内找上一轮。第 4 轮被强制记成冷，旧稿 71% 那一列是这个。"""
    prev, out = {}, []
    for r in rows:
        k = sess(r)
        if r["turn"] < t0:
            out.append(0)
            continue
        out.append(min(prev[k], r["prompt"]) if k in prev else 0)
        prev[k] = r["prompt"]
    return out


def optb_tokens(rows, budget):
    """变长条目的 McFarling OPTb：furthest next-use 应用到 cache ∪ {当前}。"""
    keys = [sess(r) for r in rows]
    n = len(keys)
    nxt, last = [n] * n, {}
    for i in range(n - 1, -1, -1):
        nxt[i] = last.get(keys[i], n)
        last[keys[i]] = i
    cache, out = {}, []
    for i, r in enumerate(rows):
        k, p = keys[i], r["prompt"]
        if k in cache:
            out.append(min(cache[k][0], p))
            cache[k] = (p, nxt[i])
        else:
            out.append(0)
            cache[k] = (p, nxt[i])
        while sum(s for s, _ in cache.values()) > budget:
            worst = max(cache, key=lambda x: cache[x][1])
            del cache[worst]
            if worst == k:
                break
    return out


def fair_optb(rows, budget, n_teams=20):
    """每队均分 token，队内再 OPTb。"""
    share = budget / n_teams
    keys = [(r["team"], r["member"]) for r in rows]
    n = len(keys)
    nxt, last = [n] * n, {}
    for i in range(n - 1, -1, -1):
        nxt[i] = last.get(keys[i], n)
        last[keys[i]] = i
    cache, out = {}, []
    for i, r in enumerate(rows):
        k, p = keys[i], r["prompt"]
        if k in cache:
            out.append(min(cache[k][0], p))
        else:
            out.append(0)
        cache[k] = (p, nxt[i])
        team = r["team"]
        while True:
            mine = {x: v for x, v in cache.items() if x[0] == team}
            if sum(s for s, _ in mine.values()) <= share:
                break
            del cache[max(mine, key=lambda x: mine[x][1])]
    return out


def bypass_sized(rows, budget):
    """满了就不再收留。均匀环上应贴近 OPTb。"""
    cache, prev, out = {}, {}, []
    for r in rows:
        k, p = sess(r), r["prompt"]
        if k in cache:
            out.append(min(prev.get(k, 0), p))
            cache[k] = p
        else:
            out.append(0)
            if sum(cache.values()) + p <= budget:
                cache[k] = p
        prev[k] = p
        while sum(cache.values()) > budget and cache:
            victims = [x for x in cache if x != k]
            if not victims:
                if sum(cache.values()) > budget:
                    del cache[k]
                break
            del cache[max(victims, key=lambda x: cache[x])]
    return out


def live_cached(rows):
    return [r["cached"] for r in rows]


def team_hits(rows, cached, t0=4):
    by = {}
    for r, c in zip(rows, cached):
        if r["turn"] < t0:
            continue
        by.setdefault(r["team"], [0, 0])
        by[r["team"]][0] += min(c, r["prompt"])
        by[r["team"]][1] += r["prompt"]
    rates = sorted(h / p for h, p in by.values() if p)
    return rates[0], rates[len(rates) // 2], rates[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lru", required=True)
    ap.add_argument("--unified", required=True)
    ap.add_argument("--pool", type=int, default=POOL_DEFAULT)
    ap.add_argument("--steady-from", type=int, default=4)
    args = ap.parse_args()

    lru = load(args.lru)
    uni = load(args.unified)
    t0 = args.steady_from

    h_l, c_l, p_l = score_window(lru, live_cached(lru), t0)
    h_u, c_u, p_u = score_window(uni, live_cached(uni), t0)
    # 该窗墙钟来自 score_e2e 已落盘的数；这里只用漏算 token 拟合外推。
    miss_l, miss_u = p_l - c_l, p_u - c_u
    wall_l, wall_u = 184.8998, 151.2215
    slope = (wall_l - wall_u) / (miss_l - miss_u)
    intercept = wall_u - slope * miss_u

    def wall(miss):
        return intercept + slope * miss

    print(f"池 {args.pool:,}  稳态 turn>={t0}")
    print(f"墙钟拟合  {intercept:.1f} + {slope * 1e6:.2f} µs/漏算token  "
          f"(由 LRU {wall_l:.1f}s / unified {wall_u:.1f}s)")
    print(f"\n{'策略':<22} {'命中':>7} {'漏算 token':>12} {'墙钟':>8} "
          f"{'对 LRU':>8} {'对 OPTb':>8}  注")

    optb = optb_tokens(lru, args.pool)
    h_opt, _, _ = score_window(lru, optb, t0)

    rows_out = []
    for name, src, hits, measured_wall, note in [
        ("LRU 实测", lru, live_cached(lru), wall_l, "真机"),
        ("unified 实测", uni, live_cached(uni), wall_u, "真机"),
        ("有限池 OPTb", lru, optb, None, "同池、知情、聚合界"),
        ("满了就不再收留", lru, bypass_sized(lru, args.pool), None,
         "环上应贴近 OPTb"),
        ("公平 OPTb（均分）", lru, fair_optb(lru, args.pool), None,
         "同池、知情、不饿死界"),
        ("无限缓存（窗前仍在）", lru, infinite_warm(lru), None,
         "容量界，生产口径"),
        ("无限缓存（窗内自起）", lru, infinite_window_only(lru, t0), None,
         "第 4 轮强制冷，旧 71% 列"),
    ]:
        h, c, p = score_window(src, hits, t0)
        miss = p - c
        w = measured_wall if measured_wall is not None else wall(miss)
        tag = "" if measured_wall is None else "  [实测]"
        print(f"{name:<22} {h * 100:6.2f}% {miss:12,.0f} {w:7.1f}s "
              f"{wall_l / w:7.2f}x {h / h_opt * 100:7.1f}%  {note}{tag}")
        rows_out.append({"name": name, "hit": h, "miss": miss, "wall": w})

    lo, mid, hi = team_hits(lru, optb, t0)
    flo, fmid, fhi = team_hits(lru, fair_optb(lru, args.pool), t0)
    print(f"\nOPTb 队级命中  min {lo * 100:.1f}%  p50 {mid * 100:.1f}%  "
          f"max {hi * 100:.1f}%")
    print(f"公平 OPTb 队级  min {flo * 100:.1f}%  p50 {fmid * 100:.1f}%  "
          f"max {fhi * 100:.1f}%")


if __name__ == "__main__":
    main()
