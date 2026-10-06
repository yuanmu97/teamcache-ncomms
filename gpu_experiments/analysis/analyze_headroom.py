#!/usr/bin/env python3
"""从 e2e trace 推导缓存 headroom 与它能兑现的墙钟上界。

不需要 GPU。回答三个问题：
  1. 上下文在一次运行里怎么长，据此哪几轮本来就装得下（LRU 白拿）
  2. 无限缓存的命中上限是多少，LRU 距它多远
  3. prefill 占端到端时间的份额，从而任何命中率改进的墙钟上界
"""
import json
import sys
from collections import defaultdict


def load(path):
    rows = []
    for line in open(path):
        d = json.loads(line)
        if d.get("status") != 200 or "rid" not in d:
            continue
        team, member, turn = d["rid"].split(":")
        d["_turn"] = int(turn)
        d["_sess"] = f"{team}:{member}"
        d["_lat"] = d["t_done"] - d["t_recv"]
        d["cached_tokens"] = d.get("cached_tokens") or 0
        rows.append(d)
    rows.sort(key=lambda r: r["t_recv"])
    return rows


def main(paths, pool=420800):
    for path in paths:
        rows = load(path)
        arm = path.split("-")[-2]
        n_sess = len({r["_sess"] for r in rows})
        print(f"\n{'='*72}\n{arm}  {len(rows)} 请求  {n_sess} 会话  池 {pool:,} token\n{'='*72}")

        # ---- 1. 上下文增长与"白拿"的轮次 ----
        by_turn = defaultdict(list)
        for r in rows:
            by_turn[r["_turn"]].append(r)
        print(f"{'轮':>3} {'中位 prompt':>11} {'全体驻留':>11} {'装得下?':>7} "
              f"{'命中率':>7} {'中位延迟':>9}")
        fit_rounds = []
        for t in sorted(by_turn):
            g = by_turn[t]
            med = sorted(r["prompt_tokens"] for r in g)[len(g) // 2]
            live = med * n_sess
            fits = live <= pool
            fit_rounds.append(fits)
            hit = sum(r["cached_tokens"] for r in g) / max(1, sum(r["prompt_tokens"] for r in g))
            lat = sorted(r["_lat"] for r in g)[len(g) // 2]
            print(f"{t:>3} {med:>11,} {live:>11,} {'是' if fits else '否':>7} "
                  f"{hit*100:>6.1f}% {lat:>8.2f}s")

        # ---- 2. 无限缓存上限 ----
        # 同一会话上一轮的 prompt 就是本轮可复用的前缀（严格延长）
        prev = {}
        reusable = total = 0
        for r in sorted(rows, key=lambda r: (r["_sess"], r["_turn"])):
            total += r["prompt_tokens"]
            if r["_sess"] in prev:
                reusable += min(prev[r["_sess"]], r["prompt_tokens"])
            prev[r["_sess"]] = r["prompt_tokens"]
        actual = sum(r["cached_tokens"] for r in rows)
        print(f"\n无限缓存上限  {reusable/total*100:5.1f}%  ({reusable:,} / {total:,})")
        print(f"实测命中      {actual/total*100:5.1f}%  ({actual:,})")
        print(f"剩余 headroom {(reusable-actual)/total*100:5.1f}pp"
              f"  = {(reusable-actual)/1e6:.2f}M token 的 prefill")

        # ---- 3. prefill 时间份额：对未命中 prompt 与解码长度做回归 ----
        # lat = a + b*uncached + c*completion
        import statistics
        xs = [(r["prompt_tokens"] - r["cached_tokens"], r["completion_tokens"], r["_lat"])
              for r in rows]
        n = len(xs)
        # 正规方程（3 参数最小二乘，手写避免依赖 numpy）
        S = [[0.0] * 4 for _ in range(3)]
        for u, c, y in xs:
            v = [1.0, u, c]
            for i in range(3):
                for j in range(3):
                    S[i][j] += v[i] * v[j]
                S[i][3] += v[i] * y
        for i in range(3):
            p = max(range(i, 3), key=lambda k: abs(S[k][i]))
            S[i], S[p] = S[p], S[i]
            for k in range(i + 1, 3):
                f = S[k][i] / S[i][i]
                for j in range(i, 4):
                    S[k][j] -= f * S[i][j]
        coef = [0.0] * 3
        for i in (2, 1, 0):
            coef[i] = (S[i][3] - sum(S[i][j] * coef[j] for j in range(i + 1, 3))) / S[i][i]
        a, b, c = coef
        tot_u = sum(x[0] for x in xs)
        tot_c = sum(x[1] for x in xs)
        t_pre, t_dec, t_fix = b * tot_u, c * tot_c, a * n
        t_all = t_pre + t_dec + t_fix
        print(f"\n回归  固定 {a*1000:.0f} ms/请求   prefill {b*1e6:.0f} µs/token"
              f"   decode {c*1000:.1f} ms/token")
        print(f"      每 token 上 decode 比 prefill 贵 {c/b:,.0f} 倍" if b > 0 else "")
        print(f"时间构成（请求内，非墙钟）  prefill {t_pre/t_all*100:4.1f}%"
              f"   decode {t_dec/t_all*100:4.1f}%   固定 {t_fix/t_all*100:4.1f}%")
        if b > 0:
            saved = (reusable - actual) * b
            print(f"\n把命中打到上限最多省 {saved:.0f}s 的 prefill"
                  f"；这一臂墙钟 {max(r['t_done'] for r in rows)-min(r['t_recv'] for r in rows):.0f}s"
                  f"  →  上界 {saved/(max(r['t_done'] for r in rows)-min(r['t_recv'] for r in rows))*100:.1f}%")
        print(f"\n注：轮次装得下的比例 {sum(fit_rounds)}/{len(fit_rounds)}，"
              f"这些轮 LRU 无需任何机制即可命中")


if __name__ == "__main__":
    main(sys.argv[1:])
