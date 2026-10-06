#!/usr/bin/env python3
"""为计划中的 e2e 工作点离线选配额并预测各臂，纯 CPU。

存在的理由：上机一组三臂约 35 分钟 GPU，而命中率只取决于请求序列与池容量，
这两样都可以从计划参数算出来。离线回放已在真实三臂上校准过（LRU 22.96%
对实测 28.43%、q=1 15.82% 对实测 18.53%、unified 与 LRU 同值对实测 +0.95pp，
方向与量级都对），所以先在这里定配额，只把定下来的配置拿去上机。

配额给的是基线**自己的最佳配置**而不是公式推出来的那个：同一条真实轨迹上
公式按峰值推出 q=1，比 LRU 差 7.1pp，而扫出来的 q=3 比 LRU 好 17.0pp。拿
q=1 去比会把基线做成靶子，而 unified 的声明是"不需要被告知参数"，所以它要
赢的必须是配置到最好的 Cohort。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_trace import BlockLRU, _Req, make_policy  # noqa: E402


def synth(teams, members, rounds, preamble, sys_tokens, gen_cap):
    """按 harness 的几何合成请求序列与 prompt 大小。

    每个团队同时只有一个请求在飞（轮转），所以全局序列是 T 个独立环的交错；
    团队按轮转推进，与实测 trace 的交错方式一致。成员的 prompt 是私有前置
    上下文加系统提示加整段共享记录，而记录每有任何成员发一轮就长一个封顶值。
    """
    refs = []
    turn = [0] * teams          # 每个团队已发生的引用数
    total = teams * members * rounds
    for i in range(total):
        t = i % teams
        n = turn[t]
        member = n % members
        size = preamble + sys_tokens + n * gen_cap
        refs.append((f"team{t}", member, n // members, size))
        turn[t] += 1
    return refs


def run(refs, pool, skip):
    cache = BlockLRU(pool)
    tot = hit = nskip = 0
    for team, member, turn, size in refs:
        sess = f"{team}:{member}"
        h, nblk = cache.lookup(sess, size)
        tot += size
        hit += h
        rid = f"{team}:{member}:{turn}"
        if skip is not None and skip(_Req(rid, size)):
            nskip += 1
        else:
            cache.insert(sess, nblk)
    return hit / tot, nskip


def ceiling(refs):
    """无限缓存下的命中上限：同会话上一轮的 prompt 即本轮可复用前缀。"""
    prev, reuse, tot = {}, 0, 0
    for team, member, _, size in refs:
        k = (team, member)
        tot += size
        if k in prev:
            reuse += min(prev[k], size)
        prev[k] = size
    return reuse / tot


def sweep(a, values):
    """负载曲线。单点会被问"是不是只在这一个点成立"，而曲线是 CPU 免费的。"""
    print(f"{'前置':>7}{'首轮过订':>10}{'末轮过订':>10}{'上限':>8}"
          f"{'LRU':>8}{'最佳固定':>10}{'unified':>9}{'差':>8}")
    for p in values:
        refs = synth(a.teams, a.members, a.rounds, p, a.sys_tokens, a.gen_cap)
        sess = a.teams * a.members
        first = p + a.sys_tokens
        last = first + (a.members * a.rounds - 1) * a.gen_cap
        base, _ = run(refs, a.pool, None)
        bq, bh = 0, -1.0
        for q in range(1, a.members):
            h, _ = run(refs, a.pool,
                       make_policy("group", a.pool, a.members,
                                   {"quota": q})[0])
            if h > bh:
                bq, bh = q, h
        uh, _ = run(refs, a.pool,
                    make_policy("unified", a.pool, a.members, {})[0])
        print(f"{p:>7}{sess*first/a.pool:>9.2f}x{sess*last/a.pool:>9.2f}x"
              f"{ceiling(refs)*100:>7.1f}%{base*100:>7.2f}%"
              f"{bh*100:>7.2f}% (q{bq}){uh*100:>8.2f}%"
              f"{(uh-bh)*100:>+7.2f}pp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teams", type=int, default=20)
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--preamble", type=int, default=9000)
    ap.add_argument("--sys-tokens", type=int, default=256)
    ap.add_argument("--gen-cap", type=int, default=144)
    ap.add_argument("--pool", type=int, default=420800)
    ap.add_argument("--sweep", default="",
                    help="逗号分隔的前置上下文长度，出负载曲线")
    a = ap.parse_args()

    if a.sweep:
        sweep(a, [int(x) for x in a.sweep.split(",")])
        return

    refs = synth(a.teams, a.members, a.rounds, a.preamble, a.sys_tokens,
                 a.gen_cap)
    sizes = [s for _, _, _, s in refs]
    first = a.preamble + a.sys_tokens
    last = first + (a.members * a.rounds - 1) * a.gen_cap
    sess = a.teams * a.members
    print(f"{len(refs)} 请求  {sess} 会话  池 {a.pool:,} token")
    print(f"上下文 首轮 {first:,} -> 末轮 {last:,}   "
          f"prompt 合计 {sum(sizes)/1e6:.2f}M")
    print(f"过订 首轮 {sess*first/a.pool:.2f}x -> 末轮 {sess*last/a.pool:.2f}x"
          f"   装得下 {a.pool//first}->{a.pool//last} 个会话")
    # 无限缓存上限：同会话上一轮的 prompt 即本轮可复用前缀
    prev, reuse = {}, 0
    for team, member, _, size in refs:
        k = (team, member)
        if k in prev:
            reuse += min(prev[k], size)
        prev[k] = size
    print(f"无限缓存上限 {reuse/sum(sizes)*100:.1f}%\n")

    base, _ = run(refs, a.pool, None)
    print(f"{'策略':<22}{'命中率':>9}{'相对 LRU':>11}{'绕过':>8}")
    print(f"{'upstream LRU':<22}{base*100:>8.2f}%{0.0:>+10.2f}pp{0:>8}")
    best = (None, -1)
    for q in range(1, a.members):
        h, ns = run(refs, a.pool,
                    make_policy("group", a.pool, a.members, {"quota": q})[0])
        print(f"{'Cohort q=' + str(q):<22}{h*100:>8.2f}%"
              f"{(h-base)*100:>+10.2f}pp{ns:>8}")
        if h > best[1]:
            best = (q, h)
    h, ns = run(refs, a.pool,
                make_policy("unified", a.pool, a.members, {})[0])
    print(f"{'unified':<22}{h*100:>8.2f}%{(h-base)*100:>+10.2f}pp{ns:>8}")

    print(f"\n基线最强配置 QUOTA={best[0]}（命中 {best[1]*100:.2f}%）")
    print(f"unified 相对它 {(h-best[1])*100:+.2f}pp")


if __name__ == "__main__":
    main()
