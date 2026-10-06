#!/usr/bin/env python3
"""在 126 格网格上比较 concentrate 规则的候选修法，纯 CPU。

为什么需要它：真实 AutoGen 轨迹上量出的最佳修法（按团队老化 + 打平稳定 +
排除当前引用 + 短 p_hit 窗）里，有两项是网格从未检验过的。而网格里的
`empirical` 就是拟合的 Claude Code 发言序，论文最要紧的泛化声明——在那上面
必须与 LRU 持平——只能在这里验。所以修法先过网格，再落 patch。

老化周期以**轮**为单位，这样 k=5 的真实轨迹与 k=7 的网格可比。
"""
import argparse
import statistics
from collections import OrderedDict, defaultdict

import optimal as O


def iso_last(seq):
    """非增等序回归（PAVA）的末段均值。

    `held` 在真实运行里近似单调非增：池压力随驻留会话数与上下文单调增，
    而会话内上下文只增不减。单调阶跃时它给出当前那一级；独立混合时相邻块
    会被不断合并，自动退回窗口均值。没有新参数。
    """
    blocks = []
    for x in seq:
        blocks.append([float(x), 1])
        while (len(blocks) > 1
               and blocks[-2][0] / blocks[-2][1] < blocks[-1][0] / blocks[-1][1]):
            s, c = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += c
    return blocks[-1][0] / blocks[-1][1] if blocks else 1.0


def run_concentrate_v(refs, capacity, members, window=200, min_samples=2,
                      margin=1.0, bypass=True, spread_bypass=True,
                      age_rounds=0.0, excl=False, stable=False,
                      iso=False):
    """与 optimal.run_concentrate 同构，多出三个候选开关。

    age_rounds=0 关老化、excl=False、stable=False 时应与原版逐格相同。
    """
    teams = max(len({t for t, _ in refs}), 1)
    share = max(int(capacity / teams), 1)
    period = int(age_rounds * members) if age_rounds else 0

    cache = OrderedDict()
    pinned, protect = set(), defaultdict(set)
    dist = defaultdict(list)
    seen = defaultdict(float)
    age_t = defaultdict(int)
    last, clock, flags = {}, 0, []
    icache = {}

    def p_iso(key):
        d = dist[key]
        c = icache.get(key)
        if c is not None and c[0] == len(d):
            return c[1]
        v = iso_last([1 if x < capacity else 0 for x in d])
        icache[key] = (len(d), v)
        return v

    def plan(team, cur):
        ms = [m for m in range(members) if len(dist[(team, m)]) >= min_samples]
        if len(ms) < members:
            return None, {}
        rate = {m: seen[(team, m)] - (1.0 if (excl and m == cur) else 0.0)
                for m in ms}
        mass = sum(rate.values()) or 1
        if iso:
            p_hit = {m: p_iso((team, m)) for m in ms}
        else:
            p_hit = {m: sum(1 for x in dist[(team, m)] if x < capacity)
                     / len(dist[(team, m)]) for m in ms}
        key = ((lambda m: (-rate[m], m)) if stable else (lambda m: -rate[m]))
        top = sorted(ms, key=key)[:share]
        concentrated = sum(rate[m] for m in top) / mass
        spread = sum(rate[m] * p_hit[m] for m in ms) / mass
        want = set(top) if concentrated > spread * margin else set()
        return want, p_hit

    for team, member in refs:
        key = (team, member)
        hit = key in cache
        flags.append(not hit)
        if hit:
            cache.move_to_end(key)

        if key in last:
            since = last[key]
            d = dist[key]
            d.append(sum(1 for v in last.values() if v > since))
            if len(d) > window:
                del d[:-window]
        last[key] = clock
        clock += 1
        seen[key] += 1
        if period:
            age_t[team] += 1
            if age_t[team] >= period:
                age_t[team] = 0
                for m in range(members):
                    seen[(team, m)] *= 0.5

        if hit:
            continue

        want, p_hit = plan(team, member)
        if want is None:
            admit, pin = True, False
        elif not want:
            for m in protect[team]:
                pinned.discard((team, m))
            protect[team].clear()
            admit = not (spread_bypass and p_hit.get(member, 1.0) <= 0.0)
            pin = False
        else:
            for m in protect[team] - want:
                pinned.discard((team, m))
            protect[team] &= want
            pin = member in want and len(pinned) < capacity
            admit = pin or not bypass

        if not admit:
            continue
        cache[key] = True
        if pin:
            pinned.add(key)
            protect[team].add(member)
        while len(cache) > capacity:
            victim = next((k for k in cache if k not in pinned), None)
            if victim is None or victim == key:
                break
            del cache[victim]
    return flags


CANDIDATES = {
    "现状": dict(),
    "稳定": dict(stable=True),
    "稳定+排除": dict(stable=True, excl=True),
    "稳定+排除+w2": dict(stable=True, excl=True, window=2),
    "稳定+排除+iso": dict(stable=True, excl=True, iso=True),
    "稳定+iso": dict(stable=True, iso=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teams", type=int, default=9)
    ap.add_argument("--members", type=int, default=7)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--context", type=int, default=9728)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--topologies", default="ring,empirical,star,debate,pairs,mixed")
    args = ap.parse_args()

    topos = args.topologies.split(",")
    sess = args.teams * args.members
    loads = [1.0, 1.67, 2.33, 3.0, 3.67, 4.33, 5.0]
    caps = [max(int(round(sess / L)), 1) for L in loads]
    print(f"teams={args.teams} members={args.members} rounds={args.rounds} "
          f"{sess} 会话   负载 "
          f"{'/'.join(f'{sess/c:.2f}x' for c in caps)}")
    print(f"{len(topos)} 拓扑 x {len(caps)} 负载 x {args.seeds} 种子 = "
          f"{len(topos)*len(caps)*args.seeds} 格\n")

    gaps = {n: [] for n in CANDIDATES}
    gaps["原版函数"] = []
    starved = {n: 0 for n in CANDIDATES}
    starved["原版函数"] = 0
    per_topo = {n: defaultdict(list) for n in list(CANDIDATES) + ["原版函数"]}

    for topo in topos:
        for seed in range(args.seeds):
            refs = O.topo_reference(topo, args.teams, args.members,
                                    args.rounds, seed=seed)
            tof = [t for t, _ in refs]
            for cap in caps:
                fixed = {
                    "LRU": O.run_lru(refs, cap),
                    "bypass": O.run_lru_bypass(refs, cap),
                    "Cohort": O.run_cohort(refs, cap, args.members),
                }
                sc = {a: O.score(a, f, tof, args.context)
                      for a, f in fixed.items()}
                best = min(sc, key=lambda a: sc[a]["miss_rate"])
                bm = sc[best]["miss_rate"]

                f0 = O.run_concentrate(refs, cap, args.members)
                s0 = O.score("x", f0, tof, args.context)
                gaps["原版函数"].append((s0["miss_rate"] - bm) * 100)
                starved["原版函数"] += s0["starved_teams"]
                per_topo["原版函数"][topo].append((s0["miss_rate"] - bm) * 100)

                for name, kw in CANDIDATES.items():
                    f = run_concentrate_v(refs, cap, args.members, **kw)
                    s = O.score("x", f, tof, args.context)
                    g = (s["miss_rate"] - bm) * 100
                    gaps[name].append(g)
                    starved[name] += s["starved_teams"]
                    per_topo[name][topo].append(g)

    print("=== 距逐格最优固定策略的差距，pp（负数即更优）===")
    print(f"{'候选':<18}{'中位':>7}{'p90':>7}{'最坏':>7}{'≤1pp':>7}{'更优':>6}{'饿死':>6}")
    n = len(gaps["原版函数"])
    for name in ["原版函数"] + list(CANDIDATES):
        g = sorted(gaps[name])
        p90 = g[min(int(0.9 * len(g)), len(g) - 1)]
        print(f"{name:<18}{statistics.median(g):>+7.2f}{p90:>+7.2f}"
              f"{max(g):>+7.2f}{sum(1 for x in g if x <= 1):>5}/{n:<3}"
              f"{sum(1 for x in g if x < 0):>5}{starved[name]:>6}")

    print(f"\n=== 各拓扑中位差距，pp ===")
    print(f"{'候选':<18}" + "".join(f"{t[:9]:>11}" for t in topos))
    for name in ["原版函数"] + list(CANDIDATES):
        print(f"{name:<18}" + "".join(
            f"{statistics.median(per_topo[name][t]):>+11.2f}" for t in topos))


if __name__ == "__main__":
    main()
