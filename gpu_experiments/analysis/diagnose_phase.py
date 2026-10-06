#!/usr/bin/env python3
"""把 unified 在真实 trace 上的每一次定价摊开。

要回答的只有一个问题：第 2→3 轮从 65% 掉到 7%，是规则判了集中之后又放手，
还是它从未真正集中、前两轮只是池子还装得下。

定价拆成三个同时能读的量：
  concentrated  份额里装得下的引用质量
  spread_hist   窗口里「LRU 会不会留住」的历史平均 —— 现在引擎用的
  spread_now    当前 token 栈上「LRU 现在会不会留住」—— 规则声称在测的那个量
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_trace import _Req, load, make_policy  # noqa: E402


def shadow_hold(stack, key, pool):
    size = stack.get(key, 0)
    if size <= 0:
        return None
    above = 0
    for k in reversed(stack):
        if k == key:
            break
        above += stack[k]
    return 1 if above + size <= pool else 0


def dump(trace, pool=420800, members=5):
    rows = load(trace)
    skip, cfg_used = make_policy("unified", pool, members, {"window_s": 1e9})
    st = skip.__globals__["_cohort_state"]

    print(f"trace={os.path.basename(trace)}  n={len(rows)}  "
          f"margin={cfg_used.adapt_margin}  "
          f"min_samples={cfg_used.adapt_min_samples}  "
          f"window={cfg_used.adapt_window}")

    by = {}
    for r in rows:
        req = _Req(r["rid"], r["prompt"])
        skipped = skip(req)
        team, member = r["team"], r["member"]
        roster = st.roster.get(team, set())
        stats = {}
        ready = True
        for m in roster:
            hist = st.held.get((team, m))
            if not hist or len(hist) < cfg_used.adapt_min_samples:
                ready = False
                break
            rate = st.rate.get((team, m), 0.0) - (1.0 if m == member else 0.0)
            bits = [b if not isinstance(b, tuple) else b[0] for b in hist]
            under = [b[1] for b in hist if isinstance(b, tuple)]
            p_hist = (sum(b for b, u in ((x if isinstance(x, tuple) else (x, True))
                                         for x in hist) if u)
                      / max(1, sum(1 for x in hist
                                   if not isinstance(x, tuple) or x[1])))
            p_hist_all = sum(bits) / float(len(bits)) if bits else 0
            p_now = shadow_hold(st.stack, (team, m), pool)
            stats[m] = (rate, p_hist, p_now, st.stack.get((team, m), 0),
                        list(hist[-6:]))
        active = max(len({t for t, _ in st.stack}), 1)
        # 与引擎同一条公式
        sizes = [v for v in st.stack.values() if v > 0]
        reserve = (sum(sizes) / len(sizes)) if sizes else 0
        share = max(pool - active * reserve, reserve) / max(active, 1)
        share_plain = pool / max(active, 1)

        footprint = sum(st.stack.values())
        rec = {
            "turn": r["turn"], "skip": skipped, "ready": ready,
            "share": share, "share_plain": share_plain,
            "reserve": reserve, "active": active,
            "n_roster": len(roster), "prompt": r["prompt"],
            "footprint": footprint, "team": team, "member": member,
        }
        if ready and stats:
            mass = sum(x[0] for x in stats.values()) or 1
            top, spent = set(), 0
            for m in sorted(stats, key=lambda x: (-stats[x][0], x)):
                size = stats[m][3]
                if top and spent + size > share:
                    continue
                top.add(m)
                spent += size
            conc = sum(stats[m][0] for m in top) / mass
            spread_h = sum(a * b for a, b, _, _, _ in stats.values()) / mass
            spread_n = sum(a * (c if c is not None else 1)
                           for a, _, c, _, _ in stats.values()) / mass
            # 排除当前成员后的即时定价
            mass_x = sum(stats[m][0] for m in stats if m != member) or 1
            spread_nx = sum(stats[m][0] * (stats[m][2] if stats[m][2] is not None else 1)
                            for m in stats if m != member) / mass_x
            rec.update({
                "conc": conc, "spread_h": spread_h, "spread_n": spread_n,
                "spread_nx": spread_nx, "n_top": len(top),
                "top": tuple(sorted(top)),
                "decide_hist": "C" if conc > spread_h * cfg_used.adapt_margin else "S",
                "decide_now": "C" if conc > spread_n * cfg_used.adapt_margin else "S",
                "decide_nx": "C" if conc > spread_nx * cfg_used.adapt_margin else "S",
                "spent": spent,
            })
        by.setdefault(r["turn"], []).append(rec)

    print(f"\n{'轮':>3} {'n':>4} {'就绪':>5} {'绕过':>5} "
          f"{'foot':>8} {'share':>7} {'装下':>5} "
          f"{'conc':>6} {'spr_h':>6} {'spr_n':>6} "
          f"{'判h':>8} {'过订':>5}")
    for t in sorted(by):
        rs = by[t]
        ready = [x for x in rs if x.get("ready")]
        n_c_h = sum(1 for x in ready if x.get("decide_hist") == "C")
        def avg(k):
            xs = [x[k] for x in ready if k in x]
            return sum(xs) / len(xs) if xs else 0
        foot = sum(x["footprint"] for x in rs) / len(rs)
        print(f"{t:>3} {len(rs):>4} {len(ready):>5} "
              f"{sum(1 for x in rs if x['skip']):>5} "
              f"{foot:>8.0f} {avg('share'):>7.0f} "
              f"{avg('n_top'):>5.1f} "
              f"{avg('conc'):>6.2f} {avg('spread_h'):>6.2f} "
              f"{avg('spread_n'):>6.2f} "
              f"{n_c_h:>3}/{len(ready):<4} "
              f"{'YES' if foot > pool else 'no':>5}")

    print("\n=== 每团队判 C/S 序列与 top 是否在转 ===")
    teams = sorted({x["team"] for rs in by.values() for x in rs})
    flips = rot = 0
    for team in teams:
        seq, tops = [], []
        for t in sorted(by):
            xs = [x for x in by[t] if x["team"] == team and x.get("ready")]
            if not xs:
                seq.append("-")
                continue
            d = xs[-1].get("decide_hist", "-")
            seq.append(d)
            if "top" in xs[-1]:
                tops.append(xs[-1]["top"])
        if len(set(x for x in seq if x in "CS")) > 1:
            flips += 1
        nuniq = len(set(tops))
        if nuniq > 1:
            rot += 1
        if team in teams[:4] or nuniq > 1 or len(set(seq)) > 1:
            print(f"  {team:<8} {' '.join(seq):<20}  tops={tops}")
    print(f"有过 C↔S 翻转的团队: {flips}/{len(teams)}   "
          f"top 集合变过的: {rot}/{len(teams)}")

    # 抽一个团队看成员级 hist
    print("\n=== 团队 0 每轮成员 hist（最近 6 个 would-hold）===")
    # 重新走一遍太贵；从最后的 state 读最终 hist，并在上面循环里没存团队。
    # 用 rate 键列出。
    team0 = None
    for k in st.held:
        if k[0].endswith("0") or k[0] == "0" or k[0].endswith("-0"):
            team0 = k[0]
            break
    # rid 的 team 是字符串，harness 用的是什么
    teams = sorted({k[0] for k in st.held})
    t0 = teams[0]
    print(f"team={t0}")
    for m in sorted({k[1] for k in st.held if k[0] == t0}):
        h = st.held[(t0, m)]
        print(f"  m={m} n={len(h)} mean={sum(h)/len(h):.2f}  {h}")


if __name__ == "__main__":
    dump(sys.argv[1])
