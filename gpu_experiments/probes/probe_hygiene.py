#!/usr/bin/env python3
"""两处"卫生"改动在真实轨迹上值多少，纯 CPU。

改动只有两条，都已在 126 格网格上量为免费（中位 +0.04pp 不变、零饿死）：
  1. `top` 打平时按成员 id 稳定排序
  2. 排序与定价时排除当前这次引用

它们修的是一个潜在退化：`observe()` 先给当前成员的引用计数加一，而 `top`
又在一个 set 上排序，于是集中子集容易恒含刚发言的那个成员——也就是 LRU
本来就留着的那个——使规则做了决定却什么都不改变。
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_trace import BlockLRU, _Req, load, make_policy  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
POOL = 420800

EDITS = [
    ("def concentrate_on(self, team, share_tokens, margin, min_samples):",
     "def concentrate_on(self, team, share_tokens, margin, min_samples,\n"
     "                       cur=None):"),
    ("            stats[m] = (self.rate.get((team, m), 0.0),",
     "            stats[m] = (self.rate.get((team, m), 0.0)\n"
     "                        - (1.0 if m == cur else 0.0),"),
    ("for m in sorted(stats, key=lambda x: -stats[x][0]):",
     "for m in sorted(stats, key=lambda x: (-stats[x][0], x)):"),
    ("""        want = st.concentrate_on(team, share, cfg.adapt_margin,
                                 cfg.adapt_min_samples)""",
     """        want = st.concentrate_on(team, share, cfg.adapt_margin,
                                 cfg.adapt_min_samples, member)"""),
]


def build_hygiene():
    src = open(os.path.join(HERE, "patch_sglang_cohort.py")).read()
    o = {}
    exec(compile(src, "patch", "exec"), o)
    h = o["HELPER"]
    applied = 0
    for a, b in EDITS:
        if a in h:
            h = h.replace(a, b)
        else:
            applied += 1          # 已落地到文件里，无需再改
    if applied and applied != len(EDITS):
        raise SystemExit(f"两处改动只落地了一部分（{applied}/{len(EDITS)}）")
    doc = {"mode": "unified", "members": 5, "pool_tokens": POOL,
           "window_s": 1e9}
    fd, c = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as fh:
        json.dump(doc, fh)
    os.environ.update({"COHORT_CONTROL": c, "COHORT_MODE": "unified",
                       "COHORT_POOL_TOKENS": str(POOL),
                       "COHORT_MEMBERS": "5", "COHORT_WINDOW_S": "1e9"})
    ns = {}
    exec(compile(h, "<h>", "exec"), ns)
    return ns["_cohort_skip_insert"]


def main():
    trace = (sys.argv[1] if len(sys.argv) > 1
             else "traces/e2e-autogen-design-lru-20t.jsonl")
    rows_all = load(trace)

    def run(skip, minturn):
        rows = [r for r in rows_all if r["turn"] >= minturn]
        c = BlockLRU(POOL)
        tot = hit = ns = 0
        for r in rows:
            s = f"{r['team']}:{r['member']}"
            h, nb = c.lookup(s, r["prompt"])
            tot += r["prompt"]
            hit += h
            if skip is not None and skip(_Req(r["rid"], r["prompt"])):
                ns += 1
            else:
                c.insert(s, nb)
        return hit / tot, ns, len(rows)

    for mt, tag in ((0, "全程 0-12 轮"), (7, "热启动 7-12 轮")):
        print(f"\n--- {tag}")
        b, _, n = run(None, mt)
        print(f"{'upstream LRU':<26}{b*100:>8.2f}%   ({n} 请求)")
        for name, skip in (
            ("unified 原版", make_policy("unified", POOL, 5, {})[0]),
            ("unified 稳定+排除", build_hygiene()),
            ("Cohort q=3 (事后最优)",
             make_policy("group", POOL, 5, {"quota": 3})[0]),
        ):
            a, ns, _ = run(skip, mt)
            print(f"{name:<26}{a*100:>8.2f}% {(a-b)*100:>+8.2f}pp  绕过 {ns}")


if __name__ == "__main__":
    main()
