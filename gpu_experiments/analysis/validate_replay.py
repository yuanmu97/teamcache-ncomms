#!/usr/bin/env python3
"""用两组结构不同的实测校验离线回放器的淘汰模型，纯 CPU。

两组数据是故意选成结构相反的：

  冷启动组   每个成员从 384 token 起爬升，没有静态前缀，整条链每轮重写
  共享前缀组 每个成员带 3,000 token 静态前置上下文，而那段前置上下文被
             20 个团队共享（trace 实证：每个成员序号只有 1 个 distinct
             prefix_sha），所以 distinct 前缀只有 5 份而不是 100 份

模型必须同时对上两组，否则它只是在某一种结构上碰巧对。共享是按 trace 实证
建模的：前置区的块键是（成员, 块号），尾部才带上团队。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import OrderedDict  # noqa: E402

from replay_trace import PAGE, load  # noqa: E402

POOL = 420800


def replay(rows, pool_tokens, shared_prefix_tokens=0):
    """块级缓存，先淘汰叶子，前置区可跨团队共享。

    命中按每条链从第 0 块起的连续驻留前缀算；淘汰从最久未用的链的尾部逐块
    进行，并在腾够空间时立刻停下，所以受害链会留下部分前缀。
    """
    cap = pool_tokens // PAGE
    pblk = shared_prefix_tokens // PAGE
    resident = {}                 # chain -> 驻留块数
    used = OrderedDict()          # chain -> 近因
    total = 0

    def keys(team, member, nblk):
        """这条 prompt 覆盖的链：共享前置区一条，私有尾部一条。"""
        if pblk and nblk > pblk:
            return [(("s", member), pblk), ((team, member), nblk - pblk)]
        if pblk:
            return [(("s", member), nblk)]
        return [((team, member), nblk)]

    tot = hit = 0
    for r in rows:
        team, member = r["team"], r["member"]
        nblk = r["prompt"] // PAGE
        tot += r["prompt"]

        # 命中：按链顺序累计连续驻留前缀，一旦某条链不满就停
        h = 0
        for chain, want in keys(team, member, nblk):
            have = min(resident.get(chain, 0), want)
            h += have
            if have < want:
                break
        if h == nblk and nblk > 0:
            h -= 1
        hit += h * PAGE

        for chain, want in keys(team, member, nblk):
            have = resident.get(chain, 0)
            if want > have:
                total += want - have
                resident[chain] = want
            used[chain] = True
            used.move_to_end(chain)

        live = {c for c, _ in keys(team, member, nblk)}
        while total > cap:
            victim = next((c for c in used if c not in live), None)
            if victim is None:
                break
            resident[victim] -= 1
            total -= 1
            if resident[victim] <= 0:
                del resident[victim]
                del used[victim]
    return hit / tot


def main():
    cases = [
        ("冷启动，无静态前缀", "archive/20t-cold-ramp/"
         "e2e-autogen-design-unified-20t.jsonl", 0, 28.43),
        ("前置 3000，跨团队共享", "traces/e2e-autogen-design-lru-20t.jsonl",
         3000, 79.51),
    ]
    print(f"{'数据':<24}{'共享前置':>9}{'回放':>8}{'实测 LRU':>10}{'偏差':>9}")
    for tag, path, shared, meas in cases:
        rows = load(path)
        got = replay(rows, POOL, shared) * 100
        print(f"{tag:<24}{shared:>9}{got:>7.2f}%{meas:>9.2f}%"
              f"{got - meas:>+8.2f}pp")
    print("\n对照：把共享前置当作私有（即我原先的假设）")
    rows = load("traces/e2e-autogen-design-lru-20t.jsonl")
    print(f"  前置 3000 视为私有 -> {replay(rows, POOL, 0)*100:.2f}%"
          f"  （实测 79.51%）")


if __name__ == "__main__":
    main()
