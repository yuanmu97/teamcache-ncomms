#!/usr/bin/env python3
"""在真实 trace 上离线回放准入策略，块粒度，纯 CPU。

存在的理由：上机一组三臂要二十分钟 GPU，而策略的命中率只取决于请求序列和
池容量，这两样 trace 里都有。所以先在这里筛，只把预测能赢的配置拿去上机。

决策代码不是重写的，是从 patch_sglang_cohort.py 里 exec 出来的那一份
`_cohort_skip_insert`，也就是引擎里真正跑的那一份。

缓存模型：每个会话一条严格延长的块链。命中是从第 0 块起连续驻留的前缀长度，
所以链上的空洞会自然截断命中，不需要额外强加"只淘汰叶子"的约束。
"""
import argparse
import json
import os
import tempfile
from collections import OrderedDict, defaultdict

PAGE = 64


def load(path):
    rows = []
    for line in open(path):
        d = json.loads(line)
        if d.get("status") != 200 or ":" not in d.get("rid", ""):
            continue
        team, member, turn = d["rid"].split(":")
        rows.append({
            "rid": d["rid"],
            "team": team,
            "member": int(member),
            "turn": int(turn),
            "prompt": d["prompt_tokens"],
            "cached": d.get("cached_tokens") or 0,
            "t": d["t_recv"],
        })
    rows.sort(key=lambda r: r["t"])
    return rows


class BlockLRU:
    """token 计的块级缓存，radix 结构，先淘汰叶子。

    为什么必须先淘汰叶子。第一版按全局块近因淘汰，把一个会话的整条链看成
    近因相同、于是连前缀一起丢。在上下文从 384 起爬升的运行里这是好近似
    （链上每一块每轮都刚被重写，全有或全无确实接近），实测 LRU 28.43% 对
    回放 22.96%。但一旦成员带着一段**静态**私有前置上下文，这个近似就彻底
    失效：真实引擎里那段前缀是有子节点的内部节点，SGLang 先淘汰叶子，所以
    前缀被保护、只有增长的尾巴被丢。实测是 LRU 79.51%，第一版回放预测
    0.00%。差 79pp，全部来自这一个被放掉的约束。

    所以这里按会话维护驻留块数与近因：命中是从第 0 块起的连续前缀（由构造
    保证连续），淘汰从最久未用的会话的**尾部**逐块进行。
    """

    def __init__(self, pool_tokens):
        self.cap = pool_tokens // PAGE
        self.resident = {}               # sess -> 驻留的前缀块数
        self.used = OrderedDict()        # sess -> 近因（末尾最新）
        self.total = 0
        self.evicted = 0

    def lookup(self, sess, prompt):
        nblk = prompt // PAGE
        hit = min(self.resident.get(sess, 0), nblk)
        # 引擎至少要留一个 token 去算，全命中要退一块
        if hit == nblk and nblk > 0:
            hit -= 1
        if sess in self.used:
            self.used.move_to_end(sess)
        return hit * PAGE, nblk

    def insert(self, sess, nblk):
        have = self.resident.get(sess, 0)
        if nblk > have:
            self.total += nblk - have
            self.resident[sess] = nblk
        self.used[sess] = True
        self.used.move_to_end(sess)
        # 逐块剥，而不是整条剥：淘汰在腾够空间的那一刻停下，于是最后一个
        # 受害会话留着一段**部分前缀**。这正是先淘汰叶子与全局块近因的差别
        # 所在，也是静态前置上下文能活下来的原因。
        while self.total > self.cap:
            victim = next((s for s in self.used if s != sess), None)
            if victim is None:
                break
            self.resident[victim] -= 1
            self.total -= 1
            self.evicted += 1
            if self.resident[victim] <= 0:
                del self.resident[victim]
                del self.used[victim]


def make_policy(mode, pool_tokens, members, extra=None):
    """把引擎里那份决策代码取出来，按 mode 配好。"""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "patch_sglang_cohort.py")).read()
    ns_outer = {}
    exec(compile(src, "patch", "exec"), ns_outer)
    helper = ns_outer["HELPER"]

    doc = {"mode": mode, "members": members, "pool_tokens": pool_tokens,
           "window_s": 1e9}
    doc.update(extra or {})
    fd, control = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as fh:
        json.dump(doc, fh)
    os.environ["COHORT_CONTROL"] = control
    os.environ["COHORT_MODE"] = mode
    os.environ["COHORT_POOL_TOKENS"] = str(pool_tokens)
    os.environ["COHORT_MEMBERS"] = str(members)
    os.environ["COHORT_WINDOW_S"] = "1e9"
    for k, v in (extra or {}).items():
        os.environ["COHORT_" + k.upper()] = str(v)

    ns = {}
    exec(compile(helper, "<helper>", "exec"), ns)
    return ns["_cohort_skip_insert"], ns["_cohort_cfg"]


class _Req:
    __slots__ = ("rid", "skip_radix_cache_insert", "origin_input_ids")

    def __init__(self, rid, tokens):
        self.rid = rid
        self.skip_radix_cache_insert = False
        self.origin_input_ids = [0] * tokens


def replay(rows, mode, pool_tokens, members, extra=None):
    cache = BlockLRU(pool_tokens)
    if mode == "upstream":
        skip = None
    else:
        skip, cfg = make_policy(mode, pool_tokens, members, extra)

    tot = hit = 0
    skipped = 0
    by_turn = defaultdict(lambda: [0, 0])
    for r in rows:
        sess = f"{r['team']}:{r['member']}"
        h, nblk = cache.lookup(sess, r["prompt"])
        tot += r["prompt"]
        hit += h
        by_turn[r["turn"]][0] += h
        by_turn[r["turn"]][1] += r["prompt"]
        if skip is None:
            cache.insert(sess, nblk)
        else:
            req = _Req(r["rid"], r["prompt"])
            if skip(req):
                skipped += 1
            else:
                cache.insert(sess, nblk)
    return {"hit": hit / tot, "hit_tok": hit, "tot": tot,
            "skipped": skipped, "n": len(rows),
            "by_turn": {k: v[0] / v[1] for k, v in sorted(by_turn.items())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--pool", type=int, default=420800)
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--turns", action="store_true", help="打印逐轮命中")
    args = ap.parse_args()

    rows = load(args.trace)
    sess = {f"{r['team']}:{r['member']}" for r in rows}
    teams = {r["team"] for r in rows}
    measured = sum(r["cached"] for r in rows) / sum(r["prompt"] for r in rows)
    print(f"{len(rows)} 请求  {len(teams)} 团队  {len(sess)} 会话  "
          f"池 {args.pool:,} token  实测命中 {measured*100:.2f}%")

    arms = [
        ("upstream (LRU)", "upstream", None),
        ("group q=1", "group", {"quota": 1}),
        ("group q=2", "group", {"quota": 2}),
        ("group q=3", "group", {"quota": 3}),
        ("unified w=40 (现)", "unified", {}),
    ] + [(f"unified w={w}", "unified", {"adapt_window": w})
         for w in (2, 3, 4, 6, 8, 12)]
    print(f"\n{'策略':<22} {'命中率':>8} {'相对 LRU':>10} {'绕过':>8}")
    base = None
    out = {}
    for name, mode, extra in arms:
        res = replay(rows, mode, args.pool, args.members, extra)
        out[name] = res
        if base is None:
            base = res["hit"]
        d = (res["hit"] - base) * 100
        print(f"{name:<22} {res['hit']*100:>7.2f}% {d:>+9.2f}pp "
              f"{res['skipped']:>7}")

    if args.turns:
        names = [a[0] for a in arms]
        print(f"\n{'轮':>3}" + "".join(f"{n[:14]:>15}" for n in names))
        for t in sorted(out[names[0]]["by_turn"]):
            print(f"{t:>3}" + "".join(
                f"{out[n]['by_turn'][t]*100:>14.1f}%" for n in names))


if __name__ == "__main__":
    main()
