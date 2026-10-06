#!/usr/bin/env python3
"""在真实 trace 上比较 unified 的修法变体，纯 CPU。

改动以源码替换的方式施加在 HELPER 上，patch_sglang_cohort.py 本身不动，
等哪个变体站得住再落回去。
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_trace import BlockLRU, _Req, load  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

STABLE_OLD = "for m in sorted(stats, key=lambda x: -stats[x][0]):"
STABLE_NEW = "for m in sorted(stats, key=lambda x: (-stats[x][0], x)):"

AGE_OLD = """        self.since_age += 1
        if self.since_age >= window * 8:
            self.since_age = 0
            for k in list(self.rate):
                self.rate[k] *= 0.5
                if self.rate[k] < 0.01:
                    del self.rate[k]"""

AGE_PERTEAM = """        ages = self.__dict__.setdefault("_age_t", {})
        ages[team] = ages.get(team, 0) + 1
        if ages[team] >= window * 8:
            ages[team] = 0
            for k in list(self.rate):
                if k[0] == team:
                    self.rate[k] *= 0.5
                    if self.rate[k] < 0.01:
                        del self.rate[k]"""

AGE_OFF = "        pass"

SIG_OLD = "def concentrate_on(self, team, share_tokens, margin, min_samples):"
SIG_NEW = ("def concentrate_on(self, team, share_tokens, margin, min_samples,"
           " cur=None):")
RATE_OLD = """            stats[m] = (self.rate.get((team, m), 0.0),"""
RATE_NEW = """            stats[m] = (self.rate.get((team, m), 0.0)
                        - (1.0 if m == cur else 0.0),"""
CALL_OLD = """        want = st.concentrate_on(team, share, cfg.adapt_margin,
                                 cfg.adapt_min_samples)"""
CALL_NEW = """        want = st.concentrate_on(team, share, cfg.adapt_margin,
                                 cfg.adapt_min_samples, member)"""
EXCL = [(SIG_OLD, SIG_NEW), (RATE_OLD, RATE_NEW), (CALL_OLD, CALL_NEW)]

VARIANTS = {
    "现状": [],
    "稳定打平": [(STABLE_OLD, STABLE_NEW)],
    "按团队老化": [(AGE_OLD, AGE_PERTEAM)],
    "按团队老化+稳定": [(AGE_OLD, AGE_PERTEAM), (STABLE_OLD, STABLE_NEW)],
    "关老化+稳定": [(AGE_OLD, AGE_OFF), (STABLE_OLD, STABLE_NEW)],
}
VARIANTS["关老化+稳定+排除当前"] = (
    [(AGE_OLD, AGE_OFF), (STABLE_OLD, STABLE_NEW)] + EXCL)
VARIANTS["按团队老化+稳定+排除当前"] = (
    [(AGE_OLD, AGE_PERTEAM), (STABLE_OLD, STABLE_NEW)] + EXCL)


def build(edits, mode, pool_tokens, members, extra=None):
    src = open(os.path.join(HERE, "patch_sglang_cohort.py")).read()
    o = {}
    exec(compile(src, "patch", "exec"), o)
    helper = o["HELPER"]
    for old, new in edits:
        if old not in helper:
            raise SystemExit(f"替换未命中:\n{old[:80]}")
        helper = helper.replace(old, new)
    doc = {"mode": mode, "members": members, "pool_tokens": pool_tokens,
           "window_s": 1e9}
    doc.update(extra or {})
    fd, ctrl = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as fh:
        json.dump(doc, fh)
    env = {"COHORT_CONTROL": ctrl, "COHORT_MODE": mode,
           "COHORT_POOL_TOKENS": str(pool_tokens),
           "COHORT_MEMBERS": str(members), "COHORT_WINDOW_S": "1e9"}
    for k, v in (extra or {}).items():
        env["COHORT_" + k.upper()] = str(v)
    os.environ.update(env)
    ns = {}
    exec(compile(helper, "<helper>", "exec"), ns)
    return ns["_cohort_skip_insert"]


def run(rows, skip, pool_tokens):
    cache = BlockLRU(pool_tokens)
    tot = hit = nskip = 0
    per_turn = {}
    for r in rows:
        sess = f"{r['team']}:{r['member']}"
        h, nblk = cache.lookup(sess, r["prompt"])
        tot += r["prompt"]
        hit += h
        a = per_turn.setdefault(r["turn"], [0, 0])
        a[0] += h
        a[1] += r["prompt"]
        if skip is not None and skip(_Req(r["rid"], r["prompt"])):
            nskip += 1
        else:
            cache.insert(sess, nblk)
    return hit / tot, nskip, {k: v[0] / v[1] for k, v in sorted(per_turn.items())}


def main():
    trace = sys.argv[1]
    pool = int(sys.argv[2]) if len(sys.argv) > 2 else 420800
    rows = load(trace)
    print(f"{len(rows)} 请求  {len({r['team'] for r in rows})} 团队  "
          f"{len({(r['team'], r['member']) for r in rows})} 会话  池 {pool:,}")

    base, _, base_turn = run(rows, None, pool)
    print(f"\n{'变体':<18} {'命中率':>8} {'相对 LRU':>10} {'绕过':>7}   后段(8-12 轮)")
    print(f"{'upstream LRU':<18} {base*100:>7.2f}% {0.0:>+9.2f}pp {0:>6}   "
          f"{'/'.join(f'{base_turn[t]*100:.0f}' for t in range(8, 13) if t in base_turn)}")

    rows_ = rows
    results = {}
    for name in ("group q=2", "group q=3"):
        q = int(name[-1])
        skip = build([], "group", pool, 5, {"quota": q})
        h, s, pt = run(rows_, skip, pool)
        results[name] = h
        print(f"{name:<18} {h*100:>7.2f}% {(h-base)*100:>+9.2f}pp {s:>6}   "
              f"{'/'.join(f'{pt[t]*100:.0f}' for t in range(8, 13) if t in pt)}")

    for win in (1, 2, 3, 4, 6):
        print(f"--- adapt_window={win}")
        for name, edits in VARIANTS.items():
            skip = build(edits, "unified", pool, 5, {"adapt_window": win})
            h, s, pt = run(rows_, skip, pool)
            results[(name, win)] = h
            print(f"{'  ' + name:<18} {h*100:>7.2f}% {(h-base)*100:>+9.2f}pp "
                  f"{s:>6}   {'/'.join(f'{pt[t]*100:.0f}' for t in range(8, 13) if t in pt)}")


if __name__ == "__main__":
    main()
