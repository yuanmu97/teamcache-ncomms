#!/usr/bin/env python3
import json
from collections import defaultdict

def turns(path):
    by = defaultdict(lambda: [0, 0])
    for line in open(path):
        d = json.loads(line)
        if d.get("status") != 200 or ":" not in d.get("rid", ""):
            continue
        t = int(d["rid"].split(":")[2])
        by[t][0] += d.get("cached_tokens") or 0
        by[t][1] += d["prompt_tokens"]
    return by

def loadj(path):
    return json.load(open(path))

ARMS = ["lru", "cohort", "unified"]
BASE = "e2e-autogen-design-{arm}-20t-p3000r8"

print("=== 逐轮命中 ===")
print(f"{'轮':>3}" + "".join(f"{a:>10}" for a in ARMS))
data = {a: turns(f"traces/{BASE.format(arm=a)}.jsonl") for a in ARMS}
for t in sorted(data["lru"]):
    print(f"{t:>3}" + "".join(
        f"{data[a][t][0] / data[a][t][1] * 100:>9.1f}%" for a in ARMS))

print("\n=== 全程 / 热启动 / 稳态 ===")
print(f"{'窗口':<8}{'LRU':>12}{'Cohort q=2':>14}{'unified':>12}{'上限':>8}")
for key, label in (("full", "全程"), ("hot", "热启动"), ("steady", "稳态")):
    row = f"{label:<8}"
    ceil = 0
    for a in ARMS:
        w = loadj(f"results/e2e/{BASE.format(arm=a)}.json")[key]
        row += f"{w['prefix_hit_rate'] * 100:>11.2f}%"
        ceil = w["infinite_hit_rate"]
    row += f"{ceil * 100:>7.1f}%"
    print(row)

print("\n=== 墙钟与 TTTA ===")
for a in ARMS:
    d = loadj(f"results/e2e/{BASE.format(arm=a)}.json")
    print(f"{a:<8} wall={d['wall_s']}s  req={d['requests']}  "
          f"ttta={d['full']['ttta_p50']:.1f}s  "
          f"hot_ttta={d['hot']['ttta_p50']:.1f}s  "
          f"steady_ttta={d['steady']['ttta_p50']:.1f}s")
