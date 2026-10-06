#!/usr/bin/env python3
"""Speaking-order statistics of every recorded trace (Fig. 1a-c).

Writes $TEAMCACHE_OUT/collections-summary.json; the deposit ships the same
file as gpu/results/collections-summary.json.
"""

from __future__ import annotations

import json

from .paths import OUT, TRACES as ROOT


def members_of(rows):
    out = []
    for r in rows:
        if "system" not in (r.get("msg_roles") or []):
            continue
        sha = (r.get("prefix_sha") or [None])[0]
        out.append((sha, r.get("prompt_tokens") or 0, r.get("completion_tokens") or 0))
    return out


def recency(members):
    idx, last, ranks, order = {}, {}, [], []
    for i, (sha, _, _) in enumerate(members):
        idx.setdefault(sha, len(idx))
        order.append(idx[sha])
        if sha in last:
            ranks.append(len({members[j][0] for j in range(last[sha] + 1, i)}))
        last[sha] = i
    k = max(len(idx), 1)
    norm = (sum(ranks) / len(ranks) / (k - 1)) if ranks and k > 1 else None
    fc = sum(1 for i, (sha, _, _) in enumerate(members)
             if any(members[j][0] == sha for j in range(i + 1, len(members))))
    return {
        "k": k, "member_turns": len(members),
        "norm_rank": norm,
        "self_repeat": (sum(1 for r in ranks if r == 0) / len(ranks)) if ranks else None,
        "false_completion": fc / len(members) if members else None,
        "prompt_median": sorted(m[1] for m in members)[len(members) // 2] if members else None,
        "gen_median": sorted(m[2] for m in members)[len(members) // 2] if members else None,
        "order": "".join(str(x) for x in order[:40]),
    }


def main():
    print(f"{'trace':<42}{'k':>3}{'n':>5}{'norm':>6}{'self':>6}{'false':>7}{'p50in':>8}{'p50out':>8}")
    rows_out = []
    for path in sorted(ROOT.glob("*.jsonl")):
        if path.name.startswith("e2e-"):
            continue
        raw = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        raw = [r for r in raw if r.get("status") == 200]
        if len(raw) < 8:
            continue
        s = recency(members_of(raw))
        s["trace"] = path.name
        rows_out.append(s)
        print(f"{path.name:<42}{s['k']:3d}{s['member_turns']:5d}"
              f"{(s['norm_rank'] or 0)*100:5.0f}%{(s['self_repeat'] or 0)*100:5.0f}%"
              f"{(s['false_completion'] or 0)*100:6.0f}%"
              f"{s['prompt_median'] or 0:8d}{s['gen_median'] or 0:8d}")
    out = OUT / "collections-summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows_out, open(out, "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
