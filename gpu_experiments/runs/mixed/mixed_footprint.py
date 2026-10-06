#!/usr/bin/env python3
"""Pick mixed-fleet size from footprint only. Do not look at policy quality."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
POOL = 420800
RING_SESS = 9728 + 512  # same growth model as star/ring closed-loop
CANDIDATES = (8, 10, 12, 16, 20)


def selector_latest_sum(path: Path) -> int:
    latest = {}
    for line in path.read_text().splitlines():
        d = json.loads(line)
        hdr = d.get("headers") or {}
        if "x-team" not in hdr:
            continue
        latest[(d["team"], d["member"])] = d["prompt_tokens"]
    return int(sum(latest.values()))


def main():
    trace = ROOT / "traces" / "selector-debug-k5-Kimi-K3.jsonl"
    sel_one = selector_latest_sum(trace)
    ring_one = 5 * RING_SESS
    rows = []
    for n in CANDIDATES:
        half = n // 2
        foot = half * ring_one + half * sel_one
        load = foot / POOL
        rows.append({
            "teams": n,
            "ring_teams": half,
            "selector_teams": half,
            "ring_team_tokens": ring_one,
            "selector_team_tokens_end_of_trace": sel_one,
            "footprint_tokens": foot,
            "pool": POOL,
            "overload": load,
            "in_target_1_2_2_5": 1.2 <= load <= 2.5,
        })
    in_range = [r for r in rows if r["in_target_1_2_2_5"]]
    chosen = None
    if in_range:
        in_range.sort(key=lambda r: (abs(r["overload"] - 1.5), r["teams"]))
        chosen = in_range[0]["teams"]
    out = {
        "policy_quality_consulted": False,
        "target_overload": [1.2, 2.5],
        "prefer_near": 1.5,
        "prefer_teams": 12,
        "candidates": rows,
        "chosen_teams": chosen,
        "stop_if_none": chosen is None,
        "note": "do not shrink pool or crop history if every candidate misses the band",
    }
    dest = HERE / "cpu" / "mixed_footprint.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps({"chosen_teams": chosen, "rows": [
        {k: r[k] for k in ("teams", "overload", "in_target_1_2_2_5")} for r in rows
    ]}, indent=2))
    print("wrote", dest)


if __name__ == "__main__":
    main()
