#!/usr/bin/env python3
"""Per-run GPU tables used by the figures, rebuilt from raw GPU result files.

Writes mixed_rows.json, sweep_rows.json, star_rows.json, reset_rows.json and
robustness.json to $TEAMCACHE_OUT/figure_data.  The deposit ships the same
tables under gpu/figure_data; `make check` compares the two.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .paths import E2E, FIGDATA_OUT, MIXED, RES, SWEEP, VLLM, ensure

WARMUP_N = 10
SEEDS = (11, 12, 13, 14)
MIXED_KEY = {
    "LRU": "lru",
    "Unpartitioned Admission": "group_blind",
    "Offline LFU": "static_rate",
    "TeamCache": "teamcache",
}
SWEEP_ARM = {
    "LRU": "lru",
    "Unpartitioned Admission": "bypass",
    "Static Partitioning": "cohort",
    "TeamCache": "unified",
}
SWEEP_TEAMS = (6, 8, 10, 12, 14, 16)


def loadj(path: Path):
    return json.loads(Path(path).read_text())


def load_jsonl(path: Path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def mixed_paths(policy_key: str, seed: int, base: Path = MIXED) -> tuple[Path, Path]:
    suf = "" if seed == 11 else f"_s{seed}"
    stem = f"mixed_{policy_key}_12t{suf}"
    return base / f"{stem}.json", base / f"{stem}.requests.jsonl"


def split_meas(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["team"]].append(r)
    warm, meas = [], []
    for xs in by.values():
        xs = sorted(xs, key=lambda x: x["t_send"])
        warm.extend(xs[:WARMUP_N])
        meas.extend(xs[WARMUP_N:])
    return warm, meas


def _rel(path: Path) -> str:
    return str(path.relative_to(MIXED.parents[1]))


def _mixed_row(jp: Path, lp: Path):
    doc = loadj(jp)
    _, meas = split_meas(load_jsonl(lp))
    kind = {}
    for kname in ("ring", "selector"):
        teams = [t for t in doc["per_team"] if t["kind"] == kname]
        cached = sum(t["prefix_tokens_cached"] for t in teams)
        prompt = sum(t["prefix_tokens_prompt"] for t in teams)
        xs = [r for r in meas if r.get("kind") == kname]
        kind[kname] = {
            "n_req": len(xs),
            "n_teams": len(teams),
            "token_reuse": cached / prompt,
            "request_hit": sum((r.get("cached") or 0) >= 0.5 for r in xs) / len(xs),
            "low_reuse_teams": sum(1 for t in teams if t["miss_rate"] >= 0.9),
            "cached": cached,
            "prompt": prompt,
        }
    return {
        "makespan_s": doc["wall_s"],
        "request_hit": doc["request_hit"],
        "token_reuse": (sum(r.get("cached_tokens") or 0 for r in meas)
                        / sum(r.get("prompt_tokens") or 0 for r in meas)),
        "low_reuse_teams": doc["low_reuse_teams"],
        "prompt": doc["prefix_tokens_prompt"],
        "cached": doc["prefix_tokens_cached"],
        "n_meas": len(meas),
        "source": _rel(jp),
        "ring": kind["ring"],
        "selector": kind["selector"],
    }


def extract_mixed():
    rows = []
    for seed in SEEDS:
        for paper, key in MIXED_KEY.items():
            jp, lp = mixed_paths(key, seed)
            rows.append({"seed": seed, "policy": paper} | _mixed_row(jp, lp))
    return rows


def extract_reset():
    """Controlled validation at seed 11 with a verified prefix-cache flush."""
    rows = []
    for paper, key in MIXED_KEY.items():
        jp, lp = mixed_paths(key, 11, MIXED / "isolation_s11")
        r = _mixed_row(jp, lp)
        rows.append({"policy": paper, "makespan_s": r["makespan_s"],
                     "request_hit": r["request_hit"], "token_reuse": r["token_reuse"],
                     "source": r["source"]})
    return rows


def extract_sweep():
    rows = []
    for n in SWEEP_TEAMS:
        for paper, arm in SWEEP_ARM.items():
            path = SWEEP / f"{arm}_{n}t_ring.json"
            d = loadj(path)
            rows.append({
                "n_teams": n,
                "policy": paper,
                "arm": arm,
                "wall_s": d["wall_s"],
                "request_hit": 1.0 - d["turn_miss_rate"],
                "low_reuse_teams": d["starved_teams"],
                "source": _rel(path),
            })
    return rows


def largest_n(sweep_rows, policy, budget):
    ok = [r["n_teams"] for r in sweep_rows
          if r["policy"] == policy and r["wall_s"] <= budget]
    if not ok:
        raise ValueError(f"no team count meets {budget}s for {policy}")
    return max(ok)


def extract_star():
    def pack(paper, path, pair):
        d = loadj(path)
        hit = d.get("request_hit")
        if hit is None:
            hit = 1.0 - d["turn_miss_rate"]
        return {"policy": paper, "makespan_s": d["wall_s"], "request_hit": hit,
                "pair": pair, "source": _rel(path)}

    return [
        pack("LRU", RES / "team-ablate" / "lru_9t_star.json", "original-locator"),
        pack("Unpartitioned Admission", RES / "team-ablate" / "bypass_9t_star.json",
             "original-locator"),
        pack("Offline LFU", MIXED / "star_static_rate_9t.json", "new-matched"),
        pack("TeamCache", MIXED / "star_teamcache_9t.json", "new-matched"),
    ]


def _team_miss(doc):
    teams = sorted(doc["per_team"], key=lambda t: t["team"])
    return [t["team"] for t in teams], [t["miss_rate"] * 100 for t in teams]


def extract_robustness():
    hi_l = loadj(E2E / "e2e-fleet-mixed-lru-10t-hicache-joint.json")
    hi_t = loadj(E2E / "e2e-fleet-mixed-unified-10t-hicache-joint.json")
    short_gpu = loadj(E2E / "e2e-autogen-design-lru-20t-p3000r8.json")
    short_l = loadj(E2E / "e2e-autogen-design-lru-20t-p3000r8-hicache.json")
    short_t = loadj(E2E / "e2e-autogen-design-unified-20t-p3000r8-hicache.json")
    gl10_l = loadj(E2E / "e2e-fleet-mixed-lru-10t-glm.json")
    gl10_t = loadj(E2E / "e2e-fleet-mixed-unified-10t-glm.json")
    gl_l = loadj(E2E / "e2e-fleet-mixed-lru-32t-glm-scale.json")
    gl_t = loadj(E2E / "e2e-fleet-mixed-unified-32t-glm-scale.json")
    v8_l = loadj(VLLM / "e2e-autogen-design-lru-8t-p3000r8-vllm-mask-280.metrics.json")
    v8_t = loadj(VLLM / "e2e-autogen-design-unified-8t-p3000r8-vllm-mask-280.metrics.json")
    v16_l = loadj(VLLM / "e2e-autogen-design-lru-8t-p3000r16-vllm-mask-280.metrics.json")
    v16_t = loadj(VLLM / "e2e-autogen-design-unified-8t-p3000r16-vllm-mask-280.metrics.json")
    hi_ids, hi_lru_miss = _team_miss(hi_l)
    _, hi_tc_miss = _team_miss(hi_t)
    gl_ids, gl_lru_miss = _team_miss(gl_l)
    _, gl_tc_miss = _team_miss(gl_t)
    return {
        "short_rounds": list(range(1, 9)),
        "short_gpu": [t["prefix_hit_rate"] * 100 for t in short_gpu["by_turn"]],
        "short_lru": [t["prefix_hit_rate"] * 100 for t in short_l["by_turn"]],
        "short_tc": [t["prefix_hit_rate"] * 100 for t in short_t["by_turn"]],
        "short_gpu_steady": short_gpu["steady"]["prefix_hit_rate"],
        "short_lru_steady": short_l["steady"]["prefix_hit_rate"],
        "short_tc_steady": short_t["steady"]["prefix_hit_rate"],
        "hicache": {
            "ids": hi_ids,
            "lru_miss": hi_lru_miss,
            "tc_miss": hi_tc_miss,
            "lru_low": hi_l["starved_teams"],
            "tc_low": hi_t["starved_teams"],
            "lru_wall": hi_l["wall_s"],
            "tc_wall": hi_t["wall_s"],
            "lru_cached": hi_l["mean_cached"],
            "tc_cached": hi_t["mean_cached"],
            "speedup": hi_l["wall_s"] / hi_t["wall_s"],
        },
        "glm": {
            "ids": gl_ids,
            "lru_miss": gl_lru_miss,
            "tc_miss": gl_tc_miss,
            "lru_low": gl_l["starved_teams"],
            "tc_low": gl_t["starved_teams"],
            "lru_wall": gl_l["wall_s"],
            "tc_wall": gl_t["wall_s"],
            "lru_cached": gl_l["mean_cached"],
            "tc_cached": gl_t["mean_cached"],
            "speedup": gl_l["wall_s"] / gl_t["wall_s"],
            "n_better": int(sum(t < l for t, l in zip(gl_tc_miss, gl_lru_miss))),
            "n_full_lru": int(sum(m >= 99.9 for m in gl_lru_miss)),
            "n_full_tc": int(sum(m >= 99.9 for m in gl_tc_miss)),
            "cached_10_lru": gl10_l["mean_cached"],
            "cached_10_tc": gl10_t["mean_cached"],
            "wall_10_lru": gl10_l["wall_s"],
            "wall_10_tc": gl10_t["wall_s"],
            "member_cached_lru": [m["mean_cached"] for m in gl_l["per_member"]],
            "member_cached_tc": [m["mean_cached"] for m in gl_t["per_member"]],
        },
        "vllm": {
            "r8_lru": v8_l["hit_frac"],
            "r8_tc": v8_t["hit_frac"],
            "r16_lru": v16_l["hit_frac"],
            "r16_tc": v16_t["hit_frac"],
            "r8_hits": (v8_l["delta_hits"], v8_t["delta_hits"]),
            "r16_hits": (v16_l["delta_hits"], v16_t["delta_hits"]),
        },
    }


def main():
    out = ensure(FIGDATA_OUT)
    tables = {
        "mixed_rows": extract_mixed(),
        "sweep_rows": extract_sweep(),
        "star_rows": extract_star(),
        "reset_rows": extract_reset(),
        "robustness": extract_robustness(),
    }
    for name, rows in tables.items():
        (out / f"{name}.json").write_text(json.dumps(rows, indent=2))
        print("wrote", out / f"{name}.json")


if __name__ == "__main__":
    main()
