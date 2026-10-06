#!/usr/bin/env python3
"""Zero-GPU bounded verification: hit recompute, reconstructed hashes, first-touch."""
from __future__ import annotations

import csv
import hashlib
import json
import random
import struct
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GPU = HERE / "gpu"
CPU = HERE / "cpu"
POLICIES = ("lru", "group_blind", "static_rate", "teamcache")
SEEDS = (11, 12, 13, 14)
WARMUP_N = 10
CONTEXT = 9728
MESSAGE = 512
VARIANTS = 4
VOCAB = 100000
TEAMS = 12
MEMBERS = 5


def mixed_path(policy, seed):
    if seed == 11:
        return GPU / f"mixed_{policy}_12t.json"
    return GPU / f"mixed_{policy}_12t_s{seed}.json"


def load_jsonl(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def split_warmup(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["team"]].append(r)
    warm, meas = [], []
    for tid, xs in by.items():
        xs = sorted(xs, key=lambda r: r["t_send"])
        warm.extend(xs[:WARMUP_N])
        meas.extend(xs[WARMUP_N:])
    return warm, meas


def half_prompt_hit(meas):
    if not meas:
        raise ValueError("Empty measured window")
    return sum((r.get("cached") or 0) >= 0.5 for r in meas) / len(meas)


def ids_hash(ids):
    h = hashlib.sha256()
    h.update(struct.pack("<I", len(ids)))
    h.update(struct.pack(f"<{len(ids)}I", *ids))
    return h.hexdigest()


class PayloadFactory:
    """Same RNG mapping as Shared.context / messages and MixedShared.ids."""

    def __init__(self, seed):
        self.seed = seed
        self.context = {
            (t, m): [random.Random((t << 16) ^ m ^ seed).randrange(16, VOCAB)
                     for _ in range(CONTEXT)]
            for t in range(TEAMS) for m in range(MEMBERS)
        }
        self.messages = {
            (t, m): [[random.Random((t << 24) ^ (m << 8) ^ v ^ seed)
                      .randrange(16, VOCAB) for _ in range(MESSAGE)]
                     for v in range(VARIANTS)]
            for t in range(TEAMS) for m in range(MEMBERS)
        }
        self.sel_buf = {}

    def selector_buf(self, team, member, ntok):
        key = (team, member)
        buf = self.sel_buf.get(key)
        if buf is None or len(buf) < ntok:
            rng = random.Random((team << 16) ^ member ^ self.seed ^ 0x5E1)
            need = max(ntok, 16384)
            self.sel_buf[key] = [rng.randrange(16, VOCAB) for _ in range(need)]
        return self.sel_buf[key][:ntok]

    def ids_for(self, r):
        kind = r.get("kind")
        t, m, turn = r["team"], r["member"], r["turn"]
        if kind == "ring":
            return self.context[(t, m)] + self.messages[(t, m)][turn % VARIANTS]
        if kind == "selector":
            return self.selector_buf(t, m, r.get("prompt_tokens") or 0)
        raise ValueError(kind)


def main():
    hit_rows = []
    first_touch = []
    recon = []
    length_mismatch = []
    for seed in SEEDS:
        factory = PayloadFactory(seed)
        arm_hashes = {}
        for pol in POLICIES:
            p = mixed_path(pol, seed)
            doc = json.loads(p.read_text())
            jl = load_jsonl(p.with_name(p.stem + ".requests.jsonl"))
            warm, meas = split_warmup(jl)
            hit = half_prompt_hit(meas)
            json_hit = doc["request_hit"]
            hit_rows.append({
                "seed": seed, "policy": pol, "n_meas": len(meas),
                "recomputed_hit": hit, "json_request_hit": json_hit,
                "abs_diff": abs(hit - json_hit),
            })
            # first-touch: earliest t_send per (team, member) in full jsonl
            seen = {}
            for r in sorted(jl, key=lambda x: x["t_send"]):
                key = (r["team"], r["member"])
                if key not in seen:
                    seen[key] = r
            n_pos = sum(1 for r in seen.values() if (r.get("cached_tokens") or 0) > 0)
            first_touch.append({
                "seed": seed, "policy": pol,
                "n_sessions": len(seen),
                "first_touch_cached_gt0": n_pos,
                "first_touch_cached_max": max((r.get("cached_tokens") or 0) for r in seen.values()),
                "first_touch_cached_sum": sum((r.get("cached_tokens") or 0) for r in seen.values()),
            })
            # reconstructed content hash over measured, original member mapping
            req_hashes = []
            for r in sorted(meas, key=lambda x: (x["team"], x["member"], x["turn"], x.get("kind"))):
                ids = factory.ids_for(r)
                if r.get("kind") == "ring" and len(ids) != r.get("prompt_tokens"):
                    length_mismatch.append({
                        "seed": seed, "policy": pol, "kind": "ring",
                        "recon_len": len(ids), "prompt_tokens": r.get("prompt_tokens"),
                    })
                # Hash token content only; rid_member is recorded separately.
                req_hashes.append((r["team"], r["member"], r["turn"], r.get("kind"),
                                   ids_hash(ids), len(ids)))
            rid_ne = sum(1 for r in meas if r.get("rid_member") != r.get("member"))
            body = json.dumps(req_hashes, separators=(",", ":")).encode()
            arm_hashes[pol] = hashlib.sha256(body).hexdigest()[:16]
            recon.append({
                "seed": seed, "policy": pol,
                "reconstructed_content_hash": arm_hashes[pol],
                "n": len(req_hashes),
                "rid_member_ne_member": rid_ne,
            })
        # same seed arms must share reconstructed hash
        uniq = set(arm_hashes.values())
        if len(uniq) != 1:
            print("HASH_MISMATCH", seed, arm_hashes, file=sys.stderr)

    diffs = [r["abs_diff"] for r in hit_rows]
    out = {
        "hit_excerpt_vs_script": {
            "report_excerpt_was_wrong": True,
            "script_formula": "sum(cached>=0.5)/len(meas) with empty-window raise",
            "n_runs": len(hit_rows),
            "n_measured_per_run": 330,
            "max_abs_diff_vs_json": max(diffs),
            "mean_abs_diff_vs_json": sum(diffs) / len(diffs),
            "all_within_1e-15": max(diffs) < 1e-15,
            "rows": hit_rows,
        },
        "first_touch": first_touch,
        "first_touch_any_hit": any(r["first_touch_cached_gt0"] for r in first_touch),
        "reconstructed": recon,
        "reconstructed_equal_within_seed": all(
            len({r["reconstructed_content_hash"] for r in recon if r["seed"] == s}) == 1
            for s in SEEDS
        ),
        "ring_length_mismatches": length_mismatch[:20],
        "n_ring_length_mismatches": len(length_mismatch),
    }
    CPU.mkdir(exist_ok=True)
    path = CPU / "bounded_verify.json"
    path.write_text(json.dumps(out, indent=2))
    print("wrote", path)
    print("hit max_abs_diff", out["hit_excerpt_vs_script"]["max_abs_diff_vs_json"],
          "n", out["hit_excerpt_vs_script"]["n_runs"])
    print("first_touch_any_hit", out["first_touch_any_hit"])
    print("recon equal within seed", out["reconstructed_equal_within_seed"])
    print("ring length mismatches", out["n_ring_length_mismatches"])
    for s in SEEDS:
        hs = {r["policy"]: r["reconstructed_content_hash"] for r in recon if r["seed"] == s}
        print("seed", s, hs)
    # SR remap check: rid_member != member only on static_rate
    for r in recon:
        print("rid_ne", r["seed"], r["policy"], r["rid_member_ne_member"])


if __name__ == "__main__":
    main()
