#!/usr/bin/env python3
"""Closed-loop GPU client with requests.jsonl dump and optional Static-rate rid remap.

Does not change skip predicates. KV content is still seeded by the original
member; only the rid's member field is rewritten when --rank-map is set.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import replay_team as rt  # noqa: E402


_ORIG_REPORT = rt.report
_DUMP = []
_LOCK = threading.Lock()
_RANK = None  # original member -> frequency rank


class RankShared(rt.Shared):
    def speak(self, team, member, turn):
        cfg = self.cfg
        msg = self.messages[(team, member)][turn % cfg.variants]
        rid_member = _RANK[member] if _RANK is not None else member
        payload = {
            "input_ids": self.context[(team, member)] + msg,
            "sampling_params": {"max_new_tokens": cfg.gen_tokens,
                                "temperature": 0},
            "rid": f"{team}:{rid_member}:{turn}",
        }
        t_send = time.monotonic()
        ttft, total, _, meta = rt.post_stream(self.url, payload)
        prompt = meta.get("prompt_tokens") or (
            len(self.context[(team, member)]) + cfg.message)
        cached = meta.get("cached_tokens")
        row = {
            "team": team,
            "member": member,
            "t_send": t_send,
            "t_first": t_send + ttft,
            "t_done": t_send + total,
            "ttft": ttft,
            "latency": total,
            "cached_tokens": cached,
            "cached": (cached / prompt) if cached else 0.0,
            "prompt_tokens": prompt,
            "rid": payload["rid"],
            "rid_member": rid_member,
        }
        with _LOCK:
            _DUMP.append({
                "team": team,
                "member": member,
                "rid_member": rid_member,
                "turn": turn,
                "rid": payload["rid"],
                "t_send": t_send,
                "ttft": ttft,
                "latency": total,
                "cached_tokens": cached,
                "cached": row["cached"],
                "prompt_tokens": prompt,
            })
        return row


def report_and_dump(cfg, teams, wall):
    _ORIG_REPORT(cfg, teams, wall)
    out = Path(cfg.out or f"team_{cfg.label}_{cfg.teams}t_{cfg.topology}.json")
    dump_path = out.with_name(out.stem + ".requests.jsonl")
    with dump_path.open("w", encoding="utf-8") as fh:
        for rec in _DUMP:
            fh.write(json.dumps(rec) + "\n")
    print(f"wrote {dump_path}  n={len(_DUMP)}")


def main():
    global _RANK
    argv = sys.argv[1:]
    rank_path = None
    kept = []
    i = 0
    while i < len(argv):
        if argv[i] == "--rank-map":
            rank_path = argv[i + 1]
            i += 2
            continue
        kept.append(argv[i])
        i += 1
    sys.argv = [sys.argv[0]] + kept
    if rank_path:
        doc = json.loads(Path(rank_path).read_text())
        raw = doc.get("rank_map_member_to_freq_rank") or doc
        _RANK = {int(k): int(v) for k, v in raw.items()}
        print(f"Static-rate rid remap { _RANK } (input_ids still original member)")
    rt.Shared = RankShared
    rt.report = report_and_dump
    rt.main()


if __name__ == "__main__":
    main()
