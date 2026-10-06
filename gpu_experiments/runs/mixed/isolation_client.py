#!/usr/bin/env python3
"""Same mixed payload as mixed_replay.py; extra dump fields only.

Does not change input_ids construction, member→buffer mapping, or unified.
"""
from __future__ import annotations

import hashlib
import struct
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import mixed_replay as mr  # noqa: E402
from replay_team import post_stream  # noqa: E402


def ids_hash(ids):
    h = hashlib.sha256()
    h.update(struct.pack("<I", len(ids)))
    h.update(struct.pack(f"<{len(ids)}I", *ids))
    return h.hexdigest()


def _post(self, team, member, turn, rid_member, payload, kind):
    t_send = time.monotonic()
    try:
        ttft, total, text_len, meta = post_stream(self.url, payload)
        status = "ok"
        err = None
    except Exception as e:  # noqa: BLE001 — record then re-raise after dump
        row = {
            "team": team, "member": member, "rid_member": rid_member,
            "turn": turn, "rid": payload.get("rid"), "kind": kind,
            "t_send": t_send, "status": "error", "error": type(e).__name__,
            "input_ids_sha256": ids_hash(payload["input_ids"]),
        }
        with self.dump_lock:
            self.dump.append(row)
        raise
    prompt = meta.get("prompt_tokens") or len(payload["input_ids"])
    cached = meta.get("cached_tokens")
    row = {
        "team": team,
        "member": member,
        "rid_member": rid_member,
        "turn": turn,
        "rid": payload["rid"],
        "kind": kind,
        "t_send": t_send,
        "t_first": t_send + ttft,
        "t_done": t_send + total,
        "ttft": ttft,
        "latency": total,
        "cached_tokens": cached,
        "prompt_tokens": prompt,
        "cached": (cached / prompt) if cached else 0.0,
        "completion_tokens": meta.get("completion_tokens"),
        "finish_reason": meta.get("finish_reason"),
        "text_len": text_len,
        "input_ids_sha256": ids_hash(payload["input_ids"]),
        "status": status,
        "error": err,
        "meta_keys": sorted(meta.keys()) if isinstance(meta, dict) else [],
    }
    with self.dump_lock:
        self.dump.append(row)
    return row


if __name__ == "__main__":
    mr.MixedShared._post = _post
    mr.main()
