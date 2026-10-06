#!/usr/bin/env python3
"""Closed-loop mixed fleet: half ring order, half Selector-debug speaking order.

Not live autonomous. Same post_stream + barrier/stagger/reset as replay_team.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import replay_team as rt  # noqa: E402
from replay_team import Shared, post_stream, pct, speaker_order  # noqa: E402

TRACE = ROOT / "traces" / "selector-debug-k5-Kimi-K3.jsonl"


def load_selector_steps():
    rows = []
    selector_calls = 0
    for line in TRACE.read_text().splitlines():
        d = json.loads(line)
        hdr = d.get("headers") or {}
        if d.get("status") != 200:
            continue
        if "x-team" not in hdr:
            selector_calls += 1
            continue
        rows.append(d)
    members = sorted({r["member"] for r in rows})
    remap = {m: i for i, m in enumerate(members)}
    steps = []
    turns = defaultdict(int)
    for r in rows:
        m = remap[r["member"]]
        steps.append({
            "member": m,
            "orig_member": r["member"],
            "turn": turns[m],
            "prompt_tokens": r["prompt_tokens"],
        })
        turns[m] += 1
    return steps, remap, selector_calls


class MixedShared(Shared):
    def __init__(self, cfg, rank_maps=None):
        super().__init__(cfg)
        self.rank_maps = rank_maps or {}
        self.sel_buf = {}
        self.dump = []
        self.dump_lock = threading.Lock()

    def ids(self, team, member, ntok):
        key = (team, member)
        buf = self.sel_buf.get(key)
        if buf is None or len(buf) < ntok:
            rng = random.Random((team << 16) ^ member ^ self.cfg.seed ^ 0x5E1)
            need = max(ntok, 16384)
            self.sel_buf[key] = [rng.randrange(16, self.cfg.vocab) for _ in range(need)]
        return self.sel_buf[key][:ntok]

    def speak_ring(self, team, member, turn):
        cfg = self.cfg
        msg = self.messages[(team, member)][turn % cfg.variants]
        rid_member = self.rank_maps.get(team, {}).get(member, member)
        payload = {
            "input_ids": self.context[(team, member)] + msg,
            "sampling_params": {"max_new_tokens": cfg.gen_tokens, "temperature": 0},
            "rid": f"{team}:{rid_member}:{turn}",
        }
        return self._post(team, member, turn, rid_member, payload, "ring")

    def speak_selector(self, team, member, turn, ntok):
        rid_member = self.rank_maps.get(team, {}).get(member, member)
        payload = {
            "input_ids": self.ids(team, member, ntok),
            "sampling_params": {"max_new_tokens": self.cfg.gen_tokens, "temperature": 0},
            "rid": f"{team}:{rid_member}:{turn}",
        }
        return self._post(team, member, turn, rid_member, payload, "selector")

    def _post(self, team, member, turn, rid_member, payload, kind):
        t_send = time.monotonic()
        ttft, total, _, meta = post_stream(self.url, payload)
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
        }
        with self.dump_lock:
            self.dump.append(row)
        return row


class RingTeam(rt.Team):
    def step(self, record, delivered):
        member = next(self.order)
        turn = self.turn[member]
        self.turn[member] += 1
        row = self.shared.speak_ring(self.tid, member, turn)
        row["wtft"] = row["t_first"] - delivered
        if record:
            with self.shared.lock:
                self.steps.append(row)
        return row["t_done"]


class SelectorTeam:
    def __init__(self, tid, cfg, shared, trace_steps):
        self.tid = tid
        self.cfg = cfg
        self.shared = shared
        self.trace_steps = trace_steps
        self.steps = []
        self.t_start = None
        self.t_end = None
        self.kind = "selector"

    def run(self):
        cfg = self.cfg
        time.sleep(self.tid * cfg.stagger)
        delivered = time.monotonic()
        warmup_n = cfg.members * cfg.warmup_rounds
        for i, spec in enumerate(self.trace_steps[:warmup_n]):
            if self.shared.stop.is_set():
                break
            row = self.shared.speak_selector(
                self.tid, spec["member"], spec["turn"], spec["prompt_tokens"])
            delivered = row["t_done"]
        self.shared.barrier.wait()
        self.t_start = delivered = time.monotonic()
        for spec in self.trace_steps[warmup_n:]:
            if self.shared.stop.is_set():
                break
            row = self.shared.speak_selector(
                self.tid, spec["member"], spec["turn"], spec["prompt_tokens"])
            row["wtft"] = row["t_first"] - delivered
            row["orig_member"] = spec["orig_member"]
            with self.shared.lock:
                self.steps.append(row)
            delivered = row["t_done"]
        self.t_end = time.monotonic()


def report(cfg, teams, wall, extra):
    rows = [r for t in teams for r in t.steps]
    if not rows:
        raise SystemExit("no steps completed")
    miss = [r for r in rows if r["cached"] < 0.5]
    per_team = []
    for t in teams:
        if not t.steps:
            continue
        span = (t.t_end or time.monotonic()) - t.t_start
        misses = sum(1 for r in t.steps if r["cached"] < 0.5)
        reuse = 0
        prompt = 0
        for r in t.steps:
            reuse += r["cached_tokens"] or 0
            prompt += r["prompt_tokens"] or 0
        per_team.append({
            "team": t.tid,
            "kind": getattr(t, "kind", getattr(t, "topology", "?")),
            "steps": len(t.steps),
            "ttta": span,
            "miss_rate": misses / len(t.steps),
            "prefix_reuse": reuse / prompt if prompt else None,
            "prefix_tokens_cached": reuse,
            "prefix_tokens_prompt": prompt,
        })
    kinds = defaultdict(list)
    for x in per_team:
        kinds[x["kind"]].append(x["ttta"])

    def kind_stats(xs):
        if not xs:
            return None
        return {"n": len(xs), "median": statistics.median(xs), "max": max(xs),
                "all": xs}

    prompt_sum = sum(r["prompt_tokens"] or 0 for r in rows)
    cached_sum = sum(r["cached_tokens"] or 0 for r in rows)
    out = {
        "label": cfg.label,
        "topology": "mixed-ring-selector",
        "closed_loop_trace_replay": True,
        "live_autonomous": False,
        "teams": cfg.teams,
        "members": cfg.members,
        "capacity": cfg.capacity,
        "wall_s": wall,
        "steps": len(rows),
        "turn_miss_rate": len(miss) / len(rows),
        "request_hit": 1.0 - len(miss) / len(rows),
        "prefix_token_reuse": cached_sum / prompt_sum if prompt_sum else None,
        "prefix_tokens_cached": cached_sum,
        "prefix_tokens_prompt": prompt_sum,
        "low_reuse_teams": sum(1 for x in per_team if x["miss_rate"] >= 0.9),
        "ttta_median": statistics.median(x["ttta"] for x in per_team),
        "ttta_max": max(x["ttta"] for x in per_team),
        "by_kind_ttta": {k: kind_stats(v) for k, v in kinds.items()},
        "per_team": per_team,
        "extra": extra,
    }
    path = Path(cfg.out)
    path.write_text(json.dumps(out, indent=2))
    dump = path.with_name(path.stem + ".requests.jsonl")
    with dump.open("w") as fh:
        for rec in teams[0].shared.dump:
            slim = {k: rec[k] for k in rec if k not in ("t_first", "t_done")}
            fh.write(json.dumps(slim, default=str) + "\n")
    print(f"wrote {path} wall={wall:.1f}s hit={out['request_hit']:.3f} "
          f"prefix={out['prefix_token_reuse']}")
    print(f"wrote {dump}")


def load_rank_maps(path, n_teams):
    if not path:
        return {}
    doc = json.loads(Path(path).read_text())
    raw = doc.get("per_team_rank_map") or {}
    maps = {}
    for t, mp in raw.items():
        maps[int(t)] = {int(k): int(v) for k, v in mp.items()}
    return maps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--teams", type=int, default=12)
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--context", type=int, default=9728)
    ap.add_argument("--message", type=int, default=512)
    ap.add_argument("--variants", type=int, default=4)
    ap.add_argument("--gen-tokens", type=int, default=144)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--warmup-rounds", type=int, default=2)
    ap.add_argument("--stagger", type=float, default=2.0)
    ap.add_argument("--capacity", type=int, required=True)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--reset-multiple", type=float, default=1.2)
    ap.add_argument("--vocab", type=int, default=100000)
    ap.add_argument("--usd-per-hour", type=float, default=112.32)
    ap.add_argument("--slo-ttta", type=float, default=120.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank-map", default="")
    ap.add_argument("--settle-rounds", type=int, default=0)
    cfg = ap.parse_args()
    # RingTeam reuses replay_team.Team, which keys speaker_order off topology.
    cfg.topology = "ring"
    cfg.team_contexts = [cfg.context] * cfg.teams
    if cfg.teams % 2:
        raise SystemExit("mixed fleet needs an even team count")

    steps, remap, n_sel = load_selector_steps()
    n_half = cfg.teams // 2
    rank_maps = load_rank_maps(cfg.rank_map, cfg.teams)
    shared = MixedShared(cfg, rank_maps=rank_maps)
    print(f"label={cfg.label} mixed {n_half} ring + {n_half} selector-replay  "
          f"selector_keep_steps={len(steps)} selector_calls_excluded={n_sel}")
    print(f"member remap {remap}  rank_maps={bool(rank_maps)}")
    if cfg.reset_multiple > 0:
        shared.reset_cache()

    teams = []
    for tid in range(cfg.teams):
        if tid < n_half:
            t = RingTeam(tid, cfg, shared)
            t.kind = "ring"
            teams.append(t)
        else:
            teams.append(SelectorTeam(tid, cfg, shared, steps))

    threads = [threading.Thread(target=t.run, daemon=True) for t in teams]
    for th in threads:
        th.start()
    tw = time.monotonic()
    shared.barrier.wait()
    print(f"warm-up done in {time.monotonic() - tw:.0f}s", flush=True)
    t0 = time.monotonic()
    target = sum(len(t.steps) for t in teams)  # 0 during measured; poll dump
    while any(th.is_alive() for th in threads):
        time.sleep(15.0)
        with shared.lock:
            now = sum(len(t.steps) for t in teams)
            recent = [r["cached"] for t in teams for r in t.steps[-6:]]
        hit = (sum(recent) / len(recent) * 100) if recent else float("nan")
        print(f"  {now} measured steps  {time.monotonic() - t0:.0f}s  "
              f"cached(recent) {hit:.1f}%", flush=True)
    for th in threads:
        th.join()
    wall = max((t.t_end or t0) for t in teams) - t0
    extra = {
        "selector_trace": str(TRACE.relative_to(ROOT)),
        "selector_remap": {str(k): v for k, v in remap.items()},
        "selector_calls_excluded": n_sel,
        "seed": cfg.seed,
    }
    report(cfg, teams, wall, extra)


if __name__ == "__main__":
    main()
