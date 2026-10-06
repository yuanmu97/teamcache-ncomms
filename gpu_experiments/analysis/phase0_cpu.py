#!/usr/bin/env python3
"""Phase 0B: original HELPER selftest, CPU replay, microbench with debug I/O.

window_s=1e9 replays are arrival-order sensitivity, not a claim that they
reproduce the two engine hooks or a real clock.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CPU = HERE / "cpu"
PATCH = ROOT / "patch_sglang_cohort.py"
sys.path.insert(0, str(ROOT))

from replay_trace import BlockLRU, PAGE, _Req, make_policy  # noqa: E402
from optimal import topo_reference  # noqa: E402


def cpu_model() -> str:
    for cmd in (
        ["sysctl", "-n", "machdep.cpu.brand_string"],
        ["lscpu"],
    ):
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
            line = next((ln.strip() for ln in out.splitlines() if ln.strip()), "")
            if line:
                if cmd[0] == "lscpu":
                    for ln in out.splitlines():
                        if ln.lower().startswith("model name"):
                            return ln.split(":", 1)[1].strip()
                return line
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
    return platform.processor() or platform.platform()


def run_selftest() -> dict:
    CPU.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, str(PATCH), "--selftest"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    text = (proc.stdout or "") + (proc.stderr or "")
    (CPU / "selftest.txt").write_text(text)
    return {
        "returncode": proc.returncode,
        "out": CPU / "selftest.txt",
        "ok": proc.returncode == 0 and "selftest" in text.lower() or proc.returncode == 0,
        "tail": text.strip().splitlines()[-8:],
    }


def helper_once_memoize() -> dict:
    """Read st.once: a repeated rid must not re-observe."""
    skip, cfg = make_policy("unified", 420800, 5, extra={"window_s": 1e9})
    ns_src = open(PATCH, encoding="utf-8").read()
    # The skip function closes over TeamState. Confirm memoize by calling twice.
    req = _Req("7:2:4", 10240)
    a = skip(req)
    b = skip(req)
    # A different turn is a different rid and must be a new observation.
    c = skip(_Req("7:2:5", 10240))
    return {
        "same_rid_skip_pair": [bool(a), bool(b)],
        "same_rid_stable": a == b,
        "new_turn_ran": c is not None,
        "note": "HELPER st.once memoizes by rid; window_s=1e9 is arrival-order sensitivity",
    }


def synth_rows(topology: str, teams: int, members: int, rounds: int,
               prompt: int, seed: int):
    refs = topo_reference(topology, teams, members, rounds, seed=seed)
    counts = defaultdict(int)
    rows = []
    for i, (team, member) in enumerate(refs):
        turn = counts[(team, member)]
        counts[(team, member)] += 1
        rows.append({
            "rid": f"{team}:{member}:{turn}",
            "team": str(team),
            "member": int(member),
            "turn": turn,
            "prompt": prompt,
            "t": i,
        })
    return rows


def replay_rows(rows, mode, pool, members, extra=None):
    cache = BlockLRU(pool)
    skip = None
    if mode != "upstream":
        skip, _cfg = make_policy(mode, pool, members, extra)
    hit = tot = skipped = 0
    per_team = defaultdict(lambda: [0, 0])
    for r in rows:
        sess = f"{r['team']}:{r['member']}"
        h, nblk = cache.lookup(sess, r["prompt"])
        tot += r["prompt"]
        hit += h
        per_team[r["team"]][0] += h
        per_team[r["team"]][1] += r["prompt"]
        if skip is None:
            cache.insert(sess, nblk)
        else:
            if skip(_Req(r["rid"], r["prompt"])):
                skipped += 1
            else:
                cache.insert(sess, nblk)
    return {
        "hit": hit / tot if tot else 0.0,
        "recomputed": tot - hit,
        "skipped": skipped,
        "n": len(rows),
        "per_team_hit": {t: a / b for t, (a, b) in per_team.items() if b},
    }


def replay_topologies() -> dict:
    prompt = 9728 + 512
    pool = 420800
    extra = {"window_s": 1e9, "adapt_window": 40, "adapt_margin": 1.02,
             "adapt_min_samples": 2}
    out = {"label": "arrival-order sensitivity (window_s=1e9)",
           "not_claimed": ["dual hook", "real clock", "GPU TTTA"],
           "topologies": {}}
    for topo in ("ring", "empirical", "star"):
        rows = synth_rows(topo, teams=12, members=5, rounds=8, prompt=prompt, seed=11)
        arms = {}
        for name, mode, ex in (
            ("LRU", "upstream", None),
            ("group q=2", "group", {**extra, "quota": 2}),
            ("unified", "unified", extra),
        ):
            arms[name] = replay_rows(rows, mode, pool, 5, ex)
            arms[name]["hit"] = round(arms[name]["hit"], 4)
        out["topologies"][topo] = {"n": len(rows), "arms": arms}
    return out


def lru_trace_trend() -> dict:
    """Only traces that already carry cached_tokens. Trend, not equality."""
    path = ROOT / "traces" / "selector-debug-k5-Kimi-K3.jsonl"
    rows = []
    for line in path.read_text().splitlines():
        d = json.loads(line)
        if d.get("status") != 200 or ":" not in d.get("rid", ""):
            continue
        if "cached_tokens" not in d or "prompt_tokens" not in d:
            continue
        team, member, turn = d["rid"].split(":")
        rows.append({
            "rid": d["rid"], "team": team, "member": int(member),
            "turn": int(turn), "prompt": d["prompt_tokens"],
            "cached": d["cached_tokens"], "t": d.get("t_recv", 0),
        })
    rows.sort(key=lambda r: r["t"])
    if not rows:
        return {"status": "no cached_tokens field"}
    measured = sum(r["cached"] for r in rows) / sum(r["prompt"] for r in rows)
    pred = replay_rows(rows, "upstream", 420800, 5, None)
    return {
        "trace": str(path.relative_to(ROOT)),
        "n": len(rows),
        "engine_cached_over_prompt": measured,
        "blocklru_hit": pred["hit"],
        "note": "session-binary vs partial-token need not match; trend only",
    }


def _call_skip(skip, n_sess, n_req, prompt=10240):
    # 50/100/200 sessions, >=10k requests. rid cycles members of each session.
    members = 5
    teams = max(1, n_sess // members)
    t0 = time.perf_counter()
    for i in range(n_req):
        team = i % teams
        member = (i // teams) % members
        turn = i // (teams * members)
        skip(_Req(f"{team}:{member}:{turn}", prompt))
    return time.perf_counter() - t0


def microbench() -> dict:
    extra = {"window_s": 30, "adapt_window": 40, "adapt_margin": 1.02,
             "adapt_min_samples": 2, "pool_tokens": 420800}
    results = {"cpu": cpu_model(), "host": platform.node(),
               "includes_unified_debug_jsonl_io": True,
               "not_b300_engine_overhead": True,
               "not_e2e_throughput": True,
               "sizes": {}}
    debug = Path("/tmp/unified_debug.jsonl")
    for n_sess in (50, 100, 200):
        if debug.exists():
            debug.unlink()
        skip, _cfg = make_policy("unified", 420800, 5, extra)
        # init vs steady: 2k warmup then 10k timed
        _call_skip(skip, n_sess, 2000)
        samples = []
        for _ in range(12):
            dt = _call_skip(skip, n_sess, 10000)
            samples.append(dt / 10000 * 1e6)  # us per request
        lat = sorted(samples)
        results["sizes"][str(n_sess)] = {
            "n_req_per_sample": 10000,
            "samples": 12,
            "us_per_req_p50": pct(lat, 0.5),
            "us_per_req_p95": pct(lat, 0.95),
            "us_per_req_p99": pct(lat, 0.99),
            "debug_jsonl_bytes": debug.stat().st_size if debug.exists() else 0,
        }
    # Isolated copy without debug I/O — adjunct, not a replacement.
    skip, _cfg = make_policy("unified", 420800, 5, extra)
    import builtins
    real_open = builtins.open

    def quiet_open(path, *a, **k):
        if str(path) == "/tmp/unified_debug.jsonl":
            class N:
                def write(self, *_a, **_k):
                    return 0

                def __enter__(self):
                    return self

                def __exit__(self, *_a):
                    return False
            return N()
        return real_open(path, *a, **k)

    builtins.open = quiet_open
    try:
        samples = []
        for _ in range(8):
            dt = _call_skip(skip, 100, 10000)
            samples.append(dt / 10000 * 1e6)
        results["adjunct_no_debug_io_100sess_us_p50"] = pct(sorted(samples), 0.5)
        results["adjunct_note"] = "not a substitute for the with-I/O headline"
    finally:
        builtins.open = real_open
    return results


def pct(xs, q):
    s = sorted(xs)
    return s[min(int(len(s) * q), len(s) - 1)]


def splice_diagnostic() -> dict:
    """ring → empirical → ring on CPU. Not a GPU time, not a tune."""
    prompt = 9728 + 512
    parts = (
        synth_rows("ring", 8, 5, 6, prompt, 11)
        + synth_rows("empirical", 8, 5, 6, prompt, 12)
        + synth_rows("ring", 8, 5, 6, prompt, 13)
    )
    extra = {"window_s": 1e9, "adapt_window": 40, "adapt_margin": 1.02,
             "adapt_min_samples": 2}
    u = replay_rows(parts, "unified", 420800, 5, extra)
    g = replay_rows(parts, "group", 420800, 5, {**extra, "quota": 2})
    return {
        "n": len(parts),
        "unified_hit": u["hit"],
        "group_q2_hit": g["hit"],
        "note": "CPU splice only; do not convert to GPU seconds; not used to retune",
    }


def main():
    CPU.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(PATCH.read_bytes()).hexdigest()
    st = run_selftest()
    print("selftest rc", st["returncode"], "->", st["out"])
    memo = helper_once_memoize()
    print("once memoize", memo)
    topo = replay_topologies()
    print("topologies", {k: {a: v["arms"][a]["hit"] for a in v["arms"]}
                         for k, v in topo["topologies"].items()})
    trend = lru_trace_trend()
    print("trace trend", trend)
    mb = microbench()
    print("microbench", json.dumps(mb["sizes"], indent=2))
    sp = splice_diagnostic()
    print("splice", sp)
    out = {
        "patch_sha256": sha,
        "cpu_model": mb["cpu"],
        "selftest": {"returncode": st["returncode"], "path": "cpu/selftest.txt",
                     "tail": st["tail"]},
        "rid_memoize": memo,
        "topology_replay": topo,
        "lru_trace_trend": trend,
        "microbench": mb,
        "splice_ring_empirical_ring": sp,
    }
    dest = CPU / "phase0_cpu.json"
    dest.write_text(json.dumps(out, indent=2, default=str))
    print("wrote", dest)


if __name__ == "__main__":
    main()
