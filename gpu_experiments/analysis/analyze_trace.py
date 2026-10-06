#!/usr/bin/env python3
"""Turn a proxy trace into the quantities the peer-agent-team analysis needs.

The proxy sees only HTTP requests -- it has no session identifier, because the
Messages API does not carry one. That absence is itself part of the research
argument, so instead of asking the harness for session ids we recover sessions
the same way an inference engine could: by matching prompt prefixes.

Each trace row carries a cumulative prefix hash chain taken at message
boundaries. Two requests from the same agent share a long chain; requests from
different agents diverge at the system prompt or the first message. That gives
us, for free:

  * session recovery       -> who issued what
  * duty cycle rho         -> how many members generate at once
  * the resend tax         -> input tokens vs. genuinely new tokens
  * false completions      -> `end_turn` replies that were not the end

Usage:  analyze_trace.py traces/team-run1.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict


def load(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["t_recv"])
    return rows


def common_prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def recover_sessions(rows):
    """Greedy: attach each request to the session whose last request shares the
    longest prompt prefix, as long as the whole of that request's prompt is a
    prefix of the new one (i.e. the conversation only ever grows)."""
    sessions = []  # list of dicts: {"rows": [...], "chain": last prefix_sha}
    for row in rows:
        chain = row["prefix_sha"]
        best, best_len = None, 0
        for sess in sessions:
            prev = sess["chain"]
            shared = common_prefix_len(prev, chain)
            # the previous turn must be fully contained in this one
            if shared == len(prev) and shared > best_len:
                best, best_len = sess, shared
        if best is None:
            best = {"rows": [], "chain": chain}
            sessions.append(best)
        best["rows"].append(row)
        best["chain"] = chain
    return sessions


def busy_intervals(rows):
    return [(r["t_recv"], r.get("t_done") or r["t_recv"]) for r in rows]


def concurrency_profile(rows):
    """Fraction of wall time with exactly n requests in flight."""
    events = []
    for start, end in busy_intervals(rows):
        events.append((start, 1))
        events.append((end, -1))
    events.sort()
    time_at = defaultdict(float)
    cur, prev_t = 0, events[0][0]
    for t, delta in events:
        time_at[cur] += t - prev_t
        cur += delta
        prev_t = t
    return time_at


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--json-out", default="")
    cfg = ap.parse_args(argv)

    rows = load(cfg.trace)
    if not rows:
        sys.exit("empty trace")

    t0 = rows[0]["t_recv"]
    t1 = max(r.get("t_done") or r["t_recv"] for r in rows)
    span = t1 - t0

    tok_in = sum((r.get("usage") or {}).get("input_tokens", 0) for r in rows)
    tok_out = sum((r.get("usage") or {}).get("output_tokens", 0) for r in rows)

    sessions = recover_sessions(rows)
    sessions.sort(key=lambda s: s["rows"][0]["t_recv"])

    print(f"trace           {cfg.trace}")
    print(f"requests        {len(rows)}")
    print(f"wall span       {span:.1f} s")
    print(f"input tokens    {tok_in:,}")
    print(f"output tokens   {tok_out:,}")
    print(f"in:out ratio    {tok_in / max(tok_out, 1):.1f} : 1")
    print(f"stop reasons    {dict(Counter(r.get('stop_reason') for r in rows))}")
    print()

    # ---- recovered sessions ----
    print(f"recovered sessions: {len(sessions)}")
    print(f"{'#':>2} {'reqs':>5} {'sys_ch':>7} {'first_t':>8} {'last_t':>8} "
          f"{'in_tok':>9} {'out_tok':>8} {'peak_in':>8}")
    for i, sess in enumerate(sessions):
        srows = sess["rows"]
        ins = [(r.get("usage") or {}).get("input_tokens", 0) for r in srows]
        outs = [(r.get("usage") or {}).get("output_tokens", 0) for r in srows]
        print(f"{i:>2} {len(srows):>5} {srows[0]['system_chars']:>7} "
              f"{srows[0]['t_recv'] - t0:>8.2f} "
              f"{(srows[-1].get('t_done') or srows[-1]['t_recv']) - t0:>8.2f} "
              f"{sum(ins):>9,} {sum(outs):>8,} {max(ins):>8,}")
    print()

    # ---- duty cycle ----
    prof = concurrency_profile(rows)
    busy = span - prof.get(0, 0.0)
    members = len(sessions)
    weighted = sum(n * t for n, t in prof.items())
    print("concurrency profile (fraction of wall time with n requests in flight)")
    for n in sorted(prof):
        print(f"  n={n}: {prof[n] / span * 100:5.1f}%  ({prof[n]:6.2f} s)")
    print(f"mean in-flight  {weighted / span:.2f}")
    print(f"any-busy        {busy / span * 100:.1f}% of wall time")
    if members:
        print(f"duty cycle rho  {weighted / span / members:.3f} "
              f"(mean in-flight / {members} members)")
        print(f"1/rho headroom  {members * span / max(weighted, 1e-9):.1f}x")
    print()

    # ---- the resend tax ----
    # Within a session, consecutive turns share a prompt prefix. The genuinely
    # new content is the tail. Approximate the token split by characters, which
    # the prompt_chars field gives us exactly.
    resent = 0
    fresh = 0
    for sess in sessions:
        prev_chars = 0
        for r in sess["rows"]:
            resent += min(prev_chars, r["prompt_chars"])
            fresh += max(r["prompt_chars"] - prev_chars, 0)
            prev_chars = r["prompt_chars"]
    total_chars = resent + fresh
    print("resend tax (prompt characters)")
    print(f"  re-sent       {resent:,} ({resent / max(total_chars,1) * 100:.1f}%)")
    print(f"  genuinely new {fresh:,} ({fresh / max(total_chars,1) * 100:.1f}%)")
    print(f"  => {resent / max(fresh,1):.1f}x more prompt is re-sent than is new")
    print()

    # ---- cross-member prefix sharing ----
    # For each request, the longest prompt prefix (in messages) it shares with
    # any earlier request from a *different* recovered session. This is the
    # ceiling on what a prefix cache can reuse across team members.
    owner = {}
    for i, sess in enumerate(sessions):
        for r in sess["rows"]:
            owner[r["seq"]] = i
    cross = []
    for idx, row in enumerate(rows):
        best = 0
        for earlier in rows[:idx]:
            if owner[earlier["seq"]] == owner[row["seq"]]:
                continue
            best = max(best, common_prefix_len(earlier["prefix_sha"], row["prefix_sha"]))
        cross.append(best)
    shared_sys = sum(1 for c in cross if c >= 1)
    print("cross-member prefix sharing (message boundaries shared with another member)")
    print(f"  requests sharing at least the system prompt: {shared_sys}/{len(rows)}")
    print(f"  distribution: {dict(Counter(cross))}")
    print()

    # ---- false completions ----
    # An `end_turn` is a *false* completion when the same session issues another
    # request later: the member was pausing, not finishing, and the engine had
    # no way to tell.
    false_end, true_end = 0, 0
    for sess in sessions:
        for j, r in enumerate(sess["rows"]):
            if r.get("stop_reason") != "end_turn":
                continue
            if j < len(sess["rows"]) - 1:
                false_end += 1
            else:
                true_end += 1
    print("protocol ambiguity")
    print(f"  end_turn replies         {false_end + true_end}")
    print(f"  ... later resumed        {false_end}  <- peer wait, looked like completion")
    print(f"  ... genuinely final      {true_end}")
    if false_end + true_end:
        print(f"  false-completion rate    {false_end / (false_end + true_end) * 100:.0f}%")

    if cfg.json_out:
        out = {
            "requests": len(rows),
            "span_s": span,
            "input_tokens": tok_in,
            "output_tokens": tok_out,
            "sessions": len(sessions),
            "mean_inflight": weighted / span,
            "rho": weighted / span / members if members else None,
            "resent_chars": resent,
            "fresh_chars": fresh,
            "false_end_turn": false_end,
            "true_end_turn": true_end,
            "concurrency_profile": {str(k): v for k, v in prof.items()},
        }
        with open(cfg.json_out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {cfg.json_out}")


if __name__ == "__main__":
    main()
