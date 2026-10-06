#!/usr/bin/env python3
"""Measure how ring-like a recorded harness actually is.

§3.4 retired "peer teams speak in a ring" as a *measured* claim: on the Claude
Code traces the mean recency rank of the next speaker is 0.79 at k=4 and 1.77
at k=7 where a ring would give 3 and 6. That left the ring as an analytical
worst case with no known population. This script tests the other half of the
question -- whether harnesses that serialise speaking by construction produce
the ring on the wire -- against traces from `harness_roundrobin.py` (AutoGen
RoundRobinGroupChat) and `harness_crewai.py` (CrewAI sequential).

Recency rank is defined exactly as the mechanism computes it in
`patch_sglang_cohort.py`: for each turn, the number of *distinct* other
members that spoke since that member's previous turn. A ring gives k-1 every
time; a member speaking twice in a row gives 0.

Members are recovered from the system prompt, which both harnesses make
unique per agent, so the attribution does not depend on prefix chaining.

stdlib only.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict


def load(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows = [r for r in rows if r.get("status") == 200 and r.get("prefix_sha")]
    rows.sort(key=lambda r: r["t_recv"])
    return rows


def member_of(row):
    """Identify the speaker by its system prompt.

    Both harnesses give every agent a distinct system message and send it as
    message 0, so `prefix_sha[0]` is a stable per-member key. Falling back to
    the first message of any role keeps the script usable on traces where the
    harness folds the role into a user turn instead.
    """
    roles = row.get("msg_roles") or []
    chain = row["prefix_sha"]
    if roles and roles[0] == "system":
        return chain[0]
    return chain[0]


def common_prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def analyse(rows, label):
    seq = [member_of(r) for r in rows]
    members = {}
    for m in seq:
        members.setdefault(m, len(members))
    ids = [members[m] for m in seq]
    k = len(members)

    ranks, last_pos = [], {}
    for i, m in enumerate(ids):
        if m in last_pos:
            ranks.append(len(set(ids[last_pos[m] + 1:i])))
        last_pos[m] = i

    print(f"\n=== {label} ===")
    print(f"requests {len(rows)}  members {k}  turns with a prior turn {len(ranks)}")
    print(f"order: {''.join(chr(ord('A') + i) for i in ids)}")
    if not ranks or k < 2:
        print("too few turns to characterise")
        return None

    mean = sum(ranks) / len(ranks)
    norm = mean / (k - 1)
    dist = Counter(ranks)
    print(f"mean recency rank {mean:.2f} of {k - 1}  (normalised {norm:.2f})")
    print("rank distribution: " + "  ".join(
        f"{r}:{dist[r] / len(ranks):.0%}" for r in sorted(dist)))
    print(f"immediate self-repeat (rank 0): {dist[0] / len(ranks):.0%}")
    print(f"perfect-ring turns (rank {k - 1}): {dist[k - 1] / len(ranks):.0%}")

    # What a prefix cache can actually reuse across members, which is the
    # premise that makes residency rather than sharing the binding resource.
    by_member = defaultdict(list)
    for r, m in zip(rows, ids):
        by_member[m].append(r)
    cross = []
    firsts = [by_member[m][0] for m in sorted(by_member)]
    for i in range(len(firsts)):
        for j in range(i + 1, len(firsts)):
            shared = common_prefix_len(firsts[i]["prefix_sha"],
                                       firsts[j]["prefix_sha"])
            cross.append(shared)
    if cross:
        print(f"cross-member shared prompt prefix (messages): "
              f"max {max(cross)}, mean {sum(cross) / len(cross):.2f}")

    pt = [r.get("prompt_tokens") for r in rows if r.get("prompt_tokens")]
    if pt:
        pt_sorted = sorted(pt)
        print(f"prompt tokens: min {pt_sorted[0]}  median "
              f"{pt_sorted[len(pt_sorted) // 2]}  max {pt_sorted[-1]}")
    ct = [r.get("completion_tokens") for r in rows if r.get("completion_tokens")]
    if ct:
        ct_sorted = sorted(ct)
        print(f"completion tokens: median {ct_sorted[len(ct_sorted) // 2]}  "
              f"max {ct_sorted[-1]}")
    fin = Counter(r.get("finish_reason") for r in rows)
    print(f"finish_reason: {dict(fin)}")

    extra = workload_shape(rows, ids, k)
    out = {"label": label, "members": k, "turns": len(ranks),
           "mean_rank": mean, "norm_rank": norm,
           "self_repeat": dist[0] / len(ranks),
           "ring_turns": dist[k - 1] / len(ranks),
           "cross_prefix_max": max(cross) if cross else 0}
    out.update(extra)
    return out


def workload_shape(rows, ids, k):
    """The §3 characterisation numbers, for any harness rather than one client.

    False completion rate and duty cycle are the two measurements that define
    the workload class (§1.3), and until now both were only computable from the
    Claude Code traces -- which is precisely limitation §12.8, since they rest
    on n=2 runs of one task on one model. Both are recoverable from any
    recording-proxy trace, so computing them here is what lets the claim be
    checked at a sample size that means something.

    False completion: a request whose finish reason says the turn ended, whose
    session then speaks again. That is the signal the engine acts on when it
    hands a still-live session to LRU, so it is measured per *session*, not per
    request, and the last turn of each session is excluded -- nothing follows
    it, so it carries no evidence either way.

    Duty cycle: generating time summed over members, divided by the wall span
    times the member count. This is the fraction of co-resident sessions that
    are doing work, i.e. the number that makes residency rather than
    concurrency the binding resource. It is measured on the span between the
    first request and the last response, so a harness that stops early is not
    credited with idleness it never had.
    """
    done_shapes = {"end_turn", "stop", "length"}
    speaks_again, terminal = 0, 0
    for i, (row, member) in enumerate(zip(rows, ids)):
        if (row.get("finish_reason") or "") not in done_shapes:
            continue
        later = any(m == member for m in ids[i + 1:])
        if later:
            speaks_again += 1
        else:
            terminal += 1
    scored = speaks_again + terminal
    false_rate = speaks_again / scored if scored else None

    spans = [(r.get("t_recv"), r.get("t_done")) for r in rows
             if r.get("t_recv") is not None and r.get("t_done") is not None]
    duty = busy = None
    if spans:
        t0 = min(s for s, _ in spans)
        t1 = max(e for _, e in spans)
        wall = t1 - t0
        busy = sum(e - s for s, e in spans)
        duty = busy / (wall * k) if wall > 0 and k else None

    pt = [r.get("prompt_tokens") or 0 for r in rows]
    ct = [r.get("completion_tokens") or 0 for r in rows]
    io_ratio = (sum(pt) / sum(ct)) if sum(ct) else None

    print(f"false completion rate: "
          + (f"{false_rate:.0%} ({speaks_again}/{scored} turns that reported "
             f"a finished turn were followed by the same session speaking "
             f"again)" if false_rate is not None else "no scoreable turns"))
    print(f"duty cycle: "
          + (f"{duty:.3f} ({busy:.0f}s generating over {k} members "
             f"across a {t1 - t0:.0f}s span)" if duty is not None
             else "no timing data"))
    print(f"input:output token ratio: "
          + (f"{io_ratio:.1f}:1" if io_ratio else "no completion tokens"))
    return {"false_completion": false_rate, "duty_cycle": duty,
            "io_ratio": io_ratio}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--json-out")
    cfg = ap.parse_args()

    out = []
    for path in cfg.traces:
        rows = load(path)
        if not rows:
            print(f"\n=== {path} ===\nno completed requests")
            continue
        res = analyse(rows, path.rsplit("/", 1)[-1])
        if res:
            out.append(res)

    if out:
        print("\n=== summary: normalised mean recency rank ===")
        print("  1.00 = perfect ring (LRU's worst case), 0.00 = always self")
        for r in out:
            print(f"  {r['label']:<28} k={r['members']}  "
                  f"{r['norm_rank']:.2f}   ring turns {r['ring_turns']:.0%}")
        print("  reference, measured Claude Code traces:")
        print("    team-run1 (k=4)             0.26")
        print("    team-run2 (k=7)             0.30")

        print("\n=== summary: the §3 characterisation, per trace ===")
        print("  Claude Code reference: false completion 77% (k=4), "
              "duty cycle 0.277, ratio 68-74:1")
        print(f"  {'trace':<28} {'false compl':>12} {'duty':>7} {'ratio':>8}")
        for r in out:
            fc = f"{r['false_completion']:.0%}" if r.get("false_completion") is not None else "-"
            dc = f"{r['duty_cycle']:.3f}" if r.get("duty_cycle") is not None else "-"
            io = f"{r['io_ratio']:.1f}:1" if r.get("io_ratio") else "-"
            print(f"  {r['label']:<28} {fc:>12} {dc:>7} {io:>8}")
    if cfg.json_out and out:
        with open(cfg.json_out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
