#!/usr/bin/env python3
"""Closed-loop peer-team replay: teams progress, not reference strings.

WHY THIS EXISTS. `replay_sglang.py` sends a pre-generated reference string
sequentially with `max_new_tokens=1`. That is a clean caching experiment and it
established the mechanism, but it cannot measure the thing the mechanism is
actually claimed to improve. Its central result is that aggregate cost cannot
distinguish a group-blind policy from a group-aware one -- measured, the
group-blind arm was 0.9% *faster* while pushing its worst team from 58.6% to
84.8% misses. If aggregate numbers cannot tell them apart, the metric has to be
team-level, and a team-level metric needs a team that can actually stall.

So here a team is a state machine, not a list of references:

  - A member is eligible only when its inbox is non-empty, so at most one
    member of a ring team is runnable at a time. Waking is point to point,
    matching the measured harness semantics, which has no broadcast primitive.
  - The next collaboration step cannot start until the previous member's output
    exists. An eviction therefore delays the whole team rather than one
    request, which is what makes per-team miss rate translate into team
    makespan.
  - Teams run concurrently, so prefill and decode interfere the way they do in
    service, and both the soft resource (cache tokens) and the hard one
    (running-request slots) are under pressure at once.
  - Generation is real. The measured input:output ratio is 68-74:1, so output
    length is derived from context length instead of being pinned to 1 token.
    This dilutes the speedup relative to the prefill-only experiment, and that
    dilution is a result, not something to hide.

Four metrics come out that the sequential replay cannot produce:

  GPR   collaboration steps per second, per team and aggregate
  CST   fraction of wall clock a team spent with an unread message and nobody
        generating -- "could have progressed and did not"
  WTFT  message delivery to that member's first token
  TTTA  team start to final answer

Both arms issue byte-identical request streams and differ only in the engine's
policy, which is read from a control file, so an arm switch needs no restart.

  python3 replay_team.py --label lru       --teams 8 --capacity 24
  python3 replay_team.py --label cohort-q3 --teams 8 --capacity 24
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import threading
import time
import urllib.request
from collections import defaultdict


def post_stream(url, payload, timeout=1800.0):
    """Send one generation and return (ttft, total, text_len, meta).

    Streaming is not a detail here: WTFT is a headline metric, and without the
    first-chunk timestamp it cannot be separated from total latency.
    """
    body = dict(payload)
    body["stream"] = True
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"content-type": "application/json"})
    t0 = time.monotonic()
    ttft = None
    meta = {}
    text = ""
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                doc = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if ttft is None:
                ttft = time.monotonic() - t0
            if isinstance(doc, dict):
                meta = doc.get("meta_info") or meta
                text = doc.get("text", text)
    total = time.monotonic() - t0
    return (ttft if ttft is not None else total), total, len(text), meta


# Maximum-likelihood fit of the recency-rank distribution over both measured
# team traces, done jointly so that each trace's own k truncates the support
# correctly (run1 has k=4 and simply cannot produce rank 4). See
# `analyze_trace.py` for the rank extraction. lambda=0 would be a uniformly
# random speaker; lambda->inf would be one member monologuing forever.
EMPIRICAL_LAMBDA = 0.3722


def speaker_order(topology, members, seed=0):
    """Who speaks next, as a function of the topology.

    `empirical` is the measured shape and should be the default for any claim
    about real behaviour. It is *not* a ring: on the two traces the mean
    recency rank of the next speaker is 0.79 (k=4) and 1.77 (k=7) against a
    ring's 3 and 6, and 48%/22% of turns are the same member speaking again.
    Real peer teams have better locality than uniform random, let alone a ring.

    Ring is the analytical worst case, kept as a control: it is the reference
    string on which LRU provably evicts exactly the next speaker, which is what
    §5.1's bound is derived on. Reporting only ring would overstate the problem.

    Star puts one synthesiser at the centre, so the hub is referenced every
    other step while a spoke waits 2(k-1) steps. That makes reuse distances
    non-uniform *within* a team, which is the case where "where the misses
    land" stops being only a fairness question and starts affecting aggregate
    cost too.
    """
    if topology == "ring":
        while True:
            for m in range(members):
                yield m
    elif topology == "star":
        while True:
            for m in range(1, members):
                yield 0
                yield m
    elif topology == "empirical":
        rng = random.Random(seed)
        weights = [math.exp(-EMPIRICAL_LAMBDA * r) for r in range(members)]
        stack = list(range(members))
        # Warm every member once, so the fitted distribution is not asked to
        # describe a stack it has not filled yet.
        for m in stack:
            yield m
        while True:
            rank = rng.choices(range(members), weights=weights)[0]
            m = stack.pop(rank)
            stack.insert(0, m)
            yield m
    else:
        raise SystemExit(f"unknown topology {topology!r}")


MIXED_KINDS = ("ring", "empirical", "star")


def team_topology(topology, tid):
    """Which order one team follows, so a node can serve several at once.

    `mixed` is the configuration a deployed node is actually in: the speaking
    order is a property of the harness, not of peer collaboration, and a
    cluster runs more than one harness. It is also the only case where no
    fixed policy is defensible -- LRU is right for the free-form teams and
    ruinous for the serialised ones, and a quota is the reverse -- so it is
    the honest setting for a policy that claims to need no configuration.
    """
    if topology != "mixed":
        return topology
    return MIXED_KINDS[tid % len(MIXED_KINDS)]


class Shared:
    """Engine access plus the per-session content both arms must share."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.url = f"http://{cfg.host}:{cfg.port}/generate"
        self.lock = threading.Lock()
        self.stop = threading.Event()
        # Teams plus the main thread, so the measured phase starts from one
        # clock for everyone rather than from whenever each team happened to
        # finish warming up.
        self.barrier = threading.Barrier(cfg.teams + 1)
        self.flushed = threading.Event()
        self.settled = threading.Barrier(cfg.teams + 1)
        # Per-team private histories. A deployed node is not 48 copies of one
        # 3k-preamble ring: it is a handful of concurrent teams whose contexts
        # already look like the collected traces (AutoGen design ~40–65k,
        # Selector/Claude Code ~8–16k). Uniform `cfg.context` stays the default
        # so older sweeps do not change.
        lengths = getattr(cfg, "team_contexts", None) or [cfg.context] * cfg.teams
        self.context = {
            (t, m): [random.Random((t << 16) ^ m ^ cfg.seed)
                     .randrange(16, cfg.vocab) for _ in range(lengths[t])]
            for t in range(cfg.teams) for m in range(cfg.members)
        }
        # A turn must strictly extend what the engine stored or it is not
        # served from cache at all: measured, a byte-identical re-request
        # recomputes everything (0.694 s against 0.704 s cold, zero cached
        # tokens) while the same prefix plus 512 tokens reads in 0.107 s. So
        # each turn appends an arriving peer message. Drawing that message
        # fresh every time would grow an unbounded fan of sibling branches
        # under the shared prefix -- 31% of the pool went to dead branches in
        # one run -- so messages cycle through a fixed set, leaving each
        # session a bounded and equal footprint.
        self.messages = {
            (t, m): [[random.Random((t << 24) ^ (m << 8) ^ v ^ cfg.seed)
                      .randrange(16, cfg.vocab) for _ in range(cfg.message)]
                     for v in range(cfg.variants)]
            for t in range(cfg.teams) for m in range(cfg.members)
        }

    def speak(self, team, member, turn):
        cfg = self.cfg
        msg = self.messages[(team, member)][turn % cfg.variants]
        payload = {
            "input_ids": self.context[(team, member)] + msg,
            "sampling_params": {"max_new_tokens": cfg.gen_tokens,
                                "temperature": 0},
            "rid": f"{team}:{member}:{turn}",
        }
        t_send = time.monotonic()
        ttft, total, _, meta = post_stream(self.url, payload)
        prompt = meta.get("prompt_tokens") or (
            len(self.context[(team, member)]) + cfg.message)
        cached = meta.get("cached_tokens")
        return {
            "team": team,
            "member": member,
            "t_send": t_send,
            "t_first": t_send + ttft,
            "t_done": t_send + total,
            "ttft": ttft,
            "latency": total,
            "cached_tokens": cached,
            "cached": (cached / prompt) if cached else 0.0,
        }

    def reset_cache(self):
        """Flush the pool with content unique to this label.

        Sharing the flush content across arms would make the second reset a run
        of cache hits that evicts nothing, leaving the previous arm's sessions
        warm and quietly faking the result.
        """
        cfg = self.cfg
        n = int(cfg.capacity * cfg.reset_multiple)
        print(f"resetting cache with {n} unique {cfg.context:,}-token "
              f"contexts...", flush=True)
        t0 = time.monotonic()
        rng = random.Random(0xC0FFEE + sum(ord(c) for c in cfg.label))
        for _ in range(n):
            post_stream(self.url, {
                "input_ids": [rng.randrange(16, cfg.vocab)
                              for _ in range(cfg.context)],
                "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            })
        print(f"reset done in {time.monotonic() - t0:.0f}s\n", flush=True)

    def evict_with_filler(self):
        """Evict measuring leftovers without resetting in-engine policy state.

        `/flush_cache` also drops `_CohortState`, so the next requests look
        like a cold start and the rule admits everyone again. Unique
        no-team-id prompts take the official insert path, turn the pool
        over, and leave `want` in place. Settle rounds then plant only the
        keep-set.
        """
        print("evicting leftovers with unlabelled filler...", flush=True)
        self.reset_cache()


class Team:
    """One peer team: a mailbox, a speaking order, and a closed loop."""

    def __init__(self, tid, cfg, shared):
        self.tid = tid
        self.cfg = cfg
        self.shared = shared
        self.topology = team_topology(cfg.topology, self.tid)
        self.order = speaker_order(self.topology, cfg.members,
                                   seed=(cfg.seed << 8) ^ self.tid)
        self.turn = defaultdict(int)
        self.steps = []
        self.t_start = None
        self.t_end = None

    def step(self, record, delivered):
        member = next(self.order)
        turn = self.turn[member]
        self.turn[member] += 1
        row = self.shared.speak(self.tid, member, turn)
        # Wake to first token: the message was already in this member's inbox
        # at `delivered`, so everything after that point is time the team had
        # something to do and was not doing it.
        row["wtft"] = row["t_first"] - delivered
        row["member"] = member
        row["turn"] = turn
        if record:
            with self.shared.lock:
                self.steps.append(row)
        return row["t_done"]

    def run(self):
        cfg = self.cfg
        # Teams do not arrive in lockstep, and the difference is not cosmetic.
        # A group-blind admission policy freezes in whoever arrived first, so
        # if every team's members interleave perfectly it admits the same first
        # q members of every team and becomes indistinguishable from a per-team
        # quota -- measured, the two arms landed on the same sessions. Staggered
        # arrival is both what actually happens, since a lead spawns teammates
        # over time and fleets arrive independently, and the condition under
        # which the two policies are different objects at all.
        time.sleep(self.tid * cfg.stagger)

        # Warm-up is not padding. SGLang splits a radix node when a *second*
        # sequence is inserted under it, not when one is matched, so the first
        # two distinct turns of a session miss even on an empty pool -- and a
        # request byte-identical to a stored path misses outright. Measuring
        # from turn zero would therefore fold a fixed cold-start transient into
        # every arm and compress the difference between them.
        delivered = time.monotonic()
        for _ in range(cfg.members * cfg.warmup_rounds):
            if self.shared.stop.is_set():
                break
            delivered = self.step(False, delivered)
        self.shared.barrier.wait()
        if cfg.settle_rounds:
            self.shared.flushed.wait()
            for _ in range(cfg.members * cfg.settle_rounds):
                if self.shared.stop.is_set():
                    break
                delivered = self.step(False, delivered)
            self.shared.settled.wait()

        # The task arrives in the first speaker's inbox. From then on every
        # step is woken by the previous member's output, so the loop is closed
        # and an eviction delays the whole team rather than one request.
        self.t_start = delivered = time.monotonic()
        for _ in range(cfg.members * cfg.rounds):
            if self.shared.stop.is_set():
                break
            delivered = self.step(True, delivered)
        self.t_end = time.monotonic()


def pct(xs, q):
    s = sorted(xs)
    return s[min(int(len(s) * q), len(s) - 1)]


def report(cfg, teams, wall):
    rows = [r for t in teams for r in t.steps]
    if not rows:
        raise SystemExit("no steps completed")

    miss = [r for r in rows if r["cached"] < 0.5]
    ttft = [r["ttft"] for r in rows]
    wtft = [r["wtft"] for r in rows]

    per_team = []
    for t in teams:
        if not t.steps:
            continue
        span = (t.t_end or time.monotonic()) - t.t_start
        stall = sum(r["wtft"] for r in t.steps)
        misses = sum(1 for r in t.steps if r["cached"] < 0.5)
        per_team.append({
            "team": t.tid,
            "steps": len(t.steps),
            "ttta": span,
            "gpr": len(t.steps) / span if span > 0 else 0.0,
            "cst": min(stall / span, 1.0) if span > 0 else 0.0,
            "miss_rate": misses / len(t.steps),
            "wtft_p50": statistics.median(r["wtft"] for r in t.steps),
            "wtft_p95": pct([r["wtft"] for r in t.steps], 0.95),
        })

    # Per-member hit rate, which is what tells a working mechanism from a
    # defeated one. Cohort keeps members 0..q-1 and bypasses the rest, so if it
    # is working the hit rate is a step function of the member index. A flat
    # row of misses instead means the retained members are being evicted
    # anyway, and no quota setting will fix that.
    per_member = []
    for m in range(cfg.members):
        got = [r for r in rows if r.get("member") == m]
        if not got:
            continue
        per_member.append({
            "member": m,
            "steps": len(got),
            "miss_rate": sum(1 for r in got if r["cached"] < 0.5) / len(got),
            "mean_cached": statistics.mean(r["cached"] for r in got),
        })

    ttta = [x["ttta"] for x in per_team]
    # teams/node at a team SLO. The SLO is a TTTA budget; a team meets it if it
    # finished its rounds within the budget. Reported for a spread of budgets
    # because the interesting comparison is the shape, not one threshold.
    slo_rows = []
    for mult in (1.0, 1.25, 1.5, 2.0):
        budget = cfg.slo_ttta * mult
        met = sum(1 for x in per_team if x["ttta"] <= budget)
        slo_rows.append((budget, met, met / len(per_team)))

    usd_per_team_task = cfg.usd_per_hour * (wall / 3600.0) / len(per_team)

    print(f"\n--- {cfg.label} ---")
    print(f"topology / teams / members    {cfg.topology} / {cfg.teams} / "
          f"{cfg.members}")
    print(f"sessions / capacity / load    {cfg.teams * cfg.members} / "
          f"{cfg.capacity} / {cfg.teams * cfg.members / cfg.capacity:.2f}x")
    print(f"wall clock                    {wall:>10.1f} s")
    print(f"collaboration steps           {len(rows):>10}")
    print(f"aggregate GPR (steps/s)       {len(rows) / wall:>10.2f}")
    print(f"turn miss rate                {len(miss) / len(rows) * 100:>9.1f}%")
    print(f"mean cached fraction          "
          f"{statistics.mean(r['cached'] for r in rows) * 100:>9.1f}%")
    print(f"TTFT   p50 / p95              {pct(ttft, .5):>10.2f} / "
          f"{pct(ttft, .95):.2f} s")
    print(f"WTFT   p50 / p95              {pct(wtft, .5):>10.2f} / "
          f"{pct(wtft, .95):.2f} s")
    print(f"TTTA   p50 / p95              {pct(ttta, .5):>10.1f} / "
          f"{pct(ttta, .95):.1f} s")
    print(f"CST    p50 / max              "
          f"{statistics.median(x['cst'] for x in per_team) * 100:>9.1f}% / "
          f"{max(x['cst'] for x in per_team) * 100:.1f}%")
    print(f"per-team GPR p50 / min        "
          f"{statistics.median(x['gpr'] for x in per_team):>10.3f} / "
          f"{min(x['gpr'] for x in per_team):.3f}")
    print(f"per-team miss p50 / max       "
          f"{statistics.median(x['miss_rate'] for x in per_team) * 100:>9.1f}% / "
          f"{max(x['miss_rate'] for x in per_team) * 100:.1f}%")
    print(f"starved teams (>=90% miss)    "
          f"{sum(1 for x in per_team if x['miss_rate'] >= 0.9):>10} / "
          f"{len(per_team)}")
    print(f"$ per team-task               {usd_per_team_task:>10.3f}")
    print("\nper-member miss rate (a working quota shows a step here):")
    for x in per_member:
        print(f"  member {x['member']}  n={x['steps']:>4}  "
              f"miss {x['miss_rate'] * 100:>5.1f}%  "
              f"cached {x['mean_cached'] * 100:>5.1f}%")
    print("\nteams meeting TTTA budget:")
    for budget, met, frac in slo_rows:
        print(f"  <= {budget:>6.0f} s   {met:>3} / {len(per_team)}  "
              f"({frac * 100:.0f}%)")

    out = cfg.out or f"team_{cfg.label}_{cfg.teams}t_{cfg.topology}.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({
            "label": cfg.label, "topology": cfg.topology,
            "teams": cfg.teams, "members": cfg.members,
            "rounds": cfg.rounds, "context": cfg.context,
            "gen_tokens": cfg.gen_tokens, "capacity": cfg.capacity,
            "load": cfg.teams * cfg.members / cfg.capacity,
            "wall_s": wall, "steps": len(rows),
            "gpr_aggregate": len(rows) / wall,
            "turn_miss_rate": len(miss) / len(rows),
            "mean_cached": statistics.mean(r["cached"] for r in rows),
            "ttft_p50": pct(ttft, .5), "ttft_p95": pct(ttft, .95),
            "wtft_p50": pct(wtft, .5), "wtft_p95": pct(wtft, .95),
            "ttta_p50": pct(ttta, .5), "ttta_p95": pct(ttta, .95),
            "cst_p50": statistics.median(x["cst"] for x in per_team),
            "cst_max": max(x["cst"] for x in per_team),
            "team_gpr_p50": statistics.median(x["gpr"] for x in per_team),
            "team_gpr_min": min(x["gpr"] for x in per_team),
            "team_miss_p50": statistics.median(x["miss_rate"]
                                               for x in per_team),
            "team_miss_max": max(x["miss_rate"] for x in per_team),
            "starved_teams": sum(1 for x in per_team
                                 if x["miss_rate"] >= 0.9),
            "usd_per_team_task": usd_per_team_task,
            "slo": [{"budget_s": b, "met": m, "frac": f}
                    for b, m, f in slo_rows],
            "per_team": per_team,
            "per_member": per_member,
        }, fh, indent=2)
    print(f"\nwrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True,
                    help="names the arm; also seeds the cache-reset content")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--teams", type=int, default=8)
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--topology", default="ring",
                    choices=("ring", "star", "empirical", "mixed"))
    ap.add_argument("--context", type=int, default=15360)
    ap.add_argument("--team-contexts", default="",
                    help="comma-separated per-team private history lengths; "
                         "empty = every team uses --context. A 10-team node "
                         "is several concurrent groups, not 48 copies of one "
                         "ring, so this is how total demand scales.")
    ap.add_argument("--message", type=int, default=512,
                    help="arriving peer message appended each turn")
    ap.add_argument("--variants", type=int, default=4,
                    help="distinct peer messages a session cycles through; "
                         "1 would make turns byte-identical, which does not hit")
    ap.add_argument("--gen-tokens", type=int, default=224,
                    help="real decode length; 224 against a 15,872-token "
                         "prompt is the measured 68-74:1 input:output ratio")
    ap.add_argument("--rounds", type=int, default=5,
                    help="measured collaboration rounds per team")
    ap.add_argument("--warmup-rounds", type=int, default=2,
                    help="unmeasured rounds first; two is the minimum that "
                         "clears SGLang's insert-time node split")
    ap.add_argument("--settle-rounds", type=int, default=0,
                    help="after warmup, flush the pool and plant this many "
                         "unmeasured rounds so a just-engaged keep-set is "
                         "resident before scoring; 0 leaves the old path")
    ap.add_argument("--stagger", type=float, default=2.0,
                    help="seconds between team arrivals; 0 makes the fleet "
                         "start in lockstep, under which a group-blind policy "
                         "and a per-team quota admit the same sessions")
    ap.add_argument("--capacity", type=int, required=True,
                    help="measured cache capacity in sessions, for load only")
    ap.add_argument("--slo-ttta", type=float, default=120.0,
                    help="team SLO: TTTA budget in seconds")
    ap.add_argument("--usd-per-hour", type=float, default=112.32)
    ap.add_argument("--reset-multiple", type=float, default=1.2)
    ap.add_argument("--vocab", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=None)
    cfg = ap.parse_args()
    if cfg.team_contexts.strip():
        xs = [int(x) for x in cfg.team_contexts.split(",") if x.strip()]
        if len(xs) == 1:
            xs = xs * cfg.teams
        if len(xs) != cfg.teams:
            raise SystemExit(
                f"--team-contexts has {len(xs)} entries, need {cfg.teams}")
        cfg.team_contexts = xs
    else:
        cfg.team_contexts = [cfg.context] * cfg.teams
    cfg.context = max(cfg.team_contexts)

    shared = Shared(cfg)
    tokens = sum(L * cfg.members for L in cfg.team_contexts)
    print(f"label={cfg.label}  topology={cfg.topology}  teams={cfg.teams}  "
          f"members={cfg.members}  rounds={cfg.rounds}")
    print(f"sessions={cfg.teams * cfg.members}  capacity={cfg.capacity}  "
          f"load={cfg.teams * cfg.members / cfg.capacity:.2f}x")
    print(f"team contexts {cfg.team_contexts}  working-set {tokens:,} tok")
    print(f"prompt peak={cfg.context + cfg.message:,} tok  gen={cfg.gen_tokens} "
          f"tok  ratio={(cfg.context + cfg.message) / cfg.gen_tokens:.0f}:1")
    print("the engine decides retention; this client is identical in both "
          "arms\n", flush=True)

    if cfg.reset_multiple > 0:
        shared.reset_cache()

    teams = [Team(t, cfg, shared) for t in range(cfg.teams)]
    threads = [threading.Thread(target=t.run, daemon=True) for t in teams]
    for th in threads:
        th.start()
    if cfg.warmup_rounds:
        print(f"warming up {cfg.warmup_rounds} rounds "
              f"({cfg.teams * cfg.members * cfg.warmup_rounds} steps)...",
              flush=True)
    tw = time.monotonic()
    shared.barrier.wait()
    print(f"warm-up done in {time.monotonic() - tw:.0f}s\n", flush=True)
    if cfg.settle_rounds:
        print(f"evicting leftovers and planting {cfg.settle_rounds} rounds...",
              flush=True)
        ts = time.monotonic()
        shared.evict_with_filler()
        shared.flushed.set()
        shared.settled.wait()
        print(f"settle done in {time.monotonic() - ts:.0f}s\n", flush=True)
    t0 = time.monotonic()

    # Progress while it runs, so a stuck arm is visible immediately rather than
    # after the whole budget is spent.
    target = cfg.teams * cfg.members * cfg.rounds
    while any(th.is_alive() for th in threads):
        time.sleep(15.0)
        with shared.lock:
            now = sum(len(t.steps) for t in teams)
            recent = [r["cached"] for t in teams for r in t.steps[-6:]]
        hit = statistics.mean(recent) * 100 if recent else float("nan")
        print(f"  {now:>5}/{target} steps  {time.monotonic() - t0:>6.0f}s  "
              f"cached(recent) {hit:>5.1f}%", flush=True)
    for th in threads:
        th.join()
    # From the last team's own finish, not from when the polling loop noticed:
    # polling every fifteen seconds would quantise the wall clock to fifteen
    # seconds and swamp the difference between arms.
    report(cfg, teams, max(t.t_end for t in teams) - t0)


if __name__ == "__main__":
    main()
