#!/usr/bin/env python3
"""Offline optimum and online policies for peer-team KV retention.

THE PROBLEM. A peer team's members take turns. Each turn re-sends that member's
whole private history, so the cost of a turn is decided by one thing: did the
member's history survive in the prefix cache since its last turn? Calibrated on
8xB300 / Kimi K3 at a 60K context that is 2.803 s if it did not and 0.129 s if
it did -- a factor of 21.7. Retention is the whole game, and retention is a
caching problem.

WHY THE OFFLINE OPTIMUM IS COMPUTABLE. Retaining a finished session is a
*decision*, not a side effect: the engine may serve a miss without admitting the
result. That makes this caching-with-bypass rather than demand paging.

  * Uniform item size and cost (all members carry comparable context):
    McFarling's OPTb -- furthest-next-use applied to cache UNION {current
    request}, so the request may decline itself. Belady's MIN is optimal only
    under the forced (demand-paging) model; with bypass allowed it is not (c=1
    on "a b a" costs 3 forced, 2 with bypass). OPTb has a sandwich bound, not a
    published optimality proof (Michaud, TACO 2016), so we use it as a strong
    lower bound. Uniform *cost* is also required and is not automatic for KV,
    where miss cost scales with prefix length; the headline runs make all
    contexts equal length, a chosen property of the controlled workload.
  * Non-uniform (a real trace, where members differ in context length): exactly
    the strongly NP-complete CACHING(FAULT, OPTIONAL) of Chrobak, Woeginger,
    Makino and Xu (ESA'10), so bypass does not rescue tractability. The exact
    optimum comes from dynamic programming over subsets of resident sessions,
    tractable for one team. That DP is ours, not a published algorithm.

WHAT THE NUMBERS SAY, in increasing order of what they cost to believe.

  1. LRU -- what vLLM and SGLang both do -- misses on *every* reference at every
     capacity below the full working set. Peer turn-taking is a cyclic reference
     string, and cyclic access is LRU's textbook worst case: the least recently
     used member is precisely the one whose turn is next. `cycle-proof`.

  2. The fix for aggregate cost is not lookahead but admission control. Simply
     declining to admit once the cache is full reaches the offline optimum on a
     uniform cycle, with no group identity and no knowledge of the future.

  3. That aggregate optimum is reached by freezing whichever sessions happened
     to arrive first, so some teams hit always and others never. Aggregate
     prefill looks optimal while team-level progress is arbitrarily unfair. This
     is what group identity buys: a per-group budget spends the same total
     capacity but degrades every team to k-of-n residency instead of starving a
     subset outright. At 3.9x overload, LRU starves 200 of 200 teams and bypass
     alone still starves 32; a per-group budget starves none. `sweep`.

  4. A measured negative: refining the resident subset with wait-graph position
     changes nothing, in every regime tried. Queue position was meant to stand
     in for furthest-next-use, but on peer strings a fixed per-group subset is
     already as good. So the protocol needs `group_id` and not `waits_on` --
     a smaller change than we set out to argue for. `waits-ablation`.

  python3 optimal.py cycle-proof
  python3 optimal.py sweep
  python3 optimal.py scalability
  python3 optimal.py waits-ablation
  python3 optimal.py trace traces/team-run2.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import OrderedDict, defaultdict

# Calibrated on 8xB300, Kimi K3, vLLM, /v1/completions with raw token ids;
# produced by measure_cost_model.py and probe_reuse.py.
COLD_INTERCEPT_S = 0.056
COLD_RATE_TOK_S = 20_680.0
WAKE_INTERCEPT_S = 0.075
CACHED_READ_TOK_S = 1_182_000.0
KV_POOL_TOKENS = 22_202_457  # vLLM, Kimi K3, 8xB300
STARVED_MISS_RATE = 0.9      # a team missing this often is not making progress


# Truncated-geometric rate fitted jointly to the two measured traces (§3.4).
EMPIRICAL_LAMBDA = 0.3722


def cold_cost_s(length):
    """Recompute the whole context."""
    return COLD_INTERCEPT_S + length / COLD_RATE_TOK_S


def wake_cost_s(length, new_tokens=512):
    """Context served from the prefix cache; only the peer message is new."""
    return (WAKE_INTERCEPT_S + length / CACHED_READ_TOK_S
            + new_tokens / COLD_RATE_TOK_S)


# --------------------------------------------------------------------------
# reference strings
# --------------------------------------------------------------------------

def synth_reference(teams, members, rounds, seed=0, jitter=0.0):
    """Interleaved peer teams, each taking turns within itself.

    Matches what the traces show: no global barrier, teams progress
    independently and their turns interleave. `jitter` lets teams run at
    unequal rates, which is the case a fixed per-group subset has to survive.
    """
    rng = random.Random(seed)
    turn = {t: 0 for t in range(teams)}
    weight = [1.0 + jitter * rng.random() for _ in range(teams)]
    live = list(range(teams))
    refs = []
    for _ in range(rounds * teams * members):
        t = rng.choices(live, weights=[weight[x] for x in live])[0]
        refs.append((t, turn[t] % members))
        turn[t] += 1
    return refs


def trace_reference(path):
    """Recover the reference string from a recording-proxy trace.

    Sessions are recovered by prompt-prefix containment, the same signal the
    engine itself has. Each request is one reference to its session; the context
    length at that moment comes from the reported input_tokens.
    """
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    rows.sort(key=lambda r: r["t_recv"])

    sessions, refs = [], []
    for row in rows:
        chain = row["prefix_sha"]
        best, best_len = None, 0
        for sess in sessions:
            shared = 0
            for x, y in zip(sess["chain"], chain):
                if x != y:
                    break
                shared += 1
            if shared == len(sess["chain"]) and shared > best_len:
                best, best_len = sess, shared
        if best is None:
            best = {"chain": chain, "id": len(sessions)}
            sessions.append(best)
        best["chain"] = chain
        usage = row.get("usage") or {}
        refs.append((best["id"], usage.get("input_tokens") or 0))
    return refs, len(sessions)


# --------------------------------------------------------------------------
# policies.  each returns one miss flag per reference.
# --------------------------------------------------------------------------

def run_lru(keys, capacity):
    """What vLLM and SGLang do: recency-ordered eviction, admit on every miss."""
    cache, flags = OrderedDict(), []
    for k in keys:
        flags.append(k not in cache)
        cache[k] = True
        cache.move_to_end(k)
        while len(cache) > capacity:
            cache.popitem(last=False)
    return flags


def run_lru_bypass(keys, capacity):
    """LRU plus the single change of not admitting on every miss.

    Isolates admission control from any notion of a group. Once full it stops
    admitting, which on a uniform cycle is exactly optimal in aggregate -- and
    exactly why it is unfair, since whoever arrived first keeps residency
    forever and the rest never get a turn in cache.
    """
    cache, flags = set(), []
    for k in keys:
        hit = k in cache
        flags.append(not hit)
        if not hit and len(cache) < capacity:
            cache.add(k)
    return flags


def run_lru_sized(keys, sizes, budget_tokens):
    """LRU under a token budget instead of a session count.

    Needed to compare against the subset DP at all: a session-count budget
    silently lets a count-based policy hold more tokens than the DP is allowed,
    which makes the "optimum" look beatable. Under non-uniform sizes the only
    common currency is tokens.
    """
    cache, flags = OrderedDict(), []
    for k in keys:
        flags.append(k not in cache)
        cache[k] = sizes[k]
        cache.move_to_end(k)
        while sum(cache.values()) > budget_tokens and len(cache) > 1:
            cache.popitem(last=False)
        if sizes[k] > budget_tokens:
            cache.pop(k, None)  # cannot be retained at all
    return flags


def run_lru_bypass_sized(keys, sizes, budget_tokens):
    """Admission control under a token budget: admit only while it still fits."""
    cache, flags = {}, []
    for k in keys:
        hit = k in cache
        flags.append(not hit)
        if not hit and sum(cache.values()) + sizes[k] <= budget_tokens:
            cache[k] = sizes[k]
    return flags


def cohort_quota(refs, capacity, members):
    """How many members of each group stay resident.

    The remainder is handed out one member at a time rather than dropped, so
    the whole budget is spent; without that a fleet just under the capacity
    wall would leave a slice of the pool idle.
    """
    groups = sorted({t for t, _ in refs})
    base, extra = divmod(capacity, max(len(groups), 1))
    return {g: min(base + (1 if i < extra else 0), members)
            for i, g in enumerate(groups)}


def run_cohort(refs, capacity, members):
    """Cohort: split the budget per group, keep a stable subset, bypass the rest.

    Needs nothing but `group_id`. The k-of-n shape is the point: under pressure
    a group gives up a few members and keeps the rest permanently resident,
    rather than thrashing all of them. A resident member still pays full price
    the first time it is seen.
    """
    quota = cohort_quota(refs, capacity, members)
    resident, flags = set(), []
    for t, m in refs:
        flags.append((t, m) not in resident)
        if m < quota[t]:
            resident.add((t, m))
    return flags


def run_cohort_waits(refs, capacity, members, queue_depth=2):
    """Ablation: let wait-graph position choose the resident subset.

    A member at position p of its group's pending queue cannot run until p peers
    ahead of it have produced output, so queue position looked like a
    protocol-supplied stand-in for furthest-next-use. Measured against the fixed
    subset above it changes nothing, which is why Cohort does not ask the
    protocol for wait edges. Kept so the claim stays falsifiable.
    """
    quota = cohort_quota(refs, capacity, members)
    cache, age, flags = set(), {}, []
    upcoming = defaultdict(int)
    for i, (team, who) in enumerate(refs):
        key = (team, who)
        flags.append(key not in cache)
        age[key] = i
        upcoming[team] = who
        # the queue the protocol would expose: the next `queue_depth` members
        rank = {(who + 1 + d) % members: d for d in range(queue_depth)}
        candidates = {m for (t, m) in cache if t == team} | {who}
        ordered = sorted(candidates,
                         key=lambda m: (rank.get(m, queue_depth + members),
                                        age.get((team, m), -1)))
        keep = {(team, m) for m in ordered[:quota[team]]}
        cache = {k for k in cache if k[0] != team} | keep
    return flags


def run_optb(keys, capacity):
    """McFarling's OPTb: furthest next use over cache UNION {current request}.

    Bypass is what separates this from demand paging. Demand paging must admit
    what it just served; here the request may decline itself, which is what
    lets the bound hold a stable working set instead of being dragged around by
    a cyclic string. Note this is *not* Belady's MIN, which is optimal only in
    the forced model; see the module docstring.
    """
    n = len(keys)
    next_use, last = [n] * n, {}
    for i in range(n - 1, -1, -1):
        next_use[i] = last.get(keys[i], n)
        last[keys[i]] = i

    cache, flags = {}, []
    for i, k in enumerate(keys):
        if k in cache:
            flags.append(False)
            cache[k] = next_use[i]
            continue
        flags.append(True)
        if len(cache) < capacity:
            cache[k] = next_use[i]
        else:
            worst = max(cache, key=cache.__getitem__)
            if cache[worst] > next_use[i]:
                del cache[worst]
                cache[k] = next_use[i]
            # else bypass: everything cached is needed sooner than this item
    return flags


def run_cohort_tokens(refs, sizes, budget_tokens, order="size"):
    """Per-group admission budgeted in tokens rather than in sessions.

    Counting sessions is only the right unit when sessions are interchangeable.
    On the measured traces they are not: the lead's context runs 2.2x the
    smallest member's, because the lead accumulates the whole conversation
    while teammates do not.

    `order` is what the budget is spent on first:

      index  member index order, which is what the deployed mechanism does. It
             needs no measurement of anything, only the id already in the rid.
      size   smallest context first, the textbook move for fitting more items
             into a fixed space.
      freq   most-referenced first, using observed counts. Informed, so it is
             an upper bound on what an LFU-style online estimator could reach,
             not a deployable policy.

    Measured, `size` loses to plain bypass and `freq` wins, because on real
    peer teams the lead is simultaneously the largest session and the most
    referenced one. Sorting by size is therefore exactly backwards, which is
    also why Cohort's index rule works: member 0 is the lead.
    """
    groups = sorted({t for t, _ in refs})
    per_group = budget_tokens / max(len(groups), 1)
    members_of = defaultdict(set)
    seen_count = defaultdict(int)
    for t, m in refs:
        members_of[t].add(m)
        seen_count[(t, m)] += 1

    if order == "index":
        def key_fn(t, m):
            return m
    elif order == "size":
        def key_fn(t, m):
            return sizes[(t, m)]
    elif order == "freq":
        def key_fn(t, m):
            return -seen_count[(t, m)] * (cold_cost_s(sizes[(t, m)])
                                          - wake_cost_s(sizes[(t, m)]))
    else:
        raise SystemExit(f"unknown order {order!r}")

    resident = set()
    for t in groups:
        spent = 0.0
        for m in sorted(members_of[t], key=lambda m: key_fn(t, m)):
            if spent + sizes[(t, m)] > per_group:
                continue
            spent += sizes[(t, m)]
            resident.add((t, m))

    seen, flags = set(), []
    for key in refs:
        flags.append(key not in seen)
        if key in resident:
            seen.add(key)
    return flags


def partial_cost_s(resident_tokens, missing_tokens):
    """Cost of a turn served partly from cache and partly recomputed.

    Checked against the one direct observation of a partially evicted session:
    at the capacity boundary a 61,440-token session was 49.6% cached and took
    1.566 s, against 2.803 s cold and 0.129 s warm. This model gives 1.660 s
    there, 6% high, which is close enough to correct a bias but not to be
    quoted as a measurement.
    """
    if missing_tokens <= 0:
        return wake_cost_s(resident_tokens, 0)
    return (COLD_INTERCEPT_S + missing_tokens / COLD_RATE_TOK_S
            + resident_tokens / CACHED_READ_TOK_S)


def simulate_blocks(keys, capacity_blocks, blocks_per_session, admit):
    """Block-granular cache: what the engine actually does.

    Session-granular simulation says a session is either resident or not.
    Measured, it is neither: eviction works on blocks, and the session sitting
    at the capacity boundary came back 49.6% cached at a latency squarely
    between cold and warm. Because a radix cache evicts leaves before their
    parents and a prefix match only credits a contiguous run from position 0,
    what survives of a session is a *prefix*, which is what this models: each
    session holds some number of leading blocks, and eviction takes tail blocks
    from the least recently used session first.

    The consequence for the paper's numbers is one-directional. A partially
    evicted session still pays only for the part it lost, so session-granular
    simulation overstates how bad LRU is, and every LRU/OPT ratio computed that
    way is an upper bound rather than a prediction.

    `admit(key)` decides whether a served request is retained, which is the one
    knob the mechanism turns.

    Returns the recomputed fraction per reference.
    """
    resident = {}          # session -> resident leading blocks
    order = OrderedDict()  # session -> recency, least recent first
    used = 0
    fracs = []
    for k in keys:
        have = resident.get(k, 0)
        fracs.append(1.0 - have / blocks_per_session)
        if not admit(k):
            # Served without being retained. Whatever prefix already survived
            # stays -- the engine releases this request's own KV, it does not
            # tear down the tree.
            continue
        used += blocks_per_session - have
        resident[k] = blocks_per_session
        order[k] = True
        order.move_to_end(k)
        while used > capacity_blocks and len(order) > 1:
            victim = next(iter(order))
            take = min(resident[victim], used - capacity_blocks)
            resident[victim] -= take
            used -= take
            if resident[victim] == 0:
                del resident[victim]
                del order[victim]
    return fracs


def run_dp_optimal(keys, sizes, capacity_tokens, max_sessions=16):
    """Exact optimum for the non-uniform case, by DP over resident subsets.

    State is the set of sessions whose context is resident. Each reference is
    charged a hit or a miss depending on whether its session is in the state,
    after which any affordable resident set may be chosen. Exponential in the
    session count, so this is for one team rather than a fleet.
    """
    distinct = sorted(set(keys))
    if len(distinct) > max_sessions:
        return None
    idx = {k: b for b, k in enumerate(distinct)}
    n = len(distinct)
    bit_size = [sizes[distinct[b]] for b in range(n)]

    def fits(mask):
        return sum(bit_size[b] for b in range(n)
                   if mask >> b & 1) <= capacity_tokens

    affordable = [m for m in range(1 << n) if fits(m)]
    INF = float("inf")
    best = [INF] * (1 << n)
    best[0] = 0.0
    for k in keys:
        b, size = idx[k], sizes[k]
        hit_c, miss_c = wake_cost_s(size), cold_cost_s(size)
        nxt = [INF] * (1 << n)
        for mask in affordable:
            base = best[mask]
            if base == INF:
                continue
            cost = base + (hit_c if mask >> b & 1 else miss_c)
            served = mask | (1 << b)
            sub = served
            while True:
                if cost < nxt[sub] and fits(sub):
                    nxt[sub] = cost
                if sub == 0:
                    break
                sub = (sub - 1) & served
        best = nxt
    return min(best)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def score(label, flags, teams_of_ref, length, new_tokens=512):
    """Aggregate cost, plus the per-team spread the aggregate hides."""
    misses = sum(flags)
    hits = len(flags) - misses
    per_team = defaultdict(lambda: [0, 0])
    for flag, team in zip(flags, teams_of_ref):
        per_team[team][0] += 1
        per_team[team][1] += flag
    rates = sorted(m / n for n, m in per_team.values())
    total = misses * cold_cost_s(length) + hits * wake_cost_s(length, new_tokens)
    return {
        "label": label,
        "miss_rate": misses / max(len(flags), 1),
        "prefill_s": total,
        "per_turn_s": total / max(len(flags), 1),
        "recomputed_tokens": misses * length,
        "team_miss_p50": statistics.median(rates) if rates else 0.0,
        "team_miss_max": max(rates) if rates else 0.0,
        "starved_teams": sum(1 for r in rates if r >= STARVED_MISS_RATE),
        "n_teams": len(per_team),
    }


ARMS = ("LRU (vLLM/SGLang)", "LRU + bypass", "Cohort", "OPTb (bound)")


def evaluate(refs, length, capacity, members):
    keys = refs
    teams_of_ref = [t for t, _ in refs]
    flags = (
        run_lru(keys, capacity),
        run_lru_bypass(keys, capacity),
        run_cohort(refs, capacity, members),
        run_optb(keys, capacity),
    )
    return {label: score(label, f, teams_of_ref, length)
            for label, f in zip(ARMS, flags)}


def print_rows(rows):
    base, opt = rows[ARMS[0]], rows[ARMS[-1]]
    print(f"{'policy':<22} {'miss':>7} {'prefill s':>11} {'vs LRU':>8} "
          f"{'vs OPT':>7} {'team miss p50':>14} {'max':>7} {'starved':>10}")
    for label in ARMS:
        r = rows[label]
        vs_lru = base["prefill_s"] / r["prefill_s"] if r["prefill_s"] else 0
        vs_opt = r["prefill_s"] / opt["prefill_s"] if opt["prefill_s"] else 0
        print(f"{label:<22} {r['miss_rate'] * 100:>6.1f}% "
              f"{r['prefill_s']:>11,.1f} {vs_lru:>7.2f}x {vs_opt:>6.2f}x "
              f"{r['team_miss_p50'] * 100:>13.1f}% "
              f"{r['team_miss_max'] * 100:>6.1f}% "
              f"{r['starved_teams']:>5}/{r['n_teams']:<4}")


# --------------------------------------------------------------------------
# topology-agnostic admission
# --------------------------------------------------------------------------

def topo_reference(topology, teams, members, rounds, seed=0):
    """Interleaved teams, each following `topology` internally.

    Teams interleave without a global barrier, as the traces show, so the
    pressure a session feels is its own team's order plus whatever the other
    teams touched meanwhile. Only the intra-team order changes here.

    `ring` and `empirical` are the two measured populations (§3.4): hard
    round-robin harnesses give a perfect ring, free-form peer messaging gives
    the fitted truncated-geometric order. The rest are the topologies we have
    *not* measured, and they are the reason a policy should not be keyed to a
    topology name at all:

    `star` a synthesiser in the middle. The hub speaks every other turn
           (recency rank 1) and each spoke waits k-1 distinct speakers, so the
           reuse distances are heterogeneous *within* one team. A team-level
           mean rank averages exactly the structure that matters away.
    `debate` two members dominate and the rest chime in occasionally. Same
           heterogeneity, different shape, and the valuable sessions are not
           the low-numbered ones.
    `pairs` disjoint pairs inside a team, i.e. sub-team locality.
    `mixed` one node serving teams from several harnesses at once, which is
           what a real deployment looks like and the case where no single
           fixed policy can be right for everyone.
    """
    if topology == "mixed":
        kinds = ["ring", "empirical", "star", "debate"]
        rng = random.Random(seed)
        gens = {t: _intra_order(kinds[t % len(kinds)], members, seed ^ (t << 8))
                for t in range(teams)}
        refs = []
        for _ in range(rounds * teams * members):
            t = rng.randrange(teams)
            refs.append((t, next(gens[t])))
        return refs
    rng = random.Random(seed)
    gens = {t: _intra_order(topology, members, seed ^ (t << 8)) for t in range(teams)}
    refs = []
    for _ in range(rounds * teams * members):
        t = rng.randrange(teams)
        refs.append((t, next(gens[t])))
    return refs


def _intra_order(topology, members, seed):
    if topology == "ring":
        while True:
            for m in range(members):
                yield m
    elif topology == "star":
        while True:
            for m in range(1, members):
                yield 0
                yield m
    elif topology == "debate":
        # Two principals alternate; a third party interjects once per cycle.
        # Deliberately puts the valuable sessions at indices 1 and 2 so that a
        # quota keyed to member order cannot get it right by luck.
        others = [m for m in range(members) if m not in (1, 2)]
        while True:
            for o in others:
                yield 1
                yield 2
                yield 1
                yield 2
                yield o
    elif topology == "pairs":
        pairs = [(m, m + 1) for m in range(0, members - 1, 2)]
        leftover = [members - 1] if members % 2 else []
        while True:
            for a, b in pairs:
                yield a
                yield b
                yield a
                yield b
            for m in leftover:
                yield m
    elif topology == "empirical":
        rng = random.Random(seed)
        weights = [math.exp(-EMPIRICAL_LAMBDA * r) for r in range(members)]
        stack = list(range(members))
        for m in stack:
            yield m
        while True:
            r = rng.choices(range(members), weights=weights)[0]
            m = stack.pop(r)
            stack.insert(0, m)
            yield m
    else:
        raise SystemExit(f"unknown topology {topology!r}")


def run_adaptive_team(refs, capacity, members, threshold=0.6, window=40):
    """The shipped rule: one scalar per team, compared to a fixed threshold.

    Reproduces `patch_sglang_cohort.py`'s `adaptive` mode, including the lower
    confidence bound on the mean normalised recency rank. Present here as the
    thing to beat, not as a baseline we believe in: it is correct on the two
    populations it was derived from and, as this mode shows, blind to any team
    whose reuse distances are heterogeneous inside the team.
    """
    quota = cohort_quota(refs, capacity, members)
    cache, flags = OrderedDict(), []
    order, ranks = defaultdict(list), defaultdict(list)
    for team, member in refs:
        key = (team, member)
        hit = key in cache
        flags.append(not hit)
        if hit:
            cache.move_to_end(key)

        hist = order[team]
        rank = None
        for j in range(len(hist) - 1, -1, -1):
            if hist[j] == member:
                rank = len(set(hist[j + 1:]))
                break
        hist.append(member)
        if len(hist) > 4 * window:
            del hist[:-2 * window]
        conf = None
        if rank is not None:
            seen = ranks[team]
            seen.append(rank)
            if len(seen) > window:
                del seen[:-window]
            if len(seen) >= members and members > 1:
                scale = float(members - 1)
                mean = sum(seen) / len(seen) / scale
                var = sum((r / scale - mean) ** 2 for r in seen) / len(seen)
                conf = mean - (var / len(seen)) ** 0.5

        engaged = conf is not None and conf > threshold
        skip = engaged and member >= quota[team]
        if not hit and not skip:
            cache[key] = True
            while len(cache) > capacity:
                cache.popitem(last=False)
    return flags


def run_unified(refs, capacity, members, window=200, margin=1.0,
                min_samples=2, cut_scale=1.0, swap_ratio=0.8):
    """One rule, no topology names: per-member reuse distance against share.

    LRU hits exactly when a session's reuse distance is below the capacity it
    competes for. So the question is never "which topology is this" but "is
    this session's distance inside its share", and that question can be asked
    per member:

      layer 1  distance < share  ->  LRU will keep it anyway; do not intervene.
      layer 2  distance >= share ->  doomed under LRU. Protect a fixed subset
                                     of these, up to the remaining budget, and
                                     bypass the rest.

    Layer 2 is not a heuristic bolted on: on a cyclic string every member has
    the same distance and the offline optimum is precisely to freeze an
    arbitrary subset (§5.1). It is a set decision, so no per-item rule can
    express it.

    The reductions are what makes this worth having. On a ring every distance
    equals k-1, everything falls to layer 2, and the rule *is* Cohort. On the
    measured order distances sit near zero, everything stays in layer 1, and
    the rule *is* LRU. On a star the hub's distance is 1 and it survives layer
    1 on its own merits rather than because it happens to be member 0 -- which
    is how the shipped quota keeps it, and that is luck of the numbering.

    `margin` scales the share before comparing, so the boundary case can be
    tuned without reintroducing a magic threshold on the statistic itself.
    """
    teams = max(len({t for t, _ in refs}), 1)
    share = max(int(capacity / teams), 1)
    flags = []
    cache = OrderedDict()               # resident, in global recency order
    pinned = set()                      # held by layer 2, exempt from eviction
    protect = defaultdict(set)          # team -> its protected members
    est = {}                            # session -> P(LRU holds it) over window
    est_dist = {}                       # session -> mean reuse distance
    dist = defaultdict(list)
    last_seen, clock = {}, 0

    def is_near(t, m):
        """Layer 1: LRU can hold this session unaided.

        The threshold is not a tunable: `share / members` is exactly the hit
        rate a team is guaranteed if it gives up on the rest and concentrates
        its whole share on a fixed subset. So the test reads "is LRU already
        doing better for this session than a guaranteed k-of-n slot would",
        which is the comparison the decision actually turns on. A fixed 0.5
        misses it -- `pairs` sits at 0.53-0.57, just above 0.5 and well below
        its 0.6 share ratio, so a 0.5 rule spreads where it should concentrate
        and reads 44.5% misses against Cohort's 31.9%.

        The threshold is not a tunable: `share / members` is exactly the hit
        rate a team is guaranteed if it gives up on the rest and concentrates
        its whole share on a fixed subset. So the test reads "is LRU already
        doing better for this session than a guaranteed k-of-n slot would",
        which is the comparison the decision actually turns on. A fixed 0.5
        misses it -- `pairs` sits at 0.53-0.57, just above 0.5 and well below
        its 0.6 share ratio, so a 0.5 rule spreads where it should concentrate
        and reads 44.5% misses against Cohort's 31.9%.

        Read from the current window every time it is asked, never cached as a
        membership set. An earlier version accumulated the layer-1 members
        monotonically and it failed for a reason worth recording: during warmup
        the cache is nearly empty, so *every* session measures a short reuse
        distance and lands in layer 1 permanently. The team's layer-2 budget
        then decays to zero and the policy degenerates. Any classifier over a
        sliding statistic has to be allowed to change its mind.
        """
        k = (t, m)
        if len(dist[k]) < min_samples:
            return True         # not enough evidence to refuse retention
        return est[k] >= cut_scale * float(share) / members

    for team, member in refs:
        key = (team, member)
        hit = key in cache
        flags.append(not hit)
        if hit:
            cache.move_to_end(key)

        # Reuse distance: distinct sessions touched since this session's own
        # previous turn. Measured globally, because LRU is a global structure
        # and the quantity that decides whether it holds a session is the
        # global distance; comparing it against a per-team share is a unit
        # error. Measured on the request stream, which the policy never alters
        # -- it changes what is retained, not who speaks -- so unlike an
        # idleness signal it cannot bias its own input (archive/notes/retired.md R7).
        if key in last_seen:
            since = last_seen[key]
            d = dist[key]
            d.append(sum(1 for t in last_seen.values() if t > since))
            if len(d) > window:
                del d[:-window]
            # The fraction of recent turns LRU would have served, not the mean
            # distance. Several topologies make a session's distance bimodal --
            # `pairs` alternates a short in-block gap with a long between-block
            # one -- and the mean of a bimodal distribution sits in a gap where
            # no sample ever lands, so a threshold on it classifies neither
            # mode. A fraction answers the question that actually matters:
            # would LRU hold this session more often than not.
            est[key] = sum(1 for x in d if x < capacity * margin) / len(d)
            est_dist[key] = sum(d) / len(d)
        last_seen[key] = clock
        clock += 1

        if hit:
            continue

        if is_near(team, member):
            protect[team].discard(member)
            pinned.discard(key)
            admit, pin = True, False          # layer 1, or still warming up
        else:
            # Layer 2: doomed under LRU, so protect a fixed subset. The budget
            # is what is left of the team's share after its layer-1 members,
            # because those occupy the cache too. Leaving that term out is not
            # a rounding error: it pinned three spokes per team on a star, left
            # one slot for ten hubs, and thrashed the single most valuable
            # session in every team.
            protect[team] &= {m for m in protect[team] if not is_near(team, m)}
            budget = max(share - sum(1 for m in range(members)
                                     if is_near(team, m)), 0)
            # Starvation guard, stated as the condition it guards rather than
            # as a floor on the budget. A team whose layer-1 members already
            # fill its share protects nobody, and if the shared pool is small
            # those members thrash there too: measured on `pairs` as one team
            # at 92.8% misses, a starved team, which is the one outcome the
            # group part of this mechanism exists to prevent. Raising the
            # budget floor to one unconditionally does fix it and costs far
            # too much elsewhere -- +11.1pp on the measured order, +8.4pp on a
            # mixed fleet -- because most teams are not starving and the
            # forced pin is taken out of the pool their healthy members need.
            # So the guard fires on the symptom: no resident sessions at all.
            if not any((team, m) in cache for m in range(members)):
                budget = max(budget, 1)
            # Bypass, not a fallback into the shared pool. We tried the
            # fallback on the theory that refusing a session frees space for
            # nobody, and measurement refuted it: the refused low-value
            # sessions go on to evict the layer-1 sessions they share the pool
            # with, which on a star is every team's hub -- 46.7% misses against
            # 25.9% with bypass, and 35.8% against 14.6% on debate. Declining
            # retention is what protects the sessions LRU is getting right.
            room = (member in protect[team]
                    or (len(protect[team]) < budget and len(pinned) < capacity))
            if not room and protect[team]:
                # Pick the protected subset by value, not by arrival. Mean
                # reuse distance is already measured and is the reciprocal of
                # reference rate, so the member seen most often is the one
                # worth a slot -- no new state. This is also what removes the
                # arbitrariness we hold against fixed quota, which seats
                # members by index: on `pairs` the low-index members happen to
                # be the valuable ones and quota gets it right by luck, while
                # here it is derived. Equal distances, as on a ring, leave the
                # choice arbitrary, which is correct: there the optimum is an
                # arbitrary subset.
                weakest = max(protect[team],
                              key=lambda m: est_dist.get((team, m), 0.0))
                # Hysteresis. Swapping on any improvement thrashes: on a ring
                # every distance is the same up to sampling noise, so a bare
                # inequality flips the protected subset back and forth and
                # each flip is a miss -- +6.4pp on ring and +4.8pp on star
                # against +0.6pp and +0.5pp without the swap at all. Requiring
                # a real margin keeps the swap for the case it was added for,
                # where the difference is a factor of 1.6.
                if est_dist.get(key, 0.0) < swap_ratio * est_dist.get(
                        (team, weakest), 0.0):
                    protect[team].discard(weakest)
                    pinned.discard((team, weakest))
                    room = True
            admit, pin = room, room

        if not admit:
            continue
        cache[key] = True
        if pin:
            pinned.add(key)
            protect[team].add(member)
        while len(cache) > capacity:
            victim = next((k for k in cache if k not in pinned), None)
            if victim is None or victim == key:
                break
            del cache[victim]
    return flags


# The one tuned constant in the mechanism. Must match COHORT_ADAPT_MARGIN in
# `patch_sglang_cohort.py`; `mode_sensitivity` reports the range over which the
# claim survives, and warns if the shipped value falls outside it.
SHIPPED_MARGIN = 1.02


def run_concentrate(refs, capacity, members, window=200, min_samples=2,
                    margin=1.0, bypass=True, spread_bypass=True, observe=None):
    """Concentrate or spread, decided per team by rate-weighted comparison.

    Synthesis of the two failures above. The marginal rule is right that the
    comparison must be rate-weighted and wrong to reason one item at a time:
    on a cyclic string the miss-ratio curve is convex over the range that
    matters -- no slot pays off until the whole working set fits -- so greedy
    marginal reasoning settles at the wrong end and reads +40.7pp on a ring.
    That is the same convexity the paper's C2 rests on, and the reason the
    optimum there is a *set* choice that no per-item rule can express. The
    two-layer rule does make a set choice and is wrong about the exchange rate:
    `share / members` is the value of concentrating only if members are
    referenced equally, which on `debate` costs +36.8pp.

    So keep the set decision and price it properly. Each team asks one
    question, in units of its own references:

      concentrating  gives  (rate mass of its top `share` members) / (its mass)
      spreading      gives  its measured LRU hit rate, rate-weighted

    and concentrates when that is the larger. The subset is then the top
    `share` members by reference rate, which is also what removes the
    arbitrariness of seating members by index: on a ring the rates are equal
    and the choice is arbitrary, which is correct, while on a star it selects
    the hub because the hub is busy, not because it is member 0.
    """
    teams = max(len({t for t, _ in refs}), 1)
    share = max(int(capacity / teams), 1)

    cache = OrderedDict()
    pinned, protect = set(), defaultdict(set)
    dist = defaultdict(list)
    seen = defaultdict(int)
    last, clock, flags = {}, 0, []

    def plan(team):
        """The team's protected subset, or empty if it should spread.

        Returns the per-member hit probability alongside it, because the
        spread branch needs it too: see `spread_bypass`.
        """
        p_hit, ms = {}, []
        for m in range(members):
            # Only reuse-distance samples taken while more sessions were
            # alive than the pool can hold. A hit recorded before that is
            # LRU not yet being a constraint, and averaging it into the
            # spread value is what makes a growing ring look like a 0.5
            # hit-rate workload after overflow. Same rule as the engine.
            bits = [1 if x < capacity else 0
                    for x, under in dist[(team, m)] if under]
            if len(bits) < min_samples:
                return None, {}             # still measuring: behave as LRU
            p_hit[m] = sum(bits) / float(len(bits))
            ms.append(m)
        rate = {m: seen[(team, m)] for m in ms}
        mass = sum(rate.values()) or 1
        top = sorted(ms, key=lambda m: -rate[m])[:share]
        concentrated = sum(rate[m] for m in top) / mass
        spread = sum(rate[m] * p_hit[m] for m in ms) / mass
        want = set(top) if concentrated > spread * margin else set()
        return want, p_hit

    for team, member in refs:
        key = (team, member)
        if observe is not None:
            observe(clock, key, cache)
        hit = key in cache
        flags.append(not hit)
        if hit:
            cache.move_to_end(key)

        if key in last:
            since = last[key]
            d = dist[key]
            reuse = sum(1 for v in last.values() if v > since)
            d.append((reuse, len(last) > capacity))
            if len(d) > window:
                del d[:-window]
        last[key] = clock
        clock += 1
        seen[key] += 1

        if hit:
            continue

        want, p_hit = plan(team)
        if want is None:
            admit, pin = True, False
        elif not want:
            for m in protect[team]:
                pinned.discard((team, m))
            protect[team].clear()
            # Spread: the shared pool serves it. But "spread" should not mean
            # "admit unconditionally" -- a member the pool provably cannot hold
            # gains nothing from being admitted and evicts one that can. This is
            # the paper's own admission-control result applied to the branch
            # that declines to concentrate, and it is group-blind. Without it
            # the rule's two options are pin-or-LRU, and on `debate` at high
            # load neither is right: plain admission control beats both by
            # 12-22pp, which is what `mode_sensitivity` surfaced.
            admit = not (spread_bypass and p_hit.get(member, 1.0) <= 0.0)
            pin = False
        else:
            for m in protect[team] - want:
                pinned.discard((team, m))
            protect[team] &= want
            pin = member in want and len(pinned) < capacity
            # Concentrate: the rest is bypassed. `bypass=False` exists to be
            # ablated -- it admits the unprotected members to the shared pool
            # instead, which looks strictly kinder and is not: they then flood
            # the pool and evict the members another team is relying on LRU to
            # hold. Measured in `mode_sensitivity`.
            admit = pin or not bypass

        if not admit:
            continue
        cache[key] = True
        if pin:
            pinned.add(key)
            protect[team].add(member)
        while len(cache) > capacity:
            victim = next((k for k in cache if k not in pinned), None)
            if victim is None or victim == key:
                break
            del cache[victim]
    return flags


def mode_sensitivity(cfg):
    """Two ablations the rule needs and the paper does not yet have.

    `margin` is the one tuned constant in the whole mechanism: a team
    concentrates when its concentrated value exceeds its spread value *times*
    this. It exists because at indifference the two options are not equally
    safe -- each team's decision to concentrate is locally correct against the
    current shared pool, and when every team makes it the pool collapses and
    the premise of the decision goes with it. So the margin buys a preference
    for the commons at the indifference point. A tuned constant is a liability
    unless we can say how wide the range is over which the conclusion holds,
    which is what the first table measures. It reports the property we actually
    claim -- never much worse than whichever fixed policy happens to be right
    for that topology -- rather than the miss rate, because a margin that wins
    on a ring and loses on a star is not a better margin.

    The second table ablates the bypass. When a team concentrates, the members
    outside its protected subset are declined rather than admitted to the
    shared pool. Admitting them looks strictly kinder -- they might get a hit
    -- and we measured it reversing on star and debate, because those members
    then flood the pool and evict the hubs that other teams were relying on
    LRU to hold. That result currently lives only in archive/notes/retired.md; it belongs in
    the paper, because "decline a request the cache could have served" is the
    single least intuitive thing the mechanism does.
    """
    margins = cfg.margins
    print(f"teams={cfg.teams} members={cfg.members} capacity={cfg.capacity} "
          f"rounds={cfg.rounds}  load="
          f"{cfg.teams * cfg.members / cfg.capacity:.2f}x\n")

    print("=== margin sensitivity: gap to the best fixed policy, in pp ===")
    print("negative is better than any fixed policy; the claim is that no cell "
          "is badly positive")
    head = "".join(f"{m:>9.2f}" for m in margins)
    print(f"{'topology':<12}{'best fixed':>12}{head}")
    worst = {m: -99.0 for m in margins}
    for topo in cfg.topologies:
        refs = topo_reference(topo, cfg.teams, cfg.members, cfg.rounds,
                              seed=cfg.seed)
        teams_of_ref = [t for t, _ in refs]
        fixed = {
            "LRU": run_lru(refs, cfg.capacity),
            "bypass": run_lru_bypass(refs, cfg.capacity),
            "Cohort": run_cohort(refs, cfg.capacity, cfg.members),
        }
        fm = {a: score(a, f, teams_of_ref, cfg.context)["miss_rate"]
              for a, f in fixed.items()}
        best = min(fm, key=fm.get)
        cells = []
        for m in margins:
            f = run_concentrate(refs, cfg.capacity, cfg.members, margin=m)
            r = score("unified", f, teams_of_ref, cfg.context)
            gap = (r["miss_rate"] - fm[best]) * 100
            worst[m] = max(worst[m], gap)
            cells.append(f"{gap:>+9.1f}")
        print(f"{topo:<12}{best:>12}{''.join(cells)}")
    print(f"{'WORST CELL':<12}{'':>12}"
          + "".join(f"{worst[m]:>+9.1f}" for m in margins))
    ok = [m for m in margins if worst[m] <= cfg.tolerance]
    if ok:
        print(f"\nmargins holding the claim within {cfg.tolerance:.0f}pp on "
              f"every topology: {ok[0]:.2f}-{ok[-1]:.2f} "
              f"(shipped: {SHIPPED_MARGIN:.2f})")
        if not (ok[0] <= SHIPPED_MARGIN <= ok[-1]):
            print("  WARNING: the shipped margin is outside that range")
    else:
        print(f"\nno margin holds the claim within {cfg.tolerance:.0f}pp -- "
              "the rule is not robust at this operating point")

    print("\n=== bypass ablation: decline the unprotected, or admit them? ===")
    print("`admit` is the kinder-looking variant; a positive delta means "
          "declining was better")
    print(f"{'topology':<12}{'bypass':>9}{'admit':>9}{'delta':>9}"
          f"{'starved (bypass/admit)':>26}")
    for topo in cfg.topologies:
        refs = topo_reference(topo, cfg.teams, cfg.members, cfg.rounds,
                              seed=cfg.seed)
        teams_of_ref = [t for t, _ in refs]
        out = {}
        for name, byp in (("bypass", True), ("admit", False)):
            f = run_concentrate(refs, cfg.capacity, cfg.members,
                                margin=SHIPPED_MARGIN, bypass=byp)
            out[name] = score(name, f, teams_of_ref, cfg.context)
        d = (out["admit"]["miss_rate"] - out["bypass"]["miss_rate"]) * 100
        print(f"{topo:<12}{out['bypass']['miss_rate'] * 100:>8.1f}%"
              f"{out['admit']['miss_rate'] * 100:>8.1f}%{d:>+8.1f}pp"
              f"{out['bypass']['starved_teams']:>18}"
              f"/{out['admit']['starved_teams']:<7}")


def mode_reductions(cfg):
    """Check the degeneracy claims mechanically rather than asserting them.

    The rule is only worth having if it *becomes* the right fixed policy
    instead of merely scoring near it, so each claim is checked on the
    decision it makes, not on the aggregate it produces:

      uniform rates, distances past capacity  ->  a frozen k-of-n subset,
                                                  which is Cohort
      distances inside capacity               ->  admits everything and pins
                                                  nothing, which is LRU
      one busy member                         ->  pins that member because it
                                                  is busy, not because it is
                                                  member 0
    """
    K, T, R = cfg.members, cfg.teams, cfg.rounds
    checks = []

    for topo, cap, want in (("ring", 31, "concentrate"),
                            ("empirical", 31, "spread"),
                            ("star", 31, "concentrate")):
        refs = topo_reference(topo, T, K, R, seed=cfg.seed)
        pins = _pins_taken(refs, cap, K)
        got = "concentrate" if pins else "spread"
        checks.append((f"{topo} at {T * K / cap:.2f}x", want, got, want == got))

    # The hub is member 0 on a star, so index and rate agree and the test is
    # vacuous. Relabel members so the busy one is last: a rule that seats by
    # index now protects the wrong session, a rule that reads rate does not.
    refs = [(t, (m + 1) % K) for t, m in topo_reference("star", T, K, R,
                                                        seed=cfg.seed)]
    hub = 1 % K
    pins = _pins_taken(refs, 31, K)
    got = "hub" if any(m == hub for _, m in pins) else "not hub"
    checks.append(("relabelled star protects the busy member", "hub", got,
                   got == "hub"))

    quota = cohort_quota(refs, 31, K)
    seated = all(m < quota[t] for t, m in pins)
    checks.append(("...and it is not the one quota would seat", "no", 
                   "yes" if seated else "no", not seated))

    print(f"{'claim':<44}{'want':>14}{'got':>14}{'':>4}")
    for name, want, got, ok in checks:
        print(f"{name:<44}{want:>14}{got:>14}{'  ok' if ok else '  FAIL':>6}")
    bad = sum(1 for *_, ok in checks if not ok)
    print(f"\n{len(checks) - bad}/{len(checks)} reductions hold")
    return bad


def _pins_taken(refs, capacity, members):
    """Which (team, member) pairs the rule ends up protecting."""
    teams = max(len({t for t, _ in refs}), 1)
    share = max(int(capacity / teams), 1)
    dist, seen, last, clock = defaultdict(list), defaultdict(int), {}, 0
    pins = set()
    for team, member in refs:
        key = (team, member)
        if key in last:
            d = dist[key]
            reuse = sum(1 for v in last.values() if v > last[key])
            d.append((reuse, len(last) > capacity))
            if len(d) > 200:
                del d[:-200]
        last[key] = clock
        clock += 1
        seen[key] += 1
    for team in range(teams):
        bits = {m: [1 if x < capacity else 0
                    for x, under in dist[(team, m)] if under]
                for m in range(members)}
        ms = [m for m in range(members) if len(bits[m]) >= 2]
        if len(ms) < members:
            continue
        mass = sum(seen[(team, m)] for m in ms) or 1
        top = sorted(ms, key=lambda m: -seen[(team, m)])[:share]
        conc = sum(seen[(team, m)] for m in top) / mass
        spread = sum(seen[(team, m)] * (sum(bits[m]) / len(bits[m]))
                     for m in ms) / mass
        if conc > spread * 1.02:
            pins |= {(team, m) for m in top}
    return pins


def mode_topology(cfg):
    """Does one rule handle topologies it was not derived from?

    The shipped adaptive rule was derived from two measured populations, ring
    and the fitted free-form order, both of which happen to be homogeneous
    inside a team. Star and debate are not, and a team-level mean cannot see
    the difference. This mode asks whether the per-member rule recovers the
    right behaviour on all of them without being told which is which.
    """
    arms = ("LRU", "bypass", "Cohort", "adaptive (team)", "two-layer",
            "unified", "OPTb")
    print(f"teams={cfg.teams} members={cfg.members} capacity={cfg.capacity} "
          f"rounds={cfg.rounds}  load="
          f"{cfg.teams * cfg.members / cfg.capacity:.2f}x")
    print("`adaptive (team)` is the shipped rule, keyed to one scalar per team; "
          "`two-layer` and\n`unified` are the per-member and the rate-weighted "
          "set rule (see archive/notes/retired.md R11)\n")

    summary = {}
    for topo in cfg.topologies:
        refs = topo_reference(topo, cfg.teams, cfg.members, cfg.rounds,
                              seed=cfg.seed)
        teams_of_ref = [t for t, _ in refs]
        flags = {
            "LRU": run_lru(refs, cfg.capacity),
            "bypass": run_lru_bypass(refs, cfg.capacity),
            "Cohort": run_cohort(refs, cfg.capacity, cfg.members),
            "adaptive (team)": run_adaptive_team(refs, cfg.capacity, cfg.members),
            "two-layer": run_unified(refs, cfg.capacity, cfg.members),
            "unified": run_concentrate(refs, cfg.capacity, cfg.members),
            "OPTb": run_optb(refs, cfg.capacity),
        }
        rows = {a: score(a, f, teams_of_ref, cfg.context) for a, f in flags.items()}
        summary[topo] = rows

        print(f"--- {topo} ---")
        print(f"{'policy':<18} {'miss':>7} {'team miss max':>14} {'starved':>10} "
              f"{'vs LRU':>8} {'vs OPTb':>8}")
        for a in arms:
            r = rows[a]
            vs_lru = (rows["LRU"]["prefill_s"] / r["prefill_s"]
                      if r["prefill_s"] else 0)
            vs_opt = (r["prefill_s"] / rows["OPTb"]["prefill_s"]
                      if rows["OPTb"]["prefill_s"] else 0)
            print(f"{a:<18} {r['miss_rate'] * 100:>6.1f}% "
                  f"{r['team_miss_max'] * 100:>13.1f}% "
                  f"{r['starved_teams']:>5}/{r['n_teams']:<4} "
                  f"{vs_lru:>7.2f}x {vs_opt:>7.2f}x")
        print()

    # The claim is not "unified wins everywhere" -- on the measured order the
    # right answer is LRU and matching it is the whole point. The claim is that
    # one rule is never much worse than whichever fixed policy happens to be
    # right for that topology, which is the only property a deployed node can
    # use.
    print("=== miss rate vs the best fixed policy for each topology ===")
    print(f"{'topology':<12} {'best fixed':>18} {'its miss':>9} "
          f"{'adaptive':>9} {'unified':>9} {'unified gap':>12}")
    for topo, rows in summary.items():
        fixed = {a: rows[a]["miss_rate"] for a in ("LRU", "bypass", "Cohort")}
        best = min(fixed, key=fixed.get)
        gap = rows["unified"]["miss_rate"] - fixed[best]
        print(f"{topo:<12} {best:>18} {fixed[best] * 100:>8.1f}% "
              f"{rows['adaptive (team)']['miss_rate'] * 100:>8.1f}% "
              f"{rows['unified']['miss_rate'] * 100:>8.1f}% "
              f"{gap * 100:>+11.1f}pp")


# --------------------------------------------------------------------------
# modes
# --------------------------------------------------------------------------

def mode_idleness(cfg):
    """Is a session's measured idleness independent of the eviction decision?

    The KV-offload line of work decides what to demote by measuring how idle a
    session has recently been. That is sound when idleness is exogenous: a
    session waiting on a tool call or on a human is idle for reasons the
    scheduler did not cause. On a peer team it is not. A member is idle because
    it is waiting for a teammate, and how long that wait is depends on whether
    the scheduler kept the *teammate's* context.

    So the loop to look for is not self-reinforcing on the victim -- evicting a
    member makes its own turns longer, which makes it look busier. It runs
    through the neighbours: evict one member, the whole team's round slows
    down, every other member of that team waits longer between turns, and their
    measured idleness rises. They become the next victims, and the damage
    concentrates on a team that is already behind.

    The simulation is event-driven over several teams, each cycling through its
    members serially, teams progressing independently as the traces show. A
    turn costs a cached read or a cold prefill. Whenever the resident set is
    over capacity, the resident session with the highest trailing-window idle
    share is dropped.

    The control replaces that score with a fixed round-robin victim order, so
    it evicts just as often and just as bluntly but without reading anything
    the scheduler itself perturbed. Any difference between the two is
    attributable to the feedback rather than to eviction pressure.
    """
    k, c, window = cfg.members, cfg.capacity, cfg.window
    teams, horizon = cfg.teams, cfg.horizon
    print(f"closed-loop idleness, {teams} teams x {k} members, "
          f"c={c} resident sessions, horizon={horizon}s")
    print(f"window={window}s, uniform {cfg.context:,} token contexts, "
          f"teams cycle independently\n")

    def run(policy):
        resident = set()
        busy = defaultdict(list)      # session -> [(start, end)]
        served = defaultdict(int)
        misses = defaultdict(int)
        # Event queue of (time_ready, team, member): when each team's next
        # speaker becomes runnable. Teams are staggered so they do not march
        # in lockstep, which is what the traces show.
        nxt = [(i * 0.5, i, 0) for i in range(teams)]
        rr = 0
        while nxt:
            nxt.sort()
            t_ready, team, m = nxt.pop(0)
            if t_ready > horizon:
                break
            sid = (team, m)
            hit = sid in resident
            dur = (wake_cost_s(cfg.context) if hit
                   else cold_cost_s(cfg.context))
            busy[sid].append((t_ready, t_ready + dur))
            served[sid] += 1
            if not hit:
                misses[sid] += 1

            if not hit:
                resident.add(sid)
                now = t_ready + dur
                while len(resident) > c:
                    if policy == "idleness":
                        lo = now - window
                        # Idle share over the trailing window. A session that
                        # has not run at all in the window scores 1.0.
                        def idle(x):
                            run_t = sum(max(0.0, e - max(s, lo))
                                        for s, e in busy[x])
                            return 1.0 - run_t / window
                        victim = max((x for x in resident if x != sid),
                                     key=idle)
                    else:
                        cand = sorted(x for x in resident if x != sid)
                        victim = cand[rr % len(cand)]
                        rr += 1
                    resident.discard(victim)
            nxt.append((t_ready + dur, team, (m + 1) % k))
        return busy, served, misses, horizon

    print(f"{'victim rule':>14} {'rounds/s':>9} {'miss':>7} "
          f"{'team miss p50':>14} {'max':>7} {'teams >=90%':>12} "
          f"{'idle: hurt half':>16} {'healthy half':>13}")
    summary = {}
    for policy in ("round-robin", "idleness"):
        busy, served, misses, end = run(policy)
        total = sum(served.values())
        lo = end - window
        team_miss, team_idle = [], []
        for team in range(teams):
            s = sum(served[(team, m)] for m in range(k))
            ms = sum(misses[(team, m)] for m in range(k))
            team_miss.append(ms / s if s else 0.0)
            # What the policy would see about this team right now.
            shares = []
            for m in range(k):
                run_t = sum(max(0.0, e - max(st, lo))
                            for st, e in busy[(team, m)])
                shares.append(1.0 - run_t / window)
            team_idle.append(statistics.mean(shares))

        order = sorted(range(teams), key=lambda t: team_miss[t])
        half = max(1, teams // 2)
        healthy = statistics.mean(team_idle[t] for t in order[:half])
        hurt = statistics.mean(team_idle[t] for t in order[-half:])
        p50, worst = statistics.median(team_miss), max(team_miss)
        starved = sum(1 for x in team_miss if x >= STARVED_MISS_RATE)
        summary[policy] = (total / end, sum(misses.values()) / total,
                           hurt, healthy)
        print(f"{policy:>14} {total / end:>9.2f} "
              f"{sum(misses.values()) / total * 100:>6.1f}% "
              f"{p50 * 100:>13.1f}% {worst * 100:>6.1f}% "
              f"{starved:>8}/{teams} {hurt * 100:>15.2f}% "
              f"{healthy * 100:>12.2f}%")

    rr_rate, rr_miss, _, _ = summary["round-robin"]
    id_rate, id_miss, hurt, healthy = summary["idleness"]
    print(f"\nTwo results, and the second one refutes the hypothesis we came in\n"
          "with, which is why it is worth reporting.\n"
          f"1. The idleness rule loses {rr_rate / id_rate:.2f}x of the throughput "
          f"outright:\n"
          f"   {id_miss * 100:.1f}% miss against {rr_miss * 100:.1f}%, and more "
          f"teams cross the starvation line.\n"
          "   On a ring the idle shares are nearly equal, so the argmax is\n"
          "   decided by which member ran least recently -- the rule degenerates\n"
          "   to recency, which is the cyclic worst case this file opens with.\n"
          "   The round-robin control wins precisely because rotating the victim\n"
          "   is uncorrelated with the cycle, which is the same reason admission\n"
          "   control wins in the modes above.\n"
          f"2. We expected the metric to be endogenous -- hurt teams looking\n"
          f"   idler and so attracting more eviction. It is not: hurt teams\n"
          f"   score {hurt / healthy:.3f}x the healthy ones, i.e. no difference at all.\n"
          f"   The reason is exact rather than noise. On a serialised ring each\n"
          f"   member is busy for exactly 1/k of its team's cycle whatever the\n"
          f"   cycle length, because the members share one cycle end to end. A\n"
          f"   hurt team cycles fewer times but each of its turns is a 6.6x\n"
          f"   longer cold prefill, and the two cancel. Measured idle share sits\n"
          f"   at {hurt * 100:.1f}% against the predicted (k-1)/k = "
          f"{(k - 1) / k * 100:.1f}% for every team.\n"
          "   The conclusion is stronger than endogeneity and simpler: on a\n"
          "   serialised peer ring, recent-activity share is pinned by the team\n"
          "   size and carries no information about the team's health, so it\n"
          "   cannot rank teams for eviction even in principle.\n"
          "This is a secondary result. It does not support the mechanism in\n"
          "this paper; it explains why the mechanism measures nothing at all\n"
          "and takes the group label as given.")


def mode_cycle_proof(cfg):
    print("pure mailbox cycle, one team, 200 rounds, uniform 60K contexts\n")
    print(f"{'k':>3} {'c':>3} {'LRU':>8} {'+bypass':>9} {'Cohort':>8} "
          f"{'OPT':>8} {'(k-c)/k':>9}")
    for k in (3, 4, 5, 8, 12):
        for c in sorted({2, k // 2, k - 1}):
            if not 1 <= c < k:
                continue
            refs = synth_reference(1, k, 200)
            keys, n = refs, len(refs)
            print(f"{k:>3} {c:>3} {sum(run_lru(keys, c)) / n * 100:>7.1f}% "
                  f"{sum(run_lru_bypass(keys, c)) / n * 100:>8.1f}% "
                  f"{sum(run_cohort(refs, c, k)) / n * 100:>7.1f}% "
                  f"{sum(run_optb(keys, c)) / n * 100:>7.1f}% "
                  f"{(k - c) / k * 100:>8.1f}%")
    print("\nLRU is the textbook cyclic worst case: the least recently used\n"
          "member is exactly the one whose turn is next. On a single uniform\n"
          "cycle, declining to admit already reaches the optimum, so group\n"
          "identity buys nothing here. What it buys is fairness across teams,\n"
          "which one team cannot show -- see `sweep`.")


def mode_synth(cfg):
    refs = synth_reference(cfg.teams, cfg.members, cfg.rounds, cfg.seed,
                           cfg.jitter)
    L = cfg.context
    n_sessions = cfg.teams * cfg.members
    capacity = min(cfg.pool // L, n_sessions)
    print(f"teams={cfg.teams}  members/team={cfg.members}  context={L:,} tok  "
          f"jitter={cfg.jitter}")
    print(f"sessions={n_sessions}  references={len(refs):,}")
    print(f"KV pool={cfg.pool:,} tok -> capacity={capacity} sessions "
          f"({capacity / n_sessions * 100:.0f}% of the working set)")
    print(f"cold={cold_cost_s(L):.3f}s  wake={wake_cost_s(L):.3f}s  "
          f"ratio={cold_cost_s(L) / wake_cost_s(L):.1f}x\n")
    print_rows(evaluate(refs, L, capacity, cfg.members))


def mode_sweep(cfg):
    cap_total = cfg.pool // cfg.context
    print(f"context={cfg.context:,} tok  pool={cfg.pool:,} tok -> {cap_total} "
          f"resident sessions = {cap_total / cfg.members:.1f} full teams")
    print(f"members/team={cfg.members}  jitter={cfg.jitter}\n")
    print(f"{'teams':>6} {'load':>6} | {'------ prefill seconds ------':^37} | "
          f"{'---- starved teams ----':^22}")
    print(f"{'':>6} {'':>6} | " +
          " ".join(f"{x:>8}" for x in ("LRU", "+byp", "Cohort", "OPT")) +
          " | " + " ".join(f"{x:>6}" for x in ("LRU", "+byp", "Cohort")))
    for teams in cfg.teams_list:
        refs = synth_reference(teams, cfg.members, cfg.rounds, cfg.seed,
                               cfg.jitter)
        n_sessions = teams * cfg.members
        rows = evaluate(refs, cfg.context, min(cap_total, n_sessions),
                        cfg.members)
        print(f"{teams:>6} {n_sessions / cap_total:>5.2f}x | " +
              " ".join(f"{rows[a]['prefill_s']:>8,.0f}" for a in ARMS) + " | " +
              " ".join(f"{rows[a]['starved_teams']:>6}" for a in ARMS[:3]))


def mode_scalability(cfg):
    """Collaborative scalability: teams per node at a team-level SLO.

    Single-request dashboards cannot express this. Teams per node is capped from
    above by pool capacity divided by team size no matter what the policy is --
    with a 22.2M-token pool and seven 60K members that wall is at 51.6 teams.
    The question a capacity planner actually faces is what happens past the
    wall, so the SLO is swept rather than fixed: at each per-turn prefill budget,
    how many teams can each policy carry while every team still progresses?
    """
    cap_total = cfg.pool // cfg.context
    grid = sorted({*range(1, 61, 1), *range(62, cfg.max_teams + 1, 2)})
    measured = {}
    for teams in grid:
        refs = synth_reference(teams, cfg.members, cfg.rounds, cfg.seed,
                               cfg.jitter)
        measured[teams] = evaluate(refs, cfg.context,
                                   min(cap_total, teams * cfg.members),
                                   cfg.members)

    uncontended = measured[max(cap_total // cfg.members // 2, 1)][ARMS[0]][
        "per_turn_s"]
    print(f"context={cfg.context:,} tok  members/team={cfg.members}  "
          f"jitter={cfg.jitter}")
    print(f"capacity wall = {cap_total} resident sessions "
          f"= {cap_total / cfg.members:.1f} teams, for any policy")
    print(f"uncontended per-turn prefill {uncontended * 1000:.0f} ms; "
          f"a team is starved above {STARVED_MISS_RATE * 100:.0f}% miss rate\n")

    def limit(label, budget):
        """Largest team count that meets the budget with nobody starved."""
        best = 0
        for teams in grid:
            r = measured[teams][label]
            if r["per_turn_s"] <= budget and r["starved_teams"] == 0:
                best = teams
            else:
                break
        return best

    print(f"{'per-turn SLO':>13} {'LRU':>7} {'+bypass':>9} {'Cohort':>8} "
          f"{'OPT':>7} {'Cohort/LRU':>12} {'Cohort/OPT':>12}")
    for budget_ms in cfg.slo_ms:
        b = budget_ms / 1000.0
        lim = {a: limit(a, b) for a in ARMS}
        coh, lru, opt = lim["Cohort"], lim[ARMS[0]], lim[ARMS[-1]]
        print(f"{budget_ms:>10} ms {lru:>7} {lim['LRU + bypass']:>9} "
              f"{coh:>8} {opt:>7} {coh / max(lru, 1):>11.2f}x "
              f"{coh / max(opt, 1):>11.2f}x")

    print("\nmax teams with zero starved teams, ignoring latency entirely:")
    for label in ARMS[:3]:
        best = 0
        for teams in grid:
            if measured[teams][label]["starved_teams"] == 0:
                best = teams
            else:
                break
        cap = ">= %d" % best if best == grid[-1] else str(best)
        print(f"  {label:<22} {cap}")


def mode_waits_ablation(cfg):
    """Does wait-graph position help pick the resident subset? Measured: no."""
    cap_total = cfg.pool // cfg.context
    print(f"{'teams':>6} {'jitter':>7} {'Cohort (fixed)':>16} "
          f"{'Cohort (waits)':>16} {'delta':>8}")
    for jitter in (0.0, 1.0, 3.0):
        for teams in (60, 80, 100, 150):
            refs = synth_reference(teams, cfg.members, cfg.rounds, cfg.seed,
                                   jitter)
            cap = min(cap_total, teams * cfg.members)
            teams_of = [t for t, _ in refs]
            fixed = score("f", run_cohort(refs, cap, cfg.members), teams_of,
                          cfg.context)["prefill_s"]
            waits = score("w", run_cohort_waits(refs, cap, cfg.members),
                          teams_of, cfg.context)["prefill_s"]
            print(f"{teams:>6} {jitter:>7.1f} {fixed:>16,.0f} "
                  f"{waits:>16,.0f} {(waits - fixed) / fixed * 100:>7.1f}%")
    print("\nWait-graph position does not improve on a fixed per-group subset,\n"
          "so Cohort asks the protocol for `group_id` only.")


def mode_block_sim(cfg):
    """Redo the numeric argument with the cache unit the engine actually uses.

    Every ratio quoted from `cycle-proof` and `sweep` was computed with whole
    sessions as the cache unit, and we know from measurement that this is
    wrong in a known direction: eviction is block-granular, a partially evicted
    session pays only for what it lost, so session-granular simulation
    overstates LRU's cost. This mode quantifies that gap so the paper can state
    a corrected number instead of a caveat.
    """
    L, block = cfg.context, cfg.block
    bps = max(1, (L + block - 1) // block)
    cold, warm = cold_cost_s(L), wake_cost_s(L)
    print(f"context={L:,} tok  block={block} tok -> {bps} blocks/session")
    print(f"cold={cold:.3f}s  warm={warm:.3f}s  ratio={cold / warm:.1f}x\n")

    def cost_blocks(fracs):
        return sum(partial_cost_s(int(L * (1 - f)), int(L * f)) for f in fracs)

    def cost_sessions(flags):
        m = sum(flags)
        return m * cold + (len(flags) - m) * warm

    print("one team, pure cycle, 200 rounds. LRU only: a policy whose resident"
          "\nset fits never evicts partially, so blocks change nothing for it."
          "\n")
    print(f"{'k':>3} {'c':>3} {'LRU sess':>10} {'LRU block':>11} "
          f"{'overstated':>11} {'OPT':>8} {'sess LRU/OPT':>13} "
          f"{'block LRU/OPT':>14}")
    for k in (3, 4, 5, 8, 12):
        for c in sorted({2, k // 2, k - 1}):
            if not 1 <= c < k:
                continue
            refs = synth_reference(1, k, 200)
            keys = refs
            sess = cost_sessions(run_lru(keys, c))
            blk = cost_blocks(simulate_blocks(keys, c * bps, bps,
                                              lambda _k: True))
            opt = cost_sessions(run_optb(keys, c))
            print(f"{k:>3} {c:>3} {sess:>10,.1f} {blk:>11,.1f} "
                  f"{sess / blk:>10.2f}x {opt:>8,.1f} {sess / opt:>12.2f}x "
                  f"{blk / opt:>13.2f}x")

    print("\nfleet of teams. Cohort and bypass are simulated on the same block"
          "\ncache, so any change to their numbers is real and not an artefact"
          "\nof the unit.\n")
    cap_total = cfg.pool // L
    print(f"pool={cfg.pool:,} tok -> {cap_total} resident sessions"
          f"  members/team={cfg.members}\n")
    print(f"{'teams':>6} {'load':>6} | {'LRU sess':>9} {'LRU blk':>9} "
          f"{'Cohort sess':>12} {'Cohort blk':>11} | "
          f"{'sess ratio':>11} {'block ratio':>12}")
    for teams in cfg.teams_list:
        refs = synth_reference(teams, cfg.members, cfg.rounds, cfg.seed,
                               cfg.jitter)
        keys = refs
        n_sessions = teams * cfg.members
        cap = min(cap_total, n_sessions)
        # Same admission rule as the session-granular arm, or the comparison
        # would measure a policy difference and call it a granularity effect.
        quota = cohort_quota(refs, cap, cfg.members)

        lru_sess = cost_sessions(run_lru(keys, cap))
        lru_blk = cost_blocks(simulate_blocks(keys, cap * bps, bps,
                                              lambda _k: True))
        coh_sess = cost_sessions(run_cohort(refs, cap, cfg.members))
        coh_blk = cost_blocks(simulate_blocks(
            keys, cap * bps, bps, lambda key: key[1] < quota[key[0]]))
        print(f"{teams:>6} {n_sessions / cap_total:>5.2f}x | "
              f"{lru_sess:>9,.0f} {lru_blk:>9,.0f} "
              f"{coh_sess:>12,.0f} {coh_blk:>11,.0f} | "
              f"{lru_sess / coh_sess:>10.2f}x {lru_blk / coh_blk:>11.2f}x")

    print("\nThe block ratio is the honest one. Where it is lower than the\n"
          "session ratio, the earlier figure was an upper bound on the\n"
          "headroom, exactly as the boundary-session measurement predicted.")


def mode_trace(cfg):
    refs, n_sessions = trace_reference(cfg.path)
    keys = [sid for sid, _ in refs]
    sizes = {}
    for sid, tok in refs:
        sizes[sid] = max(sizes.get(sid, 0), tok)
    total = sum(sizes.values())
    avg = int(total / len(sizes))
    print(f"trace={cfg.path}")
    print(f"references={len(keys)}  sessions={n_sessions}")
    print("context per session: " +
          ", ".join(f"{sizes[s]:,}" for s in sorted(sizes)))
    print(f"working set={total:,} tok  mean={avg:,} tok\n")
    spread = max(sizes.values()) / min(sizes.values())
    print(f"size spread max/min = {spread:.1f}x -- this is the regime where\n"
          "the uniformity premise behind 'bypass alone is optimal' fails.\n"
          "Every policy gets the same budget in tokens, which is the only\n"
          "common currency once sessions stop being interchangeable, and OPT\n"
          "is the exact subset DP under that same budget.\n")
    print(f"{'budget':>7} {'tokens':>9} {'LRU':>8} {'+bypass':>9} "
          f"{'idx-tok':>9} {'size-tok':>9} {'freq-tok':>9} {'OPT(DP)':>9} "
          f"{'byp/OPT':>9} {'idx/OPT':>9} {'freq/OPT':>9}")
    rows = []
    for frac in (0.3, 0.5, 0.7, 0.9):
        budget = int(total * frac)
        one_team = [(0, sid) for sid, _ in refs]
        by_key = {(0, sid): sizes[sid] for sid in sizes}

        def cost(flags):
            """Charge each reference at its own context length, not the mean.

            The mean is what hides the effect being measured: a policy that
            keeps three small members instead of one large one looks identical
            under a mean-cost model.
            """
            return sum(cold_cost_s(sizes[k]) if f else wake_cost_s(sizes[k])
                       for f, k in zip(flags, keys))

        lru = cost(run_lru_sized(keys, sizes, budget))
        byp = cost(run_lru_bypass_sized(keys, sizes, budget))
        tix = cost(run_cohort_tokens(one_team, by_key, budget, "index"))
        tsz = cost(run_cohort_tokens(one_team, by_key, budget, "size"))
        tfq = cost(run_cohort_tokens(one_team, by_key, budget, "freq"))
        dp = run_dp_optimal(keys, sizes, budget)
        dp_s = dp if dp is not None else float("nan")
        rows.append((byp / dp_s, tix / dp_s, tfq / dp_s))
        print(f"{frac * 100:>6.0f}% {budget:>9,} {lru:>8.1f} {byp:>9.1f} "
              f"{tix:>9.1f} {tsz:>9.1f} {tfq:>9.1f} {dp_s:>9.1f} "
              f"{byp / dp_s:>8.2f}x {tix / dp_s:>8.2f}x {tfq / dp_s:>8.2f}x")
    lo, hi = min(r[0] for r in rows), max(r[0] for r in rows)
    print(f"\nThree things this says, none of which the uniform cycle shows.\n"
          f"1. bypass is {lo:.2f}-{hi:.2f}x the optimum, against exactly 1.00x\n"
          "   on a uniform cycle, so the uniformity premise behind 'admission\n"
          "   control alone is optimal' is load-bearing and has to be stated.\n"
          "2. bypass is not even uniformly better than LRU any more: on the\n"
          "   cells above where LRU wins, freezing a subset chosen by arrival\n"
          "   order costs more than LRU's churn, because the frozen subset can\n"
          "   be the wrong shape when sizes differ. Admission control is a fix\n"
          "   for cyclic reference, not a strictly dominant policy.\n"
          "3. ordering within the group barely matters here and sorting by\n"
          "   size actively hurts, because the lead is both the largest context\n"
          "   and the most referenced one. Size-aware packing is backwards on\n"
          "   this workload; the deployed index rule keeps the lead for free.\n"
          "Note this is one team, so the per-group budget is the whole budget\n"
          "and index order degenerates to arrival order -- the group-aware\n"
          "column cannot separate from bypass here by construction. What the\n"
          "group split buys is measured on the fleet, not on one team.")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    def common(p):
        p.add_argument("--members", type=int, default=7)
        p.add_argument("--context", type=int, default=61440)
        p.add_argument("--rounds", type=int, default=20)
        p.add_argument("--jitter", type=float, default=0.0)
        p.add_argument("--pool", type=int, default=KV_POOL_TOKENS)
        p.add_argument("--seed", type=int, default=0)
        return p

    y = common(sub.add_parser("synth"))
    y.add_argument("--teams", type=int, default=60)
    y.set_defaults(fn=mode_synth)

    w = common(sub.add_parser("sweep"))
    w.add_argument("--teams-list", type=lambda x: [int(i) for i in x.split(",")],
                   default=[20, 40, 52, 60, 80, 100, 150, 200])
    w.set_defaults(fn=mode_sweep)

    s = common(sub.add_parser("scalability"))
    s.add_argument("--slo-ms", type=lambda x: [int(i) for i in x.split(",")],
                   default=[400, 600, 800, 1000, 1500, 2000, 2500])
    s.add_argument("--max-teams", type=int, default=300)
    s.set_defaults(fn=mode_scalability)

    common(sub.add_parser("waits-ablation")).set_defaults(fn=mode_waits_ablation)
    sub.add_parser("cycle-proof").set_defaults(fn=mode_cycle_proof)

    i = common(sub.add_parser("idleness"))
    i.add_argument("--teams", type=int, default=8)
    i.add_argument("--capacity", type=int, default=24,
                   help="resident sessions, below teams*members")
    i.add_argument("--window", type=float, default=30.0,
                   help="trailing window the idleness score is measured over")
    i.add_argument("--horizon", type=float, default=300.0)
    i.set_defaults(fn=mode_idleness)

    b = common(sub.add_parser("block-sim"))
    b.add_argument("--block", type=int, default=128,
                   help="prefix match unit; vLLM reports 128")
    b.add_argument("--teams-list",
                   type=lambda x: [int(i) for i in x.split(",")],
                   default=[20, 40, 52, 60, 80, 100, 150, 200])
    b.set_defaults(fn=mode_block_sim)

    g = common(sub.add_parser("topology"))
    g.add_argument("--teams", type=int, default=10)
    g.add_argument("--capacity", type=int, default=31,
                   help="resident sessions; default is the measured working point")
    g.add_argument("--topologies",
                   type=lambda x: x.split(","),
                   default=["ring", "empirical", "star", "debate", "pairs"])
    g.set_defaults(fn=mode_topology)

    z = common(sub.add_parser("sensitivity"))
    z.add_argument("--teams", type=int, default=9)
    z.add_argument("--capacity", type=int, default=31)
    z.add_argument("--topologies", type=lambda x: x.split(","),
                   default=["ring", "empirical", "star", "debate", "pairs",
                            "mixed"])
    z.add_argument("--margins", type=lambda x: [float(i) for i in x.split(",")],
                   default=[0.90, 0.95, 1.00, 1.02, 1.05, 1.10, 1.25, 1.50])
    z.add_argument("--tolerance", type=float, default=5.0,
                   help="how many pp worse than the best fixed policy still "
                        "counts as holding the claim")
    z.set_defaults(fn=mode_sensitivity)

    rd = common(sub.add_parser("reductions"))
    rd.add_argument("--teams", type=int, default=10)
    rd.set_defaults(fn=mode_reductions)

    t = sub.add_parser("trace")
    t.add_argument("path")
    t.set_defaults(fn=mode_trace)

    cfg = ap.parse_args(argv)
    cfg.fn(cfg)


if __name__ == "__main__":
    main()
