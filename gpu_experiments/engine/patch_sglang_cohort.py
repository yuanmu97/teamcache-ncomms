#!/usr/bin/env python3
"""Generate the Cohort patch for SGLang's KV retention path.

WHY IN THE ENGINE. Cohort was first tried entirely from a gateway, by
re-referencing the members a team needs resident to refresh their recency. That
works in isolation -- after a full pool turnover a touched session still read
from cache in 0.128 s against 2.902 s for its untouched twin -- but it collapses
under load. At 21 teams and 1.41x the measured capacity it ran 1.62x *slower*
than untouched LRU (2600.8 s against 1604.9 s) and starved more teams (7 of 21
against 3). The keeper loses a race it cannot win: pinning 84 sessions out of a
~92-104 session pool leaves too little headroom, so by the time a resident's
turn to be refreshed arrives it has already been evicted, and the refresh that
should have cost 0.15 s costs a 3.03 s prefill instead. Measured touch latency
was 5.43 s at the median. Retention has to be decided where eviction is decided.

WHAT THE PATCH DOES. Almost nothing, which is the point. SGLang already treats
"serve this but do not retain it" as a first-class per-request property,
`skip_radix_cache_insert`. The patch routes the finished-request site in
`srt/mem_cache/common.py` through one predicate that additionally bypasses
members beyond their team's residency quota. No new data structure, no
scheduler change, no eviction-order change.

A bypassed request whose prefill was chunked keeps whatever the chunked path
already inserted, because that bookkeeping cannot be skipped (see EDITS). That
is a safe degradation rather than a correctness problem, and it is why the
experiments size prompts to a single chunk, where bypass is exact.

Group and member ride on the existing request id as "<group>:<member>:<turn>",
so the protocol change is a naming convention rather than a schema change.
Every tensor-parallel rank's scheduler sees the same requests in the same order
and so reaches the same decision with no coordination. `COHORT_QUOTA=0` leaves
behaviour byte-identical to upstream, which is how the baseline arm runs.

The policy is read from a JSON control file, restat at most once per second,
rather than from the environment alone. Switching arms would otherwise need an
engine restart at roughly twelve minutes each, and the load curve plus the
ablations come to about twenty arms. Modes: `group` (Cohort), `rotate` (same
budget, resident subset rotates within each team), `online` (quota derived from
capacity over the observed active team count rather than handed in), `global`
(group-blind bypass with the same budget), `invdist` (informed baseline that
bypasses items whose observed reuse distance exceeds capacity).

  python3 patch_sglang_cohort.py --image sglang-kimi-k3:d6e7288627be \
      --out /tmp/sgl-patch
  python3 patch_sglang_cohort.py --selftest
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

TARGET = "srt/mem_cache/common.py"

ANCHOR = "from sglang.srt.utils.common import ceil_align"

HELPER = '''

# --- Cohort: per-group KV residency ---------------------------------------
# Peer team members take turns, which makes their references a cycle, and a
# cycle is LRU's worst case: the least recently used member is precisely the one
# whose turn is next, so under pressure every member misses on every turn. The
# fix is not a better eviction order but admission control with a notion of a
# group -- each team keeps a fixed quota of members resident and the rest are
# served without ever being admitted, so they cannot evict their own teammates.
#
# Aggregate cost alone does not justify the group part: plain bypass with no
# group notion reaches the same total by freezing whichever sessions arrived
# first, which starves whole teams. The per-group quota spends the same capacity
# and degrades every team to k-of-n instead.
import json as _cohort_json
import os as _cohort_os
import sys as _cohort_sys
import time as _cohort_time
from collections import OrderedDict as _cohort_OrderedDict


class _CohortConfig:
    """Policy knobs, reloaded from a control file at most once per second.

    Reading a file rather than the environment is what makes a sweep
    affordable. A quota change would otherwise need an engine restart, about
    twelve minutes each, and the load curve plus the ablations come to roughly
    twenty arms. The cost of the alternative is one stat per second per rank.

    A malformed or missing file leaves the last good values in force: a policy
    knob is never worth failing a request over. Every applied change is logged
    so an arm's policy can be confirmed from the engine log rather than
    assumed.
    """

    FIELDS = ("mode", "quota", "global_cap", "members", "capacity",
              "rotate_period", "window_s", "pool_tokens",
              "adapt_threshold", "adapt_window",
              "adapt_margin", "adapt_min_samples")

    def __init__(self):
        env = _cohort_os.environ.get
        self.mode = env("COHORT_MODE", "group")
        self.quota = int(env("COHORT_QUOTA", "0") or 0)
        self.global_cap = int(env("COHORT_GLOBAL_CAP", "0") or 0)
        self.members = int(env("COHORT_MEMBERS", "0") or 0)
        self.capacity = int(env("COHORT_CAPACITY", "0") or 0)
        self.rotate_period = int(env("COHORT_ROTATE_PERIOD", "0") or 0)
        self.window_s = float(env("COHORT_WINDOW_S", "30") or 30)
        self.pool_tokens = int(env("COHORT_POOL_TOKENS", "0") or 0)
        # 0.6 sits between the two measured populations (1.0 on a ring, 0.33
        # on the fitted trace distribution at k=5), so the classification is
        # not sensitive to it. The window has to hold at least one full round
        # before the mean means anything; 40 turns is ten rounds at k=4.
        self.adapt_threshold = float(env("COHORT_ADAPT_THRESHOLD", "0.6") or 0.6)
        self.adapt_window = int(env("COHORT_ADAPT_WINDOW", "40") or 40)
        # unified mode. The margin acts only where concentrating and spreading
        # are numerically equal; two samples is the least that distinguishes a
        # session LRU holds from one it drops, and asking for more delays the
        # first decision without making it better (measured: 3 and 5 samples
        # are 2.1pp and 3.4pp worse than 2 on the topology grid).
        self.adapt_margin = float(env("COHORT_ADAPT_MARGIN", "1.02") or 1.02)
        self.adapt_min_samples = int(env("COHORT_ADAPT_MIN_SAMPLES", "2") or 2)
        self.generation = 0
        self._path = env("COHORT_CONTROL", "")
        # None means never checked. A zero would not do: monotonic() counts
        # from process start here, so the rate limit below would swallow every
        # refresh for the first second, which is exactly when the first
        # requests of an arm arrive.
        self._checked = None
        self._mtime = None

    def snapshot(self):
        return tuple(getattr(self, f) for f in self.FIELDS)

    def refresh(self):
        if not self._path:
            return
        now = _cohort_time.monotonic()
        if self._checked is not None and now - self._checked < 1.0:
            return
        self._checked = now
        try:
            mtime = _cohort_os.stat(self._path).st_mtime
            if mtime == self._mtime:
                return
            with open(self._path, "r", encoding="utf-8") as fh:
                doc = _cohort_json.load(fh)
            if not isinstance(doc, dict):
                return
        except Exception:
            return
        before = self.snapshot()
        for field in self.FIELDS:
            if field not in doc:
                continue
            try:
                cur = getattr(self, field)
                setattr(self, field, type(cur)(doc[field]))
            except Exception:
                pass
        self._mtime = mtime
        if self.snapshot() != before:
            # A policy change starts a new arm, so the state that only makes
            # sense within an arm -- who was admitted, when a team was last
            # seen -- must not leak across the boundary.
            _cohort_state.reset()
            self.generation += 1
            print("[cohort] gen=%d %s" % (
                self.generation,
                " ".join("%s=%r" % (f, getattr(self, f)) for f in self.FIELDS)),
                file=_cohort_sys.stderr, flush=True)


class _CohortState:
    """Per-process bookkeeping for the policies that are not pure functions.

    Every tensor-parallel rank's scheduler sees the same requests in the same
    order, so each rank builds the same state and reaches the same decision
    with no coordination.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.admitted = set()      # global mode: sessions frozen in
        self.team_seen = {}        # online mode: team -> monotonic last seen
        self.size = {}             # online mode: session -> prompt tokens
        self.last_ref = {}         # invdist mode: session -> reference index
        self.refs = 0
        self.order = {}            # adaptive mode: team -> recent member ids
        self.ranks = {}            # adaptive mode: team -> recent recency ranks
        # unified mode. `stack` is the engine's own LRU order, which is what
        # lets the policy ask "would LRU have held this" instead of being told
        # which topology it is serving.
        self.stack = _cohort_OrderedDict()   # session -> prompt tokens
        self.held = {}             # session -> (would-hold, under_pressure) pairs
        self.rate = {}             # session -> aged reference count
        self.roster = {}           # team -> members ever seen
        self.since_age = 0
        # One observation and one decision per request id. The engine calls
        # the predicate at two insert sites (unfinished prefill, then
        # finished). Treating those as two references records a would-hold
        # of 1 at stack distance 0 -- the session is already MRU from the
        # first call -- and the pressure-regime p_hit never falls. Replay
        # calls the predicate once, which is why it saw the gain the live
        # engine did not.
        self.once = {}
        self.pressure_ticks = {}  # team -> under-pressure observes

    def observe(self, team, member, tokens, pool_tokens, window):
        """Record one reference and whether LRU would have served it.

        The token stack distance -- how much more recently used KV sits above
        this session -- is exactly the quantity that decides an LRU hit, so
        measuring it needs no model of the workload. Tokens rather than
        sessions because the pool is tokens and the traces show contexts
        differing 2.2x within one team.
        """
        key = (team, member)
        if tokens > 0:
            prev = self.stack.get(key)
            if prev is not None:
                above = 0
                for k in reversed(self.stack):
                    if k == key:
                        break
                    above += self.stack[k]
                # Footprint at arrival, before this session is moved to MRU.
                # `under` is the regime bit: a would-hold taken while every
                # known session still fitted is a measurement of a different
                # experiment -- LRU was not yet a constraint -- and must not
                # price the spread side once the stack no longer fits.
                footprint = sum(self.stack.values())
                hist = self.held.setdefault(key, [])
                hist.append((1 if above + prev <= pool_tokens else 0,
                             footprint > pool_tokens))
                if len(hist) > window:
                    del hist[:-window]
            self.stack[key] = tokens
            self.stack.move_to_end(key)
            # Bounded: a session that has not spoken in thousands of turns
            # cannot affect any decision, and the scan above is linear.
            while len(self.stack) > 4096:
                self.stack.popitem(last=False)
        self.rate[key] = self.rate.get(key, 0.0) + 1.0
        self.roster.setdefault(team, set()).add(member)
        # Age rather than window the rates, so a team that changes who carries
        # the conversation is followed rather than averaged over its history.
        self.since_age += 1
        if self.since_age >= window * 8:
            self.since_age = 0
            for k in list(self.rate):
                self.rate[k] *= 0.5
                if self.rate[k] < 0.01:
                    del self.rate[k]

    def concentrate_on(self, team, share_tokens, margin, min_samples,
                       cur=None):
        """The members worth freezing, empty to spread, None while measuring.

        One question per team, priced in that team's own references: is the
        reference mass of the members that fit in its share worth more than
        the LRU hit rate it is actually getting? Concentrating is the answer
        only when it is, which is what makes the same rule reduce to a frozen
        subset on a ring and to untouched LRU on the measured order.

        "The LRU hit rate it is actually getting" is the would-hold rate
        measured *while the working set did not fit*. Bits taken before the
        stack overflowed answer a different question -- LRU was not yet a
        constraint -- and averaging across that boundary is how a growing
        ring looks like a 0.5 hit-rate workload and the rule folds back.

        `cur` is the member being served right now, excluded from the pricing
        below. Without it the rule can decide to concentrate and still change
        nothing: `observe` counts this reference before the decision is taken,
        so on a ring -- where every member's true rate is equal -- the current
        speaker tops the ranking and lands in `top`, which is the one member
        LRU was holding anyway. Measured on the AutoGen ring trace, excluding
        it plus the stable tie-break below is worth +3.6pp, and on the 126-cell
        grid both together are free (median gap +0.04pp either way, 0 starved).
        """
        roster = self.roster.get(team)
        if not roster:
            return None
        stats = {}
        for m in roster:
            hist = self.held.get((team, m)) or []
            # Only bits recorded while the working set did not fit. Mixing
            # them with the fit-regime ones is what makes the rule engage
            # after overflow and then fold back: those early 1s keep the
            # spread value above the concentrated value just as the share
            # shrinks (contexts grow, reserve grows) and the comparison
            # becomes harder to win. There is no extra parameter -- the
            # pool is the regime boundary the rule already claims to use.
            bits = [bit for bit, under in hist if under]
            if len(bits) < min_samples:
                # Omit, do not block the team. Waiting for every roster
                # member kept mixed-fleet empirical teams on want=None
                # (a long-tail member speaks twice in eight rounds), so
                # they inserted 5×56k into a full pool and evicted the
                # ring keep-set. Rings are unchanged: every member is
                # measured on the same clock.
                continue
            stats[m] = (self.rate.get((team, m), 0.0)
                        - (1.0 if m == cur else 0.0),
                        sum(bits) / float(len(bits)),
                        self.stack.get((team, m), 0))
        mass = sum(r for r, _, _ in stats.values())
        if mass <= 0:
            return None
        top, spent = set(), 0
        # Member id breaks ties, because `roster` is a set and ties are the
        # normal case: on a ring every member has the same rate, and which
        # subset gets frozen then has to be stable across turns or the frozen
        # set rotates and retains nothing. The simulator got this for free --
        # it ranks over `range(members)` and Python's sort is stable -- so this
        # is the engine catching up with the design that was validated.
        for m in sorted(stats, key=lambda x: (-stats[x][0], x)):
            size = stats[m][2]
            if top and spent + size > share_tokens:
                continue
            top.add(m)
            spent += size
        concentrated = sum(stats[m][0] for m in top) / mass
        spread = sum(r * h for r, h, _ in stats.values()) / mass
        # The margin breaks ties toward leaving LRU alone, and only ties. The
        # spread value is observed; the concentrated value is self-invalidating
        # -- it assumes a pool that every team concentrating would destroy --
        # so at indifference the measured quantity is the one to trust. In
        # simulation a blanket margin is the wrong instrument (1.15 fixes a
        # mixed fleet at 1.25x by 7pp and costs 9pp on debate at 1.61x); 1.02
        # is small enough to act only where the two are numerically equal.
        return top if concentrated > spread * margin else set()

    def pool_can_hold(self, team, member, min_samples):
        """Has LRU ever actually held this member between its turns?

        Needed because "spread" must not mean "admit unconditionally". A member
        the pool has never once retained gains nothing from being inserted and
        evicts one that would have survived, so declining it is free. This is
        the paper's own admission-control result applied to the branch that
        declines to concentrate, and it is group-blind -- it asks only about
        this member's own history.

        Without it the rule's two options are pin-or-LRU, and neither is right
        on a skewed order under heavy pressure: on `debate` at 3.9x, plain
        admission control beat the rule by 21.8pp. Adding this closes 7.5pp of
        that and costs at most 0.4pp anywhere else. It also flattens the
        response to `margin`, which matters because that constant is the only
        tuned number in the mechanism: the worst cell over the whole grid moves
        by 1.4pp across margins 0.90-1.10 instead of 8.5pp.

        Unmeasured is treated as holdable, so a member is never declined on the
        strength of no evidence.
        """
        hist = self.held.get((team, member)) or []
        bits = [bit for bit, under in hist if under]
        if len(bits) < min_samples:
            return True
        return any(bits)

    def recency_rank(self, team, member, members, window):
        """Where the current speaker sat in its team's recency stack.

        This is the one statistic that separates the two regimes. A harness
        that serialises speaking (AutoGen's RoundRobinGroupChat, CrewAI's
        sequential process) drives it to members-1 every turn; free-form peer
        messaging, which is what the traces measure, keeps it near zero
        because the dominant mode is a member speaking again immediately.

        Returns a *lower confidence bound* on the mean rank over the window,
        normalised to [0, 1], or None until the window holds one full round.

        The bound rather than the mean, because the two regimes differ in
        variance as sharply as they differ in mean, and that is free accuracy.
        A ring is a zero-variance signal -- every rank is exactly members-1 --
        so one round already pins it down and the mechanism engages at once.
        The measured order is noisy, so a short window wanders: seven samples
        into the fitted distribution the running mean touches 0.61 and a
        mean-only rule misclassifies, while the bound stays at 0.49 and the
        policy correctly does nothing. The window then settles near 0.35 and
        never comes back. So the cost of being careful is paid only by the
        regime that can afford it.
        """
        hist = self.order.setdefault(team, [])
        rank = None
        for j in range(len(hist) - 1, -1, -1):
            if hist[j] == member:
                rank = len(set(hist[j + 1:]))
                break
        hist.append(member)
        if len(hist) > 4 * window:
            del hist[:-2 * window]
        if rank is None:
            return None
        seen = self.ranks.setdefault(team, [])
        seen.append(rank)
        if len(seen) > window:
            del seen[:-window]
        n = len(seen)
        if n < members or members < 2:
            return None
        scale = float(members - 1)
        mean = sum(seen) / n / scale
        var = sum((r / scale - mean) ** 2 for r in seen) / n
        return mean - (var / n) ** 0.5

    def active_teams(self, window_s):
        now = _cohort_time.monotonic()
        cutoff = now - window_s
        if len(self.team_seen) > 4096:
            self.team_seen = {t: s for t, s in self.team_seen.items()
                              if s >= cutoff}
        return sum(1 for s in self.team_seen.values() if s >= cutoff) or 1


_cohort_cfg = _CohortConfig()
_cohort_state = _CohortState()


def _cohort_skip_insert(req) -> bool:
    """True when this request's KV should be served but not retained.

    Group and member are read from the request id, by the convention
    "<group>:<member>:<turn>", so the protocol change is a naming convention.
    Requests that do not carry that shape -- every ordinary client, whose ids
    the engine generates and which contain no colon -- take the upstream path
    byte for byte. Quota 0 in group mode likewise reproduces upstream exactly,
    which is how the baseline arm runs.

    Modes:

    `group` is Cohort. The first `quota` members of each team are retained and
    the rest are served without being admitted, so a team degrades to k-of-n
    rather than thrashing as a whole.

    `rotate` is the same budget with the resident subset rotating within each
    team every `rotate_period` turns. It exists because fixed quota makes the
    same members second-class forever, and the claim being tested is about
    *where* the unavoidable misses land.

    `online` derives the budget instead of being told it: the token pool
    divided by the number of teams seen in the last `window_s` seconds, filled
    in member order until spent. Without this the mechanism has to be handed
    the answer, which is not a deployable design.

    `global` is the group-blind ablation: the same admission budget spent with
    no notion of a team, freezing the first `global_cap` distinct sessions to
    arrive. Aggregate cost says the group part does not earn its keep -- this
    policy is marginally cheaper overall -- so the comparison has to be made on
    where the misses land, and there it pushed the worst team to 84.8% misses
    against 58.6%.

    `invdist` is the informed baseline in the ScaleSim / KVFlow family: bypass
    an item whose observed reuse distance exceeds what the cache can hold,
    since it would not survive anyway. On a cycle every member's distance is
    the same, so the policy cannot discriminate and collapses to admit-all or
    bypass-all. That degeneracy is the point of running it.
    """
    if getattr(req, "skip_radix_cache_insert", False):
        return True
    _cohort_cfg.refresh()
    rid = getattr(req, "rid", None) or ""
    parts = rid.split(":")
    if len(parts) < 2 or not parts[1].isdigit():
        return False

    cfg = _cohort_cfg
    st = _cohort_state
    team, member = parts[0], int(parts[1])
    turn = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0

    if cfg.mode == "global":
        if cfg.global_cap <= 0:
            return False
        key = (team, member)
        if key in st.admitted:
            return False
        if len(st.admitted) < cfg.global_cap:
            st.admitted.add(key)
            return False
        return True

    if cfg.mode == "invdist":
        if cfg.capacity <= 0:
            return False
        key = (team, member)
        st.refs += 1
        prev = st.last_ref.get(key)
        st.last_ref[key] = st.refs
        if prev is None:
            return False
        return (st.refs - prev) > cfg.capacity

    if cfg.mode == "online":
        # The quota is derived, not handed in. Every other mode has to be told
        # how many members to keep, which quietly gives the mechanism the
        # answer; here it gets a token budget -- the pool divided by the teams
        # seen recently -- and fills it in member order until it is spent.
        #
        # Budgeting in tokens rather than sessions also handles unequal
        # contexts, which the traces show are the norm: the lead's history runs
        # 2.2x the smallest teammate's. Member order matters and is not
        # arbitrary: member 0 is the lead, which is both the largest context
        # and the most referenced, so admitting in index order keeps the one
        # worth keeping. Ordering by size instead measures worse than not
        # bothering at all.
        if cfg.pool_tokens <= 0:
            return False
        st.team_seen[team] = _cohort_time.monotonic()
        ids = getattr(req, "origin_input_ids", None)
        st.size[(team, member)] = len(ids) if ids else 0
        # The pool has to cover working KV for the requests in flight as well
        # as the contexts being retained, and with one runnable member per team
        # the concurrency is the team count. So reserve one request's worth per
        # active team. The reserve is the mean prompt actually observed, which
        # keeps the whole budget derived from what the engine sees rather than
        # configured -- and mean rather than max, because one unusually long
        # session would otherwise shrink every team's budget at once.
        #
        # Getting this wrong is not harmless in either direction. With the
        # reserve set too high the mechanism bypasses sessions that would have
        # fit and measures 1.20-1.34x *slower* than upstream at loads where
        # everything fits; too low and it behaves like LRU under pressure.
        active = st.active_teams(cfg.window_s)
        sizes = [v for v in st.size.values() if v > 0]
        reserve = (sum(sizes) / len(sizes)) if sizes else 0
        budget = max(cfg.pool_tokens - active * reserve, reserve) / active
        spent = 0
        for m in range(member + 1):
            spent += st.size.get((team, m), 0)
        # Always keep the first member: a budget too small for even one
        # session should degrade to one resident member, not to none.
        return member > 0 and spent > budget

    if cfg.mode == "rotate":
        if cfg.quota <= 0 or cfg.members <= 0 or cfg.rotate_period <= 0:
            return False
        shift = turn // cfg.rotate_period
        return ((member + shift) % cfg.members) >= cfg.quota

    if cfg.mode == "adaptive":
        # Group-aware admission is the right fix for one regime and the wrong
        # fix for the other, so the policy has to know which one it is in.
        #
        # On a serialised ring LRU evicts exactly the next speaker and quota
        # admission recovers most of the loss. On the measured order the
        # dominant mode is a member speaking again immediately -- 37% of turns
        # at k=5 -- so recency is a genuinely informative signal, LRU already
        # exploits it, and freezing a subset destroys it: measured 47.0%
        # misses against LRU's 33.0% at ten teams, and 83.9% against 53.4% at
        # sixteen. Engaging unconditionally is a regression on real traffic.
        #
        # The two regimes are separated by the normalised mean recency rank
        # (1.0 on a ring, 0.33 on the fitted trace distribution at k=5), which
        # is why one threshold suffices and why it is nowhere near either
        # population. Teams are classified independently, which matters
        # because a real node serves teams from several harnesses at once and
        # neither fixed policy is right for that mix.
        if cfg.quota <= 0 or cfg.members <= 0:
            return False
        conf = st.recency_rank(team, member, cfg.members, cfg.adapt_window)
        if conf is None or conf <= cfg.adapt_threshold:
            return False          # recency is informative: stay out of the way
        return member >= cfg.quota

    if cfg.mode == "unified":
        # `adaptive` asks which of two measured populations a team belongs to,
        # and that question is the wrong one. Its input is one scalar per team,
        # so it cannot see heterogeneity *inside* a team: on a star, where one
        # synthesiser speaks every other turn, the bound reads 0.566 against a
        # 0.6 threshold and the policy never engages, forfeiting the 1.80x that
        # group-aware admission gets there. Keying to per-member reuse distance
        # instead fixes the star and breaks elsewhere, because it prices
        # concentration at `share / members`, which holds only if members are
        # referenced equally -- +36.8pp on a debate order at 2.5x, where two
        # principals carry 81% of a team's references. Pricing it marginally
        # instead, from the reuse-distance histogram, breaks on the ring by
        # +40.7pp: a cyclic string makes the miss-ratio curve convex over the
        # range that matters, so no single slot pays off until the whole
        # working set fits and greedy marginal reasoning stops at the wrong
        # end. That is C2 restated -- the optimum there is a set decision.
        #
        # So: a set decision, priced in the team's own references. See
        # archive/notes/retired.md R11 for the grid these numbers come from; across 6
        # topologies x 7 loads x 3 seeds this rule sits 0.4pp (median) off the
        # best fixed policy chosen per cell in hindsight, and starves no team
        # in any cell where the three fixed policies starve 129, 85 and 54.
        if cfg.pool_tokens <= 0:
            return False
        if rid in st.once:
            return st.once[rid]
        st.team_seen[team] = _cohort_time.monotonic()
        # `origin_input_ids` is the full prompt on this engine, but a stale
        # bytecode cache once left it empty here and the rule sat in the
        # measuring branch forever -- indistinguishable from LRU. Prefer the
        # longest sequence field we can see, and never treat an empty array
        # as "no information" when another field has the length.
        tokens = 0
        _seq = getattr(req, "seqlen", None)
        if isinstance(_seq, int) and _seq > 0:
            tokens = _seq
        for _name in ("origin_input_ids", "origin_input_ids_unpadded",
                      "full_untruncated_fill_ids"):
            _ids = getattr(req, _name, None)
            if _ids is not None:
                try:
                    tokens = max(tokens, len(_ids))
                except TypeError:
                    pass
        st.observe(team, member, tokens, cfg.pool_tokens, cfg.adapt_window)

        active = st.active_teams(cfg.window_s)
        # Same reserve as `online`: the pool must cover working KV for the
        # requests in flight, one runnable member per team, before any of it
        # can be promised to retention.
        sizes = [v for v in st.stack.values() if v > 0]
        reserve = (sum(sizes) / len(sizes)) if sizes else 0
        share = max(cfg.pool_tokens - active * reserve, reserve) / active

        want = st.concentrate_on(team, share, cfg.adapt_margin,
                                 cfg.adapt_min_samples, member)
        try:
            _hist = st.held.get((team, member)) or []
            _press = sum(1 for _b, _u in _hist if _u) if (
                _hist and isinstance(_hist[0], tuple)) else 0
            with open("/tmp/unified_debug.jsonl", "a") as _df:
                _df.write(_cohort_json.dumps({
                    "rid": rid, "tok": tokens,
                    "foot": int(sum(st.stack.values())),
                    "share": int(share), "held": len(_hist),
                    "press": _press,
                    "want": None if want is None else sorted(want),
                }) + chr(10))
        except Exception:
            pass
        if want is None:
            # Measuring. Admitting everyone after the stack overflows
            # inserts the sessions the rule is about to bypass; a later
            # prefix match refreshes their LRU recency and evicts the
            # keep-set. While the stack fits, stand aside. Once it does
            # not, keep the lowest member ids that fit in the share — the
            # same placeholder a static quota would have used — so the
            # set we are about to freeze is not competing with junk.
            skip = False
            foot = sum(v for v in st.stack.values() if v > 0)
            if cfg.pool_tokens and tokens > 0 and foot > cfg.pool_tokens:
                st.pressure_ticks[team] = st.pressure_ticks.get(team, 0) + 1
                lost = any(
                    bit == 0
                    for m in st.roster.get(team, ())
                    for bit, under in (st.held.get((team, m)) or [])
                    if under)
                roster_n = len(st.roster.get(team, ())) or 1
                limit = roster_n * cfg.adapt_min_samples
                if lost and st.pressure_ticks[team] <= limit:
                    k = max(1, int(share // tokens))
                    skip = member >= k
        elif not want:
            # Spread: recency is informative, so stand aside -- but still
            # decline a member the pool has never managed to hold, because
            # inserting it only evicts one that would have survived.
            skip = not st.pool_can_hold(team, member, cfg.adapt_min_samples)
        else:
            skip = member not in want
        st.once[rid] = skip
        if len(st.once) > 8192:
            for k in list(st.once)[:4096]:
                del st.once[k]
        return skip

    if cfg.quota <= 0:
        return False
    return member >= cfg.quota
'''

# Two sites, and which is which matters more than it looks.
#
# The finished-request site is the obvious one: `cache_finished_req` decides
# whether a completed request's KV is inserted or freed.
#
# The unfinished site needs splitting. `maybe_cache_unfinished_req` is reached
# from two callers:
#
#   scheduler.py       with chunked=True, carrying a chunked prefill's
#                      bookkeeping -- it inserts the finished chunk and updates
#                      `req.prefix_indices` so the next chunk continues from
#                      it. Skipping this strands the first chunk's KV, neither
#                      in the tree nor freed, and corrupts the lock counts.
#                      Measured: every scheduler rank died with "Prefill out of
#                      memory, available full tokens: 0", a pool full of KV
#                      that nothing owned.
#   batch_result_...   with no chunked flag, when a prefill completes and the
#                      request continues into decode. *This* is what puts the
#                      prompt in the tree.
#
# We first patched only the finished site, and it worked -- while every request
# generated a single token and therefore finished inside its prefill batch.
# With a real 144-token decode the prompt is already in the tree by the time
# the request finishes, so bypassing at the finished site leaves it there and
# the mechanism silently becomes a no-op. Measured: per-member miss rates went
# flat at 95.6 / 95.6 / 93.3 / 91.1 / 95.6% under a quota of 2, where a working
# quota has to show a step. Retention has to be declined at every site that can
# insert, and only the chunk-accounting one is off limits.
EDITS = [
    ('is_insert=is_insert and not getattr(req, "skip_radix_cache_insert", False),',
     'is_insert=is_insert and not _cohort_skip_insert(req),'),
    ('''def maybe_cache_unfinished_req(req: Req, tree_cache: BasePrefixCache, **kwargs):
    if getattr(req, "skip_radix_cache_insert", False):
        return

    tree_cache.cache_unfinished_req(req, **kwargs)''',
     '''def maybe_cache_unfinished_req(req: Req, tree_cache: BasePrefixCache, **kwargs):
    if getattr(req, "skip_radix_cache_insert", False):
        return

    # Decline retention here too, but never on the chunked path: that call is
    # load-bearing accounting, not a retention decision.
    if not kwargs.get("chunked", False) and _cohort_skip_insert(req):
        return

    tree_cache.cache_unfinished_req(req, **kwargs)'''),
]


def read_from_image(image: str) -> str:
    script = (
        'P=$(python3 -c "import sglang,os;print(os.path.dirname(sglang.__file__))"); '
        f'cat $P/{TARGET}'
    )
    out = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "bash", image, "-c", script],
        capture_output=True, text=True, check=True)
    return out.stdout


def patch(src: str) -> str:
    if "_cohort_skip_insert" in src:
        raise SystemExit("source already patched")
    if src.count(ANCHOR) != 1:
        raise SystemExit(f"import anchor not found exactly once: {ANCHOR!r}")
    # Rewrite the call sites before inserting the helper. The helper itself
    # reads `skip_radix_cache_insert`, so inserting it first would make the
    # first edit pattern match twice -- once in real code, once in the helper.
    out = src
    for old, new in EDITS:
        if out.count(old) != 1:
            raise SystemExit(
                f"edit site not found exactly once ({out.count(old)}): {old!r}")
        out = out.replace(old, new, 1)
    out = out.replace(ANCHOR, ANCHOR + HELPER, 1)
    for _, new in EDITS:
        if out.count(new) != 1:
            raise SystemExit(f"rewritten site missing or duplicated: {new!r}")
    if 'getattr(req, "skip_radix_cache_insert", False)' not in out:
        raise SystemExit("helper lost the original skip attribute check")
    compile(out, TARGET, "exec")  # refuse to emit something that cannot import
    return out


class _Req:
    def __init__(self, rid, tokens=10000):
        self.rid = rid
        self.skip_radix_cache_insert = False
        self.origin_input_ids = [0] * tokens


def selftest(tmpdir="/tmp/cohort-selftest"):
    """Exercise the helper in isolation, including a live control-file switch.

    The helper is executed on its own rather than as part of `common.py`,
    whose imports are not available here. What matters is that the predicate
    behaves, and that a control-file edit is picked up without a restart --
    the whole reason for the file.
    """
    os.makedirs(tmpdir, exist_ok=True)
    control = os.path.join(tmpdir, "control.json")

    def write(doc):
        with open(control, "w", encoding="utf-8") as fh:
            json.dump(doc, fh)
        # The reload compares mtime, and a coarse filesystem clock can make
        # two rapid writes indistinguishable.
        os.utime(control, (time.time(), time.time() + _bump[0]))
        _bump[0] += 1

    _bump = [1]
    env = {"COHORT_CONTROL": control, "COHORT_MODE": "group",
           "COHORT_QUOTA": "0", "COHORT_CAPACITY": "24",
           "COHORT_MEMBERS": "5", "COHORT_WINDOW_S": "30"}
    os.environ.update(env)
    write({"mode": "group", "quota": 0})

    ns = {}
    exec(compile(HELPER, "<cohort-helper>", "exec"), ns)
    skip = ns["_cohort_skip_insert"]
    cfg = ns["_cohort_cfg"]

    def force_reload():
        cfg._checked = None

    checks = []

    def check(name, got, want):
        checks.append((name, got, want))

    # Ordinary traffic: engine-generated ids carry no colon, so the predicate
    # must decline to act on them under every mode.
    for mode in ("group", "rotate", "online", "global", "invdist"):
        write({"mode": mode, "quota": 3, "global_cap": 2, "capacity": 24,
               "members": 5, "rotate_period": 2})
        force_reload()
        check(f"plain rid untouched under {mode}",
              skip(_Req("a1b2c3d4e5f6")), False)

    # group: members below the quota are retained, the rest bypassed.
    write({"mode": "group", "quota": 3})
    force_reload()
    check("group m0", skip(_Req("t0:0:5")), False)
    check("group m2", skip(_Req("t0:2:5")), False)
    check("group m3", skip(_Req("t0:3:5")), True)
    check("group m4 other team", skip(_Req("t9:4:0")), True)

    # quota 0 is upstream.
    write({"mode": "group", "quota": 0})
    force_reload()
    check("quota 0 retains m4", skip(_Req("t0:4:5")), False)

    # rotate: the resident subset shifts every rotate_period turns, so a
    # member that is bypassed now is retained later.
    write({"mode": "rotate", "quota": 3, "members": 5, "rotate_period": 2})
    force_reload()
    rot = [skip(_Req(f"t0:4:{turn}")) for turn in range(10)]
    check("rotate m4 sometimes retained", any(not r for r in rot), True)
    check("rotate m4 sometimes bypassed", any(rot), True)
    resident = sum(1 for m in range(5) if not skip(_Req(f"t0:{m}:0")))
    check("rotate keeps quota resident", resident, 3)

    # global: the first `global_cap` distinct sessions freeze in, regardless of
    # which team they belong to.
    write({"mode": "global", "global_cap": 2})
    force_reload()
    check("global 1st admitted", skip(_Req("t0:0:0")), False)
    check("global 2nd admitted", skip(_Req("t1:0:0")), False)
    check("global 3rd bypassed", skip(_Req("t2:0:0")), True)
    check("global 1st still admitted", skip(_Req("t0:0:1")), False)

    # online: nothing is configured but the pool. The budget is the pool less
    # one in-flight request per active team, divided by the active teams, so
    # the quota falls as teams arrive. With 240,000 tokens, 6 teams and uniform
    # 10,000-token members the reserve is 10,000 each, leaving 180,000 to share
    # and 30,000 per team -- three members, not a fourth.
    write({"mode": "online", "pool_tokens": 240000, "window_s": 30})
    force_reload()
    first = {}
    for t in range(6):
        for m in range(5):
            first[(t, m)] = skip(_Req(f"t{t}:{m}:0"))
    # The budget follows the load. Team 0 walked its members while it was the
    # only team active, so it was allowed all five; team 5 arrived once six
    # teams were active and got three. Re-running team 0 now that the fleet is
    # up gives it three as well, which is what makes the quota derived rather
    # than configured.
    check("online team alone keeps m4", first[(0, 4)], False)
    check("online team arriving into a full fleet loses m4",
          first[(5, 4)], True)
    check("online first team follows once the fleet is up",
          skip(_Req("t0:4:1")), True)
    check("online m3 out of budget at 6 teams", skip(_Req("t0:3:1")), True)
    check("online m2 within budget at 6 teams", skip(_Req("t0:2:1")), False)
    # Unequal contexts are the norm; a big member must consume more of the
    # budget than a small one rather than counting as one slot. A 30,000-token
    # lead against a 14,000-token mean leaves a 26,000-token budget, so the
    # lead alone overruns it and the first teammate is already out -- where a
    # session count would have given the team the same number of slots as any
    # other.
    write({"mode": "online", "pool_tokens": 240000, "window_s": 30})
    force_reload()
    big = {}
    for t in range(6):
        for m in range(5):
            big[(t, m)] = skip(_Req(f"t{t}:{m}:0", 30000 if m == 0 else 10000))
    check("online big lead crowds out its own teammates", big[(0, 1)], True)
    check("online always keeps member 0", skip(_Req("t0:0:1", 999999)), False)

    # invdist: a first sighting is admitted optimistically; a reuse distance
    # beyond capacity is bypassed. On a cycle of 40 sessions against a
    # capacity of 24 every distance is 40, so the policy bypasses everything
    # after the first pass -- the degeneracy this baseline exists to show.
    write({"mode": "invdist", "capacity": 24})
    force_reload()
    cycle = [(t, m) for t in range(8) for m in range(5)]
    first = [skip(_Req(f"t{t}:{m}:0")) for t, m in cycle]
    second = [skip(_Req(f"t{t}:{m}:1")) for t, m in cycle]
    check("invdist admits first sighting", any(first), False)
    check("invdist bypasses distance>capacity", all(second), True)

    # adaptive: the same quota, engaged only where recency stops being
    # informative. Two teams are driven with the two measured speaking orders
    # and the policy has to classify them independently and oppositely.
    write({"mode": "adaptive", "quota": 3, "members": 5,
           "adapt_threshold": 0.6, "adapt_window": 40})
    force_reload()
    ring_skips, emp_skips = [], []
    # Team "r" speaks round robin, the shape a serialising harness produces.
    # Team "e" replays the fitted trace distribution, which is dominated by a
    # member speaking again immediately.
    emp_order = [0, 1, 2, 3, 4] + [0, 0, 2, 1, 1, 0, 3, 3, 0, 0,
                                   1, 1, 4, 0, 0, 2, 2, 0, 1, 1] * 4
    for turn in range(len(emp_order)):
        ring_skips.append(skip(_Req(f"r:{turn % 5}:{turn}")))
        emp_skips.append(skip(_Req(f"e:{emp_order[turn]}:{turn}")))
    # Warm-up is honest: neither team is classified until the window holds a
    # full round, so the first turns are admitted either way.
    check("adaptive engages on a ring", any(ring_skips[10:]), True)
    check("adaptive bypasses ring members over quota",
          all(s for s, t in zip(ring_skips[10:], range(10, len(emp_order)))
              if (t % 5) >= 3), True)
    check("adaptive stays out of the way on the measured order",
          any(emp_skips), False)

    # And the classification has to be per team, not per node: a real node
    # serves both harnesses at once, which is the case neither fixed policy
    # handles.
    check("adaptive classifies teams independently",
          any(ring_skips) and not any(emp_skips), True)

    # unified: the same reductions the simulator checks (optimal.py
    # reductions), verified against the code that actually runs in the engine,
    # because the two differ where it matters -- the engine budgets tokens and
    # discovers the roster, and neither is exercised by the simulator.
    #
    # The regime has to be one where the share is a real constraint, or the
    # correct behaviour is to stand aside and the test proves nothing. Five
    # teams of five 10k members against a 200k pool: the reserve for requests
    # in flight takes one member per team, leaving each team a share of 30k,
    # so three of its five members fit and two cannot.
    def unified_run(order, teams=("u0", "u1", "u2", "u3", "u4"),
                    pool=200000, sizes=None, size_at=None):
        write({"mode": "unified", "pool_tokens": pool, "window_s": 3600,
               "adapt_window": 40})
        force_reload()
        # Explicitly, because a reload only resets state when the config
        # actually changed, and these scenarios share one config. In the
        # engine an arm *is* a config change, so this is a property of the
        # test rather than of the mechanism -- but without it the ring's
        # measurements leak into the next scenario and every check passes or
        # fails for the wrong reason.
        ns["_cohort_state"].reset()
        out = {t: [] for t in teams}
        for turn, member in enumerate(order):
            for t in teams:
                n = (size_at(turn, member) if size_at
                     else (sizes or {}).get(member, 10000))
                out[t].append(skip(_Req("%s:%d:%d" % (t, member, turn), n)))
        return out

    ring = [i % 5 for i in range(120)]
    r = unified_run(ring)["u0"]
    check("unified engages on a ring", any(r[20:]), True)
    # Which members are frozen is not the claim -- on a ring the rates are
    # equal, so the subset is arbitrary and that is correct. The claim is that
    # some members are frozen and some are bypassed, i.e. k-of-n rather than
    # all-or-nothing.
    check("unified degenerates to a k-of-n subset on a ring",
          any(r[20:]) and not all(r[20:]), True)

    # The measured order: the dominant mode is a member speaking again
    # immediately, so recency is informative and the rule must stand aside.
    _rng = __import__("random").Random(7)
    emp, _stack = [], list(range(5))
    for _ in range(160):
        i = min(int(_rng.expovariate(1 / 1.6)), 4)
        m = _stack.pop(i)
        _stack.insert(0, m)
        emp.append(m)
    check("unified stands aside on the measured order",
          any(unified_run(emp)["u0"][20:]), False)

    # A star, relabelled so the busy member is *last*. This is the case the
    # shipped `adaptive` rule cannot see at all (its team-level bound reads
    # 0.566 against a 0.6 threshold) and the case a quota keyed to member
    # index gets wrong by construction.
    star = []
    while len(star) < 140:
        for spoke in (1, 2, 3):
            star += [4, spoke]
    # A star team has four sessions, not five, so the same pool leaves it no
    # pressure and standing aside would be right for the wrong reason.
    star_out = unified_run(star, pool=80000)["u0"]
    hub_skipped = [s for s, m in zip(star_out, star) if m == 4][10:]
    check("unified protects the busy member of a star", any(hub_skipped), False)
    spoke_skipped = [s for s, m in zip(star_out, star) if m != 4][20:]
    check("...and gives up some of the quiet ones", any(spoke_skipped), True)

    # And one notch of pressure lower is the case worth pinning down, because
    # it is where a mechanism that likes intervening would intervene for
    # nothing. At a 100k pool LRU already holds the hub and already loses the
    # spokes, so freezing the hub buys exactly what is already happening:
    # both sides of the comparison come to 0.500 and the rule declines to
    # concentrate.
    #
    # It does not follow that it should insert everything. The spokes are ones
    # LRU provably never holds -- `held` is the counterfactual "would a pure
    # LRU pool have kept this", read off the token stack distance, so it is a
    # property of the reference stream and not of our own decision, which is
    # what stops a declined session from being locked out by its own exclusion.
    # Declining them is free here (simulated: 50.7% either way, hub 0%, spokes
    # 100%) and worth 7.5pp on a skewed order at 3.9x. So the check is not
    # "does nothing" but the sharper pair: never give up the member LRU is
    # holding, and do not insert the ones it cannot.
    easy = unified_run(star, pool=100000)["u0"]
    easy_hub = [s for s, m in zip(easy, star) if m == 4][10:]
    easy_spoke = [s for s, m in zip(easy, star) if m != 4][20:]
    check("unified keeps the member LRU already holds", any(easy_hub), False)
    check("...and still declines the ones LRU never holds",
          all(easy_spoke), True)

    # Non-uniform contexts: the share is tokens, so one oversized member has
    # to cost the team more of its budget than a small one.
    big = unified_run(ring, sizes={0: 60000})["u0"]
    check("unified prices the share in tokens, not members",
          sum(big[20:]) > sum(r[20:]), True)

    # Growing contexts cross the pool. Five teams of five, 5k then +4k each
    # turn: 25 * 5k = 125k fits a 200k pool, 25 * 17k = 425k does not. The
    # unfiltered rule prices the fit-regime hits into the later comparison
    # and folds back. After overflow it must engage and stay engaged.
    grow = unified_run(ring, pool=200000,
                       size_at=lambda turn, _m: 5000 + (turn // 5) * 4000)["u0"]
    # turn//5 is the per-member visit index on a 5-cycle.
    # Two insert sites are one reference. Without this the second call
    # records a would-hold at distance 0 and the live engine never
    # concentrates on a ring, while the one-call replay does.
    ns["_cohort_state"].reset()
    write({"mode": "unified", "pool_tokens": 200000, "window_s": 3600})
    force_reload()
    ns["_cohort_state"].reset()
    a = skip(_Req("u0:0:0", 10000))
    b = skip(_Req("u0:0:0", 10000))
    check("two insert sites of one request agree", a == b, True)
    skip(_Req("u0:0:1", 10000))
    skip(_Req("u0:0:1", 10000))
    _h = ns["_cohort_state"].held.get(("u0", 0), [])
    check("...and record one would-hold, not two", len(_h) == 1, True)

    ns["_cohort_state"].reset()
    write({"mode": "unified", "pool_tokens": 25000, "window_s": 3600})
    force_reload()
    ns["_cohort_state"].reset()
    check("measuring admits the first member while the stack fits",
          skip(_Req("m:0:0", 10000)), False)
    skip(_Req("m:1:0", 10000))
    skip(_Req("m:2:0", 10000))
    skip(_Req("m:0:1", 10000))
    check("measuring under pressure does not flood m4",
          skip(_Req("m:4:1", 10000)), True)

    check("unified stands aside while the stack still fits",
          any(grow[:5]), False)
    check("unified engages after the stack overflows",
          any(grow[40:]), True)
    check("...and does not fold back once it has",
          any(grow[-15:]), True)

    # The control file is the reason this patch exists in this form: a change
    # must take effect in a running process.
    write({"mode": "group", "quota": 1})
    force_reload()
    check("live switch to quota 1", skip(_Req("t0:2:0")), True)
    write({"mode": "group", "quota": 5})
    force_reload()
    check("live switch to quota 5", skip(_Req("t0:2:0")), False)
    check("generation advanced", cfg.generation >= 8, True)

    bad = [(n, g, w) for n, g, w in checks if g != w]
    for name, got, want in checks:
        print(f"  {'ok  ' if got == want else 'FAIL'} {name}: "
              f"got {got!r} want {want!r}")
    print(f"\n{len(checks) - len(bad)}/{len(checks)} checks passed")
    for key in env:
        os.environ.pop(key, None)
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="sglang-kimi-k3:d6e7288627be")
    ap.add_argument("--src", help="read from this file instead of the image")
    ap.add_argument("--out", default="/tmp/sgl-patch")
    ap.add_argument("--selftest", action="store_true",
                    help="exercise the helper without touching the image")
    cfg = ap.parse_args()

    if cfg.selftest:
        return selftest()

    src = (open(cfg.src, encoding="utf-8").read() if cfg.src
           else read_from_image(cfg.image))
    patched = patch(src)

    os.makedirs(cfg.out, exist_ok=True)
    dest = os.path.join(cfg.out, "common.py")
    with open(dest, "w", encoding="utf-8") as fh:
        fh.write(patched)
    print(f"wrote {dest}  ({len(src)} -> {len(patched)} bytes)")
    print(f"bind-mount it over  <sglang>/{TARGET}")
    print("COHORT_QUOTA unset or 0 reproduces upstream behaviour exactly")


if __name__ == "__main__":
    sys.exit(main())
