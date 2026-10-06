"""TeamCache admission predicate injected into SGLang `common.py`.

This module holds the exact helper text that `teamcache install` splices
into `sglang/srt/mem_cache/common.py`. Keep the injected identifiers
(`_cohort_*`) so the engine patch sites stay a mechanical edit.

Default mode is `unified` (the paper's TeamCache policy). `pool_tokens`
must be set to the prefix-pool capacity in tokens; when it is 0 the
predicate does not intervene.
"""

from __future__ import annotations

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
        self.mode = env("COHORT_MODE", "unified")
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
        self.team_seen = _cohort_OrderedDict()  # team -> monotonic last seen
        self.size = {}             # online mode: session -> prompt tokens
        self.size_tokens = 0
        self.size_count = 0
        self.last_ref = {}         # invdist mode: session -> reference index
        self.refs = 0
        self.order = {}            # adaptive mode: team -> recent member ids
        self.ranks = {}            # adaptive mode: team -> recent recency ranks
        # unified mode. `stack` is the engine's own LRU order, which is what
        # lets the policy ask "would LRU have held this" instead of being told
        # which topology it is serving.
        self.stack = _cohort_OrderedDict()   # session -> prompt tokens
        self.stack_tokens = 0       # avoid summing the full stack on each request
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
        self.once = _cohort_OrderedDict()
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
                footprint = self.stack_tokens
                hist = self.held.setdefault(key, [])
                hist.append((1 if above + prev <= pool_tokens else 0,
                             footprint > pool_tokens))
                if len(hist) > window:
                    del hist[:-window]
            if prev is not None:
                self.stack_tokens -= prev
            self.stack[key] = tokens
            self.stack_tokens += tokens
            self.stack.move_to_end(key)
            # Bounded: a session that has not spoken in thousands of turns
            # cannot affect any decision, and the scan above is linear.
            while len(self.stack) > 4096:
                _, dropped = self.stack.popitem(last=False)
                self.stack_tokens -= dropped
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

    def mark_team(self, team):
        now = _cohort_time.monotonic()
        self.team_seen[team] = now
        self.team_seen.move_to_end(team)

    def active_teams(self, window_s):
        now = _cohort_time.monotonic()
        cutoff = now - window_s
        while self.team_seen:
            team, stamp = next(iter(self.team_seen.items()))
            if stamp >= cutoff:
                break
            del self.team_seen[team]
        return len(self.team_seen) or 1


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
        st.mark_team(team)
        ids = getattr(req, "origin_input_ids", None)
        size_key = (team, member)
        old_size = st.size.get(size_key, 0)
        new_size = len(ids) if ids else 0
        if old_size > 0:
            st.size_tokens -= old_size
        elif new_size > 0:
            st.size_count += 1
        st.size[size_key] = new_size
        st.size_tokens += new_size
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
        reserve = (st.size_tokens / st.size_count) if st.size_count else 0
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
            answer = st.once[rid]
            st.once.move_to_end(rid)
            return answer
        st.mark_team(team)
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
        reserve = (st.stack_tokens / len(st.stack)) if st.stack else 0
        share = max(cfg.pool_tokens - active * reserve, reserve) / active

        want = st.concentrate_on(team, share, cfg.adapt_margin,
                                 cfg.adapt_min_samples, member)
        if want is None:
            # Measuring. Admitting everyone after the stack overflows
            # inserts the sessions the rule is about to bypass; a later
            # prefix match refreshes their LRU recency and evicts the
            # keep-set. While the stack fits, stand aside. Once it does
            # not, keep the lowest member ids that fit in the share — the
            # same placeholder a static quota would have used — so the
            # set we are about to freeze is not competing with junk.
            skip = False
            foot = st.stack_tokens
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
            while len(st.once) > 4096:
                st.once.popitem(last=False)
        return skip

    if cfg.quota <= 0:
        return False
    return member >= cfg.quota
'''
