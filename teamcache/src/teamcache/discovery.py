"""Optional team/member inference from prompt fingerprints.

Installing TeamCache does not rewrite traffic that lacks a request id. Use this
module on a gateway or offline, then write `team:member:turn` into the request
id the engine sees. It is not wired into the SGLang patch.

A new prompt joins a member lineage when it strictly extends that lineage's
previous `prefix_sha` chain. System/root hashes that start two independent
lineages are marked as templates and never used to join. Handoff edges come
from a long non-template prefix fingerprint bound to exactly one other lineage,
a sufficiently long globally unique output length, or a 2-message prompt whose
user blob grew by exactly one other lineage's (user + output) within a small
wrapper slack. Uncertain requests stay singletons.
"""

from __future__ import annotations

import heapq
import threading
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field

Identity = namedtuple("Identity", "team_id member_id lineage_new n_handoffs")


def common_prefix_len(a, b):
    """Return the number of equal leading items in two sequences."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class _UnionFind:
    def __init__(self):
        self.p = {}

    def add(self, x):
        self.p.setdefault(x, x)

    def find(self, x):
        self.add(x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if ra < rb:
            self.p[rb] = ra
            return ra
        self.p[ra] = rb
        return rb

    def prune(self, live):
        """Drop expired nodes while preserving components among live nodes."""
        groups = defaultdict(list)
        for node in live:
            groups[self.find(node)].append(node)
        self.p = {}
        for nodes in groups.values():
            root = min(nodes)
            self.p[root] = root
            for node in nodes:
                self.p[node] = root


@dataclass
class _Lineage:
    lid: int
    tenant: str
    model: str
    chain: tuple
    sizes: tuple
    roles: tuple
    last_t: float
    last_out: int = 0
    n_obs: int = 1
    emitted_sizes: set = field(default_factory=set)
    roots: set = field(default_factory=set)
    fingerprints: set = field(default_factory=set)
    root: str = ""
    last_output_seq: int = -1


def align_spans(row):
    """Return (fingerprints, sizes, roles, out_len) aligned at the same indices.

    OpenAI-proxy traces hash messages only, so `prefix_sha[0]` is the system
    message. Anthropic-proxy traces hash the system blob first, then messages,
    so the chain is one longer than `msg_roles`.
    """
    chain = list(row.get("prefix_sha") or [])
    roles = list(row.get("msg_roles") or [])
    sizes = list(row.get("msg_sizes") or row.get("msg_chars") or [])
    n_msg = row.get("n_messages")
    if n_msg is None:
        n_msg = len(roles) or len(sizes)
    sys_chars = row.get("system_chars")
    if (
        sys_chars is not None
        and chain
        and n_msg is not None
        and len(chain) == n_msg + 1
    ):
        sizes = [int(sys_chars)] + list(sizes)
        roles = ["system"] + list(roles)
    n = len(chain)
    if len(sizes) < n:
        sizes = sizes + [0] * (n - len(sizes))
    if len(roles) < n:
        roles = roles + ["?"] * (n - len(roles))
    out = row.get("out_text_len")
    if out is None:
        out = row.get("out_chars") or 0
    return chain, sizes[:n], roles[:n], int(out or 0)


class TeamDiscovery:
    """Online, conservative member/team inference from prompt fingerprints."""

    def __init__(self, min_span_chars=1024, window_s=7200.0, stuff_slack=64):
        self.min_span = int(min_span_chars)
        self.window_s = float(window_s)
        # CrewAI concatenates the previous user blob and completion into one
        # new user message, plus a few wrapper characters (14–43 on our traces).
        # That is still "incorporates a distinctive history span"; the slack
        # is the wrapper, not a paraphrase.
        self.stuff_slack = int(stuff_slack)
        self._next_lid = 0
        self._next_seq = 0
        self._lineages = {}
        self._uf = _UnionFind()
        self._lock = threading.RLock()
        self._templates = set()
        self._scope_lids = defaultdict(set)  # (tenant, model) -> lids
        self._expiry = defaultdict(list)     # (tenant, model) -> (last_t, lid)
        # A cumulative prefix hash identifies the whole chain up to that
        # message. This turns extension lookup from O(all lineages) into
        # O(messages in this prompt).
        self._chain_lids = defaultdict(set)  # (tenant, model, length, tail) -> lids
        self._root_lids = defaultdict(set)  # (tenant, model, root_fp) -> lids
        self._fp_lids = defaultdict(set)    # (tenant, model, fp) -> lids
        self._out_lids = defaultdict(set)   # (tenant, model, size) -> lids
        self._stuffed_lids = defaultdict(set)  # (tenant, model, expected size) -> lids
        self._stuffed_expect = {}           # lid -> current expected size
        self._poisoned_out = set()          # (tenant, model, size)
        self._stats = {"requests": 0, "extension_probes": 0, "expired": 0}
        self._forgotten_since_compact = 0

    def _scope(self, tenant, model):
        return (tenant or "_", model or "_")

    def _expire(self, tenant, model, now):
        cutoff = now - self.window_s
        heap = self._expiry[(tenant, model)]
        while heap and heap[0][0] < cutoff:
            last_t, lid = heapq.heappop(heap)
            lineage = self._lineages.get(lid)
            if lineage is None or lineage.last_t != last_t:
                continue
            self._forget(lid)
            self._stats["expired"] += 1

    @staticmethod
    def _chain_key(tenant, model, chain):
        if not chain:
            return None
        return (tenant, model, len(chain), chain[-1])

    @staticmethod
    def _discard(index, key, lid):
        owners = index.get(key)
        if owners is None:
            return
        owners.discard(lid)
        if not owners:
            index.pop(key, None)

    def _forget(self, lid):
        L = self._lineages.pop(lid, None)
        if L is None:
            return
        scope = (L.tenant, L.model)
        self._scope_lids[scope].discard(lid)
        self._discard(self._chain_lids,
                      self._chain_key(L.tenant, L.model, L.chain), lid)
        for root in L.roots:
            self._discard(self._root_lids, (L.tenant, L.model, root), lid)
        for fp in L.fingerprints:
            self._discard(self._fp_lids, (L.tenant, L.model, fp), lid)
        for size in L.emitted_sizes:
            self._discard(self._out_lids, (L.tenant, L.model, size), lid)
        expect = self._stuffed_expect.pop(lid, None)
        if expect is not None:
            self._discard(self._stuffed_lids,
                          (L.tenant, L.model, expect), lid)
        self._forgotten_since_compact += 1
        if self._forgotten_since_compact >= max(1024, len(self._lineages)):
            self._uf.prune(self._lineages)
            self._forgotten_since_compact = 0

    def _new_lineage(self, tenant, model, chain, sizes, roles, t):
        lid = self._next_lid
        self._next_lid += 1
        L = _Lineage(
            lid=lid, tenant=tenant, model=model,
            chain=tuple(chain), sizes=tuple(sizes), roles=tuple(roles),
            last_t=t, root=chain[0] if chain else "",
        )
        self._lineages[lid] = L
        self._uf.add(lid)
        self._scope_lids[(tenant, model)].add(lid)
        key = self._chain_key(tenant, model, chain)
        if key is not None:
            self._chain_lids[key].add(lid)
        heapq.heappush(self._expiry[(tenant, model)], (t, lid))
        return L

    def _mark_template(self, tenant, model, root):
        if root:
            self._templates.add((tenant, model, root))

    def _is_template(self, tenant, model, root):
        return (tenant, model, root) in self._templates

    def _bind_root(self, tenant, model, root, lid):
        owners = self._root_lids[(tenant, model, root)]
        owners.add(lid)
        if len(owners) > 1:
            self._mark_template(tenant, model, root)

    def _score_extension(self, L, sizes):
        score = 0
        size_set = set(sizes)
        if L.last_out and L.last_out in size_set:
            score += 2
        score += sum(1 for sz in L.emitted_sizes if sz in size_set)
        return score

    def _choose_extension(self, cands, sizes):
        """cands is a list of (shared, lineage). Unique longest wins; ties
        break on distinctive output sizes already recorded for that lineage."""
        if not cands:
            return None
        best_shared = max(s for s, _ in cands)
        top = [L for s, L in cands if s == best_shared]
        if len(top) == 1:
            return top[0]
        scored = [(self._score_extension(L, sizes), L) for L in top]
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored[0][0] > 0 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
            return scored[0][1]
        return None

    def _extension_candidates(self, tenant, model, chain):
        # prefix_sha is cumulative, so (prefix length, last hash) identifies
        # an entire prompt prefix without copying or comparing the full tuple.
        for length in range(len(chain) - 1, 0, -1):
            self._stats["extension_probes"] += 1
            lids = self._chain_lids.get((tenant, model, length, chain[length - 1]))
            if lids:
                return [
                    (length, self._lineages[lid])
                    for lid in lids
                    if lid in self._lineages
                ]
        return []

    def _begin_locked(self, row, tenant, model):
        chain, sizes, roles, _ = align_spans(row)
        t = float(row["_t"] if row.get("_t") is not None else (row.get("t_recv") or 0.0))
        model = row.get("model_in") or row.get("model_out") or model
        tenant, model = self._scope(tenant, model)
        self._expire(tenant, model, t)
        self._stats["requests"] += 1
        seq = self._next_seq
        self._next_seq += 1

        if not chain:
            chosen = self._new_lineage(tenant, model, [], [], [], t)
            return Identity(team_id=f"t{chosen.lid}", member_id=chosen.lid,
                            lineage_new=True, n_handoffs=0), seq

        cands = self._extension_candidates(tenant, model, chain)
        chosen = self._choose_extension(cands, sizes)

        lineage_new = False
        if chosen is None:
            root = chain[0]
            joined = None
            if (
                roles
                and roles[0] == "system"
                and not self._is_template(tenant, model, root)
            ):
                owners = [
                    self._lineages[lid]
                    for lid in self._root_lids[(tenant, model, root)]
                    if lid in self._lineages
                ]
                long_private = max(sizes[1:], default=0) >= self.min_span
                # CrewAI (and similar) stuffs history into one user message, so
                # the chain never grows past length 2. Join through a distinctive
                # system hash only in that shape; a longer chain is AutoGen-style
                # and must extend a private-history prefix instead.
                stuffed = len(chain) == 2 and len(owners) == 1 and len(owners[0].chain) == 2
                if stuffed and long_private:
                    joined = owners[0]
                elif len(owners) >= 1 and not (stuffed and long_private):
                    # A second independent root with the same system prompt:
                    # treat the hash as a template and keep a singleton.
                    self._mark_template(tenant, model, root)
            if joined is not None:
                chosen = joined
            else:
                chosen = self._new_lineage(tenant, model, chain, sizes, roles, t)
                lineage_new = True

        lid = chosen.lid
        n_handoffs = self._handoffs(chosen, tenant, model, chain, sizes, roles)
        n_handoffs += self._stuffed_handoff(chosen, tenant, model, row, sizes)

        old_chain_key = self._chain_key(tenant, model, chosen.chain)
        self._discard(self._chain_lids, old_chain_key, lid)
        chosen.chain = tuple(chain)
        chosen.sizes = tuple(sizes)
        chosen.roles = tuple(roles)
        chosen.last_t = t
        chosen.n_obs += 0 if lineage_new else 1
        new_chain_key = self._chain_key(tenant, model, chosen.chain)
        if new_chain_key is not None:
            self._chain_lids[new_chain_key].add(lid)
        heapq.heappush(self._expiry[(tenant, model)], (t, lid))
        self._index_spans(chosen, tenant, model, chain, sizes, roles, 0)
        self._refresh_stuffed(chosen)
        team = self._uf.find(lid)
        return (Identity(team_id=f"t{team}", member_id=lid,
                         lineage_new=lineage_new, n_handoffs=n_handoffs), seq)

    def begin(self, row, tenant="_", model="_"):
        """Identify a request before it is sent; return ``(identity, ticket)``.

        The ticket makes response completion safe when requests finish out of
        order. Gateways should call :meth:`record_output` with it.
        """
        with self._lock:
            return self._begin_locked(row, tenant, model)

    def _record_output_locked(self, member_id, out, ticket=None):
        lineage = self._lineages.get(int(member_id))
        out = int(out or 0)
        if lineage is None:
            return False
        if out >= self.min_span:
            lineage.emitted_sizes.add(out)
            key = (lineage.tenant, lineage.model, out)
            owners = self._out_lids[key]
            owners.add(lineage.lid)
            if len(owners) > 1:
                self._poisoned_out.add(key)
        seq = self._next_seq if ticket is None else int(ticket)
        if seq >= lineage.last_output_seq:
            lineage.last_output_seq = seq
            lineage.last_out = out
            self._refresh_stuffed(lineage)
        return True

    def record_output(self, member_id, out, ticket=None):
        """Complete a request identified by :meth:`begin` (idempotent by ticket)."""
        with self._lock:
            return self._record_output_locked(member_id, out, ticket)

    def observe(self, row, tenant="_", model="_"):
        """One-shot offline API; online gateways use ``begin`` then completion."""
        with self._lock:
            _, _, _, out = align_spans(row)
            ident, ticket = self._begin_locked(row, tenant, model)
            self._record_output_locked(ident.member_id, out, ticket)
            return ident

    def _handoffs(self, chosen, tenant, model, chain, sizes, roles):
        n = 0
        lid = chosen.lid
        seen = set()
        for i, fp in enumerate(chain):
            role = roles[i] if i < len(roles) else "?"
            size = sizes[i] if i < len(sizes) else 0
            if role == "system":
                continue
            if size < self.min_span:
                continue
            others = {x for x in self._fp_lids[(tenant, model, fp)]
                      if x != lid and x in self._lineages}
            if len(others) == 1:
                other = next(iter(others))
                if other not in seen:
                    self._uf.union(lid, other)
                    seen.add(other)
                    n += 1
            key = (tenant, model, size)
            if size >= self.min_span and key not in self._poisoned_out:
                out_others = {x for x in self._out_lids[key]
                              if x != lid and x in self._lineages}
                if len(out_others) == 1:
                    other = next(iter(out_others))
                    if other not in seen:
                        self._uf.union(lid, other)
                        seen.add(other)
                        n += 1
        return n

    def _stuffed_handoff(self, chosen, tenant, model, row, sizes):
        """Join when a 2-message prompt's user blob grew by another lineage's output.

        AutoGen puts that output in its own message, so exact size matching in
        `_handoffs` already fires. CrewAI does not: the previous completion is
        spliced into one user message. Matching `user ≈ prev_user + prev_out`
        with a small unique slack recovers the first-round ring. Later CrewAI
        turns truncate context and no longer satisfy the recurrence; those
        misses are in the data, but the first-round edges already connect the
        team.
        """
        roles = row.get("msg_roles") or []
        n_msg = row.get("n_messages")
        if n_msg is None:
            n_msg = len(roles)
        if n_msg != 2 or roles[:1] != ["system"] or self.stuff_slack < 0:
            return 0
        user_sz = sizes[-1] if len(sizes) >= 2 else 0
        if user_sz < self.min_span:
            return 0
        matches = set()
        for expected in range(max(0, user_sz - self.stuff_slack),
                              user_sz + self.stuff_slack + 1):
            matches.update(self._stuffed_lids.get((tenant, model, expected), ()))
        matches.discard(chosen.lid)
        matches.intersection_update(self._lineages)
        if len(matches) != 1:
            return 0
        other = next(iter(matches))
        if self._uf.find(other) == self._uf.find(chosen.lid):
            return 0
        self._uf.union(chosen.lid, other)
        return 1

    def _refresh_stuffed(self, lineage):
        old = self._stuffed_expect.pop(lineage.lid, None)
        if old is not None:
            self._discard(
                self._stuffed_lids,
                (lineage.tenant, lineage.model, old),
                lineage.lid,
            )
        if len(lineage.sizes) < 2 or lineage.last_out < self.min_span:
            return
        expect = int(lineage.sizes[-1]) + int(lineage.last_out)
        self._stuffed_expect[lineage.lid] = expect
        self._stuffed_lids[(lineage.tenant, lineage.model, expect)].add(lineage.lid)

    def _index_spans(self, L, tenant, model, chain, sizes, roles, out):
        lid = L.lid
        if chain:
            self._bind_root(tenant, model, chain[0], lid)
            L.roots.add(chain[0])
        for i, fp in enumerate(chain):
            role = roles[i] if i < len(roles) else "?"
            size = sizes[i] if i < len(sizes) else 0
            if role == "system" or size < self.min_span:
                continue
            self._fp_lids[(tenant, model, fp)].add(lid)
            L.fingerprints.add(fp)
        if out >= self.min_span:
            key = (tenant, model, out)
            owners = self._out_lids[key]
            owners.add(lid)
            if len(owners) > 1:
                self._poisoned_out.add(key)

    def team_of(self, member_id):
        with self._lock:
            return f"t{self._uf.find(member_id)}"

    def turn_of(self, member_id):
        """Return the zero-based observation number for a live lineage."""
        with self._lock:
            lineage = self._lineages.get(int(member_id))
            if lineage is None:
                raise KeyError(member_id)
            return lineage.n_obs - 1

    def stats(self):
        """Small operational snapshot; contains counts, never prompt content."""
        with self._lock:
            return {
                **self._stats,
                "live_lineages": len(self._lineages),
                "live_scopes": sum(bool(v) for v in self._scope_lids.values()),
                "templates": len(self._templates),
            }


def self_test():
    """Run dependency-free lineage and handoff checks."""
    d = TeamDiscovery(min_span_chars=50)

    def req(t, shas, sizes, roles, out):
        row = {
            "t_recv": t,
            "prefix_sha": shas,
            "msg_sizes": sizes,
            "msg_roles": roles,
            "out_text_len": out,
        }
        return d.observe(row)

    # Member A, then B incorporates A's long output, then A extends its chain.
    a0 = req(1.0, ["sysA", "sysA+u"], [80, 40], ["system", "user"], 120)
    b0 = req(2.0, ["sysB", "sysB+u", "sysB+u+a"], [80, 40, 120],
             ["system", "user", "user"], 130)
    a1 = req(3.0, ["sysA", "sysA+u", "sysA+u+a", "sysA+u+a+b"],
             [80, 40, 120, 130], ["system", "user", "user", "user"], 90)
    assert a1.member_id == a0.member_id, (a0, a1)
    assert b0.member_id != a0.member_id
    assert d.team_of(a0.member_id) == d.team_of(b0.member_id), "ring should merge"

    # A second team with the same system hashes but different output lengths.
    d2 = TeamDiscovery(min_span_chars=50)
    t1a = d2.observe({"t_recv": 1, "prefix_sha": ["S0", "S0u"],
                      "msg_sizes": [80, 40], "msg_roles": ["system", "user"],
                      "out_text_len": 200})
    t2a = d2.observe({"t_recv": 1.1, "prefix_sha": ["S0", "S0u"],
                      "msg_sizes": [80, 40], "msg_roles": ["system", "user"],
                      "out_text_len": 300})
    t1b = d2.observe({"t_recv": 2, "prefix_sha": ["S1", "S1u", "x"],
                      "msg_sizes": [80, 40, 200],
                      "msg_roles": ["system", "user", "user"],
                      "out_text_len": 210})
    t2b = d2.observe({"t_recv": 2.1, "prefix_sha": ["S1", "S1u", "y"],
                      "msg_sizes": [80, 40, 300],
                      "msg_roles": ["system", "user", "user"],
                      "out_text_len": 310})
    # First prompts of the two teams must not collapse into one member.
    assert t1a.member_id != t2a.member_id
    assert d2.team_of(t1a.member_id) == d2.team_of(t1b.member_id)
    assert d2.team_of(t2a.member_id) == d2.team_of(t2b.member_id)
    assert d2.team_of(t1a.member_id) != d2.team_of(t2a.member_id)

    # Unrelated short prompt stays a singleton.
    lonely = d2.observe({"t_recv": 9, "prefix_sha": ["zz"],
                         "msg_sizes": [12], "msg_roles": ["user"],
                         "out_text_len": 3})
    assert d2.team_of(lonely.member_id) != d2.team_of(t1a.member_id)

    # CrewAI: one user blob grows by the previous completion plus a wrapper.
    d3 = TeamDiscovery(min_span_chars=50, stuff_slack=64)
    c0 = d3.observe({"t_recv": 1, "n_messages": 2,
                     "prefix_sha": ["C0", "C0u"], "msg_sizes": [80, 100],
                     "msg_roles": ["system", "user"], "out_text_len": 200})
    c1 = d3.observe({"t_recv": 2, "n_messages": 2,
                     "prefix_sha": ["C1", "C1u"], "msg_sizes": [80, 314],
                     "msg_roles": ["system", "user"], "out_text_len": 150})
    assert c0.member_id != c1.member_id
    assert d3.team_of(c0.member_id) == d3.team_of(c1.member_id), "stuffed handoff"
    print("self-test ok")


if __name__ == "__main__":
    self_test()
