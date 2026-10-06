#!/usr/bin/env python3
"""Prompt-lineage team / member discovery (design §Team Discovery).

The serving fast path still reads `team:member:turn` when a harness exports it.
This module is the compatibility path: infer those keys from token relationships
already present in the prompt, without client-side request rewriting.

Traces do not retain raw message text. The available exact fingerprints are the
cumulative `prefix_sha` chain at message boundaries. A new prompt joins a member
lineage when it strictly extends that lineage's previous chain. System / root
hashes that start two independent lineages are marked as templates and never
used to join. Handoff edges come from (1) a long non-template prefix fingerprint
already bound to exactly one other lineage, (2) a sufficiently long output
length that is globally unique in the tenant-scoped window, and (3) a
CrewAI-style 2-message prompt whose user blob grew by exactly one other
lineage's `(user + output)` within a small wrapper slack. Uncertain
requests stay singletons.

Usage:
    python team_discovery.py --self-test
    python team_discovery.py --traces-dir traces
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, namedtuple
from dataclasses import dataclass, field
from pathlib import Path


Identity = namedtuple("Identity", "team_id member_id lineage_new n_handoffs")


def common_prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def adjusted_rand_index(labels_a, labels_b):
    n = len(labels_a)
    if n < 2:
        return 1.0
    ta, tb = Counter(labels_a), Counter(labels_b)
    nij = Counter(zip(labels_a, labels_b))
    sum_a = sum(v * (v - 1) // 2 for v in ta.values())
    sum_b = sum(v * (v - 1) // 2 for v in tb.values())
    sum_ab = sum(v * (v - 1) // 2 for v in nij.values())
    comb_n = n * (n - 1) // 2
    if comb_n == 0:
        return 1.0
    expected = (sum_a * sum_b) / comb_n
    max_ = 0.5 * (sum_a + sum_b)
    if max_ == expected:
        return 1.0 if sum_ab == sum_a else 0.0
    return (sum_ab - expected) / (max_ - expected)


def pair_prf(labels_pred, labels_gt):
    """Pairwise same-cluster precision / recall / F1 (permutation-invariant)."""
    n = len(labels_pred)
    tp = fp = fn = 0
    for i in range(n):
        for j in range(i + 1, n):
            same_p = labels_pred[i] == labels_pred[j]
            same_g = labels_gt[i] == labels_gt[j]
            if same_p and same_g:
                tp += 1
            elif same_p:
                fp += 1
            elif same_g:
                fn += 1
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    f1 = (2 * prec * rec / (prec + rec)) if prec + rec else 0.0
    return prec, rec, f1


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
    root: str = ""


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
        self._lineages = {}
        self._uf = _UnionFind()
        self._templates = set()
        self._root_lids = defaultdict(set)  # (tenant, model, root_fp) -> lids
        self._fp_lids = defaultdict(set)    # (tenant, model, fp) -> lids
        self._out_lids = defaultdict(set)   # (tenant, model, size) -> lids
        self._poisoned_out = set()          # (tenant, model, size)

    def _scope(self, tenant, model):
        return (tenant or "_", model or "_")

    def _expire(self, tenant, model, now):
        cutoff = now - self.window_s
        dead = [
            lid for lid, L in self._lineages.items()
            if L.tenant == tenant and L.model == model and L.last_t < cutoff
        ]
        for lid in dead:
            self._forget(lid)

    def _forget(self, lid):
        L = self._lineages.pop(lid, None)
        if L is None:
            return
        key = (L.tenant, L.model)
        self._root_lids[(key[0], key[1], L.root)].discard(lid)
        drop_fp = [k for k, s in self._fp_lids.items() if lid in s]
        for k in drop_fp:
            self._fp_lids[k].discard(lid)
        drop_out = [k for k, s in self._out_lids.items() if lid in s]
        for k in drop_out:
            self._out_lids[k].discard(lid)

    def _new_lineage(self, tenant, model, chain, sizes, roles, t, out):
        lid = self._next_lid
        self._next_lid += 1
        L = _Lineage(
            lid=lid, tenant=tenant, model=model,
            chain=tuple(chain), sizes=tuple(sizes), roles=tuple(roles),
            last_t=t, last_out=out, root=chain[0] if chain else "",
        )
        if out >= self.min_span:
            L.emitted_sizes.add(out)
        self._lineages[lid] = L
        self._uf.add(lid)
        return L

    def _live(self, tenant, model):
        return [L for L in self._lineages.values()
                if L.tenant == tenant and L.model == model]

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

    def observe(self, row, tenant="_", model="_"):
        chain, sizes, roles, out = align_spans(row)
        if not chain:
            return Identity(team_id=f"t{self._next_lid}", member_id=self._next_lid,
                            lineage_new=True, n_handoffs=0)
        t = float(row["_t"] if row.get("_t") is not None else (row.get("t_recv") or 0.0))
        model = row.get("model_in") or row.get("model_out") or model
        tenant, model = self._scope(tenant, model)
        self._expire(tenant, model, t)

        cands = []
        for L in self._live(tenant, model):
            shared = common_prefix_len(L.chain, chain)
            if shared == len(L.chain) and len(chain) > len(L.chain):
                cands.append((shared, L))
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
                chosen = self._new_lineage(tenant, model, chain, sizes, roles, t, out)
                lineage_new = True

        lid = chosen.lid
        n_handoffs = self._handoffs(chosen, tenant, model, chain, sizes, roles)
        n_handoffs += self._stuffed_handoff(chosen, tenant, model, row, sizes)

        chosen.chain = tuple(chain)
        chosen.sizes = tuple(sizes)
        chosen.roles = tuple(roles)
        chosen.last_t = t
        chosen.last_out = out
        chosen.n_obs += 0 if lineage_new else 1
        if out >= self.min_span:
            chosen.emitted_sizes.add(out)
        self._index_spans(chosen, tenant, model, chain, sizes, roles, out)
        team = self._uf.find(lid)
        return Identity(team_id=f"t{team}", member_id=lid,
                        lineage_new=lineage_new, n_handoffs=n_handoffs)

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
        matches = []
        for L in self._live(tenant, model):
            if L.lid == chosen.lid or L.last_out < self.min_span:
                continue
            if len(L.sizes) < 2:
                continue
            expect = L.sizes[-1] + L.last_out
            if abs(user_sz - expect) <= self.stuff_slack:
                matches.append(L.lid)
        if len(set(matches)) != 1:
            return 0
        other = matches[0]
        if self._uf.find(other) == self._uf.find(chosen.lid):
            return 0
        self._uf.union(chosen.lid, other)
        return 1

    def _index_spans(self, L, tenant, model, chain, sizes, roles, out):
        lid = L.lid
        if chain:
            self._bind_root(tenant, model, chain[0], lid)
        for i, fp in enumerate(chain):
            role = roles[i] if i < len(roles) else "?"
            size = sizes[i] if i < len(sizes) else 0
            if role == "system" or size < self.min_span:
                continue
            self._fp_lids[(tenant, model, fp)].add(lid)
        if out >= self.min_span:
            key = (tenant, model, out)
            owners = self._out_lids[key]
            owners.add(lid)
            if len(owners) > 1:
                self._poisoned_out.add(key)

    def team_of(self, member_id):
        return f"t{self._uf.find(member_id)}"


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def recover_sessions(rows):
    sessions = []
    for row in rows:
        chain = row.get("prefix_sha") or []
        best, best_len = None, 0
        for sess in sessions:
            prev = sess["chain"]
            shared = common_prefix_len(prev, chain)
            if shared == len(prev) and shared > best_len:
                best, best_len = sess, shared
        if best is None:
            best = {"rows": [], "chain": chain, "id": len(sessions)}
            sessions.append(best)
        best["rows"].append(row)
        best["chain"] = chain
    return sessions


def overlay_rows(file_rows, filter_fn=None):
    """Overlay files from t=0 so concurrent teams share an activity window."""
    mixed = []
    for file_id, rows in file_rows:
        usable = [r for r in rows if r.get("prefix_sha")]
        if filter_fn:
            usable = [r for r in usable if filter_fn(r)]
        if not usable:
            continue
        usable = sorted(usable, key=lambda r: r.get("t_recv") or 0.0)
        t0 = usable[0].get("t_recv") or 0.0
        for seq, r in enumerate(usable):
            item = dict(r)
            item["_file"] = file_id
            item["_t"] = (r.get("t_recv") or 0.0) - t0
            item["_seq"] = seq
            mixed.append(item)
    mixed.sort(key=lambda r: (r["_t"], r["_file"], r["_seq"]))
    return mixed


def _session_id(row):
    md = row.get("metadata") or {}
    if not isinstance(md, dict):
        return None
    uid = md.get("user_id")
    if not uid:
        return None
    try:
        parsed = json.loads(uid)
        return parsed.get("session_id") or uid
    except (TypeError, json.JSONDecodeError):
        return uid


def _explicit_multi_team(file_rows):
    """e2e traces store many teams in one file; collection traces all say t0."""
    for _, rows in file_rows:
        teams = {r.get("team") for r in rows if r.get("team") not in (None, "")}
        if len(teams) > 1:
            return True
    return False


def attach_ground_truth(file_rows):
    """GT team = e2e `team` field when a file actually has many teams, else file.

    GT member = explicit member when present, else system-prompt hash for
    AutoGen/CrewAI, else recovered growing-prefix session (Claude Code).
    """
    use_explicit_team = _explicit_multi_team(file_rows)
    by_key_map = {}
    for file_id, rows in file_rows:
        sessions = recover_sessions([r for r in rows if r.get("prefix_sha")])
        by_key = {}
        for sess in sessions:
            for r in sess["rows"]:
                by_key[(r.get("t_recv"), tuple(r.get("prefix_sha") or []))] = sess["id"]
        by_key_map[file_id] = by_key

    def gt_for(row):
        file_id = row["_file"]
        roles = row.get("msg_roles") or []
        sha0 = (row.get("prefix_sha") or [None])[0]
        if use_explicit_team and row.get("team") is not None:
            team = str(row["team"])
            if row.get("member") is not None:
                return team, f"{team}:{row['member']}"
            if row.get("rid") and str(row["rid"]).count(":") >= 1:
                parts = str(row["rid"]).split(":")
                return team, f"{parts[0]}:{parts[1]}"
            return team, f"{team}:{sha0}"
        team = _session_id(row) or file_id
        if row.get("member") is not None:
            member = f"{team}:{row['member']}"
        elif roles[:1] == ["system"] and sha0:
            member = f"{team}:{sha0}"
        else:
            sid = by_key_map[file_id].get(
                (row.get("t_recv"), tuple(row.get("prefix_sha") or [])),
                sha0,
            )
            member = f"{team}:{sid}"
        return str(team), str(member)

    return gt_for


def score_stream(rows, gt_for, min_span_chars=1024):
    disc = TeamDiscovery(min_span_chars=min_span_chars)
    pred_m, pred_t, gt_m, gt_t = [], [], [], []
    n_new = n_hand = 0
    for row in rows:
        ident = disc.observe(row)
        team, member = gt_for(row)
        # Final component is applied after the stream; store lineage for remap.
        pred_m.append(ident.member_id)
        pred_t.append(ident.member_id)  # placeholder, remapped below
        gt_m.append(member)
        gt_t.append(team)
        n_new += int(ident.lineage_new)
        n_hand += ident.n_handoffs
    pred_t = [disc.team_of(lid) for lid in pred_m]
    return {
        "n": len(rows),
        "n_gt_teams": len(set(gt_t)),
        "n_gt_members": len(set(gt_m)),
        "n_pred_teams": len(set(pred_t)),
        "n_pred_members": len(set(pred_m)),
        "member_ari": adjusted_rand_index(pred_m, gt_m),
        "team_ari": adjusted_rand_index(pred_t, gt_t),
        "member_pair_prf": pair_prf(pred_m, gt_m),
        "team_pair_prf": pair_prf(pred_t, gt_t),
        "singleton_member_rate": (
            sum(1 for c in Counter(pred_m).values() if c == 1) / max(len(set(pred_m)), 1)
        ),
        "n_lineages_opened": n_new,
        "n_handoff_edges": n_hand,
    }


def _fmt_prf(prf):
    p, r, f = prf
    return f"{p:.3f}/{r:.3f}/{f:.3f}"


def self_test():
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


def _agent_rows(row):
    roles = row.get("msg_roles") or []
    return bool(roles) and roles[0] == "system"


CASES = [
    ("autogen-ring", ["autogen-ring-k5.jsonl"]),
    ("autogen-design-glm", ["autogen-design-k5-GLM-5.3.jsonl"]),
    ("autogen-debug", ["autogen-debug-k5-Kimi-K3.jsonl"]),
    ("autogen-review", ["autogen-review-k5-Kimi-K3.jsonl"]),
    ("autogen-design-k6", ["autogen-design-k6-Kimi-K3.jsonl"]),
    ("autogen-design-k8", ["autogen-design-k8-Kimi-K3.jsonl"]),
    ("mix-autogen-shared-sys",
     ["autogen-ring-k5.jsonl", "autogen-design-k5-GLM-5.3.jsonl"]),
    ("mix-autogen-shared-k6",
     ["autogen-ring-k5.jsonl", "autogen-design-k6-Kimi-K3.jsonl"]),
    ("mix-autogen-disjoint",
     ["autogen-debug-k5-Kimi-K3.jsonl", "autogen-review-k5-Kimi-K3.jsonl"]),
    ("mix-autogen-3",
     ["autogen-debug-k5-Kimi-K3.jsonl", "autogen-review-k5-Kimi-K3.jsonl",
      "autogen-ring-k5.jsonl"]),
    ("mix-autogen-4-kimi",
     ["autogen-debug-k5-Kimi-K3.jsonl", "autogen-review-k5-Kimi-K3.jsonl",
      "autogen-ring-k5.jsonl", "autogen-design-k8-Kimi-K3.jsonl"]),
    ("crewai-ring", ["crewai-ring-k5.jsonl"]),
    ("crewai-debug", ["crewai-debug-k5-Kimi-K3.jsonl"]),
    ("mix-crewai", ["crewai-ring-k5.jsonl", "crewai-debug-k5-Kimi-K3.jsonl"]),
    ("selector-debug-kimi", ["selector-debug-k5-Kimi-K3.jsonl"]),
    ("selector-debug-kimi-agents", ["selector-debug-k5-Kimi-K3.jsonl"]),
    ("selector-design-kimi-agents", ["selector-design-k5-Kimi-K3.jsonl"]),
    ("selector-debug-glm-agents", ["selector-debug-k5-GLM-5.3.jsonl"]),
    ("claude-team-run1", ["team-run1.jsonl"]),
    ("claude-team-run2", ["team-run2.jsonl"]),
    ("claude-single", ["single-run1.jsonl"]),
    ("mix-claude", ["team-run1.jsonl", "team-run2.jsonl"]),
    ("e2e-autogen-20t", ["e2e-autogen-design-lru-20t-p3000r8.jsonl"]),
    ("e2e-selector-32t-agents", ["e2e-selector-debug-lru-32t-p3000r8.jsonl"]),
]


def run_eval(traces_dir, min_span_chars=1024):
    traces_dir = Path(traces_dir)
    results = []
    for name, files in CASES:
        loaded = []
        missing = False
        for fn in files:
            p = traces_dir / fn
            if not p.exists():
                print(f"skip {name}: missing {fn}")
                missing = True
                break
            loaded.append((Path(fn).stem, load_jsonl(p)))
        if missing:
            continue
        filt = _agent_rows if name.endswith("-agents") else None
        rows = overlay_rows(loaded, filter_fn=filt)
        if not rows:
            print(f"skip {name}: empty")
            continue
        gt_for = attach_ground_truth(loaded)
        stats = score_stream(rows, gt_for, min_span_chars=min_span_chars)
        stats["case"] = name
        stats["files"] = files
        results.append(stats)
    return results


def print_table(results):
    hdr = (f"{'case':<28} {'n':>5} {'gtT':>4} {'prT':>4} {'gtM':>4} {'prM':>4} "
           f"{'mARI':>6} {'tARI':>6} {'team P/R/F1':>17} {'mem P/R/F1':>17}")
    print(hdr)
    print("-" * len(hdr))
    for s in results:
        print(
            f"{s['case']:<28} {s['n']:5d} {s['n_gt_teams']:4d} {s['n_pred_teams']:4d} "
            f"{s['n_gt_members']:4d} {s['n_pred_members']:4d} "
            f"{s['member_ari']:6.3f} {s['team_ari']:6.3f} "
            f"{_fmt_prf(s['team_pair_prf']):>17} {_fmt_prf(s['member_pair_prf']):>17}"
        )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces-dir", default=str(Path(__file__).resolve().parent / "traces"))
    ap.add_argument("--min-span", type=int, default=1024)
    ap.add_argument("--json-out", default="")
    ap.add_argument("--self-test", action="store_true")
    cfg = ap.parse_args(argv)

    self_test()
    if cfg.self_test:
        return

    results = run_eval(cfg.traces_dir, min_span_chars=cfg.min_span)
    print()
    print(f"min_span_chars={cfg.min_span}  traces={cfg.traces_dir}")
    print_table(results)
    if cfg.json_out:
        out = Path(cfg.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        serial = []
        for s in results:
            row = dict(s)
            row["member_pair_prf"] = list(s["member_pair_prf"])
            row["team_pair_prf"] = list(s["team_pair_prf"])
            serial.append(row)
        out.write_text(json.dumps(serial, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
