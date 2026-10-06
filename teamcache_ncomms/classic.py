"""Team-agnostic replacement and admission policies from the caching literature.

Every policy takes a sequence of hashable keys and a capacity in entries and
returns one miss flag per reference, the same contract as ../../optimal.py.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict, deque


def run_arc(keys, c, skip=None):
    """Adaptive Replacement Cache (Megiddo and Modha, FAST 2003)."""
    t1, t2, b1, b2 = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
    p, flags = 0.0, []

    def replace(in_b2):
        if t1 and (len(t1) > p or (in_b2 and len(t1) == p)):
            k, _ = t1.popitem(last=False)
            b1[k] = True
        else:
            k, _ = t2.popitem(last=False)
            b2[k] = True

    for i, x in enumerate(keys):
        if x in t1:
            del t1[x]
            t2[x] = True
            flags.append(False)
            continue
        if x in t2:
            t2.move_to_end(x)
            flags.append(False)
            continue
        flags.append(True)
        if skip is not None and skip[i]:
            continue
        if x in b1:
            p = min(c, p + max(len(b2) / len(b1), 1))
            replace(False)
            del b1[x]
            t2[x] = True
        elif x in b2:
            p = max(0.0, p - max(len(b1) / len(b2), 1))
            replace(True)
            del b2[x]
            t2[x] = True
        else:
            if len(t1) + len(b1) == c:
                if len(t1) < c:
                    b1.popitem(last=False)
                    replace(False)
                else:
                    t1.popitem(last=False)
            elif len(t1) + len(b1) < c:
                total = len(t1) + len(t2) + len(b1) + len(b2)
                if total >= c:
                    if total == 2 * c:
                        b2.popitem(last=False)
                    replace(False)
            t1[x] = True
    return flags


def run_lirs(keys, c, hir_frac=0.01, skip=None):
    """Low Inter-reference Recency Set (Jiang and Zhang, SIGMETRICS 2002)."""
    l_hirs = max(1, int(round(hir_frac * c)))
    l_lirs = c - l_hirs
    stack, queue = OrderedDict(), OrderedDict()   # last item is the top / newest
    lir, resident = set(), set()
    flags = []

    def prune():
        while stack:
            k = next(iter(stack))
            if k in lir:
                return
            del stack[k]

    def demote_bottom_lir():
        k = next(iter(stack))
        lir.discard(k)
        del stack[k]
        queue[k] = True
        prune()

    for i, x in enumerate(keys):
        hit = x in resident
        flags.append(not hit)
        if not hit and skip is not None and skip[i]:
            continue
        if x in lir:
            bottom = next(iter(stack)) == x
            stack[x] = True
            stack.move_to_end(x)
            if bottom:
                prune()
            continue
        if hit:
            if x in stack:
                lir.add(x)
                del queue[x]
                stack.move_to_end(x)
                demote_bottom_lir()
            else:
                stack[x] = True
                queue.move_to_end(x)
            continue
        if len(lir) < l_lirs:
            lir.add(x)
            resident.add(x)
            stack[x] = True
            stack.move_to_end(x)
            continue
        if len(queue) >= l_hirs:
            v, _ = queue.popitem(last=False)
            resident.discard(v)
        resident.add(x)
        if x in stack:
            lir.add(x)
            stack.move_to_end(x)
            demote_bottom_lir()
        else:
            stack[x] = True
            queue[x] = True
        if len(stack) > 8 * c:
            for k in [k for k in stack if k not in resident][: len(stack) - 8 * c]:
                del stack[k]
            prune()
    return flags


def run_2q(keys, c, kin=0.25, kout=0.5, skip=None):
    """Full 2Q (Johnson and Shasha, VLDB 1994)."""
    k_in, k_out = max(1, int(kin * c)), max(1, int(kout * c))
    am, a1in, a1out = OrderedDict(), OrderedDict(), OrderedDict()
    flags = []

    def reclaim():
        if len(am) + len(a1in) < c:
            return
        if len(a1in) > k_in or not am:
            k, _ = a1in.popitem(last=False)
            a1out[k] = True
            if len(a1out) > k_out:
                a1out.popitem(last=False)
        else:
            am.popitem(last=False)

    for i, x in enumerate(keys):
        if x in am:
            am.move_to_end(x)
            flags.append(False)
        elif x in a1in:
            flags.append(False)
        elif skip is not None and skip[i]:
            flags.append(True)
        elif x in a1out:
            flags.append(True)
            del a1out[x]
            reclaim()
            am[x] = True
        else:
            flags.append(True)
            reclaim()
            a1in[x] = True
    return flags


def run_s3fifo(keys, c, small=0.1, skip=None):
    """S3-FIFO (Yang et al., SOSP 2023): small FIFO, main FIFO, ghost FIFO."""
    s_cap = max(1, int(round(small * c)))
    m_cap = c - s_cap
    s, m, g = deque(), deque(), OrderedDict()
    where, freq, flags = {}, defaultdict(int), []

    def evict_m():
        while m:
            t = m.popleft()
            if freq[t] > 0:
                freq[t] -= 1
                m.append(t)
            else:
                del where[t]
                return

    def evict_s():
        while s:
            t = s.popleft()
            if freq[t] >= 1:
                freq[t] = 0
                if len(m) >= m_cap:
                    evict_m()
                m.append(t)
                where[t] = "m"
            else:
                del where[t]
                g[t] = True
                if len(g) > m_cap:
                    g.popitem(last=False)
                return

    for i, x in enumerate(keys):
        if x in where:
            freq[x] = min(freq[x] + 1, 3)
            flags.append(False)
            continue
        flags.append(True)
        if skip is not None and skip[i]:
            continue
        while len(s) + len(m) >= c:
            if len(s) >= s_cap or not m:
                evict_s()
            else:
                evict_m()
        freq[x] = 0
        if x in g:
            del g[x]
            m.append(x)
            where[x] = "m"
        else:
            s.append(x)
            where[x] = "s"
    return flags


def run_wtinylfu(keys, c, window=0.01, protected=0.8, sample=10, skip=None):
    """W-TinyLFU (Einziger et al., TOS 2017) with exact, periodically halved counts."""
    w_cap = max(1, int(round(window * c)))
    main_cap = c - w_cap
    p_cap = max(1, int(protected * main_cap))
    win, prob, prot = OrderedDict(), OrderedDict(), OrderedDict()
    freq, seen, flags = defaultdict(int), 0, []

    for i, x in enumerate(keys):
        freq[x] += 1
        seen += 1
        if seen >= sample * c:
            for k in list(freq):
                freq[k] //= 2
                if not freq[k]:
                    del freq[k]
            seen //= 2
        if x in win:
            win.move_to_end(x)
            flags.append(False)
            continue
        if x in prot:
            prot.move_to_end(x)
            flags.append(False)
            continue
        if x in prob:
            del prob[x]
            prot[x] = True
            if len(prot) > p_cap:
                k, _ = prot.popitem(last=False)
                prob[k] = True
            flags.append(False)
            continue
        flags.append(True)
        if skip is not None and skip[i]:
            continue
        win[x] = True
        if len(win) <= w_cap:
            continue
        cand, _ = win.popitem(last=False)
        if len(prob) + len(prot) < main_cap:
            prob[cand] = True
            continue
        victims = prob if prob else prot
        victim = next(iter(victims))
        if freq.get(cand, 0) > freq.get(victim, 0):
            del victims[victim]
            prob[cand] = True
    return flags


def run_teamcache(refs, c, reserve=True, window=40, min_samples=2, margin=1.02,
                  pin=False, decisions=None):
    """Count-based port of the engine's `unified` admission predicate.

    Mirrors teamcache/src/teamcache/policy.py: a shadow LRU stack
    over all histories, per-member would-hold bits recorded under pressure,
    aged reference rates, a per-team share with one in-flight request reserved
    per active team, omission of members without evidence, the bounded
    bootstrap quota while measuring, and bypass of members the pool never
    holds.  Physical eviction is plain LRU; `pin=True` instead exempts the
    chosen subset from eviction, as in the simulator rule.
    """
    cache, shadow = OrderedDict(), OrderedDict()
    held, rate = defaultdict(list), defaultdict(float)
    roster, ticks, protect = defaultdict(set), defaultdict(int), defaultdict(set)
    since_age, flags = 0, []
    teams = len({t for t, _ in refs})

    def concentrate_on(team, share, cur):
        stats = {}
        for m in roster[team]:
            bits = [b for b, under in held[(team, m)] if under]
            if len(bits) < min_samples:
                continue
            stats[m] = (rate[(team, m)] - (1.0 if m == cur else 0.0),
                        sum(bits) / len(bits))
        mass = sum(r for r, _ in stats.values())
        if mass <= 0:
            return None
        top = set()
        for m in sorted(stats, key=lambda x: (-stats[x][0], x)):
            if top and len(top) + 1 > share:
                continue
            top.add(m)
        conc = sum(stats[m][0] for m in top) / mass
        spread = sum(r * h for r, h in stats.values()) / mass
        return top if conc > spread * margin else set()

    for team, member in refs:
        key = (team, member)
        hit = key in cache
        flags.append(not hit)
        if key in shadow:
            above = 0
            for k in reversed(shadow):
                if k == key:
                    break
                above += 1
            held[key].append((1 if above + 1 <= c else 0, len(shadow) > c))
            if len(held[key]) > window:
                del held[key][:-window]
        shadow[key] = True
        shadow.move_to_end(key)
        rate[key] += 1.0
        roster[team].add(member)
        since_age += 1
        if since_age >= window * 8:
            since_age = 0
            for k in list(rate):
                rate[k] *= 0.5

        share = max(c - teams, 1) / teams if reserve else c / teams
        want = concentrate_on(team, share, member)
        if want is None:
            skip = False
            if len(shadow) > c:
                ticks[team] += 1
                lost = any(b == 0 for m in roster[team]
                           for b, under in held[(team, m)] if under)
                if lost and ticks[team] <= len(roster[team]) * min_samples:
                    skip = member >= max(1, int(share))
        elif not want:
            bits = [b for b, under in held[key] if under]
            skip = len(bits) >= min_samples and not any(bits)
        else:
            skip = member not in want
        protect[team] = want or set()
        if decisions is not None:
            decisions.append(skip)

        if hit:
            cache.move_to_end(key)
            continue
        if skip:
            continue
        cache[key] = True
        while len(cache) > c:
            victim = next(iter(cache))
            if pin:
                victim = next((k for k in cache if k[1] not in protect[k[0]] and k != key),
                              victim)
            del cache[victim]
    return flags


def flatten(refs, members):
    """Erase team identity: every history becomes a member of one global team."""
    return [(0, t * members + m) for t, m in refs]
