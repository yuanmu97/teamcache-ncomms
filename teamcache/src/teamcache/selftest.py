"""Exercise the injected helper without importing SGLang."""

from __future__ import annotations

import json
import os
import time

from teamcache.policy import HELPER


class _Req:
    def __init__(self, rid, tokens=10000):
        self.rid = rid
        self.skip_radix_cache_insert = False
        self.origin_input_ids = [0] * tokens


def selftest(tmpdir=None):
    """Exercise the helper in isolation, including a live control-file switch.

    The helper is executed on its own rather than as part of `common.py`,
    whose imports are not available here. What matters is that the predicate
    behaves, and that a control-file edit is picked up without a restart --
    the whole reason for the file.
    """
    if tmpdir is None:
        import tempfile
        tmpdir = tempfile.mkdtemp(prefix="teamcache-selftest-")
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

    # The public benchmark uses an explicit off mode while keeping the same
    # pool budget in both arms.
    write({"mode": "off", "quota": 0, "pool_tokens": 240000})
    force_reload()
    check("off mode is upstream LRU", skip(_Req("t0:4:6")), False)

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
                out[t].append(skip(_Req(f"{t}:{member}:{turn}", n)))
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
