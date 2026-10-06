"""The deployed TeamCache admission predicate, driven outside the engine.

The predicate is the exact source that the SGLang patch inserts
(``gpu_experiments/engine/patch_sglang_cohort.py``), executed in a private
namespace with a simulated clock, so cache-level experiments test the rule
that ran on the GPUs rather than a re-implementation of it.
"""

from __future__ import annotations

import builtins
import importlib.util
from functools import lru_cache
from pathlib import Path

from .paths import REPO

PATCH = REPO / "gpu_experiments" / "engine" / "patch_sglang_cohort.py"


@lru_cache(maxsize=1)
def _helper_source() -> str:
    spec = importlib.util.spec_from_file_location("_tc_patch", PATCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.HELPER


class _Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now


class _Req:
    __slots__ = ("rid", "origin_input_ids", "seqlen", "skip_radix_cache_insert")

    def __init__(self, rid, tokens):
        self.rid = rid
        self.origin_input_ids = range(tokens)
        self.seqlen = tokens
        self.skip_radix_cache_insert = False


def _no_open(*_args, **_kwargs):
    raise OSError("debug output disabled")


class EngineRule:
    """One engine instance: ``skip(team, member, turn, tokens, now)``."""

    def __init__(self, pool_tokens, mode="unified", window_s=30.0, adapt_window=40,
                 margin=1.02, min_samples=2, quota=0, global_cap=0):
        ns = {"__builtins__": dict(vars(builtins), open=_no_open)}
        exec(_helper_source(), ns)
        self._clock = _Clock()
        ns["_cohort_time"] = self._clock
        cfg = ns["_cohort_cfg"]
        cfg._path = ""
        cfg.mode = mode
        cfg.pool_tokens = int(pool_tokens)
        cfg.window_s = float(window_s)
        cfg.adapt_window = int(adapt_window)
        cfg.adapt_margin = float(margin)
        cfg.adapt_min_samples = int(min_samples)
        cfg.quota = int(quota)
        cfg.global_cap = int(global_cap)
        self._ns = ns
        self._skip = ns["_cohort_skip_insert"]
        self.state = ns["_cohort_state"]
        self.cfg = cfg
        self.want = {}                  # team -> last keep-set (None: measuring)
        plan = self.state.concentrate_on

        def _recorded(team, *args, **kwargs):
            w = plan(team, *args, **kwargs)
            self.want[team] = None if w is None else set(w)
            return w

        self.state.concentrate_on = _recorded

    def skip(self, team, member, turn, tokens, now):
        self._clock.now = float(now)
        return bool(self._skip(_Req(f"t{team}:{member}:{turn}", int(tokens))))
