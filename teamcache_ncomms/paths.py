"""Locations of the data deposit and of regenerated outputs.

TEAMCACHE_DATA points at an unpacked copy of the Zenodo data deposit
(default: ./data).  TEAMCACHE_OUT receives everything this repository
regenerates (default: ./out).  TEAMCACHE_CACHE_LEVEL selects which
cache-level results the figures read; by default the deposited ones, so that
figures can be rebuilt without rerunning the experiments.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("TEAMCACHE_DATA", REPO / "data")).resolve()
OUT = Path(os.environ.get("TEAMCACHE_OUT", REPO / "out")).resolve()

TRACES = DATA / "traces"
RES = DATA / "gpu" / "results"
E2E = RES / "e2e"
SWEEP = RES / "team-sweep"
ABLATE = RES / "team-ablate"
VLLM = RES / "vllm"
MIXED = DATA / "gpu" / "mixed"
FIGDATA = DATA / "gpu" / "figure_data"
CPU = DATA / "cpu"
CACHE_LEVEL_DEPOSIT = DATA / "cache_level"
PUBLIC_RAW = Path(os.environ.get("TEAMCACHE_PUBLIC", DATA / "public_raw")).resolve()

CACHE_LEVEL_OUT = OUT / "cache_level"
FIGDATA_OUT = OUT / "figure_data"
FIGURES_OUT = OUT / "figures"
SOURCE_DATA_OUT = OUT / "source_data"


class _WithFallback:
    """Directory whose missing files are taken from the deposit."""

    def __init__(self, primary: Path, fallback: Path):
        self.primary, self.fallback = primary, fallback

    def __truediv__(self, name):
        p = self.primary / name
        return p if p.exists() else self.fallback / name

    def __repr__(self):
        return f"{self.primary} (missing files from {self.fallback})"


def cache_level() -> _WithFallback:
    """Cache-level results read by the figures and the number check.

    TEAMCACHE_CACHE_LEVEL selects rerun outputs; experiments that were not
    rerun (for example the public-trace experiment without the public data)
    are read from the deposit.
    """
    primary = Path(os.environ.get("TEAMCACHE_CACHE_LEVEL", CACHE_LEVEL_DEPOSIT)).resolve()
    return _WithFallback(primary, CACHE_LEVEL_DEPOSIT)


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
