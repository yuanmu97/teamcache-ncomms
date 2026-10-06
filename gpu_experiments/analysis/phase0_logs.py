#!/usr/bin/env python3
"""Phase 0A: recover TTTA / hit / low-reuse from existing 12/14-team ring JSON.

Prefix-token reuse is unavailable: replay_team.report never dumped per-request
cached_tokens. Do not invent Σ cached_prefix from mean_cached × context.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SWEEP = ROOT / "results" / "team-sweep"
PLOTS = HERE / "plots"
CPU = HERE / "cpu"

# Paper names. group/online/adaptive are not TeamCache.
PAPER = {
    "lru": "LRU",
    "bypass": "Group-blind",
    "cohort": "Static-index",
    "unified": "TeamCache",
    "online": "online (not TeamCache)",
}
ARMS = ("lru", "bypass", "cohort", "unified")


def pct(xs, q):
    s = sorted(xs)
    if not s:
        return None
    return s[min(int(len(s) * q), len(s) - 1)]


def load_run(n_teams: int, arm: str) -> dict:
    path = SWEEP / f"{arm}_{n_teams}t_ring.json"
    data = json.loads(path.read_text())
    data["_path"] = str(path.relative_to(ROOT))
    return data


def summarize(data: dict) -> dict:
    per = data["per_team"]
    ttta = [x["ttta"] for x in per]
    miss = [x["miss_rate"] for x in per]
    n = len(per)
    wall = data["wall_s"]
    # Closed-loop: each team's TTTA sits on the fleet wall. Ratio ≈ 1 is a
    # property of the harness, not an independent SLO measurement.
    ratios = [t / wall for t in ttta]
    return {
        "path": data["_path"],
        "label_in_json": data["label"],
        "paper_name": PAPER[data["label"]] if data["label"] in PAPER else data["label"],
        "n_teams": data["teams"],
        "capacity": data["capacity"],
        "load": data["load"],
        "wall_s": wall,
        "turn_miss_rate": data["turn_miss_rate"],
        "request_hit": 1.0 - data["turn_miss_rate"],
        "hit_threshold": "cached < 0.5 is a miss (replay_team.report)",
        "mean_cached_fraction": data["mean_cached"],
        "mean_cached_is_not_prefix_sum": True,
        "prefix_token_reuse": "unavailable",
        "low_reuse_teams": sum(1 for x in per if x["miss_rate"] >= 0.9),
        "low_reuse_note": "JSON starved_teams; miss_rate>=0.9 is reuse, not execution starvation",
        "ttta_all": ttta,
        "ttta_median": statistics.median(ttta),
        "ttta_max": max(ttta),
        "ttta_empirical_p95": pct(ttta, 0.95),
        "ttta_n": n,
        "ttta_over_wall_median": statistics.median(ratios),
        "ttta_over_wall_min": min(ratios),
        "ttta_over_wall_max": max(ratios),
        "per_team_miss": miss,
        "per_member": data.get("per_member"),
        "solo_normalized_slowdown": "not reported (no exclusive solo-team runs)",
    }


def ecdf(ax, values, label):
    s = sorted(values)
    ys = [(i + 1) / len(s) for i in range(len(s))]
    ax.step(s, ys, where="post", label=label)


def plot_ttta(rows_by_n: dict):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    for ax, n in zip(axes, (12, 14)):
        for arm in ARMS:
            ecdf(ax, rows_by_n[n][arm]["ttta_all"], PAPER[arm])
        ax.set_title(f"{n}-team ring, closed-loop TTTA")
        ax.set_xlabel("TTTA (s)")
        ax.set_ylabel("ECDF")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = PLOTS / "phase0_ttta_ecdf.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_ratio(rows_by_n: dict):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    for ax, n in zip(axes, (12, 14)):
        u = rows_by_n[n]["unified"]["ttta_all"]
        b = rows_by_n[n]["bypass"]["ttta_all"]
        teams = list(range(len(u)))
        ratios = [uu / bb if bb else None for uu, bb in zip(u, b)]
        ax.bar(teams, ratios, color="#4C78A8")
        ax.axhline(1.0, color="black", lw=0.8)
        ax.set_title(f"{n}-team TTTA TeamCache / Group-blind")
        ax.set_xlabel("team id")
        ax.set_ylabel("TTTA ratio")
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = PLOTS / "phase0_ttta_ratio_unified_bypass.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_cached(rows_by_n: dict):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for ax, n in zip(axes, (12, 14)):
        xs = [PAPER[a] for a in ARMS]
        ys = [rows_by_n[n][a]["mean_cached_fraction"] for a in ARMS]
        ax.bar(xs, ys, color="#F58518")
        ax.set_ylim(0, 1)
        ax.set_title(f"{n}-team global mean_cached (not Σ prefix)")
        ax.set_ylabel("mean cached fraction")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = PLOTS / "phase0_mean_cached_global.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    for ax, n in zip(axes, (12, 14)):
        for arm in ARMS:
            pm = rows_by_n[n][arm]["per_member"] or []
            ax.plot([x["member"] for x in pm],
                    [x["mean_cached"] for x in pm],
                    marker="o", label=PAPER[arm])
        ax.set_title(f"{n}-team per-member mean_cached")
        ax.set_xlabel("member id")
        ax.set_ylabel("mean cached fraction")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out2 = PLOTS / "phase0_mean_cached_per_member.png"
    fig.savefig(out2, dpi=140)
    plt.close(fig)
    return out, out2


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    CPU.mkdir(parents=True, exist_ok=True)
    rows_by_n = {}
    slim = {}
    for n in (12, 14):
        rows_by_n[n] = {}
        slim[n] = {}
        for arm in ARMS + ("online",):
            path = SWEEP / f"{arm}_{n}t_ring.json"
            if not path.exists():
                continue
            s = summarize(load_run(n, arm))
            rows_by_n[n][arm] = s
            dump = dict(s)
            dump.pop("per_member", None)
            slim[n][arm] = dump

    plots = {
        "ttta_ecdf": str(plot_ttta(rows_by_n).relative_to(HERE)),
        "ttta_ratio": str(plot_ratio(rows_by_n).relative_to(HERE)),
    }
    c1, c2 = plot_cached(rows_by_n)
    plots["mean_cached_global"] = str(c1.relative_to(HERE))
    plots["mean_cached_per_member"] = str(c2.relative_to(HERE))

    ablate = ROOT / "results" / "team-ablate"
    star_note = {
        "cohort_9t_star.json": "label=cohort, mode=group+quota (static partition). wall=96.17s. NOT unified / NOT TeamCache.",
        "lru_9t_star.json": "LRU background. wall=173.36s.",
        "bypass_9t_star.json": "Group-blind background. wall=150.92s.",
        "unified_9t_star": "ABSENT. Phase 1 cannot reuse 96s as TeamCache.",
    }
    present = {p.name: star_note.get(p.name, "present") for p in sorted(ablate.glob("*9t_star.json"))}

    out = {
        "phase": "0A",
        "gpu_hours": 0,
        "prefix_token_reuse": "unavailable",
        "hit_definition": "request hit = 1 - turn_miss_rate; miss iff cached < 0.5",
        "ttta_note": "closed-loop TTTA ≈ wall; pairwise ratios ≈ wall ratios; not an independent SLO experiment",
        "solo_normalized_slowdown": "not reported",
        "star_locator": present,
        "plots": plots,
        "runs": slim,
    }
    dest = CPU / "phase0_logs.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}")
    for n in (12, 14):
        u = slim[n]["unified"]
        print(f"{n}t TeamCache hit={u['request_hit']:.3f} wall={u['wall_s']:.1f}s "
              f"low-reuse={u['low_reuse_teams']} prefix=unavailable")


if __name__ == "__main__":
    main()
