#!/usr/bin/env python3
"""Turn the closed-loop sweep into the paper's team-level tables and figures.

The point of the closed-loop replay is that aggregate numbers cannot separate a
group-blind policy from a group-aware one, so this reports the team-level view
first and the aggregate second, and derives the headline metric -- teams per
node at a team SLO -- from the sweep rather than from a single load point.

  python3 analyze_team.py /tmp/team-sweep --charts ../../bench/charts
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics

ARM_ORDER = ("lru", "bypass", "cohort", "rotate", "online", "invdist")
ARM_LABEL = {
    "lru": "upstream LRU",
    "bypass": "bypass (group-blind)",
    "cohort": "Cohort",
    "rotate": "Cohort + rotation",
    "online": "Cohort, derived quota",
    "invdist": "invocation distance",
}


def load(indir):
    runs = []
    for path in sorted(glob.glob(os.path.join(indir, "*.json"))):
        with open(path, encoding="utf-8") as fh:
            runs.append(json.load(fh))
    if not runs:
        raise SystemExit(f"no runs in {indir}")
    return runs


def key(run):
    return (run["label"], run["teams"], run["topology"])


def table_sweep(runs):
    """Load curve, one row per team count, arms side by side."""
    by = {key(r): r for r in runs}
    labels = [a for a in ARM_ORDER
              if any(r["label"] == a for r in runs)]
    counts = sorted({r["teams"] for r in runs})
    topos = sorted({r["topology"] for r in runs})

    for topo in topos:
        print(f"\n=== {topo} ===")
        print(f"{'teams':>6} {'load':>6} {'arm':<22} {'wall':>8} {'GPR':>7} "
              f"{'miss':>7} {'team miss p50':>14} {'max':>7} {'starved':>9} "
              f"{'TTTA p95':>9} {'WTFT p95':>9} {'CST p50':>8} {'$/task':>8}")
        for n in counts:
            for lab in labels:
                r = by.get((lab, n, topo))
                if r is None:
                    continue
                print(f"{n:>6} {r['load']:>5.2f}x {ARM_LABEL[lab]:<22} "
                      f"{r['wall_s']:>7.1f}s {r['gpr_aggregate']:>7.2f} "
                      f"{r['turn_miss_rate'] * 100:>6.1f}% "
                      f"{r['team_miss_p50'] * 100:>13.1f}% "
                      f"{r['team_miss_max'] * 100:>6.1f}% "
                      f"{r['starved_teams']:>4}/{r['teams']:<4} "
                      f"{r['ttta_p95']:>8.1f}s {r['wtft_p95']:>8.2f}s "
                      f"{r['cst_p50'] * 100:>7.1f}% "
                      f"{r['usd_per_team_task']:>8.3f}")
            print()


def table_pairwise(runs):
    """The comparison the paper turns on: aggregate versus worst team.

    If the aggregate columns of bypass and cohort agree while the worst-team
    column does not, aggregate metrics are not sufficient to choose a policy
    here, and that is the argument for a team-level headline metric.
    """
    by = {key(r): r for r in runs}
    print("\n=== can aggregate cost tell the two policies apart? ===")
    print(f"{'topo':>5} {'teams':>6} {'wall byp/coh':>13} "
          f"{'miss byp-coh':>13} | {'worst team byp':>15} {'coh':>7} "
          f"{'starved byp':>12} {'coh':>5}")
    for topo in sorted({r["topology"] for r in runs}):
        for n in sorted({r["teams"] for r in runs}):
            b, c = by.get(("bypass", n, topo)), by.get(("cohort", n, topo))
            if not (b and c):
                continue
            print(f"{topo:>5} {n:>6} {b['wall_s'] / c['wall_s']:>12.3f}x "
                  f"{(b['turn_miss_rate'] - c['turn_miss_rate']) * 100:>12.1f}pp"
                  f" | {b['team_miss_max'] * 100:>14.1f}% "
                  f"{c['team_miss_max'] * 100:>6.1f}% "
                  f"{b['starved_teams']:>12} {c['starved_teams']:>5}")


def table_slo(runs, budgets):
    """Teams per node at a team SLO.

    A team counts only if it finished its rounds inside the budget, so this is
    a capacity number and not a latency percentile: how many teams a node
    carries before some of them stop keeping up.
    """
    print("\n=== teams per node at a team-level SLO (TTTA budget) ===")
    topos = sorted({r["topology"] for r in runs})
    labels = [a for a in ARM_ORDER if any(r["label"] == a for r in runs)]
    for topo in topos:
        print(f"\n  {topo}")
        head = "  ".join(f"{ARM_LABEL[a]:>22}" for a in labels)
        print(f"  {'TTTA budget':>12}  {head}")
        for budget in budgets:
            cells = []
            for lab in labels:
                best = 0
                for r in sorted((x for x in runs
                                 if x["label"] == lab and
                                 x["topology"] == topo),
                                key=lambda x: x["teams"]):
                    met = sum(1 for t in r["per_team"]
                              if t["ttta"] <= budget)
                    # Every team must keep up, so the node's capacity is the
                    # largest fleet at which nobody is dropped.
                    if met == r["teams"]:
                        best = r["teams"]
                    else:
                        break
                cells.append(f"{best:>22}")
            print(f"  {budget:>10.0f}s  " + "  ".join(cells))


def charts(runs, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed; skipping charts")
        return
    os.makedirs(outdir, exist_ok=True)
    labels = [a for a in ARM_ORDER if any(r["label"] == a for r in runs)]
    colors = {"lru": "#c0392b", "bypass": "#e67e22", "cohort": "#27ae60",
              "rotate": "#2980b9", "online": "#8e44ad", "invdist": "#7f8c8d"}

    for topo in sorted({r["topology"] for r in runs}):
        sel = [r for r in runs if r["topology"] == topo]
        if len({r["teams"] for r in sel}) < 2:
            continue

        # The paper's central figure: the two panels disagree. Aggregate
        # progress puts bypass and Cohort on top of each other; the worst team
        # separates them.
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for lab in labels:
            pts = sorted((r["load"], r) for r in sel if r["label"] == lab)
            if not pts:
                continue
            x = [p for p, _ in pts]
            axes[0].plot(x, [r["gpr_aggregate"] for _, r in pts], "o-",
                         color=colors[lab], label=ARM_LABEL[lab])
            axes[1].plot(x, [r["team_miss_max"] * 100 for _, r in pts], "o-",
                         color=colors[lab], label=ARM_LABEL[lab])
        axes[0].set_xlabel("load (sessions / cache capacity)")
        axes[0].set_ylabel("aggregate group progress rate (steps/s)")
        axes[0].set_title("what the aggregate sees")
        axes[1].set_xlabel("load (sessions / cache capacity)")
        axes[1].set_ylabel("worst team miss rate (%)")
        axes[1].set_title("what the worst team sees")
        for ax in axes:
            ax.grid(alpha=.3)
            ax.legend(fontsize=8)
        fig.suptitle(f"Aggregate cost cannot choose the policy ({topo})")
        fig.tight_layout()
        p = os.path.join(outdir, f"cohort-team-aggregate-vs-worst-{topo}.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"wrote {p}")

        # The dashboard figure. Per-request latency is what an operator
        # watches, and across the collapse it barely moves, because a request
        # that waits for a teammate is not a slow request -- it is a request
        # that has not been submitted yet. The team-level rate is what falls.
        base = min((r for r in sel if r["label"] == "lru"),
                   key=lambda r: r["load"], default=None)
        if base is not None:
            pts = sorted((r["load"], r) for r in sel if r["label"] == "lru")
            fig, ax = plt.subplots(figsize=(6.4, 4.2))
            x = [p for p, _ in pts]
            ax.plot(x, [r["ttft_p50"] / base["ttft_p50"] for _, r in pts],
                    "o-", color="#27ae60", label="per-request TTFT p50")
            ax.plot(x, [r["ttft_p95"] / base["ttft_p95"] for _, r in pts],
                    "s-", color="#16a085", label="per-request TTFT p95")
            ax.plot(x, [r["team_gpr_p50"] / base["team_gpr_p50"]
                        for _, r in pts],
                    "^-", color="#c0392b", label="per-team progress rate")
            ax.axhline(1.0, color="#7f8c8d", lw=.8, ls=":")
            ax.set_xlabel("load (sessions / cache capacity)")
            ax.set_ylabel(f"relative to {base['teams']} teams")
            ax.set_title("The dashboard stays green while teams stall")
            ax.grid(alpha=.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            p = os.path.join(outdir, f"cohort-dashboard-blindspot-{topo}.png")
            fig.savefig(p, dpi=150)
            plt.close(fig)
            print(f"wrote {p}")

        # Distribution of per-team miss rates: bypass splits the fleet into
        # winners and losers, Cohort moves every team to the same k-of-n.
        heavy = max(sel, key=lambda r: r["load"])["teams"]
        sel2 = [r for r in sel if r["teams"] == heavy]
        if sel2:
            fig, ax = plt.subplots(figsize=(6, 4))
            for r in sel2:
                if r["label"] not in colors:
                    continue
                xs = sorted(t["miss_rate"] * 100 for t in r["per_team"])
                ys = [(i + 1) / len(xs) * 100 for i in range(len(xs))]
                ax.step(xs, ys, where="post", color=colors[r["label"]],
                        label=ARM_LABEL[r["label"]])
            ax.set_xlabel("per-team miss rate (%)")
            ax.set_ylabel("teams at or below (%)")
            ax.set_title(f"Where the misses land ({heavy} teams, {topo})")
            ax.grid(alpha=.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            p = os.path.join(outdir, f"cohort-team-miss-cdf-{topo}.png")
            fig.savefig(p, dpi=150)
            plt.close(fig)
            print(f"wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("indir", nargs="+")
    ap.add_argument("--charts", default=None)
    ap.add_argument("--slo", type=lambda s: [float(x) for x in s.split(",")],
                    default=[60, 90, 120, 180, 240, 360])
    cfg = ap.parse_args()

    runs = [r for d in cfg.indir for r in load(d)]
    print(f"{len(runs)} runs from {', '.join(cfg.indir)}")
    ctx = {r["context"] for r in runs}
    gen = {r["gen_tokens"] for r in runs}
    print(f"context={sorted(ctx)}  gen={sorted(gen)}  "
          f"ratio~{statistics.mean(ctx) / statistics.mean(gen):.0f}:1")
    table_sweep(runs)
    table_pairwise(runs)
    table_slo(runs, cfg.slo)
    if cfg.charts:
        charts(runs, cfg.charts)


if __name__ == "__main__":
    main()
