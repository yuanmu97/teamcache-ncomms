#!/usr/bin/env python3
"""Figures and Source Data of the TeamCache Nature Communications paper.

Every number is read from the data deposit (TEAMCACHE_DATA) or from
cache-level results (TEAMCACHE_CACHE_LEVEL); see teamcache_ncomms/paths.py.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch, Patch,
                                Polygon, Rectangle)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from teamcache_ncomms import a100 as A100  # noqa: E402
from teamcache_ncomms import optimal as O  # noqa: E402
from teamcache_ncomms import paths  # noqa: E402
from teamcache_ncomms.gpu_tables import extract_robustness  # noqa: E402

RES = paths.RES
E2E = paths.E2E
GPU = paths.MIXED
FD = Path(os.environ.get("TEAMCACHE_FIGDATA", paths.FIGDATA))
SIM = paths.cache_level()
OUT = paths.ensure(paths.FIGURES_OUT)

MM = 1 / 25.4
DOUBLE = 180 * MM
SINGLE = 88 * MM
FS = 7
WARMUP_N = 10
SEEDS = (11, 12, 13, 14)

COLORS = {
    "LRU": "#6E6E6E",
    "Unpartitioned Admission": "#D55E00",
    "Static Partitioning": "#AA4499",
    "Offline LFU": "#0072B2",
    "TeamCache": "#009E73",
}
MARKERS = {
    "LRU": "o",
    "Unpartitioned Admission": "s",
    "Static Partitioning": "v",
    "Offline LFU": "^",
    "TeamCache": "D",
}
SHORT = {
    "LRU": "LRU",
    "Unpartitioned Admission": "Unpart.",
    "Static Partitioning": "Static",
    "Offline LFU": "Off. LFU",
    "TeamCache": "TeamCache",
}
SIMKEY = {"LRU": "LRU", "Blind": "Unpartitioned Admission",
          "Static": "Static Partitioning", "TC": "TeamCache"}
TOPO_LAB = {"ring": "Ring", "empirical": "Free-form", "star": "Star",
            "debate": "Debate", "pairs": "Pairs", "mixed": "Mixed"}
REUSE_CMAP = LinearSegmentedColormap.from_list(
    "reuse", ["#F7F7F7", "#C7E9C0", "#74C476", "#238B45", "#00441B"])


def setup():
    names = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans")
                 if f in names), "DejaVu Sans")
    mpl.rcParams.update({
        "font.family": font,
        "font.size": FS,
        "axes.labelsize": FS,
        "axes.titlesize": FS,
        "xtick.labelsize": FS - 1,
        "ytick.labelsize": FS - 1,
        "legend.fontsize": FS - 1,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 600,
    })
    return font


def label(ax, s, dx=-0.02, dy=1.02):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=8, fontweight="bold",
            ha="right", va="bottom")


SOURCE = defaultdict(dict)
SD_DIR = paths.SOURCE_DATA_OUT


def sd(fig, panel, rows, note):
    SOURCE[fig][panel] = (rows, note)


def write_source_data(name):
    if name not in SOURCE:
        return
    import pandas as pd
    SD_DIR.mkdir(parents=True, exist_ok=True)
    num = name.replace("fig", "")
    path = SD_DIR / f"SourceData_Fig{num}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for panel, (rows, note) in sorted(SOURCE[name].items()):
            sheet = f"Fig{num}{panel}"
            pd.DataFrame(rows).to_excel(xw, sheet_name=sheet, index=False, startrow=2)
            ws = xw.sheets[sheet]
            ws.cell(row=1, column=1, value=f"Figure {num}{panel}. {note}")
    print("wrote", path.name)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    print("wrote", name)
    write_source_data(name)


def loadj(p):
    return json.loads(Path(p).read_text())


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def bars_with_points(ax, x, vals, pol, width=0.62):
    vals = np.asarray(vals, dtype=float)
    ax.bar(x, vals.mean(), width, color=COLORS[pol], alpha=0.85 if pol != "LRU" else 0.6,
           edgecolor="black", linewidth=0.4, zorder=2)
    if len(vals) > 1:
        sd = vals.std(ddof=1)
        ax.errorbar(x, vals.mean(), yerr=sd, color="black", lw=0.6, capsize=1.6,
                    capthick=0.6, zorder=3)
        jitter = np.linspace(-0.12, 0.12, len(vals))
        ax.scatter(x + jitter, vals, s=4, color="white", edgecolor="black",
                   linewidth=0.4, zorder=4)


def legend_handles(pols, kind="line"):
    if kind == "bar":
        return [Patch(facecolor=COLORS[p], edgecolor="black", linewidth=0.4, label=p)
                for p in pols]
    return [Line2D([], [], color=COLORS[p], marker=MARKERS[p], lw=1.0, ms=3.5, label=p)
            for p in pols]


# ------------------------------------------------------------------ layout helpers

TOPO_COL = {"ring": "#2F5D8C", "empirical": "#C8553D", "star": "#5B8E3E",
            "debate": "#9A6FB0", "pairs": "#D08C2F", "mixed": "#4A4A4A"}
MEMBER_COL = ["#4C78A8", "#F2A541", "#59A14F", "#E15759", "#76B7B2",
              "#B07AA1", "#9C755F", "#BAB0AC"]
EXAMPLES = [("autogen-review-k5-Kimi-K3.jsonl", "AutoGen sequential", "#2F5D8C"),
            ("selector-review-k5-Kimi-K3.jsonl", "AutoGen selector", "#8E6BBF"),
            ("selector-debug-k5-GLM-5.3.jsonl", "AutoGen selector", "#C8553D")]


TC_DARK = "#00795A"


def top_legend(fig, pols, lines=True):
    handles = [Line2D([], [], ls="-" if lines else "", marker=MARKERS[p], color=COLORS[p],
                      mec="black", mew=0.5 if not lines else 0.3, ms=4.5 if not lines else 3.5,
                      label=p) for p in pols]
    fig.legend(handles=handles, loc="upper center", ncol=len(pols), bbox_to_anchor=(0.5, 1.0),
               fontsize=FS - 1, handletextpad=0.3, columnspacing=1.4)


def mm_axes(fig, x, y, w, h, frame=True):
    W, H = fig.get_size_inches() / MM
    ax = fig.add_axes([x / W, y / H, w / W, h / H])
    if not frame:
        ax.set_axis_off()
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
    return ax


def mlabel(fig, x, y, s):
    W, H = fig.get_size_inches() / MM
    fig.text(x / W, y / H, s, fontsize=8, fontweight="bold", ha="left", va="top")


def rbox(ax, x, y, w, h, fc="white", ec="#555555", lw=0.6, r=1.0, ls="-", z=1, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z, alpha=alpha))


def arrow(ax, p0, p1, color="#444444", lw=0.7, rad=0.0, style="-|>", ls="-", ms=6, z=3):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, lw=lw,
                                 color=color, linestyle=ls, shrinkA=0, shrinkB=0,
                                 connectionstyle=f"arc3,rad={rad}", zorder=z))


def return_times(order):
    last, out = {}, []
    for i, m in enumerate(order):
        if m in last:
            out.append(i - last[m])
        last[m] = i
    return out


# ------------------------------------------------------------------ Figure 1

def fig1_concept(ax):
    rbox(ax, 1, 3, 37, 43, fc="#F3F5F8", ec="none", r=1.5)
    ax.text(19.5, 42.5, "Agent team", ha="center", va="center", fontweight="bold")
    roles = [("C", "coder", 7), ("R", "reviewer", 5), ("D", "debugger", 4)]
    for i, (m, role, n) in enumerate(roles):
        y = 33.5 - 10.5 * i
        col = MEMBER_COL[i]
        ax.add_patch(Circle((6, y), 2.6, fc=col, ec="white", lw=0.6, zorder=3))
        ax.text(6, y, m, color="white", fontweight="bold", ha="center", va="center", zorder=4)
        ax.text(10.4, y + 2.9, role, fontsize=FS - 1.5, color="#444444", va="center")
        for j in range(n):
            ax.add_patch(Rectangle((10.4 + 3.6 * j, y - 1.5), 3.3, 3.0, fc=col,
                                   alpha=0.3 + 0.7 * (j + 1) / n, ec="none"))
    ax.text(19.5, 7.0, "Each member keeps a private history\nthat grows at each of its turns",
            ha="center", va="center", fontsize=FS - 1.5, color="#555555", linespacing=1.15)

    arrow(ax, (39.5, 25), (55.0, 25), color="#666666", lw=1.6, ms=8)
    ax.text(47.2, 30.5, "request =\nhistory + new turn", ha="center", va="center",
            fontsize=FS - 1.5, color="#444444", linespacing=1.1)
    for j, m in enumerate("CRDCRD"):
        ax.text(41.2 + 2.3 * j, 20.2, m, ha="center", va="center", fontsize=FS - 1,
                fontweight="bold", color=MEMBER_COL["CRD".index(m)])
    ax.text(47.2, 16.6, "speaking order", ha="center", va="center", fontsize=FS - 1.5,
            color="#555555")

    rbox(ax, 56, 3, 37, 43, fc="#F1F1F1", ec="none", r=1.5)
    ax.text(74.5, 42.5, "Serving node", ha="center", va="center", fontweight="bold")
    rbox(ax, 58.5, 18, 32, 20.5, fc="white", ec="#9A9A9A", lw=0.6, r=1.0)
    ax.text(74.5, 35.8, "GPU prefix pool (2 histories)", ha="center", va="center",
            fontsize=FS - 1.5, color="#333333")
    for i, (n, y) in enumerate(((7, 29.0), (5, 22.8))):
        for j in range(n):
            ax.add_patch(Rectangle((61 + 3.6 * j, y - 1.5), 3.3, 3.0, fc=MEMBER_COL[i],
                                   alpha=0.3 + 0.7 * (j + 1) / n, ec="none", zorder=3))
    ax.text(83.2, 22.8, "reuse", fontsize=FS - 1.5, color="#007A5A", va="center")
    for j in range(4):
        ax.add_patch(Rectangle((61 + 3.6 * j, 10.0), 3.3, 3.0, fc="none",
                               ec=MEMBER_COL[2], lw=0.6, ls=(0, (2, 1.2)), zorder=3))
    ax.text(76.2, 11.5, "evicted:\nrecompute", fontsize=FS - 1.5, color="#B03A2E",
            va="center", linespacing=1.1)
    ax.text(74.5, 6.0, "Working set exceeds the pool", ha="center", va="center",
            fontsize=FS - 1.5, color="#555555")


def fig1_mechanism(ax):
    seq = "CRDCRDCRD"
    mc = {m: MEMBER_COL[i] for i, m in enumerate("CRD")}

    def run(select):
        cache, before, hits = [], [], []
        for m in seq:
            before.append(list(reversed(cache)))
            h = m in cache
            hits.append(h)
            if h:
                cache.remove(m)
                cache.append(m)
            elif select is None or m in select:
                cache.append(m)
                if len(cache) > 2:
                    cache.pop(0)
        return before, hits

    x0, dx = 25.0, 5.9
    xs = [x0 + dx * j for j in range(len(seq))]
    ax.plot([x0 + 2.5 * dx] * 2, [3.5, 47.0], color="#AAAAAA", lw=0.5, ls=(0, (2, 1.5)))
    ax.text(x0 + dx, 47.3, "warm-up", ha="center", va="bottom", fontsize=FS - 1.5, color="#777777")
    ax.text(x0 + 5.5 * dx, 47.3, "steady state", ha="center", va="bottom", fontsize=FS - 1.5,
            color="#777777")
    ax.text(1, 42, "Request", va="center", fontweight="bold")
    for x, m in zip(xs, seq):
        ax.add_patch(Circle((x, 42), 2.0, fc=mc[m], ec="white", lw=0.5, zorder=3))
        ax.text(x, 42, m, color="white", fontsize=FS - 1, fontweight="bold",
                ha="center", va="center", zorder=4)
    ax.text(x0 + dx * 8 + 5.0, 42, "steady\nhit rate", ha="left", va="center",
            fontsize=FS - 1.5, color="#555555", linespacing=1.1)
    rows = []
    for base, name, sub, select in ((26.0, "LRU", "admit every\nhistory", None),
                                    (9.0, "Selective", "admit C and R\nonly", "CR")):
        before, hits = run(select)
        rows.extend({"policy": name, "request": j + 1, "member": m,
                     "phase": "warm-up" if j < 3 else "steady state",
                     "cache_before_request": " ".join(c), "hit": bool(h)}
                    for j, (m, c, h) in enumerate(zip(seq, before, hits)))
        ax.text(1, base + 5.6, name, va="center", fontweight="bold")
        ax.text(1, base + 1.0, sub, va="center", fontsize=FS - 1.5, color="#555555",
                linespacing=1.1)
        for x, content, h in zip(xs, before, hits):
            for s in range(2):
                y = base + 5.2 - 4.4 * s
                if s < len(content):
                    m = content[s]
                    ax.add_patch(Rectangle((x - 2.3, y), 4.6, 3.8, fc=mc[m], ec="none", zorder=2))
                    ax.text(x, y + 1.9, m, color="white", fontsize=FS - 1.5, fontweight="bold",
                            ha="center", va="center", zorder=3)
                else:
                    ax.add_patch(Rectangle((x - 2.3, y), 4.6, 3.8, fc="white", ec="#BBBBBB",
                                           lw=0.5, ls=(0, (1.5, 1)), zorder=2))
            if h:
                ax.scatter([x], [base - 1.6], s=11, marker="o", color="#009E73", zorder=4, lw=0)
            else:
                ax.scatter([x], [base - 1.6], s=11, marker="x", color="#C0392B", zorder=4, lw=0.9)
        rate = 100 * sum(hits[3:]) / len(hits[3:])
        ax.text(x0 + dx * 8 + 5.0, base + 3.2, f"{rate:.0f}%", ha="left", va="center",
                fontsize=FS + 1, fontweight="bold",
                color="#C0392B" if rate == 0 else "#007A5A")
    ax.scatter([26], [1.0], s=11, marker="o", color="#009E73", lw=0)
    ax.text(27.6, 1.0, "hit", va="center", fontsize=FS - 1.5)
    ax.scatter([35], [1.0], s=11, marker="x", color="#C0392B", lw=0.9)
    ax.text(36.6, 1.0, "miss (full prefill)", va="center", fontsize=FS - 1.5)
    ax.text(1, 1.0, "Cache before each request", va="center", fontsize=FS - 1.5, color="#555555")
    sd("fig2", "b", rows, "Toy two-slot cache under a C, R, D ring; cache contents are "
       "listed from most to least recent.")


def fig1():
    fig = plt.figure(figsize=(DOUBLE, 62 * MM))
    traces = {r["trace"]: r for r in loadj(RES / "collections-summary.json")}

    mlabel(fig, 0, 61, "a")
    rows_b = []
    for i, (name, harness, col) in enumerate(EXAMPLES):
        r = traces[name]
        order = [int(c) for c in r["order"]]
        rows_b.extend({"trace": name, "harness": harness, "recency_rank": r["norm_rank"],
                       "member_turn": t, "member": m} for t, m in enumerate(order))
        ax = mm_axes(fig, 10, 45 - 16.5 * i, 72, 11)
        for t, m in enumerate(order):
            ax.add_patch(Rectangle((t + 0.1, m - 0.4), 0.8, 0.8, fc=MEMBER_COL[m], ec="none"))
        ax.set_xlim(0, len(order))
        ax.set_ylim(r["k"] - 0.5, -0.5)
        ax.set_yticks(range(r["k"]))
        ax.tick_params(axis="y", length=0, labelsize=FS - 2, pad=1)
        ax.spines["left"].set_visible(False)
        if i == len(EXAMPLES) - 1:
            ax.set_xticks(range(0, len(order) + 1, 10))
            ax.set_xlabel("Member turn", labelpad=1)
        else:
            ax.set_xticks(range(0, len(order) + 1, 10))
            ax.set_xticklabels([])
        if i == 1:
            ax.set_ylabel("Member", labelpad=2)
        ax.set_title(f"{harness}, recency rank {r['norm_rank']:.2f}", loc="left",
                     fontsize=FS - 1, color=col, pad=1.5)

    sd("fig1", "a", rows_b, "Speaking order of three recorded agent teams.")
    ax = mm_axes(fig, 150, 11, 29, 45)
    mlabel(fig, 137, 61, "c")
    rows_d = []
    for name, harness, col in EXAMPLES:
        rt = np.sort(return_times([int(c) for c in traces[name]["order"]]))
        rows_d.extend({"trace": name, "return_time_member_turns": int(v),
                       "cumulative_fraction": (j + 1) / len(rt)} for j, v in enumerate(rt))
        x = np.concatenate([[0], rt, [16]])
        y = np.concatenate([[0], np.arange(1, len(rt) + 1) / len(rt), [1]])
        ax.step(x, y, where="post", color=col, lw=1.1)
        cv = rt.std() / rt.mean()
        ax.plot([], [], color=col, lw=1.1, label=f"rank {traces[name]['norm_rank']:.2f}, CV {cv:.2f}")
    ax.annotate("ring: every member\nreturns after 5 turns", (5, 0.3), xytext=(7.0, 0.12),
                fontsize=FS - 1.5, color=EXAMPLES[0][2],
                arrowprops=dict(arrowstyle="-", lw=0.5, color=EXAMPLES[0][2]))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 1.03)
    ax.set_xticks([1, 5, 10, 15])
    ax.set_xlabel("Return time (member turns)")
    ax.set_ylabel("Cumulative fraction of returns")
    ax.legend(loc="lower right", fontsize=FS - 2, handlelength=1.2, borderaxespad=0.1,
              bbox_to_anchor=(1.02, 0.42))
    sd("fig1", "c", rows_d, "Empirical distribution of member return times.")

    ax = mm_axes(fig, 104, 11, 29, 45)
    mlabel(fig, 86, 61, "b")
    groups = [("AutoGen seq.", []), ("CrewAI seq.", []), ("AutoGen sel.", []),
              ("Claude Code", [])]
    ex = {n: c for n, _, c in EXAMPLES}
    rows_e = []
    for r in traces.values():
        name = r["trace"]
        if (r.get("member_turns") or 0) < 16 or r.get("norm_rank") is None:
            continue
        g = 0 if name.startswith("autogen") else 1 if name.startswith("crewai") else \
            2 if name.startswith("selector") else None
        if g is not None:
            groups[g][1].append((r["norm_rank"], ex.get(name)))
            rows_e.append({"source": "recorded", "system": groups[g][0], "trace": name,
                           "normalised_recency_rank": r["norm_rank"]})
    groups[3][1].extend([(0.26, None), (0.30, None)])
    rows_e.extend({"source": "recorded", "system": "Claude Code",
                   "trace": f"claude-code-session-{i + 1}", "normalised_recency_rank": rk}
                  for i, (rk, _) in enumerate(groups[3][1]))
    for gi, (_, pts) in enumerate(groups):
        pts.sort(key=lambda q: q[0])
        for rk, col in pts:
            same = [q for q in pts if abs(q[0] - rk) < 0.04]
            k = same.index((rk, col))
            off = 0 if len(same) == 1 else -0.25 + 0.5 * k / (len(same) - 1)
            ax.scatter(rk, gi + off, s=7, color=col or "#9AA5B1", edgecolor="white",
                       linewidth=0.3, zorder=3 if col is None else 4)
    public = loadj(SIM / "s8_public_traces.json")["traces"]
    systems = ["Magentic-One", "ChatDev", "AppWorld", "HyperAgent"]
    rng = np.random.default_rng(0)
    for k, name in enumerate(systems):
        gi = len(groups) + k
        vals = np.array([t["recency_rank"] for t in public if t["system"] == name])
        rows_e.extend({"source": t["source"], "system": name, "trace": t["id"],
                       "normalised_recency_rank": t["recency_rank"]}
                      for t in public if t["system"] == name)
        ax.scatter(vals, gi + rng.uniform(-0.22, 0.22, len(vals)), s=1.2, color="#9AA5B1",
                   alpha=0.45, lw=0, zorder=2)
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.add_patch(Rectangle((q1, gi - 0.3), q3 - q1, 0.6, fc="none", ec="#333333",
                               lw=0.6, zorder=3))
        ax.plot([med, med], [gi - 0.3, gi + 0.3], color="#333333", lw=1.0, zorder=3)
    sd("fig1", "b", rows_e, "Normalised recency rank of traces with at least 16 member turns. "
       "Recorded: traces recorded for this study (Claude Code values from two sessions). "
       "Public: MAST and Who&When multi-agent traces.")
    ax.axhline(len(groups) - 0.5, color="#BBBBBB", lw=0.5, ls="--")
    ax.text(0.02, -0.55, "recorded", ha="left", va="center", fontsize=FS - 2,
            color="#666666")
    ax.text(0.02, len(groups) - 0.25, "public", ha="left", va="center", fontsize=FS - 2,
            color="#666666")
    labs = [g for g, _ in groups] + systems
    ax.set_yticks(range(len(labs)))
    ax.set_yticklabels(labs, fontsize=FS - 1.5)
    ax.tick_params(axis="y", length=0, pad=1)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(len(labs) - 0.4, -0.9)
    ax.set_xlim(0, 1.1)
    ax.set_xticks([0, 0.5, 1])
    ax.axvline(1.0, color="#BBBBBB", lw=0.5, ls=":", zorder=1)
    ax.set_xlabel("Normalised recency rank")
    save(fig, "fig1")


# ------------------------------------------------------------------ Figure 2

TEAMS_S2 = 9


def ring_pmf(teams, members, tmax):
    p = 1.0 / teams
    pmf = np.zeros(tmax + 1)
    for t in range(members, tmax + 1):
        x = t - members
        pmf[t] = math.comb(x + members - 1, x) * p ** members * (1 - p) ** x
    return pmf


def che_point(pmf, n_items, cap):
    surv = 1.0 - np.cumsum(pmf)
    age = np.concatenate([[0.0], np.cumsum(surv[:-1])]) / n_items
    occ = n_items * np.minimum(age, 1.0)
    k = int(np.searchsorted(occ, cap))
    frac = (cap - occ[k - 1]) / (occ[k] - occ[k - 1])
    cdf = np.cumsum(pmf)
    return k - 1 + frac, cdf[k - 1] + frac * (cdf[k] - cdf[k - 1])


def freeform_pmf(teams, members, tmax, seed=1):
    refs = O.topo_reference("empirical", teams, members, 300, seed=seed)
    skip = 3 * teams * members
    last, rt = {}, []
    for i, key in enumerate(refs):
        if key in last and i >= skip:
            rt.append(i - last[key])
        last[key] = i
    pmf = np.bincount(np.minimum(rt, tmax), minlength=tmax + 1).astype(float)
    return pmf / pmf.sum()


def phase_grids():
    s3 = loadj(SIM / "s3_phase.json")
    cells = s3["cells"]
    ps = sorted({c["p"] for c in cells})
    caps = sorted({c["cap"] for c in cells}, reverse=True)

    def grid(key):
        return np.array([[next(c[key] for c in cells if c["p"] == p and c["cap"] == cap)
                          for cap in caps] for p in ps])

    return {"runs": s3["runs"], "ps": ps, "caps": caps, "loads": [63 / c for c in caps],
            "lru": grid("LRU"), "st": grid("Static"), "tc": grid("TC"),
            "pred": grid("che_lru") - grid("lemma_static")}


def phase_heat_axes(ax, cax, ps, loads, topo_ticks):
    ax.set_xticks(range(len(loads)))
    ax.set_xticklabels([f"{x:.1f}" for x in loads], fontsize=FS - 1 if topo_ticks else FS - 2)
    ax.set_yticks(range(0, len(ps), 2))
    labs = [f"{ps[i]:.1f}" for i in range(0, len(ps), 2)]
    if topo_ticks:
        labs[0], labs[-1] = "free-form  0.0", "ring  1.0"
    ax.set_yticklabels(labs)
    if topo_ticks:
        ticks = ax.get_yticklabels()
        ticks[0].set_color(TOPO_COL["empirical"])
        ticks[-1].set_color(TOPO_COL["ring"])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Load")
    cax.tick_params(labelsize=FS - 2, length=1.5)
    for s in cax.spines.values():
        s.set_linewidth(0.3)


def fig2():
    s2 = loadj(SIM / "s2_cliff.json")
    fig = plt.figure(figsize=(DOUBLE, 112 * MM))

    ax = mm_axes(fig, 12, 64, 56, 40)
    mlabel(fig, 0, 111, "a")
    ax.axvspan(1.0, 2.2, color="#F6E3E0", lw=0, zorder=0)
    ax.text(1.62, 103, "working set\nexceeds pool", ha="center", va="top", fontsize=FS - 1.5,
            color="#B03A2E", linespacing=1.1)
    g = s2["gpu"]
    kx, ky = [d["load"] for d in g], [100 * d["hit"] for d in g]
    rows = [loadj(E2E / f"e2e-ring-lru-{t}t-p56k-glm.json") for t in (6, 8, 10, 12, 16)
            if (E2E / f"e2e-ring-lru-{t}t-p56k-glm.json").exists()]
    pool = rows[0].get("pool_tokens", 2_695_552)

    def ws(d):
        if "working_set" in d:
            return d["working_set"]
        return d["teams"] * d["members"] * (d["context"] + 512 + d["gen_tokens"])
    gx = [ws(d) / pool for d in rows]
    gy = [100 * (1 - d["turn_miss_rate"]) for d in rows]
    ax.plot(gx, gy, color=TOPO_COL["empirical"], marker="s", ms=3.2, label="GLM 5.3",
            zorder=3)
    ax.plot(kx, ky, color=TOPO_COL["ring"], marker="o", ms=3.4, label="Kimi K3", zorder=4)
    for d, x, y in zip(g, kx, ky):
        left = d["teams"] == 10
        ax.annotate(f"{d['teams']}", (x, y), textcoords="offset points",
                    xytext=(-3, 2) if left else (0, 3.5), fontsize=FS - 2,
                    color=TOPO_COL["ring"], ha="right" if left else "center")
    for d, x, y in zip(rows, gx, gy):
        ax.annotate(f"{d['teams']}", (x, y), textcoords="offset points", xytext=(2.5, 2.5),
                    fontsize=FS - 2, color=TOPO_COL["empirical"])
    ax.text(kx[2] + 0.04, 42, "+23% working set:\n82% → 0.4% hits", fontsize=FS - 1.5,
            color=TOPO_COL["ring"], va="center", linespacing=1.1, fontweight="bold")
    ax.set_xlim(0.6, 2.05)
    ax.set_ylim(-4, 106)
    ax.set_xlabel("Working set / prefix pool")
    ax.set_ylabel("Request hit rate (%)")
    ax.legend(loc="lower left", fontsize=FS - 1.5, borderaxespad=0.1, handlelength=1.4)
    ax.set_title("LRU on an 8×B300-GPU server, five-member rings", loc="left", fontsize=FS - 1, pad=3)
    sd("fig2", "a",
       [{"model": "Kimi K3", "teams": d["teams"], "working_set_over_pool": x,
         "request_hit_rate_pct": y} for d, x, y in zip(g, kx, ky)]
       + [{"model": "GLM 5.3", "teams": d["teams"], "working_set_over_pool": x,
           "request_hit_rate_pct": y} for d, x, y in zip(rows, gx, gy)],
       "LRU request hit rate on an 8×B300-GPU server for five-member ring teams.")

    fig1_mechanism(mm_axes(fig, 90, 58, 82, 50, frame=False))
    mlabel(fig, 80, 111, "b")

    mlabel(fig, 0, 51, "c")
    tmax, teams, k, cap = 160, 9, 5, 36
    dists = [("Ring", ring_pmf(teams, k, tmax), TOPO_COL["ring"]),
             ("Free-form", freeform_pmf(teams, k, tmax), TOPO_COL["empirical"])]
    rows_c = []
    for i, (name, pmf, col) in enumerate(dists):
        ax = mm_axes(fig, 8, 28 - 19 * i, 48, 13)
        tc, hit = che_point(pmf[:tmax], teams * k, cap)
        t = np.arange(tmax)
        p = pmf[:tmax] / pmf[:tmax].max()
        rows_c.extend({"topology": name, "return_time_requests": int(tt),
                       "probability": float(pmf[tt]), "probability_scaled_to_max": float(pp),
                       "characteristic_time_T_C": tc, "predicted_hit_rate_pct": 100 * hit}
                      for tt, pp in zip(t, p))
        ax.fill_between(t, p, where=t <= tc + 0.5, step="mid", color=col, alpha=0.85, lw=0)
        ax.fill_between(t, p, where=t >= tc - 0.5, step="mid", color=col, alpha=0.18, lw=0)
        ax.axvline(tc, color="black", lw=0.7, ls="--")
        ax.text(tc + 2, 1.29, f"$T_C$ = {tc:.0f}", fontsize=FS - 0.5, va="top")
        ax.text(0.99 if i == 0 else 0.40, 0.98 if i == 0 else 0.75,
                f"{name}\nhit {100 * hit:.0f}%", transform=ax.transAxes,
                ha="right" if i == 0 else "left", va="top", fontsize=FS - 1.5, color=col,
                linespacing=1.1, fontweight="bold")
        ax.set_xlim(0, tmax - 1)
        ax.set_ylim(0, 1.32)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        if i == 0:
            ax.set_xticklabels([])
            ax.set_title(f"{teams} teams of {k}, room for {cap} of {teams * k}",
                         loc="left", fontsize=FS - 1, pad=3)
        else:
            ax.set_xlabel("Return time (requests)")
    W, H = fig.get_size_inches() / MM
    fig.text(5.5 / W, 25 / H, "Probability", rotation=90, ha="center", va="center",
             fontsize=FS)
    sd("fig2", "c", rows_c, f"Return-time distributions for {teams} teams of {k} members "
       f"with room for {cap} histories; return times below {tmax} requests are shown.")

    ax = mm_axes(fig, 71, 9, 44, 32)
    mlabel(fig, 61, 51, "d")
    topo = s2["topology"]
    rows_d = []
    ax.axvspan(1.0, 6, color="#F6E3E0", lw=0, zorder=0)
    for t in TOPO_LAB:
        rs = [r for r in topo if r["topo"] == t]
        loads = sorted({r["load"] for r in rs})
        sim = [100 * np.mean([r["lru"] for r in rs if r["load"] == L]) for L in loads]
        che = [100 * np.mean([r["che"] for r in rs if r["load"] == L]) for L in loads]
        rows_d.extend({"topology": TOPO_LAB[t], "load": L,
                       "seeds": sum(r["load"] == L for r in rs),
                       "measured_lru_hit_pct": s, "characteristic_time_hit_pct": c}
                      for L, s, c in zip(loads, sim, che))
        ax.plot(loads, che, color=TOPO_COL[t], lw=0.9, zorder=2, label=TOPO_LAB[t])
        ax.scatter(loads, sim, s=4, color="white", edgecolor=TOPO_COL[t], lw=0.5, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(0.22, 5.5)
    ax.set_xticks([0.5, 1, 2, 4])
    ax.set_xticklabels(["0.5", "1", "2", "4"])
    ax.minorticks_off()
    ax.set_ylim(-4, 106)
    ax.set_xlabel("Load (histories / capacity)")
    ax.set_ylabel("LRU hit rate (%)")
    leg = ax.legend(loc="lower left", fontsize=FS - 2, handlelength=1.0, borderaxespad=0.2,
                    labelspacing=0.15, handletextpad=0.3)
    for txt, t in zip(leg.get_texts(), TOPO_LAB):
        txt.set_color(TOPO_COL[t])
    ax.set_title(f"Six topologies, R² = {s2['summary']['topology_che_r2']:.4f}", loc="left",
                 fontsize=FS - 1, pad=3)
    sd("fig2", "d", rows_d, "Measured LRU and characteristic-time reconstruction for six "
       f"collaboration patterns; R2 = {s2['summary']['topology_che_r2']:.4f}.")

    ax = mm_axes(fig, 131, 9, 33, 32)
    cax = mm_axes(fig, 166, 9, 1.8, 32)
    mlabel(fig, 120, 51, "e")
    ph = phase_grids()
    ps, loads, lru, st, pred = ph["ps"], ph["loads"], ph["lru"], ph["st"], ph["pred"]
    diff = 100 * (lru - st)
    im = ax.imshow(diff, origin="lower", aspect="auto", cmap="PuOr_r",
                   norm=TwoSlopeNorm(0, vmin=-60, vmax=30))
    ax.contour(np.arange(len(loads)), np.arange(len(ps)), pred, levels=[0],
               colors="black", linewidths=0.9, linestyles="--")
    ax.text(2.5, 8.6, "fixed subset\nbetter", ha="center", va="center", fontsize=FS - 1,
            color="white", fontweight="bold", linespacing=1.05)
    ax.text(2.5, 1.5, "LRU better", ha="center", va="center", fontsize=FS - 1,
            fontweight="bold")
    ax.text(5.4, 5.5, "predicted", ha="right", va="bottom", fontsize=FS - 2)
    phase_heat_axes(ax, cax, ps, loads, False)
    ax.set_ylabel("Ring regularity p")
    ax.set_title("Better fixed policy", loc="left", fontsize=FS - 1, pad=3)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("LRU minus fixed subset (points)", fontsize=FS - 2)
    sd("fig2", "e",
       [{"ring_regularity_p": p, "capacity": cap, "load": L,
         "lru_hit_pct": 100 * lru[i, j], "static_hit_pct": 100 * st[i, j],
         "lru_minus_static_pp": diff[i, j], "predicted_lru_minus_static": pred[i, j]}
        for i, p in enumerate(ps) for j, (cap, L) in enumerate(zip(ph["caps"], loads))],
       "Request hit rates of LRU and Static Partitioning (mean over seeds); the dashed "
       "boundary is the zero contour of the predicted difference.")
    save(fig, "fig2")


# ------------------------------------------------------------------ Figure 3


def design_architecture(ax):
    rbox(ax, 1, 28.5, 178, 25.5, fc="#E8F4EE", ec="none", r=1.5)
    ax.text(3.5, 51.6, "TeamCache: per-team admission control", ha="left", va="center",
            fontweight="bold", color=TC_DARK)
    rbox(ax, 1, 1, 178, 24.5, fc="#F1F1F1", ec="none", r=1.5)
    ax.text(3.5, 3.6, "Serving engine (unmodified)", ha="left", va="center",
            fontweight="bold", color="#555555")

    comps = [
        ("1", "Team Discovery", "team and member of each request\nfrom harness IDs or prompt lineage"),
        ("2", "Topology Profiler", "speaking order, reference rates\nand a shadow recency order"),
        ("3", "Policy Evaluator", "LRU value: share of returns within capacity\n"
                                   "concentration value: top-$B$ coverage"),
        ("4", "Admission Selector", "concentrate if concentration value\n"
                                     "> $\\eta$ × LRU value, otherwise dynamic"),
    ]
    bx, bw, by, bh = [5, 49, 93, 137], 38, 31.0, 16.5
    for x, (num, title, sub) in zip(bx, comps):
        rbox(ax, x, by, bw, bh, fc="white", ec=TC_DARK, lw=0.7, r=1.2, z=2)
        ax.add_patch(Circle((x + 3.2, by + bh - 3.2), 2.0, fc=TC_DARK, ec="none", zorder=3))
        ax.text(x + 3.2, by + bh - 3.2, num, color="white", fontweight="bold", ha="center",
                va="center", fontsize=FS - 0.5, zorder=4)
        ax.text(x + 6.5, by + bh - 3.2, title, ha="left", va="center", fontweight="bold",
                zorder=4)
        ax.text(x + bw / 2, by + 5.4, sub, ha="center", va="center", fontsize=FS - 1.5,
                color="#333333", linespacing=1.25, zorder=4)
    for i in range(3):
        x0, x1 = bx[i] + bw, bx[i + 1]
        arrow(ax, (x0 + 0.3, by + bh / 2), (x1 - 0.3, by + bh / 2), color=TC_DARK, lw=0.8)

    sboxes = [(5, 20, "Request"), (32, 26, "Prefix matching"), (65, 26, "Model execution"),
              (98, 22, "Completed\nrequest")]
    sy, sh = 9.0, 9.5
    for x, w, t in sboxes:
        rbox(ax, x, sy, w, sh, fc="white", ec="#777777", lw=0.6, r=1.0, z=2)
        ax.text(x + w / 2, sy + sh / 2, t, ha="center", va="center", fontsize=FS - 0.5,
                zorder=3, linespacing=1.1)
    for (x, w, _), (nx, _, _) in zip(sboxes[:-1], sboxes[1:]):
        arrow(ax, (x + w + 0.3, sy + sh / 2), (nx - 0.3, sy + sh / 2), color="#555555")
    dx, dy = 128.5, sy + sh / 2
    arrow(ax, (120.3, dy), (dx - 3.2, dy), color="#555555")
    ax.add_patch(Polygon([(dx - 3, dy), (dx, dy + 3), (dx + 3, dy), (dx, dy - 3)],
                         closed=True, fc="white", ec=TC_DARK, lw=0.8, zorder=3))
    ax.text(dx, dy, "?", ha="center", va="center", fontweight="bold", color=TC_DARK, zorder=4)

    rbox(ax, 141, 13.2, 35, 10.3, fc="white", ec="#777777", lw=0.6, r=1.0, z=2)
    ax.text(158.5, 21.2, "Prefix pool (LRU eviction)", ha="center", va="center",
            fontsize=FS - 1.5, zorder=3)
    for j in range(10):
        col = TOPO_COL["ring"] if j in (0, 1, 4, 5, 8) else TOPO_COL["empirical"]
        ax.add_patch(Rectangle((143.2 + 3.1 * j, 14.8), 2.7, 3.6, fc=col, alpha=0.75,
                               ec="none", zorder=3))
    ax.text(158.5, 7.0, "no insertion; recomputed\nif the member returns", ha="center",
            va="center", fontsize=FS - 1.5, color="#B03A2E", linespacing=1.1)
    arrow(ax, (dx + 2.2, dy + 2.2), (140.6, 17.5), color=TC_DARK, lw=0.9, rad=-0.15)
    ax.text(138.6, 20.4, "admit", fontsize=FS - 1.5, color=TC_DARK, ha="center")
    arrow(ax, (dx + 2.2, dy - 2.2), (143.0, 7.3), color="#B03A2E", lw=0.9, rad=0.15)
    ax.text(134.0, 3.9, "bypass", fontsize=FS - 1.5, color="#B03A2E", ha="center")

    arrow(ax, (bx[3] + 6, by - 0.3), (dx + 0.4, dy + 3.3), color=TC_DARK, lw=0.8,
          ls=(0, (3, 1.5)))
    ax.text(bx[3] + 7.5, 28.2, "admit or bypass this history", fontsize=FS - 2,
            color=TC_DARK, ha="left", va="center")
    arrow(ax, (15, sy + sh + 0.3), (15, by - 0.3), color="#555555", lw=0.7,
          ls=(0, (3, 1.5)))
    ax.text(16.2, 24.5, "request metadata", fontsize=FS - 2, color="#555555",
            ha="left", va="center")
    ax.plot([158.5, 158.5, 45, 45], [23.5, 25.3, 25.3, sy + sh + 1.6], color="#888888",
            lw=0.6, zorder=1)
    arrow(ax, (45, sy + sh + 1.7), (45, sy + sh + 0.3), color="#888888", lw=0.6)
    ax.text(101.0, 25.9, "cached prefixes", fontsize=FS - 2, color="#777777",
            ha="center", va="bottom")


def design_shadow(ax, trace):
    cfg = trace["config"]
    refs = [tuple(r) for r in trace["refs"]]
    kinds, cap = cfg["kinds"], cfg["cap"]
    pools = {}

    def observe(clock, key, cache):
        if 600 < clock <= 1200:
            pools[clock] = list(cache)

    O.run_concentrate(refs, cap, cfg["members"], margin=cfg["margin"], observe=observe)
    last, pick = {}, None
    for clock, key in enumerate(refs):
        if key in last and clock in pools and kinds[key[0] % 2] == "ring":
            order = sorted(last, key=lambda k: -last[k])
            d = order.index(key)
            if d >= cap + 6 and key in pools[clock]:
                pick = (clock, key, order, d)
                break
        last[key] = clock
    clock, key, order, d = pick
    pool = list(reversed(pools[clock]))
    rows = [{"row": "shadow order", "position": j, "team": k[0], "member": k[1],
             "team_kind": "ring" if kinds[k[0] % 2] == "ring" else "free-form",
             "in_prefix_pool": k in pool, "returning_history": k == key}
            for j, k in enumerate(order)]
    rows += [{"row": "prefix pool", "position": j, "team": k[0], "member": k[1],
              "team_kind": "ring" if kinds[k[0] % 2] == "ring" else "free-form",
              "in_prefix_pool": True, "returning_history": k == key}
             for j, k in enumerate(pool)]
    sd("fig3", "b", rows, f"Shadow recency order and TeamCache prefix pool at request {clock}; "
       f"capacity C = {cap} histories; the returning ring history has stack distance {d}.")

    cw, x0, h = 0.95, 4.0, 4.2

    def cells(seq, y):
        for j, k in enumerate(seq):
            col = TOPO_COL[kinds[k[0] % 2]]
            held = k in pool
            ax.add_patch(Rectangle((x0 + cw * j, y), cw * 0.82, h,
                                   fc=col if held else "white", ec=col,
                                   lw=0 if held else 0.45, alpha=0.85 if held else 1.0,
                                   zorder=2))

    y1, y2 = 29.0, 12.5
    ax.text(x0, 40.5, f"Shadow LRU order: all {len(order)} histories", fontsize=FS - 1,
            va="center")
    cells(order, y1)
    ax.text(x0, 20.5, f"TeamCache prefix pool: {len(pool)} admitted histories",
            fontsize=FS - 1, va="center")
    cells(pool, y2)
    xc = x0 + cw * cap - 0.08
    ax.plot([xc, xc], [y1 - 1.0, y1 + h + 1.0], color="black", lw=0.7, ls="--", zorder=4)
    ax.text(xc - 0.6, y1 - 1.6, f"C = {cap}", fontsize=FS - 2, ha="right", va="center")
    j2 = pool.index(key)
    for x, y in ((x0 + cw * d, y1), (x0 + cw * j2, y2)):
        ax.add_patch(Rectangle((x - 0.25, y - 0.45), cw * 0.82 + 0.5, h + 0.9, fc="none",
                               ec="black", lw=0.8, zorder=3))
    xr = x0 + cw * (d + 0.41)
    ax.annotate("", xy=(xr, y1 + h + 0.6), xytext=(xr, y1 + h + 3.2),
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color="black", mutation_scale=5))
    ax.text(xr - 1.0, y1 + h + 2.6, f"returning ring member:\nd = {d} > C, LRU would miss",
            fontsize=FS - 2, ha="right", va="center", color=TOPO_COL["ring"], linespacing=1.1)
    ax.text(x0 + cw * (j2 + 1) + 1.0, y2 + h / 2, "kept by TeamCache: hit", fontsize=FS - 2,
            ha="left", va="center", color=TC_DARK)
    ax.text(x0, y1 - 1.6, "most recent", fontsize=FS - 2.5, color="#666666", va="center")
    ax.text(x0 + cw * len(order), y1 - 1.6, "least recent", fontsize=FS - 2.5,
            color="#666666", va="center", ha="right")
    ly = 4.0
    for i, (lab, col, filled) in enumerate((("ring", TOPO_COL["ring"], True),
                                           ("free-form", TOPO_COL["empirical"], True),
                                           ("bypassed, not in pool", "#777777", False))):
        x = x0 + (0, 11, 26)[i]
        ax.add_patch(Rectangle((x, ly - 1.2), 2.2, 2.4, fc=col if filled else "white", ec=col,
                               lw=0 if filled else 0.45, alpha=0.85 if filled else 1.0))
        ax.text(x + 3.0, ly, lab, fontsize=FS - 2, va="center")


def fig3():
    trace = loadj(SIM / "s5_policy_trace.json")
    cfg = trace["config"]
    fig = plt.figure(figsize=(DOUBLE, 108 * MM))
    up = 0
    design_architecture(mm_axes(fig, 0, 51 + up, 180, 55, frame=False))
    mlabel(fig, 0, 108 + up, "a")

    design_shadow(mm_axes(fig, 0, 1 + up, 62, 42, frame=False), trace)
    mlabel(fig, 0, 45 + up, "b")

    ax = mm_axes(fig, 77, 9 + up, 44, 31)
    mlabel(fig, 66, 45 + up, "c")
    vals = trace["values"]
    n = len(trace["refs"])
    grid = np.arange(0, n + 1, 28)
    rows_c = []
    for kind in cfg["kinds"]:
        teams = sorted({v["team"] for v in vals if v["kind"] == kind})
        cur = {t: None for t in teams}
        series = {"conc": [], "lru": []}
        it = iter(sorted((v for v in vals if v["kind"] == kind), key=lambda v: v["clock"]))
        pending = next(it, None)
        xs = []
        for gx in grid:
            while pending is not None and pending["clock"] <= gx:
                cur[pending["team"]] = pending
                pending = next(it, None)
            have = [c for c in cur.values() if c is not None]
            if len(have) < len(teams):
                continue
            xs.append(gx)
            for key in series:
                series[key].append([100 * c[key] for c in have])
        col = TOPO_COL[kind]
        for key, ls in (("conc", "-"), ("lru", (0, (2.5, 1.2)))):
            arr = np.array(series[key])
            rows_c.extend({"team_kind": "ring" if kind == "ring" else "free-form",
                           "value": "concentration value" if key == "conc" else "LRU value",
                           "requests": int(x), "mean_over_teams_pct": a.mean(),
                           "min_over_teams_pct": a.min(), "max_over_teams_pct": a.max()}
                          for x, a in zip(xs, arr))
            ax.plot(xs, arr.mean(axis=1), color=col, ls=ls, lw=1.1)
            ax.fill_between(xs, arr.min(axis=1), arr.max(axis=1), color=col, alpha=0.15, lw=0)
    ax.text(n * 0.98, 47, "ring: concentrate", color=TOPO_COL["ring"], fontsize=FS - 1.5,
            ha="right", va="bottom")
    ax.text(n * 0.98, 76, "free-form: dynamic", color=TOPO_COL["empirical"],
            fontsize=FS - 1.5, ha="right", va="bottom")
    ax.plot([], [], color="#444444", lw=1.1, label="concentration value")
    ax.plot([], [], color="#444444", lw=1.1, ls=(0, (2.5, 1.2)), label="LRU value")
    ax.legend(loc="upper left", fontsize=FS - 1.5, handlelength=1.8, borderaxespad=0.1,
              ncol=2, columnspacing=1.0)
    ax.set_xlim(0, n)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Requests")
    ax.set_ylabel("Estimated team hit rate (%)")
    sd("fig3", "c", rows_c, "Per-team values estimated from the shadow stack, sampled every "
       "28 requests; mean and range over the four teams of each kind.")

    ax = mm_axes(fig, 136, 9 + up, 42, 31)
    mlabel(fig, 125, 45 + up, "d")
    skip = 3 * cfg["teams"] * cfg["members"]
    refs = trace["refs"]
    rows_d = []
    for gi, kind in enumerate(cfg["kinds"]):
        for pi, pol in enumerate(("LRU", "TeamCache")):
            miss = trace["miss"][pol]
            per = []
            for t in sorted({r[0] for r in refs if cfg["kinds"][r[0] % 2] == kind}):
                idx = [i for i in range(skip, n) if refs[i][0] == t]
                per.append(100 * (1 - sum(miss[i] for i in idx) / len(idx)))
                rows_d.append({"team_kind": "ring" if kind == "ring" else "free-form",
                               "policy": pol, "team": t, "requests": len(idx),
                               "request_hit_rate_pct": per[-1]})
            x = gi * 2.6 + pi
            ax.bar(x, np.mean(per), 0.78, color=COLORS[pol],
                   alpha=0.6 if pol == "LRU" else 0.85, edgecolor="black", lw=0.4, zorder=2)
            jit = np.linspace(-0.18, 0.18, len(per))
            ax.scatter(x + jit, per, s=5, color="white", edgecolor="black", lw=0.4, zorder=3)
            ax.text(x, np.mean(per) + 7, f"{np.mean(per):.0f}", ha="center", fontsize=FS - 1.5)
    ax.set_xticks([0.5, 3.1])
    ax.set_xticklabels(["Ring teams", "Free-form teams"])
    for tl, kind in zip(ax.get_xticklabels(), cfg["kinds"]):
        tl.set_color(TOPO_COL[kind])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Request hit rate (%)")
    ax.legend(handles=legend_handles(["LRU", "TeamCache"], kind="bar"), loc="upper left",
              fontsize=FS - 1.5, borderaxespad=0.1, handlelength=1.0)
    sd("fig3", "d", rows_d, f"Per-team request hit rate after the first {skip} warm-up "
       "requests of the same reference stream.")
    save(fig, "fig3")


# ------------------------------------------------------------------ Figure 4

CLASSIC = ["LRU", "ARC", "2Q", "LIRS", "S3-FIFO", "W-TinyLFU", "TC-global", "TC"]
CLASSIC_LAB = {"TC-global": "TeamCache without teams", "TC": "TeamCache",
               "TeamCache": "TeamCache", "Static": "Static"}
CLASSIC_COL = {"LRU": COLORS["LRU"], "ARC": "#56B4E9", "2Q": "#999933", "LIRS": "#0072B2",
               "S3-FIFO": "#CC79A7", "W-TinyLFU": "#E69F00", "TC-global": "#8FD3BC",
               "TC": COLORS["TeamCache"], "TeamCache": COLORS["TeamCache"],
               "Static": COLORS["Static Partitioning"]}


def fig4():
    s6 = loadj(SIM / "s6_baselines.json")
    s8 = loadj(SIM / "s8_public_traces.json")
    over = [r for r in s6["grid"] if r["load"] > 1.01]
    fig = plt.figure(figsize=(DOUBLE, 132 * MM))

    def mean_hit(pol, topo):
        return 100 * np.mean([r["hit_mean"] for r in over
                              if r["policy"] == pol and r["topo"] == topo])

    ph = phase_grids()
    ps, loads, caps = ph["ps"], ph["loads"], ph["caps"]
    ax = mm_axes(fig, 12, 82, 64, 42)
    mlabel(fig, 0, 131, "a")
    ci = 1
    runs = ph["runs"]
    ax.axvspan(0.5, 0.6, color="#E8E8E8", lw=0, zorder=0)
    rows_e = []
    for name, pol in (("LRU", "LRU"), ("Static", "Static Partitioning"),
                      ("TC", "TeamCache")):
        ms, lo, hi = [], [], []
        for p in ps:
            v = np.array([r[name] for r in runs if r["p"] == p and r["cap"] == caps[ci]])
            m = v.mean()
            h = 1.96 * v.std(ddof=1) / math.sqrt(len(v))
            ms.append(100 * m)
            lo.append(100 * (m - h))
            hi.append(100 * (m + h))
            rows_e.append({"policy": pol, "ring_regularity_p": p, "seeds": len(v),
                           "mean_hit_pct": ms[-1], "ci95_low_pct": lo[-1],
                           "ci95_high_pct": hi[-1]})
        ax.fill_between(ps, lo, hi, color=COLORS[pol], alpha=0.2, lw=0)
        ax.plot(ps, ms, color=COLORS[pol], marker=MARKERS[pol], ms=2.6, mec="black",
                mew=0.3)
    ax.text(0.22, 86, "LRU wins", ha="center", fontsize=FS - 1.5, color="#555555")
    ax.text(0.8, 86, "Static wins", ha="center", fontsize=FS - 1.5,
            color=COLORS["Static Partitioning"])
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0\nfree-form", "0.5", "1\nring"])
    ax.get_xticklabels()[0].set_color(TOPO_COL["empirical"])
    ax.get_xticklabels()[-1].set_color(TOPO_COL["ring"])
    ax.set_xlabel("Ring regularity p", labelpad=1)
    ax.set_ylabel("Request hit rate (%)")
    ax.set_title(f"Tracking the better policy, load {loads[ci]:.2f}", loc="left",
                 fontsize=FS - 1, pad=3)
    ax.set_ylim(0, 96)
    ax.legend(handles=legend_handles(["LRU", "Static Partitioning", "TeamCache"]),
              loc="lower left", fontsize=FS - 1.5, borderaxespad=0.2, handlelength=1.6)
    sd("fig4", "a", rows_e, f"Load {loads[ci]:.2f}; mean over seeds with normal 95% CI.")


    ax = mm_axes(fig, 100, 82, 70, 42)
    mlabel(fig, 86, 131, "b")
    offs = {"LRU": (-1.2, 0, "right"), "ARC": (-1.2, 0.3, "right"), "2Q": (-1.2, 0, "right"),
            "LIRS": (-1.2, 0, "right"), "S3-FIFO": (-1.2, 0.6, "right"),
            "W-TinyLFU": (0, 2.6, "center"), "TC-global": (-0.6, 0.9, "right"),
            "TC": (-0.8, 0, "right")}
    rows_a = []
    for pol in CLASSIC:
        x, y = mean_hit(pol, "empirical"), mean_hit(pol, "ring")
        rows_a.append({"policy": pol, "free_form_hit_pct": x, "ring_hit_pct": y})
        tc = pol == "TC"
        ax.scatter(x, y, s=30 if tc else 16, marker="D" if tc else "o",
                   color=CLASSIC_COL[pol], edgecolor="black", linewidth=0.4, zorder=4)
        dx, dy, ha = offs[pol]
        ax.text(x + dx, y + dy, CLASSIC_LAB.get(pol, pol), ha=ha, va="center", fontsize=FS - 1.5,
                color=TC_DARK if tc else "#333333", fontweight="bold" if tc else "normal")
    sd("fig4", "b", rows_a, "Mean request hit rate (%) over oversubscribed loads, "
       "averaged over 24 seeds, on ring teams and on free-form teams with recorded orders.")
    ax.set_xlim(42, 60)
    ax.set_ylim(-2, 37)
    ax.set_xlabel("Hit rate, free-form teams (%)")
    ax.set_ylabel("Hit rate, ring teams (%)")
    ax.annotate("", xy=(46.0, 10), xytext=(43.2, 3),
                arrowprops=dict(arrowstyle="->", lw=0.6, color="#999999"))
    ax.text(44.6, 3.0, "better", ha="left", va="center", fontsize=FS - 2, color="#999999")

    topos = ["ring", "pairs", "mixed", "debate", "star", "empirical"]
    s14 = loadj(SIM / "s14_group_baselines.json")
    over14 = [r for r in s14["grid"] if r["load"] > 1.01]
    over1 = [c for c in loadj(SIM / "s1_grid.json")["cells"] if c["load"] > 1.01]
    blind = [("LRU", over), ("Blind", over1)] + [(p_, over) for p_ in CLASSIC[1:-1]]
    group = [("Static", over1)] + [(p_, over14) for p_ in (
        "Partitioned LRU", "Partitioned ARC", "Partitioned W-TinyLFU", "UCP (hindsight)",
        "Max-min partitioning (hindsight)")]
    rows_b = blind + group + [("TC", over)]
    mat = np.array([[100 * min(r["worst_mean"] for r in src
                               if r["policy"] == pol and r["topo"] == t)
                     for t in topos] for pol, src in rows_b])
    ax = mm_axes(fig, 30, 9, 46, 50)
    mlabel(fig, 0, 66, "c")
    ax.imshow(mat, cmap=REUSE_CMAP, vmin=0, vmax=70, aspect="auto")
    for i in range(len(rows_b)):
        for j in range(len(topos)):
            v = mat[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=FS - 2.5,
                    color="white" if v > 45 else ("#B03A2E" if v < 5 else "#333333"),
                    fontweight="bold" if v < 5 else "normal")
    ax.set_xticks(range(len(topos)))
    ax.set_xticklabels([TOPO_LAB[t] for t in topos], rotation=90, fontsize=FS - 1.5)
    ax.set_yticks(range(len(rows_b)))
    short = {"TC-global": "TeamCache w/o teams", "Blind": "Unpart. Admission",
             "Static": "Static Partitioning", "Max-min partitioning (hindsight)":
             "Max-min (hindsight)", "Partitioned W-TinyLFU": "Part. W-TinyLFU",
             "Partitioned LRU": "Part. LRU", "Partitioned ARC": "Part. ARC"}
    ax.set_yticklabels([short.get(p_, CLASSIC_LAB.get(p_, p_)) for p_, _ in rows_b],
                       fontsize=FS - 2)
    ax.get_yticklabels()[-1].set_color(TC_DARK)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.tick_params(length=0, pad=1.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axhline(len(blind) - 0.5, color="white", lw=1.6)
    ax.add_patch(Rectangle((-0.5, len(rows_b) - 1.5), len(topos), 1, fc="none",
                           ec=TC_DARK, lw=0.9))
    ax.set_title("Worst-team reuse at the worst load (%)", fontsize=FS - 1, pad=3)
    sd("fig4", "c", [{"policy": SIMKEY.get(pol, pol), "topology": t, "min_worst_team_hit_pct": mat[i, j],
                      "runs": 10 if src is over14 else 24}
                     for i, (pol, src) in enumerate(rows_b) for j, t in enumerate(topos)],
       "Lowest mean worst-team request hit rate across oversubscribed loads, per topology; "
       "partitioned and hindsight policies from Supplementary Note 4 (10 runs), others 24 runs.")

    pols = ["Static", "ARC", "LIRS", "S3-FIFO", "W-TinyLFU", "TeamCache"]
    rep = {(r["system"], r["load"], r["policy"]): 100 * r["hit_mean"] for r in s8["replay"]}
    cells = sorted({(r["system"], r["load"]) for r in s8["replay"]})
    ax = mm_axes(fig, 112, 9, 58, 50)
    mlabel(fig, 86, 66, "d")
    rows_c = []
    sys_mark = {"AppWorld": "o", "ChatDev": "s", "HyperAgent": "^", "Magentic-One": "v"}
    for i, pol in enumerate(pols):
        d = [(sy, ld, rep[(sy, ld, pol)] - rep[(sy, ld, "LRU")]) for sy, ld in cells]
        rows_c.extend({"policy": pol, "system": sy, "load": ld, "hit_minus_lru_pts": v}
                      for sy, ld, v in d)
        vs = [v for *_, v in d]
        ax.plot([min(vs), max(vs)], [i, i], color=CLASSIC_COL[pol], lw=1.2, alpha=0.5,
                solid_capstyle="round", zorder=2)
        for k, (sy, ld, v) in enumerate(d):
            ax.scatter(v, i + (0.14 if ld > 2 else -0.14), s=8, marker=sys_mark[sy],
                       color=CLASSIC_COL[pol] if ld > 2 else "white",
                       edgecolor=CLASSIC_COL[pol], linewidth=0.5, zorder=3)
    sd("fig4", "d", rows_c, "Request hit rate minus LRU (percentage points) when public "
       "multi-agent traces are replayed as nine concurrent teams at load 1.5 and 2.5.")
    ax.axvline(0, color=COLORS["LRU"], lw=0.6, ls="--", zorder=1)
    ax.set_yticks(range(len(pols)))
    ax.set_yticklabels([CLASSIC_LAB.get(p_, p_) for p_ in pols], fontsize=FS - 1.5)
    ax.get_yticklabels()[-1].set_color(TC_DARK)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(len(pols) - 0.4, -1.9)
    ax.set_xlim(-34, 12)
    ax.set_xticks([-30, -20, -10, 0, 10])
    ax.set_xlabel("Hit rate minus LRU (points)")
    hs = [Line2D([], [], ls="", marker=m, ms=3, mfc="#999999", mec="#999999", mew=0.5,
                 label=sy) for sy, m in sys_mark.items()]
    ax.legend(handles=hs, loc="upper left", ncol=2, fontsize=FS - 2, handletextpad=0.1,
              borderaxespad=0.0, labelspacing=0.2, columnspacing=0.6,
              bbox_to_anchor=(0.0, 1.02))
    save(fig, "fig4")



# ------------------------------------------------------------------ Figure 5

def ring_seed_values():
    out = {}
    for arm, pol in (("lru", "LRU"), ("cohort", "Static Partitioning"),
                     ("unified", "TeamCache")):
        hits, spans = [], []
        for s in ("", "-s1", "-s2"):
            d = loadj(E2E / f"e2e-autogen-design-{arm}-20t-p3000r8{s}.json")
            hits.append(100 * d["steady"]["prefix_hit_rate"])
            spans.append(d["steady"]["span_s"])
        out[pol] = (hits, spans)
    return out


def selector_values():
    out = {}
    for arm, pol in (("lru", "LRU"), ("cohort", "Static Partitioning"),
                     ("unified", "TeamCache")):
        d = loadj(E2E / f"e2e-selector-debug-{arm}-20t-p3000r8.json")
        out[pol] = ([100 * d["steady"]["prefix_hit_rate"]], [d["wall_s"]])
    return out


def a100_values():
    """Per-seed (reuse %, completion time s) of the A100 repetitions, by policy."""
    doc = A100.summary()
    out = {}
    for key in ("ring", "selector"):
        sec = doc.get(key, {})
        out[key] = {p: (v["reuse_pct"]["runs"], v["time_s"]["runs"]) for p, v in sec.items()
                    if isinstance(v, dict) and "reuse_pct" in v and len(v["time_s"]["runs"]) == 3}
    sec = doc.get("mixed", {})
    out["mixed"] = {p: ([r["reuse_pct"] for r in v["runs"]], [r["time_s"] for r in v["runs"]])
                    for p, v in sec.items() if isinstance(v, dict) and len(v.get("runs", [])) == 3}
    return out


T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}


def paired_ci(a, b):
    a = np.asarray(a, float)
    d = 100 * (a - np.asarray(b, float)) / a
    if len(d) < 2:
        return d, d.mean(), None, None
    h = T975[len(d)] * d.std(ddof=1) / math.sqrt(len(d))
    return d, d.mean(), d.mean() - h, d.mean() + h


def fig5_scatter(ax, data, pols, note):
    for p in pols:
        reuse, time = np.asarray(data[p][0], float), np.asarray(data[p][1], float)
        ax.scatter(reuse, time, s=7, marker=MARKERS[p], color=COLORS[p], alpha=0.35, lw=0,
                   zorder=2)
        if len(reuse) > 1:
            ax.errorbar(reuse.mean(), time.mean(), xerr=reuse.std(ddof=1),
                        yerr=time.std(ddof=1), color=COLORS[p], lw=0.6, capsize=0, zorder=3)
        ax.scatter(reuse.mean(), time.mean(), s=26, marker=MARKERS[p], color=COLORS[p],
                   edgecolor="black", lw=0.5, zorder=4)
    l, t = data["LRU"], data["TeamCache"]
    p0 = (np.mean(l[0]), np.mean(l[1]))
    p1 = (np.mean(t[0]), np.mean(t[1]))
    if abs(p1[0] - p0[0]) > 3:
        ax.annotate("", xy=p1, xytext=p0, zorder=1,
                    arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#333333",
                                    shrinkA=4, shrinkB=4, mutation_scale=7,
                                    connectionstyle="arc3,rad=-0.2"))
    ax.text(0.97, 0.97, note, transform=ax.transAxes, ha="right", va="top",
            fontsize=FS - 1, fontweight="bold", color=TC_DARK, linespacing=1.1)


def fig5():
    ring = ring_seed_values()
    sel = selector_values()
    mixed = loadj(FD / "mixed_rows.json")
    mpols = ["LRU", "Unpartitioned Admission", "Offline LFU", "TeamCache"]
    mdata = {}
    for p in mpols:
        rows = sorted((r for r in mixed if r["policy"] == p), key=lambda r: r["seed"])
        mdata[p] = ([100 * r["token_reuse"] for r in rows], [r["makespan_s"] for r in rows])
    tpols = ["LRU", "Static Partitioning", "TeamCache"]
    fig = plt.figure(figsize=(DOUBLE, 62 * MM))
    top_legend(fig, ["LRU", "Unpartitioned Admission", "Static Partitioning", "Offline LFU",
                     "TeamCache"], lines=False)

    ring_speed = np.mean(ring["LRU"][1]) / np.mean(ring["TeamCache"][1])
    mixed_cut = 100 * (1 - np.mean(mdata["TeamCache"][1]) / np.mean(mdata["LRU"][1]))
    panels = [(ring, tpols, "Ring (AutoGen), n = 3", f"{ring_speed:.2f}× faster\nthan LRU",
               (-3, 80), (135, 200)),
              (sel, tpols, "Free-form (AutoGen selector), n = 1", "matches LRU",
               (20, 70), (380, 580)),
              (mdata, mpols, "Mixed: 6 ring + 6 selector, n = 4",
               f"{mixed_cut:.1f}% faster\nthan LRU", (25, 62), (120, 162))]
    for i, (data, pols, title, note, xl, yl) in enumerate(panels):
        ax = mm_axes(fig, 10 + 39 * i, 10, 30, 39)
        mlabel(fig, 0 + 39 * i, 56, "abc"[i])
        sd("fig5", "abc"[i],
           [{"policy": p, "run": j + 1, "prefix_token_reuse_pct": r, "completion_time_s": t}
            for p in pols for j, (r, t) in enumerate(zip(*data[p]))],
           f"{title}; each row is one run, markers in the panel show the mean and s.d.")
        fig5_scatter(ax, data, pols, note)
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        ax.set_title(title, loc="left", fontsize=FS - 1, pad=3)
        ax.set_xlabel("Prefix-token reuse (%)")
        if i == 0:
            ax.set_ylabel("Completion time (s)")
            ax.annotate("better", xy=(0.92, 0.1), xytext=(0.62, 0.32),
                        xycoords="axes fraction", fontsize=FS - 1.5, color="#777777",
                        ha="center", va="center",
                        arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#999999",
                                        mutation_scale=6))

    ax = mm_axes(fig, 146, 10, 32, 39)
    mlabel(fig, 120, 56, "d")
    a100 = a100_values()
    rows = [("Ring", "B300", "LRU", ring), ("Ring", "B300", "Static Partitioning", ring),
            ("Ring", "A100", "LRU", a100["ring"]),
            ("Ring", "A100", "Static Partitioning", a100["ring"]),
            ("Free-form", "B300", "LRU", sel), ("Free-form", "B300", "Static Partitioning", sel),
            ("Free-form", "A100", "LRU", a100["selector"]),
            ("Free-form", "A100", "Static Partitioning", a100["selector"]),
            ("Mixed", "B300", "LRU", mdata), ("Mixed", "B300", "Unpartitioned Admission", mdata),
            ("Mixed", "B300", "Offline LFU", mdata),
            ("Mixed", "A100", "LRU", a100["mixed"]),
            ("Mixed", "A100", "Unpartitioned Admission", a100["mixed"]),
            ("Mixed", "A100", "Offline LFU", a100["mixed"])]
    rows = [r for r in rows if r[3] and r[2] in r[3] and "TeamCache" in r[3]]
    ys, y, prev = [], 0.0, None
    for grp, hw, _, _ in rows:
        if prev is not None:
            y += 1.6 if grp != prev[0] else (1.25 if hw != prev[1] else 1.0)
        ys.append(y)
        prev = (grp, hw)
    rows_d = []
    for y, (grp, hw, base, data) in zip(ys, rows):
        d, m, lo, hi = paired_ci(data[base][1], data["TeamCache"][1])
        rows_d.extend({"scenario": grp, "server": hw, "baseline": base, "run": j + 1,
                       "time_saved_pct": float(v), "mean_time_saved_pct": float(m),
                       "t_ci95_low_pct": None if lo is None else float(lo),
                       "t_ci95_high_pct": None if hi is None else float(hi)}
                      for j, v in enumerate(d))
        a1 = hw == "A100"
        if lo is not None:
            ax.plot([lo, hi], [y, y], color=COLORS[base], lw=1.2, solid_capstyle="butt",
                    alpha=0.55 if a1 else 1.0)
            ax.scatter(d, [y] * len(d), s=5, color="white", edgecolor=COLORS[base], lw=0.5,
                       zorder=3)
            ax.scatter([m], [y], s=20, marker=MARKERS[base], color=COLORS[base],
                       edgecolor="black", lw=0.5, zorder=4, hatch="////" if a1 else None)
        else:
            ax.scatter([m], [y], s=20, marker=MARKERS[base], facecolor="white",
                       edgecolor=COLORS[base], lw=0.8, zorder=4)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{hw} vs {SHORT[b]}" for _, hw, b, _ in rows], fontsize=FS - 2.5)
    for t, (_, hw, _, _) in zip(ax.get_yticklabels(), rows):
        if hw == "A100":
            t.set_color("#666666")
    ax.tick_params(axis="y", length=0, pad=1)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(ys[-1] + 0.7, -1.0)
    firsts = {}
    for y, (grp, *_rest) in zip(ys, rows):
        firsts.setdefault(grp, y)
    for grp, y in firsts.items():
        ax.text(-0.02, y - 0.8, grp, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=FS - 1.5, fontweight="bold", color="#444444")
    ax.set_xlabel("Completion time saved\nby TeamCache (%)")
    ax.set_title("Paired difference, 95% CI", loc="left", fontsize=FS - 1, pad=3)
    sd("fig5", "d", rows_d, "Completion time saved by TeamCache relative to each baseline, "
       "100 x (baseline - TeamCache) / baseline, paired by run; Student t 95% CI. "
       "B300: Kimi K3; A100: GLM-4.7-Flash, three paired seeds.")
    save(fig, "fig5")


# ------------------------------------------------------------------ Figure 6

def split_meas(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["team"]].append(r)
    meas = []
    for xs in by.values():
        meas.extend(sorted(xs, key=lambda x: x["t_send"])[WARMUP_N:])
    return meas


def mixed_paths(key, seed):
    suf = "" if seed == 11 else f"_s{seed}"
    return GPU / f"mixed_{key}_12t{suf}.requests.jsonl"


MIXED_KEY = {"LRU": "lru", "Unpartitioned Admission": "group_blind",
             "Offline LFU": "static_rate", "TeamCache": "teamcache"}


def per_team_reuse(pol, seed):
    suf = "" if seed == 11 else f"_s{seed}"
    doc = loadj(GPU / f"mixed_{MIXED_KEY[pol]}_12t{suf}.json")
    return [t["prefix_reuse"] for t in sorted(doc["per_team"], key=lambda t: t["team"])]


def fig6():
    mixed = loadj(FD / "mixed_rows.json")
    mpols = ["LRU", "Unpartitioned Admission", "Offline LFU", "TeamCache"]
    fig = plt.figure(figsize=(DOUBLE, 100 * MM))
    top_legend(fig, mpols)

    ax = mm_axes(fig, 12, 60, 118, 31)
    mlabel(fig, 0, 96, "a")
    xs = np.arange(12)
    ax.axvspan(-0.5, 5.5, color=TOPO_COL["ring"], alpha=0.07, lw=0, zorder=0)
    ax.axvspan(5.5, 11.5, color=TOPO_COL["empirical"], alpha=0.07, lw=0, zorder=0)
    ax.text(2.5, 103, "ring teams", ha="center", va="bottom", fontsize=FS - 1,
            color=TOPO_COL["ring"], fontweight="bold")
    ax.text(8.5, 103, "selector (free-form) teams", ha="center", va="bottom", fontsize=FS - 1,
            color=TOPO_COL["empirical"], fontweight="bold")
    rows_a = []
    for p in mpols:
        mat = np.array([per_team_reuse(p, s) for s in SEEDS]) * 100
        rows_a.extend({"policy": p, "seed": s, "team": t,
                       "team_kind": "ring" if t < 6 else "selector",
                       "prefix_token_reuse_pct": mat[si, t]}
                      for si, s in enumerate(SEEDS) for t in range(12))
        for row in mat:
            ax.plot(xs, row, color=COLORS[p], lw=0.4, alpha=0.3, zorder=2)
        ax.plot(xs, mat.mean(0), color=COLORS[p], marker=MARKERS[p], ms=3, mec="black",
                mew=0.3, zorder=3)
    ax.set_xticks(xs)
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Team index")
    ax.set_ylabel("Prefix-token reuse (%)")
    sd("fig6", "a", rows_a, "Per-team prefix-token reuse in the mixed workload, four seeds.")

    ax = mm_axes(fig, 146, 60, 32, 31)
    mlabel(fig, 136, 96, "b")
    sd("fig6", "b",
       [{"policy": r["policy"], "seed": r["seed"], "low_reuse_teams": r["low_reuse_teams"]}
        for p in mpols for r in sorted(mixed, key=lambda r: r["seed"]) if r["policy"] == p],
       "Number of the 12 teams whose request miss rate is at least 90%.")
    for i, p in enumerate(mpols):
        v = [r["low_reuse_teams"] for r in mixed if r["policy"] == p]
        bars_with_points(ax, i, v, p)
        ax.text(i, np.mean(v) + np.std(v, ddof=1) + 0.35, f"{np.mean(v):.2g}",
                ha="center", fontsize=FS - 1.5,
                color=TC_DARK if p == "TeamCache" else "#333333",
                fontweight="bold" if p == "TeamCache" else "normal")
    ax.set_xticks(range(len(mpols)))
    ax.set_xticklabels(["LRU", "Unpart.", "Off.\nLFU", "Team-\nCache"], fontsize=FS - 1.5)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Low-reuse teams (of 12)")
    ax.set_ylim(0, 6.5)
    ax.set_title("Miss rate ≥ 90%", loc="left", fontsize=FS - 1, pad=3)

    mlabel(fig, 0, 47, "c")
    side = mm_axes(fig, 4.5, 8, 3.0, 32, frame=False)
    side.add_patch(Rectangle((0, 16.15), 3.0, 15.7, fc=TOPO_COL["ring"], ec="none", alpha=0.85))
    side.add_patch(Rectangle((0, 0.15), 3.0, 15.7, fc=TOPO_COL["empirical"], ec="none",
                             alpha=0.85))
    side.text(1.5, 24, "ring", rotation=90, ha="center", va="center", fontsize=FS - 1.5,
              color="white", fontweight="bold")
    side.text(1.5, 8, "selector", rotation=90, ha="center", va="center", fontsize=FS - 1.5,
              color="white", fontweight="bold")
    im = None
    rows_c = []
    for idx, p in enumerate(mpols):
        acc = np.zeros((12, 5))
        den = np.zeros((12, 5))
        for seed in SEEDS:
            for r in split_meas(load_jsonl(mixed_paths(MIXED_KEY[p], seed))):
                acc[r["team"], r["member"]] += r.get("cached_tokens") or 0
                den[r["team"], r["member"]] += r.get("prompt_tokens") or 0
        mat = np.divide(acc, den, out=np.zeros_like(acc), where=den > 0)
        rows_c.extend({"policy": p, "team": t, "team_kind": "ring" if t < 6 else "selector",
                       "member": m, "cached_tokens": int(acc[t, m]),
                       "prompt_tokens": int(den[t, m]), "prefix_token_reuse": mat[t, m]}
                      for t in range(12) for m in range(5))
        ax = mm_axes(fig, 13 + 36 * idx, 8, 29, 32)
        im = ax.imshow(mat, vmin=0, vmax=1, cmap=REUSE_CMAP, aspect="auto")
        ax.axhline(5.5, color="white", lw=1.2)
        ax.set_xticks(range(5))
        ax.set_yticks([0, 5, 6, 11])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel("Member")
        if idx:
            ax.set_yticklabels([])
        n_cov = int((mat >= 0.10).sum())
        ax.set_title(SHORT[p], loc="left", fontsize=FS - 0.5, pad=3, color=COLORS[p],
                     fontweight="bold")
        ax.text(1.0, 1.02, f"{n_cov}/60 slots ≥ 0.1", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=FS - 1.5,
                color=TC_DARK if p == "TeamCache" else "#555555",
                fontweight="bold" if p == "TeamCache" else "normal")
    cax = mm_axes(fig, 158, 8, 1.6, 32)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Prefix-token reuse", fontsize=FS - 1)
    cax.tick_params(labelsize=FS - 2, length=1.5)
    sd("fig6", "c", rows_c, f"Prefix-token reuse per member slot, pooled over four seeds "
       f"after the first {WARMUP_N} requests of each team.")
    save(fig, "fig6")


# ------------------------------------------------------------------ Figure 7

def fig7():
    rows = loadj(FD / "sweep_rows.json")
    pols = ["LRU", "Unpartitioned Admission", "Static Partitioning", "TeamCache"]
    teams = sorted({r["n_teams"] for r in rows})
    fig = plt.figure(figsize=(DOUBLE, 60 * MM))
    top_legend(fig, pols)

    def series(p, key):
        rs = sorted((r for r in rows if r["policy"] == p), key=lambda r: r["n_teams"])
        return [r["n_teams"] for r in rs], [r[key] for r in rs]

    def overflow(ax, ytext):
        ax.axvspan(9, 16.6, color="#F6E3E0", lw=0, zorder=0)
        ax.text(12.8, ytext, "working set\nexceeds pool", ha="center", va="top",
                fontsize=FS - 1.5, color="#B03A2E", linespacing=1.1)

    ax = mm_axes(fig, 12, 11, 46, 38)
    mlabel(fig, 0, 55, "a")
    overflow(ax, 101)
    for p in pols:
        xs, ys = series(p, "request_hit")
        ax.plot(xs, [100 * y for y in ys], color=COLORS[p], marker=MARKERS[p], mec="black",
                mew=0.3, zorder=3)
    tc10 = 100 * next(r["request_hit"] for r in rows
                      if r["policy"] == "TeamCache" and r["n_teams"] == 10)
    lru10 = 100 * next(r["request_hit"] for r in rows
                       if r["policy"] == "LRU" and r["n_teams"] == 10)
    ax.text(10.4, 63, f"{tc10:.0f}% vs {lru10:.1f}%\nat ten teams", fontsize=FS - 1.5,
            color=TC_DARK, fontweight="bold", va="bottom", linespacing=1.1)
    ax.set_xticks(teams)
    ax.set_xlim(5.4, 16.6)
    ax.set_xlabel("Concurrent five-member ring teams")
    ax.set_ylabel("Request hit rate (%)")
    ax.set_ylim(-3, 104)
    ax.set_title("Reuse", loc="left", fontsize=FS - 1, pad=3)

    ax = mm_axes(fig, 72, 11, 46, 38)
    mlabel(fig, 61, 55, "b")
    overflow(ax, 232)
    for p in pols:
        xs, ys = series(p, "wall_s")
        ax.plot(xs, ys, color=COLORS[p], marker=MARKERS[p], mec="black", mew=0.3, zorder=3)
    for b in (120, 150):
        ax.axhline(b, color="#888888", ls="--", lw=0.6, zorder=1)
        ax.text(5.6, b + 3, f"{b} s budget", fontsize=FS - 1.5, color="#555555")
    ax.set_xticks(teams)
    ax.set_xlim(5.4, 16.6)
    ax.set_ylim(40, 235)
    ax.set_xlabel("Concurrent five-member ring teams")
    ax.set_ylabel("Completion time (s)")
    ax.set_title("Measured-phase completion time", loc="left", fontsize=FS - 1, pad=3)

    ax = mm_axes(fig, 134, 11, 44, 38)
    mlabel(fig, 122, 55, "c")
    budgets = [120, 150]
    w = 0.19
    feas = {p: [max(r["n_teams"] for r in rows if r["policy"] == p and r["wall_s"] <= b)
                for b in budgets] for p in pols}
    for i, p in enumerate(pols):
        ax.bar(np.arange(2) + (i - 1.5) * w, feas[p], w * 0.9, color=COLORS[p],
               edgecolor="black", linewidth=0.4, alpha=0.6 if p == "LRU" else 0.85, zorder=2)
        for j, y in enumerate(feas[p]):
            ax.text(j + (i - 1.5) * w, y + 0.25, f"{y}", ha="center", fontsize=FS - 2)
    for j in range(2):
        gain = 100 * (feas["TeamCache"][j] / feas["LRU"][j] - 1)
        ax.text(j + 1.5 * w, feas["TeamCache"][j] + 1.6, f"+{gain:.0f}%", ha="center",
                fontsize=FS - 1, color=TC_DARK, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["120 s budget", "150 s budget"])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Largest feasible number of teams")
    ax.set_ylim(0, 15.5)
    ax.set_title("Serving capacity", loc="left", fontsize=FS - 1, pad=3)
    sweep = [{"policy": p, "teams": n, "request_hit_rate_pct": 100 * h, "completion_time_s": w}
             for p in pols
             for n, h, w in zip(series(p, "request_hit")[0], series(p, "request_hit")[1],
                                series(p, "wall_s")[1])]
    sd("fig7", "a", [{k: r[k] for k in ("policy", "teams", "request_hit_rate_pct")}
                     for r in sweep], "Request hit rate against the number of ring teams.")
    sd("fig7", "b", [{k: r[k] for k in ("policy", "teams", "completion_time_s")}
                     for r in sweep], "Measured-phase completion time.")
    sd("fig7", "c", [{"policy": p, "budget_s": b, "largest_feasible_teams": feas[p][j],
                      "gain_over_lru_pct": 100 * (feas[p][j] / feas["LRU"][j] - 1)}
                     for p in pols for j, b in enumerate(budgets)],
       "Largest team count whose completion time fits the budget.")
    save(fig, "fig7")


# ------------------------------------------------------------------ Figure 8

def ecdf(ax, vals, pol):
    xs = np.sort(np.asarray(vals, dtype=float))
    ys = np.arange(1, xs.size + 1) / xs.size * 100
    ax.plot(xs, ys, color=COLORS[pol], drawstyle="steps-post", lw=1.1, label=pol)


def fig8():
    rob = extract_robustness()
    setup()
    fig = plt.figure(figsize=(DOUBLE, 64 * MM))
    handles = [Line2D([], [], color=COLORS["LRU"], marker="o", mec="black", mew=0.3, ms=3.5,
                      label="LRU"),
               Line2D([], [], color=COLORS["TeamCache"], marker="D", mec="black", mew=0.3,
                      ms=3.5, label="TeamCache"),
               Line2D([], [], color=COLORS["LRU"], ls=":", marker="o", ms=3.5, mfc="white",
                      label="LRU without host memory")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0),
               fontsize=FS - 1, handletextpad=0.3, columnspacing=1.5)
    W, H = fig.get_size_inches() / MM

    def group(x0, x1, text):
        fig.add_artist(Line2D([x0 / W, x1 / W], [52.5 / H] * 2, color="#555555", lw=0.6))
        fig.text((x0 + x1) / 2 / W, 53.3 / H, text, ha="center", va="bottom",
                 fontsize=FS, fontweight="bold", color="#333333")

    group(4, 72, "SGLang HiCache (GPU + host memory)")
    group(80, 144, "GLM 5.3, 32 teams")
    group(152, 178, "vLLM")

    def speed(ax, h):
        ax.text(0.04, 0.97, f"{h['speedup']:.2f}× faster", transform=ax.transAxes,
                ha="left", va="top", fontsize=FS - 1, color=TC_DARK, fontweight="bold")
        ax.text(0.04, 0.85, f"{h['lru_wall']:.0f} s → {h['tc_wall']:.0f} s",
                transform=ax.transAxes, ha="left", va="top", fontsize=FS - 1.5,
                color="#555555")

    ax = mm_axes(fig, 12, 11, 24, 33)
    mlabel(fig, 0, 51, "a")
    n = 5
    xs = rob["short_rounds"][:n]
    ax.plot(xs, rob["short_gpu"][:n], color=COLORS["LRU"], ls=":", marker="o", mfc="white")
    ax.plot(xs, rob["short_lru"][:n], color=COLORS["LRU"], marker="o", mec="black", mew=0.3)
    ax.plot(xs, rob["short_tc"][:n], color=COLORS["TeamCache"], marker="D", mec="black",
            mew=0.3)
    ax.set_xticks(xs)
    ax.set_ylim(0, 102)
    ax.set_xlabel("Round")
    ax.set_ylabel("Prefix-token reuse (%)")
    ax.set_title("20-team ring", loc="left", fontsize=FS - 1, pad=3)
    sd("fig8", "a",
       [{"round": r, "lru_reuse_pct": a, "teamcache_reuse_pct": b,
         "lru_without_host_memory_reuse_pct": c}
        for r, a, b, c in zip(xs, rob["short_lru"][:n], rob["short_tc"][:n],
                              rob["short_gpu"][:n])],
       "Prefix-token reuse by round, SGLang HiCache, 20 ring teams.")

    for (x, key, title, l, lx) in ((48, "hicache", "10 long-context teams", "b", 39),
                                   (120, "glm", "Per-team miss rate", "d", 111)):
        ax = mm_axes(fig, x, 11, 24, 33)
        mlabel(fig, lx, 51, l)
        h = rob[key]
        ax.axvspan(90, 102, color="#F6E3E0", lw=0, zorder=0)
        ecdf(ax, h["lru_miss"], "LRU")
        ecdf(ax, h["tc_miss"], "TeamCache")
        ax.set_xlim(right=102)
        ax.set_xlabel("Team miss rate (%)")
        ax.set_ylabel("Teams (%)")
        ax.set_ylim(0, 104)
        ax.set_title(title, loc="left", fontsize=FS - 1, pad=3)
        speed(ax, h)
        sd("fig8", l,
           [{"policy": pol, "team_rank": j + 1, "team_miss_rate_pct": float(v),
             "cumulative_teams_pct": 100 * (j + 1) / len(vals)}
            for pol, vals in (("LRU", h["lru_miss"]), ("TeamCache", h["tc_miss"]))
            for j, v in enumerate(np.sort(np.asarray(vals, float)))],
           f"{title}; per-team miss-rate ECDF. Completion time {h['lru_wall']:.0f} s (LRU) "
           f"and {h['tc_wall']:.0f} s (TeamCache), {h['speedup']:.2f}x.")

    ax = mm_axes(fig, 85, 11, 24, 33)
    mlabel(fig, 75, 51, "c")
    g = rob["glm"]
    mem = np.arange(len(g["member_cached_lru"]))
    ax.bar(mem - 0.19, [100 * v for v in g["member_cached_lru"]], 0.36,
           color=COLORS["LRU"], alpha=0.6, edgecolor="black", linewidth=0.4)
    ax.bar(mem + 0.19, [100 * v for v in g["member_cached_tc"]], 0.36,
           color=COLORS["TeamCache"], alpha=0.85, edgecolor="black", linewidth=0.4)
    ax.set_xticks(mem)
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("Member index")
    ax.set_ylabel("Prefix-token reuse (%)")
    ax.set_ylim(0, 75)
    ax.set_title("Reuse by member", loc="left", fontsize=FS - 1, pad=3)
    sd("fig8", "c",
       [{"member": int(m), "lru_reuse_pct": 100 * a, "teamcache_reuse_pct": 100 * b}
        for m, a, b in zip(mem, g["member_cached_lru"], g["member_cached_tc"])],
       "Prefix-token reuse by member index, GLM 5.3 with 32 teams.")

    ax = mm_axes(fig, 160, 11, 18, 33)
    mlabel(fig, 148, 51, "e")
    v = rob["vllm"]
    lru = [v["r8_hits"][0], v["r16_hits"][0]]
    tcv = [v["r8_hits"][1], v["r16_hits"][1]]
    x = np.arange(2)
    ax.bar(x - 0.19, np.array(lru) / 1e3, 0.36, color=COLORS["LRU"], alpha=0.6,
           edgecolor="black", linewidth=0.4)
    ax.bar(x + 0.19, np.array(tcv) / 1e3, 0.36, color=COLORS["TeamCache"], alpha=0.85,
           edgecolor="black", linewidth=0.4)
    for i in range(2):
        ax.text(x[i] + 0.19, tcv[i] / 1e3 + 10, f"{tcv[i] / lru[i]:.2f}×",
                ha="center", fontsize=FS - 1, color=TC_DARK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["8", "16"])
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Block hits (thousands)")
    ax.set_title("8 ring teams", loc="left", fontsize=FS - 1, pad=3)
    ax.set_ylim(0, 480)
    ax.set_xlim(-0.6, 1.65)
    sd("fig8", "e",
       [{"rounds": r, "lru_block_hits": a, "teamcache_block_hits": b, "ratio": b / a}
        for r, a, b in zip((8, 16), lru, tcv)],
       "vLLM prefix-cache block hits, eight ring teams.")
    save(fig, "fig8")


# ------------------------------------------------------------------ SI figures

def si_ring_cliff():
    s2 = loadj(SIM / "s2_cliff.json")
    fig = plt.figure(figsize=(SINGLE, 62 * MM))
    ax = mm_axes(fig, 12, 10, 72, 46)
    ring = s2["ring"]
    cmap = plt.get_cmap("Blues")
    for i, kk in enumerate(range(3, 9)):
        col = cmap(0.35 + 0.6 * i / 5)
        rs = [r for r in ring if r["members"] == kk]
        caps = sorted({r["cap"] for r in rs}, reverse=True)
        lx = [TEAMS_S2 * kk / c for c in caps]
        sim = [100 * np.mean([r["lru"] for r in rs if r["cap"] == c]) for c in caps]
        th = [100 * next(r["che_closed"] for r in rs if r["cap"] == c) for c in caps]
        ax.plot(lx, th, color=col, lw=0.9)
        ax.scatter(lx, sim, s=6, color=col, zorder=3, label=f"{kk}", lw=0)
    ax.plot([0.6, 1.0, 1.0, 1.8], [100, 100, 0, 0], color="black", ls="--", lw=0.7)
    ax.text(1.03, 8, "synchronised\nlimit", fontsize=FS - 1.5, va="bottom", linespacing=1.05)
    ax.set_xlim(0.6, 1.8)
    ax.set_ylim(-4, 106)
    ax.set_xlabel("Load (histories / capacity)")
    ax.set_ylabel("LRU hit rate (%)")
    ax.set_title("Rings: experiment (dots), closed form (lines)", loc="left", fontsize=FS - 1,
                 pad=3)
    ax.legend(title="members k", title_fontsize=FS - 1.5, loc="upper right", fontsize=FS - 1.5,
              handletextpad=0.05, borderaxespad=0.1, labelspacing=0.12, ncol=2,
              columnspacing=0.6, markerscale=1.4)
    save(fig, "si_ring_cliff")


def si_heatmaps():
    s1 = loadj(SIM / "s1_grid.json")
    cells = s1["cells"]
    topos = list(TOPO_LAB)
    caps = sorted({c["cap"] for c in cells}, reverse=True)[1:]
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 110 * MM),
                             gridspec_kw={"wspace": 0.25, "hspace": 0.45})
    im = None
    for ax, key, letter in zip(axes.ravel(), ("LRU", "Blind", "Static", "TC"), "abcd"):
        label(ax, letter, dx=-0.12)
        mat = np.array([[next(c for c in cells if c["policy"] == key and c["topo"] == t
                              and c["cap"] == cap) for cap in caps] for t in topos])
        val = np.vectorize(lambda c: 100 * c["worst_mean"])(mat)
        im = ax.imshow(val, vmin=0, vmax=100, cmap=REUSE_CMAP, aspect="auto")
        for i in range(len(topos)):
            for j in range(len(caps)):
                c = mat[i, j]
                low = c["worst_mean"] < 0.05
                ax.text(j, i, f"{100 * c['worst_mean']:.1f}\n[{100 * c['worst_lo']:.1f},"
                        f"{100 * c['worst_hi']:.1f}]", ha="center", va="center",
                        fontsize=FS - 2, fontweight="bold" if low else "normal",
                        color="#B03A2E" if low else
                        "white" if c["worst_mean"] > 0.6 else "black")
                if low:
                    ax.add_patch(Rectangle((j - 0.46, i - 0.46), 0.92, 0.92, fill=False,
                                           lw=0.9, edgecolor="#B03A2E"))
        ax.set_xticks(range(len(caps)))
        ax.set_xticklabels([f"{63 / c:.2f}" for c in caps])
        ax.set_yticks(range(len(topos)))
        ax.set_yticklabels([TOPO_LAB[t] for t in topos])
        h = s1["headline"][key]
        ax.set_title(f"{SIMKEY[key]}: {h['cells_below_5pct']} cells < 5%, "
                     f"minimum {100 * h['min_cell_worst']:.1f}%", pad=3)
        ax.set_xlabel("Load")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("Worst-team hit rate (%), mean of 24 seeds [95% CI]")
    save(fig, "si_worst_heatmaps")


def si_frontier_and_cost():
    s1 = loadj(SIM / "s1_grid.json")
    s4 = loadj(SIM / "s4_cost.json")
    runs = s1["runs"]
    topos = ["ring", "star", "pairs", "debate", "mixed", "empirical"]
    fig = plt.figure(figsize=(DOUBLE, 62 * MM))
    gs = fig.add_gridspec(2, 5, wspace=0.45, hspace=0.6,
                          width_ratios=[1, 1, 1, 0.25, 2.2])
    for idx, t in enumerate(topos):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        for key in ("LRU", "Blind", "Static", "TC"):
            rs = [r for r in runs if r["policy"] == key and r["topo"] == t
                  and r["load"] >= 2.2]
            ax.scatter([100 * np.mean([r["hit"] for r in rs])],
                       [100 * np.mean([r["worst"] for r in rs])],
                       color=COLORS[SIMKEY[key]], marker=MARKERS[SIMKEY[key]], s=14)
        ax.set_xlim(-4, 78)
        ax.set_ylim(-4, 64)
        ax.set_title(TOPO_LAB[t], pad=2)
        if idx == 0:
            label(ax, "a", dx=-0.45)
        if idx % 3 == 0:
            ax.set_ylabel("Worst-team hit (%)")
        if idx >= 3:
            ax.set_xlabel("Mean hit (%)")
    ax = fig.add_subplot(gs[:, 4])
    for t in topos:
        rs = sorted([r for r in s4["rows"] if r["topo"] == t and r["policy"] == "TC"],
                    key=lambda r: r["ratio"])
        ax.plot([r["ratio"] for r in rs], [r["speedup"] for r in rs], marker="o",
                ms=2.5, color=TOPO_COL[t], label=TOPO_LAB[t])
    ax.axhline(1, color="#999999", ls=":", lw=0.6)
    ax.axvline(4.875, color="#999999", ls="--", lw=0.6)
    ax.axvline(21.5, color="#999999", ls="--", lw=0.6)
    ax.text(5.0, 0.915, "calibrated\n(9.7k tokens)", fontsize=FS - 2)
    ax.text(22.5, 0.915, "isolated\n(61k tokens)", fontsize=FS - 2)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xticks([2, 5, 10, 20, 30])
    ax.set_xlabel("Cold-miss / hit request cost ratio")
    ax.set_ylabel("Estimated TeamCache speedup over LRU")
    ax.set_ylim(0.9, 1.6)
    ax.legend(loc="upper left", fontsize=FS - 1.5, ncol=3)
    label(ax, "b", dx=-0.15)
    fig.legend(handles=legend_handles(list(SIMKEY.values())), loc="upper center",
               ncol=4, bbox_to_anchor=(0.32, 1.06), fontsize=FS - 1)
    save(fig, "si_frontier_cost")


def si_phase_detail():
    ph = phase_grids()
    ps, caps = ph["ps"], ph["caps"]
    fig, axes = plt.subplots(1, 4, figsize=(DOUBLE, 52 * MM),
                             gridspec_kw={"wspace": 0.3})
    for ax, key, title, letter in zip(axes, ("lru", "st", "tc"),
                                      ("LRU", "Static Partitioning", "TeamCache"), "abc"):
        label(ax, letter, dx=-0.12)
        mat = 100 * ph[key]
        ax.imshow(mat, origin="lower", aspect="auto", cmap=REUSE_CMAP, vmin=0, vmax=85)
        for i in range(len(ps)):
            for j in range(len(caps)):
                ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center",
                        fontsize=FS - 2.5, color="white" if mat[i, j] > 55 else "black")
        ax.set_title(f"{title}: hit rate (%)", pad=3, fontsize=FS - 1)
    best = np.maximum(ph["lru"], ph["st"])
    worse = np.minimum(ph["lru"], ph["st"])
    span = best - worse
    frac = np.where(span > 0.03, (ph["tc"] - worse) / np.where(span > 0, span, 1), np.nan)
    ax = axes[3]
    label(ax, "d", dx=-0.12)
    ax.imshow(frac, origin="lower", aspect="auto", cmap=REUSE_CMAP, vmin=0, vmax=1)
    for i in range(len(ps)):
        for j in range(len(caps)):
            v = frac[i, j]
            ax.text(j, i, "·" if np.isnan(v) else f"{100 * v:.0f}", ha="center", va="center",
                    fontsize=FS - 2.5, color="#888888" if np.isnan(v) else
                    ("white" if v > 0.65 else "black"))
    ax.contour(np.arange(len(caps)), np.arange(len(ps)), ph["pred"], levels=[0],
               colors="#333333", linewidths=0.7, linestyles="--")
    ax.set_title(f"Gap recovered (%), median {100 * np.nanmedian(frac):.0f}", pad=3,
                 fontsize=FS - 1, color=TC_DARK)
    for ax in axes:
        ax.set_xticks(range(len(caps)))
        ax.set_xticklabels([f"{63 / c:.1f}" for c in caps], fontsize=FS - 2)
        ax.set_yticks(range(len(ps)))
        ax.set_yticklabels([f"{p:.1f}" for p in ps], fontsize=FS - 2)
        ax.tick_params(length=0)
        ax.set_xlabel("Load")
    axes[0].set_ylabel("Ring regularity p")
    save(fig, "si_phase_detail")


def si_ring_turns():
    pols = ["LRU", "Static Partitioning", "TeamCache"]
    arms = {"LRU": "lru", "Static Partitioning": "cohort", "TeamCache": "unified"}
    setups = [("8×B300, Kimi K3", E2E, ""), ("4×A100, GLM-4.7-Flash", A100.A100 / "e2e", "-a100")]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 52 * MM), gridspec_kw={"wspace": 0.22})
    for ax, (title, root, tag), letter in zip(axes, setups, "ab"):
        first_over = None
        for pol in pols:
            mats = []
            for s in ("", "-s1", "-s2"):
                d = loadj(root / f"e2e-autogen-design-{arms[pol]}-20t-p3000r8{tag}{s}.json")
                mats.append([100 * t["prefix_hit_rate"] for t in d["by_turn"]])
                if first_over is None:
                    first_over = next(t["turn"] for t in d["by_turn"] if not t["fits"]) + 1
            n = min(len(m) for m in mats)
            mat = np.array([m[:n] for m in mats])
            xs = np.arange(1, n + 1)
            for row in mat:
                ax.plot(xs, row, color=COLORS[pol], lw=0.4, alpha=0.35, zorder=2)
            ax.plot(xs, mat.mean(0), color=COLORS[pol], marker=MARKERS[pol], mec="black",
                    mew=0.3, zorder=3)
        ax.axvspan(first_over - 0.5, n + 0.4, color="#F6E3E0", lw=0, zorder=0)
        ax.text((first_over - 0.5 + n + 0.4) / 2, 98, "working set exceeds pool", ha="center",
                va="top", fontsize=FS - 1.5, color="#B03A2E")
        ax.set_xlim(0.6, n + 0.4)
        ax.set_xticks(xs)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Round")
        ax.set_title(f"{title}, n = 3", pad=3)
        label(ax, letter, dx=-0.09)
    axes[0].set_ylabel("Prefix-token reuse (%)")
    top_legend(fig, pols)
    fig.subplots_adjust(top=0.84)
    save(fig, "si_ring_turns")


def si_discovery():
    fig = plt.figure(figsize=(DOUBLE, 50 * MM))
    ax = mm_axes(fig, 0, 0, 180, 50, frame=False)
    grey_fc, grey_ec = "#EEF0F2", "#6B7280"
    blue_fc, blue_ec = "#DCE6F2", MEMBER_COL[0]
    orange_fc, orange_ec = "#FBE5C8", "#D08C2F"
    lineage, handoff = MEMBER_COL[0], "#C8701E"

    mlabel(fig, 1, 49.5, "a")
    ax.text(6, 46.8, "Prompt evidence seen by the gateway", fontsize=FS, va="center")
    rows = [("$A_1$", "private A", None), ("$A_2$", "private A + new turn", None),
            ("$B_1$", "private B", "span copied from A"), ("$X_1$", "private X", None)]
    ys = [38, 28, 18, 8]
    for (name, priv, span), y in zip(rows, ys):
        ax.text(9, y, name, fontsize=FS, ha="right", va="center")
        rbox(ax, 12, y - 3.2, 20, 6.4, fc=grey_fc, ec=grey_ec)
        ax.text(22, y, "shared system", fontsize=FS - 1, ha="center", va="center")
        rbox(ax, 34, y - 3.2, 27, 6.4, fc=blue_fc, ec=blue_ec)
        ax.text(47.5, y, priv, fontsize=FS - 1, ha="center", va="center")
        if span:
            rbox(ax, 63, y - 3.2, 27, 6.4, fc=orange_fc, ec=orange_ec)
            ax.text(76.5, y, span, fontsize=FS - 1, ha="center", va="center")
    arrow(ax, (62, 38), (62, 28.2), color=lineage, lw=0.9, rad=-0.6)
    ax.text(68, 33, "extends a distinctive private prefix:\nsame member lineage",
            fontsize=FS - 1.5, color=lineage, va="center", ha="left")
    arrow(ax, (54, 24.6), (72, 21.4), color=handoff, lw=0.9, ls="--", rad=0.25)
    ax.text(92, 20.5, "unique copied span:\nhandoff edge A–B", fontsize=FS - 1.5,
            color=handoff, va="center", ha="left")
    ax.text(63, 8, "matches only the shared template:\nno edge", fontsize=FS - 1.5,
            color=grey_ec, va="center", ha="left")
    ax.plot([118, 118], [3, 47], color="#CCCCCC", lw=0.5)

    mlabel(fig, 121, 49.5, "b")
    ax.text(126, 46.8, "Inferred identities", fontsize=FS, va="center")
    rbox(ax, 124, 14, 32, 27, fc="#E3F2EC", ec=TC_DARK, lw=0.7, r=1.5)
    ax.text(140, 37.5, "Team 1 (connected component)", fontsize=FS - 1.5, ha="center",
            va="center", color=TC_DARK)
    rbox(ax, 159, 14, 18, 27, fc=grey_fc, ec=grey_ec, lw=0.7, r=1.5)
    ax.text(168, 37.5, "Team 2", fontsize=FS - 1.5, ha="center", va="center", color=grey_ec)
    for x, name, ec in ((131, "A", blue_ec), (149, "B", blue_ec), (168, "X", grey_ec)):
        ax.add_patch(Circle((x, 25), 4.2, fc="white", ec=ec, lw=0.8, zorder=3))
        ax.text(x, 25, name, fontsize=FS, fontweight="bold", ha="center", va="center", zorder=4)
    ax.plot([135.2, 144.8], [25, 25], color=handoff, lw=0.9, ls="--", zorder=2)
    ax.text(140, 27.5, "handoff", fontsize=FS - 2, color=handoff, ha="center", va="bottom")
    ax.text(150.5, 6.5, "Uncertain prompts stay in their own\ncomponent; later handoffs can merge"
            " them.", fontsize=FS - 1.5, color="#555555", ha="center", va="center")
    save(fig, "si_discovery")


def si_star():
    from teamcache_ncomms.gpu_tables import extract_star
    pols = ["LRU", "Unpartitioned Admission", "Offline LFU", "TeamCache"]
    b300 = {r["policy"]: r for r in extract_star()}
    a100 = A100.summary()["star"]
    panels = [
        ("8×B300, Kimi K3, n = 1", "Completion time (s)",
         {p: [b300[p]["makespan_s"]] for p in pols}),
        ("8×B300, Kimi K3, n = 1", "Request hit rate (%)",
         {p: [100 * b300[p]["request_hit"]] for p in pols}),
        ("4×A100, GLM-4.7-Flash, n = 3", "Completion time (s)",
         {p: [r["time_s"] for r in a100[p]["runs"]] for p in pols}),
        ("4×A100, GLM-4.7-Flash, n = 3", "Request hit rate (%)",
         {p: [r["request_hit_pct"] for r in a100[p]["runs"]] for p in pols}),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(DOUBLE, 50 * MM), gridspec_kw={"wspace": 0.5})
    for ax, (title, ylab, vals), letter in zip(axes, panels, "abcd"):
        for i, p in enumerate(pols):
            bars_with_points(ax, i, vals[p], p)
        ax.set_xticks(range(len(pols)))
        ax.set_xticklabels([SHORT[p] for p in pols], rotation=35, ha="right")
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 200 if "time" in ylab else 100)
        ax.set_title(title, pad=3, fontsize=FS - 1)
        label(ax, letter, dx=-0.3)
    save(fig, "si_star")


def main():
    font = setup()
    print("font", font)
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    fig8()
    si_ring_cliff()
    si_heatmaps()
    si_frontier_and_cost()
    si_phase_detail()
    si_ring_turns()
    si_discovery()
    si_star()


if __name__ == "__main__":
    main()
