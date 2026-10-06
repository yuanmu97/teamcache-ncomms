#!/usr/bin/env python3
"""126-cell margin sweep + no-bypass ablation. CPU only."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import optimal as O

TOPOS = ["ring", "empirical", "star", "debate", "pairs", "mixed"]
LOADS = [1.0, 1.67, 2.33, 3.0, 3.67, 4.33, 5.0]
MARGINS = [0.95, 1.00, 1.02, 1.05, 1.10]
TEAMS, MEMBERS, ROUNDS, SEEDS = 9, 7, 40, 3
CONTEXT = 9728
OUT = Path(__file__).resolve().parent / "results" / "sensitivity-126.json"


def cells():
    sess = TEAMS * MEMBERS
    caps = [max(int(round(sess / L)), 1) for L in LOADS]
    rows = []
    for topo in TOPOS:
        for seed in range(SEEDS):
            refs = O.topo_reference(topo, TEAMS, MEMBERS, ROUNDS, seed=seed)
            tof = [t for t, _ in refs]
            for cap in caps:
                fixed = {
                    "LRU": O.run_lru(refs, cap),
                    "bypass": O.run_lru_bypass(refs, cap),
                    "Cohort": O.run_cohort(refs, cap, MEMBERS),
                }
                sc = {a: O.score(a, f, tof, CONTEXT) for a, f in fixed.items()}
                best = min(sc, key=lambda a: sc[a]["miss_rate"])
                bm = sc[best]["miss_rate"]
                rec = {
                    "topo": topo, "seed": seed, "cap": cap,
                    "load": sess / cap, "best_fixed": best,
                    "best_miss": bm,
                }
                for m in MARGINS:
                    f = O.run_concentrate(refs, cap, MEMBERS, margin=m)
                    s = O.score("u", f, tof, CONTEXT)
                    rec[f"m{m}"] = (s["miss_rate"] - bm) * 100
                    rec[f"m{m}_starve"] = s["starved_teams"]
                for name, byp in (("bypass", True), ("admit", False)):
                    f = O.run_concentrate(refs, cap, MEMBERS,
                                          margin=1.02, bypass=byp)
                    s = O.score(name, f, tof, CONTEXT)
                    rec[name] = s["miss_rate"]
                    rec[f"{name}_starve"] = s["starved_teams"]
                rows.append(rec)
    return rows


def summarize(rows):
    print(f"{len(rows)} cells  {len(TOPOS)} topo x {len(LOADS)} load x {SEEDS} seed")
    print("=== margin: gap to best fixed, pp ===")
    print(f"{'margin':>8}{'median':>9}{'p90':>8}{'worst':>8}{'<=1pp':>8}{'starve':>8}")
    out = {"n": len(rows), "margins": {}, "bypass": {}}
    for m in MARGINS:
        g = sorted(r[f"m{m}"] for r in rows)
        p90 = g[min(int(0.9 * len(g)), len(g) - 1)]
        starve = sum(r[f"m{m}_starve"] for r in rows)
        le1 = sum(1 for x in g if x <= 1)
        print(f"{m:8.2f}{statistics.median(g):+9.2f}{p90:+8.2f}{max(g):+8.2f}"
              f"{le1:5}/{len(g):<3}{starve:8}")
        out["margins"][str(m)] = {
            "median": statistics.median(g), "p90": p90, "worst": max(g),
            "le1": le1, "starve": starve,
        }
    print("\n=== no-bypass (admit) vs decline, miss rate ===")
    print(f"{'topo':<12}{'bypass':>8}{'admit':>8}{'delta':>8}{'starve b/a':>12}")
    by_topo = defaultdict(list)
    for r in rows:
        by_topo[r["topo"]].append(r)
    for topo in TOPOS:
        rs = by_topo[topo]
        b = statistics.mean(x["bypass"] for x in rs)
        a = statistics.mean(x["admit"] for x in rs)
        sb = sum(x["bypass_starve"] for x in rs)
        sa = sum(x["admit_starve"] for x in rs)
        print(f"{topo:<12}{b*100:7.1f}%{a*100:7.1f}%{(a-b)*100:+8.1f}pp"
              f"{sb:5}/{sa}")
        out["bypass"][topo] = {
            "bypass_miss": b, "admit_miss": a, "delta_pp": (a - b) * 100,
            "starve_bypass": sb, "starve_admit": sa,
        }
    return out


def main():
    rows = cells()
    summary = summarize(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "rows": rows}, open(OUT, "w"), indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
