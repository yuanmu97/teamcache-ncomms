#!/usr/bin/env python3
"""Pull 1K-decode and $/1M numbers out of a bench/run.sh --profile core dir."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
import parse as P  # noqa: E402

USD_H = 112.32


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else
             ROOT / "bench" / "results" / "20260914-cohort-off")
    rows = []
    for path in sorted(d.glob("*.jsonl")):
        if path.name.startswith("_"):
            continue
        raw = P._last_json(path)
        if not raw:
            continue
        out_tps = float(P._f(raw, "output_throughput", default=0) or 0)
        tot_tps = float(P._f(raw, "total_throughput", default=0) or 0)
        osl = int(P._f(raw, "random_output_len", default=0) or 0)
        isl = int(P._f(raw, "random_input_len", default=0) or 0)
        c = int(P._f(raw, "max_concurrency", default=0) or 0)
        usd_out = P.dollar_per_m(USD_H, out_tps)
        usd_tot = P.dollar_per_m(USD_H, tot_tps)
        rows.append({
            "cell": path.stem, "isl": isl, "osl": osl, "c": c,
            "out_tps": out_tps, "tot_tps": tot_tps,
            "usd_per_m_out": usd_out, "usd_per_m_tot": usd_tot,
            "p50_e2e_s": (P._f(raw, "median_e2e_latency_ms", default=0) or 0) / 1000,
        })
        print(f"{path.stem:<22} c={c:<3} {isl}/{osl}  "
              f"out={out_tps:7.1f} tok/s  "
              f"$/1M_out={usd_out:6.3f}" if usd_out else f"{path.stem} no tps")
    out = Path(__file__).resolve().parent / "results" / "bench-core-summary.json"
    json.dump({"dir": str(d), "usd_h": USD_H, "rows": rows}, open(out, "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
