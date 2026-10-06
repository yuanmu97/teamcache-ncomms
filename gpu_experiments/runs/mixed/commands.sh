#!/usr/bin/env bash
# Commands actually run and checked.
set -euo pipefail
cd "$(dirname "$0")"

# --- Phase 0, local Mac (Apple M3), 0 GPU ---
python3 phase0_logs.py
python3 phase0_cpu.py
python3 mixed_footprint.py
python3 select_static_rate.py --workload star --teams 9 --seed 11
python3 select_static_rate.py --workload mixed --teams 12 --seed 11
# plan.json frozen 2026-09-18T01:28:00Z before any B300 /generate.

# --- Phase 1–3, B300 ---
# rsync additional_validation/ -> /opt/dlami/nvme/pat/additional_validation/
# rsync deploy/serve_kimi_k3_b300_sglang_cohort.sh (debug-file volume only)
# stopped idle glm53-b300-sglang; started kimi-k3-b300-sglang-cohort
bash run_gpu.sh
# verified: star Static-rate + TeamCache seed 11; mixed 4 arms × seeds 11,12,13,14
# node_elapsed 4889s = 10.864 GPU-hours. extra star seeds not started (budget rule).
python3 plot_gpu.py
python3 collect_tables.py
python3 write_report.py
