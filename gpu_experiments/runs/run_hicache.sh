#!/usr/bin/env bash
# Phase 4: restart cohort engine with HiCache, measure host-tier hit cost.
set -uo pipefail
cd /opt/dlami/nvme/h200-llm/deploy 2>/dev/null || cd "$(dirname "$0")/../../deploy"
echo "restarting cohort engine with ENABLE_HICACHE=1 $(date -u +%H:%M:%S)"
ENABLE_HICACHE=1 ./serve_kimi_k3_b300_sglang_cohort.sh
# wait for health
for i in $(seq 1 180); do
  if curl -sf -m 3 http://127.0.0.1:30000/v1/models >/dev/null; then
    echo "engine up after ${i}s"
    break
  fi
  sleep 10
done
cd /opt/dlami/nvme/pat
python3 measure_hicache.py --port 30000 --context 3256 --pool-tokens 420800 \
  --out results/hicache_cost.json
echo "hicache measure DONE $(date -u +%H:%M:%S)"
cat results/hicache_cost.json
