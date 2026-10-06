#!/usr/bin/env bash
# Device+host jointly oversubscribed pair. The 20-team HiCache pair fit in
# host DRAM (LRU 90%). hicache_ratio=2 → joint pool ≈ 3× device = 1,262,400.
# 48 teams: first turn 0.62×, peak 1.71× — same shape as the device-only
# 20-team cliff (0.77× → 2.14×). unified is told the joint pool, not the
# device pool, so it does not concentrate while host still fits.
set -uo pipefail
cd "$(dirname "$0")"
ROOT=/opt/dlami/nvme/pat
DEPLOY="${DEPLOY:-$ROOT/deploy}"
export ENABLE_HICACHE=1
export HICACHE_RATIO="${HICACHE_RATIO:-2}"
export TEAMS="${TEAMS:-48}"
export MEMBERS=5 ROUNDS=8 PREAMBLE=3000 GEN_CAP=144 QUOTA=2
export POOL="${POOL:-1262400}"
export TAG_SUFFIX=-hicache-joint
export MODEL="${MODEL:-/opt/dlami/nvme/models/Kimi-K3}"
export ENGINE_PORT="${ENGINE_PORT:-30000}"

echo "hicache-joint restart ratio=$HICACHE_RATIO pool=$POOL teams=$TEAMS $(date -u +%H:%M:%S)"
if [[ ! -x "$DEPLOY/serve_kimi_k3_b300_sglang_cohort.sh" ]]; then
  echo "missing $DEPLOY/serve_kimi_k3_b300_sglang_cohort.sh" >&2
  exit 2
fi
docker rm -f glm53-b300-sglang kimi-k3-b300-vllm kimi-k3-b300-sglang-cohort >/dev/null 2>&1 || true
sleep 2
(cd "$DEPLOY" && ENABLE_HICACHE=1 HICACHE_RATIO="$HICACHE_RATIO" \
  ./serve_kimi_k3_b300_sglang_cohort.sh)
./wait_engine.sh "$ENGINE_PORT" 180 || exit 1

echo "hicache-joint lru $(date -u +%H:%M:%S)"
./e2e_autogen.sh lru "$TEAMS" "$MEMBERS" "$ROUNDS"
echo "hicache-joint unified $(date -u +%H:%M:%S)"
./e2e_autogen.sh unified "$TEAMS" "$MEMBERS" "$ROUNDS"
echo "hicache-joint DONE $(date -u +%H:%M:%S)"
