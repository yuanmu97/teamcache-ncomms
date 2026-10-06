#!/usr/bin/env bash
# Joint device+host oversub on a realistic node: a handful of concurrent
# 5-member teams, not 48 copies of one small-preamble ring.
#
# Demand scales by the histories those teams already carry (from collected
# medians), not by cloning the toy 3k-preamble geometry. Replay, not a new
# live-trace wave.
#
# Default mix (10 teams × 5):
#   6 × 56k  AutoGen/CrewAI design band (collected 38–65k)
#   2 × 12k  Selector debug / Claude Code band
#   2 × 16k  Selector review / mid
# Working set 1.96M vs joint pool 1.26M (hicache_ratio=2) ≈ 1.55×.
# Speaking order is mixed (ring / empirical / star) so the node is not one
# topology either.
set -uo pipefail
cd "$(dirname "$0")"
CTRL="${CTRL:-/tmp/cohort/control.json}"
ENGINE_PORT="${ENGINE_PORT:-30000}"
TEAMS="${TEAMS:-10}"
MEMBERS="${MEMBERS:-5}"
ROUNDS="${ROUNDS:-6}"
POOL="${POOL:-1262400}"
MIX="${TEAM_CONTEXTS:-56000,56000,56000,56000,56000,56000,12000,12000,16000,16000}"
OUTDIR="${OUTDIR:-/opt/dlami/nvme/pat/results/e2e}"
mkdir -p "$OUTDIR"

mean=$(python3 -c "xs=[int(x) for x in '$MIX'.split(',')]; print(sum(xs)//len(xs))")
cap=$(( POOL / mean ))
(( cap < 1 )) && cap=1
echo "fleet teams=$TEAMS mix=$MIX pool=$POOL mean=$mean cap_sessions=$cap $(date -u +%H:%M:%S)"

set_arm() {
  local mode="$1" quota="$2"
  cat >"$CTRL" <<EOF
{"mode": "$mode", "quota": $quota, "global_cap": 0, "members": $MEMBERS,
 "capacity": $cap, "rotate_period": 0, "window_s": 30,
 "pool_tokens": $POOL, "adapt_threshold": 0.6, "adapt_window": 40}
EOF
  curl -s -m 45 "http://127.0.0.1:${ENGINE_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"kimi","messages":[{"role":"user","content":"ok"}],"max_tokens":1}' \
    >/dev/null || true
  sleep 1
  echo "arm $mode quota=$quota $(date -u +%H:%M:%S)"
}

run_arm() {
  local label="$1" mode="$2" quota="$3"
  set_arm "$mode" "$quota"
  python3 replay_team.py \
    --label "$label" \
    --port "$ENGINE_PORT" \
    --teams "$TEAMS" --members "$MEMBERS" \
    --topology mixed \
    --team-contexts "$MIX" \
    --message 512 --gen-tokens 144 \
    --rounds "$ROUNDS" --warmup-rounds 2 \
    --stagger 1.5 \
    --capacity "$cap" \
    --reset-multiple 0.4 \
    --out "$OUTDIR/${label}.json"
}

run_arm e2e-fleet-mixed-lru-10t-hicache-joint group 0
run_arm e2e-fleet-mixed-unified-10t-hicache-joint unified 0
echo "hicache-fleet DONE $(date -u +%H:%M:%S)"
