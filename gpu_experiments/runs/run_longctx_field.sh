#!/usr/bin/env bash
# 61k ring on the live 420,800 pool. Bypass is exact only when the whole
# prompt is one prefill: chunked inserts always land in the radix tree, so
# 15 members x first 32k page flood the pool and the keep-set dies. The
# engine must be started with CHUNKED_PREFILL_SIZE >= 65536.
set -euo pipefail
cd "$(dirname "$0")"
CTRL="${CTRL:-/tmp/cohort/control.json}"
ENGINE_PORT="${ENGINE_PORT:-30000}"
POOL="${POOL:-420800}"
TEAMS="${TEAMS:-3}"
MEMBERS="${MEMBERS:-5}"
ROUNDS="${ROUNDS:-6}"
WARMUP="${WARMUP:-5}"
CONTEXT="${CONTEXT:-61440}"
GEN="${GEN:-144}"
MESSAGE="${MESSAGE:-512}"
OUTDIR="${OUTDIR:-/opt/dlami/nvme/pat/results/e2e}"
mkdir -p "$OUTDIR" /tmp/logs

RETAIN=$(( CONTEXT + MESSAGE + GEN ))
CAP=$(( POOL / RETAIN ))
(( CAP < 1 )) && CAP=1
LOAD=$(python3 -c "print(f'{$TEAMS * $MEMBERS / $CAP:.2f}')")
echo "longctx field  teams=$TEAMS context=$CONTEXT pool=$POOL cap=$CAP load=${LOAD}x $(date -u +%H:%M:%S)"

set_arm() {
  local mode="$1" quota="$2"
  cat >"$CTRL" <<EOF
{"mode": "$mode", "quota": $quota, "global_cap": 0, "members": $MEMBERS,
 "capacity": $CAP, "rotate_period": 0, "window_s": 30,
 "pool_tokens": $POOL, "adapt_threshold": 0.6, "adapt_window": 40}
EOF
  curl -s -m 45 "http://127.0.0.1:${ENGINE_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"kimi","messages":[{"role":"user","content":"ok"}],"max_tokens":1}' \
    >/dev/null || true
  sleep 1
  echo "arm $mode quota=$quota $(date -u +%H:%M:%S)"
}

run_one() {
  local label="$1" mode="$2" quota="$3" topo="$4"
  local out="$OUTDIR/${label}.json"
  if [[ -f "$out" ]]; then
    echo "skip $label (have $out)"
    return
  fi
  set_arm "$mode" "$quota"
  python3 replay_team.py \
    --label "$label" \
    --port "$ENGINE_PORT" \
    --teams "$TEAMS" --members "$MEMBERS" \
    --topology "$topo" \
    --context "$CONTEXT" \
    --message "$MESSAGE" --gen-tokens "$GEN" \
    --rounds "$ROUNDS" --warmup-rounds "$WARMUP" \
    --stagger 1.5 \
    --capacity "$CAP" \
    --reset-multiple 0.4 \
    --out "$out"
}

run_one e2e-longctx-ring-lru-3t-p61k-c64k      group   0 ring
run_one e2e-longctx-ring-unified-3t-p61k-c64k   unified 0 ring
run_one e2e-longctx-ring-cohort-3t-p61k-c64k    group   1 ring
echo "longctx field DONE $(date -u +%H:%M:%S)"
