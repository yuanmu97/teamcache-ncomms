# Host settings for the 4xA100-80GB repetition campaign.
# Sourced by every a100 driver; mirrors engine/common_b300.sh.
TC="${TC:-/srv/teamcache/tc-a100}"
VENV="${VENV:-/srv/teamcache/glm47-flash/venv}"
MODEL="${MODEL:-/srv/teamcache/glm47-flash/model}"
SERVED="${SERVED:-glm-4.7-flash}"
AGPY="${AGPY:-$TC/.venv-ag/bin/python}"
ENGINE_PORT="${ENGINE_PORT:-30000}"
# Same token pool as the Kimi K3 B300 runs, so every workload sits at the same
# oversubscription in tokens.
POOL="${POOL:-420800}"
CTRL="${CTRL:-/tmp/cohort/control.json}"
OUT="${OUT:-$TC/results-a100}"
LOGS="${LOGS:-$TC/logs}"
ENGINE_LOG="${ENGINE_LOG:-$LOGS/engine.log}"
DEBUG_HOST="${DEBUG_HOST:-/tmp/unified_debug.jsonl}"
mkdir -p "$OUT" "$LOGS" "$(dirname "$CTRL")"

# Write the control file and push one request through the cache path so the
# engine applies it before the arm starts.
set_policy() {  # mode quota global_cap members capacity
  "$VENV/bin/python" - "$CTRL" "$1" "$2" "$3" "$4" "$5" "$POOL" <<'PY'
import json, os, sys
path, mode, quota, gcap, members, cap, pool = sys.argv[1:8]
doc = {"mode": mode, "quota": int(quota), "global_cap": int(gcap),
       "members": int(members), "capacity": int(cap), "rotate_period": 0,
       "window_s": 30, "pool_tokens": int(pool), "adapt_threshold": 0.6,
       "adapt_window": 40, "adapt_margin": 1.02, "adapt_min_samples": 2}
with open(path, "w") as fh:
    json.dump(doc, fh)
    fh.flush(); os.fsync(fh.fileno())
PY
  sleep 1.2
  curl -s -m 60 "http://127.0.0.1:${ENGINE_PORT}/generate" \
    -H "Content-Type: application/json" \
    -d '{"input_ids":[1,2,3,4],"sampling_params":{"max_new_tokens":1},"rid":"policyprobe:0:0"}' >/dev/null || true
  sleep 0.5
  echo "policy -> $1 quota=$2 gcap=$3 :: $(grep '\[cohort\] gen=' "$ENGINE_LOG" | tail -1)"
}

flush_cache() {
  curl -s -m 60 -X POST "http://127.0.0.1:${ENGINE_PORT}/flush_cache" >/dev/null || \
  curl -s -m 60 "http://127.0.0.1:${ENGINE_PORT}/flush_cache" >/dev/null || true
  sleep 3
}

wait_health() {
  local i
  for i in $(seq 1 360); do
    if curl -sf -m 3 "http://127.0.0.1:${ENGINE_PORT}/health" >/dev/null 2>&1; then
      echo "engine healthy after $((i * 5))s"; return 0
    fi
    sleep 5
  done
  echo "engine failed to become healthy" >&2; return 1
}

log_run() {  # name status seconds note
  [[ -f "$OUT/runs.csv" ]] || echo "ts,name,status,run_s,note" >"$OUT/runs.csv"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),$1,$2,$3,${4:-}" >>"$OUT/runs.csv"
}
