#!/usr/bin/env bash
# GLM-5.3 natural pool, AutoGen-style ring, official LRU only.
# Working-set sweep for the background cliff figure: hit rate vs tokens.
#
#   ssh b300 'cd /opt/dlami/nvme/pat && ./run_glm_ring_lru.sh'
set -uo pipefail
cd "$(dirname "$0")"

exec 9>/tmp/glm-ring-lru.lock
if ! flock -n 9; then
  echo "another run_glm_ring_lru.sh holds the lock" >&2
  exit 3
fi

ROOT="${ROOT:-/opt/dlami/nvme/pat}"
DEPLOY="${DEPLOY:-$ROOT/deploy}"
ENGINE_PORT="${ENGINE_PORT:-30000}"
CTRL="${CTRL:-/tmp/cohort/control.json}"
OUTDIR="${OUTDIR:-$ROOT/results/e2e}"
MEMBERS="${MEMBERS:-5}"
ROUNDS="${ROUNDS:-5}"
WARMUP="${WARMUP:-2}"
CONTEXT="${CONTEXT:-56000}"
GEN="${GEN:-144}"
MESSAGE="${MESSAGE:-512}"
TEAM_COUNTS="${TEAM_COUNTS:-6 8 10 12}"
CONTAINER="${CONTAINER:-glm53-b300-sglang}"
mkdir -p "$OUTDIR" /tmp/logs "$ROOT/results/glm"

echo "glm-ring-lru start $(date -u +%H:%M:%S)"

if [[ "${SKIP_START:-0}" != "1" ]]; then
  python3 "$ROOT/patch_sglang_cohort.py" --out /tmp/sgl-patch
  docker rm -f kimi-k3-b300-vllm kimi-k3-b300-sglang-cohort glm53-b300-sglang \
    >/dev/null 2>&1 || true
  sleep 3
  (cd "$DEPLOY" && ./serve_glm53_b300_sglang_cohort.sh)
fi
bash "$ROOT/wait_engine.sh" "$ENGINE_PORT" 180 || exit 1

POOL=$(python3 - <<'PY'
import json, urllib.request, sys
try:
    d = json.loads(urllib.request.urlopen("http://127.0.0.1:30000/get_server_info", timeout=8).read())
    for k in ("max_total_num_tokens", "max_total_tokens"):
        if k in d:
            print(int(d[k])); raise SystemExit
        mem = d.get("memory") or {}
        if isinstance(mem, dict) and k in mem:
            print(int(mem[k])); raise SystemExit
except SystemExit:
    raise
except Exception as e:
    print(e, file=sys.stderr)
print(2695552)
PY
)
RETAIN=$(( CONTEXT + MESSAGE + GEN ))
CAP=$(( POOL / CONTEXT )); (( CAP < 1 )) && CAP=1
echo "pool=$POOL context=$CONTEXT retain=$RETAIN cap=$CAP teams=$TEAM_COUNTS"

flush_cache() {
  local code
  code=$(curl -sS -o /dev/null -w "%{http_code}" -m 20 -X POST \
    "http://127.0.0.1:${ENGINE_PORT}/flush_cache" || echo 000)
  echo "flush_cache code=${code}"
}

set_lru() {
  flush_cache || true
  cat >"$CTRL" <<EOF
{"mode": "group", "quota": 0, "global_cap": 0, "members": $MEMBERS,
 "capacity": $CAP, "rotate_period": 0, "window_s": 30,
 "pool_tokens": $POOL, "adapt_threshold": 0.6, "adapt_window": 40}
EOF
  curl -s -m 45 "http://127.0.0.1:${ENGINE_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"glm","messages":[{"role":"user","content":"ok"}],"max_tokens":1}' \
    >/dev/null || true
  sleep 2
  echo "arm lru $(date -u +%H:%M:%S)"
  docker logs --tail 20 "$CONTAINER" 2>&1 | grep '\[cohort\] gen=' | tail -2 || true
}

stamp_meta() {
  local path="$1" teams="$2"
  python3 - "$path" "$POOL" "$MESSAGE" "$teams" "$MEMBERS" "$CONTEXT" "$GEN" <<'PY'
import json, sys
path, pool, message, teams, members, context, gen = sys.argv[1:8]
d = json.load(open(path))
d["pool_tokens"] = int(pool)
d["message"] = int(message)
d["model"] = "GLM-5.3"
d["working_set"] = int(teams) * int(members) * (int(context) + int(message) + int(gen))
json.dump(d, open(path, "w"), indent=2)
print(f"  ws={d['working_set']} over={d['working_set']/d['pool_tokens']:.2f}x "
      f"hit={1-d['turn_miss_rate']:.4f} cached={d.get('mean_cached')}")
PY
}

for teams in $TEAM_COUNTS; do
  ws=$(( teams * MEMBERS * RETAIN ))
  over=$(python3 -c "print(round($ws / $POOL, 2))")
  tag="e2e-ring-lru-${teams}t-p56k-glm"
  out="$OUTDIR/${tag}.json"
  if [[ -f "$out" ]]; then
    echo "skip $tag (have $out)"
    continue
  fi
  echo
  echo "### $teams teams  ws=$ws  over=${over}x  $(date -u +%H:%M:%S)"
  set_lru
  python3 "$ROOT/replay_team.py" \
    --label "$tag" \
    --port "$ENGINE_PORT" \
    --teams "$teams" --members "$MEMBERS" \
    --topology ring \
    --context "$CONTEXT" \
    --message "$MESSAGE" --gen-tokens "$GEN" \
    --rounds "$ROUNDS" --warmup-rounds "$WARMUP" \
    --stagger 1.5 \
    --capacity "$CAP" \
    --reset-multiple 0.4 \
    --out "$out"
  stamp_meta "$out" "$teams"
done

echo "glm-ring-lru DONE pool=$POOL $(date -u +%H:%M:%S)"
ls -la "$OUTDIR"/e2e-ring-lru-*-p56k-glm.json
