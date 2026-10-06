#!/usr/bin/env bash
# GLM-5.3: stack more realistic teams against the NATURAL pool.
#
# 10 teams @ 1.55× (shrunk pool) was only 54% miss / 0 starve — not a cliff.
# Same ratio on the 2.70M natural pool is ~22 teams and will repeat that.
# 32 teams: 20×56k+6×12k+6×16k, workset 6.44M ≈ 2.39×.
# Unified share at 32 active ≈ 44k → 1-of-5 on 56k rings (still un-starves
# a 100% miss ring). 2-of-5 still fits on 12k/16k. Do not add teams past
# where 1×56k×N overflows the natural pool (~48).
set -uo pipefail
cd "$(dirname "$0")"
ROOT=/opt/dlami/nvme/pat
DEPLOY="${DEPLOY:-$ROOT/deploy}"
ENGINE_PORT="${ENGINE_PORT:-30000}"
TEAMS="${TEAMS:-32}"
MEMBERS="${MEMBERS:-5}"
ROUNDS="${ROUNDS:-6}"
# 20×56k + 6×12k + 6×16k
MIX="${TEAM_CONTEXTS:-56000,56000,56000,56000,56000,56000,56000,56000,56000,56000,56000,56000,56000,56000,56000,56000,56000,56000,56000,56000,12000,12000,12000,12000,12000,12000,16000,16000,16000,16000,16000,16000}"
OUTDIR="${OUTDIR:-$ROOT/results/e2e}"
CTRL="${CTRL:-/tmp/cohort/control.json}"
mkdir -p "$OUTDIR" results/glm

n_mix=$(python3 -c "print(len('$MIX'.split(',')))")
if [[ "$n_mix" != "$TEAMS" ]]; then
  echo "mix has $n_mix entries, TEAMS=$TEAMS" >&2
  exit 2
fi
workset=$(python3 -c "xs=[int(x) for x in '$MIX'.split(',')]; print(sum(xs)*$MEMBERS)")
echo "glm-scale teams=$TEAMS workset=$workset $(date -u +%H:%M:%S)"

if [[ "${SKIP_START:-0}" != "1" ]]; then
  python3 patch_sglang_cohort.py --out /tmp/sgl-patch
  docker rm -f kimi-k3-b300-vllm kimi-k3-b300-sglang-cohort glm53-b300-sglang >/dev/null 2>&1 || true
  sleep 3
  (cd "$DEPLOY" && ./serve_glm53_b300_sglang_cohort.sh)
fi
bash ./wait_engine.sh "$ENGINE_PORT" 180 || exit 1

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
over=$(python3 -c "print(round($workset/int('$POOL'),2))")
mean=$(python3 -c "xs=[int(x) for x in '$MIX'.split(',')]; print(sum(xs)//len(xs))")
cap=$(( POOL / mean )); (( cap < 1 )) && cap=1
echo "pool=$POOL workset=$workset oversub=${over}x cap=$cap load=$(python3 -c "print(round($TEAMS*$MEMBERS/$cap,2))")x"

flush_cache() {
  local code
  code=$(curl -sS -o /dev/null -w "%{http_code}" -m 20 -X POST \
    "http://127.0.0.1:${ENGINE_PORT}/flush_cache" || echo 000)
  echo "flush_cache code=${code}"
  if [[ "$code" != "200" && "$code" != "204" ]]; then
    echo "flush_cache failed" >&2
    return 1
  fi
}

set_arm() {
  local mode="$1" quota="$2"
  flush_cache || true
  cat >"$CTRL" <<EOF
{"mode": "$mode", "quota": $quota, "global_cap": 0, "members": $MEMBERS,
 "capacity": $cap, "rotate_period": 0, "window_s": 30,
 "pool_tokens": $POOL, "adapt_threshold": 0.6, "adapt_window": 40}
EOF
  curl -s -m 45 "http://127.0.0.1:${ENGINE_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"glm","messages":[{"role":"user","content":"ok"}],"max_tokens":1}' \
    >/dev/null || true
  sleep 1
  echo "arm $mode $(date -u +%H:%M:%S)"
  docker logs --tail 20 glm53-b300-sglang 2>&1 | grep '\[cohort\] gen=' | tail -2 || true
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
  python3 - "$OUTDIR/${label}.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
print(f"wrote {sys.argv[1]} cached={d.get('mean_cached')} "
      f"miss={d.get('turn_miss_rate')} starved={d.get('starved_teams')} "
      f"worst={d.get('team_miss_max')} wall={d.get('wall_s')}")
PY
}

TAG="e2e-fleet-mixed-lru-${TEAMS}t-glm-scale"
if [[ "${UNIFIED_ONLY:-0}" != "1" ]]; then
run_arm "$TAG" group 0
python3 - "$OUTDIR/${TAG}.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
c=float(d.get("mean_cached") or 0)
m=float(d.get("turn_miss_rate") or 0)
s=int(d.get("starved_teams") or 0)
w=float(d.get("team_miss_max") or 0)
print(f"LRU gate cached={c:.3f} miss={m:.3f} starved={s} worst={w:.3f}")
# Significant-effect gate: LRU must actually be in trouble.
ok = (c < 0.35) or (m >= 0.65) or (s >= 1) or (w >= 0.90)
if not ok:
    print("LRU_STILL_WARM")
    sys.exit(3)
PY
if [[ $? -eq 3 ]]; then
  echo "LRU still warm at ${TEAMS} teams / ${over}x; add 56k teams, keep this pool." >&2
  exit 3
fi
fi
run_arm "e2e-fleet-mixed-unified-${TEAMS}t-glm-scale" unified 0
echo "glm-scale DONE pool=$POOL teams=$TEAMS $(date -u +%H:%M:%S)"
