#!/usr/bin/env bash
# HiCache repetitions on 4xA100-80GB: restarts the engine with a host-memory
# hierarchy (ratio 2, joint pool 1,262,400 tokens as on B300) and repeats the
# twenty-team ring and the oversubscribed long-context fleet, 3 seeds each.
#
#   PHASES="hicache_ring hicache_fleet" ./campaign_ext.sh
set -uo pipefail
cd "$(dirname "$0")"
source ./common_a100.sh

PHASES="${PHASES:-hicache_ring hicache_fleet}"
PY="$VENV/bin/python"
MEMBERS=5
JOINT=$((3 * POOL))

rotate() {
  local i=$1; shift
  local arr=("$@") n=$# out=() k
  for ((k = 0; k < n; k++)); do out+=("${arr[$(((k + i) % n))]}"); done
  echo "${out[@]}"
}

run_timed() {
  local name=$1; shift
  local t0=$(date +%s)
  if timeout "${RUN_TIMEOUT:-2400}" "$@"; then
    log_run "$name" ok $(( $(date +%s) - t0 ))
  else
    log_run "$name" fail $(( $(date +%s) - t0 )) "exit $?"
    return 1
  fi
}

boot_hicache() {
  if grep -q "hicache=1" "$LOGS/engine_boot.log" 2>/dev/null && \
     curl -sf -m 3 "http://127.0.0.1:${ENGINE_PORT}/health" >/dev/null; then
    return 0
  fi
  ENABLE_HICACHE=1 HICACHE_RATIO=2 ./serve_a100.sh 2>&1 | tee "$LOGS/engine_boot.log"
}

phase_hicache_ring() {
  mkdir -p "$OUT/hicache"
  local s arm tag
  for s in 0 1 2; do
    for arm in $(rotate "$s" lru unified); do
      tag="e2e-autogen-design-${arm}-20t-p3000r8-a100-hicache"
      [[ $s != 0 ]] && tag="${tag}-s${s}"
      [[ -f "$OUT/hicache/${tag}.json" ]] && continue
      flush_cache
      : >"$DEBUG_HOST"
      HARNESS=autogen TASK=design TEAMS=20 MEMBERS=$MEMBERS ROUNDS=8 PREAMBLE=3000 \
      GEN_CAP=144 QUOTA=2 POOL=$JOINT SEED=$s OUTDIR="$OUT/hicache" MODEL="$SERVED" \
      TAG_SUFFIX=-a100-hicache ENGINE_PORT=$ENGINE_PORT \
        run_timed "$tag" ./e2e_autogen_a100.sh "$arm" 20 "$MEMBERS" 8
      sleep 10
    done
  done
}

phase_hicache_fleet() {
  mkdir -p "$OUT/hicache"
  # Contexts scale with the pool so the fleet oversubscribes the joint pool
  # 1.55-fold as on B300; POOL=210400 halves both for the A100's prefill rate.
  local mix="${FLEET_MIX:-56000,56000,56000,56000,56000,56000,12000,12000,16000,16000}"
  local mean
  mean=$(python3 -c "xs=[int(x) for x in '$mix'.split(',')]; print(sum(xs)//len(xs))")
  local cap=$((JOINT / mean)) i s arm out
  for i in 0 1 2; do
    s=$((11 + i))
    for arm in $(rotate "$i" lru unified); do
      out="$OUT/hicache/fleet_${arm}_10t_s${s}.json"
      [[ -f "$out" ]] && continue
      flush_cache
      if [[ $arm == lru ]]; then
        POOL=$JOINT set_policy group 0 0 "$MEMBERS" "$cap"
      else
        POOL=$JOINT set_policy unified 0 0 "$MEMBERS" "$cap"
      fi
      : >"$DEBUG_HOST"
      run_timed "$(basename "$out" .json)" "$PY" replay_team.py --label "fleet_${arm}" \
        --port "$ENGINE_PORT" --teams 10 --members "$MEMBERS" --topology mixed \
        --team-contexts "$mix" --message 512 --gen-tokens 144 --rounds 6 \
        --warmup-rounds 2 --stagger 1.5 --capacity "$cap" --reset-multiple 0.4 \
        --seed "$s" --out "$out"
      [[ $arm == unified ]] && cp "$DEBUG_HOST" "${out%.json}.unified_debug.jsonl" 2>/dev/null
      sleep 10
    done
  done
}

echo "ext campaign start $(date -u) phases: $PHASES"
boot_hicache || { echo "hicache boot failed"; exit 1; }
for ph in $PHASES; do
  echo "######## phase $ph $(date -u +%H:%M:%S)"
  case $ph in
    hicache_ring) phase_hicache_ring ;;
    hicache_fleet) phase_hicache_fleet ;;
    *) echo "unknown phase $ph" ;;
  esac
done
echo "ext campaign done $(date -u)"
