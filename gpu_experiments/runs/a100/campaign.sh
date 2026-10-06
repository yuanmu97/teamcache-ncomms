#!/usr/bin/env bash
# Repetition campaign on 4xA100-80GB with GLM-4.7-Flash, one engine boot.
# Every comparison runs three seeds, the policy order rotates across seeds and
# the prefix cache is flushed before every arm.  Existing outputs are skipped,
# so the script can be restarted after an interruption.
#
#   PHASES="e8 ring mixed selector star sweep" ./campaign.sh
set -uo pipefail
cd "$(dirname "$0")"
source ./common_a100.sh

PHASES="${PHASES:-e8 ring mixed selector star sweep}"
PY="$VENV/bin/python"
MEMBERS=5

# Live AutoGen driver: e2e_autogen.sh with the engine name, served model and
# log source of this host.
sed -e 's/"model":"kimi"/"model":"'"$SERVED"'"/' \
    -e 's#docker logs --tail 400 kimi-k3-b300-sglang-cohort 2>&1#tail -n 400 '"$ENGINE_LOG"'#' \
    e2e_autogen.sh >e2e_autogen_a100.sh
chmod +x e2e_autogen_a100.sh

rotate() {  # seed_index arms... -> arms rotated by seed_index
  local i=$1; shift
  local arr=("$@") n=$# out=() k
  for ((k = 0; k < n; k++)); do out+=("${arr[$(((k + i) % n))]}"); done
  echo "${out[@]}"
}

run_timed() {  # name cmd...
  local name=$1; shift
  local t0=$(date +%s)
  if timeout "${RUN_TIMEOUT:-2400}" "$@"; then
    log_run "$name" ok $(( $(date +%s) - t0 ))
  else
    log_run "$name" fail $(( $(date +%s) - t0 )) "exit $?"
    return 1
  fi
}

# ---------------------------------------------------------------- E8
phase_e8() {
  mkdir -p "$OUT/nonregress"
  local rep label
  for rep in 1 2 3; do
    for label in quota0 unified; do
      local out="$OUT/nonregress/${label}_r${rep}.json"
      [[ -f "$out" ]] && continue
      flush_cache
      if [[ $label == quota0 ]]; then set_policy group 0 0 5 0; else set_policy unified 0 0 5 0; fi
      run_timed "e8_${label}_r${rep}" "$PY" probe_nonregress.py --label "$label" \
        --seed $((16 + rep)) --out "$out"
    done
  done
}

# ---------------------------------------------------------------- live AutoGen
live_arm() {  # harness task arm seed
  local harness=$1 task=$2 arm=$3 seed=$4
  local tag="e2e-${harness}-${task}-${arm}-20t-p3000r8-a100"
  [[ $seed != 0 ]] && tag="${tag}-s${seed}"
  [[ -f "$OUT/e2e/${tag}.json" ]] && { echo "skip $tag"; return 0; }
  flush_cache
  : >"$DEBUG_HOST"
  local ctk=""
  [[ $harness == selector ]] && ctk='{"enable_thinking": false}'
  PROXY_CHAT_TEMPLATE_KWARGS=$ctk \
  HARNESS=$harness TASK=$task TEAMS=20 MEMBERS=$MEMBERS ROUNDS=8 PREAMBLE=3000 \
  GEN_CAP=144 QUOTA=2 POOL=$POOL SEED=$seed OUTDIR="$OUT/e2e" MODEL="$SERVED" \
  TAG_SUFFIX=-a100 ENGINE_PORT=$ENGINE_PORT \
    run_timed "$tag" ./e2e_autogen_a100.sh "$arm" 20 "$MEMBERS" 8
  [[ $arm == unified ]] && cp "$DEBUG_HOST" "$OUT/e2e/${tag}.unified_debug.jsonl" 2>/dev/null
  sleep 10
}

phase_live() {  # harness task
  local s arm
  for s in 0 1 2; do
    for arm in $(rotate "$s" lru cohort unified); do
      live_arm "$1" "$2" "$arm" "$s"
    done
  done
}

# ---------------------------------------------------------------- replay clients
replay_arm() {  # client label topology teams capacity seed out [rankmap]
  local client=$1 label=$2 topo=$3 teams=$4 cap=$5 seed=$6 out=$7 rank=${8:-}
  [[ -f "$out" ]] && { echo "skip $out"; return 0; }
  local extra=()
  [[ -n $rank ]] && extra+=(--rank-map "$rank")
  [[ $client == client_dump.py ]] && extra+=(--topology "$topo")
  flush_cache
  : >"$DEBUG_HOST"
  run_timed "$(basename "$out" .json)" "$PY" "mixed/$client" --label "$label" \
    --teams "$teams" --members "$MEMBERS" --rounds 5 --warmup-rounds 2 \
    --gen-tokens 144 --context 9728 --capacity "$cap" --stagger 2.0 \
    --seed "$seed" --out "$out" "${extra[@]}"
  [[ $label == teamcache || $label == unified ]] && \
    cp "$DEBUG_HOST" "${out%.json}.unified_debug.jsonl" 2>/dev/null
  return 0
}

policy_for() {  # arm quota gcap cap
  case $1 in
    lru) set_policy group 0 0 "$MEMBERS" "$4" ;;
    group_blind|bypass) set_policy global 0 "$3" "$MEMBERS" "$4" ;;
    static_rate|cohort) set_policy group "$2" 0 "$MEMBERS" "$4" ;;
    teamcache|unified) set_policy unified 0 0 "$MEMBERS" "$4" ;;
  esac
}

phase_mixed() {
  mkdir -p "$OUT/mixed"
  local i s arm rank
  for i in 0 1 2; do
    s=$((11 + i))
    for arm in $(rotate "$i" lru group_blind static_rate teamcache); do
      rank=""; [[ $arm == static_rate ]] && rank=mixed/cpu/static_rate_mixed.json
      [[ -f "$OUT/mixed/mixed_${arm}_12t_s${s}.json" ]] && continue
      policy_for "$arm" 4 24 28
      replay_arm mixed_replay.py "$arm" mixed 12 28 "$s" \
        "$OUT/mixed/mixed_${arm}_12t_s${s}.json" "$rank"
    done
  done
}

phase_star() {
  mkdir -p "$OUT/star"
  local i s arm rank
  for i in 0 1 2; do
    s=$((11 + i))
    for arm in $(rotate "$i" lru group_blind static_rate teamcache); do
      rank=""; [[ $arm == static_rate ]] && rank=mixed/cpu/static_rate_star.json
      [[ -f "$OUT/star/star_${arm}_9t_s${s}.json" ]] && continue
      policy_for "$arm" 4 27 31
      replay_arm client_dump.py "$arm" star 9 31 "$s" \
        "$OUT/star/star_${arm}_9t_s${s}.json" "$rank"
    done
  done
}

phase_sweep() {
  mkdir -p "$OUT/team-sweep"
  local retain=$((9728 + 512 + 144)) i s teams cap quota budget arm
  for i in 0 1 2; do
    s=$((11 + i))
    for teams in ${TEAM_COUNTS:-6 8 10 12 14 16}; do
      cap=$(( (POOL - teams * retain) / retain )); (( cap < 1 )) && cap=1
      quota=$(( cap / teams )); (( quota < 1 )) && quota=1
      budget=$(( teams * quota ))
      for arm in $(rotate "$i" lru bypass cohort unified); do
        [[ -f "$OUT/team-sweep/${arm}_${teams}t_ring_s${s}.json" ]] && continue
        policy_for "$arm" "$quota" "$budget" "$cap"
        replay_arm client_dump.py "$arm" ring "$teams" "$cap" "$s" \
          "$OUT/team-sweep/${arm}_${teams}t_ring_s${s}.json"
      done
    done
  done
}

echo "campaign start $(date -u) phases: $PHASES"
for ph in $PHASES; do
  echo "######## phase $ph $(date -u +%H:%M:%S)"
  case $ph in
    e8) phase_e8 ;;
    ring) phase_live autogen design ;;
    selector) phase_live selector debug ;;
    mixed) phase_mixed ;;
    star) phase_star ;;
    sweep) phase_sweep ;;
    *) echo "unknown phase $ph" ;;
  esac
done
echo "campaign done $(date -u)"
