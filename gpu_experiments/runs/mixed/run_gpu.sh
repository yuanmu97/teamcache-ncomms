#!/usr/bin/env bash
# GPU driver for additional_validation. Reads frozen plan.json.
# Does not retune unified. Budget: elapsed node seconds × 8 GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PAT="${PAT:-/opt/dlami/nvme/pat}"
# Local checkout on the laptop uses research/peer-agent-serving; on B300 the
# same files live in $PAT. Prefer the directory that contains replay_team.py.
if [[ -f "$ROOT/../replay_team.py" ]]; then
  PAT="$(cd "$ROOT/.." && pwd)"
fi
PLAN="$ROOT/plan.json"
CTRL="${CTRL:-/tmp/cohort/control.json}"
CONTAINER="${CONTAINER:-kimi-k3-b300-sglang-cohort}"
POOL="${POOL:-420800}"
MEMBERS=5
ROUNDS=5
WARMUP=2
GEN=144
CONTEXT=9728
MESSAGE=512
STAGGER=2.0
SEED=11
GPU_PER_NODE=8
BUDGET_GPU_H="${BUDGET_GPU_H:-16}"
RUNS_CSV="$ROOT/runs.csv"
DEBUG_HOST="/tmp/cohort-debug/unified_debug.jsonl"

python3 - "$PLAN" <<'PY'
import json, sys
p=json.load(open(sys.argv[1]))
assert p.get("frozen"), "plan.json is not frozen"
print("plan frozen at", p.get("frozen_at"))
print("star q*", p["star"]["q_star"], "mixed teams", p["mixed"]["teams"], "mixed q*", p["mixed"]["q_star"])
PY

STAR_Q=$(python3 -c "import json;print(json.load(open('$PLAN'))['star']['q_star'])")
MIX_TEAMS=$(python3 -c "import json;print(json.load(open('$PLAN'))['mixed']['teams'])")
MIX_Q=$(python3 -c "import json;print(json.load(open('$PLAN'))['mixed']['q_star'])")
MIX_CAP=$(python3 -c "import json;print(json.load(open('$PLAN'))['mixed']['capacity'])")
MIX_GCAP=$(python3 -c "import json;print(json.load(open('$PLAN'))['mixed']['global_cap'])")
STAR_CAP=$(python3 -c "import json;print(json.load(open('$PLAN'))['star']['capacity'])")
MERGE_MIX=$(python3 -c "import json;print('1' if json.load(open('$PLAN'))['mixed']['equivalent_to_lru'] else '0')")
NODE_START=$(date +%s)

gpu_hours() {
  python3 -c "print(f'{($1)*$GPU_PER_NODE/3600:.4f}')"
}

remaining() {
  local now elapsed used
  now=$(date +%s)
  elapsed=$(( now - NODE_START ))
  used=$(gpu_hours "$elapsed")
  python3 -c "print(float('$BUDGET_GPU_H') - float('$used'))"
}

log_run() {  # name status elapsed_s extra
  local now elapsed
  now=$(date +%s)
  elapsed=$(( now - NODE_START ))
  mkdir -p "$ROOT"
  if [[ ! -f "$RUNS_CSV" ]]; then
    echo "ts,name,status,run_s,node_elapsed_s,gpu_hours,note" >"$RUNS_CSV"
  fi
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),$1,$2,$3,$elapsed,$(gpu_hours "$elapsed"),${4:-}" >>"$RUNS_CSV"
}

set_policy() {  # mode quota global_cap members capacity
  python3 - "$CTRL" "$1" "$2" "$3" "$4" "$5" "$POOL" <<'PY'
import json, sys, os, time
path, mode, quota, gcap, members, cap, pool = sys.argv[1:8]
os.makedirs(os.path.dirname(path), exist_ok=True)
doc = {
  "mode": mode, "quota": int(quota), "global_cap": int(gcap),
  "members": int(members), "capacity": int(cap),
  "rotate_period": 0, "window_s": 30, "pool_tokens": int(pool),
  "adapt_threshold": 0.6, "adapt_window": 40,
  "adapt_margin": 1.02, "adapt_min_samples": 2,
}
# Rewrite in place so the bind-mounted inode updates.
with open(path, "w") as fh:
    json.dump(doc, fh)
    fh.flush(); os.fsync(fh.fileno())
print("wrote", path, doc)
PY
  sleep 2
  local line
  line=$(docker logs --tail 400 "$CONTAINER" 2>&1 | grep '\[cohort\] gen=' | tail -1 || true)
  echo "policy -> mode=$1 quota=$2 cap=$3"
  echo "          engine ${line:-<no cohort log line; switch NOT confirmed>}"
  if [[ -z "$line" ]]; then
    echo "WARNING: engine did not log a cohort generation bump" >&2
  fi
}

confirm_switch() {
  local before after
  before=$(docker logs --tail 800 "$CONTAINER" 2>&1 | grep -c '\[cohort\] gen=' || true)
  set_policy "$@"
  sleep 1
  after=$(docker logs --tail 800 "$CONTAINER" 2>&1 | grep -c '\[cohort\] gen=' || true)
  echo "cohort log lines $before -> $after"
}

wait_health() {
  local i
  for i in $(seq 1 180); do
    if curl -sf -m 2 "http://127.0.0.1:30000/health" >/dev/null 2>&1 \
       || curl -sf -m 2 "http://127.0.0.1:30000/get_model_info" >/dev/null 2>&1; then
      echo "engine healthy after ${i} checks"
      return 0
    fi
    sleep 5
  done
  echo "engine failed to become healthy" >&2
  return 1
}

arm_star() {  # label rank_map_or_empty [seed]
  local label=$1
  local seed="${3:-$SEED}"
  local out="$ROOT/gpu/star_${label}_9t_s${seed}.json"
  if [[ "$seed" == "$SEED" ]]; then
    out="$ROOT/gpu/star_${label}_9t.json"
  fi
  if [[ -f "$out" ]]; then
    echo "skip existing $out"
    return 0
  fi
  local extra=()
  if [[ -n "${2:-}" ]]; then
    extra+=(--rank-map "$2")
  fi
  local t0 t1
  t0=$(date +%s)
  if ! timeout 900 python3 "$ROOT/client_dump.py" \
    --label "$label" --teams 9 --members "$MEMBERS" \
    --rounds "$ROUNDS" --warmup-rounds "$WARMUP" \
    --gen-tokens "$GEN" --context "$CONTEXT" --capacity "$STAR_CAP" \
    --topology star --stagger "$STAGGER" --seed "$seed" \
    --out "$out" "${extra[@]}"; then
    log_run "star_${label}_s${seed}" fail "$(( $(date +%s)-t0 ))" "timeout_or_error"
    return 1
  fi
  t1=$(date +%s)
  log_run "star_${label}_s${seed}" ok "$((t1-t0))" ""
  if [[ "$label" == "teamcache" && -f "$DEBUG_HOST" ]]; then
    cp "$DEBUG_HOST" "$ROOT/gpu/star_${label}_s${seed}_unified_debug.jsonl" || true
  fi
}

arm_mixed() {  # label rank_map_or_empty [seed]
  local label=$1
  local seed="${3:-$SEED}"
  local out="$ROOT/gpu/mixed_${label}_${MIX_TEAMS}t_s${seed}.json"
  if [[ "$seed" == "$SEED" ]]; then
    out="$ROOT/gpu/mixed_${label}_${MIX_TEAMS}t.json"
  fi
  if [[ -f "$out" ]]; then
    echo "skip existing $out"
    return 0
  fi
  local extra=()
  if [[ -n "${2:-}" ]]; then
    extra+=(--rank-map "$2")
  fi
  : > "$DEBUG_HOST" || true
  local t0 t1
  t0=$(date +%s)
  if ! timeout 900 python3 "$ROOT/mixed_replay.py" \
    --label "$label" --teams "$MIX_TEAMS" --members "$MEMBERS" \
    --rounds "$ROUNDS" --warmup-rounds "$WARMUP" \
    --gen-tokens "$GEN" --context "$CONTEXT" --capacity "$MIX_CAP" \
    --stagger "$STAGGER" --seed "$seed" --out "$out" "${extra[@]}"; then
    log_run "mixed_${label}_s${seed}" fail "$(( $(date +%s)-t0 ))" "timeout_or_error"
    return 1
  fi
  t1=$(date +%s)
  log_run "mixed_${label}_s${seed}" ok "$((t1-t0))" ""
  if [[ "$label" == "teamcache" && -f "$DEBUG_HOST" ]]; then
    cp "$DEBUG_HOST" "$ROOT/gpu/mixed_${label}_s${seed}_unified_debug.jsonl" || true
  fi
}

mkdir -p "$ROOT/gpu"

# --- engine ---
if [[ "${SKIP_START:-0}" == "1" ]]; then
  echo "SKIP_START=1; reusing $CONTAINER"
else
  if docker ps --format '{{.Names}}' | grep -qx glm53-b300-sglang; then
    echo "idle GLM engine holds the GPUs; stopping glm53-b300-sglang only (not glm53-bench-client)"
    docker stop glm53-b300-sglang
    log_run "stop_glm53" ok 0 "idle>24h; additional_validation needs K3 cohort"
  fi
  python3 "$PAT/patch_sglang_cohort.py" --out /tmp/sgl-patch
  mkdir -p /tmp/cohort /tmp/cohort-debug
  touch "$DEBUG_HOST"
  cat >"$CTRL" <<EOF
{"mode": "group", "quota": 0, "global_cap": 0, "members": $MEMBERS,
 "capacity": $STAR_CAP, "rotate_period": 0, "window_s": 30,
 "pool_tokens": $POOL, "adapt_threshold": 0.6, "adapt_window": 40,
 "adapt_margin": 1.02, "adapt_min_samples": 2}
EOF
  UNIFIED_DEBUG_FILE="$DEBUG_HOST"
  SERVE=""
  for cand in \
      "$PAT/deploy/serve_kimi_k3_b300_sglang_cohort.sh" \
      "$(dirname "$PAT")/deploy/serve_kimi_k3_b300_sglang_cohort.sh" \
      "/opt/dlami/nvme/h200-llm/deploy/serve_kimi_k3_b300_sglang_cohort.sh"; do
    if [[ -f "$cand" ]]; then SERVE="$cand"; break; fi
  done
  if [[ -z "$SERVE" ]]; then
    echo "cannot find serve_kimi_k3_b300_sglang_cohort.sh" >&2
    exit 1
  fi
  echo "starting engine via $SERVE"
  UNIFIED_DEBUG_FILE="$DEBUG_HOST" bash "$SERVE"
fi
wait_health

# Neutralize, then star Static-rate then TeamCache (baseline first).
confirm_switch group 0 0 "$MEMBERS" "$STAR_CAP"
confirm_switch group "$STAR_Q" 0 "$MEMBERS" "$STAR_CAP"
arm_star static_rate "$ROOT/cpu/static_rate_star.json" || arm_star static_rate "$ROOT/cpu/static_rate_star.json"

confirm_switch group 0 0 "$MEMBERS" "$STAR_CAP"
confirm_switch unified 0 0 "$MEMBERS" "$STAR_CAP"
: > "$DEBUG_HOST" || true
arm_star teamcache || arm_star teamcache

# Mixed four arms: LRU, Group-blind, Static-rate, TeamCache
confirm_switch group 0 0 "$MEMBERS" "$MIX_CAP"
arm_mixed lru || arm_mixed lru

confirm_switch group 0 0 "$MEMBERS" "$MIX_CAP"
confirm_switch global 0 "$MIX_GCAP" "$MEMBERS" "$MIX_CAP"
arm_mixed group_blind || arm_mixed group_blind

if [[ "$MERGE_MIX" == "1" ]]; then
  echo "mixed Static-rate q*>=k; merged with LRU, skipping duplicate GPU arm"
  log_run "mixed_static_rate" skipped 0 "q*>=k equivalent to LRU"
else
  confirm_switch group 0 0 "$MEMBERS" "$MIX_CAP"
  confirm_switch group "$MIX_Q" 0 "$MEMBERS" "$MIX_CAP"
  arm_mixed static_rate "$ROOT/cpu/static_rate_mixed.json" || arm_mixed static_rate "$ROOT/cpu/static_rate_mixed.json"
fi

confirm_switch group 0 0 "$MEMBERS" "$MIX_CAP"
confirm_switch unified 0 0 "$MEMBERS" "$MIX_CAP"
: > "$DEBUG_HOST" || true
arm_mixed teamcache || arm_mixed teamcache

ELAPSED=$(( $(date +%s) - NODE_START ))
echo "node_elapsed=${ELAPSED}s gpu_hours=$(gpu_hours "$ELAPSED") remaining=$(remaining)"
log_run "pilot_done" ok "$ELAPSED" "star 2 + mixed 4 (or merged)"

python3 - "$ROOT" "$BUDGET_GPU_H" "$GPU_PER_NODE" "$NODE_START" <<'PY'
import csv, json, os, sys, time
root, budget, gpn, t0 = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
path = os.path.join(root, "runs.csv")
rows = list(csv.DictReader(open(path))) if os.path.exists(path) else []
ok = [r for r in rows if r["status"]=="ok" and r["name"].startswith("mixed_") and r["name"].endswith("_s11") or (
      r["status"]=="ok" and r["name"].startswith("mixed_") and r["name"] in
      ("mixed_lru","mixed_group_blind","mixed_static_rate","mixed_teamcache"))]
# Pilot mixed names: mixed_lru_s11 after change; primary seed files use mixed_* without _s
times = []
star_ok = team_ok = False
for r in rows:
    if r["status"] != "ok":
        continue
    if r["name"].startswith("mixed_") and "fail" not in r["name"]:
        try:
            times.append(float(r["run_s"]))
        except ValueError:
            pass
    if r["name"].startswith("star_static"):
        star_ok = True
    if r["name"].startswith("star_teamcache"):
        team_ok = True
mixed_files = [
    os.path.join(root, "gpu", "mixed_lru_12t.json"),
    os.path.join(root, "gpu", "mixed_group_blind_12t.json"),
    os.path.join(root, "gpu", "mixed_static_rate_12t.json"),
    os.path.join(root, "gpu", "mixed_teamcache_12t.json"),
]
star_files = [
    os.path.join(root, "gpu", "star_static_rate_9t.json"),
    os.path.join(root, "gpu", "star_teamcache_9t.json"),
]
mixed_valid = all(os.path.exists(p) for p in mixed_files)
star_valid = all(os.path.exists(p) for p in star_files)
used = (time.time() - t0) * gpn / 3600
remain = budget - used
med = sorted(times)[len(times)//2] if times else 180
need_mixed = 1.5 * med * 4 + 120  # 4 arms + flush
need_mixed_h = need_mixed * gpn / 3600
need_star_h = (1.5 * med * 2 + 120) * gpn / 3600
decision = {
    "mixed_valid": mixed_valid,
    "star_valid": star_valid,
    "remain_gpu_h": remain,
    "median_run_s": med,
    "need_mixed_group_gpu_h": need_mixed_h,
    "extra_mixed": mixed_valid and star_valid and remain >= need_mixed_h * 3,
    "extra_star": False,
}
# extra mixed 3 seeds first; only if a full group fits
decision["extra_mixed"] = bool(mixed_valid and star_valid and remain >= need_mixed_h)
decision["n_extra_mixed_seeds"] = 0
if decision["extra_mixed"]:
    n = min(3, int(remain // need_mixed_h))
    decision["n_extra_mixed_seeds"] = n
    remain2 = remain - n * need_mixed_h
    decision["extra_star"] = remain2 >= need_star_h
    decision["n_extra_star_seeds"] = min(3, int(remain2 // need_star_h)) if decision["extra_star"] else 0
else:
    decision["n_extra_star_seeds"] = 0
json.dump(decision, open(os.path.join(root, "gpu", "seed_decision.json"), "w"), indent=2)
print(json.dumps(decision, indent=2))
PY

DEC="$ROOT/gpu/seed_decision.json"
N_MIX=$(python3 -c "import json;print(json.load(open('$DEC')).get('n_extra_mixed_seeds',0))")
N_STAR=$(python3 -c "import json;print(json.load(open('$DEC')).get('n_extra_star_seeds',0))")
echo "extra mixed seeds=$N_MIX extra star seeds=$N_STAR"

SEEDS=(12 13 14)
# Rotate policy start order per extra seed: LRU/GB/SR/TC, then GB/SR/TC/LRU, then SR/TC/LRU/GB
for ((i=0; i<N_MIX; i++)); do
  s=${SEEDS[$i]}
  echo "=== extra mixed seed=$s ==="
  case $i in
    0) order="lru group_blind static_rate teamcache" ;;
    1) order="group_blind static_rate teamcache lru" ;;
    *) order="static_rate teamcache lru group_blind" ;;
  esac
  for arm in $order; do
    confirm_switch group 0 0 "$MEMBERS" "$MIX_CAP"
    case $arm in
      lru) confirm_switch group 0 0 "$MEMBERS" "$MIX_CAP"; arm_mixed lru "" "$s" ;;
      group_blind) confirm_switch global 0 "$MIX_GCAP" "$MEMBERS" "$MIX_CAP"; arm_mixed group_blind "" "$s" ;;
      static_rate) confirm_switch group "$MIX_Q" 0 "$MEMBERS" "$MIX_CAP"; arm_mixed static_rate "$ROOT/cpu/static_rate_mixed.json" "$s" ;;
      teamcache) confirm_switch unified 0 0 "$MEMBERS" "$MIX_CAP"; : > "$DEBUG_HOST" || true; arm_mixed teamcache "" "$s" ;;
    esac
  done
done

for ((i=0; i<N_STAR; i++)); do
  s=${SEEDS[$i]}
  echo "=== extra star seed=$s ==="
  confirm_switch group 0 0 "$MEMBERS" "$STAR_CAP"
  confirm_switch group "$STAR_Q" 0 "$MEMBERS" "$STAR_CAP"
  arm_star static_rate "$ROOT/cpu/static_rate_star.json" "$s"
  confirm_switch group 0 0 "$MEMBERS" "$STAR_CAP"
  confirm_switch unified 0 0 "$MEMBERS" "$STAR_CAP"
  : > "$DEBUG_HOST" || true
  arm_star teamcache "" "$s"
done

python3 "$ROOT/plot_gpu.py" || true
python3 "$ROOT/collect_tables.py" || true
python3 "$ROOT/write_report.py" || true
ELAPSED=$(( $(date +%s) - NODE_START ))
echo "DONE node_elapsed=${ELAPSED}s gpu_hours=$(gpu_hours "$ELAPSED")"
log_run "all_done" ok "$ELAPSED" ""

