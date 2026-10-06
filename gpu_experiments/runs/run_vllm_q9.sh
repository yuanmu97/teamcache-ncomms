#!/usr/bin/env bash
# Q9: rule engages as designed on vLLM (block_mask, no early return).
#
# Phase A — 8×5 @ 280: pool still fits (~240 peak). unified must PASS
# (hits keep growing, hit_frac ≈ stock LRU). Freeze = abort.
# Phase B — 8×5 @ 220: first-reuse must be nonzero, then oversub. unified
# should beat LRU on hit_frac or keep growing while LRU thrashes.
# Never 20×5; never 160 (first-reuse was 0).
set -uo pipefail
cd "$(dirname "$0")"
ROOT=/opt/dlami/nvme/pat
DEPLOY="${DEPLOY:-$ROOT/deploy}"
export ENGINE_PORT="${ENGINE_PORT:-8888}"
export MODEL="${MODEL:-moonshotai/Kimi-K3}"
export TEAMS="${TEAMS:-8}" MEMBERS="${MEMBERS:-5}" ROUNDS="${ROUNDS:-8}"
export PREAMBLE=3000 GEN_CAP=144 QUOTA=2
export PORT=8910
DCP_SIZE=8
METRIC_DIR=results/vllm
mkdir -p "$METRIC_DIR" traces results/e2e
BLOCK_SIZE=1536

metrics() {
  python3 - <<'PY'
import urllib.request
t = urllib.request.urlopen("http://127.0.0.1:8888/metrics", timeout=8).read().decode()
h = q = None
for line in t.splitlines():
    if line.startswith("vllm:prefix_cache_hits_total{"):
        h = float(line.rsplit(None, 1)[-1])
    elif line.startswith("vllm:prefix_cache_queries_total{"):
        q = float(line.rsplit(None, 1)[-1])
print(f"{h:.0f} {q:.0f}")
PY
}

start_vllm() {
  local blocks="$1"
  echo "start vLLM patched blocks=$blocks $(date -u +%H:%M:%S)"
  docker rm -f kimi-k3-b300-sglang-cohort glm53-b300-sglang kimi-k3-b300-vllm >/dev/null 2>&1 || true
  sleep 3
  python3 patch_vllm_cohort.py --image vllm-kimi-k3:5894fdf \
    --out /tmp/vllm-block_pool-cohort.py --check
  grep -q "block_mask = _extra" /tmp/vllm-block_pool-cohort.py
  (cd "$DEPLOY" && NUM_GPU_BLOCKS_OVERRIDE="$blocks" DCP_SIZE="$DCP_SIZE" \
    VLLM_BLOCK_POOL_PATCH=/tmp/vllm-block_pool-cohort.py \
    ./serve_kimi_k3_b300_vllm.sh)
  bash ./wait_engine.sh "$ENGINE_PORT" 180 || return 1
}

kill_strays() {
  for pid in $(pgrep -f "harness_roundrobin.py" || true); do kill "$pid" 2>/dev/null || true; done
  for pid in $(pgrep -f "recording-proxy-${PORT}" || true); do kill "$pid" 2>/dev/null || true; done
  for pid in $(pgrep -f "e2e_autogen.sh" || true); do kill "$pid" 2>/dev/null || true; done
  rm -f /tmp/e2e-autogen.lock
  sleep 1
}

watch_arm() {
  local arm="$1" tag="$2" h0="$3" q0="$4"
  local trace="traces/${tag}.jsonl"
  local first=$(( TEAMS * MEMBERS ))
  local reuse=$(( first * 2 ))
  local reuse_ok=0
  local last_h=-1 stall=0
  echo "watch $arm expect first=$first reuse_at=$reuse $(date -u +%H:%M:%S)"
  for _ in $(seq 1 360); do
    local n=0
    [[ -f "$trace" ]] && n=$(wc -l < "$trace")
    n=${n:-0}
    read -r h q <<<"$(metrics)"
    local dh=$(( h - h0 )) dq=$(( q - q0 ))
    local frac=0
    (( dq > 0 )) && frac=$(python3 -c "print(f'{$dh/$dq:.3f}')")
    echo "mid arm=$arm n=$n d_hits=$dh d_q=$dq hit_frac=$frac $(date -u +%H:%M:%S)"
    if (( n >= reuse && reuse_ok == 0 )); then
      if (( dq < 50000 || dh < 1000 )); then
        echo "ABORT $arm first-reuse d_hits=$dh d_q=$dq after $n requests" >&2
        kill_strays
        return 4
      fi
      echo "gate_ok $arm first-reuse d_hits=$dh hit_frac=$frac"
      reuse_ok=1
      last_h=$dh
    elif (( reuse_ok == 1 )) && [[ "$arm" == "unified" ]]; then
      # LRU itself plateaus between reuse waves. The broken hook froze
      # unified hits for the rest of the run (~230 requests). Flag that.
      if (( dh == last_h && n > reuse + 10 )); then
        stall=$(( stall + 1 ))
      else
        stall=0
      fi
      last_h=$dh
      if (( stall >= 10 && n >= reuse + 80 )); then
        echo "ABORT $arm hits frozen at $dh after $n requests" >&2
        kill_strays
        return 5
      fi
    fi
    kill -0 "$WATCH_PID" 2>/dev/null || break
    sleep 8
  done
  return 0
}

score_metrics() {
  local tag="$1" arm="$2" h0="$3" q0="$4" wall="$5" blocks="$6" pool="$7"
  read -r h1 q1 <<<"$(metrics)"
  python3 - "$METRIC_DIR/${tag}.metrics.json" <<PY
import json, pathlib, sys
h0, q0, h1, q1 = $h0, $q0, $h1, $q1
dh, dq = h1 - h0, q1 - q0
out = {
    "tag": "$tag", "arm": "$arm",
    "teams": $TEAMS, "members": $MEMBERS, "rounds": $ROUNDS,
    "preamble": $PREAMBLE, "pool_tokens": $pool,
    "num_gpu_blocks": $blocks,
    "hook": "block_mask",
    "wall_s": $wall,
    "hits0": h0, "hits1": h1, "delta_hits": dh,
    "queries0": q0, "queries1": q1, "delta_queries": dq,
    "hit_frac": (dh / dq) if dq else 0.0,
}
pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=2) + "\n")
print(f"metrics {out['arm']} blocks=$blocks hit_frac={out['hit_frac']*100:.2f}% "
      f"hits={dh:.0f}/{dq:.0f} wall={out['wall_s']}s")
PY
}

run_arm() {
  local arm="$1" blocks="$2" suffix="$3"
  local pool=$(( blocks * BLOCK_SIZE ))
  export POOL="$pool"
  export NUM_GPU_BLOCKS_OVERRIDE="$blocks"
  export TAG_SUFFIX="${suffix}"
  local tag="e2e-autogen-design-${arm}-${TEAMS}t-p${PREAMBLE}r${ROUNDS}-${suffix}"
  rm -f "traces/${tag}.jsonl"
  kill_strays
  for ep in /reset_prefix_cache /v1/reset_prefix_cache; do
    code=$(curl -sS -o /dev/null -w "%{http_code}" -m 15 -X POST \
      "http://127.0.0.1:${ENGINE_PORT}${ep}" || echo 000)
    echo "reset $ep code=$code"
    [[ "$code" == "200" || "$code" == "204" ]] && break
  done
  sleep 1
  read -r h0 q0 <<<"$(metrics)"
  echo "arm=$arm blocks=$blocks pool=$pool before hits=$h0 queries=$q0 $(date -u +%H:%M:%S)"
  local t0; t0=$(date +%s)
  ENGINE_PORT="$ENGINE_PORT" MODEL="$MODEL" PREAMBLE="$PREAMBLE" \
    GEN_CAP="$GEN_CAP" POOL="$pool" TAG_SUFFIX="-${suffix}" PORT="$PORT" \
    bash ./e2e_autogen.sh "$arm" "$TEAMS" "$MEMBERS" "$ROUNDS" \
    > "$METRIC_DIR/${tag}.log" 2>&1 &
  WATCH_PID=$!
  watch_arm "$arm" "$tag" "$h0" "$q0" || return $?
  local ec=0; wait "$WATCH_PID" || ec=$?
  local wall=$(( $(date +%s) - t0 ))
  echo "arm=$arm e2e_ec=$ec wall=${wall}s $(date -u +%H:%M:%S)"
  tail -20 "$METRIC_DIR/${tag}.log"
  score_metrics "$tag" "$arm" "$h0" "$q0" "$wall" "$blocks" "$pool"
  return 0
}

phase() {
  local blocks="$1" suffix="$2" kind="$3"
  echo "===== phase $kind blocks=$blocks $(date -u +%H:%M:%S) ====="
  if [[ "${SKIP_START:-0}" != "1" ]]; then
    start_vllm "$blocks" || return 1
  else
    echo "SKIP_START=1; reuse engine on :$ENGINE_PORT"
    bash ./wait_engine.sh "$ENGINE_PORT" 30 || return 1
    SKIP_START=0
  fi
  docker logs --tail 30 kimi-k3-b300-vllm 2>&1 | grep -E "\[cohort\]" | tail -3 || true
  run_arm lru "$blocks" "$suffix" || return $?
  run_arm unified "$blocks" "$suffix" || return $?
  python3 - "$METRIC_DIR" "$suffix" "$kind" <<'PY'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
suf, kind = sys.argv[2], sys.argv[3]
arms = {}
for f in d.glob(f"e2e-autogen-design-*-8t-*-{suf}.metrics.json"):
    x = json.loads(f.read_text())
    arms[x["arm"]] = x
    print(f"{kind} {x['arm']:8} hit={x['hit_frac']*100:6.2f}%  "
          f"hits={x['delta_hits']:.0f}/{x['delta_queries']:.0f}  wall={x['wall_s']}s")
if "lru" in arms and "unified" in arms:
    lu, uu = arms["lru"], arms["unified"]
    if kind == "pass" and uu["delta_hits"] < lu["delta_hits"] * 0.6:
        print("FAIL pass: unified hits << LRU (still dropping inserts)", flush=True)
        raise SystemExit(6)
    if kind == "pass":
        print(f"PASS do-no-harm unified/LRU hits={uu['delta_hits']/max(lu['delta_hits'],1):.2f}", flush=True)
    if kind == "oversub":
        print(f"oversub unified-lru = {(uu['hit_frac']-lu['hit_frac'])*100:+.2f}pp", flush=True)
PY
}

echo "vLLM Q9 mask hook $(date -u +%H:%M:%S)"
if [[ "${SKIP_PASS:-0}" != "1" ]]; then
  phase 280 vllm-mask-280 pass || exit $?
fi
# 220 first-reuse was 0 (same class as 160). 240 is peak-ish: first wave
# still fits, later rounds oversub.
over_blocks="${OVERSUB_BLOCKS:-240}"
phase "$over_blocks" "vllm-mask-${over_blocks}" oversub || exit $?
echo "vLLM Q9 DONE $(date -u +%H:%M:%S)"
