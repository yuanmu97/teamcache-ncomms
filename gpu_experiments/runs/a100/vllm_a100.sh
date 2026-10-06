#!/usr/bin/env bash
# vLLM port on 4xA100-80GB: eight live AutoGen ring teams for sixteen rounds
# with the block-mask hook, LRU vs TeamCache, three seeds.  The block pool is
# fixed at 430,080 tokens, the B300 setting, and the prefix
# cache counters are sampled every 10 s so that hits after eight and after
# sixteen rounds can both be read from one run.
set -uo pipefail
cd "$(dirname "$0")"
source ./common_a100.sh

VVENV="${VVENV:-$TC/.venv-vllm}"
VPORT="${VPORT:-8888}"
BLOCK=16
POOL_V="${POOL_V:-430080}"
BLOCKS=$((POOL_V / BLOCK))
VLOG="$LOGS/engine_vllm.log"
VOUT="$OUT/vllm"
TEAMS=8 MEMBERS=5 ROUNDS=16
mkdir -p "$VOUT"

sed -e 's/"model":"kimi"/"model":"'"$SERVED"'"/' \
    -e 's#docker logs --tail 400 kimi-k3-b300-sglang-cohort 2>&1#tail -n 400 '"$VLOG"'#' \
    e2e_autogen.sh >e2e_autogen_vllm.sh
chmod +x e2e_autogen_vllm.sh

metrics() {
  "$VVENV/bin/python" - "$VPORT" <<'PY'
import sys, urllib.request
t = urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/metrics", timeout=8).read().decode()
h = q = 0.0
for line in t.splitlines():
    if line.startswith("vllm:prefix_cache_hits_total{"):
        h += float(line.rsplit(None, 1)[-1])
    elif line.startswith("vllm:prefix_cache_queries_total{"):
        q += float(line.rsplit(None, 1)[-1])
print(f"{h:.0f} {q:.0f}")
PY
}

start_vllm() {
  pkill -f "sglang.launch_server" 2>/dev/null || true
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 30
  [[ -f "$CTRL" ]] || echo '{"mode": "group", "quota": 0}' >"$CTRL"
  COHORT_CONTROL="$CTRL" COHORT_MODE=group COHORT_QUOTA=0 \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
  nohup "$VVENV/bin/vllm" serve "$MODEL" \
    --served-model-name "$SERVED" \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --block-size "$BLOCK" \
    --num-gpu-blocks-override "$BLOCKS" \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization "${VLLM_MEM:-0.80}" \
    --reasoning-parser glm45 \
    --trust-remote-code \
    --host 127.0.0.1 --port "$VPORT" >"$VLOG" 2>&1 &
  echo $! >"$LOGS/engine_vllm.pid"
  local i
  for i in $(seq 1 240); do
    curl -sf -m 3 "http://127.0.0.1:${VPORT}/health" >/dev/null 2>&1 && { echo "vLLM up after $((i * 5))s"; return 0; }
    kill -0 "$(cat "$LOGS/engine_vllm.pid")" 2>/dev/null || { echo "vLLM exited"; tail -40 "$VLOG"; return 1; }
    sleep 5
  done
  return 1
}

run_arm() {  # arm seed
  local arm=$1 seed=$2
  local tag="e2e-autogen-design-${arm}-${TEAMS}t-p3000r${ROUNDS}-a100-vllm"
  [[ $seed != 0 ]] && tag="${tag}-s${seed}"
  [[ -f "$VOUT/${tag}.metrics.json" ]] && return 0
  curl -s -m 15 -X POST "http://127.0.0.1:${VPORT}/reset_prefix_cache" >/dev/null || true
  sleep 2
  read -r h0 q0 <<<"$(metrics)"
  local t0; t0=$(date +%s)
  # Thinking off: vLLM returns GLM-4.7-Flash's reasoning outside `content`, so
  # with thinking on the agents' replies were empty and histories never grew.
  PROXY_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}' \
  HARNESS=autogen TASK=design TEAMS=$TEAMS MEMBERS=$MEMBERS ROUNDS=$ROUNDS PREAMBLE=3000 \
  GEN_CAP=144 QUOTA=2 POOL=$POOL_V SEED=$seed OUTDIR="$VOUT" MODEL="$SERVED" \
  TAG_SUFFIX=-a100-vllm ENGINE_PORT=$VPORT \
    timeout 3600 ./e2e_autogen_vllm.sh "$arm" "$TEAMS" "$MEMBERS" "$ROUNDS" >"$VOUT/${tag}.log" 2>&1 &
  local pid=$!
  : >"$VOUT/${tag}.samples.jsonl"
  while kill -0 "$pid" 2>/dev/null; do
    local n=0
    [[ -f "traces/${tag}.jsonl" ]] && n=$(wc -l <"traces/${tag}.jsonl")
    read -r h q <<<"$(metrics)"
    echo "{\"t\": $(( $(date +%s) - t0 )), \"requests\": ${n:-0}, \"hits\": $((h - h0)), \"queries\": $((q - q0))}" \
      >>"$VOUT/${tag}.samples.jsonl"
    sleep 10
  done
  wait "$pid"; local ec=$?
  local wall=$(( $(date +%s) - t0 ))
  read -r h1 q1 <<<"$(metrics)"
  cat >"$VOUT/${tag}.metrics.json" <<EOF
{"tag": "$tag", "arm": "$arm", "seed": $seed, "teams": $TEAMS, "members": $MEMBERS,
 "rounds": $ROUNDS, "pool_tokens": $POOL_V, "block_size": $BLOCK, "num_gpu_blocks": $BLOCKS,
 "exit_code": $ec, "wall_s": $wall, "delta_hits": $((h1 - h0)), "delta_queries": $((q1 - q0))}
EOF
  log_run "$tag" "$([[ $ec == 0 ]] && echo ok || echo fail)" "$wall"
  sleep 10
}

echo "vLLM A100 start $(date -u)"
start_vllm || { log_run vllm_boot fail 0; exit 1; }
for s in ${SEEDS:-0 1 2}; do
  if (( s % 2 == 0 )); then arms="lru unified"; else arms="unified lru"; fi
  for arm in $arms; do run_arm "$arm" "$s"; done
done
pkill -f "vllm serve" 2>/dev/null || true
echo "vLLM A100 done $(date -u)"
