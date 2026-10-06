#!/usr/bin/env bash
# Start GLM-4.7-Flash on 4xA100 with the TeamCache engine hook.
#   ENABLE_HICACHE=1 HICACHE_RATIO=2 ./serve_a100.sh   # host-memory hierarchy
set -euo pipefail
source "$(dirname "$0")/common_a100.sh"

COMMON=$("$VENV/bin/python" -c "import sglang,os;print(os.path.dirname(sglang.__file__))")/srt/mem_cache/common.py
if ! grep -q _cohort_skip_insert "$COMMON"; then
  [[ -f "$COMMON.orig" ]] || cp "$COMMON" "$COMMON.orig"
  "$VENV/bin/python" "$TC/patch_sglang_cohort.py" --src "$COMMON.orig" --out /tmp/sgl-patch
  cp /tmp/sgl-patch/common.py "$COMMON"
  find "$(dirname "$COMMON")" -name 'common*.pyc' -delete
fi
grep -q _cohort_skip_insert "$COMMON" && echo "engine hook installed in $COMMON"

[[ -f "$CTRL" ]] || echo '{"mode": "group", "quota": 0}' >"$CTRL"
: >"$DEBUG_HOST"

EXTRA=()
if [[ "${ENABLE_HICACHE:-0}" == "1" ]]; then
  EXTRA+=(--enable-hierarchical-cache --hicache-ratio "${HICACHE_RATIO:-2}")
fi

pkill -f "sglang.launch_server" 2>/dev/null || true
sleep 5
cd "$TC"
COHORT_CONTROL="$CTRL" COHORT_MODE=group COHORT_QUOTA=0 \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
nohup "$VENV/bin/python" -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name "$SERVED" \
  --tp 4 \
  --dtype bfloat16 \
  --attention-backend triton \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.85}" \
  --max-total-tokens "$POOL" \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-32768}" \
  --enable-cache-report \
  --trust-remote-code \
  --host 127.0.0.1 \
  --port "$ENGINE_PORT" \
  "${EXTRA[@]}" >"$ENGINE_LOG" 2>&1 &
echo $! >"$LOGS/engine.pid"
echo "engine pid $(cat "$LOGS/engine.pid") pool=$POOL hicache=${ENABLE_HICACHE:-0}"
wait_health
