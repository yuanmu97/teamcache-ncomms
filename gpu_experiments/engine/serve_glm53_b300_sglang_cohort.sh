#!/usr/bin/env bash
# GLM-5.3 B300 SGLang with the same cohort/unified bind-mount as K3.
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=/dev/null
source ./common_b300.sh

if [[ ! -f "$MODEL_GLM53/config.json" ]]; then
  echo "missing $MODEL_GLM53/config.json" >&2
  exit 1
fi

NAME=glm53-b300-sglang
PORT="${SGLANG_PORT:-30000}"
COHORT_PATCH_DIR="${COHORT_PATCH_DIR:-/tmp/sgl-patch}"
SGL_COMMON=/sgl-workspace/sglang/python/sglang/srt/mem_cache/common.py
COHORT_CONTROL_DIR="${COHORT_CONTROL_DIR:-/tmp/cohort}"
COHORT_CONTROL="$COHORT_CONTROL_DIR/control.json"
mkdir -p "$COHORT_CONTROL_DIR" "$LOGDIR"
if [[ ! -f "$COHORT_CONTROL" ]]; then
  echo '{"mode":"group","quota":0,"global_cap":0,"members":5,"capacity":0,"rotate_period":0,"window_s":30}' \
    >"$COHORT_CONTROL"
fi
if [[ ! -f "$COHORT_PATCH_DIR/common.py" ]] || ! grep -q "_cohort_skip_insert" "$COHORT_PATCH_DIR/common.py"; then
  echo "missing patched $COHORT_PATCH_DIR/common.py" >&2
  exit 1
fi

MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-}"
SGL_EXTRA=()
if [[ -n "$MAX_TOTAL_TOKENS" ]]; then
  SGL_EXTRA+=(--max-total-tokens "$MAX_TOTAL_TOKENS")
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --restart unless-stopped --init \
  --gpus all \
  --network host \
  --ipc=host \
  -e COHORT_CONTROL="$COHORT_CONTROL" \
  -v "$COHORT_CONTROL_DIR:$COHORT_CONTROL_DIR:ro" \
  -v "$COHORT_PATCH_DIR/common.py:$SGL_COMMON:ro" \
  -v "$NVME/models:$NVME/models" \
  -v "$NVME/cache:/root/.cache/huggingface" \
  -v "$NVME/logs:$LOGDIR" \
  "$SGLANG_B300_IMAGE" \
  sglang serve \
    --model-path "$MODEL_GLM53" \
    --tp 8 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 6 \
    --mem-fraction-static 0.8 \
    --enable-cache-report \
    "${SGL_EXTRA[@]}" \
    --host 0.0.0.0 \
    --port "$PORT"

echo "started $NAME on :$PORT image=$SGLANG_B300_IMAGE max_total_tokens=${MAX_TOTAL_TOKENS:-<default>}"
