#!/usr/bin/env bash
# Kimi K3 B300 with Cohort per-group KV residency. Research variant of
# serve_kimi_k3_b300_sglang.sh; identical serving flags, so the two arms of the
# A/B differ only in COHORT_QUOTA.
#
# COHORT_QUOTA=0 (default) bind-mounts the patched file but leaves the predicate
# inert, which is how the baseline arm runs: same image, same mount, same code
# path, so nothing but the quota can explain a difference.
#
# Generate the patch first:
#   python3 research/peer-agent-serving/patch_sglang_cohort.py --out /tmp/sgl-patch
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=/dev/null
source ./common_b300.sh

if [[ ! -f "$MODEL_KIMI_K3/config.json" ]]; then
  echo "missing $MODEL_KIMI_K3/config.json" >&2
  exit 1
fi

NAME=kimi-k3-b300-sglang-cohort
PORT="${SGLANG_PORT:-30000}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-1048576}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
MAMBA_FULL_MEMORY_RATIO="${MAMBA_FULL_MEMORY_RATIO:-1.8}"

COHORT_QUOTA="${COHORT_QUOTA:-0}"
# Modes: group (Cohort), rotate (same budget, resident subset rotates within
# each team), online (quota derived from capacity over the observed active team
# count instead of being handed in), global (group-blind bypass with the same
# budget, to test whether grouping earns its keep on fairness rather than on
# aggregate cost), invdist (informed baseline that bypasses items whose observed
# reuse distance exceeds capacity).
COHORT_MODE="${COHORT_MODE:-group}"
COHORT_GLOBAL_CAP="${COHORT_GLOBAL_CAP:-0}"
COHORT_MEMBERS="${COHORT_MEMBERS:-0}"
COHORT_CAPACITY="${COHORT_CAPACITY:-0}"
COHORT_ROTATE_PERIOD="${COHORT_ROTATE_PERIOD:-0}"
COHORT_WINDOW_S="${COHORT_WINDOW_S:-30}"
COHORT_PATCH_DIR="${COHORT_PATCH_DIR:-/tmp/sgl-patch}"
SGL_COMMON=/sgl-workspace/sglang/python/sglang/srt/mem_cache/common.py

# The policy is re-read from this file at most once a second, so an arm can
# switch without a restart. That matters: a restart is about twelve minutes and
# the load curve plus the ablations come to roughly twenty arms. The env vars
# above stay as the initial values, so a run with no control file behaves
# exactly as before.
#
# The *directory* is mounted, not the file. A single-file bind mount binds an
# inode, so rewriting the file by rename -- which is what most editors and
# tools do -- would leave the container reading the old inode forever.
COHORT_CONTROL_DIR="${COHORT_CONTROL_DIR:-/tmp/cohort}"
COHORT_CONTROL="$COHORT_CONTROL_DIR/control.json"
mkdir -p "$COHORT_CONTROL_DIR"
if [[ ! -f "$COHORT_CONTROL" ]]; then
  cat >"$COHORT_CONTROL" <<EOF
{"mode": "$COHORT_MODE", "quota": $COHORT_QUOTA,
 "global_cap": $COHORT_GLOBAL_CAP, "members": $COHORT_MEMBERS,
 "capacity": $COHORT_CAPACITY, "rotate_period": $COHORT_ROTATE_PERIOD,
 "window_s": $COHORT_WINDOW_S}
EOF
fi

if [[ ! -f "$COHORT_PATCH_DIR/common.py" ]]; then
  echo "missing $COHORT_PATCH_DIR/common.py -- run patch_sglang_cohort.py" >&2
  exit 1
fi
if ! grep -q "_cohort_skip_insert" "$COHORT_PATCH_DIR/common.py"; then
  echo "$COHORT_PATCH_DIR/common.py is not the patched file" >&2
  exit 1
fi

# Shrinks the KV pool so contention can be reached with fewer teams. Unset
# leaves SGLang's own sizing, which is what the reported capacity is measured
# against.
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-}"
SGL_EXTRA=()
if [[ -n "$MAX_TOTAL_TOKENS" ]]; then
  SGL_EXTRA+=(--max-total-tokens "$MAX_TOTAL_TOKENS")
fi

# Bypass is only exact when a request's prefill is not chunked: the chunked path
# must keep its bookkeeping, so a chunked bypassed request retains its completed
# chunks. Sized above the replay's prompt (context + peer message) so both arms
# prefill in one shot. Applied to both arms, so it cannot explain a difference.
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-32768}"
SGL_EXTRA+=(--chunked-prefill-size "$CHUNKED_PREFILL_SIZE")
if [[ "${ENABLE_HICACHE:-0}" == "1" ]]; then
  SGL_EXTRA+=(--enable-hierarchical-cache)
  # Host / device size. Default 2 → joint pool ≈ (1+2)× device tokens.
  SGL_EXTRA+=(--hicache-ratio "${HICACHE_RATIO:-2}")
fi

mkdir -p "$LOGDIR"
# Host-side file for unified's /tmp/unified_debug.jsonl. Decision code is
# unchanged; this is only so additional_validation can read want over time.
# Keep it off the read-only control-dir mount.
UNIFIED_DEBUG_FILE="${UNIFIED_DEBUG_FILE:-/tmp/cohort-debug/unified_debug.jsonl}"
mkdir -p "$(dirname "$UNIFIED_DEBUG_FILE")"
touch "$UNIFIED_DEBUG_FILE"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --restart unless-stopped --init \
  --gpus all \
  --network host \
  --ipc=host \
  -e COHORT_QUOTA="$COHORT_QUOTA" \
  -e COHORT_MODE="$COHORT_MODE" \
  -e COHORT_GLOBAL_CAP="$COHORT_GLOBAL_CAP" \
  -e COHORT_MEMBERS="$COHORT_MEMBERS" \
  -e COHORT_CAPACITY="$COHORT_CAPACITY" \
  -e COHORT_ROTATE_PERIOD="$COHORT_ROTATE_PERIOD" \
  -e COHORT_WINDOW_S="$COHORT_WINDOW_S" \
  -e COHORT_CONTROL="$COHORT_CONTROL" \
  -v "$COHORT_CONTROL_DIR:$COHORT_CONTROL_DIR:ro" \
  -v "$COHORT_PATCH_DIR/common.py:$SGL_COMMON:ro" \
  -v "$NVME/models:$NVME/models" \
  -v "$NVME/cache:/root/.cache/huggingface" \
  -v "$NVME/logs:$LOGDIR" \
  -v "$UNIFIED_DEBUG_FILE:/tmp/unified_debug.jsonl" \
  "$SGLANG_KIMI_K3_IMAGE" \
  sglang serve \
    --trust-remote-code \
    --model-path "$MODEL_KIMI_K3" \
    --tp-size 8 \
    --context-length "$CONTEXT_LENGTH" \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --mamba-full-memory-ratio "$MAMBA_FULL_MEMORY_RATIO" \
    --weight-loader-prefetch-checkpoints \
    --reasoning-parser kimi_k3 \
    --tool-call-parser kimi_k3 \
    --enable-cache-report \
    "${SGL_EXTRA[@]}" \
    --host 0.0.0.0 \
    --port "$PORT"

echo "started $NAME on :$PORT image=$SGLANG_KIMI_K3_IMAGE"
echo "COHORT_MODE=$COHORT_MODE COHORT_QUOTA=$COHORT_QUOTA COHORT_GLOBAL_CAP=$COHORT_GLOBAL_CAP"
echo "control file $COHORT_CONTROL (re-read within 1s; edit in place to switch arms)"
echo "max_total_tokens=${MAX_TOTAL_TOKENS:-<engine default>}"
