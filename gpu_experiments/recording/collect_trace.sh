#!/usr/bin/env bash
# Collect one workload trace: harness x task x team size x model.
#
# Supersedes `collect_ring.sh`, which could only vary the team size on one task
# with two serialising harnesses. That was enough to establish the ring exists
# in deployed configurations, and not enough for anything else: limitation
# §12.8 is that every motivating number in the paper rests on two runs of one
# task on one model, and we have already been burnt once by exactly that -- the
# ring was reported as the measured shape of peer collaboration until a wider
# sample showed it is a property of the harness.
#
# Three axes matter, for three different reasons:
#
#   harness  autogen and crewai serialise by construction; selector asks a
#            model who speaks next, which is the free-form regime we otherwise
#            only have two Claude Code runs of.
#   task     `design` is the collected class, kept identical. `debug` lets
#            members defer and wait, which is where a false-completion rate of
#            77% would break if it were an artefact of a task that never lets
#            an agent stop. `review` puts one artefact in front of everybody,
#            the most favourable case we can construct for cross-member prefix
#            sharing.
#   model    the characterisation should be a property of the collaboration,
#            not of one model's verbosity. GLM 5.3 is already deployable on
#            this machine, so this is the cheap half of §12.8.
#
# Runs entirely on the inference host: a k=5 x 8-round collection is about an
# hour and an ssh forward does not survive that.
#
#   ssh b300 'cd /opt/dlami/nvme/pat && ./collect_trace.sh selector debug 5 8'
#
set -uo pipefail

HARNESS="${1:-autogen}"
TASK="${2:-design}"
MEMBERS="${3:-5}"
ROUNDS="${4:-8}"
PORT="${PORT:-8901}"
ENGINE_PORT="${ENGINE_PORT:-30000}"
MODEL="${MODEL:-/opt/dlami/nvme/models/Kimi-K3}"
MODEL_TAG="${MODEL_TAG:-$(basename "$MODEL")}"

cd "$(dirname "$0")"
mkdir -p traces

case "$HARNESS" in
  autogen)  PY=.venv-ag/bin/python;     SCRIPT=harness_roundrobin.py ;;
  selector) PY=.venv-ag/bin/python;     SCRIPT=harness_selector.py ;;
  crewai)   PY=.venv-crewai/bin/python; SCRIPT=harness_crewai.py ;;
  *) echo "unknown harness $HARNESS (want autogen|selector|crewai)" >&2; exit 2 ;;
esac

# The tag carries every axis, because a trace whose provenance is not in its
# name is a trace that will be compared against the wrong thing later.
TAG="${HARNESS}-${TASK}-k${MEMBERS}-${MODEL_TAG}"
TRACE="traces/${TAG}.jsonl"

# A collection that silently shares a proxy port with another collection would
# interleave two harnesses into one trace file, and every per-member statistic
# computed from it would be wrong in a way that still looks plausible.
exec 7>"/tmp/collect-${PORT}.lock"
if ! flock -n 7; then
  echo "another collection holds port ${PORT}; refusing to start" >&2
  exit 3
fi

# Never delete a trace the proxy may still hold open: an earlier run wrote an
# entire collection into an unlinked inode that way and it was unrecoverable.
if [ -e "$TRACE" ]; then
  mv "$TRACE" "${TRACE}.$(date -u +%Y%m%dT%H%M%SZ).bak"
fi

pkill -f "recording-proxy-${PORT}" 2>/dev/null
sleep 1
python3 spawn.py --log "/tmp/${TAG}-proxy.log" -- \
  python3 proxy_openai.py --listen-port "$PORT" \
  --upstream-port "$ENGINE_PORT" --trace "$TRACE" --inject-rid \
  --tag "recording-proxy-${PORT}"
sleep 3

python3 spawn.py --log "/tmp/${TAG}.log" -- \
  "$PY" "$SCRIPT" --base-url "http://127.0.0.1:${PORT}/v1" --model "$MODEL" \
  --task "$TASK" --members "$MEMBERS" --rounds "$ROUNDS"

echo "collecting ${TAG}: waiting for $SCRIPT"
# spawn.py backgrounds; a collection scored at t+5s is an empty file.
while pgrep -f "$SCRIPT" >/dev/null; do
  sleep 15
done
echo "trace  $TRACE  lines=$(wc -l < "$TRACE" 2>/dev/null || echo 0)"
echo "logs   /tmp/${TAG}.log  /tmp/${TAG}-proxy.log"
python3 analyze_ring.py "$TRACE" --json-out "${TRACE%.jsonl}.summary.json" || true
