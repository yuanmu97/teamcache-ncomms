#!/usr/bin/env bash
# Collect a ring trace on the inference host itself.
#
# Runs on b300 rather than over the ssh forward: a k=5 x 8-round collection is
# roughly an hour of wall time and the forward does not survive that. The
# first attempt lost two runs to ConnectionRefused, so the harness, the
# recording proxy and the engine now all sit on one machine and nothing in the
# path can drop.
#
#   ssh b300 'cd /opt/dlami/nvme/pat && ./collect_ring.sh autogen 5 8'
#
set -uo pipefail

HARNESS="${1:-autogen}"
MEMBERS="${2:-5}"
ROUNDS="${3:-8}"
PORT="${PORT:-8901}"
ENGINE_PORT="${ENGINE_PORT:-30000}"

cd "$(dirname "$0")"
mkdir -p traces

TAG="${HARNESS}-ring-k${MEMBERS}"
TRACE="traces/${TAG}.jsonl"

case "$HARNESS" in
  autogen) PY=.venv-ag/bin/python;     SCRIPT=harness_roundrobin.py ;;
  crewai)  PY=.venv-crewai/bin/python; SCRIPT=harness_crewai.py ;;
  *) echo "unknown harness $HARNESS" >&2; exit 2 ;;
esac

# Never delete a trace the proxy may still hold open: an earlier run wrote an
# entire collection into an unlinked inode that way and it was unrecoverable.
if [ -e "$TRACE" ]; then
  mv "$TRACE" "${TRACE}.$(date -u +%Y%m%dT%H%M%SZ).bak"
fi

pkill -f "recording-proxy-${PORT}" 2>/dev/null
sleep 1
python3 spawn.py --log "/tmp/${TAG}-proxy.log" -- \
  python3 proxy_openai.py --listen-port "$PORT" \
  --upstream-port "$ENGINE_PORT" --trace "$TRACE" --tag "recording-proxy-${PORT}"
sleep 3

python3 spawn.py --log "/tmp/${TAG}.log" -- \
  "$PY" "$SCRIPT" --base-url "http://127.0.0.1:${PORT}/v1" \
  --members "$MEMBERS" --rounds "$ROUNDS"

sleep 3
echo "collecting ${TAG}: $(pgrep -cf "$SCRIPT") harness proc(s)"
echo "trace  $TRACE"
echo "logs   /tmp/${TAG}.log  /tmp/${TAG}-proxy.log"
