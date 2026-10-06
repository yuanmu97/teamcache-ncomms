#!/usr/bin/env bash
# Block until an OpenAI-compatible engine answers on PORT.
PORT="${1:-30000}"
TRIES="${2:-180}"
for i in $(seq 1 "$TRIES"); do
  if curl -sf -m 4 "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    echo "engine :$PORT up after ${i} tries $(date -u +%H:%M:%S)"
    exit 0
  fi
  sleep 10
done
echo "engine :$PORT not ready after $TRIES tries" >&2
exit 1
