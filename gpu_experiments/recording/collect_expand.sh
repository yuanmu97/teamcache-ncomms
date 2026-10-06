#!/usr/bin/env bash
# Phase 3: screened collection cells. One engine, serial, flocked per port.
# Usage: ./collect_expand.sh k3|k3-k|glm
set -uo pipefail
cd "$(dirname "$0")"
WAVE="${1:-k3}"

run_one() {
  local h="$1" t="$2" k="$3" r="$4"
  echo "######## collect $h $t k=$k r=$r  $(date -u +%H:%M:%S) ########"
  ./collect_trace.sh "$h" "$t" "$k" "$r"
}

case "$WAVE" in
  k3)
    run_one selector design 5 8
    run_one autogen debug 5 8
    run_one autogen review 5 8
    run_one selector debug 5 8
    run_one selector review 5 8
    run_one crewai debug 5 8
    ;;
  k3-k)
    run_one autogen design 6 8
    run_one autogen design 8 8
    ;;
  glm)
    echo "GLM wave expects MODEL=/opt/dlami/nvme/models/GLM-5.3 MODEL_TAG=GLM-5.3" >&2
    export MODEL="${MODEL:-/opt/dlami/nvme/models/GLM-5.3}"
    export MODEL_TAG=GLM-5.3
    run_one autogen design 5 8
    run_one selector design 5 8
    run_one selector debug 5 8
    ;;
  *) echo "usage: $0 k3|k3-k|glm" >&2; exit 2 ;;
esac
echo "collect_expand $WAVE DONE $(date -u +%H:%M:%S)"
