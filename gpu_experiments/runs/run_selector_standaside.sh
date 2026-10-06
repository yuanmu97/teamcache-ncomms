#!/usr/bin/env bash
# Free-order live at the stand-aside point, not the 32-team engage point.
#
# 32-team Selector already showed unified >= LRU, but the rule had engaged
# and there was no static arm. Offline, LRU leads on this trace at 8-20
# teams. 20 teams + the ring's q=2 is the classifier-mistake setting:
# static does what it does on a ring; TeamCache should refuse to.
set -uo pipefail
cd "$(dirname "$0")"

export HARNESS=selector
export TASK=debug
export TEAMS="${TEAMS:-20}"
export MEMBERS="${MEMBERS:-5}"
export ROUNDS="${ROUNDS:-8}"
export PREAMBLE="${PREAMBLE:-3000}"
export GEN_CAP="${GEN_CAP:-144}"
export QUOTA="${QUOTA:-2}"
export ARMS="${ARMS:-lru cohort unified}"
export POOL="${POOL:-420800}"

echo "selector stand-aside  TEAMS=$TEAMS QUOTA=$QUOTA PREAMBLE=$PREAMBLE GEN_CAP=$GEN_CAP ARMS=$ARMS  $(date -u +%H:%M:%S)"
./e2e_arms.sh
echo "selector stand-aside DONE $(date -u +%H:%M:%S)"
