#!/usr/bin/env bash
# CrewAI sequential on the live engine. Collected (debug/design rings) but
# never served as a harness fleet. Same 6-team / 28k-preamble geometry as
# run_longctx_field.sh so the miss is the expensive one-page miss, not 9k.
set -euo pipefail
cd "$(dirname "$0")"
export HARNESS=crewai
export TASK="${TASK:-debug}"
export TEAMS="${TEAMS:-6}"
export MEMBERS="${MEMBERS:-5}"
export ROUNDS="${ROUNDS:-6}"
export PREAMBLE="${PREAMBLE:-28000}"
export GEN_CAP="${GEN_CAP:-144}"
export QUOTA="${QUOTA:-1}"
export POOL="${POOL:-420800}"
export ARMS="${ARMS:-lru unified}"
export TAG_SUFFIX="${TAG_SUFFIX:-}"
exec ./e2e_arms.sh
