#!/usr/bin/env bash
# Ablations on the closed-loop replay, at one load point, one engine boot.
#
# Each arm answers a question the main sweep leaves open:
#
#   rotate    Cohort's quota is by member index, so the same members are
#             second-class in every team forever. C3 claims the misses can be
#             *placed*; this asks whether they can be placed down to the member.
#             Rotating may also thrash, and that result counts too.
#   online    The quota everywhere else is a configured constant, and each load
#             point was set so teams x quota equals capacity -- the mechanism
#             is handed the answer. This derives it from a token budget divided
#             by the observed active team count.
#   invdist   The informed baseline in the ScaleSim / KVFlow family. On a cycle
#             every member's reuse distance is identical, so a distance
#             estimator cannot discriminate. Running it turns that from an
#             argument into a measurement.
#   star      Everything so far is a ring. With a hub synthesiser the reuse
#             distances inside a team stop being uniform and evicting the hub
#             blocks k-1 members instead of one.
set -uo pipefail

CTRL=${CTRL:-/tmp/cohort/control.json}
OUTDIR=${OUTDIR:-/tmp/team-ablate}
POOL_TOKENS=${POOL_TOKENS:-420800}
MEMBERS=${MEMBERS:-5}
ROUNDS=${ROUNDS:-5}
WARMUP=${WARMUP:-2}
GEN=${GEN:-144}
CONTEXT=${CONTEXT:-9728}
MESSAGE=${MESSAGE:-512}
VARIANTS=${VARIANTS:-4}
STAGGER=${STAGGER:-2.0}
# 9 teams is where the quota shows a clean step -- members 0 and 1 at 0.0% miss
# against 100% for the rest -- so any ablation that moves the step is visible
# rather than buried under a load that nothing can help.
TEAMS=${TEAMS:-9}
CONTAINER=${CONTAINER:-kimi-k3-b300-sglang-cohort}

mkdir -p "$OUTDIR"
# Same token accounting as the main sweep; see sweep_team.sh for why counting
# sessions puts the working point on the wrong side of the cliff. The older
# VARIANTS-scaled retain charged every session for all of its sibling message
# branches at once; those branches are evicted as leaves and do not accumulate,
# so it overstated the footprint by 19% and derived quota 2 where the main
# sweep derives 3, putting the ablation a full step below the load point it was
# supposed to mirror.
RETAIN=$(( CONTEXT + MESSAGE + GEN ))
INFLIGHT=$RETAIN
CAPACITY=$(( (POOL_TOKENS - TEAMS * INFLIGHT) / RETAIN ))
QUOTA=$(( CAPACITY / TEAMS )); (( QUOTA < 1 )) && QUOTA=1

set_policy() {  # json body
  printf '%s\n' "$1" >"$CTRL"
  sleep 2
  docker logs --tail 2000 "$CONTAINER" 2>&1 | grep '\[cohort\] gen=' | tail -1
}

run_arm() {  # label topology
  local label=$1 topo=${2:-ring}
  local out="$OUTDIR/${label}_${TEAMS}t_${topo}.json"
  [[ -f "$out" ]] && { echo "skip $label (have $out)"; return; }
  echo "=== $label  ${TEAMS} teams  $topo  $(date -u +%H:%M:%S) ==="
  python3 /tmp/replay_team.py \
    --label "$label" --teams "$TEAMS" --members "$MEMBERS" \
    --rounds "$ROUNDS" --warmup-rounds "$WARMUP" --gen-tokens "$GEN" \
    --context "$CONTEXT" --capacity "$CAPACITY" --topology "$topo" \
    --stagger "$STAGGER" --out "$out" 2>&1 | tail -30
}

echo "load point: $TEAMS teams x $MEMBERS members = $(( TEAMS * MEMBERS )) sessions"
echo "capacity=$CAPACITY sessions  quota=$QUOTA  pool=$POOL_TOKENS tokens"
echo

# Ring, one load point. cohort is repeated here so every ablation has a
# same-session baseline rather than being compared across sweeps.
set_policy "{\"mode\":\"group\",\"quota\":$QUOTA,\"global_cap\":0,\"members\":$MEMBERS,\"capacity\":$CAPACITY,\"rotate_period\":0,\"window_s\":30,\"pool_tokens\":0}"
run_arm cohort ring

set_policy "{\"mode\":\"rotate\",\"quota\":$QUOTA,\"global_cap\":0,\"members\":$MEMBERS,\"capacity\":$CAPACITY,\"rotate_period\":1,\"window_s\":30,\"pool_tokens\":0}"
run_arm rotate ring

set_policy "{\"mode\":\"online\",\"quota\":0,\"global_cap\":0,\"members\":$MEMBERS,\"capacity\":$CAPACITY,\"rotate_period\":0,\"window_s\":30,\"pool_tokens\":$POOL_TOKENS}"
run_arm online ring

set_policy "{\"mode\":\"invdist\",\"quota\":0,\"global_cap\":0,\"members\":$MEMBERS,\"capacity\":$CAPACITY,\"rotate_period\":0,\"window_s\":30,\"pool_tokens\":0}"
run_arm invdist ring

# Star: hub plus spokes. Three arms, because the question is whether the
# placement result generalises past the ring, not just whether it is faster.
for arm in lru bypass cohort; do
  case "$arm" in
    lru)    set_policy "{\"mode\":\"group\",\"quota\":0,\"global_cap\":0,\"members\":$MEMBERS,\"capacity\":$CAPACITY,\"rotate_period\":0,\"window_s\":30,\"pool_tokens\":0}" ;;
    bypass) set_policy "{\"mode\":\"global\",\"quota\":0,\"global_cap\":$(( TEAMS * QUOTA )),\"members\":$MEMBERS,\"capacity\":$CAPACITY,\"rotate_period\":0,\"window_s\":30,\"pool_tokens\":0}" ;;
    cohort) set_policy "{\"mode\":\"group\",\"quota\":$QUOTA,\"global_cap\":0,\"members\":$MEMBERS,\"capacity\":$CAPACITY,\"rotate_period\":0,\"window_s\":30,\"pool_tokens\":0}" ;;
  esac
  run_arm "$arm" star
done

echo
echo "=== ablations done $(date -u +%H:%M:%S) ==="
ls -la "$OUTDIR"
