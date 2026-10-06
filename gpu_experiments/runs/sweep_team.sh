#!/usr/bin/env bash
# Team-level sweep for the closed-loop replay. Runs unattended on the engine
# host; every arm switches policy by rewriting the control file, so the engine
# is booted once for the whole sweep instead of once per arm.
#
# Three arms at each team count:
#   lru      upstream behaviour (quota 0 takes the unmodified path)
#   bypass   the same admission budget spent with no notion of a team
#   cohort   the same budget divided per team
#
# The bypass arm is the one that matters. Aggregate cost cannot separate it
# from cohort -- measured previously, it was 0.9% faster -- so if the team-level
# metrics cannot separate them either, the group part of the mechanism is not
# earning its keep and the paper should say so.
set -uo pipefail

# Working point, in tokens. Counting sessions was our first attempt and it put
# the whole sweep on the wrong side of a cliff, so the accounting is worth
# spelling out.
#
# A session's lasting footprint is one branch, not all of them:
#     retain = CONTEXT + MESSAGE + GEN                       = 10,384
# An in-flight request needs working KV for the same amount, and with one
# runnable member per team the concurrency equals the team count, so
#     budget_tokens = POOL - teams * retain
#
# We got here by being wrong twice, in opposite directions.
#
# First pass used a flat 40 sessions and ignored the in-flight requests, which
# put every team count from 12 up past the cliff; all three arms then read
# exactly 100% miss, three identical numbers that look like a result and are
# not one. Second pass charged each session CONTEXT + VARIANTS * (MESSAGE +
# GEN) = 12,352, on the theory that all VARIANTS sibling branches persist.
# Measurement refutes that: at 8 teams upstream LRU reads only 12.0% miss,
# which requires it to be holding roughly 35 of the 40 sessions, impossible at
# 12,352 each. The older sibling branches are leaves and eviction takes leaves
# first, so they do not accumulate. Overstating the footprint made the quota
# too small, and the mechanism then bypassed sessions that would have fit and
# measured *slower* than upstream below the cliff -- 1.20x at 6 teams, 1.34x at
# 8. Worth stating plainly: a retention mechanism calibrated with a pessimistic
# capacity estimate is a pessimisation, which is the argument for deriving the
# budget online rather than configuring it (the `online` arm).
CTRL=${CTRL:-/tmp/cohort/control.json}
OUTDIR=${OUTDIR:-/tmp/team-sweep}
POOL=${POOL:-420800}
MEMBERS=${MEMBERS:-5}
ROUNDS=${ROUNDS:-5}
WARMUP=${WARMUP:-2}
SETTLE=${SETTLE:-0}
GEN=${GEN:-144}
CONTEXT=${CONTEXT:-9728}
MESSAGE=${MESSAGE:-512}
VARIANTS=${VARIANTS:-4}
TOPOLOGY=${TOPOLOGY:-ring}
# Finer steps than the first pass, because the interesting region is narrow:
# 8 teams still reads 79% cached and 12 teams reads 0%, so the cliff is inside
# that gap and a sweep that steps over it cannot separate the policies.
TEAM_COUNTS=${TEAM_COUNTS:-"6 8 9 10 11 12 14 16"}
ARMS=${ARMS:-"lru bypass cohort online"}
CONTAINER=${CONTAINER:-kimi-k3-b300-sglang-cohort}
# Seconds between team arrivals. Zero makes group-blind bypass admit the same
# first q members of every team as a per-team quota does, which would compare
# the mechanism against itself.
STAGGER=${STAGGER:-2.0}

mkdir -p "$OUTDIR"

RETAIN=$(( CONTEXT + MESSAGE + GEN ))
INFLIGHT=$RETAIN
echo "retain=$RETAIN tok/session  inflight=$INFLIGHT tok/request  pool=$POOL"

ADAPT_THRESHOLD=${ADAPT_THRESHOLD:-0.6}
ADAPT_WINDOW=${ADAPT_WINDOW:-40}

set_policy() {  # mode quota global_cap [members] [capacity] [rotate_period]
  cat >"$CTRL" <<EOF
{"mode": "$1", "quota": $2, "global_cap": $3,
 "members": ${4:-$MEMBERS}, "capacity": ${5:-0},
 "rotate_period": ${6:-0}, "window_s": 30, "pool_tokens": $POOL,
 "adapt_threshold": $ADAPT_THRESHOLD, "adapt_window": $ADAPT_WINDOW}
EOF
  # The engine restats at most once a second; give it two, then read back what
  # the engine itself logged. The first pass of this sweep trusted the write
  # and reported "0 changes" for every arm without anyone noticing, so this
  # prints the engine's own view of the live config and nothing else.
  sleep 2
  local line
  line=$(docker logs --tail 400 "$CONTAINER" 2>&1 \
         | grep '\[cohort\] gen=' | tail -1 || true)
  echo "policy -> want mode=$1 quota=$2 cap=$3"
  echo "          engine ${line:-<no cohort log line; switch NOT confirmed>}"
}

run_arm() {  # label teams capacity_sessions
  local label=$1 teams=$2 cap=$3
  local out="$OUTDIR/${label}_${teams}t_${TOPOLOGY}.json"
  if [[ -f "$out" ]]; then
    echo "skip $label $teams teams (have $out)"
    return
  fi
  echo "=== $label  ${teams} teams  $(date -u +%H:%M:%S) ==="
  python3 /tmp/replay_team.py \
    --label "$label" --teams "$teams" --members "$MEMBERS" \
    --rounds "$ROUNDS" --warmup-rounds "$WARMUP" \
    --settle-rounds "$SETTLE" --gen-tokens "$GEN" \
    --context "$CONTEXT" --capacity "$cap" --topology "$TOPOLOGY" \
    --stagger "$STAGGER" --out "$out" 2>&1 | tail -32
}

for teams in $TEAM_COUNTS; do
  budget_tok=$(( POOL - teams * INFLIGHT ))
  cap=$(( budget_tok / RETAIN )); (( cap < 1 )) && cap=1
  quota=$(( cap / teams )); (( quota < 1 )) && quota=1
  budget=$(( teams * quota ))
  echo
  echo "### ${teams} teams  $(( teams * MEMBERS )) sessions  "\
       "retainable=$cap  quota=$quota  budget=$budget sessions  "\
       "load=$(python3 -c "print(f'{$teams*$MEMBERS/$cap:.2f}')")x"
  for arm in $ARMS; do
    case "$arm" in
      # Quota 0 takes the unmodified upstream path.
      lru)    set_policy group  0        0 "$MEMBERS" "$cap" ;;
      # The same admission budget with no notion of a team, so any difference
      # from cohort is the group part and nothing else.
      bypass) set_policy global 0        "$budget" "$MEMBERS" "$cap" ;;
      cohort) set_policy group  "$quota" 0 "$MEMBERS" "$cap" ;;
      # Same mechanism with nothing configured but the pool size: the quota is
      # derived from the pool less one in-flight request per active team,
      # divided by the teams it actually sees. Every other arm is handed a
      # quota chosen so that teams x quota equals capacity, which quietly
      # gives the mechanism the answer.
      online) set_policy online 0        0 "$MEMBERS" "$cap" ;;
      # Same per-team quota as cohort, but only spent on teams whose measured
      # recency rank says LRU is about to fail them. On the empirical order
      # LRU wins every count -- 56.2% miss against cohort's 73.5% at sixteen
      # teams -- so a mechanism that always engages is a regression on the
      # traffic we actually recorded, and the arm that matters is the one that
      # can tell the two populations apart without being told which is which.
      adaptive) set_policy adaptive "$quota" 0 "$MEMBERS" "$cap" ;;
      # One rule for every speaking order, and told nothing but the pool size
      # -- not the quota, not the member count, not which order it is serving.
      # Each team compares the reference mass of the members that fit in its
      # share against the LRU hit rate it is measurably getting, and freezes a
      # subset only when that trade is worth taking. `adaptive` asks the
      # narrower question "ring or not", which a single per-team scalar cannot
      # answer for a team whose members are referenced unequally.
      unified) set_policy unified 0        0 "$MEMBERS" "$cap" ;;
      *)      echo "unknown arm $arm" >&2; continue ;;
    esac
    run_arm "$arm" "$teams" "$cap"
  done
done

echo
echo "=== sweep done $(date -u +%H:%M:%S) ==="
ls -la "$OUTDIR"
