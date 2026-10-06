#!/usr/bin/env bash
# End-to-end: real AutoGen teams, real engine, one policy per arm.
#
# Every other measurement in this paper goes through `replay_team.py`, which
# reproduces the wait graph and the token sizes but generates its prompts. The
# objection that answers is: does any of this survive a harness we did not
# write, driving a model that decides its own output lengths? So this arm runs
# N concurrent RoundRobinGroupChat teams -- the configuration measured at
# normalised recency rank 1.00, a perfect ring -- and times them.
#
# The only thing between AutoGen and the engine is the recording gateway,
# which stamps <team>:<member>:<turn> onto each request. AutoGen itself knows
# nothing beyond the team name it sends in a header.
#
#   ssh b300 'cd /opt/dlami/nvme/pat && ./e2e_autogen.sh lru 8 5 4'
#
set -uo pipefail

ARM="${1:-lru}"
TEAMS="${2:-8}"
MEMBERS="${3:-5}"
ROUNDS="${4:-4}"

PORT="${PORT:-8910}"
ENGINE_PORT="${ENGINE_PORT:-30000}"
MODEL="${MODEL:-/opt/dlami/nvme/models/Kimi-K3}"
CTRL="${CTRL:-/tmp/cohort/control.json}"
POOL="${POOL:-420800}"
OUTDIR="${OUTDIR:-/opt/dlami/nvme/pat/results/e2e}"

# Which population of teams to serve. The ring is the regime the mechanism was
# designed for, but the requirement we set ourselves is that a free-form order
# must not regress -- and that half had no real-engine data, only simulation.
# `selector` asks a model who speaks next instead of advancing a position
# counter, so it is the free-form population on our own hardware.
HARNESS="${HARNESS:-autogen}"
TASK="${TASK:-design}"
case "$HARNESS" in
  autogen)  HARNESS_PY=.venv-ag/bin/python; HARNESS_SCRIPT=harness_roundrobin.py ;;
  selector) HARNESS_PY=.venv-ag/bin/python; HARNESS_SCRIPT=harness_selector.py ;;
  crewai)   HARNESS_PY=.venv-crewai/bin/python; HARNESS_SCRIPT=harness_crewai.py ;;
  *) echo "unknown harness $HARNESS (want autogen|selector|crewai)" >&2; exit 2 ;;
esac

cd "$(dirname "$0")"
mkdir -p traces "$OUTDIR"

# Two arms sharing one engine do not produce two measurements, they produce one
# contaminated measurement at twice the offered load, reported under the team
# count each arm was asked for. Guard here as well as in `e2e_arms.sh`, because
# a smoke test run by hand goes straight through this file.
exec 8>/tmp/e2e-autogen.lock
if ! flock -n 8; then
  echo "another e2e_autogen.sh is running; refusing to start" >&2
  ps -eo pid,lstart,args | grep -E "e2e_autoge[n].sh" >&2
  exit 3
fi

# The lock above is defeated by `rm -f /tmp/e2e-*.lock`, which is a natural
# thing to type after a run dies holding it -- two launches then lock two
# different inodes and neither refuses. That is how the lru arm of the
# p3000r8 run came to be measured under two fleets: one started 14:07, one
# 14:08:43, 1259 requests where 800 were asked for. Check for the thing that
# actually matters instead, which no amount of lock-file surgery can hide.
STRAY=$(pgrep -cf "harness_(roundrobin|selector|crewai)\.py" || true)
if [ "${STRAY:-0}" -gt 0 ]; then
  echo "$STRAY harness processes are still alive; refusing to start" >&2
  ps -eo pid,lstart,args | grep -E "harness_(roundrobin|selector|crewai)\.p[y]" >&2
  exit 3
fi

# The budget must be derived from the context this harness actually produces,
# not from the context the traces measured. The first run of this file used
# CONTEXT=9728 -- the Claude Code median -- and was wrong by 3.8x: AutoGen on
# the `design` task generates ~2,500 tokens a turn against Claude Code's 144,
# so the measured median prompt was 36,710 (p90 66,580, max 92,355). A 420,800
# token pool holds 11 such sessions while 60 were in flight, so the offered
# load was 5.5x rather than the intended 1.6x, and a quota of 2 per team asked
# the mechanism to protect 24 sessions in a pool that fits 11. That is the
# known wrong-quota regime where the group-aware arm loses to plain LRU, and it
# did: 0.34% against 0.41%.
#
# Two things follow. Cap the generation so the operating point is the one the
# analysis is about and every arm decodes the same amount, and size the budget
# from the run's own geometry: a member's prompt is the whole shared transcript
# plus its private system prompt, so it grows by one capped turn per turn of
# any member and peaks at the end of the run.
#
# A third correction, and the one that mattered most. Starting every member
# from an empty context makes the first half of a run uncontended: at 20x5 the
# whole fleet fits in the pool until round 7, LRU hits 84% there for free, and
# any bypassing policy can only lose by throwing those hits away. That warm
# phase is an artefact -- a real member's first prompt already carries its
# tools and the slice of the repo it owns, which is why the measured Claude
# Code contexts have a median of 9,728 rather than a ramp from 380. It also
# hid the retention rule completely: averaged across the fits/does-not-fit
# transition its own p_hit reads 0.55, so it never engaged and matched LRU to
# the last digit, while on the contended rounds of the same trace it engages
# and gains +24.3pp. PREAMBLE gives each member private context up front so
# the run is contended from its first turn.
PREAMBLE="${PREAMBLE:-0}"
GEN_CAP="${GEN_CAP:-512}"
SYS_TOKENS="${SYS_TOKENS:-256}"
RETAIN=$(( PREAMBLE + MEMBERS * ROUNDS * GEN_CAP + SYS_TOKENS ))
FIRST=$(( PREAMBLE + SYS_TOKENS ))
sessions=$(( TEAMS * MEMBERS ))
fits=$(( POOL / RETAIN )); (( fits < 1 )) && fits=1
fits_first=$(( POOL / FIRST )); (( fits_first < 1 )) && fits_first=1
budget_tok=$(( POOL - TEAMS * RETAIN ))
cap=$(( budget_tok / RETAIN )); (( cap < 1 )) && cap=1
# The formula sizes the quota to the peak footprint and is too conservative to
# be a fair baseline: on the 20x5x13 trace it returns 1, which is the one
# setting where the group-aware arm *loses* to LRU (-7.1pp), while a swept
# quota of 3 gains +17.0pp. Since the claim about the adaptive rule is that it
# needs no quota, the arm it has to beat is the baseline at its own best
# setting, chosen for this workload by `plan_workload.py` on the offline
# replayer -- which reproduced all three measured arms before being trusted.
quota=$(( cap / TEAMS )); (( quota < 1 )) && quota=1
quota="${QUOTA:-$quota}"

# Print the operating point, because the failure above was invisible in the
# log: every number it reported was self-consistent and wrong.
echo "operating point: preamble=${PREAMBLE} first_ctx=${FIRST} tok  " \
     "peak_ctx=${RETAIN} tok  pool=${POOL} tok  in_flight=${sessions}  " \
     "oversubscription first=$(( sessions * 100 / fits_first ))% " \
     "peak=$(( sessions * 100 / fits ))%  quota=${quota}"

case "$ARM" in
  lru)      MODE=group;    Q=0 ;;
  cohort)   MODE=group;    Q=$quota ;;
  adaptive) MODE=adaptive; Q=$quota ;;
  # No quota, and that is the claim rather than an omission: this arm is given
  # the pool size and nothing else, and derives who to keep from the reference
  # stream. Every other arm above has to be handed the answer computed from
  # the session footprint a few lines up.
  unified)  MODE=unified;  Q=0 ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac

cat >"$CTRL" <<EOF
{"mode": "$MODE", "quota": $Q, "global_cap": 0, "members": $MEMBERS,
 "capacity": $cap, "rotate_period": 0, "window_s": 30,
 "pool_tokens": $POOL, "adapt_threshold": 0.6, "adapt_window": 40}
EOF
# refresh() only runs on the cache path. A discarded 1-token call is what
# actually applies the new file; grepping docker logs is not a proof -- after
# a restart the last logged gen is leftover from the previous arm's file.
curl -s -m 45 "http://127.0.0.1:${ENGINE_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-4.7-flash","messages":[{"role":"user","content":"ok"}],"max_tokens":1}' \
  >/dev/null || true
sleep 1
echo "arm=$ARM mode=$MODE quota=$Q cap=$cap teams=$TEAMS members=$MEMBERS rounds=$ROUNDS"
tail -n 400 "${TC:-.}/logs/engine_vllm.log" \
  | grep '\[cohort\] gen=' | tail -1

# The harness and task are in the tag: an arm compared against one collected
# under a different population is not a comparison.
# The preamble and the round count belong in the tag. Without them two runs at
# the same team count overwrite each other's trace and result, which already
# cost the trace the whole offline analysis was built on.
SEED="${SEED:-0}"
TAG_SUFFIX="${TAG_SUFFIX:-}"
TAG="e2e-${HARNESS}-${TASK}-${ARM}-${TEAMS}t-p${PREAMBLE}r${ROUNDS}${TAG_SUFFIX}"
if [ "$SEED" != "0" ]; then
  TAG="${TAG}-s${SEED}"
fi
TRACE="traces/${TAG}.jsonl"
[ -e "$TRACE" ] && mv "$TRACE" "${TRACE}.$(date -u +%Y%m%dT%H%M%SZ).bak"

pkill -f "recording-proxy-${PORT}" 2>/dev/null
sleep 1
python3 spawn.py --log "/tmp/${TAG}-proxy.log" -- \
  python3 proxy_openai.py --listen-port "$PORT" --upstream-port "$ENGINE_PORT" \
  --trace "$TRACE" --inject-rid --tag "recording-proxy-${PORT}"
sleep 3

START=$(date +%s)
pids=()
for i in $(seq 1 "$TEAMS"); do
  "$HARNESS_PY" "$HARNESS_SCRIPT" \
    --base-url "http://127.0.0.1:${PORT}/v1" --team "team${i}" \
    --model "$MODEL" \
    --task "$TASK" --members "$MEMBERS" --rounds "$ROUNDS" \
    --max-tokens "$GEN_CAP" \
    --preamble-tokens "$PREAMBLE" \
    --seed "$SEED" \
    > "/tmp/${TAG}-team${i}.log" 2>&1 &
  pids+=($!)
done
echo "launched ${#pids[@]} teams at $(date -u +%H:%M:%S)"

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$(( fail + 1 )); done
END=$(date +%s)

echo "arm=$ARM wall=$(( END - START ))s failures=$fail"
SCORE_EXTRA=()
if [ "$HARNESS" = "selector" ]; then
  SCORE_EXTRA+=(--rid-re '^team[0-9]+:')
fi
python3 score_e2e.py "$TRACE" --arm "$ARM" --teams "$TEAMS" \
  --members "$MEMBERS" --rounds "$ROUNDS" \
  --wall "$(( END - START ))" --out "$OUTDIR/${TAG}.json" \
  "${SCORE_EXTRA[@]}"
python3 - "$OUTDIR/${TAG}.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
def line(tag, w):
    print(f"  {tag:<8} hit={w['prefix_hit_rate']*100:6.2f}%  "
          f"ceil={w['infinite_hit_rate']*100:5.1f}%  "
          f"headroom={w['headroom_pp']:+5.1f}pp  "
          f"req={w['requests']}  ttta_p50={w['ttta_p50']:.1f}s")
print(f"windows contended_from={d['contended_from_turn']} "
      f"steady_from={d['engaged_from_turn']}")
line("full", d["full"])
line("hot", d["hot"])
line("steady", d["steady"])
PY
