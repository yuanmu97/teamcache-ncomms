#!/usr/bin/env bash
# Three policies against real AutoGen teams, one engine boot, back to back.
#
# The operating point is chosen so the experiment can say anything at all:
#
#   12 teams x 5 members = 60 sessions against a pool that holds about 38, so
#   1.6x oversubscribed -- the same working point the topology grid reports at.
#   Below 1.0x every policy is correct and the run measures nothing.
#
#   6 rounds, because `unified` needs three references to a member before it
#   has two samples of whether LRU held it. At 3 rounds the rule is still
#   measuring when the run ends, which is an honest property of the mechanism
#   but makes for a dishonest experiment.
#
# Arms run in one engine boot so cache state and autotuning are shared; the
# order is fixed rather than randomised because each arm starts from whatever
# the previous one left resident, and lru first is the conservative choice --
# it leaves the cache full, which is the least favourable start for the arms
# that follow.
set -uo pipefail
cd "$(dirname "$0")"

# A second concurrent instance does not just waste the machine, it silently
# doubles the offered load and the run still reports the team count it was
# asked for. We lost the first lru measurement to exactly this: two copies
# started 56s apart, so 12 teams were measured under 24 teams of pressure, and
# because `spawn.py` opens its log with O_TRUNC the evidence was a truncated
# log with two different wall times rather than an error. Refuse instead.
exec 9>/tmp/e2e-arms.lock
if ! flock -n 9; then
  echo "another e2e_arms.sh holds /tmp/e2e-arms.lock; refusing to start" >&2
  ps -eo pid,lstart,args | grep -E "e2e_arm[s].sh" >&2
  exit 3
fi

TEAMS="${TEAMS:-12}"
MEMBERS="${MEMBERS:-5}"
ROUNDS="${ROUNDS:-6}"
ARMS="${ARMS:-lru cohort unified}"

for arm in $ARMS; do
  echo "######## $arm  teams=$TEAMS rounds=$ROUNDS  $(date -u +%H:%M:%S) ########"
  ./e2e_autogen.sh "$arm" "$TEAMS" "$MEMBERS" "$ROUNDS" 2>&1
  echo
  # The engine keeps whatever the arm left resident, and a policy change
  # resets the mechanism's own state but not the radix tree. Let the next arm
  # start from a settled pool rather than mid-eviction.
  sleep 20
done
echo "ALL ARMS DONE $(date -u +%H:%M:%S)"
