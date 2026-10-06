#!/usr/bin/env bash
# Does a detached child keep its caller's flock alive after the caller exits?
#
# This is not a hypothetical. `e2e_arms.sh` and `e2e_autogen.sh` take an flock
# to stop a second copy from doubling the offered load, and they launch the
# recording proxy through `spawn.py`, which detaches it deliberately. Before
# `spawn.py` closed inherited descriptors, the proxy held the lock after its
# arm exited, so the first arm of a three-arm run succeeded and the other two
# were refused by a lock held by the first arm's own proxy. The failure mode
# reads as "another instance is running" when nothing is, and it cost a run.
#
#   ssh b300 'cd /opt/dlami/nvme/pat && ./selftest_spawn.sh'
#
set -uo pipefail
cd "$(dirname "$0")"

LOCK=/tmp/spawn-selftest.lock
rm -f "$LOCK"

# The holder has to be a subshell, not a function: `exec 8>` applies to the
# shell that runs it, so a function would pin the descriptor open for this
# script's whole life and the test would report FAIL for itself rather than
# for the detached child.
(
  exec 8>"$LOCK"
  flock -n 8 || { echo "setup failed: could not take a fresh lock" >&2; exit 1; }
  python3 spawn.py --log /tmp/spawn-selftest-child.log -- sleep 90
) || { echo "setup failed"; exit 1; }
sleep 2

if ! pgrep -f "slee[p] 90" >/dev/null; then
  echo "INCONCLUSIVE: the detached child is not running, so nothing was tested"
  rm -f "$LOCK"; exit 2
fi

# The holder function has returned, so the only candidate holder left is the
# detached child. Taking the lock now is exactly what the next arm does.
if flock -n "$LOCK" true; then
  echo "PASS: lock released once the caller exited; the detached child did"
  echo "      not inherit it, so the next arm can start"
  rc=0
else
  echo "FAIL: lock still held while the caller is gone -- the detached child"
  echo "      inherited the descriptor and will refuse every later arm"
  fuser -v "$LOCK" 2>&1 | head -5
  rc=1
fi

pkill -f "slee[p] 90" 2>/dev/null
rm -f "$LOCK"
exit $rc
