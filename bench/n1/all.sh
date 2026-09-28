#!/bin/bash
# Everything still owed, in ONE lock acquisition: stage A end-to-end, then the stage B
# and stage C quality gates. The lock is contended for hours at a time, so splitting
# these into separate acquisitions risks them landing on different days.
set -uo pipefail
BENCH=$HOME/tools/flash-next-bench
if [ -z "${N1ALL_LOCKED:-}" ]; then
  echo "WAIT for lock $(date +%T)"
  exec flock -w 28800 "$HOME/.gpu.lock" env N1ALL_LOCKED=1 bash "$0" "$@"
fi
echo "LOCK acquired $(date +%T)"
export N1AB_LOCKED=1 N1Q_LOCKED=1     # already inside the lock; do not re-acquire
# Each part runs even if an earlier one fails, but every failure is listed and makes the exit status non-zero.
FAILED=()
part() { "$@" || FAILED+=("$* (rc $?)"); }
part env MODE=full bash $BENCH/n1/ab.sh a "w16 w4"
part bash $BENCH/n1/qual.sh b w4
part bash $BENCH/n1/qual.sh c w4
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "=== FAILED $(date +%T):"; printf '    %s\n' "${FAILED[@]}"; exit 1
fi
echo "=== ALL DONE $(date +%T)"
