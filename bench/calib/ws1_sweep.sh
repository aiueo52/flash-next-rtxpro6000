#!/bin/bash
# WS1 fixed-width sweep driver: W4, W6, W8, W12, W16, W4-repeat; each arm under the GPU lock.
# A failed arm does not stop the sweep, but it is listed at the end and makes the exit status non-zero.
# runs/<label>.done is written only for arms whose ws1_validate.sh run succeeded; ws1_report.py reports
# only those arms. The label prefix must be fresh: the sweep refuses if outputs for it already exist.
set -uo pipefail
cd ~/tools/flash-next-bench || exit 1
T=$(date +%H%M)
if compgen -G "runs/ws1$T-*" > /dev/null; then
  echo "outputs for label prefix ws1$T already exist in runs/; refusing to mix runs (retry in a minute)" >&2; exit 1
fi
FAILED=(); OK=()
run() { # name profile [server args...]
  local name=$1 prof=$2; shift 2
  echo "=== arm $name ($(date +%T)) args: $*"
  if SERVER_ARGS="$*" flock -w 28800 ~/.gpu.lock ./calib/ws1_validate.sh ws1$T-$name $prof; then
    date --iso-8601=seconds > "runs/ws1$T-$name-$prof.done"; OK+=("$name")
  else
    FAILED+=("$name (rc $?)")
  fi
}
run w4  w4
run w6  w4 --speculative-num-steps 5  --speculative-num-draft-tokens 6
run w8  w4 --speculative-num-steps 7  --speculative-num-draft-tokens 8
run w12 w4 --speculative-num-steps 11 --speculative-num-draft-tokens 12
run w16 w16
run w4b w4
echo "=== sweep done ($(date +%T)) label prefix ws1$T; complete: ${OK[*]:-none}"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "=== FAILED arms (excluded by ws1_report.py): ${FAILED[*]}"; exit 1
fi
