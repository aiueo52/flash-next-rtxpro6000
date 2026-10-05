#!/bin/bash
# fc_measure.sh <label> <profile> -- against a server already held by fc_hold.sh (PORT, default 8011):
# warm-up, then per workload greedy and sampling arms (N requests each, the first one carries a
# 20-step profiler trace and is excluded from the ms/step median). Writes prof/logs/<label>-<profile>.jsonl
# and traces prof/traces/<label>-<profile>-<workload>-<mode>/.
set -uo pipefail
LABEL="$1"; PROFILE="$2"; PORT="${PORT:-8011}"; N="${N:-3}"; WORKLOADS="${WORKLOADS:-code-edit prose-en}"
MODES="${MODES:-greedy sampling}"; PROF_STEPS="${PROF_STEPS:-20}"
BENCH=~/tools/flash-next-bench; PY=~/tools/sglang-rtxpro6000/.venv/bin/python
JS=$BENCH/prof/logs/$LABEL-$PROFILE.jsonl
$PY $BENCH/prof/fc_sampling_probe.py --port $PORT --mode greedy --workload $BENCH/workloads/prose-en.txt --n 1 --max-tokens 200 >/dev/null 2>&1
for W in $WORKLOADS; do
  for M in $MODES; do
    TD=$BENCH/prof/traces/$LABEL-$PROFILE-$W-$M; rm -rf $TD
    if [ "$PROF_STEPS" -gt 0 ]; then PA="--profile-dir $TD --profile-steps $PROF_STEPS"; else PA=""; fi
    $PY $BENCH/prof/fc_sampling_probe.py --port $PORT --mode $M --workload $BENCH/workloads/$W.txt \
        --n $N $PA --json-out $JS
  done
done
