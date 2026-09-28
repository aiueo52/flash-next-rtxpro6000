#!/bin/bash
# validate.sh <label> <w4|w16> [MODE=prof|full] -- start serve-fast.sh, profile, bench, needle, stop.
# Extra env for the server can be passed via SERVER_ENV="A=1 B=2".
# Any failing step aborts the run. Only the server process group started here is ever signalled
# (also on abort, via the EXIT trap); other GPU processes are never killed -- the run refuses instead.
set -euo pipefail
LABEL="$1"; PROFILE="$2"; MODE="${MODE:-full}"
REPO=~/tools/sglang-rtxpro6000; BENCH=~/tools/flash-next-bench; PY=$REPO/.venv/bin/python
PROF=$BENCH/prof; LOG=$REPO/logs/serve-$LABEL-$PROFILE.log
. "$(dirname "$(readlink -f "$0")")/../lib/owned_group.sh"
trap owned_cleanup EXIT; trap 'exit 130' INT; trap 'exit 143' TERM
require_idle_gpu 120
cd $REPO
start_owned $LOG env ${SERVER_ENV:-} ./serve-fast.sh $PROFILE ${SERVER_ARGS:-}
for i in $(seq 1 225); do sleep 4; grep -q "ready to roll" $LOG && break; if ! kill -0 $OWNED_PID 2>/dev/null; then echo "server died"; tail -20 $LOG; exit 1; fi; done
grep -q "ready to roll" $LOG || { echo "server not ready after 15min"; tail -5 $LOG; exit 1; }   # EXIT trap stops the server group
echo "[$LABEL/$PROFILE] server up ($(date +%T))"
mkdir -p $PROF/traces
for W in code-edit prose-en; do
  OUT=$PROF/traces/$LABEL-$PROFILE-$W; rm -rf $OUT
  $PY $PROF/profile_decode2.py $OUT 20 $BENCH/workloads/$W.txt 2>&1 | { grep -v "^files" || true; } | tail -2
done
$PY $PROF/trimmed_step.py $PROF/traces/$LABEL-$PROFILE-code-edit/*.gz $PROF/traces/$LABEL-$PROFILE-prose-en/*.gz
if [ "$MODE" = "full" ]; then
  cd $BENCH && .venv-review/bin/python -m fnbench run --endpoint http://127.0.0.1:8001/v1 --engine sglang \
     --workloads code-edit,prose-en,agent-loop --repeats 2 --sampling greedy --allow-proc sglang \
     --label $LABEL-$PROFILE --out runs/$LABEL-$PROFILE.jsonl 2>&1 | tail -12
  $PY $PROF/needle_test.py 18500 0.4 $LABEL-$PROFILE
fi
stop_owned 30
report_gpu_leftovers 20
echo "[$LABEL/$PROFILE] done ($(date +%T)); gpu mem: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
