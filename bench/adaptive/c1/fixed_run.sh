#!/bin/bash
# same-window fixed-profile references for the C1 A/B
# Any failing step aborts the run. Only the server process group started here is ever signalled
# (also on abort, via the EXIT trap); other GPU processes are never killed -- the run refuses instead.
set -euo pipefail
PROFILE="$1"; LABEL="c1fix-$PROFILE"
REPO=~/tools/sglang-rtxpro6000; BENCH=~/tools/flash-next-bench; PY=$REPO/.venv/bin/python
LOG=$REPO/logs/serve-$LABEL.log
. "$(dirname "$(readlink -f "$0")")/../../lib/owned_group.sh"
trap owned_cleanup EXIT; trap 'exit 130' INT; trap 'exit 143' TERM
require_idle_gpu 1200 20
cd $REPO
start_owned $LOG env PYTHONPATH=$HOME/tools/sglang-c1/python PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ./serve-fast.sh $PROFILE
for i in $(seq 1 90); do sleep 4; grep -q "ready to roll" $LOG && break
  kill -0 $OWNED_PID 2>/dev/null || { echo died; tail -20 $LOG; exit 1; }; done
grep -q "ready to roll" $LOG || { echo "not ready"; tail -5 $LOG; exit 1; }   # EXIT trap stops the server group
echo "[$LABEL] up $(date +%T)"
cd $BENCH && rm -f runs/$LABEL.jsonl
.venv-review/bin/python -m fnbench run --endpoint http://127.0.0.1:8001/v1 --engine sglang \
   --workloads code-edit,prose-en,prose-ja,agent-loop --repeats 2 --sampling greedy \
   --allow-proc sglang --label $LABEL --out runs/$LABEL.jsonl 2>&1 | tail -4
stop_owned 30
report_gpu_leftovers 20
echo "[$LABEL] done $(date +%T)"
