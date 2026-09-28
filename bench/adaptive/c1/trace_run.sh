#!/bin/bash
# C1 trace collection: one server, one workload at a time, per-workload trace
# segment boundaries recorded so the offline simulator can split the file.
#   usage: trace_run.sh <w4|w16> <outdir>
# Any failing step aborts the run. Only the server process group started here is ever signalled
# (also on abort, via the EXIT trap); other GPU processes are never killed -- the run refuses instead.
set -euo pipefail
PROFILE="$1"; OUT="$2"
REPO=~/tools/sglang-rtxpro6000; BENCH=~/tools/flash-next-bench
mkdir -p "$OUT" "$REPO/logs"
TRACE="$OUT/trace-$PROFILE.jsonl"
rm -f "$TRACE" "$TRACE.rank0" $BENCH/runs/c1trace-$PROFILE-*.jsonl
LOG=$REPO/logs/serve-c1-trace-$PROFILE.log

# We hold ~/.gpu.lock here. On 2026-09-06 13:07 a server of ours that was loading
# weights was caught by another script's teardown, which force-killed every GPU compute
# process (that script is not published), so wait for the GPU to be genuinely empty and settle before launching
# rather than racing the handover. Bounded, so a non-flock user cannot wedge us
# forever; if the GPU stays busy we refuse (other processes are never killed).
. "$(dirname "$(readlink -f "$0")")/../../lib/owned_group.sh"
trap owned_cleanup EXIT; trap 'exit 130' INT; trap 'exit 143' TERM
require_idle_gpu 1200 20

cd $REPO
start_owned $LOG env PYTHONPATH=$HOME/tools/sglang-c1/python PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  SGLANG_ADAPTIVE_TRACE="$TRACE" ./serve-fast.sh $PROFILE
for i in $(seq 1 90); do sleep 4; grep -q "ready to roll" $LOG && break
  if ! kill -0 $OWNED_PID 2>/dev/null; then echo "server died"; tail -30 $LOG; exit 1; fi; done
grep -q "ready to roll" $LOG || { echo "not ready"; tail -5 $LOG; exit 1; }   # EXIT trap stops the server group
PGID=$OWNED_PGID
echo "[c1-trace/$PROFILE] up $(date +%T) pgid=$PGID"

: > "$OUT/segments-$PROFILE.txt"
for W in code-edit prose-en prose-ja agent-loop; do
  cd $BENCH
  .venv-review/bin/python -m fnbench run --endpoint http://127.0.0.1:8001/v1 --engine sglang \
     --workloads $W --repeats 1 --sampling greedy --allow-proc sglang \
     --label c1trace-$PROFILE-$W --out runs/c1trace-$PROFILE-$W.jsonl 2>&1 | tail -3
  N=$(wc -l < "$TRACE.rank0" 2>/dev/null || echo 0)
  echo "$W $N" >> "$OUT/segments-$PROFILE.txt"
  echo "  [$W] trace lines so far: $N"
done

stop_owned 30
report_gpu_leftovers 20
echo "[c1-trace/$PROFILE] done $(date +%T); gpu: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
