#!/bin/bash
# C1 A/B: shipped EMA config vs the confidence policy, same hour, same server args.
#   usage: ab_run.sh <label> <config.json> [POLICY]   POLICY=ema|confidence
# Any failing step aborts the run. Only the server process group started here is ever signalled
# (also on abort, via the EXIT trap); other GPU processes are never killed -- the run refuses instead.
set -euo pipefail
LABEL="$1"; CFG="$2"; POLICY="${3:-ema}"
REPO=~/tools/sglang-rtxpro6000; BENCH=~/tools/flash-next-bench; PY=$REPO/.venv/bin/python
LOG=$REPO/logs/serve-c1-$LABEL.log
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
  SGLANG_ADAPTIVE_POLICY=$POLICY ./serve-fast.sh wa \
  --speculative-adaptive-config "$CFG"
for i in $(seq 1 90); do sleep 4; grep -q "ready to roll" $LOG && break
  if ! kill -0 $OWNED_PID 2>/dev/null; then echo "server died"; tail -30 $LOG; exit 1; fi; done
grep -q "ready to roll" $LOG || { echo "not ready"; tail -8 $LOG; exit 1; }   # EXIT trap stops the server group
echo "[$LABEL] up $(date +%T) policy=$POLICY cfg=$CFG"
{ grep -m1 "free gpu memory\|Free GPU memory\|avail mem" $LOG || true; } | tail -1
cd $BENCH
.venv-review/bin/python -m fnbench run --endpoint http://127.0.0.1:8001/v1 --engine sglang \
   --workloads code-edit,prose-en,prose-ja,agent-loop --repeats 2 --sampling greedy \
   --allow-proc sglang --label $LABEL --out runs/$LABEL.jsonl 2>&1 | tail -8
$PY $BENCH/prof/needle_test.py 18500 0.4 $LABEL
SW=$(grep -c "Adaptive spec params updated" $LOG) || true   # grep -c prints 0 but exits 1 on no match
echo "[$LABEL] switches during bench: $SW"
{ grep "Adaptive spec params updated" $LOG || true; } | tail -12
stop_owned 30
report_gpu_leftovers 20
echo "[$LABEL] done $(date +%T); gpu: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
