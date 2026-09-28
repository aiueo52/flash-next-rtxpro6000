#!/bin/bash
# Caller MUST hold $HOME/.gpu.lock for the whole run.
set -euo pipefail
PROFILE=${1:?w4 or w16}; LABEL=${2:?unique label}
BENCH=$HOME/tools/flash-next-bench
REPO=$HOME/tools/sglang-p3
RUNDIR=$BENCH/runs/$LABEL-$PROFILE
mkdir "$RUNDIR"
PREFIX=$RUNDIR/recorder
LOG=$RUNDIR/server.log
SPID=
MPID=
cleanup() {
  trap - EXIT INT TERM
  if [ -n "$MPID" ]; then kill -TERM "$MPID" 2>/dev/null || true; wait "$MPID" 2>/dev/null || true; fi
  if [ -n "$SPID" ]; then
    kill -TERM -- -"$SPID" 2>/dev/null || true
    sleep 6
    kill -KILL -- -"$SPID" 2>/dev/null || true
    wait "$SPID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM
# Do not kill or interfere with compute applications that predate this run.
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo 'GPU has an existing compute application despite lock ownership; refusing start'
  exit 1
fi
export MEM_FRACTION=0.920 W16_MEM_FRACTION=0.920 WA_MEM_FRACTION=0.920 SERVE_DISPLAY_HZ= MAMBA_SLOTS=10 MAX_TOTAL_TOKENS=131072
export SGLANG_MOE_PRUNE_SINGLETON_TAU=0 FLASHINFER_MOE_PRUNE_SINGLETON_TAU=0 SGLANG_MOE_PRUNE_IN_PROLOGUE=1
export PYTHONPATH=$REPO/python:$BENCH/bench
export SGLANG_MOE_CONTRIB_LOG=$PREFIX SGLANG_MOE_CONTRIB_WIDTH=${PROFILE#w} SGLANG_MOE_CONTRIB_RING=262144
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
# The copies resolve their interpreter to the read-only production venv, while
# logs and launcher-local caches resolve inside the P3 worktree.
setsid "$REPO/serve-fast.sh" "$PROFILE" > "$LOG" 2>&1 < /dev/null &
SPID=$!
for i in $(seq 1 225); do
  sleep 4
  if rg -q 'ready to roll' "$LOG"; then break; fi
  kill -0 "$SPID" 2>/dev/null || { tail -40 "$LOG"; exit 1; }
done
rg -q 'ready to roll' "$LOG" || { tail -40 "$LOG"; exit 1; }
rg -q '\[p3-contrib\] ready' "$LOG" || { echo 'P3 recorder did not arm'; exit 1; }
# Loading transients have ended. Enforce free VRAM only now and after workloads.
check_memory() {
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  echo "steady free MiB=$free $(date -Is)" | tee -a "$RUNDIR/memory.log"
  [ "$free" -ge 4096 ] || { echo 'steady VRAM free below 4 GiB'; exit 1; }
}
check_memory
python3 "$BENCH/prof/p3_memory_watch.py" "$SPID" "$RUNDIR/memory-continuous.log" "$RUNDIR/memory-failed.txt" &
MPID=$!
for WORKLOAD in code-edit prose-en prose-ja agent-loop; do
  python3 "$BENCH/prof/p3_control.py" "$PREFIX" reset
  OUTPUT=$BENCH/runs/census-$LABEL-$PROFILE-$WORKLOAD
  [ ! -e "$OUTPUT.jsonl" ] && [ ! -e "$OUTPUT.npz" ]
  (cd "$BENCH" && .venv-review/bin/python -m fnbench run \
    --endpoint http://127.0.0.1:8001/v1 --engine sglang --workloads "$WORKLOAD" \
    --repeats 1 --sampling greedy --allow-proc sglang \
    --label "census-$LABEL-$PROFILE-$WORKLOAD" --out "$OUTPUT.jsonl") > "$RUNDIR/$WORKLOAD.client.log" 2>&1
  python3 "$BENCH/prof/p3_control.py" "$PREFIX" dump --path "$OUTPUT.npz" | tee "$RUNDIR/$WORKLOAD.capture.json"
  check_memory
  [ ! -e "$RUNDIR/memory-failed.txt" ]
  python3 - "$BENCH/specs/P3_CONTRIB_PRUNE.md" "$RUNDIR/$WORKLOAD.capture.json" "$PROFILE" "$WORKLOAD" <<'PYREPORT'
import json,sys
r=json.load(open(sys.argv[2]))
with open(sys.argv[1],'a') as f:
    f.write(f"\nCapture landed: {sys.argv[3]} / {sys.argv[4]}: `{r['path']}`, {r['stored_calls']} target-layer calls; layers {r['layers'][0]}–{r['layers'][-1]}; unchanged={r['unchanged']}; max reconstruction relative L2={r['max_reconstruction_rel']:.6f}.\n")
PYREPORT
  echo "P3 capture complete: $OUTPUT.npz $(date -Is)"
done
