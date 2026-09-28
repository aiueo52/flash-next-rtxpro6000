#!/bin/bash
# run_arm.sh <arm> [ENV...] -- start serve-fast.sh wa with the given env, run all Q1 benchmarks, stop. Wrap in flock.
# Any failing step aborts the arm. Only the server process group started here is ever signalled (also
# on abort, via the EXIT trap); other GPU processes are never killed -- if they remain, the arm refuses.
# Outputs go to a fresh runs/<arm> (an existing one is moved to runs/stale/<run id>/ first, so partial or old
# results never mix in); runs/<arm>/DONE, carrying the run id (Q1_RUN_ID from run_all.sh, else a new one), is
# written only after the whole arm succeeded. analyze.py / write_report.py ignore arms without it.
set -euo pipefail
ARM="$1"; shift
export BS="${BS:-1}"
[[ "$BS" =~ ^[1-9][0-9]*$ ]] || { echo "BS must be a positive integer" >&2; exit 2; }
REPO=$HOME/tools/sglang-rtxpro6000; PY=$REPO/.venv/bin/python; HERE=$(cd "$(dirname "$0")" && pwd)
BENCH=$HERE/../..; LOG=$REPO/logs/serve-q1-$ARM.log; OUT=$HERE/runs/$ARM
RUN_ID=${Q1_RUN_ID:-$(date +%Y%m%dT%H%M%S)-$$}
for d in "$OUT" "$OUT-smoke"; do
  if [ -e "$d" ]; then mkdir -p "$HERE/runs/stale/$RUN_ID"; mv "$d" "$HERE/runs/stale/$RUN_ID/"; fi
done
mkdir -p $OUT
. "$BENCH/lib/owned_group.sh"
trap owned_cleanup EXIT; trap 'exit 130' INT; trap 'exit 143' TERM
echo "[$ARM] start $(date +%T) BS=$BS run $RUN_ID env: $*" | tee -a $OUT/arm.log
cd $REPO
start_server() {
  # wait for the previous GPU user to release memory (desktop alone is ~5.7 GB)
  # other agents run short GPU jobs outside the lock; wait until none is visible for ~15 s
  for i in $(seq 1 40); do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); P=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    if [ "$U" -lt 8000 ] && [ "$P" -eq 0 ]; then sleep 15; [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ] && break; else sleep 4; fi; done
  # never kill another user's GPU job: refuse instead (exit, not return, so the retries below stop too)
  require_idle_gpu 0 0 2>&1 | tee -a $OUT/arm.log || exit 1
  echo "[$ARM] gpu mem before start: ${U} MiB" | tee -a $OUT/arm.log
  # 2026-09-07 00:40: with the native caching allocator the load leaves ~9 GB of inactive split blocks (reserved 86.9 vs allocated 77.7 GB
  # after the draft load; diag via torch.cuda.memory_stats), so wa/0.925 (and w4/0.935) fail the KV-pool check. expandable_segments is an
  # allocator-only setting (no effect on kernels/results); applied identically to every arm.
  start_owned $LOG env PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" "$@" ./serve-fast.sh wa --max-running-requests "$BS" || return 1
  for i in $(seq 1 150); do sleep 4; grep -q "ready to roll" $LOG && return 0; if ! kill -0 $OWNED_PID 2>/dev/null; then echo "[$ARM] server died" | tee -a $OUT/arm.log; tail -3 $LOG | tee -a $OUT/arm.log; stop_owned || exit 1; return 1; fi; done
  echo "[$ARM] server not ready after 10min" | tee -a $OUT/arm.log; stop_owned || exit 1; return 1
}
OK=0; for try in 1 2 3 4; do start_server "$@" && { OK=1; break; }; cp $LOG $LOG.try$try; sleep 45; done; [ $OK = 1 ] || exit 1
PGID=$OWNED_PGID
echo "[$ARM] server up $(date +%T) pgid $PGID; free $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)" | tee -a $OUT/arm.log
T0=$(date +%s)
for depth in 0.1 0.5 0.9; do $PY $BENCH/prof/needle_test.py 18500 $depth q1-$ARM-d$depth 2>&1 | tee -a $OUT/needle.log; done
# run_bench.py exits non-zero on any request/item error; pipefail carries that through tee.
if ! LIMIT=3 $PY $HERE/run_bench.py $ARM-smoke 2>&1 | tee -a $OUT/arm.log; then
  echo "[$ARM] SMOKE FAILED (run_bench.py error), aborting arm" | tee -a $OUT/arm.log; exit 2; fi
if ! $PY -c "import json,sys; s=json.load(open('$HERE/runs/$ARM-smoke/summary.json'))['bench']; sys.exit(0 if all(v.get('errors',0)==0 for v in s.values()) and all(v.get('mean_gen_tokens',1)>0 for v in s.values()) else 1)"; then
  echo "[$ARM] SMOKE FAILED, aborting arm" | tee -a $OUT/arm.log; exit 2; fi   # the EXIT trap stops the server group
if ! $PY $HERE/run_bench.py $ARM 2>&1 | tee -a $OUT/arm.log; then
  echo "[$ARM] BENCHMARKS FAILED (run_bench.py error; see summary.json), aborting arm without DONE" | tee -a $OUT/arm.log; exit 3; fi
for depth in 0.1 0.5 0.9; do $PY $BENCH/prof/needle_test.py 18500 $depth q1-$ARM-d$depth-post 2>&1 | tee -a $OUT/needle.log; done
echo "[$ARM] benchmarks done $(date +%T) total $(( $(date +%s) - T0 ))s" | tee -a $OUT/arm.log
cp $LOG $OUT/server.log
stop_owned 30
report_gpu_leftovers 20 2>&1 | tee -a $OUT/arm.log
echo "[$ARM] stopped $(date +%T); gpu procs: '$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')' mem $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" | tee -a $OUT/arm.log
# completion marker: only reached when every step above succeeded (set -e), and only if the summary itself
# records a finished run without errors (defence in depth; analyze.py checks the same)
$PY -c "import json,sys; s=json.load(open('$OUT/summary.json')); sys.exit(0 if s.get('finished') and not s.get('incomplete', True) and not s.get('errors') else 1)" \
  || { echo "[$ARM] summary.json is incomplete or records errors; no DONE marker" | tee -a $OUT/arm.log; exit 4; }
printf '{"run_id": "%s", "arm": "%s", "finished": "%s"}\n' "$RUN_ID" "$ARM" "$(date --iso-8601=seconds)" > $OUT/DONE.tmp
mv $OUT/DONE.tmp $OUT/DONE
