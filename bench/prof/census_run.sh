#!/bin/bash
# census_run.sh <w4|w16> [ring] -- P1 route census.
# One server with route_census armed via sitecustomize, fnbench once per
# workload, a ring snapshot copied after each.  The ring holds the last N calls,
# so each snapshot is that workload's traffic and no differencing is needed.
# Run under flock; the caller owns the GPU lock.
# Any failing step aborts the run. Only the server process group started here is ever signalled
# (also on abort, via the EXIT trap); other GPU processes are never killed -- the run refuses instead.
set -euo pipefail
PROFILE="${1:-w16}"; RING="${2:-16384}"; TAG="${TAG:-}"
REPO=~/tools/sglang-rtxpro6000; BENCH=~/tools/flash-next-bench
HOOK=/tmp/moe-census-hook; PREFIX=/tmp/census$TAG-$PROFILE
mkdir -p "$HOOK" "$BENCH/runs" "$REPO/logs"
cp "$BENCH/bench/moe_smallm/sitecustomize_census.py" "$HOOK/sitecustomize.py"
rm -f "$PREFIX".*
LOG=$REPO/logs/serve-census$TAG-$PROFILE.log
. "$(dirname "$(readlink -f "$0")")/../lib/owned_group.sh"
trap owned_cleanup EXIT; trap 'exit 130' INT; trap 'exit 143' TERM
require_idle_gpu 120
cd "$REPO"
# SERVER_ENV may carry its own PYTHONPATH (e.g. the p1 worktree); the census
# hook and the bench package are prepended to it.
EXTRA=""; PP=""
for kv in ${SERVER_ENV:-}; do
  case "$kv" in PYTHONPATH=*) PP="${kv#PYTHONPATH=}" ;; *) EXTRA="$EXTRA $kv" ;; esac
done
PP="$HOOK${PP:+:$PP}:$BENCH/bench"
echo "[census/$PROFILE] extra env:$EXTRA PYTHONPATH=$PP"
start_owned "$LOG" env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $EXTRA \
    PYTHONPATH="$PP" \
    SGLANG_MOE_CENSUS_LOG="$PREFIX" SGLANG_MOE_CENSUS_RING="$RING" \
    SGLANG_MOE_CENSUS_AUTODUMP=10 \
    ./serve-fast.sh "$PROFILE"
for i in $(seq 1 90); do
  sleep 4; grep -q "ready to roll" "$LOG" && break
  kill -0 $OWNED_PID 2>/dev/null || { echo "server died"; tail -20 "$LOG"; exit 1; }
done
grep -q "ready to roll" "$LOG" || { echo "not ready"; tail -5 "$LOG"; exit 1; }   # EXIT trap stops the server group
grep -m1 "route_census" "$LOG" || echo "WARNING: route_census never announced itself"
echo "[census/$PROFILE] server up ($(date +%T))"
for W in code-edit prose-en agent-loop prose-ja; do
  (cd "$BENCH" && .venv-review/bin/python -m fnbench run \
     --endpoint http://127.0.0.1:8001/v1 --engine sglang --workloads "$W" \
     --repeats 1 --sampling greedy --allow-proc sglang \
     --label "census$TAG-$PROFILE-$W" --out "runs/census$TAG-$PROFILE-$W.jsonl" 2>&1 | tail -3)
  sleep 15                       # let the 10 s autodump fire after the last call
  NPZ=$(ls -S "$PREFIX".*.npz 2>/dev/null | grep -v "\.tmp\." | head -1) || NPZ=""
  JSN=$(ls -S "$PREFIX".*.json 2>/dev/null | head -1) || JSN=""
  if [ -n "$NPZ" ]; then cp "$NPZ" "$BENCH/runs/census$TAG-$PROFILE-$W.npz"
    echo "  snapshot $(du -h "$NPZ" | cut -f1) -> runs/census$TAG-$PROFILE-$W.npz"; fi
  if [ -n "$JSN" ]; then cp "$JSN" "$BENCH/runs/census$TAG-$PROFILE-$W.json"; fi
done
stop_owned 30
report_gpu_leftovers 20
echo "[census/$PROFILE] done ($(date +%T)); gpu mem: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
