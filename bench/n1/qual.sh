#!/bin/bash
# N1 quality gate for the stages that change target outputs (B and C).
# Usage: qual.sh <stage b|c> <profile w4|w16>
#
# Runs the control and the arm back to back in one lock acquisition. For each, with the
# server up: step-time profile, greedy generations for token agreement, fnbench
# (acceptance + t/s), and the 18.5k needle. (The 12-task battery that was also run here
# is not published: its questions came from a set of unshown origin.) Modelled on prof/validate.sh
# but keeping the server alive long enough for the extra probes, which validate.sh
# cannot do because it owns the server lifecycle.
#
# The comparison that matters is arm-vs-control *within this run*; the within-build
# repeat agreement that agree_cmp prints is the noise floor and must be read first,
# because speculative decoding plus non-deterministic reductions make even greedy runs
# differ from themselves.
#
# Any failing step aborts the gate. Only the server process group started here is ever
# signalled (also on abort, via the EXIT trap); other GPU processes are never killed --
# the run refuses instead.
set -euo pipefail
STAGE="${1:?b|c}"; PROFILE="${2:?w4|w16}"
REPO=$HOME/tools/sglang-rtxpro6000; BENCH=$HOME/tools/flash-next-bench
PY=$REPO/.venv/bin/python; PROF=$BENCH/prof; N1PY=$HOME/tools/sglang-n1/python
R=$BENCH/n1/qual; mkdir -p "$R"

case "$STAGE" in
  b) FLAG="SGLANG_LMHEAD_NVFP4=1" ;;
  c) FLAG="SGLANG_LINEAR_ATTN_NVFP4=1 SGLANG_ATTN_NVFP4=1" ;;
  *) echo "usage: $0 {b|c} {w4|w16}" >&2; exit 2 ;;
esac

if [ -z "${N1Q_LOCKED:-}" ]; then
  echo "WAIT for lock $(date +%T) gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  exec flock -w 28800 "$HOME/.gpu.lock" env N1Q_LOCKED=1 bash "$0" "$@"
fi
. "$(dirname "$(readlink -f "$0")")/../lib/owned_group.sh"
trap owned_cleanup EXIT; trap 'exit 130' INT; trap 'exit 143' TERM
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
[ "$FREE" -lt 40000 ] && { echo "ABORT: only ${FREE}MiB free"; exit 3; }
echo "LOCK acquired $(date +%T), ${FREE}MiB free; stage=$STAGE profile=$PROFILE"

one() {  # one <label> <server_env>
  # Three separate `local`s on purpose: `local a=$1 b=$REPO/x-$a` does NOT see `a` while
  # the same statement is still being evaluated, so under `set -u` it aborts with
  # "label: unbound variable" -- which is exactly how the first stage B/C run died.
  local label="$1"
  local senv="$2"
  local log=$REPO/logs/serve-$label-$PROFILE.log
  echo "=== $label/$PROFILE start $(date +%T)"
  require_idle_gpu 120
  cd $REPO
  start_owned $log env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$N1PY $senv \
    ./serve-fast.sh $PROFILE
  for i in $(seq 1 90); do
    sleep 4; grep -q "ready to roll" $log && break
    kill -0 $OWNED_PID 2>/dev/null || { echo "server died"; tail -25 $log; return 1; }
  done
  grep -q "ready to roll" $log || { echo "not ready in 6min"; tail -5 $log; return 1; }   # EXIT trap stops the group
  echo "[$label/$PROFILE] up $(date +%T)"
  { grep -iE "NVFP4|nvfp4" $log || true; } | head -4        # confirm the flag actually took effect

  for W in code-edit prose-en; do
    local out=$PROF/traces/$label-$PROFILE-$W; rm -rf $out
    $PY $PROF/profile_decode2.py $out 20 $BENCH/workloads/$W.txt 2>&1 | tail -1
  done
  $PY $PROF/trimmed_step.py $PROF/traces/$label-$PROFILE-code-edit/*.gz \
      $PROF/traces/$label-$PROFILE-prose-en/*.gz | tee "$R/$label-$PROFILE.step.txt"

  $PY $PROF/agree_gen.py "$R/$label-$PROFILE.agree.json" 320 2 2>&1 | tail -2
  cd $BENCH && .venv-review/bin/python -m fnbench run --endpoint http://127.0.0.1:8001/v1 \
    --engine sglang --workloads code-edit,prose-en,agent-loop --repeats 2 \
    --sampling greedy --allow-proc sglang --label $label-$PROFILE \
    --out runs/$label-$PROFILE.jsonl 2>&1 | tail -10
  $PY $PROF/needle_test.py 18500 0.4 $label-$PROFILE 2>&1 | tail -3

  stop_owned 30
  report_gpu_leftovers 20
  echo "[$label/$PROFILE] done $(date +%T)"
}

# Plain calls (no `|| exit`): errexit stays active inside one(), so any failing step aborts.
one "n1${STAGE}q-off" ""
one "n1${STAGE}q-on"  "$FLAG"

echo "=== agreement (arm vs control; within-build repeats are the noise floor)"
$PY $PROF/agree_cmp.py "$R/n1${STAGE}q-off-$PROFILE.agree.json" \
                       "$R/n1${STAGE}q-on-$PROFILE.agree.json" 256 | tee "$R/agree-$STAGE-$PROFILE.txt"
echo "=== N1 quality gate done $(date +%T)"
