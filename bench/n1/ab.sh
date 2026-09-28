#!/bin/bash
# N1 server A/B: <stage> <profile> [MODE]
#
#   ab.sh a w16 prof     stage A (draft lm_head NVFP4), W16, step time only
#   ab.sh b w4 full      stage B (target lm_head NVFP4), W4, + fnbench + needle
#
# Runs the flag-OFF control and the flag-ON arm back to back in the same lock
# acquisition, so they share a clock hour, a driver state and a thermal state. BOTH
# arms run with PYTHONPATH pointing at the N1 worktree: the only difference between
# them is the env flag, otherwise the comparison would also be measuring whatever else
# differs between the two checkouts.
#
# serve-fast.sh is NOT modified. Stage B/C drop lm_head / linear_attn / attn from
# --qwen4-exp-dense-fp8 by re-passing the flag (later wins), so the target weight is
# still BF16 when the NVFP4 packer reads it and no double quantisation happens.
set -uo pipefail
STAGE="${1:?stage a|b|c}"; PROFILES="${2:?w4|w16, space separated}"; MODE="${MODE:-${3:-prof}}"
BENCH=$HOME/tools/flash-next-bench
N1PY=$HOME/tools/sglang-n1/python
R=$BENCH/n1/runs; mkdir -p "$R"

case "$STAGE" in
  a) FLAG="SGLANG_MTP_LMHEAD_NVFP4=1"; EXTRA_ARGS="" ;;
  b) FLAG="SGLANG_LMHEAD_NVFP4=1"; EXTRA_ARGS="" ;;   # lm_head stays in the FP8 list on
     # purpose: the control must be the shipped build, and the stage B hook packs from
     # the checkpoint shard anyway, so the FP8 head it replaces costs only startup time.
  c) FLAG="SGLANG_LINEAR_ATTN_NVFP4=1 SGLANG_ATTN_NVFP4=1"
     EXTRA_ARGS="--qwen4-exp-dense-fp8 shared_expert,lm_head,mtp_dense" ;;
  *) echo "usage: $0 {a|b|c} {w4|w16} [prof|full]" >&2; exit 2 ;;
esac

if [ -z "${N1AB_LOCKED:-}" ]; then
  echo "WAIT for lock $(date +%T) gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  exec flock -w 28800 "$HOME/.gpu.lock" env N1AB_LOCKED=1 bash "$0" "$@"
fi
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
[ "$FREE" -lt 40000 ] && { echo "ABORT: only ${FREE}MiB free"; exit 3; }
echo "LOCK acquired $(date +%T), ${FREE}MiB free; stage=$STAGE profiles='$PROFILES' mode=$MODE"

run() {  # run <label> <profile> <server_env>
  local label="$1" prof="$2" senv="$3"
  echo "=== $label $prof $(date +%T)"
  MODE="$MODE" SERVER_ENV="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$N1PY $senv" \
    SERVER_ARGS="$EXTRA_ARGS" \
    bash "$BENCH/prof/validate.sh" "$label" "$prof" 2>&1 | tee "$R/$label-$prof.log" \
    || { echo "ABORT: $label $prof failed (see $R/$label-$prof.log)" >&2; exit 1; }
}

# Every profile's pair runs inside this one lock acquisition: the lock can be held for
# hours by the V5 jobs, so splitting w16 and w4 into separate acquisitions risks the
# two halves of a comparison landing hours apart.
# Control first, then the arm: if the machine drifts during a pair, the drift is charged
# against the change rather than in its favour.
# REPEATS>1 interleaves the pair (off,on,off,on...). The first stage-A pair showed the
# *target verify* phase moving by 12% between arms, which stage A cannot touch -- so a
# single pair does not separate the change from run-to-run drift. Interleaved rounds do:
# a real effect is consistent across rounds, drift is not.
for r in $(seq 1 "${REPEATS:-1}"); do
  sfx=""; [ "${REPEATS:-1}" -gt 1 ] && sfx="r$r"
  for P in $PROFILES; do
    run "n1${STAGE}-off$sfx" "$P" ""
    run "n1${STAGE}-on$sfx"  "$P" "$FLAG"
  done
done
echo "=== N1 A/B done $(date +%T)"
