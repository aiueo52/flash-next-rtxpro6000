#!/bin/bash
# arm_rs2.sh <label> <to|rs|rs1> [samplings, default lmstudio,greedy; none = smoke, probes only] -- one fresh
# `serve-fast.sh wa` from the RS2 worktree (~/tools/sglang-rs2) on PORT (8029) with the stack (5 flags, RQ2 u2h) and
# min_p on in every mode: warm-up, the BN1 grid per sampling mode, profiled probes, and the evidence lines.
# to = target-only verify; rs = RS + RS2 (SGLANG_OPT_SPEC_SPARSE_RS=1); rs1 = RS with RS2 off (diagnosis only).
# SERVER_ENV goes to every mode (e.g. SGLANG_ENABLE_QSA_SHARED_TAIL_PREFIX=1); PROMPT_LIMIT = prompts per domain.
# Run under the GPU lock and a memory cap:
#   flock -w 28800 ~/.gpu.lock systemd-run --user --scope -q -p MemoryMax=110G -p MemorySwapMax=0 \
#       bash bench/rs2/arm_rs2.sh A1 to
set -uo pipefail
LABEL="$1"; MODE="$2"; SAMPLINGS="${3:-lmstudio,greedy}"
PORT="${PORT:-8029}"
WT=${WT:-$HOME/tools/sglang-rs2}; BENCH=~/tools/flash-next-bench; PY=~/tools/sglang-rtxpro6000/.venv/bin/python
OUTDIR=${OUTDIR:-$BENCH/runs/rs2}; mkdir -p "$OUTDIR"; OUTDIR=$(cd "$OUTDIR" && pwd)  # we cd to $WT below
LOG=$OUTDIR/serve-$LABEL.log
bash $BENCH/bench/memgate.sh "$LABEL" || exit 3   # MemAvailable >= 110 GiB before the server
FI=rq2u2h
ARM_ENV="SGLANG_ROUTER_FAST_TOPK=1 SGLANG_OPT_SPEC_SPARSE_VERIFY=1 SGLANG_OPT_SPEC_SPARSE_TOPK=1"
ARM_ENV="$ARM_ENV SGLANG_OPT_GDN_FRONT_OVERLAP=1 SGLANG_OPT_DRAFT_MOE_GEMV=1 SGLANG_SPEC_MIN_P=1"
EXTRA=()
case $MODE in
  to) ;;
  rs) ARM_ENV="$ARM_ENV SGLANG_RS_DRAFT_TOPK=64 SGLANG_OPT_SPEC_SPARSE_RS=1"; EXTRA=(--speculative-use-rejection-sampling) ;;
  rs1) ARM_ENV="$ARM_ENV SGLANG_RS_DRAFT_TOPK=64"; EXTRA=(--speculative-use-rejection-sampling) ;;
  *) echo "mode must be to, rs or rs1"; exit 2 ;;
esac
[ -s $HOME/.cache/sglang-$FI/.cache/flashinfer/0.6.17/120f/cached_ops/fused_moe_120/fused_moe_120.so ] ||
  { echo "[$LABEL] no FlashInfer build for $FI (bench/fc_moe/build_rq2.sh)"; exit 2; }
ARM_ENV="$ARM_ENV PYTHONPATH=$HOME/tools/flashinfer-p3:$WT/python FLASHINFER_WORKSPACE_BASE=$HOME/.cache/sglang-$FI FLASHINFER_P2_NO_NINJA=1"
for f in serve-fast.sh serve-local.sh .cache/jit; do
  [ -e "$WT/$f" ] || { echo "[$LABEL] $WT/$f missing: copy the serve scripts and .cache first"; exit 2; }
done
source "$BENCH/bench/dg1/memcheck.sh"
echo "[$LABEL] start $(date +%T) mode=$MODE samplings=$SAMPLINGS wt=$WT commit=$(git -C $WT log --oneline -1 | cut -c1-10) env='$ARM_ENV ${SERVER_ENV:-}' args='${EXTRA[*]}'"
cd $WT
env PORT=$PORT SERVE_DISPLAY_HZ= $ARM_ENV ${SERVER_ENV:-} setsid ./serve-fast.sh wa "${EXTRA[@]}" > "$LOG" 2>&1 < /dev/null &
SPID=$!
stop() {
  local pg=$SPID  # setsid in a script execs without forking: the server group id is its pid, and stays valid after it exits
  [ -n "$pg" ] && kill -TERM -- -$pg 2>/dev/null
  for i in $(seq 1 20); do kill -0 $SPID 2>/dev/null || break; sleep 1; done
  [ -n "$pg" ] && kill -KILL -- -$pg 2>/dev/null
  # our own leftovers only: processes of our process group still listening on our port
  for p in $(ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u); do
    [ "$(ps -o pgid= -p $p 2>/dev/null | tr -d ' ')" = "$pg" ] && kill -KILL $p 2>/dev/null
  done
  for i in $(seq 1 60); do
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] && break; sleep 2
  done
  echo "[$LABEL] stopped $(date +%T); gpu mem: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
}
trap stop EXIT
for i in $(seq 1 300); do   # start-up is 8-12 min under CPU load; 20 min cap
  sleep 4; grep -q "ready to roll" "$LOG" && break
  kill -0 $SPID 2>/dev/null || { echo "[$LABEL] server died"; tail -30 "$LOG"; exit 1; }
done
grep -q "ready to roll" "$LOG" || { echo "[$LABEL] server not ready after 20 min"; tail -5 "$LOG"; exit 1; }
echo "[$LABEL] server up $(date +%T)"
PG=$(ps -o pgid= -p $SPID 2>/dev/null | tr -d ' ')
for p in $(pgrep -g "$PG"); do LC_ALL=C grep -h 'fused_moe_120\.so' /proc/$p/maps 2>/dev/null; done | awk '{print $6}' | sort -u \
    | sed "s/^/[$LABEL] mapped: /"
$PY $BENCH/prof/fc_sampling_probe.py --port $PORT --mode greedy --workload $BENCH/workloads/prose-en.txt --n 1 --max-tokens 200 2>&1 | tail -1
$PY $BENCH/prof/fc_sampling_probe.py --port $PORT --mode sampling --workload $BENCH/workloads/code-edit.txt --n 1 --max-tokens 200 2>&1 | tail -1
cd $BENCH
for S in ${SAMPLINGS//,/ }; do
  [ "$S" = none ] && continue   # smoke: probes only
  .venv-review/bin/python -m fnbench run --endpoint http://127.0.0.1:$PORT/v1 --engine sglang \
      --workloads code-edit,prose-en,prose-ja,agent-loop --prompt-sets workloads/sets --prompt-limit ${PROMPT_LIMIT:-8} \
      --repeats 1 --sampling "$S" --require-acceptance --allow-proc sglang --out "$OUTDIR/$LABEL-$S.jsonl"
  echo "[$LABEL] fnbench $S rc=$? $(date +%T)"
done
TG=$OUTDIR/traces/$LABEL-g-code-edit; rm -rf $TG
$PY $BENCH/prof/fc_sampling_probe.py --port $PORT --mode greedy --workload $BENCH/workloads/code-edit.txt \
    --n 1 --profile-dir $TG --profile-steps 20 2>&1 | tail -2
for W in code-edit prose-en; do
  TD=$OUTDIR/traces/$LABEL-$W; rm -rf $TD
  $PY $BENCH/prof/fc_sampling_probe.py --port $PORT --mode sampling --workload $BENCH/workloads/$W.txt \
      --n 2 --profile-dir $TD --profile-steps 20 --json-out $OUTDIR/$LABEL-probe.jsonl 2>&1 | tail -3
done
echo "[$LABEL] error lines in server log: $(grep -cE "Traceback|Error" "$LOG")"
echo "[$LABEL] rs2 enabled lines: $(grep -c 'SGLANG_OPT_SPEC_SPARSE_RS on: sparse chain RS' "$LOG")"
echo "[$LABEL] draft MoE GEMV enabled lines: $(grep -c "draft MoE GEMV enabled" "$LOG")"
echo "[$LABEL] ST1 lines: $(grep -c 'tail after the valid prefix' "$LOG")"
$PY $BENCH/bench/dg1/dg_trace.py $TG | sed "s/^/[$LABEL] dg1 /"
$PY $BENCH/bench/rq1/prologue_modes.py $TG | sed "s/^/[$LABEL] rq /"
for W in code-edit prose-en; do
  # RS2's kernels, the dense RS chain kernel, SV1's sparse verify, and the V-wide top-k RS2 must not launch
  n=$(zcat $OUTDIR/traces/$LABEL-$W/*.trace.json.gz 2>/dev/null \
      | grep -oE '"name": *"(_draft_partial_topk_kernel|_draft_finalize_kernel|_chain_sampling_sparse_kernel|speculative_sampling_classic_kernel|_sparse_target_probs_kernel|_tree_sampling_target_only_sparse_kernel)|at::native::(sbtopk|mbtopk)::[A-Za-z]+' \
      | sed 's/"name": *"//' | sort | uniq -c | tr -s ' ' | tr '\n' ';')
  echo "[$LABEL] rs2 $W kernels in trace: ${n:-none}"
done
true   # grep -c exits 1 on zero matches
