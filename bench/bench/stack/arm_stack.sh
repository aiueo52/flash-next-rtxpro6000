#!/bin/bash
# arm_stack.sh <label> <off|on> [samplings, default lmstudio,greedy; none = smoke, probes only] -- one fresh
# `serve-fast.sh wa` from the stacked worktree (~/tools/sglang-stack, production 7b4d539f9b + RS1, min_p, SV1,
# FG1, RT1, DG1, SV2, every one default off) on PORT (8025): warm-up, the BN1 grid (4 domains x 8 prompts) per
# sampling mode, then profiled probes (one greedy, two LM-Studio sampling) and the evidence that each change ran.
# off = every flag off, production FlashInfer. on = $STACK_ENV (default below) plus the FlashInfer build
# $STACK_FI (default rq2u2h; empty = production FlashInfer), loaded as bench/rq1/arm_rq1.sh does.
# SERVER_ENV goes to both arms (e.g. SGLANG_SPEC_MIN_P=1); PROMPT_LIMIT = prompts per domain (default 8).
# Run under the GPU lock and a memory cap:
#   flock -w 28800 ~/.gpu.lock systemd-run --user --scope -q -p MemoryMax=110G -p MemorySwapMax=0 \
#       bash bench/stack/arm_stack.sh A1 off
set -uo pipefail
LABEL="$1"; MODE="$2"; SAMPLINGS="${3:-lmstudio,greedy}"
PORT="${PORT:-8025}"
WT=${WT:-$HOME/tools/sglang-stack}; BENCH=~/tools/flash-next-bench; PY=~/tools/sglang-rtxpro6000/.venv/bin/python
OUTDIR=${OUTDIR:-$BENCH/runs/stack}; mkdir -p "$OUTDIR"; OUTDIR=$(cd "$OUTDIR" && pwd)  # we cd to $WT below
LOG=$OUTDIR/serve-$LABEL.log
bash $BENCH/bench/memgate.sh "$LABEL" || exit 3   # MemAvailable >= 110 GiB before the server
DEFAULT_STACK_ENV="SGLANG_ROUTER_FAST_TOPK=1 SGLANG_OPT_SPEC_SPARSE_VERIFY=1 SGLANG_OPT_SPEC_SPARSE_TOPK=1 SGLANG_OPT_GDN_FRONT_OVERLAP=1 SGLANG_OPT_DRAFT_MOE_GEMV=1"
case $MODE in
  off) ARM_ENV="PYTHONPATH=$WT/python" ;;
  on) ARM_ENV="${STACK_ENV-$DEFAULT_STACK_ENV}"
      FI=${STACK_FI-rq2u2h}
      if [ -n "$FI" ]; then
        [ -s $HOME/.cache/sglang-$FI/.cache/flashinfer/0.6.17/120f/cached_ops/fused_moe_120/fused_moe_120.so ] ||
          { echo "[$LABEL] no FlashInfer build for $FI (bench/fc_moe/build_rq2.sh)"; exit 2; }
        ARM_ENV="$ARM_ENV PYTHONPATH=$HOME/tools/flashinfer-p3:$WT/python FLASHINFER_WORKSPACE_BASE=$HOME/.cache/sglang-$FI FLASHINFER_P2_NO_NINJA=1"
      else
        # SV2's FlashInfer topk module is built only in the private caches; the shared one would JIT it.
        case " $ARM_ENV " in *" SGLANG_OPT_SPEC_SPARSE_TOPK=1 "*)
          echo "[$LABEL] SGLANG_OPT_SPEC_SPARSE_TOPK=1 needs STACK_FI (a private FlashInfer cache)"; exit 2 ;; esac
        ARM_ENV="$ARM_ENV PYTHONPATH=$WT/python"
      fi ;;
  *) echo "mode must be off or on"; exit 2 ;;
esac
for f in serve-fast.sh serve-local.sh .cache/jit; do
  [ -e "$WT/$f" ] || { echo "[$LABEL] $WT/$f missing: copy the serve scripts and .cache first"; exit 2; }
done
source "$BENCH/bench/dg1/memcheck.sh"
echo "[$LABEL] start $(date +%T) mode=$MODE samplings=$SAMPLINGS wt=$WT commit=$(git -C $WT log --oneline -1 | cut -c1-10) env='$ARM_ENV ${SERVER_ENV:-}'"
cd $WT
env PORT=$PORT SERVE_DISPLAY_HZ= $ARM_ENV ${SERVER_ENV:-} setsid ./serve-fast.sh wa > "$LOG" 2>&1 < /dev/null &
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
# which MoE module the server mapped (sglang-<build>/ on an RQ arm, the production cache otherwise)
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
echo "[$LABEL] draft MoE GEMV enabled lines: $(grep -c "draft MoE GEMV enabled" "$LOG")"
echo "[$LABEL] GEMV fallback warnings: $(grep -c "keeps CUTLASS for this graph\|SGLANG_OPT_DRAFT_MOE_GEMV ignored" "$LOG")"
$PY $BENCH/bench/rt1/router_trace.py $TG | sed "s/^/[$LABEL] rt1 /"
$PY $BENCH/bench/dg1/dg_trace.py $TG | sed "s/^/[$LABEL] dg1 /"
$PY $BENCH/bench/fg1/gate_streams.py $TG | sed "s/^/[$LABEL] fg1 /"
$PY $BENCH/bench/rq1/prologue_modes.py $TG | sed "s/^/[$LABEL] rq /"
for W in code-edit prose-en; do
  n=$(zcat $OUTDIR/traces/$LABEL-$W/*.trace.json.gz 2>/dev/null | grep -o '"name": *"_[a-z_]*sparse[a-z_]*kernel' | sort | uniq -c | tr -s ' ' | tr '\n' ';')
  echo "[$LABEL] sv1 $W sparse-verify kernels in trace: ${n:-none}"
  n=$(zcat $OUTDIR/traces/$LABEL-$W/*.trace.json.gz 2>/dev/null | grep -oE 'flashinfer::sampling::(RadixTopKKernel|FilteredTopKUnifiedKernel|StableSortTopKByValueKernel|SortTopKByIndexKernel)|at::native::(sbtopk|mbtopk)::[A-Za-z]+' | sort | uniq -c | tr -s ' ' | tr '\n' ';')
  echo "[$LABEL] sv2 $W top-k kernels in trace: ${n:-none}"
done
true   # grep -c exits 1 on zero matches
