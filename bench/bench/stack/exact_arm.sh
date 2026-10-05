#!/bin/bash
# exact_arm.sh <label> <0|1> -- one `serve-fast.sh w4` server from ~/tools/sglang-stack-dt with the stack flags and RQ2
# u2h, SGLANG_OPT_DRAFT_TAIL=<0|1>, then bench/stack/exact_client.py (greedy, full text) into runs/stack8/exact.
# wa's greedy text changes from one server start to the next (stack8 A1 vs A2: 16/16 prompts); w4 has one width, no wa
# controller, so its text may repeat. The tuned MoE tactics fuse the finalize with atomics (two production runs
# differ, FC_fc-moe §4b), so the arms run with --disable-flashinfer-autotune (separate finalize kernel). Run it under
# the GPU lock (bench/stack/chain_exact_w4.sh).
LABEL=$1; DT=$2
WT=$HOME/tools/sglang-stack-dt; BENCH=~/tools/flash-next-bench; PY=~/tools/sglang-rtxpro6000/.venv/bin/python
PORT=${PORT:-8027}; OUT=$BENCH/runs/stack8/exact; mkdir -p $OUT; LOG=$OUT/serve-$LABEL.log
ENVS="SGLANG_ROUTER_FAST_TOPK=1 SGLANG_OPT_SPEC_SPARSE_VERIFY=1 SGLANG_OPT_SPEC_SPARSE_TOPK=1"
ENVS="$ENVS SGLANG_OPT_GDN_FRONT_OVERLAP=1 SGLANG_OPT_DRAFT_MOE_GEMV=1 SGLANG_OPT_DRAFT_TAIL=$DT"
ENVS="$ENVS PYTHONPATH=$HOME/tools/flashinfer-p3:$WT/python FLASHINFER_WORKSPACE_BASE=$HOME/.cache/sglang-rq2u2h"
ENVS="$ENVS FLASHINFER_P2_NO_NINJA=1"
source $BENCH/bench/dg1/memcheck.sh
echo "[$LABEL] start $(date +%T) w4 DT=$DT commit=$(git -C $WT log --oneline -1 | cut -c1-10)"
cd $WT
env PORT=$PORT SERVE_DISPLAY_HZ= $ENVS setsid ./serve-fast.sh w4 --disable-flashinfer-autotune > "$LOG" 2>&1 < /dev/null &
SPID=$!
stop() {
  local pg; pg=$(ps -o pgid= -p $SPID 2>/dev/null | tr -d ' ')
  [ -n "$pg" ] && kill -TERM -- -$pg 2>/dev/null
  for i in $(seq 1 20); do kill -0 $SPID 2>/dev/null || break; sleep 1; done
  [ -n "$pg" ] && kill -KILL -- -$pg 2>/dev/null
  for p in $(ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u); do kill -KILL $p 2>/dev/null; done
  for i in $(seq 1 60); do
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] && break; sleep 2
  done
  echo "[$LABEL] stopped $(date +%T)"
}
trap stop EXIT
for i in $(seq 1 300); do
  sleep 4; grep -q "ready to roll" "$LOG" && break
  kill -0 $SPID 2>/dev/null || { echo "[$LABEL] server died"; tail -30 "$LOG"; exit 1; }
done
grep -q "ready to roll" "$LOG" || { echo "[$LABEL] server not ready after 20 min"; tail -5 "$LOG"; exit 1; }
echo "[$LABEL] server up $(date +%T)"
echo "[$LABEL] draft tail enabled lines: $(grep -c 'SGLANG_OPT_DRAFT_TAIL on: draft-extend select True' "$LOG")"
$PY $BENCH/bench/stack/exact_client.py --port $PORT --out $OUT/$LABEL.jsonl --max-tokens 2048 --repeat 2 \
    $BENCH/workloads/code-edit.txt $BENCH/workloads/prose-en.txt $BENCH/workloads/prose-ja.txt \
    $BENCH/workloads/agent-loop.txt 2>&1 | sed "s/^/[$LABEL] /"
echo "[$LABEL] error lines in server log: $(grep -cE "Traceback|Error" "$LOG")"
