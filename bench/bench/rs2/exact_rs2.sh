#!/bin/bash
# exact_rs2.sh <label> <to|rs> -- RS2's greedy exactness arm (specs/RS2_SPARSE_RS_2026-10-01.md F2): one
# `serve-fast.sh w4 --disable-flashinfer-autotune` from ~/tools/sglang-rs2 with the stack flags, RQ2 u2h and min_p on;
# rs adds RS + RS2. Then bench/stack/exact_client.py (greedy, full text, 2 repeats) into runs/rs2/exact.
# Greedy rows keep the greedy chain under RS (spec 2.4), so to and rs should emit the same text.
# SERVER_ENV goes to both modes. Run under the GPU lock (bench/rs2/chain_rs2.sh).
LABEL=$1; MODE=$2
WT=${WT:-$HOME/tools/sglang-rs2}; BENCH=~/tools/flash-next-bench; PY=~/tools/sglang-rtxpro6000/.venv/bin/python
PORT=${PORT:-8029}; OUT=$BENCH/runs/rs2/exact; mkdir -p $OUT; LOG=$OUT/serve-$LABEL.log
ENVS="SGLANG_ROUTER_FAST_TOPK=1 SGLANG_OPT_SPEC_SPARSE_VERIFY=1 SGLANG_OPT_SPEC_SPARSE_TOPK=1"
ENVS="$ENVS SGLANG_OPT_GDN_FRONT_OVERLAP=1 SGLANG_OPT_DRAFT_MOE_GEMV=1 SGLANG_SPEC_MIN_P=1"
EXTRA=(--disable-flashinfer-autotune)
case $MODE in
  to) ;;
  rs) ENVS="$ENVS SGLANG_RS_DRAFT_TOPK=64 SGLANG_OPT_SPEC_SPARSE_RS=1"; EXTRA+=(--speculative-use-rejection-sampling) ;;
  *) echo "mode must be to or rs"; exit 2 ;;
esac
ENVS="$ENVS PYTHONPATH=$HOME/tools/flashinfer-p3:$WT/python FLASHINFER_WORKSPACE_BASE=$HOME/.cache/sglang-rq2u2h"
ENVS="$ENVS FLASHINFER_P2_NO_NINJA=1"
source $BENCH/bench/dg1/memcheck.sh
echo "[$LABEL] start $(date +%T) w4 mode=$MODE commit=$(git -C $WT log --oneline -1 | cut -c1-10) env='${SERVER_ENV:-}'"
cd $WT
env PORT=$PORT SERVE_DISPLAY_HZ= $ENVS ${SERVER_ENV:-} setsid ./serve-fast.sh w4 "${EXTRA[@]}" > "$LOG" 2>&1 < /dev/null &
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
  echo "[$LABEL] stopped $(date +%T)"
}
trap stop EXIT
for i in $(seq 1 300); do
  sleep 4; grep -q "ready to roll" "$LOG" && break
  kill -0 $SPID 2>/dev/null || { echo "[$LABEL] server died"; tail -30 "$LOG"; exit 1; }
done
grep -q "ready to roll" "$LOG" || { echo "[$LABEL] server not ready after 20 min"; tail -5 "$LOG"; exit 1; }
echo "[$LABEL] server up $(date +%T)"
echo "[$LABEL] rs2 enabled lines: $(grep -c 'SGLANG_OPT_SPEC_SPARSE_RS on: sparse chain RS' "$LOG")"
$PY $BENCH/bench/stack/exact_client.py --port $PORT --out $OUT/$LABEL.jsonl --max-tokens 2048 --repeat 2 \
    $BENCH/workloads/code-edit.txt $BENCH/workloads/prose-en.txt $BENCH/workloads/prose-ja.txt \
    $BENCH/workloads/agent-loop.txt 2>&1 | sed "s/^/[$LABEL] /"
echo "[$LABEL] error lines in server log: $(grep -cE "Traceback|Error" "$LOG")"
