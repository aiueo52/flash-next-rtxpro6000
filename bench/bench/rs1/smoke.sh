#!/bin/bash
# smoke.sh <label> <to|rs> -- fresh wa server from the RS worktree, a few probe requests
# (greedy prose-en, LM-Studio sampling code-edit/prose-en/prose-ja), server-log error scan, stop.
set -uo pipefail
LABEL="$1"; MODE="$2"
PORT="${PORT:-8021}"
WT=~/tools/sglang-rs; BENCH=~/tools/flash-next-bench; PY=~/tools/sglang-rtxpro6000/.venv/bin/python
OUTDIR=${OUTDIR:-$BENCH/runs/rs1}; mkdir -p "$OUTDIR"; OUTDIR=$(cd "$OUTDIR" && pwd)  # we cd to $WT below
LOG=$OUTDIR/serve-$LABEL.log; JS=$OUTDIR/$LABEL-probe.jsonl; rm -f "$JS"
EXTRA=(); [ "$MODE" = rs ] && EXTRA=(--speculative-use-rejection-sampling)
source "$BENCH/bench/rs1/memcheck.sh"
echo "[$LABEL] start $(date +%T) mode=$MODE env='${SERVER_ENV:-}'"
cd $WT
env PORT=$PORT PYTHONPATH=$WT/python SERVE_DISPLAY_HZ= ${SERVER_ENV:-} \
    setsid ./serve-fast.sh wa "${EXTRA[@]}" > "$LOG" 2>&1 < /dev/null &
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
  echo "[$LABEL] stopped $(date +%T); gpu mem: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
}
trap stop EXIT
for i in $(seq 1 300); do   # start-up is 8-12 min under CPU load; 20 min cap
  sleep 4; grep -q "ready to roll" "$LOG" && break
  kill -0 $SPID 2>/dev/null || { echo "[$LABEL] server died"; tail -40 "$LOG"; exit 1; }
done
grep -q "ready to roll" "$LOG" || { echo "[$LABEL] server not ready after 20 min"; tail -5 "$LOG"; exit 1; }
echo "[$LABEL] server up $(date +%T)"
P="$PY $BENCH/prof/fc_sampling_probe.py --port $PORT --json-out $JS"
$P --mode greedy --workload $BENCH/workloads/prose-en.txt --n 1 --max-tokens 200 > /dev/null 2>&1   # warm-up
$P --mode sampling --workload $BENCH/workloads/code-edit.txt --n 1 --max-tokens 200 > /dev/null 2>&1 # warm-up
for W in code-edit prose-en prose-ja; do
  for M in greedy sampling; do
    echo "== $W $M"; $P --mode $M --workload $BENCH/workloads/$W.txt --n 2 --max-tokens 500 2>&1 | tail -4
  done
done
echo "[$LABEL] error lines: $(grep -cE 'Traceback|Error|error' "$LOG")"
grep -E "Traceback|Error" "$LOG" | head -5
