#!/bin/bash
# arm_rt.sh <label> <off|on> [N, default 4] -- one fresh `serve-fast.sh wa` from the RT1 worktree
# (~/tools/sglang-rt1, PYTHONPATH) on PORT (8021); warm-up, then per workload x {greedy, sampling}
# N unprofiled 600-token requests (fc_sampling_probe: ms/step and tok/step per request), one profiled
# greedy code-edit request (20-step trace for the router kernel check) and an error scan of the server log.
# off = production router (_router_triton_kernel); on = SGLANG_ROUTER_FAST_TOPK=1 (packed-key router).
# Same layout as bench/fg1/arm_fg.sh. Run under the GPU lock and a memory cap:
#   flock -w 28800 ~/.gpu.lock systemd-run --user --scope -q -p MemoryMax=110G -p MemorySwapMax=0 \
#       bash bench/rt1/arm_rt.sh A1 off
set -uo pipefail
LABEL="$1"; MODE="$2"; N="${3:-4}"
PORT="${PORT:-8021}"
WT=${WT:-$HOME/tools/sglang-rt1}; BENCH=~/tools/flash-next-bench; PY=~/tools/sglang-rtxpro6000/.venv/bin/python
OUTDIR=${OUTDIR:-$BENCH/runs/rt1}; mkdir -p "$OUTDIR"; OUTDIR=$(cd "$OUTDIR" && pwd)  # we cd to $WT below
LOG=$OUTDIR/serve-$LABEL.log
case $MODE in
  off) ARM_ENV="" ;;
  on) ARM_ENV="SGLANG_ROUTER_FAST_TOPK=1" ;;
  *) echo "mode must be off or on"; exit 2 ;;
esac
source "$BENCH/bench/rs1/memcheck.sh"
echo "[$LABEL] start $(date +%T) mode=$MODE n=$N wt=$WT commit=$(git -C $WT log --oneline -1 | cut -c1-10) env='$ARM_ENV ${SERVER_ENV:-}'"
cd $WT
env PORT=$PORT PYTHONPATH=$WT/python SERVE_DISPLAY_HZ= $ARM_ENV ${SERVER_ENV:-} \
    setsid ./serve-fast.sh wa > "$LOG" 2>&1 < /dev/null &
SPID=$!
stop() {
  local pg; pg=$(ps -o pgid= -p $SPID 2>/dev/null | tr -d ' ')
  [ -n "$pg" ] && kill -TERM -- -$pg 2>/dev/null
  for i in $(seq 1 20); do kill -0 $SPID 2>/dev/null || break; sleep 1; done
  [ -n "$pg" ] && kill -KILL -- -$pg 2>/dev/null
  # our own leftovers only: processes still listening on our port
  for p in $(ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u); do kill -KILL $p 2>/dev/null; done
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
$PY $BENCH/prof/fc_sampling_probe.py --port $PORT --mode greedy --workload $BENCH/workloads/prose-en.txt --n 1 --max-tokens 200 2>&1 | tail -1
$PY $BENCH/prof/fc_sampling_probe.py --port $PORT --mode sampling --workload $BENCH/workloads/code-edit.txt --n 1 --max-tokens 200 2>&1 | tail -1
if [ "$N" -gt 0 ]; then
  for W in code-edit prose-en prose-ja agent-loop; do
    for M in greedy sampling; do
      $PY $BENCH/prof/fc_sampling_probe.py --port $PORT --mode $M --workload $BENCH/workloads/$W.txt \
          --n $N --max-tokens 600 --json-out $OUTDIR/$LABEL-probe.jsonl 2>&1 | tail -1
    done
  done
fi
TD=$OUTDIR/traces/$LABEL-code-edit; rm -rf $TD
$PY $BENCH/prof/fc_sampling_probe.py --port $PORT --mode greedy --workload $BENCH/workloads/code-edit.txt \
    --n 1 --profile-dir $TD --profile-steps 20 2>&1 | tail -2
echo "[$LABEL] error lines in server log: $(grep -cE "Traceback|Error" "$LOG" || true)"
$PY $BENCH/bench/rt1/router_trace.py $TD | sed "s/^/[$LABEL] /"
