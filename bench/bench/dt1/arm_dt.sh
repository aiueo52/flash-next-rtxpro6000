#!/bin/bash
# arm_dt.sh <label> <off|on|conf> [N, default 4] -- one fresh `serve-fast.sh wa` from the DT1 worktree
# (~/tools/sglang-dt, PYTHONPATH) on PORT (8023): warm-up, then per workload x {greedy, sampling}
# N unprofiled 600-token requests (fc_sampling_probe: ms/step and tok/step per request), one profiled
# greedy code-edit request (20-step trace: draft-tail kernels, bench/dt1/dt_trace.py) and a log scan.
# off = production; on = SGLANG_OPT_DRAFT_TAIL=1 (bit-exact items);
# conf = on + SGLANG_OPT_DRAFT_TAIL_FUSED_CONF=1 (p0 from the fused argmax pass, not bit-exact).
# Copied from bench/dg1/arm_dg.sh (2026-10-01); NOT RUN YET. Needs the worktree's serve scripts and
# SGLang JIT cache first (see specs/DT1_DRAFT_TAIL_2026-10-01.md, "Server A/B"). Run under the GPU lock:
#   flock -w 28800 ~/.gpu.lock systemd-run --user --scope -q -p MemoryMax=110G -p MemorySwapMax=0 \
#       bash bench/dt1/arm_dt.sh A1 off
set -uo pipefail
LABEL="$1"; MODE="$2"; N="${3:-4}"
PORT="${PORT:-8023}"
WT=${WT:-$HOME/tools/sglang-dt}; BENCH=~/tools/flash-next-bench; PY=~/tools/sglang-rtxpro6000/.venv/bin/python
OUTDIR=${OUTDIR:-$BENCH/runs/dt1}; mkdir -p "$OUTDIR"; OUTDIR=$(cd "$OUTDIR" && pwd)  # we cd to $WT below
LOG=$OUTDIR/serve-$LABEL.log
case $MODE in
  off) ARM_ENV="" ;;
  on) ARM_ENV="SGLANG_OPT_DRAFT_TAIL=1" ;;
  conf) ARM_ENV="SGLANG_OPT_DRAFT_TAIL=1 SGLANG_OPT_DRAFT_TAIL_FUSED_CONF=1" ;;
  *) echo "mode must be off, on or conf"; exit 2 ;;
esac
for f in serve-fast.sh serve-local.sh .cache/jit; do
  [ -e "$WT/$f" ] || { echo "[$LABEL] $WT/$f missing: set up the worktree first (spec, Server A/B)"; exit 2; }
done
source "$BENCH/bench/dt1/memcheck.sh"
echo "[$LABEL] start $(date +%T) mode=$MODE n=$N wt=$WT commit=$(git -C $WT log --oneline -1 | cut -c1-10) env='$ARM_ENV ${SERVER_ENV:-}'"
cd $WT
env PORT=$PORT PYTHONPATH=$WT/python SERVE_DISPLAY_HZ= $ARM_ENV ${SERVER_ENV:-} \
    setsid ./serve-fast.sh wa > "$LOG" 2>&1 < /dev/null &
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
grep -cE "Traceback|Error" "$LOG" | sed "s/^/[$LABEL] error lines in server log: /"
grep -c "SGLANG_OPT_DRAFT_TAIL on: draft-extend select True" "$LOG" | sed "s/^/[$LABEL] draft tail enabled lines: /"
$PY $BENCH/bench/dt1/dt_trace.py $TD | sed "s/^/[$LABEL] /"
true   # grep -c exits 1 on zero matches
