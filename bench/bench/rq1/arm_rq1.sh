#!/bin/bash
# arm_rq1.sh <label> <prod|rq1|rq2uV> [N, default 4] -- one fresh `serve-fast.sh wa` from the RQ1 worktree
# (~/tools/sglang-rq1, production commit 7b4d539f9b) on PORT (8023); warm-up, then per workload x {greedy,
# sampling} N unprofiled 600-token requests (fc_sampling_probe: ms/step and tok/step per request), one profiled
# greedy code-edit request (20-step trace: in-server prologue modes) and an error scan of the server log.
# prod = the venv FlashInfer and the production JIT cache.
# rq1  = FlashInfer from ~/tools/flashinfer-p2 (rolled expand quantize loop, FC_fc-moe 4b.1c) with its own JIT
#        cache ~/.cache/sglang-p2, whose modules other than fused_moe_120 are byte-identical to production's.
#        FLASHINFER_P2_NO_NINJA loads those .so files as-is: a copied cache keeps the old absolute paths, so
#        ninja would rebuild every TU inside the server (the 2026-10-01 04:07 OOM).
# rq2uV = RQ2 (row staged in shared memory with cp.async; V = 1, 2, 1h or 2h: quantize loop 1 or 2 chunks per
#        trip, h = SF write address hoisted; FC_fc-moe 1d): FlashInfer python from ~/tools/flashinfer-p3
#        (identical to p2's) with the cache ~/.cache/sglang-rq2uV (bench/fc_moe/build_rq2.sh), loaded the same way.
# Same layout as bench/fg1/arm_fg.sh. Run under the GPU lock and a memory cap:
#   flock -w 28800 ~/.gpu.lock systemd-run --user --scope -q -p MemoryMax=110G -p MemorySwapMax=0 \
#       bash bench/rq1/arm_rq1.sh A1 prod
set -uo pipefail
LABEL="$1"; MODE="$2"; N="${3:-4}"
PORT="${PORT:-8023}"
WT=${WT:-$HOME/tools/sglang-rq1}; BENCH=~/tools/flash-next-bench; PY=~/tools/sglang-rtxpro6000/.venv/bin/python
OUTDIR=${OUTDIR:-$BENCH/runs/rq1}; mkdir -p "$OUTDIR"; OUTDIR=$(cd "$OUTDIR" && pwd)  # we cd to $WT below
LOG=$OUTDIR/serve-$LABEL.log
case $MODE in
  prod) ARM_ENV="PYTHONPATH=$WT/python" ;;
  rq1) ARM_ENV="PYTHONPATH=$HOME/tools/flashinfer-p2:$WT/python FLASHINFER_WORKSPACE_BASE=$HOME/.cache/sglang-p2 FLASHINFER_P2_NO_NINJA=1" ;;
  rq2u*) [ -s $HOME/.cache/sglang-$MODE/.cache/flashinfer/0.6.17/120f/cached_ops/fused_moe_120/fused_moe_120.so ] ||
           { echo "no build for $MODE (bench/fc_moe/build_rq2.sh)"; exit 2; }
         ARM_ENV="PYTHONPATH=$HOME/tools/flashinfer-p3:$WT/python FLASHINFER_WORKSPACE_BASE=$HOME/.cache/sglang-$MODE FLASHINFER_P2_NO_NINJA=1" ;;
  *) echo "mode must be prod, rq1 or rq2u<variant>"; exit 2 ;;
esac
source "$BENCH/bench/rs1/memcheck.sh"
echo "[$LABEL] start $(date +%T) mode=$MODE n=$N wt=$WT commit=$(git -C $WT log --oneline -1 | cut -c1-10) env='$ARM_ENV ${SERVER_ENV:-}'"
cd $WT
env PORT=$PORT SERVE_DISPLAY_HZ= $ARM_ENV ${SERVER_ENV:-} setsid ./serve-fast.sh wa > "$LOG" 2>&1 < /dev/null &
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
# which MoE module the server mapped (the arm check: sglang-p2 for rq1, sglang-rq2uN for rq2uN, sglang for prod)
PG=$(ps -o pgid= -p $SPID 2>/dev/null | tr -d ' ')
for p in $(pgrep -g "$PG"); do LC_ALL=C grep -h 'fused_moe_120\.so' /proc/$p/maps 2>/dev/null; done | awk '{print $6}' | sort -u \
    | sed "s/^/[$LABEL] mapped: /"
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
$PY $BENCH/bench/rq1/prologue_modes.py $TD | sed "s/^/[$LABEL] /"
true   # grep -c exits 1 on zero matches
