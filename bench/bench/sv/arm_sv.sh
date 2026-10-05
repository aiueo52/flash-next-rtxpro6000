#!/bin/bash
# arm_sv.sh <label> <dense|sparse> [samplings, default lmstudio,greedy; none = smoke, probes only] -- one fresh `serve-fast.sh wa` from
# the sparse-verify worktree (~/tools/sglang-sv, PYTHONPATH) on PORT (8021); warm-up, the BN1 grid
# (4 domains x 8 prompts) once per sampling mode, then one profiled LM-Studio-sampling probe per workload
# (20-step torch-profiler trace, for the R.eager comparison) and an error scan of the server log.
# dense = production verify; sparse = SGLANG_OPT_SPEC_SPARSE_VERIFY=1. Same layout as bench/rs1/arm.sh.
# Run under the GPU lock and a memory cap:
#   flock -w 28800 ~/.gpu.lock systemd-run --user --scope -q -p MemoryMax=110G -p MemorySwapMax=0 \
#       bash bench/sv/arm_sv.sh A1 dense
set -uo pipefail
LABEL="$1"; MODE="$2"; SAMPLINGS="${3:-lmstudio,greedy}"
PORT="${PORT:-8021}"
WT=${WT:-$HOME/tools/sglang-sv}; BENCH=~/tools/flash-next-bench; PY=~/tools/sglang-rtxpro6000/.venv/bin/python
OUTDIR=${OUTDIR:-$BENCH/runs/sv}; mkdir -p "$OUTDIR"; OUTDIR=$(cd "$OUTDIR" && pwd)  # we cd to $WT below
LOG=$OUTDIR/serve-$LABEL.log
case $MODE in
  dense) ARM_ENV="" ;;
  sparse) ARM_ENV="SGLANG_OPT_SPEC_SPARSE_VERIFY=1" ;;
  *) echo "mode must be dense or sparse"; exit 2 ;;
esac
source "$BENCH/bench/rs1/memcheck.sh"
echo "[$LABEL] start $(date +%T) mode=$MODE samplings=$SAMPLINGS wt=$WT env='$ARM_ENV ${SERVER_ENV:-}'"
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
$PY $BENCH/prof/fc_sampling_probe.py --port $PORT --mode greedy --workload $BENCH/workloads/prose-en.txt --n 1 --max-tokens 200 2>&1 | tail -2
$PY $BENCH/prof/fc_sampling_probe.py --port $PORT --mode sampling --workload $BENCH/workloads/code-edit.txt --n 1 --max-tokens 200 2>&1 | tail -2
cd $BENCH
for S in ${SAMPLINGS//,/ }; do
  [ "$S" = none ] && continue   # smoke: probes only
  .venv-review/bin/python -m fnbench run --endpoint http://127.0.0.1:$PORT/v1 --engine sglang \
      --workloads code-edit,prose-en,prose-ja,agent-loop --prompt-sets workloads/sets --prompt-limit 8 \
      --repeats 1 --sampling "$S" --require-acceptance --allow-proc sglang --out "$OUTDIR/$LABEL-$S.jsonl"
  echo "[$LABEL] fnbench $S rc=$? $(date +%T)"
done
for W in code-edit prose-en; do
  TD=$OUTDIR/traces/$LABEL-$W; rm -rf $TD
  $PY $BENCH/prof/fc_sampling_probe.py --port $PORT --mode sampling --workload $BENCH/workloads/$W.txt \
      --n 2 --profile-dir $TD --profile-steps 20 --json-out $OUTDIR/$LABEL-probe.jsonl 2>&1 | tail -3
done
grep -cE "Traceback|Error" "$LOG" | sed "s/^/[$LABEL] error lines in server log: /"
# which verify ran: sparse kernels in the profiled probes (0 on dense arms)
for W in code-edit prose-en; do
  n=$(zcat $OUTDIR/traces/$LABEL-$W/*.trace.json.gz 2>/dev/null | grep -o '"name": *"_[a-z_]*sparse[a-z_]*kernel' | sort | uniq -c | tr -s ' ' | tr '\n' ';')
  echo "[$LABEL] $W sparse-verify kernels in trace: ${n:-none}"
done
true   # grep -c exits 1 on zero matches
