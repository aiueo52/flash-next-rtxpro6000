#!/bin/bash
# arm.sh <label> <to|rs> [samplings, default lmstudio,greedy] -- one fresh `serve-fast.sh wa` from the RS worktree
# (~/tools/sglang-rs, PYTHONPATH) on PORT (8021), one greedy + one sampling warm-up, then the BN1
# grid (4 domains x 8 prompts) once per sampling mode, to $OUTDIR/<label>-<mode>.jsonl.
# to = target-only verify (production), rs = rejection sampling.
# Run under the GPU lock and a memory cap:
#   flock -w 28800 ~/.gpu.lock systemd-run --user --scope -q -p MemoryMax=110G -p MemorySwapMax=0 \
#       bash bench/rs1/arm.sh A1 to
# Extra server env: SERVER_ENV="A=1 B=2".
set -uo pipefail
LABEL="$1"; MODE="$2"; SAMPLINGS="${3:-lmstudio,greedy}"
PORT="${PORT:-8021}"
WT=${WT:-$HOME/tools/sglang-rs}; BENCH=~/tools/flash-next-bench; PY=~/tools/sglang-rtxpro6000/.venv/bin/python
OUTDIR=${OUTDIR:-$BENCH/runs/rs1}; mkdir -p "$OUTDIR"; OUTDIR=$(cd "$OUTDIR" && pwd)  # we cd to $WT below
LOG=$OUTDIR/serve-$LABEL.log
EXTRA=(); [ "$MODE" = rs ] && EXTRA=(--speculative-use-rejection-sampling)
source "$BENCH/bench/rs1/memcheck.sh"
echo "[$LABEL] start $(date +%T) mode=$MODE samplings=$SAMPLINGS wt=$WT env='${SERVER_ENV:-}'"
cd $WT
env PORT=$PORT PYTHONPATH=$WT/python SERVE_DISPLAY_HZ= ${SERVER_ENV:-} \
    setsid ./serve-fast.sh wa "${EXTRA[@]}" > "$LOG" 2>&1 < /dev/null &
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
$PY $BENCH/prof/fc_sampling_probe.py --port $PORT --mode greedy --workload $BENCH/workloads/prose-en.txt --n 1 --max-tokens 200 2>&1 | tail -2
$PY $BENCH/prof/fc_sampling_probe.py --port $PORT --mode sampling --workload $BENCH/workloads/code-edit.txt --n 1 --max-tokens 200 2>&1 | tail -2
cd $BENCH
for S in ${SAMPLINGS//,/ }; do
  .venv-review/bin/python -m fnbench run --endpoint http://127.0.0.1:$PORT/v1 --engine sglang \
      --workloads code-edit,prose-en,prose-ja,agent-loop --prompt-sets workloads/sets --prompt-limit 8 \
      --repeats 1 --sampling "$S" --require-acceptance --allow-proc sglang --out "$OUTDIR/$LABEL-$S.jsonl"
  echo "[$LABEL] fnbench $S rc=$? $(date +%T)"
done
grep -cE "Traceback|Error" "$LOG" | sed "s/^/[$LABEL] error lines in server log: /"
true   # grep -c exits 1 on zero matches
