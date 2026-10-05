#!/bin/bash
# arm_xa.sh <label> <off|on> [N, default 4] -- one fresh `serve-fast.sh wa` from the XA1 worktree
# (~/tools/sglang-xa, PYTHONPATH) on PORT (8027): warm-up, then per workload x {greedy, sampling}
# N unprofiled 600-token requests (fc_sampling_probe: ms/step and tok/step per request; the four
# standard workloads plus longctx/doc-a-8k, where every QSA row is at the 2051-column cap), a greedy
# dump for the output match rate (greedy_dump.py, 4 workloads x 2 x 320 tokens), one profiled greedy
# code-edit request (20-step trace: QSA attention kernel counts) and a server-log scan.
# off = production QSA attention (valid counts + _compact_kv + trtllm-gen XQA);
# on = SGLANG_OPT_TRITON_DECODE_ATTN=1 (one split-KV Triton kernel per attention call).
# PRESET (default wa) and SERVE_ARGS start another preset instead, e.g. the optional w4 greedy check (spec,
# Server A/B): PRESET=w4 SERVE_ARGS=--disable-flashinfer-autotune ... arm_xa.sh X1 off 0
# Copied from bench/dg1/arm_dg.sh (2026-10-01); NOT RUN YET. The worktree's serve scripts and caches
# are set up (specs/XA1_DECODE_ATTN_2026-10-01.md, "Server A/B"). Run under the GPU lock:
#   flock -w 28800 ~/.gpu.lock systemd-run --user --scope -q -p MemoryMax=110G -p MemorySwapMax=0 \
#       bash bench/xa1/arm_xa.sh A1 off
set -uo pipefail
LABEL="$1"; MODE="$2"; N="${3:-4}"
PORT="${PORT:-8027}"; PRESET="${PRESET:-wa}"
WT=${WT:-$HOME/tools/sglang-xa}; BENCH=~/tools/flash-next-bench; PY=~/tools/sglang-rtxpro6000/.venv/bin/python
OUTDIR=${OUTDIR:-$BENCH/runs/xa1}; mkdir -p "$OUTDIR"; OUTDIR=$(cd "$OUTDIR" && pwd)  # we cd to $WT below
LOG=$OUTDIR/serve-$LABEL.log
WORKLOADS="code-edit.txt prose-en.txt prose-ja.txt agent-loop.txt"
[ "${XA_LONG:-1}" = 1 ] && WORKLOADS="$WORKLOADS longctx/doc-a-8k.txt"
case $MODE in
  off) ARM_ENV="" ;;
  on) ARM_ENV="SGLANG_OPT_TRITON_DECODE_ATTN=1" ;;
  *) echo "mode must be off or on"; exit 2 ;;
esac
for f in serve-fast.sh serve-local.sh .cache/jit .cache/triton; do
  [ -e "$WT/$f" ] || { echo "[$LABEL] $WT/$f missing: set up the worktree first (spec, Server A/B)"; exit 2; }
done
source "$BENCH/bench/xa1/memcheck.sh"
echo "[$LABEL] start $(date +%T) mode=$MODE preset='$PRESET ${SERVE_ARGS:-}' n=$N wt=$WT commit=$(git -C $WT log --oneline -1 | cut -c1-10) env='$ARM_ENV ${SERVER_ENV:-}'"
cd $WT
env PORT=$PORT PYTHONPATH=$WT/python SERVE_DISPLAY_HZ= $ARM_ENV ${SERVER_ENV:-} \
    setsid ./serve-fast.sh $PRESET ${SERVE_ARGS:-} > "$LOG" 2>&1 < /dev/null &
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
for i in $(seq 1 300); do   # start-up took 4-5.5 min in the rq1 arms (10-01); 20 min cap
  sleep 4; grep -q "ready to roll" "$LOG" && break
  kill -0 $SPID 2>/dev/null || { echo "[$LABEL] server died"; tail -30 "$LOG"; exit 1; }
done
grep -q "ready to roll" "$LOG" || { echo "[$LABEL] server not ready after 20 min"; tail -5 "$LOG"; exit 1; }
echo "[$LABEL] server up $(date +%T)"
$PY $BENCH/prof/fc_sampling_probe.py --port $PORT --mode greedy --workload $BENCH/workloads/prose-en.txt --n 1 --max-tokens 200 2>&1 | tail -1
$PY $BENCH/prof/fc_sampling_probe.py --port $PORT --mode sampling --workload $BENCH/workloads/code-edit.txt --n 1 --max-tokens 200 2>&1 | tail -1
if [ "$N" -gt 0 ]; then
  for W in $WORKLOADS; do
    for M in greedy sampling; do
      $PY $BENCH/prof/fc_sampling_probe.py --port $PORT --mode $M --workload $BENCH/workloads/$W \
          --n $N --max-tokens 600 --json-out $OUTDIR/$LABEL-probe.jsonl 2>&1 | tail -1
    done
  done
fi
if [ "${GREEDY:-1}" = 1 ]; then
  $PY $BENCH/bench/xa1/greedy_dump.py --port $PORT $OUTDIR/$LABEL-greedy.json 2>&1 | tail -1
fi
TD=$OUTDIR/traces/$LABEL-code-edit; rm -rf $TD
$PY $BENCH/prof/fc_sampling_probe.py --port $PORT --mode greedy --workload $BENCH/workloads/code-edit.txt \
    --n 1 --profile-dir $TD --profile-steps 20 2>&1 | tail -2
grep -cE "Traceback|Error" "$LOG" | sed "s/^/[$LABEL] error lines in server log: /"
grep -c "SGLANG_OPT_TRITON_DECODE_ATTN: unsupported" "$LOG" | sed "s/^/[$LABEL] Triton decode fallback warnings: /"
$PY $BENCH/bench/xa1/xa_trace.py $TD | sed "s/^/[$LABEL] /"
true   # grep -c exits 1 on zero matches
