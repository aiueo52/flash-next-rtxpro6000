#!/bin/bash
# WA3: adapted from adaptive/c1/ab_run.sh and prof/validate.sh.
# Invoke the entire test-c -> control -> test-c sequence under the GPU flock.
set -euo pipefail
BENCH=$HOME/tools/flash-next-bench
REPO=$HOME/tools/sglang-rtxpro6000
PY=$REPO/.venv/bin/python
PREFIX=$1
ROOT=$BENCH/specs/wa2
NOTE=$BENCH/specs/WA2_LOG.md
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
mkdir -p "$ROOT"
PGID= MPID=
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [ -n "$PGID" ]; then
    kill -TERM -- "-$PGID" 2>/dev/null || true
    sleep 8
    if kill -0 -- "-$PGID" 2>/dev/null; then
      kill -KILL -- "-$PGID" 2>/dev/null || true
      sleep 3
    fi
  fi
  if [ -n "$MPID" ]; then kill "$MPID" 2>/dev/null || true; wait "$MPID" 2>/dev/null || true; fi
  printf '\nRun exit %s at %s; owned server process group cleaned up.\n' "$rc" "$(date --iso-8601=seconds)" >> "$NOTE"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# Single bounded blocking wait, as in C1. Never kill an unowned process.
for ((i=0; i<80; i++)); do
  PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
  [ -z "$PIDS" ] && break
  echo "Waiting for GPU: $PIDS"
  sleep 15
done
sleep 20
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[ -z "$PIDS" ] || { echo "GPU still busy after bounded wait: $PIDS"; exit 1; }

for ARM in c1 ctl c2; do
  LABEL=$PREFIX-$ARM
  OUT=$ROOT/$LABEL
  mkdir "$OUT"
  printf '\n#### Starting %s at %s\n' "$LABEL" "$(date --iso-8601=seconds)" >> "$NOTE"
  PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
  [ -z "$PIDS" ] || { echo "Unexpected GPU process: $PIDS"; exit 1; }
  if [ "$ARM" != ctl ]; then
    CFG=$BENCH/adaptive/w16_3_7_15_c.json
    A=7.942995690 B=0.555366379
  else
    CFG=$BENCH/adaptive/w16_conf.json
    A=9.74 B=0.70
  fi
  PYTHONPATH=$HOME/tools/sglang-wa2/python "$PY" -B -c 'import sglang.srt.speculative.adaptive_confidence as m; print(m.__file__)' > "$OUT/code-origin.txt"
  grep -qx $HOME/tools/sglang-wa2/python/sglang/srt/speculative/adaptive_confidence.py "$OUT/code-origin.txt"
  cd "$REPO"
  # serve-local hardcodes its cache under REPO. A mount-private writable copy
  # permits the exact production launchers while the real tree/venv stay RO.
  env PYTHONPATH=$HOME/tools/sglang-wa2/python SERVE_DISPLAY_HZ= WA_MEM_FRACTION=0.920 ADAPTIVE_CONFIG="$CFG" SGLANG_ADAPTIVE_POLICY=confidence \
    SGLANG_ADAPTIVE_STEP_A="$A" SGLANG_ADAPTIVE_STEP_B="$B" \
    SGLANG_ADAPTIVE_TRACE="$OUT/trace.jsonl" SGLANG_ADAPTIVE_DEBUG=1 \
    setsid bwrap --die-with-parent --dev-bind / / \
      --ro-bind $HOME/tools/sglang-wa2 $HOME/tools/sglang-wa2 \
      --ro-bind "$REPO" "$REPO" --bind "$ROOT/runtime-cache" "$REPO/.cache" \
      "$REPO/serve-fast.sh" wa > "$OUT/server.log" 2>&1 < /dev/null &
  SPID=$!
  PGID=$SPID
  # C1 cleanup uses PGID. setsid makes the owned leader PID its group ID.
  (
    LOW_SINCE=-1
    while kill -0 "$SPID" 2>/dev/null; do
      FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) || { kill -TERM -- "-$SPID"; exit 1; }
      PHASE=startup
      [ ! -f "$OUT/ready" ] || PHASE=steady
      [ ! -f "$OUT/stopping" ] || PHASE=cleanup
      printf '{"t":"%s","phase":"%s","free_mib":%s}\n' "$(date --iso-8601=seconds)" "$PHASE" "$FREE" >> "$OUT/memory.jsonl"
      # Supervisor clarification: loading transients are normal. After ready,
      # abort only for <1024 MiB sustained across at least 10 seconds.
      if [ "$PHASE" = steady ] && [ "$FREE" -lt 1024 ]; then
        [ "$LOW_SINCE" -ge 0 ] || LOW_SINCE=$SECONDS
        if [ "$((SECONDS - LOW_SINCE))" -ge 10 ]; then
          printf '\nABORT %s: steady VRAM below 1024 MiB for >=10 seconds (%s MiB free).\n' "$LABEL" "$FREE" >> "$NOTE"
          kill -TERM -- "-$SPID" 2>/dev/null || true
          exit 1
        fi
      else
        LOW_SINCE=-1
      fi
      sleep 2
    done
  ) &
  MPID=$!
  for ((i=0; i<225; i++)); do
    sleep 4
    if ! kill -0 "$SPID" 2>/dev/null; then tail -40 "$OUT/server.log"; exit 1; fi
    grep -q 'ready to roll' "$OUT/server.log" && break
  done
  grep -q 'ready to roll' "$OUT/server.log" || { echo 'Server startup exceeded 15 min'; tail -30 "$OUT/server.log"; exit 1; }
  ACTUAL=$(ps -o pgid= -p "$SPID" | tr -d ' ')
  [ "$ACTUAL" = "$PGID" ] || { echo "Unexpected process group $ACTUAL"; exit 1; }
  touch "$OUT/ready"
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits > "$OUT/ready-free-mib.txt"
  printf '\n%s ready: steady-state free VRAM %s MiB at %s.\n' "$LABEL" "$(cat "$OUT/ready-free-mib.txt")" "$(date --iso-8601=seconds)" >> "$NOTE"
  echo "[$LABEL] ready $(date --iso-8601=seconds) A=$A B=$B"
  cd "$BENCH"
  WA2_EVENTS="$OUT/events.jsonl" .venv-review/bin/python -B calib/wa2_fnbench.py run \
    --endpoint http://127.0.0.1:8001/v1 --engine sglang \
    --workloads code-edit,prose-en,prose-ja,agent-loop --repeats 3 --sampling greedy \
    --allow-proc sglang --label "$LABEL" --out "$OUT/runs.jsonl" 2>&1 | tee "$OUT/fnbench.log"
  "$PY" -B "$BENCH/prof/needle_test.py" 18500 0.4 "$LABEL" | tee "$OUT/needle.log"
  grep -q 'PASS=True' "$OUT/needle.log"
  touch "$OUT/stopping"
  kill -TERM -- "-$PGID" 2>/dev/null || true
  sleep 8
  if kill -0 -- "-$PGID" 2>/dev/null; then kill -KILL -- "-$PGID" 2>/dev/null || true; sleep 3; fi
  wait "$SPID" 2>/dev/null || true
  kill "$MPID" 2>/dev/null || true
  wait "$MPID" 2>/dev/null || true
  PGID= MPID=
  "$PY" -B "$BENCH/calib/wa3_report.py" arm "$OUT"
  echo "[$LABEL] complete $(date --iso-8601=seconds)"
  sleep 20
done
"$PY" -B "$BENCH/calib/wa3_report.py" final "$ROOT/$PREFIX-c1" "$ROOT/$PREFIX-ctl" "$ROOT/$PREFIX-c2"
