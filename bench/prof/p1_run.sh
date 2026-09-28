#!/bin/bash
# p1_run.sh <label> <w4|w16>   -- one server, then the P1 measurement set.
#   MODE=speed    profile traces + trimmed_step + short fnbench + route census (D)
#   MODE=quality  fnbench x2 + needle + greedy-agreement dump
#   MODE=both     everything
# Server env via SERVER_ENV="A=1 B=2"; CENSUS=1 arms route_census (D per call).
# Run under flock; the caller owns the GPU lock.
# Any failing step aborts the run. Only the server process group started here is ever signalled
# (also on abort, via the EXIT trap); other GPU processes are never killed -- the run refuses instead.
set -euo pipefail
LABEL="$1"; PROFILE="$2"; MODE="${MODE:-both}"
REPO=~/tools/sglang-rtxpro6000; BENCH=~/tools/flash-next-bench
PY=$REPO/.venv/bin/python; PROF=$BENCH/prof
LOG=$REPO/logs/serve-$LABEL-$PROFILE.log
mkdir -p "$REPO/logs" "$PROF/traces" "$BENCH/runs"
# Split PYTHONPATH out of SERVER_ENV so the census hook can be prepended to it
# (a plain string append would land after the last variable, not inside the path).
ENV_EXTRA=""; PP=""
for kv in ${SERVER_ENV:-}; do
  case "$kv" in PYTHONPATH=*) PP="${kv#PYTHONPATH=}" ;; *) ENV_EXTRA="$ENV_EXTRA $kv" ;; esac
done
if [ "${CENSUS:-0}" = 1 ]; then
  HOOK=/tmp/moe-census-hook-$LABEL; mkdir -p "$HOOK"
  cp "$BENCH/bench/moe_smallm/sitecustomize_census.py" "$HOOK/sitecustomize.py"
  PP="$HOOK${PP:+:$PP}:$BENCH/bench"
  ENV_EXTRA="$ENV_EXTRA SGLANG_MOE_CENSUS_LOG=/tmp/census-$LABEL SGLANG_MOE_CENSUS_RING=16384 SGLANG_MOE_CENSUS_AUTODUMP=10"
  rm -f /tmp/census-$LABEL.*
fi
[ -n "$PP" ] && ENV_EXTRA="$ENV_EXTRA PYTHONPATH=$PP"
echo "[$LABEL/$PROFILE] env: $ENV_EXTRA"
. "$(dirname "$(readlink -f "$0")")/../lib/owned_group.sh"
trap owned_cleanup EXIT; trap 'exit 130' INT; trap 'exit 143' TERM
require_idle_gpu 120
cd "$REPO"
start_owned "$LOG" env $ENV_EXTRA ./serve-fast.sh "$PROFILE"
for i in $(seq 1 90); do
  sleep 4; grep -q "ready to roll" "$LOG" && break
  kill -0 $OWNED_PID 2>/dev/null || { echo "server died"; tail -20 "$LOG"; exit 1; }
done
grep -q "ready to roll" "$LOG" || { echo "server not ready"; tail -8 "$LOG"; exit 1; }   # EXIT trap stops the server group
echo "[$LABEL/$PROFILE] server up ($(date +%T))"

if [ "$MODE" = speed ] || [ "$MODE" = both ]; then
  for W in code-edit prose-en; do
    OUT=$PROF/traces/$LABEL-$PROFILE-$W; rm -rf "$OUT"
    $PY $PROF/profile_decode2.py "$OUT" 20 "$BENCH/workloads/$W.txt" 2>&1 | { grep -v "^files" || true; } | tail -2
  done
  $PY $PROF/trimmed_step.py $PROF/traces/$LABEL-$PROFILE-code-edit/*.gz \
                            $PROF/traces/$LABEL-$PROFILE-prose-en/*.gz
fi

REPEATS=1; if [ "$MODE" = quality ] || [ "$MODE" = both ]; then REPEATS=2; fi
(cd "$BENCH" && .venv-review/bin/python -m fnbench run \
   --endpoint http://127.0.0.1:8001/v1 --engine sglang \
   --workloads code-edit,prose-en,agent-loop,prose-ja --repeats $REPEATS \
   --sampling greedy --allow-proc sglang --label "$LABEL-$PROFILE" \
   --out "runs/$LABEL-$PROFILE.jsonl" 2>&1 | tail -8)

if [ "$MODE" = quality ] || [ "$MODE" = both ]; then
  $PY $PROF/needle_test.py 18500 0.4 "$LABEL-$PROFILE"
  $PY $PROF/agree_gen.py "$BENCH/runs/agree-$LABEL-$PROFILE.json" 320 2
fi

if [ "${CENSUS:-0}" = 1 ]; then
  sleep 15
  N=$(ls -S /tmp/census-$LABEL.*.npz 2>/dev/null | grep -v "\.tmp\." | head -1) || N=""
  if [ -n "$N" ]; then cp "$N" "$BENCH/runs/census-$LABEL-$PROFILE.npz"; fi
  J=$(ls -S /tmp/census-$LABEL.*.json 2>/dev/null | head -1) || J=""
  if [ -n "$J" ]; then
    cp "$J" "$BENCH/runs/census-$LABEL-$PROFILE.json"
    $PY -c "import json;d=json.load(open('$BENCH/runs/census-$LABEL-$PROFILE.json'));
print('D per call:', {t:round(v['mean_distinct'],2) for t,v in d['by_T'].items()},
      'calls:', {t:v['calls'] for t,v in d['by_T'].items()})"
  fi
fi

stop_owned 30
report_gpu_leftovers 20
echo "[$LABEL/$PROFILE] done ($(date +%T)); gpu mem: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
