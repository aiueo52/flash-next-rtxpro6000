#!/bin/bash
# abba_rq1.sh [tag] -- production FlashInfer vs RQ1 (rolled expand quantize loop, ~/tools/flashinfer-p2), ABBA
# order (A1 prod, B1 rq1, B2 rq1, A2 prod); env B=rq2u<variant> (e.g. B=rq2u2h) tests an RQ2 build instead.
# Each arm takes its own GPU-lock hold (bench/rq1/arm_rq1.sh), then the paired ms/step table
# (bench/fg1/fg_table.py) and the in-server prologue modes of the four traces.
# RQ1 and RQ2 do the same arithmetic, so greedy requests take the same steps up to the MoE's own run-to-run noise
# (finalize atomics): greedy ms/step is the primary metric and tok/step should move no more than between the A
# arms. Stops after B1 if that arm produced no probe rows.
set -uo pipefail
cd ~/tools/flash-next-bench
TAG=${1:-}; OUT=${OUTDIR:-runs/rq1}; B=${B:-rq1}
PY=~/tools/sglang-rtxpro6000/.venv/bin/python
for arm in A1:prod B1:$B B2:$B A2:prod; do
  L=${arm%%:*}$TAG; M=${arm##*:}
  echo "[abba-rq1] $L ($M) queued $(date +%T)"
  flock -w 28800 ~/.gpu.lock systemd-run --user --scope -q -p MemoryMax=110G -p MemorySwapMax=0 \
      env OUTDIR=$OUT bash bench/rq1/arm_rq1.sh "$L" "$M" || echo "[abba-rq1] arm $L rc=$?"
  if [ "${L%$TAG}" = B1 ] && [ ! -s "$OUT/$L-probe.jsonl" ]; then
    echo "[abba-rq1] B1 produced no probe rows; stopping"; exit 1
  fi
  sleep 30   # let other lock waiters in between arms
done
$PY bench/fg1/fg_table.py $OUT/A1$TAG-probe.jsonl $OUT/B1$TAG-probe.jsonl $OUT/B2$TAG-probe.jsonl $OUT/A2$TAG-probe.jsonl
$PY bench/rq1/prologue_modes.py $OUT/traces/{A1,B1,B2,A2}$TAG-code-edit
echo "[abba-rq1] done $(date +%T)"
