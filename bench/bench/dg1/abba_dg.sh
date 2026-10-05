#!/bin/bash
# abba_dg.sh [tag] -- draft MoE GEMV off vs on, ABBA order (A1 off, B1 on, B2 on, A2 off); each arm takes
# its own GPU-lock hold (bench/dg1/arm_dg.sh), then the paired table (bench/dg1/dg_table.py).
# Only draft numerics change, so drafts (and with them tok/step) differ between arms in greedy and
# sampling alike: the primary metric is ms per output token (ms/step / tok/step) per request.
# Copied from bench/fg1/abba_fg.sh (2026-10-01); NOT RUN YET.
set -uo pipefail
cd ~/tools/flash-next-bench
TAG=${1:-}; OUT=${OUTDIR:-runs/dg1}
PY=~/tools/sglang-rtxpro6000/.venv/bin/python
for arm in A1:off B1:on B2:on A2:off; do
  L=${arm%%:*}$TAG; M=${arm##*:}
  echo "[abba-dg] $L ($M) queued $(date +%T)"
  flock -w 28800 ~/.gpu.lock systemd-run --user --scope -q -p MemoryMax=110G -p MemorySwapMax=0 \
      env OUTDIR=$OUT bash bench/dg1/arm_dg.sh "$L" "$M" || echo "[abba-dg] arm $L rc=$?"
  sleep 30   # let other lock waiters in between arms
done
$PY bench/dg1/dg_table.py $OUT/A1$TAG-probe.jsonl $OUT/B1$TAG-probe.jsonl $OUT/B2$TAG-probe.jsonl $OUT/A2$TAG-probe.jsonl
echo "[abba-dg] done $(date +%T)"
