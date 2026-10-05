#!/bin/bash
# abba_dt.sh [tag] [bmode: on|conf, default on] -- draft-tail glue off vs on, ABBA order (A1 off, B1 on,
# B2 on, A2 off); each arm takes its own GPU-lock hold (bench/dt1/arm_dt.sh), then the paired table
# (bench/dt1/dt_table.py) and the per-arm trace summaries (bench/dt1/dt_trace.py).
# `on` is bit-exact, so greedy tok/step should match A; `conf` moves p0 by ~1e-6 relative, which can
# flip a rare adaptive width decision. Primary metric either way: ms per output token per request.
# Copied from bench/dg1/abba_dg.sh (2026-10-01); NOT RUN YET.
set -uo pipefail
cd ~/tools/flash-next-bench
TAG=${1:-}; BMODE=${2:-on}; OUT=${OUTDIR:-runs/dt1}
PY=~/tools/sglang-rtxpro6000/.venv/bin/python
for arm in A1:off B1:$BMODE B2:$BMODE A2:off; do
  L=${arm%%:*}$TAG; M=${arm##*:}
  echo "[abba-dt] $L ($M) queued $(date +%T)"
  flock -w 28800 ~/.gpu.lock systemd-run --user --scope -q -p MemoryMax=110G -p MemorySwapMax=0 \
      env OUTDIR=$OUT bash bench/dt1/arm_dt.sh "$L" "$M" || echo "[abba-dt] arm $L rc=$?"
  sleep 30   # let other lock waiters in between arms
done
$PY bench/dt1/dt_table.py $OUT/A1$TAG-probe.jsonl $OUT/B1$TAG-probe.jsonl $OUT/B2$TAG-probe.jsonl $OUT/A2$TAG-probe.jsonl
for L in A1 B1 B2 A2; do
  echo "== trace $L$TAG"; $PY bench/dt1/dt_trace.py $OUT/traces/$L$TAG-code-edit
done
echo "[abba-dt] done $(date +%T)"
