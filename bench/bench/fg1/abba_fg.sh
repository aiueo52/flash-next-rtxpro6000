#!/bin/bash
# abba_fg.sh [tag] -- GDN front overlap off vs on, ABBA order (A1 off, B1 on, B2 on, A2 off); each arm takes
# its own GPU-lock hold (bench/fg1/arm_fg.sh), then the paired ms/step table (bench/fg1/fg_table.py).
# The change is bit-exact, so greedy requests repeat the same steps in every arm: greedy ms/step is the
# primary metric and tok/step must not move. Sampling rows vary with the draws and are secondary.
set -uo pipefail
cd ~/tools/flash-next-bench
TAG=${1:-}; OUT=${OUTDIR:-runs/fg1}
PY=~/tools/sglang-rtxpro6000/.venv/bin/python
for arm in A1:off B1:on B2:on A2:off; do
  L=${arm%%:*}$TAG; M=${arm##*:}
  echo "[abba-fg] $L ($M) queued $(date +%T)"
  flock -w 28800 ~/.gpu.lock systemd-run --user --scope -q -p MemoryMax=110G -p MemorySwapMax=0 \
      env OUTDIR=$OUT bash bench/fg1/arm_fg.sh "$L" "$M" || echo "[abba-fg] arm $L rc=$?"
  sleep 30   # let other lock waiters in between arms
done
$PY bench/fg1/fg_table.py $OUT/A1$TAG-probe.jsonl $OUT/B1$TAG-probe.jsonl $OUT/B2$TAG-probe.jsonl $OUT/A2$TAG-probe.jsonl
echo "[abba-fg] done $(date +%T)"
