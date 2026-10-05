#!/bin/bash
# abba_xa.sh [tag] -- QSA decode attention off (XQA) vs on (SGLANG_OPT_TRITON_DECODE_ATTN=1) in ORDER
# (default ABBA: A1 off, B1 on, B2 on, A2 off; ORDER="A1:off B1:on B2:on A2:off B3:on A3:off A4:off B4:on"
# for 8 server starts, which the expected ~1-2% per step needs: STACK spec, one start moves ms/step by a
# few percent). Each arm takes its own GPU-lock hold (bench/xa1/arm_xa.sh), then the ANCOVA per sampling
# mode (bench/stats/ancova_ab.py; the arm-level line with 5+ arms), the paired table for the 4-arm default
# (bench/xa1/xa_table.py) and the greedy output match rates (prof/agree_cmp.py).
# Primary metric: ms per output token. "on" also fixes the index-shared draft rows (holes mid-row, see the
# spec), so drafts and with them tok/step can change; target numerics change too (another kernel).
# Copied from bench/dg1/abba_dg.sh (2026-10-01); NOT RUN YET.
set -uo pipefail
cd ~/tools/flash-next-bench
TAG=${1:-}; OUT=${OUTDIR:-runs/xa1}
ORDER=${ORDER:-A1:off B1:on B2:on A2:off}
PY=~/tools/sglang-rtxpro6000/.venv/bin/python
for arm in $ORDER; do
  L=${arm%%:*}$TAG; M=${arm##*:}
  echo "[abba-xa] $L ($M) queued $(date +%T)"
  flock -w 28800 ~/.gpu.lock systemd-run --user --scope -q -p MemoryMax=110G -p MemorySwapMax=0 \
      env OUTDIR=$OUT bash bench/xa1/arm_xa.sh "$L" "$M" || echo "[abba-xa] arm $L rc=$?"
  sleep 30   # let other lock waiters in between arms
done
LABELS=$(for arm in $ORDER; do echo -n "${arm%%:*}$TAG "; done)
echo "== ancova (probe rows, per sampling mode)"
$PY bench/stats/ancova_ab.py $(for L in $LABELS; do echo -n "$L=$OUT/$L-probe.jsonl "; done) 2>&1
if [ "$ORDER" = "A1:off B1:on B2:on A2:off" ]; then
  echo "== paired table"
  $PY bench/xa1/xa_table.py $OUT/A1$TAG-probe.jsonl $OUT/B1$TAG-probe.jsonl $OUT/B2$TAG-probe.jsonl \
      $OUT/A2$TAG-probe.jsonl
fi
# Greedy match rate over the first 256 tokens: the within-arm repeat floor is printed first; A1 vs A2
# and B1 vs B2 are the between-server floors, A vs B the change.
for pair in "A1 A2" "B1 B2" "A1 B1" "A2 B2"; do
  set -- $pair
  [ -e $OUT/$1$TAG-greedy.json ] && [ -e $OUT/$2$TAG-greedy.json ] || continue
  echo "[abba-xa] greedy match $1 vs $2"
  $PY prof/agree_cmp.py $OUT/$1$TAG-greedy.json $OUT/$2$TAG-greedy.json 256
done
echo "[abba-xa] done $(date +%T)"
