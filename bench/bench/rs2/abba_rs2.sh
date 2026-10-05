#!/bin/bash
# abba_rs2.sh [tag] -- RS2's server A/B (specs/RS2_SPARSE_RS_2026-10-01.md 6): A = the stack, target-only verify;
# B = the stack + RS + RS2; min_p on in both (bench/rs2/arm_rs2.sh, SERVER_ENV / PROMPT_LIMIT pass through), in ORDER
# (default 8 starts, BAAB-balanced), each arm its own GPU-lock hold; then the ANCOVA per sampling mode and the clipped
# R.eager of the sampling traces. RS changes acceptance, so read t/s and ms/token, not ms/step alone.
set -uo pipefail
cd ~/tools/flash-next-bench
TAG=${1:-}; OUT=${OUTDIR:-runs/rs2}; mkdir -p $OUT
ORDER=${ORDER:-A1:to B1:rs B2:rs A2:to B3:rs A3:to A4:to B4:rs}
PY=~/tools/sglang-rtxpro6000/.venv/bin/python
for arm in $ORDER; do
  L=${arm%%:*}$TAG; M=${arm##*:}
  echo "[abba-rs2] $L ($M) queued $(date +%T)"
  flock -w 28800 ~/.gpu.lock systemd-run --user --scope -q -p MemoryMax=110G -p MemorySwapMax=0 \
      env OUTDIR=$OUT bash bench/rs2/arm_rs2.sh "$L" "$M" lmstudio,greedy || echo "[abba-rs2] arm $L rc=$?"
  sleep 30   # let other lock waiters in between arms
done
LABELS=$(for arm in $ORDER; do echo -n "${arm%%:*}$TAG "; done)
for S in lmstudio greedy; do
  echo "== ancova $S"
  $PY bench/stats/ancova_ab.py $(for L in $LABELS; do echo -n "$L=$OUT/$L-$S.jsonl "; done) 2>&1
done
echo "== clipped R.eager (us/step, sampling traces)"
$PY prof/fc_eager_clip.py $(for L in $LABELS; do echo -n "$OUT/traces/$L-code-edit "; done) \
    $(for L in $LABELS; do echo -n "$OUT/traces/$L-prose-en "; done) 2>&1
echo "[abba-rs2] done $(date +%T)"
