#!/bin/bash
# abba_rs2d.sh [tag] -- the RS line's server A/B (specs/RS2D_DRAFT_SHARPEN_2026-10-01.md 4.3): A = the stack,
# target-only verify; B = the stack + RS + RS2 + B_ENV (RS2d knobs, G1, RS3 BV). Both arms run WT (default
# ~/tools/sglang-rs3) with min_p on (bench/rs2/arm_rs2.sh). B_ENV goes to the B arms only: SGLANG_RS_BLOCK_VERIFY
# without SGLANG_OPT_SPEC_SPARSE_RS stops the server. ORDER, ANCOVA and R.eager as in bench/rs2/abba_rs2.sh.
set -uo pipefail
cd ~/tools/flash-next-bench
TAG=${1:-}; OUT=${OUTDIR:-runs/rs2d-abba}; mkdir -p $OUT
: "${B_ENV:?set B_ENV, e.g. SGLANG_RS_DRAFT_TEMP_SCALE=0.7 SGLANG_RS_GREEDY_FAST=1 SGLANG_RS_BLOCK_VERIFY=1}"
export WT=${WT:-$HOME/tools/sglang-rs3}
ORDER=${ORDER:-A1:to B1:rs B2:rs A2:to B3:rs A3:to A4:to B4:rs}
PY=~/tools/sglang-rtxpro6000/.venv/bin/python
echo "[abba-rs2d] wt=$WT commit=$(git -C $WT log --oneline -1 | cut -c1-10) B_ENV='$B_ENV' order='$ORDER'"
for arm in $ORDER; do
  L=${arm%%:*}$TAG; M=${arm##*:}
  ARM_SERVER_ENV=""; [ "$M" = rs ] && ARM_SERVER_ENV=$B_ENV
  echo "[abba-rs2d] $L ($M) queued $(date +%T)"
  flock -w 28800 ~/.gpu.lock systemd-run --user --scope -q -p MemoryMax=110G -p MemorySwapMax=0 \
      env OUTDIR=$OUT SERVER_ENV="$ARM_SERVER_ENV" bash bench/rs2/arm_rs2.sh "$L" "$M" lmstudio,greedy ||
      echo "[abba-rs2d] arm $L rc=$?"
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
echo "[abba-rs2d] done $(date +%T)"
