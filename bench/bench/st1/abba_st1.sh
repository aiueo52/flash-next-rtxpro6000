#!/bin/bash
# abba_st1.sh [tag] -- ST1's server A/B (specs/ST1_SHARED_TAIL_2026-10-01.md §3): A = the stack (5 flags, RQ2 u2h), B = A +
# SGLANG_ENABLE_QSA_SHARED_TAIL_PREFIX=1, both from ~/tools/sglang-st1 through bench/stack/arm_stack.sh (mode on), in
# ORDER (default ABBA), each arm its own GPU-lock hold; then the ANCOVA per sampling mode (tok/step first), the paired
# BN1 table for the 4-arm default and the clipped R.eager of the sampling traces. min_p off, as in stack8.
set -uo pipefail
cd ~/tools/flash-next-bench
TAG=${1:-}; OUT=${OUTDIR:-runs/st1}; mkdir -p $OUT
ORDER=${ORDER:-A1:a B1:b B2:b A2:a}
PY=~/tools/sglang-rtxpro6000/.venv/bin/python
BASE_ENV="SGLANG_ROUTER_FAST_TOPK=1 SGLANG_OPT_SPEC_SPARSE_VERIFY=1 SGLANG_OPT_SPEC_SPARSE_TOPK=1"
BASE_ENV="$BASE_ENV SGLANG_OPT_GDN_FRONT_OVERLAP=1 SGLANG_OPT_DRAFT_MOE_GEMV=1"
for arm in $ORDER; do
  L=${arm%%:*}$TAG; M=${arm##*:}
  E="$BASE_ENV"; [ "$M" = b ] && E="$E SGLANG_ENABLE_QSA_SHARED_TAIL_PREFIX=1"
  echo "[abba-st1] $L ($M) queued $(date +%T)"
  flock -w 28800 ~/.gpu.lock systemd-run --user --scope -q -p MemoryMax=110G -p MemorySwapMax=0 \
      env OUTDIR=$OUT WT=$HOME/tools/sglang-st1 PORT=8031 PROMPT_LIMIT=${PROMPT_LIMIT:-4} STACK_ENV="$E" \
      bash bench/stack/arm_stack.sh "$L" on lmstudio,greedy || echo "[abba-st1] arm $L rc=$?"
  echo "[abba-st1] $L ST1 log lines: $(grep -c 'tail after the valid prefix' $OUT/serve-$L.log)"
  sleep 30   # let other lock waiters in between arms
done
LABELS=$(for arm in $ORDER; do echo -n "${arm%%:*}$TAG "; done)
for S in lmstudio greedy; do
  echo "== ancova $S"
  $PY bench/stats/ancova_ab.py $(for L in $LABELS; do echo -n "$L=$OUT/$L-$S.jsonl "; done) 2>&1
  [ "$ORDER" = "A1:a B1:b B2:b A2:a" ] || continue
  echo "== paired $S"
  $PY -B bench/stats/paired_ab.py --schedule ABBA \
      --arms $OUT/A1$TAG-$S.jsonl $OUT/B1$TAG-$S.jsonl $OUT/B2$TAG-$S.jsonl $OUT/A2$TAG-$S.jsonl \
      --out $OUT/paired$TAG-$S.json 2>&1 | tail -25
  $PY bench/sv/paired_table.py $OUT/paired$TAG-$S.json
done
echo "== clipped R.eager (us/step, sampling traces)"
$PY prof/fc_eager_clip.py $(for L in $LABELS; do echo -n "$OUT/traces/$L-code-edit "; done) \
    $(for L in $LABELS; do echo -n "$OUT/traces/$L-prose-en "; done) 2>&1
echo "[abba-st1] done $(date +%T)"
