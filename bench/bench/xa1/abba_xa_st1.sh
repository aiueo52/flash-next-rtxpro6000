#!/bin/bash
# abba_xa_st1.sh [tag] -- XA1's server A/B on top of ST1 (specs/XA1_DECODE_ATTN_2026-10-01.md 7b):
# A = the stack (5 flags, RQ2 u2h) + SGLANG_ENABLE_QSA_SHARED_TAIL_PREFIX=1; B = A + SGLANG_OPT_TRITON_DECODE_ATTN=1,
# both from ~/tools/sglang-xa-st1 through bench/stack/arm_stack.sh (mode on), in ORDER (default 8 starts, the expected
# effect is about 1%), each arm its own GPU-lock hold; per arm the XA1 kernel evidence (xa_trace.py on the greedy
# trace) and the fallback warnings; then the ANCOVA per sampling mode and the clipped R.eager. min_p off, as in stack8.
set -uo pipefail
cd ~/tools/flash-next-bench
TAG=${1:-}; OUT=${OUTDIR:-runs/xa1-st1}; mkdir -p $OUT
ORDER=${ORDER:-A1:a B1:b B2:b A2:a B3:b A3:a A4:a B4:b}
PY=~/tools/sglang-rtxpro6000/.venv/bin/python
BASE_ENV="SGLANG_ROUTER_FAST_TOPK=1 SGLANG_OPT_SPEC_SPARSE_VERIFY=1 SGLANG_OPT_SPEC_SPARSE_TOPK=1"
BASE_ENV="$BASE_ENV SGLANG_OPT_GDN_FRONT_OVERLAP=1 SGLANG_OPT_DRAFT_MOE_GEMV=1 SGLANG_ENABLE_QSA_SHARED_TAIL_PREFIX=1"
for arm in $ORDER; do
  L=${arm%%:*}$TAG; M=${arm##*:}
  E="$BASE_ENV"; [ "$M" = b ] && E="$E SGLANG_OPT_TRITON_DECODE_ATTN=1"
  echo "[abba-xa-st1] $L ($M) queued $(date +%T)"
  flock -w 28800 ~/.gpu.lock systemd-run --user --scope -q -p MemoryMax=110G -p MemorySwapMax=0 \
      env OUTDIR=$OUT WT=$HOME/tools/sglang-xa-st1 PORT=8033 PROMPT_LIMIT=${PROMPT_LIMIT:-4} STACK_ENV="$E" \
      bash bench/stack/arm_stack.sh "$L" on lmstudio,greedy || echo "[abba-xa-st1] arm $L rc=$?"
  echo "[abba-xa-st1] $L ST1 log lines: $(grep -c 'tail after the valid prefix' $OUT/serve-$L.log)," \
       "XA1 fallback warnings: $(grep -c 'SGLANG_OPT_TRITON_DECODE_ATTN: unsupported' $OUT/serve-$L.log)"
  $PY bench/xa1/xa_trace.py $OUT/traces/$L-g-code-edit 2>&1 | sed "s/^/[abba-xa-st1] $L /"
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
echo "[abba-xa-st1] done $(date +%T)"
