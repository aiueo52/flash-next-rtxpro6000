#!/bin/bash
# abba_stack.sh [tag] -- production (every flag off) vs the stacked changes in ORDER (default ABBA: A1 off, B1 on,
# B2 on, A2 off; e.g. ORDER="A1:off B1:on B2:on A2:off B3:on A3:off A4:off B4:on" for 8 server starts); each arm
# takes its own GPU-lock hold (bench/stack/arm_stack.sh, STACK_ENV / STACK_FI / SERVER_ENV / PROMPT_LIMIT pass
# through), then the ANCOVA per sampling mode (bench/stats/ancova_ab.py; arm level with 5+ arms), the paired BN1
# analysis for the 4-arm default, and the clipped R.eager of the sampling traces.
# The stack changes draft numerics (DG1) and the sampling verify (SV1), so tok/step may move: read ms/token.
set -uo pipefail
cd ~/tools/flash-next-bench
TAG=${1:-}; OUT=${OUTDIR:-runs/stack}
ORDER=${ORDER:-A1:off B1:on B2:on A2:off}
PY=~/tools/sglang-rtxpro6000/.venv/bin/python
for arm in $ORDER; do
  L=${arm%%:*}$TAG; M=${arm##*:}
  echo "[abba-stack] $L ($M) queued $(date +%T)"
  flock -w 28800 ~/.gpu.lock systemd-run --user --scope -q -p MemoryMax=110G -p MemorySwapMax=0 \
      env OUTDIR=$OUT bash bench/stack/arm_stack.sh "$L" "$M" lmstudio,greedy || echo "[abba-stack] arm $L rc=$?"
  sleep 30   # let other lock waiters in between arms
done
LABELS=$(for arm in $ORDER; do echo -n "${arm%%:*}$TAG "; done)
for S in lmstudio greedy; do
  echo "== ancova $S"
  $PY bench/stats/ancova_ab.py $(for L in $LABELS; do echo -n "$L=$OUT/$L-$S.jsonl "; done) 2>&1
  [ "$ORDER" = "A1:off B1:on B2:on A2:off" ] || continue
  echo "== paired $S"
  $PY -B bench/stats/paired_ab.py --schedule ABBA \
      --arms $OUT/A1$TAG-$S.jsonl $OUT/B1$TAG-$S.jsonl $OUT/B2$TAG-$S.jsonl $OUT/A2$TAG-$S.jsonl \
      --out $OUT/paired$TAG-$S.json 2>&1 | tail -25
  $PY bench/sv/paired_table.py $OUT/paired$TAG-$S.json
done
echo "== clipped R.eager (us/step, sampling traces)"
$PY prof/fc_eager_clip.py $(for L in $LABELS; do echo -n "$OUT/traces/$L-code-edit "; done) \
    $(for L in $LABELS; do echo -n "$OUT/traces/$L-prose-en "; done) 2>&1
echo "[abba-stack] done $(date +%T)"
