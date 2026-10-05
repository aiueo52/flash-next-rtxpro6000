#!/bin/bash
# abba_sv.sh [tag] -- sparse vs dense target-only verify, ABBA order (A1 dense, B1 sparse, B2 sparse,
# A2 dense); each arm takes its own GPU-lock hold (bench/sv/arm_sv.sh), then paired analysis per
# sampling mode and the clipped R.eager per step from each arm's traces.
# Only LM-Studio sampling requests take the sparse path; the greedy grid is the arm-noise control.
set -uo pipefail
cd ~/tools/flash-next-bench
TAG=${1:-}; OUT=${OUTDIR:-runs/sv}
PY=~/tools/sglang-rtxpro6000/.venv/bin/python
for arm in A1:dense B1:sparse B2:sparse A2:dense; do
  L=${arm%%:*}$TAG; M=${arm##*:}
  echo "[abba-sv] $L ($M) queued $(date +%T)"
  flock -w 28800 ~/.gpu.lock systemd-run --user --scope -q -p MemoryMax=110G -p MemorySwapMax=0 \
      env OUTDIR=$OUT bash bench/sv/arm_sv.sh "$L" "$M" lmstudio,greedy || echo "[abba-sv] arm $L rc=$?"
  sleep 30   # let other lock waiters in between arms
done
for S in lmstudio greedy; do
  echo "== paired $S"
  $PY -B bench/stats/paired_ab.py --schedule ABBA \
      --arms $OUT/A1$TAG-$S.jsonl $OUT/B1$TAG-$S.jsonl $OUT/B2$TAG-$S.jsonl $OUT/A2$TAG-$S.jsonl \
      --out $OUT/paired$TAG-$S.json 2>&1 | tail -25
  $PY bench/sv/paired_table.py $OUT/paired$TAG-$S.json
done
echo "== clipped R.eager (us/step)"
$PY prof/fc_eager_clip.py $OUT/traces/*$TAG-* 2>&1
echo "[abba-sv] done $(date +%T)"
