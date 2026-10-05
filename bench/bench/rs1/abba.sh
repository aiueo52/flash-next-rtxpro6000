#!/bin/bash
# abba.sh [tag] -- RS vs target-only, ABBA order (A1 to, B1 rs, B2 rs, A2 to); each arm takes its own
# GPU-lock hold (fresh server, LM-Studio sampling grid then greedy grid), then paired analysis per mode.
# Extra server env for both arms: SERVER_ENV=...; other worktree: WT=...
set -uo pipefail
cd ~/tools/flash-next-bench
TAG=${1:-}; OUT=${OUTDIR:-runs/rs1}
PY=~/tools/sglang-rtxpro6000/.venv/bin/python
for arm in A1:to B1:rs B2:rs A2:to; do
  L=${arm%%:*}$TAG; M=${arm##*:}
  echo "[abba] $L ($M) queued $(date +%T)"
  flock -w 28800 ~/.gpu.lock systemd-run --user --scope -q -p MemoryMax=110G -p MemorySwapMax=0 \
      bash bench/rs1/arm.sh "$L" "$M" lmstudio,greedy || echo "[abba] arm $L failed rc=$?"
  sleep 30   # let other lock waiters in between arms
done
for S in lmstudio greedy; do
  echo "== paired $S"
  $PY -B bench/stats/paired_ab.py --schedule ABBA \
      --arms $OUT/A1$TAG-$S.jsonl $OUT/B1$TAG-$S.jsonl $OUT/B2$TAG-$S.jsonl $OUT/A2$TAG-$S.jsonl \
      --out $OUT/paired$TAG-$S.json 2>&1 | tail -25
done
echo "[abba] done $(date +%T)"
