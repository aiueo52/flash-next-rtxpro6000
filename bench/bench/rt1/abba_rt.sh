#!/bin/bash
# abba_rt.sh [tag] -- Triton router vs packed-key router (SGLANG_ROUTER_FAST_TOPK=1), ABBA order (A1 off,
# B1 on, B2 on, A2 off); each arm takes its own GPU-lock hold (bench/rt1/arm_rt.sh), then the paired
# ms/step table (bench/fg1/fg_table.py layout, copied as bench/rt1/rt_table.py). The router is
# bit-exact, so greedy requests repeat the same steps in every arm (up to the MoE finalize atomics):
# greedy ms/step is the primary metric and tok/step must not move.
set -uo pipefail
cd ~/tools/flash-next-bench
TAG=${1:-}; OUT=${OUTDIR:-runs/rt1}
PY=~/tools/sglang-rtxpro6000/.venv/bin/python
for arm in A1:off B1:on B2:on A2:off; do
  L=${arm%%:*}$TAG; M=${arm##*:}
  echo "[abba-rt] $L ($M) queued $(date +%T)"
  flock -w 28800 ~/.gpu.lock systemd-run --user --scope -q -p MemoryMax=110G -p MemorySwapMax=0 \
      env OUTDIR=$OUT bash bench/rt1/arm_rt.sh "$L" "$M" || echo "[abba-rt] arm $L rc=$?"
  sleep 30   # let other lock waiters in between arms
done
$PY bench/rt1/rt_table.py $OUT/A1$TAG-probe.jsonl $OUT/B1$TAG-probe.jsonl $OUT/B2$TAG-probe.jsonl $OUT/A2$TAG-probe.jsonl
echo "[abba-rt] done $(date +%T)"
