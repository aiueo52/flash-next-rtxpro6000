#!/bin/bash
# fc-moe GPU session 1: prologue env variants, side-stream contention, tactic sweep. Run under flock.
set -u
source ~/tools/flash-next-bench/bench/fc_moe/env.sh
cd ~/tools/flash-next-bench/bench
O=~/tools/flash-next-bench/runs/fc_moe/s1.jsonl
PY="python -m fc_moe.fc_moe_bench"
nvidia-smi --query-gpu=memory.used,clocks.sm,power.draw --format=csv,noheader
date +%T
$PY --T 1,4,8,16 --label prod --out $O 2>&1 | grep -E '^\{|\[fc\]|Error|error' 
FLASHINFER_MOE_PRUNE_SINGLETON_TAU=0 $PY --T 1,4,8,16 --label noprune --out $O 2>&1 | grep -E '^\{|Error'
FLASHINFER_MOE_PACK_GROUPS=0 $PY --T 1,4,16 --label nopack --out $O 2>&1 | grep -E '^\{|Error'
FLASHINFER_MOE_FOLD_EXPAND=0 $PY --T 1,4,16 --label noexpand --out $O 2>&1 | grep -E '^\{|Error'
FLASHINFER_MOE_FOLD_EXPAND=0 FLASHINFER_MOE_PACK_GROUPS=0 FLASHINFER_MOE_PRUNE_SINGLETON_TAU=0 $PY --T 1,4,16 --label g2off_packoff_pruneoff --out $O 2>&1 | grep -E '^\{|Error'
$PY --T 1,4,16 --side-mb 8 --label side8 --out $O 2>&1 | grep -E '^\{|Error'
$PY --T 4,16 --side-mb 32 --label side32 --out $O 2>&1 | grep -E '^\{|Error'
$PY --T 1,4,8,16 --label prod_again --out $O 2>&1 | grep -E '^\{|Error'
date +%T
for T in 4 8 16 1; do
  $PY --T $T --tactics 0-30:30-80 --seconds 0.3 --label sweep --out ~/tools/flash-next-bench/runs/fc_moe/sweep.jsonl 2>&1 | grep -E '^\{' | grep -v error
done
date +%T
nvidia-smi --query-gpu=memory.used,clocks.sm,power.draw --format=csv,noheader
date +%T
python -m fc_moe.draft_gemv_proto --sweep --out ~/tools/flash-next-bench/runs/fc_moe/draft_gemv.jsonl 2>&1 | grep -vE "^\s*$" | tail -30
date +%T
