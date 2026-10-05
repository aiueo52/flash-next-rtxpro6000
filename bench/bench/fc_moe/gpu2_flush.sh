#!/bin/bash
# fc-moe GPU session 2 (2026-10-01): does L2 eviction between calls make the prologue slow?
# In-server the prologue is 8.3 us (T=4) or 12.5 us, and 11.5-12.4 us for every draft T=1 call, while
# the microbench (L2 warm: one layer's experts, 16 calls back to back) gives 7.5 / 8.4 us.
# A serial read before each call (same stream, no overlap) evicts L2 like the server's weight streaming.
# Run under flock + a 40G scope. Appends to runs/fc_moe/s2.jsonl.
set -u
source ~/tools/flash-next-bench/bench/fc_moe/env.sh
cd ~/tools/flash-next-bench/bench
O=~/tools/flash-next-bench/runs/fc_moe/s2.jsonl
PY="python -m fc_moe.fc_moe_bench"
nvidia-smi --query-gpu=memory.used,clocks.sm,clocks.mem,power.draw --format=csv,noheader
date +%T
$PY --T 1,4,16 --label prod_fref --out $O 2>&1 | grep -E '^\{|\[fc\]|Error|error'
$PY --T 1,4,16 --flush-mb 256 --label flush256 --out $O 2>&1 | grep -E '^\{|Error'
$PY --T 1,4,16 --flush-mb 64 --label flush64 --out $O 2>&1 | grep -E '^\{|Error'
$PY --T 1,4,16 --flush-mb 256 --side-mb 8 --label flush256_side8 --out $O 2>&1 | grep -E '^\{|Error'
$PY --T 1,4,16 --label prod_fref2 --out $O 2>&1 | grep -E '^\{|Error'
date +%T
