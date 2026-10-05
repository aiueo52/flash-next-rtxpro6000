#!/bin/bash
# fc-moe GPU session 3 (2026-10-01): which cold access makes the routing prologue slow?
# Session 2: a 256 MB read before each call adds ~4 us to the prologue at T=1/4, like the in-server
# slow mode. The fused prologue (k=10, fold) is 105 KB of SASS (79 KB unfolded) and reads ~1.3 KB of
# kernel parameters, so it can miss on its inputs, the layer's small static tensors, its code or
# its parameters. Each variant re-warms one more of these after the flush:
#   +in      the call's hidden states and top-k ids/weights (in-server: just written, warm)
#   +static  the layer's input global scales and alphas
#   +dummy   one T=1 call of the same layer first: code and static data warm, own params still cold
# Run under flock + a 40G scope. Appends to runs/fc_moe/s3.jsonl.
set -u
source ~/tools/flash-next-bench/bench/fc_moe/env.sh
cd ~/tools/flash-next-bench/bench
O=~/tools/flash-next-bench/runs/fc_moe/s3.jsonl
PY="python -m fc_moe.fc_moe_bench --T 1,4 --out $O"
F="--flush-mb 256"
nvidia-smi --query-gpu=memory.used,clocks.sm,clocks.mem,power.draw --format=csv,noheader
date +%T
$PY --label warm 2>&1 | grep -E '^\{|\[fc\]|Error|error'
$PY $F --label flush 2>&1 | grep -E '^\{|Error|error'
$PY $F --touch-inputs --label flush_in 2>&1 | grep -E '^\{|Error|error'
$PY $F --touch-inputs --touch-static --label flush_in_static 2>&1 | grep -E '^\{|Error|error'
$PY $F --touch-inputs --touch-static --dummy-call --label flush_in_static_dummy 2>&1 | grep -E '^\{|Error|error'
$PY $F --dummy-call --label flush_dummy 2>&1 | grep -E '^\{|Error|error'
export FLASHINFER_MOE_FOLD_EXPAND=0
$PY --label noexp_warm 2>&1 | grep -E '^\{|\[fc\]|Error|error'
$PY $F --label noexp_flush 2>&1 | grep -E '^\{|Error|error'
$PY $F --touch-inputs --touch-static --label noexp_flush_in_static 2>&1 | grep -E '^\{|Error|error'
$PY $F --touch-inputs --touch-static --dummy-call --label noexp_flush_in_static_dummy 2>&1 | grep -E '^\{|Error|error'
unset FLASHINFER_MOE_FOLD_EXPAND
$PY --label warm2 2>&1 | grep -E '^\{|Error|error'
date +%T
