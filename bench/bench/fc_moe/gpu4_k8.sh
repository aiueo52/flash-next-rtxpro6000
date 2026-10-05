#!/bin/bash
# fc-moe GPU session 4 (2026-10-01): is the fused prologue's cold penalty its own code?
# Session 3: a T=1 dummy call of the same layer removes the whole ~3.5 us penalty, warming the
# inputs and static tensors does not. A top-k 8 dummy runs another instantiation (other code) but
# shares the workspace pool, the layer's static data and the module's code pages. A k=10 dummy with its
# own workspace buffer shares the code and static data but not the measured call's pool workspace.
#   k8 slow, ownws fast -> the k=10 kernel's own instruction lines
#   k8 fast, ownws slow -> the per-call workspace (lines or TLB)
#   both fast           -> layer static data that --touch-static misses (the layer-5 dummy checks it)
# Run under flock + a 40G scope. Appends to runs/fc_moe/s4.jsonl.
set -u
source ~/tools/flash-next-bench/bench/fc_moe/env.sh
cd ~/tools/flash-next-bench/bench
O=~/tools/flash-next-bench/runs/fc_moe/s4.jsonl
PY="python -m fc_moe.fc_moe_bench --T 1,4 --out $O"
F="--flush-mb 256 --touch-inputs --touch-static"
nvidia-smi --query-gpu=memory.used,clocks.sm,clocks.mem,power.draw --format=csv,noheader
date +%T
$PY --label warm 2>&1 | grep -E '^\{|\[fc\]|Error|error'
$PY $F --label flush_in_static 2>&1 | grep -E '^\{|Error|error'
$PY $F --dummy-call --label dummy_k10 2>&1 | grep -E '^\{|Error|error'
$PY $F --dummy-call --dummy-k 8 --label dummy_k8 2>&1 | grep -E '^\{|Error|error'
$PY $F --dummy-call --dummy-own-ws --label dummy_ownws 2>&1 | grep -E '^\{|Error|error'
$PY $F --dummy-call --dummy-layer 5 --label dummy_L5 2>&1 | grep -E '^\{|Error|error'
$PY $F --dummy-call --dummy-k 4 --label dummy_k4 2>&1 | grep -E '^\{|Error|error'
date +%T
