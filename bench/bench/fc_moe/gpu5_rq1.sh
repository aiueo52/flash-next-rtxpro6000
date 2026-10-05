#!/bin/bash
# fc-moe GPU session 5 (2026-10-01): RQ1 = the fused prologue with its expand quantize loop rolled
# (~/tools/flashinfer-p2, FC_fc-moe 4b.1c) against production. Prologue warm and cold (256 MB read,
# then the call's inputs and the layer's static data) at T=1/4/16 in ABBA order, then a bit-exact check.
# Run under flock + a 40G scope. Appends to runs/fc_moe/s5.jsonl.
set -u
cd ~/tools/flash-next-bench/bench
R=~/tools/flash-next-bench/runs/fc_moe
F="--flush-mb 256 --touch-inputs --touch-static"
nvidia-smi --query-gpu=memory.used,clocks.sm,clocks.mem,power.draw --format=csv,noheader
date +%T
for arm in prod rq1 rq1b prodb; do
  (
    if [[ $arm == rq1* ]]; then source fc_moe/env_rq1.sh; else source fc_moe/env.sh; fi
    PY="python -m fc_moe.fc_moe_bench --T 1,4,16 --out $R/s5.jsonl"
    $PY --label ${arm}_warm 2>&1 | grep -E '^\{|\[fc\]|Error|error'
    $PY $F --label ${arm}_cold 2>&1 | grep -E '^\{|Error|error'
  )
done
(source fc_moe/env.sh; python -m fc_moe.dump_out --out $R/dump-prod.pt 2>&1 | grep -E '\[dump\]|Error|error')
(source fc_moe/env_rq1.sh; python -m fc_moe.dump_out --out $R/dump-rq1.pt 2>&1 | grep -E '\[dump\]|Error|error')
(source fc_moe/env.sh; python -m fc_moe.dump_out --compare $R/dump-prod.pt $R/dump-rq1.pt)
date +%T
