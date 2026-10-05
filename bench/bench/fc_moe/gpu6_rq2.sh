#!/bin/bash
# fc-moe GPU session 6 (2026-10-01): RQ2 (row staged in shared memory with cp.async, quantize loop 1 or 2
# chunks per trip, "h" = SF write address hoisted out of the loop; ~/tools/flashinfer-p3, FC_fc-moe 1d)
# against RQ1 and production. Prologue warm and cold (256 MB read, then the call's inputs and the layer's
# static data) at T=1/4/16, arms in mirrored order, then the output check of each variant against production.
# Run under flock + a 40G scope. Appends to runs/fc_moe/s6.jsonl.
set -u
cd ~/tools/flash-next-bench/bench
R=~/tools/flash-next-bench/runs/fc_moe
F="--flush-mb 256 --touch-inputs --touch-static"
nvidia-smi --query-gpu=memory.used,clocks.sm,clocks.mem,power.draw --format=csv,noheader
date +%T
for arm in prod rq1 u1 u2 u1h u2h u2hb u1hb u2b u1b rq1b prodb; do
  (
    case $arm in
      rq1*) source fc_moe/env_rq1.sh ;;
      u*) V=${arm#u}; RQ2_U=${V%b} source fc_moe/env_rq2.sh ;;
      *) source fc_moe/env.sh ;;
    esac
    PY="python -m fc_moe.fc_moe_bench --T 1,4,16 --out $R/s6.jsonl"
    $PY --label ${arm}_warm 2>&1 | grep -E '^\{|\[fc\]|Error|error'
    $PY $F --label ${arm}_cold 2>&1 | grep -E '^\{|Error|error'
  )
done
[ -s $R/dump-prod.pt ] || (source fc_moe/env.sh; python -m fc_moe.dump_out --out $R/dump-prod.pt 2>&1 | grep -E '\[dump\]|Error|error')
for V in 1 2 1h 2h; do
  (RQ2_U=$V source fc_moe/env_rq2.sh; python -m fc_moe.dump_out --out $R/dump-rq2u$V.pt 2>&1 | grep -E '\[dump\]|Error|error')
  (source fc_moe/env.sh; python -m fc_moe.dump_out --compare $R/dump-prod.pt $R/dump-rq2u$V.pt)
done
date +%T
