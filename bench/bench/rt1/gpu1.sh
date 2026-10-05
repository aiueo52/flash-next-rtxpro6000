#!/bin/bash
# RT1 session 1 (run under the GPU lock): bit-exact check, then chain timing.
cd ~/tools/flash-next-bench/bench
source rt1/env.sh
mkdir -p ../runs/rt1
nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader
python -m rt1.route_bench --check 2>&1 | tee ../runs/rt1/check.jsonl
python -m rt1.route_bench --time --rounds 7 2>&1 | tee ../runs/rt1/time1.jsonl
nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader
