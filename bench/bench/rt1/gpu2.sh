#!/bin/bash
# RT1 session 2 (one GPU-lock hold, 110G scope): the bit-exact check again, now with the worktree's
# server module itself and the nan-rank case, then an in-server smoke of SGLANG_ROUTER_FAST_TOPK=1.
#   flock -w 28800 ~/.gpu.lock systemd-run --user --scope -q -p MemoryMax=110G -p MemorySwapMax=0 \
#       bash bench/rt1/gpu2.sh > runs/rt1/smoke-rt.log 2>&1
cd ~/tools/flash-next-bench/bench
( source rt1/env.sh; python -m rt1.route_bench --check ) 2>&1 | tee ../runs/rt1/check2.jsonl
grep -q '"check": "PASS"' ../runs/rt1/check2.jsonl || { echo "[rt1] check failed; no smoke"; exit 1; }
cd ~/tools/flash-next-bench
OUTDIR=runs/rt1 bash bench/rt1/arm_rt.sh smoke-rt on 0
