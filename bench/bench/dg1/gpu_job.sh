#!/bin/bash
# dg1 GPU job body. Run only through the GPU lock, e.g.
#   flock -w 28800 ~/.gpu.lock systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 \
#     bash -c 'source ~/tools/flash-next-bench/bench/fc_moe/env.sh; bash ~/tools/flash-next-bench/bench/dg1/gpu_job.sh job1 --sweep'
# Args: <label> [dg1_bench args...]. Worktree sglang first; production PDL setting; private Triton cache.
set -u
label=${1:?label}
shift
export PYTHONPATH=$HOME/tools/sglang-dg/python:$PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1
export SGLANG_TRITON_PDL=1
export TRITON_CACHE_DIR=$HOME/.cache/dg1-triton
runs=$HOME/tools/flash-next-bench/runs/dg1
mkdir -p "$runs"
cd "$HOME/tools/flash-next-bench/bench" || exit 1
{
  date -Is
  nvidia-smi --query-gpu=memory.used,memory.total,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu --format=csv
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
  # 280 s keeps the whole lock hold under the 5-minute budget.
  timeout 280 python -m dg1.dg1_bench --label "$label" "$@"
  echo "exit=$?"
  nvidia-smi --query-gpu=memory.used,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader
  date -Is
} 2>&1 | tee "$runs/$label.log"
