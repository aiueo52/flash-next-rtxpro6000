#!/bin/bash
# xa1 GPU job body. Run only through the GPU lock, e.g.
#   flock -w 28800 ~/.gpu.lock systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 \
#     bash ~/tools/flash-next-bench/bench/xa1/gpu_job.sh job1 --phases validate,sweep,main,profile
# Args: <label> [args...]; XA1_MODULE picks the module (default xa1.bench; xa1.ablate),
# XA1_TIMEOUT the python timeout in seconds (default 540).
# Production fork sglang (read only), private Triton/FlashInfer caches from env.sh (the prod
# XQA .so is copied, never built), PDL on as in production.
set -u
label=${1:?label}
shift
source "$HOME/tools/flash-next-bench/bench/xa1/env.sh"
runs=$HOME/tools/flash-next-bench/runs/xa1
mkdir -p "$runs"
cd "$HOME/tools/flash-next-bench/bench" || exit 1
{
  date -Is
  nvidia-smi --query-gpu=memory.used,memory.total,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu --format=csv
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
  # The whole lock hold stays under 10 minutes (XA1_TIMEOUT splits it when one hold runs two).
  timeout "${XA1_TIMEOUT:-540}" python -m "${XA1_MODULE:-xa1.bench}" --label "$label" "$@"
  echo "exit=$?"
  nvidia-smi --query-gpu=memory.used,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader
  date -Is
} 2>&1 | tee "$runs/$label.log"
