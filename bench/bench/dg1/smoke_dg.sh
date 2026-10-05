#!/bin/bash
# smoke_dg.sh -- one `on` arm without the request grid (N=0) under the GPU lock, then the pass check:
# no server errors, one "enabled" line, no fallback warnings, GEMV kernels in the 20-step trace.
cd ~/tools/flash-next-bench
S=runs/dg1/smoke-dg.log
flock -w 28800 ~/.gpu.lock systemd-run --user --scope -q -p MemoryMax=110G -p MemorySwapMax=0 \
    env OUTDIR=runs/dg1 bash bench/dg1/arm_dg.sh smoke-dg on 0 > $S 2>&1
echo "[smoke-dg] done $(date +%T)"
grep -E "server up|error lines|enabled lines|fallback warnings|gemv_k|cutlass_|died|not ready" $S
if grep -q "error lines in server log: 0" $S && grep -q "draft MoE GEMV enabled lines: [1-9]" $S \
   && grep -q "GEMV fallback warnings: 0" $S && grep -qE "gemv_k1 +count +[1-9]" $S; then
  echo "[smoke-dg] CLEAN"
else
  echo "[smoke-dg] NOT CLEAN"
fi
