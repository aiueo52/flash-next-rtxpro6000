#!/bin/bash
# smoke_dt.sh [on|conf, default on] -- one B arm without the request grid (N=0) under the GPU lock,
# then the pass check: no server errors, the "on" log line, the fused select kernels in the 20-step
# trace, no cast between the draft lm_head and the topk1 partial, fewer captured copies per draft
# forward than production (8.17 per forward in runs/sv/traces/A1-code-edit; DT-a + DT-b leave ~1.2), and
# the chain tree kernel in place of build_tree_efficient (DT-t).
cd ~/tools/flash-next-bench
MODE=${1:-on}
S=runs/dt1/smoke-dt-$MODE.log
flock -w 28800 ~/.gpu.lock systemd-run --user --scope -q -p MemoryMax=110G -p MemorySwapMax=0 \
    env OUTDIR=runs/dt1 bash bench/dt1/arm_dt.sh smoke-dt-$MODE $MODE 0 > $S 2>&1
echo "[smoke-dt] done $(date +%T)"
grep -E "server up|error lines|enabled lines|draft forwards|graph_copies|lmhead_to_partial|epilogue|de_tail|select_|tree|died|not ready" $S
copies=$(grep -oP "graph_copies_per_forward +n +[0-9]+ +median +\K[0-9.]+" $S)
if grep -q "error lines in server log: 0" $S && grep -q "draft tail enabled lines: [1-9]" $S \
   && grep -qE "draft_extend_select_partial_kernel count [1-9]" $S \
   && grep -qE "lmhead_to_partial_kernels +n +[1-9][0-9]* +median +0\.00" $S \
   && grep -qE "chain_tree_topk1_kernel +count [1-9]" $S && grep -qE "build_tree_efficient +count 0$" $S \
   && [ -n "$copies" ] && awk -v c="$copies" 'BEGIN { exit !(c < 4) }'; then
  echo "[smoke-dt] CLEAN"
else
  echo "[smoke-dt] NOT CLEAN"
fi
