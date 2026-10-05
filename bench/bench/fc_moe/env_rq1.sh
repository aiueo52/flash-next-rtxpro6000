# fc-moe RQ1 env: fc_moe/env.sh, but FlashInfer from the private copy ~/tools/flashinfer-p2 (rolled expand
# quantize loop, FC_fc-moe 4b.1c) with that copy's own JIT cache. Source this, never in a production shell.
source ~/tools/flash-next-bench/bench/fc_moe/env.sh
export SGLANG_CACHE_DIR=/home/user/.cache/sglang-p2
export FLASHINFER_WORKSPACE_BASE=$SGLANG_CACHE_DIR
export PYTHONPATH=/home/user/tools/flashinfer-p2:$PYTHONPATH
