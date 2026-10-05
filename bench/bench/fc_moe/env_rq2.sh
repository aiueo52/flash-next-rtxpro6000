# fc-moe RQ2 env: fc_moe/env.sh, but FlashInfer from ~/tools/flashinfer-p3 (python identical to p2; the header
# has the RQ2 edit, FC_fc-moe 1d) with the variant's private JIT cache ~/.cache/sglang-rq2u$RQ2_U, built by
# bench/fc_moe/build_rq2.sh. Set RQ2_U to a built variant (1, 2, 1h, 2h) and source this, never in a production
# shell.
source ~/tools/flash-next-bench/bench/fc_moe/env.sh
export SGLANG_CACHE_DIR=/home/user/.cache/sglang-rq2u${RQ2_U:?set RQ2_U to a build_rq2.sh variant}
export FLASHINFER_WORKSPACE_BASE=$SGLANG_CACHE_DIR
[ -s $SGLANG_CACHE_DIR/.cache/flashinfer/0.6.17/120f/cached_ops/fused_moe_120/fused_moe_120.so ] ||
  echo "env_rq2: no fused_moe_120.so in $SGLANG_CACHE_DIR" >&2
export PYTHONPATH=/home/user/tools/flashinfer-p3:$PYTHONPATH
