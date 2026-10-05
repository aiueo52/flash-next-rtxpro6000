# xa1 bench env: production venv + production sources (read-only), private Triton / FlashInfer /
# CUDA / inductor caches. Source this; never export these into a production shell.
export PATH="/home/user/tools/sglang-rtxpro6000/.venv/bin:$PATH"
# XA1_SGLANG_PY=~/tools/sglang-xa/python selects the integration worktree (integ_check.py).
export PYTHONPATH=/home/user/tools/flash-next-bench/bench:${XA1_SGLANG_PY:-/home/user/tools/sglang-rtxpro6000/python}${PYTHONPATH:+:$PYTHONPATH}
export PYTHONDONTWRITEBYTECODE=1
export TRITON_CACHE_DIR=$HOME/.cache/xa1-triton
# sglang derives FlashInfer/inductor/CUDA cache dirs from SGLANG_CACHE_DIR (environ.py
# third_party_cache_defaults); all of them stay private here.
export SGLANG_CACHE_DIR=$HOME/.cache/xa1-sglang
export FLASHINFER_WORKSPACE_BASE=$SGLANG_CACHE_DIR
export TORCHINDUCTOR_CACHE_DIR=$SGLANG_CACHE_DIR/inductor
export CUDA_CACHE_PATH=$SGLANG_CACHE_DIR/nv
export FLASHINFER_CUDA_ARCH_LIST=12.0f
# serve-fast.sh: Triton PDL on.
export SGLANG_TRITON_PDL=1
