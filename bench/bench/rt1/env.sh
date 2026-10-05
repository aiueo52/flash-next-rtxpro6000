# RT1 bench env: production venv + production sources (read-only), private Triton cache.
# Source this, never export these into a production shell.
export PATH="/home/user/tools/sglang-rtxpro6000/.venv/bin:$PATH"
export PYTHONPATH=/home/user/tools/flash-next-bench/bench:/home/user/tools/sglang-rtxpro6000/python${PYTHONPATH:+:$PYTHONPATH}
export TRITON_CACHE_DIR=/home/user/.cache/rt1-triton
# serve-fast.sh: Triton PDL on, router GEMV on
export SGLANG_TRITON_PDL=1 SGLANG_ROUTER_GEMV=1
