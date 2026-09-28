# source this before running the benches: `. bench/w8a16v2/env.sh`
export CUDA_HOME=/home/user/tools/mamba/envs/cuda13
export PATH=/home/user/tools/sglang-rtxpro6000/.venv/bin:$CUDA_HOME/bin:$PATH
export TRITON_CACHE_DIR=/home/user/tools/sglang-rtxpro6000/.cache/triton
export PYTHONPATH=/home/user/tools/sglang-qsa-ring/python
