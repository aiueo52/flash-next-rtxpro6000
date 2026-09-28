#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$HOME/tools/sglang-p3/python:$HOME/tools/flash-next-bench/bench
export CUDA_HOME=$HOME/tools/mamba/envs/cuda13
export PATH="$HOME/tools/sglang-rtxpro6000/.venv/bin:$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++ TORCH_CUDA_ARCH_LIST=12.0
export SGLANG_MOE_PRUNE_SINGLETON_TAU=0 FLASHINFER_MOE_PRUNE_SINGLETON_TAU=0 SGLANG_MOE_PRUNE_IN_PROLOGUE=1
export FLASHINFER_MOE_PACK_GROUPS=1 MEM_FRACTION=0.920 W16_MEM_FRACTION=0.920 WA_MEM_FRACTION=0.920 SERVE_DISPLAY_HZ=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec $HOME/tools/sglang-rtxpro6000/.venv/bin/python "$@"
