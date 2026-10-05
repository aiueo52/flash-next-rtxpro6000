#!/bin/bash
# Usage: jitprobe.sh <sglang worktree>   (CPU only; reads the cache, builds nothing)
# Same compile env as serve-local.sh, so the deps hashes resolve against the same headers.
set -eu
WT=$(realpath "$1")
HERE=$(dirname "$(realpath "$0")")
export CUDA_HOME=$HOME/tools/mamba/envs/cuda13 CUDACXX=$HOME/tools/mamba/envs/cuda13/bin/nvcc
export PATH="$HOME/tools/sglang-rtxpro6000/.venv/bin:$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++ TORCH_CUDA_ARCH_LIST=12.0
export SGLANG_JIT_CACHE_DIR=$WT/.cache/jit PYTHONPATH=$WT/python CUDA_VISIBLE_DEVICES=""
cd "$WT"
exec python "$HERE/jitprobe.py" "$WT"
