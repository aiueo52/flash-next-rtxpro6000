# Environment for the megakernel (K2) microbenchmarks.
# Usage:  . bench/megakernel/env.sh   then  flock -w 14400 ~/.gpu.lock python -m ...
FORK="$HOME/tools/sglang-rtxpro6000"
export CUDA_HOME=$HOME/tools/mamba/envs/cuda13
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH="$FORK/.venv/bin:$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++
export TORCH_CUDA_ARCH_LIST=12.0
export PYTHONPATH="$FORK/python${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$FORK/.cache/hf" XDG_CACHE_HOME="$FORK/.cache/xdg"
export TRITON_CACHE_DIR="$FORK/.cache/triton" SGLANG_JIT_CACHE_DIR="$FORK/.cache/jit"
export PY="$FORK/.venv/bin/python"
