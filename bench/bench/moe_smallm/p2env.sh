# P2 isolated environment: private flashinfer copy (a0+a3+g2-1+g2-2+g1-pack-only + p2-prune-in-prologue),
# private JIT cache, and the sglang-p2 worktree (fable/prune-in-prologue) ahead of the venv's sglang.
export CUDA_HOME=$HOME/tools/mamba/envs/cuda13
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH="$HOME/tools/sglang-rtxpro6000/.venv/bin:$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++
export TORCH_CUDA_ARCH_LIST=12.0
export SGLANG_CACHE_DIR=$HOME/.cache/sglang-p2
export PYTHONPATH=$HOME/tools/flashinfer-p2:$HOME/tools/sglang-p2/python:$HOME/tools/flash-next-bench/bench${PYTHONPATH:+:$PYTHONPATH}
export FLASHINFER_MOE_PACK_GROUPS=${FLASHINFER_MOE_PACK_GROUPS:-1}
