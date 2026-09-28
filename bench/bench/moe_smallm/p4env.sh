# P4 isolated environment: private flashinfer copy (a0+a3+g2-1+g2-2+g1-pack-only + p2-prune-in-prologue + p4-contrib-prune),
# private JIT cache, and the sglang-p4 worktree (codex/p4-contrib-prune) ahead of the venv's sglang.
export CUDA_HOME=$HOME/tools/mamba/envs/cuda13
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH="$HOME/tools/sglang-rtxpro6000/.venv/bin:$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++
export TORCH_CUDA_ARCH_LIST=12.0
export SGLANG_CACHE_DIR=$HOME/.cache/sglang-p4
export PYTHONPATH=$HOME/tools/flashinfer-p4:$HOME/tools/sglang-p4/python:$HOME/tools/flash-next-bench/bench${PYTHONPATH:+:$PYTHONPATH}
export FLASHINFER_MOE_PACK_GROUPS=${FLASHINFER_MOE_PACK_GROUPS:-1}
export SGLANG_MOE_PRUNE_NORM_MANIFEST=$HOME/tools/flash-next-bench/prune/p4-manifest.json
export MEM_FRACTION=0.920 W4_MEM_FRACTION=0.920 W16_MEM_FRACTION=0.920 WA_MEM_FRACTION=0.920 SERVE_DISPLAY_HZ=
export PYTHONDONTWRITEBYTECODE=1
