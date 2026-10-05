# fc-moe isolated env: production venv FlashInfer (read-only) + private JIT/autotune cache copy.
# Source this, never export these into a production shell.
export CUDA_HOME=$HOME/tools/mamba/envs/cuda13
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH="/home/user/tools/flash-next-bench/bench/fc_moe/guardbin:/home/user/tools/sglang-rtxpro6000/.venv/bin:$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++
export MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST=12.0
export SGLANG_CACHE_DIR=/home/user/.cache/sglang-fcmoe
export PYTHONPATH=/home/user/tools/flash-next-bench/bench:/home/user/tools/sglang-rtxpro6000/python${PYTHONPATH:+:$PYTHONPATH}
# production defaults (serve-fast.sh): pack on, folds on (csrc default), prune tau 0.08 in-prologue
export FLASHINFER_MOE_PACK_GROUPS=${FLASHINFER_MOE_PACK_GROUPS-1}
export FLASHINFER_MOE_PRUNE_SINGLETON_TAU=${FLASHINFER_MOE_PRUNE_SINGLETON_TAU-0.08}
export FLASHINFER_CUDA_ARCH_LIST="12.0f"
export FLASHINFER_WORKSPACE_BASE=$SGLANG_CACHE_DIR
