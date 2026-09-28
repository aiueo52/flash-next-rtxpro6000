# G1 isolated environment (private flashinfer copy + private JIT cache)
# source this, never export these into a production shell.
export CUDA_HOME=$HOME/tools/mamba/envs/cuda13
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH="$HOME/tools/sglang-rtxpro6000/.venv/bin:$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++
export TORCH_CUDA_ARCH_LIST=12.0
export SGLANG_CACHE_DIR=$HOME/.cache/sglang-g1
export PYTHONPATH=$HOME/tools/flashinfer-g1:$HOME/tools/flash-next-bench/bench:$HOME/tools/sglang-rtxpro6000/python${PYTHONPATH:+:$PYTHONPATH}
export MOE_BENCH_MARKER=/tmp/gpu-handover-g1
