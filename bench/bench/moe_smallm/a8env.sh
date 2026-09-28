# A8 isolated environment (private flashinfer copy + private JIT cache)
export CUDA_HOME=$HOME/tools/mamba/envs/cuda13
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH="$HOME/tools/sglang-rtxpro6000/.venv/bin:$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++
export TORCH_CUDA_ARCH_LIST=12.0
export SGLANG_CACHE_DIR=$HOME/.cache/sglang-a8
export PYTHONPATH=$HOME/tools/flashinfer-a8:$HOME/tools/flash-next-bench/bench${PYTHONPATH:+:$PYTHONPATH}
