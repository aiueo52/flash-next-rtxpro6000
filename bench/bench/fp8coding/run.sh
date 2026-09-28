#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
export PYTHONDONTWRITEBYTECODE=1
exec nice -n 19 $HOME/tools/sglang-rtxpro6000/.venv/bin/python survey.py "$@"
