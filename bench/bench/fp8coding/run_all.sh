#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
export PYTHONDONTWRITEBYTECODE=1
FP8C_PYTHON=$HOME/tools/sglang-rtxpro6000/.venv/bin/python
nice -n 19 "$FP8C_PYTHON" survey.py
nice -n 19 "$FP8C_PYTHON" validate_huffman.py
nice -n 19 "$FP8C_PYTHON" audit_conversion.py
nice -n 19 "$FP8C_PYTHON" time_weights.py
nice -n 19 "$FP8C_PYTHON" report.py
