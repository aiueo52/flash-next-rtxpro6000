"""Test E's verify timings alone: in test E the proposal budget assert stops the run before benchmark_verify.

Same environment as chain_rs2.sh's CUDA test (DEV=cuda, WT, the server's prebuilt FlashInfer).
"""

import importlib.util
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location(name="rs2_test", location=Path(__file__).with_name("test_sparse_rs.py"))
test = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test)
assert test.DEV == "cuda"
torch.manual_seed(20261001)
test.benchmark_verify()
