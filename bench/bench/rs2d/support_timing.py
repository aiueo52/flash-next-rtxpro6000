"""CUDA graph timing of draft proposal support sizes.

Usage: WT=<worktree> python support_timing.py
Needs the GPU; the supervisor runs it under ~/.gpu.lock.
"""
import importlib.util
import itertools
import os
from pathlib import Path
import statistics
import sys

import torch

WT = Path(os.environ["WT"])


def load_module(*, name, relative_path):
    spec = importlib.util.spec_from_file_location(name=name, location=WT / "python/sglang" / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rs = load_module(name="support_sparse_rs", relative_path="kernels/ops/speculative/sparse_rs.py")


def graph_time(*, operation, calls, repeats=100):
    for _ in range(3):
        operation()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            operation()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(7):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / (calls * repeats))
    return statistics.median(times)


def main():
    torch.manual_seed(20261001)
    results = {}
    for dtype, n in itertools.product((torch.float32, torch.bfloat16), (1, 2)):
        vocab, target_vocab, steps = 49152, 248320, 7
        logits = torch.randn(size=(n, vocab), dtype=dtype, device="cuda")
        hot = torch.randperm(n=target_vocab, device="cuda")[:vocab].long()
        uniforms = torch.rand(size=(n,), device="cuda")
        positions = torch.zeros(size=(n,), dtype=torch.int64, device="cuda")
        chain = torch.empty(size=(n, steps + 1), dtype=torch.int64, device="cuda")
        dtype_name = "fp32" if dtype == torch.float32 else "bf16"
        for rows in ("sampling", "greedy"):
            temperatures = torch.full(size=(n,), fill_value=0.8 if rows == "sampling" else 1.0,
                                      dtype=torch.float32, device="cuda")
            top_ks = torch.full(size=(n,), fill_value=40 if rows == "sampling" else 1,
                               dtype=torch.int32, device="cuda")
            top_ps = torch.full(size=(n,), fill_value=0.95, dtype=torch.float32, device="cuda")
            min_ps = torch.full(size=(n,), fill_value=0.05, dtype=torch.float32, device="cuda")
            for k in ((64, 32, 16, 8) if rows == "sampling" else (64, 16)):
                support_p = torch.empty(size=(n, k), dtype=torch.float32, device="cuda")
                support_t = torch.empty(size=(n, k), dtype=torch.int64, device="cuda")

                def proposal():
                    rs.rs_draft_proposal_sparse(
                        next_token_logits=logits, temperatures=temperatures, top_ks=top_ks,
                        top_ps=top_ps, min_ps=min_ps, uniforms=uniforms, k=k, hot_token_id=hot,
                        positions=positions, draft_tokens=chain, draft_token_column=1,
                        draft_support_probs=support_p, draft_support_tokens=support_t,
                        temp_scale=0.7, onehot_above=0.9, greedy_fast=True,
                    )

                gpu_us = graph_time(operation=proposal, calls=steps)
                results[(dtype, n, rows, k)] = gpu_us
                delta = gpu_us - results[(dtype, n, rows, 64)]
                print(f"support dtype={dtype_name} n={n} rows={rows} K={k} "
                      f"GPU={gpu_us:.3f} us/call (vs K=64 {delta:+.2f} us)", flush=True)
    baseline = results[(torch.float32, 1, "sampling", 64)]
    summary = "; ".join(
        f"K={k} {results[(torch.float32, 1, 'sampling', k)]:.3f} us/call "
        f"(vs K=64 {results[(torch.float32, 1, 'sampling', k)] - baseline:+.2f} us)"
        for k in (64, 32, 16, 8)
    )
    print(f"support summary dtype=fp32 n=1 rows=sampling: {summary}", flush=True)


if __name__ == "__main__":
    main()
