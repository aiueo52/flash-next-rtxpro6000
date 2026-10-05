"""G1d (RS2d spec section 5): CUDA graph timing of the RS2 draft proposal for all-greedy rows.

Usage: WT=<worktree with G1> python g1d_timing.py   (needs the GPU; run under ~/.gpu.lock)

Greedy rows are stored as top_k 1, temperature 1.0. Rows: the full proposal (greedy_fast False), the G1 fast
path (greedy_fast True) and the target-only top-1 kernel. Budget: fast path <= 6 us per call at n=1, fp32.
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


rs = load_module(name="g1d_sparse_rs", relative_path="kernels/ops/speculative/sparse_rs.py")
topk1 = load_module(name="g1d_topk1", relative_path="kernels/ops/speculative/topk1.py")


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
    for dtype, n in itertools.product((torch.float32, torch.bfloat16), (1, 2, 4)):
        vocab, target_vocab, steps = 49152, 248320, 7
        logits = torch.randn(size=(n, vocab), dtype=dtype, device="cuda")
        hot = torch.randperm(n=target_vocab, device="cuda")[:vocab].long()
        temperatures = torch.ones(size=(n,), dtype=torch.float32, device="cuda")
        top_ks = torch.ones(size=(n,), dtype=torch.int32, device="cuda")
        top_ps = torch.ones(size=(n,), dtype=torch.float32, device="cuda")
        uniforms = torch.rand(size=(n,), device="cuda")
        positions = torch.zeros(size=(n,), dtype=torch.int64, device="cuda")
        chain = torch.empty(size=(n, steps + 1), dtype=torch.int64, device="cuda")
        support_p = torch.empty(size=(n, 64), dtype=torch.float32, device="cuda")
        support_t = torch.empty(size=(n, 64), dtype=torch.int64, device="cuda")

        def proposal(*, greedy_fast):
            rs.rs_draft_proposal_sparse(
                next_token_logits=logits, temperatures=temperatures, top_ks=top_ks, top_ps=top_ps,
                uniforms=uniforms, k=64, hot_token_id=hot, positions=positions, draft_tokens=chain,
                draft_token_column=1, draft_support_probs=support_p, draft_support_tokens=support_t,
                greedy_fast=greedy_fast,
            )

        operations = {
            "full": lambda: proposal(greedy_fast=False),
            "greedy_fast": lambda: proposal(greedy_fast=True),
            "target-only": lambda: topk1.draft_topk1_postprocess(
                next_token_logits=logits, positions=positions, draft_tokens=chain, draft_token_column=1,
                hot_token_id=hot),
        }
        for name, operation in operations.items():
            results[(dtype, n, name)] = graph_time(operation=operation, calls=steps)
            print(f"G1d dtype={dtype} n={n} Vd={vocab} {name:12s} GPU={results[(dtype, n, name)]:.3f} us/call",
                  flush=True)
    fast = results[(torch.float32, 1, "greedy_fast")]
    print(f"{'PASS' if fast <= 6 else 'WARN'} G1d: greedy_fast n=1 fp32 {fast:.3f} us/call (budget 6 us)")


if __name__ == "__main__":
    main()
