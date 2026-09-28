"""Measured DRAM read roofline on this box (GPU-ONLY).

The MoE grouped GEMM's marginal cost per distinct expert is a *pure streaming
read*.  Interpreting it needs the real achievable read rate, not the 1792 GB/s
data-sheet number: `MOE_SMALLM_SPEC.md` quotes a 1461 GB/s *copy* roof (which is
read+write and therefore not the right comparison).

Runs a few candidate readers over a buffer far larger than the 128 MiB L2 and
reports the best.
"""
from __future__ import annotations

import argparse
import json

import torch


def _time(fn, warmup=3, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e-3  # seconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gb", type=float, default=4.0)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    n = int(args.gb * 1e9) // 4
    x = torch.randn(n, dtype=torch.float32, device="cuda")
    xb = x.view(torch.bfloat16)
    xu = x.view(torch.uint8)
    nbytes = x.numel() * 4
    out = {}

    cands = {
        "sum_fp32": lambda: torch.sum(x),
        "sum_bf16": lambda: torch.sum(xb),
        "max_fp32": lambda: torch.max(x),
        "norm_fp32": lambda: torch.linalg.vector_norm(x),
        "dot_fp32(2x bytes)": lambda: torch.dot(x, x),
    }
    for name, fn in cands.items():
        sec = _time(fn, iters=args.iters)
        b = nbytes * (2 if "2x" in name else 1)
        out[name] = round(b / sec / 1e9, 1)

    # read+write copy, for comparison with the 1461 GB/s figure in the spec
    y = torch.empty_like(x)
    sec = _time(lambda: y.copy_(x), iters=args.iters)
    out["copy(read+write)"] = round(2 * nbytes / sec / 1e9, 1)

    clk = torch.cuda.clock_rate() if hasattr(torch.cuda, "clock_rate") else None
    print(json.dumps({"buffer_GB": nbytes / 1e9, "GBps": out, "sm_clock_kHz": clk}, indent=2))


if __name__ == "__main__":
    main()
