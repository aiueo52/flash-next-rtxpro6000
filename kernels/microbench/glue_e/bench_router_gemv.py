#!/usr/bin/env python3
"""glue-E item 4 (analysis): MoE router gate, cuBLAS vs. the tuned skinny BF16 GEMV.

    python bench/glue_e/bench_router_gemv.py 1 16

`SGLANG_ROUTER_GEMV=1` routes qwen2_moe._forward_router_logits through
`bf16_gemv` instead of `ReplicatedLinear` (cuBLAS gemm + splitKreduce). The
supervisor A/Bs the flag in-server; this only shows the isolated kernel picture at
the served shape, N = num_experts = 512, K = hidden_size = 2560, bias-free.
"""

from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _harness import clocks_note, cupti_by_kernel, make_copies, n_copies, report  # noqa: E402

N = 512
K = 2560


def main(ms):
    from sglang.srt.layers.quantization.w8a16_gemv import _plan, bf16_gemv

    print(clocks_note())
    g = torch.Generator(device="cuda").manual_seed(0)
    w0 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda", generator=g)
    copies = make_copies(w0, n_copies(w0.numel() * 2))
    print(f"weight working set: {len(copies)} x {w0.numel() * 2 / (1 << 20):.1f} MB")
    for M in ms:
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", generator=g)
        cfg = _plan(M, N, K, w0.stride(0) == 1, 2, 188)
        print(f"\n=== M={M}  tuned cfg (BLOCK_N,BLOCK_K,SPLITS,USE_DOT,W_KN,warps,stages)={cfg}")
        ref = torch.nn.functional.linear(x, copies[0])
        got = bf16_gemv(x, copies[0])
        d = (ref.float() - got.float()).abs().max().item()
        print(f"  max |cublas - gemv| = {d:.3e}  (bf16 ulp at |x|~1 is 7.8e-3)")

        res_c, tot_c = cupti_by_kernel(lambda w: torch.nn.functional.linear(x, w), copies)
        report("cuBLAS linear", res_c, tot_c)
        res_g, tot_g = cupti_by_kernel(lambda w: bf16_gemv(x, w), copies)
        report("bf16_gemv (SGLANG_ROUTER_GEMV=1)", res_g, tot_g)
        calls = 49  # router calls per decode step in the W16/W4 traces
        print(
            f"  delta = {tot_g - tot_c:+.2f} us/call -> "
            f"{(tot_g - tot_c) * calls:+.0f} us/step over {calls} router calls"
        )


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [1, 16])
