#!/usr/bin/env python3
"""glue-E item 3: draft lm_head GEMV + logits copy vs. GEMV writing the buffer.

    python bench/glue_e/bench_draft_logits.py

Shape is the served draft head: [M, 2560] x [32768, 2560] fp8 per-channel, with the
fp32 next-token logits buffer as the destination. The rotating weight working set
follows bench/w8a16v2/harness.py (>= 4x L2), so the GEMV time is the HBM-bound one
the server sees.
"""

from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _harness import clocks_note, cupti_by_kernel, make_copies, n_copies, report  # noqa: E402

N = 32768
K = 2560


def main(ms):
    from sglang.srt.layers.quantization.w8a16_gemv import w8a16_gemv

    print(clocks_note())
    g = torch.Generator(device="cuda").manual_seed(0)
    wf = torch.randn(N, K, dtype=torch.float32, device="cuda", generator=g)
    w0 = (wf / wf.abs().amax(dim=1, keepdim=True) * 400).to(torch.float8_e4m3fn)
    scale = torch.rand(N, 1, device="cuda", generator=g).add_(0.5).float()
    copies = make_copies(w0, n_copies(w0.numel()))
    print(f"weight working set: {len(copies)} x {w0.numel() / (1 << 20):.0f} MB")

    for M in ms:
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", generator=g)
        buf = torch.empty(M, N, dtype=torch.float32, device="cuda")
        print(f"\n=== M={M}")

        def old(w):
            y = w8a16_gemv(x, w, scale)
            buf.copy_(y)

        def new(w):
            w8a16_gemv(x, w, scale, out=buf)

        w8a16_gemv(x, copies[0], scale, out=buf)
        want = w8a16_gemv(x, copies[0], scale).float()
        print(f"  bit-identical: {bool(torch.equal(want, buf))}")

        res_o, tot_o = cupti_by_kernel(old, copies)
        report("gemv + buffer.copy_", res_o, tot_o)
        res_n, tot_n = cupti_by_kernel(new, copies)
        report("gemv out=buffer", res_n, tot_n)
        steps = 14  # draft-head calls per W16 decode step in the profile
        print(
            f"  delta = {tot_n - tot_o:+.2f} us/call -> "
            f"{(tot_n - tot_o) * steps:+.0f} us/step over {steps} draft steps"
        )


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [1])
