#!/usr/bin/env python3
"""glue-E item 2: fused HC combine (1 launch) vs. hc_combine_gate + hc_combine_apply.

    python bench/glue_e/bench_hc_combine.py            # rows 1 and 16
    python bench/glue_e/bench_hc_combine.py 1 2 4 8 16

Metric: CUPTI kernel duration over CUDA-graph replay (bench/w8a16v2/harness.py),
which is what the in-server torch profile reports. The rotating working set is a
ring of distinct (block_output, residual, normed, out) quadruples several times the
128 MB L2, so the numbers reflect real memory traffic rather than an L2-resident
micro-benchmark.
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _harness import clocks_note, cupti_by_kernel, report  # noqa: E402

HC = 4
HS = 2560
ROW = HC * HS
L2 = 128 << 20


def build(rows: int):
    """A working set of >= 4x L2 of distinct residual/normed/out triples."""
    per_call = rows * (2 * ROW + HS + ROW) * 2  # r + n + y read, out written
    n = max(8, min(96, -(-4 * L2 // max(per_call, 1))))
    g = torch.Generator(device="cuda").manual_seed(0)
    w = (
        torch.randn(HC, ROW, dtype=torch.float32, device="cuda", generator=g)
        .mul_(0.02)
        .to(torch.bfloat16)
    )
    args = []
    for _ in range(n):
        y = torch.randn(rows, HS, dtype=torch.bfloat16, device="cuda", generator=g)
        r = torch.randn(rows, ROW, dtype=torch.bfloat16, device="cuda", generator=g)
        nn = torch.randn(rows, ROW, dtype=torch.bfloat16, device="cuda", generator=g)
        out = torch.empty_like(r)
        args.append((y, r, nn, out))
    return w, args


def main(rows_list):
    from sglang.kernels.ops.elementwise.hc_combine import hc_combine_split
    from sglang.srt.layers.hc_combine_fused_triton import (
        hc_combine_fused,
        hc_combine_fused_supported,
    )

    print(clocks_note())
    for rows in rows_list:
        w, args = build(rows)
        print(f"\n=== rows={rows}  hc={HC} hidden={HS}  working set x{len(args)}")
        if not hc_combine_fused_supported(args[0][0], args[0][1], args[0][2], w, HC, HS):
            print("  fused path unsupported at this row count")
            continue

        def split(a):
            y, r, n, out = a
            hc_combine_split(y, r, n, w, HC, HS, out=out)

        def fused(a):
            y, r, n, out = a
            hc_combine_fused(y, r, n, w, HC, HS, out=out)

        # correctness first, so a broken variant never reports a time
        y, r, n, out = args[0]
        ref = hc_combine_split(y, r, n, w, HC, HS)
        got = hc_combine_fused(y, r, n, w, HC, HS)
        d = (ref.float() - got.float()).abs().max().item()
        print(f"  max |split - fused| = {d:.3e}")

        res_s, tot_s = cupti_by_kernel(split, args)
        report("split (gate + apply)", res_s, tot_s)
        res_f, tot_f = cupti_by_kernel(fused, args)
        report("fused (one Triton launch)", res_f, tot_f)
        print(
            f"  delta = {tot_f - tot_s:+.2f} us/call   "
            f"-> {(tot_f - tot_s) * 96:+.0f} us/step at 96 combines"
        )


if __name__ == "__main__":
    rows = [int(a) for a in sys.argv[1:]] or [1, 16]
    main(rows)
