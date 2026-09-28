#!/usr/bin/env python3
"""glue-E item 1: GDN WY verify kernel reading q/k/v strided vs. .contiguous()-first.

    python bench/glue_e/bench_gdn_wy_t16.py           # T=16 (W16 verify) and T=4

Loads the installed vendored wrapper as the baseline and a patched copy (from
bench/glue_e/gdn_wy_T16_strided.patch, applied in a temp dir - the venv file is
never touched) with FLASHINFER_GDN_WY_STRIDED_QKV=1 as the candidate, then reports
CUPTI per-call device time. The baseline's three q/k/v copies show up as separate
elementwise kernels, which is exactly the cost the patch removes.
"""

from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "test" / "srt" / "layers"))
from _harness import clocks_note, cupti_by_kernel, report  # noqa: E402
from test_gdn_wy_t16_strided import (  # noqa: E402
    CONV_DIM,
    HV,
    K_DIM,
    V_DIM,
    _load_modules,
    _make_inputs,
)

L2 = 128 << 20


def build(B: int, T: int, copies: int):
    # pool=B so the allocated bytes equal the touched bytes: the kernel only reads
    # the B state slots it is indexed at, and the working set has to be real.
    return [_make_inputs(B, T, seed=i, pool=B) for i in range(copies)]


def main(cases):
    ref, pat = _load_modules()
    print(clocks_note())
    for B, T in cases:
        per_call = B * (HV * V_DIM * K_DIM * 2 + T * CONV_DIM * 2)
        copies = max(8, min(320, -(-4 * L2 // max(per_call, 1))))
        sets = build(B, T, copies)
        ws_mb = copies * per_call / (1 << 20)
        print(f"\n=== B={B} T={T}  working set x{copies} (~{ws_mb:.0f} MB, L2 is 128 MB)")

        def run_ref(a):
            kw = {k: v for k, v in a.items() if k != "conv_out"}
            ref.gated_delta_rule_mtp(**kw)

        def run_pat(a):
            kw = {k: v for k, v in a.items() if k != "conv_out"}
            pat.gated_delta_rule_mtp(**kw)

        a0 = sets[0]
        kw = {k: v for k, v in a0.items() if k != "conv_out"}
        o_ref = ref.gated_delta_rule_mtp(**kw).clone()
        o_pat = pat.gated_delta_rule_mtp(**kw).clone()
        print(f"  bit-identical: {bool(torch.equal(o_ref, o_pat))}")

        res_r, tot_r = cupti_by_kernel(run_ref, sets)
        report("contiguous (3 copies + kernel)", res_r, tot_r)
        res_p, tot_p = cupti_by_kernel(run_pat, sets)
        report("strided (kernel only)", res_p, tot_p)
        n_layers = 36  # GDN layers in Qwen3.8-Flash-Next (48 * 3/4)
        print(
            f"  delta = {tot_p - tot_r:+.2f} us/call  -> "
            f"{(tot_p - tot_r) * n_layers:+.0f} us/step over {n_layers} GDN layers"
        )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cases = [tuple(int(x) for x in a.split("x")) for a in sys.argv[1:]]
    else:
        cases = [(1, 16), (1, 4)]
    main(cases)
