"""xa1: CPU-only checks in the Triton interpreter (no GPU, no lock).

1. Candidate B (gather) against the fp32 reference: fresh rows (verify / draft-extend layout) and
   index-shared draft rows (holes mid-row), short and capped contexts, 1..16 splits, both tiles.
2. Candidate A (packed) against the XQA contract: the production valid-count + _compact_kv
   kernels (prod source, interpreted) build the scratch, A must match fp32 attention over packed
   tokens [0, count) and, for fresh rows, the full reference.
3. The production packing on index-shared rows: which columns the XQA contract really attends.

The Triton 3.7 interpreter multiplies bf16 tiles as raw bits, so bf16 (v1) configs run with
F32_DOT=True (fp32 operands, P still rounded through bf16); f16 configs (v2 default) use the
native f16 dot, which the interpreter computes correctly.

  source bench/xa1/env.sh; TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= python -m xa1.selftest_cpu
"""
from __future__ import annotations

import os
import sys
import time

import torch

from xa1 import cases
from xa1.kernels import XA1Config, XA1Workspace, xa1_decode_gather, xa1_decode_packed


def _err(out, ref):
    d = (out.float() - ref.float()).abs()
    rel = d.max() / ref.float().abs().max().clamp_min(1e-12)
    return float(d.max()), float(rel)


def _pack_prod(case, lay):
    """Production valid counts + _compact_kv into a page-aligned scratch (server layout)."""
    from sglang.srt.layers.attention.qsa.sparse_attn import (
        qwen_sparse_kv_extraction_compact_triton, qwen_sparse_valid_counts_triton)
    rows, topk = lay.topk.shape
    stride = (topk + cases.PAGE - 1) // cases.PAGE * cases.PAGE
    counts = torch.empty(rows, dtype=torch.int32)
    qwen_sparse_valid_counts_triton(lay.seq_lens, lay.topk, counts, rows, topk)
    cu = torch.arange(rows + 1, dtype=torch.int32) * stride
    pk = torch.zeros((rows * stride, cases.HKV, cases.D), dtype=torch.float8_e4m3fn)
    pv = torch.zeros_like(pk)
    qwen_sparse_kv_extraction_compact_triton(lay.k_pool, lay.v_pool, case.req_to_token,
                                             case.row_req, lay.topk, lay.seq_lens, cu, pk, pv,
                                             rows, topk)
    return pk, pv, counts, stride


def main():
    if os.environ.get("TRITON_INTERPRET") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        sys.exit("run with TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES=")
    t0 = time.time()
    ws = XA1Workspace(max_groups=32, max_splits=40, head_dim=cases.D, device="cpu")
    fails = 0
    tol = 2e-2  # bf16 output + truncated bf16 P in the interpreter

    print("== B (gather) vs fp32 reference")
    specs = [
        dict(rows=4, ctx=600, mode="fresh"),
        dict(rows=3, ctx=2300, mode="fresh"),
        dict(rows=1, ctx=2600, mode="fresh"),
        dict(rows=1, ctx=900, mode="shared", n_tail=5),
        dict(rows=1, ctx=2600, mode="shared", n_tail=2),
        dict(rows=2, ctx=2700, mode="shared", n_tail=16),
    ]
    # vc runs tl's cast here (no inline asm in the interpreter); same values as the asm cvt.
    cfgs = [XA1Config(1, 64, 4, 1), XA1Config(3, 32, 4, 1, part16=True), XA1Config(8, 64, 4, 1),
            XA1Config(16, 32, 4, 1, part16=True), XA1Config(5, 32, 4, 1, f16=False),
            XA1Config(6, 32, 4, 1, red3d=True, vcvt=True),
            XA1Config(4, 64, 4, 1, red3d=True, part16=True),
            # v4 lse; 22 splits of 128 leave empty splits at 2051 columns, 33 is one tile each.
            XA1Config(11, 64, 4, 1, vcvt=True, lse=True), XA1Config(17, 64, 4, 1, lse=True),
            XA1Config(22, 64, 4, 1, lse=True), XA1Config(33, 32, 4, 1, vcvt=True, lse=True)]
    for i, spec in enumerate(specs):
        case = cases.Case(layers=1, seed=10 + i, **spec)
        lay = case.layers[0]
        ref = cases.reference(case, lay)
        # Fresh rows also run without the prefix clip (same result, more empty columns).
        for prefix_valid in ((True, False) if spec["mode"] == "fresh" else (False,)):
            for cfg in cfgs:
                out = xa1_decode_gather(lay.q, lay.k_pool, lay.v_pool, case.req_to_token,
                                        case.row_req, lay.topk, lay.seq_lens,
                                        bmm1_scale=cases.SCALING, bmm2_scale=1.0, cfg=cfg, ws=ws,
                                        pdl=False, prefix_valid=prefix_valid,
                                        f32_dot=not cfg.f16)
                a, r = _err(out, ref)
                ok = (r < tol and bool(torch.isfinite(out.float()).all())
                      and int(ws.cnt.abs().sum()) == 0)
                fails += not ok
                print(f"  {spec} pv={int(prefix_valid)} {cfg.key}: max_abs={a:.2e} rel={r:.2e} "
                      f"{'ok' if ok else 'FAIL'}", flush=True)

    print("== A (packed, production _compact_kv) vs XQA contract / reference")
    for i, spec in enumerate(specs):
        case = cases.Case(layers=1, seed=10 + i, **spec)
        lay = case.layers[0]
        pk, pv, counts, stride = _pack_prod(case, lay)
        pref = cases.prefix_reference(case, lay, pk, pv, counts, stride)
        ref = cases.reference(case, lay)
        for cfg in cfgs[:3] + cfgs[-2:-1]:
            out = xa1_decode_packed(lay.q, pk, pv, counts, stride=stride,
                                    bmm1_scale=cases.SCALING, bmm2_scale=1.0, cfg=cfg, ws=ws,
                                    pdl=False, f32_dot=not cfg.f16)
            a, r = _err(out, pref)
            ok = r < tol and int(ws.cnt.abs().sum()) == 0
            fails += not ok
            ra = _err(out, ref)[1]
            print(f"  {spec} {cfg.key}: vs-contract rel={r:.2e} {'ok' if ok else 'FAIL'}; "
                  f"vs-reference rel={ra:.2e}")
        if spec["mode"] == "fresh":
            fails += not _err(pref, ref)[1] < 1e-3

    print("== production packing on index-shared draft rows (what the XQA contract attends)")
    for ctx, n_tail in ((2600, 1), (2601, 4), (2602, 16), (2603, 3), (900, 5)):
        case = cases.Case(rows=1, ctx=ctx, layers=1, mode="shared", n_tail=n_tail, seed=7)
        lay = case.layers[0]
        valid = case.valid_mask(lay)[0]
        cols = valid.nonzero().flatten()
        pk, pv, counts, stride = _pack_prod(case, lay)
        n = int(counts[0])
        attended = torch.arange(n)
        holes = int((~valid[:n]).sum())
        dropped = int((cols >= n).sum())
        pref = cases.prefix_reference(case, lay, pk, pv, counts, stride)
        rel = _err(pref, cases.reference(case, lay))[1]
        newest = int(lay.topk[0, cols[-1]])
        print(f"  ctx={ctx} anchor_L={ctx - n_tail} n_tail={n_tail}: valid={cols.numel()} "
              f"count={n} attended-holes={holes} dropped-valid={dropped} "
              f"(newest pos {newest} {'dropped' if cols[-1] >= n else 'kept'}); "
              f"prefix-vs-reference rel={rel:.2e}")
        del attended
    print(f"[xa1-selftest] {'PASS' if fails == 0 else f'{fails} FAIL'} in {time.time() - t0:.1f}s")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
