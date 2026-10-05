"""dg1: CPU-only check of the GEMV kernels in the Triton interpreter against the fp32 reference.

Runs the module's real kernels (TRITON_INTERPRET=1, USE_PDL=False) on a 12-expert slice of the
mtpft5 MTP layer: three tile configs (incl. splits2=1 and a route with id -1).  Validates layout
(up/gate halves, nibble order), the swizzled-scale reads and the split reduction before GPU time.

The Triton 3.7 interpreter multiplies bf16 tensors as raw uint16 bits (bf16 * bf16 -> garbage),
so the two helpers that return bf16 tiles are wrapped to return fp32; on the GPU these products
(e2m1 x e4m3, at most 6 significant bits) are exact in bf16, so the arithmetic is the same.
Its fp32 -> bf16 cast truncates (no rounding), so the emulation reference truncates as well.

  TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= python -m dg1.selftest_cpu
"""
from __future__ import annotations

import os
import sys
import time

import torch
import triton
import triton.language as tl

import sglang.srt.layers.quantization  # noqa: F401  (import order: see moe_smallm.runners)
import sglang.srt.layers.moe.moe_runner.flashinfer_trtllm  # noqa: F401
from sglang.srt.layers.moe import draft_moe_gemv as dmg

_unpack_bf16 = dmg._unpack_dequant
_scales_bf16 = dmg._swizzled_scales


@triton.jit
def _unpack_f32(b, BLOCK_N: tl.constexpr, BLOCK_KB: tl.constexpr):
    return _unpack_bf16(b, BLOCK_N, BLOCK_KB).to(tl.float32)


@triton.jit
def _scales_f32(sf_ptr, sf_rows, kb0, BN: tl.constexpr, BK: tl.constexpr):
    return _scales_bf16(sf_ptr, sf_rows, kb0, BN, BK).to(tl.float32)


def _ordered_bits(t: torch.Tensor) -> torch.Tensor:
    """bf16 -> integers that step by one per bf16 value (sign-magnitude bits made monotone)."""
    b = t.view(torch.int16).int()
    return torch.where(b < 0, -(b & 0x7FFF), b)


def main():
    if os.environ.get("TRITON_INTERPRET") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        sys.exit("run with TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES=")
    t0 = time.time()
    from dg1.grid import H, TOPK, with_k1, with_k2
    from dg1.layer import bf16_trunc, build_layer
    from dg1.proxies import patched

    dmg._unpack_dequant = _unpack_f32
    dmg._swizzled_scales = _scales_f32
    n_exp = 12
    g = torch.Generator().manual_seed(0)
    ids = torch.stack([torch.randperm(n_exp, generator=g)[:TOPK] for _ in range(3)]).to(torch.int32)
    ids[1, 3] = -1
    ids[1, 7] = -1
    wts = torch.rand((3, TOPK), generator=g).softmax(-1)
    X = torch.randn((3, H), generator=g).to(torch.bfloat16)
    # swizzle_blockscale ends in .cuda(); keep its permutation on the CPU.
    orig_cuda = torch.Tensor.cuda
    torch.Tensor.cuda = lambda self, *a, **k: self
    try:
        L, refs = build_layer(X=X, ids=ids, wts=wts, device="cpu", chunk=n_exp, num_experts=n_exp,
                              refs=("dq", "dqt"))
    finally:
        torch.Tensor.cuda = orig_cuda
    print(f"[selftest] layer built {time.time() - t0:.1f}s", flush=True)

    base = dmg.DEFAULT_CONFIG
    cases = [
        (0, base),
        (1, with_k2(with_k1(base, (16, 256, 4, 2)), (8, 64, 5, 2, 3))),
        (2, with_k2(with_k1(base, (4, 512, 2, 2)), (32, 128, 1, 4, 3))),
    ]
    ok = True
    for row, cfg in cases:
        ws = dmg.make_workspace(num_routes=TOPK, hidden_size=H, intermediate_size=L.w13_weight.shape[1] // 2,
                                device="cpu", cfg=cfg)
        with patched(dmg, pdl=False):
            y = dmg.draft_moe_gemv(
                hidden_states=X[row:row + 1], topk_ids=ids[row:row + 1], topk_weights=wts[row:row + 1],
                w13_weight=L.w13_weight, w13_blockscale_swizzled=L.w13_blockscale_swizzled,
                w13_weight_scale_2=L.w13_weight_scale_2, w2_weight=L.w2_weight,
                w2_blockscale_swizzled=L.w2_blockscale_swizzled, w2_weight_scale_2=L.w2_weight_scale_2,
                workspace=ws, cfg=cfg)
        ref = refs["dq"][row]
        rel = ((y[0].float() - ref).norm() / ref.norm()).item()
        # dqt with the output truncated too is the kernel's arithmetic under the interpreter's
        # casts; fp32 summation order still flips some act/output truncations (a few ulps on
        # near-zero outputs), far below the 6e-3 that the two bf16 roundings cost vs dq.
        emu = bf16_trunc(refs["dqt"][row])
        rel_emu = ((y[0].float() - emu).norm() / emu.norm()).item()
        ulp = (_ordered_bits(y[0]) - _ordered_bits(emu.bfloat16())).abs()
        ulps = int((ulp > 1).sum())
        cnt = int(ws.counters.abs().sum())
        good = rel_emu < 1e-3 and rel < 2e-2 and cnt == 0
        ok &= good
        print(f"[selftest] row {row} cfg {cfg} rel_l2 vs dq={rel:.2e} vs emulation={rel_emu:.2e} "
              f"max_ulp={int(ulp.max())} elems>1ulp={ulps} counters_left={cnt} {'OK' if good else 'FAIL'} "
              f"({time.time() - t0:.0f}s)", flush=True)
    print("[selftest] PASS" if ok else "[selftest] FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
