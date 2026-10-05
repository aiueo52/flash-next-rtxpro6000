"""dg1: the mtpft5 MTP MoE layer in the server's CUTLASS NVFP4 format, plus fp32 references.

The server quantizes this bf16 layer online (nvfp4_online.py: gate/up pair under one amax, ModelOpt
convention); here quantize_nvfp4 (w4a16_nvfp4_gemv.py, the fork's torch reference with the same
convention) does it, then w13 is stored [up; gate] (load_up_proj_weight_first) and the block scales
are swizzled with the server's swizzle_blockscale.  References per routed row, all fp32:

  dq  dequantized weights, bf16 x, fp32 activations   (the GEMV up to its two bf16 roundings)
  dqb as dq with act rounded to bf16 (RNE)             (the GEMV's exact arithmetic on the GPU, up
                                                        to fp32 summation order and the output cast)
  dqt as dq with act truncated to bf16                 (same, for the Triton interpreter's casts)
  a4  dequantized weights, x and act fake-quantized to NVFP4 with global scale 1.0 and the GEMM1
      output rounded to bf16                             (what CUTLASS W4A4 computes)
  bf  original bf16 checkpoint weights                  (weight quantization included)
"""
from __future__ import annotations

import json
import os
import types

import torch
import torch.nn.functional as F

from dg1.grid import E, H, I, TOPK

CKPT = os.path.expanduser("~/models/RadixArk/Qwen3.8-Flash-Next-NVFP4-mtpft5")
PREFIX = "mtp.layers.0.mlp.experts."


def _ckpt_file(name: str) -> str:
    with open(os.path.join(CKPT, "model.safetensors.index.json")) as f:
        return os.path.join(CKPT, json.load(f)["weight_map"][name])


def fake_nvfp4(a: torch.Tensor) -> torch.Tensor:
    """[M, K] fp32 -> NVFP4 round trip with global encode scale 1.0 (input_scale_quant of the draft)."""
    from sglang.srt.layers.quantization.w4a16_nvfp4_gemv import _E2M1_LEVELS, _round_e2m1

    m, k = a.shape
    blk = a.reshape(m, k // 16, 16)
    sf = (blk.abs().amax(-1, keepdim=True) / 6.0).clamp(max=448.0).to(torch.float8_e4m3fn).float()
    y = torch.where(sf > 0, blk / sf.clamp(min=1e-38), torch.zeros_like(blk))
    lv = torch.tensor(_E2M1_LEVELS, dtype=torch.float32, device=a.device)[_round_e2m1(y.abs()).long()]
    return (torch.where(y < 0, -lv, lv) * sf).reshape(m, k)


def bf16_rne(t: torch.Tensor) -> torch.Tensor:
    return t.bfloat16().float()


def bf16_trunc(t: torch.Tensor) -> torch.Tensor:
    """fp32 -> bf16 -> fp32 dropping the low 16 bits: the Triton 3.7 interpreter's cast (no rounding)."""
    return (t.contiguous().view(torch.int32) & -65536).view(torch.float32)


def expert_fp32(x, w13_gate_up, w2, *, a4: bool = False, act_round=None):
    """x [n, H] fp32, w13 [2I, H] with gate rows first (checkpoint order), w2 [H, I] -> [n, H] fp32."""
    if a4:
        x = fake_nvfp4(x)
    h = x @ w13_gate_up.t()
    if a4:
        h = h.bfloat16().float()
    act = F.silu(h[:, :I]) * h[:, I:]
    if a4:
        act = fake_nvfp4(act)
    if act_round is not None:
        act = act_round(act)
    return act @ w2.t()


# expert_fp32 options of the references on dequantized weights ("bf" uses the checkpoint's).
_DQ_REFS = {"dq": {}, "dqb": {"act_round": bf16_rne}, "dqt": {"act_round": bf16_trunc}, "a4": {"a4": True}}


class _RouteIndex:
    """(row, weight) pairs of every route, grouped by expert id."""

    def __init__(self, ids: torch.Tensor, wts: torch.Tensor):
        flat = ids.reshape(-1).long()
        order = torch.argsort(flat, stable=True)
        self.bounds = torch.searchsorted(flat[order], torch.arange(E + 1, device=ids.device)).tolist()
        self.rows = order // ids.shape[1]
        self.wts = wts.reshape(-1)[order]

    def of(self, e: int):
        lo, hi = self.bounds[e], self.bounds[e + 1]
        return self.rows[lo:hi], self.wts[lo:hi, None]


def build_layer(*, X, ids, wts, device, chunk: int = 32, num_experts: int = E,
                refs=("dq", "dqb", "a4", "bf")):
    """Quantized layer (duck of FusedMoE's fields after process_weights_after_loading) + references.

    X [N, H] bf16, ids [N, TOPK] int32, wts [N, TOPK] fp32, all on `device`.
    """
    from safetensors import safe_open

    from sglang.srt.layers.quantization.utils import swizzle_blockscale
    from sglang.srt.layers.quantization.w4a16_nvfp4_gemv import dequantize_nvfp4, quantize_nvfp4

    u8 = dict(dtype=torch.uint8, device=device)
    w13q = torch.empty((num_experts, 2 * I, H // 2), **u8)
    w13s = torch.empty((num_experts, 2 * I, H // 16), **u8)
    ws13 = torch.empty((num_experts, 2), dtype=torch.float32, device=device)
    w2q = torch.empty((num_experts, H, I // 2), **u8)
    w2s = torch.empty((num_experts, H, I // 16), **u8)
    ws2 = torch.empty((num_experts,), dtype=torch.float32, device=device)
    out = {k: torch.zeros((X.shape[0], H), dtype=torch.float32, device=device) for k in refs}
    routes = _RouteIndex(ids, wts)
    gu_name, dn_name = PREFIX + "gate_up_proj", PREFIX + "down_proj"
    with safe_open(_ckpt_file(gu_name), framework="pt", device="cpu") as f:
        gu_sl, dn_sl = f.get_slice(gu_name), f.get_slice(dn_name)
        assert list(gu_sl.get_shape()) == [E, 2 * I, H] and list(dn_sl.get_shape()) == [E, H, I]
        for e0 in range(0, num_experts, chunk):
            gu = gu_sl[e0:min(e0 + chunk, num_experts)].to(device)
            dn = dn_sl[e0:min(e0 + chunk, num_experts)].to(device)
            for i in range(gu.shape[0]):
                e = e0 + i
                q, bs, gs = quantize_nvfp4(gu[i])
                # checkpoint rows: gate [0, I), up [I, 2I); server: up first.
                w13q[e, :I], w13q[e, I:] = q[I:], q[:I]
                w13s[e, :I], w13s[e, I:] = bs[I:], bs[:I]
                ws13[e] = gs
                q2, bs2, gs2 = quantize_nvfp4(dn[i])
                w2q[e], w2s[e], ws2[e:e + 1] = q2, bs2, gs2
                rows, wr = routes.of(e)
                if rows.numel() == 0:
                    continue
                xb = X[rows].float()
                w13_dq, w2_dq = dequantize_nvfp4(q, bs, gs), dequantize_nvfp4(q2, bs2, gs2)
                for name, acc in out.items():
                    if name == "bf":
                        y = expert_fp32(xb, gu[i].float(), dn[i].float())
                    else:
                        y = expert_fp32(xb, w13_dq, w2_dq, **_DQ_REFS[name])
                    acc.index_add_(0, rows, wr * y)
            del gu, dn
    one = torch.ones((), dtype=torch.float32, device=device)
    layer = types.SimpleNamespace(
        layer_id=0,
        w13_weight=w13q,
        w2_weight=w2q,
        w13_weight_scale=w13s.view(torch.float8_e4m3fn),
        w2_weight_scale=w2s.view(torch.float8_e4m3fn),
        w13_blockscale_swizzled=swizzle_blockscale(w13s.view(torch.float8_e4m3fn)).to(device),
        w2_blockscale_swizzled=swizzle_blockscale(w2s.view(torch.float8_e4m3fn)).to(device),
        w13_weight_scale_2=ws13,
        w2_weight_scale_2=ws2,
        # modelopt_quant.py (CUTLASS): input scale 1.0 -> alphas = weight scales, quant = 1 / 1.0.
        g1_alphas=ws13[:, 0].contiguous(),
        g2_alphas=ws2.clone(),
        w13_input_scale_quant=one,
        w2_input_scale_quant=one.clone(),
        moe_ep_size=1,
        moe_ep_rank=0,
        moe_tp_size=1,
        moe_tp_rank=0,
    )
    return layer, out
