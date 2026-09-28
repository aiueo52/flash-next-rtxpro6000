"""BF16 dequantised reference for the NVFP4 MoE.  Runs on CPU on a tiny slice.

NVFP4 (modelopt, group_size=16) decodes as::

    W[n, k] = e2m1_decode(code[n, k]) * float(block_sf[n, k//16]) * weight_scale_2

``code`` is packed 2-per-byte, **low nibble first** (k even in the low nibble),
matching ``cvt_fp4x2_to_*`` in the TRT-LLM/FlashInfer kernels and the
``[N, K//2] uint8`` checkpoint layout.

The production kernel additionally quantises the activations to NVFP4 and
computes a per-16 activation block scale, so an exact bit-match is not
expected; ``compare()`` reports relative error against the *dequantised BF16*
math, which is the right correctness bar (max rel err should sit at the FP4
activation-quantisation noise floor, ~2-5%).
"""

from __future__ import annotations

import torch

# e2m1: sign(1) exponent(2) mantissa(1).  Values for codes 0..7, negated for 8..15.
E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """[..., K/2] uint8 -> [..., K] float32 e2m1 values (low nibble = even k)."""
    lo = (packed & 0x0F).to(torch.long)
    hi = (packed >> 4).to(torch.long)
    lut = E2M1_LUT.to(packed.device)
    out = torch.stack([lut[lo], lut[hi]], dim=-1)
    return out.reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def dequant_weight(packed: torch.Tensor, block_sf: torch.Tensor,
                   global_scale: float, block: int = 16) -> torch.Tensor:
    """[N, K/2] u8 + [N, K/16] e4m3 + scalar -> [N, K] float32."""
    vals = unpack_fp4(packed)                                   # [N, K]
    sf = block_sf.to(torch.float32)                             # [N, K/16]
    n, k = vals.shape
    assert sf.shape == (n, k // block), (sf.shape, (n, k // block))
    out = vals.reshape(n, k // block, block) * sf.unsqueeze(-1) * global_scale
    return out.reshape(n, k)


def moe_reference(lw, x: torch.Tensor, topk_ids: torch.Tensor,
                  topk_weights: torch.Tensor) -> torch.Tensor:
    """Dense BF16 reference for one MoE call.  float32 math, returns bf16.

    Only the experts named in `topk_ids` are dequantised, so this stays cheap
    even for the full 512-expert bank.
    """
    T, K = x.shape
    I = lw.shape.intermediate
    xf = x.to(torch.float32)
    out = torch.zeros((T, K), dtype=torch.float32, device=x.device)
    for t in range(T):
        acc = torch.zeros(K, dtype=torch.float32, device=x.device)
        for j in range(topk_ids.shape[1]):
            e = int(topk_ids[t, j])
            w13 = dequant_weight(lw.w13_weight[e], lw.w13_weight_scale[e],
                                 float(lw.w13_weight_scale_2[e, 0]))
            # gate and up may carry different global scales
            gs_up = float(lw.w13_weight_scale_2[e, 1])
            gs_gate = float(lw.w13_weight_scale_2[e, 0])
            w13[I:] *= (gs_up / gs_gate)
            w2 = dequant_weight(lw.w2_weight[e], lw.w2_weight_scale[e],
                                float(lw.w2_weight_scale_2[e]))
            h = w13 @ xf[t]                       # [2I]
            gate, up = h[:I], h[I:]
            act = torch.nn.functional.silu(gate) * up
            acc += (w2 @ act) * float(topk_weights[t, j])
        out[t] = acc
    return out.to(torch.bfloat16)


def compare(got: torch.Tensor, ref: torch.Tensor) -> dict:
    g = got.to(torch.float32)
    r = ref.to(torch.float32)
    den = r.abs().max().clamp_min(1e-6)
    err = (g - r).abs()
    cos = torch.nn.functional.cosine_similarity(g.flatten(), r.flatten(), dim=0)
    return {
        "max_abs_err": float(err.max()),
        "max_rel_err": float(err.max() / den),
        "mean_rel_err": float(err.mean() / den),
        "cosine": float(cos),
    }
