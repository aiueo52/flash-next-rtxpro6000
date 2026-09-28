"""Decode-time skinny-GEMM shapes of Qwen3.8-Flash-Next (qwen4_exp), single GPU, TP=1.

Derived from models/RadixArk/Qwen3.8-Flash-Next-NVFP4/config.json text_config:
hidden=2560, linear_num_key_heads=16 * linear_key_head_dim=128 -> key_dim=2048,
linear_num_value_heads=48 * linear_value_head_dim=128 -> value_dim=6144,
num_attention_heads=24, head_dim=256, num_key_value_heads=2, attn_output_gate=True,
shared_expert_intermediate_size=640, num_experts=512, vocab=248320.

Weight layout matters for the kernel and is reproduced here:
  fp8  (Fp8LinearMethod.apply) passes ``layer.weight.t()``: storage is [K, N]
       contiguous, so the [N, K] view has stride (1, N) -- N is the contiguous axis.
  bf16 (bf16_gemv) passes ``linear.weight``: [N, K] contiguous, stride (K, 1).
"""

import torch

HIDDEN = 2560
KEY_DIM = 16 * 128
VALUE_DIM = 48 * 128
QKV_OUT = (24 * 2) * 256 + 2 * 2 * 256  # q+gate, k, v with attn_output_gate
SHARED_INTER = 640
VOCAB = 248320

# name, N, K, weight dtype tag, calls in the verify pass (M = spec width),
# calls per draft step (M = 1). 229 verify + 5 per draft step reproduces the observed
# 244 calls/step at W4 (3 draft steps) and ~304 at W16 (14 draft steps).
SHAPES = [
    ("shared_gate_up", 2 * SHARED_INTER, HIDDEN, "fp8", 48, 1),
    ("shared_down", HIDDEN, SHARED_INTER, "fp8", 48, 1),
    ("gdn_in_proj_qkvz", 2 * KEY_DIM + 2 * VALUE_DIM, HIDDEN, "fp8", 36, 0),
    ("gdn_in_proj_ba", 2 * 48, HIDDEN, "bf16", 36, 0),
    ("gdn_out_proj", HIDDEN, VALUE_DIM, "fp8", 36, 0),
    ("attn_qkv", QKV_OUT, HIDDEN, "fp8", 12, 1),
    ("attn_o", HIDDEN, 24 * 256, "fp8", 12, 1),
    ("moe_router_gate", 512, HIDDEN, "bf16", 0, 0),  # measured, not in the call model
    ("draft_head", 32768, HIDDEN, "fp8", 0, 1),
    ("lm_head", VOCAB, HIDDEN, "fp8", 1, 0),
]

# M values that actually occur: 1 on draft steps, spec width on the verify pass.
SHAPE_MS = {
    "lm_head": (1, 4, 16),
    "draft_head": (1,),
}
DEFAULT_MS = (1, 4, 16)

# spec width on the verify pass, number of draft steps
MODES = {"W16": (16, 14), "W4": (4, 3)}


def make_weight(N, K, tag, device="cuda", seed=0):
    """Return (w_view[N,K], scale, ref_w_fp32[N,K]) in the exact layout the server uses."""
    g = torch.Generator(device=device).manual_seed(seed)
    if tag == "bf16":
        w = (torch.randn(N, K, device=device, generator=g) * 0.02).to(torch.bfloat16)
        return w, torch.ones(1, dtype=torch.float32, device=device), w.float()
    wb = (torch.randn(N, K, device=device, generator=g) * 0.02).float()
    scale = (wb.abs().amax(dim=1).clamp(min=1e-8) / 448.0).contiguous()
    q = (wb / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    # Server layout: fp8 weight is stored [K, N] and viewed transposed.
    storage = q.t().contiguous()
    w = storage.t()
    return w, scale.reshape(N, 1), (w.float() * scale[:, None])
