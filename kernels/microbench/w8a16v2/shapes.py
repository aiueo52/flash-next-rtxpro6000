"""Decode-time skinny-GEMM shapes of Qwen3.8-Flash-Next (qwen4_exp), single GPU, TP=1.

Weight layout matters for the kernel and is reproduced exactly:

  fp8  Fp8LinearMethod.process_weights_after_loading does
         layer.weight = Parameter(weight.t())
       on the [N, K]-CONTIGUOUS quantized weight, and apply() then calls
         w8a16_gemv(x, layer.weight.t(), layer.weight_scale)
       so the kernel receives an [N, K] view of [N, K]-contiguous storage,
       stride (K, 1).  N is *not* the contiguous axis: w.stride(0) == K, hence
       _plan()'s `contig_n` is False and W_KN is False for every server fp8 call.
       (An earlier version of this file claimed "storage is [K, N] contiguous";
       that was wrong and every (1, True, ...) tuning entry it produced was dead
       code in the server.  See git history / the v3 tuning notes.)
  bf16 bf16_gemv passes ``linear.weight``: [N, K] contiguous, stride (K, 1) --
       same layout, so contig_n is False there too.

Scales are [N, 1] fp32 per output channel, as Fp8LinearMethod stores them.

`grid_n` is cdiv(N, 32), the N extent of the launch grid the server actually uses
(the fallback plan is BLOCK_N=32), so the rows line up with the per-launch-grid
medians recorded by torch.profiler in the serving profile.
"""

import torch

HIDDEN = 2560
KEY_DIM = 16 * 128
VALUE_DIM = 48 * 128
QKV_OUT = (24 * 2) * 256 + 2 * 2 * 256
SHARED_INTER = 640
VOCAB = 248320

# name, N, K, dtype tag, calls per verify pass (M = spec width), calls per draft step
#
# Call counts are read off the serving torch profile (W16 decode, 20 steps): the
# verify count is the number of M=spec_width calls in one decode step, the draft
# count is the number of M=1 calls per draft step (the MTP layer runs once per
# draft step plus once inside the verify pass, i.e. draft_steps + 1 times).
SHAPES = [
    # --- primary fp8 targets (w8a16_gemv) --------------------------------------
    # GDN out_proj (36 layers) + attn o_proj (12 layers) + MTP o_proj: one shape.
    ("o_proj", HIDDEN, VALUE_DIM, "fp8", 48, 1),
    ("shared_down", HIDDEN, SHARED_INTER, "fp8", 48, 1),
    ("shared_gate_up", 2 * SHARED_INTER, HIDDEN, "fp8", 48, 1),
    ("attn_qkv", QKV_OUT, HIDDEN, "fp8", 12, 1),
    ("gdn_in_proj_qkvz", 2 * KEY_DIM + 2 * VALUE_DIM, HIDDEN, "fp8", 36, 0),
    ("draft_head", 32768, HIDDEN, "fp8", 0, 1),
    ("lm_head", VOCAB, HIDDEN, "fp8", 1, 0),
    # --- primary bf16 target (bf16_gemv) ---------------------------------------
    ("gdn_in_proj_ba", 2 * 48, HIDDEN, "bf16", 36, 0),
    # --- secondary bf16 targets ------------------------------------------------
    # MoE router gate: currently cuBLAS in the server (~4.8 us) unless the
    # router-gemv flag is on; 48 MoE layers per verify pass + 1 in the MTP layer.
    ("moe_router_gate", 512, HIDDEN, "bf16", 48, 1),
    # MTP fc_embedding / fc_hidden: two calls per draft step, M=1 (or M=hc_count).
    ("mtp_fc", HIDDEN, HIDDEN, "bf16", 0, 2),
]

SECONDARY = {"moe_router_gate", "mtp_fc"}

# Ground truth: in-server medians of the CUPTI kernel durations (torch.profiler,
# CUDA-graph decode, 20 steps), microseconds.  Keyed (shape, M).  Every one of
# these ran the *fallback* plan (BLOCK_N=32, BLOCK_K=128, SPLITS=planner,
# USE_DOT=True, W_KN=False, 4 warps, 3 stages) because the tuned table's keys
# assumed the wrong layout.
#
# W16 profile (spec width 16, 14 draft steps).
GROUND_TRUTH = {
    ("gdn_in_proj_qkvz", 16): 32.4,
    ("draft_head", 1): 65.4,
    ("o_proj", 16): 13.1,
    ("o_proj", 1): 13.1,
    ("attn_qkv", 16): 25.6,
    ("attn_qkv", 1): 25.6,
    ("shared_gate_up", 16): 6.5,
    ("shared_gate_up", 1): 6.5,
    ("lm_head", 16): 407.0,
    ("gdn_in_proj_ba", 16): 7.9,
    ("shared_down", 16): 4.1,
    ("shared_down", 1): 4.1,
}
# W4 profile (spec width 4, 3 draft steps); the M=16 rows become M=4 there.
GROUND_TRUTH_W4 = {
    ("gdn_in_proj_qkvz", 4): 29.3,
    ("draft_head", 1): 64.9,
    ("o_proj", 4): 13.1,
    ("attn_qkv", 4): 26.2,
    ("shared_gate_up", 4): 6.5,
    ("lm_head", 4): 396.0,
    ("gdn_in_proj_ba", 4): 5.5,
    ("shared_down", 4): 4.0,
}
# The launch grid the profile recorded, for cross-checking the reproduction.
GROUND_TRUTH_GRID = {
    "gdn_in_proj_qkvz": (512, 1),
    "draft_head": (1024, 1),
    "o_proj": (80, 6),
    "attn_qkv": (416, 1),
    "shared_gate_up": (40, 10),
    "lm_head": (7760, 1),
    "gdn_in_proj_ba": (3, 1),
    "shared_down": (80, 5),
}

SHAPE_MS = {
    "lm_head": (16,),
    "draft_head": (1,),
    "gdn_in_proj_qkvz": (16,),
    "gdn_in_proj_ba": (16,),
    "mtp_fc": (1,),
}
DEFAULT_MS = (16, 1)
MODES = {"W16": (16, 14), "W4": (4, 3)}


def make_weight(N, K, tag, device="cuda", seed=0):
    """(w_view[N,K], scale[N,1], ref_w_fp32[N,K]) in the exact layout the server hands over.

    fp8: the server quantizes an [N, K] weight, keeps that [N, K]-contiguous
    storage, stores the [K, N] transposed *view* on the layer, and transposes it
    back before the call -- so what arrives is [N, K] with stride (K, 1).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    if tag == "bf16":
        w = (torch.randn(N, K, device=device, generator=g) * 0.02).to(torch.bfloat16)
        return w, torch.ones(1, dtype=torch.float32, device=device), w.float()
    wb = (torch.randn(N, K, device=device, generator=g) * 0.02).float()
    scale = (wb.abs().amax(dim=1).clamp(min=1e-8) / 448.0).contiguous()
    q = (wb / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous()
    # Server: qweight is [N, K] contiguous; layer.weight = qweight.t() ([K, N]
    # view); apply() passes layer.weight.t(), i.e. back to this [N, K] view.
    w = q.t().t()
    assert w.shape == (N, K) and w.stride() == (K, 1)
    return w, scale.reshape(N, 1), (w.float() * scale[:, None])


# The plan the server ran before the v3 retune (commit 1eb2831e5c): _BY_SHAPE's fp8
# keys were contig_n=True, which no call matches, so every fp8 shape fell through to
# _plan's fallback tile of [32, 128] with the planner's split count, 4 warps, 3 stages.
# The two bf16 keys were already contig_n=False, so those entries really were live.
PRE_V3_TUNED = {
    (2, 96, 2560): (32, 512, 1, True, None, 4, 3),
    (2, 512, 2560): (16, 128, 8, True, None, 4, 3),
}


def pre_v3_plan(M, N, K, w_bytes, sms=188):
    """The (BLOCK_N, BLOCK_K, SPLITS, USE_DOT, W_KN, warps, stages) the server used."""
    import triton

    from sglang.srt.layers.quantization import w8a16_gemv as k

    pinned = PRE_V3_TUNED.get((w_bytes, N, K))
    if pinned is not None:
        return pinned
    block_n, block_k, warps, stages = 32, 128, 4, 3
    n_blocks = triton.cdiv(N, block_n)
    if n_blocks >= sms:
        return (block_n, block_k, 1, True, None, warps, stages)
    want = min(k._MAX_SPLITS, max(1, (2 * sms) // n_blocks))
    block_k = min(block_k, max(64, k._prev_pow2(max(1, K // want))))
    n_kb = triton.cdiv(K, block_k)
    splits = min(want, n_kb)
    for cand in range(min(n_kb, splits + 2), max(1, splits - 3), -1):
        if n_kb % cand == 0 and cand <= k._MAX_SPLITS:
            splits = cand
            break
    return k._fit(M, N, (block_n, block_k, splits, True, None, warps, stages))
