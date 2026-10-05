"""RT1: softmax top-k router with one packed-key reduction per pick (FC_fc-moe #3, step 1).

Production's `_router_triton_kernel` (sglang/kernels/ops/moe/moe_fused_gate.py) picks each of
the 10 experts with three dependent warp reductions: max of the biased logits, min lane id among
the maxima (lowest id wins ties), and a masked sum to fetch the winner's softmax weight.  Each is
16 in-thread steps plus 5 shuffle levels, so the pick loop is ~30 shuffle levels per expert.

Here every logit is packed with its inverted lane id into one int64 key,

    key = (order-preserving int32 of the float) << 32 | (BLOCK_N - 1 - lane)

so a single int64 max gives both the winner and the tie-break (lowest id first), and the winner's
weight is recomputed from the float decoded out of the key with the same three instructions
production uses for every lane (fsub, fmul by log2e + ex2.approx, div.full by the row sum).

Bit-exactness against production rests on:
  * everything up to `row_sum` is the production code verbatim (same layout, so the same
    16-step in-thread chain + xor 16/8/4/2/1 butterfly that the 2026-10-01 LLIR shows);
  * the winner set and order are those of production: the key order is the float order with
    ties broken by the lower lane, and -0.0 never occurs because the logit gets `+ bias`
    (production passes a zero fp32 bias for softmax, and -0.0 + 0.0 = +0.0);
  * the routed sum and the renormalize division are the production expressions on a
    [1, BLOCK_K] tensor of the same layout (EXPLICIT_SUM=1 spells the order out instead:
    slots 0..7 in sequence, plus slots 8 + 9, which is what the LLIR shows for K=10).
A NaN logit makes production's row sum, so every weight of that row, NaN; the same holds here, and
the ids rank NaN at production's -1e30 floor.  `route_bench.py --check` compares the outputs bit for
bit (NaN == NaN).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _route_softmax_fast_kernel(
    scores_ptr,  # [M, N] bf16/fp32 router logits
    bias_ptr,  # [N] fp32 (production: zeros)
    out_weights_ptr,  # [M, K] fp32
    out_indices_ptr,  # [M, K] int32
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    USE_PDL: tl.constexpr,
    EXPLICIT_SUM: tl.constexpr,
    stride_sm,
    stride_sn,
    stride_wm,
    stride_wk,
    stride_im,
    stride_ik,
):
    # ---- production `_router_triton_kernel`, SCORING_FUNC == 2, BLOCK_M == 1, verbatim ----
    pid = tl.program_id(0)
    offs_m = pid * 1 + tl.arange(0, 1)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)

    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    row_ptr = scores_ptr + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
    mask2d = mask_m[:, None] & mask_n[None, :]
    scores = tl.load(row_ptr, mask=mask2d, other=0.0).to(tl.float32)

    logit = scores
    biased = logit + bias[None, :]
    biased = tl.where(mask_n[None, :], biased, -float("inf"))
    row_max = tl.max(biased, axis=1)[:, None]  # [1, 1]
    exp_row = tl.where(mask_n[None, :], tl.exp(biased - row_max), 0.0)
    row_sum = tl.sum(exp_row, axis=1)[:, None]  # [1, 1]

    biased = tl.where(mask_n[None, :], biased, -float("inf"))
    biased = tl.where(biased == biased, biased, -1e30)

    # ---- packed-key pick loop ----
    bits = biased.to(tl.int32, bitcast=True)
    sbits = bits ^ ((bits >> 31) & 0x7FFFFFFF)  # signed-int order == float order (no NaN, no -0)
    lo = (BLOCK_N - 1) - offs_n  # lower lane -> larger key on ties
    keys = (sbits.to(tl.int64) << 32) | lo[None, :].to(tl.int64)
    gone = tl.full([1, BLOCK_N], -9223372036854775807, tl.int64)

    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    selected_vals = tl.zeros([1, BLOCK_K], dtype=tl.float32)
    selected_idx = tl.zeros([1, BLOCK_K], dtype=tl.int32)
    cur = keys
    for k in tl.static_range(K):
        kmax = tl.max(cur, axis=1)[:, None]  # [1, 1] int64
        cur = tl.where(cur == kmax, gone, cur)
        win_lane = (BLOCK_N - 1) - kmax.to(tl.int32)  # low word = BLOCK_N - 1 - lane
        hi = (kmax >> 32).to(tl.int32)
        wb = (hi ^ ((hi >> 31) & 0x7FFFFFFF)).to(tl.float32, bitcast=True)
        win_activated = tl.exp(wb - row_max) / row_sum  # production's activated[win]
        slot = offs_k[None, :] == k
        selected_vals = tl.where(slot, win_activated, selected_vals)
        selected_idx = tl.where(slot, win_lane, selected_idx)
        if EXPLICIT_SUM:
            if k == 0:
                s_lo = win_activated
            elif k < 8:
                s_lo = s_lo + win_activated
            elif k == 8:
                s_hi = win_activated
            else:
                s_hi = s_hi + win_activated

    if EXPLICIT_SUM:
        routed_sum = s_lo + s_hi  # K == 10 only (asserted by the wrapper)
    else:
        routed_sum = tl.sum(tl.where(mask_k[None, :], selected_vals, 0.0), axis=1)[:, None]

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    if RENORMALIZE:
        norm = tl.where(routed_sum > 0.0, routed_sum, 1.0)
        selected_vals = selected_vals / norm

    out_w_ptr = out_weights_ptr + offs_m[:, None] * stride_wm + offs_k[None, :] * stride_wk
    out_i_ptr = out_indices_ptr + offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik
    store_mask = mask_m[:, None] & mask_k[None, :]
    tl.store(out_w_ptr, selected_vals, mask=store_mask)
    tl.store(out_i_ptr, selected_idx, mask=store_mask)


def route_softmax_fast(
    scores: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    renormalize: bool = True,
    explicit_sum: bool = False,
    use_pdl: bool = True,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
):
    """Drop-in for moe_fused_gate(scores, bias, topk, "softmax", renormalize=...) with no
    shared experts, groups, softcap or scaling (the Qwen3.8-Flash-Next call)."""
    assert scores.ndim == 2 and bias.ndim == 1 and scores.size(1) == bias.size(0)
    assert bias.dtype == torch.float32
    M, N = scores.shape
    assert not explicit_sum or topk == 10, "EXPLICIT_SUM spells out the K=10 order only"
    BLOCK_N = triton.next_power_of_2(N)
    assert BLOCK_N <= 512, "production uses 1 warp only up to BLOCK_N 512"
    BLOCK_K = triton.next_power_of_2(topk)
    if out is None:
        weights = torch.empty((M, topk), dtype=torch.float32, device=scores.device)
        indices = torch.empty((M, topk), dtype=torch.int32, device=scores.device)
    else:
        weights, indices = out
    extra = {"launch_pdl": True} if use_pdl else {}
    _route_softmax_fast_kernel[(M,)](
        scores,
        bias,
        weights,
        indices,
        M,
        N=N,
        K=topk,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        RENORMALIZE=bool(renormalize),
        USE_PDL=use_pdl,
        EXPLICIT_SUM=explicit_sum,
        stride_sm=scores.stride(0),
        stride_sn=scores.stride(1),
        stride_wm=weights.stride(0),
        stride_wk=weights.stride(1),
        stride_im=indices.stride(0),
        stride_ik=indices.stride(1),
        num_warps=1,
        **extra,
    )
    return weights, indices


@triton.jit
def _consume_kernel(scores_ptr, out_ptr, stride_sm, BLOCK_N: tl.constexpr, USE_PDL: tl.constexpr):
    """Dependency floor: wait for the producer, read the row, one reduction, one store."""
    pid = tl.program_id(0)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    x = tl.load(scores_ptr + pid * stride_sm + tl.arange(0, BLOCK_N)).to(tl.float32)
    m = tl.max(x, axis=0)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(out_ptr + pid, m)


def consume(scores: torch.Tensor, out: torch.Tensor, use_pdl: bool = True):
    M, N = scores.shape
    extra = {"launch_pdl": True} if use_pdl else {}
    _consume_kernel[(M,)](scores, out, scores.stride(0), BLOCK_N=N, USE_PDL=use_pdl,
                          num_warps=1, **extra)
    return out
