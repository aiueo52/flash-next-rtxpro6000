"""RT1 fast32: the packed-key router (route_fast.py) on 32-bit keys, for bf16 logits and a zero bias.

With bf16 logits and the zero fp32 bias that fused_topk passes, ``biased = float(logit) + 0.0`` is
a bf16 value (-0.0 becomes +0.0), so its fp32 bits have 16 zero low bits and the order of the
logits is the order of the high halves.  The key is then

    key = v << 9 | (511 - lane),  v = 2 * (order-preserving int16 of the high half)

17 + 9 bits, so every pick is an int32 max (one IMNMX per step and one shuffle per level instead of
the int64 compare/select pairs and two shuffles).  NaN logits, which production floors to -1e30
(fp32, strictly between two bf16 values), get the odd v between those two, so the ranking is still
production's.  Everything with an order (row max, row sum, routed sum, divisions) is the production
code or route_fast.py's, as there.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# -1e30 = 0xF149F2CA: between bf16 0xF149 (-9.95e29) and 0xF14A (-1.0003e30); the order-preserving
# int16 of 0xF149 is -29002, so the slot between is 2 * -29002 - 1.
_NAN_V = -58005


@triton.jit
def _route_softmax_fast32_kernel(
    scores_ptr,  # [M, N] bf16 router logits
    bias_ptr,  # [N] fp32 zeros
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

    # ---- 32-bit keys ----
    bits = biased.to(tl.int32, bitcast=True)
    h = bits >> 16  # sign-extended high half (the bf16 pattern)
    s = h ^ ((h >> 15) & 0x7FFF)  # order-preserving int16
    v = tl.where(biased == biased, s * 2, -58005)  # _NAN_V: production's -1e30 floor
    lo = (BLOCK_N - 1) - offs_n
    keys = (v << 9) | lo[None, :]
    gone = tl.full([1, BLOCK_N], -2147483648, tl.int32)

    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    selected_vals = tl.zeros([1, BLOCK_K], dtype=tl.float32)
    selected_idx = tl.zeros([1, BLOCK_K], dtype=tl.int32)
    cur = keys
    for k in tl.static_range(K):
        kmax = tl.max(cur, axis=1)[:, None]  # [1, 1] int32
        cur = tl.where(cur == kmax, gone, cur)
        win_lane = (BLOCK_N - 1) - (kmax & 511)
        wv = kmax >> 9
        ws = wv >> 1
        wh = ws ^ ((ws >> 15) & 0x7FFF)
        wb = (wh << 16).to(tl.float32, bitcast=True)
        wb = tl.where((wv & 1) != 0, -1e30, wb)
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


def route_softmax_fast32(
    scores: torch.Tensor,
    zero_bias: torch.Tensor,
    topk: int,
    renormalize: bool = True,
    explicit_sum: bool = False,
    use_pdl: bool = True,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
):
    """moe_fused_gate(scores, zero_bias, topk, "softmax", renormalize=...) for bf16 scores and an
    all-zero fp32 bias (the caller vouches for the zeros; fused_topk's _get_zero_bias)."""
    assert scores.ndim == 2 and zero_bias.ndim == 1 and scores.size(1) == zero_bias.size(0)
    assert scores.dtype == torch.bfloat16 and zero_bias.dtype == torch.float32
    M, N = scores.shape
    assert not explicit_sum or topk == 10, "EXPLICIT_SUM spells out the K=10 order only"
    BLOCK_N = triton.next_power_of_2(N)
    assert BLOCK_N <= 512, "9 lane bits; production uses 1 warp only up to BLOCK_N 512"
    BLOCK_K = triton.next_power_of_2(topk)
    if out is None:
        weights = torch.empty((M, topk), dtype=torch.float32, device=scores.device)
        indices = torch.empty((M, topk), dtype=torch.int32, device=scores.device)
    else:
        weights, indices = out
    extra = {"launch_pdl": True} if use_pdl else {}
    _route_softmax_fast32_kernel[(M,)](
        scores,
        zero_bias,
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
