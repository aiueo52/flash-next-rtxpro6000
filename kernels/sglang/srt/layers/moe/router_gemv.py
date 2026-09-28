"""Small-batch BF16 router projection for Qwen MoE decode/verify."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _router_gemv_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    m,
    K: tl.constexpr,
    N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    rows = tl.arange(0, BLOCK_M)
    row_mask = rows < m
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(
            x_ptr + rows[:, None] * K + k[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        weight = tl.load(weight_ptr + n[:, None] * K + k[None, :])
        acc += tl.dot(x, tl.trans(weight), out_dtype=tl.float32)

    if HAS_BIAS:
        acc += tl.load(bias_ptr + n)[None, :].to(tl.float32)
    tl.store(
        out_ptr + rows[:, None] * N + n[None, :],
        acc,
        mask=row_mask[:, None],
    )


def router_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``x @ weight.T + bias`` into an FP32 router-logit tensor."""
    if x.dim() != 2 or weight.dim() != 2:
        raise ValueError("x and weight must be 2-D")
    m, k = x.shape
    n, weight_k = weight.shape
    if not 1 <= m <= 16:
        raise ValueError("router GEMV requires 1 <= M <= 16")
    if k != weight_k or k % 128 != 0 or n % 32 != 0:
        raise ValueError("router GEMV requires matching K % 128 == 0 and N % 32 == 0")
    tensors = (x, weight) if bias is None else (x, weight, bias)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("router GEMV requires CUDA tensors")
    if any(tensor.device != x.device for tensor in tensors):
        raise ValueError("all router GEMV tensors must be on the same device")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("router GEMV supports BF16 inputs and weights only")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("router GEMV inputs and weights must be contiguous")
    if bias is not None:
        if bias.shape != (n,) or bias.dtype != torch.bfloat16:
            raise ValueError("router GEMV bias must be contiguous BF16 with shape [N]")
        if not bias.is_contiguous():
            raise ValueError("router GEMV bias must be contiguous")

    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    bias_arg = bias if bias is not None else weight
    _router_gemv_kernel[(n // 32,)](
        x,
        weight,
        bias_arg,
        out,
        m,
        K=k,
        N=n,
        HAS_BIAS=bias is not None,
        BLOCK_M=16,
        BLOCK_N=32,
        BLOCK_K=256,
        num_warps=8,
    )
    return out
