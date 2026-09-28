"""Fused Qwen4-Exp MTP entry normalization and projections."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _mtp_entry_kernel(
    embedding_ptr,
    hidden_ptr,
    embedding_norm_weight_ptr,
    hidden_norm_weight_ptr,
    embedding_proj_weight_ptr,
    hidden_proj_weight_ptr,
    out_ptr,
    eps,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NORM_BLOCK: tl.constexpr,
):
    row = tl.arange(0, BLOCK_M)
    row_mask = row < ROWS

    # First pass: one row at a time keeps the norm working set one-dimensional.
    # The 4096-wide tile covers embedding K in one masked load and hidden K in
    # three loads, while retaining only the 16 inverse-RMS scalars afterwards.
    norm_offset = tl.arange(0, NORM_BLOCK)
    embedding_inv_rms = tl.zeros((BLOCK_M,), dtype=tl.float32)
    hidden_inv_rms = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for m in range(0, ROWS):
        embedding = tl.load(
            embedding_ptr + m * K + norm_offset,
            mask=norm_offset < K,
            other=0.0,
        ).to(tl.float32)
        embedding_sum_sq = tl.sum(embedding * embedding, axis=0)
        embedding_row_inv_rms = tl.rsqrt(embedding_sum_sq / K + eps)
        embedding_inv_rms = tl.where(
            row == m, embedding_row_inv_rms, embedding_inv_rms
        )

        hidden_sum_sq = 0.0
        for hidden_k0 in range(0, 4 * K, NORM_BLOCK):
            hidden_k = hidden_k0 + norm_offset
            hidden = tl.load(
                hidden_ptr + m * (4 * K) + hidden_k,
                mask=hidden_k < 4 * K,
                other=0.0,
            ).to(tl.float32)
            hidden_sum_sq += tl.sum(hidden * hidden, axis=0)
        hidden_row_inv_rms = tl.rsqrt(hidden_sum_sq / (4 * K) + eps)
        hidden_inv_rms = tl.where(row == m, hidden_row_inv_rms, hidden_inv_rms)

    j = tl.program_id(0) * BLOCK_J + tl.arange(0, BLOCK_J)
    k_off = tl.arange(0, BLOCK_K)
    embedding_acc = tl.zeros((BLOCK_M, BLOCK_J), dtype=tl.float32)
    hidden_acc_0 = tl.zeros((BLOCK_M, BLOCK_J), dtype=tl.float32)
    hidden_acc_1 = tl.zeros((BLOCK_M, BLOCK_J), dtype=tl.float32)
    hidden_acc_2 = tl.zeros((BLOCK_M, BLOCK_J), dtype=tl.float32)
    hidden_acc_3 = tl.zeros((BLOCK_M, BLOCK_J), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + k_off
        embedding = tl.load(
            embedding_ptr + row[:, None] * K + k[None, :],
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        embedding_norm_weight = tl.load(embedding_norm_weight_ptr + k).to(
            tl.float32
        )
        normalized_embedding = (
            embedding
            * embedding_inv_rms[:, None]
            * (1.0 + embedding_norm_weight[None, :])
        ).to(tl.bfloat16)
        embedding_weight = tl.load(
            embedding_proj_weight_ptr + j[:, None] * K + k[None, :]
        )
        embedding_acc += tl.dot(
            normalized_embedding,
            tl.trans(embedding_weight),
            out_dtype=tl.float32,
        )

        # Reuse one weight chunk for all four HC branches, but feed each dot
        # sequentially so only one [16, BLOCK_K] activation tile is live.
        hidden_weight = tl.load(hidden_proj_weight_ptr + j[:, None] * K + k[None, :])
        hidden = tl.load(
            hidden_ptr + row[:, None] * (4 * K) + k[None, :],
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        hidden_norm_weight = tl.load(hidden_norm_weight_ptr + k).to(tl.float32)
        normalized_hidden = (
            hidden
            * hidden_inv_rms[:, None]
            * (1.0 + hidden_norm_weight[None, :])
        ).to(tl.bfloat16)
        hidden_acc_0 += tl.dot(
            normalized_hidden, tl.trans(hidden_weight), out_dtype=tl.float32
        )

        hidden = tl.load(
            hidden_ptr + row[:, None] * (4 * K) + K + k[None, :],
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        hidden_norm_weight = tl.load(hidden_norm_weight_ptr + K + k).to(tl.float32)
        normalized_hidden = (
            hidden
            * hidden_inv_rms[:, None]
            * (1.0 + hidden_norm_weight[None, :])
        ).to(tl.bfloat16)
        hidden_acc_1 += tl.dot(
            normalized_hidden, tl.trans(hidden_weight), out_dtype=tl.float32
        )

        hidden = tl.load(
            hidden_ptr + row[:, None] * (4 * K) + 2 * K + k[None, :],
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        hidden_norm_weight = tl.load(hidden_norm_weight_ptr + 2 * K + k).to(
            tl.float32
        )
        normalized_hidden = (
            hidden
            * hidden_inv_rms[:, None]
            * (1.0 + hidden_norm_weight[None, :])
        ).to(tl.bfloat16)
        hidden_acc_2 += tl.dot(
            normalized_hidden, tl.trans(hidden_weight), out_dtype=tl.float32
        )

        hidden = tl.load(
            hidden_ptr + row[:, None] * (4 * K) + 3 * K + k[None, :],
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        hidden_norm_weight = tl.load(hidden_norm_weight_ptr + 3 * K + k).to(
            tl.float32
        )
        normalized_hidden = (
            hidden
            * hidden_inv_rms[:, None]
            * (1.0 + hidden_norm_weight[None, :])
        ).to(tl.bfloat16)
        hidden_acc_3 += tl.dot(
            normalized_hidden, tl.trans(hidden_weight), out_dtype=tl.float32
        )

    # Both linears materialize BF16 before the reference BF16 add.
    embedding_result = embedding_acc.to(tl.bfloat16).to(tl.float32)
    hidden_result_0 = hidden_acc_0.to(tl.bfloat16).to(tl.float32)
    hidden_result_1 = hidden_acc_1.to(tl.bfloat16).to(tl.float32)
    hidden_result_2 = hidden_acc_2.to(tl.bfloat16).to(tl.float32)
    hidden_result_3 = hidden_acc_3.to(tl.bfloat16).to(tl.float32)
    tl.store(
        out_ptr + row[:, None] * (4 * K) + j[None, :],
        (embedding_result + hidden_result_0).to(tl.bfloat16),
        mask=row_mask[:, None],
    )
    tl.store(
        out_ptr + row[:, None] * (4 * K) + K + j[None, :],
        (embedding_result + hidden_result_1).to(tl.bfloat16),
        mask=row_mask[:, None],
    )
    tl.store(
        out_ptr + row[:, None] * (4 * K) + 2 * K + j[None, :],
        (embedding_result + hidden_result_2).to(tl.bfloat16),
        mask=row_mask[:, None],
    )
    tl.store(
        out_ptr + row[:, None] * (4 * K) + 3 * K + j[None, :],
        (embedding_result + hidden_result_3).to(tl.bfloat16),
        mask=row_mask[:, None],
    )


def mtp_entry_fused(
    input_embeds: torch.Tensor,
    hidden_states: torch.Tensor,
    embedding_norm_weight: torch.Tensor,
    hidden_norm_weight: torch.Tensor,
    embedding_proj_weight: torch.Tensor,
    hidden_proj_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fuse the W16 Qwen4-Exp MTP entry chain into one Triton launch."""
    if input_embeds.dim() != 2 or hidden_states.dim() != 2:
        raise ValueError("MTP entry inputs must be 2-D")
    rows, hidden_size = input_embeds.shape
    tensors = (
        input_embeds,
        hidden_states,
        embedding_norm_weight,
        hidden_norm_weight,
        embedding_proj_weight,
        hidden_proj_weight,
    )
    if not 1 <= rows <= 16:
        raise ValueError("fused MTP entry requires 1 <= M <= 16")
    if hidden_size != 2560 or hidden_states.shape != (rows, 4 * hidden_size):
        raise ValueError("fused MTP entry supports the W16 (4 x 2560) layout only")
    if embedding_norm_weight.shape != (hidden_size,):
        raise ValueError("invalid embedding norm weight shape")
    if hidden_norm_weight.shape != (4 * hidden_size,):
        raise ValueError("invalid hidden norm weight shape")
    if embedding_proj_weight.shape != (hidden_size, hidden_size):
        raise ValueError("invalid embedding projection weight shape")
    if hidden_proj_weight.shape != (hidden_size, hidden_size):
        raise ValueError("invalid hidden projection weight shape")
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("fused MTP entry requires CUDA tensors")
    if any(tensor.device != input_embeds.device for tensor in tensors):
        raise ValueError("all fused MTP entry tensors must be on one device")
    if any(tensor.dtype != torch.bfloat16 for tensor in tensors):
        raise ValueError("fused MTP entry supports BF16 only")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all fused MTP entry tensors must be contiguous")

    out = torch.empty_like(hidden_states)
    _mtp_entry_kernel[(hidden_size // 64,)](
        input_embeds,
        hidden_states,
        embedding_norm_weight,
        hidden_norm_weight,
        embedding_proj_weight,
        hidden_proj_weight,
        out,
        eps,
        K=hidden_size,
        ROWS=rows,
        BLOCK_M=16,
        BLOCK_J=64,
        BLOCK_K=128,
        NORM_BLOCK=4096,
        num_warps=8,
        num_stages=1,
    )
    return out
