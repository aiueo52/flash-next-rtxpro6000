# SPDX-License-Identifier: Apache-2.0
# Portions adapted from SGLang (https://github.com/sgl-project/sglang):
#   python/sglang/srt/layers/hyperconnection.py (GroupedGemmaRMSNorm, GatedResidual)
#   python/sglang/srt/layers/layernorm.py (GemmaRMSNorm)
#   python/sglang/srt/models/qwen3_5.py, qwen4_exp.py, qwen4_exp_mtp.py
#     (gated attention, MoE with shared expert, decoder layer structure)
# Copyright 2023-2026 SGLang Team; Copyright 2025 Qwen Team.
# Licensed under the Apache License, Version 2.0.  Rewritten here as plain,
# differentiable PyTorch for training; see NOTICE.
"""Building blocks of the Qwen3.8-Flash-Next MTP draft layer.

Every parameter is named exactly like the corresponding ``mtp.*`` checkpoint
tensor so that ``MTPHead.state_dict()`` round-trips through
``writeback/write_mtp.py`` without a rename table.

Numerics follow SGLang's reference (non-fused) paths:
``python/sglang/srt/layers/hyperconnection.py``,
``python/sglang/srt/models/qwen3_5.py`` and ``qwen4_exp.py``; they were
checked against the served head with the project's parity tools (not
included in this directory).

Norm weights keep the **checkpoint convention**: the tensor stores the delta and
the forward applies ``(1 + w)`` (Gemma style), as SGLang does at runtime.
Keeping the delta means gradients land on exactly the tensor we write back.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MTPConfig


# --------------------------------------------------------------------- norms
class GemmaRMSNorm(nn.Module):
    """RMS over the last ``size`` features, scaled by ``(1 + weight)``.

    Matches ``sglang.srt.layers.layernorm.GemmaRMSNorm``.
    """

    def __init__(self, size: int, eps: float = 1e-6):
        super().__init__()
        self.size = size
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * (1.0 + self.weight.float())).to(dtype)


class GroupedGemmaRMSNorm(nn.Module):
    """Per-HC-stream RMS (group of ``group_size``) with one ``(1 + w)`` scale.

    ``sglang.srt.layers.hyperconnection.GroupedGemmaRMSNorm`` with
    ``group_size=hidden_size`` (``hc_per_branch_norm=True`` for Qwen4Exp).
    """

    def __init__(self, size: int, group_size: int, eps: float = 1e-6):
        super().__init__()
        if size % group_size:
            raise ValueError(f"{size} not divisible by group {group_size}")
        self.size = size
        self.group_size = group_size
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        grouped = xf.reshape(*xf.shape[:-1], xf.shape[-1] // self.group_size, self.group_size)
        grouped = grouped * torch.rsqrt(
            grouped.pow(2).mean(-1, keepdim=True) + self.eps
        )
        xn = grouped.flatten(-2)
        return (xn * (1.0 + self.weight.float())).to(dtype)


# ---------------------------------------------------------------- rotary
class RotaryEmbedding(nn.Module):
    """Partial NeoX rotary embedding (rotary_dim = head_dim * partial factor).

    ``get_rope(head_size=head_dim, rotary_dim=head_dim, partial_rotary_factor=
    0.25, is_neox_style=True, base=rope_theta)``.  The checkpoint declares
    ``mrope_interleaved`` with sections [11,11,10], but text-only serving passes
    a 1-D ``positions`` tensor, so ``MRotaryEmbedding.forward_native`` skips the
    section logic entirely -> ordinary 1-D RoPE.  (Even with a 3-row position
    tensor whose rows are equal, ``apply_interleaved_rope`` is the identity.)
    """

    def __init__(self, head_dim: int, rotary_dim: int, base: float, max_position: int):
        super().__init__()
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.base = base
        self.max_position = max_position
        inv_freq = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def cos_sin(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        freqs = positions.float().unsqueeze(-1) * self.inv_freq.to(positions.device)
        return freqs.cos(), freqs.sin()

    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: [..., T, H, rotary_dim];  cos/sin: [..., T, rotary_dim//2]
        x1, x2 = torch.chunk(x, 2, dim=-1)
        cos = cos.unsqueeze(-2).to(x.dtype)
        sin = sin.unsqueeze(-2).to(x.dtype)
        return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)

    def forward(
        self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """q/k: [..., T, heads, head_dim]; positions: [..., T]."""
        cos, sin = self.cos_sin(positions)
        rd = self.rotary_dim
        q = torch.cat((self._rotate(q[..., :rd], cos, sin), q[..., rd:]), dim=-1)
        k = torch.cat((self._rotate(k[..., :rd], cos, sin), k[..., rd:]), dim=-1)
        return q, k


# -------------------------------------------------------- hyper connections
class GatedResidual(nn.Module):
    """Qwen4Exp hyper-connection block.

    ``mix``   : (hc_state[.., 4H]) -> (mixed[.., H], (hc_state, normed))
    ``combine``: (block_out[.., H], (hc_state, normed)) -> hc_state[.., 4H]

    mix      = mean_s( sigmoid(W_up @ silu(W_down @ n / hc)) * n )   over streams
    combine  = hc_state + block_out (x) 2*sigmoid(W_inject @ n / hc)

    where ``n = GroupedGemmaRMSNorm(hc_state)``.  Note both the mix gate and the
    inject gate are computed from the *normalised* residual, while ``combine``
    adds into the *raw* residual.
    """

    def __init__(self, config: MTPConfig, use_combine: bool = True):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count
        hc_size = config.hc_size
        self.hc_norm = GroupedGemmaRMSNorm(
            hc_size, config.hidden_size, config.rms_norm_eps
        )
        self.input_mix_weight_down = nn.Linear(hc_size, config.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(config.hc_lowrank, hc_size, bias=False)
        self.block_inject_weight = (
            nn.Linear(hc_size, config.hc_count, bias=False) if use_combine else None
        )

    def mix(self, hyper_input: torch.Tensor):
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        normed = self.hc_norm(hyper_input)
        w = F.silu(self.input_mix_weight_down(normed) / self.hc_count)
        w = torch.sigmoid(self.input_mix_weight_up(w))
        mixed = (
            w.unflatten(-1, (self.hc_count, self.hidden_size))
            * normed.unflatten(-1, (self.hc_count, self.hidden_size))
        ).mean(dim=-2)
        return mixed, (hyper_input, normed)

    def combine(self, block_output: torch.Tensor, residuals) -> torch.Tensor:
        hyper_input, normed = residuals
        assert self.block_inject_weight is not None
        inject = 2.0 * torch.sigmoid(
            self.block_inject_weight(normed) / self.hc_count
        )
        streams = hyper_input.unflatten(-1, (self.hc_count, self.hidden_size))
        return (streams + block_output.unsqueeze(-2) * inject.unsqueeze(-1)).flatten(-2)


# -------------------------------------------------------------- attention
class QSAIndexer(nn.Module):
    """Frozen QSA indexer.

    Never executed: for sequences <= indexer_budget * compress_ratio (8192) the
    block selection covers the whole prefix, so QSA attention is *exactly* dense
    causal attention.  The parameters exist only so the write-back can emit all
    31 ``mtp.*`` tensors unchanged.
    """

    def __init__(self, config: MTPConfig):
        super().__init__()
        out = (config.indexer_n_heads + config.indexer_kv_heads) * config.indexer_head_dim
        self.index_qk_proj = nn.Linear(config.hidden_size, out, bias=False)
        self.q_layernorm = GemmaRMSNorm(config.indexer_head_dim, config.rms_norm_eps)
        self.k_layernorm = GemmaRMSNorm(config.indexer_head_dim, config.rms_norm_eps)
        for p in self.parameters():
            p.requires_grad_(False)


class MTPAttention(nn.Module):
    """One QSA layer executed as dense causal SDPA with an output gate.

    ``q_proj`` emits ``2 * num_heads * head_dim`` rows laid out **per head** as
    ``[q_h (head_dim) | gate_h (head_dim)]``; this is the split SGLang performs
    (``view(-1, num_heads, 2*head_dim).chunk(2, -1)``).
    """

    def __init__(self, config: MTPConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = config.head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, 2 * config.q_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.kv_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.kv_size, bias=False)
        self.o_proj = nn.Linear(config.q_size, config.hidden_size, bias=False)
        self.q_norm = GemmaRMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(config.head_dim, config.rms_norm_eps)
        self.indexer = QSAIndexer(config)
        self.rotary = RotaryEmbedding(
            config.head_dim,
            config.rotary_dim,
            config.rope_theta,
            config.max_position_embeddings,
        )

    def project(self, hidden_states: torch.Tensor, positions: torch.Tensor):
        """-> q,k,v [B,T,heads,D] (rope applied) and gate [B,T,q_size]."""
        b, t, _ = hidden_states.shape
        qg = self.q_proj(hidden_states).view(b, t, self.num_heads, 2 * self.head_dim)
        q, gate = torch.chunk(qg, 2, dim=-1)
        k = self.k_proj(hidden_states).view(b, t, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(b, t, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary(positions, q, k)
        return q, k, v, gate.reshape(b, t, self.num_heads * self.head_dim)

    def attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        b, tq = q.shape[0], q.shape[1]
        rep = self.num_heads // self.num_kv_heads
        qh = q.transpose(1, 2)  # [B, Hq, Tq, D]
        kh = k.transpose(1, 2).repeat_interleave(rep, dim=1)
        vh = v.transpose(1, 2).repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(
            qh,
            kh,
            vh,
            attn_mask=attn_mask,
            is_causal=is_causal and attn_mask is None,
            scale=self.scaling,
        )
        out = out.transpose(1, 2).reshape(b, tq, self.num_heads * self.head_dim)
        out = out * torch.sigmoid(gate)
        return self.o_proj(out)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q, k, v, gate = self.project(hidden_states, positions)
        return self.attend(q, k, v, gate, attn_mask=attn_mask)


# ------------------------------------------------------------------- MoE
def _grouped_mm_available() -> bool:
    return hasattr(torch, "_grouped_mm") and torch.cuda.is_available()


def _grouped_mm_shapes_ok(hidden_size: int, intermediate_size: int, dtype) -> bool:
    """_grouped_mm requires 16-byte-aligned strides.

    With BF16 that means every leading dimension must be a multiple of 8; the
    real config (H=2560, I=640) qualifies, the reduced test config (I=12) does
    not, so the tiny path stays on the reference loop instead of crashing.
    """
    elems = 16 // max(1, dtype.itemsize)
    return all(d % elems == 0 for d in (hidden_size, intermediate_size, 2 * intermediate_size))


class MTPExperts(nn.Module):
    """512 routed SwiGLU experts stored as two stacked tensors.

    ``gate_up_proj`` [E, 2I, H] with rows ``[gate(I) | up(I)]``;
    ``down_proj``    [E, H, I].  Both are plain ``nn.Linear``-style matrices, in
    the checkpoint's own layout so the write-back needs no permutation.

    Two dispatch paths, numerically identical:

    * **grouped** (CUDA): sort the token/expert assignments by expert, gather
      once, and run two ``torch._grouped_mm`` calls.  Two kernels instead of
      thousands -- the per-expert Python loop issues several launches per
      expert per matmul and dominates the step.
    * **loop** (CPU, or ``force_loop``): the reference implementation, kept for
      the tiny tests, for CPU runs, and as the parity reference.
    """

    def __init__(self, config: MTPConfig):
        super().__init__()
        e, h, i = config.num_experts, config.hidden_size, config.moe_intermediate_size
        self.num_experts = e
        self.intermediate_size = i
        self.gate_up_proj = nn.Parameter(torch.empty(e, 2 * i, h))
        self.down_proj = nn.Parameter(torch.empty(e, h, i))
        self.force_loop = False

    # ------------------------------------------------------------ reference
    def _forward_loop(self, x, topk_ids, topk_weights):
        n, h = x.shape
        k = topk_ids.shape[1]
        flat_ids = topk_ids.reshape(-1)
        flat_w = topk_weights.reshape(-1)
        token_of = torch.arange(n, device=x.device).repeat_interleave(k)

        order = torch.argsort(flat_ids)
        sorted_ids = flat_ids[order]
        sorted_tokens = token_of[order]
        sorted_w = flat_w[order]
        counts = torch.bincount(sorted_ids, minlength=self.num_experts)
        offsets = torch.cumsum(counts, 0) - counts

        out = torch.zeros(n, h, dtype=x.dtype, device=x.device)
        counts_l = counts.tolist()
        offsets_l = offsets.tolist()
        for e in range(self.num_experts):
            c = counts_l[e]
            if c == 0:
                continue
            s = offsets_l[e]
            idx = sorted_tokens[s : s + c]
            xe = x.index_select(0, idx)
            gu = F.linear(xe, self.gate_up_proj[e])
            g, u = gu.chunk(2, dim=-1)
            ye = F.linear(F.silu(g) * u, self.down_proj[e])
            out = out.index_add(0, idx, ye * sorted_w[s : s + c].unsqueeze(-1).to(ye.dtype))
        return out

    # -------------------------------------------------------------- grouped
    def _forward_grouped(self, x, topk_ids, topk_weights):
        n, h = x.shape
        k = topk_ids.shape[1]
        flat_ids = topk_ids.reshape(-1)
        flat_w = topk_weights.reshape(-1)

        order = torch.argsort(flat_ids)
        token_of = torch.div(order, k, rounding_mode="floor")
        counts = torch.bincount(flat_ids[order], minlength=self.num_experts)
        offs = torch.cumsum(counts, 0).to(torch.int32)

        # _grouped_mm is not autocast-aware, so cast explicitly; the cast is a
        # graph node, so the gradient still reaches the FP32 master weights.
        gate_up = self.gate_up_proj.transpose(1, 2)
        down = self.down_proj.transpose(1, 2)
        if gate_up.dtype != x.dtype:
            gate_up = gate_up.to(x.dtype)
            down = down.to(x.dtype)

        xs = x.index_select(0, token_of)
        gu = torch._grouped_mm(xs, gate_up, offs=offs)
        g, u = gu.chunk(2, dim=-1)
        act = (F.silu(g) * u).contiguous()
        ys = torch._grouped_mm(act, down, offs=offs)
        ys = ys * flat_w[order].unsqueeze(-1).to(ys.dtype)
        return torch.zeros(n, h, dtype=ys.dtype, device=x.device).index_add(
            0, token_of, ys
        )

    def forward(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> torch.Tensor:
        """x: [N, H]; topk_ids/weights: [N, K] -> [N, H]."""
        if (
            not self.force_loop
            and x.is_cuda
            and _grouped_mm_available()
            and _grouped_mm_shapes_ok(x.shape[-1], self.intermediate_size, x.dtype)
        ):
            return self._forward_grouped(x, topk_ids, topk_weights)
        return self._forward_loop(x, topk_ids, topk_weights)


class MTPSharedExpert(nn.Module):
    def __init__(self, config: MTPConfig):
        super().__init__()
        i = config.shared_expert_intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, i, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, i, bias=False)
        self.down_proj = nn.Linear(i, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, scalar_gate: torch.Tensor) -> torch.Tensor:
        act = F.silu(self.gate_proj(x)) * self.up_proj(x)
        # Applying the scalar sigmoid gate before the bias-free down projection
        # is algebraically identical to applying it after.
        return self.down_proj(act * torch.sigmoid(scalar_gate))


class MTPMoE(nn.Module):
    """Softmax router, top-10, renormalised; plus a sigmoid-gated shared expert.

    ``renormalize=True`` over a full softmax is identical to a softmax taken
    over the selected logits only, which is what the fused kernels compute.

    **Inference-only draft knobs.**  ``top_k`` may be lowered below the
    checkpoint's ``num_experts_per_tok`` to emulate a server that routes the
    *draft* MoE to fewer experts per token (the target is untouched, so only
    acceptance length can move).  ``top_k_recur`` applies a different value
    while ``recur_mode`` is set, which ``MTPHead.chain`` turns on for the
    recursive chain steps 2..S only.  ``renormalize`` switches between the
    checkpoint's convention (weights renormalised over the selected experts,
    i.e. a softmax over the selected logits) and simply dropping the tail of
    the full softmax, which leaves the routed branch scaled down by the mass
    that was cut -- both are plausible server behaviours, so both are measured.
    """

    def __init__(self, config: MTPConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.top_k_recur = None
        self.recur_mode = False
        self.renormalize = True
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.gate.weight.requires_grad_(False)  # router is frozen
        self.experts = MTPExperts(config)
        self.shared_expert = MTPSharedExpert(config)
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    @property
    def active_top_k(self) -> int:
        if self.recur_mode and self.top_k_recur is not None:
            return self.top_k_recur
        return self.top_k

    def route(self, x: torch.Tensor):
        logits = self.gate(x)
        probs = torch.softmax(logits.float(), dim=-1)
        w, ids = torch.topk(probs, self.active_top_k, dim=-1)
        if self.renormalize:
            w = w / w.sum(dim=-1, keepdim=True)
        return ids, w.to(x.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        ids, w = self.route(x)
        routed = self.experts(x, ids, w)
        shared = self.shared_expert(x, self.shared_expert_gate(x))
        return (shared + routed).reshape(shape)


# ------------------------------------------------------------------ layer
class MTPDecoderLayer(nn.Module):
    def __init__(self, config: MTPConfig):
        super().__init__()
        self.self_attn = MTPAttention(config)
        self.mlp = MTPMoE(config)
        self.attn_hyper_connection = GatedResidual(config, use_combine=True)
        self.mlp_hyper_connection = GatedResidual(config, use_combine=True)

    def forward(
        self,
        hc_state: torch.Tensor,
        positions: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mixed, residuals = self.attn_hyper_connection.mix(hc_state)
        attn_out = self.self_attn(mixed, positions, attn_mask=attn_mask)
        hc_state = self.attn_hyper_connection.combine(attn_out, residuals)

        mixed, residuals = self.mlp_hyper_connection.mix(hc_state)
        mlp_out = self.mlp(mixed)
        return self.mlp_hyper_connection.combine(mlp_out, residuals)


__all__ = [
    "GemmaRMSNorm",
    "GroupedGemmaRMSNorm",
    "RotaryEmbedding",
    "GatedResidual",
    "QSAIndexer",
    "MTPAttention",
    "MTPExperts",
    "MTPSharedExpert",
    "MTPMoE",
    "MTPDecoderLayer",
]
