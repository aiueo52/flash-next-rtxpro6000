# SPDX-License-Identifier: Apache-2.0
# The input fusion and layer order follow SGLang's
# python/sglang/srt/models/qwen4_exp_mtp.py (Copyright 2023-2026 SGLang Team,
# Apache-2.0); see NOTICE.
"""The trainable Qwen3.8-Flash-Next MTP draft head.

Row convention (checked against the served head with the project's parity
tools, which are not included here):

    row t consumes  hc_hidden[t]  (the target's 10240-wide pre-mixer residual)
              and  embed(input_ids[t+1])
    at RoPE position ``positions[t] == t``
    and predicts   input_ids[t+2]  /  target_argmax[t+1]

``forward`` returns the draft logits and the draft's *own* pre-mixer HC state,
which is the chain input of the next draft step at serving time.

With ``config.tap_layers`` non-empty the entry additionally consumes the HC
state of those target layers (option 1 of docs/lab-notes/DRAFT_V2_SPEC.md,
EAGLE-3-style multi-layer fusion).  Each tap gets its own ``pre_fc_norm_tap_i`` / ``fc_hidden_tap_i`` and
is summed into the existing fused entry.  The projections are zero-initialised,
so a freshly built fused head reproduces the shipped head bit for bit; only the
chain's first forward sees the taps, because steps 2..S recurse on the head's
own HC state and no target layer is available to them.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MTPConfig
from .layers import GatedResidual, GemmaRMSNorm, MTPDecoderLayer


class MTPHead(nn.Module):
    """31 trainable ``mtp.*`` tensors + the frozen shared embed / lm_head.

    ``state_dict()`` keys are exactly the checkpoint names minus the ``mtp.``
    prefix; ``embed_tokens`` / ``lm_head`` are non-persistent buffers and never
    appear there, so the write-back cannot accidentally emit them.
    """

    def __init__(self, config: MTPConfig):
        super().__init__()
        self.config = config
        self.pre_fc_norm_embedding = GemmaRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        # NOTE: one global RMS over all 4*H features (NOT per stream); the
        # stream split happens after normalisation and fc_hidden is shared.
        self.pre_fc_norm_hidden = GemmaRMSNorm(config.hc_size, config.rms_norm_eps)
        self.fc_embedding = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.fc_hidden = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        # Zero-init so the fused head is numerically identical to the shipped
        # one at step 0 of training; see ``fuse_inputs``.
        self.pre_fc_norm_tap = nn.ModuleList(
            [GemmaRMSNorm(config.hc_size, config.rms_norm_eps)
             for _ in range(config.n_taps)]
        )
        self.fc_hidden_tap = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.hidden_size, bias=False)
             for _ in range(config.n_taps)]
        )
        for linear in self.fc_hidden_tap:
            nn.init.zeros_(linear.weight)
        self.layers = nn.ModuleList([MTPDecoderLayer(config)])
        self.hyper_connection_mixer = GatedResidual(config, use_combine=False)

        self.register_buffer(
            "embed_tokens", torch.zeros(0, config.hidden_size), persistent=False
        )
        self.register_buffer(
            "lm_head", torch.zeros(0, config.hidden_size), persistent=False
        )

    # ------------------------------------------------------------- helpers
    def set_frozen_heads(self, embed: torch.Tensor, lm_head: torch.Tensor) -> None:
        assert embed.shape == (self.config.vocab_size, self.config.hidden_size)
        assert lm_head.shape == (self.config.vocab_size, self.config.hidden_size)
        self.embed_tokens = embed
        self.lm_head = lm_head

    def freeze_non_trainable(self) -> None:
        """Router + indexer frozen; every other ``mtp.*`` tensor trainable."""
        for p in self.parameters():
            p.requires_grad_(True)
        layer = self.layers[0]
        layer.mlp.gate.weight.requires_grad_(False)
        for p in layer.self_attn.indexer.parameters():
            p.requires_grad_(False)

    def tap_parameter_names(self) -> list:
        """The fusion entry's parameters -- the only ones phase 1 trains."""
        return [n for n, _ in self.named_parameters()
                if n.startswith(("pre_fc_norm_tap.", "fc_hidden_tap."))]

    def freeze_all_but_taps(self) -> None:
        """Tap-fusion probe: the shipped 2.607 B head is frozen entirely and only
        the new zero-init fusion entry learns, so any acceptance change is
        attributable to the extra taps and nothing else."""
        if not self.config.n_taps:
            raise ValueError("freeze_all_but_taps needs config.tap_layers")
        wanted = set(self.tap_parameter_names())
        for name, p in self.named_parameters():
            p.requires_grad_(name in wanted)

    # --------------------------------------------------- draft-only MoE knobs
    def set_moe_topk(self, top_k=None, top_k_recur=None, renormalize=True):
        """Inference-only: route the draft MoE to fewer experts per token.

        ``top_k`` applies to every draft forward (including the teacher-forced
        prefix pass, which is the serving ``draft_extend`` phase);
        ``top_k_recur`` overrides it on the recursive chain steps 2..S only --
        ``chain`` sets ``recur_mode`` for those.  ``None`` restores the
        checkpoint's ``num_experts_per_tok``.  The router weights are frozen,
        so this changes no parameter, only the dispatch.
        """
        moe = self.layers[0].mlp
        moe.top_k = (
            self.config.num_experts_per_tok if top_k is None else int(top_k)
        )
        moe.top_k_recur = None if top_k_recur is None else int(top_k_recur)
        moe.renormalize = bool(renormalize)
        moe.recur_mode = False
        return moe

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def num_parameters(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, train

    # -------------------------------------------------------------- fusion
    def fuse_inputs(
        self,
        next_token_ids: torch.Tensor,
        hc_hidden: torch.Tensor,
        tap_hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """(-> [B,T,4H]) ``_fuse_residual_linear_shared`` from qwen4_exp_mtp.py.

        ``tap_hidden`` [..., n_taps*4H] adds the extra target layers, each
        through its own norm and 2560->2560 projection, summed into the same
        per-stream fusion.  ``None`` (the recursive chain steps, or a head
        without taps) leaves the entry exactly as shipped.
        """
        cfg = self.config
        # Dumps are stored BF16; the module may run in FP32 (or under autocast).
        hc_hidden = hc_hidden.to(self.fc_hidden.weight.dtype)
        embeds = F.embedding(next_token_ids, self.embed_tokens).to(
            self.fc_embedding.weight.dtype
        )
        e = self.fc_embedding(self.pre_fc_norm_embedding(embeds))
        normed = self.pre_fc_norm_hidden(hc_hidden)
        streams = normed.unflatten(-1, (cfg.hc_count, cfg.hidden_size))
        h4 = self.fc_hidden(streams)
        if tap_hidden is not None and cfg.n_taps:
            taps = tap_hidden.to(self.fc_hidden.weight.dtype).unflatten(
                -1, (cfg.n_taps, cfg.hc_size)
            )
            for i in range(cfg.n_taps):
                normed_i = self.pre_fc_norm_tap[i](taps[..., i, :])
                streams_i = normed_i.unflatten(-1, (cfg.hc_count, cfg.hidden_size))
                h4 = h4 + self.fc_hidden_tap[i](streams_i)
        return (e.unsqueeze(-2) + h4).flatten(-2)

    def project_logits(self, mixed: torch.Tensor) -> torch.Tensor:
        # The head is kept BF16 even when the trainable weights are FP32 masters,
        # so the activation has to be cast: evaluation runs under no_grad without
        # autocast, and would otherwise hand F.linear an FP32/BF16 pair.
        return F.linear(mixed.to(self.lm_head.dtype), self.lm_head)

    # ------------------------------------------------------------- forward
    def forward_mixed(
        self,
        next_token_ids: torch.Tensor,
        hc_hidden: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        tap_hidden: Optional[torch.Tensor] = None,
    ):
        """-> (mixed [B,T,H], own_hc [B,T,4H]).

        The 2560-wide mixed state is what the lm_head consumes.  Training and
        evaluation stop here and project chunk-wise (see mtptrain/loss.py): a
        full [B,T,248320] logit tensor would dominate memory.
        """
        if next_token_ids.dim() == 1:
            next_token_ids = next_token_ids.unsqueeze(0)
            hc_hidden = hc_hidden.unsqueeze(0)
            if tap_hidden is not None:
                tap_hidden = tap_hidden.unsqueeze(0)
        b, t = next_token_ids.shape
        if positions is None:
            positions = torch.arange(t, device=next_token_ids.device).expand(b, t)
        hc_state = self.fuse_inputs(next_token_ids, hc_hidden, tap_hidden)
        for layer in self.layers:
            hc_state = layer(hc_state, positions, attn_mask=attn_mask)
        own_hc = hc_state
        mixed, _ = self.hyper_connection_mixer.mix(own_hc)
        return mixed, own_hc

    def forward(
        self,
        next_token_ids: torch.Tensor,
        hc_hidden: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        tap_hidden: Optional[torch.Tensor] = None,
    ):
        """next_token_ids [B,T]; hc_hidden [B,T,4H] -> (logits [B,T,V], own_hc)."""
        mixed, own_hc = self.forward_mixed(
            next_token_ids, hc_hidden, positions, attn_mask, tap_hidden
        )
        return self.project_logits(mixed), own_hc


    # ------------------------------------------------------- rollout support
    def _attend_with_keys(self, layer, mixed, positions, past_kv, valid, self_only):
        """Attention for one layer against a supplied K/V, plus the new rows.

        ``self_only`` picks the rollout geometry: every query attends to the
        teacher-forced rows 0..t and to its own new key, but not to the other
        new rows.  With ``self_only=False`` the new rows are causally visible to
        each other, which is the ordinary teacher-forced pass.
        """
        q, k, v, gate = layer.self_attn.project(mixed, positions)
        b, t = q.shape[0], q.shape[1]
        dev = q.device
        if past_kv is None:
            keys, vals, tp = k, v, 0
        else:
            keys = torch.cat((past_kv[0], k), dim=1)
            vals = torch.cat((past_kv[1], v), dim=1)
            tp = past_kv[0].shape[1]
        i = torch.arange(t, device=dev).view(-1, 1)
        if tp:
            j = torch.arange(tp, device=dev).view(1, -1)
            left = j <= i
            right = (
                torch.eye(t, dtype=torch.bool, device=dev)
                if self_only
                else (torch.arange(t, device=dev).view(1, -1) <= i)
            )
            base = torch.cat((left, right), dim=1)
        else:
            base = torch.arange(t, device=dev).view(1, -1) <= i
        mask = base.view(1, 1, t, keys.shape[1])
        if valid is not None:
            key_valid = valid if not tp else torch.cat((valid, valid), dim=1)
            mask = mask & key_valid.view(b, 1, 1, keys.shape[1])
        out = layer.self_attn.attend(q, keys, vals, gate, attn_mask=mask, is_causal=False)
        return out, (k, v)

    def forward_train(
        self, next_token_ids, hc_hidden, positions, valid=None, tap_hidden=None
    ):
        """Teacher-forced pass that also hands back the layer's K/V.

        -> (mixed [B,T,H], own_hc [B,T,4H], (k, v)).  The K/V is what the
        rollout steps attend to.  Every row here is target-fed, so this is the
        pass the taps belong to.
        """
        layer = self.layers[0]
        hc_state = self.fuse_inputs(next_token_ids, hc_hidden, tap_hidden)
        mixed, res = layer.attn_hyper_connection.mix(hc_state)
        attn_out, kv = self._attend_with_keys(
            layer, mixed, positions, None, valid, self_only=False
        )
        hc_state = layer.attn_hyper_connection.combine(attn_out, res)
        mixed2, res2 = layer.mlp_hyper_connection.mix(hc_state)
        own_hc = layer.mlp_hyper_connection.combine(layer.mlp(mixed2), res2)
        out, _ = self.hyper_connection_mixer.mix(own_hc)
        return out, own_hc, kv

    def forward_rolled(self, tokens, hc_in, positions, past_kv, valid=None):
        """One chain step for every row in parallel.

        Row t's query sits at ``positions[t]`` and attends to the teacher-forced
        draft KV of rows 0..t plus its own new key.  The chain's own
        intermediate rows (t+1 .. t+j-1) are *not* in the context; with K=3 that
        is at most two rows missing from a prefix of up to 2048, which is the
        one approximation this rollout makes against serving.
        """
        layer = self.layers[0]
        hc_state = self.fuse_inputs(tokens, hc_in)
        mixed, res = layer.attn_hyper_connection.mix(hc_state)
        attn_out, _ = self._attend_with_keys(
            layer, mixed, positions, past_kv, valid, self_only=True
        )
        hc_state = layer.attn_hyper_connection.combine(attn_out, res)
        mixed2, res2 = layer.mlp_hyper_connection.mix(hc_state)
        own_hc = layer.mlp_hyper_connection.combine(layer.mlp(mixed2), res2)
        out, _ = self.hyper_connection_mixer.mix(own_hc)
        return out, own_hc

    # ------------------------------------------------- cached / chain paths
    def forward_with_cache(
        self,
        next_token_ids: torch.Tensor,
        hc_hidden: torch.Tensor,
        positions: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_mask: Optional[torch.Tensor] = None,
        return_logits: bool = True,
        tap_hidden: Optional[torch.Tensor] = None,
    ):
        """Same as ``forward`` but returns/extends an explicit KV cache.

        Used by the chain-acceptance simulator.  ``past_kv`` holds K/V for the
        rows already in the draft's cache, shaped [B, T_past, kv_heads, D].
        ``past_mask`` [B, T_past] hides padded past rows (chains that start at
        different offsets share one padded cache).
        """
        layer = self.layers[0]
        hc_state = self.fuse_inputs(next_token_ids, hc_hidden, tap_hidden)

        mixed, residuals = layer.attn_hyper_connection.mix(hc_state)
        q, k, v, gate = layer.self_attn.project(mixed, positions)
        t_past = 0 if past_kv is None else past_kv[0].shape[1]
        if past_kv is not None:
            k = torch.cat((past_kv[0], k), dim=1)
            v = torch.cat((past_kv[1], v), dim=1)
        b, tq, tk = q.shape[0], q.shape[1], k.shape[1]
        # Causal by *row index*: the new rows sit at t_past .. tk-1.
        i = torch.arange(tk - tq, tk, device=q.device).unsqueeze(-1)
        j = torch.arange(tk, device=q.device).unsqueeze(0)
        mask = (j <= i).view(1, 1, tq, tk).expand(b, 1, tq, tk)
        if past_mask is not None:
            key_valid = torch.cat(
                (
                    past_mask,
                    torch.ones(b, tk - t_past, dtype=torch.bool, device=q.device),
                ),
                dim=1,
            )
            mask = mask & key_valid.view(b, 1, 1, tk)
        attn_out = layer.self_attn.attend(q, k, v, gate, attn_mask=mask, is_causal=False)
        hc_state = layer.attn_hyper_connection.combine(attn_out, residuals)

        mixed, residuals = layer.mlp_hyper_connection.mix(hc_state)
        mlp_out = layer.mlp(mixed)
        own_hc = layer.mlp_hyper_connection.combine(mlp_out, residuals)

        mixed, _ = self.hyper_connection_mixer.mix(own_hc)
        out = self.project_logits(mixed) if return_logits else mixed
        return out, own_hc, (k, v)

    @torch.no_grad()
    def chain(
        self,
        start_hc: torch.Tensor,
        start_token: torch.Tensor,
        start_position: torch.Tensor,
        past_kv: Tuple[torch.Tensor, torch.Tensor],
        steps: int,
        past_mask: Optional[torch.Tensor] = None,
        allowed: Optional[torch.Tensor] = None,
        start_taps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Roll the topk=1 draft chain forward, exactly like serving.

        ``allowed`` [vocab] bool: restrict the argmax to these ids (the
        serving-time ``--speculative-token-map`` hot vocabulary).

        ``start_hc``/``start_token``/``start_position`` describe chain step 0
        (its inputs come from the *target*); every later step feeds the draft's
        own HC state and its own previous prediction, with the position
        incremented by one.  ``start_taps`` [B, n_taps*4H] likewise belongs to
        step 0 only -- the recursive steps have no target layers to read, which
        is exactly the serving geometry.  Returns predicted ids [B, steps].
        """
        hc = start_hc.unsqueeze(1)
        tok = start_token.unsqueeze(1)
        pos = start_position.unsqueeze(1)
        taps = None if start_taps is None else start_taps.unsqueeze(1)
        kv = past_kv
        mask = past_mask
        preds = []
        moe = self.layers[0].mlp
        for step in range(steps):
            # Steps 2..S recurse on the draft's own HC state; a reduced
            # ``top_k_recur`` applies to them only (step 0 reads the target).
            moe.recur_mode = step > 0
            logits, own_hc, kv = self.forward_with_cache(
                tok, hc, pos, past_kv=kv, past_mask=mask, tap_hidden=taps
            )
            if mask is not None:
                mask = torch.cat(
                    (
                        mask,
                        torch.ones(
                            mask.shape[0], 1, dtype=torch.bool, device=mask.device
                        ),
                    ),
                    dim=1,
                )
            last = logits[:, -1]
            if allowed is not None:
                last = last.masked_fill(~allowed, float("-inf"))
            nxt = last.argmax(dim=-1)
            preds.append(nxt)
            tok = nxt.unsqueeze(1)
            hc = own_hc[:, -1:]
            pos = pos + 1
            taps = None
        moe.recur_mode = False
        return torch.stack(preds, dim=1)


__all__ = ["MTPHead"]
