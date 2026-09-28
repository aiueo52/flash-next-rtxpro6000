"""Geometry of the Qwen3.8-Flash-Next MTP draft head.

Every field is read from the serving checkpoint's ``config.json`` (the real
values live under ``text_config``; the top level only carries the multimodal
wrapper).  ``MTPConfig.tiny()`` returns a reduced-dimension twin used by the
CPU unit tests.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from typing import Any

# Default location of the serving checkpoint.  A placeholder: every script also
# takes --model-dir, and MTP_MODEL_DIR overrides this default.  An MTP_MODEL_DIR
# that is set but empty stays empty (and every tool refuses it); it is never
# read as "unset".
DEFAULT_MODEL_DIR = (os.environ["MTP_MODEL_DIR"] if "MTP_MODEL_DIR" in os.environ
                     else os.path.expanduser("~/models/RadixArk/Qwen3.8-Flash-Next-NVFP4"))


@dataclass(frozen=True)
class MTPConfig:
    hidden_size: int = 2560
    hc_count: int = 4
    hc_lowrank: int = 320
    head_dim: int = 256
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    vocab_size: int = 248320
    rms_norm_eps: float = 1e-6
    partial_rotary_factor: float = 0.25
    rope_theta: float = 10_000_000.0
    max_position_embeddings: int = 262144
    # Frozen QSA indexer.  Never executed in training: for sequences no longer
    # than ``dense_attention_max_len`` (2048) every candidate token is selected,
    # so attention is exactly dense causal.  The parameters must still exist so
    # the write-back can round-trip all 31 tensors.
    indexer_n_heads: int = 4
    indexer_kv_heads: int = 1
    indexer_head_dim: int = 128
    indexer_budget: int = 2048
    indexer_compress_ratio: int = 4
    # EAGLE-3-style multi-layer entry (option 1 of docs/lab-notes/DRAFT_V2_SPEC.md).  ``tap_layers``
    # names the target decoder layers whose output HC state the entry also
    # consumes on the chain's FIRST forward only; the recursive steps keep the
    # single-tap entry.  Empty = the shipped head, bit for bit.  These ids are
    # not in the checkpoint's config.json: they are a property of the dump
    # (SGLANG_MTP_DUMP_TAPS) and are threaded in explicitly so a head can never
    # be trained against one tap set and served with another.
    tap_layers: tuple = ()

    @property
    def n_taps(self) -> int:
        return len(self.tap_layers)

    @property
    def hc_size(self) -> int:
        return self.hc_count * self.hidden_size

    @property
    def rotary_dim(self) -> int:
        return round(self.head_dim * self.partial_rotary_factor)

    @property
    def q_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def dense_attention_max_len(self) -> int:
        """Longest sequence for which QSA degenerates to dense causal attention.

        ``indexer_budget`` is *tokens selected per query row*, not blocks.  ``block_topk = budget // compress_ratio`` = 512
        blocks, each covering ``compress_ratio`` = 4 tokens, and the expansion
        is truncated back to ``token_topk = budget`` tokens
        (sglang/srt/layers/attention/qsa/{config.py:64-73, qsa_indexer.py:59-61,
        kernel.py:120}).  A query row therefore sees at most 2048 context
        tokens, so dense causal SDPA is exact for sequences <= 2048, not 8192.
        """
        return self.indexer_budget

    # ---------------------------------------------------------------- loading
    @staticmethod
    def _text_config(raw: dict[str, Any]) -> dict[str, Any]:
        return raw.get("text_config", raw)

    @classmethod
    def from_hf_config(cls, raw: dict[str, Any]) -> "MTPConfig":
        text = cls._text_config(raw)
        rope = text.get("rope_parameters") or {}
        mtp = text.get("mtp") or {}
        # The MTP block may override rope_theta; in the shipped checkpoint both
        # are 10_000_000.
        rope_theta = float(
            mtp.get("rope_theta")
            or rope.get("rope_theta")
            or text.get("rope_theta", 10_000.0)
        )
        n_mtp_layers = int(text.get("mtp_num_hidden_layers", 1) or 1)
        if n_mtp_layers != 1:
            raise ValueError(f"only one MTP layer is supported, got {n_mtp_layers}")
        if text.get("mtp_use_dedicated_embeddings", False):
            raise ValueError(
                "checkpoint declares dedicated MTP embeddings; this trainer "
                "assumes the target's embed_tokens / lm_head are shared"
            )
        return cls(
            hidden_size=int(text["hidden_size"]),
            hc_count=int(text["hc_count"]),
            hc_lowrank=int(text["hc_lowrank"]),
            head_dim=int(text["head_dim"]),
            num_attention_heads=int(text["num_attention_heads"]),
            num_key_value_heads=int(text["num_key_value_heads"]),
            num_experts=int(text["num_experts"]),
            num_experts_per_tok=int(text["num_experts_per_tok"]),
            moe_intermediate_size=int(text["moe_intermediate_size"]),
            shared_expert_intermediate_size=int(
                text["shared_expert_intermediate_size"]
            ),
            vocab_size=int(text["vocab_size"]),
            rms_norm_eps=float(text["rms_norm_eps"]),
            partial_rotary_factor=float(
                rope.get("partial_rotary_factor", text.get("partial_rotary_factor", 1.0))
            ),
            rope_theta=rope_theta,
            max_position_embeddings=int(text["max_position_embeddings"]),
            indexer_n_heads=int(text["indexer_n_heads"]),
            indexer_kv_heads=int(text["indexer_kv_heads"]),
            indexer_head_dim=int(text["indexer_head_dim"]),
            indexer_budget=int(text["indexer_budget"]),
            indexer_compress_ratio=int(text["indexer_compress_ratio"]),
        )

    @classmethod
    def from_pretrained(cls, model_dir: str | os.PathLike) -> "MTPConfig":
        with open(os.path.join(str(model_dir), "config.json"), "r") as handle:
            return cls.from_hf_config(json.load(handle))

    @classmethod
    def tiny(cls) -> "MTPConfig":
        """Small twin with the same structure for CPU tests.

        Ratios that matter are preserved: 4 HC streams, GQA (4 q heads / 2 kv
        heads), partial rotary 0.25, gated shared expert, top-k routing.
        """
        return cls(
            hidden_size=32,
            hc_count=4,
            hc_lowrank=8,
            head_dim=16,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_experts=8,
            num_experts_per_tok=3,
            moe_intermediate_size=12,
            shared_expert_intermediate_size=12,
            vocab_size=64,
            rms_norm_eps=1e-6,
            partial_rotary_factor=0.25,
            rope_theta=10_000_000.0,
            max_position_embeddings=512,
            indexer_n_heads=2,
            indexer_kv_heads=1,
            indexer_head_dim=8,
            indexer_budget=32,
            indexer_compress_ratio=4,
        )

    def with_(self, **kwargs: Any) -> "MTPConfig":
        return replace(self, **kwargs)


__all__ = ["MTPConfig", "DEFAULT_MODEL_DIR"]
