"""Load / save the 31 ``mtp.*`` tensors and the two frozen shared matrices.

The module's ``state_dict()`` keys are the checkpoint names with the ``mtp.``
prefix stripped -- a pure 1:1 identity mapping, deliberately: no rename table
can drift out of sync with the write-back.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Tuple

import torch

MTP_PREFIX = "mtp."
EMBED_KEY = "model.language_model.embed_tokens.weight"
LM_HEAD_KEY = "lm_head.weight"
INDEX_FILE = "model.safetensors.index.json"

# Tensors the training loop must never update (they still round-trip verbatim).
FROZEN_SUFFIXES = (
    "layers.0.mlp.gate.weight",
    "layers.0.self_attn.indexer.index_qk_proj.weight",
    "layers.0.self_attn.indexer.q_layernorm.weight",
    "layers.0.self_attn.indexer.k_layernorm.weight",
)


# Parameters that exist in the module but never in a shipped checkpoint: the
# frozen shared heads, and the EAGLE-3 fusion entry, which is created
# zero-initialised (see model.py) and only ever comes from a training run.
NOT_IN_CHECKPOINT = ("embed_tokens", "lm_head")
TAP_PREFIXES = ("pre_fc_norm_tap.", "fc_hidden_tap.")


def unloaded_names(missing: Iterable[str]) -> List[str]:
    """The subset of ``load_state_dict`` misses that is a real problem."""
    return [
        m
        for m in missing
        if m not in NOT_IN_CHECKPOINT and not m.startswith(TAP_PREFIXES)
    ]


def read_weight_map(model_dir: str) -> Dict[str, str]:
    with open(os.path.join(model_dir, INDEX_FILE), "r") as handle:
        return json.load(handle)["weight_map"]


def mtp_tensor_names(model_dir: str) -> List[str]:
    return sorted(k for k in read_weight_map(model_dir) if k.startswith(MTP_PREFIX))


def _load_from_shards(
    model_dir: str, keys: Iterable[str], device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    from safetensors import safe_open

    weight_map = read_weight_map(model_dir)
    wanted = list(keys)
    by_shard: Dict[str, List[str]] = {}
    for key in wanted:
        if key not in weight_map:
            raise KeyError(f"{key} is not in {INDEX_FILE}")
        by_shard.setdefault(weight_map[key], []).append(key)
    out: Dict[str, torch.Tensor] = {}
    for shard, shard_keys in by_shard.items():
        with safe_open(os.path.join(model_dir, shard), framework="pt", device=device) as f:
            for key in shard_keys:
                out[key] = f.get_tensor(key)
    return out


def load_mtp_state_dict(model_dir: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
    raw = _load_from_shards(model_dir, mtp_tensor_names(model_dir), device=device)
    return {k[len(MTP_PREFIX) :]: v for k, v in raw.items()}


def load_shared_heads(
    model_dir: str, device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(embed_tokens, lm_head).

    Both live BF16 inside the NVFP4 serving checkpoint (they are on the
    quantizer's ignore list), so the 131-shard BF16 mirror is not needed.
    """
    raw = _load_from_shards(model_dir, [EMBED_KEY, LM_HEAD_KEY], device=device)
    return raw[EMBED_KEY], raw[LM_HEAD_KEY]


def load_into(
    model,
    model_dir: str,
    device: str = "cpu",
    dtype=torch.bfloat16,
    head_dtype=torch.bfloat16,
):
    """Populate an ``MTPHead`` in place; returns the list of loaded names.

    ``dtype`` applies to the 31 trainable tensors (FP32 for master weights);
    ``head_dtype`` to the frozen embed / lm_head, which stay BF16 -- promoting
    them would cost 5.1 GiB and buy nothing, and the chunked CE backward
    multiplies in the head's dtype.
    """
    state = load_mtp_state_dict(model_dir, device=device)
    state = {k: v.to(dtype) for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = unloaded_names(missing)
    if missing or unexpected:
        raise RuntimeError(f"MTP load mismatch: missing={missing} unexpected={unexpected}")
    embed, lm_head = load_shared_heads(model_dir, device=device)
    model.set_frozen_heads(embed.to(head_dtype), lm_head.to(head_dtype))
    return sorted(state)


def model_identity(model_dir: str) -> dict:
    """Content identity of a serving checkpoint (not its path: a moved or
    written-back checkpoint with the same files is the same model).

    Every file under the directory counts, by the digest of its full content
    (symlinks followed): the weight shards and their index, ``config.json``,
    and the tokenizer files (``tokenizer.json``, ``tokenizer_config.json``,
    the chat template, special-tokens maps), so a swapped shard or tokenizer
    changes the identity even at the same name and size.  Hidden files and
    directories (download caches) are skipped.  The first call hashes the
    whole checkpoint; ``file_digest`` caches each digest keyed on the file's
    size, mtime, inode and device.
    """
    from .fileio import digest_parts, file_digest

    d = os.path.expanduser(model_dir)
    if not os.path.isdir(d):
        raise SystemExit(f"model dir {model_dir}: no such directory")
    files = []
    for root, dirs, names in os.walk(d):
        dirs[:] = sorted(x for x in dirs if not x.startswith("."))
        for n in sorted(names):
            path = os.path.join(root, n)
            if n.startswith(".") or not os.path.isfile(path):
                continue
            files.append((os.path.relpath(path, d), file_digest(path)))
    if not files:
        raise SystemExit(f"model dir {model_dir}: no file to identify it by")
    tok = [f for f in files if "token" in f[0].lower() or "chat_template" in f[0].lower()
           or f[0].endswith((".jinja", ".model", ".tiktoken"))]
    # "tokenizer" is part of "files" too; it only makes a diff readable
    return {"files": digest_parts(sorted(files)), "tokenizer": digest_parts(sorted(tok))}


def export_state_dict(model, dtype=torch.bfloat16) -> Dict[str, torch.Tensor]:
    """Module state -> checkpoint-named BF16 tensors ready for the write-back."""
    return {
        MTP_PREFIX + k: v.detach().to(dtype).contiguous()
        for k, v in model.state_dict().items()
    }


__all__ = [
    "MTP_PREFIX",
    "NOT_IN_CHECKPOINT",
    "TAP_PREFIXES",
    "unloaded_names",
    "EMBED_KEY",
    "LM_HEAD_KEY",
    "FROZEN_SUFFIXES",
    "read_weight_map",
    "mtp_tensor_names",
    "load_mtp_state_dict",
    "load_shared_heads",
    "load_into",
    "export_state_dict",
    "model_identity",
]
