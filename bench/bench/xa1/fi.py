"""xa1: production FlashInfer XQA without any JIT build.

The server's QSA path calls flashinfer.decode.trtllm_batch_decode_with_kv_cache, which on sm120
picks backend "xqa" and loads the module below from the JIT cache. Here the production .so is
copied (read-only source) into the private FLASHINFER_WORKSPACE_BASE from env.sh, and
JitSpec.build is replaced so nothing is ever compiled and ninja never runs on any cache.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import shutil

XQA_NAME = ("xqa_input_bf16_kv_cache_e4m3_output_bf16_page_size_64_head_dim_256_head_group_ratio_12"
            "_use_sliding_window_False_use_spec_dec_False_spec_q_seq_len_1")
PROD_JIT_DIR = pathlib.Path.home() / ".cache/sglang/.cache/flashinfer/0.6.17/120f/cached_ops"


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_trtllm_decode():
    base = os.environ.get("FLASHINFER_WORKSPACE_BASE", "")
    if not base or os.path.realpath(base).startswith(os.path.realpath(
            pathlib.Path.home() / ".cache/sglang")):
        raise RuntimeError("source xa1/env.sh first (private FLASHINFER_WORKSPACE_BASE)")
    import flashinfer.jit.core as core
    import flashinfer.jit.env as jit_env

    def _build(self, *a, **k):
        if self.is_compiled:
            return
        raise RuntimeError(f"xa1 nojit: {self.name} is not compiled; refusing to build it")

    for cls in (core.JitSpec, *core.JitSpec.__subclasses__()):
        if "build" in cls.__dict__:
            cls.build = _build

    src = PROD_JIT_DIR / XQA_NAME / f"{XQA_NAME}.so"
    dst = jit_env.FLASHINFER_JIT_DIR / XQA_NAME / f"{XQA_NAME}.so"
    if not str(dst).startswith(base.rstrip("/") + "/"):
        raise RuntimeError(f"unexpected FlashInfer JIT dir {dst}")
    if not dst.exists() or _sha(dst) != _sha(src):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache

    return trtllm_batch_decode_with_kv_cache, {"xqa_so": str(src), "sha256_16": _sha(src),
                                                "private_copy": str(dst)}
