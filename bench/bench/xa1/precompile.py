"""xa1: compile every Triton variant the GPU job launches, on the CPU only (no GPU, no lock).

Same approach as dg1/precompile.py: a CPU process that reports sm_120 fills the Triton cache the
GPU job reads (cache key = source, target, options, env; not the device). Covers the candidate
kernel (A and B, the sweep grid) and the two production QSA kernels the server path launches.

  source bench/xa1/env.sh; CUDA_VISIBLE_DEVICES= python -m xa1.precompile [--threads 12]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import os
import sys
import time

import torch


def _fake_sm120_driver():
    import triton
    from triton.backends.nvidia.driver import CudaDriver

    drv = CudaDriver()
    drv.get_current_device = lambda: 0
    drv.get_current_stream = lambda idx=None: 0
    drv.get_device_capability = lambda device=None: (12, 0)
    triton.runtime.driver.set_active(drv)


class _Warmup:
    """kernel[grid](...) -> kernel.warmup(..., grid=grid): compile or load, never launch."""

    def __init__(self, fn):
        self._fn = fn

    def __getitem__(self, grid):
        return lambda *a, **kw: self._fn.warmup(*a, grid=grid, **kw)


@contextlib.contextmanager
def _prod_kernels_warmup():
    from sglang.srt.layers.attention.qsa import sparse_attn as sa

    saved = (sa._fa2_valid_counts, sa._compact_kv)
    sa._fa2_valid_counts, sa._compact_kv = _Warmup(saved[0]), _Warmup(saved[1])
    try:
        yield sa
    finally:
        sa._fa2_valid_counts, sa._compact_kv = saved


def _warm_prod(sa, topk, r2t_width, rows=4):
    from xa1 import cases

    lens = torch.ones(rows, dtype=torch.int32)
    idx = torch.zeros((rows, topk), dtype=torch.int32)
    counts = torch.zeros(rows, dtype=torch.int32)
    sa.qwen_sparse_valid_counts_triton(lens, idx, counts, rows, topk)
    pool = torch.zeros((64, cases.HKV, cases.D), dtype=torch.float8_e4m3fn)
    r2t = torch.zeros((2, r2t_width), dtype=torch.int32)
    cu = torch.zeros(rows + 1, dtype=torch.int32)
    sa.qwen_sparse_kv_extraction_compact_triton(pool, pool, r2t, lens, idx, lens, cu, pool, pool,
                                                rows, topk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        sys.exit("run with CUDA_VISIBLE_DEVICES= (CPU only)")
    from xa1 import cases
    from xa1.grid import (MAX_SPLITS, PACKED_STRIDE, R2T_WIDTH, SMEM_LIMIT, TOPK_FRESH,
                          TOPK_SHARED_W4, TOPK_SHARED_W16, TRITON_CACHE, variants)
    if os.path.realpath(os.environ.get("TRITON_CACHE_DIR", "")) != os.path.realpath(TRITON_CACHE):
        sys.exit(f"export TRITON_CACHE_DIR={TRITON_CACHE}")
    _fake_sm120_driver()
    from triton.runtime._async_compile import AsyncCompileMode

    from xa1.kernels import XA1Workspace, xa1_decode_gather, xa1_decode_packed

    t0 = time.time()
    ws = XA1Workspace(max_groups=2, max_splits=MAX_SPLITS, head_dim=cases.D, device="cpu")
    q = torch.zeros((1, cases.HQ, cases.D), dtype=torch.bfloat16)
    pool = torch.zeros((64, cases.HKV, cases.D), dtype=torch.float8_e4m3fn)
    lens = torch.ones(1, dtype=torch.int32)
    r2t = torch.zeros((2, R2T_WIDTH), dtype=torch.int32)
    jobs = variants()
    compiled = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as pool_ex:
        with AsyncCompileMode(pool_ex):
            for kind, ncols, cfg in jobs:
                common = dict(bmm1_scale=cases.SCALING, bmm2_scale=1.0, cfg=cfg, ws=ws, pdl=True,
                              out=torch.empty_like(q), warmup=True)
                if kind == "B":
                    idx = torch.zeros((1, ncols), dtype=torch.int32)
                    k = xa1_decode_gather(q, pool, pool, r2t, lens, idx, lens,
                                          prefix_valid=ncols == TOPK_FRESH, **common)
                else:
                    k = xa1_decode_packed(q, pool, pool, lens, stride=PACKED_STRIDE, **common)
                compiled.append((kind, ncols, cfg, k))
            with _prod_kernels_warmup() as sa:
                for topk in (TOPK_FRESH, TOPK_SHARED_W16, TOPK_SHARED_W4):
                    _warm_prod(sa, topk, R2T_WIDTH)
    print(f"[xa1-precompile] {len(jobs)} candidate variants + 6 prod QSA kernels in "
          f"{time.time() - t0:.1f}s -> {TRITON_CACHE}", flush=True)
    # A launch over the smem limit raises OutOfResources and ends the whole GPU job.
    big = [(kind, ncols, cfg.key, k.metadata.shared) for kind, ncols, cfg, k in compiled
           if k.metadata.shared > SMEM_LIMIT]
    for b in big:
        print(f"[xa1-precompile] OVER SMEM LIMIT {b}", flush=True)
    sys.exit(1 if big else 0)


if __name__ == "__main__":
    main()
