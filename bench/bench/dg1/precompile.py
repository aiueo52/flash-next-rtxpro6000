"""dg1: compile every Triton variant the GPU job launches, on the CPU only (no GPU, no lock).

Triton's cache key covers source, target (cuda sm_120), options and Triton env vars, not the device,
so a CPU process that reports sm_120 fills the same cache the GPU job reads; the job then only loads
cubins instead of compiling ~60 kernels while it holds the GPU lock.

  CUDA_VISIBLE_DEVICES= TRITON_CACHE_DIR=~/.cache/dg1-triton python -m dg1.precompile
"""
from __future__ import annotations

import concurrent.futures
import os
import sys
import time


def _fake_sm120_driver():
    import triton
    from triton.backends.nvidia.driver import CudaDriver

    drv = CudaDriver()
    drv.get_current_device = lambda: 0
    drv.get_current_stream = lambda idx=None: 0
    drv.get_device_capability = lambda device=None: (12, 0)
    triton.runtime.driver.set_active(drv)


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        sys.exit("run with CUDA_VISIBLE_DEVICES= (CPU only)")
    from dg1.grid import TRITON_CACHE, all_configs, gemv_args
    if os.path.realpath(os.environ.get("TRITON_CACHE_DIR", "")) != os.path.realpath(TRITON_CACHE):
        sys.exit(f"export TRITON_CACHE_DIR={TRITON_CACHE}")
    _fake_sm120_driver()
    import sglang.srt.layers.quantization  # noqa: F401  (import order: see moe_smallm.runners)
    import sglang.srt.layers.moe.moe_runner.flashinfer_trtllm  # noqa: F401
    from triton.runtime._async_compile import AsyncCompileMode

    from dg1.proxies import Warmup, patched
    from sglang.srt.layers.moe import draft_moe_gemv as dmg

    t0 = time.time()
    jobs = [(cfg, True) for cfg in all_configs()] + [(all_configs()[0], False)]
    k1, k2 = Warmup(dmg._draft_moe_up_gate_kernel), Warmup(dmg._draft_moe_down_kernel)
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        with AsyncCompileMode(pool):
            for cfg, pdl in jobs:
                with patched(dmg, k1=k1, k2=k2, pdl=pdl):
                    dmg.draft_moe_gemv(**gemv_args(device="cpu", num_experts=1, cfg=cfg))
    print(f"[dg1-precompile] {len(jobs)} configs (2 kernels each) in {time.time() - t0:.1f}s "
          f"-> {TRITON_CACHE}", flush=True)


if __name__ == "__main__":
    main()
