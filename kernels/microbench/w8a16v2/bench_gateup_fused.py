#!/usr/bin/env python3
"""Shared-expert gate_up: two-kernel (GEMV + act_and_mul) vs the fused GEMV.

Same methodology as bench_w8a16v2.py: CUPTI kernel-duration medians over
CUDA-graph replay with a rotating weight working set >= 4x the 128 MB L2.

  python bench/w8a16v2/bench_gateup_fused.py            # head to head, M=1,4,16
  python bench/w8a16v2/bench_gateup_fused.py --sweep    # fused-tile sweep
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import triton  # noqa: E402
from harness import cupti_kernel_us, make_copies, n_copies  # noqa: E402

from sglang.srt.layers.quantization.w8a16_gemv import (  # noqa: E402
    _fused_plan,
    _num_sms,
    _plan,
    w8a16_gemv,
    w8a16_gemv_silu_mul,
)

# (N, K) of the shared-expert gate_up in flash-next: [gate 640 | up 640] x 2560.
SHAPES = [(1280, 2560)]


def _act():
    """The activation the unfused path actually launches."""
    try:
        from sglang.srt.layers.activation import silu_and_mul

        return silu_and_mul
    except Exception:  # pragma: no cover - CPU-only import failure
        def _torch_silu_and_mul(inp, out=None):
            h = inp.shape[-1] // 2
            y = (
                torch.nn.functional.silu(inp[..., :h].float()) * inp[..., h:].float()
            ).to(inp.dtype)
            return y if out is None else out.copy_(y)

        return _torch_silu_and_mul


def _weights(N, K, copies):
    torch.manual_seed(7)
    w = torch.randn(N, K, device="cuda", dtype=torch.float32) * 0.02
    scale = (w.abs().amax(dim=1) / 448.0).clamp(min=1e-12).float().contiguous()
    q = (w / scale[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
    del w
    return make_copies(q, copies), scale


def bench(M, N, K, min_seconds=0.25, fused_cfg=None):
    copies = n_copies(N * K)
    ws, scale = _weights(N, K, copies)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    silu_and_mul = _act()

    def two_kernel(w):
        return silu_and_mul(w8a16_gemv(x, w, scale))

    def fused(w):
        return w8a16_gemv_silu_mul(x, w, scale, cfg=fused_cfg)

    def _m(call, name_sub):
        try:
            us, n = cupti_kernel_us(call, ws, min_seconds=min_seconds, name_sub=name_sub)
            return us * 1e6, n
        except RuntimeError as exc:
            print(f"  !! no events for {name_sub!r}: {exc}")
            return float("nan"), 0

    gemv_us, _ = _m(two_kernel, "_w8a16_gemv_kernel")
    all_us, n_all = _m(two_kernel, None)
    act_us, n_act = _m(two_kernel, "act_and_mul")
    if n_act == 0:  # kernel name differs; take the non-gemv events
        act_us, n_act = _m(two_kernel, "and_mul")
    fused_us, _ = _m(fused, "_w8a16_gemv_silu_kernel")
    del ws
    torch.cuda.empty_cache()
    return {
        "gemv": gemv_us,
        "act": act_us,
        "act_events": n_act,
        "any": all_us,
        "fused": fused_us,
        "copies": copies,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="*", default=[1, 4, 16])
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--min-seconds", type=float, default=0.25)
    args = ap.parse_args()

    torch.cuda.init()
    print(f"device: {torch.cuda.get_device_name(0)}")

    for N, K in SHAPES:
        if not args.sweep:
            for M in args.rows:
                r = bench(M, N, K, args.min_seconds)
                two = r["gemv"] + r["act"]
                print(
                    f"N={N} K={K} M={M:>2}  ({r['copies']} copies)  "
                    f"gemv {r['gemv']:6.2f} + act {r['act']:6.2f} = {two:6.2f}  "
                    f"->  fused {r['fused']:6.2f} us   "
                    f"(saving {two - r['fused']:+6.2f} us/call)"
                )
                print(
                    f"    plan unfused {_plan(M, N, K, False, 1, _num_sms('cuda'))}"
                    f"  fused {_fused_plan(M, N, K, _num_sms('cuda'))}"
                )
            continue

        copies = n_copies(N * K)
        ws, scale = _weights(N, K, copies)
        for M in args.rows:
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            use_dot = M > 1
            m_pad = 16 if use_dot else 1

            def try_cfg(cfg):
                if 2 * triton.cdiv(N // 2, cfg[0]) * cfg[2] * m_pad * cfg[0] > (1 << 21):
                    return None

                def call(w, cfg=cfg):
                    return w8a16_gemv_silu_mul(x, w, scale, cfg=cfg)

                try:
                    us, _ = cupti_kernel_us(
                        call,
                        ws,
                        min_seconds=args.min_seconds,
                        name_sub="_w8a16_gemv_silu_kernel",
                    )
                except (triton.runtime.errors.OutOfResources, RuntimeError):
                    return None
                return us * 1e6

            # Stage 1: geometry at a fixed (warps, stages).
            stage1 = []
            for bn, bk, splits in itertools.product(
                (16, 32, 64), (64, 128, 256), (1, 2, 5, 10)
            ):
                cfg = (bn, bk, splits, use_dot, None, 4, 3)
                us = try_cfg(cfg)
                if us is not None:
                    stage1.append((us, cfg))
            stage1.sort(key=lambda t: t[0])
            print(f"\n== N={N} K={K} M={M} stage 1 (warps=4, stages=3) ==")
            for us, cfg in stage1[:6]:
                print(f"  {us:6.2f} us  {cfg}")

            # Stage 2: warps/stages around the top 3 geometries.
            stage2 = list(stage1[:3])
            for _, cfg in stage1[:3]:
                for warps, stages in itertools.product((2, 4, 8), (2, 3, 4)):
                    if (warps, stages) == (4, 3):
                        continue
                    c = (cfg[0], cfg[1], cfg[2], use_dot, None, warps, stages)
                    us = try_cfg(c)
                    if us is not None:
                        stage2.append((us, c))
            stage2.sort(key=lambda t: t[0])
            print(f"== N={N} K={K} M={M} best fused tiles ==")
            for us, cfg in stage2[:8]:
                print(f"  {us:6.2f} us  {cfg}")
        del ws


if __name__ == "__main__":
    main()
