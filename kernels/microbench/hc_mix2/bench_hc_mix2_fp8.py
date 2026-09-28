#!/usr/bin/env python3
"""CUPTI per-kernel medians for the three-kernel HC norm+mix, bf16 vs fp8 weights.

Methodology (see bench/w8a16v2/harness.py): CUDA-graph replay under continuous
load with a rotating weight working set >= 4x the 128 MB L2, and the reported
number is the median CUPTI kernel duration -- the same metric the in-server
torch profile shows. CUDA-event timing is ~25% higher because of launch gaps and
is not comparable.

  python bench/hc_mix2/bench_hc_mix2_fp8.py                 # bf16 vs fp8, M=1,16
  python bench/hc_mix2/bench_hc_mix2_fp8.py --sweep down    # fp8 K1 geometry
  python bench/hc_mix2/bench_hc_mix2_fp8.py --sweep up      # fp8 K2 geometry
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

os.environ.setdefault("SGLANG_HC_FUSED", "0")
os.environ.setdefault("SGLANG_HC_MIX2", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "w8a16v2"))

import torch  # noqa: E402
import triton  # noqa: E402
from harness import cupti_kernel_us, n_copies  # noqa: E402

from sglang.srt.layers.hc_mix2_triton import (  # noqa: E402
    HCMix2Config,
    _DEFAULT_CONFIG,
    _DEFAULT_CONFIG_FP8,
    hc_norm_mix2,
    quantize_hc_mix2_weights_fp8,
)

HC = 4
HS = 2560
K = HC * HS
LR = 320
EPS = 1e-6

KERNELS = ("_hc_branch_stats_kernel", "_hc_down_kernel", "_hc_up_kernel")
LABEL = {"_hc_branch_stats_kernel": "K0", "_hc_down_kernel": "K1", "_hc_up_kernel": "K2"}


def make_weight_set(fp8: bool):
    """Rotating copies sized so the weights alone exceed 4x L2."""
    per_call = LR * K + K * LR  # elements
    bytes_per_call = per_call * (1 if fp8 else 2)
    copies = n_copies(bytes_per_call)
    sets = []
    for i in range(copies):
        torch.manual_seed(1000 + i)
        norm_w = torch.randn(K, device="cuda", dtype=torch.bfloat16) * 0.02
        w_down = torch.randn(LR, K, device="cuda", dtype=torch.bfloat16) * 0.02
        w_up = torch.randn(K, LR, device="cuda", dtype=torch.bfloat16) * 0.02
        if fp8:
            wd, sd, wu, su = quantize_hc_mix2_weights_fp8(w_down, w_up)
            sets.append((norm_w, wd, wu, sd, su))
            del w_down, w_up
        else:
            sets.append((norm_w, w_down, w_up, None, None))
    torch.cuda.empty_cache()
    mb = copies * bytes_per_call / (1 << 20)
    return sets, mb


def call_fn(x, config):
    def run(ws):
        norm_w, w_down, w_up, sd, su = ws
        return hc_norm_mix2(x, norm_w, EPS, w_down, w_up, HC, HS, config, sd, su)

    return run


def measure(x, sets, config, min_seconds=0.25, only=None):
    """CUPTI medians per kernel; `only` restricts to one label (K0/K1/K2)."""
    run = call_fn(x, config)
    out = {}
    for name in KERNELS:
        if only is not None and LABEL[name] != only:
            continue
        us, n = cupti_kernel_us(run, sets, min_seconds=min_seconds, name_sub=name)
        out[LABEL[name]] = us * 1e6
    if only is None:
        out["total"] = sum(out[LABEL[k]] for k in KERNELS)
    return out


def fmt(tag, r):
    return (
        f"{tag:<28} K0 {r['K0']:6.2f}  K1 {r['K1']:6.2f}  K2 {r['K2']:6.2f}"
        f"  total {r['total']:6.2f} us"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="*", default=[16, 1])
    ap.add_argument("--sweep", choices=["down", "up"], default=None)
    ap.add_argument("--min-seconds", type=float, default=0.25)
    args = ap.parse_args()

    torch.cuda.init()
    print(f"device: {torch.cuda.get_device_name(0)}")

    if args.sweep is None:
        for fp8 in (False, True):
            sets, mb = make_weight_set(fp8)
            cfg = _DEFAULT_CONFIG_FP8 if fp8 else _DEFAULT_CONFIG
            print(
                f"\n--- {'fp8' if fp8 else 'bf16'} weights, {len(sets)} copies, "
                f"{mb:.0f} MB working set ---"
            )
            for rows in args.rows:
                x = torch.randn(rows, K, device="cuda", dtype=torch.bfloat16)
                r = measure(x, sets, cfg, args.min_seconds)
                print(fmt(f"M={rows}", r))
            del sets
            torch.cuda.empty_cache()
        return

    sets, mb = make_weight_set(True)
    print(f"fp8 sweep, {len(sets)} copies, {mb:.0f} MB working set")
    base = _DEFAULT_CONFIG_FP8
    configs = []
    if args.sweep == "down":
        for bn, bk, bg, w, st in itertools.product(
            (16, 32, 64), (128, 256), (1, 2, 4), (4, 8), (2, 3, 4)
        ):
            if HS % (bk * bg) or LR % bn:
                continue
            configs.append(
                HCMix2Config(
                    block_n=bn,
                    block_k=bk,
                    block_g=bg,
                    down_warps=w,
                    down_stages=st,
                    block_j=base.block_j,
                    block_r=base.block_r,
                    up_warps=base.up_warps,
                    up_stages=base.up_stages,
                )
            )
        key = "K1"
    else:
        for bj, br, w, st in itertools.product(
            (8, 16, 32, 64), (64, 128, 256), (4, 8), (2, 4, 6)
        ):
            configs.append(
                HCMix2Config(
                    block_n=base.block_n,
                    block_k=base.block_k,
                    block_g=base.block_g,
                    down_warps=base.down_warps,
                    down_stages=base.down_stages,
                    block_j=bj,
                    block_r=br,
                    up_warps=w,
                    up_stages=st,
                )
            )
        key = "K2"

    for rows in args.rows:
        x = torch.randn(rows, K, device="cuda", dtype=torch.bfloat16)
        results = []
        for cfg in configs:
            try:
                r = measure(x, sets, cfg, args.min_seconds, only=key)
            except Exception as exc:  # OutOfResources / CompilationError / ...
                print(f"  skip: {type(exc).__name__}")
                continue
            results.append((r[key], cfg, r))
        results.sort(key=lambda t: t[0])
        print(f"\n== M={rows} best {key} ==")
        for us, cfg, r in results[:10]:
            if args.sweep == "down":
                tag = f"bn{cfg.block_n} bk{cfg.block_k} bg{cfg.block_g} w{cfg.down_warps} s{cfg.down_stages}"
            else:
                tag = f"bj{cfg.block_j} br{cfg.block_r} w{cfg.up_warps} s{cfg.up_stages}"
            print(f"  {us:6.2f} us  {tag}")


if __name__ == "__main__":
    main()
