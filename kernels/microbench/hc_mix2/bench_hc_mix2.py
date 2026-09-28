#!/usr/bin/env python3
"""Cold-weight timing and accuracy for the two-kernel HC norm+mix path.

Baseline is the production chain: the JIT ``grouped_gemma_rmsnorm`` followed by
the persistent ``fused_hc_mix``. Timing replays a CUDA graph that holds one call
per weight copy, so the weights come from HBM rather than L2.
"""

from __future__ import annotations

import argparse
import itertools
import os

os.environ.setdefault("SGLANG_HC_FUSED", "0")
os.environ.setdefault("SGLANG_HC_MIX_FP8", "0")

import torch
import triton.runtime.errors

from sglang.kernels.ops.layernorm.grouped_gemma_rmsnorm import grouped_gemma_rmsnorm
from sglang.srt.layers.hc_mix2_triton import HCMix2Config, hc_norm_mix2
from sglang.srt.layers.hc_mix_triton import fused_hc_mix

HC = 4
HS = 2560
K = HC * HS
LR = 320
EPS = 1e-6
WEIGHT_COPIES = 24
GRAPH_REPLAYS = 50


def _make_weights() -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    copies = []
    for _ in range(WEIGHT_COPIES):
        norm_w = torch.randn(K, device="cuda", dtype=torch.bfloat16) * 0.02
        w_down = torch.randn(LR, K, device="cuda", dtype=torch.bfloat16) * 0.02
        w_up = torch.randn(K, LR, device="cuda", dtype=torch.bfloat16) * 0.02
        copies.append((norm_w, w_down, w_up))
    return copies


def _time_us(fn) -> float:
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for i in range(WEIGHT_COPIES):
            fn(i)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(WEIGHT_COPIES):
            fn(i)
    graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(GRAPH_REPLAYS):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / (GRAPH_REPLAYS * WEIGHT_COPIES)


def _errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    error = (actual.float() - expected.float()).abs()
    return error.max().item(), error.mean().item()


def _baseline_fn(x: torch.Tensor, weights):
    def run(index: int):
        norm_w, w_down, w_up = weights[index]
        normed = grouped_gemma_rmsnorm(x, norm_w, HS, EPS)
        return fused_hc_mix(normed, w_down, w_up, HC, HS), normed

    return run


def _mix2_fn(x: torch.Tensor, weights, config: HCMix2Config):
    def run(index: int):
        norm_w, w_down, w_up = weights[index]
        return hc_norm_mix2(x, norm_w, EPS, w_down, w_up, HC, HS, config)

    return run


def _sweep_configs(stage: str) -> list[HCMix2Config]:
    configs: list[HCMix2Config] = []
    if stage in ("down", "all"):
        for block_n, block_k, block_g, warps in itertools.product(
            (16, 32, 64), (128, 256), (1, 2, 4), (4, 8)
        ):
            if HS % (block_k * block_g):
                continue
            configs.append(
                HCMix2Config(
                    block_n=block_n,
                    block_k=block_k,
                    block_g=block_g,
                    down_warps=warps,
                )
            )
    if stage in ("stats", "all"):
        for mode in ("norm", "stats", "redundant"):
            configs.append(HCMix2Config(stats_mode=mode))
    if stage in ("up", "all"):
        for block_j, block_r, warps, stages in itertools.product(
            (8, 16, 32), (64, 128), (4, 8), (2, 4, 6)
        ):
            configs.append(
                HCMix2Config(
                    block_j=block_j, block_r=block_r, up_warps=warps, up_stages=stages
                )
            )
    return configs


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep", choices=("none", "down", "up", "stats", "all"), default="none"
    )
    parser.add_argument("--rows", type=int, nargs="*", default=[1, 4, 16])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(0)
    weights = _make_weights()
    inputs = {rows: torch.randn(rows, K, device="cuda", dtype=torch.bfloat16) for rows in args.rows}

    if args.sweep != "none":
        configs = _sweep_configs(args.sweep)
        print(f"sweeping {len(configs)} configs over rows={args.rows}")
        print(
            "bn  bk  bg dw mode      bj  br  uw us "
            + " ".join(f"M={r:<5d}" for r in args.rows)
        )
        for config in configs:
            times = []
            for rows in args.rows:
                try:
                    times.append(_time_us(_mix2_fn(inputs[rows], weights, config)))
                except triton.runtime.errors.OutOfResources:
                    times.append(float("nan"))
            print(
                f"{config.block_n:<3d} {config.block_k:<3d} {config.block_g:<2d} "
                f"{config.down_warps:<2d} {config.stats_mode:<9s} "
                f"{config.block_j:<3d} {config.block_r:<3d} "
                f"{config.up_warps:<2d} {config.up_stages:<2d} "
                + " ".join(f"{t:7.3f}" for t in times)
            )
        return

    config = HCMix2Config()
    timings = []
    for rows in args.rows:
        x = inputs[rows]
        norm_w, w_down, w_up = weights[0]
        ref_normed = grouped_gemma_rmsnorm(x, norm_w, HS, EPS)
        ref_mixed = fused_hc_mix(ref_normed, w_down, w_up, HC, HS)
        mixed, normed = hc_norm_mix2(x, norm_w, EPS, w_down, w_up, HC, HS, config)
        torch.cuda.synchronize()
        mixed_max, mixed_mean = _errors(mixed, ref_mixed)
        normed_max, normed_mean = _errors(normed, ref_normed)
        print(
            f"M={rows:<3d} mixed  max_abs={mixed_max:.6g} mean_abs={mixed_mean:.6g}\n"
            f"      normed max_abs={normed_max:.6g} mean_abs={normed_mean:.6g}"
        )
        baseline_us = _time_us(_baseline_fn(x, weights))
        mix2_us = _time_us(_mix2_fn(x, weights, config))
        timings.append((rows, baseline_us, mix2_us))

    print(
        f"\nCUDA-graph timing, {WEIGHT_COPIES} rotating weight copies per graph, "
        f"{GRAPH_REPLAYS} replays"
    )
    print(f"config: {config}")
    print("M   baseline_us  mix2_us  speedup")
    for rows, baseline_us, mix2_us in timings:
        print(
            f"{rows:<3d} {baseline_us:11.3f}  {mix2_us:7.3f}  "
            f"{baseline_us / mix2_us:6.2f}x"
        )


if __name__ == "__main__":
    main()
