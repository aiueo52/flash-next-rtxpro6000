#!/usr/bin/env python3
"""Correctness and cold-weight timing for the fused HC boundary kernels."""

from __future__ import annotations

import os

# This script uses GatedResidual itself as the baseline/reference chain.
os.environ["SGLANG_HC_FUSED"] = "0"
os.environ["SGLANG_HC_MIX_FP8"] = "0"

import torch

from sglang.srt.layers.hc_fused_triton import (
    hc_fused_combine_norm_mix,
    hc_fused_norm_mix,
)
from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig

HC = 4
HS = 2560
LR = 320
EPS = 1e-6
WEIGHT_COPIES = 24


def _make_modules() -> tuple[GatedResidual, GatedResidual]:
    config = HyperConnectionConfig(
        hc_count=HC,
        hidden_size=HS,
        params_dtype=torch.bfloat16,
        hc_lowrank=LR,
        rms_norm_eps=EPS,
        hc_per_branch_norm=True,
    )
    previous = (
        GatedResidual(config, use_mix=False, use_combine=True).cuda().to(torch.bfloat16)
    )
    following = (
        GatedResidual(config, use_mix=True, use_combine=False).cuda().to(torch.bfloat16)
    )
    with torch.no_grad():
        previous.block_inject_weight.weight.normal_(std=0.02)
        following.hc_norm.weight.normal_(std=0.02)
        following.input_mix_weight_down.weight.normal_(std=0.02)
        following.input_mix_weight_up.weight.normal_(std=0.02)
    return previous, following


def _errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    error = (actual.float() - expected.float()).abs()
    return error.max().item(), error.mean().item()


def _print_error(label: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    maximum, mean = _errors(actual, expected)
    print(f"  {label:16s} max_abs={maximum:.7g}  mean_abs={mean:.7g}")


GRAPH_REPLAYS = 50


def _time_us(fn) -> float:
    """GPU time per call, measured by replaying a CUDA graph that holds one call
    per weight copy (removes Python/launch overhead, like the serving graphs)."""
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


@torch.inference_mode()
def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(0)
    modules = [_make_modules() for _ in range(WEIGHT_COPIES)]
    timings: list[tuple[int, str, float, float]] = []

    for rows in (1, 16):
        block_output = torch.randn(rows, HS, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn(rows, HC * HS, device="cuda", dtype=torch.bfloat16)
        normed_prev = torch.randn_like(residual)
        previous, following = modules[0]

        ref_norm_mixed, (_, ref_normed) = following.mix(residual)
        fused_norm_mixed, fused_normed = hc_fused_norm_mix(
            residual,
            following.hc_norm.weight,
            EPS,
            following.input_mix_weight_down.weight,
            following.input_mix_weight_up.weight,
            HC,
            HS,
        )

        ref_new_residual = previous.combine(block_output, (residual, normed_prev))
        ref_combined_mixed, (_, ref_combined_normed) = following.mix(ref_new_residual)
        (
            fused_combined_mixed,
            fused_new_residual,
            fused_combined_normed,
        ) = hc_fused_combine_norm_mix(
            block_output,
            residual,
            normed_prev,
            previous.block_inject_weight.weight,
            following.hc_norm.weight,
            EPS,
            following.input_mix_weight_down.weight,
            following.input_mix_weight_up.weight,
            HC,
            HS,
        )
        torch.cuda.synchronize()

        print(f"M={rows} numerical errors versus GatedResidual baseline")
        _print_error("norm/mixed", fused_norm_mixed, ref_norm_mixed)
        _print_error("norm/normed", fused_normed, ref_normed)
        _print_error("combine/mixed", fused_combined_mixed, ref_combined_mixed)
        _print_error("combine/normed", fused_combined_normed, ref_combined_normed)
        _print_error("combine/residual", fused_new_residual, ref_new_residual)

        def baseline_norm(index: int):
            return modules[index][1].mix(residual)

        def fused_norm(index: int):
            nxt = modules[index][1]
            return hc_fused_norm_mix(
                residual,
                nxt.hc_norm.weight,
                EPS,
                nxt.input_mix_weight_down.weight,
                nxt.input_mix_weight_up.weight,
                HC,
                HS,
            )

        def baseline_combine(index: int):
            prev, nxt = modules[index]
            combined = prev.combine(block_output, (residual, normed_prev))
            return nxt.mix(combined)

        def fused_combine(index: int):
            prev, nxt = modules[index]
            return hc_fused_combine_norm_mix(
                block_output,
                residual,
                normed_prev,
                prev.block_inject_weight.weight,
                nxt.hc_norm.weight,
                EPS,
                nxt.input_mix_weight_down.weight,
                nxt.input_mix_weight_up.weight,
                HC,
                HS,
            )

        baseline_us = _time_us(baseline_norm)
        fused_us = _time_us(fused_norm)
        timings.append((rows, "norm+mix", baseline_us, fused_us))
        baseline_us = _time_us(baseline_combine)
        fused_us = _time_us(fused_combine)
        timings.append((rows, "combine+norm+mix", baseline_us, fused_us))

    print(
        f"\nCUDA-graph timing, {WEIGHT_COPIES} rotating weight copies per graph, "
        f"{GRAPH_REPLAYS} replays"
    )
    print("M   path                 baseline_us  fused_us  speedup")
    for rows, name, baseline_us, fused_us in timings:
        print(
            f"{rows:<3d} {name:<20s} {baseline_us:11.3f}  "
            f"{fused_us:8.3f}  {baseline_us / fused_us:6.2f}x"
        )


if __name__ == "__main__":
    main()
