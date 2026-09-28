#!/usr/bin/env python3
"""CUDA-graph replay benchmark for Glue-D router and MTP entry fusions."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sglang.srt.layers.moe.router_gemv import router_gemv
from sglang.srt.layers.mtp_entry import mtp_entry_fused

HIDDEN_SIZE = 2560
EXPERTS = 512
HC = 4
EPS = 1e-6
WEIGHT_COPIES = 24
GRAPH_REPLAYS = 50


def _time_us(fn) -> float:
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for index in range(WEIGHT_COPIES):
            fn(index)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for index in range(WEIGHT_COPIES):
            fn(index)
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


def _gemma_rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    x_fp32 = x.float()
    variance = x_fp32.square().mean(dim=-1, keepdim=True)
    return (
        x_fp32 * torch.rsqrt(variance + EPS) * (1.0 + weight.float())
    ).to(torch.bfloat16)


@torch.inference_mode()
def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(0)

    router_weights = [
        torch.randn(
            EXPERTS, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
        ).mul_(0.02)
        for _ in range(WEIGHT_COPIES)
    ]
    mtp_weights = [
        (
            torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16).mul_(
                0.02
            ),
            torch.randn(
                HC * HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
            ).mul_(0.02),
            torch.randn(
                HIDDEN_SIZE, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
            ).mul_(0.02),
            torch.randn(
                HIDDEN_SIZE, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
            ).mul_(0.02),
        )
        for _ in range(WEIGHT_COPIES)
    ]

    print(
        f"CUDA-graph timing: {WEIGHT_COPIES} rotating weight copies, "
        f"{GRAPH_REPLAYS} replays"
    )
    print("part  M   reference_us  fused_us  speedup")
    for rows in (1, 5, 16):
        x = torch.randn(rows, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)

        def router_reference(index: int):
            return F.linear(x, router_weights[index])

        def router_fused(index: int):
            return router_gemv(x, router_weights[index])

        reference_us = _time_us(router_reference)
        fused_us = _time_us(router_fused)
        print(
            f"A     {rows:<3d} {reference_us:12.3f}  {fused_us:8.3f}  "
            f"{reference_us / fused_us:6.2f}x"
        )

        input_embeds = torch.randn(
            rows, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
        )
        hidden_states = torch.randn(
            rows, HC * HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
        )

        def mtp_reference(index: int):
            emb_norm, hidden_norm, emb_proj, hidden_proj = mtp_weights[index]
            embedding = F.linear(
                _gemma_rms_norm(input_embeds, emb_norm), emb_proj
            )
            hidden = _gemma_rms_norm(hidden_states, hidden_norm).view(
                rows, HC, HIDDEN_SIZE
            )
            hidden = F.linear(hidden, hidden_proj)
            return (embedding.unsqueeze(1) + hidden).view_as(hidden_states)

        def mtp_fused(index: int):
            emb_norm, hidden_norm, emb_proj, hidden_proj = mtp_weights[index]
            return mtp_entry_fused(
                input_embeds,
                hidden_states,
                emb_norm,
                hidden_norm,
                emb_proj,
                hidden_proj,
                EPS,
            )

        reference_us = _time_us(mtp_reference)
        fused_us = _time_us(mtp_fused)
        print(
            f"C     {rows:<3d} {reference_us:12.3f}  {fused_us:8.3f}  "
            f"{reference_us / fused_us:6.2f}x"
        )


if __name__ == "__main__":
    main()
