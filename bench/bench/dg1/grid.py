"""dg1: shapes, tile grids and the CPU-side argument set shared by precompile.py and dg1_bench.py."""
from __future__ import annotations

import os

import torch

from sglang.srt.layers.moe import draft_moe_gemv as dmg
from sglang.srt.layers.moe.draft_moe_gemv import DEFAULT_CONFIG, DraftMoeGemvConfig

# Qwen3.8-Flash-Next MTP MoE layer (config.json of the mtpft5 checkpoint).
E, H, I, TOPK = 512, 2560, 640, 10

# Triton cache shared by the CPU precompile and the GPU job (keys cover source, target, options, env).
TRITON_CACHE = os.path.expanduser("~/.cache/dg1-triton")

# Around the prototype optimum (fc_moe/draft_gemv_proto.py: k1 8/512/4w/2s, k2 16/128/S10/2w/3s).
K1_GRID = [(bn, bk, w, s) for bn in (4, 8, 16) for bk in (256, 512) for w in (2, 4, 8) for s in (2, 3)]
K2_GRID = [(bn, bk, sp, w, 3) for bn in (8, 16, 32) for bk in (64, 128) for sp in (5, 10) for w in (2, 4)]


def with_k1(base: DraftMoeGemvConfig, k1) -> DraftMoeGemvConfig:
    bn, bk, w, s = k1
    return DraftMoeGemvConfig(block_n1=bn, block_k1=bk, warps1=w, stages1=s, block_n2=base.block_n2,
                              block_k2=base.block_k2, splits2=base.splits2, warps2=base.warps2,
                              stages2=base.stages2)


def with_k2(base: DraftMoeGemvConfig, k2) -> DraftMoeGemvConfig:
    bn, bk, sp, w, s = k2
    return DraftMoeGemvConfig(block_n1=base.block_n1, block_k1=base.block_k1, warps1=base.warps1,
                              stages1=base.stages1, block_n2=bn, block_k2=bk, splits2=sp, warps2=w,
                              stages2=s)


def k1_of(cfg: DraftMoeGemvConfig):
    return (cfg.block_n1, cfg.block_k1, cfg.warps1, cfg.stages1)


def k2_of(cfg: DraftMoeGemvConfig):
    return (cfg.block_n2, cfg.block_k2, cfg.splits2, cfg.warps2, cfg.stages2)


def all_configs() -> list[DraftMoeGemvConfig]:
    """Every config the GPU job may launch: the k1 sweep, the k2 sweep, and DEFAULT."""
    cfgs = [DEFAULT_CONFIG]
    cfgs += [with_k1(DEFAULT_CONFIG, k1) for k1 in K1_GRID]
    cfgs += [with_k2(DEFAULT_CONFIG, k2) for k2 in K2_GRID]
    return cfgs


def gemv_args(*, device, num_experts: int, cfg: DraftMoeGemvConfig) -> dict:
    """draft_moe_gemv kwargs with the server's dtypes, strides and 16 B alignment (values unused)."""
    u8 = dict(dtype=torch.uint8, device=device)
    return dict(
        hidden_states=torch.zeros((1, H), dtype=torch.bfloat16, device=device),
        topk_ids=torch.zeros((1, TOPK), dtype=torch.int32, device=device),
        topk_weights=torch.zeros((1, TOPK), dtype=torch.float32, device=device),
        w13_weight=torch.zeros((num_experts, 2 * I, H // 2), **u8),
        w13_blockscale_swizzled=torch.zeros((num_experts, 2 * I, H // 16), **u8).view(torch.float8_e4m3fn),
        # [E, 2] as on the CUTLASS path (stride 2); w2's is [E] (stride 1, Triton specializes it).
        w13_weight_scale_2=torch.ones((num_experts, 2), dtype=torch.float32, device=device),
        w2_weight=torch.zeros((num_experts, H, I // 2), **u8),
        w2_blockscale_swizzled=torch.zeros((num_experts, H, I // 16), **u8).view(torch.float8_e4m3fn),
        w2_weight_scale_2=torch.ones((num_experts,), dtype=torch.float32, device=device),
        workspace=dmg.make_workspace(num_routes=TOPK, hidden_size=H, intermediate_size=I,
                                     device=device, cfg=cfg),
        cfg=cfg,
    )
