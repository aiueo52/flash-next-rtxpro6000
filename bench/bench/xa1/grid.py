"""xa1: shapes, sweep grid and the argument sets shared by precompile.py and bench.py."""
from __future__ import annotations

import os

from xa1.kernels import XA1Config

TRITON_CACHE = os.path.expanduser("~/.cache/xa1-triton")
# Fixed req_to_token width so every case shares one Triton specialization (16448 % 16 == 0).
R2T_WIDTH = 16448
ROWS = (1, 4, 8, 16)
CTXS = (512, 1024, 2048, 4096, 8192)
TOPK_FRESH = 2051
TOPK_SHARED_W16 = 2067  # 2051 frozen + speculative_num_steps + 1 (wa: 15 steps)
TOPK_SHARED_W4 = 2055  # W4: 3 steps
PACKED_STRIDE = 2112  # cdiv(topk, 64) * 64 for every width above
SMEM_LIMIT = 101376  # sm120 opt-in shared memory per block (precompile checks every variant)
MAX_GROUPS, MAX_SPLITS = 64, 40
# job3 (v3) grid: split counts that leave no empty split at 2051 columns (33 tiles of 64, 65 of
# 32); B pipelines K/V (num_stages = K/V depth, see kernels.py). vc tiles stay under sm120's 99 KB
# of smem per block (BN64 with two K/V buffers and the f16 V operand needs 110 KB).
SPLITS_V3 = {1: (7, 9, 11, 13, 17, 22), 4: (6, 7, 9, 11, 13, 17), 8: (4, 5, 6, 7, 9, 11),
             16: (3, 4, 5, 6, 7, 9)}
VC_TILES = [(32, w, st) for w in (4, 8) for st in (2, 3, 4)] + [(64, w, 2) for w in (4, 8)]
PLAIN_TILES = [(32, 4, 3), (64, 4, 2), (64, 4, 3)]  # V as fp8 operand (v2 style), for comparison
# job4 (v4, lse combine): cheaper splits make more splits (fewer tiles per CTA) worth testing:
# 17 splits = 2 tiles of 64, 33 = 1. R16 9 / 11 with 32-wide tiles: two CTAs per SM fit in smem.
SPLITS_V4 = {1: (9, 11, 13, 17, 22, 33), 4: (9, 11, 13, 17, 22), 8: (7, 9, 11),
             16: (3, 4, 5, 9, 11)}
LSE_TILES = [(64, 4, 2, True), (64, 8, 2, True), (32, 4, 2, True), (32, 4, 3, True),
             (32, 4, 4, True), (64, 4, 3, False), (32, 4, 3, False)]
# job3 winners per row count, re-timed next to v4 in the same run.
V3_REF = {1: ("11/64/4/2/h/vc", "11/64/4/2/r3/vc"), 4: ("11/64/4/2/h/vc", "11/64/4/2/r3/vc"),
          8: ("11/64/4/2/h/vc",), 16: ("5/64/4/3", "5/32/4/4/h/vc")}


def v3_configs(rows: int):
    out = [XA1Config(s, bn, w, st, part16=h, vcvt=True) for s in SPLITS_V3[rows]
           for bn, w, st in VC_TILES for h in (False, True)]
    out += [XA1Config(s, bn, w, st) for s in SPLITS_V3[rows] for bn, w, st in PLAIN_TILES]
    out += [XA1Config(s, 64, 4, 2, red3d=True, vcvt=True) for s in SPLITS_V3[rows]]
    return out


def sweep_configs(rows: int):
    """job4: the v4 grid plus job3's best v3 configs."""
    out = [parse_cfg(k) for k in V3_REF[rows]]
    out += [XA1Config(s, bn, w, st, vcvt=vc, lse=True) for s in SPLITS_V4[rows]
            for bn, w, st, vc in LSE_TILES]
    return out


def a_configs(rows: int):
    """Candidate A is timed on the vc tiles only (it replaces XQA alone)."""
    return [c for c in sweep_configs(rows) if c.vcvt and not c.red3d]


def parse_cfg(text: str) -> XA1Config:
    """'s/bn/w/st' + optional '/bf' (bf16 MMA), '/h' (f16 partials), '/r3' (v2 combine),
    '/vc' (V converted before smem), '/l' (v4 lse partials and combine)."""
    parts = text.split("/")
    s, bn, w, st = (int(x) for x in parts[:4])
    flags = parts[4:]
    return XA1Config(s, bn, w, st, f16="bf" not in flags, part16="h" in flags,
                     red3d="r3" in flags, vcvt="vc" in flags, lse="l" in flags)


def variants():
    """(kind, ncols, cfg) for every Triton variant the GPU job may launch (PDL on).

    B with ncols TOPK_FRESH runs with prefix_valid (fresh rows), the shared widths without."""
    out = []
    every_b = {c for r in ROWS for c in sweep_configs(r)}
    every_a = {c for r in ROWS for c in a_configs(r)}
    out += [("B", TOPK_FRESH, c) for c in sorted(every_b)]
    out += [("A", 0, c) for c in sorted(every_a)]
    for ncols in (TOPK_SHARED_W16, TOPK_SHARED_W4):
        out += [("B", ncols, c) for c in sweep_configs(1)]
    return out
