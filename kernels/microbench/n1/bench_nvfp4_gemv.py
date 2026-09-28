"""NVFP4 vs FP8 skinny-GEMV head-to-head at the dense decode shapes.

Metric and working-set discipline are `bench/w8a16v2/harness.py`'s, unchanged: CUPTI
kernel-duration medians over a CUDA-graph replay that rotates over enough distinct
weight copies to make the set several times the 128 MB L2, under continuous load so the
SMs stay at boost. That is the number the in-server torch profile reports.

Why the head-to-head and not just GB/s: the two kernels move different byte counts for
the same math, so the only decision-relevant figure is *time at the same shape*. Bytes
per weight element are 1.0 (FP8 e4m3) against 0.5 + 1/16 = 0.5625 (FP4 codes + per-16
E4M3 block scales), i.e. a 1.778x byte advantage. EXCLUSIVE_TIME_MAP.md 5 measures the
FP8 kernel at 96-98 % of this card's 1615 GB/s read roof on both lm_head shapes, so
NVFP4 breaks even only at >= 0.5625 * 0.97 = ~55 % of roof and needs ~69 % for a 20 %
kernel win. `--sweep` tunes the tile before that verdict is taken.
"""

import argparse
import itertools
import sys

import torch
import triton

sys.path.insert(0, "/home/user/tools/sglang-n1/python")
sys.path.insert(0, "/home/user/tools/sglang-n1/bench/w8a16v2")

import harness  # noqa: E402

from sglang.srt.layers.quantization.w4a16_nvfp4_gemv import (  # noqa: E402
    quantize_nvfp4,
    w4a16_nvfp4_gemv,
)
from sglang.srt.layers.quantization.w8a16_gemv import w8a16_gemv  # noqa: E402

DEV = "cuda"
#: measured pure-read roof for this card (G1_LOG.md 3), used only to express a
#: measured time as a fraction of roof -- never to predict one.
READ_ROOF_GBS = 1615.0

SHAPES = [
    ("draft lm_head (hot2_49152)", 49152, 2560),
    ("target lm_head", 248320, 2560),
    ("GDN in_proj_qkvz", 16384, 2560),
    ("GDN out_proj", 2560, 6144),
    ("attn qkv", 13312, 2560),
]


def fp8_bytes(N, K):
    return N * K


def fp4_bytes(N, K):
    return N * K // 2 + N * K // 16


def build_fp8(N, K):
    w = (torch.randn(N, K, device=DEV, dtype=torch.float32) * 0.02).bfloat16()
    amax = w.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
    s = (amax / 448.0).reshape(-1).contiguous()
    q = (w.float() / (amax / 448.0)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return w, q, s


def measure_fp8(M, N, K, copies, cfg=None):
    _, q, s = build_fp8(N, K)
    ws = harness.make_copies(q, copies)
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    out = torch.empty(M, N, device=DEV, dtype=torch.bfloat16)
    sec, n = harness.cupti_kernel_us(
        lambda w: w8a16_gemv(x, w, s, cfg=cfg, out=out), ws, name_sub="w8a16"
    )
    del ws
    torch.cuda.empty_cache()
    return sec


def measure_fp4(M, N, K, copies, cfg=None):
    w = (torch.randn(N, K, device=DEV, dtype=torch.float32) * 0.02).bfloat16()
    wq, bs, gs = quantize_nvfp4(w)
    del w
    torch.cuda.empty_cache()
    qs = harness.make_copies(wq, copies)
    ss = harness.make_copies(bs, copies)
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    out = torch.empty(M, N, device=DEV, dtype=torch.bfloat16)
    pairs = list(zip(qs, ss))
    sec, n = harness.cupti_kernel_us(
        lambda p: w4a16_nvfp4_gemv(x, p[0], p[1], gs, cfg=cfg, out=out),
        pairs, name_sub="w4a16",
    )
    del qs, ss, pairs
    torch.cuda.empty_cache()
    return sec


def copies_for(nbytes, mem_gb=20):
    n = max(3, -(-(4 * harness.L2_BYTES) // nbytes))
    return int(min(n, 24, max(2, (mem_gb << 30) // nbytes)))


def run_head_to_head(ms, shapes):
    print(f"{'shape':32s} {'M':>3s} {'fp8 us':>8s} {'fp8 GB/s':>9s} {'%roof':>6s} "
          f"{'fp4 us':>8s} {'fp4 GB/s':>9s} {'%roof':>6s} {'speedup':>8s}")
    for name, N, K in shapes:
        b8, b4 = fp8_bytes(N, K), fp4_bytes(N, K)
        c8, c4 = copies_for(b8), copies_for(b4)
        for M in ms:
            t8 = measure_fp8(M, N, K, c8)
            t4 = measure_fp4(M, N, K, c4)
            g8, g4 = b8 / t8 / 1e9, b4 / t4 / 1e9
            print(f"{name:32s} {M:3d} {t8*1e6:8.2f} {g8:9.1f} {100*g8/READ_ROOF_GBS:5.1f}% "
                  f"{t4*1e6:8.2f} {g4:9.1f} {100*g4/READ_ROOF_GBS:5.1f}% {t8/t4:7.3f}x")


def run_sweep(M, N, K):
    """Tile sweep for the NVFP4 kernel. BLOCK_K must divide K and be a multiple of 16."""
    b4 = fp4_bytes(N, K)
    c4 = copies_for(b4)
    # Kept deliberately small: every distinct config is a fresh Triton compile
    # (~5-20 s), so the full cross product is hours of wall clock for a decision that a
    # few dozen points already make. Split-K is only offered where the n grid does NOT
    # already fill the machine -- at N=49152, BLOCK_N=32 is 1536 CTAs on 188 SMs, so
    # splitting can only add reduction traffic.
    n_full = triton.cdiv(N, 32) >= 2 * 188
    splits = (1,) if n_full else (1, 2, 5, 10)
    cands = []
    for bn, bk, sp, warps, stages in itertools.product(
        (32, 64, 128), (128, 256, 512), splits, (4, 8), (2, 3)
    ):
        if K % bk or (bk // 16) * bn > 8192:
            continue
        if sp > 1 and (K // bk) % sp:
            continue
        use_dot = M > 1
        m_pad = 16 if use_dot else 1
        if triton.cdiv(N, bn) * sp * m_pad * bn > (1 << 21):
            continue
        cands.append((bn, bk, sp, use_dot, warps, stages))
    print(f"sweep M={M} N={N} K={K}: {len(cands)} candidates, {c4} weight copies")
    rows = []
    for cfg in cands:
        try:
            t = measure_fp4(M, N, K, c4, cfg=cfg)
        except Exception as e:
            print(f"  {cfg} FAIL {type(e).__name__}: {str(e)[:90]}")
            continue
        g = b4 / t / 1e9
        rows.append((t, cfg, g))
        print(f"  {str(cfg):40s} {t*1e6:8.2f} us {g:8.1f} GB/s {100*g/READ_ROOF_GBS:5.1f}% roof")
    rows.sort()
    print("\nbest:")
    for t, cfg, g in rows[:8]:
        print(f"  {str(cfg):40s} {t*1e6:8.2f} us {g:8.1f} GB/s {100*g/READ_ROOF_GBS:5.1f}% roof")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--sweep", type=int, nargs=3, metavar=("M", "N", "K"), default=None)
    ap.add_argument("--shapes", type=int, default=0, help="how many SHAPES rows (0 = all)")
    ap.add_argument("--idle-check", action="store_true")
    a = ap.parse_args()
    if a.idle_check:
        harness.require_idle_gpu()
    print(f"{torch.cuda.get_device_name(0)}, roof {READ_ROOF_GBS} GB/s")
    if a.sweep:
        run_sweep(*a.sweep)
    else:
        run_head_to_head(a.ms, SHAPES[: a.shapes] if a.shapes else SHAPES)
