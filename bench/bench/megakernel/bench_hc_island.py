"""K2 experiment 3: the real HC-boundary island, separate kernels vs one
persistent kernel.

The chain measured here is the one that runs 96 times per verify step
(hyperconnection.py: `combine()` at the end of a block, then `mix()` at the
start of the next one), at the production shapes: hidden 2560, hc_count 4,
K = hc*hs = 10240, low rank 320, M = 4 and 16 rows.

    combine gate -> combine apply -> hc branch stats/norm (K0) -> down (K1)
                 -> up + gate + mean (K2)

Variants:

  prod        production today: `hc_combine_split` (2 CUDA kernels, the apply is
              launched with PDL) + `hc_norm_mix2` with FP8 mix weights (3 Triton
              kernels). 5 launches.
  bf16_split  the same with BF16 mix weights -- the apples-to-apples baseline for
              the persistent kernel, which is BF16-only.
  bf16_fused  `hc_fused_combine_norm_mix`: the fork's existing megakernel for
              exactly this island (combine + norm + down + grid barrier + up),
              one launch. This is the prior art the K2 question is about.
  mix_split / mix_fused
              the same pair without the combine stage (`hc_norm_mix2` vs
              `hc_fused_norm_mix`), isolating the 3-stage sub-island.

Reported per call: CUDA-graph wall (`time_graph`), the CUPTI per-kernel medians,
and their sum. The wall is the number that matters -- the sum of CUPTI durations
double-counts nothing here (the chain is serial) but omits the inter-kernel gap.

Run:  flock -w 14400 ~/.gpu.lock python bench/megakernel/bench_hc_island.py
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import ab_rounds, cupti_by_kernel, emit, gpu_note, make_copies  # noqa: E402

HS = 2560
HC = 4
K = HC * HS
LOWRANK = 320
EPS = 1e-6


def build(dev, rows, copies):
    from sglang.srt.layers.hc_mix2_triton import quantize_hc_mix2_weights_fp8

    g = torch.Generator(device=dev).manual_seed(0)
    hyper = (torch.randn(rows, K, generator=g, device=dev, dtype=torch.bfloat16) / 8)
    normed_prev = (torch.randn(rows, K, generator=g, device=dev, dtype=torch.bfloat16) / 8)
    block_out = (torch.randn(rows, HS, generator=g, device=dev, dtype=torch.bfloat16) / 8)
    inject = (torch.randn(HC, K, generator=g, device=dev, dtype=torch.bfloat16) / 64)
    norm_w = (torch.randn(K, generator=g, device=dev, dtype=torch.bfloat16) / 16)
    w_down = (torch.randn(LOWRANK, K, generator=g, device=dev, dtype=torch.bfloat16) / 32)
    w_up = (torch.randn(K, LOWRANK, generator=g, device=dev, dtype=torch.bfloat16) / 32)
    wd8, sd, wu8, su = quantize_hc_mix2_weights_fp8(w_down, w_up)

    # Rotate BOTH the mix weights (6.5 MB each bf16, 3.3 MB fp8 -- the traffic of
    # K1/K2) and the activations (the traffic of the two combine kernels), so no
    # stage of the island is measured out of a warm L2.
    downs = make_copies(w_down, copies)
    ups = make_copies(w_up, copies)
    downs8 = make_copies(wd8, copies)
    ups8 = make_copies(wu8, copies)
    hypers = make_copies(hyper, copies)
    normeds = make_copies(normed_prev, copies)
    blocks = make_copies(block_out, copies)
    return dict(
        hypers=hypers, normeds=normeds, blocks=blocks, inject=inject,
        norm_w=norm_w, downs=downs, ups=ups, downs8=downs8, ups8=ups8, sd=sd, su=su,
    )


def make_variants(t, rows):
    from sglang.kernels.ops.elementwise.hc_combine import hc_combine_split
    from sglang.srt.layers.hc_fused_triton import (
        hc_fused_combine_norm_mix,
        hc_fused_norm_mix,
    )
    from sglang.srt.layers.hc_mix2_triton import hc_norm_mix2

    idx = list(range(len(t["downs"])))

    def prod(i):
        res = hc_combine_split(t["blocks"][i], t["hypers"][i], t["normeds"][i], t["inject"], HC, HS)
        hc_norm_mix2(res, t["norm_w"], EPS, t["downs8"][i], t["ups8"][i], HC, HS,
                     s_down=t["sd"], s_up=t["su"])

    def bf16_split(i):
        res = hc_combine_split(t["blocks"][i], t["hypers"][i], t["normeds"][i], t["inject"], HC, HS)
        hc_norm_mix2(res, t["norm_w"], EPS, t["downs"][i], t["ups"][i], HC, HS)

    def bf16_fused(i):
        hc_fused_combine_norm_mix(
            t["blocks"][i], t["hypers"][i], t["normeds"][i], t["inject"], t["norm_w"],
            EPS, t["downs"][i], t["ups"][i], HC, HS,
        )

    def mix_split_fp8(i):
        hc_norm_mix2(t["hypers"][i], t["norm_w"], EPS, t["downs8"][i], t["ups8"][i], HC, HS,
                     s_down=t["sd"], s_up=t["su"])

    def mix_split_bf16(i):
        hc_norm_mix2(t["hypers"][i], t["norm_w"], EPS, t["downs"][i], t["ups"][i], HC, HS)

    def mix_fused_bf16(i):
        hc_fused_norm_mix(t["hypers"][i], t["norm_w"], EPS, t["downs"][i], t["ups"][i], HC, HS)

    return {
        "prod(combine+mix fp8, 5 launches)": (prod, idx),
        "bf16_split(combine+mix, 5 launches)": (bf16_split, idx),
        "bf16_fused(combine+mix, 1 launch)": (bf16_fused, idx),
        "mix_split_fp8(3 launches)": (mix_split_fp8, idx),
        "mix_split_bf16(3 launches)": (mix_split_bf16, idx),
        "mix_fused_bf16(1 launch)": (mix_fused_bf16, idx),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--min-seconds", type=float, default=0.3)
    ap.add_argument("--rows", default="4,16")
    ap.add_argument("--copies", type=int, default=40)  # 40 x 13 MB = 520 MB > 4x L2
    ap.add_argument("--out", default="bench/megakernel/results/hc_island.json")
    args = ap.parse_args()

    dev = torch.device("cuda")
    note = gpu_note()
    print(f"[gpu] {note}")
    payload = {"note": note, "results": {}}

    for rows in [int(v) for v in args.rows.split(",")]:
        t = build(dev, rows, args.copies)
        variants = make_variants(t, rows)

        wall = ab_rounds(variants, rounds=args.rounds, min_seconds=args.min_seconds)
        print(f"\n=== M={rows} (graph wall per island call, {args.rounds} interleaved rounds) ===")
        for tag, r in wall.items():
            print(f"  {tag:<38s} {r['median']:7.2f} us   rounds={r['rounds']}")

        detail = {}
        for tag, (call, arglist) in variants.items():
            res, total = cupti_by_kernel(call, arglist, min_seconds=0.25)
            detail[tag] = {
                "device_us": round(total, 3),
                "kernels": {k: [round(v[0], 3), round(v[1], 2)] for k, v in res.items()},
            }
            print(f"\n  [{tag}] device time {total:.2f} us")
            for name, (med, n) in sorted(res.items(), key=lambda kv: -kv[1][0] * kv[1][1]):
                print(f"      {name[:58]:<58s} n={n:4.1f} med={med:7.2f} us")

        payload["results"][f"M{rows}"] = {"wall_us": wall, "cupti": detail}
        del t
        torch.cuda.empty_cache()

    emit(args.out, payload)


if __name__ == "__main__":
    main()
