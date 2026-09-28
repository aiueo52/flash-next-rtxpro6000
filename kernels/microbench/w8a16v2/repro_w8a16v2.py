"""Reproduction gate: does the harness reproduce the in-server kernel medians?

Runs every ground-truth row with `shapes.pre_v3_plan`, the config the server was
actually running when the profile was taken, and reports the CUPTI kernel-duration
median (the metric the serving torch profile reports) next to the CUDA-event replay
median. Pass --tuned to measure what `_plan` picks now instead.

  repro_w8a16v2.py                 # W16 ground truth
  repro_w8a16v2.py --w4            # W4 ground truth (M=4 rows)
  repro_w8a16v2.py --clocks        # also sample clocks.sm during each row
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../python"))

import torch
import triton

from harness import (
    cupti_kernel_us,
    make_copies,
    n_copies,
    require_idle_gpu,
    time_graph,
)
from shapes import (
    GROUND_TRUTH,
    GROUND_TRUTH_GRID,
    GROUND_TRUTH_W4,
    SHAPES,
    make_weight,
    pre_v3_plan,
)

from sglang.srt.layers.quantization import w8a16_gemv as new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w4", action="store_true")
    ap.add_argument("--clocks", action="store_true")
    ap.add_argument("--secs", type=float, default=0.6)
    ap.add_argument("--events", type=int, default=600)
    ap.add_argument("--tuned", action="store_true",
                    help="measure _plan's current choice instead of the pre-v3 plan")
    a = ap.parse_args()
    require_idle_gpu()
    torch.manual_seed(0)
    dev = "cuda"
    p = torch.cuda.get_device_properties(0)
    print(f"device: {p.name}  SMs={p.multi_processor_count}  "
          f"L2={p.L2_cache_size / 1e6:.0f} MB  vram={p.total_memory / 1e9:.0f} GB")

    gt = GROUND_TRUTH_W4 if a.w4 else GROUND_TRUTH
    by_name = {s[0]: s for s in SHAPES}
    rows = []
    for name in [s[0] for s in SHAPES]:
        Ms = sorted({m for (n, m) in gt if n == name})
        if not Ms:
            continue
        _, N, K, tag, _, _ = by_name[name]
        w, scale, wref = make_weight(N, K, tag, dev)
        wbytes = N * K * (1 if tag == "fp8" else 2)
        copies = make_copies(w, n_copies(wbytes))
        for M in Ms:
            x = (torch.randn(M, K, device=dev) * 0.5).to(torch.bfloat16)
            if a.tuned:
                cfg = new._plan(M, N, K, w.stride(0) == 1, w.element_size(),
                                new._num_sms(dev))
            else:
                cfg = pre_v3_plan(M, N, K, w.element_size(), new._num_sms(dev))
            if tag == "fp8":
                fn = lambda arr: new.w8a16_gemv(x, arr, scale, cfg=cfg)
            else:
                fn = lambda arr: new.bf16_gemv(x, arr, cfg=cfg)
            grid = (triton.cdiv(N, cfg[0]), cfg[2])
            ev, clk = time_graph(fn, copies, min_seconds=a.secs, clocks=a.clocks)
            cu, nev = cupti_kernel_us(fn, copies, target_events=a.events)
            rows.append((name, N, K, tag, M, wbytes, len(copies), grid, cfg,
                         ev, cu, nev, clk, gt[(name, M)]))
            del x
        del copies, w, wref
        torch.cuda.empty_cache()

    hdr = (f"{'shape':<18}{'N':>7}{'K':>6}{'M':>3}{'cp':>4}{'MB':>7}"
           f"{'grid':>11}{'cupti us':>10}{'event us':>10}{'TB/s':>7}"
           f"{'server':>9}{'ratio':>8}" + ("   sm MHz" if a.clocks else ""))
    print()
    print(hdr)
    print("-" * len(hdr))
    worst = 0.0
    for (name, N, K, tag, M, wb, ncp, grid, cfg, ev, cu, nev, clk, g) in rows:
        r = cu * 1e6 / g
        if tag == "fp8":
            worst = max(worst, abs(r - 1.0))
        c = f"   {clk[0]}-{clk[1]}" if a.clocks else ""
        gs = GROUND_TRUTH_GRID.get(name)
        mark = "" if gs is None or tuple(grid) == gs else "  !grid"
        print(f"{name:<18}{N:>7}{K:>6}{M:>3}{ncp:>4}{wb / 1e6:>7.1f}"
              f"{str(grid):>11}{cu * 1e6:>10.2f}{ev * 1e6:>10.2f}{wb / cu / 1e12:>7.2f}"
              f"{g:>9.1f}{r:>7.2f}x{c}{mark}")
    print()
    print("configs (BLOCK_N, BLOCK_K, SPLITS, USE_DOT, W_KN, warps, stages):")
    seen = set()
    for (name, N, K, tag, M, wb, ncp, grid, cfg, *_r) in rows:
        if (name, M) in seen:
            continue
        seen.add((name, M))
        print(f"  {name:<18} M={M:<3} {cfg}")
    print(f"\nworst fp8 deviation from server: {worst * 100:.1f}% "
          f"({'PASS' if worst <= 0.25 else 'FAIL'}, gate is +-25%)")
    return 0 if worst <= 0.25 else 1


if __name__ == "__main__":
    sys.exit(main())
