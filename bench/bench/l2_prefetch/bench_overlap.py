"""Phase A item 2 -- is a side-stream prefetch hidden behind a latency-bound chain,
and what does it cost a DRAM-bound kernel that it happens to overlap?

Per round r (R rounds rotate the prefetch range over a 256 MB pool so it is always
DRAM-cold):   fork -> side: touch X MB (grid PG)   main: chain   -> join

  chain = 40 tiny Triton kernels (grid G)           -- the glue emulation
        | 4 x qkvz GEMV over rotating 42 MB copies  -- DRAM-bound consumer
        | touch 77 MB (grid 376)                    -- the MoE GEMM proxy

Reports: main-only, side-only, both, both with node priorities (main high, side low;
graph instantiated with cudaGraphInstantiateFlagUseNodePriority) -- all per round,
plus CUPTI medians of the chain kernels with/without the side branch.

Run:  . bench/l2_prefetch/env.sh; flock -w 28800 ~/.gpu.lock $PY bench/l2_prefetch/bench_overlap.py
"""

from __future__ import annotations

import argparse
import os
import statistics

import torch

from common import (MB, MODE_NORMAL, Graph, L2Ctl, banner, cupti_positions, emit, gemv_fns, gpu_note,
                    make_weight, rate_gbs, sector_range, short, time_replay)
from kernels import tiny

R = 4
POOL_MB = 256


def build(ctl, main_call, side_fn, x_mb, pg, prio_mode, pool, gemv_bufs=None):
    """prio_mode: None (plain), 'prio' (stream priorities + node-priority instantiate),
    'attr' (launch-attribute priority on the touch + node-priority instantiate)."""
    q = ctl.query()
    hi, lo = q["prio_greatest"], q["prio_least"]
    main_s = torch.cuda.Stream(priority=hi if prio_mode else 0)
    side_s = torch.cuda.Stream(priority=lo if prio_mode else 0)
    pb, _ = sector_range(pool)
    xb = x_mb * MB

    def call():
        for r in range(R):
            cur = torch.cuda.current_stream()
            if side_fn is not None:
                side_s.wait_stream(cur)
                base = pb + r * xb
                ctl.touch_ptr(side_s, base, xb, mode=MODE_NORMAL, grid=pg,
                              prio=(lo if prio_mode == "attr" else None))
            if main_call is not None:
                main_call(r)
            if side_fn is not None:
                cur.wait_stream(side_s)

    return Graph(ctl, call, node_priority=bool(prio_mode), capture_stream=main_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x-mb", type=int, default=32)
    ap.add_argument("--grids", default="8,16,32,64,128,188,376")
    ap.add_argument("--n-tiny", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results", "overlap.json"))
    args = ap.parse_args()
    torch.cuda.init()
    ctl = L2Ctl()
    q = ctl.query()
    gemv = gemv_fns()
    pool = torch.empty(POOL_MB * MB, dtype=torch.uint8, device="cuda")
    buf20 = torch.zeros(20 * 256, device="cuda")
    buf160 = torch.zeros(160 * 256, device="cuda")
    gemm_proxy = torch.empty(4 * 77 * MB, dtype=torch.uint8, device="cuda")
    gpb, _ = sector_range(gemm_proxy)
    ws = [make_weight(16384, 2560, 4) for _ in range(4)]
    results = {"gpu": gpu_note(), "limits": q, "x_mb": args.x_mb, "rows": []}
    xb = args.x_mb * MB

    chains = {
        "tiny20": lambda r: [tiny(buf20, 20) for _ in range(args.n_tiny)],
        "tiny160": lambda r: [tiny(buf160, 160) for _ in range(args.n_tiny)],
        "gemv4": lambda r: [gemv(ws[i][2], ws[i][0], ws[i][1], out=ws[i][3]) for i in range(4)],
        "gemm77": lambda r: ctl.touch_ptr(torch.cuda.current_stream(), gpb + r * 77 * MB, 77 * MB, grid=376),
    }
    grids = [int(v) for v in args.grids.split(",")]

    # side-only per grid
    banner("side-only prefetch rate")
    side_only = {}
    for pg in grids:
        gr = build(ctl, None, True, args.x_mb, pg, None, pool)
        t = time_replay(gr) / R
        gr.close()
        side_only[pg] = t
        print(f"  grid {pg:4d}: {t:8.2f} us / {args.x_mb} MB = {rate_gbs(xb, t):6.0f} GB/s", flush=True)
        results["rows"].append({"chain": "none", "grid": pg, "prio": None, "us": round(t, 2), "gbs": round(rate_gbs(xb, t))})

    for cname, cfn in chains.items():
        banner(f"chain {cname}")
        gr = build(ctl, cfn, None, args.x_mb, 1, None, pool)
        t_main = time_replay(gr) / R
        pos, _ = cupti_positions(gr, reps=20)
        gr.close()
        ref = {}
        for n, t, _ in pos:
            ref.setdefault(short(n), []).append(t)
        ref = {k: statistics.median(v) for k, v in ref.items()}
        print(f"  main-only: {t_main:8.2f} us/round   kernels {[(k, round(v, 2)) for k, v in ref.items()]}", flush=True)
        results["rows"].append({"chain": cname, "grid": 0, "prio": None, "us": round(t_main, 2), "kernels": ref})
        for pg in grids:
            for prio in (None, "prio", "attr"):
                if prio == "attr" and cname != "tiny20":
                    continue
                gr = build(ctl, cfn, True, args.x_mb, pg, prio, pool)
                t_both = time_replay(gr) / R
                pos, _ = cupti_positions(gr, reps=20)
                nodes = gr.nodes() if (prio and pg == grids[0]) else None
                gr.close()
                by = {}
                for n, t, _ in pos:
                    by.setdefault(short(n), []).append(t)
                by = {k: statistics.median(v) for k, v in by.items()}
                touch_us = next((v for k, v in by.items() if k.startswith("touch")), float("nan"))
                chain_slow = {k: round(by[k] / ref[k], 3) for k in ref if k in by}
                extra = t_both - t_main
                row = {"chain": cname, "grid": pg, "prio": prio, "us": round(t_both, 2),
                       "extra_us": round(extra, 2), "side_only_us": round(side_only[pg], 2),
                       "hidden_frac": round(1 - extra / side_only[pg], 3),
                       "touch_us_concurrent": round(touch_us, 2), "chain_kernel_slowdown": chain_slow}
                if nodes:
                    row["node_priorities"] = [(short(n["name"]), n["priority"]) for n in nodes if n["type"] == 0][:6]
                results["rows"].append(row)
                print(f"  grid {pg:4d} prio={str(prio):5s}: both {t_both:8.2f} (+{extra:6.2f} vs main; side alone "
                      f"{side_only[pg]:7.2f}) hidden {row['hidden_frac']:5.2f}  touch {touch_us:7.2f} us  "
                      f"chain x{chain_slow}", flush=True)
                if nodes:
                    print("     node prios:", row["node_priorities"], flush=True)
    emit(args.out, results)


if __name__ == "__main__":
    main()
