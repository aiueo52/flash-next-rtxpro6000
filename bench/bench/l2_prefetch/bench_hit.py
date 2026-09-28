"""Phase A item 1 -- does the production W8A16 GEMV get faster when its weight is
L2-resident, and can the weight be kept resident across the MoE expert stream?

For each real shape (GDN qkvz 16384x2560 = 42 MB, out_proj 2560x6144 = 15.7 MB) and
M in {4, 16}, one captured graph per condition:

  evict(320 MB, normal) -> [precondition] -> [stream Y MB of other data] -> GEMV

  cold            no precondition
  touched         touch W (normal loads)
  touched+stream  touch W, then stream Y MB (normal loads)               <- default policy
  persist+stream  touch W under a persisting access-policy window, stream Y (normal)
  persist+cs      same, stream Y with ld.global.cs (evict-first)
  persist+swin    same, stream Y under a streaming access-policy window
  elast+stream    touch W with the evict_last L2 cache hint, stream Y (normal)
  elast+cs        evict_last touch, evict-first stream
  swin-consumer   window set on the *capture stream* (inherited by the Triton GEMV node?)

The GEMV's own CUPTI duration is the number. `--sweep` adds the retention curve:
touch X MB, stream Y MB, re-touch X; hit fraction from the re-touch time.

Run:  . bench/l2_prefetch/env.sh; flock -w 28800 ~/.gpu.lock $PY bench/l2_prefetch/bench_hit.py
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

import torch

from common import (MB, MODE_CS, MODE_EVICT_LAST, MODE_NORMAL, NORMAL, PERSISTING, STREAMING, Graph, L2Ctl,
                    banner, cupti_positions, emit, gemv_fns, gpu_note, make_weight, rate_gbs, sector_range, short)

SHAPES = {"qkvz": (16384, 2560), "out_proj": (2560, 6144), "draft_head": (49152, 2560), "attn_qkv": (13312, 2560)}
EVICT_MB = 320
GRID_BIG = 376  # 2 CTAs / SM: full-bandwidth touch


def run_cond(ctl, cond, w, s, x, y, gemv, evict, other, y_mb, hit_ratio, capture_stream, limit_bytes):
    wbase, wbytes = sector_range(w)
    ybytes = y_mb * MB

    def call():
        st = torch.cuda.current_stream()
        ctl.touch(st, evict, mode=MODE_NORMAL, grid=GRID_BIG)
        if cond == "cold":
            pass
        elif cond.startswith("touched"):
            ctl.touch(st, w, mode=MODE_NORMAL, grid=GRID_BIG)
        elif cond.startswith("persistpfx"):
            # deterministic prefix window: the first min(limit, weight) bytes, hitRatio 1
            ctl.touch(st, w, mode=MODE_NORMAL, grid=GRID_BIG,
                      window=(wbase, min(wbytes, limit_bytes), 1.0, PERSISTING, STREAMING))
        elif cond.startswith("persist"):
            ctl.touch(st, w, mode=MODE_NORMAL, grid=GRID_BIG,
                      window=(wbase, wbytes, hit_ratio, PERSISTING, STREAMING))
        elif cond.startswith("elast"):
            ctl.touch(st, w, mode=MODE_EVICT_LAST, grid=GRID_BIG)
        elif cond == "swin-consumer":
            ctl.touch(st, w, mode=MODE_NORMAL, grid=GRID_BIG)
        if cond.endswith("+stream") or cond == "swin-consumer":
            if ybytes:
                ctl.touch(st, other, mode=MODE_NORMAL, grid=GRID_BIG, nbytes=ybytes)
        elif cond.endswith("+cs"):
            if ybytes:
                ctl.touch(st, other, mode=MODE_CS, grid=GRID_BIG, nbytes=ybytes)
        elif cond.endswith("+swin"):
            if ybytes:
                ob, _ = sector_range(other)
                # a window cannot exceed cudaDevAttrMaxAccessPolicyWindowSize (128 MiB here)
                ctl.touch(st, other, mode=MODE_NORMAL, grid=GRID_BIG, nbytes=ybytes,
                          window=(ob, min(ybytes, 128 * MB), 1.0, STREAMING, STREAMING))
        gemv(x, w, s, out=y)

    ctl.reset_persisting()
    if cond == "swin-consumer":
        ctl.stream_window(capture_stream, wbase, wbytes, hit_ratio, PERSISTING, STREAMING)
    try:
        gr = Graph(ctl, call, capture_stream=capture_stream)
    finally:
        if cond == "swin-consumer":
            ctl.stream_window(capture_stream, 0, 0, 0.0, NORMAL, NORMAL)
    nodes = gr.nodes()
    from harness import ClockSampler
    cs = ClockSampler(period=0.05); cs.start()
    pos, _ = cupti_positions(gr, reps=40)
    clk = cs.stop()
    gr.close()
    ctl.reset_persisting()
    gemv_us = [t for n, t, _ in pos if "_w8a16_gemv_kernel" in n]
    touches = [(short(n), round(t, 2)) for n, t, _ in pos if "touch" in n]
    row = {
        "cond": cond, "y_mb": y_mb, "hit_ratio": hit_ratio, "limit_mb": limit_bytes // MB, "sm_clk": clk,
        "gemv_us": round(gemv_us[-1], 3) if gemv_us else None,
        "gemv_gbs": round(rate_gbs(wbytes, gemv_us[-1]), 0) if gemv_us else None,
        "touch_us": touches,
        "nodes": [(short(n["name"]), n["win_bytes"] // MB if n["win_bytes"] else 0, n["hit_prop"], n["miss_prop"])
                  for n in nodes if n["type"] == 0],
    }
    return row


def sweep(ctl, evict, pool, other, limit_max):
    """Retention curve: touch X, stream Y, re-touch X (timed)."""
    rows = []
    xs = [8, 16, 24, 32, 48, 64, 80, 96, 112]
    ys = [0, 40, 77, 120, 190]
    for policy in ("normal", "persist", "elast"):
        if policy == "persist":
            ctl.set_persisting(limit_max)
        else:
            ctl.set_persisting(0)
        for x_mb in xs:
            pb, _ = sector_range(pool)
            xb = x_mb * MB
            for y_mb in ys:
                yb = y_mb * MB

                def call():
                    st = torch.cuda.current_stream()
                    ctl.touch(st, evict, mode=MODE_NORMAL, grid=GRID_BIG)
                    if policy == "persist":
                        ctl.touch_ptr(st, pb, xb, mode=MODE_NORMAL, grid=GRID_BIG,
                                      window=(pb, xb, 1.0, PERSISTING, STREAMING))
                    elif policy == "elast":
                        ctl.touch_ptr(st, pb, xb, mode=MODE_EVICT_LAST, grid=GRID_BIG)
                    else:
                        ctl.touch_ptr(st, pb, xb, mode=MODE_NORMAL, grid=GRID_BIG)
                    if yb:
                        ctl.touch(st, other, mode=MODE_CS if policy == "elast" else MODE_NORMAL,
                                  grid=GRID_BIG, nbytes=yb)
                    ctl.touch_ptr(st, pb, xb, mode=MODE_NORMAL, grid=GRID_BIG)  # timed re-touch

                ctl.reset_persisting()
                gr = Graph(ctl, call)
                pos, _ = cupti_positions(gr, reps=30)
                gr.close()
                ts = [t for n, t, _ in pos if "touch" in n]
                # positions: evict, first touch (cold), [stream], re-touch
                cold = ts[1]
                re = ts[-1]
                rows.append({"policy": policy, "x_mb": x_mb, "y_mb": y_mb, "cold_us": round(cold, 2),
                             "re_us": round(re, 2), "stream_us": round(ts[2], 2) if yb else None})
                print(f"  {policy:8s} X={x_mb:4d} Y={y_mb:4d}  cold {cold:8.2f}  re {re:8.2f}"
                      + (f"  stream {ts[2]:8.2f}" if yb else ""), flush=True)
    ctl.reset_persisting()
    ctl.set_persisting(0)
    # hot reference per X: Y=0 rows
    hot = {(r["policy"], r["x_mb"]): r["re_us"] for r in rows if r["y_mb"] == 0}
    for r in rows:
        h = hot[(r["policy"], r["x_mb"])]
        c = r["cold_us"]
        r["hit_frac"] = round(max(0.0, min(1.0, (c - r["re_us"]) / max(c - h, 1e-6))), 3)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--shapes", default="qkvz,out_proj,attn_qkv")
    ap.add_argument("--ms", default="4,16")
    ap.add_argument("--m1-shapes", default="draft_head", help="extra shapes measured at M=1 only")
    ap.add_argument("--ys", default="77,190")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results", "hit.json"))
    args = ap.parse_args()
    torch.cuda.init()
    ctl = L2Ctl()
    q = ctl.query()
    banner(f"device limits: {json.dumps(q)}")
    limit_max = ctl.set_persisting(q["max_persisting_bytes"])
    print(f"persisting limit requested {q['max_persisting_bytes']//MB} MB -> actual {limit_max//MB} MB")
    ctl.set_persisting(0)
    gemv = gemv_fns()
    evict = torch.empty(EVICT_MB * MB, dtype=torch.uint8, device="cuda")
    other = torch.empty(200 * MB, dtype=torch.uint8, device="cuda")
    capture_stream = torch.cuda.Stream()
    results = {"gpu": gpu_note(), "limits": q, "persisting_limit_actual": limit_max, "rows": []}

    jobs = [(sh, int(m)) for sh in args.shapes.split(",") for m in args.ms.split(",")]
    jobs += [(sh, 1) for sh in args.m1_shapes.split(",") if sh]
    for shape, M in jobs:
        N, K = SHAPES[shape]
        if True:
            w, s, x, y = make_weight(N, K, M)
            wbase, wbytes = sector_range(w)
            banner(f"{shape} N={N} K={K} M={M} weight {wbytes/MB:.1f} MB")
            plan = []
            # default L2 policy, no carve-out
            plan += [("cold", 0, 0, 1.0), ("touched", 0, 0, 1.0)]
            for y_mb in (int(v) for v in args.ys.split(",")):
                plan.append(("touched+stream", y_mb, 0, 1.0))
                plan.append(("elast+stream", y_mb, 0, 1.0))
                plan.append(("elast+cs", y_mb, 0, 1.0))
            # with the persisting carve-out
            if limit_max > 0:
                hr = 1.0 if wbytes <= limit_max else limit_max / wbytes
                plan.append(("touched", 0, limit_max, 1.0))
                for y_mb in (int(v) for v in args.ys.split(",")):
                    plan.append(("touched+stream", y_mb, limit_max, 1.0))
                    plan.append(("persist+stream", y_mb, limit_max, hr))
                    plan.append(("persist+cs", y_mb, limit_max, hr))
                    plan.append(("persist+swin", y_mb, limit_max, hr))
                    plan.append(("swin-consumer", y_mb, limit_max, hr))
                    if hr < 1.0:
                        plan.append(("persist+stream", y_mb, limit_max, 1.0))
                        plan.append(("persistpfx+stream", y_mb, limit_max, 1.0))
            for cond, y_mb, lim, hr in plan:
                ctl.set_persisting(lim)
                row = run_cond(ctl, cond, w, s, x, y, gemv, evict, other, y_mb, hr, capture_stream, lim)
                row.update({"shape": shape, "N": N, "K": K, "M": M, "w_mb": round(wbytes / MB, 1)})
                results["rows"].append(row)
                print(f"  {cond:16s} Y={y_mb:4d} lim={lim//MB:3d} hr={hr:.2f}  gemv {row['gemv_us']:8.2f} us "
                      f"= {row['gemv_gbs']:6.0f} GB/s   touches {row['touch_us']} clk {row['sm_clk']}", flush=True)
                if cond == "swin-consumer":
                    print("     nodes:", row["nodes"], flush=True)
            del w, s, x, y
            ctl.set_persisting(0)

    if args.sweep:
        banner("retention sweep")
        pool = torch.empty(128 * MB, dtype=torch.uint8, device="cuda")
        results["sweep"] = sweep(ctl, evict, pool, other, limit_max)
    emit(args.out, results)


if __name__ == "__main__":
    main()
