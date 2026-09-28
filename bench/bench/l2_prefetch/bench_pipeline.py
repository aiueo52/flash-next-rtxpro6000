"""Phase A end-to-end emulation -- a rotating chain of emulated decoder layers, with
and without a side-stream prefetch of the NEXT layer's dense weights.

Per layer (W4 medians from EXCLUSIVE_TIME_MAP.md, glue emulated with tiny Triton
kernels of ~2 us):
   tiny x6 | GEMV qkvz (42 MB) | tiny x5 | GEMV out_proj (15.7 MB) | tiny x12 (HC + routing)
   | touch E MB (MoE GEMM proxy, 77 W4 / 190 W16) | tiny x8 (activation + HC post)

Each layer's two weights live in one contiguous per-layer arena (the Phase B layout)
so one access-policy window covers them. NL layers rotate (NL x 58 MB >= 3.6x L2), so
without prefetch every GEMV is DRAM-cold, as in the server.

Prefetch variants (side stream, grid PG, next layer's arena):
   none            no prefetch (the control)
   start           whole arena forked at layer start, normal policy
   start+persist   same, persisting window (carve-out = max)
   postgemm        whole arena forked right after the GEMM proxy, normal policy
   postgemm+persist
   jit             qkvz(next) after the GEMM proxy, out_proj(next) after next's qkvz GEMV -- normal policy

Run:  . bench/l2_prefetch/env.sh; flock -w 28800 ~/.gpu.lock $PY bench/l2_prefetch/bench_pipeline.py
"""

from __future__ import annotations

import argparse
import os
import statistics

import torch

from common import (MB, MODE_NORMAL, PERSISTING, STREAMING, Graph, L2Ctl, banner, cupti_positions, emit,
                    gemv_fns, gpu_note, rate_gbs, sector_range, short, time_replay)
from kernels import tiny

NL = 8
QKVZ = (16384, 2560)
OUT = (2560, 6144)


def make_layer(M, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    n1 = QKVZ[0] * QKVZ[1]
    n2 = OUT[0] * OUT[1]
    arena = torch.empty(n1 + n2, dtype=torch.float8_e4m3fn, device="cuda")
    # same value distribution as make_weight (random *bytes* gave 1.8x slower GEMVs: NaN/large
    # magnitudes toggle every bit and the Max-Q power cap throttles the clocks)
    for lo in range(0, n1 + n2, 1 << 24):
        hi = min(lo + (1 << 24), n1 + n2)
        arena[lo:hi] = (torch.randn(hi - lo, device="cuda", generator=g) / 8).to(torch.float8_e4m3fn)
    w1 = arena[:n1].view(QKVZ[0], QKVZ[1])  # [N,K] contiguous (production layout)
    w2 = arena[n1:].view(OUT[0], OUT[1])
    s1 = (torch.rand(QKVZ[0], device="cuda", generator=g) + 0.5).float() / 64
    s2 = (torch.rand(OUT[0], device="cuda", generator=g) + 0.5).float() / 64
    x1 = torch.randn(M, QKVZ[1], device="cuda", dtype=torch.bfloat16, generator=g) / 8
    x2 = torch.randn(M, OUT[1], device="cuda", dtype=torch.bfloat16, generator=g) / 8
    y1 = torch.empty(M, QKVZ[0], device="cuda", dtype=torch.bfloat16)
    y2 = torch.empty(M, OUT[0], device="cuda", dtype=torch.bfloat16)
    return dict(arena=arena, w1=w1, s1=s1, x1=x1, y1=y1, w2=w2, s2=s2, x2=x2, y2=y2)


def run_variant(ctl, gemv, layers, variant, e_mb, pg, buf, gemm_pool, tiny_grid, prio, glue_scale=1.0, join_each=True):
    q = ctl.query()
    hi, lo = q["prio_greatest"], q["prio_least"]
    main_s = torch.cuda.Stream(priority=hi if prio else 0)
    side_s = torch.cuda.Stream(priority=lo if prio else 0)
    gpb, _ = sector_range(gemm_pool)
    persist = variant.endswith("+persist")
    n6, n5, n12, n8 = (max(1, round(v * glue_scale)) for v in (6, 5, 12, 8))

    def prefetch(L, which):
        cur = torch.cuda.current_stream()
        side_s.wait_stream(cur)
        if which == "arena":
            base, nb = sector_range(L["arena"])
        elif which == "w1":
            base, nb = sector_range(L["w1"])
        else:
            base, nb = sector_range(L["w2"])
        win = (base, nb, 1.0, PERSISTING, STREAMING) if persist else None
        ctl.touch_ptr(side_s, base, nb, mode=MODE_NORMAL, grid=pg, window=win)

    def join():
        torch.cuda.current_stream().wait_stream(side_s)

    def call():
        for i in range(NL):
            L = layers[i]
            Ln = layers[(i + 1) % NL]
            if variant.startswith("start"):
                prefetch(Ln, "arena")
            for _ in range(n6):
                tiny(buf, tiny_grid)
            gemv(L["x1"], L["w1"], L["s1"], out=L["y1"])
            if variant == "jit":
                prefetch(L, "w2")
            for _ in range(n5):
                tiny(buf, tiny_grid)
            gemv(L["x2"], L["w2"], L["s2"], out=L["y2"])
            for _ in range(n12):
                tiny(buf, tiny_grid)
            ctl.touch_ptr(torch.cuda.current_stream(), gpb + (i % 4) * e_mb * MB, e_mb * MB, grid=376)
            if variant.startswith("postgemm"):
                prefetch(Ln, "arena")
            elif variant == "jit":
                prefetch(Ln, "w1")
            for _ in range(n8):
                tiny(buf, tiny_grid)
            if variant != "none" and (join_each or i == NL - 1):
                join()

    ctl.reset_persisting()
    gr = Graph(ctl, call, node_priority=prio, capture_stream=main_s)
    from harness import ClockSampler
    cs = ClockSampler(); cs.start()
    t = time_replay(gr, min_seconds=0.6) / NL
    clk = cs.stop()
    pos, _ = cupti_positions(gr, reps=12)
    gr.close()
    ctl.reset_persisting()
    by = {}
    for n, d, _ in pos:
        by.setdefault(short(n), []).append(d)
    gemvs = sorted(by.get("_w8a16_gemv_kernel", []))
    # the two GEMV shapes separate by duration (qkvz ~29 us cold vs out_proj ~13)
    half = len(gemvs) // 2
    small, big = gemvs[:half], gemvs[half:]
    touches = sorted(sum((v for k, v in by.items() if k.startswith("touch_kernel")), []))
    return {
        "variant": variant, "e_mb": e_mb, "grid": pg, "prio": prio, "us_per_layer": round(t, 2), "sm_clk": clk,
        "gemv_qkvz_med": round(statistics.median(big), 2) if big else None,
        "gemv_out_med": round(statistics.median(small), 2) if small else None,
        "tiny_med": round(statistics.median(by.get("_tiny_kernel", [0])), 2),
        "touch_meds": [round(statistics.median(v), 2) for v in (touches[:NL], touches[NL:]) if v],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="4,16")
    ap.add_argument("--es", default="77,190")
    ap.add_argument("--grids", default="32,64,188")
    ap.add_argument("--tiny-grid", type=int, default=20)
    ap.add_argument("--variants", default="none,start,start+persist,postgemm,postgemm+persist,jit")
    ap.add_argument("--glue-scale", type=float, default=1.0, help="tiny kernels per glue window x this (0.8us each)")
    ap.add_argument("--no-join-each", action="store_true", help="join the side stream once at the end, not per layer")
    ap.add_argument("--prios", default="0,1")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results", "pipeline.json"))
    args = ap.parse_args()
    torch.cuda.init()
    ctl = L2Ctl()
    q = ctl.query()
    gemv = gemv_fns()
    buf = torch.zeros(160 * 256, device="cuda")
    gemm_pool = torch.empty(4 * 190 * MB, dtype=torch.uint8, device="cuda")
    results = {"gpu": gpu_note(), "limits": q, "rows": []}
    for M in (int(v) for v in args.ms.split(",")):
        layers = [make_layer(M, 100 + i) for i in range(NL)]
        for e_mb in (int(v) for v in args.es.split(",")):
            banner(f"M={M} expert stream {e_mb} MB/layer")
            for variant in args.variants.split(","):
                ctl.set_persisting(q["max_persisting_bytes"] if variant.endswith("+persist") else 0)
                for pg in ([0] if variant == "none" else [int(v) for v in args.grids.split(",")]):
                    for prio in ([False] if variant == "none" else [bool(int(v)) for v in args.prios.split(",")]):
                        row = run_variant(ctl, gemv, layers, variant, e_mb, pg, buf, gemm_pool, args.tiny_grid, prio,
                                          glue_scale=args.glue_scale, join_each=not args.no_join_each)
                        row["M"] = M
                        results["rows"].append(row)
                        print(f"  {variant:18s} grid {pg:4d} prio={int(prio)}: {row['us_per_layer']:8.2f} us/layer  "
                              f"qkvz {row['gemv_qkvz_med']} out {row['gemv_out_med']} tiny {row['tiny_med']} "
                              f"touch {row['touch_meds']} clk {row['sm_clk']}", flush=True)
                ctl.set_persisting(0)
        del layers
        torch.cuda.empty_cache()
    emit(args.out, results)


if __name__ == "__main__":
    main()
