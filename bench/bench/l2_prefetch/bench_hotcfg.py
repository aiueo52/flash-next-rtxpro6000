"""Phase A item 1b -- how fast can the production GEMV read an L2-resident weight if its
tile is re-tuned for the hit case? (The DRAM-tuned tile reaches only ~2.2 TB/s from L2
in bench_hit.py while a plain touch kernel reads L2 at ~5.8 TB/s.)

Sweeps (BLOCK_N, BLOCK_K, SPLITS, warps, stages) on the L2-hot weight, then re-measures
the best few cold, so the hot/cold pair per tile is known. Same graph/CUPTI protocol as
bench_hit.py.

Run:  . bench/l2_prefetch/env.sh; flock -w 28800 ~/.gpu.lock $PY bench/l2_prefetch/bench_hotcfg.py
"""

from __future__ import annotations

import argparse
import itertools
import os

import torch

from common import MB, MODE_NORMAL, Graph, L2Ctl, banner, cupti_positions, emit, gemv_fns, gpu_note, make_weight, rate_gbs, sector_range

SHAPES = {"qkvz": (16384, 2560), "out_proj": (2560, 6144), "attn_qkv": (13312, 2560)}
GRID_BIG = 376


def measure(ctl, gemv, w, s, x, y, evict, cfg, hot):
    def call():
        st = torch.cuda.current_stream()
        ctl.touch(st, evict, mode=MODE_NORMAL, grid=GRID_BIG)
        if hot:
            ctl.touch(st, w, mode=MODE_NORMAL, grid=GRID_BIG)
        gemv(x, w, s, cfg=cfg, out=y)

    gr = Graph(ctl, call)
    pos, _ = cupti_positions(gr, reps=30)
    gr.close()
    t = [d for n, d, _ in pos if "_w8a16_gemv_kernel" in n]
    return t[-1] if t else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="qkvz,out_proj,attn_qkv")
    ap.add_argument("--ms", default="4,16")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results", "hotcfg.json"))
    args = ap.parse_args()
    torch.cuda.init()
    ctl = L2Ctl()
    gemv = gemv_fns()
    from sglang.srt.layers.quantization.w8a16_gemv import _fit, _num_sms, _plan
    evict = torch.empty(320 * MB, dtype=torch.uint8, device="cuda")
    results = {"gpu": gpu_note(), "rows": []}
    for shape in args.shapes.split(","):
        N, K = SHAPES[shape]
        for M in (int(v) for v in args.ms.split(",")):
            w, s, x, y = make_weight(N, K, M)
            _, wbytes = sector_range(w)
            prod = _plan(M, N, K, False, 1, _num_sms(x.device))
            banner(f"{shape} N={N} K={K} M={M}: production cfg {prod}")
            ref_cold = measure(ctl, gemv, w, s, x, y, evict, prod, hot=False)
            ref_hot = measure(ctl, gemv, w, s, x, y, evict, prod, hot=True)
            y_ref = y.clone()
            print(f"  production: cold {ref_cold:7.2f} us ({rate_gbs(wbytes, ref_cold):5.0f} GB/s)  hot {ref_hot:7.2f} us "
                  f"({rate_gbs(wbytes, ref_hot):5.0f} GB/s)", flush=True)
            results["rows"].append({"shape": shape, "M": M, "cfg": list(prod), "production": True,
                                    "cold_us": round(ref_cold, 3), "hot_us": round(ref_hot, 3)})
            seen = {tuple(prod)}
            hot_rows = []
            for bn, bk, sp, wp, stg in itertools.product((16, 32, 64, 128), (64, 128, 256), (1, 2, 4, 8), (4, 8), (3,)):
                if bn == 128 and bk == 256 and wp == 4:
                    continue
                cfg = _fit(M, N, (bn, bk, sp, True, None, wp, stg))
                if tuple(cfg) in seen:
                    continue
                seen.add(tuple(cfg))
                try:
                    t = measure(ctl, gemv, w, s, x, y, evict, cfg, hot=True)
                except Exception as e:  # tile does not compile / fit
                    print(f"  skip {cfg}: {str(e)[:60]}", flush=True)
                    continue
                ok = torch.allclose(y.float(), y_ref.float(), rtol=2e-2, atol=1e-2)
                hot_rows.append((t, cfg, ok))
            hot_rows.sort(key=lambda r: r[0])
            for t, cfg, ok in hot_rows[: args.top]:
                cold = measure(ctl, gemv, w, s, x, y, evict, cfg, hot=False)
                print(f"  {cfg}: hot {t:7.2f} us ({rate_gbs(wbytes, t):5.0f} GB/s)  cold {cold:7.2f} us "
                      f"({rate_gbs(wbytes, cold):5.0f} GB/s)  match={ok}", flush=True)
                results["rows"].append({"shape": shape, "M": M, "cfg": list(cfg), "production": False,
                                        "cold_us": round(cold, 3), "hot_us": round(t, 3), "match": ok})
            results["rows"].append({"shape": shape, "M": M, "all_hot": [(round(t, 3), list(c)) for t, c, _ in hot_rows]})
            del w, s, x, y
    emit(args.out, results)


if __name__ == "__main__":
    main()
