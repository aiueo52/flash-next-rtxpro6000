"""Config sweep for one (shape, M) of the W8A16 skinny GEMV, event-timed.

Superseded by tune_v3.py, which measures CUPTI kernel durations (what the serving
profile reports) instead of the graph-replay wall time this script uses; kept for
quick single-shape sweeps.

  tune_w8a16v2.py o_proj --m 16
  tune_w8a16v2.py gdn_in_proj_ba --m 16 --bn 16,32 --splits 1,8,16,20,32

Timing is the same CUDA-graph-over-24-copies replay as bench_w8a16v2.py but with a
shorter window per candidate; re-check the winner with bench_w8a16v2.py before pinning
it in `_TUNED`. Configs whose max-abs error exceeds the baseline kernel's by more than
2 bf16 ulps are dropped.
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../python"))

import torch
import triton

import baseline_v1 as base
from harness import capture, make_copies, n_copies, require_idle_gpu, time_graph
from shapes import SHAPES, make_weight

from sglang.srt.layers.quantization import w8a16_gemv as new


def ints(s):
    return [int(v) for v in s.split(",") if v]


def candidates(a, N, K, M, w_bytes):
    n_blocks_of = lambda bn: triton.cdiv(N, bn)
    out = []
    for bn in ints(a.bn):
        if bn < 16 and M > 1:
            continue
        for bk in ints(a.bk):
            if bn * bk > 65536:
                continue
            n_kb = triton.cdiv(K, bk)
            for sp in ints(a.splits):
                if sp > n_kb or sp > new._MAX_SPLITS:
                    continue
                m_pad = 16 if M > 1 else 1
                if sp > 1 and n_blocks_of(bn) * sp * m_pad * bn > new._WS_FLOATS:
                    continue
                for ud in ([True] if M > 1 else ints(a.dot)):
                    for wk in ints(a.wkn):
                        for nw in ints(a.warps):
                            for ns in ints(a.stages):
                                out.append((bn, bk, sp, bool(ud), bool(wk), nw, ns))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shape")
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--bn", default="32,64,128")
    ap.add_argument("--bk", default="64,128,256,512")
    ap.add_argument("--splits", default="1,2,4,8,16")
    ap.add_argument("--warps", default="4")
    ap.add_argument("--stages", default="3")
    ap.add_argument("--dot", default="1")
    ap.add_argument("--wkn", default="")
    ap.add_argument("--secs", type=float, default=0.15)
    ap.add_argument("--spin", type=float, default=0.06)
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()
    require_idle_gpu()

    name, N, K, tag = next((s[0], s[1], s[2], s[3]) for s in SHAPES if s[0] == a.shape)
    dev = "cuda"
    w, scale, wref = make_weight(N, K, tag, dev)
    if not a.wkn:
        a.wkn = "1" if w.stride(0) == 1 else "0"
    wbytes = N * K * (1 if tag == "fp8" else 2)
    copies = make_copies(w, n_copies(wbytes))
    M = a.m
    x = (torch.randn(M, K, device=dev) * 0.5).to(torch.bfloat16)
    ref = x.float() @ wref.t()
    mx = ref.abs().max().item()
    ulp = 2.0 ** (math.floor(math.log2(mx)) - 7)

    if tag == "fp8":
        run = lambda arr, cfg: new.w8a16_gemv(x, arr, scale, cfg=cfg)
        eb = (base.w8a16_gemv(x, w, scale).float() - ref).abs().max().item()
    else:
        run = lambda arr, cfg: new.bf16_gemv(x, arr, cfg=cfg)
        eb = (base.bf16_gemv(x, w).float() - ref).abs().max().item()

    cands = candidates(a, N, K, M, 1 if tag == "fp8" else 2)
    print(f"{name} N={N} K={K} M={M} {tag} {wbytes/1e6:.1f} MB, {len(copies)} copies, "
          f"{len(cands)} candidates, baseline err {eb:.3e}, 2ulp {2*ulp:.3e}")
    res = []
    for i, cfg in enumerate(cands):
        try:
            y = run(copies[0], cfg)
            torch.cuda.synchronize()
            err = (y.float() - ref).abs().max().item()
            if not math.isfinite(err) or err > eb + 2 * ulp:
                print(f"  [{i+1}/{len(cands)}] {cfg} bad err {err:.3e}")
                continue
            t, _ = time_graph(lambda arr: run(arr, cfg), copies,
                              min_seconds=a.secs, spin_seconds=a.spin)
            res.append((t, cfg, err))
        except Exception as e:
            print(f"  [{i+1}/{len(cands)}] {cfg} {type(e).__name__}: {str(e)[:90]}")
    res.sort()
    print(f"\n{'us':>9}{'TB/s':>8}  (BLOCK_N, BLOCK_K, SPLITS, USE_DOT, W_KN, warps, stages)"
          f"   ctas  err")
    for t, cfg, err in res[: a.top]:
        print(f"{t*1e6:>9.2f}{wbytes/t/1e12:>8.2f}  {cfg}   "
              f"{triton.cdiv(N, cfg[0]) * cfg[2]:>5}  {err:.2e}")


if __name__ == "__main__":
    main()
