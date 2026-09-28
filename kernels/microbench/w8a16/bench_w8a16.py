"""A/B benchmark for the W8A16 skinny GEMV: frozen baseline vs. the current dispatcher.

Each measurement replays a CUDA graph that rotates over >=24 distinct weight copies
(sized so the working set is ~4x the 128 MB L2), so the achieved TB/s is HBM traffic.

Usage:
  bench_w8a16.py                 # all shapes, all M
  bench_w8a16.py --shapes shared_gate_up,shared_down --m 1,16
"""

import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../python"))

import torch
import baseline_w8a16 as base
from harness import make_copies, n_copies, require_idle_gpu, time_graph
from shapes import DEFAULT_MS, MODES, SHAPE_MS, SHAPES, make_weight

from sglang.srt.layers.quantization import w8a16_gemv as new


def bench_one(name, N, K, tag, Ms, iters):
    dev = "cuda"
    w, scale, wref = make_weight(N, K, tag, dev)
    esz = 1 if tag == "fp8" else 2
    wbytes = N * K * esz
    ncp = n_copies(wbytes)
    copies = make_copies(w, ncp)
    rows = []
    for M in Ms:
        x = (torch.randn(M, K, device=dev) * 0.5).to(torch.bfloat16)
        ref = x.float() @ wref.t()
        rowdata = {}
        for label, mod in (("before", base), ("after", new)):
            if tag == "fp8":
                fn = lambda a, m=mod: m.w8a16_gemv(x, a, scale)
            else:
                fn = lambda a, m=mod: m.bf16_gemv(x, a)
            y = fn(copies[0])
            err = (y.float() - ref).abs().max().item()
            rowdata[label] = (time_graph(fn, copies, iters=iters), err)
            # Same graph shape, one weight: the draft steps re-read the same weights
            # 3-14x per decode step, so those calls run L2-warm in the server.
            rowdata[label + "_warm"] = (time_graph(fn, [copies[0]] * 24, iters=iters), err)
        cfg = new._plan(M, N, K, w.stride(0) == 1, new._num_sms(dev), w.element_size())
        rows.append((name, N, K, tag, M, wbytes, rowdata, cfg))
        del x, ref
    del copies, w, wref
    torch.cuda.empty_cache()
    return rows


def fmt(rows):
    hdr = (f"{'shape':<18}{'N':>7}{'K':>7}{'M':>3} {'before us':>10}{'after us':>10}"
           f"{'speedup':>9}{'TB/s bef':>9}{'TB/s aft':>9}{'warm us':>9}  "
           f"{'err bef':>9}{'err aft':>9}  config")
    print(hdr)
    print("-" * len(hdr))
    for name, N, K, tag, M, wbytes, d, cfg in rows:
        b, a, aw = d["before"], d["after"], d["after_warm"]
        print(f"{name:<18}{N:>7}{K:>7}{M:>3} {b[0]*1e6:>10.2f}{a[0]*1e6:>10.2f}"
              f"{b[0]/a[0]:>8.2f}x{wbytes/b[0]/1e12:>9.2f}{wbytes/a[0]/1e12:>9.2f}"
              f"{aw[0]*1e6:>9.2f}  {b[1]:>9.2e}{a[1]:>9.2e}  {cfg}")


def savings(rows):
    """ms/step saved, using the verify-pass + draft-step call model in shapes.py.

    "cold" charges every call the L2-defeating time. "mixed" is the closer model of the
    server: the verify pass streams each weight once (cold), while the draft steps rerun
    one MTP layer and the draft head n_draft times over weights that stay in L2 (warm).
    """
    by = {(r[0], r[4]): r[6] for r in rows}
    print()
    for mode, (spec_w, n_draft) in MODES.items():
        for how, draft_key in (("cold ", ""), ("mixed", "_warm")):
            tot_b = tot_a = 0.0
            calls = 0
            missing = []
            for name, n_verify, n_draft_calls in ((s[0], s[4], s[5]) for s in SHAPES):
                for M, c, key in ((spec_w, n_verify, ""), (1, n_draft_calls * n_draft, draft_key)):
                    if c == 0:
                        continue
                    d = by.get((name, M))
                    if d is None:
                        missing.append((name, M))
                        continue
                    tot_b += d["before" + key][0] * c
                    tot_a += d["after" + key][0] * c
                    calls += c
            print(f"{mode} {how}: spec_width={spec_w} draft_steps={n_draft} calls={calls}  "
                  f"before {tot_b*1e3:.3f} -> after {tot_a*1e3:.3f} ms/step  "
                  f"(saved {(tot_b-tot_a)*1e3:.3f} ms/step, {100*(tot_b-tot_a)/tot_b:.1f}%)")
            if missing:
                print(f"   (not measured, excluded: {missing})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="")
    ap.add_argument("--m", default="")
    ap.add_argument("--iters", type=int, default=50)
    a = ap.parse_args()
    require_idle_gpu()
    torch.manual_seed(0)
    print(f"device: {torch.cuda.get_device_name()}  "
          f"SMs={torch.cuda.get_device_properties(0).multi_processor_count}")
    want = set(a.shapes.split(",")) if a.shapes else None
    forced = [int(v) for v in a.m.split(",")] if a.m else None
    rows = []
    for name, N, K, tag, _, _ in SHAPES:
        if want and name not in want:
            continue
        Ms = forced or list(SHAPE_MS.get(name, DEFAULT_MS))
        rows += bench_one(name, N, K, tag, Ms, a.iters)
    fmt(rows)
    if not want and not forced:
        savings(rows)
