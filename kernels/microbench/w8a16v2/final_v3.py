"""Before/after table for the v3 tuning, in CUPTI kernel-duration medians.

"before" is the config the server runs today at 1eb2831e5c: `_plan`'s *fallback*
for the true [N, K] layout (the tuned table's keys assumed the wrong layout, so
every fp8 entry in it was dead). "after" is whatever `_plan` returns now.

  final_v3.py                 # full table + per-decode-step model
  final_v3.py --shapes o_proj,attn_qkv --ms 1,16
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../python"))

import torch

import baseline_v1 as base
from harness import cupti_kernel_us, make_copies, n_copies, require_idle_gpu
from shapes import GROUND_TRUTH, MODES, SECONDARY, SHAPES, make_weight, pre_v3_plan

from sglang.srt.layers.quantization import w8a16_gemv as new

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="")
    ap.add_argument("--ms", default="1,4,16")
    ap.add_argument("--secs", type=float, default=0.3)
    ap.add_argument("--events", type=int, default=2000)
    ap.add_argument("--v1", action="store_true", help="also time the pre-v2 kernel")
    a = ap.parse_args()
    require_idle_gpu()
    torch.manual_seed(0)
    dev = "cuda"
    sms = new._num_sms(dev)
    p = torch.cuda.get_device_properties(0)
    print(f"device: {p.name}  SMs={sms}  L2={p.L2_cache_size / 1e6:.0f} MB\n")

    by = {s[0]: s for s in SHAPES}
    want = a.shapes.split(",") if a.shapes else [s[0] for s in SHAPES]
    Ms = [int(v) for v in a.ms.split(",") if v]
    rows = []
    for name in want:
        _, N, K, tag, nv, nd = by[name]
        w, scale, wref = make_weight(N, K, tag, dev)
        wbytes = N * K * (1 if tag == "fp8" else 2)
        copies = make_copies(w, n_copies(wbytes))
        for M in Ms:
            x = (torch.randn(M, K, device=dev) * 0.5).to(torch.bfloat16)
            ref = x.float() @ wref.t()
            refmax = ref.abs().max().item()
            if tag == "fp8":
                run = lambda arr, cfg: new.w8a16_gemv(x, arr, scale, cfg=cfg)
                v1 = lambda arr: base.w8a16_gemv(x, arr, scale)
            else:
                run = lambda arr, cfg: new.bf16_gemv(x, arr, cfg=cfg)
                v1 = lambda arr: base.bf16_gemv(x, arr)
            fb = pre_v3_plan(M, N, K, 1 if tag == "fp8" else 2, sms)
            tuned = new._plan(M, N, K, w.stride(0) == 1, w.element_size(), sms)
            kw = dict(min_seconds=a.secs, max_events=a.events)
            tb, _ = cupti_kernel_us(lambda arr: run(arr, fb), copies, **kw)
            ta, _ = cupti_kernel_us(lambda arr: run(arr, tuned), copies, **kw)
            t1 = None
            if a.v1:
                t1, _ = cupti_kernel_us(v1, copies, **kw)
            rel = (run(copies[0], tuned).float() - ref).abs().max().item() / refmax
            torch.cuda.synchronize()
            rows.append((name, N, K, tag, M, wbytes, nv, nd, fb, tuned, tb, ta, t1, rel))
            del x, ref
        del copies, w, wref
        torch.cuda.empty_cache()

    hdr = (f"{'shape':<18}{'N':>7}{'K':>6}{'M':>3}{'MB':>7}{'before':>9}{'after':>9}"
           f"{'gain':>7}{'TB/s':>7}{'server':>8}{'relerr':>10}  after cfg")
    print(hdr)
    print("-" * len(hdr))
    for (name, N, K, tag, M, wb, nv, nd, fb, tn, tb, ta, t1, rel) in rows:
        g = GROUND_TRUTH.get((name, M))
        print(f"{name:<18}{N:>7}{K:>6}{M:>3}{wb/1e6:>7.1f}{tb*1e6:>9.2f}{ta*1e6:>9.2f}"
              f"{tb/ta:>6.2f}x{wb/ta/1e12:>7.2f}"
              f"{(f'{g:>8.1f}' if g else '       -')}{rel:>10.2e}  {tn}")
    if a.v1:
        print("\nv1 (pre-v2 kernel, BLOCK_K=256/512, no split-K) for reference:")
        for (name, N, K, tag, M, wb, nv, nd, fb, tn, tb, ta, t1, rel) in rows:
            print(f"  {name:<18} M={M:<3} v1 {t1*1e6:8.2f}us   today {tb*1e6:8.2f}us"
                  f"   tuned {ta*1e6:8.2f}us")

    by_key = {(r[0], r[4]): r for r in rows}
    print()
    for mode, (spec_w, n_draft) in MODES.items():
        for label, only in (("all shapes  ", None), ("primary only", SECONDARY)):
            tb = ta = 0.0
            calls = 0
            for name, N, K, tag, nv, nd in SHAPES:
                if only and name in only:
                    continue
                for M, c in ((spec_w, nv), (1, nd * (n_draft + 1))):
                    r = by_key.get((name, M))
                    if c == 0 or r is None:
                        continue
                    tb += r[10] * c
                    ta += r[11] * c
                    calls += c
                if calls and (name, spec_w) not in by_key and nv:
                    print(f"    (missing {name} M={spec_w})")
            if calls:
                print(f"{mode} {label}: spec_width={spec_w} draft_steps={n_draft} "
                      f"calls={calls}  before {tb*1e3:.3f} -> after {ta*1e3:.3f} ms/step "
                      f"(saved {(tb-ta)*1e3:.3f} ms, {100*(tb-ta)/tb:.1f}%)")


if __name__ == "__main__":
    main()
