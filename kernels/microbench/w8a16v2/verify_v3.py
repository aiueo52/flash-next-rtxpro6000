"""Correctness gate for the configs `_plan` actually selects, in the true layout.

For every (shape, M in {1, 4, 16}):
  * max|y - ref_fp32| / max|ref_fp32| <= 2e-2 against an fp32 reference;
  * two consecutive CUDA-graph replays with the same input are bit-identical
    (this is what proves the in-launch split-K counters self-reset -- a leaked
    counter makes the second replay read stale partials);
  * a replay with a changed input still matches its own fp32 reference;
  * the split-K counter array is all zero afterwards.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../python"))

import torch

from harness import require_idle_gpu
from shapes import SHAPES, make_weight

from sglang.srt.layers.quantization import w8a16_gemv as new

REL_TOL = 2e-2


def main():
    require_idle_gpu()
    torch.manual_seed(0)
    dev = "cuda"
    ok = True
    for name, N, K, tag, _, _ in SHAPES:
        w, scale, wref = make_weight(N, K, tag, dev)
        for M in (1, 4, 16):
            x = (torch.randn(M, K, device=dev) * 0.5).to(torch.bfloat16)
            ref = x.float() @ wref.t()
            refmax = ref.abs().max().item()
            fn = ((lambda a: new.w8a16_gemv(x, a, scale)) if tag == "fp8"
                  else (lambda a: new.bf16_gemv(x, a)))
            cfg = new._plan(M, N, K, w.stride(0) == 1, w.element_size(), new._num_sms(dev))
            rel = (fn(w).float() - ref).abs().max().item() / refmax
            torch.cuda.synchronize()

            # graph capture over a mutable input, then two identical replays
            xs = torch.empty_like(x)
            xs.copy_(x)
            gfn = ((lambda a: new.w8a16_gemv(xs, a, scale)) if tag == "fp8"
                   else (lambda a: new.bf16_gemv(xs, a)))
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    gfn(w)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                yg = gfn(w)
            g.replay()
            torch.cuda.synchronize()
            y1 = yg.clone()
            g.replay()
            torch.cuda.synchronize()
            y2 = yg.clone()
            bitwise = bool(torch.equal(y1, y2))
            rel_g = (y1.float() - ref).abs().max().item() / refmax

            x2 = (torch.randn(M, K, device=dev) * 0.5).to(torch.bfloat16)
            xs.copy_(x2)
            g.replay()
            torch.cuda.synchronize()
            r2 = x2.float() @ wref.t()
            rel2 = (yg.float() - r2).abs().max().item() / r2.abs().max().item()
            counters = int(new._workspace(dev)[1].abs().sum().item())

            good = (math.isfinite(rel) and rel <= REL_TOL and bitwise
                    and rel_g <= REL_TOL and rel2 <= REL_TOL and counters == 0)
            ok &= good
            print(f"  {'ok ' if good else 'FAIL'} {name:<18} M={M:<3} "
                  f"rel={rel:.2e} rel_graph={rel_g:.2e} rel_newinput={rel2:.2e} "
                  f"bitwise_replay={bitwise} counters={counters} cfg={cfg}")
            del g, x, ref
        del w, wref
        torch.cuda.empty_cache()
    print("VERIFY", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
