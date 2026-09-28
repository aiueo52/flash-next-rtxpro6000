"""Per-(shape, M-bucket) config sweep for the W8A16 / BF16 skinny GEMV, v3.

v3 differences from tune_w8a16v2.py:
  * the weight layout is the true server one ([N, K] contiguous, see shapes.py);
  * the metric is the CUPTI kernel-duration median, the same number the serving
    torch profile reports, not the CUDA-event wall time of a graph replay
    (which is ~25% higher because it includes the inter-kernel gaps);
  * correctness is checked against an fp32 reference (max|dy| / max|ref| <= 2e-2)
    and by requiring two consecutive graph replays to be bit-identical, which is
    what proves the in-launch split-K counters self-reset.

  tune_v3.py --shapes o_proj,shared_down --ms 1,16
  tune_v3.py --all --out /path/results.json

Sweep is two-phase to keep the compile bill bounded:
  A. (BLOCK_N, BLOCK_K, SPLITS) at warps=4 stages=3 (plus USE_DOT=False variants
     when M == 1), <= ~60 configs;
  B. the best few from A x num_warps {2,4,8} x num_stages {2,3,4}.
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../python"))

import torch
import triton

from harness import cupti_kernel_us, make_copies, n_copies, require_idle_gpu
from shapes import SHAPES, make_weight

from sglang.srt.layers.quantization import w8a16_gemv as new

SPLITS_ALL = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 32)
BN_ALL = (16, 32, 64, 128)
BK_ALL = (64, 128, 256, 512)
MAX_TILE = 32768  # fp8 elements per weight tile; 128x512 spills on 4 warps


def _fit(M, N, cfg):
    """new._fit, but the M=1/USE_DOT=False M_PAD is honoured (new._fit already does)."""
    return new._fit(M, N, cfg)


def phase_a(N, K, M, sms):
    """(BLOCK_N, BLOCK_K, SPLITS, USE_DOT, W_KN, 4, 3) candidates."""
    out = []
    for bn in BN_ALL:
        nb = triton.cdiv(N, bn)
        for bk in BK_ALL:
            if bn * bk > MAX_TILE:
                continue
            n_kb = triton.cdiv(K, bk)
            divisors = [s for s in SPLITS_ALL if s <= n_kb and n_kb % s == 0]
            others = [s for s in SPLITS_ALL if s <= n_kb and n_kb % s != 0]
            picks = {1}
            # the split count landing closest to ~2x and ~5x SM occupancy
            for target in (2 * sms, 5 * sms):
                pool = divisors or others
                best = min(pool, key=lambda s: abs(nb * s - target)) if pool else 1
                picks.add(best)
            # if every divisor still leaves the GPU under-occupied, take the
            # largest one that fits rather than nothing
            if nb * max(picks) < sms and divisors:
                picks.add(max(divisors))
            for sp in sorted(picks):
                out.append((bn, bk, sp, True, False, 4, 3))
    if M == 1:
        # broadcast-multiply path: M_PAD=1, so the split-K scratch is 16x smaller
        # and much deeper splits fit. Sweep it on a reduced (bn, bk) grid.
        for bn in (16, 32, 64, 128):
            nb = triton.cdiv(N, bn)
            for bk in (128, 256):
                if bn * bk > MAX_TILE:
                    continue
                n_kb = triton.cdiv(K, bk)
                divisors = [s for s in SPLITS_ALL if s <= n_kb and n_kb % s == 0]
                picks = {1}
                for target in (2 * sms, 5 * sms):
                    if divisors:
                        picks.add(min(divisors, key=lambda s: abs(nb * s - target)))
                for sp in sorted(picks):
                    out.append((bn, bk, sp, False, False, 4, 3))
    # dedupe after _fit, which may collapse a too-large split plan to SPLITS=1
    seen, uniq = set(), []
    for cfg in out:
        c = _fit(M, N, cfg)
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def phase_b(best, M, N, top=4):
    out, seen = [], set()
    for _, cfg, _ in best[:top]:
        bn, bk, sp, ud, wk, _, _ = cfg
        for nw in (2, 4, 8):
            if bn * bk // (32 * nw) < 4:  # < 4 elements per lane: pointless
                continue
            for ns in (2, 3, 4):
                c = _fit(M, N, (bn, bk, sp, ud, wk, nw, ns))
                if c not in seen:
                    seen.add(c)
                    out.append(c)
    return out


def measure(run, copies, cfg, ref, refmax, secs, events):
    y = run(copies[0], cfg)
    torch.cuda.synchronize()
    err = (y.float() - ref).abs().max().item() / refmax
    if not math.isfinite(err) or err > 2e-2:
        return None, err
    t, _ = cupti_kernel_us(lambda a: run(a, cfg), copies,
                           min_seconds=secs, spin_seconds=0.06, max_events=events)
    return t, err


def tune_one(name, N, K, tag, M, args, dev="cuda"):
    sms = new._num_sms(dev)
    w, scale, wref = make_weight(N, K, tag, dev)
    wbytes = N * K * (1 if tag == "fp8" else 2)
    copies = make_copies(w, n_copies(wbytes))
    x = (torch.randn(M, K, device=dev) * 0.5).to(torch.bfloat16)
    ref = x.float() @ wref.t()
    refmax = ref.abs().max().item()
    if tag == "fp8":
        run = lambda a, cfg: new.w8a16_gemv(x, a, scale, cfg=cfg)
    else:
        run = lambda a, cfg: new.bf16_gemv(x, a, cfg=cfg)

    fallback = new._plan(M, N, K, w.stride(0) == 1, w.element_size(), sms)
    res, seen = [], set()

    def sweep(cands, label):
        for i, cfg in enumerate(cands):
            if cfg in seen:
                continue
            seen.add(cfg)
            try:
                t, err = measure(run, copies, cfg, ref, refmax, args.secs, args.events)
            except Exception as e:
                print(f"  [{label} {i+1}/{len(cands)}] {cfg} {type(e).__name__}: "
                      f"{str(e)[:80]}", flush=True)
                continue
            if t is None:
                print(f"  [{label} {i+1}/{len(cands)}] {cfg} bad relerr {err:.2e}",
                      flush=True)
                continue
            res.append((t, cfg, err))
            if args.verbose:
                print(f"  [{label} {i+1}/{len(cands)}] {t*1e6:8.2f}us {cfg}", flush=True)

    cands_a = phase_a(N, K, M, sms)
    if fallback not in cands_a:
        cands_a.append(fallback)
    print(f"{name} N={N} K={K} M={M} {tag} {wbytes/1e6:.1f} MB, {len(copies)} copies, "
          f"phase A {len(cands_a)} cfgs, fallback={fallback}", flush=True)
    sweep(cands_a, "A")
    res.sort(key=lambda r: r[0])
    cands_b = phase_b(res, M, N, top=args.top_b)
    print(f"  phase B {len(cands_b)} cfgs from top {args.top_b}", flush=True)
    sweep(cands_b, "B")
    res.sort(key=lambda r: r[0])

    fb = next((t for t, c, _ in res if c == fallback), None)
    print(f"\n  {'us':>9}{'TB/s':>8}  cfg{' ' * 44}ctas   relerr   vs fallback")
    for t, cfg, err in res[: args.top]:
        sp = f"{fb/t:.2f}x" if fb else "-"
        print(f"  {t*1e6:>9.2f}{wbytes/t/1e12:>8.2f}  {str(cfg):<47}"
              f"{triton.cdiv(N, cfg[0]) * cfg[2]:>5}  {err:.2e}  {sp}")
    print(flush=True)

    out = {"name": name, "N": N, "K": K, "tag": tag, "M": M, "bytes": wbytes,
           "fallback": list(fallback), "fallback_us": fb * 1e6 if fb else None,
           "results": [[t * 1e6, list(c), e] for t, c, e in res[:20]]}
    del x, ref, copies, w, wref
    torch.cuda.empty_cache()
    return out


def confirm(name, N, K, tag, M, cands, args, dev="cuda"):
    """Short sweep of an explicit candidate list (used for the M=4 bucket)."""
    sms = new._num_sms(dev)
    w, scale, wref = make_weight(N, K, tag, dev)
    wbytes = N * K * (1 if tag == "fp8" else 2)
    copies = make_copies(w, n_copies(wbytes))
    x = (torch.randn(M, K, device=dev) * 0.5).to(torch.bfloat16)
    ref = x.float() @ wref.t()
    refmax = ref.abs().max().item()
    if tag == "fp8":
        run = lambda a, cfg: new.w8a16_gemv(x, a, scale, cfg=cfg)
    else:
        run = lambda a, cfg: new.bf16_gemv(x, a, cfg=cfg)
    fallback = new._plan(M, N, K, w.stride(0) == 1, w.element_size(), sms)
    res, seen = [], set()
    for cfg in list(cands) + [fallback]:
        cfg = _fit(M, N, tuple(cfg))
        if M > 1:
            cfg = (cfg[0], cfg[1], cfg[2], True, cfg[4], cfg[5], cfg[6])
            cfg = _fit(M, N, cfg)
        if cfg in seen:
            continue
        seen.add(cfg)
        try:
            t, err = measure(run, copies, cfg, ref, refmax, args.secs, args.events)
        except Exception as e:
            print(f"  {cfg} {type(e).__name__}: {str(e)[:80]}", flush=True)
            continue
        if t is None:
            continue
        res.append((t, cfg, err))
    res.sort(key=lambda r: r[0])
    fb = next((t for t, c, _ in res if c == fallback), None)
    print(f"{name} M={M} confirm: fallback={fallback} {fb*1e6 if fb else -1:.2f}us")
    for t, cfg, err in res[:6]:
        print(f"  {t*1e6:>9.2f}  {str(cfg):<47} {(fb/t if fb else 0):.2f}x  {err:.2e}")
    print(flush=True)
    out = {"name": name, "N": N, "K": K, "tag": tag, "M": M, "bytes": wbytes,
           "fallback": list(fallback), "fallback_us": fb * 1e6 if fb else None,
           "results": [[t * 1e6, list(c), e] for t, c, e in res[:20]]}
    del x, ref, copies, w, wref
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="")
    ap.add_argument("--ms", default="1,16")
    ap.add_argument("--confirm4", default="")
    ap.add_argument("--secs", type=float, default=0.12)
    ap.add_argument("--events", type=int, default=1500)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--top-b", type=int, default=4)
    ap.add_argument("--out", default="")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    require_idle_gpu()
    torch.manual_seed(0)
    p = torch.cuda.get_device_properties(0)
    print(f"device: {p.name}  SMs={p.multi_processor_count}  "
          f"L2={p.L2_cache_size / 1e6:.0f} MB\n", flush=True)

    by = {s[0]: s for s in SHAPES}
    want = a.shapes.split(",") if a.shapes else list(by)
    out = []
    for name in want:
        _, N, K, tag, _, _ = by[name]
        for M in [int(v) for v in a.ms.split(",") if v]:
            out.append(tune_one(name, N, K, tag, M, a))
    if a.confirm4:
        prev = json.load(open(a.confirm4))
        for name in want:
            _, N, K, tag, _, _ = by[name]
            cands = []
            for r in prev:
                if r["name"] == name:
                    cands += [c for _, c, _ in r["results"][:6]]
            if cands:
                out.append(confirm(name, N, K, tag, 4, cands, a))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
