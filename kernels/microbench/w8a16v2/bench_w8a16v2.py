"""A/B benchmark for the W8A16 skinny GEMV: frozen pre-optimization kernel vs. current.

  bench_w8a16v2.py --before            # "before" table only, vs. the serving profile
  bench_w8a16v2.py                     # before/after table + ms/step model
  bench_w8a16v2.py --selftest          # split-K correctness / graph-replay / counter checks
  bench_w8a16v2.py --shapes o_proj,shared_gate_up --m 16

Every row is a CUDA-graph replay over >= 24 distinct weight copies (working set several
times the 128 MB L2), replayed for >= --secs of continuous load after a spin-up, so the
SMs are at their boost clock; --clocks prints the sampled SM clock per row.
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../python"))

import torch

import baseline_v1 as base
from harness import make_copies, n_copies, require_idle_gpu, time_graph
from shapes import DEFAULT_MS, GROUND_TRUTH, MODES, SHAPE_MS, SHAPES, make_weight

from sglang.srt.layers.quantization import w8a16_gemv as new


def _fn(mod, tag, x, scale):
    if tag == "fp8":
        return lambda a: mod.w8a16_gemv(x, a, scale)
    return lambda a: mod.bf16_gemv(x, a)


def bench_one(name, N, K, tag, Ms, args, which):
    dev = "cuda"
    w, scale, wref = make_weight(N, K, tag, dev)
    wbytes = N * K * (1 if tag == "fp8" else 2)
    copies = make_copies(w, n_copies(wbytes))
    rows = []
    for M in Ms:
        x = (torch.randn(M, K, device=dev) * 0.5).to(torch.bfloat16)
        ref = x.float() @ wref.t()
        d = {"ncopies": len(copies)}
        for label, mod in (("before", base), ("after", new)):
            if label not in which:
                continue
            fn = _fn(mod, tag, x, scale)
            y = fn(copies[0])
            d[label + "_err"] = (y.float() - ref).abs().max().item()
            d[label], d[label + "_clk"] = time_graph(
                fn, copies, min_seconds=args.secs, clocks=args.clocks)
            # Draft steps re-read one MTP layer's weights 3-14x per decode step, so
            # those calls run L2-warm in the server; same graph shape, one weight.
            d[label + "_warm"], _ = time_graph(
                fn, [copies[0]] * len(copies), min_seconds=args.secs)
        cfg = new._plan(M, N, K, w.stride(0) == 1, w.element_size(),
                        new._num_sms(dev)) if "after" in which else None
        rows.append((name, N, K, tag, M, wbytes, d, cfg))
        del x, ref
    del copies, w, wref
    torch.cuda.empty_cache()
    return rows


def fmt_before(rows, clocks):
    hdr = (f"{'shape':<18}{'N':>7}{'K':>7}{'M':>3}{'grid':>6}{'MB':>8}{'cp':>4}"
           f"{'us':>9}{'TB/s':>7}{'warm us':>9}{'profile':>9}{'ratio':>7}"
           + ("   sm MHz" if clocks else ""))
    print(hdr)
    print("-" * len(hdr))
    for name, N, K, tag, M, wbytes, d, _ in rows:
        gt = GROUND_TRUTH.get((name, M))
        t = d["before"]
        ratio = f"{t * 1e6 / gt:>6.2f}x" if gt else "     -"
        clk = f"   {d['before_clk'][0]}-{d['before_clk'][1]}" if clocks else ""
        print(f"{name:<18}{N:>7}{K:>7}{M:>3}{-(-N // 32):>6}{wbytes / 1e6:>8.1f}"
              f"{d['ncopies']:>4}{t * 1e6:>9.2f}{wbytes / t / 1e12:>7.2f}"
              f"{d['before_warm'] * 1e6:>9.2f}{(f'{gt:>9.1f}' if gt else '        -')}"
              f"{ratio}{clk}")


def fmt_ab(rows):
    hdr = (f"{'shape':<18}{'N':>7}{'K':>7}{'M':>3}{'before':>9}{'after':>9}{'gain':>7}"
           f"{'TB/s b':>8}{'TB/s a':>8}{'warm a':>8}{'err b':>10}{'err a':>10}  config")
    print(hdr)
    print("-" * len(hdr))
    for name, N, K, tag, M, wbytes, d, cfg in rows:
        b, a = d["before"], d["after"]
        print(f"{name:<18}{N:>7}{K:>7}{M:>3}{b * 1e6:>9.2f}{a * 1e6:>9.2f}{b / a:>6.2f}x"
              f"{wbytes / b / 1e12:>8.2f}{wbytes / a / 1e12:>8.2f}"
              f"{d['after_warm'] * 1e6:>8.2f}{d['before_err']:>10.2e}{d['after_err']:>10.2e}"
              f"  {cfg}")


def savings(rows):
    """ms/step from the call model in shapes.py; "mixed" runs the draft calls L2-warm."""
    by = {(r[0], r[4]): r[6] for r in rows}
    print()
    for mode, (spec_w, n_draft) in MODES.items():
        for how, dk in (("cold ", ""), ("mixed", "_warm")):
            tb = ta = 0.0
            calls = 0
            for name, nv, nd in ((s[0], s[4], s[5]) for s in SHAPES):
                for M, c, key in ((spec_w, nv, ""), (1, nd * n_draft, dk)):
                    d = by.get((name, M))
                    if c == 0 or d is None or "after" not in d:
                        continue
                    tb += d["before" + key] * c
                    ta += d["after" + key] * c
                    calls += c
            if calls:
                print(f"{mode} {how}: spec_width={spec_w} draft_steps={n_draft} "
                      f"calls={calls}  before {tb * 1e3:.3f} -> after {ta * 1e3:.3f} ms/step "
                      f"(saved {(tb - ta) * 1e3:.3f}, {100 * (tb - ta) / tb:.1f}%)")


def selftest():
    """Split-K numerics, CUDA-graph capture/replay with changing inputs, counter reset."""
    dev = "cuda"
    ok = True

    # 1. Triton scalar atomic_add must fire exactly once per CTA and broadcast the old
    #    value to every thread; the whole fixup depends on it.
    import triton
    import triton.language as tl

    @triton.jit
    def _probe(cnt, out, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        old = tl.atomic_add(cnt + 0, 1, sem="acq_rel", scope="gpu")
        tl.store(out + pid * BLOCK + tl.arange(0, BLOCK), tl.zeros((BLOCK,), tl.int32) + old)

    cnt = torch.zeros(1, dtype=torch.int32, device=dev)
    out = torch.zeros(64 * 256, dtype=torch.int32, device=dev)
    _probe[(64,)](cnt, out, BLOCK=256, num_warps=4)
    torch.cuda.synchronize()
    per_cta = out.view(64, 256)
    uniform = bool((per_cta == per_cta[:, :1]).all())
    print(f"  scalar atomic: counter={cnt.item()} (want 64)  broadcast_uniform={uniform}  "
          f"distinct_old={len(set(per_cta[:, 0].tolist()))} (want 64)")
    ok &= cnt.item() == 64 and uniform and len(set(per_cta[:, 0].tolist())) == 64

    for name, N, K, tag, _, _ in SHAPES:
        w, scale, wref = make_weight(N, K, tag, dev)
        for M in sorted(set(SHAPE_MS.get(name, DEFAULT_MS)) | {1, 4, 16}):
            x = (torch.randn(M, K, device=dev) * 0.5).to(torch.bfloat16)
            ref = x.float() @ wref.t()
            fn = _fn(new, tag, x, scale)
            fb = _fn(base, tag, x, scale)
            ya, yb = fn(w), fb(w)
            ea = (ya.float() - ref).abs().max().item()
            eb = (yb.float() - ref).abs().max().item()
            # One bf16 ulp at the output magnitude (7 explicit mantissa bits).
            mx = ref.abs().max().item()
            ulp = 2.0 ** (math.floor(math.log2(mx)) - 7) if mx > 0 else 0.0
            cfg = new._plan(M, N, K, w.stride(0) == 1, w.element_size(), new._num_sms(dev))
            good = ea <= eb + 2 * ulp
            ok &= good

            # Graph capture + replay with changing inputs, then counters back to zero.
            xs = torch.empty_like(x)
            xs.copy_(x)
            gfn = _fn(new, tag, xs, scale)
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    gfn(w)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            with torch.cuda.graph(g):
                yg = gfn(w)
            replay_ok = True
            for seed in range(3):
                x2 = (torch.randn(M, K, device=dev, generator=torch.Generator(
                    device=dev).manual_seed(seed)) * 0.5).to(torch.bfloat16)
                xs.copy_(x2)
                g.replay()
                torch.cuda.synchronize()
                r2 = x2.float() @ wref.t()
                e2 = (yg.float() - r2).abs().max().item()
                m2 = r2.abs().max().item()
                u2 = 2.0 ** (math.floor(math.log2(m2)) - 7) if m2 > 0 else 0.0
                replay_ok &= e2 <= max(eb, 2 * u2) + 2 * u2
            cnt_zero = int(new._workspace(x.device)[1].abs().sum().item()) == 0
            # Stress the split-K fixup: many back-to-back launches sharing one scratch,
            # which is where a missing fence or a leaked counter would show up.
            stress_ok = True
            if cfg[2] > 1:
                gs = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gs):
                    ys = [gfn(w) for _ in range(32)]
                for _ in range(50):
                    gs.replay()
                torch.cuda.synchronize()
                r3 = xs.float() @ wref.t()
                m3 = r3.abs().max().item()
                u3 = 2.0 ** (math.floor(math.log2(m3)) - 7) if m3 > 0 else 0.0
                stress_ok = all((t.float() - r3).abs().max().item() <= max(eb, 2 * u3) + 2 * u3
                                for t in ys)
                stress_ok &= int(new._workspace(x.device)[1].abs().sum().item()) == 0
                del gs, ys
            ok &= replay_ok and cnt_zero and stress_ok
            del g
            flag = "ok " if (good and replay_ok and cnt_zero and stress_ok) else "FAIL"
            print(f"  {flag} {name:<18} M={M:<3} err_after={ea:.3e} err_before={eb:.3e} "
                  f"2ulp={2*ulp:.2e} replay={replay_ok} counters_zero={cnt_zero} "
                  f"stress={stress_ok} cfg={cfg}")
        del w, wref
        torch.cuda.empty_cache()
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="")
    ap.add_argument("--m", default="")
    ap.add_argument("--secs", type=float, default=0.6)
    ap.add_argument("--before", action="store_true")
    ap.add_argument("--after", action="store_true")
    ap.add_argument("--clocks", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    require_idle_gpu()
    torch.manual_seed(0)
    p = torch.cuda.get_device_properties(0)
    print(f"device: {p.name}  SMs={p.multi_processor_count}  "
          f"L2={p.L2_cache_size / 1e6:.0f} MB  vram={p.total_memory / 1e9:.0f} GB")
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    which = {"before"} if a.before else ({"after"} if a.after else {"before", "after"})
    want = set(a.shapes.split(",")) if a.shapes else None
    forced = [int(v) for v in a.m.split(",")] if a.m else None
    rows = []
    for name, N, K, tag, _, _ in SHAPES:
        if want and name not in want:
            continue
        rows += bench_one(name, N, K, tag, forced or list(SHAPE_MS.get(name, DEFAULT_MS)),
                          a, which)
    print()
    if which == {"before"}:
        fmt_before(rows, a.clocks)
    else:
        fmt_ab(rows)
        if not want and not forced:
            savings(rows)
