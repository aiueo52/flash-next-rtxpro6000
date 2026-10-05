"""FC-glue: the GDN layer front (HC mix -> combine gate -> in_proj qkvz || ba -> out_proj),
with the real fork kernels, under CUDA-graph replay, in several stream arrangements.

Production (2026-10-01, serve-fast.sh): K0/K1/K2 (hc_norm_mix2, fp8) -> hc_combine_gate (GATE_EARLY=2)
-> qkvz fp8 GEMV on the main stream || in_proj_ba bf16 GEMV [3 CTAs] on the alt stream -> join.
In the 09-07 traces ba starts ~26 us after qkvz (its 3 CTAs wait for qkvz's to drain) and sticks
out 1.5-3.4 us past it; the gate runs alone for ~1.6 us between K2 and qkvz.

Variants (per iteration; all bit-identical -- same kernels, same inputs, only stream placement):
  prod        main: K0 K1 K2 gate qkvz          alt: ba (fork after gate)      join -> out_proj
  swap        main: K0 K1 K2 gate ba            alt: qkvz (fork after gate)    join -> out_proj
  gside       main: K0 K1 K2 qkvz               altG: gate (fork after K0)  altB: ba (fork after K2)
  gside_swap  main: K0 K1 K2 ba                 altG: gate (fork after K0)  altQ: qkvz (fork after K2)
  serial      main: K0 K1 K2 gate qkvz ba       (no alt stream)
  no_ba       main: K0 K1 K2 gate qkvz          (lower bound: ba deleted)
  bare        main: K0 K1 K2 qkvz               (lower bound: ba and gate deleted)

Run: . bench/megakernel/env.sh; SGLANG_TRITON_PDL=1 flock -w 28800 ~/.gpu.lock $PY bench/fc_glue/bench_gdn_front.py
"""
from __future__ import annotations

import argparse, os, statistics, sys

os.environ.setdefault("SGLANG_TRITON_PDL", "1")
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "megakernel"))
from harness import ab_rounds, capture, emit, gpu_note, make_copies  # noqa: E402

HS, HC, K, LR = 2560, 4, 10240, 320
N_QKVZ, N_BA, K_OUT = 16384, 96, 6144
EPS = 1e-6


def build(dev, M, copies):
    from sglang.srt.layers.hc_mix2_triton import quantize_hc_mix2_weights_fp8
    g = torch.Generator(device=dev).manual_seed(0)
    r = lambda *s, sc=1.0: (torch.randn(*s, generator=g, device=dev, dtype=torch.bfloat16) * sc)
    wd8, sd, wu8, su = quantize_hc_mix2_weights_fp8(r(LR, K, sc=1 / 32), r(K, LR, sc=1 / 32))
    wq = r(N_QKVZ, HS, sc=0.05).to(torch.float8_e4m3fn)
    wo = r(HS, K_OUT, sc=0.05).to(torch.float8_e4m3fn)
    t = dict(
        hyper=make_copies(r(M, K, sc=1 / 8), copies),
        norm_w=r(K, sc=1 / 16), inject=r(HC, K, sc=1 / 64),
        wd=make_copies(wd8, copies), wu=make_copies(wu8, copies), sd=sd, su=su,
        wq=make_copies(wq, copies), sq=torch.rand(N_QKVZ, device=dev) * 0.01,
        wba=make_copies(r(N_BA, HS, sc=0.02), copies),
        wo=make_copies(wo, copies), so=torch.rand(HS, device=dev) * 0.01,
        part=[torch.empty((M, 8, HC), dtype=torch.float32, device=dev) for _ in range(copies)],
        yq=[torch.empty((M, N_QKVZ), dtype=torch.bfloat16, device=dev) for _ in range(copies)],
        yba=[torch.empty((M, N_BA), dtype=torch.bfloat16, device=dev) for _ in range(copies)],
        yo=[torch.empty((M, HS), dtype=torch.bfloat16, device=dev) for _ in range(copies)],
    )
    return t


class _DownProxy:
    """Stands in for hc_mix2_triton._hc_down_kernel so a hook can fork a side stream between
    K0 and K1 (the gate only needs K0's `normed`, argument 4)."""

    def __init__(self, real):
        self.real, self.hook = real, None

    def __getitem__(self, grid):
        launch = self.real[grid]

        def run(*a, **kw):
            if self.hook is not None:
                self.hook(a[4])
            return launch(*a, **kw)
        return run


def make_variants(t, streams):
    import sglang.srt.layers.hc_mix2_triton as m2
    from sglang.kernels.ops.elementwise.hc_combine import hc_combine_gate
    from sglang.srt.layers.quantization.w8a16_gemv import bf16_gemv, w8a16_gemv
    if not isinstance(m2._hc_down_kernel, _DownProxy):
        m2._hc_down_kernel = _DownProxy(m2._hc_down_kernel)
    proxy = m2._hc_down_kernel
    altG, altB, altQ = streams
    idx = list(range(len(t["hyper"])))

    def mix(i, hook=None):
        proxy.hook = hook
        try:
            return m2.hc_norm_mix2(t["hyper"][i], t["norm_w"], EPS, t["wd"][i], t["wu"][i], HC, HS,
                                   s_down=t["sd"], s_up=t["su"])
        finally:
            proxy.hook = None

    gate = lambda i, normed: hc_combine_gate(normed, t["inject"], HC, HS, partials=t["part"][i])
    qkvz = lambda i, x: w8a16_gemv(x, t["wq"][i], t["sq"], out=t["yq"][i])
    ba = lambda i, x: bf16_gemv(x, t["wba"][i], out=t["yba"][i])
    outp = lambda i: w8a16_gemv(t["yq"][i][:, :K_OUT], t["wo"][i], t["so"], out=t["yo"][i])

    def on(stream, fn):
        cur = torch.cuda.current_stream()
        stream.wait_stream(cur)
        with torch.cuda.stream(stream):
            fn()

    def join(*ss):
        cur = torch.cuda.current_stream()
        for s in ss:
            cur.wait_stream(s)

    def prod(i):
        x, n = mix(i)
        gate(i, n)
        # production order (_forward_input_proj_direct): alt waits, qkvz on main, then ba on alt
        cur = torch.cuda.current_stream(); altB.wait_stream(cur)
        qkvz(i, x)
        with torch.cuda.stream(altB):
            ba(i, x)
        join(altB); outp(i)

    def swap(i):
        x, n = mix(i)
        gate(i, n)
        on(altQ, lambda: qkvz(i, x))
        ba(i, x)
        join(altQ); outp(i)

    def gside(i):
        x, n = mix(i, hook=lambda nrm: on(altG, lambda: gate(i, nrm)))
        cur = torch.cuda.current_stream(); altB.wait_stream(cur)
        qkvz(i, x)
        with torch.cuda.stream(altB):
            ba(i, x)
        join(altB, altG); outp(i)

    def gside_swap(i):
        x, n = mix(i, hook=lambda nrm: on(altG, lambda: gate(i, nrm)))
        on(altQ, lambda: qkvz(i, x))
        ba(i, x)
        join(altQ, altG); outp(i)

    def serial(i):
        x, n = mix(i); gate(i, n); qkvz(i, x); ba(i, x); outp(i)

    def no_ba(i):
        x, n = mix(i); gate(i, n); qkvz(i, x); outp(i)

    def bare(i):
        x, n = mix(i); qkvz(i, x); outp(i)

    return {k: (v, idx) for k, v in dict(prod=prod, swap=swap, gside=gside, gside_swap=gside_swap,
                                          serial=serial, no_ba=no_ba, bare=bare).items()}


def timeline(call, args, pick=3):
    """One replay under the profiler: kernel starts relative to the pick-th iteration's K0."""
    from torch.profiler import ProfilerActivity, profile
    g = capture(call, args)
    for _ in range(20):
        g.replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        g.replay(); torch.cuda.synchronize()
    import json, tempfile
    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        prof.export_chrome_trace(f.name)
        tr = json.load(open(f.name))
    ks = sorted((e["ts"], e["dur"], e["name"], e.get("args", {}).get("stream", -1))
                for e in tr["traceEvents"] if e.get("cat") == "kernel")
    k0 = [k for k in ks if "_hc_branch_stats" in k[2]]
    t0, t1 = k0[pick][0], k0[pick + 1][0]
    rows = [(round(s - t0, 2), round(d, 2), n[:40], sid) for s, d, n, sid in ks if t0 <= s < t1]
    del g
    return rows, t1 - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="4,16")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--copies", type=int, default=10)
    ap.add_argument("--min-seconds", type=float, default=0.4)
    ap.add_argument("--out", default="bench/fc_glue/results/gdn_front.json")
    a = ap.parse_args()
    dev = torch.device("cuda")
    note = gpu_note(); print("[gpu]", note)
    streams = (torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream())
    payload = {"note": note, "results": {}}
    for M in [int(v) for v in a.rows.split(",")]:
        t = build(dev, M, a.copies)
        var = make_variants(t, streams)
        wall = ab_rounds(var, rounds=a.rounds, min_seconds=a.min_seconds)
        base = wall["prod"]["median"]
        print(f"\n=== M={M}: graph wall per iteration (us), median of {a.rounds} interleaved rounds ===")
        for k, r in wall.items():
            d = [x - y for x, y in zip(r["rounds"], wall["prod"]["rounds"])]
            print(f"  {k:<11s} {r['median']:8.2f}   vs prod {r['median'] - base:+6.2f}   paired-median {statistics.median(d):+6.2f}"
                  f"   rounds {r['rounds']}")
        tl = {}
        for k in ("prod", "swap", "gside", "gside_swap"):
            rows, per = timeline(*var[k])
            tl[k] = rows
            print(f"\n  [{k}] one iteration = {per:.2f} us")
            for r in rows:
                print(f"     {r[0]:7.2f} {r[1]:6.2f}  s{r[3]}  {r[2]}")
        payload["results"][f"M{M}"] = {"wall": wall, "timeline": tl}
        del t, var
        torch.cuda.empty_cache()
    emit(a.out, payload)


if __name__ == "__main__":
    main()
