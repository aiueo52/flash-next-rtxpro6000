"""RT1 bench: bit-exact check and CUDA-graph chain timing for the packed-key router.

  python -m rt1.route_bench --check            # production vs fast*, bit for bit (many rows); "server" is
                                               # the RT1 worktree's moe_router_softmax_fast.py itself
  python -m rt1.route_bench --time --rounds 7  # whole-chain wall per call, interleaved rounds

Timing variants (each captured as one CUDA graph of R back-to-back calls, PDL on like serve-fast.sh):
  gemv          router GEMV alone (bf16_gemv, the SGLANG_ROUTER_GEMV=1 path)
  gemv+prod     GEMV + production _router_triton_kernel (moe_fused_gate softmax)
  gemv+fast     GEMV + packed-key router (tl.sum routed sum)
  gemv+fastx    GEMV + packed-key router (explicit routed-sum order)
  gemv+fast32   GEMV + 32-bit packed-key router (bf16 logits, zero bias; route_fast32.py)
  gemv+fast32x  the same with the explicit routed-sum order
  gemv+consume  GEMV + a 1-warp PDL kernel that only reads the row (dependency floor)
(gemv+prod) - (gemv+consume) is what a faster separate router can win at most;
(gemv+consume) - (gemv) is what only folding the router into the GEMV can win.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import time

import torch

from rt1.route_fast import consume, route_softmax_fast
from rt1.route_fast32 import route_softmax_fast32

N_EXP, HIDDEN, TOPK = 512, 2560, 10
IMPLS = {  # tag -> (fn, explicit_sum)
    "fast": (route_softmax_fast, False),
    "fastx": (route_softmax_fast, True),
    "fast32": (route_softmax_fast32, False),
    "fast32x": (route_softmax_fast32, True),
}
# the server's own module (RT1 worktree), loaded by path so production stays first on sys.path
SERVER_MODULE = os.environ.get(
    "RT1_SERVER_MODULE",
    os.path.expanduser("~/tools/sglang-rt1/python/sglang/kernels/ops/moe/moe_router_softmax_fast.py"),
)


def _server_impl():
    spec = importlib.util.spec_from_file_location("rt1_server_router", SERVER_MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def route(scores, zero_bias, topk, renormalize, explicit_sum=False, out=None):
        assert mod.covered(scores, zero_bias, topk) and not explicit_sum and out is None
        return mod.route_softmax_fast(scores, zero_bias, topk, renormalize=renormalize)

    return route


if os.path.exists(SERVER_MODULE):
    IMPLS["server"] = (_server_impl(), False)


def _bad_rows(fw, fi, pw, pi):
    """rows whose ids differ or whose weights differ bit for bit (NaN == NaN: a NaN logit makes
    production's row sum, so every weight of the row, NaN; the payload is not compared)"""
    nan_both = torch.isnan(fw) & torch.isnan(pw)
    w_bad = ((fw.view(torch.int32) != pw.view(torch.int32)) & ~nan_both).any(dim=1)
    return (fi != pi).any(dim=1), w_bad


def _nan_rank(rows, dev):
    # fewer than 10 lanes above production's -1e30 NaN floor, so the picks reach the NaN lanes and their
    # bf16 neighbours -9.953e29 (above the floor) and -1.0003e30 (below); NaN in about half the rows
    vals = torch.tensor([-9.953e29, float("nan"), -1.0003e30, -3e38, float("-inf")], device=dev)
    u = torch.rand(rows, N_EXP, device=dev)
    idx = torch.bucketize(u, torch.tensor([0.01, 0.03, 0.33, 0.63], device=dev))
    no_nan = (torch.rand(rows, 1, device=dev) < 0.5) & (idx == 1)
    return torch.where(no_nan, torch.full_like(u, -1.0003e30), vals[idx])


def _prod_route(lg, zb):
    from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate

    return moe_fused_gate(lg, zb, TOPK, scoring_func="softmax", renormalize=True)


def check(args):
    torch.manual_seed(0)
    dev = "cuda"
    zb = torch.zeros(N_EXP, dtype=torch.float32, device=dev)
    rows = args.check_rows
    cases = {
        "randn*1": lambda: torch.randn(rows, N_EXP, device=dev),
        "randn*0.25": lambda: torch.randn(rows, N_EXP, device=dev) * 0.25,
        "randn*4": lambda: torch.randn(rows, N_EXP, device=dev) * 4,
        "randn*16": lambda: torch.randn(rows, N_EXP, device=dev) * 16,
        "ties(int*0.5)": lambda: torch.randint(-6, 6, (rows, N_EXP), device=dev).float() * 0.5,
        "ties(few)": lambda: torch.randint(0, 3, (rows, N_EXP), device=dev).float(),
        "all-equal": lambda: torch.full((rows, N_EXP), 0.75, device=dev),
        "zeros(+-0)": lambda: torch.where(torch.rand(rows, N_EXP, device=dev) < 0.5,
                                          torch.zeros(rows, N_EXP, device=dev),
                                          -torch.zeros(rows, N_EXP, device=dev)),
        "spiky": lambda: torch.randn(rows, N_EXP, device=dev)
        + 30 * (torch.rand(rows, N_EXP, device=dev) < 0.01).float(),
        # NaN logits (production floors them to -1e30 for the ranking), some rows only
        "nan": lambda: torch.where((torch.rand(rows, N_EXP, device=dev) < 0.02)
                                   & (torch.rand(rows, 1, device=dev) < 0.5),
                                   torch.full((rows, N_EXP), float("nan"), device=dev),
                                   torch.randn(rows, N_EXP, device=dev)),
        # around the -1e30 floor: bf16 -9.95e29 / -1.0003e30 neighbours, -inf, NaN
        "nan-floor": lambda: torch.tensor([-9.953e29, -1.0003e30, float("-inf"), float("nan"), -3e38, -1e29],
                                          device=dev)[torch.randint(0, 6, (rows, N_EXP), device=dev)],
        "nan-rank": lambda: _nan_rank(rows, dev),
    }
    # real-shaped logits: the router GEMV on random hidden states and weights
    from sglang.srt.layers.quantization.w8a16_gemv import bf16_gemv

    w = (torch.randn(N_EXP, HIDDEN, device=dev) * 0.02).bfloat16()
    gemv_rows = []
    for _ in range(max(1, rows // 16)):
        x = torch.randn(16, HIDDEN, device=dev).bfloat16()
        gemv_rows.append(bf16_gemv(x, w))
    gl = torch.cat(gemv_rows)
    cases["gemv(x,W)"] = lambda: gl.float()

    ok_all = True
    for name, gen in cases.items():
        lg = gen().bfloat16().contiguous()
        pw, pi = _prod_route(lg, zb)
        res = {"case": name, "rows": lg.shape[0]}
        for tag, (fn, ex) in IMPLS.items():
            fw, fi = fn(lg, zb, TOPK, True, explicit_sum=ex)
            torch.cuda.synchronize()
            id_bad, w_bad = _bad_rows(fw, fi, pw, pi)
            res[tag] = {"ids_bad_rows": int(id_bad.sum()), "w_bad_rows": int(w_bad.sum())}
            if id_bad.any() or w_bad.any():
                ok_all = False
                r = int(torch.nonzero(id_bad | w_bad)[0, 0])
                res[tag]["first_bad_row"] = r
                res[tag]["prod_ids"] = pi[r].tolist()
                res[tag]["fast_ids"] = fi[r].tolist()
                res[tag]["max_abs_w"] = float((fw[r] - pw[r]).abs().max())
        print(json.dumps(res), flush=True)
    # the server's own launch shapes: M is a Triton specialization key (1 -> constexpr,
    # 16 -> divisible by 16, 4 -> plain), so check each one separately as well
    for M in (1, 4, 16):
        bad = {tag: 0 for tag in IMPLS}
        n = 0
        for it in range(args.check_iters):
            scale = (0.25, 1.0, 4.0)[it % 3]
            lg = (torch.randn(M, N_EXP, device=dev) * scale).bfloat16()
            if it % 4 == 3:
                lg = (torch.randint(-4, 4, (M, N_EXP), device=dev).float() * 0.5).bfloat16()
            pw, pi = _prod_route(lg, zb)
            for tag, (fn, ex) in IMPLS.items():
                fw, fi = fn(lg, zb, TOPK, True, explicit_sum=ex)
                id_bad, w_bad = _bad_rows(fw, fi, pw, pi)
                bad[tag] += int((id_bad | w_bad).sum())
            n += M
        torch.cuda.synchronize()
        if any(bad.values()):
            ok_all = False
        print(json.dumps({"case": f"server-shape M={M}", "rows": n, "bad_rows": bad}), flush=True)
    print(json.dumps({"check": "PASS" if ok_all else "FAIL"}), flush=True)
    return ok_all


def time_graph(g, ncalls, seconds):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    s.record()
    g.replay()
    e.record()
    torch.cuda.synchronize()
    per = max(s.elapsed_time(e) / 1e3, 1e-6)
    iters = max(10, int(seconds / per))
    s.record()
    for _ in range(iters):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / 1e3 / (iters * ncalls) * 1e6


def timing(args):
    from sglang.srt.layers.quantization.w8a16_gemv import bf16_gemv, prealloc

    dev = torch.device("cuda")
    prealloc(dev)
    torch.manual_seed(1)
    w = (torch.randn(N_EXP, HIDDEN, device=dev) * 0.02).bfloat16()
    zb = torch.zeros(N_EXP, dtype=torch.float32, device=dev)
    R = args.reps
    graphs = {}
    keep = []
    for M in args.M:
        x = torch.randn(M, HIDDEN, device=dev).bfloat16()
        lg = torch.empty(M, N_EXP, dtype=torch.bfloat16, device=dev)
        fw = torch.empty(M, TOPK, dtype=torch.float32, device=dev)
        fi = torch.empty(M, TOPK, dtype=torch.int32, device=dev)
        cbuf = torch.empty(M, dtype=torch.float32, device=dev)
        keep.append((x, lg, fw, fi, cbuf))  # the graphs hold these addresses

        def mk(variant):
            def call():
                bf16_gemv(x, w, out=lg)
                if variant == "gemv+prod":
                    _prod_route(lg, zb)
                elif variant == "gemv+fast":
                    route_softmax_fast(lg, zb, TOPK, True, out=(fw, fi))
                elif variant == "gemv+fastx":
                    route_softmax_fast(lg, zb, TOPK, True, explicit_sum=True, out=(fw, fi))
                elif variant == "gemv+fast32":
                    route_softmax_fast32(lg, zb, TOPK, True, out=(fw, fi))
                elif variant == "gemv+fast32x":
                    route_softmax_fast32(lg, zb, TOPK, True, explicit_sum=True, out=(fw, fi))
                elif variant == "gemv+consume":
                    consume(lg, cbuf)

            return call

        for v in args.variants:
            f = mk(v)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    f()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(R):
                    f()
            torch.cuda.synchronize()
            graphs[(M, v)] = g
    res = {k: [] for k in graphs}
    for rnd in range(args.rounds):
        keys = list(graphs)
        if rnd % 2:
            keys.reverse()
        for k in keys:
            res[k].append(time_graph(graphs[k], R, args.seconds))
    out = []
    for M in args.M:
        base = statistics.median(res[(M, "gemv")]) if (M, "gemv") in res else None
        for v in args.variants:
            vals = res[(M, v)]
            med = statistics.median(vals)
            rec = {"M": M, "variant": v, "median_us": round(med, 3),
                   "min_us": round(min(vals), 3), "max_us": round(max(vals), 3)}
            if base is not None:
                rec["minus_gemv_us"] = round(med - base, 3)
            out.append(rec)
            print(json.dumps(rec), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--check-rows", type=int, default=65536)
    ap.add_argument("--check-iters", type=int, default=200)
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--M", type=lambda s: [int(v) for v in s.split(",")], default=[1, 4, 16])
    ap.add_argument("--variants", type=lambda s: s.split(","),
                    default=["gemv", "gemv+prod", "gemv+fast", "gemv+fastx", "gemv+fast32", "gemv+fast32x",
                             "gemv+consume"])
    ap.add_argument("--reps", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--seconds", type=float, default=0.3)
    args = ap.parse_args()
    t0 = time.time()
    print(json.dumps({"start": time.strftime("%T"), "device": torch.cuda.get_device_name()}), flush=True)
    ok = True
    if args.check:
        ok = check(args)
    if args.time:
        timing(args)
    print(json.dumps({"done": time.strftime("%T"), "secs": round(time.time() - t0, 1)}), flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
