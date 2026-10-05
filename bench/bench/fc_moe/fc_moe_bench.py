"""fc-moe (2026-10-01): MoE-chain fixed-cost microbench on the production FlashInfer build.

One process = one env variant (the csrc reads FLASHINFER_MOE_* once).  Routing is replayed
from the recorded census (pre-prune ids + weights), so P2's in-prologue prune sees real
weights.  Per-kernel durations come from a chrome trace of CUDA-graph replays; wall/call
from event timing of the same graph.

  python -m fc_moe.fc_moe_bench --T 1,4,8,16 --label prod
  python -m fc_moe.fc_moe_bench --T 4 --tactics 0-30:40-64 --label sweep4
  python -m fc_moe.fc_moe_bench --T 4 --side-mb 8 --label side8
  python -m fc_moe.fc_moe_bench --T 1,4,16 --flush-mb 256 --label flush256
"""
from __future__ import annotations

import argparse, gzip, json, os, statistics as st, sys, tempfile, time

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/tools/flash-next-bench/bench"))
from moe_smallm import model_shapes as MS  # noqa: E402
from moe_smallm.harness import capture  # noqa: E402
from moe_smallm import runners as RN  # noqa: E402
from moe_smallm.runners import (build_inputs, ensure_sglang_imports, force_tactics,  # noqa: E402
                                get, load_autotune_cache)
from moe_smallm.weights import load_layer, prepare_runtime_scales  # noqa: E402

CENSUS = os.path.expanduser("~/tools/flash-next-bench/runs/census-{w}-code-edit.npz")
ROT = 16


def census_routings(T: int, n: int, offset: int = 0):
    """T=4 from the W4 census, T=16 from W16, T=8 = first 8 rows of W16 calls, T=1 = row 0."""
    src_T = 4 if T <= 4 else 16
    z = np.load(CENSUS.format(w="w4" if src_T == 4 else "w16"))
    rows = np.nonzero(z["T"] == src_T)[0]
    step = max(1, len(rows) // (n * 7))
    ids, ws = [], []
    for j in range(n):
        r = rows[(offset + j * step) % len(rows)]
        k = int(z["k"][r])
        a = z["ids"][r][: src_T * k].reshape(src_T, k)[:T].astype(np.int32)
        w = z["w"][r][: src_T * k].reshape(src_T, k)[:T].astype(np.float32)
        ids.append(torch.from_numpy(a.copy()))
        ws.append(torch.from_numpy(w.copy()))
    return ids, ws


def classify(name: str, grid):
    if "fusedBuildExpertMapsSortF" in name:
        return "prologue"
    if "device_kernel" in name and "cutlass" in name and list(grid)[1] == 188:
        return "gemm"
    if "doActivation" in name:
        return "act"
    if "finalizeMoeRouting" in name:
        return "finalize"
    if "expandInputRows" in name:
        return "expand"
    if "memset" in name.lower() or name == "Memset (Device)":
        return "memset"
    if "side_read" in name or "reduce_kernel" in name or "Reduce" in name:
        return "side"
    return "other:" + name[:50]


def trace_stats(path: str, stride: int = 1):
    """stride=2 keeps every second MoE call: the measured one after a --dummy-call."""
    ev = json.load(gzip.open(path) if path.endswith(".gz") else open(path))["traceEvents"]
    K = sorted([e for e in ev if e.get("cat") in ("kernel", "gpu_memset", "gpu_memcpy")],
               key=lambda e: e["ts"])
    per = {}
    calls, cur = [], None
    for k in K:
        c = classify(k["name"], k.get("args", {}).get("grid", [0, 0, 0]))
        if c == "prologue":
            cur = {"prologue": k, "gemms": [], "start": k["ts"]}
            calls.append(cur)
        elif cur is not None and c == "gemm":
            cur["gemms"].append(k)
        elif cur is not None:
            cur.setdefault(c, k)
    n_all = len(calls)
    calls = calls[stride - 1::stride]
    out = {}
    for role in ("prologue", "act", "expand", "memset", "finalize"):
        v = [c[role]["dur"] for c in calls if role in c]
        if v:
            out[role] = st.median(v)
            if role == "prologue" and len(v) >= 10:  # the in-server prologue is bimodal (fast/slow)
                q = st.quantiles(v, n=10)
                out["prologue_p10"], out["prologue_p90"] = q[0], q[-1]
    g1 = [c["gemms"][0]["dur"] for c in calls if len(c["gemms"]) >= 2]
    g2 = [c["gemms"][1]["dur"] for c in calls if len(c["gemms"]) >= 2]
    if g1:
        out["gemm1"], out["gemm2"] = st.median(g1), st.median(g2)
    span = [c["gemms"][-1]["ts"] + c["gemms"][-1]["dur"] - c["start"] for c in calls if c["gemms"]]
    if span:
        out["chain_span"] = st.median(span)
    return out, n_all


def time_replays(g, ncalls, seconds=0.5):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(20):
        g.replay()
    torch.cuda.synchronize()
    s.record(); g.replay(); e.record(); torch.cuda.synchronize()
    per = max(s.elapsed_time(e) / 1e3, 1e-6)
    iters = max(10, int(seconds / per))
    s.record()
    for _ in range(iters):
        g.replay()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / 1e3 / (iters * ncalls) * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", default="1,4,8,16")
    ap.add_argument("--label", default="run")
    ap.add_argument("--autotune-cache", default=os.path.expanduser(
        "~/.cache/sglang/flashinfer/autotune/0.6.17/sm120/594f7c285c817365/rank_tp0_pp0_dp0.json"))
    ap.add_argument("--tactics", default=None, help="g1lo-g1hi:g2lo-g2hi sweep (forced)")
    ap.add_argument("--force", default=None, help="g1,g2 forced tactics")
    ap.add_argument("--side-mb", type=float, default=0.0,
                    help="concurrent side-stream read of this many MB forked before each MoE call")
    ap.add_argument("--flush-mb", type=float, default=0.0,
                    help="serial read of this many MB on the same stream before each MoE call "
                         "(evicts L2 like the server's weight streaming between layers)")
    ap.add_argument("--touch-inputs", action="store_true",
                    help="after the flush, read the call's hidden states, top-k ids and weights "
                         "(in-server these were just written by the previous kernels)")
    ap.add_argument("--touch-static", action="store_true",
                    help="after the flush, read the layer's small per-expert tensors "
                         "(input global scales, alphas)")
    ap.add_argument("--dummy-call", action="store_true",
                    help="after the flush, run one T=1 MoE call of the same layer with another "
                         "routing: warms the kernels' code and static data, not the measured "
                         "call's own kernel parameters")
    ap.add_argument("--dummy-k", type=int, default=0,
                    help="dummy call keeps only its first K routes per token: another top-k "
                         "instantiation (other code), same workspace pool and static data")
    ap.add_argument("--dummy-own-ws", action="store_true",
                    help="dummy call goes straight through flashinfer with its own workspace buffer: "
                         "same code and static data, not the measured call's graph-pool workspace")
    ap.add_argument("--dummy-layer", type=int, default=None,
                    help="dummy call on this layer instead: same code and workspace pool, "
                         "other static data and weights")
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seconds", type=float, default=0.5)
    args = ap.parse_args()

    ensure_sglang_imports()
    import fc_moe.nojit  # noqa: F401  -- never compile inside the GPU scope
    shape = MS.read_checkpoint_shape(MS.DEFAULT_CKPT)
    lw = prepare_runtime_scales(load_layer(args.layer, MS.DEFAULT_CKPT, device="cuda", shape=shape))
    load_autotune_cache(args.autotune_cache)
    make = get("flashinfer_cutlass")
    env = {k: v for k, v in os.environ.items() if k.startswith("FLASHINFER_MOE")}
    print(f"[fc] label={args.label} env={env}", flush=True)

    side_buf = None
    if args.side_mb > 0:
        side_buf = torch.randn(int(args.side_mb * 1e6 / 2), device="cuda", dtype=torch.bfloat16)
        side_out = torch.empty((), device="cuda", dtype=torch.float32)
        side_stream = torch.cuda.Stream()

    flush_buf = None
    if args.flush_mb > 0:
        flush_buf = torch.randn(int(args.flush_mb * 1e6 / 2), device="cuda", dtype=torch.bfloat16)
        flush_out = torch.empty((), device="cuda", dtype=torch.float32)
    touch_out = torch.empty((), device="cuda", dtype=torch.float32)
    static = [lw.w13_input_scale_quant, lw.g1_alphas, lw.w2_input_scale_quant, lw.g2_alphas]
    dlw = lw
    if args.dummy_layer is not None:
        dlw = prepare_runtime_scales(load_layer(args.dummy_layer, MS.DEFAULT_CKPT, device="cuda",
                                                shape=shape))
    pre = "+".join(n for n, on in (("flush", flush_buf is not None), ("in", args.touch_inputs),
                                   ("static", args.touch_static), ("dummy", args.dummy_call)) if on)
    if args.dummy_call and args.dummy_k:
        pre += f"_k{args.dummy_k}"
    if args.dummy_call and args.dummy_layer is not None:
        pre += f"_L{args.dummy_layer}"
    own_ws = None
    if args.dummy_call and args.dummy_own_ws:
        pre += "_ownws"
        # Arbitrary 64 MB, far above the T=1 workspace; allocated outside the graph pool.
        own_ws = torch.empty(64 << 20, device="cuda", dtype=torch.uint8)

    def touch(tensors):
        for t in tensors:
            torch.sum(t.reshape(-1), dim=0, dtype=torch.float32, out=touch_out)

    results = []

    def run_point(T, tag):
        ids, ws = census_routings(T, ROT)
        ins = build_inputs(lw, ids, ws, device="cuda")
        base = make(lw, ins)
        dummies = [None] * len(base)
        if args.dummy_call:  # other census rows: same kernels, different routes
            dids, dws = census_routings(1, ROT, offset=5 * ROT)
            if args.dummy_k:
                dids = [r[:, :args.dummy_k].contiguous() for r in dids]
                dws = [w[:, :args.dummy_k].contiguous() for w in dws]
            dins = build_inputs(dlw, dids, dws, device="cuda", seed=1)
            if own_ws is not None:
                RN.DIRECT_OPTS["workspace_buffer"] = own_ws
                dummies = get("flashinfer_direct")(dlw, dins)
            else:
                dummies = make(dlw, dins)
        if side_buf is not None:
            def wrap(c):
                def f():
                    cur = torch.cuda.current_stream()
                    side_stream.wait_stream(cur)
                    with torch.cuda.stream(side_stream):
                        torch.sum(side_buf, dim=0, dtype=torch.float32, out=side_out)
                    r = c()
                    cur.wait_stream(side_stream)
                    return r
                return f
            calls = [wrap(c) for c in base]
        else:
            calls = base
        if flush_buf is not None:
            def wrapf(c, ci, dummy):
                def f():
                    torch.sum(flush_buf, dim=0, dtype=torch.float32, out=flush_out)
                    if args.touch_inputs:
                        touch([ci.hidden_states, ci.topk_ids, ci.topk_weights])
                    if args.touch_static:
                        touch(static)
                    if dummy is not None:
                        dummy()
                    return c()
                return f
            calls = [wrapf(c, ci, d) for c, ci, d in zip(calls, ins, dummies)]
        dpre = st.mean(int(torch.unique(r[r >= 0]).numel()) for r in ids)
        g = capture(calls)
        # with --dummy-call this includes the dummy call
        wall = time_replays(g, len(calls), args.seconds)
        # post-prune D: the csrc writes pruned ids back as -1 in place
        dpost = st.mean(int(torch.unique(ci.topk_ids[ci.topk_ids >= 0]).numel()) for ci in ins)
        with tempfile.TemporaryDirectory() as td:
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(10):
                    g.replay()
                torch.cuda.synchronize()
            p = os.path.join(td, "t.json")
            prof.export_chrome_trace(p)
            ks, n = trace_stats(p, stride=2 if args.dummy_call and flush_buf is not None else 1)
        rec = {"label": args.label, "tag": tag, "pre": pre, "T": T, "n_prologues": n,
               "D_pre": round(dpre, 2),
               "D_post": round(dpost, 2), "wall_us": round(wall, 2),
               **{k: round(v, 2) for k, v in ks.items()}}
        print(json.dumps(rec), flush=True)
        results.append(rec)
        del g

    Ts = [int(x) for x in args.T.split(",")]
    if args.tactics:
        a, b = args.tactics.split(":")
        g1r = range(int(a.split("-")[0]), int(a.split("-")[1]) + 1)
        g2r = range(int(b.split("-")[0]), int(b.split("-")[1]) + 1)
        for T in Ts:
            for t1 in g1r:  # gemm1 sweep with the cached gemm2
                force_tactics(t1, None)
                try:
                    run_point(T, f"g1={t1}")
                except Exception as ex:  # invalid tactic for this stage
                    print(json.dumps({"T": T, "tag": f"g1={t1}", "error": str(ex)[:120]}), flush=True)
            for t2 in g2r:
                force_tactics(None, t2)
                try:
                    run_point(T, f"g2={t2}")
                except Exception as ex:
                    print(json.dumps({"T": T, "tag": f"g2={t2}", "error": str(ex)[:120]}), flush=True)
    else:
        if args.force:
            p = [None if x in ("", "none") else int(x) for x in args.force.split(",")]
            force_tactics(p[0], p[1])
        for T in Ts:
            run_point(T, "point")
    if args.out:
        with open(args.out, "a") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
