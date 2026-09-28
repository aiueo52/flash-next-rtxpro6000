#!/usr/bin/env python3
"""Microbenchmark for the routed-expert grouped GEMM at speculative-verify widths.

Reports us/call and achieved GB/s for one layer of Qwen3.8-Flash-Next-NVFP4,
running the production FlashInfer CUTLASS path exactly as SGLang calls it.

Everything except ``--mode run`` works on CPU.  Use ``--mode dryrun`` to check
argument parsing, weight loading, routing generation and the bytes model with
``CUDA_VISIBLE_DEVICES=""``.

Examples (GPU, run only when the GPU is free)::

  # 1. baseline at the three widths, real weights, independent routing
  python -m moe_smallm.bench_moe --mode run --widths 4,8,16

  # 2. THE key experiment: kernel time vs number of distinct experts.
  #    The slope gives achieved DRAM bandwidth; the production D can then be
  #    read back off the fit using the trace's per-layer time.
  python -m moe_smallm.bench_moe --mode run --widths 16 \
      --sweep-distinct 10,20,40,60,80,100,120,139,160

  # 3. correctness against the BF16 dequant reference (tiny slice)
  python -m moe_smallm.bench_moe --mode check --num-experts 16 --widths 4
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

from . import model_shapes as MS
from . import routing as R


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="bench_moe", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("run", "dryrun", "check", "model", "profile"),
                   default="dryrun",
                   help="run=GPU benchmark, dryrun=CPU plumbing check, "
                        "check=correctness vs BF16 reference, model=bytes model only, "
                        "profile=per-kernel breakdown of one configuration")
    p.add_argument("--ckpt", default=MS.DEFAULT_CKPT)
    p.add_argument("--layer", type=int, default=4, help="which decoder layer's experts to load")
    p.add_argument("--num-experts", type=int, default=None,
                   help="load only the first N experts (CPU tests); default = all 512")
    p.add_argument("--synthetic", action="store_true",
                   help="skip the checkpoint, use random weights of the same shape")
    p.add_argument("--widths", default="4,16",
                   help="comma-separated T (draft tokens per verify), e.g. 4,8,16")
    p.add_argument("--candidate", default="flashinfer_cutlass",
                   help="flashinfer_cutlass | flashinfer_direct | triton_grouped")
    p.add_argument("--routing", choices=("independent", "correlated", "hot", "replay", "distinct"),
                   default="independent")
    p.add_argument("--carry", type=float, default=0.0, help="routing=correlated chain parameter")
    p.add_argument("--n-hot", type=int, default=5, help="routing=hot shared-expert count")
    p.add_argument("--replay", default=None, help="routing=replay: route_logger jsonl path")
    p.add_argument("--sweep-distinct", default=None,
                   help="comma-separated D values; forces routing=distinct and sweeps them")
    p.add_argument("--rotation", type=int, default=16,
                   help="calls per captured graph; also how far the working set is spread")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-seconds", type=float, default=0.6)
    p.add_argument("--clocks", action="store_true", help="sample clocks.sm during timing")
    p.add_argument("--no-graph", action="store_true", help="eager timing instead of CUDA graph")
    p.add_argument("--tune-max-num-tokens", type=int, default=None,
                   help="flashinfer_direct: override the autotune bucket ceiling")
    p.add_argument("--no-fused-finalize", action="store_true",
                   help="flashinfer_direct: use_fused_finalize=False (halves the GEMM2 tactic list)")
    p.add_argument("--autotune-cache", default="auto",
                   help="SGLang FlashInfer tactic cache to load ('auto', a path, "
                        "or 'none'); without it the harness silently runs the "
                        "default tactic instead of the production one")
    p.add_argument("--profile-iters", type=int, default=20,
                   help="--mode profile: graph replays inside the profiler")
    p.add_argument("--hidden", type=int, default=None,
                   help="override hidden_size (K of gemm1); forces --synthetic. "
                        "G1 probe: (H/2, 2D) has the same tile geometry as a "
                        "split-K=2 of (H, D), so it prices split-K without a kernel.")
    p.add_argument("--force-tactics", default=None,
                   help="'g1,g2' CUTLASS tactic ids, pinned past the AutoTuner "
                        "(mandatory whenever --hidden changes the shape: the "
                        "on-disk cache would miss and silently run tactic -1)")
    p.add_argument("--json", default=None, help="write results to this path")
    return p.parse_args(argv)


def bytes_model_table(shape: MS.MoEShape) -> list[dict]:
    """Per-layer / per-step bytes and the achieved-bandwidth inversion."""
    # Robust (median-of-per-position) verify-phase kernel times from
    # prof/traces/prof-base32k-{w4,w16}-code-edit, in us per layer.
    TRACE_US = {4: 42.4 + 25.3, 8: None, 16: 87.3 + 54.0}
    rows = []
    for t in (1, 2, 4, 8, 16):
        d = MS.expected_distinct_experts(shape, t)
        row = {
            "T": t,
            "E_distinct_independent": round(d, 2),
            "bytes_per_layer_MB_indep": round(d * shape.bytes_per_expert / 1e6, 1),
            "bytes_per_step_GB_indep": round(
                d * shape.bytes_per_expert * shape.num_layers / 1e9, 3),
        }
        us = TRACE_US.get(t)
        if us:
            row["trace_us_per_layer"] = us
            row["trace_ms_per_step"] = round(us * shape.num_layers / 1e3, 3)
            row["GBps_if_independent"] = round(MS.achieved_gbps(shape, us, d), 1)
            row["pct_peak_if_independent"] = round(
                100 * MS.achieved_gbps(shape, us, d) / MS.PEAK_DRAM_GBPS, 1)
            row["D_max_at_peak"] = round(
                MS.implied_distinct_experts(shape, us, MS.PEAK_DRAM_GBPS), 1)
            row["D_at_measured_copy_roof"] = round(
                MS.implied_distinct_experts(shape, us, MS.MEASURED_COPY_GBPS), 1)
            row["floor_ms_per_step_at_90pct_peak"] = round(
                d * shape.bytes_per_expert * shape.num_layers
                / (0.9 * MS.PEAK_DRAM_GBPS * 1e9) * 1e3, 3)
        rows.append(row)
    return rows


def _routings(args, shape, t, n, distinct=None):
    if distinct is not None:
        import torch
        g = torch.Generator().manual_seed(args.seed)
        return [R.with_distinct(t, shape.top_k, shape.num_experts, distinct, g)
                for _ in range(n)]
    if args.routing == "hot":
        import torch
        g = torch.Generator().manual_seed(args.seed)
        return [R.hot_mixture(t, shape.top_k, shape.num_experts, args.n_hot, g)
                for _ in range(n)]
    return R.make_routings(args.routing, n, t, shape.top_k, shape.num_experts,
                           seed=args.seed, carry=args.carry, replay_path=args.replay)


def _load_weights(args, shape, device):
    from .weights import load_layer, prepare_runtime_scales, synth_layer

    if args.synthetic:
        n = args.num_experts or shape.num_experts
        lw = synth_layer(shape, n, device=device, seed=args.seed)
    else:
        lw = load_layer(args.layer, args.ckpt, device=device,
                        num_experts=args.num_experts, shape=shape)
    return prepare_runtime_scales(lw)


def _apply_force_tactics(args):
    if not args.force_tactics:
        return
    from .runners import force_tactics
    parts = [None if x.strip() in ("", "none") else int(x)
             for x in args.force_tactics.split(",")]
    while len(parts) < 2:
        parts.append(None)
    force_tactics(parts[0], parts[1])


def cmd_model(args, shape):
    rows = bytes_model_table(shape)
    print(f"shape: E={shape.num_experts} topk={shape.top_k} H={shape.hidden} "
          f"I={shape.intermediate} layers={shape.num_layers}")
    print(f"bytes/expert: gemm1={shape.bytes_gemm1_per_expert} "
          f"gemm2={shape.bytes_gemm2_per_expert} total={shape.bytes_per_expert} "
          f"({shape.bytes_per_expert/1e6:.4f} MB); all 512 experts/layer = "
          f"{shape.bytes_all_experts_per_layer/1e9:.3f} GB")
    print(f"peak DRAM {MS.PEAK_DRAM_GBPS:.0f} GB/s (28 Gbps x 512 bit); "
          f"measured copy roof {MS.MEASURED_COPY_GBPS:.0f} GB/s; L2 {MS.L2_BYTES>>20} MiB")
    for r in rows:
        print(json.dumps(r))
    return rows


def cmd_dryrun(args, shape):
    import torch

    print(f"[dryrun] torch {torch.__version__}, cuda_visible="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES','<unset>')}")
    lw = _load_weights(args, shape, "cpu")
    print(f"[dryrun] weights: w13 {tuple(lw.w13_weight.shape)} {lw.w13_weight.dtype}, "
          f"w13_sf {tuple(lw.w13_weight_scale.shape)}, "
          f"w13_sf_swizzled {tuple(lw.w13_blockscale_swizzled.shape)}, "
          f"w2 {tuple(lw.w2_weight.shape)}, "
          f"w2_sf_swizzled {tuple(lw.w2_blockscale_swizzled.shape)}")
    print(f"[dryrun] g1_alphas {tuple(lw.g1_alphas.shape)} "
          f"g2_alphas {tuple(lw.g2_alphas.shape)} "
          f"w13_input_scale_quant={float(lw.w13_input_scale_quant):.6g}")
    from .harness import check_working_set
    from .runners import build_inputs
    sweep = ([int(x) for x in args.sweep_distinct.split(",")]
             if args.sweep_distinct else [None])
    for t in [int(x) for x in args.widths.split(",")]:
        for D in sweep:
            rs = _routings(args, shape, t, args.rotation, distinct=D)
            rs = R.disjointify(rs, lw.num_experts)
            ws = R.uniform_weights(rs)
            ins = build_inputs(lw, rs, ws, device="cpu", seed=args.seed)
            d = R.distinct_counts(rs)
            union = len({int(v) for r in rs for v in r.flatten()})
            touched = union * shape.bytes_per_expert
            print(f"[dryrun] T={t} D={D}: {len(ins)} calls, "
                  f"x{tuple(ins[0].hidden_states.shape)}, distinct/call "
                  f"mean={statistics.mean(d):.1f} min={min(d)} max={max(d)}, "
                  f"rotation union={union} experts = {touched/2**20:.0f} MiB")
            w = check_working_set(touched, f"T={t}")
            if w:
                print("[dryrun] " + w)
    print("[dryrun] OK -- plumbing is sound; run with --mode run on a free GPU.")


def cmd_check(args, shape):
    """Correctness vs the BF16 dequant reference.  GPU for the kernel, CPU for the ref."""
    import torch

    from . import refs
    from .harness import require_idle_gpu
    from .runners import build_inputs, ensure_sglang_imports, get

    require_idle_gpu()
    # Import SGLang in the server's order before anything else; see
    # runners.ensure_sglang_imports.  Doing it up front means an import problem
    # surfaces in a second instead of after a 1.4 GB weight load.
    ensure_sglang_imports()
    lw_cpu = _load_weights(args, shape, "cpu")
    lw = _load_weights(args, shape, "cuda")
    t = int(args.widths.split(",")[0])
    ne = lw.num_experts
    g = torch.Generator().manual_seed(args.seed)
    ids = R.independent(t, min(shape.top_k, ne), ne, g)
    wts = R.uniform_weights([ids])[0]
    ins_gpu = build_inputs(lw, [ids], [wts], device="cuda", seed=args.seed)
    calls = get(args.candidate)(lw, ins_gpu)
    calls[0]()
    torch.cuda.synchronize()
    got = ins_gpu[0].output.cpu()
    ref = refs.moe_reference(lw_cpu, ins_gpu[0].hidden_states.cpu(), ids, wts)
    res = refs.compare(got, ref)
    print(json.dumps({"mode": "check", "T": t, "num_experts": ne, **res}, indent=2))
    return res


def cmd_run(args, shape):
    import torch

    from .harness import check_working_set, require_idle_gpu, time_eager, time_graph
    from .runners import (
        DIRECT_OPTS, build_inputs, ensure_sglang_imports, get, load_autotune_cache,
    )

    require_idle_gpu()
    # See runners.ensure_sglang_imports: standalone imports of the MoE runner
    # hit a circular import that the server's import order hides.
    ensure_sglang_imports()
    if args.autotune_cache != "none":
        load_autotune_cache(args.autotune_cache)
    _apply_force_tactics(args)
    if args.tune_max_num_tokens is not None:
        DIRECT_OPTS["tune_max_num_tokens"] = args.tune_max_num_tokens
    if args.no_fused_finalize:
        DIRECT_OPTS["use_fused_finalize"] = False

    lw = _load_weights(args, shape, "cuda")
    make = get(args.candidate)
    timer = time_eager if args.no_graph else time_graph
    sweep = ([int(x) for x in args.sweep_distinct.split(",")]
             if args.sweep_distinct else [None])
    results = []
    for t in [int(x) for x in args.widths.split(",")]:
        for D in sweep:
            rs = _routings(args, shape, t, args.rotation, distinct=D)
            rs = R.disjointify(rs, lw.num_experts)
            ws = R.uniform_weights(rs)
            ins = build_inputs(lw, rs, ws, device="cuda", seed=args.seed)
            calls = make(lw, ins)
            d = R.distinct_counts(rs)
            dmean = statistics.mean(d)
            union = len({int(v) for r in rs for v in r.flatten()})
            warn = check_working_set(union * shape.bytes_per_expert, f"T={t} D={D}")
            kw = {"min_seconds": args.min_seconds}
            if not args.no_graph:
                kw["clocks"] = args.clocks
            sec, clk = timer(calls, **kw)
            us = sec * 1e6
            gbps = dmean * shape.bytes_per_expert / sec / 1e9
            row = {
                "candidate": args.candidate, "T": t, "requested_distinct": D,
                "distinct_mean": round(dmean, 2), "us_per_call": round(us, 2),
                "ms_per_step_48_layers": round(us * shape.num_layers / 1e3, 3),
                "achieved_GBps": round(gbps, 1),
                "pct_peak": round(100 * gbps / MS.PEAK_DRAM_GBPS, 1),
                "rotation_union_experts": union,
                "sm_clock_min_max": clk,
            }
            if warn:
                row["warning"] = warn
                print(warn, file=sys.stderr)
            print(json.dumps(row), flush=True)
            results.append(row)
    if len(sweep) > 1:
        _fit(results)
    return results


def cmd_profile(args, shape):
    """Per-kernel breakdown of one (T, D) point.  GPU-ONLY.

    Answers "what is the D-independent intercept actually made of": the six glue
    kernels, the two GEMMs, and the launch gaps between them.  Also reports the
    achieved SM clock, because a pure-MoE replay loop is power-limited on this
    325 W card while the server interleaves cheaper kernels.
    """
    import collections

    import torch
    from torch.profiler import ProfilerActivity, profile

    from .harness import capture, require_idle_gpu, time_graph
    from .runners import (
        build_inputs,
        ensure_sglang_imports,
        get,
        load_autotune_cache,
    )

    require_idle_gpu()
    ensure_sglang_imports()
    if args.autotune_cache != "none":
        load_autotune_cache(args.autotune_cache)
    _apply_force_tactics(args)
    lw = _load_weights(args, shape, "cuda")
    make = get(args.candidate)
    sweep = ([int(x) for x in args.sweep_distinct.split(",")]
             if args.sweep_distinct else [None])
    results = []
    for t in [int(x) for x in args.widths.split(",")]:
        for D in sweep:
            rs = R.disjointify(_routings(args, shape, t, args.rotation, distinct=D),
                               lw.num_experts)
            ins = build_inputs(lw, rs, R.uniform_weights(rs), device="cuda",
                               seed=args.seed)
            calls = make(lw, ins)
            dmean = statistics.mean(R.distinct_counts(rs))
            # short burst (clocks still high) vs sustained (power-limited)
            burst, clk_b = time_graph(calls, min_seconds=0.03, spin_seconds=0.0,
                                      clocks=True)
            sust, clk_s = time_graph(calls, min_seconds=args.min_seconds,
                                     spin_seconds=0.25, clocks=True)
            g = capture(calls)
            for _ in range(20):
                g.replay()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(args.profile_iters):
                    g.replay()
                torch.cuda.synchronize()
            agg = collections.defaultdict(lambda: [0, 0.0])
            for e in prof.key_averages():
                if e.device_type.name != "CUDA" and e.self_device_time_total <= 0:
                    continue
                if e.self_device_time_total <= 0:
                    continue
                agg[e.key][0] += e.count
                agg[e.key][1] += e.self_device_time_total
            ncalls = args.profile_iters * len(calls)
            rows = sorted(agg.items(), key=lambda kv: -kv[1][1])
            print(json.dumps({
                "profile": True, "T": t, "distinct": round(dmean, 2),
                "us_per_call_burst": round(burst * 1e6, 2),
                "us_per_call_sustained": round(sust * 1e6, 2),
                "sm_clock_burst": clk_b, "sm_clock_sustained": clk_s,
                "burst_over_sustained": round(sust / burst, 3),
            }))
            tot = 0.0
            for name, (cnt, us) in rows:
                if us / ncalls < 0.05:
                    continue
                tot += us / ncalls
                print(f"    {us/ncalls:8.2f} us/call  n={cnt/ncalls:5.2f}  {name[:150]}")
            print(f"    {tot:8.2f} us/call  TOTAL kernel time "
                  f"(vs {sust*1e6:.2f} wall -> {sust*1e6-tot:.2f} us of gaps)")
            results.append({"T": t, "distinct": dmean,
                            "us_burst": burst * 1e6, "us_sustained": sust * 1e6,
                            "kernels": {k: v[1] / ncalls for k, v in rows}})
    return results


def _fit(rows):
    """Least-squares t = a + b*D per width; b -> achieved bandwidth."""
    from collections import defaultdict

    by = defaultdict(list)
    for r in rows:
        by[r["T"]].append((r["distinct_mean"], r["us_per_call"]))
    shape = MS.read_checkpoint_shape() if os.path.exists(
        os.path.join(MS.DEFAULT_CKPT, "config.json")) else MS.MoEShape()
    for t, pts in sorted(by.items()):
        n = len(pts)
        if n < 2:
            continue
        sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
        sxx = sum(p[0] ** 2 for p in pts); sxy = sum(p[0] * p[1] for p in pts)
        b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        a = (sy - b * sx) / n
        gbps = shape.bytes_per_expert / (b * 1e-6) / 1e9
        print(json.dumps({
            "fit": True, "T": t,
            "us_per_call": f"{a:.2f} + {b:.3f}*D",
            "fixed_overhead_us": round(a, 2),
            "marginal_us_per_expert": round(b, 3),
            "streaming_GBps_from_slope": round(gbps, 1),
            "pct_peak": round(100 * gbps / MS.PEAK_DRAM_GBPS, 1),
        }), flush=True)


def main(argv=None):
    args = parse_args(argv)
    if args.sweep_distinct:
        args.routing = "distinct"
    shape = (MS.read_checkpoint_shape(args.ckpt)
             if os.path.exists(os.path.join(args.ckpt, "config.json")) else MS.MoEShape())
    if args.hidden is not None and args.hidden != shape.hidden:
        import dataclasses
        shape = dataclasses.replace(shape, hidden=args.hidden)
        args.synthetic = True
        print(f"[bench] hidden overridden to {args.hidden}; --synthetic forced")
    fn = {"model": cmd_model, "dryrun": cmd_dryrun, "check": cmd_check,
          "run": cmd_run, "profile": cmd_profile}[args.mode]
    out = fn(args, shape)
    if args.json and out is not None:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
