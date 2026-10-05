"""dg1 (2026-10-01): draft MoE one-token GEMV vs the production FlashInfer CUTLASS chain on the
mtpft5 MTP MoE layer (real weights in the server's format, see dg1/layer.py).

Phases, one process and one GPU-lock hold (< 5 min):
  correct   every T=1 routing of the W16 code-edit census (the draft's one-token calls), random
            bf16 x: GEMV and CUTLASS vs the fp32 references, GEMV vs CUTLASS, determinism
  dispatch  ModelOptNvFp4FusedMoEMethod.apply with the hook: eager, inside a captured CUDA graph,
            first call inside a capture (must fall back), T=2 (must fall back)
  time      CUDA-graph replay of 48 calls with disjoint expert sets (480 of 512 experts, 1.33 GB
            per replay >> 128 MB L2): wall/call from events, per-kernel from a profiler trace;
            back to back and with a 256 MB L2-evicting read before every call
  sweep     (--sweep) k1 and k2 tile grids (dg1/grid.py), top-3 x top-3 jointly, best vs DEFAULT

  python -m dg1.dg1_bench --label job1 --sweep          (under the GPU lock; see dg1/gpu_job.sh)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import statistics as st
import sys
import time
import traceback
import types

import numpy as np
import torch

T0 = time.time()
BENCH = os.path.expanduser("~/tools/flash-next-bench")
RUNS = os.path.join(BENCH, "runs", "dg1")
CENSUS = os.path.join(BENCH, "runs", "census-w16-code-edit.npz")
# The fc_moe private copy of the server's tactic cache (same file as production's, cmp-checked):
# T=1 -> gemm1 tactic 17, gemm2 tactic 52 with the FINALIZE epilogue.
AUTOTUNE = os.path.expanduser(
    "~/.cache/sglang-fcmoe/flashinfer/autotune/0.6.17/sm120/594f7c285c817365/rank_tp0_pp0_dp0.json")
CALLS = 48              # disjoint top-10 sets per timing graph: 480 of the 512 experts
FLUSH_MB = 256          # fc_moe's L2 eviction read (2x the 128 MB L2)
TARGET_MOE_LAYERS = 48  # census T=16 rows = (48 target + 1 draft-extend) calls per step


def log(msg: str) -> None:
    print(f"[dg1 {time.time() - T0:6.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- inputs ----------------------------

def census_t1():
    """Pre-prune ids/weights of every one-token call, and one-token calls per verify step."""
    from dg1.grid import TOPK

    z = np.load(CENSUS)
    rows = np.nonzero(z["T"] == 1)[0]
    assert (z["k"][rows] == TOPK).all()
    ids = torch.from_numpy(z["ids"][rows][:, :TOPK].astype(np.int32))
    wts = torch.from_numpy(z["w"][rows][:, :TOPK].astype(np.float32))
    steps = int((z["T"] == 16).sum()) / (TARGET_MOE_LAYERS + 1)
    return ids, wts, len(rows) / steps


def padded(ids, wts):
    """[N, 16] buffers: row j's [1, 10] view is contiguous and 64 B aligned like a server call."""
    from dg1.grid import TOPK

    n = ids.shape[0]
    ids16 = torch.full((n, 16), -1, dtype=torch.int32)
    w16 = torch.zeros((n, 16), dtype=torch.float32)
    ids16[:, :TOPK], w16[:, :TOPK] = ids, wts
    return ids16.cuda(), w16.cuda()


# ---------------------------------------------------------------- the two paths ---------------------

def gemv(dmg, L, x, ids, wts, ws, cfg):
    return dmg.draft_moe_gemv(
        hidden_states=x, topk_ids=ids, topk_weights=wts, w13_weight=L.w13_weight,
        w13_blockscale_swizzled=L.w13_blockscale_swizzled, w13_weight_scale_2=L.w13_weight_scale_2,
        w2_weight=L.w2_weight, w2_blockscale_swizzled=L.w2_blockscale_swizzled,
        w2_weight_scale_2=L.w2_weight_scale_2, workspace=ws, cfg=cfg)


def runner_config(activation: str = "silu"):
    from dg1.grid import E, H, I, TOPK
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig

    return MoeRunnerConfig(num_experts=E, num_local_experts=E, hidden_size=H,
                           intermediate_size_per_partition=I, top_k=TOPK,
                           params_dtype=torch.bfloat16, activation=activation, is_gated=True,
                           routed_scaling_factor=None)


class Cutlass:
    """The production chain: _run_flashinfer_cutlass with the quant info that apply() builds."""

    def __init__(self, L):
        from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
            FlashInferCutlassMoeQuantInfo, _run_flashinfer_cutlass)

        self.qi = FlashInferCutlassMoeQuantInfo(
            quant_type="fp4", w13_weight=L.w13_weight, w2_weight=L.w2_weight,
            output_dtype=torch.bfloat16,
            quant_scales=[L.w13_input_scale_quant, L.w13_blockscale_swizzled, L.g1_alphas,
                          L.w2_input_scale_quant, L.w2_blockscale_swizzled, L.g2_alphas],
            moe_ep_size=1, moe_ep_rank=0, moe_tp_size=1, moe_tp_rank=0,
            apply_routed_scaling_factor=False)
        self.rc = runner_config()
        self.run = _run_flashinfer_cutlass

    def __call__(self, x, ids, wts, out, quant_info=None):
        # output= skips the symmetric-memory allocation (needs a TP group); else as in-server.
        d = types.SimpleNamespace(hidden_states=x, hidden_states_scale=None,
                                  topk_output=types.SimpleNamespace(topk_weights=wts, topk_ids=ids))
        y = self.run(dispatch_output=d, quant_info=quant_info or self.qi, runner_config=self.rc,
                     output=out)
        assert y.data_ptr() == out.data_ptr(), "cutlass_fused_moe did not write into output="
        return y


# ---------------------------------------------------------------- stats -----------------------------

def rel_rows(a, b):
    a, b = a.float(), b.float()
    return (a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-30)


def summary(a, b) -> dict:
    r = rel_rows(a, b).double().cpu()
    a, b = a.float(), b.float()
    return {"rel_l2_max": r.max().item(), "rel_l2_mean": r.mean().item(),
            "rel_l2_p99": r.quantile(0.99).item(),
            "maxabs_over_maxref": ((a - b).abs().max() / b.abs().max()).item()}


def ordered_bits(t):
    """bf16 -> integers that step by one per bf16 value."""
    b = t.contiguous().view(torch.int16).int()
    return torch.where(b < 0, -(b & 0x7FFF), b)


# ---------------------------------------------------------------- phase: correctness ----------------

def phase_correct(ctx) -> dict:
    from dg1.layer import bf16_rne
    from dg1.proxies import patched

    from dg1.grid import TOPK

    dmg, L, refs, X, ids16, w16, cut, n = (ctx.dmg, ctx.L, ctx.refs, ctx.X, ctx.ids16, ctx.w16,
                                          ctx.cut, ctx.n)
    k = TOPK
    ws = ctx.make_ws(ctx.cfg0)
    out_g = torch.empty((n, X.shape[1]), dtype=torch.bfloat16, device="cuda")
    out_c = torch.empty_like(out_g)
    ids_before, w_before = ids16.clone(), w16.clone()
    for j in range(n):
        out_g[j:j + 1] = gemv(dmg, L, X[j:j + 1], ids16[j:j + 1, :k], w16[j:j + 1, :k], ws, ctx.cfg0)
        cut(X[j:j + 1], ids16[j:j + 1, :k], w16[j:j + 1, :k], out_c[j:j + 1])
    torch.cuda.synchronize()
    ctx.out_g, ctx.out_c = out_g, out_c
    emu = bf16_rne(refs["dqb"])
    ulp = (ordered_bits(out_g) - ordered_bits(emu.bfloat16())).abs()
    res = {
        "rows": n,
        "gemv_vs_dq": summary(out_g, refs["dq"]),
        "gemv_vs_emulation": {**summary(out_g, emu), "elems_gt1ulp": int((ulp > 1).sum()),
                              "elems_eq": int((ulp == 0).sum()), "elems": ulp.numel()},
        "gemv_vs_a4": summary(out_g, refs["a4"]),
        "gemv_vs_bf16_weights": summary(out_g, refs["bf"]),
        "cutlass_vs_dq": summary(out_c, refs["dq"]),
        "cutlass_vs_a4": summary(out_c, refs["a4"]),
        "cutlass_vs_bf16_weights": summary(out_c, refs["bf"]),
        "gemv_vs_cutlass": summary(out_g, out_c),
        "a4_vs_dq": summary(refs["a4"], refs["dq"]),
        "dq_vs_bf16_weights": summary(refs["dq"], refs["bf"]),
        "finite": bool(torch.isfinite(out_g.float()).all() and torch.isfinite(out_c.float()).all()),
        "routing_unchanged": bool(torch.equal(ids16, ids_before) and torch.equal(w16, w_before)),
        "counters_zero": bool((ws.counters == 0).all()),
    }
    # Determinism on the first 256 rows: GEMV rerun, GEMV without PDL, CUTLASS rerun.
    m = min(n, 256)
    rg = torch.empty((m, X.shape[1]), dtype=torch.bfloat16, device="cuda")
    rp, rc = torch.empty_like(rg), torch.empty_like(rg)
    for j in range(m):
        rg[j:j + 1] = gemv(dmg, L, X[j:j + 1], ids16[j:j + 1, :k], w16[j:j + 1, :k], ws, ctx.cfg0)
        with patched(dmg, pdl=False):
            rp[j:j + 1] = gemv(dmg, L, X[j:j + 1], ids16[j:j + 1, :k], w16[j:j + 1, :k], ws, ctx.cfg0)
        cut(X[j:j + 1], ids16[j:j + 1, :k], w16[j:j + 1, :k], rc[j:j + 1])
    torch.cuda.synchronize()
    diff_c = (rc != out_c[:m]).any(dim=1)
    res["determinism"] = {
        "rows": m,
        "gemv_rerun_bitwise": bool(torch.equal(rg, out_g[:m])),
        "gemv_pdl_off_bitwise": bool(torch.equal(rp, out_g[:m])),
        "cutlass_rerun_rows_differing": int(diff_c.sum()),
        "cutlass_rerun_rel_l2_max": rel_rows(rc, out_c[:m]).max().item(),
    }
    return res


# ---------------------------------------------------------------- phase: dispatch -------------------

class _LogTap(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _method(ctx, runner):
    """ModelOptNvFp4FusedMoEMethod state that apply() reads on the FlashInfer CUTLASS path."""
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.utils import MoeRunnerBackend

    cut = ctx.cut

    class _Runner:  # MoeRunner.run -> fused_experts_none_to_flashinfer_cutlass, with output=
        def run(self, dispatch_output, quant_info):
            out = torch.empty_like(dispatch_output.hidden_states)
            cut(dispatch_output.hidden_states, dispatch_output.topk_output.topk_ids,
                dispatch_output.topk_output.topk_weights, out, quant_info=quant_info)
            return StandardCombineInput(hidden_states=out)

    return types.SimpleNamespace(
        moe_runner_config=cut.rc, _moe_runner_backend=MoeRunnerBackend.FLASHINFER_CUTLASS,
        enable_flashinfer_trtllm_moe=False, enable_flashinfer_cutedsl_moe=False,
        enable_flashinfer_cutlass_moe=True, draft_moe_gemv=runner, runner=_Runner(),
        quant_config=None)


def phase_dispatch(ctx) -> dict:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4FusedMoEMethod

    from dg1.grid import TOPK

    dmg, L, X, ids16, w16 = ctx.dmg, ctx.L, ctx.X, ctx.ids16, ctx.w16
    apply = ModelOptNvFp4FusedMoEMethod.apply
    k = TOPK
    calls = {"gemv": 0}
    real = dmg.draft_moe_gemv

    def counted(**kw):
        calls["gemv"] += 1
        return real(**kw)

    def dispatch(x, ids, wts):
        return StandardDispatchOutput(hidden_states=x, hidden_states_scale=None,
                                      topk_output=StandardTopKOutput(topk_weights=wts, topk_ids=ids,
                                                                     router_logits=None))

    tap = _LogTap()
    logging.getLogger(dmg.__name__).addHandler(tap)
    dmg.draft_moe_gemv = counted
    res = {}
    try:
        # (a) eager one-token calls take the GEMV and equal the direct GEMV bit for bit.
        m = _method(ctx, dmg.DraftMoeGemvRunner())
        bad = 0
        for j in range(64):
            y = apply(m, L, dispatch(X[j:j + 1], ids16[j:j + 1, :k], w16[j:j + 1, :k])).hidden_states
            bad += int(not torch.equal(y, ctx.out_g[j:j + 1]))
        torch.cuda.synchronize()
        res["eager"] = {"calls": 64, "gemv_calls": calls["gemv"], "rows_not_bitwise": bad}

        # (b) captured graph of 4 one-token calls, replayed over 50 routing sets.
        m = _method(ctx, dmg.DraftMoeGemvRunner())
        xs = torch.empty((4, X.shape[1]), dtype=torch.bfloat16, device="cuda")
        si = torch.empty((4, 16), dtype=torch.int32, device="cuda")
        sw = torch.empty((4, 16), dtype=torch.float32, device="cuda")

        def load(base):
            xs.copy_(X[base:base + 4])
            si.copy_(ids16[base:base + 4])
            sw.copy_(w16[base:base + 4])

        def four():
            return [apply(m, L, dispatch(xs[i:i + 1], si[i:i + 1, :k], sw[i:i + 1, :k])).hidden_states
                    for i in range(4)]

        load(0)
        four()  # eager first call: the runner allocates its workspace outside the capture
        torch.cuda.synchronize()
        before = calls["gemv"]
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            outs = four()
        captured = calls["gemv"] - before
        bad = 0
        for rep in range(50):
            base = 4 * rep
            load(base)
            g.replay()
            torch.cuda.synchronize()
            bad += sum(int(not torch.equal(outs[i], ctx.out_g[base + i:base + i + 1])) for i in range(4))
        res["graph"] = {"captured_gemv_calls": captured, "replays": 50, "rows_not_bitwise": bad,
                        "counters_zero": bool((m.draft_moe_gemv._workspace.counters == 0).all())}
        del g

        # (c) the very first one-token call inside a capture: CUTLASS for that graph, a warning.
        runner = dmg.DraftMoeGemvRunner()
        m = _method(ctx, runner)
        xc = X[300:301].clone()
        ic, wc = ids16[300:301, :k].clone(), w16[300:301, :k].clone()
        warm = torch.empty_like(xc)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                ctx.cut(xc, ic, wc, warm)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        before, n_msgs = calls["gemv"], len(tap.messages)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            yc = apply(m, L, dispatch(xc, ic, wc)).hidden_states
        g.replay()
        torch.cuda.synchronize()
        res["capture_first"] = {
            "gemv_calls": calls["gemv"] - before, "workspace_none": runner._workspace is None,
            "warned": runner._warned_capture, "log": tap.messages[n_msgs:],
            "rel_l2_vs_eager_cutlass": rel_rows(yc, ctx.out_c[300:301]).max().item(),
        }
        del g

        # (d) two tokens: not a one-token call, CUTLASS (copies: P2 may prune them in place).
        m = _method(ctx, dmg.DraftMoeGemvRunner())
        x2, i2, w2 = X[0:2].clone(), ids16[0:2, :k].clone(), w16[0:2, :k].clone()
        before = calls["gemv"]
        y2 = apply(m, L, dispatch(x2, i2, w2)).hidden_states
        torch.cuda.synchronize()
        res["two_tokens"] = {"gemv_calls": calls["gemv"] - before,
                             "finite": bool(torch.isfinite(y2.float()).all())}
    finally:
        dmg.draft_moe_gemv = real
        logging.getLogger(dmg.__name__).removeHandler(tap)

    # enable-time checks on a stand-in for the FusedMoE layer
    ok_layer = types.SimpleNamespace(moe_runner_config=runner_config(), moe_ep_size=1, moe_tp_size=1,
                                     w13_weight=L.w13_weight)
    gelu_layer = types.SimpleNamespace(moe_runner_config=runner_config("gelu"), moe_ep_size=1,
                                       moe_tp_size=1, w13_weight=L.w13_weight)
    res["enable"] = {
        "reason_silu": dmg._layer_unsupported_reason(ok_layer),
        "reason_gelu": dmg._layer_unsupported_reason(gelu_layer),
        "enable_on_linear": dmg.enable_draft_moe_gemv(torch.nn.Linear(2, 2)),
    }
    return res


# ---------------------------------------------------------------- phase: timing ---------------------

def capture(calls, warmup: int = 3) -> torch.cuda.CUDAGraph:
    """Graph running every callable once, in order (moe_smallm.harness.capture)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            for c in calls:
                c()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for c in calls:
            c()
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    return g


def time_replays(g, ncalls: int, seconds: float) -> float:
    """us per call (fc_moe_bench.time_replays)."""
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(20):
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


_ROLES = (("_draft_moe_up_gate_kernel", "k1"), ("_draft_moe_down_kernel", "k2"),
          ("fusedBuildExpertMaps", "prologue"), ("doActivation", "act"),
          ("finalizeMoeRouting", "finalize"), ("expandInputRows", "expand"),
          ("reduce_kernel", "flush"))


def _role(name: str) -> str:
    for key, role in _ROLES:
        if key in name:
            return role
    if "cutlass" in name or "device_kernel" in name:
        return "gemm"
    return "memset" if "memset" in name.lower() else "other"


def kernel_breakdown(g, replays: int = 5) -> dict:
    """Median us per kernel role per call from a profiler trace of graph replays."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(replays):
            g.replay()
        torch.cuda.synchronize()
    path = os.path.join(RUNS, f".trace-{os.getpid()}.json")
    prof.export_chrome_trace(path)
    with open(path) as f:
        ev = json.load(f)["traceEvents"]
    os.remove(path)
    ks = sorted((e for e in ev if e.get("cat") in ("kernel", "gpu_memset")), key=lambda e: e["ts"])
    calls, cur, names = [], None, {}
    for kk in ks:
        role = _role(kk["name"])
        names[kk["name"][:90]] = names.get(kk["name"][:90], 0) + 1
        if role in ("prologue", "k1"):
            cur = {"start": kk["ts"], "end": kk["ts"] + kk["dur"], "durs": {}, "gemms": 0}
            calls.append(cur)
        if cur is None:
            continue
        if role == "gemm":
            cur["gemms"] += 1
            role = f"gemm{cur['gemms']}"
        if role != "flush":
            cur["end"] = max(cur["end"], kk["ts"] + kk["dur"])
        cur["durs"][role] = cur["durs"].get(role, 0.0) + kk["dur"]
    roles = sorted({r for c in calls for r in c["durs"]})
    out = {r: round(st.median(c["durs"][r] for c in calls if r in c["durs"]), 2) for r in roles}
    out["span"] = round(st.median(c["end"] - c["start"] for c in calls), 2)
    out["calls_seen"] = len(calls)
    out["kernels_per_replay"] = {k: v / replays for k, v in names.items()}
    return out


def timing_inputs(w_census):
    from dg1.grid import E, H, TOPK

    g = torch.Generator().manual_seed(2)
    ids = torch.randperm(E, generator=g)[:CALLS * TOPK].view(CALLS, TOPK).to(torch.int32)
    x = torch.randn((CALLS, H), generator=g).to(torch.bfloat16).cuda()
    ids16, w16 = padded(ids, w_census[:CALLS])
    return x, ids16, w16


def gemv_calls(ctx, x, ids16, w16, cfg, ws):
    from dg1.grid import TOPK

    dmg, L = ctx.dmg, ctx.L
    return [lambda j=j: gemv(dmg, L, x[j:j + 1], ids16[j:j + 1, :TOPK], w16[j:j + 1, :TOPK], ws, cfg)
            for j in range(CALLS)]


def phase_time(ctx, seconds: float = 0.5, rounds: int = 3) -> dict:
    from dg1.grid import H, TOPK
    from dg1.proxies import Skip, patched

    dmg = ctx.dmg
    x, ids16, w16 = timing_inputs(ctx.w_census)
    outc = torch.empty((CALLS, H), dtype=torch.bfloat16, device="cuda")
    flush_buf = torch.randn(FLUSH_MB * (1 << 20) // 2, device="cuda").to(torch.bfloat16)
    flush_out = torch.empty((), dtype=torch.float32, device="cuda")

    def flush():
        torch.sum(flush_buf, dim=0, dtype=torch.float32, out=flush_out)

    def with_flush(calls):
        return [lambda c=c: (flush(), c()) for c in calls]

    cut = [lambda j=j: ctx.cut(x[j:j + 1], ids16[j:j + 1, :TOPK], w16[j:j + 1, :TOPK], outc[j:j + 1])
           for j in range(CALLS)]
    ws = ctx.make_ws(ctx.cfg0)
    gv = gemv_calls(ctx, x, ids16, w16, ctx.cfg0, ws)
    graphs = {"cutlass": capture(cut), "gemv": capture(gv)}
    with patched(dmg, pdl=False):
        graphs["gemv_pdl_off"] = capture(gv)
    with patched(dmg, k2=Skip()):
        graphs["gemv_k1_only"] = capture(gv)
    graphs["flush_only"] = capture([flush] * CALLS)
    graphs["cutlass+flush"] = capture(with_flush(cut))
    graphs["gemv+flush"] = capture(with_flush(gv))
    log(f"time: {len(graphs)} graphs captured")
    samples = {k: [] for k in graphs}
    for _ in range(rounds):
        for name, g in graphs.items():
            samples[name].append(time_replays(g, CALLS, seconds))
    us = {k: round(st.median(v), 2) for k, v in samples.items()}
    res = {"calls_per_graph": CALLS, "us_per_call": us,
           "us_samples": {k: [round(v, 2) for v in vs] for k, vs in samples.items()}}
    fl = us["flush_only"]
    res["us_per_call_after_flush"] = {"cutlass": round(us["cutlass+flush"] - fl, 2),
                                      "gemv": round(us["gemv+flush"] - fl, 2)}
    res["kernels"] = {}
    for name in ("cutlass", "gemv", "cutlass+flush", "gemv+flush"):
        res["kernels"][name] = kernel_breakdown(graphs[name])
    nbytes = ctx.bytes_per_call
    res["GBps"] = {k: round(nbytes / (us[k] * 1e-6) / 1e9, 1) for k in ("cutlass", "gemv")}
    # Saving per verify step: one-token draft calls per step (census W16; 2 at W4 / 3 draft steps).
    saved = {"back_to_back": us["cutlass"] - us["gemv"],
             "after_flush": res["us_per_call_after_flush"]["cutlass"]
             - res["us_per_call_after_flush"]["gemv"]}
    res["saved_us_per_call"] = {k: round(v, 2) for k, v in saved.items()}
    res["saved_ms_per_step"] = {f"{k}_w16": round(v * ctx.per_step / 1e3, 3) for k, v in saved.items()}
    res["saved_ms_per_step"].update({f"{k}_w4": round(v * 2 / 1e3, 3) for k, v in saved.items()})
    return res


# ---------------------------------------------------------------- phase: sweep ----------------------

def phase_sweep(ctx, deadline: float) -> dict:
    from dg1.grid import K1_GRID, K2_GRID, k1_of, k2_of, with_k1, with_k2
    from dg1.proxies import Skip, patched

    dmg = ctx.dmg
    x, ids16, w16 = timing_inputs(ctx.w_census)
    res = {"k1": [], "k2": [], "joint": [], "skipped": 0}

    def measure(cfg, *, k1=None, k2=None, seconds=0.25, act_fill=False):
        ws = ctx.make_ws(cfg)
        if act_fill:
            ws.act.copy_(torch.randn(ws.act.shape, device="cuda") * 0.1)
        with patched(dmg, k1=k1, k2=k2):
            g = capture(gemv_calls(ctx, x, ids16, w16, cfg, ws))
        us = time_replays(g, CALLS, seconds)
        del g
        return us

    for k1 in K1_GRID:
        if time.time() > deadline:
            res["skipped"] += 1
            continue
        res["k1"].append([measure(with_k1(ctx.cfg0, k1), k2=Skip()), list(k1)])
    for k2 in K2_GRID:
        if time.time() > deadline:
            res["skipped"] += 1
            continue
        res["k2"].append([measure(with_k2(ctx.cfg0, k2), k1=Skip(), act_fill=True), list(k2)])
    log(f"sweep: k1 {len(res['k1'])} + k2 {len(res['k2'])} configs, skipped {res['skipped']}")
    top1 = [tuple(k) for _, k in sorted(res["k1"])[:3]]
    top2 = [tuple(k) for _, k in sorted(res["k2"])[:3]]
    for k1 in top1:
        for k2 in top2:
            if time.time() > deadline:
                res["skipped"] += 1
                continue
            cfg = with_k2(with_k1(ctx.cfg0, k1), k2)
            res["joint"].append([measure(cfg, seconds=0.3), list(k1), list(k2)])
    if not res["joint"]:
        return res
    _, k1, k2 = min(res["joint"])
    best = with_k2(with_k1(ctx.cfg0, tuple(k1)), tuple(k2))
    a, b = [], []
    for _ in range(3):  # alternate best and DEFAULT
        a.append(measure(best, seconds=0.5))
        b.append(measure(ctx.cfg0, seconds=0.5))
    res["best"] = {"k1": list(k1_of(best)), "k2": list(k2_of(best)),
                   "us_best": [round(v, 2) for v in a], "us_default": [round(v, 2) for v in b],
                   "median_best": round(st.median(a), 2), "median_default": round(st.median(b), 2)}
    if best != ctx.cfg0:
        res["best"]["correct"] = check_config(ctx, best)
    return res


def check_config(ctx, cfg) -> dict:
    """The first 512 rows with another tile config: vs the RNE emulation and vs DEFAULT."""
    from dg1.grid import TOPK
    from dg1.layer import bf16_rne

    dmg, L, X, ids16, w16 = ctx.dmg, ctx.L, ctx.X, ctx.ids16, ctx.w16
    m = min(ctx.n, 512)
    ws = ctx.make_ws(cfg)
    y = torch.empty((m, X.shape[1]), dtype=torch.bfloat16, device="cuda")
    for j in range(m):
        y[j:j + 1] = gemv(dmg, L, X[j:j + 1], ids16[j:j + 1, :TOPK], w16[j:j + 1, :TOPK], ws, cfg)
    torch.cuda.synchronize()
    return {"rows": m, "vs_emulation": summary(y, bf16_rne(ctx.refs["dqb"][:m])),
            "vs_default": summary(y, ctx.out_g[:m]), "counters_zero": bool((ws.counters == 0).all())}


# ---------------------------------------------------------------- main ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run")
    ap.add_argument("--rows", type=int, default=0, help="correctness rows (0 = all census T=1 rows)")
    ap.add_argument("--phases", default="correct,dispatch,time")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--budget", type=float, default=240.0, help="seconds after start for the sweep")
    args = ap.parse_args()
    os.makedirs(RUNS, exist_ok=True)
    phases = args.phases.split(",") + (["sweep"] if args.sweep else [])

    free, total = torch.cuda.mem_get_info()
    log(f"label={args.label} phases={phases} gpu free {free / 2**30:.1f}/{total / 2**30:.1f} GiB")
    if free < 12 * 2**30:
        sys.exit("[dg1] < 12 GiB free on the GPU; not running")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    import sglang.srt.layers.quantization  # noqa: F401  (import order: moe_smallm.runners)
    import sglang.srt.layers.moe.moe_runner.flashinfer_trtllm  # noqa: F401
    import dg1.nojit  # noqa: F401  -- never JIT-compile FlashInfer inside the GPU scope
    from flashinfer.autotuner import AutoTuner

    from dg1.grid import DEFAULT_CONFIG, E, H, I, TOPK
    from dg1.layer import build_layer
    from sglang.srt.layers.moe import draft_moe_gemv as dmg

    assert dmg.PDL is True, "export SGLANG_TRITON_PDL=1 (production setting)"
    AutoTuner.get().load_configs(AUTOTUNE)
    import flashinfer
    import triton
    log(f"flashinfer {flashinfer.__version__} triton {triton.__version__} torch {torch.__version__} "
        f"sglang from {os.path.dirname(dmg.__file__)}")
    triton_cache = os.environ.get("TRITON_CACHE_DIR", "")
    n_cache0 = len(os.listdir(triton_cache)) if os.path.isdir(triton_cache) else -1

    ids, wts, per_step = census_t1()
    n = args.rows or ids.shape[0]
    ids, wts = ids[:n], wts[:n]
    g = torch.Generator().manual_seed(1)
    X = torch.randn((n, H), generator=g).to(torch.bfloat16).cuda()
    ids16, w16 = padded(ids, wts)
    L, refs = build_layer(X=X, ids=ids16[:, :TOPK], wts=w16[:, :TOPK], device="cuda")
    torch.cuda.synchronize()
    log(f"layer built: {n} rows, {per_step:.2f} one-token calls per W16 step")

    ctx = types.SimpleNamespace(
        dmg=dmg, L=L, refs=refs, X=X, ids16=ids16, w16=w16, n=n, cfg0=DEFAULT_CONFIG,
        cut=Cutlass(L), w_census=wts, out_g=None, out_c=None, per_step=per_step,
        bytes_per_call=TOPK * (2 * I * H // 2 + 2 * I * H // 16 + H * I // 2 + H * I // 16),
        make_ws=lambda cfg: dmg.make_workspace(num_routes=TOPK, hidden_size=H, intermediate_size=I,
                                               device="cuda", cfg=cfg))
    result = {"label": args.label, "num_experts": E, "one_token_calls_per_w16_step": round(per_step, 2),
              "default_config": [DEFAULT_CONFIG.block_n1, DEFAULT_CONFIG.block_k1, DEFAULT_CONFIG.warps1,
                                 DEFAULT_CONFIG.stages1, DEFAULT_CONFIG.block_n2, DEFAULT_CONFIG.block_k2,
                                 DEFAULT_CONFIG.splits2, DEFAULT_CONFIG.warps2, DEFAULT_CONFIG.stages2],
              "env": {k: v for k, v in os.environ.items()
                      if k.startswith(("FLASHINFER_MOE", "SGLANG_", "TRITON_CACHE"))}}
    out_path = os.path.join(RUNS, f"{args.label}.json")
    funcs = {"correct": phase_correct, "dispatch": phase_dispatch, "time": phase_time,
             "sweep": lambda c: phase_sweep(c, T0 + args.budget)}
    for ph in phases:
        try:
            result[ph] = funcs[ph](ctx)
            log(f"{ph}: done")
        except Exception:
            result[ph] = {"error": traceback.format_exc()}
            log(f"{ph}: FAILED\n{result[ph]['error']}")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=1)
    n_cache1 = len(os.listdir(triton_cache)) if os.path.isdir(triton_cache) else -1
    result["triton_cache_new_entries"] = n_cache1 - n_cache0
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "env"}, indent=1)[:12000], flush=True)
    log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
