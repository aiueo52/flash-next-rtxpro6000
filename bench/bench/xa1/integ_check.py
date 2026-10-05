"""xa1: the integrated kernel (worktree ~/tools/sglang-xa, SGLANG_OPT_TRITON_DECODE_ATTN).

Needs the worktree's sglang on PYTHONPATH: XA1_SGLANG_PY=~/tools/sglang-xa/python before env.sh.

* --mode cpu (TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES=): qsa.decode_attn.qsa_decode_attention
  against the fp32 reference on fresh rows (verify / draft-extend) and index-shared draft rows
  (holes mid-row), capped and short contexts; arrival counters must be zero afterwards.
* --mode precompile (CUDA_VISIBLE_DEVICES=): fill the private Triton cache for the gpu mode.
* --mode gpu (only through gpu_job.sh, XA1_MODULE=xa1.integ_check): the real
  QwenSparseAttnBackend._forward_paged_attention, built through __init__ with the flag on and off,
  on synthetic rows vs the reference; a CUDA graph of 12 chained layer calls replayed twice must
  equal eager bitwise; then graph timing per call (on vs off, warm and after an L2 flush).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
import types

import torch

from xa1 import cases

WORKTREE_PY = os.path.expanduser("~/tools/sglang-xa/python")
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
LAYERS = 12
# (rows, ctx, mode, n_tail): fresh = verify / draft-extend rows (draft decode for one row when
# index sharing is off), shared = draft decode rows (2051 frozen + 16 tail columns).
GPU_SHAPES = [(1, 8192, "shared", 8), (1, 1024, "shared", 4), (1, 8192, "fresh", 0),
              (4, 8192, "fresh", 0), (8, 8192, "fresh", 0), (16, 8192, "fresh", 0),
              (4, 1024, "fresh", 0), (16, 2048, "fresh", 0)]
# Rows past 16 (bs > 1) take the fallback launch config (fewer splits); CPU only.
CPU_SHAPES = [(1, 8192, "shared", 8), (1, 300, "shared", 4), (4, 8192, "fresh", 0),
              (8, 2048, "fresh", 0), (16, 1024, "fresh", 0), (32, 2048, "fresh", 0),
              (40, 1024, "fresh", 0)]
TAIL_WIDTH = 16


def _name(rows, ctx, mode, n_tail):
    if mode == "fresh":
        return f"R{rows}-ctx{ctx}"
    return f"R{rows}-ctx{ctx}-shared{2051 + TAIL_WIDTH}-t{n_tail}"


def _case(rows, ctx, mode, n_tail, *, layers, device, seed):
    from xa1.grid import R2T_WIDTH

    return cases.Case(rows=rows, ctx=ctx, layers=layers, mode=mode, tail_width=TAIL_WIDTH,
                      n_tail=n_tail, device=device, seed=seed + rows * 7 + ctx,
                      r2t_width=R2T_WIDTH)


def _check_worktree():
    import sglang

    if not os.path.realpath(sglang.__file__).startswith(os.path.realpath(WORKTREE_PY)):
        sys.exit(f"sglang comes from {sglang.__file__}; export XA1_SGLANG_PY={WORKTREE_PY}")


def _rel_err(out, ref):
    o = out.float().reshape(ref.shape[0], -1)
    r = ref.float().reshape(ref.shape[0], -1)
    row_abs = (o - r).abs().amax(1)
    return float((row_abs / r.abs().amax(1).clamp_min(1e-12)).max()), float(row_abs.max())


# ------------------------------------------------------------------ cpu


def cpu_mode(args, log):
    if os.environ.get("TRITON_INTERPRET") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        sys.exit("run with TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES=")
    from sglang.srt.layers.attention.qsa import decode_attn as da

    ws = da.QSADecodeAttnWorkspace(num_kv_heads=cases.HKV, head_dim=cases.D, device="cpu")
    out, ok = {}, True
    for shape in CPU_SHAPES:
        rows, ctx, mode, n_tail = shape
        case = _case(*shape, layers=1, device="cpu", seed=args.seed)
        lay = case.layers[0]
        t0 = time.time()
        got = da.qsa_decode_attention(
            q=lay.q, k_buffer=lay.k_pool, v_buffer=lay.v_pool, req_to_token=case.req_to_token,
            row_req_pool_indices=case.row_req, topk_indices=lay.topk, seq_lens=lay.seq_lens,
            sm_scale=cases.SCALING, workspace=ws, prefix_valid=mode == "fresh")
        rel, abs_ = _rel_err(got, cases.reference(case, lay))
        zero = int(ws.arrivals.abs().sum()) == 0
        good = rel < 2e-2 and zero
        ok &= good
        out[_name(*shape)] = dict(config=da._launch_config(rows), max_rel=rel, max_abs=abs_,
                                  counters_zero=zero, ok=good, seconds=time.time() - t0)
        log(f"[cpu] {_name(*shape)} cfg={da._launch_config(rows)} max_rel={rel:.2e} "
            f"max_abs={abs_:.2e} counters_zero={zero} {'OK' if good else 'FAIL'} "
            f"({time.time() - t0:.0f}s)")
    return out, ok


# ------------------------------------------------------------------ precompile


def precompile_mode(args, log):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        sys.exit("run with CUDA_VISIBLE_DEVICES= (CPU only)")
    from xa1.grid import R2T_WIDTH, SMEM_LIMIT, TRITON_CACHE
    from xa1.precompile import _fake_sm120_driver, _prod_kernels_warmup, _warm_prod

    if os.path.realpath(os.environ.get("TRITON_CACHE_DIR", "")) != os.path.realpath(TRITON_CACHE):
        sys.exit(f"export TRITON_CACHE_DIR={TRITON_CACHE}")
    _fake_sm120_driver()
    from triton.runtime._async_compile import AsyncCompileMode

    from sglang.srt.layers.attention.qsa import decode_attn as da

    # The CPU-side probe saw no device, so PDL came out False; the GPU run has it on.
    da.PDL = True
    kernel = da._qsa_decode_attn_kernel
    compiled = []

    class _Record:
        def __getitem__(self, grid):
            def run(*a, **kw):
                compiled.append(kernel.warmup(*a, grid=grid, **kw))
            return run

    da._qsa_decode_attn_kernel = _Record()
    ws = da.QSADecodeAttnWorkspace(num_kv_heads=cases.HKV, head_dim=cases.D, device="cpu")
    pool = torch.zeros((64, cases.HKV, cases.D), dtype=torch.float8_e4m3fn)
    r2t = torch.zeros((2, R2T_WIDTH), dtype=torch.int32)
    variants = sorted({(rows, 2051 + (TAIL_WIDTH if mode == "shared" else 0), mode == "fresh")
                       for rows, _, mode, _ in GPU_SHAPES})
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as ex:
        with AsyncCompileMode(ex):
            for rows, ncols, prefix_valid in variants:
                da.qsa_decode_attention(
                    q=torch.zeros((rows, cases.HQ, cases.D), dtype=torch.bfloat16),
                    k_buffer=pool, v_buffer=pool, req_to_token=r2t,
                    row_req_pool_indices=torch.ones(rows, dtype=torch.int32),
                    topk_indices=torch.zeros((rows, ncols), dtype=torch.int32),
                    seq_lens=torch.ones(rows, dtype=torch.int32), sm_scale=cases.SCALING,
                    workspace=ws, prefix_valid=prefix_valid)
            with _prod_kernels_warmup() as sa:
                # Row counts 1, 16 and others are separate integer specializations.
                for topk in (2051, 2051 + TAIL_WIDTH):
                    for rows in (1, 4, 16):
                        _warm_prod(sa, topk, R2T_WIDTH, rows=rows)
    shared = [k.metadata.shared for k in compiled]
    log(f"[precompile] {len(variants)} integrated variants {variants} + prod QSA kernels in "
        f"{time.time() - t0:.1f}s; smem {shared}")
    return dict(variants=variants, smem=shared), max(shared) <= SMEM_LIMIT


# ------------------------------------------------------------------ gpu


class _Pool:
    def __init__(self, layers):
        self._layers = layers

    def get_key_buffer(self, layer_id):
        return self._layers[layer_id].k_pool

    def get_value_buffer(self, layer_id):
        return self._layers[layer_id].v_pool


def _backend(case, *, flag, mode, rows):
    """A real QwenSparseAttnBackend (through __init__) over the case's pools, plus a caller."""
    from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    os.environ["SGLANG_OPT_TRITON_DECODE_ATTN"] = "1" if flag else "0"
    runner = types.SimpleNamespace(
        token_to_kv_pool=_Pool(case.layers), device=torch.device("cuda"), model_config=None,
        req_to_token_pool=types.SimpleNamespace(req_to_token=case.req_to_token))
    be = QwenSparseAttnBackend(runner)
    assert be._triton_decode_attn == flag
    be.forward_metadata = types.SimpleNamespace(
        sequence_lengths=case.layers[0].seq_lens, row_req_pool_indices=case.row_req,
        is_cuda_graph=False, fa2_valid_counts=None)
    if mode == "shared":
        # Any state object: draft decode then reuses the draft-extend selection.
        be._mtp_shared_sparse_indices = object()
    fmode = ForwardMode.DECODE if mode == "shared" or rows == 1 else ForwardMode.TARGET_VERIFY
    fb = types.SimpleNamespace(forward_mode=fmode, req_pool_indices=case.row_req)
    layers = [types.SimpleNamespace(layer_id=i, scaling=cases.SCALING)
              for i in range(len(case.layers))]

    def call(i):
        lay = case.layers[i]
        return be._forward_paged_attention(lay.q, layers[i], fb, lay.topk)

    return be, call


def _graph_matches_eager(call, capture):
    eager = [call(i).clone() for i in range(LAYERS)]
    held = {}

    def run():
        held["outs"] = [call(i) for i in range(LAYERS)]

    g = capture(run)
    g.replay()
    torch.cuda.synchronize()
    first = [o.clone() for o in held["outs"]]
    g.replay()
    torch.cuda.synchronize()
    return (all(torch.equal(a, e) for a, e in zip(first, eager))
            and all(torch.equal(a, e) for a, e in zip(held["outs"], eager)))


def gpu_mode(args, log):
    from xa1.bench import Flush, _diff, capture, chain, spin, time_graphs
    from xa1.fi import load_trtllm_decode

    # Puts the production XQA .so in the private JIT dir and forbids any build; the backend
    # then imports the same flashinfer function.
    _, xqa_info = load_trtllm_decode()
    from sglang.srt.layers.attention.qsa import decode_attn as da

    if not da.PDL:
        sys.exit("PDL is off: source env.sh (SGLANG_TRITON_PDL=1)")
    flush = Flush()
    out, ok = dict(xqa=xqa_info), True
    for shape in GPU_SHAPES:
        rows, ctx, mode, n_tail = shape
        name = _name(*shape)
        case = _case(*shape, layers=LAYERS, device="cuda", seed=args.seed)
        be_on, on = _backend(case, flag=True, mode=mode, rows=rows)
        _, off = _backend(case, flag=False, mode=mode, rows=rows)
        lay = case.layers[0]
        ref = cases.reference(case, lay)
        o_on = on(0).view_as(lay.q).clone()
        o_off = off(0).view_as(lay.q).clone()
        torch.cuda.synchronize()
        rel_on, abs_on = _rel_err(o_on, ref)
        rel_off, abs_off = _rel_err(o_off, ref)
        res = dict(config=da._launch_config(rows),
                   valid_cols=case.valid_mask(lay).sum(1).tolist(),
                   on_vs_ref=dict(max_rel=rel_on, max_abs=abs_on),
                   off_vs_ref=dict(max_rel=rel_off, max_abs=abs_off),
                   on_vs_off=_diff(o_on, o_off),
                   counters_zero=int(be_on._decode_attn_workspace.arrivals.abs().sum()) == 0,
                   graph_equals_eager=_graph_matches_eager(on, capture))
        res["counters_zero_after_graph"] = (
            int(be_on._decode_attn_workspace.arrivals.abs().sum()) == 0)
        good = (rel_on < 2e-2 and res["counters_zero"] and res["graph_equals_eager"]
                and res["counters_zero_after_graph"])
        ok &= good
        warm = {k: (capture(chain(f, 2 * LAYERS)), 2 * LAYERS) for k, f in (("on", on), ("off", off))}
        cold = {k: (capture(chain(f, LAYERS)), LAYERS) for k, f in (("on", on), ("off", off))}
        spin(0.5)
        res["warm"] = time_graphs(warm, rounds=args.rounds)
        res["cold"] = time_graphs(cold, rounds=args.rounds, flush=flush)
        res["ok"] = good
        out[name] = res
        w, c = res["warm"], res["cold"]
        log(f"[gpu] {name} cfg={res['config']} on_vs_ref={rel_on:.2e} off_vs_ref={rel_off:.2e} "
            f"rows_neq={res['on_vs_off']['frac_rows_neq']:.2f} "
            f"graph_eq={res['graph_equals_eager']} counters0={res['counters_zero_after_graph']} | "
            f"warm on {w['on']['median']:.2f} off {w['off']['median']:.2f} | "
            f"cold on {c['on']['median']:.2f} off {c['off']['median']:.2f} us/call "
            f"{'OK' if good else 'FAIL'}")
        del warm, cold
        torch.cuda.synchronize()
    return out, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="xa1-integ")
    ap.add_argument("--mode", choices=("cpu", "precompile", "gpu"), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()
    _check_worktree()

    def log(msg):
        print(msg, flush=True)

    fn = dict(cpu=cpu_mode, precompile=precompile_mode, gpu=gpu_mode)[args.mode]
    res, ok = fn(args, log)
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"{args.label}-{args.mode}.json")
    with open(path, "w") as f:
        json.dump(dict(mode=args.mode, ok=ok, results=res), f, indent=1, default=str)
    log(f"[{args.mode}] {'OK' if ok else 'FAIL'} -> {path}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
