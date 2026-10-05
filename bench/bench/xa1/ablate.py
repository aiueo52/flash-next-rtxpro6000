"""xa1: where the candidate-B time goes. Ablated and timer-instrumented copies of the B kernel.

abl1 instrumented v2 (fp32 partials, 3D combine, masked K/V, single-buffered); abl3 the v3 kernel
(unmasked K/V loads pipelined at 3 * st - 2 stages, V converted by inline asm, 2D combine, f16
partials with cfg.part16) and v4 (cfg.lse: acc / l + lse partials, barrier-free combine). Same
stamps and flags for all. abl2 never ran: nogather at 3 * st - 2 stages needs 138 KB of smem.

Flags (each removes one part; outputs are wrong on purpose, timing only):
  nored    the last arriver only resets the counter (partials + atomic kept)
  nosplit  no partials, atomic or reduction: every split stores acc/l to out (racy)
  nogather slots from the column number (no top-k index / req_to_token loads)
  noload   K/V tiles synthesised in registers (no KV loads at all)
  noloop   no main loop
  timed    per-CTA %globaltimer stamps: entry, after griddepcontrol.wait, after the last tile's
           softmax, after the arrival atomic, after the reduction (last arriver), exit; + %smid

  source bench/xa1/env.sh; CUDA_VISIBLE_DEVICES= python -m xa1.ablate --precompile
  (GPU, lock)  XA1_MODULE=xa1.ablate bash bench/xa1/gpu_job.sh abl1
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch
import triton
import triton.language as tl

from sglang.kernels.triton_pdl import pdl_trigger, pdl_wait

from xa1 import cases
from xa1.kernels import (BLOCK_H, LOG2E, XA1Config, XA1Workspace, _combine_lse,
                         _e4m3_to_f16_asm, _split_totals)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
LAYERS = 12
NCOLS = 2051
STAMPS = 8
MAX_CTAS = 512
FLAGS = ("nored", "nosplit", "nogather", "noload", "noloop", "timed")


@triton.jit
def _gtime(dep):
    # The read is predicated on dep, so it cannot issue before dep exists.
    return tl.inline_asm_elementwise(
        "{ .reg .pred p; setp.ne.s32 p, $1, -1234567; mov.u64 $0, 0; "
        "@p mov.u64 $0, %globaltimer; }",
        "=l,r", [dep], dtype=tl.int64, is_pure=False, pack=1)


@triton.jit
def _smid(dep):
    return tl.inline_asm_elementwise(
        "{ .reg .u32 t; mov.u32 t, $1; mov.u32 $0, %smid; }",
        "=r,r", [dep], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _stamp(dbg_ptr, pid, k: tl.constexpr, t, offs_h):
    tl.store(dbg_ptr + pid * 8 + k + offs_h * 0, t.to(tl.int64), mask=offs_h == 0)


@triton.jit
def _abl_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr, len_ptr, idx_ptr, r2t_ptr, req_ptr,
    ws_o_ptr, ws_ml_ptr, cnt_ptr, dbg_ptr,
    qk_scale, out_scale, idx_stride, r2t_stride,
    NCOLS: tl.constexpr, HKV: tl.constexpr, GROUP: tl.constexpr, BLOCK_H: tl.constexpr,
    D: tl.constexpr, NUM_SPLITS: tl.constexpr, SPLITS_P2: tl.constexpr, BLOCK_N: tl.constexpr,
    USE_PDL: tl.constexpr, PART_F16: tl.constexpr, V_ASM: tl.constexpr, LSE: tl.constexpr,
    NORED: tl.constexpr, NOSPLIT: tl.constexpr, NOGATHER: tl.constexpr, NOLOAD: tl.constexpr,
    NOLOOP: tl.constexpr, TIMED: tl.constexpr,
):
    sid = tl.program_id(0)
    kvh = tl.program_id(1)
    row = tl.program_id(2)
    grp = row * HKV + kvh
    pid = grp * NUM_SPLITS + sid
    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    hmask = offs_h < GROUP
    qo_off = (grp * GROUP + offs_h)[:, None] * D + offs_d[None, :]
    hd_off = kvh * D + offs_d
    if TIMED:
        _stamp(dbg_ptr, pid, 0, _gtime(offs_h), offs_h)
        _stamp(dbg_ptr, pid, 6, _smid(offs_h), offs_h)
    pdl_wait(USE_PDL)
    if TIMED:
        _stamp(dbg_ptr, pid, 1, _gtime(offs_h), offs_h)
    q = tl.load(q_ptr + qo_off, mask=hmask[:, None], other=0.0).to(tl.float16)
    length = tl.load(len_ptr + row)
    req = tl.load(req_ptr + row).to(tl.int64)
    chunk = tl.cdiv(tl.cdiv(NCOLS, NUM_SPLITS), BLOCK_N) * BLOCK_N
    start = sid * chunk
    end = tl.minimum(start + chunk, NCOLS)
    if NOLOOP:
        end = start
    m_i = tl.full([BLOCK_H], -1.0e30, tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    acc = tl.zeros([BLOCK_H, D], tl.float32)
    for n0 in range(start, end, BLOCK_N):
        cols = n0 + offs_n
        cmask = cols < end
        if NOLOAD:
            valid = cmask
            k = ((offs_n[:, None] * 3 + offs_d[None, :] + n0) % 7).to(tl.float16)
            v = ((offs_n[:, None] + offs_d[None, :] * 5 + n0) % 5).to(tl.float16)
        else:
            if NOGATHER:
                valid = cmask
                slot = (cols * 5 + row * 131) % 16384
            else:
                pos = tl.load(idx_ptr + row * idx_stride + cols, mask=cmask, other=-1)
                valid = (pos >= 0) & (pos < length)
                slot = tl.load(r2t_ptr + req * r2t_stride + pos, mask=valid, other=0)
            kv_off = slot.to(tl.int64)[:, None] * (HKV * D) + hd_off[None, :]
            k = tl.load(k_ptr + kv_off).to(tl.float16)
            if V_ASM:
                v = _e4m3_to_f16_asm(tl.load(v_ptr + kv_off))
            else:
                v = tl.load(v_ptr + kv_off).to(tl.float16)
        s = tl.dot(q, tl.trans(k))
        s = tl.where(valid[None, :], s * qk_scale, -1.0e30)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.where(valid[None, :], tl.exp2(s - m_new[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(tl.float16), v)
        m_i = m_new
    if TIMED:
        _stamp(dbg_ptr, pid, 2, _gtime(l_i.to(tl.int32, bitcast=True)), offs_h)
        _stamp(dbg_ptr, pid, 7, tl.cdiv(end - start, BLOCK_N) + offs_h * 0, offs_h)
    if NOSPLIT:
        o = acc / tl.where(l_i > 0, l_i, 1.0)[:, None] * out_scale
        tl.store(out_ptr + qo_off, o.to(out_ptr.dtype.element_ty), mask=hmask[:, None])
    elif LSE:
        _abl_finish_lse(acc, m_i, l_i, out_ptr, ws_o_ptr, ws_ml_ptr, cnt_ptr, dbg_ptr, grp, pid,
                        out_scale, qo_off, offs_h, offs_d, hmask, GROUP=GROUP, BLOCK_H=BLOCK_H,
                        D=D, NUM_SPLITS=NUM_SPLITS, NORED=NORED, TIMED=TIMED)
    else:
        o_off = (pid * BLOCK_H + offs_h)[:, None] * D + offs_d[None, :]
        if PART_F16:
            acc = acc / tl.where(l_i > 0, l_i, 1.0)[:, None]
        tl.store(ws_o_ptr + o_off, acc.to(ws_o_ptr.dtype.element_ty), mask=hmask[:, None],
                 cache_modifier=".cg")
        tl.store(ws_ml_ptr + pid * 2 * BLOCK_H + offs_h, m_i, cache_modifier=".cg")
        tl.store(ws_ml_ptr + pid * 2 * BLOCK_H + BLOCK_H + offs_h, l_i, cache_modifier=".cg")
        tl.debug_barrier()
        done = tl.atomic_add(cnt_ptr + grp, 1, sem="acq_rel", scope="gpu")
        if TIMED:
            _stamp(dbg_ptr, pid, 3, _gtime(done + offs_h * 0), offs_h)
        if done == NUM_SPLITS - 1:
            if not NORED:
                _, _, m_tot, l_tot = _split_totals(ws_ml_ptr, grp, offs_h, BLOCK_H, NUM_SPLITS,
                                                   SPLITS_P2)
                o = tl.zeros([BLOCK_H, D], tl.float32)
                for s_i in tl.static_range(NUM_SPLITS):
                    part = grp * NUM_SPLITS + s_i
                    w = tl.exp2(tl.load(ws_ml_ptr + part * 2 * BLOCK_H + offs_h,
                                        cache_modifier=".cg") - m_tot)
                    if PART_F16:
                        w = w * tl.load(ws_ml_ptr + part * 2 * BLOCK_H + BLOCK_H + offs_h,
                                        cache_modifier=".cg")
                    o_s = tl.load(ws_o_ptr + (part * BLOCK_H + offs_h)[:, None] * D
                                  + offs_d[None, :], mask=hmask[:, None], other=0.0,
                                  cache_modifier=".cg")
                    o += o_s.to(tl.float32) * w[:, None]
                o = o * (out_scale / tl.where(l_tot > 0, l_tot, 1.0))[:, None]
                tl.store(out_ptr + qo_off, o.to(out_ptr.dtype.element_ty), mask=hmask[:, None])
                if TIMED:
                    _stamp(dbg_ptr, pid, 4, _gtime(tl.sum(o, 1).to(tl.int32, bitcast=True)),
                           offs_h)
            tl.store(cnt_ptr + grp, 0)
    if TIMED:
        _stamp(dbg_ptr, pid, 5, _gtime(offs_h), offs_h)
    pdl_trigger(USE_PDL)


@triton.jit
def _abl_finish_lse(acc, m_i, l_i, out_ptr, ws_o_ptr, ws_ml_ptr, cnt_ptr, dbg_ptr, grp, pid,
                    out_scale, qo_off, offs_h, offs_d, hmask, GROUP: tl.constexpr,
                    BLOCK_H: tl.constexpr, D: tl.constexpr, NUM_SPLITS: tl.constexpr,
                    NORED: tl.constexpr, TIMED: tl.constexpr):
    """kernels._finish (LSE branch) with the stamps."""
    o_off = (pid * BLOCK_H + offs_h)[:, None] * D + offs_d[None, :]
    o = acc / tl.where(l_i > 0, l_i, 1.0)[:, None]
    tl.store(ws_o_ptr + o_off, o.to(ws_o_ptr.dtype.element_ty), mask=hmask[:, None],
             cache_modifier=".cg")
    lse = tl.where(l_i > 0, m_i + tl.log2(tl.where(l_i > 0, l_i, 1.0)), -1.0e30)
    tl.store(ws_ml_ptr + pid * BLOCK_H + offs_h, lse, cache_modifier=".cg")
    tl.debug_barrier()
    done = tl.atomic_add(cnt_ptr + grp, 1, sem="acq_rel", scope="gpu")
    if TIMED:
        _stamp(dbg_ptr, pid, 3, _gtime(done + offs_h * 0), offs_h)
    if done == NUM_SPLITS - 1:
        if not NORED:
            o = _combine_lse(ws_o_ptr, ws_ml_ptr, out_ptr, grp, out_scale, offs_h, offs_d,
                             hmask, GROUP=GROUP, BLOCK_H=BLOCK_H, D=D, NUM_SPLITS=NUM_SPLITS)
            if TIMED:
                _stamp(dbg_ptr, pid, 4, _gtime(tl.sum(o, 1).to(tl.int32, bitcast=True)), offs_h)
        tl.store(cnt_ptr + grp, 0)


def launch(*, q, k_pool, v_pool, out, seq_lens, topk, r2t, row_req, ws, dbg, cfg: XA1Config,
           flags=(), pdl=True, warmup=False):
    rows, hq, d = q.shape
    hkv = k_pool.shape[1]
    splits_p2 = triton.next_power_of_2(cfg.num_splits)
    ws_o = ws.ws_o.view(torch.float16) if cfg.part16 or cfg.lse else ws.ws_o
    # Without the pos -> slot chain the pipeliner gives K/V all num_stages - 1 stages.
    chained = "nogather" not in flags and "noload" not in flags
    args = (q, k_pool, v_pool, out, seq_lens, topk, r2t, row_req, ws_o, ws.ws_ml, ws.cnt, dbg,
            cases.SCALING * LOG2E, 1.0, topk.stride(0), r2t.stride(0))
    kw = dict(NCOLS=topk.shape[1], HKV=hkv, GROUP=hq // hkv, BLOCK_H=BLOCK_H, D=d,
              NUM_SPLITS=cfg.num_splits, SPLITS_P2=splits_p2, BLOCK_N=cfg.block_n, USE_PDL=pdl,
              PART_F16=cfg.part16, V_ASM=cfg.vcvt, LSE=cfg.lse, num_warps=cfg.num_warps,
              num_stages=3 * cfg.num_stages - 2 if chained else cfg.num_stages,
              **{f.upper(): f in flags for f in FLAGS})
    if pdl:
        kw["launch_pdl"] = True
    grid = (cfg.num_splits, hkv, rows)
    if warmup:
        return _abl_kernel.warmup(*args, grid=grid, **kw)
    _abl_kernel[grid](*args, **kw)
    return out


@triton.jit
def _tres_kernel(out_ptr, N: tl.constexpr):
    offs = tl.arange(0, 32)
    t_prev = _gtime(offs)
    for i in range(N):
        t = _gtime((t_prev & 0x7FFFFFFF).to(tl.int32))
        tl.store(out_ptr + i + offs * 0, t, mask=offs == 0)
        t_prev = t


# ------------------------------------------------------------------ what runs

# abl3: job3's best v3 configs and v4 (lse) candidates (abl1 ran v2's 12/64/4/3, 12/64/4/3,
# 10/64/4/3, 5/64/8/3).
BEST = {1: XA1Config(11, 64, 4, 2, part16=True, vcvt=True),
        4: XA1Config(11, 64, 4, 2, part16=True, vcvt=True),
        8: XA1Config(11, 64, 4, 2, part16=True, vcvt=True),
        16: XA1Config(5, 32, 4, 4, part16=True, vcvt=True)}
V4 = {1: (XA1Config(11, 64, 4, 2, vcvt=True, lse=True), XA1Config(17, 64, 4, 2, vcvt=True, lse=True)),
      4: (XA1Config(11, 64, 4, 2, vcvt=True, lse=True), XA1Config(17, 64, 4, 2, vcvt=True, lse=True)),
      8: (XA1Config(11, 64, 4, 2, vcvt=True, lse=True),),
      16: (XA1Config(5, 64, 4, 2, vcvt=True, lse=True), XA1Config(5, 64, 4, 3, lse=True))}
ABLATIONS = [(), ("nored",), ("nosplit",), ("nogather",), ("nogather", "nosplit"), ("noload",),
             ("noload", "nosplit"), ("noloop",), ("noloop", "nosplit")]
V4_ABLATIONS = [(), ("nored",), ("nosplit",), ("noloop",)]
# Per-tile slope: one CTA per (row, kv head) walks ceil(2051 / splits / 64) tiles, no reduction.
SLOPE_SPLITS = (1, 2, 4, 8)
SLOPE_FLAGS = [("nosplit",), ("nogather", "nosplit"), ("noload", "nosplit")]


def jobs():
    """(rows, cfg, flags, pdl) for every launch of the GPU run."""
    out = []
    for rows, cfg in BEST.items():
        out += [(rows, cfg, f, True) for f in ABLATIONS]
        out += [(rows, cfg, (), False), (rows, cfg, ("timed",), True)]
        # fp32 partials: how much of the combine is partial bytes.
        out += [(rows, cfg._replace(part16=False), (), True)]
        for c4 in V4[rows]:
            out += [(rows, c4, f, True) for f in V4_ABLATIONS]
            out += [(rows, c4, ("timed",), True)]
    for s in SLOPE_SPLITS:
        out += [(1, XA1Config(s, 64, 4, 2, part16=True, vcvt=True), f, True) for f in SLOPE_FLAGS]
    return out


def _name(rows, cfg, flags, pdl):
    return f"R{rows} {cfg.key} {'+'.join(flags) or 'full'}{'' if pdl else ' nopdl'}"


def precompile():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        sys.exit("run with CUDA_VISIBLE_DEVICES= (CPU only)")
    import concurrent.futures

    from triton.runtime._async_compile import AsyncCompileMode

    from xa1.grid import MAX_SPLITS, R2T_WIDTH, SMEM_LIMIT
    from xa1.precompile import _fake_sm120_driver

    _fake_sm120_driver()
    ws = XA1Workspace(max_groups=2, max_splits=MAX_SPLITS, head_dim=cases.D, device="cpu")
    pool = torch.zeros((64, cases.HKV, cases.D), dtype=torch.float8_e4m3fn)
    lens = torch.ones(1, dtype=torch.int32)
    r2t = torch.zeros((2, R2T_WIDTH), dtype=torch.int32)
    idx = torch.zeros((1, NCOLS), dtype=torch.int32)
    dbg = torch.zeros(MAX_CTAS * STAMPS, dtype=torch.int64)
    q = torch.zeros((1, cases.HQ, cases.D), dtype=torch.bfloat16)
    t0 = time.time()
    js = jobs()
    compiled = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        with AsyncCompileMode(ex):
            for rows, cfg, flags, pdl in js:
                k = launch(q=q, k_pool=pool, v_pool=pool, out=torch.empty_like(q), seq_lens=lens,
                           topk=idx, r2t=r2t, row_req=lens, ws=ws, dbg=dbg, cfg=cfg, flags=flags,
                           pdl=pdl, warmup=True)
                compiled.append((_name(rows, cfg, flags, pdl), k))
            _tres_kernel.warmup(dbg, N=256, grid=(1,), num_warps=1)
    print(f"[ablate-precompile] {len(js)} variants in {time.time() - t0:.1f}s", flush=True)
    big = [(name, k.metadata.shared) for name, k in compiled if k.metadata.shared > SMEM_LIMIT]
    for b in big:
        print(f"[ablate-precompile] OVER SMEM LIMIT {b}", flush=True)
    sys.exit(1 if big else 0)


# ------------------------------------------------------------------ GPU side


def _timeline(dbg, n_calls, n_ctas):
    """Per call stats (us) from the stamps of one replay; call i follows call i-1 in the graph."""
    st = dbg.view(n_calls, MAX_CTAS, STAMPS)[:, :n_ctas].cpu().long()
    out = []
    for c in range(n_calls):
        s = st[c]
        t0, t1, t2, t3, t4, t5 = (s[:, i] for i in range(6))
        last = t4 > 0
        rec = dict(
            launch_spread=float(t0.max() - t0.min()) / 1e3,
            wait_spread=float(t1.max() - t1.min()) / 1e3,
            loop_med=float((t2 - t1).median()) / 1e3, loop_max=float((t2 - t1).max()) / 1e3,
            atomic_med=float((t3 - t2).median()) / 1e3 if bool((t3 > 0).all()) else None,
            red=float((t4[last] - t3[last]).max()) / 1e3 if bool(last.any()) else None,
            tail=float(t5.max() - (t4[last].max() if bool(last.any()) else t3.max())) / 1e3,
            dur=float(t5.max() - t0.min()) / 1e3, run=float(t5.max() - t1.min()) / 1e3,
            sms=int(torch.unique(s[:, 6]).numel()), tiles_max=int(s[:, 7].max()))
        if c:
            prev_end = int(st[c - 1][:, 5].max())
            rec.update(start_vs_prev_end=float(int(t0.min()) - prev_end) / 1e3,
                       release_vs_prev_end=float(int(t1.min()) - prev_end) / 1e3,
                       end_to_end=float(int(t5.max()) - prev_end) / 1e3)
        out.append(rec)
    keys = out[1].keys()
    return {k: statistics.median([r[k] for r in out[1:] if r.get(k) is not None] or [0.0])
            for k in keys}


def gpu_main(args):
    from xa1.bench import Flush, Shape, capture, spin, time_graphs

    os.makedirs(RESULTS, exist_ok=True)
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    t_start = time.time()
    res = {}
    dbg_t = torch.zeros(4096, dtype=torch.int64, device="cuda")
    _tres_kernel[(1,)](dbg_t, N=256, num_warps=1)
    torch.cuda.synchronize()
    d = (dbg_t[1:256] - dbg_t[:255]).cpu()
    res["timer_steps_ns"] = sorted(set(int(x) for x in d.tolist()))[:12]
    log(f"[ablate] globaltimer steps (ns): {res['timer_steps_ns']}")
    ws = XA1Workspace(max_groups=64, max_splits=32, head_dim=cases.D, device="cuda")
    n_warm = 2 * LAYERS
    dbg = torch.zeros(n_warm * MAX_CTAS * STAMPS, dtype=torch.int64, device="cuda")
    spin(1.5)
    by_rows = {}
    for rows, cfg, flags, pdl in jobs():
        by_rows.setdefault(rows, []).append((cfg, flags, pdl))
    for rows, js in by_rows.items():
        shape = Shape(rows, 8192)
        case = shape.build(layers=LAYERS, seed=args.seed + rows * 7 + 8192, overlap=0.75)
        outs = [torch.empty_like(lay.q) for lay in case.layers]

        def call_fn(cfg, flags, pdl):
            def call(i):
                lay = case.layers[i % LAYERS]
                launch(q=lay.q, k_pool=lay.k_pool, v_pool=lay.v_pool, out=outs[i % LAYERS],
                       seq_lens=lay.seq_lens, topk=lay.topk, r2t=case.req_to_token,
                       row_req=case.row_req, ws=ws,
                       dbg=dbg[i * MAX_CTAS * STAMPS:(i + 1) * MAX_CTAS * STAMPS], cfg=cfg,
                       flags=flags, pdl=pdl)
            return call

        def chain(call, n):
            def run():
                for i in range(n):
                    call(i)
            return run

        warm, cold, timed = {}, {}, []
        for cfg, flags, pdl in js:
            name = _name(rows, cfg, flags, pdl)
            fn = call_fn(cfg, flags, pdl)
            if "timed" in flags:
                timed.append((name, capture(chain(fn, n_warm)), cfg))
                continue
            warm[name] = (capture(chain(fn, n_warm)), n_warm)
            cold[name] = (capture(chain(fn, LAYERS)), LAYERS)
        tw = time_graphs(warm, rounds=args.rounds)
        tc = time_graphs(cold, rounds=args.rounds, flush=Flush())
        for name in warm:
            res[name] = dict(warm=tw[name]["median"], cold=tc[name]["median"])
            log(f"[ablate] {name:40s} warm {tw[name]['median']:6.2f} cold {tc[name]['median']:6.2f}")
        for name, g, cfg in timed:
            n_ctas = cfg.num_splits * cases.HKV * rows
            for _ in range(3):
                g.replay()
            torch.cuda.synchronize()
            per_rep = []
            for _ in range(5):
                dbg.zero_()
                g.replay()
                torch.cuda.synchronize()
                per_rep.append(_timeline(dbg, n_warm, n_ctas))
            tl_med = {k: statistics.median([r[k] for r in per_rep]) for k in per_rep[0]}
            res[name] = tl_med
            log(f"[ablate] {name:40s} timeline (us, median call): " + ", ".join(
                f"{k} {v:.2f}" for k, v in tl_med.items()))
        del case, outs
        torch.cuda.empty_cache()
    res["elapsed_s"] = time.time() - t_start
    with open(os.path.join(RESULTS, f"{args.label}.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(RESULTS, f"{args.label}.log"), "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"[ablate] done in {res['elapsed_s']:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precompile", action="store_true")
    ap.add_argument("--label", default="abl")
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    if args.precompile:
        precompile()
    else:
        gpu_main(args)


if __name__ == "__main__":
    main()
