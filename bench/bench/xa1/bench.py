"""xa1: GPU validation and CUDA-graph timing of QSA decode attention (XQA vs Triton candidates).

Run only through xa1/gpu_job.sh (GPU lock, private caches, timeout). Phases (--phases):

* validate: per shape, layer-0 outputs of the production path (valid counts + _compact_kv + XQA,
  the real QwenSparseAttnBackend._forward_trtllm_sparse), candidate A behind the same server
  function, and candidate B, against the fp32 reference; per-row max abs / rel error and how
  often the bf16 outputs differ from production.
* sweep: candidate configs per row count at the QSA cap (ctx 8192): B warm + cold, A on a hot
  packed scratch (it replaces XQA only).
* main: per shape, CUDA graphs of chained calls over 12 layers' KV, interleaved rounds, median
  per call. warm = 24 calls (2 passes) replayed back to back; cold = 12 calls after a 2 x 256 MB
  L2 flush (zero one buffer, read another, so no dirty lines remain). Chains: xqa / A alone on one
  hot packed scratch (the server's view: _compact_kv just wrote it), srv (production path),
  srvA (production path with A instead of XQA), B (gather, replaces all three kernels).
* profile: torch.profiler kernel durations, each call isolated (sync before and after).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch

from xa1 import cases
from xa1.fi import load_trtllm_decode
from xa1.grid import (CTXS, MAX_GROUPS, MAX_SPLITS, R2T_WIDTH, ROWS, TOPK_SHARED_W4,
                      TOPK_SHARED_W16, a_configs, parse_cfg, sweep_configs)
from xa1.kernels import (XA1Config, XA1Workspace, xa1_decode_gather, xa1_decode_packed,
                         xqa_compatible)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
LAYERS = 12
# v4 defaults (vc, K/V double-buffered, lse combine); the sweep replaces them. job3 (v3) picked
# 11/64/4/2/h/vc for 1-8 rows and 5/64/4/3 for 16; job2 (v2) 12/64/4/3, 12/64/4/3, 10/64/4/3,
# 5/64/8/3; job1 (v1) ran 16/12/8/4 splits with 64/8/2 tiles.
DEFAULT_B = {1: XA1Config(11, 64, 4, 2, vcvt=True, lse=True),
             4: XA1Config(11, 64, 4, 2, vcvt=True, lse=True),
             8: XA1Config(11, 64, 4, 2, vcvt=True, lse=True),
             16: XA1Config(5, 64, 4, 2, vcvt=True, lse=True)}
DEFAULT_A = dict(DEFAULT_B)


class Shape:
    def __init__(self, rows, ctx, mode="fresh", n_tail=8, tail_width=16):
        self.rows, self.ctx, self.mode, self.n_tail, self.tail_width = (
            rows, ctx, mode, n_tail, tail_width)

    @property
    def name(self):
        if self.mode == "fresh":
            return f"R{self.rows}-ctx{self.ctx}"
        return f"R{self.rows}-ctx{self.ctx}-shared{2051 + self.tail_width}-t{self.n_tail}"

    def build(self, *, layers, seed, overlap):
        return cases.Case(rows=self.rows, ctx=self.ctx, layers=layers, mode=self.mode,
                          overlap=overlap, tail_width=self.tail_width, n_tail=self.n_tail,
                          device="cuda", seed=seed, r2t_width=R2T_WIDTH)


# ------------------------------------------------------------------ paths for one shape


class Paths:
    """Every implementation of one attention call over a Case, with the server's buffers."""

    def __init__(self, case, trtllm_decode, ws, cfg_a, cfg_b):
        self.case, self.dec, self.ws, self.cfg_a, self.cfg_b = case, trtllm_decode, ws, cfg_a, cfg_b
        rows = case.rows
        self.stub = cases.server_stub(case.req_to_token, rows)
        self.stub_a = cases.server_stub(case.req_to_token, rows)
        self.counts = torch.zeros(rows, dtype=torch.int32, device="cuda")
        self.counts_a = torch.zeros(rows, dtype=torch.int32, device="cuda")
        # One hot packed scratch (layer 0) for the XQA / A alone chains.
        lay0 = case.layers[0]
        self.srv(0)
        torch.cuda.synchronize()
        pk, pv = next(iter(self.stub._fa2_scratch.values()))
        topk = lay0.topk.shape[1]
        self.stride = (topk + cases.PAGE - 1) // cases.PAGE * cases.PAGE
        self.hot_k = pk[: rows * self.stride].clone()
        self.hot_v = pv[: rows * self.stride].clone()
        self.hot_counts = self.counts.clone()
        self.block_tables = (torch.arange(rows * self.stride // cases.PAGE, dtype=torch.int32,
                                          device="cuda").view(rows, -1).contiguous())
        self.kv_hnd = tuple(t.view(-1, cases.PAGE, cases.HKV, cases.D).permute(0, 2, 1, 3)
                            for t in (self.hot_k, self.hot_v))
        self.xqa_ws = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        self.dec_a = xqa_compatible(cfg_a, ws, pdl=True)

    def srv(self, li):
        lay = self.case.layers[li]
        return cases.server_call(self.stub, self.case, lay, self.counts, self.dec)

    def srv_a(self, li):
        lay = self.case.layers[li]
        return cases.server_call(self.stub_a, self.case, lay, self.counts_a, self.dec_a)

    def b(self, li, out=None):
        lay = self.case.layers[li]
        return xa1_decode_gather(lay.q, lay.k_pool, lay.v_pool, self.case.req_to_token,
                                 self.case.row_req, lay.topk, lay.seq_lens,
                                 bmm1_scale=cases.SCALING, bmm2_scale=1.0, cfg=self.cfg_b,
                                 ws=self.ws, pdl=True, prefix_valid=self.case.mode == "fresh",
                                 out=out)

    def xqa(self, li):
        return self.dec(query=self.case.layers[li].q, kv_cache=self.kv_hnd,
                        workspace_buffer=self.xqa_ws, block_tables=self.block_tables,
                        seq_lens=self.hot_counts, max_seq_len=self.stride,
                        bmm1_scale=cases.SCALING, bmm2_scale=1.0)

    def a(self, li, out=None, cfg=None):
        return xa1_decode_packed(self.case.layers[li].q, self.hot_k, self.hot_v, self.hot_counts,
                                 stride=self.stride, bmm1_scale=cases.SCALING, bmm2_scale=1.0,
                                 cfg=cfg or self.cfg_a, ws=self.ws, pdl=True, out=out)


# ------------------------------------------------------------------ numerics


def _ordered(x):
    i = x.contiguous().view(torch.int16).int()
    return torch.where(i < 0, -(i & 0x7FFF), i)


def _err(out, ref):
    o = out.float().reshape(ref.shape[0], -1)
    r = ref.float().reshape(ref.shape[0], -1)
    row_abs = (o - r).abs().amax(1)
    row_rel = row_abs / r.abs().amax(1).clamp_min(1e-12)
    return dict(max_abs=float(row_abs.max()), max_rel=float(row_rel.max()),
                mean_rel=float(row_rel.mean()))


def _diff(a, b):
    """How often bf16 outputs a and b differ (elements, rows) and by how many bf16 ulps."""
    a = a.reshape(b.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    ulps = (_ordered(a) - _ordered(b)).abs()
    return dict(frac_elems_neq=float((ulps > 0).float().mean()),
                frac_elems_gt1ulp=float((ulps > 1).float().mean()),
                frac_elems_gt4ulp=float((ulps > 4).float().mean()),
                frac_rows_neq=float((ulps > 0).any(1).float().mean()),
                max_ulp=int(ulps.max()), max_abs=float((a.float() - b.float()).abs().max()))


def validate_shape(paths):
    case = paths.case
    lay = case.layers[0]
    ref = cases.reference(case, lay)
    srv = paths.srv(0).view_as(lay.q).clone()
    srv_a = paths.srv_a(0).view_as(lay.q).clone()
    b = paths.b(0).clone()
    torch.cuda.synchronize()
    pk, pv = next(iter(paths.stub._fa2_scratch.values()))
    counts = paths.counts.cpu()
    contract = cases.prefix_reference(case, lay, pk, pv, counts, paths.stride)
    valid = case.valid_mask(lay)
    return dict(
        valid_cols=valid.sum(1).tolist(), packed_counts=counts.tolist(),
        srv_vs_ref=_err(srv, ref), srvA_vs_ref=_err(srv_a, ref), B_vs_ref=_err(b, ref),
        srv_vs_contract=_err(srv, contract), srvA_vs_contract=_err(srv_a, contract),
        contract_vs_ref=_err(contract, ref),
        srvA_vs_srv=_diff(srv_a, srv), B_vs_srv=_diff(b, srv),
        counters_zero=int(paths.ws.cnt.abs().sum()) == 0)


# ------------------------------------------------------------------ timing


class Flush:
    def __init__(self):
        self.w = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
        self.r = torch.ones(64 << 20, dtype=torch.int32, device="cuda")

    def __call__(self):
        self.w.zero_()
        self.r.amax()


def capture(fn):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def chain(call, n_calls, *, n_layers=LAYERS):
    def run():
        for i in range(n_calls):
            call(i % n_layers)
    return run


def time_graphs(graphs, *, rounds, flush=None):
    """graphs: name -> (CUDAGraph, calls). Interleaved, rotating order; median us per call."""
    names = list(graphs)
    ev = {n: [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(rounds)] for n in names}
    for g, _ in graphs.values():
        g.replay()
    torch.cuda.synchronize()
    for r in range(rounds):
        k = r % len(names)
        for n in names[k:] + names[:k]:
            if flush is not None:
                flush()
            s, e = ev[n][r]
            s.record()
            graphs[n][0].replay()
            e.record()
    torch.cuda.synchronize()
    out = {}
    for n in names:
        per = [s.elapsed_time(e) * 1e3 / graphs[n][1] for s, e in ev[n]]
        out[n] = dict(median=statistics.median(per), p10=sorted(per)[len(per) // 10],
                      p90=sorted(per)[(len(per) * 9) // 10])
    return out


def spin(seconds=1.0):
    """Bring clocks up before timing (a few big matmuls)."""
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    t0 = time.time()
    while time.time() - t0 < seconds:
        for _ in range(20):
            a = (a @ a).clamp_(-1, 1)
        torch.cuda.synchronize()


def time_shape(paths, *, rounds, kinds):
    warm, cold = {}, {}
    fns = dict(xqa=paths.xqa, A=paths.a, srv=paths.srv, srvA=paths.srv_a, B=paths.b)
    for k in kinds:
        warm[k] = (capture(chain(fns[k], 2 * LAYERS)), 2 * LAYERS)
        if k in ("srv", "srvA", "B"):
            cold[k] = (capture(chain(fns[k], LAYERS)), LAYERS)
    res = dict(warm=time_graphs(warm, rounds=rounds))
    if cold:
        res["cold"] = time_graphs(cold, rounds=rounds, flush=Flush())
    return res


# ------------------------------------------------------------------ phases


def _paths(shape, dec, ws, cfg_a, cfg_b, args, layers=LAYERS):
    case = shape.build(layers=layers, seed=args.seed + shape.rows * 7 + shape.ctx, overlap=args.overlap)
    return Paths(case, dec, ws, cfg_a, cfg_b)


def sweep_phase(dec, ws, args, log):
    best_a, best_b, rows_out = {}, {}, {}
    shapes = [Shape(r, args.sweep_ctx) for r in ROWS]
    shapes.append(Shape(1, args.sweep_ctx, mode="shared", n_tail=8, tail_width=16))
    for shape in shapes:
        p = _paths(shape, dec, ws, DEFAULT_A[shape.rows], DEFAULT_B[shape.rows], args)
        cfgs = sweep_configs(shape.rows)
        a_set = set(a_configs(shape.rows))
        warm, cold, warm_a = {}, {}, {}
        for c in cfgs:
            key = c.key
            p.cfg_b = c
            warm[key] = (capture(chain(p.b, 2 * LAYERS)), 2 * LAYERS)
            cold[key] = (capture(chain(p.b, LAYERS)), LAYERS)
            if shape.mode == "fresh" and c in a_set:
                warm_a[key] = (capture(chain(lambda li, c=c: p.a(li, cfg=c), 2 * LAYERS)),
                               2 * LAYERS)
        tw = time_graphs(warm, rounds=args.sweep_rounds)
        tc = time_graphs(cold, rounds=args.sweep_rounds, flush=Flush())
        score = {k: tw[k]["median"] + tc[k]["median"] for k in tw}
        kb = min(score, key=score.get)
        entry = dict(B_warm={k: v["median"] for k, v in tw.items()},
                     B_cold={k: v["median"] for k, v in tc.items()}, best_B=kb)
        if warm_a:
            ta = time_graphs(warm_a, rounds=args.sweep_rounds)
            ka = min(ta, key=lambda k: ta[k]["median"])
            entry.update(A_hot={k: v["median"] for k, v in ta.items()}, best_A=ka)
            best_a[shape.rows] = parse_cfg(ka)
            best_b[shape.rows] = parse_cfg(kb)
        else:
            best_b[("shared", shape.rows)] = parse_cfg(kb)
        rows_out[shape.name] = entry
        top = sorted(score, key=score.get)[:5]
        fam = {}
        picks = [("v3", lambda k: not k.endswith("/l")), ("lse vc", lambda k: k.endswith("/vc/l")),
                 ("lse plain", lambda k: k.endswith("/l") and "/vc" not in k)]
        picks += [(f"lse S{n}", lambda k, n=n: k.endswith("/l") and k.split("/")[0] == str(n))
                  for n in sorted({c.num_splits for c in cfgs if c.lse})]
        for name, pick in picks:
            ks = [k for k in score if pick(k)]
            if ks:
                fam[name] = min(ks, key=score.get)
        log(f"[sweep] {shape.name}: B best {kb} warm {tw[kb]['median']:.2f} cold "
            f"{tc[kb]['median']:.2f} | top5 " + ", ".join(
                f"{k}:{tw[k]['median']:.2f}/{tc[k]['median']:.2f}" for k in top)
            + " | by family " + ", ".join(
                f"{n} {k}:{tw[k]['median']:.2f}/{tc[k]['median']:.2f}" for n, k in fam.items())
            + (f" | A best {entry['best_A']} hot {entry['A_hot'][entry['best_A']]:.2f}"
               if warm_a else ""))
        del p
        torch.cuda.empty_cache()
    return best_a, best_b, rows_out


def main_shapes(args):
    shapes = [Shape(r, c) for r in ROWS for c in CTXS]
    for ctx in (1024, 4096, 8192):
        shapes.append(Shape(1, ctx, mode="shared", n_tail=8, tail_width=TOPK_SHARED_W16 - 2051))
    shapes.append(Shape(1, 8192, mode="shared", n_tail=2, tail_width=TOPK_SHARED_W4 - 2051))
    if args.quick:
        shapes = [s for s in shapes if s.ctx in (1024, 8192)]
    return shapes


def main_phase(dec, ws, args, best_a, best_b, log):
    out = {}
    for shape in main_shapes(args):
        cfg_b = best_b.get(("shared", 1)) if shape.mode == "shared" else None
        cfg_b = cfg_b or best_b.get(shape.rows) or DEFAULT_B[shape.rows]
        cfg_a = best_a.get(shape.rows) or DEFAULT_A[shape.rows]
        p = _paths(shape, dec, ws, cfg_a, cfg_b, args)
        val = validate_shape(p)
        tim = time_shape(p, rounds=args.rounds, kinds=("xqa", "A", "srv", "srvA", "B"))
        out[shape.name] = dict(cfg_A=cfg_a.key, cfg_B=cfg_b.key, validate=val, time=tim)
        w, c = tim["warm"], tim["cold"]
        log(f"[main] {shape.name:28s} warm us/call: xqa {w['xqa']['median']:6.2f} A "
            f"{w['A']['median']:6.2f} srv {w['srv']['median']:6.2f} srvA {w['srvA']['median']:6.2f}"
            f" B {w['B']['median']:6.2f} | cold: srv {c['srv']['median']:6.2f} srvA "
            f"{c['srvA']['median']:6.2f} B {c['B']['median']:6.2f} | rel vs ref: srv "
            f"{val['srv_vs_ref']['max_rel']:.1e} srvA {val['srvA_vs_ref']['max_rel']:.1e} B "
            f"{val['B_vs_ref']['max_rel']:.1e} | B!=srv elems {val['B_vs_srv']['frac_elems_neq']:.3f}"
            f" >1ulp {val['B_vs_srv']['frac_elems_gt1ulp']:.3f} max {val['B_vs_srv']['max_ulp']}ulp")
        del p
        torch.cuda.empty_cache()
    return out


def profile_phase(dec, ws, args, best_a, best_b, log):
    from torch.profiler import ProfilerActivity, profile

    out = {}
    for shape in [Shape(r, 8192) for r in ROWS] + [Shape(1, 8192, mode="shared")]:
        cfg_b = best_b.get(("shared", 1)) if shape.mode == "shared" else None
        cfg_b = cfg_b or best_b.get(shape.rows) or DEFAULT_B[shape.rows]
        p = _paths(shape, dec, ws, best_a.get(shape.rows) or DEFAULT_A[shape.rows], cfg_b, args)
        fns = dict(srv=p.srv, xqa=p.xqa, A=p.a, B=p.b)
        for f in fns.values():
            f(0)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(2):
                for f in fns.values():
                    for li in range(LAYERS):
                        f(li)
                        torch.cuda.synchronize()
        per = {}
        for e in prof.profiler.kineto_results.events():
            if e.device_type() == torch.autograd.DeviceType.CUDA and "Mem" not in e.name()[:8]:
                per.setdefault(e.name()[:60], []).append(e.duration_ns() / 1e3)
        out[shape.name] = {k: dict(n=len(v), median=statistics.median(v)) for k, v in per.items()}
        log(f"[profile] {shape.name}: " + "; ".join(
            f"{k[:28]} x{v['n']} {v['median']:.2f}" for k, v in out[shape.name].items()))
        del p
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--phases", default="validate,sweep,main,profile")
    ap.add_argument("--rounds", type=int, default=25)
    ap.add_argument("--sweep-rounds", type=int, default=9)
    ap.add_argument("--sweep-ctx", type=int, default=8192)
    ap.add_argument("--overlap", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cfg-a", action="append", default=[], help="rows=s/bn/w/st[/bf][/h]")
    ap.add_argument("--cfg-b", action="append", default=[],
                    help="rows=s/bn/w/st[/bf][/h] or shared=...")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    t0 = time.time()
    dec, info = load_trtllm_decode()
    prop = torch.cuda.get_device_properties(0)
    meta = dict(label=args.label, device=prop.name, sms=prop.multi_processor_count,
                l2_bytes=getattr(prop, "L2_cache_size", None), torch=torch.__version__,
                xqa=info, args=vars(args), pdl_env=os.environ.get("SGLANG_TRITON_PDL"))
    log(f"[xa1] {json.dumps(meta)}")
    ws = XA1Workspace(max_groups=MAX_GROUPS, max_splits=MAX_SPLITS, head_dim=cases.D,
                      device="cuda")
    best_a = {int(k): parse_cfg(v) for k, v in (s.split("=") for s in args.cfg_a)}
    best_b = {}
    for s in args.cfg_b:
        k, v = s.split("=")
        best_b[("shared", 1) if k == "shared" else int(k)] = parse_cfg(v)
    res = dict(meta=meta)
    phases = args.phases.split(",")
    spin(1.5)
    if "validate" in phases:
        # Small shapes first: catches a broken path before the long phases.
        val = {}
        for shape in (Shape(4, 600), Shape(16, 2300), Shape(1, 2601, mode="shared", n_tail=4)):
            p = _paths(shape, dec, ws, DEFAULT_A[shape.rows], DEFAULT_B[shape.rows], args,
                       layers=1)
            val[shape.name] = validate_shape(p)
            log(f"[validate] {shape.name}: {json.dumps(val[shape.name])}")
        res["validate"] = val
    if "sweep" in phases:
        sa, sb, res["sweep"] = sweep_phase(dec, ws, args, log)
        best_a.update({k: v for k, v in sa.items() if k not in best_a})
        best_b.update({k: v for k, v in sb.items() if k not in best_b})
    res["best_A"] = {str(k): v.key for k, v in best_a.items()}
    res["best_B"] = {str(k): v.key for k, v in best_b.items()}
    def save():
        res["elapsed_s"] = time.time() - t0
        with open(os.path.join(RESULTS, f"{args.label}.json"), "w") as f:
            json.dump(res, f, indent=1)
        with open(os.path.join(RESULTS, f"{args.label}.log"), "w") as f:
            f.write("\n".join(lines) + "\n")

    save()
    if "main" in phases:
        res["main"] = main_phase(dec, ws, args, best_a, best_b, log)
        save()
    if "profile" in phases:
        res["profile"] = profile_phase(dec, ws, args, best_a, best_b, log)
    log(f"[xa1] done in {time.time() - t0:.0f}s -> {RESULTS}/{args.label}.json")
    save()


if __name__ == "__main__":
    main()
