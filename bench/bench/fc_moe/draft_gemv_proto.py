"""fc-moe: draft-side (T=1) route-parallel W4A16 NVFP4 MoE GEMV prototype, timing only.

Replaces prologue + GEMM1 + doActivation + GEMM2 of the T=1 draft MoE call with two kernels:

  k1: grid (I/BN1, R)   act[r, i] = silu(gs1[e_r] * <W1_gate[e_r, i], x>) * gs1[e_r] * <W1_up[e_r, i], x>
  k2: grid (H/BN2, S2)  out[n]    = sum_r w_r * gs2[e_r] * <W2[e_r, n], act[r]>   (routes split S2 ways,
                                    fp32 partials, last-arriver reduce -> deterministic)

At T=1 the route-parallel traffic equals the grouped traffic (routes == distinct experts), so this
reads exactly the bytes CUTLASS reads.  Weights are random codes with *linear* e4m3 block scales
(the CUTLASS copy uses the 128x4 swizzled layout; reading that in-kernel is the same bytes, only
the index arithmetic differs).  All 512 experts are resident and each captured call draws a fresh
set of 10 experts so every call streams from DRAM, as in the server (1.4 GB >> 128 MB L2).

Usage (under the GPU lock): python -m fc_moe.draft_gemv_proto [--sweep]
"""
import argparse, itertools, json, statistics as st, sys

import torch
import triton
import triton.language as tl

from sglang.srt.layers.quantization.w4a16_nvfp4_gemv import _unpack_dequant, _FP4_TRICK

E, H, I, R = 512, 2560, 640, 10


@triton.jit
def _scales(s_ptr, rows, kb0, stride, BN: tl.constexpr, BK: tl.constexpr):
    NB: tl.constexpr = BK // 16
    s = tl.load(s_ptr + rows[:, None] * stride + (kb0 + tl.arange(0, NB))[None, :]).to(tl.bfloat16)
    s = tl.broadcast_to(s[:, :, None], (BN, NB, 16))
    return tl.reshape(s, (BN, BK))


@triton.jit
def _k1(x_ptr, ids_ptr, q_ptr, s_ptr, g_ptr, act_ptr,
        K: tl.constexpr, I: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_n = tl.program_id(0)
    r = tl.program_id(1)
    e = tl.load(ids_ptr + r).to(tl.int64)
    BKB: tl.constexpr = BK // 2
    rows = pid_n * BN + tl.arange(0, BN)
    qb = q_ptr + e * (2 * I * (K // 2))
    sb = s_ptr + e * (2 * I * (K // 16))
    accg = tl.zeros((BN,), tl.float32)
    accu = tl.zeros((BN,), tl.float32)
    offs_kb = tl.arange(0, BKB)
    for k0 in range(0, K, BK):
        xv = tl.load(x_ptr + k0 + tl.arange(0, BK)).to(tl.float32)
        wg = _unpack_dequant(tl.load(qb + rows[:, None] * (K // 2) + (k0 // 2 + offs_kb)[None, :]), BN, BKB)
        wu = _unpack_dequant(tl.load(qb + (rows + I)[:, None] * (K // 2) + (k0 // 2 + offs_kb)[None, :]), BN, BKB)
        wg = wg * _scales(sb, rows, k0 // 16, K // 16, BN, BK)
        wu = wu * _scales(sb, rows + I, k0 // 16, K // 16, BN, BK)
        accg += tl.sum(wg.to(tl.float32) * xv[None, :], axis=1)
        accu += tl.sum(wu.to(tl.float32) * xv[None, :], axis=1)
    g = tl.load(g_ptr + e) * _FP4_TRICK
    accg = accg * g
    accu = accu * g
    a = accg / (1.0 + tl.exp(-accg)) * accu
    tl.store(act_ptr + r * I + rows, a.to(tl.bfloat16))


@triton.jit
def _k2(act_ptr, ids_ptr, w_ptr, q_ptr, s_ptr, g_ptr, out_ptr, ws_ptr, cnt_ptr,
        H: tl.constexpr, I: tl.constexpr, R: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
        S2: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    BKB: tl.constexpr = BK // 2
    rows = pid_n * BN + tl.arange(0, BN)
    offs_kb = tl.arange(0, BKB)
    acc = tl.zeros((BN,), tl.float32)
    for r in range(pid_s, R, S2):
        e = tl.load(ids_ptr + r).to(tl.int64)
        qb = q_ptr + e * (H * (I // 2))
        sb = s_ptr + e * (H * (I // 16))
        part = tl.zeros((BN,), tl.float32)
        for k0 in range(0, I, BK):
            av = tl.load(act_ptr + r * I + k0 + tl.arange(0, BK)).to(tl.float32)
            w = _unpack_dequant(tl.load(qb + rows[:, None] * (I // 2) + (k0 // 2 + offs_kb)[None, :]), BN, BKB)
            w = w * _scales(sb, rows, k0 // 16, I // 16, BN, BK)
            part += tl.sum(w.to(tl.float32) * av[None, :], axis=1)
        acc += part * (tl.load(g_ptr + e) * _FP4_TRICK * tl.load(w_ptr + r))
    if S2 == 1:
        tl.store(out_ptr + rows, acc.to(tl.bfloat16))
        return
    base = ws_ptr + pid_n * (S2 * BN)
    tl.store(base + pid_s * BN + tl.arange(0, BN), acc, cache_modifier=".cg")
    tl.debug_barrier()
    done = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
    if done == S2 - 1:
        tot = tl.zeros((BN,), tl.float32)
        for s_i in tl.static_range(S2):
            tot += tl.load(base + s_i * BN + tl.arange(0, BN), cache_modifier=".cg")
        tl.store(out_ptr + rows, tot.to(tl.bfloat16))
        tl.store(cnt_ptr + pid_n, 0)


def reference(x, ids, wts, q1, s1, g1, q2, s2, g2):
    from sglang.srt.layers.quantization.w4a16_nvfp4_gemv import dequantize_nvfp4
    out = torch.zeros(H, dtype=torch.float32, device=x.device)
    for r in range(R):
        e = int(ids[r])
        W1 = dequantize_nvfp4(q1[e], s1[e].view(torch.uint8), g1[e:e + 1])
        h = W1 @ x.float()
        a = torch.nn.functional.silu(h[:I]) * h[I:]
        a = a.bfloat16().float()
        W2 = dequantize_nvfp4(q2[e], s2[e].view(torch.uint8), g2[e:e + 1])
        out += wts[r] * (W2 @ a)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--calls", type=int, default=48)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(0)
    q1 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=dev)
    s1 = (torch.rand(E, 2 * I, H // 16, device=dev) * 2 + 0.5).to(torch.float8_e4m3fn)
    q2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=dev)
    s2 = (torch.rand(E, H, I // 16, device=dev) * 2 + 0.5).to(torch.float8_e4m3fn)
    g1 = torch.rand(E, device=dev) * 1e-3
    g2 = torch.rand(E, device=dev) * 1e-3
    x = torch.randn(H, device=dev, dtype=torch.bfloat16)
    C = a.calls
    ids = torch.stack([torch.randperm(E, device=dev)[:R] for _ in range(C)]).to(torch.int32)
    wts = torch.rand(C, R, device=dev).softmax(-1)
    act = torch.empty(C, R, I, device=dev, dtype=torch.bfloat16)
    out = torch.empty(C, H, device=dev, dtype=torch.bfloat16)
    ws = torch.empty(1 << 20, device=dev, dtype=torch.float32)
    cnt = torch.zeros(4096, device=dev, dtype=torch.int32)
    flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)

    def run(c, cfg):
        bn1, bk1, w1, st1, bn2, bk2, s2n, w2, st2 = cfg
        _k1[(I // bn1, R)](x, ids[c], q1, s1, g1, act[c], K=H, I=I, BN=bn1, BK=bk1,
                           num_warps=w1, num_stages=st1)
        _k2[(H // bn2, s2n)](act[c], ids[c], wts[c], q2, s2, g2, out[c], ws, cnt,
                             H=H, I=I, R=R, BN=bn2, BK=bk2, S2=s2n, num_warps=w2, num_stages=st2)

    def timeit(cfg, split=False):
        # one graph = C calls with distinct expert sets; per-kernel times from events per call
        run(0, cfg); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for c in range(C):
                run(c, cfg)
        ts = []
        for _ in range(7):
            flush.zero_()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1000 / C)
        return st.median(ts[1:])

    base = (16, 512, 4, 2, 16, 128, 2, 4, 2)
    # correctness vs fp32 reference on call 0
    run(0, base); torch.cuda.synchronize()
    ref = reference(x, ids[0], wts[0], q1, s1, g1, q2, s2, g2)
    rel = ((out[0].float() - ref).norm() / ref.norm()).item()
    print(json.dumps({"check_rel_err": rel}), flush=True)
    res = []
    if a.sweep:
        grid1 = [(bn, bk, w, s) for bn in (8, 16, 32) for bk in (256, 512) for w in (4, 8) for s in (2, 3)]
        grid2 = [(bn, bk, sp, w, s) for bn in (8, 16, 32) for bk in (128,) for sp in (1, 2, 5, 10) for w in (2, 4) for s in (2, 3)]
        best1 = None
        for c1 in grid1:
            cfg = c1 + base[4:]
            try:
                t = timeit(cfg)
            except Exception as ex:  # noqa: BLE001
                print("k1 fail", c1, str(ex)[:80]); continue
            res.append({"cfg": cfg, "us": t})
            if best1 is None or t < best1[1]:
                best1 = (c1, t)
        print(json.dumps({"best_k1_stage": best1}), flush=True)
        best = None
        for c2 in grid2:
            cfg = best1[0] + c2
            try:
                t = timeit(cfg)
            except Exception as ex:  # noqa: BLE001
                print("k2 fail", c2, str(ex)[:80]); continue
            res.append({"cfg": cfg, "us": t})
            if best is None or t < best[1]:
                best = (cfg, t)
        print(json.dumps({"best": best}), flush=True)
        base = best[0]
    # final: split timing k1 alone vs k1+k2
    t_all = timeit(base)
    bn1, bk1, w1, st1 = base[:4]

    def k1only(c, cfg):
        _k1[(I // bn1, R)](x, ids[c], q1, s1, g1, act[c], K=H, I=I, BN=bn1, BK=bk1,
                           num_warps=w1, num_stages=st1)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for c in range(C):
            k1only(c, base)
    ts = []
    for _ in range(7):
        flush.zero_()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000 / C)
    t1 = st.median(ts[1:])
    b1 = R * 2 * I * (H // 2 + H // 16)
    b2 = R * H * (I // 2 + I // 16)
    rec = {"cfg": base, "us_total": t_all, "us_k1": t1, "us_k2": t_all - t1,
           "GBs_k1": b1 / t1 / 1e3, "GBs_k2": b2 / (t_all - t1) / 1e3,
           "GBs_total": (b1 + b2) / t_all / 1e3, "check_rel_err": rel}
    print(json.dumps(rec), flush=True)
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps({"final": rec, "sweep": res}) + "\n")


if __name__ == "__main__":
    main()
