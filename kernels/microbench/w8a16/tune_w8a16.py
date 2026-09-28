"""Config sweep for the W8A16 skinny GEMV. Prints _TUNED table entries.

Usage: tune_w8a16.py <shape_name> [<shape_name> ...] [--m 1,16]
Run one shape per process: a full sweep is dominated by Triton compilation.
"""

import argparse, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../python"))

import torch, triton
from harness import make_copies, n_copies, require_idle_gpu, time_graph
from shapes import SHAPES, make_weight

from sglang.srt.layers.quantization import w8a16_gemv as mod


def coarse_configs(M, N, K, contig_n, sms, w_bytes):
    bns = [16, 32, 64, 128, 256] if contig_n else [8, 16, 32, 64, 128]
    out = []
    for bn in bns:
        if M > 1 and bn < 16:
            continue
        if bn >= 2 * triton.next_power_of_2(N) and bn > 16:
            continue
        n_tiles = triton.cdiv(N, bn)
        for bk in (64, 128, 256, 512):
            if bn * bk > 65536 or bk > 2 * triton.next_power_of_2(K):
                continue
            if M == 1 and bn * bk > 16384:  # fp32 broadcast temp lives in registers
                continue
            if mod._stage_bytes(bn, bk, M, w_bytes) > mod._SMEM_PER_STAGE:
                continue
            n_kblk = triton.cdiv(K, bk)
            for sk in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40):
                if n_kblk % sk or sk > n_kblk:
                    continue
                ctas = n_tiles * sk
                if sk > 1 and not (sms // 2 <= ctas <= 6 * sms):
                    continue
                if sk == 1 and ctas < sms // 8:
                    continue
                out.append((bn, bk, sk, 4, min(4, max(2, n_kblk // sk))))
    return out


def refine_configs(base):
    bn, bk, sk, _, ns0 = base
    return [(bn, bk, sk, nw, ns) for nw in (2, 4, 8) for ns in {2, ns0, 4}]


def run(name, N, K, tag, Ms, iters, budget_s):
    dev = "cuda"
    w, scale, wref = make_weight(N, K, tag, dev)
    esz = 1 if tag == "fp8" else 2
    ncp = n_copies(N * K * esz)
    copies = make_copies(w, ncp)
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    contig_n = w.stride(0) == 1
    gib = N * K * esz / 1e9
    print(f"# {name}: N={N} K={K} {tag} contig_n={contig_n} copies={ncp} "
          f"({ncp * N * K * esz / 1e6:.0f} MB)", flush=True)

    for M in Ms:
        x = (torch.randn(M, K, device=dev) * 0.5).to(torch.bfloat16)
        ref = x.float() @ wref.t()
        fn = (lambda a, s=scale: mod.w8a16_gemv(x, a, s, cfg=cfg)) if tag == "fp8" else \
             (lambda a: mod.bf16_gemv(x, a, cfg=cfg))
        results = []
        t0 = time.time()
        cands = coarse_configs(M, N, K, contig_n, sms, 1 if tag == 'fp8' else 2)
        stage = "coarse"
        i = 0
        while i < len(cands):
            cfg = cands[i]
            i += 1
            try:
                y = fn(copies[0])
                err = (y.float() - ref).abs().max().item()
                if not (err == err):
                    continue
                dt = time_graph(fn, copies, iters=iters)
            except Exception as e:
                print(f"#   skip {cfg}: {type(e).__name__}: {str(e)[:90]}", flush=True)
                continue
            results.append((dt, cfg, err))
            if stage == "coarse" and i == len(cands):
                results.sort()
                stage = "refine"
                cands = cands + [c for b in results[:2] for c in refine_configs(b[1])]
            if time.time() - t0 > budget_s:
                print(f"#   (budget hit after {i} configs)", flush=True)
                break
        results.sort()
        best = results[0]
        for dt, cfg, err in results[:5]:
            print(f"#   M={M:2d} {cfg} {dt*1e6:8.2f} us  {gib/dt/1e3:5.2f} TB/s  err={err:.3e}",
                  flush=True)
        print(f"    ({contig_n}, {mod._m_bucket(M)}, {N}, {K}): {best[1]},  "
              f"# {name} M={M} {best[0]*1e6:.1f}us", flush=True)
        del x, ref
    del copies, w, wref
    torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--m", default="1,16")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--budget", type=float, default=45.0, help="seconds per (shape, M)")
    a = ap.parse_args()
    require_idle_gpu()
    by_name = {s[0]: s for s in SHAPES}
    Ms = [int(v) for v in a.m.split(",")]
    for n in a.names:
        _, N, K, tag, _, _ = by_name[n]
        run(n, N, K, tag, Ms, a.iters, a.budget)
