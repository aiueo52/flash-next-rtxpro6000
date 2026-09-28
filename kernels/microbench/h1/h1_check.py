"""H1 numerics + micro-benchmark for the two GEMV prologue/epilogue items.

  h1_check.py numerics   -- correctness only (works with the GPU busy)
  h1_check.py bench      -- CUPTI kernel medians over CUDA-graph replays

Item C (SGLANG_GEMV_BA_SPLIT_GRID): the two-destination store's column split is
re-gridded so no CTA straddles it. Every output column is still produced by one
CTA running the same k-loop, so the result must be torch.equal to the flag-off
one.

Item B (SGLANG_NORM_INTO_GEMV): the GDN gated RMSNorm is folded into the
out_proj GEMV's A-load. The fp32 sum of squares is re-associated, so the
comparison is against `rms_norm_gated` + `w8a16_gemv` in ulps of bf16.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../w8a16v2"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../python"))

import torch

HIDDEN = 2560
VALUE_DIM = 48 * 128
HEAD_V = 128
NUM_V = 48
MS = (1, 4, 16)


def _fp8_weight(N, K, seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    wb = (torch.randn(N, K, device=device, generator=g) * 0.02).float()
    scale = (wb.abs().amax(dim=1).clamp(min=1e-8) / 448.0).contiguous()
    q = (wb / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous()
    w = q.t().t()
    assert w.shape == (N, K) and w.stride() == (K, 1)
    return w, scale.reshape(N, 1)


def _bf16_weight(N, K, seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    return (torch.randn(N, K, device=device, generator=g) * 0.02).to(torch.bfloat16)


def _ulp_report(a, b):
    """(max abs ulp distance in bf16, mismatching element fraction)."""
    ai = a.view(torch.int16).to(torch.int32)
    bi = b.view(torch.int16).to(torch.int32)
    # map to a monotone ordering so ulp distance is meaningful across zero
    ai = torch.where(ai < 0, torch.tensor(-0x8000, device=ai.device) - ai, ai)
    bi = torch.where(bi < 0, torch.tensor(-0x8000, device=bi.device) - bi, bi)
    d = (ai - bi).abs()
    return int(d.max().item()), float((d != 0).float().mean().item())


def numerics():
    import sglang.srt.layers.quantization.w8a16_gemv as gv
    from sglang.kernels.ops.attention.fla.layernorm_gated import rms_norm_gated

    dev = "cuda"
    ok = True

    # ---- item C: segmented split grid ------------------------------------
    print("== item C: SGLANG_GEMV_BA_SPLIT_GRID (bit-exactness) ==")
    cases = [
        ("gdn_in_proj_ba", 96, HIDDEN, 48, "bf16"),
        ("unaligned 37/96", 96, HIDDEN, 37, "bf16"),
        ("qkvz(aligned)", 16384, HIDDEN, 10240, "fp8"),
        ("fp8 unaligned", 2560, 640, 700, "fp8"),
    ]
    for name, N, K, sp, tag in cases:
        if tag == "bf16":
            w = _bf16_weight(N, K)
            call = lambda x, o, o2: gv.bf16_gemv(x, w, out=o, out2=o2, split_n=sp)
        else:
            w, sc = _fp8_weight(N, K)
            call = lambda x, o, o2: gv.w8a16_gemv(x, w, sc, out=o, out2=o2, split_n=sp)
        for M in MS:
            x = (torch.randn(M, K, device=dev) * 0.1).to(torch.bfloat16)
            outs = []
            for flag in (False, True):
                gv.BA_SPLIT_GRID = flag
                o = torch.empty(M, sp, dtype=torch.bfloat16, device=dev)
                o2 = torch.empty(M, N - sp, dtype=torch.bfloat16, device=dev)
                call(x, o, o2)
                torch.cuda.synchronize()
                outs.append((o.clone(), o2.clone()))
            gv.BA_SPLIT_GRID = False
            same = torch.equal(outs[0][0], outs[1][0]) and torch.equal(
                outs[0][1], outs[1][1]
            )
            # and against the unsplit single-destination reference
            ref = (
                gv.bf16_gemv(x, w)
                if tag == "bf16"
                else gv.w8a16_gemv(x, w, sc)
            )
            ref_ok = torch.equal(ref[:, :sp], outs[1][0]) and torch.equal(
                ref[:, sp:], outs[1][1]
            )
            ok &= same and ref_ok
            print(
                f"  {name:16s} N={N:6d} split={sp:6d} M={M:2d}  "
                f"equal(off,on)={same}  equal(unsplit,on)={ref_ok}"
            )

    # ---- item B: fused gated RMSNorm prologue ----------------------------
    print("== item B: SGLANG_NORM_INTO_GEMV (<= 1 ulp) ==")
    gv.NORM_INTO_GEMV = True
    w, sc = _fp8_weight(HIDDEN, VALUE_DIM, seed=3)
    nw = (torch.randn(HEAD_V, device=dev) * 0.3 + 1.0).to(torch.bfloat16)
    # The served checkpoint has output_gate_type=sigmoid; swish is the fla
    # default, so both gates are covered.
    for act, M in [(a, m) for a in ("sigmoid", "swish") for m in MS]:
        x = (torch.randn(M * NUM_V, HEAD_V, device=dev) * 0.5).to(torch.bfloat16)
        z = (torch.randn(M * NUM_V, HEAD_V, device=dev) * 0.5).to(torch.bfloat16)
        normed = rms_norm_gated(
            x=x, weight=nw, bias=None, z=z, eps=1e-6, group_size=None,
            norm_before_gate=True, is_rms_norm=True, activation=act,
        )
        ref = gv.w8a16_gemv(normed.view(M, VALUE_DIM), w, sc)
        got = gv.w8a16_gemv_norm_gated(
            x.view(M, VALUE_DIM), w, sc, z.view(M, VALUE_DIM), nw, HEAD_V, 1e-6,
            sigmoid_gate=(act == "sigmoid"),
        )
        torch.cuda.synchronize()
        u, frac = _ulp_report(ref, got)
        rel = ((ref.float() - got.float()).abs().max()
               / ref.float().abs().max()).item()
        # Control: the same plain GEMV run on the fused plan's tile, to
        # separate the split-K re-partition from the norm re-association.
        cfg = gv._norm_plan(M, HIDDEN, VALUE_DIM, HEAD_V, 1, gv._num_sms(dev))
        yc = torch.empty(M, HIDDEN, dtype=torch.bfloat16, device=dev)
        gv._launch(normed.view(M, VALUE_DIM), w, sc.reshape(-1).float(), yc, M,
                   HIDDEN, VALUE_DIM, True, cfg)
        torch.cuda.synchronize()
        uc, fc = _ulp_report(ref, yc)
        un, fn = _ulp_report(yc, got)
        ok &= un <= 1
        print(
            f"  {act:7s} M={M:2d}  vs server-plan ref: max_ulp={u} ({frac*100:.2f}%)  "
            f"max_rel={rel:.2e}   | split-K retile alone: {uc} ulp ({fc*100:.2f}%)"
            f"   | norm fold alone: {un} ulp ({fn*100:.2f}%)   plan={cfg}"
        )
    gv.NORM_INTO_GEMV = False
    print("NUMERICS", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def bench():
    from harness import cupti_kernel_us, make_copies, n_copies
    import sglang.srt.layers.quantization.w8a16_gemv as gv
    from sglang.kernels.ops.attention.fla.layernorm_gated import rms_norm_gated

    dev = "cuda"
    print(f"{'case':38s} {'M':>3s} {'us/call':>9s}")

    # -- item C: in_proj_ba ------------------------------------------------
    N, K, sp = 96, HIDDEN, 48
    w = _bf16_weight(N, K)
    ws = make_copies(w, n_copies(w.numel() * 2))
    for M in MS:
        x = (torch.randn(M, K, device=dev) * 0.1).to(torch.bfloat16)
        o = torch.empty(M, sp, dtype=torch.bfloat16, device=dev)
        o2 = torch.empty(M, N - sp, dtype=torch.bfloat16, device=dev)
        for label, flag in (("ba split (off)", False), ("ba split-grid (on)", True)):
            gv.BA_SPLIT_GRID = flag
            t, _ = cupti_kernel_us(
                lambda ww: gv.bf16_gemv(x, ww, out=o, out2=o2, split_n=sp),
                ws, name_sub="_w8a16_gemv_kernel")
            print(f"{label:38s} {M:3d} {t*1e6:9.2f}")
        gv.BA_SPLIT_GRID = False
        o1 = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
        t, _ = cupti_kernel_us(lambda ww: gv.bf16_gemv(x, ww, out=o1), ws,
                               name_sub="_w8a16_gemv_kernel")
        print(f"{'ba unsplit (reference)':38s} {M:3d} {t*1e6:9.2f}")

    # -- item B: norm + out_proj -------------------------------------------
    w8, sc = _fp8_weight(HIDDEN, VALUE_DIM, seed=3)
    w8s = make_copies(w8, n_copies(w8.numel()))
    nw = (torch.randn(HEAD_V, device=dev) * 0.3 + 1.0).to(torch.bfloat16)
    gv.NORM_INTO_GEMV = True
    for M in MS:
        x = (torch.randn(M * NUM_V, HEAD_V, device=dev) * 0.5).to(torch.bfloat16)
        z = (torch.randn(M * NUM_V, HEAD_V, device=dev) * 0.5).to(torch.bfloat16)
        x2, z2 = x.view(M, VALUE_DIM), z.view(M, VALUE_DIM)
        normed = torch.empty_like(x)

        def unfused(ww):
            n = rms_norm_gated(x=x, weight=nw, bias=None, z=z, eps=1e-6,
                               group_size=None, norm_before_gate=True,
                               is_rms_norm=True, activation="swish")
            return gv.w8a16_gemv(n.view(M, VALUE_DIM), ww, sc)

        tn, _ = cupti_kernel_us(unfused, w8s, name_sub="_layer_norm_fwd_1pass")
        tg, _ = cupti_kernel_us(unfused, w8s, name_sub="_w8a16_gemv_kernel")
        tf, _ = cupti_kernel_us(
            lambda ww: gv.w8a16_gemv_norm_gated(x2, ww, sc, z2, nw, HEAD_V, 1e-6),
            w8s, name_sub="_w8a16_gemv_kernel")
        print(f"{'out_proj: norm kernel':38s} {M:3d} {tn*1e6:9.2f}")
        print(f"{'out_proj: gemv (unfused)':38s} {M:3d} {tg*1e6:9.2f}")
        print(f"{'out_proj: gemv+norm fused':38s} {M:3d} {tf*1e6:9.2f}"
              f"   delta={(tf-tn-tg)*1e6:+.2f}")
    gv.NORM_INTO_GEMV = False


def sweep():
    """Fused-norm out_proj over (BLOCK_N, SPLITS, warps, stages).

    The norm is recomputed once per n block, so the redundancy is
    cdiv(N, BLOCK_N); a wider n block trades that against a fatter tile.
    """
    from harness import cupti_kernel_us, make_copies, n_copies
    import sglang.srt.layers.quantization.w8a16_gemv as gv
    from sglang.kernels.ops.attention.fla.layernorm_gated import rms_norm_gated

    dev = "cuda"
    w8, sc = _fp8_weight(HIDDEN, VALUE_DIM, seed=3)
    w8s = make_copies(w8, n_copies(w8.numel()))
    nw = (torch.randn(HEAD_V, device=dev) * 0.3 + 1.0).to(torch.bfloat16)
    gv.NORM_INTO_GEMV = True
    n_kb = VALUE_DIM // HEAD_V  # 48 k blocks with BLOCK_K == group size
    cands = []
    for bn in (16, 32, 64, 128, 256):
        for sp in (1, 2, 3, 4, 6, 8, 12, 16, 24):
            if n_kb % sp:
                continue
            ctas = -(-HIDDEN // bn) * sp
            if not (120 <= ctas <= 1600):
                continue
            for warps in (4, 8):
                for stages in (2, 3, 4):
                    cands.append((bn, HEAD_V, sp, True, None, warps, stages))
    for M in (1, 4, 16):
        x = (torch.randn(M * NUM_V, HEAD_V, device=dev) * 0.5).to(torch.bfloat16)
        z = (torch.randn(M * NUM_V, HEAD_V, device=dev) * 0.5).to(torch.bfloat16)
        x2, z2 = x.view(M, VALUE_DIM), z.view(M, VALUE_DIM)

        def unfused(ww):
            n = rms_norm_gated(x=x, weight=nw, bias=None, z=z, eps=1e-6,
                               group_size=None, norm_before_gate=True,
                               is_rms_norm=True, activation="swish")
            return gv.w8a16_gemv(n.view(M, VALUE_DIM), ww, sc)

        tn, _ = cupti_kernel_us(unfused, w8s, name_sub="_layer_norm_fwd_1pass")
        tg, _ = cupti_kernel_us(unfused, w8s, name_sub="_w8a16_gemv_kernel")
        base = tn + tg
        print(f"-- M={M} baseline norm {tn*1e6:.2f} + gemv {tg*1e6:.2f} = {base*1e6:.2f} us")
        rows = []
        y = torch.empty(M, HIDDEN, dtype=torch.bfloat16, device=dev)
        for cfg in cands:
            try:
                gv._launch(x2, w8s[0], sc.reshape(-1).float(), y, M, HIDDEN,
                           VALUE_DIM, True, cfg,
                           norm=(z2, nw, HEAD_V, 1e-6, False))
                torch.cuda.synchronize()
            except Exception:
                continue
            try:
                t, _ = cupti_kernel_us(
                    lambda ww: gv._launch(x2, ww, sc.reshape(-1).float(), y, M,
                                          HIDDEN, VALUE_DIM, True, cfg,
                                          norm=(z2, nw, HEAD_V, 1e-6, False)),
                    w8s, name_sub="_w8a16_gemv_kernel")
            except Exception:
                continue
            rows.append((t, cfg))
        rows.sort()
        for t, cfg in rows[:10]:
            print(f"   {str(cfg):40s} {t*1e6:8.2f} us  vs base {base*1e6:6.2f} "
                  f"({(t-base)*1e6:+.2f})")
    gv.NORM_INTO_GEMV = False


def _sweep_plain(name, N, K, ms, tag="fp8"):
    """Plain (unfused) GEMV tile sweep for one shape, CUPTI medians."""
    import triton
    from harness import cupti_kernel_us, make_copies, n_copies
    import sglang.srt.layers.quantization.w8a16_gemv as gv

    dev = "cuda"
    if tag == "fp8":
        w, sc = _fp8_weight(N, K, seed=3)
        s1 = sc.reshape(-1).float()
        per_ch = True
        wsz = w.numel()
    else:
        w = _bf16_weight(N, K, seed=3)
        s1 = torch.ones(1, dtype=torch.float32, device=dev)
        per_ch = False
        wsz = w.numel() * 2
    ws = make_copies(w, n_copies(wsz))
    sms = gv._num_sms(dev)
    for M in ms:
        x = (torch.randn(M, K, device=dev) * 0.1).to(torch.bfloat16)
        y = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
        cur = gv._plan(M, N, K, False, 1 if tag == "fp8" else 2, sms)
        cands = {cur}
        for bn in (16, 32, 64, 128, 256):
            for bk in (64, 128, 256, 512):
                if bn * bk > 32768:
                    continue
                n_kb = triton.cdiv(K, bk)
                for sp in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32):
                    if sp > n_kb or n_kb % sp:
                        continue
                    ctas = triton.cdiv(N, bn) * sp
                    if not (150 <= ctas <= 2000):
                        continue
                    for warps in (4, 8):
                        for stages in (3, 4):
                            c = gv._fit(M, N, (bn, bk, sp, M > 1, None, warps, stages))
                            if c[2] == sp:
                                cands.add(c)
        rows = []
        for cfg in sorted(cands):
            try:
                gv._launch(x, ws[0], s1, y, M, N, K, per_ch, cfg)
                torch.cuda.synchronize()
                t, _ = cupti_kernel_us(
                    lambda ww: gv._launch(x, ww, s1, y, M, N, K, per_ch, cfg),
                    ws, name_sub="_w8a16_gemv_kernel")
            except Exception:
                continue
            rows.append((t, cfg))
        rows.sort()
        base = dict((c, t) for t, c in rows).get(cur, float("nan"))
        print(f"-- {name} N={N} K={K} M={M}  server plan {cur} = {base*1e6:.2f} us")
        for t, cfg in rows[:6]:
            print(f"   {str(cfg):40s} {t*1e6:8.2f} us  ({(t-base)*1e6:+.2f})")


def optune():
    _sweep_plain("o_proj", HIDDEN, VALUE_DIM, (1, 4, 16))
    _sweep_plain("gdn_in_proj_qkvz", 2 * 16 * 128 + 2 * VALUE_DIM, HIDDEN, (4, 16))
    _sweep_plain("attn_qkv", 24 * 2 * 256 + 2 * 2 * 256, HIDDEN, (4, 16))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "numerics"
    torch.cuda.init()
    if mode == "numerics":
        sys.exit(numerics())
    sys.exit({"bench": bench, "sweep": sweep, "optune": optune}[mode]() or 0)
