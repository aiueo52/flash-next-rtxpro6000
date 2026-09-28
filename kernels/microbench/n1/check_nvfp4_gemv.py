"""Numerics check for w4a16_nvfp4_gemv against a torch reference.

Two separate claims are checked, because they can fail independently:

1. **The quantiser and the kernel agree on the format.** `dequantize_nvfp4` (a plain
   torch LUT gather, written from `modelopt_quant.py:810-828`) is the reference decoder.
   The kernel's fp16 bit-trick decode must reproduce it *bit-exactly* for all 16 codes at
   every block scale -- this is a discrete map, so anything but exactness is a bug, not a
   tolerance question.
2. **The GEMV matches a reference matmul on the dequantised weight.** Here the fp32
   accumulation order differs (tiled/split-K vs cuBLAS), so the comparison is a relative
   error against an fp64 reference, reported alongside the same figure for the FP8 kernel
   at the same shape -- the FP8 number is the bar, since that path is what ships today.
"""

import os
import sys

import torch

sys.path.insert(0, "/home/user/tools/sglang-n1/python")

from sglang.srt.layers.quantization.w4a16_nvfp4_gemv import (  # noqa: E402
    _E2M1_LEVELS,
    dequantize_nvfp4,
    quantize_nvfp4,
    w4a16_nvfp4_gemv,
)
from sglang.srt.layers.quantization.w8a16_gemv import w8a16_gemv  # noqa: E402

DEV = "cuda"
torch.manual_seed(0)


#: Code 8 is *negative zero*. `quantize_nvfp4` emits the sign bit from `a < 0`, which is
#: False for -0.0, so a well-behaved encoder never produces it -- +0 and -0 both map to
#: code 0. Its absence is correct, not a gap in coverage.
_UNREACHABLE_CODES = (8,)


def check_code_coverage():
    """Every reachable code must decode exactly, at a spread of block scales.

    The scales are powers of two spanning 16x on purpose. The global scale is shared by
    the whole tensor, so a wide spread would push the small rows' block scales toward
    e4m3 underflow and the round trip would stop being exact for reasons that have
    nothing to do with the codec -- which is what a first version of this check ran
    into (17000x spread, 2.1e-3 error). Here every block scale (28, 56, 112, 224, 448)
    is exactly representable in e4m3, so exactness is a real assertion about the codec.
    """
    lvl = torch.tensor(_E2M1_LEVELS, dtype=torch.float32)
    row = torch.cat([lvl, -lvl])  # 16 values: all 8 magnitudes x both signs
    scales = [0.25, 0.5, 1.0, 2.0, 4.0]
    w = torch.stack([row * s for s in scales]).to(DEV).bfloat16()
    wq, bs, gs = quantize_nvfp4(w)
    deq = dequantize_nvfp4(wq, bs, gs)
    # exact by construction: each row's values are exactly representable
    rel = ((deq - w.float()).abs().max() / w.float().abs().max()).item()
    codes_seen = set(int(c) for c in torch.cat([wq & 0x0F, wq >> 4]).flatten().tolist())
    return rel, sorted(codes_seen)


def check_kernel_vs_reference(M, N, K, per_channel=False):
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    w = (torch.randn(N, K, device=DEV, dtype=torch.float32) * 0.02).bfloat16()

    wq, bs, gs = quantize_nvfp4(w)
    deq = dequantize_nvfp4(wq, bs, gs)  # fp32, the exact value the kernel must use

    y = w4a16_nvfp4_gemv(x, wq, bs, gs)
    # Reference: same dequantised weight, fp64 accumulation.
    ref = (x.double() @ deq.double().t())
    err = ((y.float().double() - ref).abs() / ref.abs().clamp(min=1e-9)).max().item()
    rms = ((y.float().double() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()

    # Same figure for the FP8 kernel, as the bar.
    amax = w.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
    s8 = amax / 448.0
    w8 = (w.float() / s8).clamp(-448, 448).to(torch.float8_e4m3fn)
    y8 = w8a16_gemv(x, w8, s8.reshape(-1))
    ref8 = (x.double() @ (w8.float().double() * s8.double()).t())
    err8 = ((y8.float().double() - ref8).abs() / ref8.abs().clamp(min=1e-9)).max().item()
    rms8 = ((y8.float().double() - ref8).pow(2).mean().sqrt() / ref8.pow(2).mean().sqrt()).item()

    # And the quantisation error itself (nvfp4 vs fp8 vs bf16), the quality-relevant number.
    qerr4 = ((deq - w.float()).pow(2).mean().sqrt() / w.float().pow(2).mean().sqrt()).item()
    qerr8 = (((w8.float() * s8) - w.float()).pow(2).mean().sqrt() / w.float().pow(2).mean().sqrt()).item()
    return dict(kernel_max_rel=err, kernel_rms=rms, fp8_kernel_max_rel=err8,
                fp8_kernel_rms=rms8, weight_rms_nvfp4=qerr4, weight_rms_fp8=qerr8)


def check_bitexact_decode(N=256, K=2560):
    """The kernel's decode must equal the reference decode bit-for-bit.

    Isolated from the dot by using a one-hot activation: y[0, n] = deq[n, k0] exactly
    (a single product, no accumulation, no reassociation), for every k0 in a stride
    that covers all 16 lanes of a block and several blocks.
    """
    w = (torch.randn(N, K, device=DEV, dtype=torch.float32) * 0.05).bfloat16()
    wq, bs, gs = quantize_nvfp4(w)
    deq = dequantize_nvfp4(wq, bs, gs)
    bad = 0
    tested = 0
    for k0 in list(range(0, 16)) + [16, 17, 31, 100, 1279, 2559]:
        x = torch.zeros(1, K, device=DEV, dtype=torch.bfloat16)
        x[0, k0] = 1.0
        y = w4a16_nvfp4_gemv(x, wq, bs, gs)[0].float()
        want = deq[:, k0].bfloat16().float()  # kernel rounds the fp32 acc to bf16
        bad += int((y != want).sum())
        tested += N
    return bad, tested


def smoke_compile():
    """Compile+launch every tile shape the planner can emit, before anything else.

    A Triton compile error costs a whole GPU-lock window if it only surfaces after the
    codec tests, so this runs first. It also *collects* failures instead of raising on
    the first one: two windows were already spent learning one compile error at a time,
    and the two scale-broadcast paths generate different IR, so one of them failing is
    exactly when you most want to know whether the other works.
    """
    import sglang.srt.layers.quantization.w4a16_nvfp4_gemv as K

    w = (torch.randn(256, 512, device=DEV) * 0.02).bfloat16()
    wq, bs, gs = quantize_nvfp4(w)
    ok, bad = [], []
    for gather in (False, True):
        K.SCALE_GATHER = gather
        for M in (1, 4, 16):
            for cfg in ((32, 128, 1, M > 1, 4, 3), (64, 256, 1, True, 8, 3),
                        (16, 128, 2, M > 1, 4, 3)):
                x = torch.randn(M, 512, device=DEV, dtype=torch.bfloat16)
                tag = f"gather={gather} M={M} cfg={cfg}"
                try:
                    w4a16_nvfp4_gemv(x, wq, bs, gs, cfg=cfg)
                    torch.cuda.synchronize()
                    ok.append(tag)
                except Exception as e:
                    first = str(e).strip().splitlines()
                    bad.append((tag, f"{type(e).__name__}: {first[-1][:150] if first else ''}"))
    K.SCALE_GATHER = False
    for tag, err in bad:
        print(f"  [smoke FAIL] {tag}\n              {err}")
    print(f"[smoke] {len(ok)} ok, {len(bad)} failed"
          + (f"; gather=False ok={sum('gather=False' in t for t in ok)}/9"
             f" gather=True ok={sum('gather=True' in t for t in ok)}/9"))
    if not ok:
        raise SystemExit("every tile variant failed to compile; aborting")
    return len(ok), len(bad)


if __name__ == "__main__":
    print(f"torch {torch.__version__} device {torch.cuda.get_device_name(0)}")
    smoke_compile()
    rel, codes = check_code_coverage()
    print(f"[codes] all-16-code round trip: max rel err {rel:.3e}, codes seen {codes}")
    want = [c for c in range(16) if c not in _UNREACHABLE_CODES]
    assert codes == want, f"expected codes {want}, saw {codes}"
    assert rel == 0.0, f"exactly-representable weights must round-trip exactly, got {rel}"

    bad, tested = check_bitexact_decode()
    print(f"[decode] one-hot bit-exactness vs reference dequant: {bad}/{tested} mismatched")

    for (M, N, K) in [(1, 49152, 2560), (4, 49152, 2560), (16, 49152, 2560),
                      (1, 2560, 2560), (16, 13312, 2560)]:
        r = check_kernel_vs_reference(M, N, K)
        print(f"[gemv] M={M:2d} N={N:6d} K={K}: "
              f"nvfp4 kernel rms {r['kernel_rms']:.2e} (fp8 {r['fp8_kernel_rms']:.2e}) | "
              f"weight rms nvfp4 {r['weight_rms_nvfp4']:.4f} fp8 {r['weight_rms_fp8']:.4f}")
    print("OK" if bad == 0 else f"DECODE MISMATCH: {bad}")
