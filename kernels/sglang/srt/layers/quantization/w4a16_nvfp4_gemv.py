"""Triton W4A16 NVFP4 skinny GEMV: y[M,N] = x[M,K](bf16) @ dequant(W)[N,K]^T, M <= 16.

The weight is the same NVFP4 format the experts already use in this fork:

    W[n, k] = e2m1_code(n, k) * block_scale_e4m3(n, k // 16) * global_scale

stored as ``wq[N, K//2]`` uint8 (two 4-bit codes per byte, **low nibble = even k**),
``bs[N, K//16]`` uint8 holding the e4m3 bit patterns, and one fp32 global scale
(per tensor) or ``[N]`` (per output channel).

Structure is deliberately the same as ``w8a16_gemv``: one CTA per (n block, k split),
fp32 accumulator, split-K partials reduced in the same launch by whoever observes the
last counter increment, split-store epilogue, optional PDL.  What is new is the
weight A-load:

- **Unpack.** A ``[BLOCK_N, BLOCK_K//2]`` uint8 tile is split into low/high nibbles and
  ``tl.join``-ed back into ``[BLOCK_N, BLOCK_K]`` -- a pure register shuffle, no extra
  traffic, and it reproduces the standard even/odd nibble order.
- **Dequant by bit trick** (same as the in-house engine's ``nvfp4_linear.py``): the 4-bit code's
  mantissa bit and 2 exponent bits are shifted into fp16 bits [11:9] and the sign into
  [15], so a bitcast yields exactly ``e2m1_value * 2^-14``.  fp16's subnormal encoding
  makes the ``e == 0`` codes land on the same 2^-14 factor, so the map is exact for all
  16 codes with no table and no branch.  The ``2^14`` is folded into the *global* scale
  (applied once per output, outside the k loop), not into the per-16 block scale.
- **Exactness.** ``e2m1`` carries 2 significand bits and ``e4m3`` 4, so their product
  needs 6 -- bf16 has 8.  The dequantised weight is therefore *exact* in bf16, and the
  dot sees the same class of operand the FP8 path feeds it (an exactly-represented
  weight, fp32 accumulate).  The only inexactness versus a reference fp32 dequant is the
  final ``acc * global_scale``, which the FP8 path also has.

Bytes per weight element: 0.5 (codes) + 1/16 (block scale) = **0.5625**, against 1.0 for
FP8.  So at equal bandwidth efficiency this kernel is 1.78x faster; the FP8 kernel is
measured at 96-98 % of this card's 1615 GB/s read roof on the lm_head shapes
(EXCLUSIVE_TIME_MAP.md 5), so break-even needs ~55 % of roof here, not 40 %.
"""

import functools
import os
from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.triton_pdl import PDL, pdl_trigger, pdl_wait

FP4_SCALE_BLOCK = 16
#: The same 16, for use *inside* jitted code. Triton 3.7 refuses to read a plain int
#: module global from a kernel ("Cannot access global variable ... only global variables
#: instantiated as constexpr"), and an annotation (`x: tl.constexpr = 16`) does not
#: count -- it has to be a `tl.constexpr` value.
_FP4_BLK = tl.constexpr(16)
#: The bit trick decodes to `e2m1_value * 2^-14`; this undoes it. It is applied ONCE
#: per output in the epilogue, together with the global scale, rather than inside the k
#: loop -- the block scale stays the raw e4m3 value there, which keeps `fp4 * scale`
#: exactly representable in bf16 (2 + 4 significand bits against bf16's 8).
_FP4_TRICK = tl.constexpr(16384.0)
#: e2m1 -> the fp16 bit trick returns value * 2^-14; fold the reciprocal into the
#: global scale so the k loop never touches it.
FP4_TRICK_SCALE = 16384.0
#: e4m3 max (448) and e2m1 max (6): the ModelOpt global-scale convention.
E4M3_MAX = 448.0
E2M1_MAX = 6.0


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


@triton.jit
def _unpack_dequant(b, BLOCK_N: tl.constexpr, BLOCK_KB: tl.constexpr):
    """[BLOCK_N, BLOCK_KB] packed uint8 -> [BLOCK_N, 2*BLOCK_KB] bf16 = e2m1 * 2^-14.

    Low nibble is the even k, high nibble the odd k, matching the NVFP4 checkpoint
    layout.  ``tl.join`` puts them on a new trailing axis in exactly that order, so the
    reshape below is the identity permutation and costs no shuffle beyond the join.
    """
    v = b.to(tl.int32)
    lo = v & 0x0F
    hi = (v >> 4) & 0x0F
    # magnitude -> fp16 [11:9] (exponent low bits + mantissa top bit), sign -> [15].
    rl = ((lo << 9) & 0x0E00) | ((lo & 0x08) << 12)
    rh = ((hi << 9) & 0x0E00) | ((hi & 0x08) << 12)
    fl = rl.to(tl.uint16).to(tl.float16, bitcast=True)
    fh = rh.to(tl.uint16).to(tl.float16, bitcast=True)
    w = tl.join(fl, fh)  # [BLOCK_N, BLOCK_KB, 2], last axis = (even k, odd k)
    return tl.reshape(w, (BLOCK_N, 2 * BLOCK_KB)).to(tl.bfloat16)


@triton.jit
def _load_block_scales(
    s_ptr, offs_n, kb0, n_mask, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    stride_sn, NB: tl.constexpr, GATHER: tl.constexpr,
):
    """[BLOCK_N, BLOCK_K//16] e4m3 scales, broadcast to [BLOCK_N, BLOCK_K] bf16.

    GATHER False loads NB scales per row and expands them in registers; GATHER True
    re-reads each scale 16 times (an L1 hit) with no reshape. The first is fewer
    instructions, the second is the fallback if a Triton version refuses to reshape a
    broadcast value. They are numerically identical.

    One assignment and one return, deliberately: Triton type-checks *both* arms of a
    constexpr `if` that contains an early `return`, so a branch whose `other=` operand
    only makes sense for the taken arm is a compile error even when it is dead. (That is
    exactly how the E4M3-native branch this function used to carry failed -- `other=0`
    on an fp8 pointer.) The non-native e4m3 path is gone with it; this fork targets
    sm_120, where fp8e4nv loads are native.
    """
    if GATHER:
        offs_k = tl.arange(0, BLOCK_K)
        p = s_ptr + offs_n[:, None] * stride_sn + (kb0 + offs_k // _FP4_BLK)[None, :]
        out = tl.load(p, mask=n_mask[:, None], other=0.0).to(tl.bfloat16)
    else:
        p = s_ptr + offs_n[:, None] * stride_sn + (kb0 + tl.arange(0, NB))[None, :]
        s = tl.load(p, mask=n_mask[:, None], other=0.0).to(tl.bfloat16)
        # broadcast each scale over its 16 contiguous k
        s = tl.broadcast_to(s[:, :, None], (BLOCK_N, NB, _FP4_BLK))
        out = tl.reshape(s, (BLOCK_N, BLOCK_K))
    return out


@triton.jit
def _w4a16_nvfp4_gemv_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    g_ptr,
    y_ptr,
    y2_ptr,
    ws_ptr,
    cnt_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_qn,
    stride_sn,
    stride_ym,
    stride_yn,
    stride_y2m,
    SPLIT_N: tl.constexpr,
    PER_CHANNEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    M_PAD: tl.constexpr,
    SPLITS: tl.constexpr,
    USE_DOT: tl.constexpr,
    SCALE_GATHER: tl.constexpr = False,
    USE_PDL: tl.constexpr = False,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    BLOCK_KB: tl.constexpr = BLOCK_K // 2
    NB: tl.constexpr = BLOCK_K // _FP4_BLK

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, M_PAD)
    n_mask = offs_n < N
    m_mask = offs_m < M
    offs_kb = tl.arange(0, BLOCK_KB)

    pdl_wait(USE_PDL)
    acc = tl.zeros((M_PAD, BLOCK_N), dtype=tl.float32)
    # K is a multiple of BLOCK_K for every shape this kernel is planned for
    # (asserted host-side), so the k loop needs no masking at all.
    for k0 in range(pid_k * BLOCK_K, K, SPLITS * BLOCK_K):
        qb = tl.load(
            q_ptr + offs_n[:, None] * stride_qn + (k0 // 2 + offs_kb)[None, :],
            mask=n_mask[:, None],
            other=0,
        )
        w = _unpack_dequant(qb, BLOCK_N, BLOCK_KB)
        s = _load_block_scales(
            s_ptr, offs_n, k0 // _FP4_BLK, n_mask,
            BLOCK_N, BLOCK_K, stride_sn, NB, SCALE_GATHER,
        )
        w = w * s
        if USE_DOT:
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + (k0 + tl.arange(0, BLOCK_K))[None, :] * stride_xk,
                mask=m_mask[:, None],
                other=0.0,
            )
            acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)
        else:
            xv = tl.load(x_ptr + (k0 + tl.arange(0, BLOCK_K)) * stride_xk).to(tl.float32)
            acc += tl.sum(w.to(tl.float32) * xv[None, :], axis=1)[None, :]

    if SPLITS == 1:
        if PER_CHANNEL:
            acc = acc * (tl.load(g_ptr + offs_n, mask=n_mask, other=0.0) * _FP4_TRICK)[None, :]
        else:
            acc = acc * (tl.load(g_ptr) * _FP4_TRICK)
        _nvfp4_store(acc, y_ptr, y2_ptr, offs_m, offs_n, stride_ym, stride_yn,
                     stride_y2m, m_mask, n_mask, SPLIT_N)
        pdl_trigger(USE_PDL)
        return

    slot = offs_m[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    base = ws_ptr + (pid_n * SPLITS) * (M_PAD * BLOCK_N)
    tl.store(base + pid_k * (M_PAD * BLOCK_N) + slot, acc, cache_modifier=".cg")
    tl.debug_barrier()
    done = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
    if done == SPLITS - 1:
        tot = tl.zeros((M_PAD, BLOCK_N), dtype=tl.float32)
        for s_i in tl.static_range(SPLITS):
            tot += tl.load(base + s_i * (M_PAD * BLOCK_N) + slot, cache_modifier=".cg")
        if PER_CHANNEL:
            tot = tot * (tl.load(g_ptr + offs_n, mask=n_mask, other=0.0) * _FP4_TRICK)[None, :]
        else:
            tot = tot * (tl.load(g_ptr) * _FP4_TRICK)
        _nvfp4_store(tot, y_ptr, y2_ptr, offs_m, offs_n, stride_ym, stride_yn,
                     stride_y2m, m_mask, n_mask, SPLIT_N)
        tl.store(cnt_ptr + pid_n, 0)
    pdl_trigger(USE_PDL)


@triton.jit
def _nvfp4_store(acc, y_ptr, y2_ptr, offs_m, offs_n, stride_ym, stride_yn,
                 stride_y2m, m_mask, n_mask, SPLIT_N: tl.constexpr):
    """Same epilogue contract as w8a16_gemv._store_split (single/two destination)."""
    val = acc.to(tl.bfloat16).to(y_ptr.dtype.element_ty)
    if SPLIT_N > 0:
        lo = offs_n < SPLIT_N
        tl.store(
            y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
            val, mask=m_mask[:, None] & n_mask[None, :] & lo[None, :],
        )
        n2 = tl.maximum(offs_n - SPLIT_N, 0)
        tl.store(
            y2_ptr + offs_m[:, None] * stride_y2m + n2[None, :] * stride_yn,
            val, mask=m_mask[:, None] & n_mask[None, :] & (offs_n >= SPLIT_N)[None, :],
        )
    else:
        tl.store(
            y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
            val, mask=m_mask[:, None] & n_mask[None, :],
        )


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------

# The split-K scratch is shared with w8a16_gemv on purpose: the invariant that
# matters is "no two split-K launches overlap", which the decode path already
# guarantees by issuing every linear on one stream (and by scratch_slot(1) for
# the MoE alt stream). A second buffer would not make an overlap safe, it would
# only cost memory.
from sglang.srt.layers.quantization.w8a16_gemv import (  # noqa: E402
    _MAX_SPLITS,
    _WS_COUNTERS,
    _WS_FLOATS,
    _num_sms,
    _workspace,
)

#: Fallback scale broadcast (see _load_block_scales); set if the reshape path fails
#: to compile on a given Triton build.
SCALE_GATHER = _env_flag("SGLANG_NVFP4_GEMV_SCALE_GATHER")


def _m_bucket(M: int) -> int:
    return 1 if M == 1 else (4 if M <= 4 else 16)


# Measured with bench/n1/bench_nvfp4_gemv.py (CUPTI kernel-duration medians over a
# CUDA-graph replay whose weight working set is several times the 128 MB L2), the same
# metric bench/w8a16v2 uses for the FP8 table. Re-run after a kernel change.
#
# key:   (M bucket, N, K)  value: (BLOCK_N, BLOCK_K, SPLITS, USE_DOT, num_warps, num_stages)
#
# BLOCK_K 512 is the pattern here and it is not the FP8 table's: at 512 the per-16 block
# scales for one row are a 32 B tile, exactly one sector, where BLOCK_K 256 fetches 32 B
# to use 16. The scales are only 1/8 of the weight bytes but at the wrong granularity
# they cost a second sector each.
_TUNED: dict = {
    # draft lm_head, hot2_49152. Planner default (32,256,...) measured 56.19 us;
    # this is 52.64 us = 1344.6 GB/s = 83.3 % of the 1615 GB/s roof (sweep 2026-09-06).
    (1, 49152, 2560): (32, 512, 1, False, 4, 2),
}


def _plan(M: int, N: int, K: int, sms: int):
    tuned = _TUNED.get((_m_bucket(M), N, K))
    if tuned is not None:
        return _fit(M, N, tuned)
    # A wide-N shape (both lm_heads) already fills the machine on N alone; a long
    # BLOCK_K then keeps more bytes in flight and amortises the unpack over more
    # of the k loop.
    if triton.cdiv(N, 128) * 2 >= sms:
        if M == 1:
            return (32, 256, 1, False, 4, 3)
        return (64, 256, 1, True, 8, 3)
    use_dot = M > 1
    block_n = 16 if M == 1 else 32
    block_k = 128
    if triton.cdiv(N, block_n) >= sms:
        return (block_n, block_k, 1, use_dot, 4, 3)
    per_sm = 5 if M == 1 else 2
    want = min(_MAX_SPLITS, max(1, (per_sm * sms) // triton.cdiv(N, block_n)))
    n_kb = K // block_k
    splits = max(1, min(want, n_kb))
    for cand in range(min(n_kb, splits + 2), max(1, splits - 3), -1):
        if n_kb % cand == 0 and cand <= _MAX_SPLITS:
            splits = cand
            break
    return _fit(M, N, (block_n, block_k, splits, use_dot, 4, 3))


def _fit(M: int, N: int, cfg):
    block_n, block_k, splits, use_dot, warps, stages = cfg
    if splits == 1:
        return cfg
    m_pad = 16 if (use_dot or M > 1) else 1
    n_blocks = triton.cdiv(N, block_n)
    if (
        splits > _MAX_SPLITS
        or n_blocks > _WS_COUNTERS
        or n_blocks * splits * m_pad * block_n > _WS_FLOATS
    ):
        return (block_n, block_k, 1, use_dot, warps, stages)
    return cfg


def _out_or_new(out, M: int, N: int, device):
    if out is None:
        return torch.empty((M, N), dtype=torch.bfloat16, device=device)
    if (
        tuple(out.shape) != (M, N)
        or out.dtype not in (torch.bfloat16, torch.float32)
        or out.device != device
        or out.stride(1) != 1
    ):
        raise ValueError(
            f"nvfp4 gemv out= must be a [{M}, {N}] row-contiguous bf16/fp32 CUDA tensor"
        )
    return out


def w4a16_nvfp4_gemv(
    x: torch.Tensor,
    wq: torch.Tensor,
    bs: torch.Tensor,
    gscale: torch.Tensor,
    cfg=None,
    out: Optional[torch.Tensor] = None,
    out2: Optional[torch.Tensor] = None,
    split_n: int = 0,
):
    """x: [M,K] bf16; wq: [N,K//2] uint8; bs: [N,K//16] e4m3; gscale: fp32 scalar or [N].

    ``gscale`` is the *plain* global scale; the 2^14 of the dequant bit trick is applied
    here, not by the caller.
    """
    M, K = x.shape
    N = wq.shape[0]
    assert wq.shape[1] == K // 2 and bs.shape == (N, K // FP4_SCALE_BLOCK)
    assert M <= 16 and wq.stride(1) == 1 and bs.stride(1) == 1
    if out2 is None:
        y, y2, split_n = _out_or_new(out, M, N, x.device), None, 0
    else:
        if not (0 < split_n < N):
            raise ValueError(f"nvfp4 gemv split_n must be in (0, {N}); got {split_n}")
        y = _out_or_new(out, M, split_n, x.device)
        y2 = _out_or_new(out2, M, N - split_n, x.device)

    g = gscale.reshape(-1)
    if g.dtype != torch.float32 or not g.is_contiguous():
        g = g.contiguous().float()
    if cfg is None:
        cfg = _plan(M, N, K, _num_sms(x.device))
    block_n, block_k, splits, use_dot, num_warps, num_stages = cfg
    assert K % block_k == 0, f"K={K} must be a multiple of BLOCK_K={block_k}"
    use_dot = use_dot or M > 1
    m_pad = 16 if use_dot else 1
    ws, cnt = _workspace(x.device)
    sview = bs.view(torch.float8_e4m3fn)

    _w4a16_nvfp4_gemv_kernel[(triton.cdiv(N, block_n), splits)](
        x, wq, sview, g, y, y if y2 is None else y2, ws, cnt,
        M, N, K,
        x.stride(0), x.stride(1),
        wq.stride(0), bs.stride(0),
        y.stride(0), y.stride(1),
        y.stride(0) if y2 is None else y2.stride(0),
        SPLIT_N=split_n,
        PER_CHANNEL=g.numel() > 1,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        M_PAD=m_pad,
        SPLITS=splits,
        USE_DOT=use_dot,
        SCALE_GATHER=SCALE_GATHER,
        USE_PDL=PDL,
        launch_pdl=PDL,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return y if y2 is None else (y, y2)


# ---------------------------------------------------------------------------
# Load-time quantisation (deterministic, ModelOpt convention)
# ---------------------------------------------------------------------------

#: e2m1 magnitudes, in code order.
_E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
#: midpoints between consecutive levels; ties round to even (the boundary index
#: parity), which is what fp4 hardware does.
_E2M1_MID = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_E2M1_TIE_UP = (0.75, 1.75, 3.5)


def _round_e2m1(a: torch.Tensor) -> torch.Tensor:
    """|a| in units of the block scale -> uint8 code in [0, 7], round-to-nearest-even."""
    bnd = torch.tensor(_E2M1_MID, dtype=torch.float32, device=a.device)
    # right=False makes an exact boundary hit round *down*; bump the boundaries whose
    # even neighbour is above (0.75 -> 1.0, 1.75 -> 2.0, 3.5 -> 4.0).
    idx = torch.bucketize(a, bnd, right=False)
    up = (a == _E2M1_TIE_UP[0]) | (a == _E2M1_TIE_UP[1]) | (a == _E2M1_TIE_UP[2])
    return (idx + up).clamp_(0, 7).to(torch.uint8)


def quantize_nvfp4(
    w: torch.Tensor, block: int = FP4_SCALE_BLOCK, row_chunk: int = 4096
):
    """[N, K] weight -> (wq [N, K//2] uint8, bs [N, K//16] uint8 e4m3 bits, gscale fp32).

    ModelOpt convention: ``gscale = amax(|W|) / (448 * 6)``, per-16-block scale
    ``amax_block / 6 / gscale`` rounded to e4m3, codes ``round_e2m1(W / (bs * gscale))``.
    Deterministic (no atomics, no nondeterministic reductions) and chunked over rows so
    the peak transient is ``row_chunk * K`` fp32 rather than the whole tensor.

    The returned ``gscale`` is the plain global scale; ``w4a16_nvfp4_gemv`` folds in the
    2^14 of the dequant bit trick itself.
    """
    assert w.dim() == 2 and w.shape[1] % block == 0
    N, K = w.shape
    # The global amax is taken chunk-wise and `w` is never cast to fp32 whole. A single
    # `w.float()` is 2x the weight -- 2.5 GB for the 248320-row target head -- and the
    # server reaches this with ~2 GB free at mem_fraction 0.935, which is exactly how
    # stage B first died (torch.OutOfMemoryError: tried to allocate 2.37 GiB).
    gmax = w.new_zeros((), dtype=torch.float32)
    for r0 in range(0, N, row_chunk):
        gmax = torch.maximum(gmax, w[r0 : r0 + row_chunk].float().abs().amax())
    gscale = (gmax / (E4M3_MAX * E2M1_MAX)).clamp_(min=1e-30)
    # The *encode* scale, and multiplication by it, are the fork's own convention
    # (nvfp4_online.py:296 hands flashinfer `1.0 / weight_scale_2`). The form matters
    # beyond taste: `amax/6/gscale` and `amax/6*enc` differ by one fp32 ulp, and e4m3's
    # 4-bit significand is coarse enough that ~1 % of block scales land exactly on a
    # tie point, where that ulp flips the rounding. Using the same expression as
    # the project's reference NVFP4 emulation is what keeps the two implementations
    # bit-identical.
    enc = 1.0 / gscale
    wq = torch.empty((N, K // 2), dtype=torch.uint8, device=w.device)
    bs = torch.empty((N, K // block), dtype=torch.uint8, device=w.device)
    for r0 in range(0, N, row_chunk):
        r1 = min(N, r0 + row_chunk)
        blk = w[r0:r1].float().view(r1 - r0, K // block, block)
        amax = blk.abs().amax(dim=-1)
        sb = (amax / E2M1_MAX * enc).clamp_(min=0.0, max=E4M3_MAX)
        sb8 = sb.to(torch.float8_e4m3fn)
        bs[r0:r1] = sb8.view(torch.uint8)
        eff = (sb8.float() / enc).unsqueeze(-1)
        # A zero block (or one whose scale underflowed e4m3) dequantises to zero.
        a = torch.where(eff > 0, blk / eff.clamp(min=1e-38), torch.zeros_like(blk))
        code = _round_e2m1(a.abs()) | (a < 0).to(torch.uint8) * 8
        code = code.reshape(r1 - r0, K)
        wq[r0:r1] = code[:, 0::2] | (code[:, 1::2] << 4)
    return wq, bs, gscale.reshape(1)


#: Per-device e2m1 level LUT. Built once: `torch.tensor(list, device="cuda")` is a
#: host->device copy, which is illegal inside a CUDA graph capture, and the draft-extend
#: graph does capture this path.
_LEVELS: dict = {}


def _levels(device) -> torch.Tensor:
    t = _LEVELS.get(device)
    if t is None:
        t = torch.tensor(_E2M1_LEVELS, dtype=torch.float32, device=device)
        _LEVELS[device] = t
    return t


def dequantize_nvfp4(wq: torch.Tensor, bs: torch.Tensor, gscale: torch.Tensor,
                     block: int = FP4_SCALE_BLOCK) -> torch.Tensor:
    """Reference dequant (fp32) for the numerics check, the offline emulation, and the
    wide-M serving fallback.

    CUDA-graph safe: the level table is cached per device and the global scale stays a
    tensor. An earlier version built the table per call and used ``gscale.item()``,
    which made draft-extend graph capture fail with "Cannot copy between CPU and CUDA
    tensors during CUDA graph capture".
    """
    N, KH = wq.shape
    K = KH * 2
    lvl = _levels(wq.device)
    code = torch.empty((N, K), dtype=torch.uint8, device=wq.device)
    code[:, 0::2] = wq & 0x0F
    code[:, 1::2] = wq >> 4
    mag = lvl[(code & 0x07).long()]
    val = torch.where((code & 0x08) > 0, -mag, mag)
    g = gscale.to(wq.device, torch.float32).reshape(-1)
    g = g.reshape(-1, 1) if g.numel() > 1 else g.reshape(1, 1)
    s = bs.view(torch.float8_e4m3fn).float() * g
    return val * s.repeat_interleave(block, dim=1)
