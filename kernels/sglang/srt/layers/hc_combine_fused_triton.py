"""Single-launch HyperConnection combine (gate + apply in one Triton kernel).

``GatedResidual.combine`` currently runs the sgl-kernel split pair
``hc_combine_gate`` + ``hc_combine_apply`` (``jit/csrc/elementwise/hc_combine.cuh``).
At the decode/verify widths of Qwen3.8-Flash-Next that is 2 launches x 96
boundaries per step, and each launch costs ~1 us of fixed time on top of a kernel
that only moves ~1 MB.  This module does the same math in one launch:

    a[m, c] = 2 * sigmoid(dot(normed_residual[m, :], inject_weight[c, :]) / HC)
    out[m, c*HS + i] = residual[m, c*HS + i] + a[m, c] * block_output[m, i]

Geometry.  Programs are indexed by (row m, column slice q): slice q is the same
``SUB``-wide column window in every one of the HC branches, with SUB a power of two
dividing the per-branch hidden size, so every access is a contiguous, 16 B-vectorised
block and the ``block_output`` row is read once per program instead of once per
branch.  Phase 1 accumulates the slice's contribution to all HC gate dots into a
persistent fp32 ``partials`` buffer; a grid barrier publishes them; phase 2 reduces
the Q partials per branch, forms the gates and writes the slice of all HC branches.

That layout also cuts traffic relative to the CUDA pair: the normed row and the
block output are each read exactly once (the reference gate kernel reads the normed
row once per branch, and its apply kernel re-reads the block output once per branch),
~3.6 MB -> ~2.4 MB per 16-row call.

Grid barrier safety.  ``rows * Q`` is kept <= the SM count, so every CTA of the
launch is resident before the barrier is reached (a CUDA-graph replay serialises the
stream, and PDL only lets a successor start once this grid is fully launched).  The
wrapper falls back to the two-kernel path whenever that does not hold.  The two
counters are restored to 0 by the last CTA out, so a graph replay starts clean.

Numerics.  Everything is fp32 with a bf16 round at the store, exactly as the CUDA
pair does, but the fp32 dot is reduced in a different order: a tl.sum tree over the
[HC, SUB] tile and then over the Q slices, versus the reference's per-thread
sequential sum of 40 products, warp butterfly and sequential fold over its 8 splits.
The gate therefore differs by ~1e-7 relative, and the bf16 output only moves when the
fp32 result sits within that of a bf16 rounding boundary -- then by exactly one bf16
ulp.  A CPU emulation of both reduction orders puts that at ~2e-5 of elements at 16
rows and 0 at 4 rows; ``test/srt/layers/test_hc_combine_fused.py`` asserts the <= 1
ulp bound on device and prints the measured rate.

MEASURED RESULT: this is a REGRESSION on RTX PRO 6000 Blackwell -- keep the flag off.
CUPTI medians over graph replay (bench/glue_e/bench_hc_combine.py, 96-deep rotating
working set): the CUDA pair is 2.14 us/call at 1 row and 2.85 us at 16 rows; this
kernel is 3.26 us and 3.74 us, i.e. +1.3 us/call or ~+130 us/step over 96 combines.
Removing the barrier entirely (diagnostic build) still leaves 2.53 us / 2.88 us, so
the Triton body alone already costs as much as BOTH hand-written CUDA kernels: the
pair is not paying two full launches, because sgl-kernel launches the apply kernel
with PDL so its prologue overlaps the gate kernel's tail. There is no launch overhead
left here for a fusion to reclaim. Kept, flag-gated and off, as the recorded negative
result; it would need a Triton-side PDL equivalent (or a CUDA rewrite) to win.

Enable with ``SGLANG_HC_COMBINE_FUSED=1``.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl

try:  # __nv_expf, matching the reference's math::exp -> expf
    from triton.language.extra.libdevice import exp as _exp_f32
except Exception:  # pragma: no cover - older Triton
    _exp_f32 = tl.exp

# Column-slice widths tried from the narrowest (most parallel) up; the first whose
# grid fits the machine wins. SUB must be a power of two dividing the per-branch
# hidden size so a slice never straddles a branch, and >= 128 so each thread still
# moves at least 4 elements.
_SUB_CANDIDATES = (128, 256, 512)
_MAX_ROWS = 32
_MAX_CTAS = 256  # partials/counters are sized for this many programs


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


# Bench hooks: pin the slice width / warp count instead of letting _plan choose.
_FORCE_SUB = _env_int("SGLANG_HC_COMBINE_FUSED_SUB", 0)
_FORCE_WARPS = _env_int("SGLANG_HC_COMBINE_FUSED_WARPS", 0)


@triton.jit
def _hc_combine_fused_kernel(
    y_ptr,  # block_output      [M, HS]      bf16/fp16
    r_ptr,  # residual          [M, HC*HS]
    n_ptr,  # normed_residual   [M, HC*HS]
    w_ptr,  # inject_weight     [HC, HC*HS]
    o_ptr,  # out               [M, HC*HS]
    part_ptr,  # fp32           [>= M*Q, HC]
    cnt_ptr,  # int32           [2]
    n_ctas,
    HC: tl.constexpr,
    HS: tl.constexpr,
    Q: tl.constexpr,
    QP: tl.constexpr,  # Q rounded up to a power of two (tl.arange needs one)
    SUB: tl.constexpr,
):
    ROW: tl.constexpr = HC * HS
    pid = tl.program_id(0)
    m = pid // Q
    q = pid % Q

    # [HC, SUB] tile: the same column window in each of the HC branches. SUB divides
    # HS, so a tile never straddles a branch and each row is a contiguous 16 B-aligned
    # run (HS is a multiple of 8).
    tile = tl.arange(0, HC)[:, None] * HS + (q * SUB + tl.arange(0, SUB))[None, :]
    nb = tl.load(n_ptr + m * ROW + tile).to(tl.float32)

    # ---- phase 1: this slice's contribution to each branch's gate dot ----
    for c in tl.static_range(HC):
        wb = tl.load(w_ptr + c * ROW + tile).to(tl.float32)
        tl.store(
            part_ptr + pid * HC + c,
            tl.sum(tl.sum(nb * wb, axis=1), axis=0),
            cache_modifier=".cg",
        )

    # ---- grid barrier ----
    tl.debug_barrier()
    tl.atomic_add(cnt_ptr, 1, sem="release", scope="gpu")
    # Poll with a volatile load, not an atomic RMW: every spinning CTA hammering one
    # address with read-modify-writes serialises in L2, which cost more than the
    # launch this kernel saves. A volatile load can be served from L2 without it.
    while tl.load(cnt_ptr, volatile=True) < n_ctas:
        pass
    tl.debug_barrier()

    # ---- phase 2: reduce the gate dots and stream this slice of every branch ----
    qi = tl.arange(0, QP)
    part = tl.load(
        part_ptr + (m * Q + qi)[:, None] * HC + tl.arange(0, HC)[None, :],
        mask=(qi < Q)[:, None],
        other=0.0,
        cache_modifier=".cg",
    )
    # 2 * sigmoid(dot / HC), the reference's `2.0f / (1.0f + expf(-total / HC))`.
    a = 2.0 / (1.0 + _exp_f32(-tl.sum(part, axis=0) / HC))  # [HC]
    yv = tl.load(y_ptr + m * HS + q * SUB + tl.arange(0, SUB)).to(tl.float32)
    rb = tl.load(r_ptr + m * ROW + tile).to(tl.float32)
    tl.store(
        o_ptr + m * ROW + tile,
        (rb + a[:, None] * yv[None, :]).to(o_ptr.dtype.element_ty),
    )

    # ---- restore the counters for the next launch / graph replay ----
    ticket = tl.atomic_add(cnt_ptr + 1, 1, sem="acq_rel", scope="gpu")
    if ticket == n_ctas - 1:
        tl.store(cnt_ptr, 0)
        tl.store(cnt_ptr + 1, 0)


_STATE: dict = {}


def _num_sms(device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _state(device, hc_count: int):
    key = (str(device), hc_count)
    st = _STATE.get(key)
    if st is None:
        # Outside inference_mode: the kernel mutates both buffers in place and
        # CUDA-graph capture rejects in-place writes to inference tensors.
        with torch.inference_mode(False):
            partials = torch.zeros(
                (_MAX_CTAS, hc_count), dtype=torch.float32, device=device
            )
            counters = torch.zeros(2, dtype=torch.int32, device=device)
        st = (partials, counters)
        _STATE[key] = st
    return st


def _plan(rows: int, hidden_size: int, hc_count: int, sms: int):
    """(Q, SUB, num_warps) or None when no slicing fits the grid barrier."""
    if hc_count < 1 or hc_count & (hc_count - 1):
        return None  # the [HC, SUB] tile needs a power-of-two branch count
    for sub in _SUB_CANDIDATES:
        if _FORCE_SUB and sub != _FORCE_SUB:
            continue
        if hidden_size % sub:  # a slice must stay inside one branch
            continue
        q = hidden_size // sub
        if rows * q > sms:  # every CTA must be resident at the barrier
            continue
        warps = _FORCE_WARPS or max(1, min(8, sub // 256))
        return q, sub, warps
    return None


def hc_combine_fused_supported(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    normed_residual: torch.Tensor,
    inject_weight: torch.Tensor,
    hc_count: int,
    hidden_size: int,
) -> bool:
    """Whether `hc_combine_fused` can serve this call (else use the two-kernel path).

    The kernel addresses every tensor with compact strides and needs its whole grid
    resident at the barrier, so anything non-contiguous or too wide falls back.
    """
    row_size = hc_count * hidden_size
    tensors = (block_output, residual, normed_residual, inject_weight)
    if not all(t.is_cuda and t.is_contiguous() for t in tensors):
        return False
    if residual.dtype not in (torch.bfloat16, torch.float16):
        return False
    if not all(t.dtype == residual.dtype for t in tensors):
        return False
    if tuple(inject_weight.shape) != (hc_count, row_size):
        return False
    rows = residual.numel() // row_size
    if rows < 1 or rows > _MAX_ROWS or rows * row_size != residual.numel():
        return False
    if block_output.numel() != rows * hidden_size:
        return False
    if normed_residual.numel() != residual.numel():
        return False
    return _plan(rows, hidden_size, hc_count, _num_sms(residual.device)) is not None


def hc_combine_fused(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    normed_residual: torch.Tensor,
    inject_weight: torch.Tensor,
    hc_count: int,
    hidden_size: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """One-launch equivalent of ``hc_combine`` / ``hc_combine_split``."""
    row_size = hc_count * hidden_size
    y = block_output.reshape(-1, hidden_size)
    r = residual.reshape(-1, row_size)
    n = normed_residual.reshape(-1, row_size)
    o = torch.empty_like(r) if out is None else out.reshape(-1, row_size)
    rows = r.shape[0]

    plan = _plan(rows, hidden_size, hc_count, _num_sms(r.device))
    if plan is None:
        raise ValueError(
            f"hc_combine_fused: no slicing fits rows={rows} hidden_size={hidden_size}"
        )
    q, sub, warps = plan
    n_ctas = rows * q
    partials, counters = _state(r.device, hc_count)

    _hc_combine_fused_kernel[(n_ctas,)](
        y,
        r,
        n,
        inject_weight,
        o,
        partials,
        counters,
        n_ctas,
        HC=hc_count,
        HS=hidden_size,
        Q=q,
        QP=triton.next_power_of_2(q),
        SUB=sub,
        num_warps=warps,
        num_stages=1,
    )
    return o.reshape(residual.shape)
