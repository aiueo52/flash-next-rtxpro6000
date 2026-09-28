"""Non-persistent HC per-branch norm + low-rank mix for decode-size batches.

The persistent variant in ``hc_fused_triton`` serializes the down projection,
a grid barrier, and the up projection inside one launch, so at M<=16 neither
weight stream ever runs at full rate. This module splits the same math into
three independent launches, each of which streams one thing:

* K0 (grid = M * hc) -- one CTA per (row, branch): the sum of squares, the
  ``inv_rms`` it implies, the bf16 ``normed`` row chunk, and a slice of the
  fp32 split-K workspace cleared for K1's atomics. Folding the clear in here
  is what keeps the graph free of a separate memset node. Two optional stages
  ride along on the branch slice this CTA already owns: ``FUSE_APPLY`` runs
  the *previous* HC boundary's combine apply as a prologue (``x_ptr`` is then
  the output residual, ``apply_resid_ptr`` the input one) and ``FUSE_GATE``
  dots ``normed`` against the inject weight as an epilogue, leaving the gate
  partials the *next* boundary's apply needs.
* K1 (grid = k_groups x n_blocks) -- ``BLOCK_G`` chunks of ``BLOCK_K`` columns
  against the matching ``w_down`` tile, accumulated in registers and pushed to
  ``t_raw`` with one device-scope atomic per CTA.
* K2 (grid = j_blocks) -- ``silu(t_raw / hc)`` rounded to bf16, one ``tl.dot``
  per r-chunk against ``w_up``, then the sigmoid-gated mean over the hc
  branches.

``stats_mode`` picks where the norm happens. "norm" (the default, above) is the
fastest measured: K1 reading a materialized ``normed`` keeps its tile in the
dot's own layout, which is worth more than the extra 327 KB round trip. "stats"
has K0 emit only ``inv_rms`` and K1 normalize on the fly; "redundant" drops K0
entirely and makes every K1 CTA re-read its whole 2560-wide branch, which is
far slower at M=16. See bench/hc_mix2 for the numbers.

A CTA's whole column range must lie inside one branch, hence the
``hs % (BLOCK_K * BLOCK_G) == 0`` requirement: one ``inv_rms`` per row suffices.

The two mix weights (6.5 MB each in bf16) are the whole cost of K1 and K2 at
decode widths, so they can be carried as fp8 e4m3 with one fp32 scale per output
row (``quantize_hc_mix2_weights_fp8``); K1 scales its fp32 partial by the per-n
scale before the atomic add and K2 scales the finished accumulator by the per-j
scale, both exact w.r.t. the split. ``SGLANG_HC_MIX2_FP8=1`` turns it on in
``GatedResidual``; ``normed`` is unaffected (bit-identical) either way.
"""

from __future__ import annotations

import os

import msgspec
import torch
import triton
import triton.language as tl

from sglang.kernels.triton_pdl import PDL, pdl_trigger, pdl_wait

_HC_MIX2_MAX_ROWS = 16


class HCMix2Config(msgspec.Struct, frozen=True):
    """Launch geometry for the three kernels; swept by bench/hc_mix2."""

    # K1: BLOCK_K must divide hs so a chunk never straddles two branches.
    block_n: int = 16
    block_k: int = 128
    block_g: int = 2
    block_s: int = 512
    down_warps: int = 8
    down_stages: int = 2
    # K2
    block_j: int = 16
    block_r: int = 64
    up_warps: int = 4
    up_stages: int = 4
    # K0 mode: "stats" writes inv_rms only, "norm" also materializes normed so
    # K1 reads it back, "redundant" drops K0 (K1 recomputes the branch
    # statistics per CTA) and zeroes the workspace with a memset instead.
    stats_mode: str = "norm"
    stats_block: int = 4096
    stats_warps: int = 8


# FP8 (e4m3) weights halve the bytes each K1/K2 tile streams, so the geometry
# that was bandwidth-optimal for bf16 is not optimal here. Swept with
# bench/hc_mix2/bench_hc_mix2_fp8.py --sweep down/up (CUPTI medians, 82 weight
# copies = 512 MB working set, RTX PRO 6000 Blackwell Max-Q):
#
#   K1  bf16 default tile (16,128,2,w8,s2) on fp8 weights   5.31 us
#       [64, 128] x 4 k-groups, 8 warps, 4 stages           3.81 us   <- pinned
#   K2  the bf16 tile is still the winner (16, 64, w4, s4)  4.64 us
#
# The wide n block is what fp8 buys: the same 2 KB tile now spans 64 lowrank
# columns instead of 16, so K1 needs a quarter of the CTAs and each one keeps
# four k-chunks in flight.
_FP8_BASE = HCMix2Config(
    block_n=64,
    block_k=128,
    block_g=4,
    down_warps=8,
    down_stages=4,
    block_j=16,
    block_r=64,
    up_warps=4,
    up_stages=4,
)


def _env_config(
    default: HCMix2Config | None = None, prefix: str = "SGLANG_HC_MIX2_"
) -> HCMix2Config:
    def _int(name: str, value: int) -> int:
        return int(os.environ.get(prefix + name, str(value)))

    if default is None:
        default = HCMix2Config()
    return HCMix2Config(
        block_n=_int("BLOCK_N", default.block_n),
        block_k=_int("BLOCK_K", default.block_k),
        block_g=_int("BLOCK_G", default.block_g),
        block_s=_int("BLOCK_S", default.block_s),
        down_warps=_int("DOWN_WARPS", default.down_warps),
        down_stages=_int("DOWN_STAGES", default.down_stages),
        block_j=_int("BLOCK_J", default.block_j),
        block_r=_int("BLOCK_R", default.block_r),
        up_warps=_int("UP_WARPS", default.up_warps),
        up_stages=_int("UP_STAGES", default.up_stages),
        stats_mode=os.environ.get(prefix + "STATS_MODE", default.stats_mode),
        stats_block=_int("STATS_BLOCK", default.stats_block),
        stats_warps=_int("STATS_WARPS", default.stats_warps),
    )


_DEFAULT_CONFIG = _env_config()
# `SGLANG_HC_MIX2_FP8_BLOCK_N` etc. override the fp8 geometry only; the bare
# `SGLANG_HC_MIX2_FP8` flag (read in hyperconnection.py) turns the path on.
_DEFAULT_CONFIG_FP8 = _env_config(_FP8_BASE, "SGLANG_HC_MIX2_FP8_")

@triton.jit
def _hc_branch_stats_kernel(
    x_ptr,
    norm_w_ptr,
    inv_rms_ptr,
    normed_ptr,
    t_raw_ptr,
    inject_w_ptr,
    gate_partials_ptr,
    apply_resid_ptr,
    apply_block_ptr,
    apply_part_ptr,
    apply_shared_ptr,
    apply_sgate_ptr,
    num_tasks,
    zero_span,
    K,
    HS,
    eps,
    HC: tl.constexpr,
    BLOCK_S: tl.constexpr,
    ZERO_BLOCK: tl.constexpr,
    WRITE_NORMED: tl.constexpr,
    SINGLE_TILE: tl.constexpr,
    FUSE_GATE: tl.constexpr,
    FUSE_APPLY: tl.constexpr,
    FUSE_SHARED: tl.constexpr,
    PREV_SPLITS: tl.constexpr,
    USE_PDL: tl.constexpr = False,
):
    pid = tl.program_id(0)
    m = pid // HC
    c = pid % HC
    base = m * K + c * HS
    offs = tl.arange(0, BLOCK_S)
    # `x` is the previous kernel's output; nothing above this point touches
    # memory, so the CTAs can be scheduled while that kernel drains.
    pdl_wait(USE_PDL)

    if SINGLE_TILE:
        # One branch fits in one tile: issue the x and weight loads together so
        # the weight latency overlaps the sum-of-squares reduction instead of
        # starting a second dependent round trip after it.
        mask_s = offs < HS
        if FUSE_APPLY:
            # Previous boundary's HC combine apply (SGLANG_HC_APPLY_MIX_FUSED):
            # this CTA owns branch `c` of the row, which is exactly the slice
            # the apply would write, so form the new residual here and keep it
            # in registers instead of storing it and reading it back.
            if FUSE_SHARED:
                # R6's shared-expert join (`routed + gate * shared`) rides in
                # the same prologue at the layer->layer boundary. The three row
                # loads are issued *before* the gate reduction here: with the
                # shared row there are three of them and the reduction's own
                # chain of scalar loads is long enough that leaving them behind
                # it costs 0.18 us a launch at M=16 (measured; the two-load
                # `else` below keeps R7's original order, which measures the
                # same either way).
                y = tl.load(
                    apply_block_ptr + m * HS + offs, mask=mask_s, other=0.0
                ).to(tl.float32)
                sh = tl.load(
                    apply_shared_ptr + m * HS + offs, mask=mask_s, other=0.0
                ).to(tl.float32)
                g = tl.load(apply_sgate_ptr + m)
                r = tl.load(
                    apply_resid_ptr + base + offs, mask=mask_s, other=0.0
                ).to(tl.float32)
                total = 0.0
                for ps in tl.static_range(PREV_SPLITS):
                    total += tl.load(
                        apply_part_ptr + (m * PREV_SPLITS + ps) * HC + c
                    )
                a = 2.0 / (1.0 + tl.exp(-total / HC))
                # Round the sum back to the storage dtype first: that
                # reproduces the bf16 store `fused_gate_sigmoid_mul_add` would
                # have made before the combine read it, which is what
                # `hc_combine_apply`'s `kUseShared` path does too.
                y = (y + g * sh).to(x_ptr.dtype.element_ty).to(tl.float32)
            else:
                total = 0.0
                for ps in tl.static_range(PREV_SPLITS):
                    total += tl.load(
                        apply_part_ptr + (m * PREV_SPLITS + ps) * HC + c
                    )
                a = 2.0 / (1.0 + tl.exp(-total / HC))
                y = tl.load(
                    apply_block_ptr + m * HS + offs, mask=mask_s, other=0.0
                ).to(tl.float32)
                r = tl.load(
                    apply_resid_ptr + base + offs, mask=mask_s, other=0.0
                ).to(tl.float32)
            xb = (r + a * y).to(x_ptr.dtype.element_ty)
            tl.store(x_ptr + base + offs, xb, mask=mask_s)
            x = xb.to(tl.float32)
        else:
            x = tl.load(x_ptr + base + offs, mask=mask_s, other=0.0).to(tl.float32)
        if WRITE_NORMED:
            w = tl.load(norm_w_ptr + c * HS + offs, mask=mask_s, other=0.0).to(
                tl.float32
            )
        inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / HS + eps)
        tl.store(inv_rms_ptr + pid, inv_rms)
        if WRITE_NORMED:
            nrm = (x * inv_rms * (1.0 + w)).to(normed_ptr.dtype.element_ty)
            tl.store(normed_ptr + base + offs, nrm, mask=mask_s)
            if FUSE_GATE:
                # HC-combine gate epilogue (SGLANG_HC_GATE_EARLY): this CTA
                # already holds branch `c` of the normed row, so the branch's
                # slice of each of the HC inject-weight rows is dotted here and
                # left as one partial per (row, branch, gate). The apply stage
                # after the block sums the HC partials; the standalone gate
                # kernel is then never launched.
                nrm32 = nrm.to(tl.float32)
                for cc in tl.static_range(HC):
                    gw = tl.load(
                        inject_w_ptr + cc * K + c * HS + offs,
                        mask=mask_s,
                        other=0.0,
                    ).to(tl.float32)
                    tl.store(
                        gate_partials_ptr + pid * HC + cc,
                        tl.sum(nrm32 * gw, axis=0),
                    )
    else:
        # FUSE_APPLY / FUSE_GATE are gated to the single-tile shape by the
        # launcher, so this branch stays the plain statistics pass.
        acc = tl.zeros((BLOCK_S,), dtype=tl.float32)
        for s0 in range(0, HS, BLOCK_S):
            s = s0 + offs
            x = tl.load(x_ptr + base + s, mask=s < HS, other=0.0).to(tl.float32)
            acc += x * x
        inv_rms = tl.rsqrt(tl.sum(acc, axis=0) / HS + eps)
        tl.store(inv_rms_ptr + pid, inv_rms)
        if WRITE_NORMED:
            for s0 in range(0, HS, BLOCK_S):
                s = s0 + offs
                mask_s = s < HS
                x = tl.load(x_ptr + base + s, mask=mask_s, other=0.0).to(tl.float32)
                w = tl.load(norm_w_ptr + c * HS + s, mask=mask_s, other=0.0).to(
                    tl.float32
                )
                tl.store(
                    normed_ptr + base + s,
                    (x * inv_rms * (1.0 + w)).to(normed_ptr.dtype.element_ty),
                    mask=mask_s,
                )

    offs_z = tl.arange(0, ZERO_BLOCK)
    for z0 in range(pid * ZERO_BLOCK, zero_span, num_tasks * ZERO_BLOCK):
        idx = z0 + offs_z
        tl.store(t_raw_ptr + idx, 0.0, mask=idx < zero_span)
    # K1 atomically accumulates into t_raw; its own gdc_wait keeps its stores
    # behind this clear, so triggering here only lets it start scheduling.
    pdl_trigger(USE_PDL)


@triton.jit
def _hc_down_kernel(
    x_ptr,
    norm_w_ptr,
    w_down_ptr,
    inv_rms_ptr,
    normed_ptr,
    t_raw_ptr,
    s_down_ptr,
    num_rows,
    K,
    HS,
    LOWRANK,
    eps,
    ROWS: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_S: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    REDUNDANT_STATS: tl.constexpr,
    READ_NORMED: tl.constexpr,
    W_FP8: tl.constexpr,
    USE_PDL: tl.constexpr = False,
):
    kg = tl.program_id(0)
    nb = tl.program_id(1)
    pdl_wait(USE_PDL)
    offs_m = tl.arange(0, ROWS)
    mask_m = offs_m < num_rows
    k0 = kg * (BLOCK_K * BLOCK_G)
    c = k0 // HS
    branch_base = c * HS
    offs_k = tl.arange(0, BLOCK_K)

    if READ_NORMED:
        pass
    elif REDUNDANT_STATS:
        offs_s = tl.arange(0, BLOCK_S)
        acc_sq = tl.zeros((ROWS,), dtype=tl.float32)
        for s0 in range(0, HS, BLOCK_S):
            s = s0 + offs_s
            xs = tl.load(
                x_ptr + offs_m[:, None] * K + (branch_base + s)[None, :],
                mask=mask_m[:, None] & (s < HS)[None, :],
                other=0.0,
            ).to(tl.float32)
            acc_sq += tl.sum(xs * xs, axis=1)
        inv_rms = tl.rsqrt(acc_sq / HS + eps)
    else:
        inv_rms = tl.load(inv_rms_ptr + offs_m * HC + c, mask=mask_m, other=1.0)

    n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = n < LOWRANK
    acc = tl.zeros((ROWS, BLOCK_N), dtype=tl.float32)
    for g in tl.range(0, BLOCK_G, num_stages=NUM_STAGES):
        k = k0 + g * BLOCK_K + offs_k
        if READ_NORMED:
            normed = tl.load(
                normed_ptr + offs_m[:, None] * K + k[None, :],
                mask=mask_m[:, None],
                other=0.0,
            )
        else:
            x = tl.load(
                x_ptr + offs_m[:, None] * K + k[None, :],
                mask=mask_m[:, None],
                other=0.0,
            ).to(tl.float32)
            w = tl.load(norm_w_ptr + k).to(tl.float32)
            normed = (x * inv_rms[:, None] * (1.0 + w[None, :])).to(
                x_ptr.dtype.element_ty
            )
            if nb == 0:
                tl.store(
                    normed_ptr + offs_m[:, None] * K + k[None, :],
                    normed,
                    mask=mask_m[:, None],
                )
        w_down = tl.load(
            w_down_ptr + n[:, None] * K + k[None, :], mask=mask_n[:, None], other=0.0
        )
        if W_FP8:
            acc = tl.dot(normed, tl.trans(w_down.to(tl.bfloat16)), acc)
        else:
            acc = tl.dot(normed, tl.trans(w_down), acc)
    if W_FP8:
        # One fp32 scale per output row n. The split-K partials are summed with
        # atomics, and scaling is linear, so applying it here is exact.
        acc = acc * tl.load(s_down_ptr + n, mask=mask_n, other=0.0)[None, :]
    tl.atomic_add(
        t_raw_ptr + offs_m[:, None] * LOWRANK + n[None, :],
        acc,
        mask=mask_m[:, None] & mask_n[None, :],
        sem="relaxed",
        scope="gpu",
    )
    pdl_trigger(USE_PDL)


@triton.jit
def _hc_up_kernel(
    normed_ptr,
    w_up_ptr,
    t_raw_ptr,
    out_ptr,
    s_up_ptr,
    num_rows,
    K,
    HS,
    LOWRANK,
    inv_hc,
    ROWS: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_R: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    W_FP8: tl.constexpr,
    USE_PDL: tl.constexpr = False,
):
    jb = tl.program_id(0)
    pdl_wait(USE_PDL)
    offs_m = tl.arange(0, ROWS)
    mask_m = offs_m < num_rows
    offs_j = tl.arange(0, BLOCK_J)
    offs_c = tl.arange(0, HC)
    j = jb * BLOCK_J + offs_j
    mask_j = j < HS
    cj = tl.reshape(offs_c[:, None] * HS + j[None, :], (HC * BLOCK_J,))
    mask_cj = tl.reshape(
        tl.broadcast_to(mask_j[None, :], (HC, BLOCK_J)), (HC * BLOCK_J,)
    )

    acc = tl.zeros((ROWS, HC * BLOCK_J), dtype=tl.float32)
    offs_r = tl.arange(0, BLOCK_R)
    for r0 in tl.range(0, LOWRANK, BLOCK_R, num_stages=NUM_STAGES):
        r = r0 + offs_r
        mask_r = r < LOWRANK
        a = tl.load(
            t_raw_ptr + offs_m[:, None] * LOWRANK + r[None, :],
            mask=mask_m[:, None] & mask_r[None, :],
            other=0.0,
        )
        a = a * inv_hc
        t = (a * tl.sigmoid(a)).to(out_ptr.dtype.element_ty)
        w = tl.load(
            w_up_ptr + cj[:, None] * LOWRANK + r[None, :],
            mask=mask_cj[:, None] & mask_r[None, :],
            other=0.0,
        )
        if W_FP8:
            acc = tl.dot(t, tl.trans(w.to(tl.bfloat16)), acc)
        else:
            acc = tl.dot(t, tl.trans(w), acc)

    if W_FP8:
        # One fp32 scale per output column (the flattened c*HS + j row of w_up);
        # the whole r reduction for that column lives in this CTA, so scaling the
        # finished accumulator is exact.
        acc = acc * tl.load(s_up_ptr + cj, mask=mask_cj, other=0.0)[None, :]
    gate = tl.sigmoid(tl.reshape(acc, (ROWS, HC, BLOCK_J)))
    xg = tl.load(
        normed_ptr
        + offs_m[:, None, None] * K
        + offs_c[None, :, None] * HS
        + j[None, None, :],
        mask=mask_m[:, None, None] & mask_j[None, None, :],
        other=0.0,
    ).to(tl.float32)
    out = tl.sum(gate * xg, axis=1) * inv_hc
    tl.store(
        out_ptr + offs_m[:, None] * HS + j[None, :],
        out.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_j[None, :],
    )
    pdl_trigger(USE_PDL)


def quantize_hc_mix2_weights_fp8(
    w_down: torch.Tensor, w_up: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-output-row FP8 E4M3 quantization of the mix2 weights.

    Mirrors ``hc_mix_triton.quantize_hc_mix_weights_fp8`` but takes ``w_up`` in
    the natural ``[hc * hs, lowrank]`` nn.Linear layout the three-kernel path
    reads (no permute/pad). Returns ``(w_down_fp8[320, K], s_down[320],
    w_up_fp8[K, 320], s_up[K])`` with fp32 scales.
    """

    def _q(w: torch.Tensor):
        wf = w.float()
        amax = wf.abs().amax(dim=1).clamp(min=1e-12)
        scale = amax / 448.0
        q = (wf / scale[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        return q.contiguous(), scale.contiguous()

    wd, sd = _q(w_down)
    wu, su = _q(w_up)
    return wd, sd, wu, su


def dequantize_hc_mix2_weight(w_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """bf16 view of an fp8 mix2 weight (for the prefill / fallback GEMM paths)."""
    return (w_fp8.float() * scale[:, None]).to(torch.bfloat16)


def hc_norm_mix2_supported(
    hyper_input: torch.Tensor,
    norm_w: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    hc: int,
    hs: int,
    config: HCMix2Config | None = None,
) -> bool:
    if config is None:
        config = (
            _DEFAULT_CONFIG_FP8
            if w_down.dtype == torch.float8_e4m3fn
            else _DEFAULT_CONFIG
        )
    if not (
        hyper_input.is_cuda
        and hyper_input.dim() == 2
        and hyper_input.dtype == torch.bfloat16
        and 1 <= hyper_input.shape[0] <= _HC_MIX2_MAX_ROWS
        and hyper_input.shape[1] == hc * hs
        and hyper_input.is_contiguous()
        and hs % (config.block_k * config.block_g) == 0
        and w_down.shape[0] % config.block_n == 0
    ):
        return False
    if not (
        norm_w.is_cuda
        and norm_w.device == hyper_input.device
        and norm_w.dtype == torch.bfloat16
        and norm_w.is_contiguous()
    ):
        return False
    # The mix weights are either both bf16 or both fp8 e4m3 (weight-only, with
    # per-output-row fp32 scales supplied by the caller).
    w_dtype = w_down.dtype
    if w_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        return False
    return all(
        w.is_cuda
        and w.device == hyper_input.device
        and w.dtype == w_dtype
        and w.is_contiguous()
        for w in (w_down, w_up)
    )


def hc_apply_norm_mix2_supported(
    hs: int,
    w_down: torch.Tensor,
    config: HCMix2Config | None = None,
) -> bool:
    """Whether `hc_norm_mix2` can absorb the previous boundary's combine apply.

    K0 must exist (not the "redundant" stats mode) and one branch must fit in
    one tile, which is what lets the applied row stay in registers.
    """
    if config is None:
        config = (
            _DEFAULT_CONFIG_FP8
            if w_down.dtype == torch.float8_e4m3fn
            else _DEFAULT_CONFIG
        )
    return config.stats_mode != "redundant" and config.stats_block >= hs


def hc_norm_mix2(
    hyper_input: torch.Tensor,
    norm_w: torch.Tensor,
    eps: float,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    hc: int,
    hs: int,
    config: HCMix2Config | None = None,
    s_down: torch.Tensor | None = None,
    s_up: torch.Tensor | None = None,
    inject_weight: torch.Tensor | None = None,
    apply_inputs: tuple | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-branch Gemma RMSNorm followed by the gated low-rank mix.

    Returns ``(mixed[M, hs], normed[M, hc * hs])`` in the input dtype. ``t_raw``
    is accumulated with device-scope atomics, so the summation order of the
    down projection is not reproducible across launches.

    ``w_down``/``w_up`` may be fp8 e4m3 (weight-only), in which case ``s_down``
    (one fp32 scale per lowrank row) and ``s_up`` (one per hc*hs row) are
    required; ``normed`` is bit-identical to the bf16 path either way.

    ``inject_weight`` adds the next combine's gate to K0 and makes the return
    ``(mixed, normed, gate_partials[M, hc, hc])``. ``apply_inputs`` is
    ``(block_output, gate_partials)`` -- or ``(block_output, gate_partials,
    shared_output, shared_gate)`` to fold R6's shared-expert join in -- of the
    *previous* boundary: K0 then applies that combine to ``hyper_input`` first,
    and the return grows to ``(mixed, normed, gate_partials, applied)`` where
    ``applied`` is the combined residual. Check
    `hc_apply_norm_mix2_supported` before passing it.
    """
    w_fp8 = w_down.dtype == torch.float8_e4m3fn
    if config is None:
        config = _DEFAULT_CONFIG_FP8 if w_fp8 else _DEFAULT_CONFIG
    if w_fp8:
        assert s_down is not None and s_up is not None
        assert w_up.dtype == torch.float8_e4m3fn
    else:
        # W_FP8 is a constexpr, so the scale loads are not even traced on the
        # bf16 path; pass the weights themselves rather than allocating a dummy
        # buffer here (an allocation inside a CUDA-graph capture would come from
        # the graph's private pool).
        s_down, s_up = w_down, w_up
    rows, k = hyper_input.shape
    lowrank = w_down.shape[0]
    device = hyper_input.device
    rows_pad = _HC_MIX2_MAX_ROWS

    mixed = torch.empty((rows, hs), dtype=hyper_input.dtype, device=device)
    redundant_stats = config.stats_mode == "redundant"
    read_normed = config.stats_mode == "norm"
    single_tile = config.stats_block >= hs
    fuse_apply = apply_inputs is not None and not redundant_stats and single_tile
    if fuse_apply:
        # `hyper_input` is the *previous* boundary's residual: K0 runs that
        # boundary's combine apply on it and writes the result to `applied`,
        # which is this boundary's hyper input and the tensor the next combine
        # reads back. Nothing else re-reads the row in between.
        prev_block_output, prev_partials = apply_inputs[0], apply_inputs[1]
        prev_shared = apply_inputs[2] if len(apply_inputs) > 2 else None
        prev_sgate = apply_inputs[3] if len(apply_inputs) > 3 else None
        prev_splits = prev_partials.shape[1]
        applied = torch.empty_like(hyper_input)
    else:
        prev_block_output = prev_partials = hyper_input
        prev_shared = prev_sgate = None
        prev_splits = 1
        applied = None
    fuse_shared = fuse_apply and prev_shared is not None
    if fuse_shared:
        assert prev_sgate is not None
    else:
        # FUSE_SHARED is a constexpr, so these loads are never traced; pass a
        # live tensor rather than allocating a dummy inside a graph capture.
        prev_shared = prev_sgate = hyper_input
    if apply_inputs is not None and not fuse_apply:
        # The caller must have checked `hc_apply_norm_mix2_supported`; running
        # the mix on the un-applied row would be silently wrong.
        raise RuntimeError("hc_norm_mix2: this config cannot absorb the apply")
    x_for_mix = applied if fuse_apply else hyper_input
    fuse_gate = (
        inject_weight is not None
        and not redundant_stats
        and read_normed
        and single_tile
    )
    gate_partials = (
        torch.empty((rows, hc, hc), dtype=torch.float32, device=device)
        if fuse_gate
        else None
    )
    normed = torch.empty_like(hyper_input)
    if redundant_stats:
        # No K0 to fold the clear into, so the graph carries a 20 KB memset.
        t_raw = torch.zeros((rows_pad, lowrank), dtype=torch.float32, device=device)
        inv_rms = t_raw
    else:
        t_raw = torch.empty((rows_pad, lowrank), dtype=torch.float32, device=device)
        inv_rms = torch.empty((rows_pad, hc), dtype=torch.float32, device=device)
        num_tasks = rows * hc
        zero_span = rows * lowrank
        _hc_branch_stats_kernel[(num_tasks,)](
            x_for_mix,
            norm_w,
            inv_rms,
            normed,
            t_raw,
            inject_weight if fuse_gate else normed,
            gate_partials if fuse_gate else t_raw,
            hyper_input,
            prev_block_output,
            prev_partials,
            prev_shared,
            prev_sgate,
            num_tasks,
            zero_span,
            k,
            hs,
            eps,
            HC=hc,
            BLOCK_S=config.stats_block,
            ZERO_BLOCK=max(64, triton.next_power_of_2(zero_span // num_tasks)),
            WRITE_NORMED=read_normed,
            SINGLE_TILE=single_tile,
            FUSE_GATE=fuse_gate,
            FUSE_APPLY=fuse_apply,
            FUSE_SHARED=fuse_shared,
            PREV_SPLITS=prev_splits,
            USE_PDL=PDL,
            launch_pdl=PDL,
            num_warps=config.stats_warps,
        )

    _hc_down_kernel[
        (k // (config.block_k * config.block_g), triton.cdiv(lowrank, config.block_n))
    ](
        x_for_mix,
        norm_w,
        w_down,
        inv_rms,
        normed,
        t_raw,
        s_down,
        rows,
        k,
        hs,
        lowrank,
        eps,
        ROWS=rows_pad,
        HC=hc,
        BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        BLOCK_G=config.block_g,
        BLOCK_S=config.block_s,
        NUM_STAGES=config.down_stages,
        REDUNDANT_STATS=redundant_stats,
        READ_NORMED=read_normed,
        W_FP8=w_fp8,
        USE_PDL=PDL,
        launch_pdl=PDL,
        num_warps=config.down_warps,
    )

    _hc_up_kernel[(triton.cdiv(hs, config.block_j),)](
        normed,
        w_up,
        t_raw,
        mixed,
        s_up,
        rows,
        k,
        hs,
        lowrank,
        1.0 / hc,
        ROWS=rows_pad,
        HC=hc,
        BLOCK_J=config.block_j,
        BLOCK_R=config.block_r,
        NUM_STAGES=config.up_stages,
        W_FP8=w_fp8,
        USE_PDL=PDL,
        launch_pdl=PDL,
        num_warps=config.up_warps,
    )
    if apply_inputs is not None:
        return mixed, normed, gate_partials, applied
    if inject_weight is not None:
        return mixed, normed, gate_partials
    return mixed, normed
