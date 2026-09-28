"""Persistent fused HyperConnection combine, per-branch norm, and mix.

Decode and target-verify batches are small enough that launching the individual
combine, RMSNorm, and low-rank mix kernels costs more than their arithmetic.
This module keeps the bf16 materialization points of that chain while executing
all of its phases in one persistent Triton launch. The fp32 split-K workspace
is zeroed once when it is allocated, then the ticket-winning CTA clears it in
the epilogue so the next launch (including a CUDA-graph replay) starts at zero.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

_HC_FUSED_MAX_ROWS = 16


def _env_block_size(name: str, default: int, allowed: tuple[int, ...]) -> int:
    value = int(os.environ.get(name, str(default)))
    if value not in allowed:
        choices = ", ".join(str(item) for item in allowed)
        raise ValueError(f"{name} must be one of: {choices}")
    return value


_HC_FUSED_BLOCK_J = _env_block_size("SGLANG_HC_FUSED_BLOCK_J", 16, (8, 16, 32))
_HC_FUSED_BLOCK_R = _env_block_size("SGLANG_HC_FUSED_BLOCK_R", 64, (64, 128))


@triton.jit
def _grid_barrier(counter_ptr, num_ctas):
    tl.atomic_add(counter_ptr, 1, sem="acq_rel", scope="gpu")
    while tl.atomic_add(counter_ptr, 0, sem="acq_rel", scope="gpu") < num_ctas:
        pass


@triton.jit
def _consume_prefetched_bf16(value):
    # Triton's dot-layout propagation otherwise consumes a distant tl.load at
    # its source, making the load block before the independent phase. The tied
    # impure asm emits no instruction, but keeps the raw bf16 register tile in
    # its blocked layout until this explicit post-work consumption point.
    return tl.inline_asm_elementwise(
        asm="",
        constraints="=r,0",
        args=[value],
        dtype=tl.bfloat16,
        is_pure=False,
        pack=2,
    )


@triton.jit
def _hc_fused_persistent_kernel(
    hyper_input_ptr,
    block_output_ptr,
    normed_prev_ptr,
    inject_w_ptr,
    norm_w_ptr,
    w_down_ptr,
    w_up_ptr,
    new_residual_ptr,
    normed_ptr,
    t_raw_ptr,
    mixed_ptr,
    counters_ptr,
    K,
    LOWRANK,
    HS,
    num_rows,
    num_ctas,
    eps,
    inv_hc,
    ROWS: tl.constexpr,
    HC: tl.constexpr,
    HAS_COMBINE: tl.constexpr,
    BLOCK_P0: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, ROWS)
    mask_m = offs_m < num_rows

    # The production down projection has 400 tiles. With 188 CTAs every CTA
    # owns two and the first 24 own a third. The first static load is issued
    # before P0; the second follows each CTA's local P0 work but precedes the
    # barrier. Together they cover 94% of W_down while limiting live registers
    # on active branch CTAs.
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    n_blocks = tl.cdiv(LOWRANK, BLOCK_N)
    k_chunks = tl.cdiv(K, BLOCK_K)
    down_tiles = n_blocks * k_chunks

    down_tile0 = pid
    down_nb0 = down_tile0 % n_blocks
    down_kc0 = down_tile0 // n_blocks
    down_n0 = down_nb0 * BLOCK_N + offs_n
    down_k0 = down_kc0 * BLOCK_K + offs_k
    down_valid0 = down_tile0 < down_tiles
    down_mask_n0 = (down_n0 < LOWRANK) & down_valid0
    down_w0 = tl.load(
        w_down_ptr + down_n0[:, None] * K + down_k0[None, :],
        mask=down_mask_n0[:, None],
        other=0.0,
    )

    down_tile1 = pid + num_ctas
    down_nb1 = down_tile1 % n_blocks
    down_kc1 = down_tile1 // n_blocks
    down_n1 = down_nb1 * BLOCK_N + offs_n
    down_k1 = down_kc1 * BLOCK_K + offs_k
    down_valid1 = down_tile1 < down_tiles
    down_mask_n1 = (down_n1 < LOWRANK) & down_valid1
    # P0b: one task owns one (row, HC branch). Wide register-resident tiles
    # turn the production K=10240 gate into three reductions and H=2560 norm
    # into one. Rnew is rounded once to bf16, then that same register value is
    # reused for its square and normalized store; there is no global reload.
    offs_p0 = tl.arange(0, BLOCK_P0)
    branch_tasks = num_rows * HC
    for task in range(pid, branch_tasks, num_ctas):
        m = task // HC
        c = task % HC
        branch_base = m * K + c * HS

        if HAS_COMBINE:
            gate_acc = 0.0
            for k0 in range(0, K, BLOCK_P0):
                k = k0 + offs_p0
                mask_k = k < K
                n = tl.load(normed_prev_ptr + m * K + k, mask=mask_k, other=0.0).to(
                    tl.float32
                )
                w = tl.load(inject_w_ptr + c * K + k, mask=mask_k, other=0.0).to(
                    tl.float32
                )
                gate_acc += tl.sum(n * w, axis=0)
            gate = 2.0 * tl.sigmoid(gate_acc * inv_hc)

        h = offs_p0
        mask_h = h < HS
        if HAS_COMBINE:
            r = tl.load(hyper_input_ptr + branch_base + h, mask=mask_h, other=0.0).to(
                tl.float32
            )
            y = tl.load(block_output_ptr + m * HS + h, mask=mask_h, other=0.0).to(
                tl.float32
            )
            x_bf16 = (r + gate * y).to(hyper_input_ptr.dtype.element_ty)
            tl.store(new_residual_ptr + branch_base + h, x_bf16, mask=mask_h)
        else:
            x_bf16 = tl.load(hyper_input_ptr + branch_base + h, mask=mask_h, other=0.0)
        x = x_bf16.to(tl.float32)
        sum_sq = tl.sum(x * x, axis=0)
        inv_rms = tl.rsqrt(sum_sq / HS + eps)
        w = tl.load(norm_w_ptr + c * HS + h, mask=mask_h, other=0.0).to(tl.float32)
        n = x * inv_rms * (1.0 + w)
        tl.store(
            normed_ptr + branch_base + h,
            n.to(normed_ptr.dtype.element_ty),
            mask=mask_h,
        )

    # P0-idle CTAs issue their second down-tile loads immediately, overlapping
    # the active branch tasks without carrying both tiles through P0 itself.
    down_w1 = tl.load(
        w_down_ptr + down_n1[:, None] * K + down_k1[None, :],
        mask=down_mask_n1[:, None],
        other=0.0,
    )

    _grid_barrier(counters_ptr + 0, num_ctas)

    # Consume the first preloaded down tile before W_up becomes live. This
    # bounds the overlap footprint while the second down tile remains available
    # to hide W_up latency.
    down_w0 = _consume_prefetched_bf16(down_w0)
    down_x0 = tl.load(
        normed_ptr + offs_m[:, None] * K + down_k0[None, :],
        mask=mask_m[:, None] & down_valid0,
        other=0.0,
    )
    down_acc0 = tl.dot(down_x0, tl.trans(down_w0))
    tl.atomic_add(
        t_raw_ptr + offs_m[:, None] * LOWRANK + down_n0[None, :],
        down_acc0,
        mask=mask_m[:, None] & down_mask_n0[None, :],
        sem="relaxed",
        scope="gpu",
    )

    # Issue the first PB tile's W_up loads before the remaining PA work. At
    # BLOCK_J=16, j_blocks=160, so one static tile covers every output CTA and
    # the entire 6.5 MB up matrix can overlap the down projection.
    offs_j = tl.arange(0, BLOCK_J)
    offs_c = tl.arange(0, HC)
    j_blocks = tl.cdiv(HS, BLOCK_J)
    pb_jb = pid
    pb_valid = pb_jb < j_blocks
    pb_j = pb_jb * BLOCK_J + offs_j
    pb_mask_j = (pb_j < HS) & pb_valid
    pb_cj = offs_c[:, None] * HS + pb_j[None, :]
    pb_cj_flat = tl.reshape(pb_cj, (HC * BLOCK_J,))
    pb_mask_cj = tl.reshape(
        tl.broadcast_to(pb_mask_j[None, :], (HC, BLOCK_J)),
        (HC * BLOCK_J,),
    )

    if BLOCK_R == 128:
        pb_offs_r = tl.arange(0, 128)
        pb_r0 = pb_offs_r
        pb_r1 = 128 + pb_offs_r
        pb_w0 = tl.load(
            w_up_ptr + pb_cj_flat[:, None] * LOWRANK + pb_r0[None, :],
            mask=pb_mask_cj[:, None] & (pb_r0[None, :] < LOWRANK),
            other=0.0,
        )
        pb_w1 = tl.load(
            w_up_ptr + pb_cj_flat[:, None] * LOWRANK + pb_r1[None, :],
            mask=pb_mask_cj[:, None] & (pb_r1[None, :] < LOWRANK),
            other=0.0,
        )
        pb_offs_tail = tl.arange(0, 64)
        pb_r2 = 256 + pb_offs_tail
        pb_w2 = tl.load(
            w_up_ptr + pb_cj_flat[:, None] * LOWRANK + pb_r2[None, :],
            mask=pb_mask_cj[:, None] & (pb_r2[None, :] < LOWRANK),
            other=0.0,
        )
    else:
        pb_offs_r = tl.arange(0, 64)
        pb_r0 = pb_offs_r
        pb_r1 = 64 + pb_offs_r
        pb_r2 = 128 + pb_offs_r
        pb_r3 = 192 + pb_offs_r
        pb_r4 = 256 + pb_offs_r
        pb_w0 = tl.load(
            w_up_ptr + pb_cj_flat[:, None] * LOWRANK + pb_r0[None, :],
            mask=pb_mask_cj[:, None] & (pb_r0[None, :] < LOWRANK),
            other=0.0,
        )
        pb_w1 = tl.load(
            w_up_ptr + pb_cj_flat[:, None] * LOWRANK + pb_r1[None, :],
            mask=pb_mask_cj[:, None] & (pb_r1[None, :] < LOWRANK),
            other=0.0,
        )
        pb_w2 = tl.load(
            w_up_ptr + pb_cj_flat[:, None] * LOWRANK + pb_r2[None, :],
            mask=pb_mask_cj[:, None] & (pb_r2[None, :] < LOWRANK),
            other=0.0,
        )
        pb_w3 = tl.load(
            w_up_ptr + pb_cj_flat[:, None] * LOWRANK + pb_r3[None, :],
            mask=pb_mask_cj[:, None] & (pb_r3[None, :] < LOWRANK),
            other=0.0,
        )
        pb_w4 = tl.load(
            w_up_ptr + pb_cj_flat[:, None] * LOWRANK + pb_r4[None, :],
            mask=pb_mask_cj[:, None] & (pb_r4[None, :] < LOWRANK),
            other=0.0,
        )

    # Finish PA while the preloaded W_up requests are in flight.
    down_w1 = _consume_prefetched_bf16(down_w1)
    down_x1 = tl.load(
        normed_ptr + offs_m[:, None] * K + down_k1[None, :],
        mask=mask_m[:, None] & down_valid1,
        other=0.0,
    )
    down_acc1 = tl.dot(down_x1, tl.trans(down_w1))
    tl.atomic_add(
        t_raw_ptr + offs_m[:, None] * LOWRANK + down_n1[None, :],
        down_acc1,
        mask=mask_m[:, None] & down_mask_n1[None, :],
        sem="relaxed",
        scope="gpu",
    )

    # Retain a generic tail for devices with fewer SMs or other supported
    # dimensions. Only 24 CTAs load one 16 KiB tile on the production shape.
    for tile in range(pid + 2 * num_ctas, down_tiles, num_ctas):
        nb = tile % n_blocks
        kc = tile // n_blocks
        n = nb * BLOCK_N + offs_n
        k = kc * BLOCK_K + offs_k
        mask_n = n < LOWRANK
        x = tl.load(
            normed_ptr + offs_m[:, None] * K + k[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )
        w = tl.load(
            w_down_ptr + n[:, None] * K + k[None, :],
            mask=mask_n[:, None],
            other=0.0,
        )
        acc = tl.dot(x, tl.trans(w))
        tl.atomic_add(
            t_raw_ptr + offs_m[:, None] * LOWRANK + n[None, :],
            acc,
            mask=mask_m[:, None] & mask_n[None, :],
            sem="relaxed",
            scope="gpu",
        )

    _grid_barrier(counters_ptr + 1, num_ctas)

    # Delay dot-layout conversion until after barrier 1. The global loads above
    # can therefore overlap PA instead of being consumed at their source.
    pb_w0 = _consume_prefetched_bf16(pb_w0)
    pb_w1 = _consume_prefetched_bf16(pb_w1)
    pb_w2 = _consume_prefetched_bf16(pb_w2)
    if BLOCK_R == 64:
        pb_w3 = _consume_prefetched_bf16(pb_w3)
        pb_w4 = _consume_prefetched_bf16(pb_w4)

    # PB: consume the preloaded W_up fragments. All five production weight
    # fragments were issued before PA completed, so their cold HBM latency is
    # no longer inside these dependent dot rounds. Each next t_raw load is
    # issued before the preceding dot to pipeline the remaining small reads.
    pb_acc = tl.zeros((ROWS, HC * BLOCK_J), dtype=tl.float32)
    pb_a0 = tl.load(
        t_raw_ptr + offs_m[:, None] * LOWRANK + pb_r0[None, :],
        mask=mask_m[:, None] & pb_valid & (pb_r0[None, :] < LOWRANK),
        other=0.0,
    )
    pb_a0 = pb_a0 * inv_hc
    pb_t0 = (pb_a0 * tl.sigmoid(pb_a0)).to(normed_ptr.dtype.element_ty)
    pb_a1 = tl.load(
        t_raw_ptr + offs_m[:, None] * LOWRANK + pb_r1[None, :],
        mask=mask_m[:, None] & pb_valid & (pb_r1[None, :] < LOWRANK),
        other=0.0,
    )
    pb_acc = tl.dot(pb_t0, tl.trans(pb_w0), pb_acc)

    pb_a1 = pb_a1 * inv_hc
    pb_t1 = (pb_a1 * tl.sigmoid(pb_a1)).to(normed_ptr.dtype.element_ty)
    pb_a2 = tl.load(
        t_raw_ptr + offs_m[:, None] * LOWRANK + pb_r2[None, :],
        mask=mask_m[:, None] & pb_valid & (pb_r2[None, :] < LOWRANK),
        other=0.0,
    )
    pb_acc = tl.dot(pb_t1, tl.trans(pb_w1), pb_acc)

    pb_a2 = pb_a2 * inv_hc
    pb_t2 = (pb_a2 * tl.sigmoid(pb_a2)).to(normed_ptr.dtype.element_ty)
    if BLOCK_R == 64:
        pb_a3 = tl.load(
            t_raw_ptr + offs_m[:, None] * LOWRANK + pb_r3[None, :],
            mask=mask_m[:, None] & pb_valid & (pb_r3[None, :] < LOWRANK),
            other=0.0,
        )
    pb_acc = tl.dot(pb_t2, tl.trans(pb_w2), pb_acc)

    if BLOCK_R == 64:
        pb_a3 = pb_a3 * inv_hc
        pb_t3 = (pb_a3 * tl.sigmoid(pb_a3)).to(normed_ptr.dtype.element_ty)
        pb_a4 = tl.load(
            t_raw_ptr + offs_m[:, None] * LOWRANK + pb_r4[None, :],
            mask=mask_m[:, None] & pb_valid & (pb_r4[None, :] < LOWRANK),
            other=0.0,
        )
        pb_acc = tl.dot(pb_t3, tl.trans(pb_w3), pb_acc)

        pb_a4 = pb_a4 * inv_hc
        pb_t4 = (pb_a4 * tl.sigmoid(pb_a4)).to(normed_ptr.dtype.element_ty)
        pb_acc = tl.dot(pb_t4, tl.trans(pb_w4), pb_acc)

    # The production LR=320 is fully covered above. Preserve the generic
    # multiple-of-64 contract with a pipelined on-demand tail.
    tail_offs_r = tl.arange(0, BLOCK_R)
    for r0 in tl.range(320, LOWRANK, BLOCK_R, num_stages=3):
        r = r0 + tail_offs_r
        mask_r = r < LOWRANK
        a = tl.load(
            t_raw_ptr + offs_m[:, None] * LOWRANK + r[None, :],
            mask=mask_m[:, None] & pb_valid & mask_r[None, :],
            other=0.0,
        )
        a = a * inv_hc
        t = (a * tl.sigmoid(a)).to(normed_ptr.dtype.element_ty)
        w = tl.load(
            w_up_ptr + pb_cj_flat[:, None] * LOWRANK + r[None, :],
            mask=pb_mask_cj[:, None] & mask_r[None, :],
            other=0.0,
        )
        pb_acc = tl.dot(t, tl.trans(w), pb_acc)

    pb_gate = tl.sigmoid(tl.reshape(pb_acc, (ROWS, HC, BLOCK_J)))
    pb_x = tl.load(
        normed_ptr
        + offs_m[:, None, None] * K
        + offs_c[None, :, None] * HS
        + pb_j[None, None, :],
        mask=mask_m[:, None, None] & pb_mask_j[None, None, :],
        other=0.0,
    ).to(tl.float32)
    pb_out = tl.sum(pb_gate * pb_x, axis=1) * inv_hc
    tl.store(
        mixed_ptr + offs_m[:, None] * HS + pb_j[None, :],
        pb_out.to(mixed_ptr.dtype.element_ty),
        mask=mask_m[:, None] & pb_mask_j[None, :],
    )

    # Only BLOCK_J=8 has a second tile per CTA on the 188-SM target. Keep a
    # pipelined fallback so the tuning option and smaller devices stay valid.
    offs_r = tl.arange(0, BLOCK_R)
    for jb in range(pid + num_ctas, j_blocks, num_ctas):
        j = jb * BLOCK_J + offs_j
        mask_j = j < HS
        cj = offs_c[:, None] * HS + j[None, :]
        cj_flat = tl.reshape(cj, (HC * BLOCK_J,))
        mask_cj = tl.reshape(
            tl.broadcast_to(mask_j[None, :], (HC, BLOCK_J)),
            (HC * BLOCK_J,),
        )
        up_acc = tl.zeros((ROWS, HC * BLOCK_J), dtype=tl.float32)
        for r0 in tl.range(0, LOWRANK, BLOCK_R, num_stages=3):
            r = r0 + offs_r
            mask_r = r < LOWRANK
            a = tl.load(
                t_raw_ptr + offs_m[:, None] * LOWRANK + r[None, :],
                mask=mask_m[:, None] & mask_r[None, :],
                other=0.0,
            )
            a = a * inv_hc
            t = (a * tl.sigmoid(a)).to(normed_ptr.dtype.element_ty)
            w = tl.load(
                w_up_ptr + cj_flat[:, None] * LOWRANK + r[None, :],
                mask=mask_cj[:, None] & mask_r[None, :],
                other=0.0,
            )
            up_acc = tl.dot(t, tl.trans(w), up_acc)

        gate = tl.sigmoid(tl.reshape(up_acc, (ROWS, HC, BLOCK_J)))
        x = tl.load(
            normed_ptr
            + offs_m[:, None, None] * K
            + offs_c[None, :, None] * HS
            + j[None, None, :],
            mask=mask_m[:, None, None] & mask_j[None, None, :],
            other=0.0,
        ).to(tl.float32)
        out = tl.sum(gate * x, axis=1) * inv_hc
        tl.store(
            mixed_ptr + offs_m[:, None] * HS + j[None, :],
            out.to(mixed_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_j[None, :],
        )

    # The last CTA reaches this point only after every PB reader is done. It
    # restores the shared split-K workspace and counters; kernel completion
    # stream-orders those stores before the next launch or graph replay.
    ticket = tl.atomic_add(counters_ptr + 2, 1, sem="acq_rel", scope="gpu")
    if ticket == num_ctas - 1:
        clear_offsets = tl.arange(0, 1024)
        # PA masks padded rows, so only rows touched by this launch need to be
        # restored. The untouched padding stays zero from initial allocation.
        clear_span = num_rows * LOWRANK
        for clear0 in range(0, clear_span, 1024):
            clear_idx = clear0 + clear_offsets
            tl.store(t_raw_ptr + clear_idx, 0.0, mask=clear_idx < clear_span)
        tl.store(counters_ptr + 0, 0)
        tl.store(counters_ptr + 1, 0)
        tl.store(counters_ptr + 2, 0)


_counters_cache: dict[torch.device, torch.Tensor] = {}
_workspace_cache: dict[tuple[torch.device, int], torch.Tensor] = {}


def _get_counters(device: torch.device) -> torch.Tensor:
    """Return this kernel's graph-replay-safe counters (not the old mixer's)."""
    buf = _counters_cache.get(device)
    if buf is None:
        buf = torch.zeros(3, dtype=torch.int32, device=device)
        _counters_cache[device] = buf
    return buf


def _get_workspace(device: torch.device, lowrank: int) -> torch.Tensor:
    """Get the same-stream scratch buffer, initialized once and reset in-kernel."""
    key = (device, lowrank)
    buf = _workspace_cache.get(key)
    if buf is None:
        buf = torch.zeros(
            (_HC_FUSED_MAX_ROWS, lowrank), dtype=torch.float32, device=device
        )
        _workspace_cache[key] = buf
    return buf


def _validate_common(
    hyper_input: torch.Tensor,
    norm_w: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    hc: int,
    hs: int,
) -> tuple[int, int, int]:
    if hyper_input.dim() != 2:
        raise ValueError("hyper_input must be 2-D")
    rows, k = hyper_input.shape
    lowrank = w_down.shape[0] if w_down.dim() == 2 else -1
    tensors = (hyper_input, norm_w, w_down, w_up)
    if not 1 <= rows <= _HC_FUSED_MAX_ROWS:
        raise ValueError(f"row count must be in [1, {_HC_FUSED_MAX_ROWS}]")
    if k != hc * hs or k % 2048 != 0:
        raise ValueError("HC*H must match the input width and be divisible by 2048")
    if lowrank <= 0 or lowrank % 64 != 0:
        raise ValueError("low-rank width must be a positive multiple of 64")
    if norm_w.shape != (k,) or w_down.shape != (lowrank, k):
        raise ValueError("invalid norm or down-projection weight shape")
    if w_up.shape != (k, lowrank):
        raise ValueError("invalid up-projection weight shape")
    if any(t.device != hyper_input.device for t in tensors):
        raise ValueError("all tensors must be on the same device")
    if any(not t.is_cuda for t in tensors):
        raise ValueError("the fused HC kernel requires CUDA tensors")
    if any(t.dtype != torch.bfloat16 for t in tensors):
        raise ValueError("the fused HC kernel supports bf16 only")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("all tensors must be contiguous")
    return rows, k, lowrank


def _launch(
    hyper_input: torch.Tensor,
    block_output: torch.Tensor,
    normed_prev: torch.Tensor,
    inject_w: torch.Tensor,
    norm_w: torch.Tensor,
    eps: float,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    hc: int,
    hs: int,
    *,
    has_combine: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    rows, k, lowrank = _validate_common(hyper_input, norm_w, w_down, w_up, hc, hs)
    if has_combine:
        extra = (block_output, normed_prev, inject_w)
        if block_output.shape != (rows, hs):
            raise ValueError("block_output has the wrong shape")
        if normed_prev.shape != (rows, k) or inject_w.shape != (hc, k):
            raise ValueError("invalid previous norm or inject-weight shape")
        if any(t.device != hyper_input.device for t in extra):
            raise ValueError("all tensors must be on the same device")
        if any(t.dtype != torch.bfloat16 for t in extra):
            raise ValueError("the fused HC kernel supports bf16 only")
        if any(not t.is_contiguous() for t in extra):
            raise ValueError("all tensors must be contiguous")

    device = hyper_input.device
    rows_pad = 16
    num_ctas = torch.cuda.get_device_properties(device).multi_processor_count
    new_residual = torch.empty_like(hyper_input) if has_combine else None
    normed = torch.empty_like(hyper_input)
    mixed = torch.empty((rows, hs), dtype=torch.bfloat16, device=device)
    t_raw = _get_workspace(device, lowrank)

    # Unused no-combine operands are valid device pointers so argument
    # marshalling stays graph-capturable; HAS_COMBINE eliminates their loads.
    block_arg = block_output if has_combine else hyper_input
    normed_prev_arg = normed_prev if has_combine else hyper_input
    inject_arg = inject_w if has_combine else norm_w
    new_residual_arg = new_residual if has_combine else hyper_input
    _hc_fused_persistent_kernel[(num_ctas,)](
        hyper_input,
        block_arg,
        normed_prev_arg,
        inject_arg,
        norm_w,
        w_down,
        w_up,
        new_residual_arg,
        normed,
        t_raw,
        mixed,
        _get_counters(device),
        k,
        lowrank,
        hs,
        rows,
        num_ctas,
        eps,
        1.0 / hc,
        ROWS=rows_pad,
        HC=hc,
        HAS_COMBINE=has_combine,
        BLOCK_P0=4096,
        BLOCK_N=32,
        BLOCK_K=256,
        BLOCK_J=_HC_FUSED_BLOCK_J,
        BLOCK_R=_HC_FUSED_BLOCK_R,
        num_warps=8,
    )
    return mixed, normed, new_residual


def hc_fused_norm_mix(
    hyper_input: torch.Tensor,
    norm_w: torch.Tensor,
    eps: float,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    hc: int,
    hs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run per-branch Gemma RMSNorm and HC low-rank mixing in one kernel."""
    mixed, normed, _ = _launch(
        hyper_input,
        hyper_input,
        hyper_input,
        norm_w,
        norm_w,
        eps,
        w_down,
        w_up,
        hc,
        hs,
        has_combine=False,
    )
    return mixed, normed


def hc_fused_combine_norm_mix(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    normed_prev: torch.Tensor,
    inject_w: torch.Tensor,
    norm_w: torch.Tensor,
    eps: float,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    hc: int,
    hs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run gated combine, per-branch norm, and HC mixing in one kernel."""
    mixed, normed, new_residual = _launch(
        residual,
        block_output,
        normed_prev,
        inject_w,
        norm_w,
        eps,
        w_down,
        w_up,
        hc,
        hs,
        has_combine=True,
    )
    assert new_residual is not None
    return mixed, new_residual, normed
