"""xa1: split-KV Triton decode attention for the QSA sparse path (candidates A and B).

One kernel, two front ends:

* A (``xa1_decode_packed``): drop-in for the XQA call in ``_forward_trtllm_sparse``. Reads the
  page-aligned packed scratch written by ``_compact_kv`` (row r at tokens [r*stride, r*stride+n)),
  attends the first ``seq_lens[r]`` packed tokens -- exactly XQA's contract with the static
  arange block table the server passes.
* B (``xa1_decode_gather``): replaces valid-count + ``_compact_kv`` + XQA. Reads the KV pool
  directly through top-k logical indices and req_to_token; a column is attended iff
  0 <= index < seq_lens[row] (the semantics of ``_logical_to_physical`` + the torch reference).

Grid (splits, kv_heads, rows); each program covers a contiguous column chunk for one kv head and
all GROUP query heads (padded to BLOCK_H=16 for mma). QK^T and PV accumulate in fp32.

MMA input type (cfg.f16): f16 (default since v2) converts fp8 e4m3 K/V with one packed
cvt.f16x2.e4m3x2 per two elements; bf16 (v1) costs one scalar F2F per element (the v1 SASS has
128 F2F per thread per tile, the main per-tile cost). Q bf16 -> f16 is exact for |q| in
[6.1e-5, 65504]; e4m3 -> f16 is exact; P in [0, 1] keeps 11 bits in f16 (8 in bf16).

Pipelining (v3): Triton's pipeliner gives each load of an indirect chain
(num_stages - 1) // (levels + 1) stages, so B's pos -> slot -> K/V chain needs num_stages 7 for
double-buffered K/V (v1/v2 ran B at 2-3: one K/V buffer, every tile waited for its own loads).
cfg.num_stages is the K/V depth for both paths: A launches it as is, B as 3 * num_stages - 2.
B gathers K/V unmasked (invalid columns read slot 0, the pool's padding slot, and are masked
in the scores), so the loads are plain 16-byte cp.async.

V operand (cfg.vcvt): with tl's e4m3 -> f16 cast Triton keeps V as fp8 in shared memory and
loads the PV B operand byte by byte (V is d-contiguous, the MMA wants token pairs): 128 LDS.U8
per thread per 64-token tile, the largest per-tile cost in the v2 SASS. vcvt converts V with an
opaque inline-asm cvt in the load layout, so the operand goes through smem as f16 (ldmatrix.trans).

Splits combine deterministically: every program stores its partial (acc, m, l) -- fp32 acc, or
with cfg.part16 the normalised acc/l in f16 -- the last arriver per (row, kv head), found with one
acq_rel atomic counter, sums the partials in split order, writes bf16 and resets the counter to 0
(CUDA-graph replay safe; DG1 pattern). v3 sums [heads, D] tiles split by split (in-thread adds);
v2 (cfg.red3d) loaded [splits, heads, d-chunk] tiles and reduced across threads; v1 summed one
split per dependent L2 round trip (~0.3 us each).

v4 (cfg.lse): the v3 combine still paid one shared-memory layout conversion (2 barriers) per
split to move the [splits, heads] softmax weights into the row slice of the [heads, D] tile (31
BAR.SYNC at 11 splits). v4 stores acc / l in f16 plus lse = m + log2(l) and loads each split's lse
with a 1D load keyed by the head offsets only, which Triton places directly in that row slice:
the combine is a running max, then sum_s o_s * 2^(lse_s - max) / sum_s 2^(lse_s - max), all
in-thread, no smem and no barrier. Split order is fixed, so the result stays deterministic.
"""
from __future__ import annotations

import os
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from sglang.kernels.triton_pdl import pdl_trigger, pdl_wait

LOG2E = 1.4426950408889634
BLOCK_H = 16
_INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"  # no inline asm in the interpreter


class XA1Config(NamedTuple):
    num_splits: int
    block_n: int
    num_warps: int
    num_stages: int  # K/V pipeline depth (B launches 3 * num_stages - 2, see module doc)
    f16: bool = True
    part16: bool = False
    red3d: bool = False
    vcvt: bool = False
    lse: bool = False  # v4: f16 acc / l partials + lse, combined without layout conversions

    @property
    def key(self):
        flags = (("" if self.f16 else "/bf") + ("/h" if self.part16 else "")
                 + ("/r3" if self.red3d else "") + ("/vc" if self.vcvt else "")
                 + ("/l" if self.lse else ""))
        return "/".join(map(str, self[:4])) + flags


class XA1Workspace:
    """Split partials + per-(row, kv head) arrival counters; counters must start at 0."""

    def __init__(self, *, max_groups: int, max_splits: int, head_dim: int, device):
        self.max_groups = max_groups
        self.max_splits = max_splits
        self.ws_o = torch.empty(max_groups * max_splits * BLOCK_H * head_dim, dtype=torch.float32,
                                device=device)
        self.ws_ml = torch.empty(max_groups * max_splits * 2 * BLOCK_H, dtype=torch.float32,
                                 device=device)
        self.cnt = torch.zeros(max_groups, dtype=torch.int32, device=device)


@triton.jit
def _e4m3_to_f16_asm(x):
    # Same cvt as tl's cast, but opaque: Triton cannot move it past the dot operand's smem copy.
    return tl.inline_asm_elementwise(
        "{ .reg .b16 lo, hi; mov.b32 {lo, hi}, $2; cvt.rn.f16x2.e4m3x2 $0, lo; "
        "cvt.rn.f16x2.e4m3x2 $1, hi; }",
        "=r,=r,r", [x.to(tl.uint8, bitcast=True)], dtype=tl.float16, is_pure=False, pack=4)


@triton.jit
def _split_totals(ws_ml_ptr, grp, offs_h, BLOCK_H: tl.constexpr, NUM_SPLITS: tl.constexpr,
                  SPLITS_P2: tl.constexpr):
    """Global max and sum of exp-weighted l over the group's splits (padded splits weigh 0)."""
    offs_s = tl.arange(0, SPLITS_P2)
    smask = offs_s < NUM_SPLITS
    ml = ws_ml_ptr + ((grp * NUM_SPLITS + offs_s) * 2 * BLOCK_H)[:, None] + offs_h[None, :]
    m_all = tl.load(ml, mask=smask[:, None], other=-1.0e30, cache_modifier=".cg")
    l_all = tl.load(ml + BLOCK_H, mask=smask[:, None], other=0.0, cache_modifier=".cg")
    m_tot = tl.max(m_all, 0)
    l_tot = tl.sum(tl.exp2(m_all - m_tot[None, :]) * l_all, 0)
    return m_all, l_all, m_tot, l_tot


@triton.jit
def _combine_2d(ws_o_ptr, ws_ml_ptr, out_ptr, grp, out_scale, offs_h, offs_d, hmask,
                GROUP: tl.constexpr, BLOCK_H: tl.constexpr, D: tl.constexpr,
                NUM_SPLITS: tl.constexpr, SPLITS_P2: tl.constexpr, PART_F16: tl.constexpr):
    """v3: one [heads, D] partial per split, added in split order (in-thread)."""
    _, _, m_tot, l_tot = _split_totals(ws_ml_ptr, grp, offs_h, BLOCK_H, NUM_SPLITS, SPLITS_P2)
    o = tl.zeros([BLOCK_H, D], tl.float32)
    for s in tl.static_range(NUM_SPLITS):
        part = grp * NUM_SPLITS + s
        w = tl.exp2(tl.load(ws_ml_ptr + part * 2 * BLOCK_H + offs_h, cache_modifier=".cg") - m_tot)
        if PART_F16:
            w = w * tl.load(ws_ml_ptr + part * 2 * BLOCK_H + BLOCK_H + offs_h, cache_modifier=".cg")
        o_s = tl.load(ws_o_ptr + (part * BLOCK_H + offs_h)[:, None] * D + offs_d[None, :],
                      mask=hmask[:, None], other=0.0, cache_modifier=".cg")
        o += o_s.to(tl.float32) * w[:, None]
    o = o * (out_scale / tl.where(l_tot > 0, l_tot, 1.0))[:, None]
    tl.store(out_ptr + (grp * GROUP + offs_h)[:, None] * D + offs_d[None, :],
             o.to(out_ptr.dtype.element_ty), mask=hmask[:, None])


@triton.jit
def _combine_3d(ws_o_ptr, ws_ml_ptr, out_ptr, grp, out_scale, offs_h, hmask,
                GROUP: tl.constexpr, BLOCK_H: tl.constexpr, D: tl.constexpr,
                NUM_SPLITS: tl.constexpr, SPLITS_P2: tl.constexpr, RED_D: tl.constexpr,
                PART_F16: tl.constexpr):
    """v2: [splits, heads, d-chunk] tiles reduced over the split axis (fixed tree)."""
    m_all, l_all, m_tot, l_tot = _split_totals(ws_ml_ptr, grp, offs_h, BLOCK_H, NUM_SPLITS,
                                               SPLITS_P2)
    w = tl.exp2(m_all - m_tot[None, :])
    if PART_F16:
        w = w * l_all
    w = w * (out_scale / tl.where(l_tot > 0, l_tot, 1.0))[None, :]
    offs_s = tl.arange(0, SPLITS_P2)
    smask = offs_s < NUM_SPLITS
    offs_r = tl.arange(0, RED_D)
    o_base = ((grp * NUM_SPLITS + offs_s)[:, None] * BLOCK_H + offs_h[None, :]) * D
    omask = smask[:, None, None] & hmask[None, :, None]
    for d0 in tl.static_range(0, D, RED_D):
        o_s = tl.load(ws_o_ptr + o_base[:, :, None] + d0 + offs_r[None, None, :], mask=omask,
                      other=0.0, cache_modifier=".cg")
        o = tl.sum(o_s.to(tl.float32) * w[:, :, None], 0)
        tl.store(out_ptr + (grp * GROUP + offs_h)[:, None] * D + d0 + offs_r[None, :],
                 o.to(out_ptr.dtype.element_ty), mask=hmask[:, None])


@triton.jit
def _combine_lse(ws_o_ptr, ws_ml_ptr, out_ptr, grp, out_scale, offs_h, offs_d, hmask,
                 GROUP: tl.constexpr, BLOCK_H: tl.constexpr, D: tl.constexpr,
                 NUM_SPLITS: tl.constexpr):
    """v4: partials are acc / l in f16 plus lse = m + log2(l). The lse loads land straight in
    the row slice of the [heads, D] layout: no smem, no barrier."""
    lse_max = tl.full([BLOCK_H], -1.0e30, tl.float32)
    for s in tl.static_range(NUM_SPLITS):
        lse = tl.load(ws_ml_ptr + (grp * NUM_SPLITS + s) * BLOCK_H + offs_h, cache_modifier=".cg")
        lse_max = tl.maximum(lse_max, lse)
    den = tl.zeros([BLOCK_H], tl.float32)
    o = tl.zeros([BLOCK_H, D], tl.float32)
    for s in tl.static_range(NUM_SPLITS):
        part = grp * NUM_SPLITS + s
        w = tl.exp2(tl.load(ws_ml_ptr + part * BLOCK_H + offs_h, cache_modifier=".cg") - lse_max)
        den += w
        o_s = tl.load(ws_o_ptr + (part * BLOCK_H + offs_h)[:, None] * D + offs_d[None, :],
                      mask=hmask[:, None], other=0.0, cache_modifier=".cg")
        o += o_s.to(tl.float32) * w[:, None]
    o = o * (out_scale / den)[:, None]
    tl.store(out_ptr + (grp * GROUP + offs_h)[:, None] * D + offs_d[None, :],
             o.to(out_ptr.dtype.element_ty), mask=hmask[:, None])
    return o


@triton.jit
def _finish(acc, m_i, l_i, out_ptr, ws_o_ptr, ws_ml_ptr, cnt_ptr, grp, sid, out_scale, qo_off,
            offs_h, offs_d, hmask, GROUP: tl.constexpr, BLOCK_H: tl.constexpr, D: tl.constexpr,
            NUM_SPLITS: tl.constexpr, SPLITS_P2: tl.constexpr, RED_D: tl.constexpr,
            PART_F16: tl.constexpr, RED3D: tl.constexpr, LSE: tl.constexpr):
    """Store the output (one split) or this split's partial; the last arriver combines."""
    if NUM_SPLITS == 1:
        o = acc / tl.where(l_i > 0, l_i, 1.0)[:, None] * out_scale
        tl.store(out_ptr + qo_off, o.to(out_ptr.dtype.element_ty), mask=hmask[:, None])
    elif LSE:
        part = grp * NUM_SPLITS + sid
        o_off = (part * BLOCK_H + offs_h)[:, None] * D + offs_d[None, :]
        o = acc / tl.where(l_i > 0, l_i, 1.0)[:, None]
        tl.store(ws_o_ptr + o_off, o.to(ws_o_ptr.dtype.element_ty), mask=hmask[:, None],
                 cache_modifier=".cg")
        # An empty split (no valid column) weighs 2^-1e30 = 0.
        lse = tl.where(l_i > 0, m_i + tl.log2(tl.where(l_i > 0, l_i, 1.0)), -1.0e30)
        tl.store(ws_ml_ptr + part * BLOCK_H + offs_h, lse, cache_modifier=".cg")
        tl.debug_barrier()
        done = tl.atomic_add(cnt_ptr + grp, 1, sem="acq_rel", scope="gpu")
        if done == NUM_SPLITS - 1:
            _combine_lse(ws_o_ptr, ws_ml_ptr, out_ptr, grp, out_scale, offs_h, offs_d, hmask,
                         GROUP=GROUP, BLOCK_H=BLOCK_H, D=D, NUM_SPLITS=NUM_SPLITS)
            tl.store(cnt_ptr + grp, 0)
    else:
        part = grp * NUM_SPLITS + sid
        o_off = (part * BLOCK_H + offs_h)[:, None] * D + offs_d[None, :]
        if PART_F16:
            acc = acc / tl.where(l_i > 0, l_i, 1.0)[:, None]
        tl.store(ws_o_ptr + o_off, acc.to(ws_o_ptr.dtype.element_ty), mask=hmask[:, None],
                 cache_modifier=".cg")
        tl.store(ws_ml_ptr + part * 2 * BLOCK_H + offs_h, m_i, cache_modifier=".cg")
        tl.store(ws_ml_ptr + part * 2 * BLOCK_H + BLOCK_H + offs_h, l_i, cache_modifier=".cg")
        tl.debug_barrier()
        done = tl.atomic_add(cnt_ptr + grp, 1, sem="acq_rel", scope="gpu")
        if done == NUM_SPLITS - 1:
            if RED3D:
                _combine_3d(ws_o_ptr, ws_ml_ptr, out_ptr, grp, out_scale, offs_h, hmask,
                            GROUP=GROUP, BLOCK_H=BLOCK_H, D=D, NUM_SPLITS=NUM_SPLITS,
                            SPLITS_P2=SPLITS_P2, RED_D=RED_D, PART_F16=PART_F16)
            else:
                _combine_2d(ws_o_ptr, ws_ml_ptr, out_ptr, grp, out_scale, offs_h, offs_d, hmask,
                            GROUP=GROUP, BLOCK_H=BLOCK_H, D=D, NUM_SPLITS=NUM_SPLITS,
                            SPLITS_P2=SPLITS_P2, PART_F16=PART_F16)
            tl.store(cnt_ptr + grp, 0)


@triton.jit
def _xa1_decode_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    len_ptr, idx_ptr, r2t_ptr, req_ptr,
    ws_o_ptr, ws_ml_ptr, cnt_ptr,
    qk_scale, out_scale,
    idx_stride, r2t_stride,
    NCOLS: tl.constexpr,
    PACKED_STRIDE: tl.constexpr,
    HKV: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_H: tl.constexpr,
    D: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    SPLITS_P2: tl.constexpr,
    RED_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GATHER: tl.constexpr,
    PREFIX_VALID: tl.constexpr,
    USE_PDL: tl.constexpr,
    MMA_F16: tl.constexpr,
    PART_F16: tl.constexpr,
    RED3D: tl.constexpr,
    V_ASM: tl.constexpr,
    F32_DOT: tl.constexpr,
    LSE: tl.constexpr,
):
    sid = tl.program_id(0)
    kvh = tl.program_id(1)
    row = tl.program_id(2)
    grp = row * HKV + kvh
    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    hmask = offs_h < GROUP
    qo_off = (grp * GROUP + offs_h)[:, None] * D + offs_d[None, :]
    hd_off = kvh * D + offs_d

    pdl_wait(USE_PDL)
    q = tl.load(q_ptr + qo_off, mask=hmask[:, None], other=0.0)
    if MMA_F16:
        q = q.to(tl.float16)
    if GATHER:
        length = tl.load(len_ptr + row)
        req = tl.load(req_ptr + row).to(tl.int64)
        # Every column is checked on its own: index-shared draft rows have -1 holes mid-row.
        # Fresh rows keep their valid columns in front (all inside [0, seq_len)): skip the rest.
        ncols = tl.minimum(length, NCOLS) if PREFIX_VALID else NCOLS
        chunk = tl.cdiv(tl.cdiv(ncols, NUM_SPLITS), BLOCK_N) * BLOCK_N
        start = sid * chunk
        end = tl.minimum(start + chunk, ncols)
    else:
        n_valid = tl.load(len_ptr + row)
        chunk = tl.cdiv(tl.cdiv(n_valid, NUM_SPLITS), BLOCK_N) * BLOCK_N
        start = sid * chunk
        end = tl.minimum(start + chunk, n_valid)

    m_i = tl.full([BLOCK_H], -1.0e30, tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    acc = tl.zeros([BLOCK_H, D], tl.float32)
    for n0 in range(start, end, BLOCK_N):
        cols = n0 + offs_n
        cmask = cols < end
        if GATHER:
            pos = tl.load(idx_ptr + row * idx_stride + cols, mask=cmask, other=-1)
            valid = (pos >= 0) & (pos < length)
            slot = tl.load(r2t_ptr + req * r2t_stride + pos, mask=valid, other=0)
            kv_off = slot.to(tl.int64)[:, None] * (HKV * D) + hd_off[None, :]
            k = tl.load(k_ptr + kv_off)
            v = tl.load(v_ptr + kv_off)
        else:
            valid = cmask
            kv_off = (row * PACKED_STRIDE + cols)[:, None] * (HKV * D) + hd_off[None, :]
            k = tl.load(k_ptr + kv_off, mask=valid[:, None], other=0.0)
            v = tl.load(v_ptr + kv_off, mask=valid[:, None], other=0.0)
        if MMA_F16:
            k = k.to(tl.float16)
            if V_ASM:
                v = _e4m3_to_f16_asm(v)
            else:
                v = v.to(tl.float16)
        else:
            k = k.to(tl.bfloat16)
            v = v.to(tl.bfloat16)
        if F32_DOT:
            s = tl.dot(q.to(tl.float32), tl.trans(k.to(tl.float32)), input_precision="ieee")
        else:
            s = tl.dot(q, tl.trans(k))
        s = tl.where(valid[None, :], s * qk_scale, -1.0e30)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.where(valid[None, :], tl.exp2(s - m_new[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        p = p.to(v.dtype)
        if F32_DOT:
            acc += tl.dot(p.to(tl.float32), v.to(tl.float32), input_precision="ieee")
        else:
            acc += tl.dot(p, v)
        m_i = m_new

    _finish(acc, m_i, l_i, out_ptr, ws_o_ptr, ws_ml_ptr, cnt_ptr, grp, sid, out_scale, qo_off,
            offs_h, offs_d, hmask, GROUP=GROUP, BLOCK_H=BLOCK_H, D=D, NUM_SPLITS=NUM_SPLITS,
            SPLITS_P2=SPLITS_P2, RED_D=RED_D, PART_F16=PART_F16, RED3D=RED3D, LSE=LSE)
    pdl_trigger(USE_PDL)


def _launch(*, q, k, v, out, lens, idx, r2t, req, idx_stride, r2t_stride, ncols, packed_stride,
            bmm1_scale, bmm2_scale, cfg: XA1Config, ws: XA1Workspace, gather: bool, pdl: bool,
            f32_dot: bool, prefix_valid: bool = False, warmup: bool = False):
    rows, hq, d = q.shape
    hkv = k.shape[1]
    if cfg.num_splits > 1 and (rows * hkv > ws.max_groups or cfg.num_splits > ws.max_splits):
        raise ValueError(f"xa1 workspace too small for rows={rows} splits={cfg.num_splits}")
    grid = (cfg.num_splits, hkv, rows)
    splits_p2 = triton.next_power_of_2(cfg.num_splits)
    ws_o = ws.ws_o.view(torch.float16) if cfg.part16 or cfg.lse else ws.ws_o
    args = (q, k, v, out, lens, idx, r2t, req, ws_o, ws.ws_ml, ws.cnt,
            float(bmm1_scale) * LOG2E, float(bmm2_scale), idx_stride, r2t_stride)
    kwargs = dict(NCOLS=ncols, PACKED_STRIDE=packed_stride, HKV=hkv, GROUP=hq // hkv,
                  BLOCK_H=BLOCK_H, D=d, NUM_SPLITS=cfg.num_splits, SPLITS_P2=splits_p2,
                  RED_D=max(16, min(d, 512 // splits_p2)), BLOCK_N=cfg.block_n, GATHER=gather,
                  PREFIX_VALID=prefix_valid,
                  USE_PDL=pdl, MMA_F16=cfg.f16, PART_F16=cfg.part16, RED3D=cfg.red3d,
                  V_ASM=cfg.vcvt and cfg.f16 and not _INTERPRET, F32_DOT=f32_dot, LSE=cfg.lse,
                  num_warps=cfg.num_warps,
                  num_stages=3 * cfg.num_stages - 2 if gather else cfg.num_stages)
    if pdl:
        kwargs["launch_pdl"] = True
    if warmup:
        return _xa1_decode_kernel.warmup(*args, grid=grid, **kwargs)
    _xa1_decode_kernel[grid](*args, **kwargs)
    return out


def xa1_decode_packed(q, packed_k, packed_v, seq_lens, *, stride, bmm1_scale, bmm2_scale,
                      cfg: XA1Config, ws: XA1Workspace, pdl: bool, out=None, f32_dot=False,
                      warmup=False):
    """Candidate A. q [R, HQ, D] bf16; packed_k/v [>= R*stride, HKV, D] fp8 (NHD tokens)."""
    q = q.contiguous()
    if out is None:
        out = torch.empty_like(q)
    return _launch(q=q, k=packed_k, v=packed_v, out=out, lens=seq_lens, idx=seq_lens, r2t=seq_lens,
                   req=seq_lens, idx_stride=0, r2t_stride=0, ncols=0, packed_stride=stride,
                   bmm1_scale=bmm1_scale, bmm2_scale=bmm2_scale, cfg=cfg, ws=ws, gather=False,
                   pdl=pdl, f32_dot=f32_dot, warmup=warmup)


def xa1_decode_gather(q, k_buffer, v_buffer, req_to_token, row_req, topk_indices, seq_lens, *,
                      bmm1_scale, bmm2_scale, cfg: XA1Config, ws: XA1Workspace, pdl: bool,
                      prefix_valid: bool, out=None, f32_dot=False, warmup=False):
    """Candidate B. k/v_buffer [slots, HKV, D] fp8 pool; topk_indices [R, C] int32 logical.

    prefix_valid: rows are fresh QSA rows (expand kernel output: valid columns first, all of
    them < seq_len), so columns >= seq_len are skipped. False for index-shared draft rows."""
    q = q.contiguous()
    if out is None:
        out = torch.empty_like(q)
    return _launch(q=q, k=k_buffer, v=v_buffer, out=out, lens=seq_lens, idx=topk_indices,
                   r2t=req_to_token, req=row_req, idx_stride=topk_indices.stride(0),
                   r2t_stride=req_to_token.stride(0), ncols=topk_indices.shape[1],
                   packed_stride=0, bmm1_scale=bmm1_scale, bmm2_scale=bmm2_scale, cfg=cfg, ws=ws,
                   gather=True, pdl=pdl, f32_dot=f32_dot, prefix_valid=prefix_valid,
                   warmup=warmup)


def xqa_compatible(cfg: XA1Config, ws: XA1Workspace, pdl: bool):
    """Candidate A behind the exact call the server makes to trtllm_batch_decode_with_kv_cache."""

    def decode(*, query, kv_cache, workspace_buffer, block_tables, seq_lens, max_seq_len,
               bmm1_scale, bmm2_scale):
        kc, vc = kv_cache  # HND views [pages, HKV, page, D] of NHD packed scratch
        page = kc.shape[2]
        if max_seq_len != block_tables.shape[1] * page:
            raise ValueError("xa1: expected the static page-aligned block table of the QSA path")
        base_k = kc.permute(0, 2, 1, 3).reshape(-1, kc.shape[1], kc.shape[3])
        base_v = vc.permute(0, 2, 1, 3).reshape(-1, vc.shape[1], vc.shape[3])
        return xa1_decode_packed(query, base_k, base_v, seq_lens, stride=max_seq_len,
                                 bmm1_scale=bmm1_scale, bmm2_scale=bmm2_scale, cfg=cfg, ws=ws,
                                 pdl=pdl)

    return decode
