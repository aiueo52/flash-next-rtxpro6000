"""Triton W8A16 skinny GEMM: y[M,N] = x[M,K](bf16) @ W[N,K](fp8 e4m3)^T * scale[N] (fp32).
Targets decode/verify shapes (M <= 16). Weight-only FP8: no activation quantization.

Two things decide throughput here, both measured in bench/w8a16/:

- Weight layout. Fp8LinearMethod stores the weight [K, N] and hands over the [N, K]
  transpose, so N is the contiguous axis and a [BLOCK_K, BLOCK_N] tile is the natural
  load; bf16_gemv passes a [N, K] weight, where [BLOCK_N, BLOCK_K] is. Loading in the
  wrong orientation costs 25% on large N and 3x on N=2560/K=6144.
- Grid size. Small-N shapes (shared-expert gate_up/down, GDN out_proj, in_proj_ba)
  launch far fewer CTAs than the machine has SMs when each CTA owns a whole K
  reduction, so the dispatcher splits K across CTAs and reduces fp32 partials in a
  second launch. That pays for the extra launch only while M is small.
"""
import functools

import torch, triton, triton.language as tl

@triton.jit
def _w8a16_gemv_kernel(x_ptr, w_ptr, s_ptr, y_ptr, M, N, K,
                       stride_xm, stride_xk, stride_wn, stride_wk, stride_ym, stride_yn,
                       PER_CHANNEL: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                       M_PAD: tl.constexpr, SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr,
                       PARTIAL: tl.constexpr, W_KN: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, M_PAD)
    offs_k = tl.arange(0, BLOCK_K)
    n_mask = offs_n < N
    m_mask = offs_m < M
    x_base = x_ptr + offs_m[:, None] * stride_xm
    acc = tl.zeros((M_PAD, BLOCK_N), dtype=tl.float32)
    for k0 in range(pid_k * BLOCK_K, K, SPLIT_K * BLOCK_K):
        kk = k0 + offs_k
        k_mask = kk < K
        if EVEN_K:
            x = tl.load(x_base + kk[None, :] * stride_xk, mask=m_mask[:, None], other=0.0)
        else:
            x = tl.load(x_base + kk[None, :] * stride_xk,
                        mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        # W_KN: the [N,K] view has stride (1, N), so a [BLOCK_K, BLOCK_N] tile is the
        # contiguous one and tl.dot takes it directly instead of transposing in smem.
        if W_KN:
            wp = w_ptr + kk[:, None] * stride_wk + offs_n[None, :] * stride_wn
            wm = n_mask[None, :] if EVEN_K else (k_mask[:, None] & n_mask[None, :])
            acc += tl.dot(x, tl.load(wp, mask=wm, other=0.0).to(tl.bfloat16),
                          out_dtype=tl.float32)
        else:
            wp = w_ptr + offs_n[:, None] * stride_wn + kk[None, :] * stride_wk
            wm = n_mask[:, None] if EVEN_K else (n_mask[:, None] & k_mask[None, :])
            acc += tl.dot(x, tl.trans(tl.load(wp, mask=wm, other=0.0).to(tl.bfloat16)),
                          out_dtype=tl.float32)
    if PARTIAL:
        # y_ptr is the fp32 [SPLIT_K, M_PAD, N] workspace; scale is applied by the reducer.
        p = y_ptr + pid_k * (M_PAD * N) + offs_m[:, None] * N + offs_n[None, :]
        tl.store(p, acc, mask=m_mask[:, None] & n_mask[None, :])
        return
    if PER_CHANNEL:
        acc = acc * tl.load(s_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
    else:
        acc = acc * tl.load(s_ptr)
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _w8a16_gemv_m1_kernel(x_ptr, w_ptr, s_ptr, y_ptr, N, K,
                          stride_wn, stride_wk, stride_xk, stride_yn,
                          PER_CHANNEL: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                          SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr, PARTIAL: tl.constexpr,
                          W_KN: tl.constexpr):
    """M == 1: broadcast-multiply + reduction instead of a 16-row tl.dot."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(pid_k * BLOCK_K, K, SPLIT_K * BLOCK_K):
        kk = k0 + offs_k
        k_mask = kk < K
        if EVEN_K:
            xv = tl.load(x_ptr + kk * stride_xk)
        else:
            xv = tl.load(x_ptr + kk * stride_xk, mask=k_mask, other=0.0)
        if W_KN:
            wp = w_ptr + kk[:, None] * stride_wk + offs_n[None, :] * stride_wn
            wm = n_mask[None, :] if EVEN_K else (k_mask[:, None] & n_mask[None, :])
            w = tl.load(wp, mask=wm, other=0.0)
            acc += tl.sum(w.to(tl.float32) * xv[:, None].to(tl.float32), axis=0)
        else:
            wp = w_ptr + offs_n[:, None] * stride_wn + kk[None, :] * stride_wk
            wm = n_mask[:, None] if EVEN_K else (n_mask[:, None] & k_mask[None, :])
            w = tl.load(wp, mask=wm, other=0.0)
            acc += tl.sum(w.to(tl.float32) * xv[None, :].to(tl.float32), axis=1)
    if PARTIAL:
        tl.store(y_ptr + pid_k * N + offs_n, acc, mask=n_mask)
        return
    if PER_CHANNEL:
        acc = acc * tl.load(s_ptr + offs_n, mask=n_mask, other=0.0)
    else:
        acc = acc * tl.load(s_ptr)
    tl.store(y_ptr + offs_n * stride_yn, acc.to(tl.bfloat16), mask=n_mask)


@triton.jit
def _w8a16_reduce_kernel(p_ptr, s_ptr, y_ptr, M, N, stride_ym, stride_yn,
                         PER_CHANNEL: tl.constexpr, SPLIT_K: tl.constexpr,
                         BLOCK_N: tl.constexpr, M_PAD: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, M_PAD)
    n_mask = offs_n < N
    mask = (offs_m < M)[:, None] & n_mask[None, :]
    base = p_ptr + offs_m[:, None] * N + offs_n[None, :]
    acc = tl.zeros((M_PAD, BLOCK_N), dtype=tl.float32)
    for i in tl.static_range(SPLIT_K):
        acc += tl.load(base + i * (M_PAD * N), mask=mask, other=0.0)
    if PER_CHANNEL:
        acc = acc * tl.load(s_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
    else:
        acc = acc * tl.load(s_ptr)
    tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
             acc.to(tl.bfloat16), mask=mask)


_NUM_SMS = None


def _num_sms(device) -> int:
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count
    return _NUM_SMS


# Measured on RTX PRO 6000 Blackwell (sm_120, 188 SMs) for the Qwen3.8-Flash-Next decode
# shapes; re-run bench/w8a16/tune_w8a16.py after a kernel or hardware change.
# key: (contiguous_n, w_bytes, M_bucket, N, K)
#   -> (BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages)
_TUNED = {
    # fp8 weight stored [K, N] (Fp8LinearMethod passes layer.weight.t())
    (True, 1, 1, 1280, 2560): (64, 256, 10, 2, 2),    # shared_expert gate_up
    (True, 1, 4, 1280, 2560): (64, 64, 8, 4, 4),
    (True, 1, 16, 1280, 2560): (32, 64, 4, 4, 4),
    (True, 1, 1, 2560, 640): (32, 512, 2, 4, 2),      # shared_expert down
    (True, 1, 4, 2560, 640): (32, 64, 2, 4, 4),
    (True, 1, 16, 2560, 640): (16, 64, 1, 4, 4),
    (True, 1, 1, 2560, 6144): (64, 256, 12, 4, 2),    # GDN out_proj, attention o_proj
    (True, 1, 4, 2560, 6144): (128, 64, 8, 8, 4),
    (True, 1, 16, 2560, 6144): (64, 64, 4, 8, 4),
    (True, 1, 1, 16384, 2560): (128, 128, 4, 2, 2),   # GDN in_proj_qkvz
    (True, 1, 4, 16384, 2560): (64, 64, 1, 4, 4),
    (True, 1, 16, 16384, 2560): (128, 64, 1, 4, 4),
    (True, 1, 1, 13312, 2560): (128, 128, 4, 4, 4),   # attention qkv_proj (with output gate)
    (True, 1, 4, 13312, 2560): (128, 64, 5, 2, 2),
    (True, 1, 16, 13312, 2560): (256, 64, 2, 4, 4),
    (True, 1, 1, 32768, 2560): (64, 256, 1, 2, 4),    # draft head
    (True, 1, 1, 248320, 2560): (128, 128, 1, 2, 4),  # lm_head
    (True, 1, 4, 248320, 2560): (128, 64, 1, 4, 4),
    (True, 1, 16, 248320, 2560): (128, 64, 1, 4, 4),
    # bf16 weight stored [N, K] (bf16_gemv passes linear.weight)
    (False, 2, 1, 96, 2560): (8, 256, 10, 4, 2),      # GDN in_proj_ba
    (False, 2, 4, 96, 2560): (16, 128, 20, 2, 2),
    (False, 2, 16, 96, 2560): (16, 128, 20, 4, 2),
    (False, 2, 1, 512, 2560): (8, 256, 10, 4, 2),     # MoE router gate
    (False, 2, 4, 512, 2560): (16, 128, 5, 2, 4),
    (False, 2, 16, 512, 2560): (16, 256, 1, 2, 4),
}


def _m_bucket(M: int) -> int:
    return 1 if M == 1 else (4 if M <= 4 else 16)


def _largest_divisor_le(n: int, cap: int) -> int:
    d = 1
    for c in range(1, min(n, cap) + 1):
        if n % c == 0:
            d = c
    return d


# One pipeline stage buffers the w tile plus the x tile in shared memory. sm_120 offers
# 100 KB/CTA; cap a single stage well under that and derive the depth from what is left.
_SMEM_PER_STAGE = 40960
_SMEM_TOTAL = 98304


def _stage_bytes(bn: int, bk: int, M: int, w_bytes: int) -> int:
    return bk * (bn * w_bytes + (2 if M == 1 else 2 * 16))


@functools.lru_cache(maxsize=None)
def _plan(M: int, N: int, K: int, contig_n: bool, sms: int, w_bytes: int):
    """Pick (BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages) for one call."""
    tuned = _TUNED.get((contig_n, w_bytes, _m_bucket(M), N, K))
    if tuned is not None and _stage_bytes(tuned[0], tuned[1], M, w_bytes) * tuned[4] <= _SMEM_TOTAL:
        return tuned
    min_n = 16 if (M > 1 or contig_n) else 8  # tl.dot needs a 16-wide tile
    target = 2 * sms
    best, best_score = None, -1.0
    for bn in (128, 64, 32, 16, 8):
        if bn < min_n or (bn > 16 and bn >= 2 * triton.next_power_of_2(N)):
            continue
        n_tiles = triton.cdiv(N, bn)
        # Each split writes and re-reads M*N fp32; keep that under half the weight bytes.
        want = max(1, min(target // n_tiles, K * w_bytes // (16 * M)))
        for bk in (512, 256, 128, 64, 32, 16):
            if bn * bk > (16384 if M == 1 else 65536) or bk > max(16, 2 * triton.next_power_of_2(K)):
                continue
            stage = _stage_bytes(bn, bk, M, w_bytes)
            if stage > _SMEM_PER_STAGE:
                continue
            n_kblk = triton.cdiv(K, bk)
            # SPLIT_K must divide the k-block count or the splits do unequal work.
            sk = _largest_divisor_le(n_kblk, want)
            ctas = n_tiles * sk
            # One contiguous run per load is bn bytes for the fp8 [K,N] layout and
            # 2*bk bytes for the bf16 [N,K] layout; 128 B is a full cache line.
            run = bn if contig_n else 2 * bk
            score = (min(ctas, target) / target) * (min(run, 128) / 128)
            score *= 1.0 if sk == 1 else 0.93  # the reducer costs one extra launch
            score *= 1.0 if n_kblk // sk > 1 else 0.9
            score *= 1.0 if ctas <= 4 * sms else 0.9
            if score > best_score:
                best, best_score = (bn, bk, sk, n_kblk // sk), score
    bn, bk, sk, iters = best
    stages = min(4, max(2, _SMEM_TOTAL // _stage_bytes(bn, bk, M, w_bytes)), max(2, iters))
    return (bn, bk, sk, 4, stages)


def _launch(x, w, s, y, M, N, K, per_channel, cfg):
    block_n, block_k, split_k, num_warps, num_stages = cfg
    even_k = K % (block_k * split_k) == 0
    grid = (triton.cdiv(N, block_n), split_k)
    use_m1 = M == 1
    m_pad = 1 if use_m1 else 16
    if split_k > 1:
        part = torch.empty((split_k, m_pad, N), dtype=torch.float32, device=x.device)
        out, sy0, sy1, partial = part, 0, 0, True
    else:
        out, sy0, sy1, partial = y, y.stride(0), y.stride(1), False
    if use_m1:
        _w8a16_gemv_m1_kernel[grid](x, w, s, out, N, K,
                                    w.stride(0), w.stride(1), x.stride(1), sy1,
                                    PER_CHANNEL=per_channel, BLOCK_N=block_n, BLOCK_K=block_k,
                                    SPLIT_K=split_k, EVEN_K=even_k, PARTIAL=partial,
                                    W_KN=w.stride(0) == 1,
                                    num_warps=num_warps, num_stages=num_stages)
    else:
        _w8a16_gemv_kernel[grid](x, w, s, out, M, N, K,
                                 x.stride(0), x.stride(1), w.stride(0), w.stride(1), sy0, sy1,
                                 PER_CHANNEL=per_channel, BLOCK_N=block_n, BLOCK_K=block_k,
                                 M_PAD=16, SPLIT_K=split_k, EVEN_K=even_k, PARTIAL=partial,
                                 W_KN=w.stride(0) == 1,
                                 num_warps=num_warps, num_stages=num_stages)
    if split_k > 1:
        rblock = 256 if N >= 256 else triton.next_power_of_2(N)
        _w8a16_reduce_kernel[(triton.cdiv(N, rblock),)](
            part, s, y, M, N, y.stride(0), y.stride(1),
            PER_CHANNEL=per_channel, SPLIT_K=split_k, BLOCK_N=rblock, M_PAD=m_pad, num_warps=4)
    return y


def w8a16_gemv(x: torch.Tensor, w: torch.Tensor, scale: torch.Tensor, cfg=None):
    """x: [M,K] bf16; w: [N,K] fp8_e4m3 (any strides); scale: [N] or [N,1] or scalar fp32."""
    M, K = x.shape
    N = w.shape[0]
    assert w.shape[1] == K and M <= 16
    y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    s = scale.reshape(-1)
    per_channel = s.numel() > 1
    if cfg is None:
        cfg = _plan(M, N, K, w.stride(0) == 1, _num_sms(x.device), w.element_size())
    return _launch(x, w, s, y, M, N, K, per_channel, cfg)


_ONES = {}


def bf16_gemv(x: torch.Tensor, w: torch.Tensor, cfg=None):
    """Skinny BF16 GEMM y = x @ w^T for tiny-N linears where cuBLAS picks a poor kernel.
    x: [M,K] bf16 (M<=16); w: [N,K] bf16."""
    M, K = x.shape
    N = w.shape[0]
    assert w.shape[1] == K and M <= 16
    y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    one = _ONES.get(x.device)
    if one is None:
        one = torch.ones(1, dtype=torch.float32, device=x.device)
        _ONES[x.device] = one
    if cfg is None:
        cfg = _plan(M, N, K, w.stride(0) == 1, _num_sms(x.device), w.element_size())
    return _launch(x, w, one, y, M, N, K, False, cfg)
