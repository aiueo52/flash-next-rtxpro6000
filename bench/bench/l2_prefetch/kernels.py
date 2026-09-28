import torch
"""Tiny Triton kernels used to emulate the latency-bound glue chain."""

import triton
import triton.language as tl


@triton.jit
def _tiny_kernel(p, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    v = tl.load(p + offs, mask=m, other=0.0)
    tl.store(p + offs, v * 0.999 + 1.0, mask=m)


def tiny(buf, grid, block=256):
    _tiny_kernel[(grid,)](buf, buf.numel(), BLOCK=block)


@triton.jit
def _hcish_kernel(w, x, y, K, N, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """A small bandwidth-ish kernel: y[n] = sum_k w[n,k]*x[k] over an fp8 [N,K] slab
    (the HC low-rank mix reads ~3 MB per call)."""
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        wt = tl.load(w + offs_n[:, None] * K + offs_k[None, :], mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0.0)
        xv = tl.load(x + offs_k, mask=offs_k < K, other=0.0)
        acc += tl.sum(wt.to(tl.float32) * xv.to(tl.float32)[None, :], axis=1)
    tl.store(y + offs_n, acc, mask=offs_n < N)


def hcish(w, x, y, block_n=8, block_k=256):
    N, K = w.shape
    _hcish_kernel[(triton.cdiv(N, block_n),)](w, x, y, K, N, BLOCK_N=block_n, BLOCK_K=block_k)


@triton.jit
def _touch_kernel(p, nsec, sink, BLOCK: tl.constexpr):
    """One 4-byte load per 32-byte sector over [p, p + nsec*32) (p: int32*), grid-strided."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    acc = tl.zeros((BLOCK,), dtype=tl.int32)
    for start in range(pid * BLOCK, nsec, nprog * BLOCK):
        offs = start + tl.arange(0, BLOCK)
        acc += tl.load(p + offs * 8, mask=offs < nsec, other=0, volatile=True)
    tl.store(sink + pid * BLOCK + tl.arange(0, BLOCK), acc)


def triton_touch(t, sink, grid=32, block=1024, num_warps=8):
    """Touch every sector of tensor t's storage with a Triton kernel (no launch attributes)."""
    from l2ctl import sector_range
    base, nb = sector_range(t)
    _touch_kernel[(grid,)](_ptr_tensor(t, base), nb // 32, sink, BLOCK=block, num_warps=num_warps)


def _ptr_tensor(t, base):
    """An int32 view over t's untyped storage starting at `base` (32-byte aligned, <= data_ptr)."""
    st = t.untyped_storage()
    off = base - st.data_ptr()
    n = (st.nbytes() - off) // 4
    v = torch.empty(0, dtype=torch.int32, device=t.device)
    v.set_(st, off // 4, (n,), (1,))
    return v
