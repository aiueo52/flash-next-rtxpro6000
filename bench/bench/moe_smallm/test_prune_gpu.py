#!/usr/bin/env python3
"""GPU unit test for sglang.srt.layers.moe.prune_singleton.

    SGLANG_MOE_PRUNE_SINGLETON_TAU=0.03 \
    SGLANG_PYTHON=<flash-next-fast checkout>/python \
    python \
        bench/moe_smallm/test_prune_gpu.py

(SGLANG_PYTHON is optional when the fork's sglang is installed in the venv.)

Checks the Triton kernel against a numpy reference, and that the same kernel
inside a CUDA graph gives the same answer on replay with fresh inputs.
"""
if __name__ != "__main__":  # collected by pytest: this is a GPU experiment script, not a unit test
    import pytest
    pytest.skip("GPU experiment script: run it directly (see the docstring)", allow_module_level=True)
import os
import sys

os.environ.setdefault("SGLANG_MOE_PRUNE_SINGLETON_TAU", "0.03")
import numpy as np  # noqa: E402
import torch  # noqa: E402

if os.environ.get("SGLANG_PYTHON"):  # python/ of a flash-next-fast checkout; else the installed sglang
    sys.path.insert(0, os.environ["SGLANG_PYTHON"])
from sglang.srt.layers.moe import prune_singleton as ps  # noqa: E402


def ref(ids, w, tau, min_rank):
    n, k = ids.shape
    flat = ids.ravel()
    cnt = np.array([(flat == e).sum() for e in flat]).reshape(n, k)
    order = np.argsort(-w, axis=1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(k)[None, :], axis=1)
    prune = (cnt == 1) & (w < tau) & (rank >= min_rank)
    ids2, w2 = ids.copy(), w.copy()
    ids2[prune] = -1
    w2[prune] = 0.0
    return ids2, w2, prune


def make(M, k, E=512, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.stack([rng.choice(E, k, replace=False) for _ in range(M)])
    w = rng.random((M, k))
    w = np.sort(w, axis=1)[:, ::-1].copy()
    w /= w.sum(1, keepdims=True)
    return ids.astype(np.int32), w.astype(np.float32)


def run(M, k, dtype_ids, seed):
    ids, w = make(M, k, seed=seed)
    ti = torch.from_numpy(ids).to("cuda", dtype_ids)
    tw = torch.from_numpy(w).cuda()
    ps.maybe_prune_singleton_routes(ti, tw)
    torch.cuda.synchronize()
    ei, ew, pr = ref(ids, w, ps.TAU, ps.MIN_RANK)
    oi = ti.cpu().numpy().astype(np.int32)
    ow = tw.cpu().numpy()
    ok = (oi == ei).all() and np.allclose(ow, ew, atol=0, rtol=0)
    print(f"  M={M:<4} k={k} ids={str(dtype_ids):<14} pruned={pr.sum():>4} "
          f"({pr.sum()/max(1,M):.2f}/row)  match={ok}")
    if not ok:
        bad = np.argwhere(oi != ei)[:5]
        print("   first mismatches:", bad, oi[oi != ei][:5], ei[oi != ei][:5])
    return ok


print(f"config: {ps.describe()}  enabled={ps.ENABLED}")
allok = True
for M in (2, 4, 8, 16, 32):
    for dt in (torch.int32, torch.int64):
        allok &= run(M, 10, dt, seed=M + (0 if dt == torch.int32 else 100))

# --- CUDA graph: capture once, replay with fresh inputs ---------------------
ids, w = make(16, 10, seed=7)
ti = torch.from_numpy(ids).cuda().to(torch.int32)
tw = torch.from_numpy(w).cuda()
ps.maybe_prune_singleton_routes(ti.clone(), tw.clone())   # warm the JIT
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        ps.maybe_prune_singleton_routes(ti, tw)
torch.cuda.current_stream().wait_stream(s)
with torch.cuda.graph(g):
    ps.maybe_prune_singleton_routes(ti, tw)
for seed in (11, 12, 13):
    ids2, w2 = make(16, 10, seed=seed)
    ti.copy_(torch.from_numpy(ids2).cuda().to(torch.int32))
    tw.copy_(torch.from_numpy(w2).cuda())
    g.replay()
    torch.cuda.synchronize()
    ei, ew, pr = ref(ids2, w2, ps.TAU, ps.MIN_RANK)
    ok = (ti.cpu().numpy() == ei).all() and np.allclose(tw.cpu().numpy(), ew)
    allok &= ok
    print(f"  graph replay seed={seed} pruned={pr.sum():>4} match={ok}")

# --- device time, measured inside a CUDA graph (no Python in the loop) ------
import time  # noqa: E402


def device_us(M, k, reps=200):
    ids, w = make(M, k, seed=3)
    ti = torch.from_numpy(ids).cuda().to(torch.int32)
    tw = torch.from_numpy(w).cuda()
    for _ in range(5):
        ps.maybe_prune_singleton_routes(ti, tw)
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            ps.maybe_prune_singleton_routes(ti, tw)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            ps.maybe_prune_singleton_routes(ti, tw)
    g.replay(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / (10 * reps) * 1e6


print(f"\n  device time inside a CUDA graph ({ps.describe()}):")
for M in (4, 16, 32):
    print(f"    M={M:<3} k=10: {device_us(M, 10):6.2f} us/call")

print("ALL OK" if allok else "FAILURES")
sys.exit(0 if allok else 1)
