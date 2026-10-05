"""SV2 acceptance test on the CPU (Triton interpreter): sparse_topk(logits, kp) against torch.topk.
Needs bench/sv2/sol_triton_topk.patch applied to the worktree (an alternative Triton top-k kernel; rejected, not in b92809af70).

usage: CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 python check_topk_cpu.py <worktree>

Contract checked per row:
  * values equal torch.topk(..., sorted=True) values bit for bit (same dtype, sorted descending);
  * logits.gather(idx) == values, indices unique and int64;
  * every index whose logit is strictly above the kp-th value is in the set (ties at the kp-th value may
    pick any of the tied ids).
"""
import os
import sys
import time

os.environ.setdefault("TRITON_INTERPRET", "1")
wt = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/tools/sglang-sv2")
sys.path.insert(0, os.path.join(wt, "python"))

import torch  # noqa: E402

from sglang.kernels.ops.speculative.sparse_verify import sparse_topk  # noqa: E402

torch.manual_seed(1234)
V_REAL = 248320


def make(mode, n, v, dtype):
    if mode == "randn":
        x = torch.randn(n, v) * 4
    elif mode == "ties":
        x = torch.randint(0, 40, (n, v)).float()
    elif mode == "peaked":
        x = torch.randn(n, v)
        x[:, torch.randint(0, v, (5,))] += 30
    elif mode == "masked":
        x = torch.full((n, v), float("-inf"))
        keep = torch.randint(0, v, (n, 20))
        x.scatter_(1, keep, torch.randn(n, 20))
    elif mode == "edges":
        x = torch.randn(n, v)
        x[:, :70] = 100.0  # many equal maxima at the low ids
        x[:, -3:] = 1e30
    else:
        raise ValueError(mode)
    return x.to(dtype).contiguous()


def check(mode, n, v, kp, dtype):
    x = make(mode, n, v, dtype)
    t0 = time.time()
    vals, idx = sparse_topk(x, kp)
    dt = time.time() - t0
    rv, _ = torch.topk(x, kp, dim=-1, largest=True, sorted=True)
    bad = []
    if vals.shape != (n, kp) or idx.shape != (n, kp):
        bad.append(f"shape {tuple(vals.shape)} {tuple(idx.shape)}")
    if vals.dtype != x.dtype or idx.dtype != torch.int64:
        bad.append(f"dtype {vals.dtype} {idx.dtype}")
    if not bad:
        if not torch.equal(vals, rv):
            bad.append("values differ from torch.topk")
        if not torch.equal(x.gather(1, idx), vals):
            bad.append("gather(idx) != values")
        for r in range(n):
            ids = idx[r].tolist()
            if len(set(ids)) != kp:
                bad.append(f"row {r}: duplicate indices")
                break
            kth = rv[r, -1]
            must = set(torch.nonzero(x[r] > kth).flatten().tolist())
            if not must <= set(ids):
                bad.append(f"row {r}: misses {len(must - set(ids))} ids above the kp-th value")
                break
    status = "ok " if not bad else "BAD"
    print(f"{status} {mode:7s} n={n:2d} v={v:6d} kp={kp:3d} {str(dtype)[6:]:8s} {dt:6.1f}s {'; '.join(bad)}")
    return not bad


cases = []
for dtype in (torch.float32, torch.bfloat16):
    for mode in ("randn", "ties", "peaked", "masked", "edges"):
        cases.append((mode, 4, 20000, 64, dtype))
cases += [
    ("randn", 16, 20000, 64, torch.float32),
    ("ties", 8, 5000, 128, torch.bfloat16),
    ("randn", 1, 3000, 16, torch.float32),
    ("randn", 4, V_REAL, 64, torch.float32),
    ("ties", 4, V_REAL, 64, torch.bfloat16),
]
ok = all([check(*c) for c in cases])
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
