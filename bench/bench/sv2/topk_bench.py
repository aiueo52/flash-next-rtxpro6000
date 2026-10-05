"""SV2 candidates at the sparse verify's shapes (fp32 logits, V = 248320): torch.topk vs FlashInfer's radix
top_k. Correctness against torch.topk, then GPU time per call. The "triton" rows in runs/sv2/topk1.jsonl come
from the alternative Triton sparse_topk (bench/sv2/alt_triton_topk.patch on 4f9cf50619; rejected, slower than torch at n >= 8).

usage (p3 FlashInfer with the rq2u2h cache, no ninja; GPU lock held by the caller):
  PYTHONPATH=~/tools/flashinfer-p3:<wt>/python FLASHINFER_WORKSPACE_BASE=~/.cache/sglang-rq2u2h \
  FLASHINFER_P2_NO_NINJA=1 TRITON_CACHE_DIR=~/.cache/sv2-triton python topk_bench.py <out.jsonl>

Timing per call, median of 7 rounds of 20 calls:
  graph  CUDA graph replay (GPU time only);
  queued eager calls queued behind a ~4 ms torch.cuda._sleep, timed from the end of the sleep (the server's
         case: the CPU is ahead of the GPU, so launch cost is hidden);
  sync   eager calls with a sync after each one (CPU launch cost exposed; pessimistic).
"""
import json
import statistics
import sys

import torch

import flashinfer

V = 248320
ROWS = (1, 4, 8, 16, 32)
KPS = (64, 128)
MODES = ("randn", "peaked", "ties", "masked")
ROUNDS, CALLS = 7, 20

VARIANTS = {
    "torch": lambda x, k: torch.topk(x, k, dim=-1, largest=True, sorted=True),
    "fi_sorted": lambda x, k: flashinfer.top_k(x, k, sorted=True),
    "fi_det_sorted": lambda x, k: flashinfer.top_k(x, k, sorted=True, deterministic=True),
    "fi_unsorted": lambda x, k: flashinfer.top_k(x, k, sorted=False),
}


def make(mode, n):
    g = torch.Generator(device="cuda").manual_seed(n * 7 + len(mode))
    if mode == "ties":
        return torch.randint(0, 40, (n, V), device="cuda", generator=g).float()
    if mode == "masked":
        x = torch.full((n, V), float("-inf"), device="cuda")
        keep = torch.randint(0, V, (n, 300), device="cuda", generator=g)
        return x.scatter_(1, keep, torch.randn(n, 300, device="cuda", generator=g))
    x = torch.randn(n, V, device="cuda", generator=g) * 4
    if mode == "peaked":
        x[:, torch.randint(0, V, (12,), device="cuda", generator=g)] += 30
    return x


def check(x, k, vals, idx, sort_vals):
    ref, _ = torch.topk(x, k, dim=-1, largest=True, sorted=True)
    bad = []
    if vals.shape != ref.shape or idx.shape != ref.shape or vals.dtype != x.dtype or idx.dtype != torch.int64:
        return [f"shape/dtype {tuple(vals.shape)} {vals.dtype} {tuple(idx.shape)} {idx.dtype}"]
    v = torch.sort(vals, dim=-1, descending=True).values if sort_vals else vals
    if not torch.equal(v, ref):
        bad.append("values differ from torch.topk")
    if not torch.equal(x.gather(1, idx), vals):
        bad.append("gather(idx) != values")
    s = torch.sort(idx, dim=-1).values
    if bool((s[:, 1:] == s[:, :-1]).any()):
        bad.append("duplicate ids")
    sel = torch.zeros_like(x, dtype=torch.bool).scatter_(1, idx, True)
    if bool(((x > ref[:, -1:]) & ~sel).any()):
        bad.append("misses an id above the pivot")
    return bad


def t_graph(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(CALLS):
            fn()
    g.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(ROUNDS):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        out.append(a.elapsed_time(b) * 1000 / CALLS)
    return statistics.median(out)


def t_queued(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(ROUNDS):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(10_000_000)
        a.record()
        for _ in range(CALLS):
            fn()
        b.record()
        b.synchronize()
        out.append(a.elapsed_time(b) * 1000 / CALLS)
    return statistics.median(out)


def t_sync(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(ROUNDS):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(CALLS):
            fn()
            torch.cuda.synchronize()
        b.record()
        b.synchronize()
        out.append(a.elapsed_time(b) * 1000 / CALLS)
    return statistics.median(out)


def main():
    f = open(sys.argv[1], "w")
    failed = False
    for k in KPS:
        for n in ROWS:
            for mode in MODES:
                x = make(mode, n)
                for name, var in VARIANTS.items():
                    rec = {"k": k, "n": n, "mode": mode, "variant": name}
                    fn = lambda: var(x, k)
                    try:
                        vals, idx = fn()
                        rec["errors"] = check(x, k, vals, idx, sort_vals=name == "fi_unsorted")
                    except Exception as e:
                        rec["errors"] = [f"{type(e).__name__}: {e}"]
                    failed |= bool(rec["errors"])
                    if mode in ("randn", "peaked") and not rec["errors"]:
                        for tm, t in (("graph", t_graph), ("queued", t_queued), ("sync", t_sync)):
                            try:
                                rec[tm] = round(t(fn), 2)
                            except Exception as e:
                                rec[tm] = f"{type(e).__name__}: {str(e)[:120]}"
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    print(json.dumps(rec), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
