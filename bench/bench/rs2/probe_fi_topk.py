"""RS2b probe: FlashInfer's radix top-k (as SV2 calls it) for the draft proposal's top-64 (Vd 49152).

Checks its order against RS2's int64 keys (value desc, index asc), that a CUDA graph replay on new logits matches
eager, and times it against RS2's split kernels and a finalize over 64 candidates. Same environment as
bench_e_verify.py.
"""

import importlib.util
from pathlib import Path

import torch
from flashinfer import top_k as flashinfer_top_k


def load(name):
    spec = importlib.util.spec_from_file_location(name=name, location=Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = load("sweep_proposal")
test, rs2 = sweep.test, sweep.rs2
K = sweep.K
# (deterministic, tie_break, sorted): tie_break 1 prefers smaller indices and implies deterministic. Unsorted
# modes only need the same set: a finalize over the 64 entries can sort them by RS2's keys itself.
MODES = {"det": (True, 0, True), "det+small": (True, 1, True), "det+small unsorted": (True, 1, False),
         "fast": (False, 0, True), "fast unsorted": (False, 0, False)}


def rs2_order(logits):
    keys = rs2._proposal_topk_keys(logits=logits, k_block=K, topk_impl="torch")
    return keys, 0xFFFFFFFF - (keys & 0xFFFFFFFF)


def fi(rows, mode):
    deterministic, tie_break, ordered = MODES[mode]
    return flashinfer_top_k(input=rows, k=K, sorted=ordered, deterministic=deterministic, tie_break=tie_break)


def graph_matches_eager(logits, mode):
    out = {}
    fi(logits, mode)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out["vals"], out["idx"] = fi(logits, mode)
    same = True
    for _ in range(3):
        logits.copy_(torch.randn_like(logits))
        graph.replay()
        vals, idx = fi(logits, mode)
        torch.cuda.synchronize()
        same &= torch.equal(out["vals"], vals) and torch.equal(out["idx"], idx)
    return same


def main():
    torch.manual_seed(20261001)
    for n in (1, 4):
        case = sweep.make_case(dtype=torch.float32, n=n)
        logits = case["logits"]
        odd = logits.clone()
        odd[0, 5], odd[-1, 7] = float("nan"), -0.0
        if n > 1:
            odd[1], odd[2] = float("nan"), -float("inf")
        for name, rows in (("random", logits), ("ties", (logits * 4).round() / 4), ("nan", odd)):
            _, ref = rs2_order(rows)
            for mode in MODES:
                try:
                    vals, idx = fi(rows, mode)
                except Exception as error:
                    print(f"  order n={n} {name:6s} {mode:18s}: ERROR {type(error).__name__}: {error}", flush=True)
                    continue
                print(f"  order n={n} {name:6s} {mode:18s}: same order {torch.equal(idx.long(), ref)}, "
                      f"same set {torch.equal(idx.long().sort(dim=1).values, ref.sort(dim=1).values)}, "
                      f"values {torch.equal(vals.sort(dim=1, descending=True).values, torch.gather(rows, 1, ref))}, "
                      f"idx dtype {idx.dtype}", flush=True)
        for mode in MODES:
            scratch = torch.randn_like(logits)
            try:
                print(f"  graph n={n} {mode:18s}: replay on new logits matches eager "
                      f"{graph_matches_eager(scratch, mode)}", flush=True)
            except Exception as error:
                print(f"  graph n={n} {mode:18s}: ERROR {type(error).__name__}: {error}", flush=True)
        case["keys"][K], _ = rs2_order(logits)
        operations = {f"flashinfer {mode}": (lambda mode=mode: fi(logits, mode)) for mode in MODES}
        operations |= {
            "torch.topk": lambda: torch.topk(input=logits, k=K, dim=-1, sorted=True),
            "RS2 partial 2048/4": lambda: sweep.partial(case=case, block=2048, warps=4),
            "RS2 final 1536/4": lambda: sweep.final(case=case, block=2048, warps=4),
        }
        for warps in (1, 2, 4):
            operations[f"final over 64 /{warps}"] = lambda warps=warps: sweep.final(case=case, block=K, warps=warps)
        for name, operation in operations.items():
            try:
                gpu_us, wall_us = test.graph_time(operation=operation, calls=14)
            except Exception as error:
                print(f"  time n={n} {name:22s} ERROR {type(error).__name__}: {str(error).splitlines()[0][:160]}",
                      flush=True)
                continue
            print(f"  time n={n} {name:22s} GPU={gpu_us:7.3f} us/call wall={wall_us:7.3f} us/call", flush=True)


if __name__ == "__main__":
    main()
