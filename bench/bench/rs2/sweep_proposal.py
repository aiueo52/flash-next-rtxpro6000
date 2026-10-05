"""RS2 draft proposal: sweep the split block and num_warps of its two kernels; outputs must stay bit-identical.

Same environment as bench_e_verify.py. Times as in test E: a CUDA graph of 14 calls, median GPU us per call.
"""

import importlib.util
import itertools
from pathlib import Path

import torch
import triton

spec = importlib.util.spec_from_file_location(name="rs2_test", location=Path(__file__).with_name("test_sparse_rs.py"))
test = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test)
assert test.DEV == "cuda"
rs2 = test.rs2

VOCAB, TARGET_VOCAB, K = 49152, 248320, 64
BLOCKS, WARPS_PARTIAL, WARPS_FINAL = (512, 1024, 2048, 4096), (2, 4, 8, 16), (4, 8, 16)


def make_case(*, dtype, n):
    sampling = test.params(temperatures=[0.8] * n, top_ks=[40] * n, top_ps=[0.95] * n, min_ps=[0.05] * n)
    return dict(
        logits=torch.randn(size=(n, VOCAB), dtype=dtype, device="cuda"), sampling=sampling,
        hot=torch.randperm(n=TARGET_VOCAB, device="cuda")[:VOCAB].long(),
        uniforms=torch.rand(size=(n,), device="cuda"),
        positions=torch.zeros(size=(n,), dtype=torch.int64, device="cuda"),
        chain=torch.zeros(size=(n, 14), dtype=torch.int64, device="cuda"),
        probs=torch.zeros(size=(n, K), dtype=torch.float32, device="cuda"),
        tokens=torch.zeros(size=(n, K), dtype=torch.int64, device="cuda"),
        topk_p=torch.zeros(size=(n, 1), dtype=torch.float32, device="cuda"),
        topk_index=torch.zeros(size=(n, 1), dtype=torch.int64, device="cuda"),
        keys={block: torch.empty(size=(n, triton.cdiv(VOCAB, block) * K), dtype=torch.int64, device="cuda")
              for block in BLOCKS},
    )


def partial(*, case, block, warps):
    n, splits = case["logits"].shape[0], triton.cdiv(VOCAB, block)
    rs2._draft_partial_topk_kernel[(n, splits)](
        Logits=case["logits"], Keys=case["keys"][block], stride_l=case["logits"].stride(0), VOCAB=VOCAB,
        SPLITS=splits, KB=K, BLOCK=block, num_warps=warps,
    )


def final(*, case, block, warps):
    keys, s = case["keys"][block], case["sampling"]
    rs2._draft_finalize_kernel[(keys.shape[0],)](
        Keys=keys, Temps=s["temperatures"], TopKs=s["top_ks"], TopPs=s["top_ps"], MinPs=s["min_ps"],
        Uniforms=case["uniforms"], HotTokens=case["hot"], Probs=case["probs"], Tokens=case["tokens"],
        TopkP=case["topk_p"], TopkIndex=case["topk_index"], Positions=case["positions"],
        DraftTokens=case["chain"], stride_q=case["probs"].stride(0), stride_t=case["tokens"].stride(0),
        stride_u=case["uniforms"].stride(0), stride_chain=case["chain"].stride(0), column=0,
        CANDIDATES=keys.shape[1], K=K, KB=K, HAS_MIN_P=True, HAS_MAP=True,
        WRITE_POSITION=True, WRITE_CHAIN=True, BLOCK=triton.next_power_of_2(keys.shape[1]), num_warps=warps,
    )


def outputs(*, case, block, warps_partial, warps_final):
    case["positions"].zero_()
    partial(case=case, block=block, warps=warps_partial)
    final(case=case, block=block, warps=warps_final)
    torch.cuda.synchronize()
    return [case[name].clone() for name in ("probs", "tokens", "topk_p", "topk_index", "chain", "positions")]


def main():
    torch.manual_seed(20261001)
    for dtype, n in itertools.product((torch.float32, torch.bfloat16), (1, 4)):
        case = make_case(dtype=dtype, n=n)
        reference = outputs(case=case, block=2048, warps_partial=4, warps_final=4)
        best = []
        for block, wp, wf in itertools.product(BLOCKS, WARPS_PARTIAL, WARPS_FINAL):
            try:
                same = all(torch.equal(a, b) for a, b in zip(
                    outputs(case=case, block=block, warps_partial=wp, warps_final=wf), reference))
                pair, _ = test.graph_time(operation=lambda: (partial(case=case, block=block, warps=wp),
                                                             final(case=case, block=block, warps=wf)), calls=14)
                part, _ = test.graph_time(operation=lambda: partial(case=case, block=block, warps=wp), calls=14)
                fin, _ = test.graph_time(operation=lambda: final(case=case, block=block, warps=wf), calls=14)
            except Exception as error:  # a config may not compile (registers, shared memory)
                print(f"  sweep dtype={dtype} n={n} block={block} wp={wp} wf={wf} ERROR {type(error).__name__}: "
                      f"{str(error).splitlines()[0][:120]}", flush=True)
                continue
            if same:
                best.append((pair, block, wp, wf))
            print(f"  sweep dtype={dtype} n={n} block={block:4d} wp={wp:2d} wf={wf:2d} pair={pair:7.3f} "
                  f"partial={part:7.3f} final={fin:7.3f} us identical={same}", flush=True)
        for pair, block, wp, wf in sorted(best)[:5]:
            print(f"BEST dtype={dtype} n={n} pair={pair:.3f} us block={block} wp={wp} wf={wf}", flush=True)


if __name__ == "__main__":
    main()
