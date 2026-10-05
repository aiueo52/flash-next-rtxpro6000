"""Production QSA packing on index-shared draft decode rows, on the CPU with the production code.

The production lookup (QSAMTPSharedSparseIndices, CPU path) builds the draft decode rows from a fresh
anchor row; the production valid-count pass and _compact_kv (Triton interpreter) pack them exactly as
_forward_trtllm_sparse does (column c of row r -> packed slot r * stride + c), and XQA then attends
packed slots [0, valid_count). K holds position + 1 in a float32 pool, so a packed 0 is a slot that
no kernel wrote (stale scratch in the server). Prints, per draft step, which visible positions the
production path drops and how many stale slots it attends instead.

  source bench/xa1/env.sh; TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= python -m xa1.hole_demo
"""
import json
import os
import sys

import torch

from xa1 import cases

HERE = os.path.dirname(os.path.abspath(__file__))


def _packed_positions(row, seq_len, ctx):
    from sglang.srt.layers.attention.qsa.sparse_attn import (
        qwen_sparse_kv_extraction_compact_triton, qwen_sparse_valid_counts_triton)

    topk = row.numel()
    stride = (topk + cases.PAGE - 1) // cases.PAGE * cases.PAGE
    k = torch.zeros((ctx, cases.HKV, cases.D), dtype=torch.float32)
    k[:, :, 0] = torch.arange(ctx, dtype=torch.float32)[:, None] + 1
    r2t = torch.zeros((2, ctx), dtype=torch.int32)
    r2t[1] = torch.arange(ctx, dtype=torch.int32)  # identity page table
    seq = torch.tensor([seq_len], dtype=torch.int32)
    counts = torch.empty(1, dtype=torch.int32)
    qwen_sparse_valid_counts_triton(seq, row[None], counts, 1, topk)
    out_k = torch.zeros((stride, cases.HKV, cases.D), dtype=torch.float32)
    out_v = torch.zeros_like(out_k)
    cu = torch.tensor([0, stride], dtype=torch.int32)
    qwen_sparse_kv_extraction_compact_triton(
        k, k, r2t, torch.tensor([1], dtype=torch.int32), row[None].contiguous(), seq, cu,
        out_k, out_v, 1, topk)
    n = int(counts[0])
    return n, out_k[:n, 0, 0].long() - 1  # -1 = slot never written


def main():
    if os.environ.get("TRITON_INTERPRET") != "1":
        sys.exit("run with TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES=")
    from sglang.srt.layers.attention.qwen_sparse_attn_backend import QSAMTPSharedSparseIndices

    gen = torch.Generator().manual_seed(0)
    out = {}
    # (anchor length, steps): W8 runs 7 draft steps, tail_width = steps + 1 (eagle_worker_v2.py).
    for anchor_len, steps in ((1000, 7), (8184, 7), (8190, 7), (8191, 7)):
        ctx = anchor_len + steps + 1
        base = torch.randperm(ctx // cases.RATIO, generator=gen)[: cases.BLOCK_TOPK]
        fresh = cases.fresh_row(anchor_len, base, 0.75, gen)
        state = QSAMTPSharedSparseIndices(layer_ids=[0], num_requests=2,
                                          token_topk=cases.FINAL_TOPK, tail_width=steps + 1,
                                          device="cpu")
        req = torch.tensor([1])
        state.capture(fresh[None], req, torch.tensor([anchor_len]), 0)
        rows = []
        # Draft decode forwards run for steps 1 .. steps-1 (the first token comes from draft extend).
        for j in range(1, steps):
            position = anchor_len + j - 1
            row = state.lookup(req, torch.tensor([position]), 0)[0].to(torch.int32)
            visible = set(int(p) for p in row[(row >= 0) & (row <= position)])
            n, packed = _packed_positions(row, position + 1, ctx)
            attended = [int(p) for p in packed]
            dropped = sorted(visible - set(attended))
            stale = sum(1 for p in attended if p < 0)
            rows.append(dict(step=j, position=position, width=row.numel(), valid_count=n,
                             dropped=dropped, stale_slots=stale,
                             current_token_dropped=position in dropped))
            print(f"anchor {anchor_len:5d} step {j}: width {row.numel()} valid_count {n} "
                  f"dropped {dropped} stale slots {stale}")
        out[str(anchor_len)] = rows
    path = os.path.join(HERE, "results", "xa1-hole-demo.json")
    json.dump(out, open(path, "w"), indent=1)
    print("saved", path)


if __name__ == "__main__":
    main()
