"""XA1 on ST1 rows: prefix_valid=True vs False on index-shared draft rows whose valid columns form a prefix.

ST1 places the drafted tail right after the anchor row's valid prefix, so every valid column sits below
seq_len and the kernel may stop at min(seq_len, NCOLS). Checks accuracy against the fp32 reference and
times both settings in CUDA graphs.

  export XA1_SGLANG_PY=~/tools/sglang-xa-st1/python; source bench/xa1/env.sh
  flock -w 28800 ~/.gpu.lock python bench/xa1/st1_prefix_check.py   # 18 s; results/xa1-st1-prefix.log
"""
import statistics

import torch

from xa1 import cases
from sglang.srt.layers.attention.qsa.decode_attn import (
    QSADecodeAttnWorkspace, qsa_decode_attention, qsa_decode_attention_supported)


def st1_layout(topk):
    out = torch.full_like(topk, -1)
    for r in range(topk.shape[0]):
        valid = topk[r][topk[r] >= 0]
        out[r, : valid.numel()] = valid
    return out


def call(case, lay, topk, ws, prefix_valid):
    return qsa_decode_attention(
        q=lay.q, k_buffer=lay.k_pool, v_buffer=lay.v_pool, req_to_token=case.req_to_token,
        row_req_pool_indices=case.row_req, topk_indices=topk, seq_lens=lay.seq_lens,
        sm_scale=cases.SCALING, workspace=ws, prefix_valid=prefix_valid)


def graph_us(fn, calls=20, replays=50):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls):
            fn()
    for _ in range(5):
        g.replay()
    times = []
    for _ in range(7):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(replays):
            g.replay()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e) * 1000 / (calls * replays))
    return statistics.median(times)


def main():
    ws = QSADecodeAttnWorkspace(num_kv_heads=cases.HKV, head_dim=cases.D, device="cuda")
    worst_gap = 0.0
    for ctx in (300, 700, 1500, 2100, 4000, 8192):
        for rows, tail_width, n_tail in ((1, 4, 2), (2, 4, 3), (1, 8, 6)):
            case = cases.Case(rows=rows, ctx=ctx, layers=1, mode="shared", tail_width=tail_width,
                              n_tail=n_tail, device="cuda", seed=ctx * 10 + rows)
            lay = case.layers[0]
            hole = lay.topk
            st1 = st1_layout(hole)
            assert qsa_decode_attention_supported(q=lay.q, k_buffer=lay.k_pool, v_buffer=lay.v_pool,
                                                  topk_indices=st1)
            assert bool(((st1 >= 0).sum(1) <= lay.seq_lens).all()), "valid count exceeds seq_len"
            ref = cases.reference(case, lay)
            outs = {
                "hole,scan-all": call(case, lay, hole, ws, False),
                "st1,scan-all": call(case, lay, st1, ws, False),
                "st1,prefix": call(case, lay, st1, ws, True),
            }
            torch.cuda.synchronize()
            err = {k: (v.float() - ref).abs().max().item() for k, v in outs.items()}
            gap = (outs["st1,prefix"].float() - outs["st1,scan-all"].float()).abs().max().item()
            worst_gap = max(worst_gap, gap)
            t_all = graph_us(lambda: call(case, lay, st1, ws, False))
            t_pre = graph_us(lambda: call(case, lay, st1, ws, True))
            print(f"ctx={ctx:5d} rows={rows} tail_width={tail_width} n_tail={n_tail} cols={st1.shape[1]} "
                  f"max|err| vs fp32: hole {err['hole,scan-all']:.2e} st1/all {err['st1,scan-all']:.2e} "
                  f"st1/prefix {err['st1,prefix']:.2e}; prefix vs all {gap:.2e}; "
                  f"time all {t_all:.2f} us prefix {t_pre:.2f} us delta {t_pre - t_all:+.2f}", flush=True)
            assert err["st1,prefix"] <= max(2 * err["st1,scan-all"], 2e-2), "prefix path less accurate"
    print(f"PASS: prefix_valid on ST1 rows matches the full scan (worst gap {worst_gap:.2e})", flush=True)


if __name__ == "__main__":
    main()
