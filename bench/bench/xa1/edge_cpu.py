"""Edge rows for the integrated kernel (worktree qsa/decode_attn.py), CPU interpreter only.

Rows with nothing visible (seq_len 0, all -1) must give zeros, a row whose only valid column sits in
the last split must give that V row, and a row with holes mid-row plus columns past seq_len must match
the fp32 reference; arrival counters must be zero afterwards.

  cd bench; export XA1_SGLANG_PY=~/tools/sglang-xa/python; source xa1/env.sh
  TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= python -m xa1.edge_cpu
"""
import os
import sys

if os.environ.get("TRITON_INTERPRET") != "1":
    sys.exit("run with TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES=")

import torch  # noqa: E402
from sglang.srt.layers.attention.qsa.decode_attn import (  # noqa: E402
    QSADecodeAttnWorkspace, qsa_decode_attention)

torch.manual_seed(0)
HQ, HKV, D, NC, CTX = 24, 2, 256, 2067, 3000
pool = CTX + 64
k = (torch.randn(pool, HKV, D) * 2).to(torch.float8_e4m3fn)
v = (torch.randn(pool, HKV, D) * 2).to(torch.float8_e4m3fn)
r2t = torch.zeros(3, CTX + 64, dtype=torch.int32)
r2t[1, :CTX] = torch.randperm(CTX, dtype=torch.int32) + 1
rows = 4
q = torch.randn(rows, HQ, D).to(torch.bfloat16)
idx = torch.full((rows, NC), -1, dtype=torch.int32)
seq = torch.tensor([0, 2500, 2500, 2500], dtype=torch.int32)
idx[0, :100] = torch.arange(100)            # seq_len 0: nothing visible
# row 1: all -1
idx[2, NC - 1] = 2499                        # one valid column, last split only
idx[3, :2051] = torch.randperm(2400, dtype=torch.int32)[:2051]
idx[3, 1000:1003] = -1                       # holes mid-row
idx[3, 2051:2060] = torch.arange(2400, 2409, dtype=torch.int32)
idx[3, 2060:] = 2600                         # past seq_len: masked
reqs = torch.ones(rows, dtype=torch.int32)
ws = QSADecodeAttnWorkspace(num_kv_heads=HKV, head_dim=D, device="cpu")


def ref(r):
    pos = idx[r]
    ok = (pos >= 0) & (pos < seq[r])
    if not ok.any():
        return torch.zeros(HQ, D)
    slots = r2t[1, pos[ok].long()].long()
    kk, vv = k[slots].float(), v[slots].float()
    out = []
    for h in range(HQ):
        s = (q[r, h].float() @ kk[:, h // 12].T) * D ** -0.5
        out.append(torch.softmax(s, 0) @ vv[:, h // 12])
    return torch.stack(out)


for prefix_valid in (False,):
    out = qsa_decode_attention(q=q, k_buffer=k, v_buffer=v, req_to_token=r2t, row_req_pool_indices=reqs,
                               topk_indices=idx, seq_lens=seq, sm_scale=D ** -0.5, workspace=ws,
                               prefix_valid=prefix_valid)
    for r in range(rows):
        rr = ref(r)
        err = (out[r].float() - rr).abs().max().item()
        print(f"prefix_valid={prefix_valid} row {r}: finite={bool(torch.isfinite(out[r].float()).all())} "
              f"max_abs_err={err:.3e} ref_max={rr.abs().max().item():.3f}")
    print("counters zero:", bool((ws.arrivals == 0).all()))
# prefix_valid=True on rows 0 and 1 only (fresh-row contract holds there)
out = qsa_decode_attention(q=q[:2].contiguous(), k_buffer=k, v_buffer=v, req_to_token=r2t,
                           row_req_pool_indices=reqs[:2], topk_indices=idx[:2].contiguous(), seq_lens=seq[:2],
                           sm_scale=D ** -0.5, workspace=ws, prefix_valid=True)
print("prefix_valid=True empty rows all zero:", bool((out.float() == 0).all()), "counters zero:", bool((ws.arrivals == 0).all()))
