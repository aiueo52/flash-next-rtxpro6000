"""xa1: synthetic QSA decode inputs, the fp32 reference and the real server call path.

Shapes are the served model's full-attention layers (config.json text_config): 24 q heads,
2 kv heads (GQA 12), head_dim 256, fp8 e4m3 KV pool [slots, 2, 256] per layer, page 64.
Top-k rows follow qsa/kernel.py torch_expand_qsa_block_indices: 512 blocks x 4 tokens in
top-k (score) order, then the uncompressed tail (L % 4 tokens), then -1, 2051 columns. Index-shared
draft rows follow QSAMTPSharedSparseIndices.lookup: those 2051 frozen columns of the anchor row
plus tail_width columns [anchor+1 .. position], -1 beyond.
"""
from __future__ import annotations

import types

import torch

HQ, HKV, D = 24, 2, 256
GROUP = HQ // HKV
SCALING = D ** -0.5
RATIO, TOKEN_TOPK = 4, 2048
BLOCK_TOPK = TOKEN_TOPK // RATIO
FINAL_TOPK = TOKEN_TOPK + RATIO - 1
PAGE = 64


def fresh_row(L: int, base_blocks: torch.Tensor, overlap: float, gen: torch.Generator) -> torch.Tensor:
    """One expanded top-k row for a query whose visible length is L (position L - 1)."""
    nb = L // RATIO
    if nb <= BLOCK_TOPK:
        blocks = torch.randperm(nb, generator=gen)
    else:
        keep = base_blocks[base_blocks < nb]
        keep = keep[torch.randperm(keep.numel(), generator=gen)[: int(round(overlap * BLOCK_TOPK))]]
        taken = torch.zeros(nb, dtype=torch.bool)
        taken[keep] = True
        rest = (~taken).nonzero().flatten()
        rest = rest[torch.randperm(rest.numel(), generator=gen)[: BLOCK_TOPK - keep.numel()]]
        blocks = torch.cat([keep, rest])[torch.randperm(BLOCK_TOPK, generator=gen)]
    toks = (blocks[:, None] * RATIO + torch.arange(RATIO)[None, :]).flatten()
    tail = torch.arange(nb * RATIO, L)
    row = torch.full((FINAL_TOPK,), -1, dtype=torch.int32)
    vals = torch.cat([toks, tail]).to(torch.int32)
    row[: vals.numel()] = vals
    return row


def shared_row(anchor_L: int, position: int, tail_width: int, base_blocks, gen) -> torch.Tensor:
    frozen = fresh_row(anchor_L, base_blocks, 1.0, gen)
    tail = anchor_L + torch.arange(tail_width)
    tail = torch.where(tail <= position, tail, torch.full_like(tail, -1)).to(torch.int32)
    return torch.cat([frozen, tail])


class Layer:
    """Inputs of one attention call (one layer): q, the layer's KV pool, top-k rows, lengths."""

    def __init__(self, *, q, k_pool, v_pool, topk, seq_lens):
        self.q = q
        self.k_pool = k_pool
        self.v_pool = v_pool
        self.topk = topk
        self.seq_lens = seq_lens


class Case:
    """rows query rows of one request at context ctx (rows are its last positions) over layers."""

    def __init__(self, *, rows, ctx, layers, mode="fresh", overlap=0.75, tail_width=16, n_tail=4,
                 device="cpu", seed=0, k_std=2.0, pool_pages=None, r2t_width=None):
        gen = torch.Generator().manual_seed(seed)
        # Index rows are drawn on the CPU; q/k/v on the target device (CPU: same generator).
        dev_gen = gen if str(device) == "cpu" else torch.Generator(device=device).manual_seed(seed)
        self.rows, self.ctx, self.mode = rows, ctx, mode
        n_pages = pool_pages or 2 * ((ctx + PAGE - 1) // PAGE) + 4
        slots = n_pages * PAGE
        page_map = torch.randperm(n_pages, generator=gen)[: (ctx + PAGE - 1) // PAGE]
        pos = torch.arange(ctx)
        r2t_row = (page_map[pos // PAGE] * PAGE + pos % PAGE).to(torch.int32)
        # req_to_token row 0 is a dummy (the server reserves req-pool slot 0 for padding).
        self.req_to_token = torch.zeros((2, r2t_width or ctx + 64), dtype=torch.int32)
        self.req_to_token[1, :ctx] = r2t_row
        self.row_req = torch.ones(rows, dtype=torch.int32)
        self.layers = []
        for _ in range(layers):
            q = torch.randn((rows, HQ, D), generator=dev_gen, device=device).to(torch.bfloat16)
            k = torch.randn((slots, HKV, D), generator=dev_gen, device=device) * k_std
            k = k.to(torch.float8_e4m3fn)
            v = torch.randn((slots, HKV, D), generator=dev_gen, device=device).to(torch.float8_e4m3fn)
            base = torch.randperm(max(ctx // RATIO, 1), generator=gen)[:BLOCK_TOPK]
            if mode == "fresh":
                lens = torch.arange(ctx - rows + 1, ctx + 1, dtype=torch.int32)
                topk = torch.stack([fresh_row(int(L), base, overlap, gen) for L in lens])
            elif mode == "shared":
                lens = torch.full((rows,), ctx, dtype=torch.int32)
                anchor_L = ctx - n_tail
                topk = torch.stack([shared_row(anchor_L, ctx - 1, tail_width, base, gen)
                                    for _ in range(rows)])
            else:
                raise ValueError(mode)
            self.layers.append(Layer(q=q, k_pool=k, v_pool=v, topk=topk.contiguous(),
                                     seq_lens=lens))
        self.to(device)

    def to(self, device):
        self.req_to_token = self.req_to_token.to(device)
        self.row_req = self.row_req.to(device)
        for lay in self.layers:
            for name in ("q", "k_pool", "v_pool", "topk", "seq_lens"):
                setattr(lay, name, getattr(lay, name).to(device))
        return self

    def valid_mask(self, lay: Layer) -> torch.Tensor:
        return (lay.topk >= 0) & (lay.topk < lay.seq_lens[:, None])


def reference(case: Case, lay: Layer, *, scale=SCALING) -> torch.Tensor:
    """fp32 attention over every valid column (qsa_sparse_attention_reference semantics)."""
    valid = case.valid_mask(lay)
    safe = lay.topk.clamp(min=0).long()
    slots = case.req_to_token[case.row_req.long()[:, None], safe].long()
    k = lay.k_pool[slots].float()  # [R, C, HKV, D]
    v = lay.v_pool[slots].float()
    q = lay.q.float().view(case.rows, HKV, GROUP, D)
    s = torch.einsum("rhgd,rchd->rhgc", q, k) * scale
    s = s.masked_fill(~valid[:, None, None, :], float("-inf"))
    p = torch.softmax(s, dim=-1)
    p = torch.nan_to_num(p, nan=0.0)
    o = torch.einsum("rhgc,rchd->rhgd", p, v)
    return o.reshape(case.rows, HQ, D)


def prefix_reference(case: Case, lay: Layer, packed_k, packed_v, counts, stride, *, scale=SCALING):
    """fp32 attention over packed tokens [0, count) of each row: the XQA contract in the server."""
    outs = []
    for r in range(case.rows):
        n = int(counts[r])
        k = packed_k[r * stride: r * stride + n].float()
        v = packed_v[r * stride: r * stride + n].float()
        q = lay.q[r].float().view(HKV, GROUP, D)
        s = torch.einsum("hgd,chd->hgc", q, k) * scale
        p = torch.softmax(s, dim=-1)
        outs.append(torch.einsum("hgc,chd->hgd", p, v).reshape(HQ, D))
    return torch.stack(outs)


# ---------------------------------------------------------------- server call path (read-only)

def server_stub(req_to_token, max_rows):
    """Bare object carrying exactly the attributes QwenSparseAttnBackend._forward_trtllm_sparse
    reads, with the backend's own helper methods bound to it."""
    from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend as B

    class _Stub:
        _get_trtllm_sparse_tables = B._get_trtllm_sparse_tables
        _get_fa2_scratch = B._get_fa2_scratch
        _forward_trtllm_sparse = B._forward_trtllm_sparse

    stub = _Stub()
    stub._trtllm_sparse_tables = {}
    stub._fa2_scratch = {}
    stub._trtllm_workspace = None
    stub._cuda_graph_max_tokens = max_rows
    stub.req_to_token_pool = types.SimpleNamespace(req_to_token=req_to_token)
    return stub


def server_metadata(case: Case, lay: Layer, valid_counts):
    """CUDA-graph flavour of QwenSparseAttnMetadata as _forward_trtllm_sparse reads it."""
    return types.SimpleNamespace(sequence_lengths=lay.seq_lens, is_cuda_graph=True,
                                 fa2_valid_counts=valid_counts,
                                 row_req_pool_indices=case.row_req)


def server_call(stub, case: Case, lay: Layer, valid_counts, trtllm_decode):
    """The production QSA paged attention for one layer: valid counts + _compact_kv + decode."""
    return stub._forward_trtllm_sparse(
        lay.q, lay.k_pool, lay.v_pool, types.SimpleNamespace(scaling=SCALING),
        types.SimpleNamespace(req_pool_indices=case.row_req),
        server_metadata(case, lay, valid_counts), lay.topk, trtllm_decode)
