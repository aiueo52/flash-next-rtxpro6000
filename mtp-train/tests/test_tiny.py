"""CPU-only unit tests for the standalone MTP module (tiny config).

Run with:
    CUDA_VISIBLE_DEVICES="" python -m pytest tests -q
or directly:
    CUDA_VISIBLE_DEVICES="" python tests/test_tiny.py

Nothing here touches CUDA.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import torch
import torch.nn.functional as F

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mtptrain.config import DEFAULT_MODEL_DIR, MTPConfig  # noqa: E402
from mtptrain.data import Sample, causal_mask_from_valid, collate  # noqa: E402
from mtptrain.layers import GatedResidual, GemmaRMSNorm  # noqa: E402
from mtptrain.model import MTPHead  # noqa: E402
from mtptrain.weights import MTP_PREFIX  # noqa: E402

MODEL_DIR = DEFAULT_MODEL_DIR  # real checkpoint, optional (env MTP_MODEL_DIR)


def _build(seed: int = 7, dtype=torch.float32):
    torch.manual_seed(seed)
    cfg = MTPConfig.tiny()
    model = MTPHead(cfg).to(dtype)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float32).to(dtype) * 0.05)
    model.set_frozen_heads(
        (torch.randn(cfg.vocab_size, cfg.hidden_size, generator=g) * 0.05).to(dtype),
        (torch.randn(cfg.vocab_size, cfg.hidden_size, generator=g) * 0.05).to(dtype),
    )
    model.freeze_non_trainable()
    return cfg, model, g


def _inputs(cfg, g, b=2, t=12, dtype=torch.float32):
    ids = torch.randint(0, cfg.vocab_size, (b, t), generator=g)
    hc = torch.randn(b, t, cfg.hc_size, generator=g, dtype=torch.float32).to(dtype)
    pos = torch.arange(t).expand(b, t).contiguous()
    return ids, hc, pos


# ------------------------------------------------------------------ shapes
def test_shapes():
    cfg, model, g = _build()
    ids, hc, pos = _inputs(cfg, g)
    logits, own_hc = model(ids, hc, pos)
    assert logits.shape == (2, 12, cfg.vocab_size), logits.shape
    assert own_hc.shape == (2, 12, cfg.hc_size), own_hc.shape
    assert torch.isfinite(logits).all()


def test_parameter_names_match_checkpoint():
    """state_dict keys must be exactly the ``mtp.*`` names, 1:1, 31 of them."""
    cfg, model, _ = _build()
    ours = {MTP_PREFIX + k for k in model.state_dict()}
    assert len(ours) == 31, sorted(ours)
    index = os.path.join(MODEL_DIR, "model.safetensors.index.json")
    if not os.path.exists(index):
        print("  (checkpoint absent; skipped the real-name comparison)")
        return
    with open(index) as handle:
        real = {
            k for k in json.load(handle)["weight_map"] if k.startswith(MTP_PREFIX)
        }
    assert ours == real, (sorted(ours - real), sorted(real - ours))


def test_shapes_match_checkpoint():
    """Full-size config: every parameter shape equals the checkpoint tensor."""
    index = os.path.join(MODEL_DIR, "model.safetensors.index.json")
    if not os.path.exists(index):
        import pytest

        pytest.skip(f"checkpoint not found at {MODEL_DIR} (set MTP_MODEL_DIR)")
    import struct

    with open(index) as handle:
        wm = json.load(handle)["weight_map"]
    shapes = {}
    for shard in sorted({v for k, v in wm.items() if k.startswith(MTP_PREFIX)}):
        with open(os.path.join(MODEL_DIR, shard), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        for k, v in header.items():
            if k.startswith(MTP_PREFIX):
                shapes[k] = tuple(v["shape"])
    cfg = MTPConfig.from_pretrained(MODEL_DIR)
    model = MTPHead(cfg)
    for k, v in model.state_dict().items():
        want = shapes[MTP_PREFIX + k]
        assert tuple(v.shape) == want, (k, tuple(v.shape), want)
    assert sum(v.numel() for v in model.state_dict().values()) == sum(
        int(torch.tensor(s).prod()) for s in shapes.values()
    )


# ------------------------------------------------------------- input fusion
def test_fuse_inputs_matches_literal():
    """Literal transcription of ``_fuse_residual_linear_shared``."""
    cfg, model, g = _build()
    ids, hc, _ = _inputs(cfg, g, b=3, t=5)

    def rms(x, w):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps)
        return xf * (1.0 + w.float())

    emb = F.embedding(ids, model.embed_tokens)
    e = F.linear(rms(emb, model.pre_fc_norm_embedding.weight), model.fc_embedding.weight)
    n = rms(hc, model.pre_fc_norm_hidden.weight)
    streams = n.reshape(3, 5, cfg.hc_count, cfg.hidden_size)
    h4 = F.linear(streams, model.fc_hidden.weight)
    expected = (e.unsqueeze(-2) + h4).reshape(3, 5, cfg.hc_size)
    got = model.fuse_inputs(ids, hc)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


def test_pre_fc_norm_hidden_is_global_not_grouped():
    """The hidden branch normalises over all 4H features, unlike hc_norm."""
    cfg, model, g = _build()
    x = torch.randn(1, 1, cfg.hc_size, generator=g)
    with torch.no_grad():
        model.pre_fc_norm_hidden.weight.zero_()
    got = model.pre_fc_norm_hidden(x)
    global_rms = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps)
    torch.testing.assert_close(got, global_rms, rtol=1e-5, atol=1e-5)
    grouped = x.reshape(1, 1, cfg.hc_count, cfg.hidden_size)
    grouped = grouped * torch.rsqrt(
        grouped.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps
    )
    assert not torch.allclose(got, grouped.reshape_as(got), rtol=1e-3, atol=1e-3)


# --------------------------------------------------------- hyper connection
def test_gated_residual_matches_literal():
    cfg = MTPConfig.tiny()
    torch.manual_seed(3)
    hc = GatedResidual(cfg, use_combine=True)
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for p in hc.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.1)
    x = torch.randn(4, cfg.hc_size, generator=g)
    block = torch.randn(4, cfg.hidden_size, generator=g)

    normed = hc.hc_norm(x)
    w = F.silu(F.linear(normed, hc.input_mix_weight_down.weight) / cfg.hc_count)
    w = torch.sigmoid(F.linear(w, hc.input_mix_weight_up.weight))
    mixed_ref = (
        w.view(4, cfg.hc_count, cfg.hidden_size)
        * normed.view(4, cfg.hc_count, cfg.hidden_size)
    ).mean(dim=1)
    inject = 2 * torch.sigmoid(
        F.linear(normed, hc.block_inject_weight.weight) / cfg.hc_count
    )
    combined_ref = (
        x.view(4, cfg.hc_count, cfg.hidden_size)
        + block.unsqueeze(1) * inject.unsqueeze(-1)
    ).reshape(4, cfg.hc_size)

    mixed, residuals = hc.mix(x)
    torch.testing.assert_close(mixed, mixed_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(hc.combine(block, residuals), combined_ref, rtol=1e-5, atol=1e-5)
    # combine adds into the RAW residual, not the normalised one.
    assert not torch.allclose(residuals[0], residuals[1])


def test_hc_norm_is_per_stream():
    cfg = MTPConfig.tiny()
    hc = GatedResidual(cfg, use_combine=True)
    g = torch.Generator().manual_seed(5)
    x = torch.randn(2, cfg.hc_size, generator=g)
    x[:, cfg.hidden_size :] *= 100.0  # scale streams 1..3 only
    normed = hc.hc_norm(x)
    per_stream = normed.view(2, cfg.hc_count, cfg.hidden_size)
    rms = per_stream.pow(2).mean(-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), rtol=2e-3, atol=2e-3)


# ------------------------------------------------------------------- MoE
def test_router_top_k_and_renormalisation():
    cfg, model, g = _build()
    moe = model.layers[0].mlp
    x = torch.randn(6, cfg.hidden_size, generator=g)
    ids, w = moe.route(x)
    assert ids.shape == (6, cfg.num_experts_per_tok)
    torch.testing.assert_close(w.sum(-1), torch.ones(6), rtol=1e-5, atol=1e-5)
    # softmax-all + renormalise == softmax over the selected logits
    logits = moe.gate(x)
    sel = torch.gather(logits, 1, ids)
    torch.testing.assert_close(w, torch.softmax(sel.float(), -1), rtol=1e-4, atol=1e-4)
    assert (ids == torch.topk(logits, cfg.num_experts_per_tok, -1).indices).all()


def test_shared_expert_gate_is_sigmoid_scalar():
    cfg, model, g = _build()
    moe = model.layers[0].mlp
    x = torch.randn(4, cfg.hidden_size, generator=g)
    scalar = moe.shared_expert_gate(x)
    assert scalar.shape == (4, 1)
    got = moe.shared_expert(x, scalar)
    act = F.silu(moe.shared_expert.gate_proj(x)) * moe.shared_expert.up_proj(x)
    ref = moe.shared_expert.down_proj(act) * torch.sigmoid(scalar)
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)


def test_expert_dispatch_matches_dense_reference():
    cfg, model, g = _build()
    experts = model.layers[0].mlp.experts
    x = torch.randn(5, cfg.hidden_size, generator=g)
    ids = torch.stack(
        [torch.randperm(cfg.num_experts, generator=g)[: cfg.num_experts_per_tok]
         for _ in range(5)]
    )
    w = torch.softmax(torch.randn(5, cfg.num_experts_per_tok, generator=g), -1)
    got = experts(x, ids, w)
    ref = torch.zeros_like(x)
    for i in range(5):
        for j in range(cfg.num_experts_per_tok):
            e = int(ids[i, j])
            gu = F.linear(x[i], experts.gate_up_proj[e])
            a, b = gu.chunk(2, -1)
            ref[i] += w[i, j] * F.linear(F.silu(a) * b, experts.down_proj[e])
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)


# ------------------------------------------------------------- attention
def test_q_gate_split_is_per_head():
    """q_proj rows are [q_h | gate_h] *within each head*, not two blocks."""
    cfg, model, g = _build()
    attn = model.layers[0].self_attn
    h, d = cfg.num_attention_heads, cfg.head_dim
    with torch.no_grad():
        wq = torch.zeros(2 * h * d, cfg.hidden_size)
        for head in range(h):
            wq[head * 2 * d : head * 2 * d + d, 0] = head + 1  # q rows
            wq[head * 2 * d + d : (head + 1) * 2 * d, 0] = -(head + 1)  # gate rows
        attn.q_proj.weight.copy_(wq)
        attn.q_norm.weight.zero_()
        attn.k_norm.weight.zero_()
    x = torch.zeros(1, 1, cfg.hidden_size)
    x[0, 0, 0] = 1.0
    qg = attn.q_proj(x).view(1, 1, h, 2 * d)
    q, gate = torch.chunk(qg, 2, dim=-1)
    for head in range(h):
        assert torch.allclose(q[0, 0, head], torch.full((d,), float(head + 1)))
        assert torch.allclose(gate[0, 0, head], torch.full((d,), -float(head + 1)))


def test_rope_is_partial_and_neox():
    cfg, model, _ = _build()
    rot = model.layers[0].self_attn.rotary
    assert rot.rotary_dim == cfg.head_dim // 4 == 4
    q = torch.zeros(1, 2, 1, cfg.head_dim)
    q[..., :] = 1.0
    k = q.clone()
    pos = torch.tensor([[0, 1]])
    q2, _ = rot(pos, q, k)
    # position 0 is the identity, and the non-rotary tail is untouched.
    torch.testing.assert_close(q2[0, 0], q[0, 0])
    torch.testing.assert_close(q2[0, 1, :, rot.rotary_dim :], q[0, 1, :, rot.rotary_dim :])
    assert not torch.allclose(q2[0, 1, :, : rot.rotary_dim], q[0, 1, :, : rot.rotary_dim])


def test_gqa_repeat():
    cfg, model, g = _build()
    attn = model.layers[0].self_attn
    assert cfg.num_attention_heads % cfg.num_key_value_heads == 0
    x = torch.randn(1, 6, cfg.hidden_size, generator=g)
    pos = torch.arange(6).view(1, 6)
    q, k, v, gate = attn.project(x, pos)
    assert q.shape == (1, 6, cfg.num_attention_heads, cfg.head_dim)
    assert k.shape == (1, 6, cfg.num_key_value_heads, cfg.head_dim)
    assert gate.shape == (1, 6, cfg.num_attention_heads * cfg.head_dim)


# --------------------------------------------------------------- causality
def test_causal_masking():
    """Perturbing row j must not move any output row i < j."""
    cfg, model, g = _build()
    ids, hc, pos = _inputs(cfg, g, b=1, t=10)
    base, _ = model(ids, hc, pos)
    j = 6
    ids2 = ids.clone()
    ids2[0, j] = (ids2[0, j] + 5) % cfg.vocab_size
    hc2 = hc.clone()
    hc2[0, j] += 3.0
    perturbed, _ = model(ids2, hc2, pos)
    torch.testing.assert_close(base[:, :j], perturbed[:, :j], rtol=2e-4, atol=2e-4)
    assert not torch.allclose(base[:, j], perturbed[:, j], rtol=1e-3, atol=1e-3)


def test_padding_mask_does_not_leak():
    cfg, model, g = _build()
    ids, hc, pos = _inputs(cfg, g, b=1, t=10)
    valid = torch.ones(1, 10, dtype=torch.bool)
    valid[0, 7:] = False
    mask = causal_mask_from_valid(valid)
    out_masked, _ = model(ids, hc, pos, attn_mask=mask)
    ids2, hc2 = ids.clone(), hc.clone()
    ids2[0, 7:] = (ids2[0, 7:] + 11) % cfg.vocab_size
    hc2[0, 7:] += 5.0
    out2, _ = model(ids2, hc2, pos, attn_mask=mask)
    torch.testing.assert_close(out_masked[:, :7], out2[:, :7], rtol=2e-4, atol=2e-4)
    # and the unpadded prefix equals a plain causal run of length 7
    short, _ = model(ids[:, :7], hc[:, :7], pos[:, :7])
    torch.testing.assert_close(out_masked[:, :7], short, rtol=2e-4, atol=2e-4)


def test_cache_path_matches_dense_path():
    cfg, model, g = _build()
    ids, hc, pos = _inputs(cfg, g, b=2, t=9)
    dense, own_dense = model(ids, hc, pos)
    cached, own_cached, (k, v) = model.forward_with_cache(ids, hc, pos)
    torch.testing.assert_close(dense, cached, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(own_dense, own_cached, rtol=2e-4, atol=2e-4)
    assert k.shape == (2, 9, cfg.num_key_value_heads, cfg.head_dim)
    # incremental extension reproduces the full pass
    _, _, kv0 = model.forward_with_cache(ids[:, :5], hc[:, :5], pos[:, :5])
    step, _, _ = model.forward_with_cache(
        ids[:, 5:6], hc[:, 5:6], pos[:, 5:6], past_kv=kv0
    )
    torch.testing.assert_close(step[:, 0], dense[:, 5], rtol=2e-4, atol=2e-4)


def test_chain_step0_matches_teacher_forcing():
    """Chain step 0 consumes the target state; it must equal the TF row."""
    cfg, model, g = _build()
    ids, hc, pos = _inputs(cfg, g, b=2, t=9)
    dense, _ = model(ids, hc, pos)
    t0 = 4
    past_mask = torch.arange(9).view(1, -1) < t0
    _, _, (k, v) = model.forward_with_cache(ids, hc, pos)
    preds = model.chain(
        hc[:, t0], ids[:, t0], pos[:, t0], (k, v), steps=3,
        past_mask=past_mask.expand(2, 9),
    )
    assert preds.shape == (2, 3)
    assert (preds[:, 0] == dense[:, t0].argmax(-1)).all()


# ----------------------------------------------------------------- freezing
def test_router_and_indexer_are_frozen_and_gradients_flow():
    cfg, model, g = _build()
    ids, hc, pos = _inputs(cfg, g, b=2, t=8)
    labels = torch.randint(0, cfg.vocab_size, (2, 8), generator=g)
    logits, _ = model(ids, hc, pos)
    loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), labels.reshape(-1))
    loss.backward()

    frozen = {
        "layers.0.mlp.gate.weight",
        "layers.0.self_attn.indexer.index_qk_proj.weight",
        "layers.0.self_attn.indexer.q_layernorm.weight",
        "layers.0.self_attn.indexer.k_layernorm.weight",
    }
    no_grad_expected = set()
    for name, p in model.named_parameters():
        if name in frozen:
            assert not p.requires_grad, name
            assert p.grad is None, name
        else:
            assert p.requires_grad, name
            if p.grad is None or p.grad.abs().sum() == 0:
                no_grad_expected.add(name)
    assert not no_grad_expected, sorted(no_grad_expected)

    total, trainable = model.num_parameters()
    assert total - trainable == sum(
        p.numel() for n, p in model.named_parameters() if n in frozen
    )
    # embed / lm_head never enter the parameter list
    names = {n for n, _ in model.named_parameters()}
    assert "embed_tokens" not in names and "lm_head" not in names


def test_frozen_weights_do_not_move_under_an_optimizer_step():
    cfg, model, g = _build()
    ids, hc, pos = _inputs(cfg, g, b=1, t=6)
    labels = torch.randint(0, cfg.vocab_size, (1, 6), generator=g)
    before = model.layers[0].mlp.gate.weight.clone()
    before_idx = model.layers[0].self_attn.indexer.index_qk_proj.weight.clone()
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-2)
    logits, _ = model(ids, hc, pos)
    F.cross_entropy(logits.reshape(-1, cfg.vocab_size), labels.reshape(-1)).backward()
    opt.step()
    assert torch.equal(before, model.layers[0].mlp.gate.weight)
    assert torch.equal(
        before_idx, model.layers[0].self_attn.indexer.index_qk_proj.weight
    )
    assert not torch.equal(before_idx * 0 + 1, model.fc_hidden.weight)


def test_loss_decreases_on_a_single_batch():
    """Overfit one batch: the objective must be learnable end to end."""
    cfg, model, g = _build()
    ids, hc, pos = _inputs(cfg, g, b=1, t=8)
    labels = torch.randint(0, cfg.vocab_size, (1, 8), generator=g)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=3e-2)
    losses = []
    for _ in range(30):
        opt.zero_grad(set_to_none=True)
        logits, _ = model(ids, hc, pos)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), labels.reshape(-1))
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5, losses[:: max(1, len(losses) // 6)]


# ------------------------------------------------------------- chunked loss
def test_chunked_ce_matches_naive_and_grads():
    from mtptrain.loss import chunked_argmax, chunked_ce, chunked_eval

    g = torch.Generator().manual_seed(21)
    n, h, v = 37, 16, 53
    hidden = (torch.randn(n, h, generator=g) * 0.4).requires_grad_(True)
    weight = torch.randn(v, h, generator=g) * 0.3
    labels = torch.randint(0, v, (n,), generator=g)
    labels[3] = -100
    labels[20] = -100

    naive_hidden = hidden.detach().clone().requires_grad_(True)
    naive = F.cross_entropy(
        F.linear(naive_hidden, weight), labels, ignore_index=-100
    )
    naive.backward()

    for chunk in (1, 5, 64):
        if hidden.grad is not None:
            hidden.grad = None
        loss = chunked_ce(hidden, weight, labels, chunk=chunk)
        torch.testing.assert_close(loss, naive.detach(), rtol=1e-5, atol=1e-6)
        loss.backward()
        torch.testing.assert_close(
            hidden.grad, naive_hidden.grad, rtol=1e-4, atol=1e-6
        )

    correct, count, loss_sum = chunked_eval(hidden.detach(), weight, labels, chunk=7)
    assert count == n - 2
    torch.testing.assert_close(
        torch.tensor(loss_sum / count), naive.detach(), rtol=1e-5, atol=1e-5
    )
    ref = F.linear(hidden.detach(), weight).argmax(-1)
    assert torch.equal(chunked_argmax(hidden.detach(), weight, chunk=9), ref)
    valid = labels != -100
    assert correct == int(((ref == labels) & valid).sum())


def test_forward_mixed_then_project_equals_forward():
    cfg, model, g = _build()
    ids, hc, pos = _inputs(cfg, g, b=2, t=7)
    logits, own_a = model(ids, hc, pos)
    mixed, own_b = model.forward_mixed(ids, hc, pos)
    torch.testing.assert_close(model.project_logits(mixed), logits)
    torch.testing.assert_close(own_a, own_b)
    assert mixed.shape == (2, 7, cfg.hidden_size)


def test_fp32_module_with_bf16_heads():
    """Training keeps FP32 masters while embed/lm_head stay BF16; evaluation
    runs without autocast, so every path must cast rather than assume a match."""
    cfg = MTPConfig.tiny()
    torch.manual_seed(5)
    model = MTPHead(cfg).to(torch.float32)
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    model.set_frozen_heads(
        (torch.randn(cfg.vocab_size, cfg.hidden_size, generator=g) * 0.05).to(torch.bfloat16),
        (torch.randn(cfg.vocab_size, cfg.hidden_size, generator=g) * 0.05).to(torch.bfloat16),
    )
    model.freeze_non_trainable()
    ids, hc, pos = _inputs(cfg, g, b=2, t=9, dtype=torch.float32)

    logits, own = model(ids, hc, pos)               # forward -> project_logits
    assert torch.isfinite(logits).all() and logits.shape[-1] == cfg.vocab_size
    mixed, _ = model.forward_mixed(ids, hc, pos)
    assert mixed.dtype == torch.float32
    out, _, kv = model.forward_with_cache(ids, hc, pos)   # cached path
    assert torch.isfinite(out).all()
    past_mask = torch.arange(9).view(1, -1).expand(2, 9) < 4
    preds = model.chain(hc[:, 4], ids[:, 4], pos[:, 4], kv, steps=3,
                        past_mask=past_mask)         # chain path
    assert preds.shape == (2, 3)

    from mtptrain.evaluate import evaluate
    samples = [
        Sample(next_token_ids=ids[0], hc_hidden=hc[0], positions=pos[0],
               labels=ids[0], input_ids=ids[0],
               greedy_consistent=torch.ones(9, dtype=torch.bool),
               source="t", doc_hash=f"d{i}")
        for i in range(2)
    ]
    res = evaluate(model, iter(samples), batch_size=2, chain_ks=(3,), starts_per_sample=1)
    assert res.summary()["rows"] > 0


# ------------------------------------------------------------------ data
def test_sample_alignment_and_collate():
    from mtptrain.data import load_dump_file  # noqa: F401  (import smoke)

    ids = torch.arange(10)
    tgt = torch.arange(10) + 100
    t = 10
    nxt = ids[1:t]
    labels = tgt[1:t]
    assert nxt.tolist() == list(range(1, 10))
    assert labels.tolist() == [100 + i for i in range(1, 10)]

    cfg = MTPConfig.tiny()
    samples = [
        Sample(
            next_token_ids=torch.arange(n) % cfg.vocab_size,
            hc_hidden=torch.zeros(n, cfg.hc_size),
            positions=torch.arange(n),
            labels=torch.arange(n) % cfg.vocab_size,
            input_ids=torch.arange(n) % cfg.vocab_size,
            greedy_consistent=torch.ones(n, dtype=torch.bool),
            source="t",
            doc_hash=str(n),
        )
        for n in (4, 7)
    ]
    batch = collate(samples)
    assert batch["labels"].shape == (2, 7)
    assert batch["labels"][0, 4:].eq(-100).all()
    assert batch["valid"][0].tolist() == [True] * 4 + [False] * 3


def test_evaluate_runs_on_tiny():
    from mtptrain.evaluate import evaluate

    cfg, model, g = _build()
    samples = []
    for i in range(4):
        n = 24
        samples.append(
            Sample(
                next_token_ids=torch.randint(0, cfg.vocab_size, (n,), generator=g),
                hc_hidden=torch.randn(n, cfg.hc_size, generator=g),
                positions=torch.arange(n),
                labels=torch.randint(0, cfg.vocab_size, (n,), generator=g),
                input_ids=torch.zeros(n, dtype=torch.int64),
                greedy_consistent=torch.ones(n, dtype=torch.bool),
                source="t",
                doc_hash=f"d{i}",
            )
        )
    res = evaluate(model, iter(samples), batch_size=2, chain_ks=(3, 15), starts_per_sample=2)
    summary = res.summary()
    assert summary["rows"] > 0
    assert 0.0 <= summary["agreement@1"] <= 1.0
    assert summary["accept@3"] <= 3.0 and summary["accept@15"] <= 15.0
    print("  eval summary:", json.dumps(summary))


def test_training_loop_smoke():
    from mtptrain.train import build_parser, train

    tmp = tempfile.mkdtemp(prefix="mtptrain-smoke-")
    out = os.path.join(tmp, "out")
    try:
        args = build_parser().parse_args(
            [
                "--tiny", "--synthetic", "4", "--steps", "6", "--batch-size", "2",
                "--max-len", "24", "--warmup", "2", "--log-every", "2",
                "--eval-every", "3", "--eval-batches", "1", "--eval-starts", "2",
                "--ckpt-every", "3", "--out", out, "--device", "cpu",
            ]
        )
        stats = train(args)
        assert stats["steps"] == 6
        assert os.path.exists(os.path.join(out, "latest.pt"))
    finally:
        shutil.rmtree(tmp)


def main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except BaseException as exc:  # noqa: BLE001  (includes pytest's Skipped)
            if type(exc).__name__ == "Skipped":
                print(f"SKIP {fn.__name__}: {exc}")
                continue
            failures += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
            import traceback

            traceback.print_exc()
    print(f"\n{len(fns) - failures}/{len(fns)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
