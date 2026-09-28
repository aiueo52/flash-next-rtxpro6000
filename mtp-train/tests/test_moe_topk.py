"""CPU tests for the draft-only MoE top-k knob (MTPHead.set_moe_topk).

Three properties matter for the sweep to mean anything:

* the default is a no-op -- ``set_moe_topk()`` reproduces the shipped routing
  bit for bit, so a "baseline" row of the sweep is the shipped head;
* a reduced ``top_k`` really dispatches fewer experts, and the renormalise
  switch changes the routed branch's scale as documented;
* ``top_k_recur`` reaches the recursive chain steps **only** -- step 0 (which
  consumes the target's hidden state) and the teacher-forced prefix pass keep
  the full top-k, which is what "reduce only the cheap steps" has to mean.
"""
from __future__ import annotations

import os
import sys

import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from tests.test_rollout import _batch, _model  # noqa: E402


def test_default_is_a_no_op():
    cfg, m, g = _model()
    batch, _ = _batch(cfg, g, b=2, n=6)
    with torch.no_grad():
        before, _ = m(batch["next_token_ids"], batch["hc_hidden"], batch["positions"])
        m.set_moe_topk()
        after, _ = m(batch["next_token_ids"], batch["hc_hidden"], batch["positions"])
    assert torch.equal(before, after)
    assert m.layers[0].mlp.top_k == cfg.num_experts_per_tok


def test_reduced_top_k_dispatches_fewer_experts_and_changes_the_output():
    cfg, m, g = _model()
    batch, _ = _batch(cfg, g, b=2, n=6)
    moe = m.layers[0].mlp
    x = torch.randn(5, cfg.hidden_size, generator=g)
    with torch.no_grad():
        m.set_moe_topk(top_k=cfg.num_experts_per_tok)
        ids_full, w_full = moe.route(x)
        m.set_moe_topk(top_k=2)
        ids_cut, w_cut = moe.route(x)
        base, _ = m.forward(batch["next_token_ids"], batch["hc_hidden"], batch["positions"])
    assert ids_full.shape[1] == cfg.num_experts_per_tok
    assert ids_cut.shape[1] == 2
    # the survivors are the same experts, in the same order
    assert torch.equal(ids_cut, ids_full[:, :2])
    # renormalised by default: the kept weights sum to one
    assert torch.allclose(w_cut.float().sum(-1), torch.ones(5), atol=1e-5)
    with torch.no_grad():
        m.set_moe_topk(top_k=2, renormalize=False)
        _, w_raw = moe.route(x)
    assert torch.all(w_raw.float().sum(-1) < 1.0)
    # without renormalisation the weights are the raw full-softmax probabilities
    probs = torch.softmax(moe.gate(x).float(), dim=-1)
    assert torch.allclose(w_raw.float(), probs.gather(1, ids_cut), atol=1e-6)
    # ... and renormalising them reproduces the renormalised branch
    assert torch.allclose(w_cut.float(),
                          w_raw.float() / w_raw.float().sum(-1, keepdim=True),
                          atol=1e-5)
    with torch.no_grad():
        m.set_moe_topk()
        restored, _ = m.forward(batch["next_token_ids"], batch["hc_hidden"],
                                batch["positions"])
    assert not torch.equal(base, restored)  # base was taken at top_k=2
    assert m.layers[0].mlp.active_top_k == cfg.num_experts_per_tok


def test_top_k_recur_reaches_only_the_recursive_chain_steps():
    cfg, m, g = _model()
    batch, _ = _batch(cfg, g, b=2, n=6)
    moe = m.layers[0].mlp
    seen = []
    real_route = moe.route

    def spy(x):
        seen.append(moe.active_top_k)
        return real_route(x)

    moe.route = spy
    m.set_moe_topk(top_k=None, top_k_recur=2)
    with torch.no_grad():
        _, _, kv = m.forward_with_cache(
            batch["next_token_ids"], batch["hc_hidden"], batch["positions"],
            return_logits=False,
        )
        prefix_calls = len(seen)
        m.chain(batch["hc_hidden"][:, 0], batch["next_token_ids"][:, 0],
                batch["positions"][:, 0], kv, steps=4)
    moe.route = real_route
    # the prefix pass (serving: draft_extend) keeps the full top-k
    assert seen[:prefix_calls] == [cfg.num_experts_per_tok] * prefix_calls
    chain_calls = seen[prefix_calls:]
    assert len(chain_calls) == 4
    assert chain_calls == [cfg.num_experts_per_tok, 2, 2, 2]
    assert moe.recur_mode is False  # reset on the way out


def test_chain_with_a_flat_reduced_top_k_uses_it_everywhere():
    cfg, m, g = _model()
    batch, _ = _batch(cfg, g, b=2, n=6)
    moe = m.layers[0].mlp
    seen = []
    real_route = moe.route
    moe.route = lambda x: (seen.append(moe.active_top_k), real_route(x))[1]
    m.set_moe_topk(top_k=3)
    with torch.no_grad():
        _, _, kv = m.forward_with_cache(
            batch["next_token_ids"], batch["hc_hidden"], batch["positions"],
            return_logits=False,
        )
        m.chain(batch["hc_hidden"][:, 0], batch["next_token_ids"][:, 0],
                batch["positions"][:, 0], kv, steps=3)
    moe.route = real_route
    assert set(seen) == {3}
