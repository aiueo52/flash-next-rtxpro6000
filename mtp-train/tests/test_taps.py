"""The EAGLE-3 fusion entry (option 1 of docs/lab-notes/DRAFT_V2_SPEC.md).

The probe's whole design rests on one property: with the two new projections
zero-initialised, the fused head must be the shipped head *bit for bit*, so a
measured acceptance change can only come from what the taps learned.  These
tests pin that, plus the two things a wiring mistake would silently break --
that the taps reach only the chain's first forward, and that a non-zero
projection actually changes the output.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mtptrain.config import MTPConfig  # noqa: E402
from mtptrain.data import Sample, collate  # noqa: E402
from mtptrain.model import MTPHead  # noqa: E402


def _heads(taps=(3, 23), seed=0):
    """(plain head, fused head) sharing every shipped weight exactly."""
    base_cfg = MTPConfig.tiny()
    g = torch.Generator().manual_seed(seed)
    plain = MTPHead(base_cfg).to(torch.float32)
    with torch.no_grad():
        for p in plain.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    embed = torch.randn(base_cfg.vocab_size, base_cfg.hidden_size, generator=g) * 0.05
    head = torch.randn(base_cfg.vocab_size, base_cfg.hidden_size, generator=g) * 0.05
    plain.set_frozen_heads(embed, head)

    fused = MTPHead(base_cfg.with_(tap_layers=taps)).to(torch.float32)
    missing, unexpected = fused.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected, unexpected
    assert all(m.startswith(("pre_fc_norm_tap.", "fc_hidden_tap.")) for m in missing), missing
    fused.set_frozen_heads(embed, head)
    return plain, fused


def _inputs(cfg, n_taps, b=2, t=9, seed=1):
    g = torch.Generator().manual_seed(seed)
    return dict(
        next_token_ids=torch.randint(0, cfg.vocab_size, (b, t), generator=g),
        hc_hidden=torch.randn(b, t, cfg.hc_size, generator=g),
        positions=torch.arange(t).expand(b, t).contiguous(),
        tap_hidden=torch.randn(b, t, n_taps * cfg.hc_size, generator=g),
    )


def test_zero_init_is_bit_identical():
    plain, fused = _heads()
    cfg = plain.config
    x = _inputs(cfg, 2)
    a, ha = plain.forward(x["next_token_ids"], x["hc_hidden"], x["positions"])
    b, hb = fused.forward(
        x["next_token_ids"], x["hc_hidden"], x["positions"],
        tap_hidden=x["tap_hidden"],
    )
    assert torch.equal(a, b), (a - b).abs().max()
    assert torch.equal(ha, hb)


def test_zero_init_chain_is_bit_identical():
    plain, fused = _heads()
    cfg = plain.config
    x = _inputs(cfg, 2)
    _, _, kv = plain.forward_with_cache(
        x["next_token_ids"], x["hc_hidden"], x["positions"], return_logits=False
    )
    _, _, kv2 = fused.forward_with_cache(
        x["next_token_ids"], x["hc_hidden"], x["positions"],
        return_logits=False, tap_hidden=x["tap_hidden"],
    )
    assert torch.equal(kv[0], kv2[0]) and torch.equal(kv[1], kv2[1])
    args = (x["hc_hidden"][:, 0], x["next_token_ids"][:, 0], x["positions"][:, 0])
    p1 = plain.chain(*args, kv, steps=4)
    p2 = fused.chain(*args, kv2, steps=4, start_taps=x["tap_hidden"][:, 0])
    assert torch.equal(p1, p2)


def test_taps_reach_only_the_first_chain_forward():
    """A non-zero tap projection must move the output, and ``chain`` must hand
    the taps to step 0 alone -- steps 2..S recurse on ``own_hc`` and have no
    target layer to read, which is the serving geometry."""
    plain, fused = _heads()
    cfg = plain.config
    x = _inputs(cfg, 2)
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        fused.fc_hidden_tap[0].weight.copy_(
            torch.randn(cfg.hidden_size, cfg.hidden_size, generator=g) * 0.05
        )
    a, _ = plain.forward(x["next_token_ids"], x["hc_hidden"], x["positions"])
    b, _ = fused.forward(
        x["next_token_ids"], x["hc_hidden"], x["positions"],
        tap_hidden=x["tap_hidden"],
    )
    assert not torch.allclose(a, b)

    seen = []
    real = fused.fuse_inputs

    def spy(next_token_ids, hc_hidden, tap_hidden=None):
        seen.append(tap_hidden is not None)
        return real(next_token_ids, hc_hidden, tap_hidden)

    fused.fuse_inputs = spy
    try:
        _, _, kv = fused.forward_with_cache(
            x["next_token_ids"], x["hc_hidden"], x["positions"],
            return_logits=False, tap_hidden=x["tap_hidden"],
        )
        seen.clear()
        fused.chain(
            x["hc_hidden"][:, 0], x["next_token_ids"][:, 0], x["positions"][:, 0],
            kv, steps=4, start_taps=x["tap_hidden"][:, 0],
        )
    finally:
        del fused.fuse_inputs
    assert seen == [True, False, False, False], seen


def test_freeze_all_but_taps():
    _, fused = _heads()
    fused.freeze_all_but_taps()
    total, trainable = fused.num_parameters()
    names = {n for n, p in fused.named_parameters() if p.requires_grad}
    assert names == set(fused.tap_parameter_names())
    cfg = fused.config
    expected = cfg.n_taps * (cfg.hidden_size**2 + cfg.hc_size)
    assert trainable == expected, (trainable, expected)
    assert total > trainable


def test_collate_and_truncate_carry_taps():
    from mtptrain.data import truncate

    cfg = MTPConfig.tiny().with_(tap_layers=(3, 23))
    n = 6
    s = Sample(
        next_token_ids=torch.arange(n),
        hc_hidden=torch.randn(n, cfg.hc_size).to(torch.bfloat16),
        positions=torch.arange(n),
        labels=torch.arange(n),
        input_ids=torch.arange(n),
        greedy_consistent=torch.ones(n, dtype=torch.bool),
        source="t",
        doc_hash="h",
        tap_hidden=torch.randn(n, 2 * cfg.hc_size).to(torch.bfloat16),
    )
    batch = collate([s, truncate(s, 4)])
    assert batch["tap_hidden"].shape == (2, 6, 2 * cfg.hc_size)
    assert torch.equal(batch["tap_hidden"][0], s.tap_hidden)
    assert torch.equal(batch["tap_hidden"][1, :4], s.tap_hidden[:4])
    assert batch["tap_hidden"][1, 4:].abs().sum() == 0

    plain = Sample(
        next_token_ids=torch.arange(n), hc_hidden=s.hc_hidden,
        positions=torch.arange(n), labels=torch.arange(n),
        input_ids=torch.arange(n),
        greedy_consistent=torch.ones(n, dtype=torch.bool), source="t", doc_hash="p",
    )
    assert "tap_hidden" not in collate([plain])
    try:
        collate([s, plain])
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("a mixed batch must be rejected")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
