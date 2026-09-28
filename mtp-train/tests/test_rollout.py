"""CPU tests for the chain-aware rollout objective (tiny config, no CUDA)."""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mtptrain.config import MTPConfig  # noqa: E402
from mtptrain.data import Sample, collate  # noqa: E402
from mtptrain.loss import chunked_ce  # noqa: E402
from mtptrain.model import MTPHead  # noqa: E402
from mtptrain.rollout import IGNORE, chain_losses, shift_left, step_weights  # noqa: E402


def _model(seed=11):
    cfg = MTPConfig.tiny()
    torch.manual_seed(seed)
    m = MTPHead(cfg).to(torch.float32)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    m.set_frozen_heads(
        torch.randn(cfg.vocab_size, cfg.hidden_size, generator=g) * 0.05,
        torch.randn(cfg.vocab_size, cfg.hidden_size, generator=g) * 0.05,
    )
    m.freeze_non_trainable()
    return cfg, m, g


def _batch(cfg, g, b=2, n=12, cons_all=True):
    samples = []
    for i in range(b):
        c = torch.ones(n, dtype=torch.bool) if cons_all else torch.zeros(n, dtype=torch.bool)
        samples.append(Sample(
            next_token_ids=torch.randint(0, cfg.vocab_size, (n,), generator=g),
            hc_hidden=torch.randn(n, cfg.hc_size, generator=g),
            positions=torch.arange(n),
            labels=torch.randint(0, cfg.vocab_size, (n,), generator=g),
            input_ids=torch.zeros(n, dtype=torch.int64),
            greedy_consistent=c, source="t", doc_hash=f"d{i}"))
    return collate(samples), samples


def test_shift_left():
    x = torch.tensor([[1, 2, 3, 4]])
    assert shift_left(x, 0, -100).tolist() == [[1, 2, 3, 4]]
    assert shift_left(x, 1, -100).tolist() == [[2, 3, 4, -100]]
    assert shift_left(x, 3, -100).tolist() == [[4, -100, -100, -100]]
    assert shift_left(x, 9, -100).tolist() == [[-100] * 4]
    b = torch.tensor([[True, True, False, True]])
    assert shift_left(b, 1, False).tolist() == [[True, False, True, False]]


def test_step_weights():
    assert step_weights(3, 0.5) == [1.0, 0.5, 0.5, 0.5]
    assert step_weights(0, 0.5) == [1.0]


def test_forward_train_matches_forward_mixed():
    """forward_train is the teacher-forced pass; it must equal forward_mixed."""
    cfg, m, g = _model()
    b, _ = _batch(cfg, g)
    mask = torch.ones(1, 1, 12, 12, dtype=torch.bool).tril()
    a_mixed, a_own = m.forward_mixed(b["next_token_ids"], b["hc_hidden"],
                                     b["positions"], attn_mask=mask)
    c_mixed, c_own, kv = m.forward_train(b["next_token_ids"], b["hc_hidden"],
                                         b["positions"], b["valid"])
    torch.testing.assert_close(a_mixed, c_mixed, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(a_own, c_own, rtol=2e-5, atol=2e-5)
    assert kv[0].shape == (2, 12, cfg.num_key_value_heads, cfg.head_dim)


def test_forward_rolled_shapes_and_context():
    """Row t must see teacher-forced rows <= t and itself, nothing later."""
    cfg, m, g = _model()
    b, _ = _batch(cfg, g)
    _, _, kv = m.forward_train(b["next_token_ids"], b["hc_hidden"],
                               b["positions"], b["valid"])
    kv = (kv[0].detach(), kv[1].detach())
    tok = torch.randint(0, cfg.vocab_size, (2, 12), generator=g)
    hc = torch.randn(2, 12, cfg.hc_size, generator=g)
    out, own = m.forward_rolled(tok, hc, b["positions"] + 1, kv, b["valid"])
    assert out.shape == (2, 12, cfg.hidden_size)
    assert own.shape == (2, 12, cfg.hc_size)

    t = 5
    for row, should_change in ((t - 1, True), (t, True), (t + 1, False)):
        k2 = kv[0].clone(); v2 = kv[1].clone()
        k2[:, row] += 7.0; v2[:, row] += 7.0
        out2, _ = m.forward_rolled(tok, hc, b["positions"] + 1, (k2, v2), b["valid"])
        moved = not torch.allclose(out[:, t], out2[:, t], rtol=1e-4, atol=1e-4)
        assert moved == should_change, (row, should_change)


def test_masks_consec_and_accepted():
    """Rolled step j is scored only where cons[t..t+j-1] holds (and, in
    'accepted' mode, where every earlier rolled prediction matched)."""
    cfg, m, g = _model()
    b, _ = _batch(cfg, g, b=1, n=10)
    cons = torch.ones(1, 10, dtype=torch.bool)
    cons[0, 4] = False
    b["greedy_consistent"] = cons
    seen = {}

    def on_term(j, w, loss, rec):
        seen[j] = rec["rows"]
        loss.backward()

    chain_losses(m, b, 2, step_weights(2, 0.5), chunk=8,
                 mask_mode="consistent", on_term=on_term)
    # step 1 needs cons[t]; row 4 drops out (plus the tail where t+1 is invalid)
    assert seen[1] == 8, seen           # rows 0..8 valid minus row 4
    # step 2 needs cons[t] & cons[t+1]; rows 3 and 4 both drop
    assert seen[2] == 6, seen


def test_accepted_mask_is_a_subset_of_consistent():
    """'accepted' additionally requires the chain to still be on an accepted
    prefix, so it can only ever score fewer rows.  On an untrained model the
    argmax essentially never matches the label, so it collapses to zero at
    step 1 -- correct, and the reason the objective needs a warm start."""
    cfg, m, g = _model()
    b, _ = _batch(cfg, g, b=2, n=14)
    counts = {}
    for mode in ("consistent", "accepted"):
        got = {}

        def on_term(j, w, loss, rec, _g=got):
            _g[j] = rec["rows"]
            loss.backward()

        m.zero_grad(set_to_none=True)
        stats = chain_losses(m, b, 3, step_weights(3, 0.5), chunk=8,
                             mask_mode=mode, on_term=on_term)
        counts[mode] = {r["step"]: r["rows"] for r in stats}
    for j in (1, 2, 3):
        assert counts["accepted"].get(j, 0) <= counts["consistent"].get(j, 0), (j, counts)
    assert counts["consistent"].get(1, 0) > 0
    assert counts["accepted"].get(1, 0) == 0   # random model: nothing accepted


def test_per_term_backward_needs_no_retain_graph():
    """The rolled inputs are detached, so each term's graph is independent and
    backward-as-produced (no retain_graph) must not raise."""
    cfg, m, g = _model()
    b, _ = _batch(cfg, g, b=2, n=12)
    calls = []

    def on_term(j, w, loss, rec):
        calls.append(j)
        (w * loss).backward()          # retain_graph=False on purpose

    chain_losses(m, b, 3, step_weights(3, 0.5), chunk=8,
                 mask_mode="consistent", on_term=on_term)
    assert calls == [0, 1, 2, 3], calls
    grads = {n: p.grad for n, p in m.named_parameters() if p.requires_grad}
    assert all(gr is not None and torch.isfinite(gr).all() for gr in grads.values())
    assert any(gr.abs().sum() > 0 for gr in grads.values())


def test_rolled_terms_add_gradient_beyond_the_teacher_forced_step():
    cfg, m, g = _model()
    b, _ = _batch(cfg, g, b=2, n=12)

    def run(k):
        m.zero_grad(set_to_none=True)
        chain_losses(m, b, k, step_weights(k, 0.5), chunk=8,
                     mask_mode="consistent",
                     on_term=lambda j, w, loss, rec: (w * loss).backward())
        return {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}

    g0, g3 = run(0), run(3)
    same = [n for n in g0 if torch.allclose(g0[n], g3[n], rtol=1e-6, atol=1e-8)]
    assert len(same) < len(g0) / 2, f"rolled steps changed almost nothing: {len(same)}/{len(g0)}"


def test_rollout_loss_decreases_on_an_overfit_batch():
    cfg, m, g = _model()
    b, _ = _batch(cfg, g, b=1, n=12)
    opt = torch.optim.AdamW(m.trainable_parameters(), lr=3e-2)
    totals = []
    for _ in range(40):
        opt.zero_grad(set_to_none=True)
        acc = []

        def on_term(j, w, loss, rec, _a=acc):
            _a.append(w * float(loss.detach()))
            (w * loss).backward()

        chain_losses(m, b, 2, step_weights(2, 0.5), chunk=8,
                     mask_mode="consistent", on_term=on_term)
        opt.step()
        totals.append(sum(acc))
    assert totals[-1] < totals[0] * 0.5, totals[:: max(1, len(totals) // 6)]


def test_teacher_forced_term_matches_plain_ce():
    cfg, m, g = _model()
    b, _ = _batch(cfg, g, b=2, n=12)
    mixed, _, _ = m.forward_train(b["next_token_ids"], b["hc_hidden"],
                                  b["positions"], b["valid"])
    ref = chunked_ce(mixed.reshape(-1, cfg.hidden_size), m.lm_head,
                     b["labels"].reshape(-1), chunk=8, ignore_index=IGNORE)
    got = chain_losses(m, b, 0, step_weights(0, 0.5), chunk=8,
                       on_term=lambda j, w, loss, rec: None)
    assert abs(got[0]["loss"] - float(ref.detach())) < 1e-5, (got[0]["loss"], float(ref.detach()))


def main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            import traceback
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{len(fns) - failures}/{len(fns)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
