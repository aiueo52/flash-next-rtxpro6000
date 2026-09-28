"""CPU tests for the chunked soft / mixed cross-entropy (mtptrain/loss.py).

Everything is checked against a naive dense implementation that materialises
the full [N, V] logits, in **value and in gradient**, at several chunk sizes --
the same contract test_tiny.py holds ``chunked_ce`` to.  Plus the two
reductions that pin the semantics: a one-hot soft target must equal the hard
CE, and alpha = 0 must equal ``chunked_ce`` exactly.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mtptrain.data import NEG_INF, Sample, collate  # noqa: E402
from mtptrain.loss import (  # noqa: E402
    chunked_ce,
    chunked_mixed_ce,
    chunked_soft_ce,
    soft_targets,
    soft_targets_from_batch,
)

IGNORE = -100


# --------------------------------------------------------------- reference
def naive_soft_ce(hidden, weight, topk_ids, q, soft_valid):
    """-inf-free dense reference: mean over soft rows of -sum_k q_k log p(i_k)."""
    logits = F.linear(hidden.to(weight.dtype), weight).float()
    logp = torch.log_softmax(logits, dim=-1)
    per_row = -(q * logp.gather(1, topk_ids)).sum(dim=-1)
    return per_row.sum() / max(1, int(soft_valid.sum()))


def naive_mixed_ce(hidden, weight, labels, topk_ids, q, soft_valid, alpha):
    logits = F.linear(hidden.to(weight.dtype), weight).float()
    hard = F.cross_entropy(logits, labels, reduction="sum", ignore_index=IGNORE)
    hard = hard / max(1, int((labels != IGNORE).sum()))
    return (1.0 - alpha) * hard + alpha * naive_soft_ce(
        hidden, weight, topk_ids, q, soft_valid
    )


def _case(n=17, h=6, v=41, k=4, seed=3, n_ignore=3):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(n, h, generator=g, dtype=torch.float32)
    weight = torch.randn(v, h, generator=g, dtype=torch.float32) * 0.3
    labels = torch.randint(0, v, (n,), generator=g)
    labels[:n_ignore] = IGNORE
    topk_ids = torch.stack(
        [torch.randperm(v, generator=g)[:k] for _ in range(n)]
    )
    topk_logits = torch.sort(
        torch.randn(n, k, generator=g) * 2.0, dim=-1, descending=True
    ).values
    lse = torch.logsumexp(topk_logits, dim=-1) + 0.25  # some mass outside the K
    return hidden, weight, labels, topk_ids, topk_logits, lse


# ----------------------------------------------------------- soft_targets
def test_soft_targets_mass_and_shape():
    _, _, _, _, topk_logits, lse = _case()
    q = soft_targets(topk_logits, lse)
    assert q.shape == topk_logits.shape
    want = torch.exp(torch.logsumexp(topk_logits, dim=-1) - lse)
    torch.testing.assert_close(q.sum(dim=-1), want, rtol=1e-5, atol=1e-6)
    # the shape inside the top-K is the target's own softmax
    torch.testing.assert_close(
        q / q.sum(dim=-1, keepdim=True),
        torch.softmax(topk_logits, dim=-1),
        rtol=1e-5, atol=1e-6,
    )


def test_soft_targets_without_lse_normalise_to_one():
    _, _, _, _, topk_logits, _ = _case()
    q = soft_targets(topk_logits, None)
    torch.testing.assert_close(q.sum(dim=-1), torch.ones(q.shape[0]), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(q, torch.softmax(topk_logits, dim=-1), rtol=1e-6, atol=1e-6)


def test_soft_targets_lse_known_mask_falls_back_to_one():
    _, _, _, _, topk_logits, lse = _case()
    known = torch.zeros(topk_logits.shape[0], dtype=torch.bool)
    known[::2] = True
    q = soft_targets(topk_logits, lse, known)
    mass = q.sum(dim=-1)
    torch.testing.assert_close(mass[~known], torch.ones(int((~known).sum())),
                               rtol=1e-6, atol=1e-6)
    assert (mass[known] < 1.0).all()


def test_soft_targets_temperature_sharpens_without_moving_the_mass():
    _, _, _, _, topk_logits, lse = _case()
    q1 = soft_targets(topk_logits, lse, temperature=1.0)
    qs = soft_targets(topk_logits, lse, temperature=0.5)
    # the stored mass is a property of the target, not of tau
    torch.testing.assert_close(q1.sum(-1), qs.sum(-1), rtol=1e-5, atol=1e-6)
    # ... but the top entry takes a larger share of it
    assert (qs[:, 0] >= q1[:, 0] - 1e-6).all()
    assert (qs[:, 0] > q1[:, 0]).any()


def test_soft_targets_padding_and_masked_rows_are_zero_not_nan():
    _, _, _, _, topk_logits, lse = _case()
    topk_logits = topk_logits.clone()
    topk_logits[2, 2:] = NEG_INF          # a short K
    topk_logits[5, :] = NEG_INF           # a fully padded row
    valid = torch.ones(topk_logits.shape[0], dtype=torch.bool)
    valid[7] = False
    q = soft_targets(topk_logits, lse, valid=valid)
    assert torch.isfinite(q).all()
    assert (q[5] == 0).all() and (q[7] == 0).all()
    assert (q[2, 2:] == 0).all() and q[2, 0] > 0


# -------------------------------------------------------------- soft loss
def test_soft_ce_value_and_grad_match_the_dense_reference():
    hidden, weight, _, ids, logits, lse = _case()
    q = soft_targets(logits, lse)
    ok = torch.ones(hidden.shape[0], dtype=torch.bool)
    ref_h = hidden.clone().requires_grad_(True)
    ref = naive_soft_ce(ref_h, weight, ids, q, ok)
    ref.backward()
    for chunk in (1, 3, 8, 64):
        h = hidden.clone().requires_grad_(True)
        got = chunked_soft_ce(h, weight, ids, q, ok, chunk=chunk)
        got.backward()
        torch.testing.assert_close(got, ref.detach(), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(h.grad, ref_h.grad, rtol=1e-5, atol=1e-6)


def test_soft_ce_with_a_one_hot_target_equals_hard_ce():
    """q = one-hot on the label (mass 1) must reproduce the hard CE exactly,
    value and gradient -- the sanity check that the sparse scatter and the
    ``Q * p - q_sparse`` gradient are the right ones."""
    hidden, weight, labels, _, _, _ = _case(n_ignore=0)
    n, k = hidden.shape[0], 3
    ids = torch.zeros(n, k, dtype=torch.int64)
    ids[:, 0] = labels
    q = torch.zeros(n, k)
    q[:, 0] = 1.0
    ok = torch.ones(n, dtype=torch.bool)

    h1 = hidden.clone().requires_grad_(True)
    soft = chunked_soft_ce(h1, weight, ids, q, ok, chunk=5)
    soft.backward()
    h2 = hidden.clone().requires_grad_(True)
    hard = chunked_ce(h2, weight, labels, chunk=5, ignore_index=IGNORE)
    hard.backward()
    torch.testing.assert_close(soft, hard, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(h1.grad, h2.grad, rtol=1e-5, atol=1e-6)


def test_soft_ce_ignores_masked_rows():
    """Masked-out rows contribute neither loss nor gradient, and the mean is
    taken over the scored rows only."""
    hidden, weight, _, ids, logits, lse = _case()
    n = hidden.shape[0]
    ok = torch.ones(n, dtype=torch.bool)
    ok[3:7] = False
    q = soft_targets(logits, lse, valid=ok)
    h = hidden.clone().requires_grad_(True)
    got = chunked_soft_ce(h, weight, ids, q, ok, chunk=4)
    got.backward()
    assert (h.grad[3:7] == 0).all()
    # the same subset scored alone gives the same number
    keep = ok.nonzero(as_tuple=True)[0]
    h2 = hidden[keep].clone().requires_grad_(True)
    ref = chunked_soft_ce(h2, weight, ids[keep], q[keep],
                          torch.ones(keep.numel(), dtype=torch.bool), chunk=4)
    torch.testing.assert_close(got, ref, rtol=1e-6, atol=1e-6)


# ------------------------------------------------------------- mixed loss
def test_mixed_ce_value_and_grad_match_the_dense_reference():
    hidden, weight, labels, ids, logits, lse = _case()
    q = soft_targets(logits, lse)
    ok = torch.ones(hidden.shape[0], dtype=torch.bool)
    for alpha in (0.0, 0.25, 0.5, 1.0):
        ref_h = hidden.clone().requires_grad_(True)
        ref = naive_mixed_ce(ref_h, weight, labels, ids, q, ok, alpha)
        ref.backward()
        for chunk in (1, 5, 64):
            h = hidden.clone().requires_grad_(True)
            got = chunked_mixed_ce(h, weight, labels, ids, q, ok, alpha=alpha,
                                   chunk=chunk, ignore_index=IGNORE)
            got.backward()
            torch.testing.assert_close(got, ref.detach(), rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(h.grad, ref_h.grad, rtol=1e-5, atol=1e-6)


def test_mixed_ce_alpha_zero_is_exactly_chunked_ce():
    hidden, weight, labels, ids, logits, lse = _case()
    q = soft_targets(logits, lse)
    ok = torch.ones(hidden.shape[0], dtype=torch.bool)
    h1 = hidden.clone().requires_grad_(True)
    a = chunked_mixed_ce(h1, weight, labels, ids, q, ok, alpha=0.0, chunk=6,
                         ignore_index=IGNORE)
    a.backward()
    h2 = hidden.clone().requires_grad_(True)
    b = chunked_ce(h2, weight, labels, chunk=6, ignore_index=IGNORE)
    b.backward()
    torch.testing.assert_close(a, b, rtol=1e-7, atol=1e-8)
    torch.testing.assert_close(h1.grad, h2.grad, rtol=1e-6, atol=1e-8)


def test_missing_lse_path_matches_the_reference():
    """A dump written before the hook stored `lse`: the K entries renormalise
    to 1 and everything downstream still agrees with the dense reference."""
    hidden, weight, labels, ids, logits, _ = _case()
    q = soft_targets(logits, None)
    ok = torch.ones(hidden.shape[0], dtype=torch.bool)
    ref_h = hidden.clone().requires_grad_(True)
    ref = naive_mixed_ce(ref_h, weight, labels, ids, q, ok, 0.5)
    ref.backward()
    h = hidden.clone().requires_grad_(True)
    got = chunked_mixed_ce(h, weight, labels, ids, q, ok, alpha=0.5, chunk=7,
                           ignore_index=IGNORE)
    got.backward()
    torch.testing.assert_close(got, ref.detach(), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(h.grad, ref_h.grad, rtol=1e-5, atol=1e-6)


def test_gradient_of_the_soft_term_is_q_p_minus_q_sparse():
    """Pin the analytic gradient the chunked backward implements, against
    autograd on the dense logits themselves."""
    hidden, weight, _, ids, logits_k, lse = _case(n=5, v=23, k=3)
    q = soft_targets(logits_k, lse)
    z = F.linear(hidden, weight).detach().requires_grad_(True)
    logp = torch.log_softmax(z, dim=-1)
    (-(q * logp.gather(1, ids)).sum() / z.shape[0]).backward()
    p = torch.softmax(z.detach(), dim=-1)
    want = p * q.sum(-1, keepdim=True)
    want.scatter_add_(1, ids, -q)
    torch.testing.assert_close(z.grad, want / z.shape[0], rtol=1e-5, atol=1e-7)


# ------------------------------------------------------- batch plumbing
def _sample(n, k, hc=8, vocab=41, seed=0, with_lse=True, with_soft=True):
    g = torch.Generator().manual_seed(seed)
    kw = {}
    if with_soft:
        kw = dict(
            topk_ids=torch.randint(0, vocab, (n, k), generator=g),
            topk_logits=torch.randn(n, k, generator=g),
        )
        if with_lse:
            kw["lse"] = torch.logsumexp(kw["topk_logits"], dim=-1) + 0.2
    return Sample(
        next_token_ids=torch.randint(0, vocab, (n,), generator=g),
        hc_hidden=torch.randn(n, hc, generator=g),
        positions=torch.arange(n),
        labels=torch.randint(0, vocab, (n,), generator=g),
        input_ids=torch.zeros(n, dtype=torch.int64),
        greedy_consistent=torch.ones(n, dtype=torch.bool),
        source="t", doc_hash=f"d{seed}", **kw,
    )


def test_soft_targets_from_batch_respects_padding_and_missing_soft():
    batch = collate([_sample(9, 4, seed=1), _sample(5, 4, seed=2),
                     _sample(7, 4, seed=3, with_soft=False)])
    ids, q, ok = soft_targets_from_batch(batch, valid=batch["valid"])
    b, t = batch["valid"].shape
    q, ok = q.view(b, t, -1), ok.view(b, t)
    assert ok[0].sum() == 9 and ok[1].sum() == 5 and ok[2].sum() == 0
    assert (q[1, 5:] == 0).all() and (q[2] == 0).all()
    assert torch.isfinite(q).all()
    assert ids.shape[-1] == 4


def test_soft_targets_from_batch_is_none_without_topk():
    batch = collate([_sample(6, 4, seed=4, with_soft=False)])
    assert soft_targets_from_batch(batch) is None


def test_ragged_k_pads_with_zero_probability():
    """A batch mixing K=4 and K=2 dumps: the short rows must put no mass on the
    padded columns."""
    batch = collate([_sample(6, 4, seed=5), _sample(6, 2, seed=6)])
    assert batch["topk_ids"].shape[-1] == 4
    ids, q, ok = soft_targets_from_batch(batch, valid=batch["valid"])
    q = q.view(2, 6, 4)
    assert (q[1, :, 2:] == 0).all()
    assert (q[1, :, :2].sum(-1) > 0).all()


# ---------------------------------------------------------------- rollout
def test_rollout_step0_uses_soft_targets_and_rolled_steps_do_not():
    """soft_steps="first": only the teacher-forced term changes when alpha
    turns on; with "all" the rolled terms move too."""
    from tests.test_rollout import _batch, _model
    from mtptrain.rollout import chain_losses, step_weights

    cfg, m, g = _model()
    batch, samples = _batch(cfg, g, b=2, n=12)
    k = 8
    soft = collate([
        Sample(
            next_token_ids=s.next_token_ids, hc_hidden=s.hc_hidden,
            positions=s.positions, labels=s.labels, input_ids=s.input_ids,
            greedy_consistent=s.greedy_consistent, source=s.source,
            doc_hash=s.doc_hash,
            topk_ids=torch.randint(0, cfg.vocab_size, (len(s), k), generator=g),
            topk_logits=torch.randn(len(s), k, generator=g),
            lse=torch.zeros(len(s)) + 5.0,
        )
        for s in samples
    ])

    def run(alpha, steps="first"):
        got = {}
        chain_losses(m, soft, 2, step_weights(2, 0.5), chunk=8,
                     mask_mode="consistent", soft_alpha=alpha, soft_steps=steps,
                     on_term=lambda j, w, loss, rec, _g=got: _g.__setitem__(
                         j, float(loss.detach())))
        return got

    base, mixed, everywhere = run(0.0), run(0.5), run(0.5, "all")
    assert abs(mixed[0] - base[0]) > 1e-4, (base[0], mixed[0])
    assert abs(mixed[1] - base[1]) < 1e-6, (base[1], mixed[1])
    assert abs(everywhere[1] - base[1]) > 1e-4, (base[1], everywhere[1])


def test_rollout_soft_is_a_no_op_without_topk_tensors():
    from tests.test_rollout import _batch, _model
    from mtptrain.rollout import chain_losses, step_weights

    cfg, m, g = _model()
    batch, _ = _batch(cfg, g, b=2, n=12)

    def run(alpha):
        got = {}
        chain_losses(m, batch, 1, step_weights(1, 0.5), chunk=8,
                     mask_mode="consistent", soft_alpha=alpha,
                     on_term=lambda j, w, loss, rec, _g=got: _g.__setitem__(
                         j, float(loss.detach())))
        return got

    assert run(0.0) == run(0.7)


# ------------------------------------------------------------ train loop
def test_training_loop_with_soft_alpha():
    """The CPU smoke test of the whole loop with distillation switched on."""
    import shutil

    from mtptrain.train import build_parser, train

    out = os.path.join(os.environ.get("TMPDIR", "/tmp"), "mtptrain-soft-smoke")
    shutil.rmtree(out, ignore_errors=True)
    args = build_parser().parse_args(
        ["--tiny", "--synthetic", "4", "--steps", "6", "--batch-size", "2",
         "--max-len", "24", "--warmup", "2", "--log-every", "3",
         "--eval-every", "3", "--eval-batches", "1", "--eval-starts", "2",
         "--soft-alpha", "0.5", "--soft-temp", "1.5", "--synthetic-topk", "6",
         "--out", out, "--device", "cpu"]
    )
    stats = train(args)
    assert stats["steps"] == 6
    assert stats["best"]["step"] > 0
    losses = [h["loss"] for h in stats["history"] if "loss" in h]
    assert losses and all(l == l for l in losses)  # no NaN

    # ... and with the rollout objective on top
    args.rollout_k = 2
    args.out = out + "-roll"
    shutil.rmtree(args.out, ignore_errors=True)
    stats = train(args)
    assert stats["steps"] == 6
    shutil.rmtree(out, ignore_errors=True)
    shutil.rmtree(args.out, ignore_errors=True)


def main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            import traceback

            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{len(fns) - failures}/{len(fns)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
