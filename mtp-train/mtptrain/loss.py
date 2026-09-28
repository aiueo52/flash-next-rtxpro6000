"""Chunked cross-entropy through the frozen lm_head.

The draft head's output vocabulary is 248 320 wide.  Materialising logits for a
4096-token batch costs 2.0 GiB in BF16 and 4.1 GiB in FP32 -- and the same again
for the gradient -- which dwarfs the 2.6 B-parameter model itself.  Because the
lm_head is *frozen*, the only gradient that has to leave the projection is the
one with respect to the 2560-wide mixed hidden state, so the logits can be
recomputed chunk-wise in the backward pass and never stored.

Peak logit memory becomes ``chunk x vocab`` instead of ``tokens x vocab``
(512 x 248320 fp32 = 486 MiB at the default chunk size).

``chunked_mixed_ce`` adds distribution distillation on top: besides the hard
label it takes the target's *sparse* soft target -- the K ids the dump stored
plus their probabilities -- and pays ``-sum_k q_k log p_draft(i_k)``.  It runs
in the same chunked, recompute-in-backward style, so the extra term costs no
extra logit memory: one chunk of logits serves both terms.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class _ChunkedCEFromHidden(torch.autograd.Function):
    """mean CE( hidden @ weight.T , labels ) with recomputed logits."""

    @staticmethod
    def forward(ctx, hidden, weight, labels, chunk, ignore_index):
        n = hidden.shape[0]
        keep = labels != ignore_index
        count = int(keep.sum())
        if count == 0:  # a mean over no row is not a loss of 0
            raise ValueError("chunked_ce: no labelled row")
        loss_sum = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            lab = labels[start:end]
            logits = F.linear(hidden[start:end].to(weight.dtype), weight).float()
            loss_sum += F.cross_entropy(
                logits, lab, reduction="sum", ignore_index=ignore_index
            )
            del logits
        ctx.save_for_backward(hidden, weight, labels)
        ctx.chunk = chunk
        ctx.ignore_index = ignore_index
        ctx.count = max(1, count)
        return loss_sum / max(1, count)

    @staticmethod
    def backward(ctx, grad_out):
        hidden, weight, labels = ctx.saved_tensors
        chunk, ignore_index, count = ctx.chunk, ctx.ignore_index, ctx.count
        scale = (grad_out / count).to(torch.float32)
        grad_hidden = torch.empty_like(hidden)
        n = hidden.shape[0]
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            lab = labels[start:end]
            h = hidden[start:end]
            logits = F.linear(h.to(weight.dtype), weight).float()
            probs = torch.softmax(logits, dim=-1)
            del logits
            valid = lab != ignore_index
            safe = torch.where(valid, lab, torch.zeros_like(lab))
            probs.scatter_add_(
                1, safe.unsqueeze(1), torch.full_like(safe, -1, dtype=probs.dtype).unsqueeze(1)
            )
            probs = probs * valid.unsqueeze(1).to(probs.dtype) * scale
            # Matmul in the weight's dtype: materialising an FP32 copy of the
            # [248320, 2560] lm_head per chunk would cost 2.5 GiB each time.
            grad_hidden[start:end] = (probs.to(weight.dtype) @ weight).to(hidden.dtype)
            del probs
        return grad_hidden, None, None, None, None


def chunked_ce(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    chunk: int = 512,
    ignore_index: int = -100,
) -> torch.Tensor:
    """hidden [N,H] (2560-wide mixed state), weight [V,H] frozen, labels [N]."""
    return _ChunkedCEFromHidden.apply(hidden, weight, labels, chunk, ignore_index)


@torch.no_grad()
def chunked_eval(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    chunk: int = 512,
    ignore_index: int = -100,
) -> Tuple[int, int, float]:
    """-> (correct, count, loss_sum) without materialising all logits."""
    n = hidden.shape[0]
    correct = 0
    count = 0
    loss_sum = 0.0
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        lab = labels[start:end]
        logits = F.linear(hidden[start:end].to(weight.dtype), weight).float()
        valid = lab != ignore_index
        correct += int(((logits.argmax(-1) == lab) & valid).sum())
        count += int(valid.sum())
        loss_sum += float(
            F.cross_entropy(logits, lab, reduction="sum", ignore_index=ignore_index)
        )
        del logits
    return correct, count, loss_sum


@torch.no_grad()
def chunked_argmax(
    hidden: torch.Tensor, weight: torch.Tensor, chunk: int = 512
) -> torch.Tensor:
    n = hidden.shape[0]
    out = torch.empty(n, dtype=torch.int64, device=hidden.device)
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        logits = F.linear(hidden[start:end].to(weight.dtype), weight)
        out[start:end] = logits.argmax(-1)
        del logits
    return out


# --------------------------------------------------------------- soft targets
def soft_targets(
    topk_logits: torch.Tensor,
    lse: Optional[torch.Tensor] = None,
    lse_known: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sparse target probabilities from a dumped top-K.  -> q [N, K] fp32.

    ``q_k = softmax(topk_logits / tau)_k * mass`` with

        mass = exp(logsumexp(topk_logits) - lse)

    i.e. the *true* probability mass the target put on those K ids, so the rest
    of the vocabulary is simply not supervised (rather than being implicitly
    forced to zero, which a renormalised top-K would do).  ``tau`` reshapes the
    distribution inside the top-K only; the mass is always computed at tau = 1,
    because it is a property of the target, not of the distillation.

    ``lse`` is None (or ``lse_known`` False) for a dump written before the hook
    stored it; the K entries are then renormalised to sum to 1.

    Padding columns of ``topk_logits`` are -inf and get exactly zero
    probability.  Rows that are entirely padding, or masked out by ``valid``,
    come back all-zero -- never NaN.
    """
    finite = torch.isfinite(topk_logits)
    rows = finite.any(dim=-1)
    if valid is not None:
        rows = rows & valid
    # Replace fully-masked rows before the softmax: softmax over all -inf is NaN.
    logits = torch.where(rows.unsqueeze(-1), topk_logits, torch.zeros_like(topk_logits))
    q = torch.softmax(logits / temperature, dim=-1)
    if lse is not None:
        # <= 0 in exact arithmetic; the dumped fp16 lse can round the other way.
        mass = torch.exp((torch.logsumexp(logits, dim=-1) - lse).clamp(max=0.0))
        if lse_known is not None:
            mass = torch.where(lse_known, mass, torch.ones_like(mass))
        q = q * mass.unsqueeze(-1)
    return q * rows.unsqueeze(-1).to(q.dtype)


def soft_targets_from_batch(
    batch: dict, temperature: float = 1.0, valid: Optional[torch.Tensor] = None
):
    """Collated batch -> (topk_ids [N,K], q [N,K], soft_valid [N]), all flat.

    Returns ``None`` when the batch carries no soft targets at all (an old dump
    without ``SGLANG_MTP_DUMP_TOPK``), which is the caller's signal to fall back
    to the plain hard CE.
    """
    if "topk_ids" not in batch:
        return None
    ids = batch["topk_ids"]
    k = ids.shape[-1]
    ok = batch["soft_valid"]
    if valid is not None:
        ok = ok & valid
    lse = batch.get("lse")
    lse_known = batch.get("lse_known")
    q = soft_targets(
        batch["topk_logits"].reshape(-1, k),
        None if lse is None else lse.reshape(-1),
        None if lse_known is None else lse_known.reshape(-1),
        temperature=temperature,
        valid=ok.reshape(-1),
    )
    return ids.reshape(-1, k), q, ok.reshape(-1)


class _ChunkedMixedCEFromHidden(torch.autograd.Function):
    """(1-alpha) * mean hard CE  +  alpha * mean soft CE, logits recomputed.

    Per row the soft term is ``-sum_k q_k log p(i_k)`` with ``p`` the full-vocab
    softmax of ``hidden @ weight.T``.  Writing ``Q = sum_k q_k`` (the stored
    top-K mass, 1 when the dump had no ``lse``) and ``q_sparse`` the scatter of
    ``q`` into the vocabulary,

        d(soft)/d(logits) = Q * p - q_sparse

    which reduces to the familiar ``p - q_sparse`` when the top-K carries all
    the mass.  Both terms share one chunk of logits, so distillation costs no
    additional logit memory over ``chunked_ce``.
    """

    @staticmethod
    def forward(ctx, hidden, weight, labels, topk_ids, q, soft_valid, alpha, chunk,
                ignore_index):
        n = hidden.shape[0]
        n_hard = max(1, int((labels != ignore_index).sum()))
        n_soft = max(1, int(soft_valid.sum()))
        hard_sum = hidden.new_zeros((), dtype=torch.float32)
        soft_sum = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            logits = F.linear(hidden[start:end].to(weight.dtype), weight).float()
            if alpha < 1.0:
                hard_sum += F.cross_entropy(
                    logits, labels[start:end], reduction="sum",
                    ignore_index=ignore_index,
                )
            if alpha > 0.0:
                logp = torch.log_softmax(logits, dim=-1)
                gathered = logp.gather(1, topk_ids[start:end])
                soft_sum -= (q[start:end] * gathered).sum()
                del logp, gathered
            del logits
        ctx.save_for_backward(hidden, weight, labels, topk_ids, q, soft_valid)
        ctx.chunk = chunk
        ctx.ignore_index = ignore_index
        ctx.alpha = alpha
        ctx.n_hard = n_hard
        ctx.n_soft = n_soft
        return (1.0 - alpha) * hard_sum / n_hard + alpha * soft_sum / n_soft

    @staticmethod
    def backward(ctx, grad_out):
        hidden, weight, labels, topk_ids, q, soft_valid = ctx.saved_tensors
        chunk, ignore_index, alpha = ctx.chunk, ctx.ignore_index, ctx.alpha
        g = grad_out.to(torch.float32)
        sh = (1.0 - alpha) * g / ctx.n_hard
        ss = alpha * g / ctx.n_soft
        grad_hidden = torch.empty_like(hidden)
        n = hidden.shape[0]
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            lab = labels[start:end]
            logits = F.linear(hidden[start:end].to(weight.dtype), weight).float()
            probs = torch.softmax(logits, dim=-1)
            del logits
            hard_ok = (lab != ignore_index).to(probs.dtype)
            qc = q[start:end]
            # Coefficient multiplying p: sh per hard row, ss * Q per soft row.
            coef = sh * hard_ok + ss * qc.sum(dim=-1)
            probs *= coef.unsqueeze(1)
            if alpha < 1.0:
                safe = torch.where(lab != ignore_index, lab, torch.zeros_like(lab))
                probs.scatter_add_(
                    1, safe.unsqueeze(1), (-sh * hard_ok).unsqueeze(1)
                )
            if alpha > 0.0:
                probs.scatter_add_(1, topk_ids[start:end], -ss * qc)
            # Matmul in the weight's dtype: an FP32 copy of the [V, H] lm_head
            # per chunk would cost 2.5 GiB each time.
            grad_hidden[start:end] = (probs.to(weight.dtype) @ weight).to(hidden.dtype)
            del probs
        return grad_hidden, None, None, None, None, None, None, None, None


def chunked_mixed_ce(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    topk_ids: torch.Tensor,
    q: torch.Tensor,
    soft_valid: torch.Tensor,
    alpha: float = 1.0,
    chunk: int = 512,
    ignore_index: int = -100,
) -> torch.Tensor:
    """``(1-alpha) * chunked_ce + alpha * soft CE`` in a single chunk loop.

    hidden [N,H], weight [V,H] frozen, labels [N], topk_ids [N,K],
    q [N,K] (from ``soft_targets``), soft_valid [N] bool -- the rows the soft
    mean averages over.  ``alpha == 0`` reproduces ``chunked_ce`` exactly and
    ``alpha == 1`` is the pure distillation loss.
    """
    # A term with no row to average over is unmeasured, not a loss of 0: with
    # no soft row the batch is the hard objective alone (and vice versa),
    # instead of the other term scaled down by (1 - alpha) or alpha.
    n_hard = int((labels != ignore_index).sum())
    n_soft = int(soft_valid.sum())
    if alpha > 0.0 and n_soft == 0:
        if n_hard == 0:
            raise ValueError("chunked_mixed_ce: no labelled row (hard or soft)")
        alpha = 0.0
    elif alpha < 1.0 and n_hard == 0:
        if n_soft == 0 or alpha == 0.0:
            raise ValueError("chunked_mixed_ce: no labelled row for the requested loss")
        alpha = 1.0
    if alpha > 0.0:
        # The denominator counts ``soft_valid`` rows, so any mass left on a row
        # outside it would be averaged over the wrong count.  Cheap insurance:
        # O(N*K) against a caller passing a q that disagrees with the mask.
        q = q * soft_valid.unsqueeze(-1).to(q.dtype)
    return _ChunkedMixedCEFromHidden.apply(
        hidden, weight, labels, topk_ids, q, soft_valid, alpha, chunk, ignore_index
    )


def chunked_soft_ce(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    topk_ids: torch.Tensor,
    q: torch.Tensor,
    soft_valid: torch.Tensor,
    chunk: int = 512,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Pure soft cross-entropy (``chunked_mixed_ce`` with alpha = 1)."""
    labels = torch.full(
        (hidden.shape[0],), ignore_index, dtype=torch.int64, device=hidden.device
    )
    return chunked_mixed_ce(
        hidden, weight, labels, topk_ids, q, soft_valid, alpha=1.0, chunk=chunk,
        ignore_index=ignore_index,
    )


__all__ = [
    "chunked_ce",
    "chunked_eval",
    "chunked_argmax",
    "soft_targets",
    "soft_targets_from_batch",
    "chunked_mixed_ce",
    "chunked_soft_ce",
]
