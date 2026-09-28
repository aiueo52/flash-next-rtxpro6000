"""Chain-aware training objective.

A head trained only on the teacher-forced step can improve single-step
agreement and still get *worse* on long chains.  That is the expected failure
mode of an objective that only ever trains one teacher-forced step: nothing in
it tells the head what happens when its own output is fed back.

This module adds that.  After the teacher-forced step it rolls the head forward
``k`` more steps exactly as the server does -- next input token = argmax of the
previous step, HC state = the head's own ``own_hc``, position += 1 -- and scores
each rolled step against the target's argmax at that position.

Design choices, and why:

* **Rolled inputs are detached.**  Token ids are discrete (argmax), so there is
  no gradient path through them anyway.  ``own_hc`` is detached too in this
  first version: keeping it attached would make step j's graph contain every
  earlier step, so activations and backward cost would grow with k instead of
  staying flat.  Each term is therefore an independent per-step objective
  ("given this self-generated state and token, predict the target's next
  token"), which is what the accept@k metric actually measures.
* **Per-term backward.**  Because the terms are independent, running backward
  as each loss is produced is exactly equivalent to summing first, and keeps
  only one step's graph alive.  That is what ``on_term`` is for.
* **Masking.**  A rolled step is scorable only where the dumped
  ``target_argmax`` is the right label for the context the chain actually built:
  ``greedy_consistent[t .. t+j-1]`` (see the derivation in evaluate.py).  With
  ``mask_mode="accepted"`` the step must additionally still be on an accepted
  prefix -- every earlier rolled prediction matched its label -- which mirrors
  serving, where a chain stops paying off the moment one token is rejected.
* **One approximation.**  ``forward_rolled`` attends to the teacher-forced KV
  plus the query's own key, but not to the chain's own intermediate rows.  At
  k=3 that is at most two missing rows out of a prefix up to 2048 long.
* **Soft targets only on step 0** (``soft_alpha > 0``).  The dumped top-K at
  row t+j is the target's distribution *conditioned on the corpus token* at
  that row, which is the right conditioning for the teacher-forced step and
  (on self-generated greedy data) still the right one for the rolled steps --
  but the rolled steps already carry the ``greedy_consistent`` /
  ``accepted`` masking machinery, and mixing a second approximation into them
  buys little.  They keep the hard label; ``soft_steps`` makes that a flag.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import torch

from .loss import chunked_argmax, chunked_ce, chunked_mixed_ce, soft_targets_from_batch

IGNORE = -100


def shift_left(x: torch.Tensor, j: int, fill) -> torch.Tensor:
    """x[b, t] -> x[b, t+j], right-padded with `fill`."""
    if j == 0:
        return x
    out = torch.full_like(x, fill)
    if j < x.shape[1]:
        out[:, : x.shape[1] - j] = x[:, j:]
    return out


def step_weights(k: int, rolled: float, first: float = 1.0) -> List[float]:
    return [first] + [rolled] * k


@torch.no_grad()
def _argmax(model, mixed: torch.Tensor, chunk: int) -> torch.Tensor:
    b, t, h = mixed.shape
    return chunked_argmax(mixed.reshape(-1, h), model.lm_head, chunk).view(b, t)


def chain_losses(
    model,
    batch: Dict[str, torch.Tensor],
    k: int,
    weights: Sequence[float],
    chunk: int = 512,
    mask_mode: str = "accepted",
    on_term: Optional[Callable[[int, float, torch.Tensor, dict], None]] = None,
    soft_alpha: float = 0.0,
    soft_temp: float = 1.0,
    soft_steps: str = "first",
) -> List[dict]:
    """Teacher-forced step plus `k` rolled steps.

    ``on_term(j, weight, loss, stats)`` receives each loss still attached and
    must run backward before returning.

    ``soft_alpha > 0`` distils the target's dumped distribution into the step-0
    loss: ``(1 - alpha) * hard CE + alpha * soft CE`` (see loss.py).
    ``soft_steps`` selects which steps use it -- ``"first"`` (step 0 only, the
    default and what the dumped top-K's conditioning justifies) or ``"all"``.
    Batches without top-K tensors silently keep the hard loss.
    """
    if mask_mode not in ("accepted", "consistent"):
        raise ValueError(f"mask_mode must be 'accepted' or 'consistent', not {mask_mode!r}")
    if soft_steps not in ("first", "all"):
        raise ValueError(f"soft_steps must be 'first' or 'all', not {soft_steps!r}")
    next_ids = batch["next_token_ids"]
    hc = batch["hc_hidden"]
    pos = batch["positions"]
    labels = batch["labels"]
    valid = batch["valid"]
    cons = batch.get("greedy_consistent")
    if cons is None:
        cons = torch.zeros_like(valid)
    stats: List[dict] = []

    soft = None
    if soft_alpha > 0.0:
        soft = soft_targets_from_batch(batch, temperature=soft_temp, valid=valid)

    def _loss(mixed_flat, lab_flat, shift: int):
        """Hard CE, or the alpha-mixed soft CE when this step uses soft targets.

        ``shift`` is the rolled-step offset j: the soft target for the row a
        step-j prediction must match sits j rows further along, exactly like
        ``labels``.
        """
        use_soft = soft is not None and (soft_steps == "all" or shift == 0)
        if not use_soft:
            return chunked_ce(
                mixed_flat, model.lm_head, lab_flat, chunk=chunk, ignore_index=IGNORE
            )
        ids, q, ok = soft
        b_, t_ = labels.shape
        ids = shift_left(ids.view(b_, t_, -1), shift, 0).reshape(-1, ids.shape[-1])
        q = shift_left(q.view(b_, t_, -1), shift, 0.0).reshape(-1, q.shape[-1])
        ok = shift_left(ok.view(b_, t_), shift, False).reshape(-1)
        ok = ok & (lab_flat != IGNORE)
        q = q * ok.unsqueeze(-1).to(q.dtype)
        return chunked_mixed_ce(
            mixed_flat, model.lm_head, lab_flat, ids, q, ok,
            alpha=soft_alpha, chunk=chunk, ignore_index=IGNORE,
        )

    # Taps feed the teacher-forced (target-fed) step only; the rolled steps
    # recurse on the head's own HC state, exactly as serving does.
    mixed, own_hc, kv = model.forward_train(
        next_ids, hc, pos, valid, tap_hidden=batch.get("tap_hidden")
    )
    lab0 = torch.where(valid, labels, torch.full_like(labels, IGNORE))
    loss0 = _loss(mixed.reshape(-1, mixed.shape[-1]), lab0.reshape(-1), 0)
    rec = {"step": 0, "rows": int(valid.sum()), "loss": float(loss0.detach())}
    stats.append(rec)
    if on_term is not None:
        on_term(0, weights[0], loss0, rec)

    if k <= 0:
        return stats

    pred = _argmax(model, mixed.detach(), chunk)
    own_prev = own_hc.detach()
    kv = (kv[0].detach(), kv[1].detach())
    del mixed, own_hc, loss0

    consec = valid.clone()
    accepted = valid.clone()
    for j in range(1, k + 1):
        consec = consec & shift_left(cons, j - 1, False)
        accepted = accepted & (pred == shift_left(labels, j - 1, IGNORE))
        ok = valid & shift_left(valid, j, False) & consec
        if mask_mode == "accepted":
            ok = ok & accepted
        lab_j = torch.where(ok, shift_left(labels, j, IGNORE),
                            torch.full_like(labels, IGNORE))
        # rows with a label: a term over none is unmeasured (loss None), not 0
        n_ok = int((lab_j != IGNORE).sum())
        rec = {"step": j, "rows": n_ok, "loss": None}
        stats.append(rec)
        if n_ok == 0:
            break
        out_j, own_j = model.forward_rolled(pred, own_prev, pos + j, kv, valid)
        loss_j = _loss(out_j.reshape(-1, out_j.shape[-1]), lab_j.reshape(-1), j)
        rec["loss"] = float(loss_j.detach())
        if on_term is not None:
            on_term(j, weights[j], loss_j, rec)
        if j < k:
            pred = _argmax(model, out_j.detach(), chunk)
            own_prev = own_j.detach()
        del out_j, own_j, loss_j
    return stats


__all__ = ["chain_losses", "step_weights", "shift_left", "IGNORE"]
