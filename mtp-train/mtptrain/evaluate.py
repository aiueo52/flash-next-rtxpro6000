"""Held-out evaluation: teacher-forced agreement and simulated chain acceptance.

Two metrics, both computed on data the trainer never saw:

``agreement@1``
    fraction of rows whose top-1 draft token equals ``target_argmax[t+1]``.
    It upper-bounds the first draft step's acceptance.

``accept@k``
    mean number of consecutive accepted tokens when the topk=1 chain is rolled
    ``k`` steps exactly as the server does: step 0 consumes the *target's* HC
    state, every later step consumes the draft's own pre-mixer HC state and its
    own previous prediction, positions increment by one, and the draft attends
    to its own KV.  Only windows whose corpus continuation coincides with the
    target's greedy continuation are counted -- beyond step 1 the dumped
    ``target_argmax`` is conditioned on the corpus token, so a window where the
    corpus diverges cannot be scored (see ``Sample.greedy_consistent``).  On
    self-generated data every window is consistent by construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F

from .data import Sample, causal_mask_from_valid, collate
from .loss import chunked_eval


@dataclass
class EvalResult:
    rows: int = 0
    correct: int = 0
    loss_sum: float = 0.0
    accept: Dict[int, List[float]] = field(default_factory=dict)
    windows: Dict[int, int] = field(default_factory=dict)

    # A metric with nothing to average over is *unmeasured*: ``None`` (JSON
    # null) in the summary, next to the count that is zero.  It is never 0.0,
    # which would read as "the head accepted nothing".

    @property
    def agreement(self) -> Optional[float]:
        return self.correct / self.rows if self.rows else None

    @property
    def loss(self) -> Optional[float]:
        return self.loss_sum / self.rows if self.rows else None

    def summary(self) -> Dict[str, Optional[float]]:
        loss = self.loss
        out = {
            "rows": float(self.rows),
            "agreement@1": self.agreement,
            "ce": loss,
            "ppl": None if loss is None else math.exp(min(20.0, loss)),
        }
        for k, vals in sorted(self.accept.items()):
            out[f"accept@{k}"] = sum(vals) / len(vals) if vals else None
            out[f"accept@{k}_windows"] = float(self.windows.get(k, 0))
        return out


@torch.no_grad()
def teacher_forced(
    model,
    batch: Dict[str, torch.Tensor],
    result: Optional[EvalResult] = None,
    chunk: int = 512,
) -> EvalResult:
    result = result or EvalResult()
    mask = causal_mask_from_valid(batch["valid"])
    mixed, _ = model.forward_mixed(
        batch["next_token_ids"], batch["hc_hidden"], batch["positions"],
        attn_mask=mask, tap_hidden=batch.get("tap_hidden"),
    )
    valid = batch["valid"]
    correct, count, loss_sum = chunked_eval(
        mixed[valid], model.lm_head, batch["labels"][valid], chunk=chunk
    )
    result.rows += count
    result.correct += correct
    result.loss_sum += loss_sum
    return result


@torch.no_grad()
def chain_acceptance(
    model,
    samples: Sequence[Sample],
    k: int,
    starts_per_sample: int = 8,
    seed: int = 0,
    result: Optional[EvalResult] = None,
) -> EvalResult:
    """Batched chain simulation: one start position per sample, repeated."""
    result = result or EvalResult()
    result.accept.setdefault(k, [])
    result.windows.setdefault(k, 0)
    if not samples:
        return result
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed)

    batch = collate(samples)
    batch = {kk: vv.to(device) for kk, vv in batch.items()}
    mask = causal_mask_from_valid(batch["valid"])
    _, _, (kc, vc) = model.forward_with_cache(
        batch["next_token_ids"],
        batch["hc_hidden"],
        batch["positions"],
        tap_hidden=batch.get("tap_hidden"),
        return_logits=False,
    )
    # The cached K/V above used a plain causal mask; padded rows are never read
    # because ``past_mask`` below hides every index >= the chain start.
    b, t_past = batch["valid"].shape
    lengths = batch["valid"].sum(dim=1)

    # ``labels[t]`` is the target token the draft row t must produce.
    labels = batch["labels"]
    cons = torch.zeros(b, t_past, dtype=torch.bool, device=device)
    for i, s in enumerate(samples):
        cons[i, : len(s)] = s.greedy_consistent.to(device)

    for rep in range(starts_per_sample):
        starts = []
        usable = []
        for i in range(b):
            n = int(lengths[i])
            hi = n - k - 1
            if hi <= 1:
                starts.append(0)
                usable.append(False)
                continue
            t = int(torch.randint(1, hi, (1,), generator=generator))
            # Step j (0-indexed) of a chain from t consumes the draft's own
            # predictions at positions t+2 .. t+j+1, so the dumped
            # target_argmax is the right label only while the corpus token
            # equals what the draft was meant to emit:
            #     next_token_ids[t+1+i] == labels[t+i]  for i < j,
            # which is exactly greedy_consistent[t .. t+j-1].
            # A k-step chain therefore needs cons[t .. t+k-2].
            ok = bool(cons[i, t : t + k - 1].all()) if k > 1 else True
            starts.append(t)
            usable.append(ok)
        if not any(usable):
            continue
        idx = torch.tensor(
            [i for i, ok in enumerate(usable) if ok], dtype=torch.int64, device=device
        )
        st = torch.tensor(
            [starts[int(i)] for i in idx], dtype=torch.int64, device=device
        )
        past_mask = (
            torch.arange(t_past, device=device).view(1, -1) < st.view(-1, 1)
        )
        taps = batch.get("tap_hidden")
        preds = model.chain(
            batch["hc_hidden"][idx, st],
            batch["next_token_ids"][idx, st],
            batch["positions"][idx, st],
            (kc[idx][:, :t_past], vc[idx][:, :t_past]),
            steps=k,
            past_mask=past_mask,
            start_taps=None if taps is None else taps[idx, st],
        )
        gold = torch.stack(
            [labels[i, s : s + k] for i, s in zip(idx.tolist(), st.tolist())]
        )
        matches = preds == gold
        # Accepted length = number of leading matches.
        accepted = (~matches).float().cumsum(dim=1).eq(0).sum(dim=1)
        result.accept[k].extend(accepted.float().tolist())
        result.windows[k] += int(idx.numel())
    return result


@torch.no_grad()
def evaluate(
    model,
    samples: Iterable[Sample],
    batch_size: int = 4,
    chain_ks: Sequence[int] = (3, 15),
    starts_per_sample: int = 8,
    max_batches: Optional[int] = None,
    seed: int = 0,
) -> EvalResult:
    model.eval()
    device = next(model.parameters()).device
    result = EvalResult()
    for k in chain_ks:  # reported (as unmeasured if need be) even with no sample
        result.accept.setdefault(k, [])
        result.windows.setdefault(k, 0)
    buf: List[Sample] = []
    n_batches = 0
    for sample in samples:
        buf.append(sample)
        if len(buf) < batch_size:
            continue
        batch = {k: v.to(device) for k, v in collate(buf).items()}
        teacher_forced(model, batch, result)
        for k in chain_ks:
            chain_acceptance(
                model, buf, k, starts_per_sample=starts_per_sample,
                seed=seed + n_batches, result=result,
            )
        buf = []
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
    if buf:
        batch = {k: v.to(device) for k, v in collate(buf).items()}
        teacher_forced(model, batch, result)
        for k in chain_ks:
            chain_acceptance(
                model, buf, k, starts_per_sample=starts_per_sample,
                seed=seed + n_batches, result=result,
            )
    return result


def chain_window_count(samples: Iterable[Sample], k: int) -> int:
    """How many samples can host at least one scorable k-step chain window.

    Mirrors ``chain_acceptance``: a start t in [1, n-k-2] whose
    ``greedy_consistent[t .. t+k-2]`` all hold.  Zero means ``accept@k``
    cannot be measured on these samples, whatever the model.
    """
    n_ok = 0
    for s in samples:
        n = len(s)
        cons = s.greedy_consistent.to(torch.bool)
        for t in range(1, n - k - 1):
            if k <= 1 or bool(cons[t : t + k - 1].all()):
                n_ok += 1
                break
    return n_ok


__all__ = ["EvalResult", "teacher_forced", "chain_acceptance", "chain_window_count",
           "evaluate"]
