"""Acceptance-length evaluation by the serving renewal process, offline.

Serving measures ``completion_tokens / spec_verify_calls``: every verify call
accepts ``a`` draft tokens plus one bonus token from the target, and the next
draft chain starts right after that bonus token.  Two things made the old
``accept@k`` estimate biased against that number:

* it started chains at *uniformly random* rows, which over-samples the inside
  of long accepted runs (length-biased sampling); serving weights each verify
  call equally, and a verify always starts right after a rejection;
* it only scored windows where the corpus continuation matched the target's
  greedy continuation, i.e. the easy positions.

This module replays the renewal process exactly on self-generated greedy
data: cursor = first generated row, chain k steps, count leading matches
``a`` against the target's argmax, advance ``a + 1``, repeat until the
sequence's generated tokens are exhausted.  The per-sequence estimate is
``generated_tokens / verifies`` -- the same quantity the server reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

from .data import Sample, collate


def _fmt(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.3f}"


@dataclass
class SeqStat:
    doc_hash: str
    gen_rows: int  # generated tokens: the prefill's first token plus those the verifies produced
    verifies: int
    accepted: int  # sum of a
    hist: List[int]  # count of verifies with a = 0..k

    @property
    def accept_len(self) -> Optional[float]:
        """Generated tokens per verify, the server's completion_tokens /
        verify steps (the first token comes from the prefill, not a verify).
        None when the sequence had no verify (nothing was measured)."""
        return self.gen_rows / self.verifies if self.verifies else None


@dataclass
class RenewalResult:
    k: int
    seqs: List[SeqStat] = field(default_factory=list)

    def summary(self) -> Dict[str, float]:
        v = sum(s.verifies for s in self.seqs)
        a = sum(s.accepted for s in self.seqs)
        tok = sum(s.gen_rows for s in self.seqs if s.verifies)
        hist = [0] * (self.k + 1)
        for s in self.seqs:
            for i, c in enumerate(s.hist):
                hist[i] += c
        per_seq = [s.accept_len for s in self.seqs if s.accept_len is not None]
        # With no verify at all nothing was measured: None, never 0.
        return {
            "sequences": len(self.seqs),
            "verifies": v,
            "tokens": tok,
            "accept_len": tok / v if v else None,  # == server's completion/verify
            "mean_matches": a / v if v else None,
            "per_seq_mean": sum(per_seq) / len(per_seq) if per_seq else None,
            "hist": hist,
        }


@torch.no_grad()
def renewal_batch(
    model,
    samples: Sequence[Sample],
    gen_start_rows: Sequence[int],
    k: int,
    result: Optional[RenewalResult] = None,
    allowed: Optional[torch.Tensor] = None,
) -> RenewalResult:
    """Run the renewal simulation for one batch of samples.

    ``gen_start_rows[i]`` is the first draft row of the generated region:
    ``prompt_tokens - 1`` (that row consumes hc[L-1] and embed(token L), the
    first generated token, and predicts token L+1).
    """
    result = result or RenewalResult(k=k)
    if not samples:
        return result
    device = next(model.parameters()).device
    batch = {kk: vv.to(device) for kk, vv in collate(samples).items()}
    b, t_past = batch["valid"].shape
    lengths = batch["valid"].sum(dim=1)  # rows per sample (= T-1)
    taps = batch.get("tap_hidden")
    _, _, (kc, vc) = model.forward_with_cache(
        batch["next_token_ids"], batch["hc_hidden"], batch["positions"],
        return_logits=False, tap_hidden=taps,
    )
    labels = batch["labels"]
    ar = torch.arange(t_past, device=device)

    cursor = torch.tensor(list(gen_start_rows), dtype=torch.int64, device=device)
    verifies = torch.zeros(b, dtype=torch.int64, device=device)
    accepted = torch.zeros(b, dtype=torch.int64, device=device)
    hist = torch.zeros(b, k + 1, dtype=torch.int64, device=device)

    # Row t consumes token t+1 and is scored against token t+2.  The token
    # at gen_start_rows+1 (the first generated one) comes from the prefill,
    # not from a verify.  A verify at cursor c still has lengths-1-c tokens
    # to produce (tokens c+2 .. T-1); once that is 0 generation is over, so
    # no further verify is counted and no draft is scored past the last
    # generated token.
    while True:
        remaining = lengths - 1 - cursor
        active = remaining >= 1
        if not bool(active.any()):
            break
        idx = active.nonzero(as_tuple=False).squeeze(1)
        st = cursor[idx]
        past_mask = ar.view(1, -1) < st.view(-1, 1)
        preds = model.chain(
            batch["hc_hidden"][idx, st],
            batch["next_token_ids"][idx, st],
            batch["positions"][idx, st],
            (kc[idx], vc[idx]),
            steps=k,
            past_mask=past_mask,
            allowed=allowed,
            start_taps=None if taps is None else taps[idx, st],
        )  # [n, k]
        j = torch.arange(k, device=device).view(1, -1)
        gidx = (st.view(-1, 1) + j).clamp(max=t_past - 1)
        gold = labels[idx.view(-1, 1).expand_as(gidx), gidx]
        avail = j < remaining[idx].view(-1, 1)  # predicted token exists
        matches = (preds == gold) & avail
        a = (~matches).long().cumsum(dim=1).eq(0).sum(dim=1)
        a = torch.minimum(a, remaining[idx])
        verifies[idx] += 1
        accepted[idx] += a
        hist[idx, a] += 1
        cursor[idx] = st + a + 1

    gen_rows = (lengths - torch.tensor(list(gen_start_rows), device=device)).tolist()
    for i, s in enumerate(samples):
        result.seqs.append(
            SeqStat(
                doc_hash=s.doc_hash,
                gen_rows=int(gen_rows[i]),
                verifies=int(verifies[i]),
                accepted=int(accepted[i]),
                hist=hist[i].tolist(),
            )
        )
    return result


@torch.no_grad()
def renewal_evaluate(
    model,
    samples: Sequence[Sample],
    gen_start_rows: Sequence[int],
    ks: Sequence[int] = (3, 15),
    batch_size: int = 32,
    progress: bool = False,
    allowed: Optional[torch.Tensor] = None,
) -> Dict[int, RenewalResult]:
    """Batched over samples, sorted by row count to limit straggler waste."""
    model.eval()
    order = sorted(range(len(samples)), key=lambda i: len(samples[i]))
    results = {k: RenewalResult(k=k) for k in ks}
    for start in range(0, len(order), batch_size):
        ids = order[start : start + batch_size]
        group = [samples[i] for i in ids]
        starts = [gen_start_rows[i] for i in ids]
        for k in ks:
            renewal_batch(model, group, starts, k, results[k], allowed=allowed)
        if progress:
            done = min(len(order), start + batch_size)
            print(f"[renewal] {done}/{len(order)} " + " ".join(
                f"k={k}:{_fmt(results[k].summary()['accept_len'])}" for k in ks), flush=True)
    return results


__all__ = ["SeqStat", "RenewalResult", "renewal_batch", "renewal_evaluate"]
