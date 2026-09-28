"""Routing generators for the small-M MoE benchmark.

Three modes:

``independent``
    Each of the T tokens draws top-k uniformly at random without replacement.
    E[#distinct] = E*(1-(1-k/E)^T).  This is the analytic upper bound; the
    trace-derived numbers in ``specs/MOE_SMALLM_SPEC.md`` show the real chain
    is well below it.

``correlated``
    A one-parameter chain model that reproduces the fact that consecutive
    speculative tokens route to overlapping experts: token t keeps a Bernoulli(
    ``carry``) subset of token t-1's experts and redraws the rest.  ``carry=0``
    degenerates to ``independent``; ``carry=1`` gives D = k for every T.
    Pick ``carry`` so that the resulting E[D(16)]/E[D(4)] matches the ratio
    measured from the traces (~2.1); ``solve_carry()`` does that.

``replay``
    Replays real per-layer routing logged by ``route_logger.py`` from a live
    server run.  This is the mode that makes the bytes model exact.
"""

from __future__ import annotations

import json
from typing import Iterator, Optional

import torch


def independent(t_tokens: int, top_k: int, num_experts: int,
                generator: torch.Generator) -> torch.Tensor:
    """[T, top_k] int32 expert ids, per-token top-k without replacement."""
    scores = torch.rand((t_tokens, num_experts), generator=generator)
    return scores.topk(top_k, dim=-1).indices.to(torch.int32)


def correlated(t_tokens: int, top_k: int, num_experts: int, carry: float,
               generator: torch.Generator) -> torch.Tensor:
    """Chain model: token t reuses each of token t-1's experts w.p. `carry`."""
    out = torch.empty((t_tokens, top_k), dtype=torch.int32)
    prev = independent(1, top_k, num_experts, generator)[0]
    out[0] = prev
    for t in range(1, t_tokens):
        keep = torch.rand(top_k, generator=generator) < carry
        kept = prev[keep]
        need = top_k - int(kept.numel())
        # redraw `need` experts not already kept
        mask = torch.zeros(num_experts, dtype=torch.bool)
        mask[kept.long()] = True
        scores = torch.rand(num_experts, generator=generator)
        scores[mask] = -1.0
        new = scores.topk(need).indices.to(torch.int32) if need else kept[:0]
        cur = torch.cat([kept, new])
        out[t] = cur
        prev = cur
    return out


def expected_distinct(t_tokens: int, top_k: int, num_experts: int,
                      carry: float, trials: int = 2000, seed: int = 0) -> float:
    g = torch.Generator().manual_seed(seed)
    tot = 0
    for _ in range(trials):
        ids = correlated(t_tokens, top_k, num_experts, carry, g)
        tot += int(torch.unique(ids).numel())
    return tot / trials


def solve_carry(target_ratio: float, top_k: int, num_experts: int,
                t_lo: int = 4, t_hi: int = 16, trials: int = 400,
                seed: int = 0) -> float:
    """Find `carry` s.t. E[D(t_hi)]/E[D(t_lo)] == target_ratio (bisection)."""
    lo, hi = 0.0, 0.95
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        r = (expected_distinct(t_hi, top_k, num_experts, mid, trials, seed)
             / expected_distinct(t_lo, top_k, num_experts, mid, trials, seed))
        if r > target_ratio:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def load_replay(path: str) -> list[torch.Tensor]:
    """Read a route_logger jsonl: one record per (step, layer)."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ids = torch.tensor(rec["topk_ids"], dtype=torch.int32)
            if ids.ndim == 1:
                ids = ids.reshape(-1, rec["top_k"])
            out.append(ids)
    if not out:
        raise ValueError(f"no routing records in {path}")
    return out


def make_routings(mode: str, n: int, t_tokens: int, top_k: int, num_experts: int,
                  seed: int = 0, carry: float = 0.0,
                  replay_path: Optional[str] = None) -> list[torch.Tensor]:
    """`n` independent routing draws, one per benchmark call in the graph."""
    if mode == "replay":
        recs = load_replay(replay_path)
        recs = [r for r in recs if r.shape[0] == t_tokens] or recs
        return [recs[i % len(recs)] for i in range(n)]
    g = torch.Generator().manual_seed(seed)
    if mode == "independent":
        return [independent(t_tokens, top_k, num_experts, g) for _ in range(n)]
    if mode == "correlated":
        return [correlated(t_tokens, top_k, num_experts, carry, g) for _ in range(n)]
    raise ValueError(f"unknown routing mode {mode!r}")


def distinct_counts(routings: list[torch.Tensor]) -> list[int]:
    return [int(torch.unique(r).numel()) for r in routings]


def disjointify(routings: list[torch.Tensor], num_experts: int,
                seed: int = 0) -> list[torch.Tensor]:
    """Shift each routing by a distinct offset so the union over the whole
    rotation covers as much of the expert bank as possible.

    Needed because a single call touches only ~40 experts x 2.7 MB = 108 MB,
    which *fits* in the 128 MB L2; replaying the same routing would measure L2
    bandwidth, not DRAM.  Offsetting successive calls spreads the working set
    over the full 1.4 GB bank.
    """
    out = []
    span = max(1, num_experts // max(1, len(routings)))
    for i, r in enumerate(routings):
        out.append(((r.long() + i * span) % num_experts).to(torch.int32))
    return out


def uniform_weights(routings: list[torch.Tensor]) -> list[torch.Tensor]:
    """Router probabilities; magnitudes do not affect kernel time."""
    return [torch.full(r.shape, 1.0 / r.shape[1], dtype=torch.float32) for r in routings]


def with_distinct(t_tokens: int, top_k: int, num_experts: int, distinct: int,
                  generator: torch.Generator, tries: int = 200) -> torch.Tensor:
    """[T, top_k] ids whose union is EXACTLY `distinct` experts.

    This is the workhorse for the ``--sweep-distinct`` experiment: kernel time
    is (to first order) ``t = a + b*D`` with ``b = bytes_per_expert / BW``, so
    sweeping D and fitting the slope measures the achieved DRAM bandwidth
    *without* needing to know the real routing distribution.  The production D
    can then be read back off the fit using the trace's per-layer kernel time.
    """
    if not (top_k <= distinct <= t_tokens * top_k):
        raise ValueError(f"distinct={distinct} must be in [{top_k}, {t_tokens*top_k}]")
    for _ in range(tries):
        pool = torch.randperm(num_experts, generator=generator)[:distinct]
        # deal every pool member out at least once, round-robin over the T*k slots
        order = pool[torch.randperm(distinct, generator=generator)]
        out = torch.full((t_tokens, top_k), -1, dtype=torch.long)
        slots = [(t, j) for j in range(top_k) for t in range(t_tokens)]
        for i, (t, j) in enumerate(slots[:distinct]):
            out[t, j] = order[i]
        ok = True
        for t in range(t_tokens):
            used = {int(v) for v in out[t] if v >= 0}
            free = [int(v) for v in pool.tolist() if v not in used]
            need = int((out[t] < 0).sum())
            if need > len(free):
                ok = False
                break
            pick = torch.randperm(len(free), generator=generator)[:need]
            vals = [free[int(i)] for i in pick]
            out[t][out[t] < 0] = torch.tensor(vals, dtype=torch.long)
        if ok and int(torch.unique(out).numel()) == distinct:
            return out.to(torch.int32)
    raise RuntimeError(f"could not build routing with exactly {distinct} distinct experts")


def hot_mixture(t_tokens: int, top_k: int, num_experts: int, n_hot: int,
                generator: torch.Generator) -> torch.Tensor:
    """`n_hot` experts shared by every token; the other k-n_hot drawn i.i.d.

    A second sanity model for the routing prior.  Note it produces a *higher*
    D(16)/D(4) ratio than the trace implies, so it is a poor fit on its own --
    see specs/MOE_SMALLM_SPEC.md section 3.
    """
    hot = torch.randperm(num_experts, generator=generator)[:n_hot].to(torch.int32)
    rest = top_k - n_hot
    out = torch.empty((t_tokens, top_k), dtype=torch.int32)
    for t in range(t_tokens):
        scores = torch.rand(num_experts, generator=generator)
        scores[hot.long()] = -1.0
        cold = scores.topk(rest).indices.to(torch.int32) if rest else hot[:0]
        out[t] = torch.cat([hot, cold])
    return out
