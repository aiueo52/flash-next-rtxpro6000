from __future__ import annotations


def mtp_expected_confirmed(steps: int, alpha: float) -> float:
    """One target token plus a truncated geometric run of accepted draft tokens."""
    return sum(alpha**power for power in range(steps + 1))


def mtp_chain_tps(steps: int, alpha: float, c1_ms: float, delta_ms: float, draft_ms: float) -> float:
    elapsed = c1_ms + steps * (delta_ms + draft_ms)
    return 1000.0 * mtp_expected_confirmed(steps, alpha) / elapsed


def ngram_tps(
    *, hit_rate: float, expected_confirmed: float, steps: int, c1_ms: float, delta_ms: float
) -> float:
    elapsed = c1_ms + hit_rate * steps * delta_ms
    return 1000.0 * expected_confirmed / elapsed


def hybrid_tps(
    *,
    hit_rate: float,
    hit_confirmed_sum_per_position: float,
    ngram_steps: int,
    mtp_steps: int,
    alpha: float,
    c1_ms: float,
    delta_ms: float,
    draft_ms: float,
) -> float:
    no_hit = 1.0 - hit_rate
    expected_tokens = hit_confirmed_sum_per_position + no_hit * mtp_expected_confirmed(mtp_steps, alpha)
    ngram_time = hit_rate * (c1_ms + ngram_steps * delta_ms)
    mtp_time = no_hit * (c1_ms + mtp_steps * (delta_ms + draft_ms))
    return 1000.0 * expected_tokens / (ngram_time + mtp_time)
