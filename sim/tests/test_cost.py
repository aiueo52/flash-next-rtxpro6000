import pytest

from ngramsim.cost import hybrid_tps, mtp_chain_tps, mtp_expected_confirmed, ngram_tps


def test_mtp_geometric_expectation_and_throughput():
    assert mtp_expected_confirmed(2, 0.5) == pytest.approx(1.75)
    assert mtp_chain_tps(2, 0.5, c1_ms=10.0, delta_ms=1.0, draft_ms=2.0) == pytest.approx(
        1000.0 * 1.75 / 16.0
    )


def test_ngram_and_hybrid_cost_small_hand_example():
    # Half the positions verify two extra rows: E[time] = 10 + .5*2*1 = 11 ms.
    assert ngram_tps(
        hit_rate=0.5, expected_confirmed=1.5, steps=2, c1_ms=10.0, delta_ms=1.0
    ) == pytest.approx(1500.0 / 11.0)
    # Hit contribution is 1.0 token/position; misses use 1 + alpha = 1.5.
    # E[tokens]=1+.5*1.5=1.75. Hit time=.5*12; miss MTP time=.5*13.
    assert hybrid_tps(
        hit_rate=0.5,
        hit_confirmed_sum_per_position=1.0,
        ngram_steps=2,
        mtp_steps=1,
        alpha=0.5,
        c1_ms=10.0,
        delta_ms=1.0,
        draft_ms=2.0,
    ) == pytest.approx(1750.0 / 12.5)
