from __future__ import annotations

import pytest

from fnbench.parsers import GenericParser, LlamaCppParser, SGLangParser, prometheus_delta


def test_llamacpp_timings_fixture() -> None:
    payloads = [
        {"choices": [{"delta": {"content": "x"}}]},
        {
            "choices": [],
            "timings": {
                "prompt_ms": 125.0,
                "predicted_ms": 80.0,
                "predicted_per_second": 100.0,
            },
        },
    ]
    parsed = LlamaCppParser().parse(payloads)
    assert parsed["timings"]["prompt_ms"] == 125.0
    assert parsed["timings"]["predicted_per_second"] == 100.0


def test_sglang_meta_info_and_accept_length_fixture() -> None:
    payloads = [
        {
            "choices": [{"delta": {"content": "ok"}}],
            "meta_info": {
                "prompt_tokens": 50,
                "spec_accept_length": 3.5,
                "finish_reason": "length",
            },
        }
    ]
    parsed = SGLangParser().parse(payloads)
    assert parsed["meta_info"]["prompt_tokens"] == 50
    assert parsed["accept_length"] == pytest.approx(3.5)


def test_prometheus_text_delta_fixture() -> None:
    before = """# HELP requests_total Total requests
requests_total{route="generate"} 10
spec_accepted_tokens_total 120
queue_depth 2
ignored_without_value
"""
    after = """requests_total{route="generate"} 12
spec_accepted_tokens_total 150
queue_depth 1
new_metric 99
"""
    assert prometheus_delta(before, after) == {
        'requests_total{route="generate"}': 2.0,
        "spec_accepted_tokens_total": 30.0,
        "queue_depth": -1.0,
    }


def test_generic_fallback_and_hook() -> None:
    parsed = GenericParser().parse(
        [{"debug": {"accepted_length": 4}, "timings": {"predicted_ms": 20}}]
    )
    assert parsed == {"timings": {"predicted_ms": 20}, "accept_length": 4.0}
