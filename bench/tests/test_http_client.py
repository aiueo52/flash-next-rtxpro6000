from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest

from fnbench.http_client import OpenAIStreamClient
from fnbench.models import Sampling


class FakeResponse:
    text = ""

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self, **kwargs: Any) -> Iterator[str]:
        del kwargs
        for token in ("A", "B", "C"):
            yield "data: " + json.dumps(
                {"choices": [{"delta": {"content": token}}]}
            )
            yield ""
        yield "data: " + json.dumps(
            {"choices": [], "usage": {"completion_tokens": 3, "total_tokens": 7}}
        )
        yield "data: [DONE]"

    def close(self) -> None:
        return None


class FakeSession:
    def post(self, *args: Any, **kwargs: Any) -> FakeResponse:
        del args, kwargs
        return FakeResponse()


class ReasoningResponse(FakeResponse):
    def iter_lines(self, **kwargs: Any) -> Iterator[str]:
        del kwargs
        for reasoning in ("r1", "r2", "r3"):
            yield "data: " + json.dumps(
                {"choices": [{"delta": {"reasoning_content": reasoning}}]}
            )
        for content in ("A", "B"):
            yield "data: " + json.dumps(
                {"choices": [{"delta": {"content": content}}]}
            )
        yield "data: " + json.dumps(
            {"choices": [], "usage": {"completion_tokens": 5, "total_tokens": 9}}
        )
        yield "data: [DONE]"


class ReasoningSession:
    def post(self, *args: Any, **kwargs: Any) -> ReasoningResponse:
        del args, kwargs
        return ReasoningResponse()


def test_known_sse_chunk_rate() -> None:
    times = iter((0.0, 0.1, 0.2, 0.3))
    client = OpenAIStreamClient(
        "http://example.invalid/v1",
        session=FakeSession(),  # type: ignore[arg-type]
        clock=lambda: next(times),
    )
    result = client.complete(
        prompt="test",
        max_tokens=3,
        sampling=Sampling(temperature=0),
        model="mock",
    )
    assert result.ttft_seconds == pytest.approx(0.1)
    assert result.decode_seconds == pytest.approx(0.2)
    assert result.decode_tps == pytest.approx(10.0)
    assert result.timeline == [[100, 1], [100, 1], [100, 1]]
    assert result.reasoning_chars == 0
    assert result.output_text == "ABC"


def test_reasoning_chunks_start_ttft_and_decode_window() -> None:
    times = iter((0.0, 0.05, 0.10, 0.15, 0.20, 0.25))
    client = OpenAIStreamClient(
        "http://example.invalid/v1",
        session=ReasoningSession(),  # type: ignore[arg-type]
        clock=lambda: next(times),
    )
    result = client.complete(
        prompt="test",
        max_tokens=5,
        sampling=Sampling(temperature=0),
        model="mock",
    )
    assert result.ttft_seconds == pytest.approx(0.05)
    assert result.decode_seconds == pytest.approx(0.20)
    assert result.decode_tps == pytest.approx(20.0)
    assert result.timeline == [[50, 2], [50, 2], [50, 2], [50, 1], [50, 1]]
    assert result.reasoning_chars == 6
    assert result.output_chars == 2
