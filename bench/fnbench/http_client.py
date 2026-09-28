from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import requests

from .models import JsonObject, Sampling

type Clock = Callable[[], float]


class EndpointError(RuntimeError):
    """Raised when an endpoint cannot complete a benchmark request."""


@dataclass(frozen=True)
class StreamMeasurement:
    ttft_seconds: float | None
    decode_seconds: float | None
    decode_tps: float | None
    usage: JsonObject
    timeline: list[list[int]]
    payloads: list[JsonObject]
    output_chars: int
    reasoning_chars: int
    output_text: str
    output_sha256: str


def calculate_decode_tps(
    completion_tokens: int | None,
    first_token_at: float | None,
    last_token_at: float | None,
) -> tuple[float | None, float | None]:
    if (
        completion_tokens is None
        or completion_tokens < 2
        or first_token_at is None
        or last_token_at is None
    ):
        return None, None
    elapsed = last_token_at - first_token_at
    if elapsed <= 0:
        return None, None
    return elapsed, (completion_tokens - 1) / elapsed


def _texts_from_chunk(payload: JsonObject) -> tuple[str, str]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return "", ""
    first = choices[0]
    if not isinstance(first, dict):
        return "", ""
    delta = first.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        reasoning = delta.get("reasoning_content")
        return (
            content if isinstance(content, str) else "",
            reasoning if isinstance(reasoning, str) else "",
        )
    content = first.get("text")
    return (content if isinstance(content, str) else ""), ""


def _usage_from_payload(payload: JsonObject) -> JsonObject | None:
    usage = payload.get("usage")
    return dict(usage) if isinstance(usage, dict) else None


class OpenAIStreamClient:
    def __init__(
        self,
        endpoint: str,
        *,
        timeout: tuple[float, float] = (5.0, 300.0),
        session: requests.Session | None = None,
        clock: Clock = time.perf_counter,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.url = f"{self.endpoint}/chat/completions"
        self.timeout = timeout
        self.session = session or requests.Session()
        self.clock = clock

    def resolve_model(self) -> str:
        url = f"{self.endpoint}/models"
        try:
            response = self.session.get(url, timeout=self.timeout[0])
            response.raise_for_status()
            payload: Any = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise EndpointError(
                f"could not auto-detect a model from {url}: {exc}; pass --model explicitly"
            ) from exc
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or not data:
            raise EndpointError(f"{url} returned no models; pass --model explicitly")
        first = data[0]
        model = first.get("id") if isinstance(first, dict) else None
        if not isinstance(model, str) or not model:
            raise EndpointError(f"{url} returned an invalid model id; pass --model explicitly")
        return model

    def complete(
        self,
        *,
        prompt: str,
        max_tokens: int,
        sampling: Sampling,
        model: str,
    ) -> StreamMeasurement:
        body: JsonObject = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            **sampling.as_request_fields(),
        }
        started = self.clock()
        try:
            response = self.session.post(
                self.url,
                json=body,
                stream=True,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            detail = ""
            if getattr(exc, "response", None) is not None:
                detail = f": {exc.response.text[:500]}"
            raise EndpointError(f"request to {self.url} failed: {exc}{detail}") from exc

        first_token_at: float | None = None
        last_token_at: float | None = None
        previous_token_at = started
        timeline: list[list[int]] = []
        payloads: list[JsonObject] = []
        output_parts: list[str] = []
        reasoning_chars = 0
        usage: JsonObject = {}
        saw_done = False
        try:
            for raw_line in response.iter_lines(chunk_size=1, decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    saw_done = True
                    break
                try:
                    decoded: Any = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise EndpointError(f"invalid SSE JSON from {self.url}: {data[:200]}") from exc
                if not isinstance(decoded, dict):
                    continue
                payload: JsonObject = decoded
                payloads.append(payload)
                chunk_usage = _usage_from_payload(payload)
                if chunk_usage is not None:
                    usage = chunk_usage
                content, reasoning = _texts_from_chunk(payload)
                token_bearing_chars = len(content) + len(reasoning)
                if token_bearing_chars == 0:
                    continue
                arrived = self.clock()
                if first_token_at is None:
                    first_token_at = arrived
                delta_ms = round((arrived - previous_token_at) * 1000)
                timeline.append([delta_ms, token_bearing_chars])
                previous_token_at = arrived
                last_token_at = arrived
                if content:
                    output_parts.append(content)
                reasoning_chars += len(reasoning)
        except requests.RequestException as exc:
            raise EndpointError(f"stream from {self.url} failed: {exc}") from exc
        finally:
            response.close()

        if not saw_done:
            raise EndpointError(f"stream from {self.url} ended before the [DONE] event")
        if first_token_at is None:
            raise EndpointError(f"stream from {self.url} contained no token-bearing chunks")

        completion_raw = usage.get("completion_tokens")
        completion_tokens = (
            int(completion_raw)
            if isinstance(completion_raw, (int, float)) and not isinstance(completion_raw, bool)
            else None
        )
        decode_seconds, decode_tps = calculate_decode_tps(
            completion_tokens,
            first_token_at,
            last_token_at,
        )
        output = "".join(output_parts)
        return StreamMeasurement(
            ttft_seconds=(first_token_at - started) if first_token_at is not None else None,
            decode_seconds=decode_seconds,
            decode_tps=decode_tps,
            usage=usage,
            timeline=timeline,
            payloads=payloads,
            output_chars=len(output),
            reasoning_chars=reasoning_chars,
            output_text=output,
            output_sha256=hashlib.sha256(output.encode("utf-8")).hexdigest(),
        )
