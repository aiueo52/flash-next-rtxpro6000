from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from typing import Any, Protocol

from .models import JsonObject

PROMETHEUS_SAMPLE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{[^}]*\})?\s+"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[+-]Inf|NaN)"
)
ACCEPT_KEYS = (
    "spec_accept_length",
    "speculative_accept_length",
    "accept_length",
    "accepted_length",
    "mean_accept_length",
)


def parse_prometheus(text: str) -> dict[str, float]:
    samples: dict[str, float] = {}
    for line in text.splitlines():
        match = PROMETHEUS_SAMPLE.match(line.strip())
        if not match:
            continue
        key = f"{match.group('name')}{match.group('labels') or ''}"
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        if math.isfinite(value):
            samples[key] = value
    return samples


def prometheus_delta(before: str, after: str) -> dict[str, float]:
    before_samples = parse_prometheus(before)
    after_samples = parse_prometheus(after)
    return {
        key: value - before_samples[key]
        for key, value in after_samples.items()
        if key in before_samples and value != before_samples[key]
    }


def _walk_objects(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_objects(child)


def _first_mapping(payloads: Iterable[JsonObject], key: str) -> JsonObject | None:
    for payload in payloads:
        for obj in _walk_objects(payload):
            value = obj.get(key)
            if isinstance(value, Mapping):
                return dict(value)
    return None


def _find_accept_length(value: Any) -> float | None:
    for obj in _walk_objects(value):
        for key in ACCEPT_KEYS:
            candidate = obj.get(key)
            if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                number = float(candidate)
                if number > 0 and math.isfinite(number):
                    return number
    return None


class EngineParser(Protocol):
    def parse(self, payloads: list[JsonObject]) -> JsonObject: ...


class LlamaCppParser:
    def parse(self, payloads: list[JsonObject]) -> JsonObject:
        timings = _first_mapping(payloads, "timings")
        result: JsonObject = {}
        if timings is not None:
            result["timings"] = timings
        accept_length = _find_accept_length(payloads)
        if accept_length is not None:
            result["accept_length"] = accept_length
        return result


class SGLangParser:
    def parse(self, payloads: list[JsonObject]) -> JsonObject:
        meta_info = _first_mapping(payloads, "meta_info")
        result: JsonObject = {}
        if meta_info is not None:
            result["meta_info"] = meta_info
        accept_length = _find_accept_length(meta_info if meta_info is not None else payloads)
        if accept_length is not None:
            result["accept_length"] = accept_length
        return result


class GenericParser:
    """Fallback parser for other OpenAI-compatible servers: generic `timings` and accept-length fields."""

    def parse_debug(self, payloads: list[JsonObject]) -> JsonObject:
        del payloads
        return {}

    def parse(self, payloads: list[JsonObject]) -> JsonObject:
        result: JsonObject = self.parse_debug(payloads)
        timings = _first_mapping(payloads, "timings")
        if timings is not None:
            result["timings"] = timings
        accept_length = _find_accept_length(payloads)
        if accept_length is not None:
            result["accept_length"] = accept_length
        return result


def get_engine_parser(engine: str) -> EngineParser:
    parsers: dict[str, EngineParser] = {
        "llamacpp": LlamaCppParser(),
        "sglang": SGLangParser(),
        "generic": GenericParser(),
    }
    return parsers[engine]


def counter_acceptance(before: str | None, after: str | None,
                       completion_tokens: int | None) -> JsonObject:
    """Completed tokens/verify, including target bonus; isolated BS=1 requests only.

    These counters are published on request completion. Missing/new/reset series
    are never silently interpreted as zero. A discarded warmup initializes them.
    """
    result: JsonObject = {"status": "unavailable", "tokens_per_verify": None,
                          "source": "prometheus_completed_request_deltas"}
    if not isinstance(before, str) or not isinstance(after, str):
        return result
    snapshots = [parse_prometheus(x) for x in (before, after)]
    deltas = []
    for name in ("sglang:generation_tokens_total", "sglang:spec_verify_calls_total"):
        series = [{k: v for k, v in snap.items() if k.split("{")[0] == name}
                  for snap in snapshots]
        if not series[0] or series[0].keys() != series[1].keys():
            result["status"] = "missing_or_changed_series"
            return result
        changes = [series[1][k] - v for k, v in series[0].items()]
        if min(changes) < 0:
            result["status"] = "counter_reset"
            return result
        deltas.append(sum(changes))
    tokens, verifies = deltas
    result.update(generation_tokens=tokens, verify_calls=verifies)
    if tokens == 0 or verifies == 0 or (completion_tokens is not None and tokens < completion_tokens):
        result["status"] = "pending"
    elif completion_tokens is None or tokens != completion_tokens:
        result["status"] = "completion_mismatch_or_concurrent_requests"
    else:
        result.update(status="ok", tokens_per_verify=tokens / verifies)
    return result
