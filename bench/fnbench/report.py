from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from .models import JsonObject


class ReportError(ValueError):
    """Raised for malformed or empty benchmark result files."""


def _load(path: Path) -> list[JsonObject]:
    records: list[JsonObject] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value: Any = json.loads(line)
                if not isinstance(value, dict):
                    raise ReportError(f"{path}:{line_number}: expected a JSON object")
                records.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"cannot read {path}: {exc}") from exc
    if not records:
        raise ReportError(f"no benchmark records in {path}")
    return records


def percentile(values: list[float], percent: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _number(record: JsonObject, *keys: str) -> float | None:
    value: Any = record
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def summarize(records: list[JsonObject]) -> dict[str, JsonObject]:
    grouped: dict[str, list[JsonObject]] = defaultdict(list)
    for record in records:
        workload = record.get("workload")
        if isinstance(workload, str):
            grouped[workload].append(record)
    result: dict[str, JsonObject] = {}
    for workload, items in grouped.items():
        rates = [
            value
            for item in items
            if (value := _number(item, "client", "decode_tps")) is not None
        ]
        ttfts = [
            value
            for item in items
            if (value := _number(item, "client", "ttft_seconds")) is not None
        ]
        tokens = [
            value
            for item in items
            if (value := _number(item, "client", "usage", "completion_tokens")) is not None
        ]
        accepts = [
            value
            for item in items
            if (value := _number(item, "server", "accept_length")) is not None
        ]
        forwards = [
            value
            for item in items
            if (value := _number(item, "derived", "effective_forward_per_second"))
            is not None
        ]
        result[workload] = {
            "count": len(items),
            "decode_median": median(rates) if rates else None,
            "decode_p10": percentile(rates, 0.10) if rates else None,
            "decode_p90": percentile(rates, 0.90) if rates else None,
            "ttft_median_ms": median(ttfts) * 1000 if ttfts else None,
            "completion_tokens_median": median(tokens) if tokens else None,
            "accept_length_median": median(accepts) if accepts else None,
            "effective_forward_median": median(forwards) if forwards else None,
        }
    return result


def _fmt(value: Any, digits: int = 2) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "—"
    return f"{value:.{digits}f}"


def _delta(current: Any, baseline: Any) -> str:
    if not isinstance(current, (int, float)) or not isinstance(baseline, (int, float)):
        return "—"
    if baseline == 0:
        return "—"
    value = (current / baseline - 1) * 100
    return f"{value:+.1f}%"


def render_report(path: Path, baseline_path: Path | None = None) -> str:
    current = summarize(_load(path))
    baseline = summarize(_load(baseline_path)) if baseline_path is not None else {}
    title = f"# fnbench report: {path.name}"
    lines = [title, ""]
    headers = [
        "workload",
        "n",
        "decode median t/s",
        "p10",
        "p90",
        "TTFT median ms",
        "tokens median",
        "accept len median",
        "effective forward/s",
    ]
    if baseline_path is not None:
        headers.append("Δ vs baseline")
        lines.extend([f"Baseline: `{baseline_path}`", ""])
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |")
    for workload in sorted(current):
        item = current[workload]
        row = [
            workload,
            str(item["count"]),
            _fmt(item["decode_median"]),
            _fmt(item["decode_p10"]),
            _fmt(item["decode_p90"]),
            _fmt(item["ttft_median_ms"]),
            _fmt(item["completion_tokens_median"], 1),
            _fmt(item["accept_length_median"]),
            _fmt(item["effective_forward_median"]),
        ]
        if baseline_path is not None:
            base_item = baseline.get(workload, {})
            row.append(_delta(item["decode_median"], base_item.get("decode_median")))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"
