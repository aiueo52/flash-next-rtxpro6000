from __future__ import annotations

import hashlib
import json
import time
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO
from urllib.parse import urlsplit, urlunsplit

import requests

from .gpu_guard import check_gpu_guard, format_block_reason
from .http_client import EndpointError, OpenAIStreamClient
from .models import JsonObject, Sampling, Workload
from .parsers import get_engine_parser, prometheus_delta, counter_acceptance


@dataclass(frozen=True)
class RunConfig:
    endpoint: str
    engine: str
    workloads: list[Workload]
    repeats: int
    out: Path
    label: str | None
    force: bool
    sampling_mode: str
    model: str | None = None
    allow_proc: tuple[str, ...] = ()
    require_acceptance: bool = False


def metrics_url_for_endpoint(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    return urlunsplit((parsed.scheme, parsed.netloc, "/metrics", "", ""))


def fetch_metrics(session: requests.Session, endpoint: str) -> JsonObject:
    url = metrics_url_for_endpoint(endpoint)
    try:
        response = session.get(url, timeout=(2.0, 5.0))
        response.raise_for_status()
    except requests.RequestException as exc:
        return {"url": url, "error": str(exc), "raw": None}
    return {"url": url, "error": None, "raw": response.text}


def _sampling(workload: Workload, mode: str) -> Sampling:
    if mode == "greedy":
        return Sampling(temperature=0.0)
    return workload.sampling


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_record(stream: TextIO, record: JsonObject) -> None:
    stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    stream.flush()


def run_benchmark(config: RunConfig, *, stderr: TextIO = sys.stderr) -> int:
    if config.repeats <= 0:
        raise ValueError("--repeats must be positive")
    guard = check_gpu_guard(force=config.force, allow_proc=config.allow_proc)
    if guard.warning:
        print(f"warning: {guard.warning}", file=stderr)
    if not guard.allowed:
        raise RuntimeError(format_block_reason(guard))
    if guard.forced:
        print(f"warning: {format_block_reason(guard)} Continuing due to --force.", file=stderr)

    config.out.parent.mkdir(parents=True, exist_ok=True)
    if config.out.exists():
        raise FileExistsError(f"output already exists: {config.out}")

    session = requests.Session()
    client = OpenAIStreamClient(config.endpoint, session=session)
    parser = get_engine_parser(config.engine)
    model = config.model or client.resolve_model()

    try:
        with config.out.open("x", encoding="utf-8") as output:
            for workload in config.workloads:
                prompt = workload.read_prompt()
                sampling = _sampling(workload, config.sampling_mode)
                if workload.prompt_set is None:
                    print(f"warming up {workload.name} ...", file=stderr)
                    client.complete(
                        prompt=prompt,
                        max_tokens=workload.max_tokens,
                        sampling=sampling,
                        model=model,
                    )
                for repeat_index in range(config.repeats):
                    before = (
                        fetch_metrics(session, config.endpoint)
                        if config.engine == "sglang"
                        else None
                    )
                    measured_at = _iso_now()
                    measurement = client.complete(
                        prompt=prompt,
                        max_tokens=workload.max_tokens,
                        sampling=sampling,
                        model=model,
                    )
                    after = (
                        fetch_metrics(session, config.endpoint)
                        if config.engine == "sglang"
                        else None
                    )
                    server = parser.parse(measurement.payloads)
                    if before is not None and after is not None:
                        metrics: JsonObject = {"before": before, "after": after, "delta": {}}
                        before_raw = before.get("raw")
                        after_raw = after.get("raw")
                        if isinstance(before_raw, str) and isinstance(after_raw, str):
                            metrics["delta"] = prometheus_delta(before_raw, after_raw)
                        server["metrics"] = metrics
                    if before is not None and after is not None:
                        deadline = time.monotonic() + 2.0
                        while True:
                            acceptance = counter_acceptance(before.get("raw"), after.get("raw"),
                                                            measurement.usage.get("completion_tokens"))
                            if acceptance["status"] != "pending" or time.monotonic() >= deadline:
                                break
                            time.sleep(0.05)
                            after = fetch_metrics(session, config.endpoint)
                        server["acceptance"] = acceptance
                        server["metrics"]["after"] = after
                        if isinstance(before.get("raw"), str) and isinstance(after.get("raw"), str):
                            server["metrics"]["delta"] = prometheus_delta(before["raw"], after["raw"])
                        server["accept_length"] = acceptance["tokens_per_verify"]
                    accept_raw = server.get("accept_length")
                    accept_length = (
                        float(accept_raw)
                        if isinstance(accept_raw, (int, float)) and not isinstance(accept_raw, bool)
                        else None
                    )
                    effective_forward = (
                        measurement.decode_tps / accept_length
                        if measurement.decode_tps is not None
                        and accept_length is not None
                        and accept_length > 0
                        else None
                    )
                    record: JsonObject = {
                        "schema_version": 1,
                        "prompt_id": workload.prompt_id or workload.name,
                        "prompt_set": workload.prompt_set,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "input_tokens_manifest": workload.input_tokens,
                        "timestamp": measured_at,
                        "label": config.label,
                        "engine": config.engine,
                        "endpoint": config.endpoint,
                        "model": model,
                        "workload": workload.name,
                        "workload_description": workload.description,
                        "repeat": repeat_index + 1,
                        "sampling_mode": config.sampling_mode,
                        "sampling": sampling.as_request_fields(),
                        "max_tokens": workload.max_tokens,
                        "prompt_chars": len(prompt),
                        "client": {
                            "ttft_seconds": measurement.ttft_seconds,
                            "decode_seconds": measurement.decode_seconds,
                            "decode_tps": measurement.decode_tps,
                            "usage": measurement.usage,
                            "timeline": measurement.timeline,
                            "timeline_encoding": (
                                "[delta_ms_from_previous_token_or_start,chunk_chars]"
                            ),
                            "output_chars": measurement.output_chars,
                            "reasoning_chars": measurement.reasoning_chars,
                            "output_sha256": measurement.output_sha256,
                        },
                        "server": server,
                        "derived": {"effective_forward_per_second": effective_forward},
                    }
                    _write_record(output, record)
                    if config.require_acceptance and (accept_length is None):
                        raise RuntimeError(f"invalid per-prompt acceptance: {server.get('acceptance')}")
                    rate = measurement.decode_tps
                    rate_text = f"{rate:.2f} t/s" if rate is not None else "n/a"
                    progress = (
                        f"measured {workload.prompt_id or workload.name} "
                        f"{repeat_index + 1}/{config.repeats}: {rate_text}"
                    )
                    print(progress, file=stderr)
    except Exception:
        if config.out.exists() and config.out.stat().st_size == 0:
            config.out.unlink()
        raise
    finally:
        session.close()
    return 0


__all__ = ["EndpointError", "RunConfig", "run_benchmark"]
