from __future__ import annotations

import json
from pathlib import Path

import pytest

from fnbench.gpu_guard import GuardResult
from fnbench.models import Sampling, Workload
from fnbench.runner import RunConfig, run_benchmark
from tests.mock_server import mock_sse_server


def test_run_writes_jsonl_at_mock_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "fnbench.runner.check_gpu_guard",
        lambda force, allow_proc: GuardResult(allowed=True, warning=None, blocked=()),
    )
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Return a sequence.", encoding="utf-8")
    workload = Workload(
        name="mock",
        prompt_path=prompt,
        max_tokens=6,
        sampling=Sampling(temperature=0),
        description="mock workload",
    )
    output = tmp_path / "run.jsonl"
    with mock_sse_server() as (endpoint, handler):
        config = RunConfig(
            endpoint=endpoint,
            engine="llamacpp",
            workloads=[workload],
            repeats=1,
            out=output,
            label="integration",
            force=False,
            sampling_mode="greedy",
        )
        assert run_benchmark(config) == 0
        expected_rate = 1 / handler.token_interval

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["client"]["decode_tps"] == pytest.approx(expected_rate, rel=0.10)
    assert record["client"]["usage"]["total_tokens"] == 16
    assert record["client"]["reasoning_chars"] == 0
    assert record["model"] == "mock-model"
    assert record["server"]["timings"]["predicted_per_second"] == expected_rate
    assert len(record["client"]["timeline"]) == handler.token_count
