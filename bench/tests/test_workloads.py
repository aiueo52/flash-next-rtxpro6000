from __future__ import annotations

from fnbench.workloads import load_workloads


def test_committed_workload_shapes() -> None:
    workloads = load_workloads()
    assert set(workloads) == {"code-edit", "prose-ja", "prose-en", "agent-loop", "long-ctx"}
    assert len(workloads["code-edit"].read_prompt().splitlines()) >= 200

    long_prompt = workloads["long-ctx"].read_prompt()
    assert 40_000 <= len(long_prompt) <= 120_000
    assert long_prompt.count("Write a continuation of 700-900 words") == 1
    assert workloads["prose-ja"].sampling.temperature > 0
