from __future__ import annotations

import json
from pathlib import Path

from fnbench.report import render_report


def _write(path: Path, rates: list[float]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for rate in rates:
            record = {
                "workload": "code-edit",
                "client": {
                    "decode_tps": rate,
                    "ttft_seconds": 0.01,
                    "usage": {"completion_tokens": 100},
                },
                "server": {"accept_length": 4},
                "derived": {"effective_forward_per_second": rate / 4},
            }
            stream.write(json.dumps(record) + "\n")


def test_report_has_percentiles_and_baseline_delta(tmp_path: Path) -> None:
    current = tmp_path / "current.jsonl"
    baseline = tmp_path / "baseline.jsonl"
    _write(current, [180.0, 200.0, 220.0])
    _write(baseline, [90.0, 100.0, 110.0])
    report = render_report(current, baseline)
    assert "decode median t/s" in report
    assert "code-edit" in report
    assert "200.00" in report
    assert "+100.0%" in report
