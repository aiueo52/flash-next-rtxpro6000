"""Validators never pass on skipped checks or empty input.

A validator that checked nothing must say so ("unverified" / "not measured") and must not give an
overall PASS.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1]
REPO = BENCH.parent


def load(path: Path, name: str, *extra_path: Path):
    sys.path[:0] = [str(p) for p in extra_path]
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        del sys.path[:len(extra_path)]


@pytest.fixture(scope="module")
def study():
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    stats = BENCH / "bench/stats"
    return load(stats / "validate_study.py", "validate_study_under_test", stats)


METHOD = dict(method="normalized containment and 50-char shingles", scope="listed frozen sources")


def test_published_method_only_audit_is_unverified(study) -> None:
    published = json.loads((BENCH / "workloads/sets/provenance-audit.json").read_text())
    check = study.provenance_check(published)
    assert check["status"] == "unverified"
    assert study.overall_status({"source_provenance": check, "frozen_inputs": {"status": "verified"}},
                                requests=120, duration_exceptions=[]) == "UNVERIFIED"


def test_empty_source_list_is_unverified(study) -> None:
    check = study.provenance_check(dict(METHOD, status="PASS", sources=[]))
    assert check == dict(status="unverified", sources=0, reason="the audit lists no sources")


def test_unhashed_sources_are_unverified(study) -> None:
    check = study.provenance_check(dict(METHOD, status="PASS", sources=[{"path": "/nonexistent"}]))
    assert check["status"] == "unverified"


def test_real_audit_with_matching_hashes_is_verified(study, tmp_path: Path) -> None:
    source = tmp_path / "prompts.jsonl"
    source.write_text('{"prompt": "x"}\n')
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    audit = dict(METHOD, status="PASS", sources=[{"path": str(source), "sha256": sha}])
    check = study.provenance_check(audit, digest=lambda p: hashlib.sha256(p.read_bytes()).hexdigest())
    assert check["status"] == "verified"
    checks = {"source_provenance": check, "frozen_inputs": study.frozen_check(
        {"hashes": {"a": "b"}, "model_inventory": {"m": {}}})}
    assert study.overall_status(checks, requests=10, duration_exceptions=[]) == "PASS"
    assert study.overall_status(checks, requests=10, duration_exceptions=["x"]) == "PASS_WITH_DURATION_EXCEPTION"
    source.write_text("changed\n")
    with pytest.raises(AssertionError, match="source changed"):
        study.provenance_check(audit, digest=lambda p: hashlib.sha256(p.read_bytes()).hexdigest())


def test_failed_audit_verdict_is_an_error(study) -> None:
    with pytest.raises(AssertionError, match="REVIEW_REQUIRED"):
        study.provenance_check(dict(METHOD, status="REVIEW_REQUIRED", sources=[]))


def test_empty_frozen_inputs_and_zero_requests_never_pass(study) -> None:
    assert study.frozen_check({"hashes": {}, "model_inventory": {}})["status"] == "unverified"
    verified = {"a": {"status": "verified"}}
    assert study.overall_status(verified, requests=0, duration_exceptions=[]) == "UNVERIFIED"
    assert study.overall_status({}, requests=0, duration_exceptions=[]) == "UNVERIFIED"


def test_audit_sets_with_nothing_compared_is_unverified() -> None:
    pytest.importorskip("tokenizers")
    audit = load(BENCH / "bench/stats/audit_sets.py", "audit_sets_under_test")
    assert audit.audit_status({}, [], [], []) == "UNVERIFIED"
    assert audit.audit_status({"p": "x"}, [], [], []) == "UNVERIFIED"
    assert audit.audit_status({}, [{"path": "s"}], [], []) == "UNVERIFIED"
    assert audit.audit_status({"p": "x"}, [{"path": "s"}], [], []) == "PASS"
    assert audit.audit_status({"p": "x"}, [{"path": "s"}], [], ["p"]) == "REVIEW_REQUIRED"


def test_p4_speed_gate_needs_every_cell_and_step_metric() -> None:
    report = load(BENCH / "bench/moe_smallm/p4_report.py", "p4_report_under_test")
    full = [dict(width=w, workload=wl, acceptance_change_pct=0.1, step_faster_pct=4.0)
            for w, wl in sorted(report.GATE_CELLS)]
    assert report.speed_gate(full)
    assert not report.speed_gate([])
    assert not report.speed_gate(full[1:])
    no_step = [dict(c) for c in full]
    del next(c for c in no_step if c["workload"] == "code-edit")["step_faster_pct"]
    assert not report.speed_gate(no_step)


def test_q1_verdict_without_paired_comparison_is_not_measured(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    shutil.copytree(REPO / "results/quality-q1/prod", runs / "prod")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, str(BENCH / "bench/quality/analyze.py"), "--runs", str(runs),
                        "--allow-unmarked", "prod", "prod"], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stderr
    verdict = [line for line in r.stdout.splitlines() if line.startswith("VERDICT:")]
    assert verdict and verdict[0].startswith("VERDICT: not measured"), verdict
    assert "no significant degradation" not in r.stdout
    assert "| (none) | - | - | - | - | - | not measured | 0 |" in r.stdout
