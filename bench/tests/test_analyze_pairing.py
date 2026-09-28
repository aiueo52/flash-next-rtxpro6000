"""analyze.py (both copies) refuses mismatched, missing or duplicate pairing keys (CPU, published Q1 results)."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1]
REPO = BENCH.parent
Q1 = REPO / "results" / "quality-q1"
SCRIPTS = [BENCH / "bench" / "quality" / "analyze.py", BENCH / "bench" / "hc1" / "quality" / "analyze.py"]


def _run(script: Path, runs: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-B", str(script), "--runs", str(runs), "--allow-unmarked"],
                          capture_output=True, text=True, timeout=300, cwd=runs)


def _copy(tmp_path: Path) -> Path:
    runs = tmp_path / "q1"
    shutil.copytree(Q1, runs)
    return runs


def _drop_first(path: Path) -> dict:
    lines = path.read_text().splitlines(keepends=True)
    path.write_text("".join(lines[1:]))
    return json.loads(lines[0])


@pytest.mark.parametrize("script", SCRIPTS, ids=["quality", "hc1"])
def test_published_results_pass(script, tmp_path):
    r = _run(script, _copy(tmp_path))
    assert r.returncode == 0, r.stderr[-2000:]


@pytest.mark.parametrize("script", SCRIPTS, ids=["quality", "hc1"])
def test_one_item_deleted_on_one_side_is_refused(script, tmp_path):
    runs = _copy(tmp_path)
    _drop_first(runs / "noprune" / "humaneval.jsonl")
    r = _run(script, runs)
    assert r.returncode != 0 and "summary.json says" in r.stderr


@pytest.mark.parametrize("script", SCRIPTS, ids=["quality", "hc1"])
def test_item_sets_that_differ_are_refused_even_with_a_matching_summary(script, tmp_path):
    runs = _copy(tmp_path)
    rec = _drop_first(runs / "noprune" / "humaneval.jsonl")
    sp = runs / "noprune" / "summary.json"
    sm = json.loads(sp.read_text())
    sm["bench"]["humaneval"]["n"] -= 1
    sm["bench"]["humaneval"]["correct"] -= int(bool(rec["correct"]))
    rest = [json.loads(x) for x in (runs / "noprune" / "humaneval.jsonl").read_text().splitlines() if x.strip()]
    h = sm["bench"]["humaneval"]  # keep every derived field consistent so only the item sets differ
    h["acc"] = h["correct"] / h["n"]
    h["truncated"] = sum(x["finish"] == "length" for x in rest)
    h["mean_gen_tokens"] = round(sum(x["gen_tokens"] for x in rest) / len(rest), 1)
    sp.write_text(json.dumps(sm))
    r = _run(script, runs)
    assert r.returncode != 0 and "cover different items" in r.stderr


@pytest.mark.parametrize("script", SCRIPTS, ids=["quality", "hc1"])
def test_duplicate_id_is_refused(script, tmp_path):
    runs = _copy(tmp_path)
    p = runs / "prod" / "jcqa.jsonl"
    first = p.read_text().splitlines(keepends=True)[0]
    p.write_text(p.read_text() + first)
    r = _run(script, runs)
    assert r.returncode != 0 and "duplicate id" in r.stderr


@pytest.mark.parametrize("script", SCRIPTS, ids=["quality", "hc1"])
def test_missing_per_item_file_is_refused(script, tmp_path):
    runs = _copy(tmp_path)
    (runs / "legacy" / "mmlu.jsonl").unlink()
    r = _run(script, runs)
    assert r.returncode != 0 and "is missing" in r.stderr


def _edit_summary(runs: Path, arm: str, bench: str, **fields) -> None:
    p = runs / arm / "summary.json"
    s = json.loads(p.read_text())
    s["bench"][bench].update(fields)
    p.write_text(json.dumps(s))


@pytest.mark.parametrize("script", SCRIPTS, ids=["quality", "hc1"])
@pytest.mark.parametrize("field,value", [("acc", 0.1234), ("truncated", 0), ("mean_gen_tokens", 999.0)])
def test_summary_field_not_matching_items_is_refused(script, field, value, tmp_path):
    # Editing only a displayed field (e.g. acc 12.34% for 1169/1319) must not reach the report.
    runs = _copy(tmp_path)
    _edit_summary(runs, "prod", "gsm8k", **{field: value})
    r = _run(script, runs)
    assert r.returncode != 0 and field in r.stderr and "12.34%" not in r.stdout


@pytest.mark.parametrize("script", SCRIPTS, ids=["quality", "hc1"])
def test_accuracy_is_computed_from_items(script, tmp_path):
    r = _run(script, _copy(tmp_path))
    assert "| gsm8k | 88.63% [86.8, 90.2] (1169/1319) |" in r.stdout


@pytest.mark.parametrize("script", SCRIPTS, ids=["quality", "hc1"])
def test_sanity_rows_need_sanity_jsonl(script, tmp_path):
    runs = _copy(tmp_path)
    r = _run(script, runs)
    assert "Sanity rows not shown for prod" in r.stdout and "tokens/maxrun/uniq/rep4" not in r.stdout
    items = json.loads((runs / "prod" / "summary.json").read_text())["bench"]["sanity"]["items"]
    rows = [dict(id=k, **v) for k, v in items.items()]
    (runs / "prod" / "sanity.jsonl").write_text("\n".join(map(json.dumps, rows)))
    r = _run(script, runs)
    assert r.returncode == 0 and "tokens/maxrun/uniq/rep4" in r.stdout
    rows[0]["maxrun"] = 12345
    (runs / "prod" / "sanity.jsonl").write_text("\n".join(map(json.dumps, rows)))
    r = _run(script, runs)
    assert r.returncode != 0 and "sanity" in r.stderr


@pytest.mark.parametrize("script", SCRIPTS, ids=["quality", "hc1"])
@pytest.mark.parametrize("need,field", [("gen_tokens", "mean_gen_tokens"), ("finish", "truncated")])
def test_one_item_missing_a_verifying_field_is_refused(script, need, field, tmp_path):
    # With one item lacking gen_tokens, a summary mean of 999 must not slip through unverified.
    runs = _copy(tmp_path)
    _edit_summary(runs, "prod", "gsm8k", mean_gen_tokens=999.0)
    p = runs / "prod" / "gsm8k.jsonl"
    lines = p.read_text().splitlines()
    first = json.loads(lines[0]); first.pop(need)
    p.write_text("\n".join([json.dumps(first)] + lines[1:]))
    r = _run(script, runs)
    assert r.returncode != 0 and field in r.stderr and need in r.stderr
