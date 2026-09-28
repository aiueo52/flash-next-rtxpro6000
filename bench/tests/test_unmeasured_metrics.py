"""Missing measurements are refused or None, never 0 (CPU)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]


def test_g1_parsers_refuse_a_trace_with_a_missing_gemm(tmp_path):
    p = tmp_path / "g1.json"
    p.write_text(json.dumps([{"T": 4, "distinct": 10, "kernels": {"_ZN7cutlass_gemm": 5.0, "glue": 1.0},
                              "us_per_call_sustained": 7.0, "sm_clock_sustained": 2000}]))
    r = subprocess.run([sys.executable, "-B", str(BENCH / "bench/moe_smallm/g1/cmp_json.py"), f"a={p}", f"b={p}"],
                       capture_output=True, text=True, timeout=120, cwd=tmp_path)
    assert r.returncode != 0 and "expected 2" in (r.stderr + r.stdout)


def test_sanity_metrics_of_an_empty_output_are_unmeasured():
    import ast

    src = (BENCH / "bench/quality/run_bench.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "metrics")
    ns: dict = {}
    mod = ast.fix_missing_locations(ast.Module(body=[ast.Import(names=[ast.alias(name="re")]), fn], type_ignores=[]))
    exec(compile(mod, "m", "exec"), ns)
    empty = ns["metrics"]("")
    assert empty["uniq_words"] is None and empty["rep_4gram"] is None
    full = ns["metrics"]("a b c d e f g")
    assert full["uniq_words"] == 1.0 and full["rep_4gram"] == 0.0
