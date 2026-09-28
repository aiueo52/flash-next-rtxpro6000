"""Every analysis/report script either runs on the published results or refuses cleanly.

Each script runs (CPU only, CUDA_VISIBLE_DEVICES="", HOME in a temp dir) inside a temporary copy of
the repository. A script that works on the published files must exit 0, write its output only into
the per-test temp dir and change nothing in the repository copy. A script that needs unpublished
data (raw runs, traces, private prompt sources) must exit non-zero with a message saying so and
no traceback, again without touching the repository copy.
"""
from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2]
PY = sys.executable

# (id, cwd relative to the repository, argv after the interpreter, expectation, output file in {OUT}
# the script must produce (None: a report on stdout; "-": only the exit status), modules it imports).
OK, REFUSE = "ok", "refuse"
CASES = [
    ("make_tables", ".", ["results/make_tables.py", "--out", "{OUT}/TABLES.md"], OK, "TABLES.md", ()),
    ("make_tables_check", ".", ["results/make_tables.py", "--check"], OK, "-", ()),
    ("fnbench_report", "bench", ["-m", "fnbench", "report", "{R}/results/runs/a0full-w4.jsonl"], OK, None, ()),
    ("q1_analyze", "bench", ["bench/quality/analyze.py", "--runs", "{R}/results/quality-q1", "--allow-unmarked"],
     OK, None, ()),
    ("q1_analyze_raw_runs", "bench", ["bench/quality/analyze.py"], REFUSE, None, ()),
    ("q1_write_report", "bench", ["bench/quality/write_report.py", "--runs", "{R}/results/quality-q1",
                                  "--allow-unmarked", "--out", "{OUT}/q1.md"], OK, "q1.md", ()),
    ("q1_write_report_raw_runs", "bench", ["bench/quality/write_report.py"], REFUSE, None, ()),
    ("hc1_analyze", "bench", ["bench/hc1/quality/analyze.py", "--runs", "{R}/results/quality-q1",
                              "--allow-unmarked"], OK, None, ()),
    ("hc1_report", "bench", ["bench/hc1/report.py"], REFUSE, None, ()),
    ("hc1_status", "bench", ["bench/hc1/status.py"], REFUSE, None, ()),
    ("agree_cmp", "bench", ["prof/agree_cmp.py", "{R}/bench/n1/qual/n1bq-off-w4.agree.json",
                            "{R}/bench/n1/qual/n1bq-on-w4.agree.json"], REFUSE, None, ("transformers",)),
    *[(f"prof_{n}", "bench", [f"prof/{n}.py"], REFUSE, None, ()) for n in (
        "analyze_trace", "attribute", "exclusive_time", "gridstats", "l2_compare", "moe_chain",
        "phase_kernels", "stack_agg", "stack_kernels", "trimmed_step", "p1_moe", "ql1_analyze", "ql1_finalize")],
    ("sse_wall", "bench", ["prof/sse_wall.py", "{R}/results/runs/a0full-w4.jsonl"], REFUSE, None, ()),
    ("paired_ab", "bench", ["bench/stats/paired_ab.py", "--arms", "{R}/results/bn1-study-v2/w4-A1.jsonl",
                            "{R}/results/bn1-study-v2/wa-A1.jsonl", "{R}/results/bn1-study-v2/w4-A2.jsonl",
                            "{R}/results/bn1-study-v2/wa-A2.jsonl", "--draws", "1000", "--out", "{OUT}/p.json"],
     OK, "p.json", ("numpy", "scipy")),
    ("variance", "bench", ["bench/stats/variance.py", "{R}/results/bn1-study-v2", "--out", "{OUT}/v.json"],
     OK, "v.json", ("numpy",)),
    ("render_variance", "bench", ["bench/stats/render_variance.py", "{OUT}/none.json"], REFUSE, None, ()),
    ("validate_study", "bench", ["bench/stats/validate_study.py", "{R}/results/bn1-study-v2", "--out",
                                 "{OUT}/vs.json"], REFUSE, None, ("numpy", "scipy")),
    ("audit_sets", "bench", ["bench/stats/audit_sets.py", "--out", "{OUT}/as.json"], REFUSE, None, ("tokenizers",)),
    ("wj1_history", "bench", ["bench/stats/wj1_history.py"], REFUSE, None, ("numpy", "scipy")),
    ("wj1_report", "bench", ["bench/stats/wj1_report.py", "{R}/results/bn1-study-v2"], REFUSE, None,
     ("numpy", "scipy")),
    ("analyze_census", "bench", ["bench/moe_smallm/analyze_census.py"], REFUSE, None, ("numpy",)),
    ("analyze_routes", "bench", ["bench/moe_smallm/analyze_routes.py"], REFUSE, None, ()),
    ("g1_cmp_json", "bench", ["bench/moe_smallm/g1/cmp_json.py"], REFUSE, None, ()),
    ("g1_compare", "bench", ["bench/moe_smallm/g1/compare.py"], REFUSE, None, ()),
    ("p4_audit", "bench", ["bench/moe_smallm/p4_audit.py", "x"], REFUSE, None, ()),
    ("p4_quality_report", "bench", ["bench/moe_smallm/p4_quality_report.py", "x"], REFUSE, None, ()),
    ("p4_report", "bench", ["bench/moe_smallm/p4_report.py", "x"], REFUSE, None, ()),
    *[(n, "bench", [f"calib/{n}.py"], REFUSE, None, ()) for n in (
        "wa2_report", "wa3_diagnose", "wa3_report", "wa3_select", "wa4_diagnose", "wa4_gate", "wa4_report",
        "wa5_audit", "wa5_gate", "wa5_report", "wa6_analyze", "wa6_audit", "wa6_costs", "wa6_finalize",
        "wa6_gate", "wa6_mechanism")],
    ("ws1_report", "bench", ["calib/ws1_report.py", "--allow-unmarked", "--runs", "{R}/results/runs", "ws11337",
                             "{R}/bench/calib/logs/ws1_sweep.log"], OK, None, ()),
    ("ws1_report_raw_runs", "bench", ["calib/ws1_report.py", "ws11337", "{R}/bench/calib/logs/ws1_sweep.log"],
     REFUSE, None, ()),
    ("fp8coding_audit_conversion", "bench", ["bench/fp8coding/audit_conversion.py"], REFUSE, None,
     ("numpy", "torch", "safetensors")),
    ("fp8coding_report", "bench", ["bench/fp8coding/report.py"], REFUSE, None, ("numpy", "torch")),
    ("prune_policy_sim", "bench", ["bench/quality/prune_policy_sim.py"], REFUSE, None, ("numpy",)),
    ("calibrate_contrib_proxy", "bench", ["bench/quality/calibrate_contrib_proxy.py", "--artifact-dir",
                                          "{OUT}/a", "--report", "{OUT}/r.md"], REFUSE, None, ("numpy",)),
    ("build_p4_norm_table", "bench", ["bench/quality/build_p4_norm_table.py"], REFUSE, None, ("numpy",)),
    ("adaptive_c1_sim", "bench", ["adaptive/c1/sim.py"], OK, None, ()),
]


def snapshot(root: Path) -> dict[str, tuple[int, int]]:
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file() or p.is_symlink()}


@pytest.fixture(scope="module")
def repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("published") / "repo"
    shutil.copytree(SRC, root, symlinks=True, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    return root


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_script_on_published_results(case, repo: Path, tmp_path: Path) -> None:
    name, cwd, argv, expect, output, modules = case
    for module in modules:
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"{module} not installed")
    out, home = tmp_path / "out", tmp_path / "home"
    out.mkdir()
    home.mkdir()
    args = [a.replace("{R}", str(repo)).replace("{OUT}", str(out)) for a in argv]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CUDA", "PYTHON"))}
    env.update(HOME=str(home), CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2",
               MKL_NUM_THREADS="2", TORCH_NUM_THREADS="2", DISPLAY=":99")
    before = snapshot(repo)
    r = subprocess.run([PY, *args], cwd=repo / cwd, env=env, capture_output=True, text=True, timeout=600)
    after = snapshot(repo)
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    assert not changed, f"{name} wrote into the repository: {changed[:10]}"
    text = r.stdout + r.stderr
    assert "Traceback" not in r.stderr, r.stderr[-3000:]
    if expect == OK:
        assert r.returncode == 0, text[-3000:]
        if output and output != "-":
            assert (out / output).is_file() and (out / output).stat().st_size, f"no {output} in the temp dir"
        elif output is None:
            assert r.stdout.strip(), "no report on stdout"
    else:
        assert r.returncode != 0, f"{name} should refuse without unpublished data:\n{text[-2000:]}"
        assert "not published" in text, f"refusal of {name} does not say what is missing:\n{text[-2000:]}"


ANALYSIS_NAME = re.compile(
    r"(report|analy[sz]e|audit|gate|diagnose|compare|cmp_json|finalize|validate_study|render_|history|mechanism|"
    r"costs|select|stack_|trace|attribute|gridstats|moe_chain|phase_kernels|trimmed_step|p1_moe|l2_compare|"
    r"sse_wall|exclusive_time|variance|paired_ab|policy_sim|contrib_proxy|norm_table|make_tables)")
# Matching names that are not analysis/report entry points.
NOT_ANALYSIS = {
    "bench/bench/moe_smallm/route_census.py",       # GPU-side route capture hook
    "bench/bench/moe_smallm/sitecustomize_census.py",  # GPU-side capture bootstrap
    "bench/bench/stats/run_variance.py",            # GPU run driver (writes the study, not a report of it)
    "bench/bench/quality/contrib_policy_sim.py",    # library for prune_policy_sim / calibrate_contrib_proxy
    "bench/fnbench/report.py",                      # run as `python -m fnbench report` (case fnbench_report)
}


def test_every_analysis_script_is_covered() -> None:
    scripts = {str(p.relative_to(SRC)) for p in [*SRC.glob("bench/**/*.py"), *SRC.glob("results/*.py")]
               if "tests" not in p.parts and "__pycache__" not in p.parts
               and not p.name.startswith(("test_", "tests_")) and ANALYSIS_NAME.search(p.name)}
    scripts.add("bench/adaptive/c1/sim.py")
    covered = {("" if cwd == "." else cwd + "/") + argv[0] for _, cwd, argv, *_ in CASES if argv[0].endswith(".py")}
    missing = sorted(scripts - covered - NOT_ANALYSIS)
    assert not missing, f"analysis scripts without a published-data case: {missing}"
