"""CPU-only quality-gate and client-concurrency regression checks.

    python3 -B -m unittest discover -s bench/quality -p test_q2.py -v
HTTP is mocked; no server, GPU or generated-code execution is used.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


class QualityGateTests(unittest.TestCase):
    def analyze(self, x, y, margin="0.5", n=100, bench="gsm8k"):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root = Path(tmp)
            shutil.copyfile(HERE / "analyze.py", root / "analyze.py")
            for arm in ("prod", "noprune"):
                dest = root / "runs" / arm
                dest.mkdir(parents=True)
                rows = [{"id": i, "correct": int(i < x if arm == "prod" else x <= i < x+y),
                         "finish": "stop", "gen_tokens": 1, "output": str(i)} for i in range(n)]
                (dest / f"{bench}.jsonl").write_text("\n".join(map(json.dumps, rows)))
                k = sum(r["correct"] for r in rows)
                (dest / "summary.json").write_text(json.dumps({"bench": {bench: {
                    "n": n, "correct": k, "acc": k/n, "wall_s": 0, "mean_gen_tokens": 1}}}))
                (dest / "DONE").write_text(json.dumps({"run_id": "r1", "arm": arm}))
            return subprocess.run([sys.executable, str(root / "analyze.py"), "--margin-pp", margin],
                                  capture_output=True, text=True, check=True).stdout

    def test_base_degradation_and_noninferiority_fail(self):
        output = self.analyze(0, 20)
        self.assertIn("VERDICT: significant degradation in gsm8k", output)
        self.assertIn("| -20.00 | -27.33 | -14.25 | 0.5 | FAIL | 100 |", output)

    def test_base_improvement_is_not_degradation(self):
        output = self.analyze(20, 0)
        self.assertIn("VERDICT: no significant degradation", output)
        self.assertIn("| +20.00 | +14.25 | +27.33 | 0.5 | PASS | 100 |", output)

    def test_inconclusive_and_strict_boundary(self):
        self.assertIn("| 0.5 | INCONCLUSIVE | 100 |", self.analyze(5, 5))
        # Zero discordances in 100 items cannot establish a 0.5 pp margin (Tango lower bound about -2.6 pp);
        # a Wald interval would collapse to [0, 0] and wrongly PASS.
        self.assertIn("| 0.5 | INCONCLUSIVE | 100 |", self.analyze(0, 0))
        self.assertIn("| 0.5 | PASS | 5000 |", self.analyze(0, 0, n=5000))
        self.assertIn("| 0 | INCONCLUSIVE | 100 |", self.analyze(0, 0, "0"))

    def test_humaneval_one_favourable_zero_unfavourable_is_not_ni(self):
        # Q1 HumanEval prod vs noprune: 1 favourable / 0 unfavourable discordant pair out of 164.
        out = self.analyze(1, 0, n=164, bench="humaneval")
        self.assertIn("| humaneval | noprune | +0.61 | -1.02 | +2.69 | 0.5 | INCONCLUSIVE | 164 |", out)
        self.assertNotIn("| 0.5 | PASS |", out)

    def test_margin_validation(self):
        for margin in ("-1", "nan", "inf"):
            result = subprocess.run([sys.executable, str(HERE / "analyze.py"), "--margin-pp", margin],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("finite and nonnegative", result.stderr)


def make_arm(root, arm, run_id=None, correct=50):
    dest = root / "runs" / arm
    dest.mkdir(parents=True)
    rows = [{"id": i, "correct": int(i < correct), "finish": "stop", "gen_tokens": 1, "output": str(i)} for i in range(100)]
    (dest / "gsm8k.jsonl").write_text("\n".join(map(json.dumps, rows)))
    (dest / "summary.json").write_text(json.dumps({"bench": {"gsm8k": {
        "n": 100, "correct": correct, "acc": correct/100, "wall_s": 0, "mean_gen_tokens": 1, "truncated": 0}}}))
    (dest / "arm.log").write_text(f"[{arm}] start 00:00:00 BS=1 run {run_id} env: X=1\n[{arm}] benchmarks done\n[{arm}] stopped 00:01:00; x\n")
    if run_id: (dest / "DONE").write_text(json.dumps({"run_id": run_id, "arm": arm}))


class CompletenessTests(unittest.TestCase):
    """Only arms with a DONE marker from the current run are aggregated."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=HERE)
        self.root = Path(self.tmp.name) / "bench" / "quality"
        self.root.mkdir(parents=True)
        for name in ("analyze.py", "write_report.py", "report_template.md"):
            shutil.copyfile(HERE / name, self.root / name)
        (self.root / "runs").mkdir()
        (self.root / "runs" / "CURRENT_RUN").write_text("new\n")
        make_arm(self.root, "prod", "new")
        make_arm(self.root, "noprune", "old")      # stale: completed in an earlier run
        make_arm(self.root, "legacy", None)        # failed / interrupted: no marker
        make_arm(self.root, "prod2", "new")

    def tearDown(self):
        self.tmp.cleanup()

    def analyze(self, *argv):
        return subprocess.run([sys.executable, str(self.root / "analyze.py"), *argv], capture_output=True, text=True)

    def test_default_selection_skips_stale_and_unmarked(self):
        r = self.analyze()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("| benchmark | prod | prod2 |", r.stdout)
        self.assertIn("skipping arm noprune: stale", r.stderr)
        self.assertIn("skipping arm legacy: no DONE marker", r.stderr)
        self.assertEqual(self.analyze("--list-arms").stdout.split(), ["prod", "prod2"])
        self.assertEqual(self.analyze("--list-arms", "--run-id", "any").stdout.split(), ["prod", "noprune", "prod2"])

    def test_explicit_incomplete_arm_or_base_is_refused(self):
        for argv in (("prod", "noprune"), ("prod", "legacy"), ("legacy",)):
            r = self.analyze(*argv)
            self.assertEqual(r.returncode, 1, argv)
            self.assertIn("refusing incomplete arms", r.stderr)
        self.assertEqual(self.analyze("--allow-unmarked", "prod", "legacy").returncode, 0)

    def test_write_report_uses_only_complete_arms(self):
        (self.root.parent.parent / "specs").mkdir()
        r = subprocess.run([sys.executable, str(self.root / "write_report.py")], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("| benchmark | prod | prod2 |", r.stdout)
        self.assertIn("- `prod2`: X=1", r.stdout)
        self.assertNotIn("- `noprune`:", r.stdout); self.assertNotIn("- `legacy`:", r.stdout)
        self.assertNotIn("| noprune |", r.stdout); self.assertNotIn("| legacy |", r.stdout)
        (self.root / "runs" / "prod" / "DONE").unlink()
        r = subprocess.run([sys.executable, str(self.root / "write_report.py")], capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("prod is not complete", r.stderr)


@unittest.skipUnless(shutil.which("flock"), "needs flock(1)")
class RunAllTests(unittest.TestCase):
    """run_all.sh with a stub run_arm.sh: failures are summarised and reflected in the exit status."""
    STUB = """#!/bin/bash
arm=$1; out="$(dirname "$0")/runs/$arm"; mkdir -p "$out"; echo "$arm" >> "$(dirname "$0")/ran"
case " ${FAIL_ARMS:-} " in *" $arm "*) exit 3 ;; esac
case " ${NOMARK_ARMS:-} " in *" $arm "*) exit 0 ;; esac
printf '{"run_id": "%s", "arm": "%s"}\\n' "$Q1_RUN_ID" "$arm" > "$out/DONE"
"""

    def run_all(self, **env):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root = Path(tmp)
            shutil.copyfile(HERE / "run_all.sh", root / "run_all.sh")
            (root / "run_arm.sh").write_text(self.STUB); (root / "run_arm.sh").chmod(0o755)
            e = dict(os.environ, GPU_LOCK=str(root / "lock"), **env)
            r = subprocess.run(["bash", str(root / "run_all.sh")], capture_output=True, text=True, env=e)
            ran = (root / "ran").read_text().split() if (root / "ran").exists() else []
            return r, ran

    def test_all_ok(self):
        r, ran = self.run_all()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(ran, ["prod", "noprune", "legacy", "prod2"])
        self.assertIn("all arms done", r.stdout)

    def test_non_prod_failure_continues_but_fails_the_run(self):
        r, ran = self.run_all(FAIL_ARMS="noprune", NOMARK_ARMS="prod2")
        self.assertEqual(r.returncode, 1)
        self.assertEqual(ran, ["prod", "noprune", "legacy", "prod2"])
        self.assertIn("FAILED:   noprune (rc 3)", r.stdout)
        self.assertIn("FAILED:   prod2 (exited 0 but no DONE marker for this run)", r.stdout)
        self.assertIn("complete: prod legacy", r.stdout)

    def test_prod_failure_stops(self):
        r, ran = self.run_all(FAIL_ARMS="prod")
        self.assertEqual(r.returncode, 1)
        self.assertEqual(ran, ["prod"])
        self.assertIn("not run (prod failed): noprune legacy prod2", r.stdout)

    def test_unknown_arm(self):
        r, ran = self.run_all(ARMS="prod bogus")
        self.assertEqual(r.returncode, 1)
        self.assertIn("bogus (unknown arm)", r.stdout)


def bench_copy(root):
    """A runnable copy of run_bench.py (+ he_exec.py) under root/bench/quality with a 4-item GSM8K fixture."""
    q = root / "bench" / "quality"
    (q / "data").mkdir(parents=True)
    for name in ("run_bench.py", "he_exec.py"):
        shutil.copyfile(HERE / name, q / name)
    (q / "data" / "gsm8k.jsonl").write_text("\n".join(
        json.dumps({"id": i, "question": f"question {i}", "answer": "18"}) for i in range(4)) + "\n")
    return q


GOOD = {"choices": [{"message": {"content": "#### 18"}, "finish_reason": "stop"}], "usage": {"completion_tokens": 3}}


class RequestErrorTests(unittest.TestCase):
    """A failed request or item must fail run_bench.py and keep the arm out of the analysis."""
    def run_bench(self, bad):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            q = bench_copy(Path(tmp))
            calls = []

            def post(endpoint, json, timeout):
                calls.append(json["messages"][0]["content"])
                if "question 2" not in json["messages"][0]["content"]:
                    return types.SimpleNamespace(status_code=200, json=lambda: GOOD)
                return bad()

            env = {k: v for k, v in os.environ.items() if k not in ("BS", "CONC", "MOCK", "LIMIT")}
            env.update(RETRY_SLEEP="0")
            with patch.dict(os.environ, env, clear=True), \
                 patch.dict(sys.modules, {"requests": types.SimpleNamespace(post=post)}), \
                 patch.object(sys, "argv", [str(q / "run_bench.py"), "armx", "gsm8k"]), \
                 patch.object(sys, "path", list(sys.path)), contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as cm:
                    runpy.run_path(str(q / "run_bench.py"), run_name="__main__")
            summary = json.loads((q / "runs" / "armx" / "summary.json").read_text())
            rows = [json.loads(l) for l in (q / "runs" / "armx" / "gsm8k.jsonl").read_text().splitlines()]
            return cm.exception.code, summary, rows, sum("question 2" in c for c in calls)

    def check(self, bad, error):
        code, summary, rows, tries = self.run_bench(bad)
        self.assertEqual(code, 1)
        self.assertTrue(summary["incomplete"]); self.assertEqual(summary["errors"], 1)
        self.assertEqual(summary["bench"]["gsm8k"]["errors"], 1)
        self.assertEqual(summary["bench"]["gsm8k"]["correct"], 3)     # what was answered is still written
        self.assertEqual([r["finish"] for r in rows], ["stop", "stop", "error", "stop"])
        self.assertTrue(rows[2]["error"].startswith(error), rows[2]["error"])
        self.assertEqual(tries, 3)                                    # retried, then given up

    def test_exception_after_retries(self):
        def bad(): raise ConnectionError("refused")
        self.check(bad, "ConnectionError")

    def test_non_200(self):
        self.check(lambda: types.SimpleNamespace(status_code=500, json=lambda: GOOD), "BadResponse: HTTP status 500")

    def test_missing_usage_or_choices(self):
        self.check(lambda: types.SimpleNamespace(status_code=200, json=lambda: {"choices": GOOD["choices"]}),
                   "BadResponse: missing usage")
        self.check(lambda: types.SimpleNamespace(status_code=200, json=lambda: {"usage": GOOD["usage"]}),
                   "BadResponse: missing choices")

    def test_malformed_json(self):
        def raise_value(): raise ValueError("not json")
        self.check(lambda: types.SimpleNamespace(status_code=200, json=raise_value), "ValueError")


class ArmErrorAnalysisTests(unittest.TestCase):
    """analyze.py refuses arms whose results record errors, even with a DONE marker for the current run."""
    def test_refuses_errors_despite_done(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root = Path(tmp)
            shutil.copyfile(HERE / "analyze.py", root / "analyze.py")
            (root / "runs").mkdir(); (root / "runs" / "CURRENT_RUN").write_text("r\n")
            for arm in ("prod", "noprune", "legacy", "prod2"):
                make_arm(root, arm, "r")
            s = json.loads((root / "runs" / "noprune" / "summary.json").read_text())
            s.update(incomplete=True, errors=1); s["bench"]["gsm8k"]["errors"] = 1
            (root / "runs" / "noprune" / "summary.json").write_text(json.dumps(s))
            with open(root / "runs" / "legacy" / "gsm8k.jsonl", "a") as f:      # summary clean, items not
                f.write("\n" + json.dumps({"id": 999, "correct": 0, "finish": "error", "output": ""}))
            r = subprocess.run([sys.executable, str(root / "analyze.py")], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("| benchmark | prod | prod2 |", r.stdout)
            self.assertIn("skipping arm noprune: summary.json marks the run incomplete", r.stderr)
            self.assertIn("skipping arm legacy: error items in gsm8k.jsonl", r.stderr)
            for argv in (["prod", "noprune"], ["prod", "legacy"], ["--allow-unmarked", "prod", "noprune"]):
                r = subprocess.run([sys.executable, str(root / "analyze.py"), *argv], capture_output=True, text=True)
                self.assertEqual(r.returncode, 1, argv)
                self.assertIn("refusing incomplete arms", r.stderr)


@unittest.skipUnless(shutil.which("flock") and shutil.which("tee"), "needs flock(1) and tee(1)")
class RunAllBenchFailureTests(unittest.TestCase):
    """run_all.sh -> stub arm (set -euo pipefail, run_bench.py | tee, then DONE, as in run_arm.sh) -> real
    run_bench.py with a fake requests module: a failed request fails the arm, no DONE, run_all exits non-zero."""
    STUB = """#!/bin/bash
set -euo pipefail
arm=$1; here=$(cd "$(dirname "$0")" && pwd); echo "$arm" >> "$here/ran"
case " ${BENCH_FAIL_ARMS:-} " in *" $arm "*) export FAKE_FAIL=1 ;; esac
PYTHONPATH="$here/fakemods" RETRY_SLEEP=0 python3 -B "$here/run_bench.py" "$arm" gsm8k 2>&1 | tee -a "$here/arm.log"
printf '{"run_id": "%s", "arm": "%s"}\\n' "$Q1_RUN_ID" "$arm" > "$here/runs/$arm/DONE"
"""
    FAKE_REQUESTS = """import json as _json, os
class _R:
    status_code = 200
    def json(self):
        return %s
def post(url, json, timeout):
    if os.environ.get("FAKE_FAIL") == "1" and "question 1" in json["messages"][0]["content"]:
        raise ConnectionError("fake connection refused")
    return _R()
""" % repr(GOOD)

    def test_bench_failure_propagates(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            q = bench_copy(Path(tmp))
            shutil.copyfile(HERE / "run_all.sh", q / "run_all.sh")
            (q / "run_arm.sh").write_text(self.STUB); (q / "run_arm.sh").chmod(0o755)
            (q / "fakemods").mkdir(); (q / "fakemods" / "requests.py").write_text(self.FAKE_REQUESTS)
            env = dict(os.environ, GPU_LOCK=str(q / "lock"), ARMS="prod noprune", BENCH_FAIL_ARMS="noprune",
                       PYTHONDONTWRITEBYTECODE="1")
            env.pop("LIMIT", None); env.pop("MOCK", None)
            r = subprocess.run(["bash", str(q / "run_all.sh")], capture_output=True, text=True, env=env)
            self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
            self.assertIn("FAILED:   noprune (rc 1)", r.stdout)
            self.assertIn("complete: prod", r.stdout)
            self.assertTrue((q / "runs" / "prod" / "DONE").exists())
            self.assertFalse(json.loads((q / "runs" / "prod" / "summary.json").read_text())["incomplete"])
            self.assertFalse((q / "runs" / "noprune" / "DONE").exists())
            self.assertTrue(json.loads((q / "runs" / "noprune" / "summary.json").read_text())["incomplete"])


class HeExecCompletionTests(unittest.TestCase):
    """he_exec.run_program passes only when the program ran to its end (completion nonce) with exit 0.
    These fixed snippets run WITHOUT the sandbox (the launcher is patched to `env -C <dir>`) so the
    completion protocol is tested even where bwrap/cgroups are missing; `he_exec.py --self-test` repeats
    the same cases inside the real sandbox."""
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(HERE))
        import he_exec
        cls.hx = he_exec
        cls.patches = [patch.object(he_exec, "_cgroup_prefix", lambda: []),
                       patch.object(he_exec, "_limit_prefix", lambda: []),
                       patch.object(he_exec, "_sandbox_argv", lambda d: ["/usr/bin/env", "-C", d])]
        for p in cls.patches: p.start()

    @classmethod
    def tearDownClass(cls):
        for p in cls.patches: p.stop()
        sys.path.remove(str(HERE))

    PROMPT = 'def add(a, b):\n    """Return a + b."""\n'
    TEST = "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(-1, 1) == 0\n"

    def run_he(self, body):
        return self.hx.run_program(self.hx.build_program(self.PROMPT, "```python\n" + body + "```", self.TEST, "add"))

    def test_normal_pass_and_fail(self):
        self.assertEqual(self.run_he("def add(a, b):\n    return a + b\n"), ("pass", ""))
        self.assertEqual(self.run_he("def add(a, b):\n    return a - b\n"), ("fail", "AssertionError"))

    def test_sys_exit_at_module_level(self):
        for exit_call in ("import sys\nsys.exit(0)", "sys_exit = SystemExit\nraise sys_exit(0)", "exit()", "quit()"):
            with self.subTest(exit_call=exit_call):
                got = self.run_he("def add(a, b):\n    return a - b\n" + exit_call + "\n")
                self.assertEqual(got, ("fail", self.hx.EARLY_EXIT))

    def test_sys_exit_inside_the_function(self):
        self.assertEqual(self.run_he("def add(a, b):\n    import sys\n    sys.exit(0)\n"), ("fail", self.hx.EARLY_EXIT))
        self.assertEqual(self.run_he("def add(a, b):\n    raise SystemExit\n"), ("fail", self.hx.EARLY_EXIT))

    def test_os_exit(self):
        self.assertEqual(self.run_he("import os\ndef add(a, b):\n    os._exit(0)\n"), ("fail", self.hx.EARLY_EXIT))
        self.assertEqual(self.hx.run_program("import os\nos._exit(0)\n"), ("fail", self.hx.EARLY_EXIT))

    def test_stdout_closed_or_redirected(self):
        # closing stdout/stderr and exiting early is not a pass
        self.assertEqual(self.run_he("import os, sys\nos.close(1)\nos.close(2)\ndef add(a, b):\n    sys.exit(0)\n"),
                         ("fail", self.hx.EARLY_EXIT))
        # redirecting stdout does not affect a correct solution (the proof is on a private pipe)
        self.assertEqual(self.run_he("import io, sys\nsys.stdout = io.StringIO()\ndef add(a, b):\n    return a + b\n"),
                         ("pass", ""))
        self.assertEqual(self.run_he("import io, sys\nsys.stdout = io.StringIO()\ndef add(a, b):\n    sys.exit(0)\n"),
                         ("fail", self.hx.EARLY_EXIT))

    def test_closing_or_forging_the_completion_pipe(self):
        self.assertEqual(self.hx.run_program("import os\nos.closerange(3, 1024)\n")[0], "fail")
        forge = "import os\nfor fd in range(3, 64):\n    try: os.write(fd, b'0' * 32)\n    except OSError: pass\nos._exit(0)\n"
        self.assertEqual(self.hx.run_program(forge), ("fail", self.hx.EARLY_EXIT))


class ClientConcurrencyTests(unittest.TestCase):
    @unittest.skipUnless((HERE / "data" / "gsm8k.jsonl").exists(),
                         "needs bench data: run prep_data.py first (datasets are not in the repository)")
    def test_default_and_explicit_caps_in_benchmarks_and_sanity(self):
        for settings, expected in (({}, 1), ({"BS": "3", "CONC": "8"}, 3),
                                   ({"BS": "4", "CONC": "2"}, 2)):
            with self.subTest(settings=settings), tempfile.TemporaryDirectory(dir=HERE) as tmp:
                lock = threading.Lock()
                active = {"gsm8k": 0, "sanity": 0}
                peak = dict(active)

                def post(endpoint, json, timeout):
                    name = "gsm8k" if "#### <number>" in json["messages"][0]["content"] else "sanity"
                    with lock:
                        active[name] += 1
                        peak[name] = max(peak[name], active[name])
                    time.sleep(.02)
                    with lock:
                        active[name] -= 1
                    return types.SimpleNamespace(status_code=200, json=lambda: {
                        "choices": [{"message": {"content": "#### 18"}, "finish_reason": "stop"}],
                        "usage": {"completion_tokens": 3}})

                env = {k: v for k, v in os.environ.items() if k not in ("BS", "CONC", "MOCK")}
                env.update(settings, LIMIT="3")
                with patch.dict(os.environ, env, clear=True), \
                     patch.dict(sys.modules, {"requests": types.SimpleNamespace(post=post)}), \
                     patch.object(sys, "argv", [str(HERE / "run_bench.py"), tmp, "gsm8k,sanity"]), \
                     patch.object(sys, "path", list(sys.path)), contextlib.redirect_stdout(io.StringIO()):
                    runpy.run_path(str(HERE / "run_bench.py"), run_name="__main__")
                self.assertEqual(peak, {"gsm8k": expected, "sanity": expected})
                summary = json.loads((Path(tmp) / "summary.json").read_text())
                self.assertEqual(summary["bs"], int(settings.get("BS", 1)))
                self.assertEqual(summary["conc"], expected)


if __name__ == "__main__":
    unittest.main()
