"""CPU tests: the token-map scripts refuse to write inside the repository unless --allow-in-repo."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name", ["collect_selfgen", "build_hot_vocab"])
def test_in_repo_output_is_refused(name):
    mod = load(name)
    assert mod.repository_root() == REPO
    for inside in (REPO / "gen.jsonl", REPO / "tokenmaps", REPO / "results" / "x" / "hot.pt"):
        with pytest.raises(SystemExit) as e:
            mod.outside_repo(inside, False, "output")
        assert "--allow-in-repo" in str(e.value)
        assert mod.outside_repo(inside, True, "output") == inside.resolve()


@pytest.mark.parametrize("name", ["collect_selfgen", "build_hot_vocab"])
def test_outside_repo_output_is_allowed(name, tmp_path):
    mod = load(name)
    assert mod.outside_repo(tmp_path / "gen.jsonl", False, "output") == (tmp_path / "gen.jsonl").resolve()


def test_collect_selfgen_cli_refuses_before_contacting_the_server(tmp_path):
    prompts = tmp_path / "p.txt"
    prompts.write_text("hello\n")
    r = subprocess.run([sys.executable, "-B", str(HERE / "collect_selfgen.py"), "--endpoint", "http://127.0.0.1:9/v1",
                        "--prompts", str(prompts), "--out", str(REPO / "gen.jsonl")],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "--allow-in-repo" in r.stderr
    assert not (REPO / "gen.jsonl").exists()
