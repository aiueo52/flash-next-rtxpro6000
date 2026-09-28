from pathlib import Path

import pytest

from ngramsim import cli

REPO = cli._repository_root()


def test_out_is_required(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "--conversations", "x", "--tokenizer", "t"])
    assert exc.value.code == 2
    assert "--out" in capsys.readouterr().err


@pytest.mark.parametrize("inside", ["results", "sim/results", "."])
def test_output_inside_repository_is_refused(inside, capsys, tmp_path):
    target = REPO / inside
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "--conversations", str(tmp_path), "--tokenizer", "t", "--out", str(target)])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "refusing to write simulator output inside the repository" in err
    assert "--allow-in-repo" in err


def test_output_outside_repository_and_explicit_opt_in(tmp_path):
    if tmp_path.resolve().is_relative_to(REPO):
        pytest.skip("pytest tmp_path is inside the repository")
    assert cli._output_dir(str(tmp_path / "out")) == (tmp_path / "out").resolve()
    assert cli._output_dir(str(REPO / "results"), allow_in_repo=True) == (REPO / "results").resolve()
    with pytest.raises(ValueError):
        cli._output_dir(str(REPO / "results"))


def test_report_regeneration_inside_repository_is_refused(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["report", str(REPO / "results" / "raw_stats.json")])
    assert exc.value.code == 2
    assert "inside the repository" in capsys.readouterr().err


def test_repository_root_is_git_toplevel():
    assert (REPO / ".git").exists() or REPO == Path(cli.__file__).resolve().parent.parent
