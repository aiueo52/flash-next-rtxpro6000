"""--resume / --out / --overwrite: every combination either refuses clearly or
ends with an --out whose pointers all agree (tiny synthetic CPU config).

"Agree" means: ``latest.pt``'s best, ``best.json`` and the
``best-mtp.safetensors`` header name the same run, step and score; that file
exists; ``history.json`` equals ``latest.pt``'s history and holds the best
step's eval; and every weights file left in --out belongs to that run.
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
import sys
import tempfile

import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mtptrain.train import (  # noqa: E402
    SimulatedCrash,
    best_from_snapshot,
    build_parser,
    train,
    weights_header,
)

RUN = ["--tiny", "--synthetic", "4", "--synthetic-repeat", "2", "--tokens-per-step", "0",
       "--batch-size", "2", "--max-len", "24", "--warmup", "1", "--log-every", "1",
       "--steps", "4", "--epochs", "10", "--eval-every", "2", "--eval-batches", "1",
       "--eval-starts", "2", "--ckpt-every", "1", "--snapshot-every-eval",
       "--device", "cpu"]


def _args(out, *extra):
    return build_parser().parse_args([*RUN, "--out", out, *extra])


def _listing(root):
    out = {}
    for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        path = os.path.join(root, name)
        out[name] = os.path.getmtime(path) if os.path.isfile(path) else "dir"
    return out


def assert_consistent(out):
    ck = torch.load(os.path.join(out, "latest.pt"), map_location="cpu", weights_only=True)
    run_id = ck["run_id"]
    assert run_id
    history = json.load(open(os.path.join(out, "history.json")))
    assert history == ck["history"]
    best = ck["best"]
    assert best["run_id"] == run_id
    best_file = os.path.join(out, "best-mtp.safetensors")
    if best["step"] >= 0:
        head = weights_header(best_file)
        assert head.get("mtp_run_id") == run_id
        disk = best_from_snapshot(best_file)
        on_json = json.load(open(os.path.join(out, "best.json")))
        for d in (disk, on_json):
            assert (d["run_id"], d["step"], d["score"]) == (run_id, best["step"], best["score"])
        assert best["step"] in [h["step"] for h in history if "eval" in h]
    else:
        assert not os.path.exists(best_file) and not os.path.exists(
            os.path.join(out, "best.json"))
    for name in os.listdir(out):
        if name.endswith(".safetensors"):
            assert weights_header(os.path.join(out, name)).get("mtp_run_id") == run_id, name


def _foreign_best(out):
    """Stamp a very good score into the best file of whatever run is in out."""
    from safetensors.torch import load_file, save_file

    path = os.path.join(out, "best-mtp.safetensors")
    head = weights_header(path)
    best = dict(json.loads(head["mtp_best"]), score=1e9)
    head["mtp_best"] = json.dumps(best)
    save_file(load_file(path), path, metadata=head)


def _prepare(d, out_state):
    """-> (out, this_latest) for an --out that is empty / holds run R / holds R's
    checkpoint next to another run's weights and best."""
    out = os.path.join(d, "out")
    if out_state == "empty":
        os.makedirs(out)
        return out, os.path.join(out, "latest.pt")
    if out_state == "this":
        train(_args(out))
        return out, os.path.join(out, "latest.pt")
    # "another": another run's best/snapshots, but R's checkpoint and history
    train(_args(out))
    _foreign_best(out)
    r = os.path.join(d, "r")
    try:
        train(_args(r), _crash_after_step=1)  # before R's first eval
    except SimulatedCrash:
        pass
    for name in ("latest.pt", "history.json"):
        shutil.copy2(os.path.join(r, name), os.path.join(out, name))
    return out, os.path.join(out, "latest.pt")


CASES = list(itertools.product(("none", "same", "other", "missing"),
                               ("empty", "this", "another"), (False, True)))


def _expected(resume, out_state, overwrite):
    if resume in ("other", "missing"):
        return "refuse"
    if resume == "same":
        return "refuse" if overwrite or out_state == "empty" else "ok"
    return "ok" if out_state == "empty" or overwrite else "refuse"


@pytest.mark.parametrize("resume,out_state,overwrite", CASES)
def test_resume_out_overwrite_matrix(resume, out_state, overwrite):
    with tempfile.TemporaryDirectory() as d:
        out, latest = _prepare(d, out_state)
        extra = ["--overwrite"] if overwrite else []
        if resume == "same":
            extra += ["--resume", latest]
        elif resume == "missing":
            extra += ["--resume", os.path.join(d, "nowhere", "latest.pt")]
        elif resume == "other":
            other = os.path.join(d, "other")
            train(_args(other))
            extra += ["--resume", os.path.join(other, "latest.pt")]
        before = _listing(out)
        expected = _expected(resume, out_state, overwrite)
        if expected == "refuse":
            with pytest.raises(SystemExit):
                train(_args(out, *extra))
            assert _listing(out) == before  # nothing touched
            return
        stats = train(_args(out, *extra))
        assert stats["steps"] == 4
        assert_consistent(out)
        assert stats["best"]["score"] != 1e9  # never another run's best
        if out_state != "empty" and resume == "none":
            kept = [n for n in os.listdir(out) if n.startswith(".previous-")]
            assert len(kept) == 1 and "best-mtp.safetensors" in os.listdir(
                os.path.join(out, kept[0]))


def test_overwrite_then_resume_never_adopts_the_old_runs_best():
    """The reviewer's reproduction: run A, a fresh --overwrite run B that is
    interrupted before its first eval, then B resumed."""
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "out")
        a = train(_args(out))
        _foreign_best(out)  # A's best now looks unbeatable
        try:
            train(_args(out, "--overwrite"), _crash_after_step=1)
        except SimulatedCrash:
            pass
        assert not os.path.exists(os.path.join(out, "best-mtp.safetensors"))
        prev = [n for n in os.listdir(out) if n.startswith(".previous-")]
        assert len(prev) == 1
        moved = os.path.join(out, prev[0])
        assert weights_header(os.path.join(moved, "best-mtp.safetensors"))[
            "mtp_run_id"] == a["run_id"]  # A's files kept, not deleted
        b = train(_args(out, "--resume", os.path.join(out, "latest.pt")))
        assert b["run_id"] != a["run_id"] and b["best"]["score"] != 1e9
        assert_consistent(out)


def test_resume_into_another_out_is_refused():
    with tempfile.TemporaryDirectory() as d:
        run1, run2 = os.path.join(d, "run1"), os.path.join(d, "run2")
        train(_args(run1))
        with pytest.raises(SystemExit, match="not inside --out"):
            train(_args(run2, "--resume", os.path.join(run1, "latest.pt")))
        assert not os.path.exists(run2)
        # the documented way to branch: copy the directory, resume the copy
        shutil.copytree(run1, run2)
        stats = train(_args(run2, "--steps", "6", "--resume",
                            os.path.join(run2, "latest.pt")))
        assert stats["steps"] == 6
        assert_consistent(run2)


# ------------------------------------------ other pointer files of the pipeline
def test_eval_set_is_a_strict_pointer_into_its_dump():
    from extract.build_eval_set import build_eval_set
    from mtptrain.data import load_fixed_eval
    from tests.test_joined_docs import _fixture

    with tempfile.TemporaryDirectory() as d:
        dump = os.path.join(d, "dump")
        _fixture(dump)
        rel_cwd = os.getcwd()
        try:
            os.chdir(d)  # a relative dump path is stored absolute
            build_eval_set("dump", None, rows=10_000, holdout=1.0)
        finally:
            os.chdir(rel_cwd)
        path = os.path.join(dump, "eval_fixed.json")
        frozen = json.load(open(path))
        assert os.path.realpath(frozen["dump_dir"]) == os.path.realpath(dump)
        assert len(load_fixed_eval(path)) == 4
        # the dump moved (with its eval set): resolved next to the eval set
        moved = os.path.join(d, "moved")
        os.rename(dump, moved)
        assert len(load_fixed_eval(os.path.join(moved, "eval_fixed.json"))) == 4
        # a lost piece is an error, not a silently smaller eval set
        os.remove(os.path.join(moved, "dump-7-00000003.safetensors"))
        with pytest.raises(SystemExit, match="rebuild the eval set"):
            load_fixed_eval(os.path.join(moved, "eval_fixed.json"))
