"""Epoch-aligned resume: a long run may be chopped into shorter GPU stretches
(``--stop-after-epochs``), so a resumed process must continue at the same step, the same epoch (hence the
same data order), and must not forget which checkpoint was best."""

import glob
import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mtptrain.train import (  # noqa: E402
    load_checkpoint,
    lr_at,
    save_checkpoint,
    selection_score,
)


class _Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.lin(x)

    def trainable_parameters(self):
        return list(self.parameters())


def test_checkpoint_roundtrip_carries_epoch_best_history():
    model = _Toy()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.randn(2, 4)).sum().backward()
    opt.step()
    best = {"score": 2.5, "step": 200, "summary": {"accept_len@7": 2.5},
            "criterion": "renewal@7"}
    history = [{"step": 100, "eval": {"accept_len@7": 3.1}}]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "latest.pt")
        save_checkpoint(path, model, opt, 250, {"args": {}}, epoch=1,
                        best=best, history=history)
        model2, opt2 = _Toy(), None
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        st = load_checkpoint(path, model2, opt2)
    assert st["step"] == 250 and st["epoch"] == 1
    assert st["best"]["step"] == 200 and st["best"]["score"] == 2.5
    assert st["history"] == history
    for a, b in zip(model.parameters(), model2.parameters()):
        assert torch.equal(a, b)


def test_old_checkpoint_without_epoch_still_loads():
    """A latest.pt written before the epoch field existed must still resume."""
    model = _Toy()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "old.pt")
        torch.save(
            {"step": 7, "model": model.state_dict(),
             "optimizer": opt.state_dict(), "meta": {},
             "torch_rng": torch.get_rng_state(),
             "py_rng": __import__("random").getstate()},
            path,
        )
        st = load_checkpoint(path, _Toy(), torch.optim.AdamW(_Toy().parameters()))
    assert {k: st[k] for k in ("step", "epoch", "best", "history", "epoch_batches",
                               "accum", "format")} == {
        "step": 7, "epoch": 0, "best": None, "history": [], "epoch_batches": 0,
        "accum": 0, "format": 1}


def test_selection_metric_renewal_at_k():
    s = {"accept_len@3": 2.0, "accept_len@7": 3.0, "renewal_mean": 2.5,
         "agreement@1": 0.8}
    name, score = selection_score(s, "renewal@7")
    assert name == "renewal@7" and abs(score - (3.0 + 1e-4 * 0.8)) < 1e-9
    assert selection_score(s, "renewal")[1] > 2.5  # unchanged behaviour


def test_lr_schedule_is_a_pure_function_of_step():
    """A stretch boundary must not perturb the cosine: lr_at() only reads the
    global step, so resuming at step N gives the same lr an uninterrupted run
    would have used."""
    kw = dict(base_lr=5e-6, warmup=100, total=3000, min_ratio=0.1)
    assert lr_at(250, **kw) == lr_at(250, **kw)
    assert lr_at(0, **kw) < lr_at(100, **kw)
    assert abs(lr_at(2999, **kw) - 5e-6 * 0.1) < 2e-9


# ------------------------------------------- --resume / --out guards in train
def _tiny_args(out, *extra):
    from mtptrain.train import build_parser

    return build_parser().parse_args(
        ["--tiny", "--synthetic", "4", "--steps", "2", "--batch-size", "2",
         "--max-len", "24", "--warmup", "1", "--log-every", "1",
         "--out", out, "--device", "cpu", *extra]
    )


def _expect_exit(fn, word):
    try:
        fn()
    except SystemExit as exc:
        assert word in str(exc), str(exc)
    else:
        raise AssertionError(f"expected a SystemExit mentioning {word!r}")


def test_missing_resume_checkpoint_is_an_error():
    from mtptrain.train import train

    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "run")
        _expect_exit(lambda: train(_tiny_args(out, "--resume", os.path.join(d, "nope.pt"))),
                     "no such checkpoint")
        assert not os.path.exists(out)  # nothing was started


def test_existing_outputs_are_not_overwritten_by_accident():
    from mtptrain.train import train

    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "run")
        train(_tiny_args(out))
        latest = os.path.join(out, "latest.pt")
        final = os.path.join(out, "final-mtp.safetensors")
        stamp = (os.path.getmtime(latest), open(final, "rb").read())
        # a second fresh run into the same --out is refused, files untouched
        _expect_exit(lambda: train(_tiny_args(out)), "already holds a run")
        assert (os.path.getmtime(latest), open(final, "rb").read()) == stamp
        # resuming from a checkpoint in another directory is refused too
        other = os.path.join(d, "other")
        train(_tiny_args(other))
        _expect_exit(lambda: train(_tiny_args(out, "--resume",
                                              os.path.join(other, "latest.pt"))),
                     "not inside --out")
        # continuing this run (resume from its own latest.pt) is allowed
        stats = train(_tiny_args(out, "--steps", "3", "--resume", latest))
        assert stats["steps"] == 3
        # and --overwrite starts afresh, after moving the old run aside
        stats = train(_tiny_args(out, "--overwrite"))
        assert stats["steps"] == 2
        assert any(n.startswith(".previous-") for n in os.listdir(out))
        # a snapshot alone counts as an earlier run as well
        snap = os.path.join(d, "snap")
        os.makedirs(snap)
        open(os.path.join(snap, "mtp-step100.safetensors"), "wb").close()
        _expect_exit(lambda: train(_tiny_args(snap)), "mtp-step100")


# ------------------------------------ interrupted run == uninterrupted run
# 4 synthetic samples x 2 passes / batch 2 = 4 micro-batches per epoch, and
# --grad-accum 3 makes optimizer steps straddle epoch boundaries, so both the
# mid-epoch position and a pending gradient accumulation are exercised.
_RUN = ["--synthetic-repeat", "2", "--tokens-per-step", "0", "--batch-size", "2",
        "--grad-accum", "3", "--epochs", "10", "--steps", "6",
        "--eval-every", "2", "--eval-batches", "1", "--eval-starts", "2"]


def _comparable(history):
    return [{k: v for k, v in h.items() if k != "tok_s"} for h in history]


def _final_state(out):
    return torch.load(os.path.join(out, "latest.pt"), map_location="cpu",
                      weights_only=True)


def _uninterrupted(d):
    from mtptrain.train import train

    out = os.path.join(d, "straight")
    stats = train(_tiny_args(out, *_RUN))
    return out, stats


def _interrupted(d, name, crash, ckpt_every, *extra):
    from mtptrain.train import SimulatedCrash, train

    out = os.path.join(d, name)
    run = [*_RUN, "--ckpt-every", str(ckpt_every), *extra]
    try:
        train(_tiny_args(out, *run), _crash_after_step=crash)
    except SimulatedCrash:
        pass
    else:
        raise AssertionError("the crash hook did not fire")
    return out, run


def _assert_same_run(ref_out, ref_stats, out, stats):
    assert stats["steps"] == ref_stats["steps"]
    assert (stats["epoch"], stats["epoch_batches"]) == (
        ref_stats["epoch"], ref_stats["epoch_batches"])
    assert _comparable(stats["history"]) == _comparable(ref_stats["history"])
    assert {k: stats["best"][k] for k in ("score", "step", "criterion")} == {
        k: ref_stats["best"][k] for k in ("score", "step", "criterion")}
    a, b = _final_state(ref_out), _final_state(out)
    assert (a["epoch"], a["epoch_batches"], a["step"]) == (
        b["epoch"], b["epoch_batches"], b["step"])
    for k in a["model"]:
        assert torch.equal(a["model"][k], b["model"][k]), k
    hist = json.load(open(os.path.join(out, "history.json")))
    assert _comparable(hist) == _comparable(ref_stats["history"])


def test_resume_after_a_crash_matches_an_uninterrupted_run():
    from mtptrain.train import train

    with tempfile.TemporaryDirectory() as d:
        ref_out, ref = _uninterrupted(d)
        # (crash after step, --ckpt-every): mid-epoch position; an epoch-end
        # checkpoint holding a pending accumulation; evals after the checkpoint
        for i, (crash, every) in enumerate(((3, 1), (3, 0), (5, 3))):
            out, run = _interrupted(d, f"crash{i}", crash, every)
            st = _final_state(out)
            assert st["step"] <= crash
            stats = train(_tiny_args(out, *run, "--resume",
                                     os.path.join(out, "latest.pt")))
            _assert_same_run(ref_out, ref, out, stats)
        # the (3, 0) case really resumed with a half-full accumulation
        assert _final_state(ref_out)["format"] == 2


def test_periodic_checkpoint_carries_epoch_position_best_and_history():
    from mtptrain.train import load_checkpoint, build_model, make_optimizer, resolve_args

    with tempfile.TemporaryDirectory() as d:
        out, run = _interrupted(d, "periodic", 3, 1)
        args = resolve_args(_tiny_args(out, *run))
        model = build_model(args)
        st = load_checkpoint(os.path.join(out, "latest.pt"), model,
                             make_optimizer(args, model.trainable_parameters()))
        # step 3 = micro-batch 9 = epoch index 2, first micro-batch
        assert (st["step"], st["epoch"], st["epoch_batches"], st["accum"]) == (3, 2, 1, 0)
        assert st["best"] and st["best"]["step"] == 2
        assert [h["step"] for h in st["history"] if "eval" in h] == [2]
        assert st["tokens_seen"] > 0


def test_resume_never_replaces_a_better_best_written_after_the_checkpoint():
    from safetensors.torch import load_file, save_file

    from mtptrain.train import train

    with tempfile.TemporaryDirectory() as d:
        out, run = _interrupted(d, "best", 5, 3)
        path = os.path.join(out, "best-mtp.safetensors")
        # pretend the eval after latest.pt produced an excellent head
        run_id = _final_state(out)["run_id"]
        from mtptrain.train import weights_header

        head = weights_header(path)
        better = dict(json.loads(head["mtp_best"]), score=1e9, step=4, run_id=run_id)
        meta = dict(head, mtp_best=json.dumps(better), mtp_run_id=run_id)
        save_file(load_file(path), path, metadata=meta)  # same tensors, new header
        before = open(path, "rb").read()
        stats = train(_tiny_args(out, *run, "--resume", os.path.join(out, "latest.pt")))
        assert stats["best"]["score"] == 1e9 and stats["best"]["step"] == 4
        assert open(path, "rb").read() == before  # never overwritten by a worse eval
        assert _final_state(out)["best"]["score"] == 1e9


def test_resume_does_not_believe_a_best_header_over_other_tensors():
    """A best file whose tensors are not the ones its header was written for
    (edited, or swapped under the same header) is moved aside, not adopted."""
    from safetensors.torch import load_file, save_file

    from mtptrain.train import train, weights_header

    with tempfile.TemporaryDirectory() as d:
        out, run = _interrupted(d, "best", 5, 3)
        path = os.path.join(out, "best-mtp.safetensors")
        head = weights_header(path)
        better = dict(json.loads(head["mtp_best"]), score=1e9, step=4)
        tensors = load_file(path)
        name = sorted(tensors)[0]
        tensors[name] = tensors[name] + 1
        meta = dict(head, mtp_best=json.dumps(better))
        save_file(tensors, path, metadata=meta)
        stats = train(_tiny_args(out, *run, "--resume", os.path.join(out, "latest.pt")))
        assert stats["best"]["score"] != 1e9
        assert glob.glob(os.path.join(out, ".previous-*", "best-mtp.safetensors"))


def test_stretches_with_stop_after_epochs_match_an_uninterrupted_run():
    from mtptrain.train import train

    with tempfile.TemporaryDirectory() as d:
        ref_out, ref = _uninterrupted(d)
        out = os.path.join(d, "stretch")
        latest = os.path.join(out, "latest.pt")
        train(_tiny_args(out, *_RUN, "--stop-after-epochs", "2"))
        assert (_final_state(out)["epoch"], _final_state(out)["epoch_batches"]) == (2, 0)
        assert _final_state(out)["accum"] == 2  # micro-batches 7-8 are pending
        stats = train(_tiny_args(out, *_RUN, "--resume", latest))
        _assert_same_run(ref_out, ref, out, stats)
