"""Explicit inputs are used or refused -- never replaced, defaulted or ignored.

Three classes of silent failure, checked across the pipeline:

(a) an explicit path that does not exist (or is the wrong type) is silently
    replaced by a default or ignored;
(b) an empty or partial input (empty manifest, glob matching nothing, filters
    removing everything) is treated as "use everything" or produces an empty
    output that looks like success;
(c) settings changed between runs of a resumable tool go undetected (train
    resume, client resume, eval --append, eval set re-runs).

CPU only, tiny configs, temporary directories.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types

import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from mtptrain.config import MTPConfig  # noqa: E402
from mtptrain.data import DumpDataset  # noqa: E402
from mtptrain.fileio import check_run_settings, read_jsonl  # noqa: E402
from mtptrain.train import build_parser, train  # noqa: E402
from tests.test_renewal import _fake_selfgen_dump  # noqa: E402

CFG = MTPConfig.tiny()


@pytest.fixture
def tmp():
    d = tempfile.mkdtemp(prefix="mtpvalid-")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _dump(root, docs=12, topk=0):
    os.makedirs(root, exist_ok=True)
    return _fake_selfgen_dump(root, docs=docs, length=24, prompt_tokens=8,
                              vocab=CFG.vocab_size, hc_dim=CFG.hc_size, topk=topk,
                              consistent=True)


def _write_jsonl(path, rows):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _exit(fn, *words):
    with pytest.raises(SystemExit) as ei:
        fn()
    msg = str(ei.value)
    for w in words:
        assert w in msg, msg
    return msg


# ======================================================================= train
DUMP_RUN = ["--tiny", "--steps", "2", "--tokens-per-step", "0", "--batch-size", "2",
            "--max-len", "24", "--warmup", "1", "--log-every", "1", "--holdout", "0.5",
            "--eval-every", "1", "--eval-batches", "1", "--eval-starts", "2",
            "--device", "cpu"]
SYN_RUN = ["--tiny", "--synthetic", "4", "--synthetic-repeat", "2", "--tokens-per-step",
           "0", "--batch-size", "2", "--max-len", "24", "--warmup", "1", "--log-every",
           "1", "--eval-every", "2", "--eval-batches", "1", "--eval-starts", "2",
           "--ckpt-every", "1", "--device", "cpu"]


def _args(*argv):
    return build_parser().parse_args(list(argv))


@pytest.mark.parametrize("flag,kind", [
    ("--eval-set", "file"), ("--dump-dir", "dir"), ("--init", "file"),
    ("--token-map", "file"), ("--renewal-dump-dir", "dir"),
    ("--renewal-manifest", "file"),
])
def test_train_refuses_missing_explicit_path_before_touching_out(tmp, flag, kind):
    dump = os.path.join(tmp, "dump")
    _dump(dump)
    out = os.path.join(tmp, "out")
    extra = [flag, os.path.join(tmp, "nope")]
    if flag.startswith(("--renewal", "--token")):
        extra += ["--eval-renewal"]
    if flag == "--init":
        extra = ["--model-dir", tmp] + extra  # --init needs a real model; refused first
    argv = [*DUMP_RUN, "--dump-dir", dump, "--out", out, *extra]
    if flag == "--dump-dir":
        argv = [*DUMP_RUN, "--out", out, *extra]
    _exit(lambda: train(_args(*argv)), flag, "no such")
    assert not os.path.exists(out)


def test_train_eval_set_is_never_replaced_by_the_dump_split(tmp):
    """Flag 2: a missing --eval-set used to fall back to the dump's split."""
    dump = os.path.join(tmp, "dump")
    _dump(dump)
    out = os.path.join(tmp, "out")
    os.makedirs(out)
    open(os.path.join(out, "latest.pt"), "w").close()  # an earlier run
    _exit(lambda: train(_args(*DUMP_RUN, "--dump-dir", dump, "--out", out,
                              "--overwrite", "--eval-set",
                              os.path.join(dump, "eval_fixed.json"))),
          "--eval-set", "no such file")
    assert os.listdir(out) == ["latest.pt"]  # --overwrite moved nothing


@pytest.mark.parametrize("extra,word", [
    (["--synthetic", "4", "--eval-set", "x.json"], "--eval-set is not used with --synthetic"),
    (["--synthetic", "4", "--dump-dir", "."], "--dump-dir is not read with --synthetic"),
    (["--synthetic", "4", "--exclude-buckets", "a"], "--exclude-buckets is not used"),
    (["--synthetic", "4", "--renewal-dump-dir", "."], "only used with --eval-renewal"),
    (["--synthetic", "4", "--eval-every", "0", "--snapshot-every-eval", "--out", "o"],
     "needs --eval-every"),
    (["--synthetic", "4", "--eval-every", "0", "--select-metric", "agreement"],
     "needs --eval-every"),
    (["--synthetic", "4", "--ckpt-every", "5"], "--ckpt-every needs --out"),
    (["--synthetic", "4", "--init", __file__], "--init is not applied with --tiny"),
    (["--synthetic", "4", "--reset-best"], "--reset-best only applies to --resume"),
    (["--synthetic", "4", "--steps", "0"], "--steps must be >= 1"),
    (["--synthetic", "4", "--holdout", "1.5"], "--holdout must be in [0, 1]"),
    ([], "--dump-dir is required"),
])
def test_train_refuses_options_that_would_be_ignored(tmp, extra, word):
    argv = ["--tiny", "--device", "cpu", *extra]
    _exit(lambda: train(_args(*argv)), word)


@pytest.mark.parametrize("metric,extra,word", [
    ("renewal", [], "needs --eval-renewal"),
    ("7", [], "not in --chain-ks"),
    ("combined", ["--chain-ks", "3"], "accept@3 and accept@15"),
    ("bogus", [], "use renewal"),
])
def test_train_refuses_unknown_select_metric(tmp, metric, extra, word):
    argv = [*SYN_RUN, "--steps", "1", "--out", os.path.join(tmp, "o"),
            "--select-metric", metric, *extra]
    _exit(lambda: train(_args(*argv)), word)


def test_dump_dataset_empty_manifest_selects_nothing(tmp):
    """Flag 1: an empty manifest meant "every file" before."""
    _dump(tmp)
    everything = DumpDataset(tmp, "train", holdout_frac=0.0).documents()
    assert len(everything) == 12
    open(os.path.join(tmp, "manifest.jsonl"), "w").close()
    assert DumpDataset(tmp, "train", holdout_frac=0.0).documents() == []
    # no manifest at all: a dump made without the client -- every document
    os.remove(os.path.join(tmp, "manifest.jsonl"))
    assert len(DumpDataset(tmp, "train", holdout_frac=0.0).documents()) == 12
    _exit(lambda: DumpDataset(tmp, manifest=os.path.join(tmp, "gone.jsonl")),
          "no such file")
    _exit(lambda: DumpDataset(tmp, exclude_buckets=["code"]), "need a manifest")
    _exit(lambda: DumpDataset(os.path.join(tmp, "nope")), "no such directory")


def test_dump_dataset_rows_not_on_disk_select_nothing(tmp):
    _dump(tmp)
    _write_jsonl(os.path.join(tmp, "manifest.jsonl"),
                 [{"doc_hash": "0" * 16, "split_key": "x", "bucket": "code"}])
    assert DumpDataset(tmp, "train", holdout_frac=0.0).documents() == []
    _exit(lambda: DumpDataset(tmp, include_buckets=["cod"]), "unknown bucket", "cod")


@pytest.mark.parametrize("case,word", [
    ("empty-manifest", "no eval documents"),
    ("holdout-1", "no training documents"),
    ("exclude-all", "no training documents"),
])
def test_train_refuses_zero_documents(tmp, case, word):
    dump = os.path.join(tmp, "dump")
    _dump(dump)
    extra = []
    if case == "empty-manifest":
        open(os.path.join(dump, "manifest.jsonl"), "w").close()
    elif case == "holdout-1":
        extra = ["--holdout", "1.0"]
    else:
        extra = ["--exclude-buckets", "code,prose"]
    out = os.path.join(tmp, "out")
    _exit(lambda: train(_args(*DUMP_RUN, "--dump-dir", dump, "--out", out, *extra)), word)
    assert not os.path.exists(out)


def test_train_on_a_dump_and_its_eval_set(tmp):
    from extract.build_eval_set import build_eval_set

    dump = os.path.join(tmp, "dump")
    _dump(dump)
    ev = DumpDataset(dump, "eval", holdout_frac=0.5).documents()
    tr = DumpDataset(dump, "train", holdout_frac=0.5).documents()
    assert ev and tr  # the fixture splits both ways
    out = os.path.join(tmp, "out")
    stats = train(_args(*DUMP_RUN, "--dump-dir", dump, "--out", out))
    assert stats["steps"] == 2 and stats["best"]["step"] > 0
    assert json.load(open(os.path.join(out, "best.json")))["compare"]["select_metric"] \
        == "combined"
    build_eval_set(dump, None, rows=10_000, holdout=0.5)
    eval_set = os.path.join(dump, "eval_fixed.json")
    out2 = os.path.join(tmp, "out2")
    stats = train(_args(*DUMP_RUN, "--dump-dir", dump, "--out", out2,
                        "--eval-set", eval_set))
    assert stats["steps"] == 2
    # an eval set built with another split seed holds training documents
    _exit(lambda: train(_args(*DUMP_RUN, "--dump-dir", dump, "--out",
                              os.path.join(tmp, "o3"), "--eval-set", eval_set,
                              "--seed", "7")), "not held out")
    # soft targets asked for, but the dump has none
    _exit(lambda: train(_args(*DUMP_RUN, "--dump-dir", dump, "--out",
                              os.path.join(tmp, "o4"), "--soft-alpha", "0.5")),
          "top-K")
    # --tap-layers that the dump does not carry
    _exit(lambda: train(_args(*DUMP_RUN, "--dump-dir", dump, "--out",
                              os.path.join(tmp, "o5"), "--tap-layers", "3")),
          "--tap-layers")


# ---------------------------------------------------------- resume comparability
def _first_run(out, *extra):
    return train(_args(*SYN_RUN, "--steps", "2", "--out", out, *extra))


def _resume(out, *extra, steps="4"):
    return train(_args(*SYN_RUN, "--steps", steps, "--out", out, "--resume",
                       os.path.join(out, "latest.pt"), *extra))


def test_resume_refuses_a_different_select_metric(tmp):
    """Flag 3: the old best score used to be carried over."""
    out = os.path.join(tmp, "out")
    _first_run(out)
    before = sorted(os.listdir(out))
    msg = _exit(lambda: _resume(out, "--select-metric", "agreement"),
                "select_metric", "--reset-best")
    assert "'combined'" in msg and "'agreement'" in msg
    assert sorted(os.listdir(out)) == before  # nothing touched
    _exit(lambda: _resume(out, "--eval-starts", "3"), "eval_starts")
    _exit(lambda: _resume(out, "--chain-ks", "3", "15", "5"), "chain_ks")
    _exit(lambda: _resume(out, "--seed", "5"), "seed")
    # logging / speed knobs may change
    stats = _resume(out, "--log-every", "2", "--prefetch", "2", "--ckpt-every", "2")
    assert stats["steps"] == 4


def test_resume_with_reset_best_restarts_selection(tmp):
    out = os.path.join(tmp, "out")
    _first_run(out)
    old_best = json.load(open(os.path.join(out, "best.json")))
    assert old_best["step"] == 2
    stats = _resume(out, "--select-metric", "agreement", "--reset-best")
    assert stats["best"]["criterion"] == "agreement"
    assert stats["best"]["step"] == 4  # selected afresh, not compared with step 2
    marker = [h for h in stats["history"] if "reset_best" in h]
    assert marker == [{"step": 2, "reset_best": ["select_metric"]}]
    aside = glob.glob(os.path.join(out, ".reset-best-*", "best.json"))
    assert len(aside) == 1 and json.load(open(aside[0]))["step"] == 2
    best = json.load(open(os.path.join(out, "best.json")))
    assert best["compare"]["select_metric"] == "agreement"
    ck = torch.load(os.path.join(out, "latest.pt"), map_location="cpu", weights_only=True)
    assert ck["meta"]["compare"]["select_metric"] == "agreement"
    # the new settings are the run's now: resuming with them needs no flag
    assert _resume(out, "--select-metric", "agreement", steps="5")["steps"] == 5


def test_resume_never_adopts_a_best_scored_under_another_record(tmp):
    """Round 11: --reset-best to another metric, a new best is saved, the run
    dies before its next checkpoint; resuming the older latest.pt (the old
    metric) must not adopt that best."""
    from mtptrain.train import SimulatedCrash

    out = os.path.join(tmp, "out")
    _first_run(out)  # combined; latest.pt at step 2
    with pytest.raises(SimulatedCrash):
        train(_args(*SYN_RUN, "--steps", "4", "--out", out, "--resume",
                    os.path.join(out, "latest.pt"), "--select-metric", "agreement",
                    "--reset-best", "--eval-every", "1", "--ckpt-every", "100"),
              _crash_after_step=3)
    best = json.load(open(os.path.join(out, "best.json")))
    assert best["criterion"] == "agreement" and best["step"] == 3
    ck = torch.load(os.path.join(out, "latest.pt"), map_location="cpu", weights_only=True)
    assert ck["step"] == 2 and ck["meta"]["compare"]["select_metric"] == "combined"
    stats = _resume(out)  # the checkpoint's own settings
    assert stats["best"]["criterion"] == "combined"
    evals = [h["step"] for h in stats["history"] if "eval" in h]
    assert stats["best"]["step"] in evals and stats["best"]["step"] > 2
    aside = glob.glob(os.path.join(out, ".previous-*", "best.json"))
    assert any(json.load(open(p))["criterion"] == "agreement" for p in aside)
    header = json.loads(__import__("mtptrain.train", fromlist=["x"]).weights_header(
        os.path.join(out, "best-mtp.safetensors"))["mtp_best"])
    assert header["criterion"] == "combined"


def test_resume_of_an_older_checkpoint_without_a_record_needs_reset_best(tmp):
    """No record: nothing shows the old best is comparable, so it is refused."""
    out = os.path.join(tmp, "out")
    _first_run(out)
    path = os.path.join(out, "latest.pt")
    ck = torch.load(path, map_location="cpu", weights_only=True)
    del ck["meta"]["compare"]
    ck["best"].pop("compare", None)
    torch.save(ck, path)
    _exit(lambda: _resume(out), "no comparability record", "--reset-best")
    stats = _resume(out, "--reset-best")
    assert stats["steps"] == 4 and stats["best"]["step"] > 2


def test_resume_refuses_changed_renewal_settings_and_token_map(tmp):
    dump = os.path.join(tmp, "dump")
    _dump(dump, docs=6)
    tm = os.path.join(tmp, "map.pt")
    torch.save(list(range(CFG.vocab_size)), tm)
    renew = ["--eval-renewal", "--renewal-dump-dir", dump, "--renewal-ks", "3",
             "--renewal-docs", "2", "--renewal-min-gen", "4", "--holdout", "1.0",
             "--token-map", tm]
    out = os.path.join(tmp, "out")
    stats = _first_run(out, *renew)
    assert stats["best"]["criterion"] == "renewal"
    _exit(lambda: _resume(out, *renew[:-2], "--renewal-docs", "1", "--token-map", tm),
          "renewal")
    torch.save(list(range(CFG.vocab_size // 2)), tm)
    _exit(lambda: _resume(out, *renew), "token_map")
    # zero documents selected is an error, not a renewal score of 0
    _exit(lambda: train(_args(*SYN_RUN, "--steps", "1", "--out", os.path.join(tmp, "o2"),
                              "--eval-renewal", "--renewal-dump-dir", dump,
                              "--renewal-min-gen", "1000", "--holdout", "1.0")),
          "no held-out self-generated document")
    _exit(lambda: train(_args(*SYN_RUN, "--steps", "1", "--out", os.path.join(tmp, "o3"),
                              "--eval-renewal", "--renewal-dump-dir", dump,
                              "--renewal-buckets", "nosuch", "--holdout", "1.0")),
          "unknown bucket")


def test_model_identity_is_content_not_path(tmp):
    from mtptrain.weights import model_identity

    a = os.path.join(tmp, "a")
    os.makedirs(os.path.join(a, ".cache"))
    json.dump({"x": 1}, open(os.path.join(a, "config.json"), "w"))
    json.dump({"weight_map": {}}, open(os.path.join(a, "model.safetensors.index.json"), "w"))
    open(os.path.join(a, "model-00001.safetensors"), "wb").write(b"0" * 10)
    open(os.path.join(a, "tokenizer.json"), "w").write('{"v": "aaaa"}')
    b = os.path.join(tmp, "b")
    shutil.copytree(a, b)
    assert model_identity(a) == model_identity(b)
    open(os.path.join(b, ".cache", "junk"), "w").write("download cache")
    assert model_identity(a) == model_identity(b)  # hidden files do not count
    # a shard swapped for another of the same name and size
    open(os.path.join(b, "model-00001.safetensors"), "wb").write(b"1" * 10)
    assert model_identity(a)["files"] != model_identity(b)["files"]
    shutil.copy(os.path.join(a, "model-00001.safetensors"), b)
    assert model_identity(a) == model_identity(b)
    # a tokenizer swapped at the same size
    open(os.path.join(b, "tokenizer.json"), "w").write('{"v": "bbbb"}')
    ib = model_identity(b)
    assert ib["files"] != model_identity(a)["files"]
    assert ib["tokenizer"] != model_identity(a)["tokenizer"]


# ====================================================================== fileio
def test_read_jsonl_missing_file_is_an_error_unless_allowed(tmp):
    _exit(lambda: read_jsonl(os.path.join(tmp, "nope.jsonl")), "no such file")
    assert read_jsonl(os.path.join(tmp, "nope.jsonl"), missing_ok=True) == []
    _exit(lambda: read_jsonl(tmp), "not a file")


def test_check_run_settings_names_the_difference():
    rows = [{"settings": {"a": 1, "b": 2}}, {"no": "settings"}]
    check_run_settings(rows, "settings", {"a": 1, "b": 2}, "log")
    _exit(lambda: check_run_settings(rows, "settings", {"a": 1, "b": 3}, "log"),
          "'b': (2, 3)")


# ================================================================ dump tools
def test_index_of_a_missing_or_empty_dump(tmp):
    from extract import build_index
    from mtptrain.dumpindex import load_index, scan

    _exit(lambda: load_index(os.path.join(tmp, "nope")), "no such directory")
    _exit(lambda: scan(os.path.join(tmp, "nope")), "no such directory")
    assert build_index.main([tmp]) == 1  # written, but reported as unusable


def test_build_eval_set_refuses_empty_and_silent_replacement(tmp):
    from extract.build_eval_set import build_eval_set

    _dump(tmp)
    out = os.path.join(tmp, "eval_fixed.json")
    build_eval_set(tmp, out, rows=10_000, holdout=0.5)
    first = open(out).read()
    build_eval_set(tmp, out, rows=10_000, holdout=0.5)  # same settings: fine
    assert open(out).read() == first
    _exit(lambda: build_eval_set(tmp, out, rows=10, holdout=0.5), "different eval set")
    assert open(out).read() == first
    build_eval_set(tmp, out, rows=10, holdout=0.5, overwrite=True)
    assert open(out).read() != first
    assert glob.glob(os.path.join(tmp, ".previous-*", "eval_fixed.json"))
    _exit(lambda: build_eval_set(tmp, out, rows=0), "positive")
    _exit(lambda: build_eval_set(tmp, out, holdout=0.0), "holdout")
    open(os.path.join(tmp, "manifest.jsonl"), "w").close()
    _exit(lambda: build_eval_set(tmp, os.path.join(tmp, "e2.json"), holdout=0.5),
          "no held-out document")
    assert not os.path.exists(os.path.join(tmp, "e2.json"))


def test_verify_dump_refuses_when_nothing_is_on_disk(tmp):
    from extract.verify_dump import verify

    _dump(tmp)
    assert verify(tmp, docs=3)["checked"] == 12
    _write_jsonl(os.path.join(tmp, "manifest.jsonl"),
                 [{"doc_hash": "0" * 16, "prompt_tokens": 4}])
    _exit(lambda: verify(tmp), "nothing was verified")
    os.remove(os.path.join(tmp, "manifest.jsonl"))
    _exit(lambda: verify(tmp), "no such file")


def test_select_refuses_unknown_bucket_split_and_missing_manifest(tmp):
    from mtptrain.select import load_token_map, select_selfgen_docs

    _dump(tmp)
    assert len(select_selfgen_docs(tmp, split="all", min_gen=4)[0]) == 12
    _exit(lambda: select_selfgen_docs(tmp, split="all", buckets=["cod"]), "unknown bucket")
    _exit(lambda: select_selfgen_docs(tmp, split="dev"), "split must be")
    _exit(lambda: select_selfgen_docs(tmp, manifest=os.path.join(tmp, "x.jsonl")),
          "no such file")
    tm = os.path.join(tmp, "tm.pt")
    torch.save([], tm)
    _exit(lambda: load_token_map(tm, 64), "empty")
    torch.save([1, 99], tm)
    _exit(lambda: load_token_map(tm, 64), "outside")


# =================================================================== client
def test_client_corpus_sources_must_exist_and_contribute(tmp):
    from extract.client import load_corpus, load_jsonl_docs

    _exit(lambda: load_corpus(claude=[os.path.join(tmp, "nope")]), "--claude-sessions",
          "no such")
    _exit(lambda: load_corpus(files=[os.path.join(tmp, "*.md")]), "matches no file")
    open(os.path.join(tmp, "x.pdf"), "w").write("binary")
    _exit(lambda: load_corpus(files=[os.path.join(tmp, "*.pdf")]), "no usable document")
    open(os.path.join(tmp, "a.md"), "w").write("some prose " * 50)
    docs = load_corpus(files=[os.path.join(tmp, "*.md")])
    assert len(docs) == 1
    # one good source does not excuse a broken one
    _exit(lambda: load_corpus(files=[os.path.join(tmp, "*.md"),
                                     os.path.join(tmp, "*.txt")]), "matches no file")
    jl = os.path.join(tmp, "t.jsonl")
    _write_jsonl(jl, [{"text": "x" * 300}])
    doc = load_jsonl_docs([jl])[0]
    assert doc["split_key"] and doc["content_hash"]  # dedup and the split need them
    _write_jsonl(jl, [{"text": "short"}])
    _exit(lambda: load_jsonl_docs([jl]), "no record")
    _exit(lambda: load_jsonl_docs([os.path.join(tmp, "nope.jsonl")]), "no such file")


def _client_args(tmp, *argv):
    from extract.client import build_parser as client_parser

    return client_parser().parse_args(["--model-dir", tmp, "--manifest",
                                       os.path.join(tmp, "m.jsonl"), *argv])


@pytest.mark.parametrize("argv,word", [
    (["--mode", "selfgen", "--dry-run"], "--dry-run only applies"),
    (["--mode", "selfgen", "--max-len", "600", "--gen-tokens", "512",
      "--min-len", "100"], "leaves less than --min-len"),
    (["--token-budget", "0"], "--token-budget"),
    (["--limit-docs", "3"], "--limit-docs only applies"),
    (["--chat-kwargs", "[1]"], "not a JSON object"),
])
def test_client_check_args(tmp, argv, word):
    from extract.client import check_args

    _exit(lambda: check_args(_client_args(tmp, *argv)), word)
    args = _client_args(tmp)
    args.model_dir = os.path.join(tmp, "nope")
    _exit(lambda: check_args(args), "--model-dir")


def test_client_resume_refuses_other_settings_for_the_same_bucket(tmp):
    from extract.client import resume_state

    man = os.path.join(tmp, "m.jsonl")
    _write_jsonl(man, [{"doc_hash": "a", "split_key": "s", "mode": "corpus",
                        "bucket": "code", "tokens": 10, "settings": {"max_len": 2048}},
                       {"doc_hash": "b", "split_key": "t", "mode": "corpus",
                        "bucket": "prose", "tokens": 10, "settings": {"max_len": 512}}])
    args = types.SimpleNamespace(manifest=man, mode="extract", bucket="code",
                                 dry_run=False, settings={"max_len": 2048})
    assert resume_state(args)[0] == {"a"}
    args.settings = {"max_len": 1024}
    _exit(lambda: resume_state(args), "max_len")


# ==================================================================== scripts
def test_eval_renewal_refuses_missing_inputs_and_existing_outputs(tmp):
    from scripts import eval_renewal

    per = os.path.join(tmp, "per.jsonl")
    open(per, "w").close()
    rn = argparse.Namespace(dump_dir=tmp, model_dir=tmp, manifest=None, ckpt=None,
                            token_map=None, ks=[3], batch_size=4, per_bucket=0,
                            min_gen=0, max_len=0, holdout_frac=0.1, per_seq=per,
                            append=None)
    _exit(lambda: eval_renewal.check_args(rn), "--per-seq")
    rn.per_seq, rn.dump_dir = None, os.path.join(tmp, "nope")
    _exit(lambda: eval_renewal.check_args(rn), "--dump-dir")
    rn.dump_dir, rn.ckpt = tmp, os.path.join(tmp, "nope.safetensors")
    _exit(lambda: eval_renewal.check_args(rn), "--ckpt")


def test_eval_renewal_seed_is_not_the_split_seed(tmp):
    from scripts import eval_renewal

    parse = eval_renewal.build_parser().parse_args
    base = ["--dump-dir", tmp, "--model-dir", tmp]
    args = parse(base)
    eval_renewal.check_args(args)
    assert args.split_seed == eval_renewal.DEFAULT_SPLIT_SEED and args.seed == 0
    # a non-default --seed with an implicit split seed is refused (it does not change the split)
    _exit(lambda: eval_renewal.check_args(parse(base + ["--seed", "7"])), "--split-seed")
    args = parse(base + ["--seed", "7", "--split-seed", "7"])
    eval_renewal.check_args(args)
    assert (args.seed, args.split_seed) == (7, 7)


# ======================================================= write-back / server
def test_writeback_refuses_missing_inputs(tmp):
    from writeback.write_mtp import load_trained, resolve_dirs

    _exit(lambda: resolve_dirs(tmp, os.path.join(tmp, "..", "dst")), "not a checkpoint")
    _exit(lambda: load_trained(os.path.join(tmp, "nope.pt")), "no such file")


def test_serve_sh_refuses_missing_dump_tree(tmp):
    launch = os.path.join(tmp, "launch")
    os.makedirs(launch)
    open(os.path.join(launch, "serve-local.sh"), "w").close()
    env = {"SGLANG_DIR": tmp, "SGLANG_DUMP_TREE": os.path.join(tmp, "nope"),
           "LAUNCH_DIR": launch, "SGLANG_MTP_DUMP_DIR": tmp}
    r = subprocess.run(["bash", os.path.join(REPO, "parity", "serve.sh"), "start-extract",
                        os.path.join(tmp, "x.log")], capture_output=True, text=True,
                       env=dict(os.environ, **env), timeout=60)
    assert r.returncode == 2 and "no python/" in r.stderr
    r = subprocess.run(["bash", os.path.join(REPO, "parity", "serve.sh"), "stop",
                        os.path.join(tmp, "none.pgid")], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 2


def test_rollout_refuses_unknown_modes():
    from mtptrain.rollout import chain_losses

    with pytest.raises(ValueError, match="mask_mode"):
        chain_losses(None, {}, 1, [1.0, 0.5], mask_mode="acepted")
    with pytest.raises(ValueError, match="soft_steps"):
        chain_losses(None, {}, 1, [1.0, 0.5], soft_steps="some")
