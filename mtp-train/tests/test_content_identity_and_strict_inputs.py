"""Inputs are identified by content, checked before anything drops them, and an
explicit empty value is never read as "not given".

(a) every corpus input is checked against the manifest before dedup, filters,
    skips or budgets can drop it;
(b) identities are content: eval data (every tensor the eval reads), model
    directories (shards, config, tokenizer), weight files, write-back output;
    a digest cache is trusted only under an unchanged (size, mtime, ctime,
    inode, device) key;
(c) an option given with no value, an empty value or an empty list item is
    refused.

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

from mtptrain.fileio import read_jsonl  # noqa: E402
from mtptrain.train import build_parser, train  # noqa: E402
from tests.test_input_validation import DUMP_RUN, _dump, _exit  # noqa: E402
from tests.test_overlap_and_unmeasured import _selfgen  # noqa: E402


@pytest.fixture
def tmp():
    d = tempfile.mkdtemp(prefix="mtpident-")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _args(*argv):
    return build_parser().parse_args(list(argv))


# ============================================ (a) check before anything drops
def _texts():
    return [f"question {i}: " + "why? " * 20 for i in range(3)]


def _key_of(manifest, name):
    return next(r["split_key"] for r in read_jsonl(manifest) if r["name"].startswith(name))


def test_an_input_edited_into_a_copy_of_another_is_refused_before_dedup(tmp):
    """Dedup used to drop the edited copy before the check ever saw it."""
    texts = _texts()
    args = _selfgen(tmp, texts)
    key = _key_of(args.manifest, "chat1.jsonl")
    texts[1] = texts[0]
    _exit(lambda: _selfgen(tmp, texts), f"split key {key}", "changed since")
    assert len(read_jsonl(args.manifest)) == 3


def test_an_input_the_loader_now_drops_is_refused(tmp):
    """Edited into something the loader drops (here: no reply left)."""
    args = _selfgen(tmp, _texts())
    key = _key_of(args.manifest, "chat2.jsonl")
    with open(os.path.join(tmp, "chat2.jsonl"), "w") as fh:
        fh.write(json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n")
    # (_selfgen rewrites chat0/chat1 unchanged and leaves the edited chat2)
    _exit(lambda: _selfgen(tmp, _texts()[:2]), f"split key {key}",
          "loads no such document")


def test_an_edited_input_is_refused_even_when_a_filter_would_drop_it(tmp):
    from extract.client import build_parser as client_parser, prepare_docs

    texts = _texts()
    args = _selfgen(tmp, texts)
    texts[1] = "an edited question: " + "how? " * 20
    for i, t in enumerate(texts):
        with open(os.path.join(tmp, f"chat{i}.jsonl"), "w") as fh:
            fh.write(json.dumps({"messages": [
                {"role": "user", "content": t},
                {"role": "assistant", "content": "because " * 20}]}) + "\n")
    # --workload code drops every (prose) document: the check still runs first
    filtered = client_parser().parse_args(
        ["--mode", "selfgen", "--model-dir", tmp, "--manifest", args.manifest,
         "--files", os.path.join(tmp, "*.jsonl"), "--workload", "code"])
    _exit(lambda: prepare_docs(filtered, 1), "changed since", "chat1.jsonl")


# ================================================== (b) content identities
def _rewrite_dump_tensor(dump, name, fn):
    """Change one tensor in every dump file, keeping ids and metadata."""
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    for path in glob.glob(os.path.join(dump, "*.safetensors")):
        with safe_open(path, framework="pt") as f:
            meta = f.metadata()
        t = load_file(path)
        t[name] = fn(t[name])
        save_file(t, path, metadata=meta)


@pytest.mark.parametrize("tensor,fn", [
    ("hc_hidden", lambda t: (t.float() * 0.5).to(t.dtype)),
    ("target_argmax", lambda t: (t + 1) % 64),
])
def test_resume_refuses_eval_data_changed_under_the_same_ids(tmp, tensor, fn):
    """Same doc hashes, other hidden states or teacher labels: not the same eval."""
    dump = os.path.join(tmp, "dump")
    _dump(dump)
    out = os.path.join(tmp, "out")
    # agreement@1 stays measurable when the labels change
    run = [*DUMP_RUN, "--dump-dir", dump, "--out", out, "--ckpt-every", "1",
           "--select-metric", "agreement"]
    train(_args(*run))
    _rewrite_dump_tensor(dump, tensor, fn)
    _exit(lambda: train(_args(*run, "--steps", "4", "--resume",
                              os.path.join(out, "latest.pt"))), "eval_data")


def test_resume_accepts_a_moved_dump_with_the_same_content(tmp, capsys):
    """Paths are not identities: a copied dump directory is the same data."""
    dump = os.path.join(tmp, "dump")
    _dump(dump)
    out = os.path.join(tmp, "out")
    train(_args(*DUMP_RUN, "--dump-dir", dump, "--out", out, "--ckpt-every", "1"))
    moved = os.path.join(tmp, "moved")
    shutil.copytree(dump, moved)
    shutil.rmtree(dump)
    capsys.readouterr()
    stats = train(_args(*DUMP_RUN, "--dump-dir", moved, "--out", out, "--ckpt-every",
                        "1", "--steps", "4", "--resume", os.path.join(out, "latest.pt")))
    assert stats["steps"] == 4
    assert "WARNING" not in capsys.readouterr().out


def test_samples_digest_covers_every_tensor_but_not_locations():
    from mtptrain.data import samples_digest
    from mtptrain.config import MTPConfig
    from mtptrain.train import synthetic_samples

    samples = synthetic_samples(MTPConfig.tiny(), 2, 16, seed=0, topk=4)
    d = samples_digest(samples)
    assert d == samples_digest(synthetic_samples(MTPConfig.tiny(), 2, 16, seed=0, topk=4))
    samples[0].pieces, samples[0].source = [("elsewhere.safetensors", 3)], "moved"
    assert samples_digest(samples) == d
    for name in ("hc_hidden", "labels", "topk_logits", "greedy_consistent"):
        s2 = synthetic_samples(MTPConfig.tiny(), 2, 16, seed=0, topk=4)
        t = getattr(s2[1], name)
        setattr(s2[1], name, (~t) if t.dtype == torch.bool else (t + 1).to(t.dtype))
        assert samples_digest(s2) != d, name
    assert samples_digest(samples, [3, 4]) != samples_digest(samples, [3, 5])


def test_file_digest_is_full_content_and_its_cache_follows_the_file(tmp):
    from mtptrain.fileio import digest_cache_path, file_digest

    path = os.path.join(tmp, "w.bin")
    open(path, "wb").write(b"a" * 4096 + b"b")
    d1 = file_digest(path)
    assert os.path.isfile(digest_cache_path())
    assert file_digest(path) == d1  # served from the cache
    st = os.stat(path)
    # the last byte changes in place; size and mtime are put back
    with open(path, "r+b") as fh:
        fh.seek(4096)
        fh.write(b"c")
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert os.stat(path).st_size == st.st_size
    assert file_digest(path) != d1  # ctime moved: recomputed
    # a cache entry under the current key is trusted -- and only then
    cache = json.load(open(digest_cache_path()))
    real = os.path.realpath(path)
    cache[real]["digest"] = "forged"
    json.dump(cache, open(digest_cache_path(), "w"))
    assert file_digest(path) == "forged"
    shutil.copy(path, path + ".x")
    os.replace(path + ".x", path)  # new inode, same bytes
    assert file_digest(path) not in ("forged", d1)


def test_digest_cache_env_set_but_empty_is_refused(tmp, monkeypatch):
    from mtptrain.fileio import file_digest

    path = os.path.join(tmp, "f")
    open(path, "w").write("x")
    monkeypatch.setenv("MTP_DIGEST_CACHE", "")
    _exit(lambda: file_digest(path), "MTP_DIGEST_CACHE")


def test_client_resume_refuses_a_swapped_tokenizer(tmp):
    """The client's run settings identify --model-dir by content."""
    import extract.client as client

    model = os.path.join(tmp, "model")
    os.makedirs(model)
    for name, text in (("config.json", "{}"), ("tokenizer.json", '{"v": "aaaa"}')):
        open(os.path.join(model, name), "w").write(text)
    man = os.path.join(tmp, "m.jsonl")
    args = client.build_parser().parse_args(
        ["--model-dir", model, "--manifest", man, "--files", "x"])
    settings = client.run_settings(args)
    with open(man, "w") as fh:
        fh.write(json.dumps({"doc_hash": "a", "split_key": "s", "mode": "corpus",
                             "bucket": args.bucket, "tokens": 5, "input_hash": "h",
                             "settings": settings}) + "\n")
    args.settings = client.run_settings(args)
    client.resume_state(args)  # unchanged: fine
    open(os.path.join(model, "tokenizer.json"), "w").write('{"v": "bbbb"}')  # same size
    args.settings = client.run_settings(args)
    _exit(lambda: client.resume_state(args), "model")


def test_eval_renewal_settings_are_content(tmp):
    from mtptrain.config import MTPConfig
    from mtptrain.train import synthetic_samples
    from scripts.eval_renewal import build_parser as renewal_parser, run_settings

    open(os.path.join(tmp, "config.json"), "w").write("{}")
    a = renewal_parser().parse_args(["--dump-dir", tmp, "--model-dir", tmp])
    sel = types.SimpleNamespace(samples=synthetic_samples(MTPConfig.tiny(), 2, 16, seed=0),
                                gen_start_rows=[4, 4], buckets=["code", "prose"])
    s1 = run_settings(a, sel)
    assert "dump_dir" not in s1 and "manifest" not in s1  # paths are not identities
    b = renewal_parser().parse_args(["--dump-dir", os.path.join(tmp, ".."),
                                     "--model-dir", tmp])
    assert run_settings(b, sel) == s1
    sel.samples[0].hc_hidden = sel.samples[0].hc_hidden * 2
    assert run_settings(a, sel)["data"] != s1["data"]


def test_writeback_verify_compares_side_files_by_content(tmp):
    from tests.test_writeback import _make_fake_checkpoint, _trained_state
    from writeback.write_mtp import build, verify

    src, dst = os.path.join(tmp, "src"), os.path.join(tmp, "dst")
    _make_fake_checkpoint(src)
    trained = _trained_state()
    report = build(src, dst, trained)
    assert report["verify"]["unchanged_files_checked"] >= 4
    # a side file replaced by another of the same size, not a link any more
    tok = os.path.join(dst, "tokenizer.json")
    os.remove(tok)
    open(tok, "w").write("[]\n")
    _exit(lambda: verify(src, dst, trained), "tokenizer.json", "content differs")
    os.remove(tok)
    shutil.copy(os.path.join(src, "tokenizer.json"), tok)  # a copy is fine
    verify(src, dst, trained)
    os.remove(tok)
    _exit(lambda: verify(src, dst, trained), "missing", "tokenizer.json")


def test_writeback_refuses_an_empty_src_or_dst(tmp):
    from writeback.write_mtp import resolve_dirs

    _exit(lambda: resolve_dirs("", os.path.join(tmp, "d")), "--src is empty")
    _exit(lambda: resolve_dirs(tmp, " "), "--dst is empty")


# ============================================ (c) explicit empty values
def _parse_error(parser, argv, capsys, word=""):
    with pytest.raises(SystemExit) as ei:
        parser.parse_args(argv)
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert word in err, err
    return err


@pytest.mark.parametrize("argv,word", [
    (["--eval-set", ""], "empty value"),
    (["--dump-dir", " "], "empty value"),
    (["--renewal-buckets", ""], "empty value"),
    (["--exclude-buckets", ""], "empty value"),
    (["--chain-ks"], "expected at least one"),
    (["--renewal-ks"], "expected at least one"),
    (["--tap-layers"], "expected at least one"),
])
def test_train_refuses_explicit_empty_values(argv, word, capsys):
    _parse_error(build_parser(), argv, capsys, word)


@pytest.mark.parametrize("argv,word", [
    (["--jsonl"], "expected at least one"),
    (["--files", ""], "empty value"),
    (["--arrow-aux-field", ""], "empty value"),
    (["--workload", ""], "empty value"),
    (["--manifest", ""], "empty value"),
])
def test_client_refuses_explicit_empty_values(argv, word, capsys):
    from extract.client import build_parser as client_parser

    _parse_error(client_parser(), argv, capsys, word)


@pytest.mark.parametrize("argv,word", [
    (["--buckets"], "expected at least one"),
    (["--buckets", ""], "empty value"),
    (["--tap-layers"], "expected at least one"),
    (["--manifest", ""], "empty value"),
])
def test_eval_renewal_refuses_explicit_empty_values(argv, word, capsys, tmp):
    from scripts.eval_renewal import build_parser as renewal_parser

    _parse_error(renewal_parser(), ["--dump-dir", tmp, *argv], capsys, word)


def test_comma_lists_refuse_empty_items(tmp):
    from extract.client import filter_docs
    from mtptrain.cli import split_list

    assert split_list("", "--x") == [] and split_list("a,b", "--x") == ["a", "b"]
    _exit(lambda: split_list("a,,b", "--x"), "empty item")
    _exit(lambda: split_list("a,", "--x"), "empty item")
    _exit(lambda: filter_docs([], workload="code,"), "--workload")
    dump = os.path.join(tmp, "dump")
    _dump(dump)
    _exit(lambda: train(_args(*DUMP_RUN, "--dump-dir", dump, "--out",
                              os.path.join(tmp, "o"), "--exclude-buckets", "code,,x")),
          "--exclude-buckets", "empty item")


def test_named_sources_refuse_an_empty_name_or_path(tmp):
    from extract.client import load_corpus

    _exit(lambda: load_corpus(files=["=" + os.path.join(tmp, "*.md")]), "empty NAME")
    _exit(lambda: load_corpus(files=["docs="]), "empty NAME")


def test_build_index_refuses_an_empty_argument():
    from extract.build_index import main

    _exit(lambda: main([""]), "empty")


def test_empty_environment_defaults_are_not_widened(tmp, monkeypatch):
    env = dict(os.environ, MTP_MODEL_DIR="", PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run(
        [sys.executable, "-c",
         "from mtptrain.config import DEFAULT_MODEL_DIR; print(repr(DEFAULT_MODEL_DIR))"],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
    assert out.stdout.strip() == "''", out  # set-but-empty stays empty: refused later
    import corpus.formats as F

    monkeypatch.setitem(sys.modules, "ngramsim", None)  # force the fallback path
    monkeypatch.setenv("NGRAMSIM_PATH", "")
    _exit(F._import_ngramsim, "NGRAMSIM_PATH")


def test_argparse_namespace_is_untouched_when_nothing_is_given():
    from mtptrain.cli import StrictParser

    p = StrictParser()
    p.add_argument("--xs", nargs="+", default=None)
    p.add_argument("--s", default="")
    p.add_argument("--flag", action="store_true")
    assert vars(p.parse_args([])) == {"xs": None, "s": "", "flag": False}
    assert vars(p.parse_args(["--xs", "a", "--flag"]))["xs"] == ["a"]
    assert isinstance(p, argparse.ArgumentParser)
