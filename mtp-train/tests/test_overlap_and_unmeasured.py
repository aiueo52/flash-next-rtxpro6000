"""Three failure classes that each looked like success.

(i)   eval documents that are also training documents (another dump, another
      split key, the renewal set): the eval then rewards memorisation;
(ii)  an input whose content changed under an unchanged key (manifest rows on
      resume, eval sets, checkpoints, the base weights);
(iii) a metric with nothing to measure reported as 0 and then used to pick
      the best checkpoint.

CPU only, tiny configs, temporary directories.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import sys
import tempfile
import types

import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from mtptrain.data import DumpDataset, is_holdout_key  # noqa: E402
from mtptrain.fileio import read_jsonl  # noqa: E402
from mtptrain.train import build_parser, train  # noqa: E402
from tests.test_input_validation import DUMP_RUN, SYN_RUN, _dump, _exit, _write_jsonl  # noqa: E402

SEED = 20260903


@pytest.fixture
def tmp():
    d = tempfile.mkdtemp(prefix="mtpoverlap-")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _args(*argv):
    return build_parser().parse_args(list(argv))


def _clone_dump(src, dst, key=lambda r: r["split_key"], rows=None):
    """The same dumped documents under other manifest rows (another dump of
    the same corpus: other file names, so other split keys)."""
    os.makedirs(dst, exist_ok=True)
    for f in glob.glob(os.path.join(src, "*.safetensors")):
        shutil.copy(f, dst)
    rows = rows if rows is not None else [
        dict(r, split_key=key(r)) for r in read_jsonl(os.path.join(src, "manifest.jsonl"))]
    _write_jsonl(os.path.join(dst, "manifest.jsonl"), rows)


def _key(seed, held):
    """A split key that is (held=True) or is not held out at --holdout 0.5."""
    for i in range(1000):
        k = f"key-{i}"
        if is_holdout_key(k, seed, 0.5) == held:
            return k
    raise AssertionError


# ============================================================ (i) overlap
def test_eval_set_from_another_dump_sharing_documents_is_refused(tmp):
    from extract.build_eval_set import build_eval_set

    a, b = os.path.join(tmp, "a"), os.path.join(tmp, "b")
    _dump(a)
    _clone_dump(a, b, key=lambda r: "other:" + r["split_key"])
    train_docs = {h for h, _ in DumpDataset(a, "train", holdout_frac=0.5).documents()}
    build_eval_set(b, None, rows=10_000, holdout=0.5)
    ev = json.load(open(os.path.join(b, "eval_fixed.json")))
    assert {d["doc_hash"] for d in ev["docs"]} & train_docs  # the fixture overlaps
    out = os.path.join(tmp, "out")
    _exit(lambda: train(_args(*DUMP_RUN, "--dump-dir", a, "--out", out, "--eval-set",
                              os.path.join(b, "eval_fixed.json"))),
          "overlap the training data", "--eval-set", "same token ids")
    assert not os.path.exists(out)
    # the eval set of the training dump itself is disjoint and accepted
    build_eval_set(a, None, rows=10_000, holdout=0.5)
    stats = train(_args(*DUMP_RUN, "--dump-dir", a, "--out", out, "--eval-set",
                        os.path.join(a, "eval_fixed.json")))
    assert stats["steps"] == 2


def test_eval_documents_from_a_training_source_document_are_refused():
    """Other token ids (another window/chunking) of a training source document."""
    from mtptrain.train import check_train_eval_overlap

    train_ids = {"h1": {"k1"}, "h2": {"k2"}}
    check_train_eval_overlap(train_ids, [("eval", {"h3": {"k3"}, "h4": set()})])
    _exit(lambda: check_train_eval_overlap(train_ids, [("eval", {"h3": {"k3", "k2"}})]),
          "from a training source document", "h3")
    _exit(lambda: check_train_eval_overlap(train_ids, [("the renewal documents",
                                                        {"h1": set()})]),
          "the renewal documents", "same token ids")


def test_eval_set_split_keys_are_checked_against_training(tmp):
    """A document held out in one dump under key K and trained on in another
    under the same K plus another key: caught through the eval set's keys."""
    from extract.build_eval_set import build_eval_set

    a = os.path.join(tmp, "a")
    _dump(a)
    rows = read_jsonl(os.path.join(a, "manifest.jsonl"))
    held = _key(SEED, True)
    # dump b: a *different* document (other ids) that also carries a key of
    # one of a's training documents, plus a held-out key of its own
    b = os.path.join(tmp, "b")
    os.makedirs(b)
    from mtptrain.config import MTPConfig
    from mtptrain.data import load_dump_file
    from tests.test_dump_roundtrip import _write_dump

    cfg = MTPConfig.tiny()
    ids = [(j * 3 + 1) % cfg.vocab_size for j in range(24)]
    tgt = ids[1:] + [0]
    path = os.path.join(b, "dump-1-00000000.safetensors")
    _write_dump(path, [(ids, tgt, 0)], hc_dim=cfg.hc_size)
    h = load_dump_file(path)[0].doc_hash
    assert h not in {r["doc_hash"] for r in rows}
    train_key = next(r["split_key"] for r in rows
                     if not is_holdout_key(r["split_key"], SEED, 0.5))
    _write_jsonl(os.path.join(b, "manifest.jsonl"), [
        {"doc_hash": h, "split_key": held, "mode": "corpus"},
        {"doc_hash": h, "split_key": train_key, "mode": "corpus"}])
    build_eval_set(b, None, rows=10_000, holdout=0.5)
    ev = json.load(open(os.path.join(b, "eval_fixed.json")))
    assert ev["docs"][0]["split_keys"] == sorted([held, train_key])
    _exit(lambda: train(_args(*DUMP_RUN, "--dump-dir", a, "--out",
                              os.path.join(tmp, "out"), "--eval-set",
                              os.path.join(b, "eval_fixed.json"))),
          "from a training source document", h)


def test_renewal_documents_from_another_dump_are_checked(tmp):
    a, b = os.path.join(tmp, "a"), os.path.join(tmp, "b")
    _dump(a)
    _clone_dump(a, b, key=lambda r: "other:" + r["split_key"])
    _exit(lambda: train(_args(*DUMP_RUN, "--dump-dir", a, "--out",
                              os.path.join(tmp, "out"), "--eval-renewal",
                              "--renewal-dump-dir", b, "--renewal-min-gen", "4",
                              "--renewal-ks", "3", "--select-metric", "combined")),
          "the renewal documents", "same token ids")


def test_holdout_is_by_any_key_of_a_document(tmp):
    """One document under two keys, one held out: it is eval, never train
    (the first key alone used to decide, so it could be both)."""
    from mtptrain.select import select_selfgen_docs

    d = os.path.join(tmp, "d")
    _dump(d, docs=4)
    rows = read_jsonl(os.path.join(d, "manifest.jsonl"))
    held, kept = _key(SEED, True), _key(SEED, False)
    others = [dict(r, split_key=kept) for r in rows[1:]]
    two = [dict(rows[0], split_key=kept), dict(rows[0], split_key=held)]
    _write_jsonl(os.path.join(d, "manifest.jsonl"), two + others)
    h = rows[0]["doc_hash"]
    ev = DumpDataset(d, "eval", holdout_frac=0.5, seed=SEED)
    tr = DumpDataset(d, "train", holdout_frac=0.5, seed=SEED)
    assert h in {x for x, _ in ev.documents()}
    assert h not in {x for x, _ in tr.documents()}
    assert ev.keys_of(h) == {held, kept}
    recs, _ = select_selfgen_docs(d, split="eval", holdout_frac=0.5, split_seed=SEED,
                                  min_gen=0)
    assert [r["doc_hash"] for r in recs] == [h]  # once, not once per row
    recs, _ = select_selfgen_docs(d, split="train", holdout_frac=0.5, split_seed=SEED,
                                  min_gen=0)
    assert h not in {r["doc_hash"] for r in recs}


# ======================================================= (ii) changed input
def _selfgen(tmp, texts, crash_after=None):
    import requests

    import extract.client as client
    from tests.test_crash_safety import _Resp, _StubRenderer

    for i, t in enumerate(texts):
        with open(os.path.join(tmp, f"chat{i}.jsonl"), "w") as fh:
            fh.write(json.dumps({"messages": [
                {"role": "user", "content": t},
                {"role": "assistant", "content": "because " * 20}]}) + "\n")

    def post(self, url, json=None, timeout=None):  # noqa: A002
        out = [(0.0, 7 + i, None) for i in range(json["sampling_params"]["max_new_tokens"])]
        return _Resp({"meta_info": {"output_token_logprobs": out}})

    args = client.build_parser().parse_args(
        ["--mode", "selfgen", "--model-dir", tmp, "--manifest",
         os.path.join(tmp, "dump", "manifest.jsonl"), "--files",
         os.path.join(tmp, "*.jsonl"), "--gen-tokens", "40", "--max-len", "400",
         "--min-len", "16", "--concurrency", "1"])
    real = (client.Renderer, requests.Session.post)
    client.Renderer, requests.Session.post = _StubRenderer, post
    try:
        client.run_selfgen(args)
    finally:
        client.Renderer, requests.Session.post = real
    return args


def test_client_resume_refuses_a_document_changed_under_its_key(tmp):
    texts = [f"question {i}: " + "why? " * 20 for i in range(3)]
    args = _selfgen(tmp, texts)
    rows = read_jsonl(args.manifest)
    assert len(rows) == 3 and all(r["input_hash"] for r in rows)
    _selfgen(tmp, texts)  # unchanged: resumes, nothing new
    assert len(read_jsonl(args.manifest)) == 3
    texts[1] = "an edited question: " + "how? " * 20
    key = next(r["split_key"] for r in rows if r["name"].startswith("chat1.jsonl"))
    _exit(lambda: _selfgen(tmp, texts), f"split key {key}", "chat1.jsonl", "changed since")
    assert len(read_jsonl(args.manifest)) == 3  # nothing appended


def test_client_refuses_manifest_rows_without_input_hash(tmp):
    from extract.client import check_inputs

    man = os.path.join(tmp, "m.jsonl")
    _write_jsonl(man, [{"doc_hash": "a", "split_key": "s", "mode": "corpus"}])
    doc = {"split_key": "s", "input_hash": "x", "source": "--files", "name": "a"}
    args = types.SimpleNamespace(manifest=man)
    _exit(lambda: check_inputs(args, [doc]), "no input_hash", "older client.py")
    _write_jsonl(man, [{"doc_hash": "a", "split_key": "s", "mode": "corpus",
                        "input_hash": "x"}])
    check_inputs(args, [doc])
    # two documents of one run under one key with different content
    _exit(lambda: check_inputs(args, [doc, dict(doc, input_hash="y", name="b")]),
          "split key s names two different documents")


def test_client_file_names_are_relative_to_the_glob_root(tmp):
    """a/x.md and b/x.md used to share the name x.md, hence the split key."""
    from extract.client import load_corpus

    for sub in ("a", "b"):
        os.makedirs(os.path.join(tmp, sub))
        with open(os.path.join(tmp, sub, "x.md"), "w") as fh:
            fh.write(f"prose from {sub} " * 40)
    docs = load_corpus(files=[os.path.join(tmp, "*", "x.md")])
    assert len(docs) == 2 and len({d["split_key"] for d in docs}) == 2
    assert sorted(d["name"].split(":")[0] for d in docs) == [os.path.join("a", "x.md"),
                                                             os.path.join("b", "x.md")]


def test_resume_refuses_changed_base_weights(tmp, monkeypatch):
    """--model-dir identity is content (the loaded weights), not path + size."""
    import mtptrain.train as T

    out = os.path.join(tmp, "out")
    train(_args(*SYN_RUN, "--steps", "2", "--out", out))
    assert json.load(open(os.path.join(out, "best.json")))["compare"]["base_weights"]
    monkeypatch.setattr(T, "weights_digest", lambda model: "0" * 32)
    _exit(lambda: train(_args(*SYN_RUN, "--steps", "4", "--out", out, "--resume",
                              os.path.join(out, "latest.pt"))), "base_weights")


def test_weights_digest_is_content():
    from mtptrain.config import MTPConfig
    from mtptrain.model import MTPHead
    from mtptrain.train import weights_digest

    torch.manual_seed(0)
    m = MTPHead(MTPConfig.tiny())
    d = weights_digest(m)
    assert d == weights_digest(m)
    with torch.no_grad():
        next(iter(m.parameters())).view(-1)[0] += 1.0
    assert weights_digest(m) != d


def test_eval_renewal_defaults_to_the_held_out_split_and_records_the_ckpt_content(tmp):
    from mtptrain.renewal import RenewalResult, SeqStat
    from scripts.eval_renewal import build_parser as renewal_parser, renewal_output

    a = renewal_parser().parse_args(["--dump-dir", tmp])
    assert a.split == "eval"
    res = {3: RenewalResult(3, [SeqStat("h1", 0, 0, 0, [0] * 4),
                                SeqStat("h2", 4, 2, 2, [0, 2, 0, 0])])}
    out = renewal_output(res, {"h1": "code", "h2": "prose"}, {"ckpt_digest": "d"})
    assert out["ckpt_digest"] == "d"
    assert out["k3"]["by_bucket"]["code"] == {"sequences": 1, "verifies": 0,
                                              "accept_len": None}
    assert out["k3"]["by_bucket"]["prose"]["accept_len"] == 2.0


# ====================================================== (iii) unmeasured
def test_eval_summary_keeps_unmeasured_apart_from_zero():
    from mtptrain.evaluate import EvalResult
    from mtptrain.renewal import RenewalResult

    s = EvalResult(accept={3: [0.0, 0.0], 15: []}, windows={3: 2, 15: 0}).summary()
    assert s["accept@3"] == 0.0 and s["accept@3_windows"] == 2
    assert s["accept@15"] is None and s["accept@15_windows"] == 0
    assert s["agreement@1"] is None and s["ce"] is None and s["ppl"] is None
    r = RenewalResult(3).summary()
    assert r["accept_len"] is None and r["mean_matches"] is None and r["verifies"] == 0


def test_selection_score_is_none_when_a_needed_metric_is_unmeasured():
    from mtptrain.train import selection_score

    assert selection_score({"accept@3": 0.5, "accept@15": None,
                            "agreement@1": 0.9}, "combined")[1] is None
    zero = selection_score({"accept@3": 0.5, "accept@15": 0.0,
                            "agreement@1": 0.9}, "combined")[1]
    assert zero is not None and 0.5 <= zero < 0.51  # a measured 0 is a score
    assert selection_score({"renewal_mean": None}, "renewal")[1] is None
    assert selection_score({}, "3")[1] is None


def test_train_refuses_a_select_metric_the_eval_data_cannot_measure(tmp):
    # --max-len 16: no 15-step chain window fits, so accept@15 never measures
    run = [a if a != "24" else "16" for a in SYN_RUN]
    _exit(lambda: train(_args(*run, "--steps", "2", "--out", os.path.join(tmp, "o"))),
          "--select-metric combined", "accept@15 cannot be measured")
    assert not os.path.exists(os.path.join(tmp, "o"))
    # selecting on accept@3 works; accept@15 is only reported as unmeasured
    stats = train(_args(*run, "--steps", "2", "--out", os.path.join(tmp, "o2"),
                        "--select-metric", "3"))
    ev = [h["eval"] for h in stats["history"] if "eval" in h][-1]
    assert ev["accept@15"] is None and ev["accept@15_windows"] == 0
    assert ev["accept@3"] is not None


def test_an_unmeasured_eval_never_becomes_or_is_compared_with_the_best(tmp, monkeypatch):
    import mtptrain.train as T

    real = T.run_eval
    calls = []

    def run_eval(model, args, samples):
        s = real(model, args, samples)
        calls.append(1)
        if len(calls) == 1:
            s["accept@15"] = None  # as if no window had been scored
        return s

    monkeypatch.setattr(T, "run_eval", run_eval)
    out = os.path.join(tmp, "out")
    stats = train(_args(*SYN_RUN, "--steps", "4", "--out", out))
    evals = [h for h in stats["history"] if "eval" in h]
    assert evals[0]["unscored"] == ["accept@15"] and "unscored" not in evals[1]
    assert stats["best"]["step"] == evals[1]["step"]
    assert json.load(open(os.path.join(out, "best.json")))["step"] == evals[1]["step"]


def test_verify_dump_reports_no_generated_row_as_unmeasured(tmp):
    from extract.verify_dump import verify
    from mtptrain.config import MTPConfig
    from tests.test_renewal import _fake_selfgen_dump

    cfg = MTPConfig.tiny()
    _fake_selfgen_dump(tmp, docs=2, length=24, prompt_tokens=24, vocab=cfg.vocab_size,
                       hc_dim=cfg.hc_size)
    r = verify(tmp)
    assert r["agreement"] is None and r["agreement_rows"] == 0


def test_losses_over_no_row_are_refused_not_zero():
    from mtptrain.loss import chunked_ce, chunked_mixed_ce

    torch.manual_seed(0)
    h, w = torch.randn(6, 8), torch.randn(10, 8)
    none = torch.full((6,), -100)
    with pytest.raises(ValueError, match="no labelled row"):
        chunked_ce(h, w, none)
    labels = torch.randint(0, 10, (6,))
    ids = torch.zeros(6, 3, dtype=torch.long)
    q = torch.zeros(6, 3)
    ok = torch.zeros(6, dtype=torch.bool)
    # alpha > 0 but no soft row: the hard CE alone, not (1-alpha) * hard + 0
    got = chunked_mixed_ce(h, w, labels, ids, q, ok, alpha=0.5)
    assert torch.allclose(got, chunked_ce(h, w, labels), atol=1e-5)
    with pytest.raises(ValueError, match="no labelled row"):
        chunked_mixed_ce(h, w, none, ids, q, ok, alpha=0.5)
