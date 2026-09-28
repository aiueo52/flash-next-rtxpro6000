"""CPU tests for the renewal (serving-process) acceptance evaluator.

Also covers ``mtptrain/select.py`` -- the document picker both
``scripts/eval_renewal.py`` and the training loop's ``--eval-renewal`` go
through, so the number printed mid-training is comparable with an offline run.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mtptrain.config import MTPConfig  # noqa: E402
from mtptrain.renewal import renewal_batch  # noqa: E402
from mtptrain.select import (  # noqa: E402
    load_selfgen_samples,
    select_selfgen_docs,
)
from tests.test_dump_roundtrip import _write_dump  # noqa: E402
from tests.test_rollout import _batch, _model  # noqa: E402


def _fake_selfgen_dump(tmp: str, docs: int = 6, length: int = 24,
                       prompt_tokens: int = 8, vocab: int = 64,
                       hc_dim: int = 128, topk: int = 0, consistent: bool = False):
    """A dump directory plus the selfgen manifest ``select.py`` reads.

    ``consistent`` makes the target's argmax equal the next dumped token, as
    on real self-generated greedy text, so chain windows are scorable."""
    from mtptrain.data import load_dump_file

    records = []
    for i in range(docs):
        path = os.path.join(tmp, f"dump-1-{i:08d}.safetensors")
        ids = [(i * 7 + j) % vocab for j in range(length)]
        tgt = ([(i * 7 + j + 1) % vocab for j in range(length)] if consistent
               else [(i * 5 + j) % vocab for j in range(length)])
        _write_dump(path, [(ids, tgt, 0)], hc_dim=hc_dim, topk=topk)
        for s in load_dump_file(path):
            records.append({
                "doc_hash": s.doc_hash, "split_key": s.doc_hash,
                "mode": "selfgen", "prompt_tokens": prompt_tokens,
                "generated_tokens": length - prompt_tokens,
                "bucket": "code" if i % 2 else "prose",
            })
    with open(os.path.join(tmp, "manifest.jsonl"), "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return records


def test_token_accounting_and_bounds():
    cfg, m, g = _model()
    _, samples = _batch(cfg, g, b=3, n=14)
    starts = [3, 0, 9]
    for k in (1, 3, 5):
        res = renewal_batch(m, samples, starts, k)
        assert len(res.seqs) == 3
        for s, st, smp in zip(res.seqs, starts, samples):
            # the first generated token comes from the prefill; each verify
            # produces a+1 tokens, except that a final all-accepted verify has
            # no bonus token past the end of generation
            assert len(smp) - st == s.gen_rows
            assert s.gen_rows - 1 <= s.accepted + s.verifies <= s.gen_rows
            assert sum(s.hist) == s.verifies and len(s.hist) == k + 1
            assert 1.0 < s.accept_len <= k + 2
        summ = res.summary()
        assert summ["tokens"] == sum(len(x) - st for x, st in zip(samples, starts))


def test_perfect_draft_accepts_everything():
    """If the labels are what the chain predicts, a = k on every full window."""
    cfg, m, g = _model()
    _, samples = _batch(cfg, g, b=1, n=10)
    s = samples[0]
    k = 3
    # Make labels equal to the chain's own predictions from row 0, step by step,
    # by running the renewal once and copying predictions back as labels.
    from mtptrain.data import collate

    batch = collate([s])
    _, _, kv = m.forward_with_cache(batch["next_token_ids"], batch["hc_hidden"], batch["positions"], return_logits=False)
    cur = 0
    while cur <= len(s) - 1:
        preds = m.chain(batch["hc_hidden"][:, cur], batch["next_token_ids"][:, cur], batch["positions"][:, cur],
                        kv, steps=k, past_mask=torch.arange(len(s)).view(1, -1) < cur)[0]
        for j in range(k):
            if cur + j <= len(s) - 1:
                s.labels[cur + j] = preds[j]
        cur += k + 1
    res = renewal_batch(m, [s], [0], k)
    st = res.seqs[0]
    # 10 rows: windows at 0 (3 accepted), 4 (3), 8 (1 capped) -> 3 verifies
    assert st.verifies == 3 and st.accepted == 7
    assert res.summary()["accept_len"] == 10 / 3


def _perfect_labels(m, s, k):
    from mtptrain.data import collate

    batch = collate([s])
    _, _, kv = m.forward_with_cache(batch["next_token_ids"], batch["hc_hidden"], batch["positions"], return_logits=False)
    cur = 0
    while cur <= len(s) - 1:
        preds = m.chain(batch["hc_hidden"][:, cur], batch["next_token_ids"][:, cur], batch["positions"][:, cur],
                        kv, steps=k, past_mask=torch.arange(len(s)).view(1, -1) < cur)[0]
        for j in range(k):
            if cur + j <= len(s) - 1:
                s.labels[cur + j] = preds[j]
        cur += k + 1


def test_generation_end_is_not_an_extra_verify():
    """5 generated tokens, k=3, all accepted: the prefill gives the first token
    and ONE verify gives the other four, so accept_len is 5, not 2.5."""
    cfg, m, g = _model()
    _, samples = _batch(cfg, g, b=1, n=5)
    s = samples[0]
    _perfect_labels(m, s, 3)
    res = renewal_batch(m, [s], [0], 3)
    st = res.seqs[0]
    assert st.gen_rows == 5
    assert st.verifies == 1 and st.accepted == 3 and st.hist == [0, 0, 0, 1]
    assert st.accept_len == 5.0 and res.summary()["accept_len"] == 5.0


def test_a_single_generated_token_has_no_verify():
    cfg, m, g = _model()
    _, samples = _batch(cfg, g, b=1, n=4)
    res = renewal_batch(m, samples, [3], 3)
    st = res.seqs[0]
    assert st.gen_rows == 1 and st.verifies == 0 and st.accept_len is None
    assert res.summary()["accept_len"] is None


def test_random_labels_mostly_reject():
    cfg, m, g = _model()
    _, samples = _batch(cfg, g, b=2, n=20)
    res = renewal_batch(m, samples, [0, 0], 3)
    assert res.summary()["accept_len"] < 1.5


# ------------------------------------------------------------ select.py
def test_select_selfgen_docs_caps_per_bucket_and_is_deterministic():
    tmp = tempfile.mkdtemp(prefix="mtpsel-")
    try:
        _fake_selfgen_dump(tmp, docs=8)
        kw = dict(split="all", min_gen=4)
        allofthem, index = select_selfgen_docs(tmp, **kw)
        assert len(allofthem) == 8 and len(index) == 8
        capped, _ = select_selfgen_docs(tmp, per_bucket=2, **kw)
        assert len(capped) == 4  # 2 buckets x 2
        again, _ = select_selfgen_docs(tmp, per_bucket=2, **kw)
        assert [r["doc_hash"] for r in again] == [r["doc_hash"] for r in capped]
        only, _ = select_selfgen_docs(tmp, buckets=["prose"], **kw)
        assert {r["bucket"] for r in only} == {"prose"}
    finally:
        shutil.rmtree(tmp)


def test_select_greedy_only_drops_sampled_documents():
    tmp = tempfile.mkdtemp(prefix="mtpsel-")
    try:
        recs = _fake_selfgen_dump(tmp, docs=6)
        for i, r in enumerate(recs):
            r["temperature"] = 0.7 if i % 3 == 0 else 0.0
        with open(os.path.join(tmp, "manifest.jsonl"), "w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        allofthem, _ = select_selfgen_docs(tmp, split="all", min_gen=4)
        greedy, _ = select_selfgen_docs(tmp, split="all", min_gen=4, greedy_only=True)
        assert len(allofthem) == 6 and len(greedy) == 4
    finally:
        shutil.rmtree(tmp)


def test_select_split_partitions_the_documents():
    tmp = tempfile.mkdtemp(prefix="mtpsel-")
    try:
        _fake_selfgen_dump(tmp, docs=12)
        kw = dict(min_gen=4, holdout_frac=0.5)
        ev, _ = select_selfgen_docs(tmp, split="eval", **kw)
        tr, _ = select_selfgen_docs(tmp, split="train", **kw)
        assert ev and tr
        assert not ({r["doc_hash"] for r in ev} & {r["doc_hash"] for r in tr})
        assert len(ev) + len(tr) == 12
    finally:
        shutil.rmtree(tmp)


def test_load_selfgen_samples_marks_the_generated_region():
    tmp = tempfile.mkdtemp(prefix="mtpsel-")
    try:
        _fake_selfgen_dump(tmp, docs=4, length=30, prompt_tokens=10)
        chosen, index = select_selfgen_docs(tmp, split="all", min_gen=4)
        sel = load_selfgen_samples(tmp, chosen, index)
        assert len(sel) == 4
        assert set(sel.gen_start_rows) == {9}      # prompt_tokens - 1
        assert sel.rows == 4 * 29
        assert sel.generated_rows == 4 * 20
        assert sel.counts() == {"code": 2, "prose": 2}
    finally:
        shutil.rmtree(tmp)



def test_load_selfgen_samples_refuses_a_changed_document():
    """1 of 2 documents mismatched: refused, never evaluated on the rest."""
    tmp = tempfile.mkdtemp(prefix="mtpsel-")
    try:
        _fake_selfgen_dump(tmp, docs=2, length=30, prompt_tokens=10)
        chosen, index = select_selfgen_docs(tmp, split="all", min_gen=4)
        assert len(chosen) == 2
        # point the first document's index entry at the second document's slot
        index = dict(index)
        index[chosen[0]["doc_hash"]] = index[chosen[1]["doc_hash"]]
        try:
            load_selfgen_samples(tmp, chosen, index)
        except SystemExit as exc:
            assert "changed" in str(exc)
        else:
            raise AssertionError("a mismatched eval document was skipped instead of refused")
    finally:
        shutil.rmtree(tmp)


# --------------------------------------------------- --eval-renewal in train
def test_training_loop_renewal_eval_selects_on_accept_len():
    from mtptrain.train import build_parser, train

    tmp = tempfile.mkdtemp(prefix="mtprenew-")
    out = os.path.join(tmp, "out")
    try:
        cfg = MTPConfig.tiny()
        _fake_selfgen_dump(tmp, docs=6, length=24, prompt_tokens=8,
                           vocab=cfg.vocab_size, hc_dim=cfg.hc_size, topk=6)
        args = build_parser().parse_args(
            ["--tiny", "--synthetic", "4", "--steps", "3", "--batch-size", "2",
             "--max-len", "24", "--warmup", "1", "--log-every", "3",
             "--eval-every", "3", "--eval-batches", "1", "--eval-starts", "2",
             "--eval-renewal", "--renewal-dump-dir", tmp,
             "--renewal-ks", "3", "15", "--renewal-docs", "2",
             "--renewal-min-gen", "4", "--holdout", "1.0",
             "--soft-alpha", "0.5", "--out", out, "--device", "cpu"]
        )
        assert args.select_metric is None
        stats = train(args)
        assert args.select_metric == "renewal"   # resolved from --eval-renewal
        evals = [h["eval"] for h in stats["history"] if "eval" in h]
        assert evals, stats["history"]
        summary = evals[-1]
        for key in ("accept_len@3", "accept_len@15", "renewal_mean"):
            assert key in summary, summary
        assert 1.0 <= summary["accept_len@3"] <= 4.0
        assert set(summary["by_bucket@3"]) <= {"code", "prose"}
        # the frozen-eval-set metrics are still reported
        assert "agreement@1" in summary and "accept@3" in summary
        assert stats["best"]["criterion"] == "renewal"
        assert stats["best"]["score"] > 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            import traceback

            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{len(fns) - failures}/{len(fns)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
