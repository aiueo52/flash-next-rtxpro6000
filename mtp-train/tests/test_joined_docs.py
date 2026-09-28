"""Documents split over several dump files are loaded whole on every path (CPU).

Chunked prefill writes one request as pieces in consecutive dump files, and
the manifest's ``doc_hash`` is the hash of the *whole* request.  These tests
build a dump where two of four documents are split that way and check that
the training/eval dataset, the frozen eval set, the renewal selection and the
index itself all see four whole documents -- none of them dropped because a
single piece's hash does not match the manifest.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import pytest

import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extract.build_eval_set import build_eval_set  # noqa: E402
from mtptrain.data import DumpDataset, _hash_ids, load_fixed_eval  # noqa: E402
from mtptrain.dumpindex import documents, is_current, load_index  # noqa: E402
from mtptrain.select import (  # noqa: E402
    is_holdout,
    load_selfgen_samples,
    select_selfgen_docs,
)
from tests.test_dump_roundtrip import _write_dump  # noqa: E402

PROMPT = 8


def _ids(seed: int, n: int) -> list:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 5000, (n,), generator=g).tolist()


def _piece(ids: list, a: int, b: int):
    return (ids[a:b], [(x + 1) % 5000 for x in ids[a:b]], a)


def _fixture(root: str) -> dict:
    """d1, d4 in one file each; d2 over files 1-2; d3 over files 2-3."""
    docs = {"d1": _ids(1, 30), "d2": _ids(2, 40), "d3": _ids(3, 45), "d4": _ids(4, 25)}
    layout = [
        [_piece(docs["d1"], 0, 30), _piece(docs["d2"], 0, 20)],
        [_piece(docs["d2"], 20, 40), _piece(docs["d3"], 0, 15)],
        [_piece(docs["d3"], 15, 45)],
        [_piece(docs["d4"], 0, 25)],
    ]
    os.makedirs(root, exist_ok=True)
    for i, seqs in enumerate(layout, start=1):
        _write_dump(os.path.join(root, f"dump-7-{i:08d}.safetensors"), seqs)
    with open(os.path.join(root, "manifest.jsonl"), "w") as fh:
        for name, ids in docs.items():
            fh.write(json.dumps({
                "doc_hash": _hash_ids(ids), "split_key": f"doc-{name}", "chunk": 0,
                "tokens": len(ids), "prompt_tokens": PROMPT,
                "generated_tokens": len(ids) - PROMPT, "temperature": 0.0,
                "mode": "selfgen", "bucket": "code" if name in ("d1", "d2") else "prose",
            }) + "\n")
    return {k: _hash_ids(v) for k, v in docs.items()}


def test_dump_dataset_yields_joined_documents_on_both_sides_of_the_split():
    tmp = tempfile.mkdtemp(prefix="mtpjoin-ds-")
    try:
        h = _fixture(tmp)
        everything = list(DumpDataset(tmp, "train", holdout_frac=0.0))
        assert [s.doc_hash for s in everything] == [h["d1"], h["d2"], h["d3"], h["d4"]]
        by_hash = {s.doc_hash: s for s in everything}
        d2 = by_hash[h["d2"]]
        assert len(d2) == 39 and len(d2.pieces) == 2
        assert torch.equal(d2.positions, torch.arange(39))
        assert len(by_hash[h["d3"]]) == 44
        assert [s.doc_hash for s in DumpDataset(tmp, "eval", holdout_frac=1.0)] == [
            h["d1"], h["d2"], h["d3"], h["d4"]]
        # a real split: by split_key, disjoint, complete, same rule as select.py
        for frac in (0.3, 0.5, 0.7):
            train = {s.doc_hash for s in DumpDataset(tmp, "train", holdout_frac=frac)}
            evals = {s.doc_hash for s in DumpDataset(tmp, "eval", holdout_frac=frac)}
            assert not train & evals and train | evals == set(h.values())
            for name, doc_hash in h.items():
                assert (doc_hash in evals) == is_holdout(f"doc-{name}", 20260903, frac)
        # bucket filters act on whole documents too
        only = {s.doc_hash for s in DumpDataset(tmp, "train", holdout_frac=0.0,
                                                include_buckets=["prose"])}
        assert only == {h["d3"], h["d4"]}
    finally:
        shutil.rmtree(tmp)


def test_build_eval_set_and_fixed_eval_keep_split_documents():
    tmp = tempfile.mkdtemp(prefix="mtpjoin-eval-")
    try:
        h = _fixture(tmp)
        out = os.path.join(tmp, "eval_fixed.json")
        summary = build_eval_set(tmp, out, rows=10_000, holdout=1.0)
        assert summary["sequences"] == 4
        frozen = json.load(open(out))
        pieces = {d["doc_hash"]: d for d in frozen["docs"]}
        assert len(pieces[h["d2"]]["files"]) == 2 and pieces[h["d2"]]["seqs"] == [1, 0]
        assert len(pieces[h["d3"]]["files"]) == 2
        assert frozen["rows"] == 29 + 39 + 44 + 24
        loaded = load_fixed_eval(out)
        assert sorted(s.doc_hash for s in loaded) == sorted(h.values())
        assert {s.doc_hash: len(s) for s in loaded}[h["d3"]] == 44
        # an older eval set (files + hashes only) still loads its single-file docs
        legacy = os.path.join(tmp, "legacy.json")
        json.dump({"files": frozen["files"], "doc_hashes": [h["d1"], h["d4"]]},
                  open(legacy, "w"))
        assert {s.doc_hash for s in load_fixed_eval(legacy)} == {h["d1"], h["d4"]}
        # ... but a document it cannot find is an error, not a smaller eval set
        json.dump({"files": frozen["files"], "doc_hashes": frozen["doc_hashes"]},
                  open(legacy, "w"))
        with pytest.raises(SystemExit, match="no longer in their files"):
            load_fixed_eval(legacy)
        json.dump({"files": frozen["files"] + [os.path.join(tmp, "gone.safetensors")],
                   "doc_hashes": [h["d1"]]}, open(legacy, "w"))
        with pytest.raises(SystemExit, match="is gone"):
            load_fixed_eval(legacy)
    finally:
        shutil.rmtree(tmp)


def test_select_loads_split_selfgen_documents():
    tmp = tempfile.mkdtemp(prefix="mtpjoin-sel-")
    try:
        h = _fixture(tmp)
        chosen, index = select_selfgen_docs(tmp, split="all", min_gen=4)
        assert {r["doc_hash"] for r in chosen} == set(h.values())
        sel = load_selfgen_samples(tmp, chosen, index)
        assert len(sel) == 4 and set(sel.gen_start_rows) == {PROMPT - 1}
        lengths = {s.doc_hash: len(s) for s in sel.samples}
        assert lengths[h["d2"]] == 39 and lengths[h["d3"]] == 44
        # with a length cap the generated region still has to survive
        capped = load_selfgen_samples(tmp, chosen, index, max_len=20)
        assert all(len(s) == 20 for s in capped.samples) and len(capped) == 4
    finally:
        shutil.rmtree(tmp)


def test_index_is_rebuilt_whole_when_the_dump_grows():
    """The first piece of a document must be re-read when its continuation
    arrives in a later file -- an incremental index that skipped already
    indexed files could never join them."""
    tmp = tempfile.mkdtemp(prefix="mtpjoin-idx-")
    try:
        h = _fixture(tmp)
        later = [os.path.join(tmp, f"dump-7-{i:08d}.safetensors") for i in (2, 3, 4)]
        parked = tempfile.mkdtemp(prefix="mtpjoin-parked-")  # outside the dump dir
        for p in later:
            shutil.move(p, parked)
        first = load_index(tmp)
        assert h["d1"] in first and h["d2"] not in first  # only d2's first piece so far
        assert is_current(tmp)
        for p in later:
            shutil.move(os.path.join(parked, os.path.basename(p)), tmp)
        os.rmdir(parked)
        assert not is_current(tmp)  # new files: the index on disk is stale
        grown = load_index(tmp)
        assert {h[k] for k in h} <= set(grown)
        assert grown[h["d2"]]["files"] == ["dump-7-00000001.safetensors",
                                           "dump-7-00000002.safetensors"]
        assert [d for d, _ in documents(grown, set(h.values()))] == [
            h["d1"], h["d2"], h["d3"], h["d4"]]
        # --reindex style rebuild gives the same answer
        assert load_index(tmp, reindex=True) == grown
        # an index.json written by an older version (no file list) is rebuilt
        os.remove(os.path.join(tmp, "index.files.json"))
        json.dump({}, open(os.path.join(tmp, "index.json"), "w"))
        assert load_index(tmp) == grown
    finally:
        shutil.rmtree(tmp)


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
