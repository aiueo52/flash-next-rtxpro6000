"""build_index on synthetic multi-chunk dumps (CPU).

Two hook-format dump directories with requests that chunked prefill split
over two or three consecutive files, a source document cut into two windows
(two manifest rows sharing a ``split_key``), a file no manifest row
references, and a manifest row that was never dumped.  The index must join
the pieces of a request within one write stream only, ``verify_dump`` must
count the undumped row without failing, and an orphan continuation or an
unreadable dump file must be refused, not skipped.
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

from extract.build_index import file_stream, scan  # noqa: E402
from mtptrain.data import _hash_ids  # noqa: E402
from tests.test_dump_roundtrip import _write_dump  # noqa: E402


def _ids(seed: int, n: int) -> list:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 5000, (n,), generator=g).tolist()


def _piece(ids: list, a: int, b: int):
    """(ids, target, start position) of ids[a:b] -- one dump sequence slot."""
    return (ids[a:b], [(x + 1) % 5000 for x in ids[a:b]], a)


def _make_dump(root: str, pid: int, layout: list, rows: list) -> None:
    """``layout``: per file, a list of sequence slots; ``rows``: manifest."""
    os.makedirs(root, exist_ok=True)
    for i, seqs in enumerate(layout, start=1):
        _write_dump(os.path.join(root, f"dump-{pid}-{i:08d}.safetensors"), seqs)
    with open(os.path.join(root, "manifest.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _row(ids: list, split_key: str, bucket: str, chunk: int = 0) -> dict:
    return {"doc_hash": _hash_ids(ids), "split_key": split_key, "chunk": chunk,
            "tokens": len(ids), "mode": "corpus", "bucket": bucket}


def _fixture(tmp: str) -> dict:
    a1, a2, a3, a4, a5 = (_ids(s, n) for s, n in ((1, 20), (2, 32), (3, 24), (4, 30), (5, 18)))
    warm = _ids(6, 10)
    src_a = os.path.join(tmp, "dump-a")
    _make_dump(src_a, 100, [
        [_piece(a1, 0, 20), _piece(a2, 0, 16)],      # a2 starts here ...
        [_piece(a2, 16, 32), _piece(a3, 0, 12)],     # ... ends here; a3 starts
        [_piece(a3, 12, 24)],                        # a3 ends
        [_piece(warm, 0, 10)],                       # in no manifest row
        [_piece(a4, 0, 30)],
    ], [
        _row(a1, "doc-a1", "code"),
        _row(a2, "doc-x", "chat", 0),                # two windows of one document
        _row(a3, "doc-x", "chat", 1),
        _row(a4, "doc-a4", "code"),
        _row(a5, "doc-a5", "code"),                  # never dumped
    ])
    b1, b2, b3 = _ids(11, 25), _ids(12, 16), _ids(13, 20)
    src_b = os.path.join(tmp, "dump-b")
    _make_dump(src_b, 200, [
        [_piece(b1, 0, 10)],
        [_piece(b1, 10, 20), _piece(b2, 0, 16)],
        [_piece(b1, 20, 25)],                        # b1 spans three files
        [_piece(b3, 0, 20)],
    ], [
        _row(b1, "doc-b1", "selfgen"),
        _row(b2, "doc-b2", "selfgen"),
        _row(b3, "doc-b3", "selfgen"),
    ])
    return {"a": src_a, "b": src_b,
            "docs": {"a1": a1, "a2": a2, "a3": a3, "a4": a4, "a5": a5,
                     "b1": b1, "b2": b2, "b3": b3}}


def test_build_index_joins_chunks_within_a_stream_only():
    tmp = tempfile.mkdtemp(prefix="mtpidx-")
    try:
        fx = _fixture(tmp)
        index, _n, bad = scan(fx["a"])
        assert not bad
        a2 = index[_hash_ids(fx["docs"]["a2"])]
        assert a2["files"] == ["dump-100-00000001.safetensors", "dump-100-00000002.safetensors"]
        assert a2["seqs"] == [1, 0] and a2["tokens"] == 32
        # the shorter prefix entry is not rewritten when a chunk extends it
        head = index[_hash_ids(fx["docs"]["a2"][:16])]
        assert head["files"] == ["dump-100-00000001.safetensors"] and head["tokens"] == 16
        assert _hash_ids(fx["docs"]["a5"]) not in index
        # a continuation in another process's files is not joined
        assert file_stream("dump-100-00000007.safetensors") == "dump-100"
        other = os.path.join(tmp, "two-streams")
        x = _ids(21, 20)
        os.makedirs(other)
        _write_dump(os.path.join(other, "dump-1-00000001.safetensors"), [_piece(x, 0, 10)])
        _write_dump(os.path.join(other, "dump-2-00000001.safetensors"), [_piece(x, 10, 20)])
        # ... it is an orphan in its own stream, and orphans are refused
        try:
            scan(other)
        except SystemExit as exc:
            assert "without their start" in str(exc) and "dump-2-00000001" in str(exc)
        else:
            raise AssertionError("an orphan continuation was accepted")
    finally:
        shutil.rmtree(tmp)


def test_verify_dump_counts_rows_that_were_never_dumped():
    from extract.verify_dump import verify

    tmp = tempfile.mkdtemp(prefix="mtpidx-")
    try:
        fx = _fixture(tmp)
        rep = verify(fx["a"], greedy_only=False)
        assert rep["mismatch"] == 0 and rep["missing"] == 1  # a5
        assert rep["checked"] == 4
    finally:
        shutil.rmtree(tmp)


def test_orphan_continuation_is_refused():
    tmp = tempfile.mkdtemp(prefix="mtpidx-")
    try:
        orphan = _ids(7, 10)
        _make_dump(tmp, 300, [[_piece(_ids(8, 12), 0, 12)], [(orphan, orphan, 16)]], [])
        try:
            scan(tmp)
        except SystemExit as exc:
            assert "radix cache" in str(exc)
        else:
            raise AssertionError("an orphan continuation was skipped instead of refused")
    finally:
        shutil.rmtree(tmp)


def test_unreadable_dump_file_is_refused():
    tmp = tempfile.mkdtemp(prefix="mtpidx-")
    try:
        _make_dump(tmp, 400, [[_piece(_ids(9, 12), 0, 12)]], [])
        with open(os.path.join(tmp, "dump-400-00000002.safetensors"), "wb") as fh:
            fh.write(b"\x10\x00\x00")  # a partial file left by a crash
        try:
            scan(tmp)
        except SystemExit as exc:
            assert "unreadable" in str(exc) and "dump-400-00000002" in str(exc)
        else:
            raise AssertionError("an unreadable dump file was skipped instead of refused")
    finally:
        shutil.rmtree(tmp)
