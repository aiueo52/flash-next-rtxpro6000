"""Index staleness and document identity (CPU).

A dump file replaced in place must not be read through an index built from
the old file: ``index.files.json`` records each file's size, mtime_ns,
ctime_ns, inode and device, and every document is re-hashed when it is loaded.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extract.build_eval_set import build_eval_set  # noqa: E402
from extract.verify_dump import verify  # noqa: E402
from mtptrain.data import DumpDataset  # noqa: E402
from mtptrain.dumpindex import (  # noqa: E402
    DocumentMismatch,
    is_current,
    load_document,
    load_index,
)
from tests.test_dump_roundtrip import _write_dump  # noqa: E402
from tests.test_joined_docs import _fixture, _ids, _piece  # noqa: E402

LAST = "dump-7-00000004.safetensors"  # holds d4 alone


def _replace_same_size(root: str, keep_mtime: bool, same_inode: bool = True) -> None:
    """Rewrite d4's file with different ids of the same length (in place, so
    the inode stays, unless ``same_inode`` is False)."""
    path = os.path.join(root, LAST)
    st = os.stat(path)
    tmp = path + ".new"
    _write_dump(tmp, [_piece(_ids(99, 25), 0, 25)])
    if same_inode:
        with open(tmp, "rb") as src, open(path, "r+b") as dst:
            dst.write(src.read())
        os.remove(tmp)
        assert os.stat(path).st_ino == st.st_ino
    else:
        os.replace(tmp, path)
    assert os.path.getsize(path) == st.st_size
    if keep_mtime:
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    else:
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))


def _expect_exit(fn, *words):
    try:
        fn()
    except SystemExit as exc:
        for w in words:
            assert w in str(exc), str(exc)
    else:
        raise AssertionError("expected a SystemExit")


def test_same_size_replacement_with_new_mtime_rebuilds_the_index():
    tmp = tempfile.mkdtemp(prefix="mtpidx-mtime-")
    try:
        h = _fixture(tmp)
        assert h["d4"] in load_index(tmp) and is_current(tmp)
        _replace_same_size(tmp, keep_mtime=False)
        assert not is_current(tmp)
        index = load_index(tmp)
        assert h["d4"] not in index and is_current(tmp)
        # the manifest document is simply absent now, and verify says so
        assert verify(tmp)["missing"] == 1
    finally:
        shutil.rmtree(tmp)


def test_a_new_file_at_the_same_size_and_mtime_makes_the_index_stale():
    """The index is keyed on inode and device too, not only name/size/mtime."""
    tmp = tempfile.mkdtemp(prefix="mtpidx-inode-")
    try:
        _fixture(tmp)
        load_index(tmp)
        _replace_same_size(tmp, keep_mtime=True, same_inode=False)
        assert not is_current(tmp)
    finally:
        shutil.rmtree(tmp)


def test_in_place_rewrite_with_the_mtime_restored_makes_the_index_stale():
    """Same name, size, mtime and inode: the ctime still moves."""
    tmp = tempfile.mkdtemp(prefix="mtpidx-ctime-")
    try:
        h = _fixture(tmp)
        load_index(tmp)
        _replace_same_size(tmp, keep_mtime=True)
        assert not is_current(tmp)
        assert h["d4"] not in load_index(tmp) and is_current(tmp)
    finally:
        shutil.rmtree(tmp)


def _pretend_current(root: str) -> None:
    """Re-stamp index.files.json with the files' present state, as if the
    index had been built after a change it does not reflect."""
    import json

    from mtptrain.dumpindex import FILES_FILE, _disk_state

    path = os.path.join(root, FILES_FILE)
    meta = json.load(open(path))
    meta["files"] = _disk_state(root)
    json.dump(meta, open(path, "w"))


def test_tampered_document_is_detected_by_every_reader():
    tmp = tempfile.mkdtemp(prefix="mtpidx-tamper-")
    try:
        h = _fixture(tmp)
        index = load_index(tmp)
        _replace_same_size(tmp, keep_mtime=True)
        _pretend_current(tmp)  # an index that does not describe the files
        assert is_current(tmp)
        # the loader
        try:
            load_document(tmp, index[h["d4"]], expect=h["d4"])
        except DocumentMismatch as exc:
            assert h["d4"] in str(exc)
        else:
            raise AssertionError("a changed document was loaded")
        # the dataset refuses it (a skip would silently shrink the data)
        ds = DumpDataset(tmp, "train", holdout_frac=0.0)
        _expect_exit(lambda: [s.doc_hash for s in ds], "changed after it was indexed")
        # verify_dump and build_eval_set refuse
        _expect_exit(lambda: verify(tmp), "do not match")
        _expect_exit(lambda: build_eval_set(tmp, os.path.join(tmp, "e.json"),
                                            holdout=1.0), h["d4"])
        assert not os.path.exists(os.path.join(tmp, "e.json"))
        # a rebuilt index no longer holds that document
        load_index(tmp, reindex=True)
        assert verify(tmp)["mismatch"] == 0
    finally:
        shutil.rmtree(tmp)


def test_retargeted_symlink_makes_the_index_stale():
    tmp = tempfile.mkdtemp(prefix="mtpidx-link-")
    try:
        src = os.path.join(tmp, "src")
        _fixture(src)
        view = os.path.join(tmp, "view")
        os.makedirs(view)
        os.symlink(os.path.join(src, LAST), os.path.join(view, LAST))
        load_index(view)
        other = os.path.join(tmp, "other.safetensors")
        shutil.copy2(os.path.join(src, LAST), other)  # same size and mtime
        os.remove(os.path.join(view, LAST))
        os.symlink(other, os.path.join(view, LAST))
        assert not is_current(view)
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
