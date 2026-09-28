"""The document index of a dump directory: doc_hash -> the pieces that hold it.

Chunked prefill splits one request over several consecutive extend forwards,
so one document can live in several dump files.  Every consumer that matches
dump sequences against the manifest's ``doc_hash`` (the hash of the request's
*full* token ids) must therefore look at joined documents, never at the raw
per-file sequences: the training and eval datasets (``data.DumpDataset``),
the frozen eval set (``extract/build_eval_set.py``, ``data.load_fixed_eval``),
the renewal selection (``select.py``) and ``extract/verify_dump.py``.
They all go through this module.

``index.json`` maps ``doc_hash -> {"file", "seq", "tokens", "files", "seqs"}``:
the first dump file and sequence slot, the length, and every piece in order.
Every prefix of a chunked request is indexed too (its first piece, the first
two, ...); ``documents()`` keeps only the longest one per starting slot.

Chunked prefill continues a request in the next forward of the *same server
process*, so pieces are joined only within one file stream: the hook names
its files ``{tag}-{pid}-{counter:08d}.safetensors`` and the stream is the name
without the trailing counter.  Within a stream, files are read in sorted = write order.

The index is always built by a full scan (reading only ``input_ids`` and
``positions``), because a request's later pieces can only be joined when its
first piece is read in the same pass.  ``index.files.json`` next to it records
the dump files the index was built from -- name, size, ``mtime_ns``,
``ctime_ns``, inode, device and, for a symlink, its target.  The index is a
cache of those files' ``input_ids``/``positions`` keyed on that state:
``load_index`` rebuilds it whenever any of it differs for any file, and never
trusts it across a changed key.  A file rewritten in place with its size and
mtime restored still has a new ctime.

Whatever the index says, a document is only accepted after its identity is
checked: ``load_document(..., expect=doc_hash)`` recomputes the hash of the
joined ``input_ids`` (the same hash the manifest uses) and raises
``DocumentMismatch`` when it differs.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .data import Sample, _hash_ids, iter_dump_files, load_concat
from .fileio import write_json_atomic

INDEX_FILE = "index.json"
FILES_FILE = "index.files.json"
INDEX_VERSION = 5


class DocumentMismatch(ValueError):
    """The ids read from disk do not hash to the doc_hash the index promised."""


def file_stream(rel: str) -> str:
    """The write stream a dump file belongs to (its name minus the counter)."""
    head, base = os.path.split(rel)
    stem = base[: -len(".safetensors")] if base.endswith(".safetensors") else base
    return os.path.join(head, stem.rsplit("-", 1)[0] if "-" in stem else "")


def _read_index_tensors(path: str):
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as f:
        ids = f.get_tensor("input_ids")
        pos = f.get_tensor("positions")
        offs = (f.get_tensor("seq_offsets").tolist()
                if "seq_offsets" in f.keys() else [0, ids.numel()])
    return ids, pos, offs


def scan(dump_dir: str) -> Tuple[dict, int, List[str]]:
    """-> (index, files read, unreadable files); writes nothing.

    An unreadable file (truncated, or a dangling symlink) is refused: skipping
    it would silently drop the requests it held -- and any continuation of
    them -- from every later step.  Move or repair it and index again.
    """
    if not os.path.isdir(dump_dir):
        raise SystemExit(f"dump dir {dump_dir!r}: no such directory")
    index: dict = {}
    bad: List[str] = []
    orphans: List[str] = []
    n = 0
    # A chunk whose first position is not 0 continues the pending chunk of the
    # same stream that ended right before it.  ``pending`` maps (stream,
    # next-expected position) -> partial doc.
    pending: dict = {}
    for path in iter_dump_files(dump_dir):
        rel = os.path.relpath(path, dump_dir)
        try:
            ids, pos, offs = _read_index_tensors(path)
        except Exception as exc:  # noqa: BLE001
            print(f"[index] unreadable {rel}: {type(exc).__name__}: {exc}")
            bad.append(rel)
            continue
        stream = file_stream(rel)
        for s in range(len(offs) - 1):
            a, b = int(offs[s]), int(offs[s + 1])
            if b <= a:
                continue
            p0, p1 = int(pos[a]), int(pos[b - 1])
            if p0 == 0:
                doc = {"files": [rel], "seqs": [s], "ids": ids[a:b].tolist()}
            elif (stream, p0) in pending:
                doc = pending.pop((stream, p0))
                doc["files"].append(rel)
                doc["seqs"].append(s)
                doc["ids"] += ids[a:b].tolist()
            else:
                orphans.append(f"{rel}#{s}")  # continuation without its start
                continue
            # Copies: a later chunk extends ``doc`` but must not change the
            # entry already recorded for this shorter prefix.
            index[_hash_ids(doc["ids"])] = {
                "file": doc["files"][0], "seq": doc["seqs"][0],
                "tokens": len(doc["ids"]),
                "files": list(doc["files"]), "seqs": list(doc["seqs"]),
            }
            pending[(stream, p1 + 1)] = doc  # a later chunk may extend it further
        n += 1
    if bad:
        raise SystemExit(
            f"[index] {dump_dir}: {len(bad)} unreadable dump file(s) ({', '.join(bad[:5])}"
            f"{', ...' if len(bad) > 5 else ''}). Skipping them would silently drop their "
            "documents; move or repair them (a crashed run can leave a partial last file) "
            "and index again.")
    if orphans:
        raise SystemExit(
            f"[index] {dump_dir}: {len(orphans)} continuation chunk(s) without their start "
            f"({', '.join(orphans[:5])}{', ...' if len(orphans) > 5 else ''}). This happens when "
            "the dump server ran with the radix cache on; re-extract with it off.")
    return index, n, bad


def _disk_state(dump_dir: str) -> Dict[str, list]:
    """{file: [size, mtime_ns, ctime_ns, inode, device, link target or None]}.

    The index is a cache of what the files' ``input_ids``/``positions`` say,
    keyed on this state: any change of it rebuilds the index, and nothing in
    an index is trusted across a changed key."""
    out = {}
    for path in iter_dump_files(dump_dir):
        try:
            st = os.stat(path)
            key = [st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino, st.st_dev]
        except OSError:
            key = [-1, -1, -1, -1, -1]  # dangling symlink
        target = os.path.realpath(path) if os.path.islink(path) else None
        out[os.path.relpath(path, dump_dir)] = key + [target]
    return out


def _files_path(out: str) -> str:
    return os.path.join(os.path.dirname(out), FILES_FILE)


def build(dump_dir: str, out: Optional[str] = None) -> dict:
    """Full scan; writes ``index.json`` and ``index.files.json`` and returns the index."""
    out = out or os.path.join(dump_dir, INDEX_FILE)
    t0 = time.time()
    state = _disk_state(dump_dir)  # before the scan: a file changed meanwhile looks stale
    index, n, _bad = scan(dump_dir)
    meta = {"version": INDEX_VERSION, "files": state}
    # index first, file list last: a crash in between leaves a file list that
    # does not describe the new index, so the next load simply rebuilds.
    write_json_atomic(out, index, indent=None)
    write_json_atomic(_files_path(out), meta, indent=None)
    print(f"[index] {len(index)} sequences, {n} files, {time.time()-t0:.1f}s -> {out}")
    return index


def is_current(dump_dir: str, out: Optional[str] = None) -> bool:
    """True when ``index.json`` was built (by this version) from the files on disk now.

    Compares name, size, mtime_ns, ctime_ns, inode, device and link target
    of every file.  Rewriting a file in place changes its ctime even when the
    size and mtime are restored, so the index is rebuilt.
    """
    out = out or os.path.join(dump_dir, INDEX_FILE)
    try:
        with open(_files_path(out)) as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        return False
    return bool(os.path.exists(out) and meta.get("version") == INDEX_VERSION
                and meta.get("files") == _disk_state(dump_dir))


def load_index(dump_dir: str, reindex: bool = False) -> dict:
    """The joined index, rebuilt when missing, stale or ``reindex`` is set.

    A dump directory that cannot be written to is indexed in memory.
    """
    dump_dir = os.path.expanduser(dump_dir)
    if not os.path.isdir(dump_dir):
        raise SystemExit(f"dump dir {dump_dir!r}: no such directory")
    out = os.path.join(dump_dir, INDEX_FILE)
    if not reindex and is_current(dump_dir, out):
        with open(out) as fh:
            return json.load(fh)
    try:
        return build(dump_dir, out)
    except OSError as exc:
        print(f"[index] cannot write {out} ({exc}); indexing in memory")
        return scan(dump_dir)[0]


def pieces(entry: dict) -> Tuple[List[str], List[int]]:
    return (list(entry.get("files") or [entry["file"]]),
            list(entry.get("seqs") or [entry["seq"]]))


def documents(index: dict, wanted: Optional[Iterable[str]] = None) -> List[Tuple[str, dict]]:
    """-> [(doc_hash, entry)] of whole documents, in dump-file order.

    Keeps, per starting slot, the longest indexed sequence -- among the
    hashes in ``wanted`` when given (the manifest), so a manifest document is
    matched as a whole, never by one of its pieces.
    """
    wanted = None if wanted is None else set(wanted)
    by_start: Dict[Tuple[str, int], Tuple[str, dict]] = {}
    for h, entry in index.items():
        if wanted is not None and h not in wanted:
            continue
        files, seqs = pieces(entry)
        key = (files[0], int(seqs[0]))
        cur = by_start.get(key)
        if cur is None or len(files) > len(pieces(cur[1])[0]):
            by_start[key] = (h, entry)
    return [by_start[k] for k in sorted(by_start)]


def load_document(
    dump_dir: str,
    entry: dict,
    min_len: int = 0,
    vocab_size: Optional[int] = 248320,
    expect: Optional[str] = None,
) -> Optional[Sample]:
    """One joined Sample for an index entry, or None when it is too short or
    holds token ids outside the vocabulary (SGLang's startup warm-up batch).

    ``expect`` is the doc_hash the caller looked up; the joined ids are hashed
    again and ``DocumentMismatch`` is raised when they do not match (a dump
    file changed under a still-current-looking index).
    """
    files, seqs = pieces(entry)
    sample = load_concat([os.path.join(dump_dir, f) for f in files], seqs)
    if expect is not None and sample.doc_hash != expect:
        raise DocumentMismatch(
            f"{expect}: the ids in {files} (slots {seqs}) hash to {sample.doc_hash}"
        )
    t = len(sample) + 1  # dump rows of the joined sequence
    if t < max(2, min_len + 1):
        return None
    if vocab_size:
        ids = sample.next_token_ids
        if int(ids.min()) < 0 or int(ids.max()) >= vocab_size:
            return None
    return sample


def read_arrays(dump_dir: str, entry: dict, names: Sequence[str]) -> Dict[str, torch.Tensor]:
    """Named per-row dump tensors of one document, concatenated over its pieces."""
    from safetensors import safe_open

    files, seqs = pieces(entry)
    parts: Dict[str, list] = {n: [] for n in names}
    for name, s in zip(files, seqs):
        with safe_open(os.path.join(dump_dir, name), framework="pt", device="cpu") as f:
            keys = set(f.keys())
            if "seq_offsets" in keys:
                offs = f.get_tensor("seq_offsets").tolist()
                a, b = int(offs[s]), int(offs[s + 1])
            else:
                a, b = 0, int(f.get_slice("input_ids").get_shape()[0])
            for n in names:
                parts[n].append(f.get_slice(n)[a:b])
    return {n: torch.cat(v) for n, v in parts.items()}


def document_hash(dump_dir: str, entry: dict) -> str:
    """The manifest's hash of one document's joined ``input_ids``, read from disk."""
    return _hash_ids(read_arrays(dump_dir, entry, ["input_ids"])["input_ids"].tolist())


__all__ = [
    "document_hash",
    "DocumentMismatch",
    "INDEX_FILE",
    "build",
    "documents",
    "file_stream",
    "is_current",
    "load_document",
    "load_index",
    "pieces",
    "read_arrays",
    "scan",
]
