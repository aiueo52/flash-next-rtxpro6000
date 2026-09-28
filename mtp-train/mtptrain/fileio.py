"""Crash-safe file writing shared by the trainer and the extraction tools.

Two patterns cover every output of the pipeline:

* **Whole files** (checkpoints, snapshots, ``best.json``, ``history.json``,
  eval sets): write a temporary file in the same directory,
  ``fsync`` it, ``rename`` it over the target, ``fsync`` the directory.  A
  reader sees the old file or the new one, never a half-written one.
* **Append-only logs** (manifests, ``--append`` result logs):
  one JSON object per line, flushed after every record, so a crash loses at
  most the record being written.  A crash can still leave a *partial last
  line*; ``open_jsonl_append`` cuts it off (with a warning) before appending,
  so the next record is never glued onto it, and ``read_jsonl`` skips it.  The
  record it held is simply redone by the resuming tool.  A last line that is a
  complete record and only lacks its newline is kept (the newline is added).
"""

from __future__ import annotations

import json
import os
from typing import Callable, Iterable, List, Optional


def fsync_dir(path: str) -> None:
    """Make a rename in ``path`` durable (best effort on odd filesystems)."""
    try:
        fd = os.open(path or ".", os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def fsync_file(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def replace_atomic(tmp: str, path: str) -> None:
    """fsync ``tmp``, rename it to ``path``, fsync the directory."""
    fsync_file(tmp)
    os.replace(tmp, path)
    fsync_dir(os.path.dirname(os.path.abspath(path)))


def write_atomic(path: str, writer: Callable[[str], None]) -> None:
    """``writer(tmp_path)`` produces the file; it then replaces ``path`` atomically."""
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        writer(tmp)
        replace_atomic(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def write_json_atomic(path: str, obj, indent: int = 2) -> None:
    def w(tmp: str) -> None:
        with open(tmp, "w") as fh:
            json.dump(obj, fh, indent=indent)
            fh.write("\n")

    write_atomic(path, w)


def write_jsonl_atomic(path: str, records: Iterable[dict]) -> None:
    def w(tmp: str) -> None:
        with open(tmp, "w") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_atomic(path, w)


def read_jsonl(path: str, missing_ok: bool = False) -> List[dict]:
    """Every complete record; a partial (crash-truncated) last line is skipped.

    A missing file is an error unless ``missing_ok`` (a log about to be
    started, e.g. the manifest a resumable tool appends to).
    """
    out: List[dict] = []
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        if missing_ok:
            return out
        raise SystemExit(f"{path}: no such file")
    if not os.path.isfile(path):
        raise SystemExit(f"{path}: not a file")
    with open(path, "rb") as fh:
        data = fh.read()
    lines = data.split(b"\n")
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        last = i == len(lines) - 1  # no trailing newline after it
        try:
            rec = json.loads(line)
            if last and not isinstance(rec, dict):
                raise ValueError  # e.g. the digits of a cut-off number
        except ValueError:
            if last:  # the interrupted write; repair_tail cuts exactly this
                print(f"[jsonl] {path}: ignoring a partial last line", flush=True)
                continue
            raise ValueError(f"{path}: line {i + 1} is not valid JSON") from None
        out.append(rec)
    return out


def repair_tail(path: str) -> int:
    """Make ``path`` end on a line boundary before appending; -> bytes cut.

    A last line without a trailing newline that is a complete JSON record
    (``read_jsonl`` returns it) only gets its newline.  Only a tail that does
    not parse -- the partial write of an interrupted run -- is cut off.
    """
    if not os.path.exists(path):
        return 0
    size = os.path.getsize(path)
    if size == 0:
        return 0
    with open(path, "rb+") as fh:
        fh.seek(-1, os.SEEK_END)
        if fh.read(1) == b"\n":
            return 0
        # find the last newline
        pos, block = size, 1 << 16
        keep = 0
        while pos > 0:
            start = max(0, pos - block)
            fh.seek(start)
            chunk = fh.read(pos - start)
            nl = chunk.rfind(b"\n")
            if nl >= 0:
                keep = start + nl + 1
                break
            pos = start
        fh.seek(keep)
        tail = fh.read()
        try:
            complete = isinstance(json.loads(tail), dict)  # every record is an object
        except ValueError:
            complete = False
        if complete:
            fh.seek(0, os.SEEK_END)
            fh.write(b"\n")
            fh.flush()
            os.fsync(fh.fileno())
            return 0
        fh.truncate(keep)
        fh.flush()
        os.fsync(fh.fileno())
    print(f"[jsonl] {path}: removed a partial last line ({size - keep} bytes) "
          "left by an interrupted run", flush=True)
    return size - keep


class JsonlAppender:
    """Append one JSON record per line, flushed per record (thread-safe use:
    call ``write`` under the caller's lock)."""

    def __init__(self, path: str):
        path = os.path.expanduser(path)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        repair_tail(path)
        self.path = path
        self._fh = open(path, "a")

    def write(self, record: dict) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh.closed:
            return
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def open_jsonl_append(path: str) -> JsonlAppender:
    return JsonlAppender(path)


def move_aside(root: str, names: Iterable[str], label: str = "previous") -> str:
    """Move ``names`` (entries of ``root``) into a new ``root/.<label>-<time>/``.

    Renames only -- nothing is deleted or copied, and everything stays on the
    same filesystem.  -> the directory used ("" when there was nothing to move).
    """
    import time

    names = sorted(n for n in names if os.path.lexists(os.path.join(root, n)))
    if not names:
        return ""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest, n = os.path.join(root, f".{label}-{stamp}"), 0
    while os.path.lexists(dest):
        n += 1
        dest = os.path.join(root, f".{label}-{stamp}-{n}")
    os.makedirs(dest)
    for name in names:
        os.rename(os.path.join(root, name), os.path.join(dest, name))
    fsync_dir(dest)
    fsync_dir(root)
    print(f"[out] moved {len(names)} earlier file(s) aside to {dest}: "
          f"{', '.join(names[:6])}{' ...' if len(names) > 6 else ''}", flush=True)
    return dest


def require_file(path: Optional[str], what: str) -> str:
    """An explicitly given file must exist (and be a file); -> expanded path."""
    full = os.path.expanduser(path or "")
    if not path or not os.path.isfile(full):
        raise SystemExit(f"{what} {path!r}: no such file")
    return full


def require_dir(path: Optional[str], what: str) -> str:
    """An explicitly given directory must exist (and be one); -> expanded path."""
    full = os.path.expanduser(path or "")
    if not path or not os.path.isdir(full):
        raise SystemExit(f"{what} {path!r}: no such directory")
    return full


def check_run_settings(rows: Iterable[dict], key: str, settings: dict, where: str) -> None:
    """Refuse to extend a log whose earlier rows were made with other settings.

    ``rows`` are the existing records; each carries ``settings`` under ``key``.
    Rows without the key (written before settings were recorded) are not
    checked.
    """
    for r in rows:
        old = r.get(key)
        if old is None or old == settings:
            continue
        diff = {k: (old.get(k), settings.get(k)) for k in sorted(set(old) | set(settings))
                if old.get(k) != settings.get(k)}
        raise SystemExit(f"{where} was written with different settings {diff} "
                         "(earlier, now); use a new output or the earlier settings")


DIGEST_CACHE_ENV = "MTP_DIGEST_CACHE"


def digest_cache_path() -> str:
    """Where ``file_digest`` keeps its cache: ``$MTP_DIGEST_CACHE``, else
    ``$XDG_CACHE_HOME/mtp-train/digests.json`` (``~/.cache/...``)."""
    if DIGEST_CACHE_ENV in os.environ:
        path = os.environ[DIGEST_CACHE_ENV]
        if not path.strip():
            raise SystemExit(f"${DIGEST_CACHE_ENV} is set but empty")
        return os.path.expanduser(path)
    root = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(root, "mtp-train", "digests.json")


def _file_key(real: str) -> list:
    st = os.stat(real)
    # ctime too: an in-place rewrite with the mtime set back still moves it
    return [st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino, st.st_dev]


def file_digest(path: str) -> str:
    """blake2b of a file's full content (symlinks followed).

    Hashing a multi-GB checkpoint takes a while, so the digest is cached per
    real path under the key (size, mtime_ns, ctime_ns, inode, device); any change of
    that key recomputes it, and a cache entry is never used across one.  The
    key is read before and after hashing: a file that changed meanwhile is
    hashed again rather than cached under a key that does not describe it.
    """
    import hashlib

    real = os.path.realpath(os.path.expanduser(path))
    cache_file = digest_cache_path()
    try:
        with open(cache_file) as fh:
            cache = json.load(fh)
        if not isinstance(cache, dict):
            cache = {}
    except (OSError, ValueError):
        cache = {}
    for _attempt in range(3):
        key = _file_key(real)
        hit = cache.get(real)
        if isinstance(hit, dict) and hit.get("key") == key:
            return hit["digest"]
        h = hashlib.blake2b(digest_size=12)
        with open(real, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 22), b""):
                h.update(block)
        if _file_key(real) == key:
            break
    else:
        raise SystemExit(f"{path}: keeps changing while it is hashed")
    digest = h.hexdigest()
    cache[real] = {"key": key, "digest": digest}
    try:  # a cache that cannot be written only costs time
        os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
        write_json_atomic(cache_file, cache, indent=None)
    except OSError:
        pass
    return digest


def digest_parts(parts: Iterable) -> str:
    """blake2b over the ``str`` of each part (order matters)."""
    import hashlib

    h = hashlib.blake2b(digest_size=12)
    for p in parts:
        h.update(str(p).encode())
        h.update(b";")
    return h.hexdigest()


__all__ = [
    "JsonlAppender",
    "digest_parts",
    "check_run_settings",
    "digest_cache_path",
    "file_digest",
    "require_dir",
    "require_file",
    "move_aside",
    "fsync_dir",
    "open_jsonl_append",
    "read_jsonl",
    "repair_tail",
    "replace_atomic",
    "write_atomic",
    "write_json_atomic",
    "write_jsonl_atomic",
]
