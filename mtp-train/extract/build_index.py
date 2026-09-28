#!/usr/bin/env python3
"""doc_hash -> dump file index (reads only input_ids and positions, so it is fast).

    python extract/build_index.py DUMP_DIR        # writes DUMP_DIR/index.json

Always a full rescan: the pieces of a request that chunked prefill split over
several files can only be joined when its first piece is read in the same
pass.  The format and the joining rule are described in
``mtptrain/dumpindex.py``; every consumer (training, eval set, renewal
selection, verify_dump) reads the index through that module and
rebuilds it when the dump directory changed.
"""
from __future__ import annotations

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mtptrain.dumpindex import build, file_stream, scan  # noqa: E402,F401


def main(argv) -> int:
    if len(argv) not in (1, 2):
        raise SystemExit("usage: build_index.py DUMP_DIR [OUT]")
    if any(not a.strip() for a in argv):
        raise SystemExit("build_index.py: an argument is empty")
    index = build(os.path.expanduser(argv[0]), argv[1] if len(argv) > 1 else None)
    if not index:  # an empty index is written, but it is not a usable dump
        print(f"[index] {argv[0]}: no documents (no readable dump files)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    sys.exit(main(sys.argv[1:]))
