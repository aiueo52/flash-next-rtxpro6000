#!/usr/bin/env python3
"""Freeze a held-out subset so every evaluation is comparable and fast.

Walks the eval split (10 % of documents, chosen by DumpDataset's by-document
hash) until it has ~`--rows` training rows, and writes an index naming just the
dump files those sequences live in, plus every chosen document with its pieces
(``docs``), so a document that chunked prefill split over several files is
loaded whole.  An eval pass then reads a small fraction of the dump instead of
all of it, and the baseline and every checkpoint are scored on exactly the
same tokens.

    python extract/build_eval_set.py DUMP_DIR          # writes DUMP_DIR/eval_fixed.json
"""
import argparse, collections, json, os, random, sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mtptrain.cli import StrictParser  # noqa: E402
from mtptrain.data import DumpDataset, _hash_ids  # noqa: E402
from mtptrain.dumpindex import pieces, read_arrays  # noqa: E402
from mtptrain.fileio import move_aside, read_jsonl, write_json_atomic  # noqa: E402

VOCAB = 248320


def build_eval_set(dump: str, out: str = None, rows: int = 200_000,
                   holdout: float = 0.1, seed: int = 20260903,
                   max_len: int = 2048, overwrite: bool = False) -> dict:
    """Write the frozen eval set; refuses an empty one.

    Re-running with the same dump and settings rewrites nothing; an existing
    eval set with other contents (another dump, other settings, a changed
    dump) is kept unless ``overwrite``, which moves it into
    ``.previous-<time>/`` next to it: runs that scored against it would
    otherwise silently change meaning.
    """
    dump = os.path.abspath(os.path.expanduser(dump))  # absolute: a pointer file
    if not os.path.isdir(dump):
        raise SystemExit(f"dump dir {dump}: no such directory")
    if rows <= 0 or max_len < 1:
        raise SystemExit("--rows and --max-len must be positive")
    if not 0.0 < holdout <= 1.0:
        raise SystemExit("--holdout must be in (0, 1]: 0 holds nothing out")
    out = os.path.abspath(os.path.expanduser(out or os.path.join(dump, "eval_fixed.json")))
    if os.path.isdir(out):
        raise SystemExit(f"--out {out} is a directory")
    man = {}
    mpath = os.path.join(dump, "manifest.jsonl")
    # No manifest: a dump made without the client (DumpDataset then uses
    # every indexed document); an empty one selects nothing (refused below).
    for r in read_jsonl(mpath, missing_ok=True):
        man[r["doc_hash"]] = r

    ds = DumpDataset(dump, split="eval", holdout_frac=holdout, seed=seed)

    # Collect the whole eval split first, from the joined index (a document
    # that chunked prefill split over several files counts once, with all its
    # pieces).  Only input_ids are read here; selecting the first N rows
    # instead would follow file order and hand back one or two buckets.
    entries = collections.defaultdict(list)
    avail = collections.Counter()
    for doc_hash, entry in ds.documents():
        if entry["tokens"] < ds.min_len + 1:
            continue
        ids = read_arrays(dump, entry, ["input_ids"])["input_ids"]
        got = _hash_ids(ids.tolist())
        if got != doc_hash:  # the dump changed under a current-looking index
            raise SystemExit(f"{doc_hash}: the ids in {entry['files']} now hash to {got}; "
                             "the dump was modified -- rebuild the index and re-check it")
        if int(ids.min()) < 0 or int(ids.max()) >= VOCAB:  # server warm-up batch
            continue
        n = min(entry["tokens"] - 1, max_len)
        rec = man.get(doc_hash, {})
        bucket = rec.get("bucket") or rec.get("mode") or "?"
        entries[bucket].append((doc_hash, entry, n))
        avail[bucket] += n
    total_avail = sum(avail.values())

    # Proportional stratification: take the same fraction of every bucket, so
    # the eval mix matches the training mix.
    frac = min(1.0, rows / max(1, total_avail))
    rng = random.Random(seed)
    files, hashes, docs, taken_rows = [], [], [], 0
    per_bucket = collections.Counter()
    seen_files = set()
    for bucket in sorted(entries):
        items = entries[bucket][:]
        rng.shuffle(items)
        budget = int(round(avail[bucket] * frac))
        taken = 0
        for doc_hash, entry, n in items:
            if taken >= budget:
                break
            hashes.append(doc_hash)
            names, seqs = pieces(entry)
            paths = [os.path.join(dump, f) for f in names]
            # the source-document keys, so a trainer can check that none of
            # them is in its training split (overlap by document, not only by hash)
            docs.append({"doc_hash": doc_hash, "pieces": names, "files": paths,
                         "seqs": seqs, "split_keys": sorted(ds.keys_of(doc_hash))})
            for src in paths:
                if src not in seen_files:
                    seen_files.add(src); files.append(src)
            taken += n; taken_rows += n; per_bucket[bucket] += n

    if not docs:
        raise SystemExit(f"{dump}: no held-out document to build an eval set from "
                         f"(holdout={holdout}, seed={seed}; empty manifest or dump?)")
    payload = {"format": 2, "dump_dir": dump,
               "rows": taken_rows, "files": files, "doc_hashes": hashes,
               "docs": docs, "holdout": holdout, "seed": seed,
               "max_len": max_len, "available_rows": total_avail,
               "fraction": frac, "by_bucket": dict(per_bucket)}
    if os.path.lexists(out):
        try:
            with open(out) as fh:
                same = json.load(fh) == json.loads(json.dumps(payload))
        except (OSError, ValueError):
            same = False
        if not same and not overwrite:
            raise SystemExit(f"{out} already holds a different eval set; runs scored "
                             "against it would change meaning.  Pass --overwrite to "
                             "move it aside, or a new --out")
        if not same:
            move_aside(os.path.dirname(out), [os.path.basename(out)])
    write_json_atomic(out, payload, indent=1)
    summary = {"out": out, "sequences": len(hashes), "rows": taken_rows,
               "files_touched": len(files), "eval_rows_available": total_avail,
               "fraction_taken": round(frac, 4), "by_bucket": dict(per_bucket)}
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = StrictParser(description=__doc__,
                     formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dump_dir")
    p.add_argument("--out", default=None)
    p.add_argument("--rows", type=int, default=200_000)
    p.add_argument("--holdout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=20260903)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--overwrite", action="store_true",
                   help="replace a different existing eval set (moved into "
                        ".previous-<time>/ next to it)")
    a = p.parse_args()
    build_eval_set(a.dump_dir, a.out, a.rows, a.holdout, a.seed, a.max_len, a.overwrite)


if __name__ == "__main__":
    main()
