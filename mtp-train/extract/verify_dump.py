#!/usr/bin/env python3
"""CPU-only correctness check on a self-generation dump.

For a document generated **greedily**, the target's argmax at row t must be the
token the server actually emitted at t+1.  A dump written by a mis-configured
server (wrong hidden state, wrong lm_head path, stale KV) fails this at once,
so it is the cheapest possible gate on a fresh extraction --- no GPU, no model.

Agreement is not exactly 1.0: generation ran with speculative decoding and an
FP8 KV cache over a growing context, the dump is a single fresh prefill of the
finished sequence, and the two differ in floating point.  Expect agreement
close to, but not exactly, 100 %; a broken dump is far off.

Before that, *every* manifest document found in the index is read back and its
joined ``input_ids`` re-hashed; a hash that differs from the manifest's
``doc_hash`` (a dump file modified after extraction) fails the run (exit 1).
The index is rebuilt whenever a file's size, mtime, ctime, inode or device
changed.

    python extract/verify_dump.py DUMP_DIR --docs 200
"""
import argparse, collections, json, os, sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from mtptrain.cli import StrictParser  # noqa: E402
from mtptrain.fileio import read_jsonl  # noqa: E402
from mtptrain.dumpindex import (  # noqa: E402
    document_hash,
    documents,
    load_index,
    read_arrays,
)


def verify(dump: str, docs: int = 200, greedy_only: bool = True) -> dict:
    d = os.path.expanduser(dump)
    if not os.path.isdir(d):
        raise SystemExit(f"{dump}: no such directory")
    if docs < 1:
        raise SystemExit("--docs must be >= 1")
    man = read_jsonl(os.path.join(d, "manifest.jsonl"))  # must exist
    by_hash = {r["doc_hash"]: r for r in man}
    want = {h: r for h, r in by_hash.items()
            if "prompt_tokens" in r
            and (not greedy_only or float(r.get("temperature", 0.0)) == 0.0)}
    print(f"[verify] {len(man)} documents in the manifest, {len(want)} greedy")

    # Whole documents from the joined index, so a sequence chunked prefill
    # split over several files is checked as one (the manifest hashes the
    # full ids).  Identity first: every manifest document on disk.
    index = load_index(d)
    present = documents(index, by_hash)
    bad = []
    for h, entry in present:
        got = document_hash(d, entry)
        if got != h:
            bad.append((h, got, entry["files"]))
    missing = len(by_hash) - len(present)
    print(f"[verify] identity: {len(present)} documents re-hashed, {len(bad)} MISMATCH, "
          f"{missing} manifest documents not on disk (never dumped, e.g. the "
          "hook's SGLANG_MTP_DUMP_MAX_GB was reached; they are ignored)")
    if not present:
        raise SystemExit(f"{dump}: none of the {len(by_hash)} manifest documents is on "
                         "disk; nothing was verified")
    for h, got, files in bad[:10]:
        print(f"  MISMATCH {h}: ids in {files} hash to {got}")
    if bad:
        raise SystemExit(f"{len(bad)} documents do not match their manifest hash; "
                         "the dump was modified after extraction")

    stats = collections.defaultdict(lambda: [0, 0])
    seen = 0
    for h, entry in [x for x in present if x[0] in want][:docs]:
        rec = want[h]
        arr = read_arrays(d, entry, ["input_ids", "target_argmax"])
        ids = arr["input_ids"].to(torch.int64)
        tgt = arr["target_argmax"].to(torch.int64)
        n, pt = int(ids.numel()), int(rec["prompt_tokens"])
        # generated region: rows pt-1 .. n-2 predict tokens pt .. end
        agree = int((tgt[pt - 1: n - 1] == ids[pt: n]).sum())
        b = stats[rec.get("bucket") or "selfgen"]
        b[0] += agree; b[1] += n - pt
        seen += 1

    if not seen:
        print(f"[verify] no {'greedy ' if greedy_only else ''}self-generated document "
              "on disk: the argmax agreement below covers nothing")
    tot = [sum(v[0] for v in stats.values()), sum(v[1] for v in stats.values())]
    def pct(a, n):  # no generated row: nothing measured, not 0 %
        return f"{100 * a / n:17.2f}%" if n else f"{'n/a':>18s}"

    print(f"{'bucket':16s} {'rows':>10s} {'argmax == emitted':>18s}")
    for b in sorted(stats):
        v = stats[b]
        print(f"{b:16s} {v[1]:10d} {pct(*v)}")
    print(f"{'TOTAL':16s} {tot[1]:10d} {pct(*tot)}  ({seen} docs)")
    return {"checked": len(present), "mismatch": len(bad), "missing": missing,
            "agreement_docs": seen, "agreement_rows": tot[1],
            "agreement": tot[0] / tot[1] if tot[1] else None}


def main() -> None:
    p = StrictParser(description=__doc__,
                     formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dump")
    p.add_argument("--docs", type=int, default=200)
    p.add_argument("--greedy-only", action=argparse.BooleanOptionalAction, default=True,
                   help="agreement over greedy documents only (--no-greedy-only: all)")
    a = p.parse_args()
    verify(a.dump, a.docs, a.greedy_only)


if __name__ == "__main__":
    main()
