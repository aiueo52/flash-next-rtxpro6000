#!/usr/bin/env python3
"""Build an FR-Spec-style reduced draft vocabulary ("hot token map") for --speculative-token-map.

The draft (MTP) head then scores only the K most frequent target tokens instead of all 248,320,
which is what makes the FP8 W8A16 draft-head GEMV cheap (see docs/optimizations.md).  The
production maps used in the measurements were built from the author's private transcripts and
are not published; this script rebuilds an equivalent map from any text you are allowed to use.

Sources are weighted and normalised per source, then blended, exactly like the production
`hot2_*` maps (which blended 50 % "generated text of agent/chat sessions" with 50 % "the target
model's own greedy outputs").  All added/special tokens of the tokenizer are always included.
10 % of the documents of every source are held out to report coverage.

Inputs (repeatable, NAME is free-form, WEIGHT defaults to 1):
  --text NAME=PATH[:WEIGHT]            plain-text file, documents separated by blank lines
  --jsonl NAME=PATH[:WEIGHT]           JSONL; uses the fields given by --field (default: text)

Example (self-generated outputs collected with collect_selfgen.py, plus a text corpus):
  python tokenmaps/build_hot_vocab.py --tokenizer $HOME/models/RadixArk/Qwen3.8-Flash-Next-NVFP4 \
      --jsonl selfgen=$HOME/tokenmap-work/gen.jsonl:1 --text corpus=my_corpus.txt:1 --sizes 32768 49152 \
      --out-dir ~/tokenmap-work/

Writes hot_<K>.pt (a sorted Python list of token ids saved with torch.save, the format SGLang's
load_token_map() reads) and prints held-out coverage per source.  CPU only.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path


def repository_root() -> Path:
    """The checkout this script lives in (tokenmaps/ sits at its top level)."""
    return Path(__file__).resolve().parent.parent


def outside_repo(path: Path, allow_in_repo: bool, what: str) -> Path:
    """Refuse an output path inside the repository unless allow_in_repo: the map is derived from the
    user's corpus, which may be private."""
    destination = path.expanduser().resolve()
    root = repository_root()
    try:
        destination.relative_to(root)
    except ValueError:
        return destination
    if not allow_in_repo:
        raise SystemExit(
            f"refusing to write {what} inside the repository ({root}): {destination}. "
            "Choose a path outside the repository, or pass --allow-in-repo (and do not commit it).")
    return destination


def parse_src(spec: str) -> tuple[str, Path, float]:
    name, sep, rest = spec.partition("=")
    if not sep:
        raise SystemExit(f"source must look like NAME=PATH[:WEIGHT], got {spec!r}")
    m = re.fullmatch(r"(.*):([0-9]*\.?[0-9]+)", rest)
    path, weight = (m.group(1), float(m.group(2))) if m else (rest, 1.0)
    return name, Path(path), weight


def iter_docs(kind: str, path: Path, fields: list[str]):
    if kind == "text":
        buf: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                buf.append(line)
            elif buf:
                yield "\n".join(buf); buf = []
        if buf:
            yield "\n".join(buf)
    else:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                parts = [r.get(k) for k in fields if isinstance(r.get(k), str)]
                if parts:
                    yield "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True, help="model directory containing tokenizer.json")
    ap.add_argument("--text", action="append", default=[])
    ap.add_argument("--jsonl", action="append", default=[])
    ap.add_argument("--field", action="append", default=None, help="JSONL text field(s); default: text")
    ap.add_argument("--sizes", type=int, nargs="+", default=[32768, 49152])
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="where hot_<K>.pt is written; the map is derived from your corpus, so a directory "
                         "inside the repository is refused unless --allow-in-repo")
    ap.add_argument("--allow-in-repo", action="store_true",
                    help="allow --out-dir inside the repository (do not commit it)")
    ap.add_argument("--prefix", default="hot")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    args.out_dir = outside_repo(args.out_dir, args.allow_in_repo, "a token map")
    fields = args.field or ["text"]

    from tokenizers import Tokenizer  # CPU-only, same package the simulator uses

    tok_dir = Path(args.tokenizer)
    tok = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    special = [t["id"] for t in json.loads((tok_dir / "tokenizer.json").read_text()).get("added_tokens", [])]

    sources = [("text",) + parse_src(s) for s in args.text] + [("jsonl",) + parse_src(s) for s in args.jsonl]
    if not sources:
        ap.error("give at least one --text or --jsonl source")
    rng = random.Random(args.seed)
    score: collections.Counter = collections.Counter()
    held: dict[str, collections.Counter] = {}
    for kind, name, path, weight in sources:
        train, ho = collections.Counter(), collections.Counter()
        ndocs = 0
        for doc in iter_docs(kind, path, fields):
            ids = tok.encode(doc, add_special_tokens=False).ids
            (ho if rng.random() < 0.1 else train).update(ids)
            ndocs += 1
        total = sum(train.values())
        print(f"source {name}: {ndocs} docs, {total} train tokens, {sum(ho.values())} held-out tokens, {len(train)} distinct")
        if total:
            for t, c in train.items():
                score[t] += weight * c / total
        held[name] = ho

    ranked = [t for t, _ in score.most_common()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for k in args.sizes:
        hot, seen = [], set()
        for t in special + ranked:
            if t not in seen:
                seen.add(t); hot.append(t)
            if len(hot) >= k:
                break
        hs = set(hot)
        cov = {n: (100 * sum(c for t, c in h.items() if t in hs) / max(1, sum(h.values()))) for n, h in held.items()}
        out = args.out_dir / f"{args.prefix}_{k}.pt"
        try:
            import torch
            torch.save(sorted(hot), out)
        except ImportError:
            out = out.with_suffix(".json")
            out.write_text(json.dumps(sorted(hot)))
            print("  (torch not installed: wrote JSON; convert with torch.save(json.load(f), 'x.pt'))")
        print(f"K={k}: rows={len(hot)} held-out coverage " + ", ".join(f"{n} {v:.2f}%" for n, v in cov.items()) + f" -> {out}")
        if len(hot) < k:
            print(f"  note: only {len(hot)} distinct tokens seen; the map is smaller than requested")


if __name__ == "__main__":
    main()
