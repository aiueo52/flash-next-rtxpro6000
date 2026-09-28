"""Picking the self-generated held-out documents the renewal evaluator scores.

``scripts/eval_renewal.py`` and the training loop must score *the same* subset,
otherwise a number printed mid-training cannot be compared with an offline run.
Both therefore go through ``select_selfgen_docs`` + ``load_selfgen_samples``
here rather than each carrying its own copy of the manifest walk.

Only ``mode == "selfgen"`` documents are eligible: the renewal simulation needs
the model's own greedy continuation (so every window's dumped ``target_argmax``
is the right label) and the manifest's ``prompt_tokens`` to know where the
generated region starts.
"""

from __future__ import annotations

import collections
import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .data import Sample, is_holdout_key, truncate


@dataclass
class SelfgenSet:
    """Everything the renewal evaluator needs, loaded once and reused."""

    samples: List[Sample] = field(default_factory=list)
    gen_start_rows: List[int] = field(default_factory=list)
    buckets: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def rows(self) -> int:
        return sum(len(s) for s in self.samples)

    @property
    def generated_rows(self) -> int:
        return sum(len(s) - g for s, g in zip(self.samples, self.gen_start_rows))

    def bucket_of(self) -> Dict[str, str]:
        return {s.doc_hash: b for s, b in zip(self.samples, self.buckets)}

    def counts(self) -> Dict[str, int]:
        return dict(collections.Counter(self.buckets))


def is_holdout(key: str, split_seed: int, holdout_frac: float) -> bool:
    """The hash rule ``DumpDataset`` uses, so both agree on the split."""
    return is_holdout_key(key, split_seed, holdout_frac)


def load_index(dump_dir: str, reindex: bool = False) -> dict:
    """doc_hash -> {file, seq, tokens, files, seqs}: the joined index
    (``dumpindex.load_index``), rebuilt by a full scan when missing or stale."""
    from .dumpindex import load_index as _load

    return _load(dump_dir, reindex=reindex)


def select_selfgen_docs(
    dump_dir: str,
    manifest: Optional[str] = None,
    split: str = "eval",
    split_seed: int = 20260903,
    holdout_frac: float = 0.1,
    buckets: Optional[Sequence[str]] = None,
    per_bucket: int = 0,
    min_gen: int = 32,
    seed: int = 0,
    reindex: bool = False,
    index: Optional[dict] = None,
    greedy_only: bool = False,
) -> Tuple[List[dict], dict]:
    """-> (manifest records, index) for self-generated docs, per-bucket capped.

    ``split`` is "eval" (the trainer's held-out documents), "train" or "all";
    ``per_bucket`` = 0 means no cap.  The per-bucket sample is drawn with a
    fixed ``seed`` so repeated evaluations score exactly the same documents.
    ``greedy_only`` keeps documents generated at temperature 0 only: on a
    sampled continuation the dumped ``target_argmax`` is still the right label
    at every row, but after a rejection the text follows the sampled token
    rather than the target's, so the replay is no longer exactly the server's.
    """
    if split not in ("eval", "train", "all"):
        raise SystemExit(f"split must be eval, train or all, not {split!r}")
    if per_bucket < 0 or min_gen < 0:
        raise SystemExit("per_bucket and min_gen must be >= 0")
    if index is None:
        index = load_index(dump_dir, reindex=reindex)
    # The self-generated documents are known only from the manifest, so it
    # must exist (the default one included); read_jsonl refuses a missing file.
    manifest = manifest or os.path.join(dump_dir, "manifest.jsonl")
    wanted = set(buckets) if buckets else None
    from .fileio import read_jsonl  # tolerates a crash-truncated last line

    rows = read_jsonl(manifest)
    if wanted:
        known = {r.get("bucket") or "selfgen" for r in rows if r.get("mode") == "selfgen"}
        if wanted - known:
            raise SystemExit(f"unknown bucket(s) {sorted(wanted - known)}; the "
                             f"self-generated documents of {manifest} are in "
                             f"{sorted(known)}")
    # A document is held out if any split key the manifest gives it is (the
    # rule DumpDataset applies), so the same token ids never land on both sides.
    keys_of: Dict[str, set] = collections.defaultdict(set)
    for r in rows:
        keys_of[r["doc_hash"]].add(r.get("split_key") or r["doc_hash"])
    recs: List[dict] = []
    seen = set()
    for r in rows:
        if r.get("mode") != "selfgen" or "prompt_tokens" not in r:
            continue
        if r["doc_hash"] in seen:  # the same document recorded twice: once
            continue
        seen.add(r["doc_hash"])
        if r["doc_hash"] not in index:
            continue
        if min_gen and r.get("generated_tokens", 0) < min_gen:
            continue
        if greedy_only and float(r.get("temperature", 0.0)) != 0.0:
            continue
        r["bucket"] = r.get("bucket") or "selfgen"
        if split != "all":
            held = any(is_holdout(k, split_seed, holdout_frac)
                       for k in keys_of[r["doc_hash"]])
            if held != (split == "eval"):
                continue
        if wanted and r["bucket"] not in wanted:
            continue
        recs.append(r)
    by_bucket: Dict[str, List[dict]] = collections.defaultdict(list)
    for r in recs:
        by_bucket[r["bucket"]].append(r)
    rng = random.Random(seed)
    chosen: List[dict] = []
    for b in sorted(by_bucket):
        rs = by_bucket[b]
        rng.shuffle(rs)
        chosen += rs[:per_bucket] if per_bucket else rs
    return chosen, index


def load_selfgen_samples(
    dump_dir: str,
    chosen: Sequence[dict],
    index: dict,
    max_len: int = 0,
) -> SelfgenSet:
    """Read the dump rows for ``chosen`` and mark where generation started.

    ``gen_start_rows[i] = prompt_tokens - 1``: that draft row consumes
    ``hc[L-1]`` and ``embed(token L)`` -- the first generated token -- and
    predicts token ``L+1``, so it is the first row of the generated region.
    Documents whose generated region does not survive ``max_len`` truncation
    are dropped.
    """
    from .dumpindex import DocumentMismatch, load_document

    out = SelfgenSet()
    for r in chosen:
        try:
            s = load_document(dump_dir, index[r["doc_hash"]], vocab_size=None,
                              expect=r["doc_hash"])
        except DocumentMismatch as exc:
            # Refused, not skipped: dropping it would silently change which
            # documents the evaluation scores.
            raise SystemExit(f"[select] a selected document's dump changed after it was "
                             f"indexed; rebuild the index or the selection: {exc}")
        if s is None:
            continue
        gs = int(r["prompt_tokens"]) - 1
        if max_len and len(s) > max_len:
            s = truncate(s, max_len)
        if gs < 0 or gs >= len(s) - 1:
            continue
        out.samples.append(s)
        out.gen_start_rows.append(gs)
        out.buckets.append(r["bucket"])
    return out


def load_token_map(path: str, vocab_size: int) -> torch.Tensor:
    """Serving hot-vocabulary list -> bool [vocab] mask for ``model.chain``."""
    ids = torch.load(os.path.expanduser(path), map_location="cpu", weights_only=True)
    ids = torch.as_tensor(list(ids), dtype=torch.int64)
    if ids.numel() == 0:
        raise SystemExit(f"token map {path}: empty (it would forbid every draft token)")
    if int(ids.min()) < 0 or int(ids.max()) >= vocab_size:
        raise SystemExit(f"token map {path}: ids outside [0, {vocab_size})")
    allowed = torch.zeros(vocab_size, dtype=torch.bool)
    allowed[ids] = True
    return allowed


__all__ = [
    "SelfgenSet",
    "is_holdout",
    "load_index",
    "select_selfgen_docs",
    "load_selfgen_samples",
    "load_token_map",
]
