"""Training samples from the extraction dumps.

Dump layout (one safetensors file per target *extend* forward, written by the
SGLang hook ``python/sglang/srt/models/mtp_dump.py`` from
``patches/sglang/mtp-dump/`` of this repository):

    input_ids      int32 [T]            tokens of the whole forward, concatenated
    positions      int32 [T]            absolute target positions
    hc_hidden      bf16  [T, 4*H]       target residual BEFORE hyper_connection_mixer
    target_argmax  int32 [T]            argmax of the target logits at every position
    seq_offsets    int32 [S+1]          per-request slice boundaries into the above
    prefix_lens    int32 [S]            radix-cache prefix length (0 = full sequence)
    topk_ids       int32 [T, K]         optional (SGLANG_MTP_DUMP_TOPK=K)
    topk_logits    fp16  [T, K]         optional
    lse            fp16  [T]            optional, written with the top-K:
                                        logsumexp of the FULL-vocab target
                                        logits, so the mass outside the top-K
                                        is known
    tap_hidden     bf16  [T, N*4*H]     optional (SGLANG_MTP_DUMP_TAPS): the
                                        output HC state of N extra target
                                        layers, concatenated; the tap ids are
                                        in the file metadata as ``tap_layers``

Row alignment for training (see model.py):

    inputs  : next_token_ids = input_ids[t+1], hc_hidden[t], positions[t]
    label   : target_argmax[t+1]
    soft    : topk_ids[t+1], topk_logits[t+1], lse[t+1]   (same row as label)
    valid t : 0 .. T-2

Old dumps carry neither ``topk_*`` nor ``lse``; the corresponding ``Sample``
fields are then ``None`` and every consumer falls back to the hard label.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import queue
import random
import threading
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence

import torch


@dataclass
class Sample:
    next_token_ids: torch.Tensor  # int64 [L]
    hc_hidden: torch.Tensor  # bf16  [L, 4H]
    positions: torch.Tensor  # int64 [L]
    labels: torch.Tensor  # int64 [L]
    input_ids: torch.Tensor  # int64 [L] (= tokens at t+1; kept for diagnostics)
    greedy_consistent: torch.Tensor  # bool [L]; input_ids[t+2] == target_argmax[t+1]
    source: str
    doc_hash: str
    # Soft targets, aligned exactly like ``labels``: row t holds the target's
    # distribution at dump row t+1.  All three are None for a dump written
    # without SGLANG_MTP_DUMP_TOPK; ``lse`` alone may be None for a top-K dump
    # written before the hook stored it.
    topk_ids: Optional[torch.Tensor] = None  # int64 [L, K]
    topk_logits: Optional[torch.Tensor] = None  # fp32  [L, K]
    lse: Optional[torch.Tensor] = None  # fp32  [L]
    # Aligned exactly like ``hc_hidden`` (row t, not t+1): both are the target
    # state the draft's first chain forward consumes at row t.
    tap_hidden: Optional[torch.Tensor] = None  # bf16 [L, N*4H]
    # [(dump file path, sequence slot)] this sample was read from, in order:
    # one entry, or several for a request chunked prefill split over files.
    pieces: Optional[List[tuple]] = None

    def __len__(self) -> int:
        return int(self.next_token_ids.numel())

    @property
    def has_soft(self) -> bool:
        return self.topk_ids is not None

    @property
    def has_taps(self) -> bool:
        return self.tap_hidden is not None


def update_digest(h, name: str, value) -> None:
    """Feed ``name`` and the full content of ``value`` (a tensor, None or a
    plain value) into the hashlib object ``h``."""
    if isinstance(value, torch.Tensor):
        t = value.detach().to("cpu").contiguous()
        h.update(f"{name}:{t.dtype}:{tuple(t.shape)};".encode())
        if t.numel():
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    else:
        h.update(f"{name}={value!r};".encode())


# What a sample's content is: every tensor any consumer reads, plus the
# document hash.  ``source`` and ``pieces`` name where it was read from.
_SAMPLE_CONTENT = ("doc_hash", "next_token_ids", "hc_hidden", "positions", "labels",
                   "input_ids", "greedy_consistent", "topk_ids", "topk_logits", "lse",
                   "tap_hidden")


def samples_digest(samples: Sequence["Sample"], extra: Optional[Sequence] = None) -> str:
    """blake2b over the full content of ``samples`` in order (and ``extra``,
    e.g. the renewal evaluator's generation start rows).

    The identity of an eval: the same doc hashes over changed hidden states,
    labels or top-K give another digest.  Reads every byte once (about a
    second per GB).
    """
    h = hashlib.blake2b(digest_size=12)
    for i, s in enumerate(samples):
        for name in _SAMPLE_CONTENT:
            update_digest(h, f"{i}.{name}", getattr(s, name))
    update_digest(h, "extra", list(extra) if extra is not None else None)
    return h.hexdigest()


def _hash_ids(ids: Sequence[int]) -> str:
    h = hashlib.blake2b(digest_size=8)
    h.update(torch.tensor(list(ids), dtype=torch.int32).numpy().tobytes())
    return h.hexdigest()


def _read_soft(f, keys, a=0, b=None):
    """-> (topk_ids int64 [n,K], topk_logits fp32 [n,K], lse fp32 [n]) or Nones.

    ``f`` is an open ``safe_open`` handle and ``a:b`` the sequence slice.  A
    dump written without SGLANG_MTP_DUMP_TOPK has none of these keys; one
    written before the hook stored ``lse`` has the first two only.
    """
    if "topk_ids" not in keys or "topk_logits" not in keys:
        return None, None, None
    ids = _rows(f, "topk_ids", a, b).to(torch.int64)
    logits = _rows(f, "topk_logits", a, b).to(torch.float32)
    lse = _rows(f, "lse", a, b).to(torch.float32) if "lse" in keys else None
    return ids, logits, lse


def _rows(f, name: str, a: int = 0, b: Optional[int] = None) -> torch.Tensor:
    """Rows ``a:b`` of one tensor; a slice read touches only those rows."""
    if a == 0 and b is None:
        return f.get_tensor(name)
    return f.get_slice(name)[a:b]


def is_holdout_key(key: str, seed: int, holdout_frac: float) -> bool:
    """The split rule: blake2b(f"{seed}:{key}") % 10000 < holdout * 10000."""
    h = hashlib.blake2b(f"{seed}:{key}".encode(), digest_size=8).hexdigest()
    return (int(h, 16) % 10_000) < int(holdout_frac * 10_000)


def iter_dump_files(dump_dir: str) -> List[str]:
    files = []
    for root, _dirs, names in os.walk(dump_dir):
        for name in sorted(names):
            if name.endswith(".safetensors"):
                files.append(os.path.join(root, name))
    return sorted(files)


def read_tap_layers(dump_dir: str) -> tuple:
    """Tap layer ids recorded by the dump hook, from the first shard's metadata.

    Training and evaluation take the geometry from the data rather than a flag,
    so a head can never be built for one tap set and fed another.
    """
    from safetensors import safe_open

    for path in iter_dump_files(dump_dir):
        with safe_open(path, framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
            if "tap_hidden" not in f.keys():
                return ()
            raw = (meta.get("tap_layers") or "").strip()
        return tuple(int(x) for x in raw.split(",") if x)
    return ()


def load_dump_file(
    path: str, min_len: int = 8, vocab_size: int = 248320
) -> List[Sample]:
    from safetensors import safe_open

    out: List[Sample] = []
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        ids = f.get_tensor("input_ids").to(torch.int64)
        hc = f.get_tensor("hc_hidden")
        pos = f.get_tensor("positions").to(torch.int64)
        tgt = f.get_tensor("target_argmax").to(torch.int64)
        offs = (
            f.get_tensor("seq_offsets").tolist()
            if "seq_offsets" in keys
            else [0, int(ids.numel())]
        )
        soft_ids, soft_logits, soft_lse = _read_soft(f, keys)
        taps = f.get_tensor("tap_hidden") if "tap_hidden" in keys else None
    for s in range(len(offs) - 1):
        a, b = int(offs[s]), int(offs[s + 1])
        t = b - a
        if t < min_len + 1:
            continue
        seq_ids = ids[a:b]
        seq_tgt = tgt[a:b]
        # SGLang's startup warmup runs a synthetic batch whose token ids are
        # outside the vocabulary; drop anything that cannot be embedded.
        if vocab_size and (
            int(seq_ids.min()) < 0 or int(seq_ids.max()) >= vocab_size
        ):
            continue
        nxt = seq_ids[1:t]
        labels = seq_tgt[1:t]
        # ``greedy_consistent[t]`` says the *corpus* continuation agrees with
        # what the target would have generated, which is what makes the chain
        # acceptance simulation exact beyond the first draft step.
        cons = torch.zeros_like(labels, dtype=torch.bool)
        cons[: t - 2] = seq_ids[2:t] == seq_tgt[1 : t - 1]
        out.append(
            Sample(
                next_token_ids=nxt,
                hc_hidden=hc[a : b - 1],
                positions=pos[a : b - 1],
                labels=labels,
                input_ids=nxt,
                greedy_consistent=cons,
                source=os.path.basename(path),
                doc_hash=_hash_ids(seq_ids.tolist()),
                # Same shift as ``labels``: row t carries dump row t+1.
                topk_ids=None if soft_ids is None else soft_ids[a + 1 : b],
                topk_logits=(
                    None if soft_logits is None else soft_logits[a + 1 : b]
                ),
                lse=None if soft_lse is None else soft_lse[a + 1 : b],
                tap_hidden=None if taps is None else taps[a : b - 1],
                pieces=[(path, s)],
            )
        )
    return out


def load_concat(paths: Sequence[str], seqs: Sequence[int]) -> Sample:
    """One Sample from consecutive chunked-prefill dump files of one request
    (positions continue across files; the last file may be short)."""
    from safetensors import safe_open

    ids, hc, pos, tgt, tap = [], [], [], [], []
    s_ids, s_logits, s_lse = [], [], []
    for path, s in zip(paths, seqs):
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            offs = f.get_tensor("seq_offsets").tolist() if "seq_offsets" in keys else None
            a, b = (int(offs[s]), int(offs[s + 1])) if offs else (0, None)
            ids.append(_rows(f, "input_ids", a, b).to(torch.int64))
            hc.append(_rows(f, "hc_hidden", a, b))
            pos.append(_rows(f, "positions", a, b).to(torch.int64))
            tgt.append(_rows(f, "target_argmax", a, b).to(torch.int64))
            ci, cl, ce = _read_soft(f, keys, a, b)
            s_ids.append(ci)
            s_logits.append(cl)
            s_lse.append(ce)
            tap.append(_rows(f, "tap_hidden", a, b) if "tap_hidden" in keys else None)
    seq_ids, seq_hc, seq_pos, seq_tgt = (torch.cat(x) for x in (ids, hc, pos, tgt))
    # A chunk without top-K makes the whole concatenated sequence unusable as a
    # soft target -- there is no per-row way to say "missing" inside one Sample.
    soft_ids = None if any(x is None for x in s_ids) else torch.cat(s_ids)
    soft_logits = None if soft_ids is None else torch.cat(s_logits)
    soft_lse = (
        None
        if soft_ids is None or any(x is None for x in s_lse)
        else torch.cat(s_lse)
    )
    seq_tap = None if any(x is None for x in tap) else torch.cat(tap)
    assert torch.equal(seq_pos, torch.arange(seq_pos[0], seq_pos[0] + seq_pos.numel())), "chunks not contiguous"
    t = int(seq_ids.numel())
    labels = seq_tgt[1:t]
    cons = torch.zeros_like(labels, dtype=torch.bool)
    cons[: t - 2] = seq_ids[2:t] == seq_tgt[1 : t - 1]
    return Sample(
        next_token_ids=seq_ids[1:t], hc_hidden=seq_hc[: t - 1], positions=seq_pos[: t - 1],
        labels=labels, input_ids=seq_ids[1:t], greedy_consistent=cons,
        source=os.path.basename(paths[0]), doc_hash=_hash_ids(seq_ids.tolist()),
        topk_ids=None if soft_ids is None else soft_ids[1:t],
        topk_logits=None if soft_logits is None else soft_logits[1:t],
        lse=None if soft_lse is None else soft_lse[1:t],
        tap_hidden=None if seq_tap is None else seq_tap[: t - 1],
        pieces=[(p, int(s)) for p, s in zip(paths, seqs)],
    )


class DumpDataset:
    """Lazy index over the dump directory with a deterministic held-out split.

    The split is by ``doc_hash`` so that re-extracting or re-chunking the same
    document can never leak it from train to eval.

    Samples are whole documents from the joined index (``dumpindex``): a
    request that chunked prefill split over several dump files is yielded
    once, as the joined sequence, and matched against the manifest by the
    hash of all its ids.  Documents come in the order of their first file.

    Every document is re-hashed when it is read; one whose ids no longer match
    the doc_hash it was indexed under (a dump file changed in place) stops
    the reader with SystemExit.  ``stats`` counts ``yielded`` and
    ``too_short`` documents.
    """

    def __init__(
        self,
        dump_dir: str,
        split: str = "train",
        holdout_frac: float = 0.1,
        seed: int = 20260903,
        min_len: int = 8,
        max_files: Optional[int] = None,
        manifest: Optional[str] = None,
        manifest_only: bool = True,
        exclude_buckets: Optional[Sequence[str]] = None,
        include_buckets: Optional[Sequence[str]] = None,
    ):
        if not os.path.isdir(os.path.expanduser(dump_dir)):
            raise SystemExit(f"dump dir {dump_dir!r}: no such directory")
        self.dump_dir = dump_dir
        self.stats: collections.Counter = collections.Counter()
        self.split = split
        self.holdout_frac = holdout_frac
        self.seed = seed
        self.min_len = min_len
        self.files = iter_dump_files(dump_dir)
        if max_files is not None:
            self.files = self.files[:max_files]
        # chunk hash -> document identity.  Several chunks come from the same
        # document; without this the split would be per *chunk* and a document
        # could land on both sides of it.
        # When a manifest is present, only sequences it lists are training data.
        # A self-generation document is prefilled twice -- once as the prompt for
        # the generation request and once as prompt+continuation for extraction
        # -- and only the second is recorded, so without this the prompt text
        # would be counted twice.  The server's startup warmup batch is excluded
        # the same way.
        self.manifest_only = manifest_only
        self.exclude_buckets = set(exclude_buckets or ())
        self.include_buckets = set(include_buckets or ())
        self.buckets: Dict[str, str] = {}
        self.split_keys: Dict[str, str] = {}
        # every split key the manifest gives a document: the same token ids can
        # appear under several source documents (a shared boilerplate window);
        # such a document is held out if any of its keys is.
        self.doc_keys: Dict[str, set] = collections.defaultdict(set)
        self.meta: Dict[str, dict] = {}
        self._docs: Optional[List[tuple]] = None
        # An explicit manifest must exist; the default one may be absent (a
        # dump made without the client), and then every indexed document is
        # data.  A manifest that exists but lists nothing (or nothing that is
        # on disk) selects nothing -- it never means "use every file".
        if manifest is None:
            default = os.path.join(dump_dir, "manifest.jsonl")
            manifest = default if os.path.exists(default) else None
        elif not os.path.isfile(os.path.expanduser(manifest)):
            raise SystemExit(f"manifest {manifest!r}: no such file")
        self.manifest = manifest
        if manifest:
            from .fileio import read_jsonl  # tolerates a crash-truncated last line

            for rec in read_jsonl(manifest):
                key = rec.get("split_key") or rec.get("doc_hash")
                self.split_keys.setdefault(rec["doc_hash"], key)
                self.doc_keys[rec["doc_hash"]].add(key)
                self.meta[rec["doc_hash"]] = rec
                self.buckets[rec["doc_hash"]] = (
                    rec.get("bucket") or rec.get("mode") or "?"
                )
        if (self.exclude_buckets or self.include_buckets) and not manifest:
            raise SystemExit(f"{dump_dir}: bucket filters need a manifest.jsonl")
        known = set(self.buckets.values())
        unknown = (self.exclude_buckets | self.include_buckets) - known
        if manifest and unknown:
            raise SystemExit(f"unknown bucket(s) {sorted(unknown)}; the manifest has "
                             f"{sorted(known)}")

    def split_key(self, doc_hash: str) -> str:
        return self.split_keys.get(doc_hash, doc_hash)

    def keys_of(self, doc_hash: str) -> set:
        """Every split key of a document (its own hash without a manifest)."""
        return set(self.doc_keys.get(doc_hash) or {doc_hash})

    def _is_holdout(self, doc_hash: str) -> bool:
        return any(is_holdout_key(k, self.seed, self.holdout_frac)
                   for k in self.keys_of(doc_hash))

    def documents(self) -> List[tuple]:
        """-> [(doc_hash, index entry)] this split would read, in file order.

        Decided from the index and the manifest alone, before any hidden
        state is read.
        """
        from .dumpindex import documents, load_index

        if self._docs is None:
            index = load_index(self.dump_dir)
            wanted = self.split_keys if (self.manifest_only and self.manifest) else None
            present = {os.path.relpath(p, self.dump_dir) for p in self.files}
            self._docs = [(h, e) for h, e in documents(index, wanted)
                          if e["file"] in present]
        want_holdout = self.split == "eval"
        out = []
        for h, entry in self._docs:
            if self.exclude_buckets or self.include_buckets:
                bucket = self.buckets.get(h, "?")
                if bucket in self.exclude_buckets:
                    continue
                if self.include_buckets and bucket not in self.include_buckets:
                    continue
            if self._is_holdout(h) == want_holdout:
                out.append((h, entry))
        return out

    def __iter__(self) -> Iterator[Sample]:
        from .dumpindex import DocumentMismatch, load_document

        for doc_hash, entry in self.documents():
            try:
                sample = load_document(self.dump_dir, entry, min_len=self.min_len,
                                       expect=doc_hash)
            except DocumentMismatch as exc:
                # Refused, not skipped: a skip would silently shrink the data
                # (and could change what an eval scores) after the checks ran.
                raise SystemExit(
                    f"{self.dump_dir}: a document's dump changed after it was indexed "
                    f"({exc}); rebuild the index (extract/build_index.py) and run "
                    "extract/verify_dump.py") from exc
            if sample is None:
                self.stats["too_short"] += 1
                continue
            self.stats["yielded"] += 1
            yield sample

    def shuffled(self, buffer_size: int = 256, seed: int = 0) -> Iterator[Sample]:
        rng = random.Random(seed)
        buf: List[Sample] = []
        for sample in self:
            buf.append(sample)
            if len(buf) >= buffer_size:
                i = rng.randrange(len(buf))
                buf[i], buf[-1] = buf[-1], buf[i]
                yield buf.pop()
        rng.shuffle(buf)
        yield from buf


# Padding column of ``topk_logits``.  ``softmax`` over the K axis maps it to
# exactly zero probability, so a short K (or a padded row) contributes nothing.
NEG_INF = float("-inf")


def collate(samples: Sequence[Sample], pad_label: int = -100):
    """Right-pad a list of samples into one batch (no sequence packing).

    When at least one sample carries soft targets the batch additionally has

        topk_ids     int64 [B, L, K]   K = max K over the batch
        topk_logits  fp32  [B, L, K]   padded columns are -inf (zero mass)
        lse          fp32  [B, L]      +inf where unknown (=> full-K mass 1)
        soft_valid   bool  [B, L]      rows with a usable soft target

    Samples without top-K (old dumps mixed into the same batch) get
    ``soft_valid=False`` on every row, so the soft term simply skips them.
    """
    b = len(samples)
    lmax = max(len(s) for s in samples)
    hc_dim = samples[0].hc_hidden.shape[-1]
    dtype = samples[0].hc_hidden.dtype
    next_ids = torch.zeros(b, lmax, dtype=torch.int64)
    hc = torch.zeros(b, lmax, hc_dim, dtype=dtype)
    pos = torch.zeros(b, lmax, dtype=torch.int64)
    labels = torch.full((b, lmax), pad_label, dtype=torch.int64)
    valid = torch.zeros(b, lmax, dtype=torch.bool)
    cons = torch.zeros(b, lmax, dtype=torch.bool)
    for i, s in enumerate(samples):
        n = len(s)
        next_ids[i, :n] = s.next_token_ids
        hc[i, :n] = s.hc_hidden
        pos[i, :n] = s.positions
        labels[i, :n] = s.labels
        valid[i, :n] = True
        cons[i, :n] = s.greedy_consistent
    out = {
        "next_token_ids": next_ids,
        "hc_hidden": hc,
        "positions": pos,
        "labels": labels,
        "valid": valid,
        "greedy_consistent": cons,
    }

    # All-or-nothing: a batch mixing tapped and untapped samples would need a
    # per-row "no taps" signal the fused entry has no way to express, and the
    # two never occur in one dump directory anyway.
    if all(s.has_taps for s in samples):
        tap_dim = samples[0].tap_hidden.shape[-1]
        taps = torch.zeros(b, lmax, tap_dim, dtype=dtype)
        for i, s in enumerate(samples):
            taps[i, : len(s)] = s.tap_hidden
        out["tap_hidden"] = taps
    elif any(s.has_taps for s in samples):
        raise ValueError("batch mixes samples with and without tap_hidden")

    kmax = max((s.topk_ids.shape[-1] for s in samples if s.has_soft), default=0)
    if kmax == 0:
        return out
    soft_ids = torch.zeros(b, lmax, kmax, dtype=torch.int64)
    soft_logits = torch.full((b, lmax, kmax), NEG_INF, dtype=torch.float32)
    # +inf lse => exp(logsumexp(topk) - lse) = 0, so an unknown lse must never
    # reach the loss; ``soft_lse_known`` gates it back to the "normalise to 1"
    # branch instead.
    soft_lse = torch.full((b, lmax), float("inf"), dtype=torch.float32)
    lse_known = torch.zeros(b, lmax, dtype=torch.bool)
    soft_valid = torch.zeros(b, lmax, dtype=torch.bool)
    for i, s in enumerate(samples):
        if not s.has_soft:
            continue
        n, k = len(s), s.topk_ids.shape[-1]
        soft_ids[i, :n, :k] = s.topk_ids
        soft_logits[i, :n, :k] = s.topk_logits
        soft_valid[i, :n] = True
        if s.lse is not None:
            soft_lse[i, :n] = s.lse
            lse_known[i, :n] = True
    out.update(
        topk_ids=soft_ids,
        topk_logits=soft_logits,
        lse=soft_lse,
        lse_known=lse_known,
        soft_valid=soft_valid,
    )
    return out


def causal_mask_from_valid(valid: torch.Tensor) -> torch.Tensor:
    """[B,T] validity -> [B,1,T,T] bool attention mask (causal + no padding)."""
    b, t = valid.shape
    causal = torch.ones(t, t, dtype=torch.bool, device=valid.device).tril()
    return causal.view(1, 1, t, t) & valid.view(b, 1, 1, t)


def truncate(sample: Sample, max_len: int) -> Sample:
    if len(sample) <= max_len:
        return sample
    return Sample(
        next_token_ids=sample.next_token_ids[:max_len],
        hc_hidden=sample.hc_hidden[:max_len],
        positions=sample.positions[:max_len],
        labels=sample.labels[:max_len],
        input_ids=sample.input_ids[:max_len],
        greedy_consistent=sample.greedy_consistent[:max_len],
        source=sample.source,
        doc_hash=sample.doc_hash,
        topk_ids=None if sample.topk_ids is None else sample.topk_ids[:max_len],
        topk_logits=(
            None if sample.topk_logits is None else sample.topk_logits[:max_len]
        ),
        lse=None if sample.lse is None else sample.lse[:max_len],
        tap_hidden=(
            None if sample.tap_hidden is None else sample.tap_hidden[:max_len]
        ),
        pieces=sample.pieces,
    )


def length_bucket_batches(
    samples: Iterator[Sample],
    tokens_per_batch: int,
    window: int = 512,
    max_len: int = 2048,
    seed: int = 0,
    min_len: int = 8,
) -> Iterator[List[Sample]]:
    """Group samples into batches of ~`tokens_per_batch` *padded* tokens.

    Sample lengths vary from a few hundred tokens (short chat turns) to the
    2048-token chunk cap (long agent sessions), so a fixed sample count would make step size -- and therefore VRAM and step time --
    swing by an order of magnitude.  Sorting inside a window before grouping
    keeps padding waste low; the resulting batches are then shuffled so the
    optimizer does not see all the short documents first.
    """
    rng = random.Random(seed)

    def flush(buf: List[Sample]) -> List[List[Sample]]:
        buf.sort(key=len)
        out: List[List[Sample]] = []
        cur: List[Sample] = []
        for s in buf:
            # buf is ascending, so s is the longest of cur + [s]
            if cur and (len(cur) + 1) * len(s) > tokens_per_batch:
                out.append(cur)
                cur = [s]
            else:
                cur.append(s)
        if cur:
            out.append(cur)
        rng.shuffle(out)
        return out

    buf: List[Sample] = []
    for sample in samples:
        if len(sample) < min_len:
            continue
        buf.append(truncate(sample, max_len))
        if len(buf) >= window:
            yield from flush(buf)
            buf = []
    if buf:
        yield from flush(buf)


class PrefetchLoader:
    """Collate batches on a background thread so the GPU never waits on I/O.

    The dump is mmap-backed, so the page-in happens inside ``collate``; doing it
    on a worker thread overlaps it with the previous step's compute.

    Always used as an iterator or a context manager -- both shut the worker down
    cleanly.  A daemon thread left blocked on ``queue.put`` at interpreter exit
    aborts the process, which is why ``close()`` drains before joining.
    """

    def __init__(self, batch_iter: Iterator[List[Sample]], depth: int = 4,
                 pad_label: int = -100):
        self._iter = batch_iter
        self._q: "queue.Queue" = queue.Queue(maxsize=depth)
        self._pad_label = pad_label
        self._err: Optional[BaseException] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._work, daemon=True)
        self._thread.start()

    def _work(self) -> None:
        try:
            for group in self._iter:
                if self._stop.is_set():
                    break
                batch = collate(group, self._pad_label)
                while not self._stop.is_set():
                    try:
                        self._q.put(batch, timeout=0.5)
                        break
                    except queue.Full:
                        continue
        except BaseException as exc:  # noqa: BLE001
            self._err = exc
        finally:
            while not self._stop.is_set():
                try:
                    self._q.put(None, timeout=0.5)
                    break
                except queue.Full:
                    continue

    def __iter__(self):
        try:
            while True:
                item = self._q.get()
                if item is None:
                    if self._err is not None:
                        raise self._err
                    return
                yield item
        finally:
            self.close()

    def close(self) -> None:
        self._stop.set()
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._thread.join(timeout=10)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def load_fixed_eval(index_path: str, max_rows: Optional[int] = None) -> List[Sample]:
    """Load the frozen held-out subset built by extract/build_eval_set.py.

    The index names only the files that actually contain eval sequences, so an
    eval pass reads a small fraction of the dump instead of all of it.

    The eval set is a pointer into one dump directory.  Its pieces are
    resolved against the recorded ``dump_dir``, or -- if the dump was moved --
    against the directory holding the eval set (its default location).  A
    piece that is missing or a document whose ids no longer hash to the
    recorded doc_hash is an error: silently scoring a smaller or different set
    would make evals incomparable.  Rebuild it with extract/build_eval_set.py.
    """
    with open(index_path) as handle:
        index = json.load(handle)
    out: List[Sample] = []
    rows = 0
    if "docs" in index:
        # Current format: every document with its pieces, so a request that
        # chunked prefill split over several files is loaded joined.
        docs = index["docs"]
        root = None
        if docs and "pieces" in docs[0]:
            for cand in (index.get("dump_dir"), os.path.dirname(os.path.abspath(index_path))):
                if cand and all(os.path.exists(os.path.join(cand, p))
                                for p in docs[0]["pieces"]):
                    root = cand
                    break
            if root is None:
                raise SystemExit(f"{index_path}: its dump ({index.get('dump_dir')}) is "
                                 "not there any more; rebuild the eval set")
        for doc in docs:
            files = ([os.path.join(root, p) for p in doc["pieces"]] if root is not None
                     else doc["files"])
            missing = [p for p in files if not os.path.exists(p)]
            if missing:
                raise SystemExit(f"{index_path}: eval document {doc['doc_hash']} lost "
                                 f"{missing[0]}; rebuild the eval set")
            sample = load_concat(files, doc["seqs"])
            if sample.doc_hash != doc["doc_hash"]:
                raise SystemExit(f"{index_path}: eval document {doc['doc_hash']} changed "
                                 f"on disk (ids now hash to {sample.doc_hash}); rebuild "
                                 "the eval set")
            out.append(sample)
            rows += len(sample)
            if max_rows and rows >= max_rows:
                return out
        return out
    # Older eval sets list files and hashes only (single-file documents).  The
    # same rule: a lost file or document is an error, not a smaller eval set.
    wanted = set(index["doc_hashes"])
    missing = [p for p in index["files"] if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"{index_path}: eval file {missing[0]} is gone "
                         f"({len(missing)} missing); rebuild the eval set")
    found = set()
    for path in index["files"]:
        for sample in load_dump_file(path):
            if sample.doc_hash not in wanted:
                continue
            found.add(sample.doc_hash)
            out.append(sample)
            rows += len(sample)
            if max_rows and rows >= max_rows:
                return out
    if wanted - found:
        raise SystemExit(f"{index_path}: {len(wanted - found)} eval document(s) are no "
                         "longer in their files; rebuild the eval set")
    return out


def write_manifest(path: str, records: Sequence[dict]) -> None:
    """Append rows (one flushed line each; a partial last line is repaired first)."""
    from .fileio import open_jsonl_append

    with open_jsonl_append(path) as handle:
        for record in records:
            handle.write(record)


__all__ = [
    "Sample",
    "NEG_INF",
    "truncate",
    "length_bucket_batches",
    "PrefetchLoader",
    "load_fixed_eval",
    "DumpDataset",
    "iter_dump_files",
    "read_tap_layers",
    "load_dump_file",
    "collate",
    "causal_mask_from_valid",
    "write_manifest",
]
