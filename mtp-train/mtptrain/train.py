#!/usr/bin/env python3
"""MTP-head fine-tuning loop.

Runs unchanged on CPU with ``--tiny --synthetic`` (the smoke test) and on one
GPU with the real checkpoint.

Base objective:  ``CE(logits[t], target_argmax[t+1])`` with the row inputs
``hc_hidden[t]`` and ``embed(input_ids[t+1])``.  Router and QSA indexer frozen,
every other ``mtp.*`` tensor trainable.

``--soft-alpha`` mixes in distribution distillation against the dumped top-K
(``(1-alpha) * hard CE + alpha * soft CE``, see loss.py); ``--eval-renewal``
selects the best checkpoint on the serving-process acceptance length
(renewal.py) instead of the length-biased ``accept@k``.  Both default to off,
so an unflagged run is the plain hard-label objective (plus ``--rollout-k`` if
given).

Checkpoints and resume
----------------------
Every ``latest.pt`` -- the periodic ``--ckpt-every`` one, the one at each
epoch boundary and the final one -- carries the whole training state: model,
optimizer (the LR schedule is a pure function of the step), step, the number
of completed epochs, the number of micro-batches already consumed in the
current epoch, a pending gradient accumulation (``accum`` and the partial
gradients, only when a save falls inside one), the token counter, the Python,
torch CPU and CUDA RNG states, ``best`` (score, step, criterion, summary) and
``history``.  Every file is written to a temporary name, fsynced and renamed.

``--resume`` restores all of it.  The data stream is a pure function of
(dump contents, ``--seed``, epoch, the batching flags), so the resumed run
rebuilds the current epoch's stream and skips the micro-batches already
consumed: steps, epochs, data order, history and best match an uninterrupted
run.  Approximate or costly parts:

* skipping re-reads (but does not train on) the skipped part of the epoch from
  the dump, so a resume late in a long epoch spends some I/O first;
* the position is exact only if the training documents (their doc hashes,
  i.e. the ids that decide the order) and the flags that shape the stream
  (``--seed``, ``--holdout``, ``--max-len``, ``--tokens-per-step``,
  ``--bucket-window``, ``--batch-size``, ``--grad-accum``,
  ``--exclude-buckets``, ``--synthetic*``) are unchanged -- a difference is
  reported at resume;
* numpy's RNG is not used by the trainer and is not saved; on GPU the kernels
  themselves are not bitwise deterministic, so a resumed GPU run follows the
  same data and schedule but not identical floating point.

Comparability: a fresh run records everything that decides what a score
means -- ``--select-metric``, the content of the eval data (a digest of every
tensor the eval reads: hidden states, labels, top-K, lse, masks),
``--eval-rows/-batch/-batches/-starts``, ``--chain-ks``, ``--holdout``,
``--seed``, the renewal settings and the content of the documents they
select, the token map's content, the tap layers, and the base model's loaded
weights and architecture config -- in ``latest.pt`` and ``best.json``.  A resume
that changes any of them is refused with the list of differences;
``--reset-best`` accepts the change, moves the old best files to
``--out/.reset-best-<time>/``, marks the break in the history and restarts the
selection.  Speed and logging knobs (``--log-every``, ``--prefetch``,
``--ckpt-every`` ...) may change freely.

Inputs are checked before anything is read or written: an explicit path that
does not exist, an option the chosen mode would ignore, an empty eval set,
zero training documents or an empty renewal selection stop the run (see
``validate_inputs``).

``best-mtp.safetensors`` records its own ``best`` entry in the file metadata.
On resume from a checkpoint in the same ``--out``, the better of that and the
checkpoint's ``best`` wins, so an eval that ran after the last checkpoint is
never overwritten by a worse one; ``best`` is only replaced by a strictly
better score.  History after the checkpoint is recomputed, not duplicated.

Run identity and --out
----------------------
A fresh run gets a ``run_id``.  It is stored in ``latest.pt``, in ``best``
(hence ``best.json``) and in the header of every weights file the run writes
(``mtp_run_id``).  The rules for ``--out``:

* a fresh run needs an --out without earlier results; with ``--overwrite``
  the earlier results (``latest.pt``, ``best*``, ``history.json``,
  ``final-mtp.safetensors``, ``mtp-step*.safetensors`` and their temp files)
  are first *moved* into ``--out/.previous-<time>/`` -- nothing is deleted;
* ``--resume`` must point at a checkpoint inside ``--out`` (to branch a run,
  copy its directory and resume the copy); ``--overwrite`` and ``--resume``
  are mutually exclusive;
* on resume, weights files and ``best.json`` from another run (a different
  ``run_id``) are moved aside the same way; ``best`` is taken from the
  ``best-mtp.safetensors`` header of *this* run, and if this run's best
  weights are gone the best selection restarts (with a warning) rather than
  pointing at a file that does not exist.  ``best.json`` and ``history.json``
  are rewritten at once, so they, the best file's header and ``latest.pt``
  agree from the first step on.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import time
import uuid
from typing import Dict, Iterator, List, Optional

import torch
import torch.nn.functional as F

from .config import DEFAULT_MODEL_DIR, MTPConfig
from .data import (
    DumpDataset,
    PrefetchLoader,
    Sample,
    causal_mask_from_valid,
    collate,
    length_bucket_batches,
    truncate,
)
from .evaluate import evaluate
from .cli import split_list
from .fileio import (
    digest_parts,
    file_digest,
    move_aside,
    write_atomic,
    write_json_atomic,
)
from .loss import chunked_ce, chunked_mixed_ce, soft_targets_from_batch
from .model import MTPHead
from .renewal import RenewalResult, renewal_evaluate
from .rollout import chain_losses, step_weights
from . import weights as W


# ----------------------------------------------------------------- schedule
def lr_at(step: int, base_lr: float, warmup: int, total: int, min_ratio: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    if total <= warmup:
        return base_lr
    progress = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return base_lr * (min_ratio + (1.0 - min_ratio) * cosine)


# -------------------------------------------------------------- synthetic
def synthetic_samples(
    config: MTPConfig, count: int, length: int, seed: int = 0, topk: int = 0
) -> List[Sample]:
    """Random but shape-correct samples so the loop can be exercised on CPU.

    ``topk`` > 0 also fabricates soft targets in the dump's layout (the argmax
    is always the top-1 id, as a real dump guarantees), so ``--soft-alpha`` has
    a code path to exercise on the smoke test.
    """
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(count):
        ids = torch.randint(0, config.vocab_size, (length,), generator=g)
        labels = torch.randint(0, config.vocab_size, (length,), generator=g)
        topk_ids = topk_logits = lse = None
        if topk > 0:
            k = min(topk, config.vocab_size)
            topk_ids = torch.randint(
                0, config.vocab_size, (length, k), generator=g
            )
            topk_ids[:, 0] = labels  # top-1 == target_argmax, as in a real dump
            topk_logits = torch.sort(
                torch.randn(length, k, generator=g) * 2.0, dim=-1, descending=True
            ).values
            # A plausible tail: the stored K never holds quite all the mass.
            lse = torch.logsumexp(topk_logits, dim=-1) + 0.1
        out.append(
            Sample(
                next_token_ids=ids,
                hc_hidden=torch.randn(
                    length, config.hc_size, generator=g, dtype=torch.float32
                ).to(torch.bfloat16),
                positions=torch.arange(length),
                labels=labels,
                input_ids=ids,
                greedy_consistent=torch.ones(length, dtype=torch.bool),
                source="synthetic",
                doc_hash=f"synthetic-{seed}-{i}",
                topk_ids=topk_ids,
                topk_logits=topk_logits,
                lse=lse,
                tap_hidden=(
                    None
                    if not config.n_taps
                    else torch.randn(
                        length, config.n_taps * config.hc_size, generator=g,
                        dtype=torch.float32,
                    ).to(torch.bfloat16)
                ),
            )
        )
    return out


def fixed_batches(samples, batch_size: int, max_len: int):
    """Fixed sample-count batching -- only used by the tiny/synthetic smoke path."""
    buf: List[Sample] = []
    for sample in samples:
        buf.append(truncate(sample, max_len))
        if len(buf) == batch_size:
            yield buf
            buf = []
    if buf:
        yield buf


# ---------------------------------------------------------------- training
def resolve_tap_layers(args) -> tuple:
    """--tap-layers, else whatever the training dump recorded (possibly none)."""
    from .data import read_tap_layers

    if args.tap_layers is not None:
        return tuple(args.tap_layers)
    if args.synthetic or not args.dump_dir:
        return ()
    return read_tap_layers(os.path.expanduser(args.dump_dir))


def weights_digest(model) -> str:
    """blake2b over every tensor of the head plus its frozen embed / lm_head.

    The frozen heads are not in ``state_dict`` (nor in a checkpoint): on
    resume they come from --model-dir again, so their content is part of what
    a best score means.  Reads each tensor once (seconds per GB on CPU).
    """
    import hashlib

    from .data import update_digest

    h = hashlib.blake2b(digest_size=12)
    items = sorted(model.state_dict().items())
    items += [("embed_tokens", model.embed_tokens), ("lm_head", model.lm_head)]
    for name, t in items:
        if t is not None:
            update_digest(h, name, t)
    return h.hexdigest()


def build_model(args) -> MTPHead:
    if args.tiny:
        config = MTPConfig.tiny()
    else:
        config = MTPConfig.from_pretrained(args.model_dir)
    config = config.with_(tap_layers=resolve_tap_layers(args))
    model = MTPHead(config)
    if args.tiny:
        g = torch.Generator().manual_seed(args.seed)
        with torch.no_grad():
            for p in model.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.02)
        model.set_frozen_heads(
            torch.randn(config.vocab_size, config.hidden_size, generator=g) * 0.02,
            torch.randn(config.vocab_size, config.hidden_size, generator=g) * 0.02,
        )
    else:
        W.load_into(model, args.model_dir, device="cpu", dtype=torch.float32)
    # content identity of the base weights (before any --init)
    model.base_digest = weights_digest(model)
    if not args.tiny:
        if args.init:
            # Warm start: keep the released embed/lm_head, replace the 31 mtp.*
            # tensors with a previous run's BF16 snapshot.  Unlike --resume this
            # starts a fresh optimizer and a fresh LR schedule.
            from safetensors.torch import load_file

            state = load_file(args.init)
            state = {k[len("mtp."):] if k.startswith("mtp.") else k: v
                     for k, v in state.items()}
            missing, unexpected = model.load_state_dict(
                {k: v.to(torch.float32) for k, v in state.items()}, strict=False
            )
            missing = W.unloaded_names(missing)
            if missing or unexpected:
                raise SystemExit(
                    f"--init mismatch: missing={missing} unexpected={unexpected}"
                )
            print(f"[init] warm start from {args.init} ({len(state)} tensors)")
    if args.train_taps_only:
        if not config.n_taps:
            raise SystemExit("--train-taps-only: the head has no tap layers (the dump "
                             "records none and --tap-layers gave none); nothing to train")
        model.freeze_all_but_taps()
    else:
        model.freeze_non_trainable()
    return model


def make_optimizer(args, params):
    if args.optimizer == "adamw8bit":
        # An explicitly chosen optimizer is used or the run refuses; a silent
        # fallback would change memory use and the optimizer state format.
        try:
            import bitsandbytes as bnb
        except Exception as exc:  # pragma: no cover - depends on the env
            raise SystemExit(f"--optimizer adamw8bit needs bitsandbytes ({exc})")
        return bnb.optim.AdamW8bit(
            params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
        )
    return torch.optim.AdamW(
        params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay, fused=False
    )


def save_weights_bf16(path: str, model, metadata: Optional[dict] = None) -> None:
    """Weights-only BF16 snapshot: 5.2 GB instead of the 31 GB full checkpoint,
    and exactly what writeback/write_mtp.py consumes.  Atomic (temp + fsync +
    rename); ``metadata`` values are stored as strings in the file header,
    next to ``mtp_tensors_digest``, the digest of the tensors written (see
    ``tensors_digest``): what a header says about the file is only believed
    for the content it was written with."""
    from safetensors.torch import save_file

    from .weights import export_state_dict

    state = export_state_dict(model)
    meta = {k: str(v) for k, v in (metadata or {}).items()}
    meta["mtp_tensors_digest"] = tensors_digest(state)
    write_atomic(path, lambda tmp: save_file(state, tmp, metadata=meta))


def tensors_digest(state: Dict[str, torch.Tensor]) -> str:
    """blake2b over every tensor (name, dtype, shape, bytes) in name order."""
    import hashlib

    from .data import update_digest

    h = hashlib.blake2b(digest_size=12)
    for name in sorted(state):
        update_digest(h, name, state[name])
    return h.hexdigest()


def file_tensors_digest(path: str) -> Optional[str]:
    """``tensors_digest`` of a safetensors file's content (None if unreadable)."""
    try:
        from safetensors.torch import load_file

        return tensors_digest(load_file(path))
    except Exception:  # noqa: BLE001
        return None


def weights_header(path: str) -> dict:
    """The string metadata of a weights file ({} if unreadable or missing)."""
    if not os.path.exists(path):
        return {}
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt") as f:
            return dict(f.metadata() or {})
    except Exception:  # noqa: BLE001
        return {}


def best_from_snapshot(path: str) -> Optional[dict]:
    """The ``best`` entry stored in a best-mtp.safetensors header, if any."""
    raw = weights_header(path).get("mtp_best")
    try:
        return json.loads(raw) if raw else None
    except ValueError:
        return None


CHECKPOINT_FORMAT = 2


def save_checkpoint(path: str, model, optimizer, step: int, meta: dict,
                    epoch: int = 0, best: Optional[dict] = None,
                    history: Optional[list] = None, epoch_batches: int = 0,
                    accum: int = 0, tokens_seen: int = 0,
                    run_id: Optional[str] = None) -> None:
    """Crash-safe: write a temp file next to the target, fsync, then rename.

    ``epoch`` is the number of *completed* passes over the training stream and
    ``epoch_batches`` the micro-batches already consumed from the current one;
    the stream is deterministic in ``seed + epoch`` (see ``make_stream``), so a
    resume skips exactly those.  ``accum`` > 0 means the save fell inside a
    gradient accumulation: the partial gradients are stored too.  ``best``
    and ``history`` ride along so a resumed stretch cannot overwrite a better
    earlier checkpoint or lose the curve.
    """
    params = list(model.trainable_parameters()) if hasattr(
        model, "trainable_parameters") else list(model.parameters())
    payload = {
        "format": CHECKPOINT_FORMAT,
        "run_id": run_id,
        "step": step,
        "epoch": epoch,
        "epoch_batches": epoch_batches,
        "accum": accum,
        "pending_grads": (
            [None if p.grad is None else p.grad.detach().cpu() for p in params]
            if accum else None
        ),
        "tokens_seen": tokens_seen,
        "best": best,
        "history": history,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "meta": meta,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": (torch.cuda.get_rng_state_all()
                     if torch.cuda.is_available() and torch.cuda.is_initialized()
                     else None),
        "py_rng": random.getstate(),
    }
    write_atomic(path, lambda tmp: torch.save(payload, tmp))


def load_checkpoint(path: str, model, optimizer) -> dict:
    # weights_only: everything save_checkpoint writes is tensors and plain
    # containers (the Python RNG state is a tuple of ints), so no pickled
    # objects -- and no code -- are loaded from the file.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng"])
    random.setstate(payload["py_rng"])
    if payload.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    accum = int(payload.get("accum", 0))
    if accum and payload.get("pending_grads") is not None:
        params = list(model.trainable_parameters()) if hasattr(
            model, "trainable_parameters") else list(model.parameters())
        for p, g in zip(params, payload["pending_grads"]):
            p.grad = None if g is None else g.to(device=p.device, dtype=p.dtype)
    return {
        "step": int(payload["step"]),
        "epoch": int(payload.get("epoch", 0)),
        "epoch_batches": int(payload.get("epoch_batches", 0)),
        "accum": accum,
        "tokens_seen": int(payload.get("tokens_seen", 0)),
        "best": payload.get("best"),
        "history": payload.get("history") or [],
        "meta": payload.get("meta") or {},
        "format": int(payload.get("format", 1)),
        "run_id": payload.get("run_id"),
    }


def selection_score(summary: dict, mode: str):
    """Which checkpoint counts as 'best'.

    ``renewal`` (the default once ``--eval-renewal`` is on) is the only
    criterion that computes the server's own metric (tokens per verify, by
    replaying the verify-and-advance process): the mean of ``accept_len`` at
    k=3 and k=15 on held-out self-generated documents.  It is still an
    offline estimate; check it against a server A/B.
    ``accept@k`` is length-biased -- it starts chains at uniformly random rows,
    over-sampling the inside of long accepted runs -- so it can rank heads
    differently from the server.
    accept@3 alone also ignores long-chain regressions, and accept@15 alone can
    be noisy, so
    ``combined`` trades them off: accept@3 + accept@15/5 (the /5 puts a 15-step
    gain on roughly the same scale as a 3-step one).  agreement@1 breaks ties.

    -> (mode, score).  ``score`` is None when a metric it needs is missing or
    unmeasured (no chain window, no verify, no row); such an eval never
    updates or is compared with the best.
    """
    needed = selection_metrics(mode)
    vals = {m: summary.get(m) for m in needed}
    if any(v is None for v in vals.values()):
        return mode, None
    if mode == "combined":
        s = vals["accept@3"] + vals["accept@15"] / 5.0
    else:
        s = vals[needed[0]]
    tie = summary.get("agreement@1")
    return mode, s + (1e-4 * tie if tie is not None else 0.0)


def _fmt_metric(x) -> str:
    return "n/a" if x is None else f"{x:.5f}"


def selection_metrics(mode: str) -> List[str]:
    """The summary keys the selection score ``mode`` is computed from."""
    if mode.startswith("renewal@"):
        return [f"accept_len@{mode.split('@', 1)[1]}"]
    if mode == "renewal":
        return ["renewal_mean"]
    if mode == "combined":
        return ["accept@3", "accept@15"]
    if mode == "agreement":
        return ["agreement@1"]
    return [f"accept@{mode}"]


def load_eval_samples(args) -> List[Sample]:
    """The fixed held-out subset, so every eval is comparable and takes minutes."""
    from .data import load_fixed_eval

    # An explicit --eval-set is used or the run refuses (validate_inputs checked
    # that it exists); only without one is the dump's held-out split scored.
    if args.eval_set:
        # Which dump it came from does not matter; that its documents are not
        # training documents is checked by content (check_train_eval_overlap).
        path = os.path.expanduser(args.eval_set)
        out = load_fixed_eval(path, max_rows=args.eval_rows or None)
        where = f"--eval-set {args.eval_set}"
    else:
        ds = DumpDataset(args.dump_dir, split="eval", holdout_frac=args.holdout,
                         seed=args.seed)
        out, rows = [], 0
        for s in ds:
            out.append(truncate(s, args.max_len))
            rows += len(out[-1])
            if args.eval_rows and rows >= args.eval_rows:
                break
        where = f"the held-out split of --dump-dir {args.dump_dir} (--holdout {args.holdout})"
    if not out:
        raise SystemExit(f"no eval documents in {where}; an eval over nothing would "
                         "score 0 at every step")
    return out


def _manifest_keys(path: Optional[str]) -> Dict[str, set]:
    """doc_hash -> every split key a manifest gives it ({} without a manifest)."""
    from .fileio import read_jsonl

    out: Dict[str, set] = {}
    if path and os.path.isfile(path):
        for r in read_jsonl(path):
            out.setdefault(r["doc_hash"], set()).add(r.get("split_key") or r["doc_hash"])
    return out


def eval_identities(args, eval_samples, renewal_set) -> List[tuple]:
    """-> [(source name, {doc_hash: split keys})] for every eval source of the run.

    The keys name the source documents; an empty set means the source's keys
    are unknown (an older eval set whose dump has no manifest), so only the
    content hash can be checked.
    """
    out = []
    if args.synthetic:
        out.append(("synthetic eval", {s.doc_hash: set() for s in eval_samples}))
    elif args.eval_set:
        path = os.path.expanduser(args.eval_set)
        with open(path) as fh:
            data = json.load(fh)
        keys = {d["doc_hash"]: set(d["split_keys"]) for d in data.get("docs", [])
                if d.get("split_keys")}
        if len(keys) < len(eval_samples):  # an older eval set: ask its dump
            for root in (data.get("dump_dir"), os.path.dirname(os.path.abspath(path))):
                if root:
                    for h, ks in _manifest_keys(os.path.join(root, "manifest.jsonl")).items():
                        keys.setdefault(h, ks)
        out.append((f"--eval-set {args.eval_set}",
                    {s.doc_hash: keys.get(s.doc_hash, set()) for s in eval_samples}))
    else:
        ds = DumpDataset(args.dump_dir, split="eval", holdout_frac=args.holdout,
                         seed=args.seed)
        out.append(("the held-out split of --dump-dir",
                    {s.doc_hash: ds.keys_of(s.doc_hash) for s in eval_samples}))
    if renewal_set is not None:
        src = args.renewal_dump_dir or args.dump_dir
        keys = _manifest_keys(os.path.expanduser(
            args.renewal_manifest or os.path.join(src, "manifest.jsonl")))
        out.append(("the renewal documents",
                    {s.doc_hash: keys.get(s.doc_hash, set()) for s in renewal_set.samples}))
    return out


def check_train_eval_overlap(train_ids: Dict[str, set], sources: List[tuple]) -> None:
    """Refuse a run whose eval documents are also training documents.

    Checked by content (the doc_hash of the token ids) and by source document
    (the manifest's split keys: another window or re-chunking of a training
    document is not held out either).  Every eval source is checked: the
    frozen eval set (which may come from another dump), the held-out split and
    the renewal documents.
    """
    train_keys = set().union(*train_ids.values()) if train_ids else set()
    problems = []
    for name, ids in sources:
        same_hash = sorted(set(ids) & set(train_ids))
        same_doc = sorted({h for h, ks in ids.items() if ks & train_keys} - set(same_hash))
        if same_hash or same_doc:
            problems.append(
                f"{name}: {len(same_hash)} of {len(ids)} documents are training "
                f"documents (same token ids{', e.g. ' + same_hash[0] if same_hash else ''})"
                f", {len(same_doc)} more come from a training source document"
                f"{' (e.g. ' + same_doc[0] + ')' if same_doc else ''}")
        unknown = sum(1 for ks in ids.values() if not ks)
        if unknown and not name.startswith("synthetic"):
            print(f"[data] note: {unknown} documents of {name} have no split key "
                  "(an older eval set without a manifest); checked by content only",
                  flush=True)
    if problems:
        raise SystemExit("eval documents overlap the training data:\n  "
                         + "\n  ".join(problems)
                         + "\nBuild the eval set from the training dump with the same "
                           "--seed/--holdout, or train on a dump that excludes them")


def _read_eval_set_header(path: str) -> dict:
    """The eval set's JSON minus the per-document lists; SystemExit if unreadable."""
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"--eval-set {path}: not a readable eval set ({exc})")
    if not isinstance(data, dict) or not ("docs" in data or "doc_hashes" in data):
        raise SystemExit(f"--eval-set {path}: not an eval set from build_eval_set.py")
    return {k: v for k, v in data.items() if k not in ("docs", "files", "doc_hashes")}


def load_renewal_set(args):
    """The fixed held-out self-generated subset the renewal evaluator scores.

    Loaded once and reused by every eval: reading the dump rows costs far more
    than the simulation itself, and re-drawing the documents would make two
    evals of the same run incomparable.
    """
    from .select import load_selfgen_samples, select_selfgen_docs

    dump_dir = args.renewal_dump_dir or args.dump_dir
    chosen, index = select_selfgen_docs(
        dump_dir,
        manifest=args.renewal_manifest,
        split="eval",
        split_seed=args.seed,
        holdout_frac=args.holdout,
        buckets=split_list(args.renewal_buckets, "--renewal-buckets") or None,
        per_bucket=args.renewal_docs,
        min_gen=args.renewal_min_gen,
        seed=args.renewal_seed,
        greedy_only=args.renewal_greedy_only,
    )
    sel = load_selfgen_samples(dump_dir, chosen, index, max_len=args.renewal_max_len)
    if not len(sel):
        raise SystemExit(
            f"--eval-renewal: no held-out self-generated document in {dump_dir} "
            f"(buckets={args.renewal_buckets or 'all'}, min_gen={args.renewal_min_gen}, "
            f"greedy_only={args.renewal_greedy_only}, --holdout {args.holdout}); "
            "renewal selection would score 0 at every eval")
    print(
        f"[renewal] {len(sel)} self-generated eval docs "
        f"({sel.generated_rows:,} generated rows) {json.dumps(sel.counts())}",
        flush=True,
    )
    return sel


@torch.no_grad()
def run_renewal(model, args, sel, allowed=None) -> dict:
    """-> {accept_len@k, verifies@k, renewal_mean, renewal_by_bucket}."""
    if sel is None or not len(sel):
        return {}
    t0 = time.time()
    results = renewal_evaluate(
        model, sel.samples, sel.gen_start_rows, ks=tuple(args.renewal_ks),
        batch_size=args.renewal_batch, allowed=allowed,
    )
    bucket_of = sel.bucket_of()
    out: dict = {}
    lens = []
    for k, res in sorted(results.items()):
        summary = res.summary()
        al = summary["accept_len"]  # None when no verify was simulated
        out[f"accept_len@{k}"] = None if al is None else round(al, 5)
        out[f"verifies@{k}"] = summary["verifies"]
        lens.append(al)
        per_bucket: dict = {}
        for s in res.seqs:
            per_bucket.setdefault(bucket_of[s.doc_hash], []).append(s)
        out[f"by_bucket@{k}"] = {}
        for b, seqs in sorted(per_bucket.items()):
            al = RenewalResult(k=k, seqs=seqs).summary()["accept_len"]
            out[f"by_bucket@{k}"][b] = None if al is None else round(al, 4)
    # the mean over ks is defined only if every k was measured
    out["renewal_mean"] = (None if not lens or any(x is None for x in lens)
                           else round(sum(lens) / len(lens), 5))
    out["renewal_seconds"] = round(time.time() - t0, 1)
    model.train()
    return out


def run_eval(model, args, eval_samples) -> dict:
    res = evaluate(
        model, iter(eval_samples), batch_size=args.eval_batch,
        chain_ks=tuple(args.chain_ks), starts_per_sample=args.eval_starts,
        max_batches=args.eval_batches or None,
    )
    model.train()
    return res.summary()


def resolve_args(args):
    """Defaults that depend on other flags."""
    if args.select_metric is None:
        args.select_metric = "renewal" if args.eval_renewal else "combined"
    if args.eval_renewal and not (args.renewal_dump_dir or args.dump_dir):
        raise SystemExit("--eval-renewal needs --dump-dir or --renewal-dump-dir")
    if not 0.0 <= args.soft_alpha <= 1.0:
        raise SystemExit("--soft-alpha must be in [0, 1]")
    if args.soft_temp <= 0.0:
        raise SystemExit("--soft-temp must be positive")
    return args


def _dump_has_tensor(dump_dir: str, name: str) -> bool:
    """True if any dump file carries ``name`` (reads safetensors headers only)."""
    from safetensors import safe_open

    from .data import iter_dump_files

    for path in iter_dump_files(os.path.expanduser(dump_dir)):
        try:
            with safe_open(path, framework="pt", device="cpu") as f:
                if name in f.keys():
                    return True
        except Exception:  # noqa: BLE001  (unreadable files are the index's business)
            continue
    return False


def validate_inputs(args) -> None:
    """Refuse, before anything is read or written, what would otherwise be
    silently replaced, defaulted or ignored.

    Every path given explicitly must exist with the right type; an option that
    the chosen mode would not use is an error, not a no-op.  Called on the
    arguments as given (before ``resolve_args`` fills defaults).
    """
    errs: List[str] = []

    def need(opt: str, path: Optional[str], kind: str) -> bool:
        full = os.path.expanduser(path)
        ok = os.path.isdir(full) if kind == "dir" else os.path.isfile(full)
        if not ok:
            errs.append(f"{opt} {path}: no such {'directory' if kind == 'dir' else 'file'}")
        return ok

    synthetic = bool(args.synthetic)
    renewal_src = args.renewal_dump_dir or args.dump_dir
    for opt, val in (("--renewal-buckets", args.renewal_buckets),
                     ("--exclude-buckets", args.exclude_buckets)):
        try:
            split_list(val or "", opt)
        except SystemExit as exc:
            errs.append(str(exc))
    if args.synthetic < 0:
        errs.append("--synthetic must be >= 0")
    if not synthetic and not args.dump_dir:
        errs.append("--dump-dir is required unless --synthetic N is given")
    dump_ok = bool(args.dump_dir) and need("--dump-dir", args.dump_dir, "dir")
    if synthetic:
        # The synthetic pool replaces the dump for training and eval.
        if args.dump_dir and not (args.eval_renewal and not args.renewal_dump_dir):
            errs.append("--dump-dir is not read with --synthetic (only as the "
                        "--eval-renewal source)")
        if args.eval_set:
            errs.append("--eval-set is not used with --synthetic")
        if args.exclude_buckets:
            errs.append("--exclude-buckets is not used with --synthetic")
    if args.eval_set and need("--eval-set", args.eval_set, "file"):
        head = _read_eval_set_header(os.path.expanduser(args.eval_set))
        for key, opt in (("holdout", "--holdout"), ("seed", "--seed")):
            want = getattr(args, key)
            if key in head and head[key] != want:
                errs.append(f"--eval-set was built with {key}={head[key]} but {opt} is "
                            f"{want}: its documents are not held out from this "
                            "run's training split")
    if not args.tiny:
        if need("--model-dir", args.model_dir, "dir"):
            need("--model-dir config.json", os.path.join(args.model_dir, "config.json"),
                 "file")
    if args.init:
        need("--init", args.init, "file")
        if args.tiny:
            errs.append("--init is not applied with --tiny")
        if args.resume:
            errs.append("--init warm-starts a new run; --resume restores the run's "
                        "own weights (pass only one)")
    if getattr(args, "reset_best", False) and not args.resume:
        errs.append("--reset-best only applies to --resume")
    if args.out and os.path.exists(os.path.expanduser(args.out)) and not os.path.isdir(
            os.path.expanduser(args.out)):
        errs.append(f"--out {args.out}: exists and is not a directory")
    # Renewal-only inputs are ignored without --eval-renewal.
    for opt, val in (("--renewal-dump-dir", args.renewal_dump_dir),
                     ("--renewal-manifest", args.renewal_manifest),
                     ("--renewal-buckets", args.renewal_buckets),
                     ("--token-map", args.token_map)):
        if val and not args.eval_renewal:
            errs.append(f"{opt} is only used with --eval-renewal")
    if args.renewal_dump_dir:
        need("--renewal-dump-dir", args.renewal_dump_dir, "dir")
    if args.renewal_manifest:
        need("--renewal-manifest", args.renewal_manifest, "file")
    if args.token_map:
        need("--token-map", args.token_map, "file")
    if args.eval_renewal:
        if not renewal_src:
            errs.append("--eval-renewal needs --dump-dir or --renewal-dump-dir")
        if not args.renewal_ks or min(args.renewal_ks) < 1:
            errs.append("--renewal-ks needs at least one chain length >= 1")
        if args.renewal_docs < 0 or args.renewal_min_gen < 0 or args.renewal_max_len < 0:
            errs.append("--renewal-docs, --renewal-min-gen and --renewal-max-len must be >= 0")
        if args.renewal_batch < 1:
            errs.append("--renewal-batch must be >= 1")
    # Options that only act at an eval, or only write into --out.
    if not args.eval_every:
        for opt, val in (("--eval-renewal", args.eval_renewal),
                         ("--snapshot-every-eval", args.snapshot_every_eval),
                         ("--select-metric", args.select_metric)):
            if val:
                errs.append(f"{opt} needs --eval-every N (no eval runs without it)")
    if not args.out:
        for opt, val in (("--ckpt-every", args.ckpt_every),
                         ("--snapshot-every-eval", args.snapshot_every_eval),
                         ("--stop-after-epochs", args.stop_after_epochs),
                         ("--resume", args.resume)):
            if val:
                errs.append(f"{opt} needs --out")
    for opt, val, lo in (("--steps", args.steps, 1), ("--epochs", args.epochs, 1),
                         ("--grad-accum", args.grad_accum, 1),
                         ("--batch-size", args.batch_size, 1),
                         ("--max-len", args.max_len, 2),
                         ("--log-every", args.log_every, 1),
                         ("--prefetch", args.prefetch, 1),
                         ("--eval-batch", args.eval_batch, 1),
                         ("--eval-starts", args.eval_starts, 1),
                         ("--tokens-per-step", args.tokens_per_step, 0),
                         ("--eval-every", args.eval_every, 0),
                         ("--ckpt-every", args.ckpt_every, 0),
                         ("--eval-rows", args.eval_rows, 0),
                         ("--eval-batches", args.eval_batches, 0),
                         ("--stop-after-epochs", args.stop_after_epochs, 0),
                         ("--rollout-k", args.rollout_k, 0),
                         ("--logit-chunk", args.logit_chunk, 1)):
        if val < lo:
            errs.append(f"{opt} must be >= {lo} (got {val})")
    if not 0.0 <= args.holdout <= 1.0:
        errs.append("--holdout must be in [0, 1]")
    if args.eval_every and (not args.chain_ks or min(args.chain_ks) < 1):
        errs.append("--chain-ks needs at least one chain length >= 1")
    if args.optimizer == "adamw8bit":
        try:
            import bitsandbytes  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            errs.append(f"--optimizer adamw8bit needs bitsandbytes ({exc})")
    if not synthetic and dump_ok:
        from .data import read_tap_layers

        if args.soft_alpha > 0 and not _dump_has_tensor(args.dump_dir, "topk_ids"):
            errs.append("--soft-alpha > 0 but no file of --dump-dir carries top-K "
                        "targets (dump with SGLANG_MTP_DUMP_TOPK)")
        if args.tap_layers is not None:
            recorded = read_tap_layers(os.path.expanduser(args.dump_dir))
            if tuple(args.tap_layers) != tuple(recorded):
                errs.append(f"--tap-layers {args.tap_layers} but the dump records "
                            f"{list(recorded) or 'no taps'}")
    if errs:
        raise SystemExit("refusing to start:\n  " + "\n  ".join(errs))


def check_select_metric(args) -> None:
    """--select-metric must name a score the evals of this run produce."""
    m = args.select_metric
    ks = set(args.chain_ks or ())
    rks = set(args.renewal_ks or ())
    if m == "renewal" or m.startswith("renewal@"):
        if not args.eval_renewal:
            raise SystemExit(f"--select-metric {m} needs --eval-renewal")
        if m != "renewal":
            k = m.split("@", 1)[1]
            if not k.isdigit() or int(k) not in rks:
                raise SystemExit(f"--select-metric {m}: {k} is not in --renewal-ks "
                                 f"{sorted(rks)}")
    elif m == "combined":
        if not {3, 15} <= ks:
            raise SystemExit("--select-metric combined uses accept@3 and accept@15; "
                             f"--chain-ks is {sorted(ks)}")
    elif m == "agreement":
        pass
    elif m.isdigit():
        if int(m) not in ks:
            raise SystemExit(f"--select-metric {m}: not in --chain-ks {sorted(ks)}")
    else:
        raise SystemExit(f"--select-metric {m!r}: use renewal, renewal@K, combined, "
                         "agreement or a --chain-ks value")


def check_metric_measurable(args, eval_samples) -> None:
    """Refuse a --select-metric that these eval samples can never measure.

    accept@k needs at least one sample that can host a scorable k-step chain
    window (long enough, greedy-consistent); without one every eval would
    report it as unmeasured and no checkpoint could ever be selected.  A
    reported but unselected chain length only gets a warning.
    """
    from .evaluate import chain_window_count

    scored = (eval_samples[: args.eval_batches * args.eval_batch]
              if args.eval_batches else eval_samples)
    needed = {int(m.split("@", 1)[1]) for m in selection_metrics(args.select_metric)
              if m.startswith("accept@")}
    for k in sorted(set(args.chain_ks) | needed):
        if chain_window_count(scored, k):
            continue
        msg = (f"accept@{k} cannot be measured: none of the {len(scored)} scored eval "
               f"samples has a {k}-step chain window (length > {k + 2} and "
               "greedy-consistent)")
        if k in needed:
            raise SystemExit(f"--select-metric {args.select_metric}: {msg}")
        print(f"[eval] WARNING: {msg}; it is reported as unmeasured", flush=True)


# ------------------------------------------------------ comparability record
def comparison_record(args, eval_samples, renewal_set, tap_layers,
                      base_digest: Optional[str] = None, config=None) -> dict:
    """Everything that decides what a best score or an eval in the history means.

    Stored at a fresh start in ``latest.pt`` (``meta["compare"]``) and in
    ``best`` (hence ``best.json``); a resume with a different record is refused
    (see ``compare_records``).  Settings that only shape speed or logging
    (--log-every, --prefetch, --ckpt-every, ...) are not in it.

    Inputs are identified by content, never by path, name, size or ids: the
    eval and renewal data by a digest of every tensor the evals read (hidden
    states, labels, top-K, lse, masks -- ``samples_digest``), the base model
    by its loaded weights (``weights_digest``) and its architecture config,
    the token map by its file content.
    """
    from dataclasses import asdict

    from .data import samples_digest

    rec = {
        "select_metric": args.select_metric,
        "eval_data": samples_digest(eval_samples),
        "eval_rows": args.eval_rows,
        "eval_batch": args.eval_batch,
        "eval_batches": args.eval_batches,
        "eval_starts": args.eval_starts,
        "chain_ks": list(args.chain_ks),
        "holdout": args.holdout,
        "seed": args.seed,
        "synthetic": args.synthetic,
        "model_config": (digest_parts([json.dumps(asdict(config), sort_keys=True,
                                                  default=str)])
                         if config is not None else None),
        # the tensors the head starts from and scores with
        "base_weights": base_digest,
        "tap_layers": list(tap_layers),
        "renewal": None,
        "token_map": None,
        "init": None,
    }
    if args.eval_renewal:
        rec["renewal"] = {
            "docs": args.renewal_docs,
            "buckets": args.renewal_buckets,
            "ks": list(args.renewal_ks),
            "batch": args.renewal_batch,
            "max_len": args.renewal_max_len,
            "min_gen": args.renewal_min_gen,
            "greedy_only": bool(args.renewal_greedy_only),
            "seed": args.renewal_seed,
            "data": samples_digest(renewal_set.samples,
                                   list(renewal_set.gen_start_rows)
                                   + list(renewal_set.buckets)),
        }
    if args.token_map:
        rec["token_map"] = file_digest(os.path.expanduser(args.token_map))
    if args.init:
        rec["init"] = {"path": os.path.abspath(os.path.expanduser(args.init)),
                       "digest": file_digest(os.path.expanduser(args.init))}
    return rec


# Provenance only: --init is refused together with --resume, so a resumed run
# inherits the record's value instead of comparing it.
_NOT_COMPARED = ("init",)


def compare_records(old: dict, new: dict) -> dict:
    """{setting: (run's value, this invocation's value)} for every difference."""
    return {k: (old.get(k), new.get(k)) for k in sorted(set(old) | set(new))
            if k not in _NOT_COMPARED and old.get(k) != new.get(k)}


# Files ``train`` writes into --out (plus mtp-step{N}.safetensors snapshots).
OUT_FILES = ("latest.pt", "best-mtp.safetensors", "best.json", "history.json",
             "final-mtp.safetensors")


def _is_snapshot(name: str) -> bool:
    return name.startswith("mtp-step") and name.endswith(".safetensors")


def _is_artifact(name: str) -> bool:
    """A file ``train`` writes into --out, or a temp file of one."""
    if name in OUT_FILES or _is_snapshot(name):
        return True
    base = name.split(".tmp", 1)[0] if ".tmp" in name else None
    return bool(base) and (base in OUT_FILES or _is_snapshot(base))


def existing_outputs(out: str) -> List[str]:
    if not out or not os.path.isdir(out):
        return []
    return sorted(n for n in os.listdir(out) if _is_artifact(n))


def check_run_paths(args, dry: bool = False) -> None:
    """Fail before any work instead of silently restarting, mixing or overwriting.

    ``dry`` only checks (nothing is moved); ``train`` calls it that way first
    and for real once every input has been loaded and checked.

    * a --resume checkpoint must exist and, when --out is given, lie in --out;
    * --overwrite (start afresh) and --resume (continue) exclude each other;
    * a fresh run into an --out with earlier results needs --overwrite, which
      moves those results into ``--out/.previous-<time>/`` (never deletes).
    """
    overwrite = getattr(args, "overwrite", False)
    if args.resume:
        resume = os.path.abspath(os.path.expanduser(args.resume))
        if not os.path.isfile(resume):
            raise SystemExit(f"--resume {args.resume}: no such checkpoint")
        if overwrite:
            raise SystemExit("--overwrite starts a new run and --resume continues one; "
                             "pass only one of them")
        if args.out and os.path.realpath(os.path.dirname(resume)) != os.path.realpath(
                os.path.expanduser(args.out)):
            raise SystemExit(
                f"--resume {args.resume} is not inside --out {args.out}: a run is "
                "continued in its own directory (its best weights, snapshots and "
                "history live there).  To branch it, copy that directory and "
                "resume the copy")
        return
    if not args.out:
        return
    found = existing_outputs(os.path.expanduser(args.out))
    if not found:
        return
    if overwrite:
        if not dry:
            move_aside(os.path.expanduser(args.out), found)
        return
    shown = ", ".join(found[:5]) + (" ..." if len(found) > 5 else "")
    raise SystemExit(
        f"--out {args.out} already holds a run ({shown}); pass --resume "
        f"{os.path.join(args.out, 'latest.pt')} to continue it, a new --out, or "
        f"--overwrite to move it aside and start afresh"
    )


def _scored_elsewhere(best: Optional[dict], record: Optional[dict]) -> bool:
    """True if ``best`` carries a comparison record other than ``record``, or
    none at all (older trainer): an unverifiable best is never adopted."""
    return bool(best and record is not None
                and (best.get("compare") is None
                     or compare_records(best["compare"], record)))


def adopt_out_on_resume(out: str, run_id: str, ckpt: dict,
                        record: Optional[dict] = None) -> Optional[dict]:
    """Make --out consistent with the checkpoint being resumed; -> best to use.

    Weights files whose header names another run, and a ``best.json`` or
    ``best-mtp.safetensors`` that does not belong to this run, are moved
    aside.  A best file of this run wins over the checkpoint's ``best`` (it
    may be newer).  Without one, a recorded best cannot be honoured -- its
    weights are gone -- so the selection restarts.

    ``record`` is this invocation's comparison record: a best -- from the
    checkpoint or from the best file's header -- scored under another record
    (e.g. after a ``--reset-best`` to another metric whose own checkpoint was
    never written) is not comparable and is never adopted; its files are moved
    aside and the selection restarts.
    """
    ck_best = ckpt.get("best")
    if _scored_elsewhere(ck_best, record):
        print("[resume] WARNING: latest.pt's best was scored under other settings; "
              "ignored", flush=True)
        ck_best = None

    aside: List[str] = []
    for name in existing_outputs(out):
        path = os.path.join(out, name)
        if name.endswith(".safetensors") and name != "best-mtp.safetensors":
            head = weights_header(path)
            if head.get("mtp_run_id") not in (None, run_id):
                aside.append(name)
        elif ".tmp" in name:
            aside.append(name)  # leftovers of an interrupted write
    best_path = os.path.join(out, "best-mtp.safetensors")
    head = weights_header(best_path)
    disk_best = best_from_snapshot(best_path)
    if disk_best is not None and (head.get("mtp_tensors_digest") is None or
                                  head["mtp_tensors_digest"]
                                  != file_tensors_digest(best_path)):
        # The header's score is only the file's if the tensors are the ones it
        # was written with (an older file without the digest cannot show it).
        print("[resume] WARNING: best-mtp.safetensors does not hold the tensors its "
              "header was written for (or predates the check); moved aside, best "
              "selection restarts", flush=True)
        ok = False
    elif _scored_elsewhere(disk_best, record):
        print(f"[resume] WARNING: best-mtp.safetensors (step {disk_best.get('step')}) "
              "was scored under other settings "
              f"({', '.join(compare_records(disk_best.get('compare') or {}, record))}); "
              "moved aside, best selection restarts", flush=True)
        ok = False
    elif head.get("mtp_run_id") is not None:
        ok = disk_best is not None and head["mtp_run_id"] == run_id
    else:  # written before run ids existed: only if it is exactly the recorded best
        ok = (disk_best is not None and ck_best is not None
              and (disk_best.get("step"), disk_best.get("score"))
              == (ck_best.get("step"), ck_best.get("score")))
    if not ok:
        aside += [n for n in ("best-mtp.safetensors", "best.json")
                  if os.path.lexists(os.path.join(out, n))]
    move_aside(out, aside)
    if ok:
        if ck_best and ck_best.get("score", -1.0) > disk_best.get("score", -1.0):
            print(f"[resume] WARNING: latest.pt records a better best (step "
                  f"{ck_best.get('step')}) than best-mtp.safetensors holds; "
                  "using the file's", flush=True)
        elif ck_best and disk_best.get("step") != ck_best.get("step"):
            print(f"[resume] best-mtp.safetensors holds a newer best (step "
                  f"{disk_best.get('step')}) than the checkpoint", flush=True)
        return dict(disk_best, run_id=run_id)
    if ck_best and ck_best.get("step", -1) >= 0:
        print(f"[resume] WARNING: the best weights of this run (step "
              f"{ck_best.get('step')}) are not in {out}; best selection restarts",
              flush=True)
    return None


# Flags that shape the training stream; a resume with different values cannot
# land on the same data position.
STREAM_FLAGS = ("seed", "holdout", "max_len", "tokens_per_step", "bucket_window",
                "batch_size", "grad_accum", "exclude_buckets", "synthetic",
                "synthetic_repeat", "synthetic_topk", "soft_alpha")


def train_dataset(args) -> DumpDataset:
    return DumpDataset(
        args.dump_dir, split="train", holdout_frac=args.holdout, seed=args.seed,
        exclude_buckets=split_list(args.exclude_buckets, "--exclude-buckets"),
    )


class SimulatedCrash(RuntimeError):
    """Raised by the ``_crash_after_step`` test hook of ``train``."""


def train(args, _crash_after_step: Optional[int] = None) -> dict:
    """``_crash_after_step`` (tests only) aborts right after that optimizer
    step and its eval/checkpoint, like a process killed at that point."""
    validate_inputs(args)  # as given, before defaults are filled in
    args = resolve_args(args)
    check_select_metric(args)
    check_run_paths(args, dry=True)  # --out is touched only once all inputs load
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    model = build_model(args).to(device)
    total, trainable = model.num_parameters()
    print(
        f"[model] params total={total/1e9:.4f}B trainable={trainable/1e6:.3f}M "
        f"frozen={(total-trainable)/1e9:.4f}B taps={list(model.config.tap_layers)}"
    )

    optimizer = make_optimizer(args, model.trainable_parameters())
    topk = args.synthetic_topk if args.soft_alpha > 0 else 0
    train_digest = None  # which documents the dump stream holds (resume warning)
    train_ids: Dict[str, set] = {}
    if args.synthetic:
        config = model.config
        train_pool = synthetic_samples(
            config, args.synthetic, args.max_len, seed=args.seed, topk=topk
        )
        eval_samples = synthetic_samples(
            config, max(2, args.synthetic // 4), args.max_len, seed=args.seed + 1,
            topk=topk,
        )
        train_ids = {s.doc_hash: set() for s in train_pool}
    else:
        train_pool = None
        eval_samples = load_eval_samples(args)
        print(
            f"[data] fixed eval set: {len(eval_samples)} sequences, "
            f"{sum(len(s) for s in eval_samples):,} rows"
        )
        # Decided from the index and manifest alone; an empty manifest, a
        # filter that removes everything or --holdout 1 is an error here, not
        # a run that trains on nothing.
        tds = train_dataset(args)
        train_docs = tds.documents()
        train_ids = {h: tds.keys_of(h) for h, _e in train_docs}
        n_train = len(train_docs)
        train_digest = digest_parts(h for h, _e in train_docs)
        if not n_train:
            raise SystemExit(
                f"no training documents in --dump-dir {args.dump_dir} (manifest "
                f"rows on disk, minus --exclude-buckets {args.exclude_buckets or '-'}, "
                f"minus the --holdout {args.holdout} split)")
        print(f"[data] {n_train} training documents")

    # Loaded once, before the first step, so a bad --renewal-* flag fails now
    # rather than after an hour of training.
    renewal_set = load_renewal_set(args) if args.eval_renewal else None
    allowed = None
    if args.token_map:
        from .select import load_token_map

        allowed = load_token_map(args.token_map, int(model.lm_head.shape[0])).to(device)
        print(f"[token-map] {int(allowed.sum())} allowed draft ids")

    check_train_eval_overlap(train_ids, eval_identities(args, eval_samples, renewal_set))
    if args.eval_every:
        check_metric_measurable(args, eval_samples)
    record = comparison_record(args, eval_samples, renewal_set,
                               model.config.tap_layers, model.base_digest, model.config)

    run_id = uuid.uuid4().hex  # replaced by the checkpoint's on --resume
    start_step = 0
    start_epoch = 0
    skip_batches = 0
    start_accum = 0
    start_tokens = 0
    resumed_best = None
    resumed_history: List[dict] = []
    if args.resume:
        # After the inputs are loaded: the record decides whether this may continue.
        resume_path = os.path.expanduser(args.resume)
        st = load_checkpoint(resume_path, model, optimizer)
        start_step, start_epoch = st["step"], st["epoch"]
        skip_batches, start_accum = st["epoch_batches"], st["accum"]
        start_tokens = st["tokens_seen"]
        resumed_best, resumed_history = st["best"], st["history"]
        if st["format"] < 2:
            print("[resume] WARNING: checkpoint from an older trainer -- no data "
                  "position; the epoch restarts from its beginning", flush=True)
        saved_args = (st["meta"] or {}).get("args") or {}
        changed = {k: (saved_args.get(k), getattr(args, k, None)) for k in STREAM_FLAGS
                   if k in saved_args and saved_args.get(k) != getattr(args, k, None)}
        old_digest = (st["meta"] or {}).get("train_docs")
        if old_digest and train_digest and old_digest != train_digest:
            print("[resume] WARNING: the training documents differ from the run's (the "
                  "dump or its manifest changed); the data order will not match the "
                  "interrupted run", flush=True)
        if changed:
            print(f"[resume] WARNING: stream flags differ from the checkpoint's "
                  f"{changed}; the data order will not match the interrupted run",
                  flush=True)
        run_id = st["run_id"] or uuid.uuid4().hex
        saved = (st["meta"] or {}).get("compare")
        if saved is None and not args.reset_best:
            raise SystemExit(
                "--resume: the checkpoint has no comparability record (older trainer), "
                "so nothing shows that its best score and eval history were scored "
                "on the same data and model; pass --reset-best to resume with a "
                "fresh best selection, or start a new run")
        if saved is not None:
            record["init"] = saved.get("init")
            diff = compare_records(saved, record)
            if diff and not args.reset_best:
                shown = "\n  ".join(f"{k}: run {a!r} -> now {b!r}"
                                     for k, (a, b) in diff.items())
                raise SystemExit(
                    "--resume: these settings differ from the run's, so its best "
                    f"score and eval history would not be comparable:\n  {shown}\n"
                    "Resume with the run's settings, start a new run (new --out), "
                    "or pass --reset-best to restart the best selection")
            if diff:
                print(f"[resume] --reset-best: settings changed ({', '.join(diff)})",
                      flush=True)
        # --out holds this run (check_run_paths): align it with the checkpoint.
        # (--reset-best moves the best files aside itself, below)
        resumed_best = adopt_out_on_resume(args.out, run_id, st,
                                           None if args.reset_best else record)
        if args.reset_best:
            # The earlier best was scored under other settings: keep its files
            # (moved aside), mark the break in the history, select afresh.
            move_aside(args.out, [n for n in ("best-mtp.safetensors", "best.json")
                                  if os.path.lexists(os.path.join(args.out, n))],
                       label="reset-best")
            resumed_best = None
            resumed_history.append({"step": start_step, "reset_best": sorted(
                compare_records(saved or {}, record))})
        if resumed_best:
            write_json_atomic(os.path.join(args.out, "best.json"), resumed_best)
        write_json_atomic(os.path.join(args.out, "history.json"), resumed_history)
        print(f"[resume] run {run_id} step {start_step} epoch {start_epoch} +{skip_batches} "
              f"micro-batches accum {start_accum} "
              f"best={None if not resumed_best else resumed_best.get('step')}")
    else:
        check_run_paths(args)  # --overwrite moves the earlier run aside only now

    autocast_dtype = torch.bfloat16 if args.bf16 else None
    history = list(resumed_history)
    step = start_step
    tokens_seen = start_tokens
    t0 = time.time()
    accum = start_accum
    if not accum:  # a resumed partial accumulation keeps its gradients
        optimizer.zero_grad(set_to_none=True)
    stop = False
    warned_soft = False
    warned_nograd = False
    best = resumed_best or {"score": -1.0, "step": -1, "summary": None,
                            "criterion": None}
    best["run_id"] = run_id
    best["compare"] = record
    epochs_this_run = 0

    def make_stream(epoch: int):
        if args.synthetic:
            return iter(train_pool * args.synthetic_repeat)
        return train_dataset(args).shuffled(seed=args.seed + epoch)

    def checkpoint(epoch_done: int, batches: int) -> None:
        save_checkpoint(
            os.path.join(args.out, "latest.pt"), model, optimizer, step,
            {"args": vars(args), "compare": record, "train_docs": train_digest},
            epoch=epoch_done, best=best, history=history,
            epoch_batches=batches, accum=accum, tokens_seen=tokens_seen,
            run_id=run_id,
        )
        write_json_atomic(os.path.join(args.out, "history.json"), history)

    # (completed epochs, micro-batches of the current one) for the final save
    position = (start_epoch, skip_batches)
    for epoch in range(start_epoch, args.epochs):
        if stop:
            break
        stream = make_stream(epoch)
        if args.tokens_per_step > 0:
            groups = length_bucket_batches(
                stream, args.tokens_per_step, window=args.bucket_window,
                max_len=args.max_len, seed=args.seed + epoch,
            )
        else:
            groups = fixed_batches(stream, args.batch_size, args.max_len)
        epoch_batches = 0
        if epoch == start_epoch and skip_batches:
            # Resume inside this epoch: the same stream, minus what was consumed.
            groups = itertools.islice(groups, skip_batches, None)
            epoch_batches = skip_batches
        loader = PrefetchLoader(groups, depth=args.prefetch)
        try:
            for batch in loader:
                if step >= args.steps:
                    stop = True
                    break
                epoch_batches += 1
                position = (epoch, epoch_batches)
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                if args.soft_alpha > 0 and "topk_ids" not in batch and not warned_soft:
                    warned_soft = True
                    print("[warn] --soft-alpha set but the dump carries no top-K; "
                          "falling back to the hard label", flush=True)
                mask = causal_mask_from_valid(batch["valid"])
                ctx = (
                    torch.autocast(device_type=device.type, dtype=autocast_dtype)
                    if autocast_dtype is not None
                    else torch.enable_grad()
                )
                model.train()
                term_losses: List[Optional[float]] = []
                term_rows: List[int] = []

                def _backward(j, w, loss, rec):
                    # Outside autocast, as the docs prescribe; the rolled terms
                    # are independent (detached inputs), so backward-as-produced
                    # is exact and keeps one step's graph alive at a time.
                    if not loss.requires_grad:
                        # --train-taps-only: the fusion entry feeds the
                        # teacher-forced step alone, and rollout.py hands the
                        # rolled steps detached inputs, so those terms are
                        # constant in the trainable parameters.  Keep logging
                        # them as a diagnostic; there is nothing to propagate.
                        nonlocal warned_nograd
                        if not warned_nograd:
                            warned_nograd = True
                            print(f"[warn] chain term {j} has no gradient path "
                                  "to any trainable parameter; logged only",
                                  flush=True)
                        return
                    with torch.autocast(device_type=device.type, enabled=False):
                        (w * loss / args.grad_accum).backward()

                if args.rollout_k > 0:
                    with ctx:
                        stats = chain_losses(
                            model, batch, args.rollout_k,
                            step_weights(args.rollout_k, args.rollout_weight),
                            chunk=args.logit_chunk, mask_mode=args.rollout_mask,
                            on_term=_backward, soft_alpha=args.soft_alpha,
                            soft_temp=args.soft_temp, soft_steps=args.soft_steps,
                        )
                    loss_value = stats[0]["loss"]
                    term_losses = [r["loss"] for r in stats]
                    term_rows = [r["rows"] for r in stats]
                else:
                    with ctx:
                        mixed, _ = model.forward_mixed(
                            batch["next_token_ids"], batch["hc_hidden"],
                            batch["positions"], attn_mask=mask,
                            tap_hidden=batch.get("tap_hidden"),
                        )
                        flat = mixed.reshape(-1, mixed.shape[-1])
                        lab = batch["labels"].reshape(-1)
                        soft = (
                            soft_targets_from_batch(
                                batch, temperature=args.soft_temp,
                                valid=batch["valid"],
                            )
                            if args.soft_alpha > 0
                            else None
                        )
                        if soft is None:
                            loss = chunked_ce(
                                flat, model.lm_head, lab, chunk=args.logit_chunk,
                                ignore_index=-100,
                            )
                        else:
                            ids_s, q_s, ok_s = soft
                            loss = chunked_mixed_ce(
                                flat, model.lm_head, lab, ids_s, q_s, ok_s,
                                alpha=args.soft_alpha, chunk=args.logit_chunk,
                                ignore_index=-100,
                            )
                    _backward(0, 1.0, loss, {})
                    loss_value = loss.item()
                    term_losses = [loss_value]
                    term_rows = [int(batch["valid"].sum())]
                accum += 1
                tokens_seen += int(batch["valid"].sum())
                if accum < args.grad_accum:
                    continue
                accum = 0

                lr = lr_at(step, args.lr, args.warmup, args.steps, args.min_lr_ratio)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                gnorm = torch.nn.utils.clip_grad_norm_(
                    model.trainable_parameters(), args.grad_clip
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % args.log_every == 0:
                    rate = tokens_seen / max(1e-6, time.time() - t0)
                    rec = {
                        "step": step, "epoch": epoch, "loss": round(loss_value, 5),
                        "lr": lr, "gnorm": round(float(gnorm), 4),
                        "tok_s": round(rate), "tokens": tokens_seen,
                        "term_losses": [None if x is None else round(x, 4)
                                        for x in term_losses],
                        "term_rows": term_rows,
                    }
                    roll = ""
                    if args.rollout_k > 0:
                        roll = "  roll " + "/".join(
                            "-" if x is None else f"{x:.3f}" for x in term_losses[1:]
                        ) + " n" + "/".join(str(n) for n in term_rows[1:])
                    print(
                        f"step {step:6d}/{args.steps} ep{epoch} loss {rec['loss']:.4f} "
                        f"lr {lr:.3e} gnorm {rec['gnorm']:.3f} tok/s {rec['tok_s']:.0f}{roll}",
                        flush=True,
                    )
                    history.append(rec)

                if args.eval_every and step % args.eval_every == 0:
                    summary = run_eval(model, args, eval_samples)
                    summary.update(run_renewal(model, args, renewal_set, allowed))
                    print(f"EVAL step {step} {json.dumps(summary)}", flush=True)
                    entry = {"step": step, "eval": summary}
                    history.append(entry)
                    # Select on accept@k, not agreement@1: the two can diverge once
                    # the head's distribution shifts, and acceptance is the goal.
                    # agreement@1 breaks ties (accept@k has far fewer windows).
                    primary, score = selection_score(summary, args.select_metric)
                    if score is None:
                        # Unmeasured is not zero: this eval neither becomes the
                        # best nor is compared with it.
                        missing = [m for m in selection_metrics(primary)
                                   if summary.get(m) is None]
                        entry["unscored"] = missing
                        print(f"[eval] WARNING step {step}: {', '.join(missing)} not "
                              f"measured (no scorable window/verify); --select-metric "
                              f"{primary} skips this eval", flush=True)
                    elif args.out and score > best["score"]:
                        best.update(score=score, step=step, summary=summary,
                                    criterion=primary)
                        # The weights carry their own best entry, so a resume
                        # knows about this one even if it postdates latest.pt.
                        save_weights_bf16(
                            os.path.join(args.out, "best-mtp.safetensors"), model,
                            metadata={"mtp_best": json.dumps(best), "mtp_run_id": run_id,
                                      "mtp_step": step},
                        )
                        write_json_atomic(os.path.join(args.out, "best.json"), best)
                        shown = " ".join(
                            f"{m} {_fmt_metric(summary.get(m))}"
                            for m in ("accept_len@3", "accept_len@15", "accept@3",
                                      "accept@15", "agreement@1") if m in summary)
                        print(f"BEST step {step} {primary}={score:.5f} ({shown})",
                              flush=True)
                    if args.out and args.snapshot_every_eval:
                        # Keep every eval's weights so an A/B needs no retraining.
                        save_weights_bf16(
                            os.path.join(args.out, f"mtp-step{step}.safetensors"), model,
                            metadata={"mtp_run_id": run_id, "mtp_step": step},
                        )

                if args.ckpt_every and step % args.ckpt_every == 0 and args.out:
                    checkpoint(epoch, epoch_batches)
                if _crash_after_step is not None and step >= _crash_after_step:
                    raise SimulatedCrash(f"simulated crash after step {step}")
        finally:
            loader.close()
        if not stop and epoch_batches == 0:
            raise SystemExit(f"epoch {epoch} produced no batch: every training "
                             "document is shorter than the minimum length")
        if not stop:
            # A full pass finished: the next epoch starts a fresh stream.
            epochs_this_run += 1
            done_epochs = epoch + 1
            position = (done_epochs, 0)
            if args.out:
                checkpoint(done_epochs, 0)
            print(f"EPOCH {done_epochs} done at step {step} "
                  f"({time.time() - t0:.0f}s in this run)", flush=True)
            if (args.stop_after_epochs
                    and epochs_this_run >= args.stop_after_epochs):
                print(f"STRETCH stop after {epochs_this_run} epoch(s), "
                      f"step {step}/{args.steps}", flush=True)
                stop = True

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        checkpoint(*position)
        save_weights_bf16(os.path.join(args.out, "final-mtp.safetensors"), model,
                          metadata={"mtp_run_id": run_id, "mtp_step": step})
    return {"steps": step, "history": history, "best": best, "run_id": run_id,
            "epoch": position[0], "epoch_batches": position[1]}


def build_parser() -> argparse.ArgumentParser:
    from .cli import StrictParser

    p = StrictParser(description=__doc__)
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                   help="serving checkpoint (env MTP_MODEL_DIR)")
    p.add_argument("--dump-dir", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--resume", default=None,
                   help="latest.pt to continue from (must exist).  Restores the "
                        "model, optimizer, step, epoch and position inside it, "
                        "partial gradient accumulation, RNG states, best and "
                        "history; must lie inside --out (copy a run directory "
                        "to branch it); the skipped part of the epoch is re-read "
                        "(not trained) from the dump.  Exact only with the same "
                        "dump and stream flags (a difference is reported); GPU "
                        "kernels are not bitwise deterministic.  Changed eval or "
                        "selection settings are refused (see --reset-best)")
    p.add_argument("--overwrite", action="store_true",
                   help="start afresh in an --out that holds an earlier run: its "
                        "files are moved into --out/.previous-<time>/ first (never "
                        "deleted).  Not combinable with --resume")
    p.add_argument("--reset-best", action="store_true",
                   help="with --resume: accept changed eval/selection settings "
                        "(which are otherwise refused) and restart the best "
                        "selection; the old best files move to --out/.reset-best-<time>/")
    p.add_argument("--init", default=None,
                   help="BF16 mtp.* snapshot to warm-start from (fresh optimizer "
                        "and LR schedule, unlike --resume)")
    p.add_argument("--tiny", action="store_true", help="use the reduced test config")
    p.add_argument("--synthetic", type=int, default=0,
                   help="N random samples instead of real dumps (CPU smoke test)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--steps", type=int, default=3200)
    p.add_argument("--batch-size", type=int, default=1,
                   help="only used when --tokens-per-step 0 (tiny/synthetic path)")
    p.add_argument("--tokens-per-step", type=int, default=8192,
                   help="padded-token budget per micro-batch (0 = fixed --batch-size)")
    p.add_argument("--bucket-window", type=int, default=512,
                   help="samples sorted by length before grouping, to cut padding")
    p.add_argument("--prefetch", type=int, default=4, help="collated batches queued ahead")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--stop-after-epochs", type=int, default=0,
                   help="stop this process after N complete passes and write "
                        "latest.pt at the epoch boundary, so --resume picks up "
                        "with the exact data order an uninterrupted run would "
                        "have had (0 = run to --steps)")
    p.add_argument("--eval-set", default=None,
                   help="fixed eval index from build_eval_set.py (must exist; "
                        "without it the dump's held-out split is scored)")
    p.add_argument("--eval-rows", type=int, default=0,
                   help="0 = the whole frozen eval set, matching the baseline")
    p.add_argument("--eval-batch", type=int, default=8)
    p.add_argument("--select-metric", default=None,
                   help="best-checkpoint criterion: renewal | combined | agreement "
                        "| 3 | 15 (default: renewal with --eval-renewal, else "
                        "combined)")
    # ------------------------------------------------------ soft targets
    p.add_argument("--soft-alpha", type=float, default=0.0,
                   help="weight of the distribution-distillation term: the loss "
                        "is (1-alpha)*hard CE + alpha*soft CE against the dumped "
                        "top-K (0 = the hard-label objective)")
    p.add_argument("--soft-temp", type=float, default=1.0,
                   help="temperature applied inside the top-K only; the stored "
                        "top-K mass is always measured at tau=1")
    p.add_argument("--soft-steps", default="first", choices=["first", "all"],
                   help="which chain steps use soft targets (see rollout.py)")
    p.add_argument("--synthetic-repeat", type=int, default=1000,
                   help="passes over the --synthetic pool per epoch (tests)")
    p.add_argument("--synthetic-topk", type=int, default=8,
                   help="K of the fabricated soft targets on the --synthetic path")
    # --------------------------------------------------- renewal eval
    p.add_argument("--eval-renewal", action="store_true",
                   help="also run the serving-process acceptance "
                        "evaluator (renewal.py) at every eval, on a fixed "
                        "held-out self-generated subset, and select on it")
    p.add_argument("--renewal-dump-dir", default=None,
                   help="dump dir holding the self-generated docs (default: --dump-dir)")
    p.add_argument("--renewal-manifest", default=None)
    p.add_argument("--renewal-docs", type=int, default=64,
                   help="per-bucket cap on scored documents (0 = all); renewal "
                        "eval is expensive, so keep this small")
    p.add_argument("--renewal-buckets", default="",
                   help="comma-separated manifest buckets to score (empty = all)")
    p.add_argument("--renewal-ks", type=int, nargs="+", default=[3, 15],
                   help="draft chain lengths; the selection score is their mean "
                        "accept_len")
    p.add_argument("--renewal-batch", type=int, default=32)
    p.add_argument("--renewal-max-len", type=int, default=0, help="0 = no truncation")
    p.add_argument("--renewal-min-gen", type=int, default=32)
    p.add_argument("--renewal-greedy-only", action="store_true",
                   help="score only self-generated documents generated greedily")
    p.add_argument("--renewal-seed", type=int, default=0,
                   help="fixes which documents the per-bucket cap keeps")
    p.add_argument("--token-map", default=None,
                   help="serving hot-vocabulary .pt; restricts the draft argmax "
                        "in the renewal eval exactly as --speculative-token-map "
                        "does at serving time")
    p.add_argument("--tap-layers", type=int, nargs="+", default=None,
                   help="EAGLE-3 fusion entry tap ids (default: read from the "
                        "dump's metadata; given, they must match it)")
    p.add_argument("--train-taps-only", action="store_true",
                   help="freeze the whole shipped head and train only the "
                        "zero-init fusion entry (phase 1 of "
                        "docs/lab-notes/DRAFT_V2_SPEC.md)")
    p.add_argument("--rollout-k", type=int, default=0,
                   help="extra self-fed chain steps trained per batch (0 = off)")
    p.add_argument("--rollout-weight", type=float, default=0.5,
                   help="loss weight of each rolled step (teacher-forced step is 1.0)")
    p.add_argument("--rollout-mask", default="accepted",
                   choices=["accepted", "consistent"],
                   help="accepted = only score a rolled step while the chain is "
                        "still on an accepted prefix, as serving would be")
    p.add_argument("--exclude-buckets", default="",
                   help="comma-separated manifest buckets to drop from TRAINING "
                        "(the frozen eval set is never filtered)")
    p.add_argument("--snapshot-every-eval", action="store_true",
                   help="keep a 5.2 GB BF16 snapshot at every eval, not just the best")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "adamw8bit"])
    p.add_argument("--bf16", action="store_true", default=False)
    p.add_argument("--holdout", type=float, default=0.1)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--eval-batches", type=int, default=0,
                   help="0 = score the whole fixed eval set (comparable to the baseline)")
    p.add_argument("--eval-starts", type=int, default=8)
    p.add_argument("--chain-ks", type=int, nargs="+", default=[3, 15])
    p.add_argument("--ckpt-every", type=int, default=0)
    p.add_argument("--logit-chunk", type=int, default=512,
                   help="rows per lm_head projection chunk (memory knob)")
    p.add_argument("--seed", type=int, default=20260903)
    return p


def main() -> None:
    train(build_parser().parse_args())  # train() validates the arguments


if __name__ == "__main__":
    main()
