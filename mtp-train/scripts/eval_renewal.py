#!/usr/bin/env python3
"""Serving-process acceptance length on self-generated dumps (see mtptrain/renewal.py).

    python scripts/eval_renewal.py --dump-dir DUMP --split eval --ks 3 15
    python scripts/eval_renewal.py --dump-dir DUMP --split eval --ckpt out/best-mtp.safetensors

Reports, per bucket and overall, ``accept_len`` = tokens / verifies, the same
number the server's ``completion_tokens / spec_verify_calls`` gives.
"""
from __future__ import annotations

import argparse, collections, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from mtptrain.cli import StrictParser  # noqa: E402
from mtptrain.config import DEFAULT_MODEL_DIR, MTPConfig  # noqa: E402
from mtptrain.model import MTPHead  # noqa: E402
from mtptrain.renewal import RenewalResult, renewal_evaluate  # noqa: E402
from mtptrain.select import (  # noqa: E402
    load_selfgen_samples,
    load_token_map,
    select_selfgen_docs,
)
from mtptrain.fileio import (  # noqa: E402
    check_run_settings,
    file_digest,
    read_jsonl,
    require_dir,
    require_file,
)
from mtptrain.data import samples_digest  # noqa: E402
from mtptrain.weights import load_into, model_identity, unloaded_names  # noqa: E402


def resolve_tap_layers(args) -> tuple:
    """--tap-layers, else whatever the dump directory recorded (possibly none)."""
    from mtptrain.data import read_tap_layers

    recorded = read_tap_layers(os.path.expanduser(args.dump_dir))
    if args.tap_layers is not None:
        if tuple(args.tap_layers) != tuple(recorded):
            raise SystemExit(f"--tap-layers {args.tap_layers} but the dump records "
                             f"{list(recorded) or 'no taps'}")
        return tuple(args.tap_layers)
    return recorded


DEFAULT_SPLIT_SEED = 20260903


def check_args(args) -> None:
    """Refuse missing inputs, impossible values and an output that exists."""
    if not args.dump_dir:
        raise SystemExit("--dump-dir (or env MTP_DUMP_DIR) is required")
    require_dir(args.dump_dir, "--dump-dir")
    require_dir(args.model_dir, "--model-dir")
    for opt, path in (("--manifest", args.manifest), ("--ckpt", args.ckpt),
                      ("--token-map", args.token_map)):
        if path:
            require_file(path, opt)
    if min(args.ks) < 1 or args.batch_size < 1:
        raise SystemExit("--ks values and --batch-size must be >= 1")
    if args.per_bucket < 0 or args.min_gen < 0 or args.max_len < 0:
        raise SystemExit("--per-bucket, --min-gen and --max-len must be >= 0")
    if not 0.0 <= args.holdout_frac <= 1.0:
        raise SystemExit("--holdout-frac must be in [0, 1]")
    # --seed only sets document sampling; the held-out split uses --split-seed. A non-default --seed with
    # an implicit split seed is most likely meant as the training split seed, so make the user say which.
    if getattr(args, "split_seed", DEFAULT_SPLIT_SEED) is None:
        if getattr(args, "seed", 0) != 0:
            raise SystemExit("--seed only sets document sampling, not the held-out split; pass --split-seed "
                             f"explicitly (the training/eval-set split seed, default {DEFAULT_SPLIT_SEED})")
        args.split_seed = DEFAULT_SPLIT_SEED
    if args.per_seq and os.path.lexists(os.path.expanduser(args.per_seq)):
        raise SystemExit(f"--per-seq {args.per_seq} exists; it is written whole per "
                         "run, so pick a new name (or remove the old file)")
    if args.append and os.path.isdir(args.append):
        raise SystemExit(f"--append {args.append} is a directory")


def run_settings(args, sel) -> dict:
    """Which documents are scored and how; every --append line carries it and a
    line with other settings is refused.  --ckpt, --tag, --dtype and
    --token-map are what a comparison varies, so they are only recorded.

    Identities are content, not paths: the scored data by a digest of every
    tensor the evaluator reads (``samples_digest``), the model directory by
    the content of its files (weights, config, tokenizer)."""
    return {
        "split": args.split, "split_seed": args.split_seed,
        "holdout_frac": args.holdout_frac, "buckets": args.buckets,
        "per_bucket": args.per_bucket, "min_gen": args.min_gen,
        "greedy_only": bool(args.greedy_only), "max_len": args.max_len,
        "seed": args.seed, "ks": list(args.ks),
        "data": samples_digest(sel.samples, list(sel.gen_start_rows) + list(sel.buckets)),
        "model": model_identity(args.model_dir),
    }


def load_model(args, tap_layers=()):
    dt = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    cfg = MTPConfig.from_pretrained(args.model_dir).with_(tap_layers=tuple(tap_layers))
    model = MTPHead(cfg).to(dt)
    load_into(model, args.model_dir, device="cpu", dtype=dt)
    step = 0
    if args.ckpt:
        if args.ckpt.endswith(".safetensors"):
            from safetensors.torch import load_file

            state = load_file(args.ckpt)
            step = None
        else:
            payload = torch.load(args.ckpt, map_location="cpu", weights_only=True)
            state = payload["model"] if "model" in payload else payload
            step = payload.get("step")
        state = {k[len("mtp."):] if k.startswith("mtp.") else k: v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(
            {k: v.to(dt) for k, v in state.items()}, strict=False
        )
        missing = unloaded_names(missing)
        if missing or unexpected:
            raise SystemExit(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
        print(f"[ckpt] loaded {len(state)} tensors from {args.ckpt}")
    model.freeze_non_trainable()
    model = model.to(args.device).eval()
    return model, step


def select_docs(args):
    """-> (records, index); the argparse face of mtptrain.select."""
    return select_selfgen_docs(
        args.dump_dir,
        manifest=args.manifest,
        split=args.split,
        split_seed=args.split_seed,
        holdout_frac=args.holdout_frac,
        buckets=args.buckets,
        per_bucket=args.per_bucket,
        min_gen=args.min_gen,
        seed=args.seed,
        reindex=args.reindex,
        greedy_only=args.greedy_only,
    )


def build_parser() -> argparse.ArgumentParser:
    p = StrictParser(description=__doc__)
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                   help="serving checkpoint (env MTP_MODEL_DIR)")
    p.add_argument("--dump-dir", default=os.environ.get("MTP_DUMP_DIR"),
                   required="MTP_DUMP_DIR" not in os.environ,
                   help="dump directory with manifest.jsonl (env MTP_DUMP_DIR)")
    p.add_argument("--manifest", default=None)
    p.add_argument("--reindex", action="store_true")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--token-map", default=None,
                   help="serving hot-vocabulary .pt (list of ids); restrict draft argmax to it")
    p.add_argument("--tap-layers", type=int, nargs="+", default=None,
                   help="EAGLE-3 fusion entry tap ids (default: read from the "
                        "dump's metadata; given, they must match it)")
    p.add_argument("--ks", type=int, nargs="+", default=[3, 15])
    p.add_argument("--buckets", nargs="+", default=None,
                   help="score only these manifest buckets (default: all)")
    p.add_argument("--per-bucket", type=int, default=0, help="0 = all")
    p.add_argument("--min-gen", type=int, default=32)
    p.add_argument("--greedy-only", action="store_true",
                   help="score only documents generated at temperature 0")
    p.add_argument("--max-len", type=int, default=0, help="0 = no truncation")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0,
                   help="document sampling seed only (not the split; see --split-seed)")
    p.add_argument("--split", default="eval", choices=["eval", "all", "train"],
                   help="the trainer's held-out documents (default; same rule as "
                        "DumpDataset), or all / the training ones, which reward "
                        "memorisation")
    p.add_argument("--split-seed", type=int, default=None,
                   help=f"held-out split seed (default {DEFAULT_SPLIT_SEED}); must match the eval set and "
                        "training (mtptrain.train --seed)")
    p.add_argument("--holdout-frac", type=float, default=0.1)
    p.add_argument("--tag", default=None)
    p.add_argument("--append", default=None)
    p.add_argument("--per-seq", default=None, help="write per-sequence stats (jsonl)")
    return p


def renewal_output(results, hash_to_bucket, head: dict) -> dict:
    """``head`` plus per-k summaries and per-bucket accept_len.

    A k or bucket with no verify is unmeasured: its accept_len is None (JSON
    null) next to ``verifies`` = 0, never 0."""
    out = dict(head)
    for k, res in results.items():
        out[f"k{k}"] = res.summary()
        buckets = collections.defaultdict(list)
        for s in res.seqs:
            buckets[hash_to_bucket[s.doc_hash]].append(s)
        out[f"k{k}"]["by_bucket"] = {}
        for b in sorted(buckets):
            bs = RenewalResult(k=k, seqs=list(buckets[b])).summary()
            out[f"k{k}"]["by_bucket"][b] = {
                "sequences": len(buckets[b]), "verifies": bs["verifies"],
                "accept_len": None if bs["accept_len"] is None else round(bs["accept_len"], 4),  # None: unmeasured
            }
    return out


def main() -> None:
    args = build_parser().parse_args()
    check_args(args)

    chosen, index = select_docs(args)
    if not chosen:
        raise SystemExit(f"no self-generated document in {args.dump_dir} matches "
                         f"(split={args.split}, buckets={args.buckets or 'all'}, "
                         f"min_gen={args.min_gen}, greedy_only={args.greedy_only})")
    print(f"[select] {len(chosen)} self-generated docs: " + json.dumps(
        collections.Counter(r["bucket"] for r in chosen)))
    taps = resolve_tap_layers(args)
    print(f"[taps] {list(taps) or 'none'}")
    model, step = load_model(args, taps)

    # Load the sequences (one file per doc mostly).
    t0 = time.time()
    sel = load_selfgen_samples(args.dump_dir, chosen, index, max_len=args.max_len)
    samples, starts = sel.samples, sel.gen_start_rows
    if not samples:
        raise SystemExit(f"none of the {len(chosen)} selected documents could be loaded "
                         "(changed on disk, or no generated row left after --max-len)")
    settings = run_settings(args, sel)
    if args.append:
        check_run_settings(read_jsonl(args.append, missing_ok=True), "settings",
                           settings, args.append)
    hash_to_bucket = sel.bucket_of()
    print(f"[load] {len(samples)} sequences, {sel.rows} rows, "
          f"{sel.generated_rows} generated rows ({time.time()-t0:.0f}s)")

    allowed = None
    if args.token_map:
        allowed = load_token_map(args.token_map, int(model.lm_head.shape[0])).to(args.device)
        print(f"[token-map] {int(allowed.sum())} allowed ids")
    t0 = time.time()
    results = renewal_evaluate(model, samples, starts, ks=args.ks,
                               batch_size=args.batch_size, progress=True, allowed=allowed)
    out = renewal_output(results, hash_to_bucket, {
        "tag": args.tag or ("baseline" if not args.ckpt else os.path.basename(args.ckpt)),
        "ckpt": args.ckpt, "step": step, "dtype": args.dtype,
        # content, not only the path: a file replaced under the same name
        # would otherwise look like the same candidate in --append logs
        "ckpt_digest": file_digest(args.ckpt) if args.ckpt else None,
        "token_map": args.token_map,
        "tap_layers": list(taps),
        "dump_dir": args.dump_dir, "sequences": len(samples),
        "seconds": round(time.time() - t0, 1),
        "settings": settings,
    })
    print(json.dumps(out, indent=1))
    from mtptrain.fileio import open_jsonl_append, write_jsonl_atomic

    if args.append:
        with open_jsonl_append(args.append) as fh:  # one flushed line per run
            fh.write(out)
    if args.per_seq:
        write_jsonl_atomic(args.per_seq, [
            {"k": k, "doc_hash": s.doc_hash, "bucket": hash_to_bucket[s.doc_hash],
             "gen_rows": s.gen_rows, "verifies": s.verifies, "accepted": s.accepted,
             "accept_len": None if s.accept_len is None else round(s.accept_len, 4),
             "hist": s.hist}
            for k, res in results.items() for s in res.seqs
        ])


if __name__ == "__main__":
    main()
