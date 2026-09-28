"""Round-trip the exact on-disk format the SGLang hook writes (CPU only).

The hook lives in another repo, so this test writes a file byte-compatible with
``MTPDumper.flush`` and asserts the loader reconstructs the training rows with
the documented alignment.  A drift in either direction fails here.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safetensors.torch import save_file  # noqa: E402

from mtptrain.data import (  # noqa: E402
    DumpDataset,
    collate,
    load_concat,
    load_dump_file,
    truncate,
)

HC = 32


def _write_dump(path: str, seqs, hc_dim: int = HC, topk: int = 0,
                with_lse: bool = True) -> None:
    """Byte-compatible with ``MTPDumper.flush``; ``topk`` adds the soft target.

    The stored logits are made strictly descending with the argmax first, which
    is what a real top-K dump always looks like.
    """
    ids, pos, tgt, offsets = [], [], [], [0]
    for seq_ids, seq_tgt, start in seqs:
        ids += list(seq_ids)
        tgt += list(seq_tgt)
        pos += list(range(start, start + len(seq_ids)))
        offsets.append(len(ids))
    n = len(ids)
    g = torch.Generator().manual_seed(4)
    tensors = {
        "input_ids": torch.tensor(ids, dtype=torch.int32),
        "positions": torch.tensor(pos, dtype=torch.int32),
        "hc_hidden": torch.randn(n, hc_dim, generator=g).to(torch.bfloat16),
        "target_argmax": torch.tensor(tgt, dtype=torch.int32),
        "seq_offsets": torch.tensor(offsets, dtype=torch.int32),
        "prefix_lens": torch.tensor([0] * len(seqs), dtype=torch.int32),
    }
    if topk:
        tk = torch.arange(n, dtype=torch.int32).view(n, 1) * 10 + torch.arange(
            topk, dtype=torch.int32
        ).view(1, topk)
        tk[:, 0] = tensors["target_argmax"]
        logits = (
            torch.linspace(4.0, -2.0, topk).view(1, topk).repeat(n, 1)
            + torch.arange(n).view(n, 1) * 0.01
        )
        tensors["topk_ids"] = tk
        tensors["topk_logits"] = logits.to(torch.float16)
        if with_lse:
            tensors["lse"] = (
                torch.logsumexp(logits, dim=-1) + 0.3
            ).to(torch.float16)
    save_file(
        tensors,
        path,
        metadata={"forward_mode": "ForwardMode.EXTEND", "tokens": str(n)},
    )


def test_row_alignment():
    tmp = tempfile.mkdtemp(prefix="mtpdump-")
    try:
        path = os.path.join(tmp, "dump-1-00000001.safetensors")
        ids = list(range(100, 120))            # 20 tokens
        tgt = [i + 1000 for i in range(20)]
        _write_dump(path, [(ids, tgt, 0)])
        samples = load_dump_file(path)
        assert len(samples) == 1
        s = samples[0]
        # T = 20 -> 19 usable rows
        assert len(s) == 19
        # row t consumes input_ids[t+1] and predicts target_argmax[t+1]
        assert s.next_token_ids.tolist() == ids[1:]
        assert s.labels.tolist() == tgt[1:]
        assert s.positions.tolist() == list(range(19))
        assert s.hc_hidden.shape == (19, HC)
    finally:
        shutil.rmtree(tmp)


def test_greedy_consistency_flag():
    tmp = tempfile.mkdtemp(prefix="mtpdump-")
    try:
        path = os.path.join(tmp, "dump-1-00000001.safetensors")
        # Build a sequence where the corpus IS the greedy continuation
        # everywhere except one position.
        ids = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        tgt = [i + 1 for i in ids]  # target_argmax[u] == input_ids[u+1]
        tgt[4] = 999                # a divergence at u = 4
        _write_dump(path, [(ids, tgt, 0)])
        s = load_dump_file(path)[0]
        cons = s.greedy_consistent.tolist()
        # greedy_consistent[t] compares input_ids[t+2] with target_argmax[t+1]
        assert cons[3] is False, cons  # t=3 -> u=4 diverges
        assert cons[0] and cons[1] and cons[2]
        assert cons[-1] is False  # last row has no t+2 token, kept False
    finally:
        shutil.rmtree(tmp)


def test_multiple_sequences_and_positions():
    tmp = tempfile.mkdtemp(prefix="mtpdump-")
    try:
        path = os.path.join(tmp, "dump-1-00000002.safetensors")
        a = (list(range(50, 70)), list(range(500, 520)), 0)
        b = (list(range(70, 85)), list(range(700, 715)), 1024)  # cached prefix
        _write_dump(path, [a, b])
        samples = load_dump_file(path)
        assert len(samples) == 2
        assert samples[0].positions[0].item() == 0
        assert samples[1].positions[0].item() == 1024
        assert samples[1].next_token_ids.tolist() == b[0][1:]
        assert samples[0].doc_hash != samples[1].doc_hash
    finally:
        shutil.rmtree(tmp)


def test_short_sequences_are_dropped():
    tmp = tempfile.mkdtemp(prefix="mtpdump-")
    try:
        path = os.path.join(tmp, "dump-1-00000003.safetensors")
        _write_dump(path, [(list(range(4)), list(range(4)), 0)])
        assert load_dump_file(path, min_len=8) == []
        assert len(load_dump_file(path, min_len=2)) == 1
    finally:
        shutil.rmtree(tmp)


def test_split_is_deterministic_and_disjoint():
    tmp = tempfile.mkdtemp(prefix="mtpdump-")
    try:
        for i in range(12):
            _write_dump(
                os.path.join(tmp, f"dump-1-{i:08d}.safetensors"),
                [(list(range(i * 100, i * 100 + 40)), list(range(40)), 0)],
            )
        train = {s.doc_hash for s in DumpDataset(tmp, "train", holdout_frac=0.25)}
        evals = {s.doc_hash for s in DumpDataset(tmp, "eval", holdout_frac=0.25)}
        assert train and evals
        assert not (train & evals)
        assert len(train) + len(evals) == 12
        again = {s.doc_hash for s in DumpDataset(tmp, "eval", holdout_frac=0.25)}
        assert again == evals
    finally:
        shutil.rmtree(tmp)


def test_collate_of_real_dump_samples():
    tmp = tempfile.mkdtemp(prefix="mtpdump-")
    try:
        path = os.path.join(tmp, "dump-1-00000004.safetensors")
        _write_dump(
            path,
            [
                (list(range(30)), list(range(30)), 0),
                (list(range(18)), list(range(18)), 0),
            ],
        )
        samples = load_dump_file(path)
        batch = collate(samples)
        assert batch["hc_hidden"].shape == (2, 29, HC)
        assert batch["valid"][1].sum().item() == 17
        assert batch["labels"][1, 17:].eq(-100).all()
    finally:
        shutil.rmtree(tmp)


def test_manifest_only_excludes_unlisted_sequences():
    """Sequences with no manifest entry (self-gen prompt prefills, the server
    warmup batch) must not become training data."""
    tmp = tempfile.mkdtemp(prefix="mtpdump-")
    try:
        for i in range(6):
            _write_dump(
                os.path.join(tmp, f"dump-1-{i:08d}.safetensors"),
                [(list(range(i * 100, i * 100 + 40)), list(range(40)), 0)],
            )
        listed = [s.doc_hash for s in DumpDataset(tmp, "train", holdout_frac=0.0,
                                                  manifest_only=False)]
        assert len(listed) == 6
        with open(os.path.join(tmp, "manifest.jsonl"), "w") as fh:
            for h in listed[:4]:
                fh.write(json.dumps({"doc_hash": h, "split_key": h,
                                     "tokens": 40}) + "\n")
        kept = list(DumpDataset(tmp, "train", holdout_frac=0.0))
        assert len(kept) == 4, len(kept)
        assert {s.doc_hash for s in kept} == set(listed[:4])
        allofthem = list(DumpDataset(tmp, "train", holdout_frac=0.0,
                                     manifest_only=False))
        assert len(allofthem) == 6
    finally:
        shutil.rmtree(tmp)


def test_chain_validity_window_is_cons_t_to_t_plus_k_minus_2():
    """Pin the chain-validity rule the accept@k scorer must use.

    Step j of a chain from t feeds the draft's own predictions at positions
    t+2..t+j+1, so the dumped target_argmax is the right label only while
    next_token_ids[t+1+i] == labels[t+i] for i < j -- i.e. greedy_consistent
    [t .. t+j-1].  A k-step chain needs cons[t .. t+k-2], not cons[t+1 .. t+k-1].
    """
    tmp = tempfile.mkdtemp(prefix="mtpdump-")
    try:
        path = os.path.join(tmp, "dump-1-00000009.safetensors")
        ids = list(range(100, 112))
        tgt = [i + 1 for i in ids]
        tgt[5] = 999                      # target diverges from the corpus at u=5
        _write_dump(path, [(ids, tgt, 0)])
        s = load_dump_file(path, min_len=2)[0]
        nxt, lab = s.next_token_ids.tolist(), s.labels.tolist()
        cons = s.greedy_consistent.tolist()
        # the identity the rule rests on
        assert all(cons[u] == (nxt[u + 1] == lab[u]) for u in range(len(lab) - 1))
        k = 3
        for t, expected in ((2, True), (3, False), (4, False), (5, True)):
            assert all(cons[t + i] for i in range(k - 1)) == expected, t
        # and the shifted window really would disagree, so this test has teeth
        assert all(cons[3 + i] for i in range(k - 1)) != all(
            cons[2 + i] for i in range(k - 1)
        )
    finally:
        shutil.rmtree(tmp)


def test_bucket_filtering():
    """exclude_buckets / include_buckets select on the manifest's bucket."""
    tmp = tempfile.mkdtemp(prefix="mtpdump-")
    try:
        hashes = []
        for i in range(8):
            path = os.path.join(tmp, f"dump-1-{i:08d}.safetensors")
            _write_dump(path, [(list(range(i * 100, i * 100 + 40)), list(range(40)), 0)])
            hashes += [s.doc_hash for s in load_dump_file(path)]
        with open(os.path.join(tmp, "manifest.jsonl"), "w") as fh:
            for i, h in enumerate(hashes):
                fh.write(json.dumps({
                    "doc_hash": h, "split_key": h, "tokens": 40,
                    "bucket": "prose-en" if i < 3 else "agent-code"}) + "\n")
        allb = list(DumpDataset(tmp, "train", holdout_frac=0.0))
        assert len(allb) == 8
        kept = list(DumpDataset(tmp, "train", holdout_frac=0.0,
                                exclude_buckets=["prose-en"]))
        assert len(kept) == 5, len(kept)
        assert all(DumpDataset(tmp, "train", holdout_frac=0.0).buckets[s.doc_hash]
                   == "agent-code" for s in kept)
        only = list(DumpDataset(tmp, "train", holdout_frac=0.0,
                                include_buckets=["prose-en"]))
        assert len(only) == 3, len(only)
    finally:
        shutil.rmtree(tmp)


def test_soft_targets_are_absent_from_old_dumps():
    """A dump written without SGLANG_MTP_DUMP_TOPK must load exactly as before."""
    tmp = tempfile.mkdtemp(prefix="mtpdump-")
    try:
        path = os.path.join(tmp, "dump-1-00000010.safetensors")
        _write_dump(path, [(list(range(100, 120)), list(range(20)), 0)])
        s = load_dump_file(path)[0]
        assert s.topk_ids is None and s.topk_logits is None and s.lse is None
        assert not s.has_soft
        batch = collate([s])
        assert "topk_ids" not in batch and "soft_valid" not in batch
    finally:
        shutil.rmtree(tmp)


def test_soft_targets_round_trip_and_align_with_labels():
    """topk row t must be dump row t+1, i.e. the same shift as ``labels``, and
    its top-1 id must be the label."""
    tmp = tempfile.mkdtemp(prefix="mtpdump-")
    try:
        path = os.path.join(tmp, "dump-1-00000011.safetensors")
        ids = list(range(100, 120))
        tgt = [i + 1000 for i in range(20)]
        _write_dump(path, [(ids, tgt, 0)], topk=8)
        s = load_dump_file(path)[0]
        assert len(s) == 19
        assert s.has_soft
        assert s.topk_ids.shape == (19, 8) and s.topk_ids.dtype == torch.int64
        assert s.topk_logits.shape == (19, 8) and s.topk_logits.dtype == torch.float32
        assert s.lse.shape == (19,) and s.lse.dtype == torch.float32
        # row t carries dump row t+1: the top-1 id equals labels[t]
        assert s.topk_ids[:, 0].tolist() == s.labels.tolist()
        # ... and the mass it implies is < 1 (the dump's lse has a tail)
        mass = torch.exp(torch.logsumexp(s.topk_logits, dim=-1) - s.lse)
        assert (mass > 0).all() and (mass < 1.0).all()
    finally:
        shutil.rmtree(tmp)


def test_topk_dump_without_lse_still_loads():
    tmp = tempfile.mkdtemp(prefix="mtpdump-")
    try:
        path = os.path.join(tmp, "dump-1-00000012.safetensors")
        _write_dump(path, [(list(range(100, 116)), list(range(16)), 0)],
                    topk=4, with_lse=False)
        s = load_dump_file(path)[0]
        assert s.has_soft and s.lse is None
        batch = collate([s])
        assert batch["soft_valid"].all()
        assert not batch["lse_known"].any()
    finally:
        shutil.rmtree(tmp)


def test_truncate_carries_the_soft_targets():
    tmp = tempfile.mkdtemp(prefix="mtpdump-")
    try:
        path = os.path.join(tmp, "dump-1-00000013.safetensors")
        _write_dump(path, [(list(range(100, 130)), list(range(30)), 0)], topk=4)
        s = truncate(load_dump_file(path)[0], 12)
        assert len(s) == 12
        assert s.topk_ids.shape == (12, 4) and s.lse.shape == (12,)
        assert s.topk_ids[:, 0].tolist() == s.labels.tolist()
    finally:
        shutil.rmtree(tmp)


def test_collate_pads_soft_targets_and_masks_missing_rows():
    tmp = tempfile.mkdtemp(prefix="mtpdump-")
    try:
        path = os.path.join(tmp, "dump-1-00000014.safetensors")
        _write_dump(path, [(list(range(30)), list(range(30)), 0),
                           (list(range(18)), list(range(18)), 0)], topk=4)
        # a third sample with no top-K at all, to exercise the mixed case
        plain = os.path.join(tmp, "dump-1-00000015.safetensors")
        _write_dump(plain, [(list(range(200, 224)), list(range(24)), 0)])
        samples = load_dump_file(path) + load_dump_file(plain)
        batch = collate(samples)
        assert batch["topk_ids"].shape == (3, 29, 4)
        assert batch["topk_logits"].shape == (3, 29, 4)
        assert batch["lse"].shape == (3, 29)
        # row-wise: sample 0 has 29 soft rows, sample 1 has 17, sample 2 none
        assert batch["soft_valid"].sum(dim=1).tolist() == [29, 17, 0]
        assert batch["lse_known"].sum(dim=1).tolist() == [29, 17, 0]
        # padded rows carry -inf logits, so they hold zero probability
        assert torch.isinf(batch["topk_logits"][1, 17:]).all()
        assert torch.isinf(batch["topk_logits"][2]).all()
    finally:
        shutil.rmtree(tmp)


def test_load_concat_carries_soft_targets_across_chunks():
    """Chunked prefill: two files, one document.  The soft targets must be
    concatenated with the same alignment as a single-file load."""
    tmp = tempfile.mkdtemp(prefix="mtpdump-")
    try:
        whole = list(range(300, 340))
        tgt = [i + 7 for i in range(40)]
        a = os.path.join(tmp, "dump-1-00000016.safetensors")
        b = os.path.join(tmp, "dump-1-00000017.safetensors")
        _write_dump(a, [(whole, tgt, 0)], topk=4)          # reference: one file
        one = load_dump_file(a)[0]
        # now the same document split in two chunks
        c1 = os.path.join(tmp, "c1.safetensors")
        c2 = os.path.join(tmp, "c2.safetensors")
        _write_dump(c1, [(whole[:24], tgt[:24], 0)], topk=4)
        _write_dump(c2, [(whole[24:], tgt[24:], 24)], topk=4)
        joined = load_concat([c1, c2], [0, 0])
        assert len(joined) == len(one)
        assert joined.labels.tolist() == one.labels.tolist()
        assert joined.has_soft and joined.topk_ids.shape == one.topk_ids.shape
        assert joined.topk_ids[:, 0].tolist() == joined.labels.tolist()
        assert joined.lse.shape == (len(joined),)
        # a chunk without top-K makes the whole concatenation soft-less
        _write_dump(b, [(whole[24:], tgt[24:], 24)])
        mixed = load_concat([c1, b], [0, 0])
        assert not mixed.has_soft and mixed.lse is None
    finally:
        shutil.rmtree(tmp)


def main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            import traceback

            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{len(fns) - failures}/{len(fns)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
