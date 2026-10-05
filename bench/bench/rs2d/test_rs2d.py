"""CPU interpreter tests for RS2d sharpening, G1 greedy proposal, dump and offline acceptance."""

import argparse
from datetime import datetime, timezone
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import msgspec
import torch
from scipy.stats import combine_pvalues

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("WT", "/home/user/tools/sglang-rs2d")
assert os.environ.get("DEV") == "cpu"
assert os.environ.get("TRITON_INTERPRET") == "1"
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "bench/rs2"))
import test_sparse_rs as base
import accept_offline as offline


def load_path(*, name, path):
    spec = importlib.util.spec_from_file_location(name=name, location=path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


reference = load_path(name="rs2_readonly_reference", path=Path(
    "/home/user/tools/sglang-rs2/python/sglang/kernels/ops/speculative/sparse_rs.py"))
dump = load_path(name="rs2d_dump", path=base.WT / "python/sglang/srt/speculative/rs_dump.py")


def call(*, module, logits, sampling, uniforms, hot, **knobs):
    n = logits.shape[0]
    positions = torch.arange(end=n, dtype=torch.int64)
    chain = torch.full(size=(n, 3), fill_value=-1, dtype=torch.int64)
    output = module.rs_draft_proposal_sparse(
        next_token_logits=logits, uniforms=uniforms, hot_token_id=hot, k=64,
        positions=positions, draft_tokens=chain, draft_token_column=1,
        **sampling, **knobs,
    )
    return (*output, positions, chain)


def identical(*, left, right):
    assert len(left) == len(right)
    for index, (a, b) in enumerate(zip(left, right)):
        assert torch.equal(input=a, other=b), (index, (a != b).sum().item())


def proposal_cases(*, full_grid=True):
    grid = list(itertools.product((0.3, 0.8, 1.5), (1, 5, 40, 0), (1.0, 0.95, 0.5)))
    if not full_grid:
        grid = [(0.3, 5, 0.95), (0.8, 40, 0.5), (1.5, 0, 1.0)]
    for vocab, tied in ((117, False), (117, True), (49152, False), (49152, True)):
        for mapped, min_p in itertools.product((False, True), (None, 0.05, 0.2)):
            if vocab == 49152:
                if not mapped or min_p != 0.05:
                    continue
                batches = [(1, [(0.8, 40, 0.95)])]
            elif full_grid:
                batches, start, index = [], 0, 0
                while start < len(grid):
                    n = (1, 2, 4, 7)[index % 4]
                    batches.append((n, [grid[(start + i) % len(grid)] for i in range(n)]))
                    start += n
                    index += 1
                assert set(row for _, rows in batches for row in rows) == set(grid)
                assert {n for n, _ in batches} == {1, 2, 4, 7}
            else:
                batches = [(n, [grid[(start + i) % len(grid)] for i in range(n)])
                           for n in (1, 2, 4, 7) for start in range(0, len(grid), n)]
            for n, rows in batches:
                logits = torch.randn(size=(n, vocab))
                if tied:
                    logits = logits.bfloat16()
                    logits[:, :80] = 8.0
                storage = torch.empty(size=(n, vocab + 11), dtype=logits.dtype)
                storage[:, :vocab] = logits
                hot = torch.randperm(n=vocab + 37)[:vocab].contiguous() if mapped else None
                sampling = base.params(temperatures=[r[0] for r in rows], top_ks=[r[1] for r in rows],
                                       top_ps=[r[2] for r in rows],
                                       min_ps=None if min_p is None else [min_p] * n)
                yield storage[:, :vocab], sampling, torch.rand(size=(n,)), hot


def test_k1_k2():
    runs, rows = 0, 0
    for logits, sampling, uniforms, hot in proposal_cases():
        args = dict(logits=logits, sampling=sampling, uniforms=uniforms, hot=hot)
        ref = call(module=reference, **args)
        identical(left=call(module=base.rs2, **args), right=ref)
        identical(left=call(module=base.rs2, temp_scale=1.0, onehot_above=0.0, **args), right=ref)
        runs += 1
        rows += logits.shape[0]
        if runs % 40 == 0:
            print(f"  K1 checked {runs} cases/{rows} rows", flush=True)
    print(f"PASS K1 defaults explicit/omitted bit-identical to read-only RS2: {runs} cases/{rows} rows; "
          "A grid, fp32/bf16 ties, n=1/2/4/7, hot on/off, min_p=None/.05/.2, positions+chain", flush=True)
    comparisons = 0
    for logits, sampling, uniforms, hot in proposal_cases(full_grid=False):
        for scale in (0.5, 0.7):
            scaled = dict(sampling, temperatures=sampling["temperatures"] * scale)
            identical(left=call(module=base.rs2, logits=logits, sampling=sampling,
                                uniforms=uniforms, hot=hot, temp_scale=scale),
                      right=call(module=reference, logits=logits, sampling=scaled,
                                 uniforms=uniforms, hot=hot))
            comparisons += 1
    print(f"PASS K2 s=.5/.7 bit-identical to reference temperatures*s: {comparisons} comparisons; "
          "fp32/bf16 ties, n=1/2/4/7, hot/min-p/mixed filters", flush=True)
    sampling = base.params(temperatures=[0, -1], top_ks=[0, 0], top_ps=[1, 1])
    logits = torch.randn(size=(2, 83))
    args = dict(logits=logits, sampling=sampling, uniforms=torch.tensor(data=[0.2, 0.8]), hot=None)
    identical(left=call(module=base.rs2, temp_scale=0.5, **args), right=call(module=reference, **args))
    for knobs in ({"temp_scale": 0}, {"temp_scale": -1}, {"temp_scale": float("nan")},
                  {"onehot_above": -0.1}, {"onehot_above": 1.1}, {"onehot_above": float("nan")}):
        try:
            call(module=base.rs2, **knobs, **args)
        except AssertionError:
            pass
        else:
            raise AssertionError(knobs)
    print("  K2 nonpositive T retains 1.0 fallback; invalid knobs rejected", flush=True)


def test_k3():
    for theta in (0.5, 0.9):
        for mapped in (False, True):
            logits = torch.tensor(data=[[9, 0, -1, -2], [1, 0.9, 0.8, 0.7]], dtype=torch.float32)
            hot = torch.tensor(data=[7, 3, 9, 2], dtype=torch.int64) if mapped else None
            sampling = base.params(temperatures=[1, 1], top_ks=[0, 0], top_ps=[1, 1], min_ps=[0.05, 0.05])
            args = dict(logits=logits, sampling=sampling, uniforms=torch.tensor(data=[1.0, 0.8]), hot=hot)
            ref = call(module=reference, **args)
            actual = call(module=base.rs2, onehot_above=theta, **args)
            selected = ref[0][:, 0] >= theta
            assert selected.tolist() == [True, False]
            for row in range(2):
                if selected[row]:
                    assert actual[0][row].tolist() == [1, 0, 0, 0]
                    assert actual[2][row].item() == 1
                    assert actual[3][row].item() == ref[1][row, 0].item()
                    assert actual[5][row, 1] == ref[1][row, 0]
                    assert torch.equal(input=actual[1][row], other=ref[1][row])
                else:
                    identical(left=[t[row] for t in actual], right=[t[row] for t in ref])
            assert torch.equal(input=actual[4], other=ref[4])
    sampling = base.params(temperatures=[1], top_ks=[0], top_ps=[1])
    actual = call(module=base.rs2, logits=torch.zeros(size=(1, 2)), sampling=sampling,
                  uniforms=torch.tensor(data=[0.9]), hot=None, onehot_above=0.5)
    assert actual[0].tolist() == [[1, 0]] and actual[3].item() == 0
    print("PASS K3 theta=.5/.9: both sides, equality threshold, hot on/off, u=1, TopkP/tokens/chain/positions", flush=True)



def test_g1a():
    runs, rows = 0, 0
    for logits, sampling, uniforms, hot in proposal_cases():
        args = dict(logits=logits, sampling=sampling, uniforms=uniforms, hot=hot)
        identical(left=call(module=base.rs2, greedy_fast=False, **args),
                  right=call(module=reference, **args))
        runs += 1
        rows += logits.shape[0]
        if runs % 40 == 0:
            print(f"  G1a checked {runs} cases/{rows} rows", flush=True)
    print(f"PASS G1a greedy_fast=False bit-identical to read-only RS2: {runs} cases/{rows} rows; full A grid", flush=True)


def check_greedy_outputs(*, actual, expected, greedy):
    identical(left=[t[~greedy] for t in actual], right=[t[~greedy] for t in expected])
    identical(left=[actual[i][greedy] for i in (0, 2, 3, 4, 5)],
              right=[expected[i][greedy] for i in (0, 2, 3, 4, 5)])
    assert torch.equal(input=actual[1][greedy, 0], other=expected[1][greedy, 0])
    assert torch.all(actual[1][greedy] == actual[1][greedy, :1])
    assert torch.all(actual[0][greedy, 1:] == 0)


def test_g1b():
    runs, rows, greedy_rows = 0, 0, 0
    for logits, sampling, uniforms, hot in proposal_cases():
        args = dict(logits=logits, sampling=sampling, uniforms=uniforms, hot=hot)
        greedy = sampling["top_ks"] == 1
        actual = call(module=base.rs2, greedy_fast=True, **args)
        expected = call(module=reference, **args)
        check_greedy_outputs(actual=actual, expected=expected, greedy=greedy)
        if logits.dtype == torch.bfloat16:
            assert torch.all(actual[1][greedy, 0] == (0 if hot is None else hot[0]))
        runs += 1
        rows += logits.shape[0]
        greedy_rows += int(greedy.sum())
        if runs % 40 == 0:
            print(f"  G1b checked {runs} cases/{rows} rows", flush=True)
    print(f"  G1b A grid complete: {runs} cases/{rows} rows", flush=True)
    for vocab in (7, 65, 4097):
        logits = torch.full(size=(4, vocab), fill_value=-3.0, dtype=torch.bfloat16)
        logits[:, [2, vocab - 1]] = 8.0
        if vocab > 2048:
            logits[:, 2049] = 8.0
        sampling = base.params(temperatures=[0, 0.8, 1.5, -1], top_ks=[1, 5, 0, 1],
                               top_ps=[0, 0.95, 0.5, 1], min_ps=[0.05, 0.2, 0, 0.05])
        for impl, mapped, knobs in itertools.product(("triton", "torch"), (False, True),
                                                    ({}, {"temp_scale": 0.7, "onehot_above": 0.8})):
            hot = torch.randperm(n=vocab + 37)[:vocab].contiguous() if mapped else None
            args = dict(logits=logits, sampling=sampling, uniforms=torch.tensor(data=[0, 0.4, 1, 1]),
                        hot=hot, topk_impl=impl, **knobs)
            actual = call(module=base.rs2, greedy_fast=True, **args)
            expected = call(module=base.rs2, greedy_fast=False, **args)
            check_greedy_outputs(actual=actual, expected=expected, greedy=sampling["top_ks"] == 1)
            assert torch.all(actual[1][[0, 3], 0] == (2 if hot is None else hot[2]))
            # Without a chain, TopkIndex stays in the draft vocabulary.
            for fast in (False, True):
                out = base.rs2.rs_draft_proposal_sparse(next_token_logits=logits, **sampling,
                    uniforms=args["uniforms"], hot_token_id=hot, k=64, topk_impl=impl,
                    greedy_fast=fast, **knobs)
                assert torch.equal(input=out[3][[0, 3]], other=torch.full(size=(2, 1), fill_value=2))
            runs += 1
            rows += 4
            greedy_rows += 2
    assert greedy_rows > 0
    print(f"PASS G1b fast/full parity: {runs} cases/{rows} rows/{greedy_rows} greedy rows; "
          "mixed A grid, bf16 max ties within/across splits, hot mapping, small vocab, knobs, u=0/1, chain/no chain", flush=True)


def test_g1c():
    torch.manual_seed(20261005)
    bs, slots, vocab, draft_vocab = 7, 4, 173, 117
    rejected, accepted, runs = 0, 0, 0
    for tied, mapped, min_p in itertools.product((False, True), (False, True), (None, 0.05)):
        sampling = base.params(temperatures=[0, 0.3, 0.8, 1.5, 0.8, 0, 0.8],
                               top_ks=[1, 5, 0, 1, 40, 1, 5], top_ps=[1, 0.95, 0.5, 0, 0.95, 1, 1],
                               min_ps=None if min_p is None else [min_p] * bs)
        hot = torch.randperm(n=vocab)[:draft_vocab].contiguous() if mapped else None
        logits = torch.randn(size=(slots - 1, bs, draft_vocab))
        if tied:
            logits = logits.bfloat16()
            logits[:, :, [2, 25, 114]] = 8.0
        uniforms = torch.rand(size=(bs, slots - 1))
        p = torch.rand(size=(bs, slots, 128))
        p /= p.sum(dim=-1, keepdim=True)
        pi = torch.stack(tensors=[torch.randperm(n=vocab)[:128] for _ in range(bs * slots)]).reshape(bs, slots, 128)
        coins = torch.rand(size=(bs, slots))
        coins[[0, 3]] = 1.0
        coins[5] = 0.0
        final = torch.rand(size=(bs,))
        retrieve = torch.randperm(n=bs * slots).reshape(bs, slots)
        outputs = []
        for fast in (False, True):
            q = torch.empty(size=(bs, slots - 1, 64))
            qi = torch.empty(size=(bs, slots - 1, 64), dtype=torch.int64)
            candidates = torch.zeros(size=(bs, slots), dtype=torch.int64)
            for step in range(slots - 1):
                base.proposal(logits=logits[step], sampling=sampling, uniforms=uniforms[:, step], hot=hot,
                              greedy_fast=fast, draft_tokens=candidates, draft_token_column=step + 1,
                              draft_support_probs=q[:, step], draft_support_tokens=qi[:, step])
            out = base.verify_outputs(bs=bs, slots=slots)
            base.sparse_verify(outputs=out, candidates=candidates, retrieve=retrieve, coins=coins,
                               coins_final=final, p=p, pi=pi, q=q, qi=qi, vocab=vocab)
            outputs.append(out)
        identical(left=list(outputs[0].values()), right=list(outputs[1].values()))
        assert torch.all(outputs[1]["accept_token_num"][[0, 3]] == 0)
        rejected += int((outputs[1]["accept_token_num"] < slots - 1).sum())
        accepted += int((outputs[1]["accept_token_num"] > 0).sum())
        runs += 1
    assert rejected > 0 and accepted > 0
    print(f"PASS G1c verify predicts/accept_index/accept_token_num bit-identical: {runs} mixed batches/{runs * bs} rows; "
          f"random targets, shared uniforms, rejected={rejected}, accepted>=1={accepted}", flush=True)


def test_k4():
    torch.manual_seed(20261004)
    m = int(os.environ.get("RS2D_LOSSLESS_M", "1024"))
    slots, steps, vocab, hot_size = 3, 2, 64, 24
    target_logits = torch.randn(size=(slots, vocab)) * 0.4
    target_logits[:, :hot_size] += 2.0
    hot = torch.randperm(n=hot_size).long()
    draft_logits = target_logits[:steps, hot] + torch.randn(size=(steps, hot_size)) * 0.25
    # Make theta=.6 exercise both one-hot and distributed proposal rows.
    peak = int(target_logits[0, hot].argmax())
    draft_logits[0, peak] += 3.0
    print(f"  K4 seed=20261004 M={m} per configuration; peaked row uses target argmax", flush=True)
    sampling = base.params(temperatures=[0.8], top_ks=[8], top_ps=[0.95], min_ps=[0.05])
    p, pi = base.sparse_target(logits=target_logits, sampling=sampling, num_slots=slots, apply_min_p=True)
    dp = base.dense_target_probs(logits=target_logits, sampling=sampling, num_slots=slots, apply_min_p=True)
    mixed = base.params(temperatures=[0.8] * m, top_ks=[8] * m, top_ps=[0.95] * m, min_ps=[0.05] * m)
    for scale, theta in ((0.5, 0), (1.0, 0.6), (0.7, 0.8)):
        q = torch.empty(size=(m, steps, hot_size))
        qi = torch.empty(size=(m, steps, hot_size), dtype=torch.int64)
        candidates = torch.zeros(size=(m, slots), dtype=torch.int64)
        for step in range(steps):
            base.proposal(logits=draft_logits[step].expand(m, hot_size).contiguous(), sampling=mixed,
                          uniforms=torch.rand(size=(m,)), hot=hot, temp_scale=scale, onehot_above=theta,
                          draft_tokens=candidates, draft_token_column=step + 1,
                          draft_support_probs=q[:, step], draft_support_tokens=qi[:, step])
        out = base.verify_outputs(bs=m, slots=slots)
        retrieve = torch.arange(end=m * slots, dtype=torch.int64).reshape(m, slots)
        base.sparse_verify(outputs=out, candidates=candidates, retrieve=retrieve,
                           coins=torch.rand(size=(m, slots)), coins_final=torch.rand(size=(m,)),
                           p=p.repeat(m, 1, 1), pi=pi.repeat(m, 1, 1), q=q, qi=qi, vocab=vocab)
        pvalues = []
        for position in (0, 1):
            mask = out["accept_token_num"] >= position
            draws = out["predicts"][out["accept_index"][mask, position].long()].long()
            n = int(mask.sum())
            assert n >= 100
            stat, dof, value = base.chi_square(counts=torch.bincount(input=draws, minlength=vocab),
                                               probs=dp[position], n=n)
            pvalues.append(value)
            print(f"  K4 s={scale} theta={theta} pos={position} n={n} chi2={stat:.4f} dof={dof} p={value:.6f}", flush=True)
        fisher_stat, fisher_p = combine_pvalues(pvalues=pvalues, method="fisher")
        assert fisher_p > 0.01, fisher_p
        dense_q = torch.zeros(size=(vocab,)).scatter_(dim=0, index=qi[0, 0], src=q[0, 0])
        theory = torch.minimum(input=dp[0], other=dense_q).sum().item()
        actual = (out["accept_token_num"] >= 1).double().mean().item()
        se = math.sqrt(theory * (1 - theory) / m)
        assert abs(actual - theory) <= 3 * se, (actual, theory, se)
        print(f"PASS K4 s={scale} theta={theta} M={m}: Fisher p={fisher_p:.6f}, accept@0={actual:.6f} "
              f"sum(min(p,q))={theory:.6f} SE={se:.6f} z={(actual-theory)/se:.3f}", flush=True)


class Request(msgspec.Struct):
    rid: str
    origin_input_ids: list[int]


def synthetic_record():
    return dict(candidates=torch.tensor(data=[[0, 10, 10]], dtype=torch.int64),
                target_probs=torch.tensor(data=[[[0.9, 0.1]] * 3]),
                target_index=torch.tensor(data=[[[10, 20]] * 3], dtype=torch.int64),
                draft_support_probs=torch.tensor(data=[[[0.6, 0.4]] * 2]),
                draft_support_tokens=torch.tensor(data=[[[10, 20]] * 2], dtype=torch.int64),
                accept_len=torch.tensor(data=[1], dtype=torch.int64),
                temperatures=torch.tensor(data=[[0.8]]), top_ks=torch.tensor(data=[2]),
                top_ps=torch.tensor(data=[0.95]), min_ps=torch.tensor(data=[0.05]))


def test_k5():
    tensors = synthetic_record()
    with tempfile.TemporaryDirectory(prefix="rs2d-dump-") as directory:
        reqs = [Request(rid="synthetic", origin_input_ids=[1, 2, 3])]
        start = time.time()
        for _ in range(50):
            dump.record_verify(directory=directory, reqs=reqs, **tensors)
        first = list(Path(directory).glob("rs-dump-*.pt"))
        assert len(first) == 1
        dump.record_verify(directory=directory, reqs=reqs, **tensors)
        tensors["target_probs"].zero_()
        dump.flush()
        chunks = sorted(Path(directory).glob("rs-dump-*.pt"))
        records = [torch.load(f=p, weights_only=True, map_location="cpu") for p in chunks]
        assert [len(r) for r in records] == [50, 1]
        expected = synthetic_record()
        for record in [records[0][0], records[1][0]]:
            assert set(record) == {"time", "rid", "input_len", *expected}
            assert start <= record["time"] <= time.time()
            assert record["rid"] == ["synthetic"] and record["input_len"] == [3]
            for name, value in expected.items():
                tensor = record[name]
                dtype = torch.int32 if name in ("candidates", "target_index", "draft_support_tokens", "accept_len", "top_ks") else torch.float32
                assert tensor.device.type == "cpu" and tensor.dtype == dtype
                assert tensor.shape == value.shape and torch.equal(input=tensor, other=value.to(dtype=dtype))
    print("PASS K5 dump: real shapes/fields/CPU dtypes, auto-flush 50 + explicit tail 1, independent snapshots", flush=True)


def test_k6():
    record = {**synthetic_record(), "input_len": [3], "rid": ["hand"], "time": time.time()}
    grid = ((1.0, 0.0), (0.5, 0.0), (1.0, 0.6))
    metrics = offline.row_metrics(record=record, row=0, grid=grid)
    assert abs(metrics["a_rs"][0] - 0.7) < 1e-6
    assert abs(metrics["a_target_only"][0] - 0.9) < 1e-6
    for pair, a in zip(grid, (0.7, 0.7923076923076923, 0.9)):
        assert abs(metrics["grid"][pair] - (a + a * a)) < 1e-6
    assert abs(offline.acceptance(p=torch.tensor(data=[0.3, 0.6, 0.1]),
                                 pi=torch.tensor(data=[10, 10, 20]), q=torch.tensor(data=[0.6, 0.4, 0.0]),
                                 qi=torch.tensor(data=[10, 20, 30])) - 0.7) < 1e-6
    q = torch.tensor(data=[0.7, 0.3, 0.0])
    assert torch.equal(input=offline.sharpen(q=q, scale=1, threshold=0, top_p=0.5, min_p=0.05), other=q.double())
    with tempfile.TemporaryDirectory(prefix="rs2d-offline-") as directory:
        torch.save(obj=[record], f=Path(directory) / "rs-dump-0-00000.pt")
        fnbench = Path(directory) / "fnbench.jsonl"
        fnbench.write_text(json.dumps({"workload": "code-edit", "prompt_id": "hand",
            "timestamp": datetime.fromtimestamp(record["time"] - 1, tz=timezone.utc).isoformat(),
            "client": {"usage": {"prompt_tokens": 3}, "ttft_seconds": 0.5, "decode_seconds": 1}}) + "\n")
        result = subprocess.run(args=[sys.executable, str(Path(__file__).with_name("accept_offline.py")),
                                      directory, str(fnbench)], capture_output=True, text=True, check=True)
        assert "code-edit: verifies=1" in result.stdout and "RS2=2.190000" in result.stdout
        assert "target-only=2.710000" in result.stdout and "Caveat:" in result.stdout
        csv = (Path(directory) / "rs2d-accept-grid.csv").read_text().splitlines()
        assert len(csv) == 1 + 2 * len(offline.GRID)
    print("PASS K6 offline CLI/CSV: a=.7/.9/.792308/.9; two-position E=a0+a0*a1; duplicate ids, s=1 exact", flush=True)


if __name__ == "__main__":
    tests = {"G1a": test_g1a, "G1b": test_g1b, "G1c": test_g1c, "K1K2": test_k1_k2, "K3": test_k3, "K4": test_k4, "K5": test_k5, "K6": test_k6}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", nargs="+", choices=list(tests), default=["K1K2", "K3", "K4", "K5", "K6"])
    args = parser.parse_args()
    torch.manual_seed(20261002)
    suite_start = time.perf_counter()
    print(f"RS2d CPU worktree={base.WT} seed=20261002", flush=True)
    for name in args.parts:
        test = tests[name]
        start = time.perf_counter()
        try:
            test()
        except Exception as error:
            print(f"FAIL {name}: {type(error).__name__}: {error}", flush=True)
            raise
        print(f"  {name} elapsed={time.perf_counter() - start:.1f}s", flush=True)
    print(f"PASS requested CPU tests={args.parts} total elapsed={time.perf_counter() - suite_start:.1f}s", flush=True)
