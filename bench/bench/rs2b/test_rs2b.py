"""RS2b correctness on CPU and parent-run real FlashInfer CUDA checks."""

import importlib.util
import contextlib
import itertools
import os
from pathlib import Path
import sys
import types
from unittest.mock import patch

import torch

DEV = os.environ.get("DEV", "cpu")
if DEV == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    assert os.environ.get("TRITON_INTERPRET") == "1"

FAKE_CALLS = []


def fake_top_k(*, input, k, sorted, deterministic, tie_break):
    assert input.device.type == "cpu"
    assert sorted is False and deterministic is True and tie_break == 1
    assert input.shape[1] >= k
    FAKE_CALLS.append((input.shape[0], input.shape[1], k))
    vals = input.float()
    bits = vals.view(dtype=torch.int32).to(dtype=torch.int64)
    ordered = torch.where(condition=bits < 0, input=(~bits) & 0xFFFFFFFF,
                          other=bits ^ 0x80000000)
    ordered = torch.where(condition=torch.isnan(input=vals),
                          input=torch.full_like(input=ordered, fill_value=0xFFFFFFFF), other=ordered)
    ids = torch.arange(end=input.shape[1], dtype=torch.int64)
    keys = ((ordered - 0x80000000) << 32) | (0xFFFFFFFF - ids)
    selected = torch.topk(input=keys, k=k, dim=-1).indices
    # Independently shuffle each row and return padded row strides.
    generator = torch.Generator(device="cpu").manual_seed(7331 + input.shape[0] + k)
    order = torch.argsort(input=torch.rand(size=selected.shape, generator=generator), dim=-1)
    selected = selected.gather(dim=1, index=order)
    indices = torch.empty(size=(input.shape[0], k + 5), dtype=torch.int64)[:, :k]
    values = torch.empty(size=(input.shape[0], k + 3), dtype=input.dtype)[:, :k]
    indices.copy_(selected)
    values.copy_(input.gather(dim=1, index=selected))
    return values, indices


if DEV == "cpu":
    fake = types.ModuleType(name="flashinfer")
    fake.top_k = fake_top_k
    sys.modules["flashinfer"] = fake

path = Path(__file__).resolve().parents[1] / "rs2/test_sparse_rs.py"
spec = importlib.util.spec_from_file_location(name="rs2b_base_tests", location=path)
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
base.IMPLS = ("triton", "torch", "flashinfer")


def assert_bits(*, current, expected, context):
    assert current.shape == expected.shape and current.dtype == expected.dtype, context
    if current.is_floating_point():
        current = current.contiguous().view(dtype=torch.int32)
        expected = expected.contiguous().view(dtype=torch.int32)
    assert torch.equal(input=current, other=expected), context


def run_outputs(*, logits, sampling, uniforms, hot, impl, chain=True):
    n = logits.shape[0]
    positions = torch.arange(end=n, dtype=torch.int64, device=DEV) + 13
    draft_tokens = torch.full(size=(n, 4), fill_value=-1, dtype=torch.int64, device=DEV)
    support_q = torch.full(size=(n, 3, min(64, logits.shape[1])), fill_value=-7.0, device=DEV)
    support_tokens = torch.full_like(input=support_q, fill_value=-1, dtype=torch.int64)
    outputs = base.proposal(
        logits=logits, sampling=sampling, uniforms=uniforms, hot=hot, impl=impl,
        positions=positions, draft_tokens=draft_tokens if chain else None,
        draft_token_column=2, draft_support_probs=support_q[:, 1],
        draft_support_tokens=support_tokens[:, 1],
    )
    assert torch.equal(input=positions, other=torch.arange(end=n, device=DEV) + 14)
    assert torch.all(support_q[:, [0, 2]] == -7)
    assert torch.all(support_tokens[:, [0, 2]] == -1)
    assert torch.all(draft_tokens[:, [0, 1, 3]] == -1)
    if chain:
        assert torch.equal(input=draft_tokens[:, 2:3], other=outputs[3])
    else:
        assert torch.all(draft_tokens == -1)
    return (*outputs, positions, draft_tokens)


@contextlib.contextmanager
def cpu_reference_keys(*, logits):
    if DEV != "cpu":
        yield
        return
    keys = base.rs2._proposal_topk_keys(logits=logits, k_block=64, topk_impl="triton")
    ranked = torch.topk(input=keys, k=64, dim=-1).values
    n = logits.shape[0]
    sampling = base.params(temperatures=[0.8] * n, top_ks=[0] * n, top_ps=[0.95] * n)
    kwargs = dict(logits=logits, sampling=sampling, uniforms=torch.zeros(size=(n,)), hot=None)
    with patch.object(target=base.rs2, attribute="_proposal_topk_keys", return_value=keys):
        original = run_outputs(impl="triton", **kwargs)

    def selector(*, logits, k_block, topk_impl):
        assert logits is kwargs["logits"] and k_block == 64 and topk_impl == "triton"
        return ranked

    # Reuse real Triton selection for unchanged logits; check compact-key finalize once.
    with patch.object(target=base.rs2, attribute="_proposal_topk_keys", new=selector):
        compact = run_outputs(impl="triton", **kwargs)
        for current, reference in zip(compact, original):
            assert_bits(current=current, expected=reference, context="CPU compact reference vs original Triton finalize")
        yield


def check_nan_outputs(*, actual, expected, hot, vocab, nan_count, top_ks):
    q, tokens, qx, draws = actual[:4]
    assert torch.isfinite(input=q).all() and torch.all(q >= 0)
    assert torch.all((q.sum(dim=-1) - 1).abs() <= 1e-5)
    if hot is None:
        assert torch.all((tokens >= 0) & (tokens < vocab))
        assert torch.all((draws >= 0) & (draws < vocab))
    else:
        assert torch.isin(elements=tokens, test_elements=hot).all()
        assert torch.isin(elements=draws, test_elements=hot).all()
    hits = tokens == draws
    assert torch.all(hits.sum(dim=1) == 1)
    assert_bits(current=qx, expected=(q * hits).sum(dim=1, keepdim=True), context="NaN q(X)")
    assert torch.all(qx > 0)
    real_ranks = 64 - nan_count
    assert_bits(current=tokens[:, :real_ranks], expected=expected[1][:, :real_ranks],
                context=("NaN real prefix", nan_count))
    assert torch.all(q[:, real_ranks:] == 0)
    for row in range(q.shape[0]):
        if 0 < top_ks[row] <= real_ranks:
            assert_bits(current=q[row, :real_ranks], expected=expected[0][row, :real_ranks],
                        context=("NaN prefix q", row, nan_count))
            assert_bits(current=qx[row], expected=expected[2][row], context="NaN capped q(X)")
            assert_bits(current=draws[row], expected=expected[3][row], context="NaN capped draw")


def test_a2():
    combinations = list(itertools.product((0.3, 0.8, 1.5), (1, 5, 40, 0), (1.0, 0.95, 0.5)))
    calls, rows_checked = 0, 0
    for vocab, dtype, n in itertools.product((49152, 117), (torch.float32, torch.bfloat16), (1, 2, 4, 7)):
        storage = torch.randn(size=(n, vocab + 11), dtype=dtype, device=DEV)
        logits = storage[:, :vocab]
        if dtype == torch.bfloat16:
            logits[:, :80] = 8.0
        hot = torch.randperm(n=vocab + 37, device=DEV)[:vocab].contiguous()
        uniforms_storage = torch.rand(size=(n, 2), device=DEV)
        uniforms_storage[:, 0] = torch.tensor(data=[0.0, 1.0, 0.4, 0.99, 0.17, 0.5, 0.83][:n], device=DEV)
        uniforms = uniforms_storage[:, 0]
        with cpu_reference_keys(logits=logits):
            for start, min_p, mapping in itertools.product(range(0, len(combinations), n), (None, 0.05, 0.2), (None, hot)):
                rows = [combinations[(start + row) % len(combinations)] for row in range(n)]
                sampling = base.params(temperatures=[r[0] for r in rows], top_ks=[r[1] for r in rows],
                                       top_ps=[r[2] for r in rows],
                                       min_ps=None if min_p is None else [min_p] * n)
                expected = run_outputs(logits=logits, sampling=sampling, uniforms=uniforms, hot=mapping, impl="triton")
                actual = run_outputs(logits=logits, sampling=sampling, uniforms=uniforms, hot=mapping, impl="flashinfer")
                for name, current, reference in zip(("q", "tokens", "q(X)", "draw", "positions", "chain"), actual, expected):
                    assert_bits(current=current, expected=reference, context=(name, vocab, dtype, n, rows, min_p, mapping is not None))
                calls += 1
                rows_checked += n
        print(f"  A2 exact Vd={vocab} dtype={dtype} n={n} all 36 sampling combinations/min_p/hot checked", flush=True)
    nan_rows = 0
    for dtype, n, nan_count in itertools.product((torch.float32, torch.bfloat16), (1, 2, 4, 7), (1, 7, 63, 83)):
        vocab = 83
        logits = torch.randn(size=(n, vocab + 11), dtype=dtype, device=DEV)[:, :vocab]
        nan_ids = torch.randperm(n=vocab, device=DEV)[:nan_count]
        logits[:, nan_ids] = float("nan")
        hot = torch.randperm(n=127, device=DEV)[:vocab].contiguous()
        uniforms = torch.rand(size=(n,), device=DEV)
        for min_p, mapping in itertools.product((None, 0.05, 0.2), (None, hot)):
            real_ranks = 64 - min(nan_count, 64)
            top_ks = [min((1, 5, 40, 0)[row % 4], real_ranks) for row in range(n)]
            sampling = base.params(temperatures=[(0.3, 0.8, 1.5)[row % 3] for row in range(n)],
                                   top_ks=top_ks, top_ps=[(1.0, 0.95, 0.5)[row % 3] for row in range(n)],
                                   min_ps=None if min_p is None else [min_p] * n)
            expected = run_outputs(logits=logits, sampling=sampling, uniforms=uniforms, hot=mapping, impl="triton")
            actual = run_outputs(logits=logits, sampling=sampling, uniforms=uniforms, hot=mapping, impl="flashinfer")
            if nan_count == vocab:
                for current, reference in zip(actual, expected):
                    assert_bits(current=current, expected=reference, context="all-NaN exact")
            else:
                check_nan_outputs(actual=actual, expected=expected, hot=mapping, vocab=vocab,
                                  nan_count=nan_count, top_ks=top_ks)
            nan_rows += n
    # Exceptional values stay away from the signed-zero selection boundary.
    for dtype in (torch.float32, torch.bfloat16):
        logits = torch.full(size=(4, 117), fill_value=-2.0, dtype=dtype, device=DEV)
        logits[:, :80] = 3.0
        logits[0, 0] = float("inf")
        logits[1, :2] = float("inf")
        logits[2] = -float("inf")
        logits[3, 110:112] = torch.tensor(data=[-0.0, 0.0], dtype=dtype, device=DEV)
        sampling = base.params(temperatures=[0, 0.3, 0.8, 1.5], top_ks=[0, 5, 40, 1], top_ps=[1, 0.95, 0.5, 0])
        for chain in (False, True):
            kwargs = dict(logits=logits, sampling=sampling, uniforms=torch.ones(size=(4,), device=DEV), hot=None, chain=chain)
            actual = run_outputs(impl="flashinfer", **kwargs)
            expected = run_outputs(impl="triton", **kwargs)
            for current, reference in zip(actual, expected):
                assert_bits(current=current, expected=reference, context="infinities/zeros/fillers/chain-off")
    if DEV == "cpu":
        before = len(FAKE_CALLS)
        logits = torch.randn(size=(2, 7), device=DEV)
        sampling = base.params(temperatures=[0.8, 1.5], top_ks=[0, 5], top_ps=[1, 0.95])
        kwargs = dict(logits=logits, sampling=sampling, uniforms=torch.zeros(size=(2,), device=DEV), hot=None)
        actual = run_outputs(impl="flashinfer", **kwargs)
        expected = run_outputs(impl="triton", **kwargs)
        assert len(FAKE_CALLS) == before, "vocab < k_block must bypass FlashInfer"
        for current, reference in zip(actual, expected):
            assert_bits(current=current, expected=reference, context="small-vocab fallback")
        assert any(vocab == 49152 and k == 64 for _, vocab, k in FAKE_CALLS)
        assert any(vocab == 83 and k == 64 for _, vocab, k in FAKE_CALLS)
        print(f"  A2 fake FlashInfer calls={len(FAKE_CALLS)}; shuffled strided values/indices; small-vocab bypass checked", flush=True)
    print(f"PASS A2 topk_impl=flashinfer vs triton: {calls} exact calls/{rows_checked} NaN-free rows; "
          f"{nan_rows} NaN rows; q/tokens/q(X)/draw/chain/positions; infinities/zeros/fillers/fallback", flush=True)


def test_g():
    if DEV == "cpu":
        print("SKIP G: CPU mode; real FlashInfer CUDA graph capture/replay requires the parent GPU run", flush=True)
        return
    for dtype, n in itertools.product((torch.float32, torch.bfloat16), (1, 2, 4, 7)):
        vocab = 49152
        logits = torch.randn(size=(n, vocab + 11), dtype=dtype, device=DEV)[:, :vocab]
        sampling = base.params(temperatures=[0.8] * n, top_ks=[40] * n, top_ps=[0.95] * n, min_ps=[0.05] * n)
        hot = torch.randperm(n=vocab + 37, device=DEV)[:vocab].contiguous()
        uniforms = torch.rand(size=(n,), device=DEV)
        positions = torch.zeros(size=(n,), dtype=torch.int64, device=DEV)
        chain = torch.full(size=(n, 4), fill_value=-1, dtype=torch.int64, device=DEV)
        q = torch.empty(size=(n, 3, 64), device=DEV)
        tokens = torch.empty(size=(n, 3, 64), dtype=torch.int64, device=DEV)

        def operation():
            return base.proposal(logits=logits, sampling=sampling, uniforms=uniforms, hot=hot, impl="flashinfer",
                                 positions=positions, draft_tokens=chain, draft_token_column=2,
                                 draft_support_probs=q[:, 1], draft_support_tokens=tokens[:, 1])

        stream = torch.cuda.Stream()
        stream.wait_stream(stream=torch.cuda.current_stream())
        with torch.cuda.stream(stream=stream):
            for _ in range(3):
                operation()
        torch.cuda.current_stream().wait_stream(stream=stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cuda_graph=graph, stream=stream):
            captured = operation()
        for replay in range(3):
            logits.copy_(torch.randn_like(input=logits))
            uniforms.copy_(torch.rand_like(input=uniforms))
            positions.copy_(torch.arange(end=n, device=DEV) + replay)
            chain.fill_(-1)
            expected = run_outputs(logits=logits, sampling=sampling, uniforms=uniforms, hot=hot, impl="flashinfer")
            # Match eager's initial positions, then compare every output after replay.
            positions.copy_(torch.arange(end=n, device=DEV) + 13)
            graph.replay()
            torch.cuda.synchronize()
            for current, reference in zip((*captured, positions, chain), expected):
                assert_bits(current=current, expected=reference, context=("G", dtype, n, replay))
        print(f"  G dtype={dtype} n={n}: 3 changed-logit/uniform replays equal eager", flush=True)
    print("PASS G: real FlashInfer capture and three changed-input replays; all outputs bit-identical to eager", flush=True)


def test_e():
    if DEV == "cpu":
        print("SKIP E: CPU mode; real FlashInfer CUDA graph GPU/wall timing requires the parent GPU run", flush=True)
        return
    results = {}
    for dtype, n in itertools.product((torch.float32, torch.bfloat16), (1, 2, 4)):
        vocab, target_vocab = 49152, 248320
        logits = torch.randn(size=(n, vocab), dtype=dtype, device=DEV)
        sampling = base.params(temperatures=[0.8] * n, top_ks=[40] * n, top_ps=[0.95] * n, min_ps=[0.05] * n)
        hot = torch.randperm(n=target_vocab, device=DEV)[:vocab].contiguous()
        uniforms = torch.rand(size=(n,), device=DEV)
        positions = torch.zeros(size=(n,), dtype=torch.int64, device=DEV)
        chain = torch.empty(size=(n, 14), dtype=torch.int64, device=DEV)

        def baseline():
            q, ids, _, _ = base.reference_proposal(
                next_token_logits=logits, temperatures=sampling["temperatures"][:, None],
                top_ks=sampling["top_ks"], top_ps=sampling["top_ps"], min_ps=sampling["min_ps"], k_cap=64)
            torch.zeros(size=(n, target_vocab), device=DEV).scatter_(dim=1, index=hot[ids], src=q)

        operations = {
            "RS2 triton": lambda: base.proposal(logits=logits, sampling=sampling, uniforms=uniforms, hot=hot,
                                              impl="triton", positions=positions, draft_tokens=chain),
            "RS2 torch": lambda: base.proposal(logits=logits, sampling=sampling, uniforms=uniforms, hot=hot,
                                             impl="torch", positions=positions, draft_tokens=chain),
            "RS2b flashinfer": lambda: base.proposal(logits=logits, sampling=sampling, uniforms=uniforms, hot=hot,
                                                  impl="flashinfer", positions=positions, draft_tokens=chain),
            "RS1+scatter": baseline,
            "greedy": lambda: base.greedy.draft_topk1_postprocess(
                next_token_logits=logits, positions=positions, hot_token_id=hot, draft_tokens=chain),
        }
        for name, operation in operations.items():
            gpu_us, wall_us = base.graph_time(operation=operation, calls=14)
            results[(dtype, n, name)] = gpu_us
            print(f"  E proposal dtype={dtype} n={n} Vd={vocab} graph_calls=14 {name:16s} "
                  f"GPU={gpu_us:.3f} us/call wall={wall_us:.3f} us/call", flush=True)
    measured = results[(torch.float32, 1, "RS2b flashinfer")]
    if measured > 10:
        print(f"WARN E: RS2b n=1 fp32 {measured:.3f} us exceeds the 10 us budget", flush=True)
    print("PASS E: graph GPU/wall measurements complete; RS2b n=1 fp32 budget is warning-only", flush=True)


if __name__ == "__main__":
    torch.manual_seed(seed=20261001)
    print(f"RS2b device={DEV} worktree={base.WT} seed=20261001", flush=True)
    if DEV == "cpu":
        print("SKIP real FlashInfer: CPU fake supplies shuffled radix-contract values/indices; CUDA uses the real module", flush=True)
    for name, test in (("A", base.test_a), ("A2", test_a2), ("B", base.test_b), ("G", test_g), ("E", test_e)):
        try:
            test()
        except Exception as error:
            print(f"FAIL {name}: {type(error).__name__}: {error}", flush=True)
            raise
    print("SKIP F: server smoke/profile and ABBA are parent-run checks", flush=True)
