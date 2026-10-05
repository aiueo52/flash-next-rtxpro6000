"""RS3 block verification CPU interpreter tests, selectable as B0-B4 and B6."""

import argparse
import ast
import itertools
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

import numpy as np
import torch
from scipy.stats import combine_pvalues

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("WT", "/home/user/tools/sglang-rs3")
assert os.environ.get("DEV") == "cpu"
assert os.environ.get("TRITON_INTERPRET") == "1"
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "bench/rs2"))
sys.path.insert(0, str(ROOT / "bench/rs2d"))
import test_sparse_rs as base
import test_rs2d as rs2d
import bv_exact

G1 = "124840a5c3"


def verify(*, module=base.rs2, block_verify=True, **batch):
    bs, slots = batch["candidates"].shape
    outputs = base.verify_outputs(bs=bs, slots=slots)
    module.chain_speculative_sampling_sparse(
        **outputs, candidates=batch["candidates"], retrive_index=batch["retrieve"],
        uniform_samples=batch["coins"], uniform_samples_for_final_sampling=batch["coins_final"],
        target_probs=batch["p"], target_index=batch["pi"], draft_support_probs=batch["q"],
        draft_support_tokens=batch["qi"], vocab_size=batch["vocab"],
        **({} if block_verify is None else {"block_verify": block_verify}),
    )
    return outputs


def same(*, actual, expected):
    rs2d.identical(left=list(actual.values()), right=list(expected.values()))


def test_b0():
    source = subprocess.run(args=["git", "-C", str(base.WT), "show",
                                 G1 + ":python/sglang/kernels/ops/speculative/sparse_rs.py"],
                            capture_output=True, text=True, check=True).stdout
    class DisabledBV(ast.NodeTransformer):
        def visit_If(self, node):
            if isinstance(node.test, ast.Name) and node.test.id == "BLOCK_VERIFY":
                return [self.visit(child) for child in node.orelse]
            return self.generic_visit(node)

    old_kernel = next(node for node in ast.parse(source).body
                      if isinstance(node, ast.FunctionDef) and node.name == "_chain_sampling_sparse_kernel")
    current_source = (base.WT / "python/sglang/kernels/ops/speculative/sparse_rs.py").read_text()
    kernel = next(node for node in ast.parse(current_source).body
                  if isinstance(node, ast.FunctionDef) and node.name == old_kernel.name)
    kernel.args.args = [arg for arg in kernel.args.args if arg.arg not in ("BLOCK_VERIFY", "SP")]
    kernel = DisabledBV().visit(kernel)
    assert ast.dump(kernel, include_attributes=False) == ast.dump(old_kernel, include_attributes=False)
    print("  B0 constexpr False kernel AST equals G1 after branch elimination", flush=True)
    comparisons = 0
    with tempfile.TemporaryDirectory(prefix="rs3-g1-") as directory:
        path = Path(directory) / "g1.py"
        path.write_text(source)
        g1 = rs2d.load_path(name="rs3_g1_reference", path=path)
        original = base.sparse_verify

        def compare(*, outputs, **batch):
            nonlocal comparisons
            original(outputs=outputs, **batch)
            expected = verify(module=g1, block_verify=None, **batch)
            same(actual=outputs, expected=expected)
            same(actual=verify(block_verify=False, **batch), expected=expected)
            comparisons += 1

        with patch.object(target=base, attribute="sparse_verify", new=compare):
            rs2d.test_g1c()
    print(f"PASS B0 G1={G1}: omitted/False identical on {comparisons} G1c batches", flush=True)


def random_batch(*, slots, kp, seed, greedy=False):
    torch.manual_seed(seed)
    bs, k, vocab = 14, 64, 193
    p = torch.rand(size=(bs, slots, kp))
    p /= p.sum(dim=-1, keepdim=True)
    pi = torch.stack(tensors=[torch.randperm(n=vocab)[:kp] for _ in range(bs * slots)]).reshape(bs, slots, kp)
    q = torch.rand(size=(bs, slots - 1, k))
    q /= q.sum(dim=-1, keepdim=True)
    qi = torch.stack(tensors=[torch.randperm(n=vocab)[:k] for _ in range(bs * (slots - 1))]).reshape(bs, slots - 1, k)
    candidates = torch.zeros(size=(bs, slots), dtype=torch.int64)
    candidates[:, 1:] = qi[:, :, 0]
    for row in range(bs):
        kind = row % 7
        if kind == 1 or greedy:
            p[row].zero_()
            p[row, :, 0] = 1
        if kind == 2:
            q[row].zero_()
            q[row, :, 0] = 1
        if kind == 3 and not greedy:
            p[row, :-1] *= ~(pi[row, :-1, :, None] == qi[row, :, None, :]).any(dim=-1)
            p[row] /= p[row].sum(dim=-1, keepdim=True)
        if kind == 4:
            q[row, :, -1] = float("nan")
        if kind == 5:
            q[row, ::2].zero_()
            qi[row, ::2, :kp] = pi[row, :-1:2]
            q[row, ::2, :kp] = p[row, :-1:2]
            candidates[row, 1::2] = pi[row, :-1:2, 0]
        if kind == 6:
            missing = next(token for token in range(vocab) if token not in pi[row, 0])
            qi[row, 0, 0] = missing
            candidates[row, 1] = missing
        if greedy:
            for step in range(slots - 1):
                if step < row % slots:
                    candidates[row, step + 1] = pi[row, step, 0]
                    qi[row, step, 0] = pi[row, step, 0]
    coins = torch.rand(size=(bs, slots))
    final = torch.rand(size=(bs,))
    final[:2] = torch.tensor(data=[0, 1])
    return dict(candidates=candidates, retrieve=torch.randperm(n=bs * slots).reshape(bs, slots),
                coins=coins, coins_final=final, p=p, pi=pi, q=q, qi=qi, vocab=vocab)


def python_reference(*, candidates, retrieve, coins, coins_final, p, pi, q, qi, vocab):
    bs, slots = candidates.shape
    outputs = base.verify_outputs(bs=bs, slots=slots)
    for row in range(bs):
        clean_q = torch.where(condition=q[row] == q[row], input=q[row], other=0).numpy()
        probs, ids, qids = p[row].numpy(), pi[row].numpy(), qi[row].numpy()
        path_pi = np.float32(1)
        pis, hs = [path_pi], []
        for step in range(1, slots):
            token = int(candidates[row, step])
            pt = probs[step - 1][ids[step - 1] == token].sum(dtype=np.float32)
            qt = clean_q[step - 1][qids[step - 1] == token].sum(dtype=np.float32)
            # Same arithmetic and edge rules as bv_exact.bv_tau_probs, in fp32.
            path_pi = np.float32(0 if pt <= 0 else (1 if qt == 0 else min(path_pi * pt / qt, 1)))
            pis.append(path_pi)
            h = path_pi
            if step < slots - 1:
                matched = np.where(ids[step, :, None] == qids[step, None, :], clean_q[step, None, :], 0).sum(axis=1, dtype=np.float32)
                n = np.maximum(path_pi * probs[step] - matched, 0).sum(dtype=np.float32)
                denominator = np.float32(n + np.float32(1) - path_pi)
                h = np.float32(n / denominator if denominator > 0 else 1)
            hs.append(h)
            if path_pi == 0:
                break
        tau = max([0] + [i for i, h in enumerate(hs, start=1) if coins[row, i - 1] < h])
        outputs["accept_token_num"][row] = tau
        outputs["accept_index"][row, :tau + 1] = retrieve[row, :tau + 1].int()
        for step in range(tau):
            outputs["predicts"][retrieve[row, step]] = candidates[row, step + 1].int()
        weights = probs[tau].copy()
        if tau < slots - 1:
            matched = np.where(ids[tau, :, None] == qids[tau, None, :], clean_q[tau, None, :], 0).sum(axis=1, dtype=np.float32)
            weights = np.maximum(pis[tau] * weights - matched, 0)
            if weights.sum(dtype=np.float32) <= 0:
                weights = probs[tau].copy()
        order = np.argsort(ids[tau], kind="stable")
        weights, tokens = weights[order], ids[tau, order]
        u = np.float32(coins_final[row]) * weights.sum(dtype=np.float32)
        hits = (weights.cumsum(dtype=np.float32) > u) & (weights > 0)
        token = min(tokens[hits], default=vocab)
        last_valid = max(tokens[weights > 0], default=-1)
        token = token if token < vocab else (last_valid if last_valid >= 0 else vocab - 1)
        outputs["predicts"][retrieve[row, tau]] = int(token)
    return outputs


def test_b1():
    rows, runs, recovered = 0, 0, 0
    for slots, kp, seed in itertools.product((2, 4, 8, 16), (16, 64), range(4)):
        batch = random_batch(slots=slots, kp=kp, seed=20261010 + seed)
        actual = verify(**batch)
        same(actual=actual, expected=python_reference(**batch))
        token = verify(block_verify=False, **batch)
        recovered += int((actual["accept_token_num"] > token["accept_token_num"]).sum())
        runs += 1
        rows += batch["candidates"].shape[0]
    assert recovered > 0
    edge_p = torch.tensor(data=[[[0.2, 0.3, 0.5, 0.0]] * 4] * 4)
    edge_pi = torch.tensor(data=[[[5, 5, 2, 7]] * 4] * 4, dtype=torch.int64)
    edge_q = edge_p[:, :-1].clone()
    edge_q[2] = float("nan")
    edge_p[3].zero_()
    batch = dict(candidates=torch.tensor(data=[[0, 5, 2, 5]] * 4, dtype=torch.int64),
                 retrieve=torch.randperm(n=16).reshape(4, 4),
                 coins=torch.tensor(data=[[0, 0, 0, 0], [1, 1, 1, 1], [0.4, 0.7, 0.2, 0], [0, 0, 0, 0]]),
                 coins_final=torch.tensor(data=[0, 1, 1, 0.5]), p=edge_p, pi=edge_pi, q=edge_q,
                 qi=edge_pi[:, :-1].clone(), vocab=8)
    actual = verify(**batch)
    same(actual=actual, expected=python_reference(**batch))
    assert actual["accept_token_num"].tolist() == [3, 0, 3, 0]
    assert actual["predicts"][batch["retrieve"][3, 0]].item() == 7
    print("  B1 edge rows: duplicate ids, p=q denominator=0, zero residual, all-NaN q, empty target, coins=0/1", flush=True)
    print(f"PASS B1 Python section-1 reference: {runs} batches/{rows} rows; slots=2/4/8/16 kp=16/64 K=64; "
          f"seven row kinds; later-success rows={recovered}; final coins=0/1", flush=True)


def test_b2():
    m = int(os.environ.get("RS3_M", "4096"))
    pvalues = []
    for case, (vocab, steps, mode) in enumerate(((4, 2, "gen"), (5, 2, "equal"), (6, 3, "gen"), (4, 3, "greedy"))):
        rng = np.random.default_rng(20261100 + case)
        P, Q = bv_exact.random_case(seed=20261100 + case, V=vocab, G=steps, mode=mode)
        if mode != "greedy":
            for prefix in P:
                equal = np.array_equal(P[prefix], Q[prefix])
                P[prefix] = 0.6 * P[prefix] + 0.4 / vocab
                Q[prefix] = P[prefix].copy() if equal else 0.6 * P[prefix] + 0.4 * Q[prefix]
        exact, expected = bv_exact.enumerate_outputs(P=P, Q=Q, V=vocab, G=steps, method="bv")
        target = bv_exact.target_distribution(P=P, V=vocab, G=steps)
        assert max(abs(exact[key] - target[key]) for key in target) < 1e-12
        paths = list(itertools.product(range(vocab), repeat=steps))
        path_weights = np.array([np.prod([Q[path[:i]][path[i]] for i in range(steps)]) for path in paths])
        second_moment = sum(weight * sum(t * t * prob for t, prob in enumerate(
            bv_exact.bv_tau_probs(path=path, P=P, Q=Q)[0])) for path, weight in zip(paths, path_weights))
        selected = rng.choice(len(paths), size=m, p=path_weights / path_weights.sum())
        slots, kp = steps + 1, 1 << (vocab - 1).bit_length()
        p = torch.zeros(size=(m, slots, kp))
        pi = torch.arange(end=kp, dtype=torch.int64).expand(m, slots, kp).clone()
        q = torch.empty(size=(m, steps, vocab))
        qi = torch.arange(end=vocab, dtype=torch.int64).expand(m, steps, vocab).clone()
        candidates = torch.zeros(size=(m, slots), dtype=torch.int64)
        for row, index in enumerate(selected):
            path = paths[index]
            candidates[row, 1:] = torch.tensor(data=path)
            for step in range(slots):
                p[row, step, :vocab] = torch.tensor(data=P[path[:step]], dtype=torch.float32)
                if step < steps:
                    q[row, step] = torch.tensor(data=Q[path[:step]], dtype=torch.float32)
        out = verify(candidates=candidates, retrieve=torch.arange(end=m * slots).reshape(m, slots),
                     coins=torch.tensor(data=rng.random((m, slots)), dtype=torch.float32),
                     coins_final=torch.tensor(data=rng.random(m), dtype=torch.float32),
                     p=p, pi=pi, q=q, qi=qi, vocab=vocab)
        counts = torch.zeros(size=(vocab ** slots,), dtype=torch.int64)
        for row in range(m):
            tau = int(out["accept_token_num"][row])
            seq = out["predicts"][row * slots:row * slots + tau + 1].tolist()
            while len(seq) < slots:
                seq.append(int(rng.choice(vocab, p=P[tuple(seq)])))
            index = sum(token * vocab ** (slots - position - 1) for position, token in enumerate(seq))
            counts[index] += 1
        stat, dof, value = base.chi_square(counts=counts, probs=torch.tensor(data=list(target.values())), n=m)
        pvalues.append(value)
        mean = out["accept_token_num"].double().mean().item()
        se = math.sqrt(max(second_moment - expected * expected, 0) / m)
        assert abs(mean - expected) <= 3 * se, (mean, expected, se)
        print(f"  B2 V={vocab} G={steps} mode={mode} paths={len(paths)} M={m} chi2={stat:.6f} dof={dof} "
              f"p={value:.6f} mean={mean:.6f} exact={expected:.6f} SE={se:.6f}", flush=True)
    stat, value = combine_pvalues(pvalues=pvalues, method="fisher")
    assert value > 0.01, value
    print(f"PASS B2 exact all-path target distribution: Fisher chi2={stat:.6f} p={value:.6f}; means within 3 SE", flush=True)


def test_b3():
    runs, rows = 0, 0
    for slots, kp, seed in itertools.product((2, 4, 8, 16), (16, 64), range(4)):
        batch = random_batch(slots=slots, kp=kp, seed=20261020 + seed, greedy=True)
        same(actual=verify(**batch), expected=verify(block_verify=False, **batch))
        runs += 1
        rows += batch["candidates"].shape[0]
    print(f"PASS B3 one-hot targets BV/token outputs identical: {runs} batches/{rows} rows", flush=True)


def test_b4():
    result = subprocess.run(args=[sys.executable, str(Path(__file__).with_name("check_integration_rs3.py"))],
                            text=True, check=True)
    assert result.returncode == 0
    print("PASS B4 runtime integration", flush=True)


SEQUENTIAL_BV = "fb1e77a05a"


def reference_h64(*, batch):
    probs = batch["p"].double().numpy()
    ids = batch["pi"].numpy()
    q = batch["q"].double().numpy()
    q = np.where(q == q, q, 0.0)
    qids = batch["qi"].numpy()
    candidates = batch["candidates"].numpy()
    bs, slots = candidates.shape
    path_pi = np.ones(shape=bs, dtype=np.float64)
    hs = np.zeros(shape=(bs, slots - 1), dtype=np.float64)
    for step in range(1, slots):
        active = path_pi > 0
        token = candidates[:, step, None]
        pt = np.where(ids[:, step - 1] == token, probs[:, step - 1], 0.0).sum(axis=1)
        qt = np.where(qids[:, step - 1] == token, q[:, step - 1], 0.0).sum(axis=1)
        ratio = np.divide(path_pi * pt, qt, out=np.ones_like(path_pi), where=qt != 0)
        path_pi = np.where(active & (pt > 0), np.minimum(ratio, 1.0), 0.0)
        h = path_pi.copy()
        if step < slots - 1:
            matched = np.where(ids[:, step, :, None] == qids[:, step, None, :],
                               q[:, step, None, :], 0.0).sum(axis=2)
            n = np.maximum(path_pi[:, None] * probs[:, step] - matched, 0.0).sum(axis=1)
            denominator = n + 1.0 - path_pi
            h = np.divide(n, denominator, out=np.ones_like(n), where=denominator > 0)
        hs[:, step - 1] = np.where(path_pi > 0, h, 0.0)
    return hs


def test_b6():
    source = subprocess.run(args=["git", "-C", str(base.WT), "show",
                                 SEQUENTIAL_BV + ":python/sglang/kernels/ops/speculative/sparse_rs.py"],
                            capture_output=True, text=True, check=True).stdout
    # Load only the CPU batch builder, without importing B5's GPU entry points.
    timing_source = Path(__file__).with_name("b5_timing.py").read_text()
    builder = next(node for node in ast.parse(timing_source).body
                   if isinstance(node, ast.FunctionDef) and node.name == "make_batch")
    namespace = {"torch": torch, "KP": 64, "K": 64, "VOCAB": 248320}
    exec(compile(ast.Module(body=[builder], type_ignores=[]), "b5_timing.make_batch", "exec"), namespace)
    total, near_total, mismatch_total = 0, 0, 0
    with tempfile.TemporaryDirectory(prefix="rs3-sequential-bv-") as directory:
        path = Path(directory) / "sequential_bv.py"
        path.write_text(source)
        sequential = rs2d.load_path(name="rs3_sequential_bv_reference", path=path)

        def compare(*, batch, label):
            nonlocal total, near_total, mismatch_total
            actual = verify(**batch)
            expected = verify(module=sequential, **batch)
            hs = reference_h64(batch=batch)
            coins = batch["coins"].double().numpy()[:, :hs.shape[1]]
            near = np.any(np.abs(coins - hs) <= 1e-5, axis=1)
            retrieve = batch["retrieve"]
            mismatch = ((actual["predicts"][retrieve] != expected["predicts"][retrieve]).any(dim=1)
                        | (actual["accept_index"] != expected["accept_index"]).any(dim=1)
                        | (actual["accept_token_num"] != expected["accept_token_num"])).numpy()
            bad = np.flatnonzero(mismatch & ~near)
            assert len(bad) == 0, (label, "non-threshold mismatch rows", bad.tolist())
            rows = len(near)
            total += rows
            near_total += int(near.sum())
            mismatch_total += int(mismatch.sum())
            print(f"  B6 {label}: rows={rows} near-threshold={int(near.sum())} "
                  f"mismatch={int(mismatch.sum())}", flush=True)

        for slots in (4, 8, 16):
            generator = torch.manual_seed(seed=20261200 + slots)
            batch = namespace["make_batch"](bs=512, slots=slots, generator=generator)
            compare(batch=dict(batch, vocab=248320), label=f"B5 slots={slots} kp=64 K=64")
            compare(batch=random_batch(slots=slots, kp=64, seed=20261300 + slots),
                    label=f"B1 kinds slots={slots} kp=64 K=64")
        edge = random_batch(slots=5, kp=16, seed=20261400)
        edge["q"] = edge["q"][:, :, :13].clone()
        edge["qi"] = edge["qi"][:, :, :13].clone()
        edge["p"][0].zero_()
        edge["p"][0, :, 0] = 1.0
        edge["candidates"][0, 1] = edge["pi"][0, 0, 1]
        edge["candidates"][0, 2:] = edge["pi"][0, 1:-1, 0]
        edge["q"][0].zero_()
        compare(batch=edge, label="masked slots=5 kp=16 K=13; zero then qt=0 reset")
    print(f"PASS B6 sequential BV={SEQUENTIAL_BV}: rows={total} near-threshold rows={near_total} "
          f"(coin within 1e-5 of float64 h), mismatch rows={mismatch_total}; "
          "all non-threshold outputs identical", flush=True)


if __name__ == "__main__":
    tests = {"B0": test_b0, "B1": test_b1, "B2": test_b2, "B3": test_b3, "B4": test_b4, "B6": test_b6}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", nargs="+", choices=list(tests), default=list(tests))
    args = parser.parse_args()
    print(f"RS3 CPU worktree={base.WT} G1={G1}", flush=True)
    suite_start = time.perf_counter()
    for name in args.parts:
        start = time.perf_counter()
        tests[name]()
        print(f"  {name} elapsed={time.perf_counter() - start:.1f}s", flush=True)
    print(f"PASS requested CPU tests={args.parts} elapsed={time.perf_counter() - suite_start:.1f}s", flush=True)
