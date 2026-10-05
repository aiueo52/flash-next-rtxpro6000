"""CPU interpreter correctness and opt-in CUDA RS2 graph benchmarks."""

import ast
import importlib.util
import itertools
import math
import os
from pathlib import Path
import statistics
import sys
import textwrap
import time
from typing import Optional

import torch
from scipy.stats import chi2, combine_pvalues

WT = Path(os.environ.get("WT", "/home/user/tools/sglang-rs2"))
DEV = os.environ.get("DEV", "cpu")
K = int(os.environ.get("RS_K", "64"))
if DEV not in ("cpu", "cuda"):
    raise ValueError("DEV must be cpu or cuda")
if DEV == "cpu":
    assert os.environ.get("TRITON_INTERPRET") == "1"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    torch.set_num_threads(4)


def load_module(*, name, relative_path, strip_flashinfer=False):
    path = WT / "python/sglang" / relative_path
    spec = importlib.util.spec_from_file_location(name=name, location=path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if strip_flashinfer:
        tree = ast.parse(path.read_text())
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.ImportFrom) and node.module == "flashinfer"
        )]
        exec(compile(tree, str(path), "exec"), module.__dict__)
    else:
        spec.loader.exec_module(module)
    return module


rs2 = load_module(name="rs2_sparse", relative_path="kernels/ops/speculative/sparse_rs.py")
rs1 = load_module(name="rs1_dense", relative_path="kernels/ops/speculative/reject_sampling.py")
sv = load_module(name="sv_target", relative_path="kernels/ops/speculative/sparse_verify.py",
                 strip_flashinfer=DEV == "cpu")
greedy = load_module(name="greedy_topk1", relative_path="kernels/ops/speculative/topk1.py")


def extract_function(*, relative_path, name, class_name=None):
    source = (WT / "python/sglang" / relative_path).read_text()
    nodes = ast.parse(source).body
    if class_name is not None:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef)
                     and node.name == class_name).body
    node = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name)
    return textwrap.dedent(ast.get_source_segment(source=source, node=node))


namespace = {"torch": torch, "Optional": Optional}
proposal_source = extract_function(relative_path="srt/speculative/spec_utils.py",
                                   name="sample_draft_proposal_truncated")
exec(proposal_source, namespace)
reference_proposal = namespace["sample_draft_proposal_truncated"]


def stable_topk(input, k, dim=-1):
    result = torch.sort(input=input, dim=dim, descending=True, stable=True)
    return result.values[..., :k], result.indices[..., :k]


# RS1 leaves tie order unspecified; only its top-k selector is canonicalized.
tie_namespace = {"torch": torch, "Optional": Optional, "stable_topk": stable_topk}
exec(proposal_source.replace("torch.topk(", "stable_topk("), tie_namespace)
tie_reference = tie_namespace["sample_draft_proposal_truncated"]


def params(*, temperatures, top_ks, top_ps, min_ps=None):
    return dict(
        temperatures=torch.tensor(data=temperatures, dtype=torch.float32, device=DEV),
        top_ks=torch.tensor(data=top_ks, dtype=torch.int32, device=DEV),
        top_ps=torch.tensor(data=top_ps, dtype=torch.float32, device=DEV),
        min_ps=None if min_ps is None else torch.tensor(data=min_ps, dtype=torch.float32, device=DEV),
    )


def proposal(*, logits, sampling, uniforms, hot=None, impl="triton", **outputs):
    return rs2.rs_draft_proposal_sparse(
        next_token_logits=logits, uniforms=uniforms, hot_token_id=hot,
        k=K, topk_impl=impl, **sampling, **outputs,
    )


def compare_proposal(*, logits, sampling, hot, impl, tied):
    ref = tie_reference if tied else reference_proposal
    q_ref, ids_ref, _, _ = ref(
        next_token_logits=logits, temperatures=sampling["temperatures"].reshape(-1, 1),
        top_ks=sampling["top_ks"], top_ps=sampling["top_ps"],
        min_ps=sampling["min_ps"], k_cap=K,
    )
    selector = stable_topk if tied else torch.topk
    top_logits, _ = selector(input=logits.float(), k=q_ref.shape[1], dim=-1)
    weights = torch.exp(input=(top_logits - top_logits[:, :1]) /
                        sampling["temperatures"].reshape(-1, 1))
    ranks = torch.arange(end=q_ref.shape[1], device=weights.device)
    top_ks = sampling["top_ks"].unsqueeze(dim=1)
    weights = weights * ((ranks < top_ks) | (top_ks <= 0))
    pre_probs = weights / weights.sum(dim=-1, keepdim=True)
    # Locate rounding-sensitive cuts using the reference's fp32 probabilities.
    exclusive = torch.cumsum(input=pre_probs.double(), dim=-1) - pre_probs.double()
    ambiguous = (exclusive - sampling["top_ps"].unsqueeze(dim=1)).abs() <= 1e-6
    if sampling["min_ps"] is not None:
        threshold = pre_probs[:, :1] * sampling["min_ps"].unsqueeze(dim=1)
        ambiguous |= (pre_probs - threshold).abs() <= 1e-6 * pre_probs[:, :1]
    ambiguous[:, 0] = False
    uniforms = torch.rand(size=(logits.shape[0],), device=DEV)
    q, tokens, qx, draws = proposal(logits=logits, sampling=sampling, uniforms=uniforms,
                                  hot=hot, impl=impl)
    if hot is not None:
        ids_ref = hot[ids_ref]
    sentinel = hot.max().item() + 1 if hot is not None else logits.shape[1]
    ref_kept = torch.where(condition=q_ref > 0, input=ids_ref, other=sentinel)
    kept = torch.where(condition=q > 0, input=tokens, other=sentinel)
    ref_order = torch.argsort(input=ref_kept, dim=-1)
    order = torch.argsort(input=kept, dim=-1)
    same = ref_kept.gather(dim=1, index=ref_order) == kept.gather(dim=1, index=order)
    matches = ids_ref.unsqueeze(dim=2) == tokens.unsqueeze(dim=1)
    impl_kept = (matches & (q > 0).unsqueeze(dim=1)).any(dim=2)
    different = (q_ref > 0) != impl_kept
    valid = (~different | ambiguous).all(dim=1)
    valid &= ((q <= 0) | matches.any(dim=1)).all(dim=1)
    assert valid.all(), (impl, tied, sampling, torch.where(condition=~valid)[0])
    boundary_flips = (~same.all(dim=1)).sum().item()
    # Renormalize on the implementation's support, then align by token id.
    expected = pre_probs * impl_kept
    expected = expected / expected.sum(dim=-1, keepdim=True)
    expected = (expected.unsqueeze(dim=2) * matches).sum(dim=1) * (q > 0)
    delta = (q - expected).abs().max().item()
    assert delta < 2e-6, delta
    assert torch.all((q.sum(dim=-1) - 1).abs() <= 1e-5)
    assert torch.all(q >= 0) and torch.isfinite(q).all()
    assert torch.allclose(input=qx, other=q_ref.new_tensor(0) + q.gather(
        dim=1, index=(tokens == (hot[draws] if hot is not None else draws)).long().argmax(dim=1, keepdim=True)),
        atol=2e-6, rtol=0)
    return delta, boundary_flips


def test_a():
    combinations = list(itertools.product((0.3, 0.8, 1.5), (1, 5, 40, 0), (1.0, 0.95, 0.5)))
    rows_checked = 0
    boundary_flips = 0
    for impl in ("triton", "torch"):
        max_delta = 0.0
        for vocab, tied in ((117, False), (117, True), (49152, False), (49152, True)):
            rows = combinations if vocab == 117 else [(0.8, 40, 0.95)]
            base = torch.randn(size=(len(rows), vocab), device=DEV)
            if tied:
                base = base.bfloat16()
                base[:, :80] = 8.0
            # A sliced storage row catches accidental use of vocab as the row stride.
            storage = torch.empty(size=(len(rows), vocab + 11), dtype=base.dtype, device=DEV)
            storage[:, :vocab] = base
            logits = storage[:, :vocab]
            hot = torch.randperm(n=vocab + 37, device=DEV)[:vocab].contiguous()
            for min_p in ((None, 0.05, 0.2) if vocab == 117 else (0.05,)):
                sampling = params(temperatures=[r[0] for r in rows], top_ks=[r[1] for r in rows],
                                  top_ps=[r[2] for r in rows],
                                  min_ps=None if min_p is None else [min_p] * len(rows))
                for mapping in ((None, hot) if vocab == 117 else (hot,)):
                    delta, flips = compare_proposal(
                        logits=logits, sampling=sampling, hot=mapping, impl=impl, tied=tied)
                    max_delta = max(max_delta, delta)
                    boundary_flips += flips
                    rows_checked += len(rows)
            print(f"  A topk_impl={impl} Vd={vocab} bf16_ties={tied} checked", flush=True)
        sampling = params(temperatures=[0, 0, 0.8, 1.5], top_ks=[0, 0, 5, 40],
                          top_ps=[0, 1.2, 0.95, 0.5], min_ps=[0.05, 0, 0.2, 0.05])
        logits = torch.randn(size=(4, 83), device=DEV)
        logits[0] = float("nan")
        logits[1] = -float("inf")
        logits[2, 0] = float("nan")
        q, tokens, _, draws = proposal(logits=logits, sampling=sampling,
                                      uniforms=torch.tensor(data=[0, 1, 0.4, 0.99], device=DEV), impl=impl)
        assert torch.isfinite(q).all() and torch.all((q.sum(dim=-1) - 1).abs() < 1e-5)
        assert torch.all((tokens >= 0) & (tokens < 83)) and torch.all((draws >= 0) & (draws < 83))
        for vocab in (7, 65):
            q, tokens, _, _ = proposal(logits=torch.randn(size=(1, vocab), device=DEV),
                                      sampling=params(temperatures=[0.8], top_ks=[0], top_ps=[1]),
                                      uniforms=torch.zeros(size=(1,), device=DEV), impl=impl)
            assert q.shape == tokens.shape == (1, min(vocab, K))
        print(f"PASS A topk_impl={impl}: mixed rows, hot on/off, fp32/bf16 boundary ties, "
              f"Vd=117/49152, fillers/NaNs/-inf, small Vd; max delta={max_delta:.3g}", flush=True)
    print(f"  A compared {rows_checked} proposal rows; boundary flips: {boundary_flips} rows; "
          "bf16 ties use the canonical RS1 selector", flush=True)


def chi_square(*, counts, probs, n):
    counts = counts.double().cpu()
    probs = probs.double().cpu()
    probs = probs / probs.sum()
    assert counts[probs == 0].sum() == 0
    expected = probs * n
    big = expected >= 5
    observed = counts[big]
    expected_big = expected[big]
    tail_expected = expected[~big].sum()
    if tail_expected >= 5:
        observed = torch.cat(tensors=[observed, counts[~big].sum().reshape(1)])
        expected_big = torch.cat(tensors=[expected_big, tail_expected.reshape(1)])
    dof = len(observed) - 1
    if dof < 1:
        return 0.0, dof, 1.0
    stat = ((observed - expected_big).square() / expected_big).sum().item()
    return stat, dof, float(chi2.sf(x=stat, df=dof))


def test_b():
    m = int(os.environ.get("DRAW_M", "2048" if DEV == "cpu" else "40000"))
    row_logits = torch.tensor(data=[[2.0, 1.7, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0],
                               [1.0, 0.5, 0.25, 0.0, -0.5, -1.0, -1.5, -2.0]], device=DEV)
    logits = row_logits.repeat_interleave(repeats=m, dim=0)
    sampling = params(temperatures=[0.8] * m + [1.5] * m, top_ks=[5] * m + [0] * m,
                      top_ps=[0.95] * m + [1] * m, min_ps=[0.05] * m + [0.2] * m)
    hot = torch.tensor(data=[17, 3, 13, 6, 9, 1, 20, 11], dtype=torch.int64, device=DEV)
    uniforms = torch.rand(size=(2 * m,), device=DEV)
    for impl in ("triton", "torch"):
        positions = torch.arange(2 * m, device=DEV, dtype=torch.int64)
        old_positions = positions.clone()
        chain = torch.full(size=(2 * m, 4), fill_value=-1, dtype=torch.int64, device=DEV)
        support_q = torch.full(size=(2 * m, 4, 8), fill_value=-7.0, device=DEV)
        support_tokens = torch.full(size=(2 * m, 4, 8), fill_value=-1, dtype=torch.int64, device=DEV)
        q, tokens, qx, draws = proposal(
            logits=logits, sampling=sampling, uniforms=uniforms, hot=hot, impl=impl,
            positions=positions, draft_tokens=chain, draft_token_column=2,
            draft_support_probs=support_q[:, 2], draft_support_tokens=support_tokens[:, 2],
        )
        assert torch.equal(input=positions, other=old_positions + 1)
        assert torch.equal(input=chain[:, 2:3], other=draws)
        assert torch.all(chain[:, [0, 1, 3]] == -1)
        assert torch.all(support_q[:, [0, 1, 3]] == -7)
        assert torch.equal(input=support_tokens[:, 2], other=tokens)
        hits = tokens == draws
        assert torch.all(hits.sum(dim=1) == 1)
        selected_probs = (q * hits).sum(dim=1, keepdim=True)
        assert torch.equal(input=qx, other=selected_probs) and torch.all(qx > 0)
        pvalues = []
        for row in range(2):
            start, end = row * m, (row + 1) * m
            counts = torch.bincount(input=draws[start:end, 0], minlength=21)
            distribution = torch.zeros(size=(21,), device=DEV).scatter_(dim=0, index=tokens[start], src=q[start])
            stat, dof, pvalue = chi_square(counts=counts, probs=distribution, n=m)
            assert pvalue > 1e-3, (stat, dof, pvalue)
            pvalues.append(pvalue)
        tie_logits = torch.full(size=(5, 117), fill_value=-3.0, device=DEV, dtype=torch.bfloat16)
        tie_logits[:, [2, 25, 114]] = 4.0
        tie_hot = torch.randperm(n=117, device=DEV).long()
        tie_sampling = params(temperatures=[0, 0.3, 0.8, 1.5, 0.8], top_ks=[1] * 5,
                              top_ps=[0, 0.5, 0.95, 1, 1])
        _, _, p, x = proposal(logits=tie_logits, sampling=tie_sampling,
                             uniforms=torch.rand(size=(5,), device=DEV), hot=tie_hot, impl=impl)
        gp, gx = greedy.draft_topk1_postprocess(
            next_token_logits=tie_logits, positions=torch.zeros(size=(5,), device=DEV, dtype=torch.int64))
        assert torch.equal(input=x, other=gx) and torch.all(x == 2) and torch.equal(input=p, other=gp)
        chain = torch.empty(size=(5, 1), device=DEV, dtype=torch.int64)
        _, _, _, mapped = proposal(logits=tie_logits, sampling=tie_sampling,
                                  uniforms=torch.rand(size=(5,), device=DEV), hot=tie_hot,
                                  impl=impl, draft_tokens=chain)
        assert torch.equal(input=mapped, other=tie_hot[gx]) and torch.equal(input=chain, other=mapped)
        print(f"PASS B topk_impl={impl}: draw M={m}/row chi2 p={pvalues}; "
              "q(X), mapped chain, in-place support, positions and greedy ties", flush=True)


def dense_target_probs(*, logits, sampling, num_slots, apply_min_p):
    repeat = lambda tensor: tensor.repeat_interleave(repeats=num_slots, dim=0)
    temperatures = repeat(sampling["temperatures"]).reshape(-1, 1)
    top_ks = repeat(sampling["top_ks"])
    top_ps = repeat(sampling["top_ps"])
    p = torch.softmax(input=logits.float() / temperatures, dim=-1)
    if DEV == "cuda":
        from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob

        p = top_k_renorm_prob(probs=p, top_k=top_ks)
        p = top_p_renorm_prob(probs=p, top_p=top_ps)
    else:
        ordered = p.sort(dim=-1, descending=True).values
        pivot = ordered.gather(dim=1, index=(top_ks.clamp(min=1, max=p.shape[1]) - 1).long().reshape(-1, 1))
        p = torch.where(condition=p >= pivot, input=p, other=0.0)
        p = p / p.sum(dim=-1, keepdim=True)
        ordered = p.sort(dim=-1, descending=True).values
        threshold = torch.where(condition=ordered.cumsum(dim=-1) >= top_ps[:, None] * p.sum(dim=-1, keepdim=True),
                                input=ordered, other=0.0).amax(dim=-1, keepdim=True)
        p = torch.where(condition=p >= threshold, input=p, other=0.0)
        p = p / p.sum(dim=-1, keepdim=True)
    if apply_min_p:
        min_ps = repeat(sampling["min_ps"]).reshape(-1, 1)
        p = torch.where(condition=p >= p.amax(dim=-1, keepdim=True) * min_ps, input=p, other=0.0)
        p = p / p.sum(dim=-1, keepdim=True)
    return p


def sparse_target(*, logits, sampling, num_slots, apply_min_p, kp=None):
    if kp is None:
        kp = sv.sparse_verify_width(int(sampling["top_ks"].max().item()))
    p, pi = sv.sparse_target_probs(
        logits=logits, temperatures=sampling["temperatures"], top_ks=sampling["top_ks"],
        top_ps=sampling["top_ps"], min_ps=sampling["min_ps"], num_draft_tokens=num_slots,
        kp=kp, apply_top_p=True, apply_min_p=apply_min_p, use_flashinfer_topk=False,
    )
    return p.reshape(-1, num_slots, kp), pi.reshape(-1, num_slots, kp)


def verify_outputs(*, bs, slots):
    return dict(
        predicts=torch.full(size=(bs * slots,), fill_value=-1, dtype=torch.int32, device=DEV),
        accept_index=torch.full(size=(bs, slots), fill_value=-1, dtype=torch.int32, device=DEV),
        accept_token_num=torch.empty(size=(bs,), dtype=torch.int32, device=DEV),
    )


def dense_verify(*, outputs, candidates, retrieve, coins, coins_final, p, q):
    rs1.chain_speculative_sampling_triton(
        **outputs, candidates=candidates, retrive_index=retrieve, retrive_next_token=None,
        retrive_next_sibling=None, uniform_samples=coins,
        uniform_samples_for_final_sampling=coins_final, target_probs=p, draft_probs=q,
        threshold_single=1.0, threshold_acc=1.0, deterministic=True,
    )


def sparse_verify(*, outputs, candidates, retrieve, coins, coins_final, p, pi, q, qi, vocab):
    rs2.chain_speculative_sampling_sparse(
        **outputs, candidates=candidates, retrive_index=retrieve, uniform_samples=coins,
        uniform_samples_for_final_sampling=coins_final, target_probs=p, target_index=pi,
        draft_support_probs=q, draft_support_tokens=qi, vocab_size=vocab,
    )


def test_c():
    cases = ("all_accept", "reject_first", "greedy", "outside_p", "q_outside", "min_p")
    mismatches = 0
    runs = 0
    rows = 0
    vocab = 193
    for steps, bs, case in itertools.product((1, 4, 7, 15), (1, 4), cases):
        slots = steps + 1
        logits = torch.randn(size=(bs * slots, vocab), device=DEV) * 1.5
        logits[:, -1] = -float("inf")
        sampling = params(temperatures=[0.8] * bs, top_ks=[1 if case == "greedy" else 12] * bs,
                          top_ps=[1 if case == "greedy" else 0.95] * bs, min_ps=[0.2] * bs)
        apply_min_p = case == "min_p" or (bs == 4 and case == "q_outside")
        p, pi = sparse_target(logits=logits, sampling=sampling, num_slots=slots, apply_min_p=apply_min_p)
        dp = dense_target_probs(logits=logits, sampling=sampling, num_slots=slots,
                                apply_min_p=apply_min_p).reshape(bs, slots, vocab)
        assert (dp.gather(dim=2, index=pi) - p).abs().max() < 2e-6
        qi = pi[:, :-1].contiguous().clone()
        q = p[:, :-1].contiguous().clone()
        candidates = torch.zeros(size=(bs, slots), dtype=torch.int64, device=DEV)
        candidates[:, 1:] = qi[:, :, 0]
        coins = torch.rand(size=(bs, slots), device=DEV)
        coins_final = torch.rand(size=(bs,), device=DEV)
        if case in ("all_accept", "greedy"):
            coins.zero_()
        if case == "reject_first":
            coins.fill_(0.99)
            q.zero_()
            q[:, :, 0] = 1
        if case in ("outside_p", "q_outside"):
            qi[:, :, -1] = vocab - 1
            q *= 0.3
            q[:, :, -1] = 0.7
            if case == "outside_p":
                candidates[:, 1] = vocab - 1
        dq = torch.zeros(size=(bs, steps, vocab), device=DEV).scatter_(dim=2, index=qi, src=q)
        retrieve = torch.randperm(n=bs * slots, device=DEV).reshape(bs, slots).long()
        dense_out = verify_outputs(bs=bs, slots=slots)
        sparse_out = verify_outputs(bs=bs, slots=slots)
        dense_verify(outputs=dense_out, candidates=candidates, retrieve=retrieve, coins=coins,
                     coins_final=coins_final, p=dp, q=dq)
        sparse_verify(outputs=sparse_out, candidates=candidates, retrieve=retrieve, coins=coins,
                      coins_final=coins_final, p=p, pi=pi, q=q, qi=qi, vocab=vocab)
        assert torch.equal(input=dense_out["accept_token_num"], other=sparse_out["accept_token_num"])
        assert torch.equal(input=dense_out["accept_index"], other=sparse_out["accept_index"])
        if case in ("all_accept", "greedy"):
            assert torch.all(sparse_out["accept_token_num"] == steps)
        if case in ("outside_p", "reject_first"):
            assert torch.all(sparse_out["accept_token_num"] == 0)
        row_mismatches = 0
        for b in range(bs):
            count = int(dense_out["accept_token_num"][b])
            path = dense_out["accept_index"][b, :count + 1].long()
            different = dense_out["predicts"][path] != sparse_out["predicts"][path]
            if different.any():
                assert not different[:-1].any(), "A draft token differs"
                w = dp[b, count] if count == steps else (dp[b, count] - dq[b, count]).clamp_min(0)
                u = coins_final[b] * w.sum()
                edge_distance = (w.cumsum(dim=0) - u).abs().min().item()
                assert edge_distance <= 2e-6, (case, edge_distance)
                row_mismatches += 1
                print(f"  C rounding edge S={steps} bs={bs} case={case} row={b} distance={edge_distance:.3g}")
        mismatches += row_mismatches
        runs += 1
        rows += bs
    assert runs == 48 and mismatches <= 1
    verify_fallbacks()
    print(f"PASS C real speculative_sampling_classic_kernel: {runs} cases/{rows} rows, "
          f"rounding-edge mismatch rows={mismatches} across 48 cases/{rows} rows (limit 1); S=1/4/7/15 bs=1/4", flush=True)


def verify_fallbacks():
    p = torch.tensor(data=[[[0.4, 0.6], [0.7, 0.3]]], device=DEV)
    pi = torch.tensor(data=[[[2, 5], [2, 5]]], device=DEV, dtype=torch.int64)
    q = p[:, :1].clone()
    qi = pi[:, :1].clone()
    c = torch.tensor(data=[[0, 2]], device=DEV, dtype=torch.int64)
    retrieve = torch.tensor(data=[[0, 1]], device=DEV, dtype=torch.int64)
    for final_coin, expected in ((0.1, 2), (1.0, 5)):
        out = verify_outputs(bs=1, slots=2)
        sparse_verify(outputs=out, candidates=c, retrieve=retrieve,
                      coins=torch.ones(size=(1, 2), device=DEV),
                      coins_final=torch.tensor(data=[final_coin], device=DEV),
                      p=p, pi=pi, q=q, qi=qi, vocab=8)
        assert out["accept_token_num"].item() == 0 and out["predicts"][0].item() == expected
    q[:] = float("nan")
    out = verify_outputs(bs=1, slots=2)
    sparse_verify(outputs=out, candidates=c, retrieve=retrieve,
                  coins=torch.ones(size=(1, 2), device=DEV), coins_final=torch.tensor(data=[0.1], device=DEV),
                  p=p, pi=pi, q=q, qi=qi, vocab=8)
    assert out["accept_token_num"].item() == 1
    print("  C sparse robustness: zero residual draws P (deliberate dense difference), u=1 fallback, NaN q", flush=True)


def test_d():
    m = int(os.environ.get("LOSSLESS_M", "4096" if DEV == "cpu" else "80000"))
    steps, vocab, hot_size = 3, 64, 24
    slots = steps + 1
    target_logits = torch.randn(size=(slots, vocab), device=DEV) * 0.4
    target_logits[:, :hot_size] += 2.0
    hot = torch.randperm(n=hot_size, device=DEV).long()
    draft_logits = target_logits[:steps, hot] + torch.randn(size=(steps, hot_size), device=DEV) * 0.25
    sampling = params(temperatures=[0.8], top_ks=[8], top_ps=[0.95], min_ps=[0.05])
    p, pi = sparse_target(logits=target_logits, sampling=sampling, num_slots=slots, apply_min_p=True)
    dp = dense_target_probs(logits=target_logits, sampling=sampling, num_slots=slots, apply_min_p=True)
    q = torch.empty(size=(m, steps, min(K, hot_size)), device=DEV)
    qi = torch.empty(size=(m, steps, min(K, hot_size)), device=DEV, dtype=torch.int64)
    candidates = torch.zeros(size=(m, slots), dtype=torch.int64, device=DEV)
    mixed = params(temperatures=[0.8] * m, top_ks=[8] * m, top_ps=[0.95] * m, min_ps=[0.05] * m)
    for step in range(steps):
        proposal(logits=draft_logits[step].expand(m, hot_size).contiguous(), sampling=mixed,
                 uniforms=torch.rand(size=(m,), device=DEV), hot=hot,
                 draft_tokens=candidates, draft_token_column=step + 1,
                 draft_support_probs=q[:, step], draft_support_tokens=qi[:, step])
    out = verify_outputs(bs=m, slots=slots)
    retrieve = torch.arange(m * slots, device=DEV, dtype=torch.int64).reshape(m, slots)
    sparse_verify(outputs=out, candidates=candidates, retrieve=retrieve,
                  coins=torch.rand(size=(m, slots), device=DEV), coins_final=torch.rand(size=(m,), device=DEV),
                  p=p.repeat(m, 1, 1), pi=pi.repeat(m, 1, 1), q=q, qi=qi, vocab=vocab)
    pvalues = []
    for position in range(slots):
        mask = out["accept_token_num"] >= position
        draws = out["predicts"][out["accept_index"][mask, position].long()].long()
        n = int(mask.sum().item())
        assert n >= 100, n
        counts = torch.bincount(input=draws, minlength=vocab)
        stat, dof, pvalue = chi_square(counts=counts, probs=dp[position], n=n)
        assert pvalue > 1e-3, (position, n, stat, dof, pvalue)
        pvalues.append(pvalue)
        print(f"  D pos={position} n={n} chi2={stat:.4f} dof={dof} p={pvalue:.6f}", flush=True)
    fisher_stat, fisher_p = combine_pvalues(pvalues=pvalues, method="fisher")
    assert fisher_p > 1e-3, fisher_p
    dense_q0 = torch.zeros(size=(vocab,), device=DEV).scatter_(dim=0, index=qi[0, 0], src=q[0, 0])
    theory = torch.minimum(input=dp[0], other=dense_q0).sum().item()
    empirical = (out["accept_token_num"] >= 1).double().mean().item()
    standard_error = math.sqrt(theory * (1 - theory) / m)
    assert abs(empirical - theory) <= 6 * standard_error + 1 / m
    print(f"PASS D losslessness: Fisher chi2={fisher_stat:.6f} p={fisher_p:.6f}; "
          f"accept@0 empirical={empirical:.6f} sum(min(p,q))={theory:.6f} "
          f"SE={standard_error:.6f} M={m}", flush=True)


def test_d2():
    m = int(os.environ.get("LOSSLESS_M", "4096" if DEV == "cpu" else "80000"))
    steps, vocab, hot_size = 3, 64, 24
    slots = steps + 1
    target_logits = torch.randn(size=(slots, vocab), device=DEV) * 0.4
    target_logits[:, :hot_size] += 2.0
    hot = torch.randperm(n=hot_size, device=DEV).long()
    draft_logits = target_logits[:steps, hot] + torch.randn(size=(steps, hot_size), device=DEV) * 0.25
    # Target helpers require an explicit full-vocab top-k and a min-p tensor.
    sampling = params(temperatures=[0.8], top_ks=[vocab], top_ps=[0.95], min_ps=[0])
    p, pi = sparse_target(logits=target_logits, sampling=sampling, num_slots=slots,
                          apply_min_p=False, kp=vocab)
    dp = dense_target_probs(logits=target_logits, sampling=sampling, num_slots=slots, apply_min_p=False)
    assert (dp.gather(dim=1, index=pi[0]) - p[0]).abs().max() < 2e-6
    q = torch.empty(size=(m, steps, min(K, hot_size)), device=DEV)
    qi = torch.empty(size=(m, steps, min(K, hot_size)), device=DEV, dtype=torch.int64)
    candidates = torch.zeros(size=(m, slots), dtype=torch.int64, device=DEV)
    mixed = params(temperatures=[0.8] * m, top_ks=[0] * m, top_ps=[0.95] * m, min_ps=None)
    for step in range(steps):
        proposal(logits=draft_logits[step].expand(m, hot_size).contiguous(), sampling=mixed,
                 uniforms=torch.rand(size=(m,), device=DEV), hot=hot,
                 draft_tokens=candidates, draft_token_column=step + 1,
                 draft_support_probs=q[:, step], draft_support_tokens=qi[:, step])
    assert torch.all((q > 0).sum(dim=-1) <= min(K, hot_size))
    assert torch.all((q.sum(dim=-1) - 1).abs() <= 1e-5)
    print(f"  D2 K={K} support <= {min(K, hot_size)} ranks; q sums to 1", flush=True)
    retrieve = torch.arange(m * slots, device=DEV, dtype=torch.int64).reshape(m, slots)
    for block_verify in (False, True):
        out = verify_outputs(bs=m, slots=slots)
        coins = torch.rand(size=(m, slots), device=DEV)
        coins_final = torch.rand(size=(m,), device=DEV)
        mode = "block" if block_verify else "token"
        if block_verify:
            rs2.chain_speculative_sampling_sparse(
                **out, candidates=candidates, retrive_index=retrieve, uniform_samples=coins,
                uniform_samples_for_final_sampling=coins_final, target_probs=p.repeat(m, 1, 1),
                target_index=pi.repeat(m, 1, 1), draft_support_probs=q, draft_support_tokens=qi,
                vocab_size=vocab, block_verify=True,
            )
        else:
            sparse_verify(outputs=out, candidates=candidates, retrieve=retrieve,
                          coins=coins, coins_final=coins_final,
                          p=p.repeat(m, 1, 1), pi=pi.repeat(m, 1, 1), q=q, qi=qi, vocab=vocab)
        pvalues = []
        for position in range(slots):
            mask = out["accept_token_num"] >= position
            draws = out["predicts"][out["accept_index"][mask, position].long()].long()
            n = int(mask.sum().item())
            assert n >= 100, n
            counts = torch.bincount(input=draws, minlength=vocab)
            stat, dof, pvalue = chi_square(counts=counts, probs=dp[position], n=n)
            assert pvalue > 1e-3, (mode, position, n, stat, dof, pvalue)
            pvalues.append(pvalue)
            print(f"  D2 K={K} {mode} pos={position} n={n} chi2={stat:.4f} "
                  f"dof={dof} p={pvalue:.6f}", flush=True)
        fisher_stat, fisher_p = combine_pvalues(pvalues=pvalues, method="fisher")
        assert fisher_p > 1e-3, (mode, fisher_p)
        if not block_verify:
            dense_q0 = torch.zeros(size=(vocab,), device=DEV).scatter_(dim=0, index=qi[0, 0], src=q[0, 0])
            theory = torch.minimum(input=dp[0], other=dense_q0).sum().item()
            empirical = (out["accept_token_num"] >= 1).double().mean().item()
            standard_error = math.sqrt(theory * (1 - theory) / m)
            assert abs(empirical - theory) <= 6 * standard_error + 1 / m
            print(f"PASS D2 K={K} {mode} losslessness: Fisher chi2={fisher_stat:.6f} p={fisher_p:.6f}; "
                  f"accept@0 empirical={empirical:.6f} sum(min(p,q))={theory:.6f} "
                  f"SE={standard_error:.6f} M={m}", flush=True)
        else:
            print(f"PASS D2 K={K} {mode} losslessness: Fisher chi2={fisher_stat:.6f} "
                  f"p={fisher_p:.6f} M={m}", flush=True)


def graph_time(*, operation, calls, repeats=100):
    for _ in range(3):
        operation()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            operation()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    gpu_times, wall_times = [], []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        wall_times.append((time.perf_counter() - wall_start) * 1e6 / (calls * repeats))
        gpu_times.append(start.elapsed_time(end) * 1000 / (calls * repeats))
    return statistics.median(gpu_times), statistics.median(wall_times)


def benchmark_proposals():
    results = {}
    for dtype, n in itertools.product((torch.float32, torch.bfloat16), (1, 2, 4)):
        vocab, target_vocab = 49152, 248320
        logits = torch.randn(size=(n, vocab), dtype=dtype, device=DEV)
        sampling = params(temperatures=[0.8] * n, top_ks=[40] * n,
                          top_ps=[0.95] * n, min_ps=[0.05] * n)
        hot = torch.randperm(n=target_vocab, device=DEV)[:vocab].long()
        uniforms = torch.rand(size=(n,), device=DEV)
        positions = torch.zeros(size=(n,), dtype=torch.int64, device=DEV)
        chain = torch.empty(size=(n, 14), dtype=torch.int64, device=DEV)

        def baseline():
            q, ids, _, _ = reference_proposal(
                next_token_logits=logits, temperatures=sampling["temperatures"][:, None],
                top_ks=sampling["top_ks"], top_ps=sampling["top_ps"],
                min_ps=sampling["min_ps"], k_cap=64,
            )
            torch.zeros(size=(n, target_vocab), device=DEV).scatter_(dim=1, index=hot[ids], src=q)

        operations = {
            "triton": lambda: proposal(logits=logits, sampling=sampling, uniforms=uniforms,
                                       hot=hot, impl="triton", positions=positions, draft_tokens=chain),
            "torch": lambda: proposal(logits=logits, sampling=sampling, uniforms=uniforms,
                                      hot=hot, impl="torch", positions=positions, draft_tokens=chain),
            "RS1+scatter": baseline,
            "greedy": lambda: greedy.draft_topk1_postprocess(
                next_token_logits=logits, positions=positions, hot_token_id=hot, draft_tokens=chain),
        }
        for name, operation in operations.items():
            gpu_us, wall_us = graph_time(operation=operation, calls=14)
            results[(dtype, n, name)] = gpu_us
            print(f"  E proposal dtype={dtype} n={n} Vd={vocab} graph_calls=14 {name:12s} "
                  f"GPU={gpu_us:.3f} us/call wall={wall_us:.3f} us/call", flush=True)
    for dtype in (torch.float32, torch.bfloat16):
        assert results[(dtype, 1, "triton")] <= 15, "RS2 exceeds the 15 us budget"
        for n in (1, 4):
            assert results[(dtype, n, "triton")] < results[(dtype, n, "RS1+scatter")], "RS2 is slower than RS1"


def benchmark_verify():
    for bs, slots in ((1, 4), (1, 8), (1, 16), (4, 4)):
        vocab = 248320
        logits = torch.randn(size=(bs * slots, vocab), device=DEV)
        sampling = params(temperatures=[0.8] * bs, top_ks=[40] * bs,
                          top_ps=[0.95] * bs, min_ps=[0.05] * bs)
        p, pi = sparse_target(logits=logits, sampling=sampling, num_slots=slots, apply_min_p=True)
        q, qi = p[:, :-1].contiguous(), pi[:, :-1].contiguous()
        dq = torch.zeros(size=(bs, slots - 1, vocab), device=DEV).scatter_(dim=2, index=qi, src=q)
        candidates = torch.cat(tensors=[torch.zeros(size=(bs, 1), device=DEV, dtype=torch.int64), qi[:, :, 0]], dim=1)
        retrieve = torch.arange(bs * slots, device=DEV, dtype=torch.int64).reshape(bs, slots)
        coins, coins_final = torch.rand(size=(bs, slots), device=DEV), torch.rand(size=(bs,), device=DEV)
        out = verify_outputs(bs=bs, slots=slots)
        next_tokens = torch.arange(1, slots + 1, device=DEV, dtype=torch.int64).repeat(bs, 1)
        next_tokens[:, -1] = -1
        siblings = torch.full_like(input=next_tokens, fill_value=-1)

        def operation(*, implementation):
            if implementation == "dense RS":
                target = dense_target_probs(logits=logits, sampling=sampling,
                                            num_slots=slots, apply_min_p=True).reshape(bs, slots, vocab)
                dense_verify(outputs=out, candidates=candidates, retrieve=retrieve, coins=coins,
                             coins_final=coins_final, p=target, q=dq)
                return
            target, indices = sparse_target(logits=logits, sampling=sampling,
                                            num_slots=slots, apply_min_p=True, kp=64)
            if implementation == "RS2":
                sparse_verify(outputs=out, candidates=candidates, retrieve=retrieve, coins=coins,
                              coins_final=coins_final, p=target, pi=indices, q=q, qi=qi, vocab=vocab)
            else:
                sv.tree_speculative_sampling_target_only_sparse(
                    **out, candidates=candidates, retrive_index=retrieve,
                    retrive_next_token=next_tokens, retrive_next_sibling=siblings,
                    uniform_samples=coins, uniform_samples_for_final_sampling=coins_final,
                    target_probs=target, target_index=indices, vocab_size=vocab,
                )
        for name in ("dense RS", "RS2", "SV1 target-only"):
            gpu_us, wall_us = graph_time(operation=lambda: operation(implementation=name), calls=1)
            print(f"  E verify bs={bs} slots={slots} {name:16s} "
                  f"GPU={gpu_us:.3f} us/verify wall={wall_us:.3f} us/verify", flush=True)


def test_e():
    if DEV == "cpu":
        print("SKIP E: CPU mode; CUDA graphs and GPU/wall microbenchmarks require a later GPU run", flush=True)
        return
    benchmark_proposals()
    benchmark_verify()
    print("PASS E: graph benchmarks complete; triton n=1 meets 15 us and n=1/4 beat RS1", flush=True)


if __name__ == "__main__":
    torch.manual_seed(20261001)
    print(f"RS2 device={DEV} worktree={WT} seed=20261001", flush=True)
    for name, test in (("A", test_a), ("B", test_b), ("C", test_c), ("D", test_d), ("D2", test_d2), ("E", test_e)):
        try:
            test()
        except Exception as error:
            print(f"FAIL {name}: {type(error).__name__}: {error}", flush=True)
            raise
    print("SKIP F: server smoke/profile and greedy exactness are parent-run checks", flush=True)
