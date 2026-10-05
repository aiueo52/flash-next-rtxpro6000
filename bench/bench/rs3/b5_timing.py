"""B5 (RS3 spec section 4): the sparse chain verify on the GPU with block verification off and on.

Usage: WT=<RS3 worktree> G1_WT=<G1 worktree> python b5_timing.py   (needs the GPU; run under ~/.gpu.lock)

Production shapes: kp 64 (lmstudio top_k 40), K 64, vocab 248320, slots 4/8/16 (chains of 3/7/15), bs 1 and 4.
Draft tokens are drawn from q, and p overlaps q, so the token-level loop stops about where it would in production.
1. Same: block_verify False against the G1 worktree's kernel on the same inputs (bit-identical outputs).
2. Same BV: block_verify True against the sequential BV kernel of fb1e77a05a (B6 on the GPU); a row may differ
   only where a coin lies within 1e-5 of its float64 h (test_rs3.reference_h64).
3. Accept: mean accepted drafts over many coin draws on fixed paths against the path expectations of
   accept_offline.py (E_tok for token-level, E_bv for BV); |z| < 4 per row.
4. Timing: CUDA graphs over 16 different batches per shape, us per verify, with the sequential BV for reference.
   Budget: BV adds <= 10 us.
"""
import ast
import importlib.util
import itertools
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

import numpy as np
import torch

WT = Path(os.environ["WT"])
G1_WT = Path(os.environ["G1_WT"])
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rs2d"))
import accept_offline as offline

KP, K, VOCAB = 64, 64, 248320
KERNEL = "python/sglang/kernels/ops/speculative/sparse_rs.py"
SEQUENTIAL_BV = "fb1e77a05a"


def load_module(*, name, path):
    spec = importlib.util.spec_from_file_location(name=name, location=path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rs3 = load_module(name="b5_rs3", path=WT / KERNEL)
g1 = load_module(name="b5_g1", path=G1_WT / KERNEL)


def load_sequential(*, directory):
    source = subprocess.run(args=["git", "-C", str(WT), "show", f"{SEQUENTIAL_BV}:{KERNEL}"],
                            capture_output=True, text=True, check=True).stdout
    path = Path(directory) / "sequential_bv.py"
    path.write_text(source)
    return load_module(name="b5_sequential", path=path)


def load_reference_h64():
    # B6's float64 h, loaded without test_rs3's CPU-only imports.
    test_source = Path(__file__).with_name("test_rs3.py").read_text()
    function = next(node for node in ast.parse(test_source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "reference_h64")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "test_rs3.reference_h64", "exec"), namespace)
    return namespace["reference_h64"]


def make_batch(*, bs, slots, generator):
    steps = slots - 1
    q_logits, q_order = torch.sort(input=2.5 * torch.randn(size=(bs, steps, K), generator=generator), dim=-1,
                                   descending=True)
    q = torch.softmax(input=q_logits, dim=-1)
    qi = torch.stack(tensors=[torch.randperm(n=VOCAB, generator=generator)[:K] for _ in range(bs * steps)])
    qi = qi.reshape(bs, steps, K)
    # p keeps q's top 48 ids with noisy log-probabilities, plus 16 ids q never proposes.
    p_logits = torch.empty(size=(bs, slots, KP))
    pi = torch.stack(tensors=[torch.randperm(n=VOCAB, generator=generator)[:KP] for _ in range(bs * slots)])
    pi = pi.reshape(bs, slots, KP)
    p_logits[:, :steps, :48] = q.log()[..., :48] + torch.randn(size=(bs, steps, 48), generator=generator)
    p_logits[:, :steps, 48:] = q.log()[..., 47:48] - 1 + torch.randn(size=(bs, steps, 16), generator=generator)
    pi[:, :steps, :48] = qi[..., :48]
    p_logits[:, steps] = 2.5 * torch.randn(size=(bs, KP), generator=generator)
    p = torch.softmax(input=p_logits, dim=-1)
    draws = torch.multinomial(input=q.reshape(-1, K), num_samples=1, generator=generator).reshape(bs, steps, 1)
    candidates = torch.zeros(size=(bs, slots), dtype=torch.int64)
    candidates[:, 1:] = qi.gather(dim=-1, index=draws).squeeze(dim=-1)
    return dict(candidates=candidates, retrieve=torch.arange(end=bs * slots).reshape(bs, slots),
                coins=torch.rand(size=(bs, slots), generator=generator),
                coins_final=torch.rand(size=(bs,), generator=generator), p=p, pi=pi, q=q, qi=qi)


def outputs_for(*, bs, slots):
    return dict(predicts=torch.full(size=(bs * slots,), fill_value=-1, dtype=torch.int32, device="cuda"),
                accept_index=torch.full(size=(bs, slots), fill_value=-1, dtype=torch.int32, device="cuda"),
                accept_token_num=torch.empty(size=(bs,), dtype=torch.int32, device="cuda"))


def verify(*, module, batch, outputs, **flags):
    module.chain_speculative_sampling_sparse(
        **outputs, candidates=batch["candidates"], retrive_index=batch["retrieve"],
        uniform_samples=batch["coins"], uniform_samples_for_final_sampling=batch["coins_final"],
        target_probs=batch["p"], target_index=batch["pi"], draft_support_probs=batch["q"],
        draft_support_tokens=batch["qi"], vocab_size=VOCAB, **flags)


def to_cuda(*, batch):
    return {name: value.cuda() for name, value in batch.items()}


def graph_time(*, operations, repeats=50):
    for operation in operations:
        operation()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for operation in operations:
            operation()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(7):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / (len(operations) * repeats))
    return statistics.median(times)


def check_same(*, generator):
    rows = 0
    for bs, slots in itertools.product((1, 4, 16), (4, 8, 16)):
        batch = to_cuda(batch=make_batch(bs=bs, slots=slots, generator=generator))
        mine, theirs = outputs_for(bs=bs, slots=slots), outputs_for(bs=bs, slots=slots)
        verify(module=rs3, batch=batch, outputs=mine, block_verify=False)
        verify(module=g1, batch=batch, outputs=theirs)
        for name in mine:
            assert torch.equal(mine[name], theirs[name]), (bs, slots, name)
        rows += bs
    print(f"PASS B5 same: block_verify=False equals the G1 kernel on {rows} rows (bs 1/4/16, slots 4/8/16)",
          flush=True)


def check_same_bv(*, sequential, generator, bs=256):
    reference_h64 = load_reference_h64()
    rows, near_rows, mismatch_rows = 0, 0, 0
    for slots in (4, 8, 16):
        batch = make_batch(bs=bs, slots=slots, generator=generator)
        hs = torch.from_numpy(reference_h64(batch=batch))
        near = (batch["coins"].double()[:, :slots - 1] - hs).abs().le(1e-5).any(dim=1)
        batch = to_cuda(batch=batch)
        mine, theirs = outputs_for(bs=bs, slots=slots), outputs_for(bs=bs, slots=slots)
        verify(module=rs3, batch=batch, outputs=mine, block_verify=True)
        verify(module=sequential, batch=batch, outputs=theirs, block_verify=True)
        retrieve = batch["retrieve"]
        mismatch = ((mine["predicts"][retrieve] != theirs["predicts"][retrieve]).any(dim=1)
                    | (mine["accept_index"] != theirs["accept_index"]).any(dim=1)
                    | (mine["accept_token_num"] != theirs["accept_token_num"])).cpu()
        bad = (mismatch & ~near).nonzero().flatten().tolist()
        assert not bad, (slots, "non-threshold mismatch rows", bad)
        rows += bs
        near_rows += int(near.sum())
        mismatch_rows += int(mismatch.sum())
        print(f"  B5 same BV slots={slots}: rows={bs} near-threshold={int(near.sum())} "
              f"mismatch={int(mismatch.sum())}", flush=True)
    print(f"PASS B5 same BV: block_verify=True equals the sequential BV kernel ({SEQUENTIAL_BV}) on {rows} rows "
          f"except near-threshold coins (near-threshold rows={near_rows}, mismatch rows={mismatch_rows})",
          flush=True)


def check_accept(*, generator, draws=4096):
    worst = 0.0
    for slots in (4, 8, 16):
        bs = 4
        batch = make_batch(bs=bs, slots=slots, generator=generator)
        record = dict(candidates=batch["candidates"], target_probs=batch["p"], target_index=batch["pi"],
                      draft_support_probs=batch["q"], draft_support_tokens=batch["qi"],
                      top_ps=torch.ones(size=(bs,)), min_ps=torch.zeros(size=(bs,)),
                      accept_len=torch.zeros(size=(bs,), dtype=torch.int32))
        expected = offline.record_metrics(record=record, grid=())
        # The same 4 paths, repeated with fresh coins.
        big = {name: value.repeat_interleave(repeats=draws, dim=0) for name, value in batch.items()
               if name not in ("coins", "coins_final", "retrieve")}
        big["retrieve"] = torch.arange(end=bs * draws * slots).reshape(bs * draws, slots)
        big["coins"] = torch.rand(size=(bs * draws, slots), generator=generator)
        big["coins_final"] = torch.rand(size=(bs * draws,), generator=generator)
        big = to_cuda(batch=big)
        for flag, key in ((False, "E_tok"), (True, "E_bv")):
            outputs = outputs_for(bs=bs * draws, slots=slots)
            verify(module=rs3, batch=big, outputs=outputs, block_verify=flag)
            accepted = outputs["accept_token_num"].double().reshape(bs, draws).cpu()
            for row in range(bs):
                mean, se = accepted[row].mean().item(), accepted[row].std().item() / draws ** 0.5
                # A path that always accepts has se 0; the floor absorbs fp32-vs-fp64 rounding there.
                z = (mean - expected[row][key]) / max(se, 1e-4)
                worst = max(worst, abs(z))
                print(f"  B5 accept slots={slots} row={row} {key}: kernel mean={mean:.4f} "
                      f"expected={expected[row][key]:.4f} z={z:+.2f}", flush=True)
    print(f"{'PASS' if worst < 4 else 'FAIL'} B5 accept: worst |z|={worst:.2f} over 3 chains x 4 paths x "
          f"{draws} coin draws, token-level and BV", flush=True)


def check_timing(*, sequential, generator, batches=16):
    worst = -float("inf")
    for bs, slots in itertools.product((1, 4), (4, 8, 16)):
        inputs = [to_cuda(batch=make_batch(bs=bs, slots=slots, generator=generator)) for _ in range(batches)]
        outputs = outputs_for(bs=bs, slots=slots)
        times = {}
        for key, module, flag in (("token", rs3, False), ("bv", rs3, True), ("sequential", sequential, True)):
            times[key] = graph_time(operations=[
                lambda batch=batch, module=module, flag=flag: verify(module=module, batch=batch, outputs=outputs,
                                                                     block_verify=flag)
                for batch in inputs])
        worst = max(worst, times["bv"] - times["token"])
        print(f"B5 timing bs={bs} slots={slots}: token-level {times['token']:.3f} us, BV {times['bv']:.3f} us, "
              f"BV adds {times['bv'] - times['token']:+.3f} us per verify "
              f"(sequential BV {times['sequential']:.3f} us)", flush=True)
    print(f"{'PASS' if worst <= 10 else 'WARN'} B5 timing: BV adds at most {worst:+.3f} us per verify "
          "(budget 10 us)", flush=True)


def main():
    generator = torch.Generator().manual_seed(20261001)
    with tempfile.TemporaryDirectory(prefix="b5-sequential-bv-") as directory:
        sequential = load_sequential(directory=directory)
        check_same(generator=generator)
        check_same_bv(sequential=sequential, generator=generator)
        check_accept(generator=generator)
        check_timing(sequential=sequential, generator=generator)


if __name__ == "__main__":
    main()
