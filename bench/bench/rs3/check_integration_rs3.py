"""RS3 CPU runtime wiring checks; no server, model or GPU is started."""

import contextlib
import importlib
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("WT", "/home/user/tools/sglang-rs3")
assert os.environ.get("DEV") == "cpu"
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
assert os.environ.get("TRITON_INTERPRET") == "1"
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "bench/rs2d"))
import check_integration_rs2d as rs2d

base = rs2d.base
G1 = "124840a5c3"


def check_block_verify(*, modules):
    utils = modules["eagle_utils"]
    flags = modules["spec_utils"]
    worker = modules["eagle_worker_v2"]
    sparse = importlib.import_module("sglang.kernels.ops.speculative.sparse_rs")
    target = importlib.import_module("sglang.kernels.ops.speculative.sparse_verify")
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput

    old = subprocess.run(args=["git", "-C", str(base.WT), "show",
                               G1 + ":python/sglang/srt/speculative/eagle_utils.py"],
                         check=True, capture_output=True, text=True).stdout
    namespace = dict(utils.__dict__)
    exec("from __future__ import annotations\n" + textwrap.dedent(base.function_source(source=old, name="eagle_sample")), namespace)
    legacy_sample = namespace["eagle_sample"]
    bs, slots, vocab = 1, 4, 8
    p = torch.tensor(data=[[[0.4, 0.6], [0.9, 0.1], [0.5, 0.5], [0.3, 0.7]]])
    pi = torch.tensor(data=[[[2, 5]] * slots], dtype=torch.int64)
    q = torch.tensor(data=[[[0.8, 0.2], [0.1, 0.9], [0.5, 0.5]]])
    candidates = torch.tensor(data=[[0, 2, 2, 5]], dtype=torch.int64)
    retrieve = torch.arange(end=slots, dtype=torch.int64).reshape(bs, slots)
    verify = modules["eagle_info"].EagleVerifyInput(
        draft_token=candidates.flatten(), custom_mask=torch.empty(size=(0,)), positions=retrieve.flatten(),
        retrieve_index=retrieve, retrieve_next_token=retrieve, retrieve_next_sibling=retrieve,
        retrieve_cum_len=None, spec_steps=slots - 1, topk=1, draft_token_num=slots,
        capture_hidden_mode=None, seq_lens_sum=3, seq_lens_cpu=torch.tensor(data=[3]),
        draft_support_probs=q, draft_support_tokens=pi[:, :-1].clone(),
    )
    sampling = rs2d.VerifySampling(temperatures=torch.tensor(data=[[0.8]]), top_ks=torch.tensor(data=[2]),
                                   top_ps=torch.tensor(data=[0.95]), min_ps=torch.tensor(data=[0.05]))
    batch = rs2d.VerifyBatch(device=torch.device("cpu"), forward_mode=ForwardMode.DECODE,
                             seq_lens=torch.tensor(data=[3]),
                             reqs=[rs2d.Request(rid="rs3-cpu", origin_input_ids=[1, 2, 3])], sampling_info=sampling)
    logits = LogitsProcessorOutput(next_token_logits=torch.zeros(size=(slots, vocab)))
    coins = torch.tensor(data=[[0.8, 0.1, 0.1, 0.5]])
    final = torch.tensor(data=[0.2])
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(target=utils, attribute="SPEC_SPARSE_RS", new=True))
        namespace["SPEC_SPARSE_RS"] = True
        namespace["RS_DUMP_DIR"] = ""
        stack.enter_context(patch.object(target=utils, attribute="RS_DUMP_DIR", new=""))
        for flag in ("_is_cpu", "_is_npu", "_is_hip", "_is_xpu"):
            stack.enter_context(patch.object(target=utils, attribute=flag, new=False))
            namespace[flag] = False
        width = stack.enter_context(patch.object(target=utils, attribute="_sparse_rs_verify_kp", return_value=2))
        coin_mock = stack.enter_context(patch.object(target=utils, attribute="_verify_coins", return_value=(coins, final)))
        namespace["_sparse_rs_verify_kp"] = width
        namespace["_verify_coins"] = coin_mock
        stack.enter_context(patch.object(target=target, attribute="sparse_target_probs", return_value=(p, pi)))
        stack.enter_context(patch(target="sglang.srt.distributed.get_tp_group", return_value=rs2d.TPGroup()))
        stack.enter_context(patch(target="sglang.srt.layers.dp_attention.is_dp_attention_enabled", return_value=False))
        stack.enter_context(patch.object(target=flags, attribute="SIMULATE_ACC_LEN", new=0))
        spy = stack.enter_context(patch.object(target=sparse, attribute="chain_speculative_sampling_sparse",
                                               wraps=sparse.chain_speculative_sampling_sparse))
        legacy = legacy_sample(verify_input=verify, batch=batch, logits_output=logits)
        legacy_kwargs = spy.call_args.kwargs
        assert "block_verify" not in legacy_kwargs
        off = utils.eagle_sample(verify_input=verify, batch=batch, logits_output=logits)
        off_kwargs = spy.call_args.kwargs
        assert off_kwargs.keys() == legacy_kwargs.keys()
        for name, value in off_kwargs.items():
            previous = legacy_kwargs[name]
            assert torch.equal(input=value, other=previous) if isinstance(value, torch.Tensor) else value == previous
        assert all(torch.equal(input=a, other=b) for a, b in zip(off, legacy))
        with patch.object(target=utils, attribute="RS_BLOCK_VERIFY", new=True):
            on = utils.eagle_sample(verify_input=verify, batch=batch, logits_output=logits)
        assert spy.call_args.kwargs["block_verify"] is True
        assert on[1].tolist() == [4] and off[1].tolist() == [1]
        # The dense RS fallback is the G1 source and gets no BV argument.
        current = base.function_source(source=(base.WT / "python/sglang/srt/speculative/eagle_utils.py").read_text(), name="eagle_sample")
        legacy_source = base.function_source(source=old, name="eagle_sample")
        marker = "        from sgl_kernel import ("
        assert current[current.index(marker):] == legacy_source[legacy_source.index(marker):]
    print("PASS RS3 integration: flag unset matches G1 call kwargs and tensors; set passes block_verify=True, real CPU BV accepts 3 vs 0; dense fallback source unchanged")
    algorithm = modules["spec_info"].SpeculativeAlgorithm
    with base.envs.SGLANG_RS_BLOCK_VERIFY.override(True), patch(
        target="sglang.srt.arg_groups.overrides.resolving_view", return_value=base.GuardConfig()
    ):
        with base.envs.SGLANG_OPT_SPEC_SPARSE_RS.override(False):
            try:
                algorithm.EAGLE.create_worker(server_args=base.GuardConfig())
            except ValueError as error:
                assert "SGLANG_RS_BLOCK_VERIFY requires SGLANG_OPT_SPEC_SPARSE_RS" in str(error)
            else:
                raise AssertionError("BV without sparse RS accepted")
        with base.envs.SGLANG_OPT_SPEC_SPARSE_RS.override(True):
            assert algorithm.EAGLE.create_worker(server_args=base.GuardConfig()) is worker.EAGLEWorkerV2
        importlib.reload(utils)
        importlib.reload(flags)
        importlib.reload(worker)
        assert utils.RS_BLOCK_VERIFY is flags.RS_BLOCK_VERIFY is worker.RS_BLOCK_VERIFY is True
    importlib.reload(utils)
    importlib.reload(flags)
    importlib.reload(worker)
    assert utils.RS_BLOCK_VERIFY is flags.RS_BLOCK_VERIFY is worker.RS_BLOCK_VERIFY is False
    print("PASS RS3 integration: startup ValueError without SPARSE_RS; import-time flag reaches verify and worker log variable, default=False")


if __name__ == "__main__":
    torch.set_num_threads(4)
    assert "SGLANG_RS_BLOCK_VERIFY" not in os.environ, "Run integration with BV flag unset"
    with base.envs.SGLANG_CACHE_DIR.override("/tmp/rs2-import-cache"):
        names = ("eagle_info", "eagle_utils", "eagle_worker_v2", "eagle_worker_common",
                 "eagle_draft_cuda_graph_runner", "spec_info", "spec_utils", "dflash_info_v2")
        modules = {name: importlib.import_module("sglang.srt.speculative." + name) for name in names}
        modules["overlap_utils"] = importlib.import_module("sglang.srt.managers.overlap_utils")
        base.check_dflash_relay(modules=modules)
        base.check_support_constructors(modules=modules)
        base.check_fields(modules=modules)
        base.check_dispatch(modules=modules)
        base.check_flag_off(modules=modules)
        check_block_verify(modules=modules)
    print("PASS RS3 CPU integration; CUDA timing/graphs and server remain parent-run")
