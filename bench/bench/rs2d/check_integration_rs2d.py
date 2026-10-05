"""CPU runtime integration checks; no server/model/GPU is started."""

import builtins
import contextlib
import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
from unittest.mock import patch

import msgspec
import torch

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("WT", "/home/user/tools/sglang-rs2d")
assert os.environ.get("DEV") == "cpu"
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
assert os.environ.get("TRITON_INTERPRET") == "1"
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "bench/rs2"))
import check_integration_cpu as base

WT = base.WT
BASELINE = "cf1e515722"
HAS_GREEDY_FAST = "greedy_fast" in importlib.import_module(
    "sglang.kernels.ops.speculative.sparse_rs").rs_draft_proposal_sparse.__code__.co_varnames


def original(*, relative):
    return subprocess.run(args=["git", "-C", str(WT), "show", BASELINE + ":python/sglang/" + relative],
                          check=True, capture_output=True, text=True).stdout


def check_knobs(*, modules):
    worker_module = modules["eagle_worker_v2"]
    utils = modules["spec_utils"]
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    old = original(relative="srt/speculative/eagle_worker_v2.py")
    namespace = dict(worker_module.__dict__)
    for name in ("_rs_sparse_proposal", "draft_forward"):
        source = base.function_source(source=old, name=name, class_name="EagleDraftWorker")
        exec(textwrap.dedent(source), namespace)
    original_methods = {name: namespace[name] for name in ("_rs_sparse_proposal", "draft_forward")}
    bs, steps, vocab = 4, 4, 83
    torch.manual_seed(1001)
    hot = torch.randperm(n=97)[:vocab].contiguous()
    sampling = base.Sampling(temperatures=torch.full(size=(bs, 1), fill_value=0.8),
                             top_ks=torch.full(size=(bs,), fill_value=40, dtype=torch.int32),
                             top_ps=torch.full(size=(bs,), fill_value=0.95),
                             min_ps=torch.full(size=(bs,), fill_value=0.05))
    logits = torch.randn(size=(bs, vocab))

    def execute(*, legacy):
        worker = object.__new__(worker_module.EagleDraftWorker)
        worker.sparse_rs = True
        worker.hot_token_id = hot
        worker.topk = 1
        worker.speculative_num_steps = steps
        worker.speculative_num_draft_tokens = steps + 1
        worker._rs_vocab_size = 97
        worker._topk1_parents_prealloc = torch.arange(start=-1, end=steps - 1).repeat(bs, 1)
        worker._topk1_score_indices_prealloc = torch.arange(end=steps).repeat(bs, 1)
        worker._chain_conf_buf = None
        worker.index_share_for_mtp_iteration = False
        worker.seed_dsa_topk_from_draft_extend = False
        worker.draft_attn_backend = base.Backend(attn_backends=[None] * steps)
        worker.draft_runner = base.DraftRunner(model_config=base.ModelConfig(hf_config=base.Architecture()), logits=logits)
        worker.target_worker = worker.draft_runner
        with contextlib.ExitStack() as stack:
            for name, method in original_methods.items():
                if legacy:
                    stack.enter_context(patch.object(target=worker_module.EagleDraftWorker, attribute=name, new=method))
            spy = stack.enter_context(patch.object(target=worker_module, attribute="rs_draft_proposal_sparse",
                                                   wraps=worker_module.rs_draft_proposal_sparse))
            stack.enter_context(patch.object(target=worker_module, attribute="get_spec", return_value=base.GuardConfig()))
            stack.enter_context(patch.object(target=worker_module.IndexTopKShareState, attribute="mtp_iteration",
                                             return_value=contextlib.nullcontext()))
            stack.enter_context(patch.object(target=worker_module, attribute="forward_context", return_value=contextlib.nullcontext()))
            if legacy:
                namespace["rs_draft_proposal_sparse"] = spy
            torch.manual_seed(1234)
            q0, qi0, p0, x0 = worker._rs_sparse_proposal(next_token_logits=logits, sampling_info=sampling)
            draft = modules["eagle_info"].EagleDraftInput(topk_p=p0, topk_index=x0,
                                                         draft_support_probs=q0, draft_support_tokens=qi0)
            batch = base.Batch(spec_info=draft, forward_mode=ForwardMode.DECODE,
                               out_cache_loc=torch.zeros(size=(bs * steps,), dtype=torch.int64),
                               positions=torch.arange(end=bs), batch_size=bs, sampling_info=sampling)
            result = worker.draft_forward(forward_batch=batch)
            return (q0, qi0, p0, x0, *result, batch.positions), spy.call_args_list

    for name in ("SGLANG_RS_DRAFT_TEMP_SCALE", "SGLANG_RS_DRAFT_ONEHOT_ABOVE", "SGLANG_RS_GREEDY_FAST"):
        assert name not in os.environ, "Run integration with sharpening knobs unset"
    legacy_results, legacy_calls = execute(legacy=True)
    current_results, current_calls = execute(legacy=False)
    assert len(current_calls) == len(legacy_calls) == steps
    for actual, expected in zip(current_results, legacy_results):
        assert (actual is None and expected is None) or torch.equal(input=actual, other=expected)
    for actual, expected in zip(current_calls, legacy_calls):
        kwargs = dict(actual.kwargs)
        assert "greedy_fast" not in kwargs
        assert kwargs.pop("temp_scale") == 1.0 and kwargs.pop("onehot_above") == 0.0
        assert kwargs.keys() == expected.kwargs.keys()
        for name, value in kwargs.items():
            previous = expected.kwargs[name]
            assert (value is None and previous is None) or (
                torch.equal(input=value, other=previous) if isinstance(value, torch.Tensor) else value == previous)
    print("PASS integration defaults: both actual call sites match cf1e515722 kwargs plus identity knobs; all output tensors bit-identical")
    with base.envs.SGLANG_RS_DRAFT_TEMP_SCALE.override(0.7), base.envs.SGLANG_RS_DRAFT_ONEHOT_ABOVE.override(0.8):
        importlib.reload(utils)
        importlib.reload(worker_module)
        assert utils.RS_DRAFT_TEMP_SCALE == 0.7 and utils.RS_DRAFT_ONEHOT_ABOVE == 0.8
        _, calls = execute(legacy=False)
        assert len(calls) == steps
        assert all(call.kwargs["temp_scale"] == 0.7 and call.kwargs["onehot_above"] == 0.8 for call in calls)
    importlib.reload(utils)
    importlib.reload(worker_module)
    print("PASS integration knobs set: import-time env values .7/.8 reach initial proposal and all draft-forward calls")
    if HAS_GREEDY_FAST:
        with base.envs.SGLANG_RS_GREEDY_FAST.override(True):
            importlib.reload(utils)
            importlib.reload(worker_module)
            assert utils.RS_GREEDY_FAST is True and worker_module.RS_GREEDY_FAST is True
            _, calls = execute(legacy=False)
            assert len(calls) == steps and all(call.kwargs["greedy_fast"] is True for call in calls)
        importlib.reload(utils)
        importlib.reload(worker_module)
        print("PASS integration greedy flag: unset preserves existing kwargs; True reaches both actual call sites")
    else:
        assert all("greedy_fast" not in call.kwargs for call in current_calls)
        print("SKIP integration greedy flag: pre-G1 worktree; existing call kwargs unchanged")


def check_guards(*, modules):
    algorithm = modules["spec_info"].SpeculativeAlgorithm
    with base.envs.SGLANG_OPT_SPEC_SPARSE_RS.override(True), patch(
        target="sglang.srt.arg_groups.overrides.resolving_view", return_value=base.GuardConfig()
    ):
        for name, knob, values in (("SGLANG_RS_DRAFT_TEMP_SCALE", base.envs.SGLANG_RS_DRAFT_TEMP_SCALE, (0, -0.1, float("nan"))),
                                   ("SGLANG_RS_DRAFT_ONEHOT_ABOVE", base.envs.SGLANG_RS_DRAFT_ONEHOT_ABOVE, (-0.1, 1.1, float("nan")))):
            for value in values:
                with knob.override(value):
                    try:
                        algorithm.EAGLE.create_worker(server_args=base.GuardConfig())
                    except ValueError as error:
                        assert name in str(error)
                    else:
                        raise AssertionError((name, value))
        for theta in (0, 1):
            with base.envs.SGLANG_RS_DRAFT_ONEHOT_ABOVE.override(theta):
                assert algorithm.EAGLE.create_worker(server_args=base.GuardConfig()) is modules["eagle_worker_v2"].EAGLEWorkerV2
    print("PASS integration startup guards: invalid scale/threshold including NaN reject; theta=0/1 accepted")


class VerifySampling(msgspec.Struct):
    temperatures: torch.Tensor
    top_ks: torch.Tensor
    top_ps: torch.Tensor
    min_ps: torch.Tensor
    is_all_greedy: bool = False
    need_top_k_sampling: bool = True
    need_top_p_sampling: bool = True
    need_min_p_sampling: bool = True
    acc_additive_penalties: object = None
    acc_scaling_penalties: object = None
    logit_bias: object = None


class Request(msgspec.Struct):
    rid: str
    origin_input_ids: list[int]


class VerifyBatch(msgspec.Struct):
    device: object
    forward_mode: object
    seq_lens: torch.Tensor
    reqs: list[Request]
    sampling_info: VerifySampling


class TPGroup(msgspec.Struct):
    world_size: int = 1


def check_dump(*, modules):
    utils = modules["eagle_utils"]
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    sparse = importlib.import_module("sglang.kernels.ops.speculative.sparse_rs")
    target = importlib.import_module("sglang.kernels.ops.speculative.sparse_verify")
    bs, slots, vocab = 1, 3, 8
    p = torch.tensor(data=[[[0.6, 0.4]] * slots])
    pi = torch.tensor(data=[[[2, 5]] * slots], dtype=torch.int64)
    candidates = torch.tensor(data=[[0, 2, 5]], dtype=torch.int64)
    retrieve = torch.arange(end=slots, dtype=torch.int64).reshape(bs, slots)
    verify = modules["eagle_info"].EagleVerifyInput(
        draft_token=candidates.flatten(), custom_mask=torch.empty(size=(0,)), positions=retrieve.flatten(),
        retrieve_index=retrieve, retrieve_next_token=retrieve, retrieve_next_sibling=retrieve,
        retrieve_cum_len=None, spec_steps=slots-1, topk=1, draft_token_num=slots,
        capture_hidden_mode=None, seq_lens_sum=3, seq_lens_cpu=torch.tensor(data=[3]),
        draft_support_probs=p[:, :-1].clone(), draft_support_tokens=pi[:, :-1].clone(),
    )
    sampling = VerifySampling(temperatures=torch.tensor(data=[[0.8]]), top_ks=torch.tensor(data=[2]),
                              top_ps=torch.tensor(data=[0.95]), min_ps=torch.tensor(data=[0.05]))
    batch = VerifyBatch(device=torch.device("cpu"), forward_mode=ForwardMode.DECODE,
                        seq_lens=torch.tensor(data=[3]), reqs=[Request(rid="cpu", origin_input_ids=[1, 2, 3])],
                        sampling_info=sampling)
    logits = LogitsProcessorOutput(next_token_logits=torch.zeros(size=(slots, vocab)))
    import_calls, order = [], []
    real_import = builtins.__import__
    real_verify = sparse.chain_speculative_sampling_sparse

    def observe_import(name, *args, **kwargs):
        if name == "sglang.srt.speculative.rs_dump":
            import_calls.append(name)
        return real_import(name, *args, **kwargs)

    def observe_verify(**kwargs):
        real_verify(**kwargs)
        order.append("verify")

    with tempfile.TemporaryDirectory(prefix="rs2d-integration-") as directory, contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(target=utils, attribute="SPEC_SPARSE_RS", new=True))
        stack.enter_context(patch.object(target=utils, attribute="SPEC_MIN_P", new=True))
        for flag in ("_is_cpu", "_is_npu", "_is_hip", "_is_xpu"):
            stack.enter_context(patch.object(target=utils, attribute=flag, new=False))
        stack.enter_context(patch.object(target=utils, attribute="_sparse_rs_verify_kp", return_value=2))
        stack.enter_context(patch.object(target=utils, attribute="_verify_coins", return_value=(
            torch.zeros(size=(bs, slots)), torch.tensor(data=[0.2]))))
        stack.enter_context(patch.object(target=target, attribute="sparse_target_probs", return_value=(p, pi)))
        stack.enter_context(patch.object(target=sparse, attribute="chain_speculative_sampling_sparse", side_effect=observe_verify))
        stack.enter_context(patch(target="sglang.srt.distributed.get_tp_group", return_value=TPGroup()))
        stack.enter_context(patch(target="sglang.srt.layers.dp_attention.is_dp_attention_enabled", return_value=False))
        stack.enter_context(patch.object(target=modules["spec_utils"], attribute="SIMULATE_ACC_LEN", new=0))
        stack.enter_context(patch.object(target=builtins, attribute="__import__", side_effect=observe_import))
        with patch.object(target=utils, attribute="RS_DUMP_DIR", new=""):
            off = utils.eagle_sample(verify_input=verify, batch=batch, logits_output=logits)
        assert import_calls == [] and "sglang.srt.speculative.rs_dump" not in sys.modules
        dump = importlib.import_module("sglang.srt.speculative.rs_dump")
        real_record = dump.record_verify

        def observe_record(**kwargs):
            assert order[-1] == "verify"
            order.append("dump")
            real_record(**kwargs)

        with patch.object(target=utils, attribute="RS_DUMP_DIR", new=directory), patch.object(
            target=dump, attribute="record_verify", side_effect=observe_record
        ):
            on = utils.eagle_sample(verify_input=verify, batch=batch, logits_output=logits)
        assert import_calls == ["sglang.srt.speculative.rs_dump"]
        assert all(torch.equal(input=a, other=b) for a, b in zip(off, on))
        dump.flush()
        record = torch.load(f=next(Path(directory).glob("rs-dump-*.pt")), weights_only=True)[0]
        assert record["accept_len"].tolist() == [2] and record["target_probs"].shape == (bs, slots, 2)
        with patch.object(target=utils, attribute="RS_DUMP_DIR", new=directory), patch(
            target="sglang.srt.distributed.get_tp_group", return_value=TPGroup(world_size=2)
        ):
            try:
                utils.eagle_sample(verify_input=verify, batch=batch, logits_output=logits)
            except ValueError as error:
                assert "TP = 1" in str(error)
            else:
                raise AssertionError("TP > 1 dump accepted")
    print("PASS integration dump: off never imports rs_dump; on runs after real CPU verify, output unchanged, TP>1 rejected")


if __name__ == "__main__":
    torch.set_num_threads(4)
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
        check_knobs(modules=modules)
        check_guards(modules=modules)
        check_dump(modules=modules)
    print("PASS RS2d CPU integration; CUDA stream ordering, graphs and server remain parent-run")
