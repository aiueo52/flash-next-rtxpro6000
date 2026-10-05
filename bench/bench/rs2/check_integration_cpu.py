"""CPU storage/dispatch checks; stream ordering and graph capture stay untested."""

import ast
import contextlib
import importlib
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from unittest.mock import patch

import msgspec
import torch

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
assert os.environ.get("TRITON_INTERPRET") == "1"
WT = Path(os.environ.get("WT", "/home/user/tools/sglang-rs2"))
os.environ["FLASHINFER_WORKSPACE_BASE"] = "/tmp/rs2-import-cache"
os.environ["XDG_CACHE_HOME"] = "/tmp/rs2-import-cache"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/rs2-import-cache/inductor"
os.environ["CUDA_CACHE_PATH"] = "/tmp/rs2-import-cache/nv"
sys.path.insert(0, str(WT / "python"))
from sglang.srt.environ import envs


class Batch(msgspec.Struct):
    spec_info: object
    forward_mode: object = None
    out_cache_loc: torch.Tensor | None = None
    positions: torch.Tensor | None = None
    batch_size: int = 0
    sampling_info: object = None
    input_ids: torch.Tensor | None = None


class Sampling(msgspec.Struct):
    temperatures: torch.Tensor
    top_ks: torch.Tensor
    top_ps: torch.Tensor
    min_ps: torch.Tensor


class ModelConfig(msgspec.Struct):
    hf_config: object
    vocab_size: int = 97


class Architecture(msgspec.Struct):
    architectures: tuple[str, ...] = ("SyntheticCPU",)


class Backend(msgspec.Struct):
    attn_backends: list[object]


class RunnerOutput(msgspec.Struct):
    logits_output: object


class DraftRunner(msgspec.Struct):
    model_config: object
    logits: torch.Tensor
    canary_manager: object = None
    inputs: list[torch.Tensor] = []

    def forward(self, forward_batch):
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        self.inputs.append(forward_batch.input_ids.clone())
        return RunnerOutput(logits_output=LogitsProcessorOutput(next_token_logits=self.logits))


class GuardConfig(msgspec.Struct):
    speculative_use_rejection_sampling: bool = True
    speculative_eagle_topk: int = 1
    enable_multi_layer_eagle: bool = False


def make_input(*, cls, n, sparse):
    return cls(
        topk_p=torch.ones((n, 1)), topk_index=torch.zeros((n, 1), dtype=torch.int64),
        bonus_tokens=torch.arange(n, dtype=torch.int32), hidden_states=None,
        draft_probs=None if sparse else torch.rand((n, 97)),
        draft_support_probs=torch.rand((n, 5)) if sparse else None,
        draft_support_tokens=torch.arange(n * 5, dtype=torch.int64).reshape(n, 5) if sparse else None,
    )


def check_fields(*, modules):
    cls = modules["eagle_info"].EagleDraftInput
    relay = modules["overlap_utils"]
    for sparse in (False, True):
        source = make_input(cls=cls, n=3, sparse=sparse)
        initial = source.draft_support_probs.clone() if sparse else source.draft_probs.clone()
        source.filter_batch(new_indices=torch.tensor([2, 0], dtype=torch.int64))
        filtered = source.draft_support_probs if sparse else source.draft_probs
        assert torch.equal(input=filtered, other=initial[[2, 0]])
        other = make_input(cls=cls, n=1, sparse=sparse)
        source.merge_batch(spec_info=other)
        assert source.topk_index.shape == (3, 1)
        merged = source.draft_support_probs if sparse else source.draft_probs
        assert torch.equal(input=merged[:2], other=filtered)
        payload = relay.RelayPayload.from_draft_input(draft_input=source)
        future = object.__new__(relay.FutureMap)
        future.device = torch.device("cpu")
        future.req_pool_size = 8
        future.spec_algo = modules["spec_info"].SpeculativeAlgorithm.EAGLE
        future._forward_buf_initialized = False
        future.dsa_topk_indices_buf = None
        future.output_tokens_buf = torch.empty((8,), dtype=torch.int64)
        indices = torch.tensor([7, 1, 4], dtype=torch.int64)
        with patch(target="sglang.srt.speculative.spec_utils.spec_need_hidden_states", return_value=False):
            future.stash(future_indices=indices, payload=payload)
        receiver = make_input(cls=cls, n=3, sparse=sparse)
        receiver.future_indices = indices[[2, 0, 1]]
        # CPU tensors cannot record CUDA streams; test only the real gather/storage.
        with patch.object(target=torch.Tensor, attribute="record_stream", new=lambda self, stream: None):
            future._resolve_spec_extras(batch=Batch(spec_info=receiver))
        resolved = receiver.draft_support_probs if sparse else receiver.draft_probs
        assert torch.equal(input=resolved, other=merged[[2, 0, 1]])
        if sparse:
            assert future.draft_probs_buf is None and receiver.draft_probs is None
            assert future.draft_support_probs_buf.shape == (8, 5)
            assert future.draft_support_tokens_buf.dtype == torch.int64
            assert torch.equal(input=receiver.draft_support_tokens, other=source.draft_support_tokens[[2, 0, 1]])
        else:
            assert future.draft_support_probs_buf is None and future.draft_support_tokens_buf is None
        with envs.SGLANG_OPT_SPEC_SPARSE_RS.override(sparse), patch(
            target="sglang.srt.speculative.eagle_info.SPEC_SPARSE_RS", new=sparse
        ), patch(
            target="sglang.srt.speculative.eagle_info.get_spec", return_value=GuardConfig()
        ):
            idle = cls.create_idle_input(device=torch.device("cpu"), hidden_size=None, dtype=None,
                                         topk=1, capture_hidden_mode=None, vocab_size=97)
        idle.merge_batch(spec_info=source)
        assert torch.equal(input=idle.topk_index, other=source.topk_index)
        if sparse:
            assert torch.equal(input=idle.draft_support_probs, other=source.draft_support_probs)
    print("PASS CPU storage: idle/filter/merge, RelayPayload and FutureMap stash/gather; CUDA stream hook stubbed")


def check_dflash_relay(*, modules):
    # Non-EAGLE inputs must relay safely with sparse RS disabled.
    with envs.SGLANG_OPT_SPEC_SPARSE_RS.override(False):
        draft = modules["dflash_info_v2"].DFlashDraftInputV2(
            topk_p=torch.ones((1, 1), device="cpu"),
            topk_index=torch.zeros((1, 1), dtype=torch.int64, device="cpu"),
            bonus_tokens=torch.zeros((1,), dtype=torch.int64, device="cpu"),
            new_seq_lens=torch.ones((1,), dtype=torch.int64, device="cpu"),
            hidden_states=torch.empty((1, 0), device="cpu"),
        )
        payload = modules["overlap_utils"].RelayPayload.from_draft_input(draft_input=draft)
        assert payload.draft_support_probs is None
        assert payload.draft_support_tokens is None
    print("PASS CPU DFlash relay: real DFlashDraftInputV2; RS2 off; both support fields None")


def check_support_constructors(*, modules):
    eagle = modules["eagle_info"]
    empty = torch.empty((0,), dtype=torch.int64, device="cpu")
    verify_kwargs = dict(
        draft_token=empty, custom_mask=empty, positions=empty,
        retrieve_index=empty, retrieve_next_token=empty, retrieve_next_sibling=empty,
        retrieve_cum_len=None, spec_steps=3, topk=1, draft_token_num=4,
        capture_hidden_mode=None, seq_lens_sum=0, seq_lens_cpu=empty,
    )
    probs = torch.ones((1, 1), device="cpu")
    tokens = torch.zeros((1, 1), dtype=torch.int64, device="cpu")
    for cls, kwargs in ((eagle.EagleDraftInput, {}), (eagle.EagleVerifyInput, verify_kwargs)):
        default = cls(**kwargs)
        assert default.draft_support_probs is None and default.draft_support_tokens is None
        explicit = cls(**kwargs, draft_support_probs=probs, draft_support_tokens=tokens)
        assert explicit.draft_support_probs is probs and explicit.draft_support_tokens is tokens
    print("PASS CPU support constructors: EagleDraftInput/EagleVerifyInput default None; explicit kwargs preserved")


def check_dispatch(*, modules):
    algorithm = modules["spec_info"].SpeculativeAlgorithm
    with envs.SGLANG_OPT_SPEC_SPARSE_RS.override(True):
        for config, cap, algo in (
            (GuardConfig(speculative_use_rejection_sampling=False), 64, algorithm.EAGLE),
            (GuardConfig(), 0, algorithm.EAGLE),
            (GuardConfig(speculative_eagle_topk=2), 64, algorithm.EAGLE),
            (GuardConfig(enable_multi_layer_eagle=True), 64, algorithm.EAGLE),
            *((GuardConfig(), 64, algo) for algo in
              (algorithm.DFLASH, algorithm.DSPARK, algorithm.NGRAM, algorithm.STANDALONE, algorithm.FROZEN_KV_MTP)),
        ):
            with envs.SGLANG_RS_DRAFT_TOPK.override(cap), patch(
                target="sglang.srt.arg_groups.overrides.resolving_view", return_value=config
            ):
                try:
                    algo.create_worker(server_args=config)
                except ValueError as error:
                    assert "SGLANG_OPT_SPEC_SPARSE_RS" in str(error)
                else:
                    raise AssertionError((config, cap, algo))
        with patch(target="sglang.srt.arg_groups.overrides.resolving_view", return_value=GuardConfig()):
            assert algorithm.EAGLE.create_worker(server_args=GuardConfig()) is modules["eagle_worker_v2"].EAGLEWorkerV2
    print("PASS CPU dispatch: invalid RS/cap/topk/multi-layer/other workers reject before model loading")


def function_source(*, source, name, class_name=None):
    nodes = ast.parse(source).body
    if class_name is not None:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == class_name).body
    node = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name)
    return ast.get_source_segment(source, node)


def check_flag_off(*, modules):
    for relative, name, class_name in (
        ("srt/speculative/spec_utils.py", "sample_draft_proposal_truncated", None),
        ("srt/speculative/spec_utils.py", "sample_draft_proposal", None),
        ("srt/speculative/eagle_worker_v2.py", "_rs_draft_proposal", "EagleDraftWorker"),
    ):
        path = "python/sglang/" + relative
        original = subprocess.run(args=["git", "show", "c868f2ee86:" + path], cwd=WT,
                                  check=True, capture_output=True, text=True).stdout
        current = (WT / path).read_text()
        assert function_source(source=original, name=name, class_name=class_name) == function_source(
            source=current, name=name, class_name=class_name)
    path = "python/sglang/kernels/ops/speculative/reject_sampling.py"
    original = subprocess.run(args=["git", "show", "c868f2ee86:" + path], cwd=WT,
                              check=True, capture_output=True).stdout
    assert original == (WT / path).read_bytes()
    cls = modules["eagle_draft_cuda_graph_runner"].EAGLEDraftCudaGraphRunner
    runner = object.__new__(cls)
    for width in (4, 6):
        outputs = tuple(torch.arange(12).reshape(3, 4) for _ in range(width))
        sliced = runner._postprocess_output_to_raw_bs(out=outputs, raw_bs=2)
        assert len(sliced) == width and all(tensor.shape == (2, 4) for tensor in sliced)
    print("PASS CPU flag off: RS1 proposal helpers unchanged; dense kernel byte-identical; four/six-result graph slicing")


def check_draft_forward(*, modules):
    worker_module = modules["eagle_worker_v2"]
    original = subprocess.run(
        args=["git", "show", "c868f2ee86:python/sglang/srt/speculative/eagle_worker_v2.py"],
        cwd=WT, check=True, capture_output=True, text=True,
    ).stdout
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    bs, steps, vocab = 4, 4, 83
    hot = torch.randperm(97)[:vocab].contiguous()
    sampling = Sampling(temperatures=torch.full((bs, 1), 0.8),
                        top_ks=torch.full((bs,), 40, dtype=torch.int32),
                        top_ps=torch.full((bs,), 0.95), min_ps=torch.full((bs,), 0.05))
    logits = torch.randn((bs, vocab))

    def make_worker(*, sparse):
        worker = object.__new__(worker_module.EagleDraftWorker)
        worker.sparse_rs = sparse
        worker.hot_token_id = hot
        worker.topk = 1
        worker.speculative_num_steps = steps
        worker.speculative_num_draft_tokens = steps + 1
        worker._rs_vocab_size = 97
        worker._topk1_parents_prealloc = torch.arange(-1, steps - 1).repeat(bs, 1)
        worker._topk1_score_indices_prealloc = torch.arange(steps).repeat(bs, 1)
        worker._chain_conf_buf = None
        worker.index_share_for_mtp_iteration = False
        worker.seed_dsa_topk_from_draft_extend = False
        worker.draft_attn_backend = Backend(attn_backends=[None] * steps)
        worker.draft_runner = DraftRunner(model_config=ModelConfig(hf_config=Architecture()), logits=logits)
        worker.target_worker = worker.draft_runner
        return worker

    def make_batch(*, draft):
        return Batch(spec_info=draft, forward_mode=ForwardMode.DECODE,
                     out_cache_loc=torch.zeros((bs * steps,), dtype=torch.int64),
                     positions=torch.arange(bs), batch_size=bs, sampling_info=sampling)

    with patch.object(target=worker_module, attribute="get_spec", return_value=GuardConfig()), patch.object(
        target=worker_module.IndexTopKShareState, attribute="mtp_iteration", return_value=contextlib.nullcontext()
    ), patch.object(target=worker_module, attribute="forward_context", return_value=contextlib.nullcontext()):
        worker = make_worker(sparse=True)
        q0, qi0, p0, x0 = worker._rs_sparse_proposal(next_token_logits=logits, sampling_info=sampling)
        cls = modules["eagle_info"].EagleDraftInput
        draft = cls(topk_p=p0, topk_index=x0, draft_support_probs=q0, draft_support_tokens=qi0)
        batch = make_batch(draft=draft)
        with patch.object(target=worker_module, attribute="select_top_k_tokens", side_effect=AssertionError("RS2 tree select")):
            result = worker.draft_forward(forward_batch=batch)
        _, _, tokens, dense_q, q, qi = result
        assert dense_q is None and q.shape == qi.shape == (bs, steps, 64)
        assert torch.equal(input=tokens[:, :1], other=hot[x0])
        assert torch.equal(input=q[:, 0], other=q0) and torch.equal(input=qi[:, 0], other=qi0)
        assert torch.all(((qi == tokens[:, :, None]) & (q > 0)).any(dim=-1))
        assert torch.equal(input=batch.positions, other=torch.arange(bs) + steps - 1)
        assert all(torch.equal(input=inputs, other=tokens[:, step])
                   for step, inputs in enumerate(worker.draft_runner.inputs))
        namespace = dict(worker_module.__dict__)
        source = function_source(source=original, name="draft_forward", class_name="EagleDraftWorker")
        exec(textwrap.dedent(source), namespace)
        dense_results = []
        for legacy in (False, True):
            worker = make_worker(sparse=False)
            torch.manual_seed(101)
            initial_q, p0, x0 = worker._rs_draft_proposal(next_token_logits=logits, sampling_info=sampling)
            batch = make_batch(draft=cls(topk_p=p0, topk_index=x0, draft_probs=initial_q))
            method = namespace["draft_forward"] if legacy else worker_module.EagleDraftWorker.draft_forward
            with patch.object(target=worker_module, attribute="rs_draft_proposal_sparse", side_effect=AssertionError("RS2 flag-off launch")):
                output = method(self=worker, forward_batch=batch)
            dense_results.append((*output, batch.positions))
        assert all(torch.equal(input=current, other=legacy)
                   for current, legacy in zip(*dense_results))
    print("PASS CPU draft-forward: real RS2 fast chain/hot map/strided uniforms; flag-off matches c868f2ee86 tensors with equal RNG")


if __name__ == "__main__":
    with envs.SGLANG_CACHE_DIR.override("/tmp/rs2-import-cache"):
        names = ("eagle_info", "eagle_utils", "eagle_worker_v2", "eagle_worker_common",
                 "eagle_draft_cuda_graph_runner", "spec_info", "spec_utils", "dflash_info_v2")
        modules = {name: importlib.import_module("sglang.srt.speculative." + name) for name in names}
        modules["overlap_utils"] = importlib.import_module("sglang.srt.managers.overlap_utils")
        check_dflash_relay(modules=modules)
        check_support_constructors(modules=modules)
        check_fields(modules=modules)
        check_dispatch(modules=modules)
        check_flag_off(modules=modules)
        check_draft_forward(modules=modules)
    print("PASS CPU integration checks; no model/server/CUDA graph started")
