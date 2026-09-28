"""Pluggable MoE candidates.  Registry so a Triton candidate can be dropped in.

``flashinfer_cutlass`` is the production path: it imports SGLang's own
``_run_flashinfer_cutlass`` and calls it with the same arguments the model
uses, so what is timed is byte-for-byte what the server runs.

Call chain being reproduced (verified in the fork, 2026-09-05)::

    FusedMoE.forward                      layers/moe/fused_moe_triton/layer.py:1421
     -> run_moe_core                                                        :1530
     -> ModelOptNvFp4FusedMoEMethod.apply  quantization/modelopt_quant.py:2990
        FlashInferCutlassMoeQuantInfo(quant_type="fp4", ...)                 :3106
        quant_scales = [w13_input_scale_quant, w13_blockscale_swizzled, g1_alphas,
                        w2_input_scale_quant,  w2_blockscale_swizzled,  g2_alphas] :3111
     -> MoeRunner.run                      layers/moe/moe_runner/runner.py:143
     -> fused_experts_none_to_flashinfer_cutlass
                                  layers/moe/moe_runner/flashinfer_cutlass.py:247
     -> _run_flashinfer_cutlass                                              :178
     -> flashinfer.fused_moe.cutlass_fused_moe                               :222

GPU-ONLY except for ``build_inputs`` (which works on CPU).
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from .weights import LayerWeights

_REGISTRY: dict[str, Callable] = {}


def ensure_sglang_imports() -> None:
    """Import SGLang's modules in the order the server uses, or the MoE runner
    hits a circular import.

    ``moe_runner/flashinfer_trtllm.py:110`` does
    ``if is_flashinfer_available(): from sglang.srt.layers.quantization.fp4_utils
    import fp4_quantize``, which pulls in ``layers/quantization/__init__.py`` ->
    ``compressed_tensors`` -> ``modelopt_quant`` -> ``fp8.py:38``, and *that*
    does ``from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import
    FlashInferTrtllmFp8MoeQuantInfo`` -- back into the module that is still
    executing its own line 110::

        ImportError: cannot import name 'FlashInferTrtllmFp8MoeQuantInfo' from
        partially initialized module ...flashinfer_trtllm (circular import)

    The server never sees it because ``srt/models/*.py`` import
    ``layers.quantization.*`` first, so ``quantization/__init__`` is already
    running (and ``fp4_utils`` is just a submodule load) by the time any
    ``moe_runner`` module is touched.  A standalone harness that imports
    ``moe_runner.flashinfer_cutlass`` first does not, and the cycle only fires
    on a machine where ``is_flashinfer_available()`` is True -- i.e. it is
    invisible on CPU unless that flag is forced (see ``tests_cpu.py``).

    Importing the quantization package to completion first is exactly what the
    model modules do, and it is cheap (already in ``sys.modules`` in-server).
    """
    import sglang.srt.layers.quantization  # noqa: F401  (import for side effect)
    import sglang.srt.layers.moe.moe_runner.flashinfer_trtllm  # noqa: F401


def register(name: str):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


def available() -> list[str]:
    return sorted(_REGISTRY)


def get(name: str) -> Callable:
    if name not in _REGISTRY:
        raise SystemExit(f"unknown candidate {name!r}; have {available()}")
    return _REGISTRY[name]


def load_autotune_cache(path: str = "auto") -> str | None:
    """Load SGLang's on-disk FlashInfer tactic cache into the AutoTuner singleton.

    **This is not optional for a representative measurement.**  Outside an
    ``autotune()`` context FlashInfer looks the tactic up in the in-process
    ``AutoTuner`` cache; a standalone harness starts with that cache empty and
    silently runs tactic ``-1`` -- the default tile, with no ``swap_ab`` and no
    FINALIZE epilogue, so a separate ``finalizeMoeRoutingKernel`` appears that
    the server does not have.  Measured cost of getting this wrong on this box:
    +33 us per MoE call at T=4 (117.6 -> 84.6 us), i.e. +1.6 ms/step.

    The server writes the cache in
    ``$SGLANG_CACHE_DIR/flashinfer/autotune/<ver>/<arch>/<cfg-hash>/rank_tp0_pp0_dp0.json``
    (``model_executor/runner/flashinfer_autotune.py:160-172``); ``auto`` picks the
    most recently modified one, which is the profile that ran last.
    """
    import glob
    import os

    from flashinfer.autotuner import AutoTuner

    if path == "auto":
        root = os.path.expanduser("~/.cache/sglang/flashinfer/autotune")
        cands = glob.glob(os.path.join(root, "*", "*", "*", "rank_tp*.json"))
        if not cands:
            print(f"[bench] no autotune cache under {root}; running default tactics")
            return None
        path = max(cands, key=os.path.getmtime)
    if not os.path.exists(path):
        print(f"[bench] autotune cache {path} missing; running default tactics")
        return None
    AutoTuner.get().load_configs(path)
    n = len(getattr(AutoTuner.get(), "profiling_cache", {}))
    print(f"[bench] loaded autotune cache ({n} entries): {path}")
    return path


def force_tactics(gemm1: int | None, gemm2: int | None) -> None:
    """Pin the CUTLASS tactic ids instead of asking the AutoTuner.

    ``cutlass_fused_moe``'s ``profile_ids`` argument is accepted but *unused* in
    FlashInfer 0.6.17 (`fused_moe/core.py:546` is never read; the tactics come
    from `tuner.choose_one` at :578/:594), so the only way to force a tactic is
    to intercept the tuner.  Needed by any experiment that changes the problem
    shape, because the on-disk cache is keyed on the shape and a miss silently
    falls back to tactic -1 (no swap_ab, no FINALIZE) -- see spec section 8.1.
    """
    from flashinfer.autotuner import AutoTuner

    if getattr(AutoTuner, "_g1_forced", None) is not None:
        AutoTuner.choose_one = AutoTuner._g1_orig_choose_one
    orig = AutoTuner.choose_one
    AutoTuner._g1_orig_choose_one = orig
    AutoTuner._g1_forced = (gemm1, gemm2)

    def patched(self, custom_op, runners, tuning_config, inputs, **kwargs):
        if custom_op == "trtllm::fused_moe::gemm1" and gemm1 is not None:
            return runners[0], gemm1
        if custom_op == "trtllm::fused_moe::gemm2" and gemm2 is not None:
            return runners[0], gemm2
        return orig(self, custom_op, runners, tuning_config, inputs, **kwargs)

    AutoTuner.choose_one = patched
    print(f"[bench] forced tactics gemm1={gemm1} gemm2={gemm2}")


@dataclass
class CallInputs:
    """One MoE call's inputs.  All tensors live on the target device."""
    hidden_states: torch.Tensor      # [T, H] bf16
    topk_ids: torch.Tensor           # [T, top_k] int32
    topk_weights: torch.Tensor       # [T, top_k] float32
    output: torch.Tensor             # [T, H] bf16, preallocated


def build_inputs(lw: LayerWeights, routings, weights, device="cpu",
                 dtype=torch.bfloat16, seed: int = 0) -> list[CallInputs]:
    """One CallInputs per routing.  Works on CPU (for shape/CLI tests)."""
    g = torch.Generator().manual_seed(seed)
    H = lw.shape.hidden
    out = []
    for r, w in zip(routings, weights):
        T = r.shape[0]
        x = (torch.randn((T, H), generator=g) * 0.5).to(dtype).to(device)
        out.append(CallInputs(
            hidden_states=x,
            topk_ids=r.to(torch.int32).to(device),
            topk_weights=w.to(torch.float32).to(device),
            output=torch.empty((T, H), dtype=dtype, device=device),
        ))
    return out


def _duck_dispatch_output(ci: CallInputs):
    """Minimal stand-in for StandardDispatchOutput / TopKOutput.

    ``_run_flashinfer_cutlass`` only reads ``.hidden_states``,
    ``.hidden_states_scale`` and ``.topk_output.{topk_weights,topk_ids}``.
    """
    topk = types.SimpleNamespace(topk_weights=ci.topk_weights, topk_ids=ci.topk_ids)
    return types.SimpleNamespace(
        hidden_states=ci.hidden_states,
        hidden_states_scale=None,   # production single-GPU value (input_sf=None)
        topk_output=topk,
    )


def _runner_config(lw: LayerWeights):
    ensure_sglang_imports()
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig

    return MoeRunnerConfig(
        num_experts=lw.shape.num_experts,
        num_local_experts=lw.num_experts,
        hidden_size=lw.shape.hidden,
        intermediate_size_per_partition=lw.shape.intermediate,
        top_k=lw.shape.top_k,
        params_dtype=torch.bfloat16,
        activation="silu",
        is_gated=True,
        routed_scaling_factor=None,
    )


@register("flashinfer_cutlass")
def flashinfer_cutlass(lw: LayerWeights, inputs: list[CallInputs]) -> list[Callable]:
    """Production path.  Returns one zero-arg callable per input."""
    ensure_sglang_imports()
    from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
        FlashInferCutlassMoeQuantInfo,
        _run_flashinfer_cutlass,
    )

    quant_info = FlashInferCutlassMoeQuantInfo(
        quant_type="fp4",
        w13_weight=lw.w13_weight,
        w2_weight=lw.w2_weight,
        output_dtype=torch.bfloat16,
        quant_scales=[
            lw.w13_input_scale_quant,
            lw.w13_blockscale_swizzled,
            lw.g1_alphas,
            lw.w2_input_scale_quant,
            lw.w2_blockscale_swizzled,
            lw.g2_alphas,
        ],
        moe_ep_size=1, moe_ep_rank=0, moe_tp_size=1, moe_tp_rank=0,
        apply_routed_scaling_factor=False,
    )
    rc = _runner_config(lw)
    calls = []
    for ci in inputs:
        d = _duck_dispatch_output(ci)
        # `output=` is passed so the symmetric-memory allocation path (which
        # needs a live TP group) is skipped -- everything else is identical.
        calls.append(lambda d=d, ci=ci: _run_flashinfer_cutlass(
            dispatch_output=d, quant_info=quant_info, runner_config=rc,
            output=ci.output,
        ))
    return calls


@register("flashinfer_direct")
def flashinfer_direct(lw: LayerWeights, inputs: list[CallInputs]) -> list[Callable]:
    """Same kernels, called straight through flashinfer.

    Useful for tactic forcing / knob sweeps that SGLang does not expose:
    ``tune_max_num_tokens``, ``use_fused_finalize``, ``workspace_buffer``.
    Knobs are read from module-level ``DIRECT_OPTS`` so the CLI can set them.
    """
    ensure_sglang_imports()
    from flashinfer.fused_moe import cutlass_fused_moe
    from flashinfer.fused_moe.core import ActivationType

    w13 = lw.w13_weight.view(torch.long)
    w2 = lw.w2_weight.view(torch.long)
    qs = [
        lw.w13_input_scale_quant,
        lw.w13_blockscale_swizzled.view(torch.int32),
        lw.g1_alphas,
        lw.w2_input_scale_quant,
        lw.w2_blockscale_swizzled.view(torch.int32),
        lw.g2_alphas,
    ]
    opts = dict(DIRECT_OPTS)
    calls = []
    for ci in inputs:
        T = ci.hidden_states.shape[0]
        kw = dict(
            output=ci.output,
            input=ci.hidden_states,
            token_selected_experts=ci.topk_ids.to(torch.int),
            token_final_scales=ci.topk_weights,
            fc1_expert_weights=w13,
            fc2_expert_weights=w2,
            output_dtype=torch.bfloat16,
            input_sf=None,
            quant_scales=qs,
            ep_size=1, ep_rank=0, tp_size=1, tp_rank=0,
            tune_max_num_tokens=opts.get("tune_max_num_tokens", 1 << (T - 1).bit_length()),
            activation_type=ActivationType.Swiglu,
            enable_alltoall=False,
            use_fused_finalize=opts.get("use_fused_finalize", True),
        )
        if opts.get("workspace_buffer") is not None:
            kw["workspace_buffer"] = opts["workspace_buffer"]
        calls.append(lambda kw=kw: cutlass_fused_moe(**kw)[0])
    return calls


#: Mutable knobs for the ``flashinfer_direct`` candidate (set by the CLI).
DIRECT_OPTS: dict = {}


@register("triton_grouped")
def triton_grouped(lw: LayerWeights, inputs: list[CallInputs]) -> list[Callable]:
    """Placeholder registry slot for a Triton NVFP4 small-M grouped kernel.

    Not implemented. A candidate must accept SGLang's swizzled 128x4 block-scale
    layout and its W4A4 activation quantisation, or document where it differs.
    """
    raise NotImplementedError("triton_grouped candidate not implemented")
