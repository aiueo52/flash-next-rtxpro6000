"""Load ONE MoE layer's real expert weights and build SGLang's stacked layout.

Reads only the tensors it needs out of the per-layer expert shards
(``layer-000NN-experts-XXXX-YYYY.safetensors``).  Works with ``device="cpu"``,
so the loader, the stacking and the swizzle are all testable without a GPU.

Checkpoint layout (verified against the header of
``layer-00004-experts-0000-0127.safetensors``)::

    model.language_model.layers.{L}.mlp.experts.{e}.gate_proj.weight        u8  [I,   H/2]
    model.language_model.layers.{L}.mlp.experts.{e}.gate_proj.weight_scale  e4m3[I,   H/16]
    model.language_model.layers.{L}.mlp.experts.{e}.gate_proj.weight_scale_2 f32 []
    model.language_model.layers.{L}.mlp.experts.{e}.gate_proj.input_scale   f32 []
    ... up_proj (same shapes) ...
    model.language_model.layers.{L}.mlp.experts.{e}.down_proj.weight        u8  [H,   I/2]
    model.language_model.layers.{L}.mlp.experts.{e}.down_proj.weight_scale  e4m3[H,   I/16]

SGLang stacks these into (``modelopt_quant.ModelOptNvFp4FusedMoEMethod.create_weights``)::

    w13_weight        [E, 2I, H/2]  uint8            rows [0:I] = gate (w1), [I:2I] = up (w3)
    w13_weight_scale  [E, 2I, H/16] float8_e4m3
    w2_weight         [E, H,  I/2]  uint8
    w2_weight_scale   [E, H,  I/16] float8_e4m3
    w13_weight_scale_2 [E, 2] f32 ; w2_weight_scale_2 [E] f32
    w13_input_scale    [E, 2] f32 ; w2_input_scale    [E] f32
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import torch

from .model_shapes import DEFAULT_CKPT, MoEShape, read_checkpoint_shape

PREFIX = "model.language_model.layers.{layer}.mlp.experts.{e}."


@dataclass
class LayerWeights:
    """One layer's routed experts in SGLang's stacked NVFP4 layout."""

    shape: MoEShape
    layer: int
    w13_weight: torch.Tensor
    w13_weight_scale: torch.Tensor
    w2_weight: torch.Tensor
    w2_weight_scale: torch.Tensor
    w13_weight_scale_2: torch.Tensor
    w2_weight_scale_2: torch.Tensor
    w13_input_scale: torch.Tensor
    w2_input_scale: torch.Tensor
    # filled by prepare_runtime_scales()
    w13_blockscale_swizzled: Optional[torch.Tensor] = None
    w2_blockscale_swizzled: Optional[torch.Tensor] = None
    g1_alphas: Optional[torch.Tensor] = None
    g2_alphas: Optional[torch.Tensor] = None
    w13_input_scale_quant: Optional[torch.Tensor] = None
    w2_input_scale_quant: Optional[torch.Tensor] = None
    meta: dict = field(default_factory=dict)

    @property
    def num_experts(self) -> int:
        return self.w13_weight.shape[0]

    def to(self, device) -> "LayerWeights":
        for k, v in list(self.__dict__.items()):
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))
        return self


def shard_files(ckpt: str, layer: int):
    pat = os.path.join(ckpt, f"layer-{layer:05d}-experts-*.safetensors")
    files = sorted(glob.glob(pat))
    if not files:
        raise FileNotFoundError(f"no expert shards for layer {layer} under {ckpt}")
    return files


def _shard_range(path: str):
    """(expert_start, expert_end) from the shard's safetensors metadata header."""
    import struct

    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    md = hdr.get("__metadata__", {})
    return int(md["expert_start"]), int(md["expert_end"])


def load_layer(
    layer: int = 4,
    ckpt: str = DEFAULT_CKPT,
    device: str = "cpu",
    num_experts: Optional[int] = None,
    shape: Optional[MoEShape] = None,
) -> LayerWeights:
    """Load `num_experts` (default: all) experts of `layer`.

    ``device="cpu"`` is fully supported and is what the CPU tests use.
    """
    from safetensors import safe_open

    shape = shape or read_checkpoint_shape(ckpt)
    E = shape.num_experts if num_experts is None else min(num_experts, shape.num_experts)
    H, I = shape.hidden, shape.intermediate

    w13 = torch.empty((E, 2 * I, H // 2), dtype=torch.uint8, device=device)
    w13sf = torch.empty((E, 2 * I, H // 16), dtype=torch.float8_e4m3fn, device=device)
    w2 = torch.empty((E, H, I // 2), dtype=torch.uint8, device=device)
    w2sf = torch.empty((E, H, I // 16), dtype=torch.float8_e4m3fn, device=device)
    w13s2 = torch.empty((E, 2), dtype=torch.float32, device=device)
    w2s2 = torch.empty((E,), dtype=torch.float32, device=device)
    w13in = torch.empty((E, 2), dtype=torch.float32, device=device)
    w2in = torch.empty((E,), dtype=torch.float32, device=device)

    seen = 0
    for path in shard_files(ckpt, layer):
        lo, hi = _shard_range(path)
        if lo >= E:
            continue
        with safe_open(path, framework="pt", device="cpu") as f:
            for e in range(lo, min(hi, E)):
                p = PREFIX.format(layer=layer, e=e)
                w13[e, :I] = f.get_tensor(p + "gate_proj.weight").to(device)
                w13[e, I:] = f.get_tensor(p + "up_proj.weight").to(device)
                w13sf[e, :I] = f.get_tensor(p + "gate_proj.weight_scale").to(device)
                w13sf[e, I:] = f.get_tensor(p + "up_proj.weight_scale").to(device)
                w2[e] = f.get_tensor(p + "down_proj.weight").to(device)
                w2sf[e] = f.get_tensor(p + "down_proj.weight_scale").to(device)
                w13s2[e, 0] = f.get_tensor(p + "gate_proj.weight_scale_2")
                w13s2[e, 1] = f.get_tensor(p + "up_proj.weight_scale_2")
                w2s2[e] = f.get_tensor(p + "down_proj.weight_scale_2")
                w13in[e, 0] = f.get_tensor(p + "gate_proj.input_scale")
                w13in[e, 1] = f.get_tensor(p + "up_proj.input_scale")
                w2in[e] = f.get_tensor(p + "down_proj.input_scale")
                seen += 1
    if seen != E:
        raise RuntimeError(f"loaded {seen} experts, expected {E}")

    return LayerWeights(
        shape=shape, layer=layer,
        w13_weight=w13, w13_weight_scale=w13sf,
        w2_weight=w2, w2_weight_scale=w2sf,
        w13_weight_scale_2=w13s2, w2_weight_scale_2=w2s2,
        w13_input_scale=w13in, w2_input_scale=w2in,
        meta={"ckpt": ckpt, "num_experts_loaded": E},
    )


def swizzle_blockscale_cpu(scale: torch.Tensor) -> torch.Tensor:
    """Device-agnostic copy of ``sglang...quantization.utils.swizzle_blockscale``.

    The SGLang version ends with ``.cuda()``; this one keeps the input device so
    the layout can be checked on CPU.  The permutation is identical:
    ``[B, M/128, 4, 32, K/4, 4] -> [B, M/128, K/4, 32, 4, 4]``.
    """
    assert scale.dtype == torch.float8_e4m3fn
    nd = scale.ndim
    if nd == 2:
        scale = scale.unsqueeze(0)
    assert scale.ndim == 3
    B, M, K = scale.shape
    up = lambda x, m: (x + m - 1) // m * m
    Mp, Kp = up(M, 128), up(K, 4)
    padded = torch.zeros((B, Mp, Kp), dtype=scale.dtype, device=scale.device)
    padded[:B, :M, :K] = scale
    padded = padded.reshape(B, Mp // 128, 4, 32, Kp // 4, 4)
    sw = padded.permute((0, 1, 4, 3, 2, 5)).contiguous()
    return sw.reshape(Mp, Kp) if nd == 2 else sw.reshape(B, Mp, Kp)


def prepare_runtime_scales(lw: LayerWeights) -> LayerWeights:
    """Reproduce ``ModelOptNvFp4FusedMoEMethod.process_weights_after_loading``
    for the ``flashinfer_cutlass`` branch.

    * ``w13_input_scale``/``w2_input_scale`` collapse to a scalar via ``.max()``
      (modelopt_quant.py, ``if self.enable_flashinfer_cutlass_moe ...``).
    * ``g1_alphas = w13_input_scale * w13_weight_scale_2[:, 0]`` (gate column),
      ``g2_alphas = w2_input_scale * w2_weight_scale_2``.
    * ``w*_input_scale_quant = 1 / w*_input_scale``.
    * block scales are swizzled into the 128x4 layout.
    """
    w13_in = lw.w13_input_scale.max().to(torch.float32)
    w2_in = lw.w2_input_scale.max().to(torch.float32)
    lw.g1_alphas = (w13_in * lw.w13_weight_scale_2[:, 0]).to(torch.float32).contiguous()
    lw.g2_alphas = (w2_in * lw.w2_weight_scale_2).to(torch.float32).contiguous()
    lw.w13_input_scale_quant = (1.0 / w13_in).to(torch.float32).reshape(())
    lw.w2_input_scale_quant = (1.0 / w2_in).to(torch.float32).reshape(())
    lw.w13_blockscale_swizzled = swizzle_blockscale_cpu(lw.w13_weight_scale)
    lw.w2_blockscale_swizzled = swizzle_blockscale_cpu(lw.w2_weight_scale)
    return lw


# --------------------------------------------------------------------------
# Synthetic weights: same shapes/dtypes, random content.  Lets the harness and
# the CPU tests run with no checkpoint and no 1.4 GB read.
# --------------------------------------------------------------------------
def synth_layer(shape: MoEShape, num_experts: int, device: str = "cpu",
                seed: int = 0) -> LayerWeights:
    g = torch.Generator(device="cpu").manual_seed(seed)
    E, H, I = num_experts, shape.hidden, shape.intermediate
    ru8 = lambda *s: torch.randint(0, 256, s, generator=g, dtype=torch.uint8).to(device)
    rsf = lambda *s: (torch.rand(s, generator=g) * 2 + 0.5).to(torch.float8_e4m3fn).to(device)
    return LayerWeights(
        shape=shape, layer=-1,
        w13_weight=ru8(E, 2 * I, H // 2), w13_weight_scale=rsf(E, 2 * I, H // 16),
        w2_weight=ru8(E, H, I // 2), w2_weight_scale=rsf(E, H, I // 16),
        w13_weight_scale_2=torch.full((E, 2), 0.01, device=device),
        w2_weight_scale_2=torch.full((E,), 0.01, device=device),
        w13_input_scale=torch.full((E, 2), 0.05, device=device),
        w2_input_scale=torch.full((E,), 0.05, device=device),
        meta={"synthetic": True, "num_experts_loaded": E},
    )
