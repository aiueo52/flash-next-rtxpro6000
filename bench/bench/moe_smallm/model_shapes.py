"""Shapes and byte accounting for the Qwen3.8-Flash-Next-NVFP4 routed experts.

Every number here is read off the checkpoint, not assumed.  CPU-only.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

DEFAULT_CKPT = os.path.expanduser(
    "~/models/RadixArk/Qwen3.8-Flash-Next-NVFP4"
)

# NVFP4: 2 fp4 codes per byte, one e4m3 block scale per 16 codes, one fp32
# per-tensor "global" scale (weight_scale_2 / input_scale) per expert-half.
FP4_PER_BYTE = 2
SF_BLOCK = 16
SF_BYTES = 1  # float8_e4m3


@dataclass(frozen=True)
class MoEShape:
    num_experts: int = 512
    top_k: int = 10
    hidden: int = 2560           # K of gemm1, N of gemm2
    intermediate: int = 640      # moe_intermediate_size; N/2 of gemm1, K of gemm2
    num_layers: int = 48
    dtype_act: str = "bfloat16"

    # ---- derived weight shapes, in SGLang's stacked layout ----
    @property
    def w13_shape(self):            # [E, 2*I, H//2] uint8
        return (self.num_experts, 2 * self.intermediate, self.hidden // FP4_PER_BYTE)

    @property
    def w13_scale_shape(self):      # [E, 2*I, H//16] float8_e4m3
        return (self.num_experts, 2 * self.intermediate, self.hidden // SF_BLOCK)

    @property
    def w2_shape(self):             # [E, H, I//2] uint8
        return (self.num_experts, self.hidden, self.intermediate // FP4_PER_BYTE)

    @property
    def w2_scale_shape(self):       # [E, H, I//16] float8_e4m3
        return (self.num_experts, self.hidden, self.intermediate // SF_BLOCK)

    # ---- bytes ----
    @property
    def bytes_gemm1_per_expert(self) -> int:
        """gate+up: fp4 codes + e4m3 block scales."""
        n = 2 * self.intermediate * self.hidden
        return n // FP4_PER_BYTE + (n // SF_BLOCK) * SF_BYTES

    @property
    def bytes_gemm2_per_expert(self) -> int:
        n = self.hidden * self.intermediate
        return n // FP4_PER_BYTE + (n // SF_BLOCK) * SF_BYTES

    @property
    def bytes_per_expert(self) -> int:
        return self.bytes_gemm1_per_expert + self.bytes_gemm2_per_expert

    @property
    def bytes_all_experts_per_layer(self) -> int:
        return self.num_experts * self.bytes_per_expert


def expected_distinct_experts(shape: MoEShape, t_tokens: int) -> float:
    """E[#distinct experts] for T tokens x top-k, assuming *independent* routing.

    P(expert e unused by one token) = 1 - k/E exactly (exchangeability of a
    top-k draw), and independence across tokens gives

        E[D] = E * (1 - (1 - k/E)^T)

    This is an **upper bound** in practice: chain tokens in a speculative
    verify share context and route to overlapping experts, so the true D is
    lower (see specs/MOE_SMALLM_SPEC.md section 3).
    """
    e, k = shape.num_experts, shape.top_k
    return e * (1.0 - (1.0 - k / e) ** t_tokens)


def implied_distinct_experts(shape: MoEShape, us_per_layer: float, gbps: float) -> float:
    """Invert the bytes model: D that a per-layer kernel time implies at `gbps`."""
    return (us_per_layer * 1e-6 * gbps * 1e9) / shape.bytes_per_expert


def achieved_gbps(shape: MoEShape, us_per_layer: float, distinct: float) -> float:
    return distinct * shape.bytes_per_expert / (us_per_layer * 1e-6) / 1e9


def read_checkpoint_shape(ckpt: str = DEFAULT_CKPT) -> MoEShape:
    """Read the real shape out of config.json.  CPU-only, no torch needed."""
    with open(os.path.join(ckpt, "config.json")) as f:
        cfg = json.load(f)
    t = cfg["text_config"]
    return MoEShape(
        num_experts=t["num_experts"],
        top_k=t["num_experts_per_tok"],
        hidden=t["hidden_size"],
        intermediate=t["moe_intermediate_size"],
        num_layers=t["num_hidden_layers"],
    )


# --- device constants (RTX PRO 6000 Blackwell Max-Q, sm_120) --------------
# GDDR7, 512-bit bus, 14001 MHz reported by nvidia-smi => 28 Gbps effective.
PEAK_DRAM_GBPS = 28.0 * 512 / 8          # 1792.0 GB/s theoretical
# Copy roof (read+write) measured on this box; roofline.py in this directory re-measures it
# (1456 GB/s in docs/lab-notes/G1_LOG.md):
MEASURED_COPY_GBPS = 1461.0
L2_BYTES = 128 << 20                      # 128 MiB
NUM_SMS = 188
