"""CPU-only tests.  Run:

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=bench \
      /home/user/tools/sglang-rtxpro6000/.venv/bin/python -m pytest \
      bench/moe_smallm/tests_cpu.py -q

Nothing here touches CUDA.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

from . import model_shapes as MS
from . import refs
from . import routing as R
from .bench_moe import bytes_model_table, main, parse_args
from .runners import build_inputs
from .weights import load_layer, prepare_runtime_scales, swizzle_blockscale_cpu, synth_layer

CKPT_PRESENT = os.path.exists(os.path.join(MS.DEFAULT_CKPT, "config.json"))


def test_shape_matches_checkpoint():
    if not CKPT_PRESENT:
        pytest.skip("checkpoint not present")
    s = MS.read_checkpoint_shape()
    assert (s.num_experts, s.top_k, s.hidden, s.intermediate, s.num_layers) == (
        512, 10, 2560, 640, 48)
    assert s.w13_shape == (512, 1280, 1280)
    assert s.w2_shape == (512, 2560, 320)
    assert s.w13_scale_shape == (512, 1280, 160)
    assert s.w2_scale_shape == (512, 2560, 40)


def test_bytes_model():
    s = MS.MoEShape()
    # 4 bits/weight + 1 e4m3 scale per 16 weights = 4.5 bits/weight
    n = 2 * 640 * 2560 + 2560 * 640
    assert s.bytes_per_expert == n * 9 // 16
    assert s.bytes_per_expert == 2_764_800
    assert s.bytes_gemm1_per_expert == 2 * s.bytes_gemm2_per_expert


def test_expected_distinct_bounds():
    s = MS.MoEShape()
    assert MS.expected_distinct_experts(s, 1) == pytest.approx(10.0)
    assert MS.expected_distinct_experts(s, 4) == pytest.approx(38.84, abs=0.05)
    assert MS.expected_distinct_experts(s, 16) == pytest.approx(138.57, abs=0.05)
    # monotone and bounded by T*k and by E
    for t in range(1, 20):
        d = MS.expected_distinct_experts(s, t)
        assert 0 < d <= min(t * s.top_k, s.num_experts)


def test_bandwidth_inversion_is_selfconsistent():
    s = MS.MoEShape()
    d, us = 38.84, 67.7
    g = MS.achieved_gbps(s, us, d)
    assert MS.implied_distinct_experts(s, us, g) == pytest.approx(d, rel=1e-9)


def test_trace_row_falsifies_independence_at_T16():
    """T=16 cannot be running independent routing: it would need >peak DRAM."""
    rows = {r["T"]: r for r in bytes_model_table(MS.MoEShape())}
    assert rows[16]["pct_peak_if_independent"] > 100.0
    assert rows[4]["pct_peak_if_independent"] < 100.0
    assert rows[16]["D_max_at_peak"] < rows[16]["E_distinct_independent"]


@pytest.mark.parametrize("t", [1, 2, 4, 8, 16])
def test_routing_shapes_and_uniqueness(t):
    g = torch.Generator().manual_seed(0)
    ids = R.independent(t, 10, 512, g)
    assert ids.shape == (t, 10) and ids.dtype == torch.int32
    for row in ids:
        assert len(set(row.tolist())) == 10


@pytest.mark.parametrize("d", [10, 25, 40, 80, 139, 160])
def test_with_distinct_is_exact(d):
    g = torch.Generator().manual_seed(1)
    ids = R.with_distinct(16, 10, 512, d, g)
    assert int(torch.unique(ids).numel()) == d
    for row in ids:
        assert len(set(row.tolist())) == 10


def test_with_distinct_rejects_impossible():
    g = torch.Generator().manual_seed(1)
    with pytest.raises(ValueError):
        R.with_distinct(4, 10, 512, 9, g)
    with pytest.raises(ValueError):
        R.with_distinct(4, 10, 512, 41, g)


def test_correlated_reduces_distinct():
    g = torch.Generator().manual_seed(2)
    a = int(torch.unique(R.correlated(16, 10, 512, 0.0, g)).numel())
    b = int(torch.unique(R.correlated(16, 10, 512, 0.9, g)).numel())
    assert b < a


def test_disjointify_keeps_ids_in_range_and_spreads_repeats():
    # the case it exists for: the same routing replayed n times
    g = torch.Generator().manual_seed(3)
    one = R.independent(4, 10, 512, g)
    rs = R.disjointify([one] * 16, 512)
    assert all(int(r.max()) < 512 and int(r.min()) >= 0 for r in rs)
    union = len({int(v) for r in rs for v in r.flatten()})
    assert union > int(torch.unique(one).numel()) * 8


def test_rotation_working_set_exceeds_4x_l2():
    s = MS.MoEShape()
    rs = R.disjointify(R.make_routings("independent", 16, 4, 10, 512, seed=3), 512)
    union = len({int(v) for r in rs for v in r.flatten()})
    assert union * s.bytes_per_expert >= 4 * MS.L2_BYTES


def test_swizzle_blockscale_shape_and_permutation():
    sf = torch.arange(1 * 128 * 8, dtype=torch.float32).reshape(1, 128, 8)
    sf = sf.to(torch.float8_e4m3fn)
    sw = swizzle_blockscale_cpu(sf)
    assert sw.shape == (1, 128, 8)
    # same multiset of values, different order
    assert sorted(sw.flatten().float().tolist()) == sorted(sf.flatten().float().tolist())
    assert not torch.equal(sw.reshape(-1), sf.reshape(-1))


def test_swizzle_pads_to_128x4():
    sf = torch.zeros(2, 130, 5, dtype=torch.float8_e4m3fn)
    assert swizzle_blockscale_cpu(sf).shape == (2, 256, 8)


def test_unpack_fp4_low_nibble_first():
    p = torch.tensor([[0x21, 0xF8]], dtype=torch.uint8)
    v = refs.unpack_fp4(p)[0].tolist()
    assert v == [0.5, 1.0, -0.0, -6.0]


def test_dequant_weight_shapes():
    n, k = 32, 64
    packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8)
    sf = torch.ones(n, k // 16, dtype=torch.float8_e4m3fn)
    w = refs.dequant_weight(packed, sf, 0.5)
    assert w.shape == (n, k)
    assert w.abs().max() <= 6.0 * 0.5 + 1e-6


def test_synth_layer_matches_sglang_layout():
    s = MS.MoEShape()
    lw = prepare_runtime_scales(synth_layer(s, 8))
    assert lw.w13_weight.shape == (8, 1280, 1280)
    assert lw.w2_weight.shape == (8, 2560, 320)
    assert lw.w13_blockscale_swizzled.shape == (8, 1280, 160)
    assert lw.w2_blockscale_swizzled.shape == (8, 2560, 40)
    assert lw.g1_alphas.shape == (8,) and lw.g2_alphas.shape == (8,)
    assert lw.w13_input_scale_quant.shape == ()


def test_load_real_layer_slice():
    if not CKPT_PRESENT:
        pytest.skip("checkpoint not present")
    lw = prepare_runtime_scales(load_layer(4, num_experts=2, device="cpu"))
    assert lw.w13_weight.shape == (2, 1280, 1280)
    assert lw.w13_weight_scale.dtype == torch.float8_e4m3fn
    assert lw.w2_weight.shape == (2, 2560, 320)
    assert torch.isfinite(lw.g1_alphas).all()
    assert float(lw.w13_input_scale_quant) > 0


def test_reference_runs_on_tiny_real_slice():
    if not CKPT_PRESENT:
        pytest.skip("checkpoint not present")
    lw = load_layer(4, num_experts=2, device="cpu")
    g = torch.Generator().manual_seed(0)
    ids = R.independent(1, 2, 2, g)
    w = R.uniform_weights([ids])[0]
    x = torch.randn((1, lw.shape.hidden), generator=g).to(torch.bfloat16)
    y = refs.moe_reference(lw, x, ids, w)
    assert y.shape == (1, lw.shape.hidden)
    assert torch.isfinite(y.float()).all()
    assert float(y.float().abs().max()) > 0


def test_compare_identical_is_perfect():
    a = torch.randn(4, 16, dtype=torch.float32).to(torch.bfloat16)
    r = refs.compare(a, a)
    assert r["max_rel_err"] == 0.0 and r["cosine"] == pytest.approx(1.0, abs=1e-6)


def test_build_inputs_cpu():
    lw = prepare_runtime_scales(synth_layer(MS.MoEShape(), 32))
    rs = R.disjointify(R.make_routings("independent", 4, 16, 10, 512, seed=0), 32)
    ins = build_inputs(lw, rs, R.uniform_weights(rs), device="cpu")
    assert len(ins) == 4
    assert ins[0].hidden_states.shape == (16, 2560)
    assert ins[0].topk_ids.dtype == torch.int32
    assert int(ins[0].topk_ids.max()) < 32
    assert ins[0].output.shape == (16, 2560)


def test_cli_parses():
    a = parse_args(["--mode", "run", "--widths", "4,8,16", "--sweep-distinct", "10,40"])
    assert a.mode == "run" and a.widths == "4,8,16"
    a2 = parse_args([])
    assert a2.mode == "dryrun" and a2.candidate == "flashinfer_cutlass"


def test_cli_model_mode_runs():
    assert main(["--mode", "model"]) == 0


def test_cli_dryrun_synthetic_runs():
    assert main(["--mode", "dryrun", "--synthetic", "--num-experts", "8",
                 "--widths", "4", "--rotation", "2"]) == 0


# --- SGLang import-order regression -------------------------------------
# `moe_runner/flashinfer_trtllm.py:110` only imports `quantization.fp4_utils`
# when `is_flashinfer_available()` is True, so the circular import is invisible
# on a CPU-only run.  These tests force that flag in a subprocess to reproduce
# the GPU failure and prove the fix, without touching CUDA.

_REPRO = r'''
import sys
import sglang.srt.utils.common as C
C.is_flashinfer_available = lambda: True
import sglang.srt.utils as U
U.is_flashinfer_available = C.is_flashinfer_available
if sys.argv[1] == "fixed":
    from moe_smallm.runners import ensure_sglang_imports
    ensure_sglang_imports()
try:
    import sglang.srt.layers.moe.moe_runner.flashinfer_cutlass  # noqa: F401
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (  # noqa: F401
        get_activation_type,
    )
    print("RESULT OK")
except ImportError as exc:
    print("RESULT ImportError", exc)
'''


def _run_repro(mode):
    import subprocess

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, "-c", _REPRO, mode], capture_output=True,
                       text=True, env=env, timeout=900)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT"):
            return line
    return f"NO RESULT (rc={r.returncode}) {r.stderr[-400:]}"


def _sglang_importable():
    import importlib.util

    return importlib.util.find_spec("sglang") is not None


@pytest.mark.slow
def test_naive_import_order_still_breaks():
    """Guards the premise: without the fix the cycle really does fire."""
    if not _sglang_importable():
        pytest.skip("sglang not importable")
    assert _run_repro("broken").startswith("RESULT ImportError")


@pytest.mark.slow
def test_ensure_sglang_imports_breaks_the_cycle():
    if not _sglang_importable():
        pytest.skip("sglang not importable")
    assert _run_repro("fixed") == "RESULT OK"


# --- route_logger / analyze_routes over the committed measurement data -----

RUNS = os.path.join(os.path.dirname(__file__), "..", "..", "runs")


def _have(profile):
    return os.path.exists(os.path.join(RUNS, f"routes-{profile}-code-edit.json"))


@pytest.mark.parametrize("profile,t,indep", [("w4", 4, 38.84), ("w16", 16, 138.57)])
def test_measured_distinct_is_far_below_independent(profile, t, indep):
    from .analyze_routes import report

    if not _have(profile):
        pytest.skip("routing snapshots not present")
    rows = report(profile, RUNS, t)
    all_ = rows["ALL"]
    assert all_["calls"] > 50_000
    assert 0 < all_["mean"] < indep, "measured D must be below the independent bound"
    # section 8.2: 73% of independent at T=4, 50% at T=16
    frac = all_["mean"] / indep
    assert 0.40 < frac < 0.85
    assert all_["max"] <= t * 10
    assert all_["min"] >= 10


def test_overlap_structure_has_no_shared_core():
    """At T=16 almost no expert is common to all 16 chain tokens (0.12 of them)."""
    from .analyze_routes import report

    if not _have("w16"):
        pytest.skip("routing snapshots not present")
    rows = report("w16", RUNS, 16)
    assert rows["ALL"]["shared_by_all"] < 1.0
    bm = rows["ALL"]["by_mult"]
    # the mean distinct count is the sum over multiplicities
    assert sum(bm.values()) == pytest.approx(rows["ALL"]["mean"], rel=0.02)


def test_analyze_routes_diff_is_a_difference():
    from .analyze_routes import _load, diff

    if not _have("w4"):
        pytest.skip("routing snapshots not present")
    base = _load(os.path.join(RUNS, "routes-w4-baseline.json"))
    first = _load(os.path.join(RUNS, "routes-w4-code-edit.json"))
    cum = diff(None, first, 4)
    inc = diff(base, first, 4)
    assert inc["calls"] < cum["calls"]
    assert inc["calls"] == cum["calls"] - base["by_T"]["4"]["calls"]
