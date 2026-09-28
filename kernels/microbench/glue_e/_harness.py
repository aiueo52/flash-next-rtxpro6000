"""Shared CUPTI plumbing for the glue-E benches.

`capture` / `cupti_kernel_us` come from bench/w8a16v2/harness.py (byte-identical to
the copy in ~/tools/sglang-qsa-ring that the spec names); `cupti_by_kernel` is the
same measurement broken out per kernel name, which is what a "did the fused kernel
replace two launches" claim needs.
"""

from __future__ import annotations

import importlib.util
import pathlib
import statistics
import sys

import torch

_HARNESS = pathlib.Path(__file__).resolve().parents[1] / "w8a16v2" / "harness.py"
_spec = importlib.util.spec_from_file_location("_glue_e_w8a16_harness", _HARNESS)
_h = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _h
_spec.loader.exec_module(_h)

capture = _h.capture
cupti_kernel_us = _h.cupti_kernel_us
time_graph = _h.time_graph
n_copies = _h.n_copies
make_copies = _h.make_copies
ClockSampler = _h.ClockSampler


def cupti_by_kernel(call, args_list, min_seconds=0.2, spin_seconds=0.25, max_events=4000):
    """{kernel name: (median us, launches per call)} over CUDA-graph replays.

    Same protocol as `cupti_kernel_us`: replay a captured graph under continuous
    load so the SMs stay at boost, then read CUPTI kernel durations. Also returns
    the summed per-call device time, which is the number a "two launches became
    one" claim lives or dies on.
    """
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    g = capture(call, args_list)
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    g.replay()
    end.record()
    torch.cuda.synchronize()
    per_replay = max(start.elapsed_time(end) / 1e3, 1e-6)
    for _ in range(max(1, int(spin_seconds / per_replay))):
        g.replay()
    torch.cuda.synchronize()

    reps = max(1, int(min_seconds / per_replay))
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            g.replay()
        torch.cuda.synchronize()

    by_name: dict[str, list[float]] = {}
    for e in prof.events():
        if e.device_type == DeviceType.CUDA:
            by_name.setdefault(e.key, []).append(e.device_time)
    del g

    calls = reps * len(args_list)
    out = {}
    total = 0.0
    for name, durs in by_name.items():
        med = statistics.median(durs)
        out[name] = (med, len(durs) / calls)
        total += sum(durs) / calls
    return out, total


def report(tag: str, res, total: float) -> None:
    print(f"  {tag:<34s} per-call device time = {total:8.2f} us")
    for name, (med, n) in sorted(res.items(), key=lambda kv: -kv[1][0] * kv[1][1]):
        print(f"      {name[:56]:<56s} n/call={n:4.1f} med={med:7.2f} us")


def clocks_note() -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        return f"SM clock now/max: {out}"
    except Exception:
        return "SM clock: unknown"
