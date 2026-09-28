"""Shared pieces for the L2-prefetch feasibility benches.

Timing protocol follows bench/megakernel/harness.py (graph replay, spin-up, CUPTI
per-kernel medians), extended with
  * `capture()` that keeps the cudaGraph_t (keep_graph=True) so node attributes can
    be inspected / edited and the graph can be instantiated with node priorities,
  * `cupti_positions()` that reports per-*position* kernel medians (kernels with the
    same name at different points of the replay are kept apart).
"""

from __future__ import annotations

import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "megakernel"))

from harness import ClockSampler, emit, gpu_note  # noqa: E402,F401
from l2ctl import (  # noqa: E402,F401
    GRAPH_FLAG_USE_NODE_PRIORITY, MODE_CS, MODE_EVICT_FIRST, MODE_EVICT_LAST, MODE_NORMAL,
    NORMAL, PERSISTING, STREAMING, L2Ctl, sector_range,
)

MB = 1 << 20
L2_BYTES = 128 << 20
DEV = "cuda"


def gemv_fns():
    from sglang.srt.layers.quantization.w8a16_gemv import prealloc, w8a16_gemv
    prealloc(torch.device(DEV))
    return w8a16_gemv


def make_weight(N: int, K: int, M: int):
    """Production layout: fp8 [N, K] *contiguous* (fp8.py stores layer.weight as the [K, N]
    transposed view of it and hands `.t()` to the GEMV), per-channel fp32 scale [N].

    NOTE: the first bench runs (21:05-21:41) used a [K, N]-contiguous weight by mistake; the
    GEMV then takes its W_KN path with a different tile and the split-K out_proj shape ran
    3.5x slower (50 us). Those numbers are superseded by the *_v2 result files."""
    g = torch.Generator(device=DEV).manual_seed(N * 7919 + K)
    w = (torch.randn(N, K, device=DEV, generator=g) / 8).to(torch.float8_e4m3fn)
    s = (torch.rand(N, device=DEV, generator=g) + 0.5).float() / 64
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16, generator=g) / 8
    y = torch.empty(M, N, device=DEV, dtype=torch.bfloat16)
    return w, s, x, y


class Graph:
    """A captured graph, replayable either through torch (plain instantiate) or through
    l2ctl with cudaGraphInstantiateFlagUseNodePriority."""

    def __init__(self, ctl: L2Ctl, call, warmup=2, node_priority=False, capture_stream=None):
        self.ctl = ctl
        self.exec = None
        s = capture_stream or torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                call()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.g = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(self.g, stream=s):
            call()
        torch.cuda.synchronize()
        if node_priority:
            self.exec = ctl.instantiate(self.g, GRAPH_FLAG_USE_NODE_PRIORITY)
        else:
            self.g.instantiate()
        self.replay()
        torch.cuda.synchronize()

    def replay(self):
        if self.exec is not None:
            self.ctl.graph_launch(self.exec, torch.cuda.current_stream())
        else:
            self.g.replay()

    def nodes(self):
        return self.ctl.nodes(self.g)

    def close(self):
        if self.exec is not None:
            torch.cuda.synchronize()
            self.ctl.exec_destroy(self.exec)
            self.exec = None
        self.g = None


def time_replay(gr: Graph, min_seconds=0.4, spin_seconds=0.2) -> float:
    """Microseconds per replay (event-timed, after a spin-up)."""
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record(); gr.replay(); end.record(); torch.cuda.synchronize()
    per = max(start.elapsed_time(end) / 1e3, 1e-6)
    for _ in range(max(1, int(spin_seconds / per))):
        gr.replay()
    torch.cuda.synchronize()
    iters = max(5, int(min_seconds / per))
    start.record()
    for _ in range(iters):
        gr.replay()
    end.record(); torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / iters


def cupti_positions(gr: Graph, reps=40, spin_seconds=0.2, max_events=6000):
    """Per-position kernel medians over `reps` replays: [(name, median_us, n)].

    Kernels are ordered by start time inside each replay; position k of every replay
    is the same graph node when the graph is a chain (for forked graphs the order of
    concurrent kernels may vary, so callers should match by name in that case).
    """
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record(); gr.replay(); end.record(); torch.cuda.synchronize()
    per = max(start.elapsed_time(end) / 1e3, 1e-6)
    for _ in range(max(1, int(spin_seconds / per))):
        gr.replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            gr.replay()
        torch.cuda.synchronize()
    evs = [e for e in prof.events() if e.device_type == DeviceType.CUDA]
    evs.sort(key=lambda e: e.time_range.start)
    if not evs or len(evs) % reps != 0:
        # fall back to by-name aggregation
        by = {}
        for e in evs:
            by.setdefault(e.key, []).append(e.device_time)
        return [(k, statistics.median(v), len(v) / reps) for k, v in by.items()], evs
    per_replay = len(evs) // reps
    out = []
    for k in range(per_replay):
        durs = [evs[r * per_replay + k].device_time for r in range(reps)]
        out.append((evs[k].key, statistics.median(durs), 1.0))
    return out, evs


def short(name: str) -> str:
    n = name.replace("void ", "")
    for tag in ("touch_table_kernel", "touch_kernel", "_w8a16_gemv_kernel", "_tiny_kernel", "_hcish_kernel"):
        if tag in n:
            i = n.find(tag)
            j = n.find("(", i)
            return n[i:j] if j > 0 else n[i:i + 24]
    return n[:40]


def rate_gbs(nbytes: float, us: float) -> float:
    return nbytes / us / 1e3


def banner(msg):
    print(f"\n=== {msg}  ({time.strftime('%H:%M:%S')})", flush=True)
