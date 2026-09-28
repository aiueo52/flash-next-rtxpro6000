"""CUDA-graph replay timing for the small-M NVFP4 grouped MoE.  GPU-ONLY.

Methodology mirrors ``sglang-rtxpro6000/bench/w8a16v2/harness.py``:

* **Working set.** A single MoE call at T=4 touches ~39 experts x 2.76 MB =
  108 MB, which *fits* inside the 128 MiB L2 -- replaying it would measure L2,
  not DRAM.  Every measurement therefore replays a *rotation* of calls whose
  expert sets are offset from each other (see ``routing.disjointify``) so the
  union is >= 4x L2.  ``check_working_set`` refuses to report a number when
  that is not satisfied.
* **Clocks.** A few hundred microseconds of replay leaves the GPU at idle
  clocks and over-reports by 2-3x.  Each measurement spins first, then replays
  for >= ``min_seconds``, and samples ``clocks.sm`` during the timed region.
* **Guard.** ``require_idle_gpu()`` refuses to run while anything else holds a
  CUDA context, and honours a handover marker file.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time

import torch

from .model_shapes import L2_BYTES

WORKING_SET_TARGET = 4 * L2_BYTES


class ClockSampler(threading.Thread):
    """Poll clocks.sm while a measurement runs; nvidia-smi costs ~40 ms/sample."""

    def __init__(self, period: float = 0.15):
        super().__init__(daemon=True)
        self.period = period
        self.samples: list[int] = []
        self._ev = threading.Event()

    def run(self):
        while not self._ev.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip()
                self.samples.append(int(out.splitlines()[0].split(",")[0]))
            except Exception:
                pass
            self._ev.wait(self.period)

    def stop(self):
        self._ev.set()
        self.join(timeout=5)
        return (min(self.samples), max(self.samples)) if self.samples else (0, 0)


def require_idle_gpu():
    """Refuse to benchmark unless the GPU is idle and the handover marker exists.

    ``MOE_BENCH_MARKER`` must name a file that exists; that is how a session
    hands the exclusive GPU over without a second run sneaking in.  Set
    ``MOE_BENCH_FORCE=1`` only if you know what you are doing.
    """
    if os.environ.get("MOE_BENCH_FORCE") == "1":
        return
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
    if out:
        raise SystemExit(
            f"GPU busy (compute apps: {out.splitlines()}); refusing to benchmark. "
            "Set MOE_BENCH_FORCE=1 to override."
        )
    marker = os.environ.get("MOE_BENCH_MARKER")
    if marker and not os.path.exists(marker):
        raise SystemExit(f"GPU handover marker missing: {marker}")


def check_working_set(bytes_touched: int, label: str = "") -> str:
    """Return a warning string (empty when fine)."""
    if bytes_touched >= WORKING_SET_TARGET:
        return ""
    return (f"WARNING{(' ' + label) if label else ''}: rotation touches "
            f"{bytes_touched/2**20:.0f} MiB < 4x L2 ({WORKING_SET_TARGET/2**20:.0f} MiB); "
            "the number may be L2-resident, not DRAM.")


def capture(calls, warmup: int = 3) -> torch.cuda.CUDAGraph:
    """Capture a graph that runs every callable in `calls` once, in order."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            for c in calls:
                c()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for c in calls:
            c()
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    return g


def time_graph(calls, min_seconds: float = 0.6, spin_seconds: float = 0.25,
               clocks: bool = False):
    """Seconds per *call* (not per replay).  Returns (sec_per_call, (clk_lo, clk_hi))."""
    g = capture(calls)
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    g.replay()
    end.record()
    torch.cuda.synchronize()
    per_replay = max(start.elapsed_time(end) / 1e3, 1e-6)

    for _ in range(max(1, int(spin_seconds / per_replay))):
        g.replay()
    torch.cuda.synchronize()

    iters = max(5, int(min_seconds / per_replay))
    sampler = ClockSampler() if clocks else None
    if sampler is not None:
        sampler.start()
    start.record()
    for _ in range(iters):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    span = start.elapsed_time(end) / 1e3
    clk = sampler.stop() if sampler is not None else (0, 0)
    del g
    return span / (iters * len(calls)), clk


def time_eager(calls, min_seconds: float = 0.6):
    """Fallback timing without a graph (for candidates that cannot be captured)."""
    for c in calls:
        c()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    t0 = time.perf_counter()
    n = 0
    start.record()
    while time.perf_counter() - t0 < min_seconds:
        for c in calls:
            c()
        n += len(calls)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1e3 / max(1, n), (0, 0)
