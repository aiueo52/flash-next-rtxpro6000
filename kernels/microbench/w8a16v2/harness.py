"""CUDA-graph replay timing for the W8A16 skinny GEMV.

Two things the previous harness got wrong and this one checks:

- Working set. Every measurement rotates over >= 24 distinct weight copies, enough of
  them that the set is ~4x the 128 MB L2, so the number reflects HBM traffic. (A weight
  already larger than 4x L2 by itself still gets the 24-copy minimum.)
- Clocks. A few hundred microseconds of replay leaves the SMs at their idle clock and
  reports 2-3x the in-server time. Each measurement spins the GPU up first and then
  replays for at least `min_seconds`, and the achieved SM clock is sampled during the
  timed region and reported, so an idle-clock run is visible instead of silent.
"""

import subprocess
import threading
import time

import torch

L2_BYTES = 128 << 20
WORKING_SET_TARGET = 4 * L2_BYTES
MEM_BUDGET = 24 << 30
MIN_COPIES = 24


def n_copies(weight_bytes: int, hi: int = 512) -> int:
    n = max(MIN_COPIES, -(-WORKING_SET_TARGET // weight_bytes))
    return int(min(n, hi, max(2, MEM_BUDGET // weight_bytes)))


def make_copies(w, n):
    """n distinct byte-identical copies of w, preserving its stride layout."""
    out = [w]
    for _ in range(n - 1):
        c = torch.empty_strided(w.shape, w.stride(), dtype=w.dtype, device=w.device)
        c.copy_(w)
        out.append(c)
    return out


class ClockSampler(threading.Thread):
    """Poll clocks.sm while a measurement runs; nvidia-smi costs ~40 ms per sample."""

    def __init__(self, period=0.15):
        super().__init__(daemon=True)
        self.period = period
        self.samples = []
        self._ev = threading.Event()

    def run(self):
        while not self._ev.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip()
                self.samples.append(int(out.splitlines()[0]))
            except Exception:
                pass
            self._ev.wait(self.period)

    def stop(self):
        self._ev.set()
        self.join(timeout=5)
        return (min(self.samples), max(self.samples)) if self.samples else (0, 0)


def capture(call, args_list, warmup=3):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            for a in args_list:
                call(a)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for a in args_list:
            call(a)
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    return g


def time_graph(call, args_list, min_seconds=0.6, spin_seconds=0.25, clocks=False):
    """Seconds per call. Replays a graph of len(args_list) calls under load.

    Returns (seconds_per_call, (sm_clock_min, sm_clock_max)); the clock pair is (0, 0)
    unless `clocks` is set, since each sample forks nvidia-smi.
    """
    g = capture(call, args_list)

    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    g.replay()
    end.record()
    torch.cuda.synchronize()
    per_replay = max(start.elapsed_time(end) / 1e3, 1e-6)

    spin = max(1, int(spin_seconds / per_replay))
    for _ in range(spin):
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
    return span / (iters * len(args_list)), clk


def require_idle_gpu():
    """Refuse to benchmark unless the GPU is idle (and any handover marker is present).

    Set W8A16_BENCH_MARKER to a path that must exist for the run to proceed; that is
    how an agent session hands the exclusive GPU over without a second run sneaking in.
    """
    import os

    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
    if out:
        raise SystemExit(f"GPU busy (compute pids: {out.splitlines()}); refusing to benchmark.")
    marker = os.environ.get("W8A16_BENCH_MARKER")
    if marker and not os.path.exists(marker):
        raise SystemExit(f"GPU handover marker missing: {marker}")


def now():
    return time.perf_counter()


def cupti_kernel_us(call, args_list, min_seconds=0.2, spin_seconds=0.25,
                    max_events=1500, name_sub=None, target_events=None):
    """(median CUPTI kernel duration in seconds, n_events) over CUDA-graph replays.

    This is the metric the in-server torch profile reports, so it is what the
    reproduction gate and the tuner compare against. The working set is whatever
    `args_list` rotates over, exactly as in `time_graph`: the graph is replayed
    under continuous load for at least `min_seconds` (so the SMs stay at boost),
    capped at `max_events` recorded kernels to keep the trace small.
    """
    import statistics

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

    if target_events is not None:  # legacy call form
        reps = max(1, -(-target_events // len(args_list)))
    else:
        reps = max(1, int(min_seconds / per_replay))
        if reps * len(args_list) > max_events:
            reps = max(1, max_events // len(args_list))
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            g.replay()
        torch.cuda.synchronize()
    durs = [
        e.device_time
        for e in prof.events()
        if e.device_type == DeviceType.CUDA
        and (name_sub is None or name_sub in e.key)
    ]
    del g
    if not durs:
        raise RuntimeError("no CUDA kernel events recorded")
    return statistics.median(durs) / 1e6, len(durs)


def launch_grid(N, cfg):
    import triton

    return (triton.cdiv(N, cfg[0]), cfg[2])
