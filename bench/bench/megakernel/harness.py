"""CUDA-graph replay timing for the megakernel (K2) feasibility benches.

Same protocol as the fork's ``bench/w8a16v2/harness.py`` (which the HC / glue-E
numbers in the fork's commit messages were taken with), reimplemented here so the
bench repo does not depend on the fork's bench tree:

* every measurement replays a **captured CUDA graph**, because that is what the
  server does and because a graph replay is the only way to see the real
  per-launch cost without the host in the picture;
* the GPU is spun up first and then kept under continuous load for at least
  ``min_seconds``, so the SMs stay at boost (a few hundred microseconds of replay
  reports 2-3x the in-server time);
* working sets rotate over >= 4x the 128 MiB L2 where weights are involved;
* clocks are sampled during the timed region and reported, because this box has
  no ``nvidia-smi -pl`` / clock pinning available without sudo;
* A/B pairs are **interleaved** round by round (``ab_rounds``) and reported as
  per-round deltas, since the GPU is shared and drifts.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import threading
import time

import torch

L2_BYTES = 128 << 20
WORKING_SET_TARGET = 4 * L2_BYTES
MEM_BUDGET = 8 << 30  # the GPU is shared; stay small
MIN_COPIES = 24


def n_copies(weight_bytes: int, hi: int = 512) -> int:
    n = max(MIN_COPIES, -(-WORKING_SET_TARGET // max(weight_bytes, 1)))
    return int(min(n, hi, max(2, MEM_BUDGET // max(weight_bytes, 1))))


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
                    [
                        "nvidia-smi",
                        "--query-gpu=clocks.sm",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout.strip()
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


def time_graph(call, args_list, min_seconds=0.5, spin_seconds=0.25, clocks=False):
    """(seconds per call, (sm_min, sm_max))."""
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


def cupti_by_kernel(call, args_list, min_seconds=0.2, spin_seconds=0.25, max_events=4000):
    """({kernel name: (median us, launches/call)}, summed device us per call)."""
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
    if reps * len(args_list) > max_events:
        reps = max(1, max_events // max(len(args_list), 1))
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
    out, total = {}, 0.0
    for name, durs in by_name.items():
        out[name] = (statistics.median(durs), len(durs) / calls)
        total += sum(durs) / calls
    return out, total


def ab_rounds(variants, rounds=5, min_seconds=0.5):
    """Interleave A/B/C... round by round. `variants` is {tag: (call, args_list)}.

    Returns {tag: {"rounds": [us...], "median": us}}. Interleaving is mandatory on
    this box: the GPU is shared and the SM clock drifts between runs.
    """
    res = {tag: [] for tag in variants}
    for _ in range(rounds):
        for tag, (call, args) in variants.items():
            sec, _ = time_graph(call, args, min_seconds=min_seconds)
            res[tag].append(sec * 1e6)
    return {
        tag: {"rounds": [round(v, 3) for v in vs], "median": round(statistics.median(vs), 3)}
        for tag, vs in res.items()
    }


def gpu_note() -> dict:
    p = torch.cuda.get_device_properties(0)
    try:
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=clocks.sm,clocks.max.sm,power.draw,temperature.gpu,memory.used",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except Exception:
        smi = "unknown"
    return {
        "device": p.name,
        "sms": p.multi_processor_count,
        "smi": smi,
        "torch": torch.__version__,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def emit(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"\n[wrote] {path}")
