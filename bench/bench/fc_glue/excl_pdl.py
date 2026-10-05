#!/usr/bin/env python3
"""prof/exclusive_time.py with the PDL spin removed.

With SGLANG_TRITON_PDL=1 a secondary kernel is launched early and spins in griddepcontrol.wait
until its producer finishes. The trace records the early start, so the plain sweep line shows the
producer and the spinning consumer as two concurrent kernels: the producer's time lands in
"shared" and its exclusive time is undercounted (e.g. verify kernel_mha 8 us/step instead of 182
at W4). Kernels on one stream are serialised, so each kernel's start is clipped to the end of the
previous kernel on the same stream; the rest is exclusive_time.py unchanged.

Usage: excl_pdl.py <trace-dir> [...] [--top N] [--phase NAME] [--noclip]
"""
import collections, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "prof"))
import exclusive_time as et  # noqa: E402

_load = et.load


def load_clipped(path):
    p, X = _load(path)
    by_stream = collections.defaultdict(list)
    for e in X:
        if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
            by_stream[((e.get("args") or {}).get("stream"), e.get("pid"), e.get("tid"))].append(e)
    clipped = 0.0
    for ks in by_stream.values():
        ks.sort(key=lambda e: e["ts"])
        end = None
        for e in ks:
            if end is not None and e["ts"] < end:
                new_ts = min(end, e["ts"] + e["dur"])
                clipped += new_ts - e["ts"]
                e["dur"] -= new_ts - e["ts"]
                e["ts"] = new_ts
            end = e["ts"] + e["dur"] if end is None else max(end, e["ts"] + e["dur"])
    print(f"[pdl-clip] {p}: removed {clipped/1e3:.1f} ms of same-stream overlap in the whole trace")
    return p, X


if __name__ == "__main__":
    if "--noclip" in sys.argv:
        sys.argv.remove("--noclip")
    else:
        et.load = load_clipped
    et.main()
