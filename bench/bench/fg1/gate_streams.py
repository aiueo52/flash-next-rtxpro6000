"""Stream check for one profiled arm: where the HC combine gate and the GEMVs ran, and whether the
gates off the main stream overlapped the HC mix K1/K2 of the main stream (the point of the fork).

  python bench/fg1/gate_streams.py runs/fg1/traces/B1-code-edit
"""
import bisect
import collections
import glob
import gzip
import json
import os
import sys

GATE = "hc_combine_gate_kernel"
GEMV = "_w8a16_gemv_kernel"
MIX_K12 = ("_hc_down_kernel", "_hc_up_kernel")


def kernels(trace_dir):
    for path in glob.glob(os.path.join(trace_dir, "*.trace.json.gz")):
        t = json.load(gzip.open(path))
        for e in t["traceEvents"] if isinstance(t, dict) else t:
            if e.get("cat") == "kernel":
                yield e["name"], e["args"].get("stream"), e["ts"], e["ts"] + e["dur"]


def main(trace_dir):
    by_stream = collections.defaultdict(collections.Counter)
    gates, k12 = [], []
    for name, stream, t0, t1 in kernels(trace_dir):
        key = GATE if GATE in name else name
        if key in (GATE, GEMV) or key in MIX_K12:
            by_stream[key][stream] += 1
        if key == GATE:
            gates.append((stream, t0, t1))
        elif key in MIX_K12:
            k12.append((stream, t0, t1))
    if not k12:
        print(f"no HC mix kernels in {trace_dir}")
        return
    main_stream = collections.Counter(s for s, _, _ in k12).most_common(1)[0][0]
    for key in (GATE, GEMV, *MIX_K12):
        print(f"{key:24s} per stream: {dict(sorted(by_stream[key].items()))}")
    # the GDN alt stream: the side stream that carries most of the GEMVs (b/a, or qkvz when on)
    alt_stream = max((s for s in by_stream[GEMV] if s != main_stream), key=lambda s: by_stream[GEMV][s])
    main_k12 = sorted((t0, t1) for s, t0, t1 in k12 if s == main_stream)
    starts = [t0 for t0, _ in main_k12]
    side = [(t0, t1) for s, t0, t1 in gates if s == alt_stream]
    overlapped = 0
    for t0, t1 in side:
        i = bisect.bisect_right(starts, t1)
        # any main K1/K2 that starts before this gate ends and ends after it starts
        overlapped += any(e > t0 for _, e in main_k12[max(0, i - 4):i])
    print(f"main stream {main_stream}, GDN alt stream {alt_stream}: gates on the alt stream {len(side)} "
          f"of {len(gates)}, overlapping main K1/K2 {overlapped}")


if __name__ == "__main__":
    main(sys.argv[1])
