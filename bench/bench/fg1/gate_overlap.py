"""FG1 overlap without stream labels: a gate counts when an HC mix K1/K2 kernel on another stream overlaps it.

gate_streams.py compares against the K1/K2 of the one stream with the most of them, but each CUDA graph (wa
captures several widths) can put the same role on its own stream id, so it undercounts (220 of 720 in the
smokes). Only one forward graph runs at a time, so a K1/K2 running during a forked gate is its own graph's.

  python bench/fg1/gate_overlap.py runs/stack8/traces/A1-g-code-edit runs/stack8/traces/B1-g-code-edit
"""
import bisect
import glob
import gzip
import json
import os
import sys

GATE = "hc_combine_gate_kernel"
MIX_K12 = ("_hc_down_kernel", "_hc_up_kernel")


def main(trace_dirs):
    for d in trace_dirs:
        gates, k12 = [], []
        for path in glob.glob(os.path.join(d, "*.trace.json.gz")):
            t = json.load(gzip.open(path))
            for e in t["traceEvents"] if isinstance(t, dict) else t:
                if e.get("cat") != "kernel":
                    continue
                span = (e["args"].get("stream"), e["ts"], e["ts"] + e["dur"])
                if GATE in e["name"]:
                    gates.append(span)
                elif e["name"] in MIX_K12:
                    k12.append(span)
        k12.sort(key=lambda x: x[1])
        starts = [t0 for _, t0, _ in k12]
        overlapped = 0
        for stream, t0, t1 in gates:
            i = bisect.bisect_right(starts, t1)
            overlapped += any(end > t0 and s != stream for s, _, end in k12[max(0, i - 6):i])
        print(f"{os.path.basename(d):26s} gates {len(gates)}, overlapping a K1/K2 on another stream {overlapped}")


if __name__ == "__main__":
    main(sys.argv[1:])
