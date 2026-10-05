"""Router kernels in one profiled arm: count and median duration of the Triton router, the RT1 fast
routers (int64 and 32-bit keys) and the router GEMV (the step's other kernels are unchanged by RT1).

  python bench/rt1/router_trace.py runs/rt1/traces/B1-code-edit
"""
import collections
import glob
import gzip
import json
import os
import statistics
import sys

NAMES = (
    "_router_triton_kernel",
    "_router_softmax_fast_kernel",
    "_router_softmax_fast32_kernel",
    "_w8a16_gemv_kernel",
)


def main(trace_dir):
    durs = collections.defaultdict(list)
    for path in glob.glob(os.path.join(trace_dir, "*.trace.json.gz")):
        t = json.load(gzip.open(path))
        for e in t["traceEvents"] if isinstance(t, dict) else t:
            if e.get("cat") == "kernel" and e["name"] in NAMES:
                durs[e["name"]].append(e["dur"])
    for n in NAMES:
        d = durs.get(n, [])
        med = f"{statistics.median(d):.2f} us" if d else "-"
        print(f"{n}: n={len(d)} median={med}")


if __name__ == "__main__":
    main(sys.argv[1])
