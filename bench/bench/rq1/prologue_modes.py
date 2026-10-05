"""RQ1: in-server routing prologue (fusedBuildExpertMapsSort...) durations per trace dir, by grid size.

grid = 16 + 10*T (26 = draft T=1, 56 = verify T=4, 176 = verify T=16). Prints n, p10/p50/p90 in us and
the share of fast calls (< 10 us: the warm mode is 7.2-8.4 us, the L2-cold mode 10.7-12.8, FC_fc-moe 4b.1).

  python bench/rq1/prologue_modes.py runs/rq1/traces/A1-code-edit runs/rq1/traces/B1-code-edit ...
"""
import collections
import glob
import gzip
import json
import statistics
import sys

FAST_US = 10.0


def durations(trace_dir):
    by_grid = collections.defaultdict(list)
    for path in sorted(glob.glob(trace_dir + "/*.json.gz")):
        for e in json.load(gzip.open(path))["traceEvents"]:
            if e.get("cat") == "kernel" and "fusedBuildExpertMapsSortF" in e["name"]:
                by_grid[e["args"]["grid"][0]].append(e["dur"])
    return by_grid


def main():
    for d in sys.argv[1:]:
        name = d.rstrip("/").split("/")[-1]
        by_grid = durations(d)
        if not by_grid:
            print(f"{name}: no prologue kernels")
            continue
        for grid, v in sorted(by_grid.items()):
            q = statistics.quantiles(v, n=10) if len(v) >= 10 else [min(v)] * 9
            fast = sum(x < FAST_US for x in v) / len(v)
            print(f"{name} grid {grid:4d} (T={(grid - 16) / 10:g}): n={len(v):5d} p10 {q[0]:6.2f} "
                  f"p50 {statistics.median(v):6.2f} p90 {q[8]:6.2f} fast {fast:6.1%}")


if __name__ == "__main__":
    main()
