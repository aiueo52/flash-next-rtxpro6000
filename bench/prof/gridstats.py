#!/usr/bin/env python3
"""Per-(kernel name substring, grid) occurrence count and median/mean duration.
Usage: gridstats.py <trace.json.gz> <name-substring> [steps=20]"""
import gzip, json, sys, collections, statistics
if len(sys.argv) < 3 or not all(__import__("os").path.isfile(a) for a in sys.argv[1:2]):
    sys.exit("usage: gridstats.py <trace.json.gz> <name-substring> [steps]  -- needs a torch-profiler trace from prof/profile_decode2.py; traces are not published in this repository")
p, pat = sys.argv[1], sys.argv[2]
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 20
ev = json.load(gzip.open(p, "rt"))["traceEvents"]
g = collections.defaultdict(list)
for e in ev:
    if e.get("ph") == "X" and e.get("cat") == "kernel" and pat in e["name"]:
        a = e.get("args") or {}
        g[(e["name"][:40], str(a.get("grid")), str(a.get("block")))].append(e["dur"])
tot = 0
for k, v in sorted(g.items(), key=lambda kv: -sum(kv[1])):
    med = statistics.median(v); s = sum(v) / steps
    tot += s
    print(f"{k[0]:40s} grid={k[1]:18s} blk={k[2]:12s} n/step={len(v)/steps:6.1f} med={med:7.1f}us mean={statistics.mean(v):7.1f} sum/step={s:8.1f}us")
print(f"total/step={tot:.1f}us")
