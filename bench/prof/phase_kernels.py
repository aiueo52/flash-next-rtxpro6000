#!/usr/bin/env python3
"""Dump the ordered kernel sequence of one occurrence of a phase (user_annotation) in a
torch-profiler chrome trace, with the innermost cpu_op that launched each kernel.
Usage: phase_kernels.py <trace.json.gz> <phase_name> [occurrence_index] [--all-occ]
"""
import gzip, json, sys, bisect, collections
if len(sys.argv) < 3 or not all(__import__("os").path.isfile(a) for a in sys.argv[1:2]):
    sys.exit("usage: phase_kernels.py <trace.json.gz> <phase> [occ] [--all-occ]  -- needs a torch-profiler trace from prof/profile_decode2.py; traces are not published in this repository")
p = sys.argv[1]; phase = sys.argv[2]
occ = int(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else 5
ev = json.load(gzip.open(p, "rt") if p.endswith(".gz") else open(p))["traceEvents"]
X = [e for e in ev if e.get("ph") == "X"]
kern = [e for e in X if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
rt = {(e.get("args") or {}).get("correlation"): e for e in X if e.get("cat") == "cuda_runtime"}
cpu = sorted([e for e in X if e.get("cat") == "cpu_op"], key=lambda e: e["ts"])
ann = sorted([e for e in X if e.get("cat") == "user_annotation" and e["name"] == phase], key=lambda e: e["ts"])
if not ann: print("phase not found; available:", collections.Counter(e["name"][:50] for e in X if e.get("cat")=="user_annotation").most_common(20)); sys.exit()
a = ann[min(occ, len(ann)-1)]
print(f"phase={phase} occurrence={occ}/{len(ann)} cpu_dur={a['dur']/1e3:.2f}ms")
cstarts = [e["ts"] for e in cpu]
def innermost(ts, tid):
    i = bisect.bisect_right(cstarts, ts) - 1
    best = None
    for j in range(i, max(-1, i - 400), -1):
        e = cpu[j]
        if e.get("tid") != tid: continue
        if e["ts"] <= ts <= e["ts"] + e["dur"]:
            if best is None or e["dur"] < best["dur"]: best = e
    return best
rows = []
for k in kern:
    c = (k.get("args") or {}).get("correlation"); r = rt.get(c)
    if r is None: continue
    if a["ts"] <= r["ts"] <= a["ts"] + a["dur"]:
        op = innermost(r["ts"], r.get("tid"))
        rows.append((r["ts"], k))
        k["_op"] = op["name"] if op else "?"; k["_api"] = r["name"]
rows.sort(key=lambda x: x[0])
tot = sum(k["dur"] for _, k in rows)
gpu_start = min(k["ts"] for _, k in rows); gpu_end = max(k["ts"] + k["dur"] for _, k in rows)
print(f"kernels={len(rows)} sum_gpu={tot:.0f}us gpu_span={gpu_end-gpu_start:.0f}us")
print(f"{'#':>3} {'gpu_us':>7} {'gap':>5} {'api':<18} {'cpu_op':<34} kernel")
prev_end = None
for i, (ts, k) in enumerate(rows):
    gap = (k["ts"] - prev_end) if prev_end is not None else 0
    prev_end = max(prev_end or 0, k["ts"] + k["dur"])
    print(f"{i:3d} {k['dur']:7.1f} {gap:5.0f} {k['_api'][:18]:<18} {k['_op'][:34]:<34} {k['name'][:70]}")
