#!/usr/bin/env python3
"""For a with_stack trace: list eager (non-graph) kernel launches inside one phase occurrence
with their innermost python frames. Usage: stack_kernels.py <trace> <phase> [occ] [nframes]"""
import gzip, json, sys, bisect, collections
if len(sys.argv) < 3 or not all(__import__("os").path.isfile(a) for a in sys.argv[1:2]):
    sys.exit("usage: stack_kernels.py <trace> <phase> [occ] [nframes]  -- needs a torch-profiler trace from prof/profile_decode2.py; traces are not published in this repository")
p, phase = sys.argv[1], sys.argv[2]
occ = int(sys.argv[3]) if len(sys.argv) > 3 else 5
nfr = int(sys.argv[4]) if len(sys.argv) > 4 else 4
ev = json.load(gzip.open(p, "rt"))["traceEvents"]
X = [e for e in ev if e.get("ph") == "X"]
cats = collections.Counter(e.get("cat") for e in X)
kern = [e for e in X if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
rt = {(e.get("args") or {}).get("correlation"): e for e in X if e.get("cat") == "cuda_runtime"}
py = sorted([e for e in X if e.get("cat") == "python_function"], key=lambda e: e["ts"])
ann = sorted([e for e in X if e.get("cat") == "user_annotation" and e["name"] == phase], key=lambda e: e["ts"])
a = ann[min(occ, len(ann) - 1)]
pstarts = [e["ts"] for e in py]
def frames(ts, tid):
    i = bisect.bisect_right(pstarts, ts) - 1
    st = []
    for j in range(i, max(-1, i - 3000), -1):
        e = py[j]
        if e.get("tid") != tid: continue
        if e["ts"] <= ts <= e["ts"] + e["dur"]: st.append(e)
    st.sort(key=lambda e: e["dur"])
    return [e["name"] for e in st[:nfr]]
rows = []
for k in kern:
    r = rt.get((k.get("args") or {}).get("correlation"))
    if r is None or not (a["ts"] <= r["ts"] <= a["ts"] + a["dur"]): continue
    rows.append((r["ts"], r["name"], k["dur"], k["name"][:50], frames(r["ts"], r.get("tid"))))
rows.sort(key=lambda x: x[0])
print(f"phase={phase} occ={occ}/{len(ann)} kernels={len(rows)} python_function events={cats.get('python_function')}")
agg = collections.OrderedDict()
for ts, api, dur, name, fr in rows:
    if api == "cudaGraphLaunch": continue
    key = " <- ".join(fr)
    agg.setdefault(key, [0, 0.0, name]); agg[key][0] += 1; agg[key][1] += dur
for key, (c, d, name) in agg.items():
    print(f"{c:4d}x {d:7.1f}us  {name}\n        {key}")
