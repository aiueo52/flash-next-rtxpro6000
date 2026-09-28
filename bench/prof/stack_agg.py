#!/usr/bin/env python3
"""Aggregate eager kernels of one phase occurrence by their innermost repo frame
(sglang/srt or flashinfer file:line). Usage: stack_agg.py <trace> <phase> [occ] [top]"""
import gzip, json, sys, bisect, collections, re
if len(sys.argv) < 3 or not all(__import__("os").path.isfile(a) for a in sys.argv[1:2]):
    sys.exit("usage: stack_agg.py <trace> <phase> [occ] [top]  -- needs a torch-profiler trace from prof/profile_decode2.py; traces are not published in this repository")
p, phase = sys.argv[1], sys.argv[2]
occ = int(sys.argv[3]) if len(sys.argv) > 3 else 5
top = int(sys.argv[4]) if len(sys.argv) > 4 else 40
ev = json.load(gzip.open(p, "rt"))["traceEvents"]
X = [e for e in ev if e.get("ph") == "X"]
kern = [e for e in X if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
rt = {(e.get("args") or {}).get("correlation"): e for e in X if e.get("cat") == "cuda_runtime"}
py = sorted([e for e in X if e.get("cat") == "python_function"], key=lambda e: e["ts"])
ann = sorted([e for e in X if e.get("cat") == "user_annotation" and e["name"] == phase], key=lambda e: e["ts"])
a = ann[min(occ, len(ann) - 1)]
pstarts = [e["ts"] for e in py]
def repo_frame(ts, tid):
    i = bisect.bisect_right(pstarts, ts) - 1
    st = []
    for j in range(i, max(-1, i - 3000), -1):
        e = py[j]
        if e.get("tid") != tid: continue
        if e["ts"] <= ts <= e["ts"] + e["dur"]: st.append(e)
    st.sort(key=lambda e: e["dur"])
    for e in st:
        n = e["name"]
        if ("sglang/srt/" in n or "flashinfer/" in n) and "torch/" not in n:
            return re.sub(r".*/(python/sglang/srt/|site-packages/flashinfer/)", r"\1", n)
    return st[0]["name"] if st else "?"
agg = collections.defaultdict(lambda: [0, 0.0, collections.Counter()])
tot_n = tot_d = 0
for k in kern:
    r = rt.get((k.get("args") or {}).get("correlation"))
    if r is None or not (a["ts"] <= r["ts"] <= a["ts"] + a["dur"]) or r["name"] == "cudaGraphLaunch": continue
    f = repo_frame(r["ts"], r.get("tid"))
    agg[f][0] += 1; agg[f][1] += k["dur"]; agg[f][2][k["name"][:28]] += 1
    tot_n += 1; tot_d += k["dur"]
print(f"phase={phase} occ={occ}/{len(ann)} kernels={tot_n} gpu_us={tot_d:.0f}")
for f, (c, d, names) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:top]:
    print(f"{c:4d}x {d:7.1f}us  {f[:95]}  [{', '.join(f'{n}:{m}' for n, m in names.most_common(3))}]")
