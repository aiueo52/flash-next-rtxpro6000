"""GPU span of every CUDA-graph replay (first kernel start to last kernel end), by phase, from torch profiler
traces, and an ABBA comparison of the median spans. A graph's span is fixed work at a fixed shape, so it moves
only with the code under test (and the GPU's state), not with acceptance; this is the low-noise measure for
changes inside the captured graphs (FG1, RT1, RQ2, XA1, DG1's draft graph).

usage: graph_span.py A1=<trace dir> B1=<dir> B2=<dir> A2=<dir>
Signature = (phase, kernel count, first kernel's grid): one captured graph at one width.
"""
import collections
import os
import statistics as st
import sys

sys.path.insert(0, os.path.expanduser("~/tools/flash-next-bench/prof"))
import exclusive_time as ET  # noqa: E402


def spans(d):
    path, X = ET.load(d)
    rt = {(e.get("args") or {}).get("correlation"): e for e in X if e.get("cat") in ("cuda_runtime", "cuda_driver")}
    ann, phase_of = ET.phase_map(X)
    groups = collections.defaultdict(list)
    for k in X:
        if k.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        c = (k.get("args") or {}).get("correlation")
        r = rt.get(c)
        if r is not None and "GraphLaunch" in r["name"]:
            groups[c].append(k)
    out = collections.defaultdict(list)
    for c, ks in groups.items():
        ph = phase_of(rt[c]["ts"])
        ph = ET.PHASE_SHORT.get(ph, ph)
        ks.sort(key=lambda k: k["ts"])
        g = (ks[0].get("args") or {}).get("grid")
        out[(ph, len(ks), str(g))].append(max(k["ts"] + k["dur"] for k in ks) - ks[0]["ts"])
    return out


arms = [a.split("=", 1) for a in sys.argv[1:]]
data = {lab: spans(d) for lab, d in arms}
labels = [a[0] for a in arms]
sigs = sorted(set.intersection(*(set(v) for v in data.values())))
print(f"{'phase':13s} {'kern':>5s} {'grid':18s} " + " ".join(f"{l:>14s}" for l in labels) + "   B/A")
per_phase = collections.defaultdict(lambda: [0.0, 0.0, 0])
for s in sigs:
    meds = {l: st.median(data[l][s]) for l in labels}
    ns = {l: len(data[l][s]) for l in labels}
    a = [meds[l] for l in labels if l.startswith("A")]
    b = [meds[l] for l in labels if l.startswith("B")]
    r = st.mean(b) / st.mean(a) - 1
    w = min(ns.values())
    pp = per_phase[s[0]]
    pp[0] += w * st.mean(a)
    pp[1] += w * st.mean(b)
    pp[2] += w
    print(f"{s[0]:13s} {s[1]:5d} {s[2][:18]:18s} " + " ".join(f"{meds[l]:8.1f} n{ns[l]:<4d}" for l in labels)
          + f" {100 * r:+6.2f}%")
for ph, (a, b, w) in per_phase.items():
    print(f"== {ph:13s} weighted B/A {100 * (b / a - 1):+6.2f}%  (A {a / w:.1f} us, B {b / w:.1f} us per replay, {w} replays)")
for ph in per_phase:
    tot = {l: sum(st.median(data[l][s]) * min(len(data[x][s]) for x in labels) for s in sigs if s[0] == ph) for l in labels}
    print(f"   {ph:13s} per arm (weighted sum, vs A1): " + "  ".join(f"{l} {100 * (tot[l] / tot[labels[0]] - 1):+.2f}%" for l in labels))
