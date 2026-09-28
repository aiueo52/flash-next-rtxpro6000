#!/usr/bin/env python3
"""Outlier-trimmed per-step budget for the two kernels the fold moves work between.
A handful of K0 launches per profile land in the hundreds of us (profiler/scheduler
artefacts); they swamp a raw sum, so drop anything over 10 us and report both."""
import gzip, json, sys, statistics
for path in sys.argv[1:]:
    ev=json.load(gzip.open(path,"rt"))["traceEvents"]
    n=len([e for e in ev if e.get("ph")=="X" and e.get("cat")=="user_annotation" and e["name"].startswith("step[")]) or 1
    out=[]
    for key,label in (("branch_stats","K0"),("hc_combine_apply2","apply")):
        d=[e["dur"] for e in ev if e.get("ph")=="X" and e.get("cat")=="kernel" and key in e["name"]]
        t=[x for x in d if x<=10.0]
        out.append(f"{label}: n={len(d)/n:5.1f}/step raw={sum(d)/n:7.1f} trimmed={sum(t)/n:7.1f} "
                   f"med={statistics.median(d):.2f} dropped={len(d)-len(t)}")
    print(f"{path.split('/')[-2]:26s} " + " | ".join(out))
