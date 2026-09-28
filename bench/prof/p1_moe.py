#!/usr/bin/env python3
"""MoE chain per-call, split into the T=16/4 verify calls and the T=1 draft calls.

``moe_chain.py`` reports one median over a mixture (48 verify + 15 draft calls
per W16 step), which is fine for the median but not for the mean and useless for
a delta.  The two populations are far apart in GEMM1 duration (D~70 vs D=10), so
a 2-means split on GEMM1 separates them cleanly.  Only the verify cluster is
affected by singleton pruning.

Usage: p1_moe.py <trace.json.gz> [more...]
"""
import collections
import gzip
import json
import statistics
import sys


def chain(path):
    ev = json.load(gzip.open(path, "rt"))["traceEvents"]
    K = sorted([e for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel"],
               key=lambda e: e["ts"])
    starts = [i for i, k in enumerate(K) if "fusedBuildExpertMapsSortF" in k["name"]]
    calls = []
    for n, i in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(K)
        rec = {"prologue": K[i]["dur"], "gemm": [], "act": 0.0, "prune": 0.0}
        for k in K[i + 1:end]:
            nm = k["name"]
            if (nm.startswith("_ZN7cutlass13device_kernel")
                    and str((k.get("args") or {}).get("grid")) == "[1, 188, 1]"):
                rec["gemm"].append(k["dur"])
            elif "doActivationKernel" in nm:
                rec["act"] += k["dur"]
        # the prune kernel is launched just BEFORE the prologue
        for k in K[max(0, i - 4):i]:
            if "prune_singleton" in k["name"]:
                rec["prune"] += k["dur"]
        if len(rec["gemm"]) == 2:
            calls.append(rec)
    return calls


# Draft (T=1, D=10) GEMM1 sits at ~17.5 us and verify (D=25..100) at 40-160 us,
# with a clean empty gap between; a fixed 30 us split is far more robust than
# clustering, which the occasional 1.4 ms preemption outlier drags off target.
SPLIT_US = 30.0


def med(v):
    return statistics.median(v) if v else float("nan")


if len(sys.argv) < 2 or not all(__import__("os").path.isfile(a) for a in sys.argv[1:]):
    sys.exit("usage: p1_moe.py <trace.json.gz> [more...]  -- needs a torch-profiler trace from prof/profile_decode2.py; traces are not published in this repository")
for p in sys.argv[1:]:
    calls = chain(p)
    thr = SPLIT_US
    groups = collections.defaultdict(list)
    for c in calls:
        groups["verify" if c["gemm"][0] >= thr else "draft"].append(c)
    print(f"\n{p}  ({len(calls)} MoE calls, verify = GEMM1 >= {thr:.0f} us)")
    print(f"{'group':<8}{'n':>6}{'prune':>8}{'prologue':>10}{'GEMM1':>9}"
          f"{'act':>7}{'GEMM2':>9}{'gemm':>9}{'chain':>9}")
    for name in ("verify", "draft"):
        c = groups[name]
        if not c:
            continue
        pr = med([x["prune"] for x in c])
        pl = med([x["prologue"] for x in c])
        a = med([x["gemm"][0] for x in c])
        ac = med([x["act"] for x in c])
        b = med([x["gemm"][1] for x in c])
        print(f"{name:<8}{len(c):>6}{pr:>8.2f}{pl:>10.2f}{a:>9.2f}"
              f"{ac:>7.2f}{b:>9.2f}{a+b:>9.2f}{pr+pl+a+ac+b:>9.2f}")
