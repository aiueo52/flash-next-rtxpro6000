#!/usr/bin/env python3
"""Outlier-robust MoE kernel time per step: median over steps of each call position.

Mean-based sums over a trace are dominated by the 1-5 preempted MoE GEMM calls per
step (MOE_SMALLM_SPEC section 4, 0.4-1.1 ms/step), which is far larger than anything
this A/B is trying to measure.
"""
import gzip, json, sys, collections, statistics

def robust(path):
    ev = json.load(gzip.open(path, "rt"))["traceEvents"]
    ks = collections.defaultdict(list)
    for e in ev:
        if e.get("ph") != "X" or e.get("cat") != "kernel":
            continue
        n = e["name"]
        if "device_kernel" in n and "Sm120Array" in n: k = "moe_gemm"
        elif "fusedBuildExpertMaps" in n: k = "moe_prologue"
        elif "expandInputRows" in n: k = "moe_expand"
        elif "doActivation" in n: k = "moe_act"
        else: continue
        ks[k].append((e["ts"], e["dur"]))
    out = {}
    for k, v in ks.items():
        v.sort()
        # call positions repeat every `per_step` launches
        n = len(v)
        # find the period from the known steps count (20 profiled steps)
        per_step = n // 20
        if per_step == 0: continue
        by_pos = collections.defaultdict(list)
        for i, (_, d) in enumerate(v[:per_step * 20]):
            by_pos[i % per_step].append(d)
        out[k] = sum(statistics.median(x) for x in by_pos.values()) / 1000.0
    return out

for p in sys.argv[1:]:
    r = robust(p)
    print(f"{p.split('/')[-2]:<28} " + "  ".join(f"{k}={v:.4f}" for k, v in sorted(r.items()))
          + f"  TOTAL={sum(r.values()):.4f} ms/step")
