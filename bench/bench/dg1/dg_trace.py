"""Draft MoE kernels in one profiled arm (bench/dg1/arm_dg.sh): counts and median durations of the GEMV
kernels and of the CUTLASS MoE chain (its routing prologue marks one MoE call). With the GEMV on, the
one-token draft calls move from the CUTLASS count to k1/k2 (14 per verify step at 15 draft steps).

  python bench/dg1/dg_trace.py runs/dg1/traces/B1-code-edit
"""
import collections
import glob
import gzip
import json
import os
import statistics
import sys

ROLES = {"_draft_moe_up_gate_kernel": "gemv_k1", "_draft_moe_down_kernel": "gemv_k2",
         "fusedBuildExpertMaps": "cutlass_prologue", "doActivation": "cutlass_act"}


def main(trace_dir):
    durs = collections.defaultdict(list)
    files = glob.glob(os.path.join(trace_dir, "*.trace.json.gz"))
    for path in files:
        t = json.load(gzip.open(path))
        for e in t["traceEvents"] if isinstance(t, dict) else t:
            if e.get("cat") != "kernel":
                continue
            for key, role in ROLES.items():
                if key in e["name"]:
                    durs[role].append(e["dur"])
    if not files:
        print(f"no traces in {trace_dir}")
        return
    for role in ROLES.values():
        v = durs.get(role, [])
        med = f"{statistics.median(v):6.2f} us" if v else "     -"
        print(f"{role:17s} count {len(v):6d}  median {med}")


if __name__ == "__main__":
    main(sys.argv[1])
