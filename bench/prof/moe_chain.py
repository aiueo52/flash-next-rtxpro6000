"""Median per-role duration of the MoE chain. Between two consecutive routing prologues,
take the first expandInputRows / doActivation / memset and the two CUTLASS grouped GEMMs
(grid [1,188,1]) in order."""
import gzip, json, sys, statistics, collections
if len(sys.argv) < 2 or not all(__import__("os").path.isfile(a) for a in sys.argv[1:2]):
    sys.exit("usage: moe_chain.py <trace.json.gz>  -- needs a torch-profiler trace from prof/profile_decode2.py; traces are not published in this repository")
p = sys.argv[1]
ev = json.load(gzip.open(p, "rt"))["traceEvents"]
K = sorted([e for e in ev if e.get("ph")=="X" and e.get("cat")=="kernel"], key=lambda e: e["ts"])
starts = [i for i,k in enumerate(K) if "fusedBuildExpertMapsSortF" in k["name"]]
roles = collections.defaultdict(list)
for n,i in enumerate(starts):
    end = starts[n+1] if n+1 < len(starts) else len(K)
    roles["prologue"].append(K[i]["dur"])
    gemms = []
    for k in K[i+1:end]:
        nm = k["name"]
        if nm.startswith("_ZN7cutlass13device_kernel") and str((k.get("args") or {}).get("grid")) == "[1, 188, 1]":
            gemms.append(k["dur"])
        elif "expandInputRows" in nm and "expand" not in [x[0] for x in []]:
            roles.setdefault("expand", []).append(k["dur"])
        elif "doActivationKernel" in nm:
            roles.setdefault("doActivation", []).append(k["dur"])
        elif nm.lower().startswith("memset") or nm == "Memset (Device)":
            roles.setdefault("memset", []).append(k["dur"])
    if len(gemms) >= 2:
        roles["GEMM1"].append(gemms[0]); roles["GEMM2"].append(gemms[1])
glue = 0.0; gemm = 0.0
for r in ("prologue","expand","GEMM1","doActivation","memset","GEMM2"):
    v = roles.get(r)
    if not v: continue
    m = statistics.median(v)
    print(f"  {r:13s} n={len(v):5d} med={m:7.2f} mean={statistics.mean(v):7.2f}")
    if r in ("GEMM1","GEMM2"): gemm += m
    else: glue += m
print(f"  glue/call={glue:.2f}us  gemm/call={gemm:.2f}us  chain/call={glue+gemm:.2f}us  calls={len(roles['prologue'])}")
