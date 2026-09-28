#!/usr/bin/env python3
"""Compare bench_moe --mode profile JSON runs: label=path ... (baseline first)."""
import os
import json, sys

def load(path):
    out = {}
    for r in json.load(open(path)):
        ks = r["kernels"]
        gemms = sorted((v for k, v in ks.items() if "device_kernel" in k or k.startswith("_ZN7cutlass")),
                       reverse=True)
        glue = sum(v for k, v in ks.items()
                   if not ("device_kernel" in k or k.startswith("_ZN7cutlass")))
        if len(gemms) < 2:  # a missing GEMM is a parse failure, not a 0 us kernel
            raise SystemExit(f"{path}: T={r['T']} D={r['distinct']}: {len(gemms)} GEMM timing(s), expected 2")
        out[(r["T"], round(r["distinct"]))] = dict(
            gemm1=gemms[0],
            gemm2=gemms[1],
            glue=glue, total=r["us_sustained"])
    return out

def main(argv):
    bad = [a for a in argv if "=" not in a or not os.path.isfile(a.split("=", 1)[1])]
    if not argv or bad:
        raise SystemExit("usage: cmp_json.py <label>=<bench_moe profile .json> ... (baseline first); " + ("missing: " + ", ".join(bad) + "; " if bad else "")
                         + "the G1 profile runs are not published in this repository")
    labels, data = [], []
    for a in argv:
        lab, path = a.split("=", 1)
        labels.append(lab); data.append(load(path))
    base = data[0]
    print(f"{'T':>3} {'D':>4} | " + " | ".join(
        f"{l+' gemm1':>13} {l+' gemm2':>13} {l+' glue':>12} {l+' tot':>13}" for l in labels))
    for k in sorted(base):
        line = f"{k[0]:>3} {k[1]:>4} |"
        for i, d in enumerate(data):
            v = d.get(k)
            if v is None:
                line += " " + " " * 56 + "|"; continue
            if i == 0:
                line += (f" {v['gemm1']:>13.2f} {v['gemm2']:>13.2f} {v['glue']:>12.2f}"
                         f" {v['total']:>13.2f} |")
            else:
                b = base[k]
                line += (f" {v['gemm1']:>7.2f}{v['gemm1']-b['gemm1']:>+6.2f}"
                         f" {v['gemm2']:>7.2f}{v['gemm2']-b['gemm2']:>+6.2f}"
                         f" {v['glue']:>6.2f}{v['glue']-b['glue']:>+6.2f}"
                         f" {v['total']:>7.2f}{v['total']-b['total']:>+6.2f} |")
        print(line)
    for i, lab in enumerate(labels[1:], 1):
        for T, Dp in ((4, 28), (16, 69)):
            k = (T, Dp)
            if k in base and k in data[i]:
                b, v = base[k], data[i][k]
                print(f"[{lab}] T={T} D={Dp}: gemm1 {v['gemm1']-b['gemm1']:+.2f}  "
                      f"gemm2 {v['gemm2']-b['gemm2']:+.2f}  glue {v['glue']-b['glue']:+.2f}  "
                      f"total {v['total']-b['total']:+.2f} us/call"
                      f"  -> {(v['total']-b['total'])*49/1000:+.3f} ms/step")

main(sys.argv[1:])
