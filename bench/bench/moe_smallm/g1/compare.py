#!/usr/bin/env python3
"""Compare G1 variants' per-kernel profiles: <label>=<log> pairs, baseline first."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse import parse

def rows(path):
    out = {}
    for r in parse(path):
        g = sorted(r["gemms"], reverse=True)
        if len(g) < 2:  # a missing GEMM is a parse failure, not a 0 us kernel
            raise SystemExit(f"{path}: T={r['T']} D={r['distinct']}: {len(g)} GEMM timing(s), expected 2")
        out[(r["T"], round(r["distinct"]))] = {
            "gemm1": g[0],
            "gemm2": g[1],
            "glue": r["glue"],
            "total": r["us_per_call_sustained"],
            "clk": r["sm_clock_sustained"],
        }
    return out

def main(argv):
    bad = [a for a in argv if "=" not in a or not os.path.isfile(a.split("=", 1)[1])]
    if not argv or bad:
        raise SystemExit("usage: compare.py <label>=<log> ... (baseline first); " + ("missing: " + ", ".join(bad) + "; " if bad else "")
                         + "the G1 profile runs are not published in this repository")
    labels, data = [], []
    for a in argv:
        lab, path = a.split("=", 1)
        labels.append(lab)
        data.append(rows(path))
    base = data[0]
    keys = sorted(base)
    hdr = f"{'T':>3} {'D':>4}"
    for lab in labels:
        hdr += f" | {lab+' g1':>11} {lab+' g2':>9} {lab+' tot':>10}"
    print(hdr)
    for k in keys:
        line = f"{k[0]:>3} {k[1]:>4}"
        for i, d in enumerate(data):
            v = d.get(k)
            if v is None:
                line += " | " + " " * 32
                continue
            if i == 0:
                line += f" | {v['gemm1']:>11.2f} {v['gemm2']:>9.2f} {v['total']:>10.2f}"
            else:
                b = base[k]
                line += (f" | {v['gemm1']:>7.2f}{v['gemm1']-b['gemm1']:>+6.2f}"
                         f" {v['gemm2']:>5.2f}{v['gemm2']-b['gemm2']:>+5.2f}"
                         f" {v['total']:>6.2f}{v['total']-b['total']:>+6.2f}")
        print(line)
    # mean delta at the production D
    for i, lab in enumerate(labels[1:], 1):
        for T, Dprod in ((4, 28), (16, 69)):
            k = (T, Dprod)
            if k in base and k in data[i]:
                b, v = base[k], data[i][k]
                print(f"[{lab}] T={T} D={Dprod}: gemm1 {v['gemm1']-b['gemm1']:+.2f} us, "
                      f"gemm2 {v['gemm2']-b['gemm2']:+.2f} us, total {v['total']-b['total']:+.2f} us "
                      f"-> {(v['total']-b['total'])*49/1000:+.3f} ms/step (49 MoE calls)")

if __name__ == "__main__":
    main(sys.argv[1:])
