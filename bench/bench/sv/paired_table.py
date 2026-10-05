"""Print paired_ab.py JSON results as one table: per domain, B/A change with 95% CI for t/s and acceptance
(tok/step), the two ABBA block means, and the A2/A1 drift of the control.

  python bench/sv/paired_table.py runs/sv/paired-lmstudio.json [more.json ...]
"""
import json
import sys


def row(name, m):
    lo, hi = m["ci_pct"]
    blocks = " / ".join(f"{b['change_pct']:+.1f}" for b in m["block_means"])
    drift = m["A2_over_A1_drift"]
    flag = " DRIFT" if drift["flag"] else ""
    return (f"  {name:<11} {m['change_pct']:+6.1f}%  [{lo:+6.1f} .. {hi:+6.1f}]  blocks {blocks:<13}"
            f" A2/A1 {drift['change_pct']:+5.1f}%{flag}")


for path in sys.argv[1:]:
    d = json.load(open(path))
    print(f"== {path} ({d['schedule']}, {d['cycles']} cycle)")
    for metric in ("tps", "acceptance"):
        print(f" {metric}")
        for dom, v in d["domains"].items():
            print(row(dom, v[metric]))
