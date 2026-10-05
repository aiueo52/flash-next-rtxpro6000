"""ABBA table for bench/fg1: per (mode, workload), the B/A change of ms/step and tok/step with a 95%
bootstrap CI over requests, the two ABBA block ratios and the A2/A1 drift; then the per-mode pooled
change (mean of the per-workload log ratios, bootstrapped jointly).

  python bench/fg1/fg_table.py runs/fg1/A1-probe.jsonl runs/fg1/B1-probe.jsonl runs/fg1/B2-probe.jsonl runs/fg1/A2-probe.jsonl
"""
import collections
import json
import math
import random
import statistics
import sys

ARMS = ("A1", "B1", "B2", "A2")
BOOT = 4000


def load(path):
    cells = collections.defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        if not r["profiled"] and r["ms_per_step"] == r["ms_per_step"]:
            cells[(r["mode"], r["workload"])].append(r)
    return cells


def log_effect(arms, metric):
    """log of mean(B1, B2) / mean(A1, A2), each arm mean over its requests."""
    m = {a: statistics.fmean(r[metric] for r in arms[a]) for a in ARMS}
    return math.log((m["B1"] + m["B2"]) / (m["A1"] + m["A2"])), m


def resample(arms, rng):
    return {a: [rng.choice(arms[a]) for _ in arms[a]] for a in ARMS}


def pct(x):
    return (math.exp(x) - 1) * 100


def main(paths):
    data = dict(zip(ARMS, (load(p) for p in paths)))
    keys = sorted(set.intersection(*(set(d) for d in data.values())))
    rng = random.Random(0)
    for metric in ("ms_per_step", "tok_per_step"):
        print(f"== {metric}: B/A change, 95% CI over requests; blocks B1/A1, B2/A2; A2/A1 drift")
        for mode in sorted({k[0] for k in keys}):
            mkeys = [k for k in keys if k[0] == mode]
            pooled_boot = [0.0] * BOOT
            pooled = 0.0
            for k in mkeys:
                arms = {a: data[a][k] for a in ARMS}
                eff, m = log_effect(arms, metric)
                boot = sorted(log_effect(resample(arms, rng), metric)[0] for _ in range(BOOT))
                # pooled CI: the same bootstrap index across workloads, averaged
                for i, b in enumerate(boot):
                    pooled_boot[i] += b / len(mkeys)
                pooled += eff / len(mkeys)
                lo, hi = boot[int(0.025 * BOOT)], boot[int(0.975 * BOOT)]
                n = "/".join(str(len(arms[a])) for a in ARMS)
                print(f"  {mode:8s} {k[1]:15s} {pct(eff):+6.2f}%  [{pct(lo):+6.2f} .. {pct(hi):+6.2f}]"
                      f"  blocks {100 * (m['B1'] / m['A1'] - 1):+5.2f} / {100 * (m['B2'] / m['A2'] - 1):+5.2f}"
                      f"  A2/A1 {100 * (m['A2'] / m['A1'] - 1):+5.2f}%  A {m['A1']:.2f}  n {n}")
            pooled_boot.sort()
            print(f"  {mode:8s} {'pooled':15s} {pct(pooled):+6.2f}%  "
                  f"[{pct(pooled_boot[int(0.025 * BOOT)]):+6.2f} .. {pct(pooled_boot[int(0.975 * BOOT)]):+6.2f}]")


if __name__ == "__main__":
    main(sys.argv[1:5])
