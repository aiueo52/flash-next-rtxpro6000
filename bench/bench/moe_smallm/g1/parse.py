#!/usr/bin/env python3
"""Pull (T, D, gemm1_us, gemm2_us, glue_us, clock) out of a --mode profile log."""
import json, re, sys

def parse(path):
    rows = []
    cur = None
    for line in open(path, errors="replace"):
        line = line.rstrip("\n")
        if '"profile": true' in line:
            try:
                cur = json.loads(line)
            except Exception:
                cur = None
                continue
            cur["gemms"] = []
            cur["glue"] = 0.0
            rows.append(cur)
            continue
        if cur is None:
            continue
        m = re.match(r"\s+([0-9.]+) us/call\s+n=\s*([0-9.]+)\s+(.*)", line)
        if not m:
            continue
        us, name = float(m.group(1)), m.group(3)
        if "TOTAL" in name:
            continue
        if "device_kernel" in name or name.startswith("_ZN7cutlass"):
            cur["gemms"].append(us)
        else:
            cur["glue"] += us
    return rows

for p in sys.argv[1:]:
    print(f"== {p}")
    print(f"{'T':>3} {'D':>5} {'gemm1':>7} {'gemm2':>7} {'glue':>6} {'total':>7} {'clk':>10}")
    for r in parse(p):
        g = sorted(r["gemms"], reverse=True)
        if len(g) < 2:  # a missing GEMM is a parse failure, not a 0 us kernel
            raise SystemExit(f"{p}: T={r['T']}: {len(g)} GEMM timing(s), expected 2")
        g1, g2 = g[0], g[1]
        print(f"{r['T']:>3} {r['distinct']:>5.0f} {g1:>7.2f} {g2:>7.2f} {r['glue']:>6.2f} "
              f"{r['us_per_call_sustained']:>7.2f} {str(r['sm_clock_sustained']):>10}")
