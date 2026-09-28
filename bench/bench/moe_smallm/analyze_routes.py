#!/usr/bin/env python3
"""Turn route_logger snapshots into per-workload routing statistics.

The logger's histograms are cumulative, so a per-workload figure is the
*difference* between consecutive snapshots (``runs/routes-<profile>-<name>.json``,
written in workload order after a ``baseline`` snapshot that holds only CUDA-graph
capture + server warmup traffic).

Usage:
    python -m moe_smallm.analyze_routes runs/routes-w4-{baseline,code-edit,...}.json
    python -m moe_smallm.analyze_routes --profile w4 --dir runs
    python -m moe_smallm.analyze_routes --ring runs/routes-w4-ring.jsonl --calls-per-step 49
CPU-only.
"""

from __future__ import annotations

import argparse
import json
import os

ORDER = ["baseline", "code-edit", "prose-en", "agent-loop", "prose-ja"]


def _load(path):
    with open(path) as f:
        return json.load(f)


def _raw(entry):
    """(n, distinct_histogram dict[int]->int, overlap list[int])."""
    n = entry["calls"]
    hist = {int(k): v for k, v in entry["distinct_histogram"].items()}
    m = entry["experts_by_multiplicity"]
    ov = {int(k): round(v * n) for k, v in m.items()}
    return n, hist, ov


def _stats(n, hist, ov, num_experts=512):
    if n <= 0:
        return None
    tot = sum(hist.values())
    mean = sum(k * v for k, v in hist.items()) / tot if tot else None  # None: nothing counted
    def pct(q):
        cum = 0
        for k in sorted(hist):
            cum += hist[k]
            if cum >= q * tot:
                return k
        return None
    tmax = max(ov) if ov else 0
    return {
        "calls": n,
        "mean": mean,
        "p50": pct(0.5),
        "p90": pct(0.9),
        "min": min(hist) if hist else None,
        "max": max(hist) if hist else None,
        "by_mult": {j: ov.get(j, 0) / n for j in range(1, tmax + 1)},
        "shared_by_all": ov.get(tmax, 0) / n,
    }


def diff(prev, cur, t):
    """Per-workload stats for row T=t between two cumulative snapshots."""
    pe = prev["by_T"].get(str(t)) if prev else None
    ce = cur["by_T"].get(str(t))
    if ce is None:
        return None
    n1, h1, o1 = _raw(ce)
    if pe is None:
        return _stats(n1, h1, o1)
    n0, h0, o0 = _raw(pe)
    h = {k: h1.get(k, 0) - h0.get(k, 0) for k in set(h1) | set(h0)}
    h = {k: v for k, v in h.items() if v > 0}
    o = {k: o1.get(k, 0) - o0.get(k, 0) for k in set(o1) | set(o0)}
    return _stats(n1 - n0, h, o)


def report(profile: str, directory: str, t: int, order=None):
    order = order or ORDER
    paths = [(w, os.path.join(directory, f"routes-{profile}-{w}.json")) for w in order]
    paths = [(w, p) for w, p in paths if os.path.exists(p)]
    if not paths:
        raise SystemExit(f"analyze_routes.py: no route_logger snapshots for profile {profile!r} in {directory} "
                         "(route snapshots are not published in this repository)")
    print(f"=== profile {profile}, verify rows T={t} "
          f"(independent-routing E[D] would be "
          f"{512*(1-(1-10/512)**t):.1f}) ===")
    print(f"{'workload':<12} {'calls':>8} {'mean':>7} {'p50':>5} {'p90':>5} "
          f"{'min':>5} {'max':>5} {'x1':>6} {'x2':>6} {'>=3':>6} {'xT':>6}")
    prev = None
    rows = {}
    for w, p in paths:
        cur = _load(p)
        s = diff(prev, cur, t)
        prev = cur
        if not s or s["calls"] <= 0:
            continue
        bm = s["by_mult"]
        ge3 = sum(v for j, v in bm.items() if j >= 3)
        print(f"{w:<12} {s['calls']:>8} {'n/a' if s['mean'] is None else format(s['mean'], '.2f'):>7} {s['p50']:>5} {s['p90']:>5} "
              f"{s['min']:>5} {s['max']:>5} {bm.get(1,0):>6.2f} {bm.get(2,0):>6.2f} "
              f"{ge3:>6.2f} {s['shared_by_all']:>6.2f}")
        rows[w] = s
    # cumulative total over the four workloads (= last snapshot minus baseline)
    if len(paths) > 1:
        base = _load(paths[0][1]) if paths[0][0] == "baseline" else None
        s = diff(base, _load(paths[-1][1]), t)
        if s:
            bm = s["by_mult"]
            ge3 = sum(v for j, v in bm.items() if j >= 3)
            print(f"{'ALL':<12} {s['calls']:>8} {'n/a' if s['mean'] is None else format(s['mean'], '.2f'):>7} {s['p50']:>5} "
                  f"{s['p90']:>5} {s['min']:>5} {s['max']:>5} {bm.get(1,0):>6.2f} "
                  f"{bm.get(2,0):>6.2f} {ge3:>6.2f} {s['shared_by_all']:>6.2f}")
            rows["ALL"] = s
    return rows


def ring_by_layer(path: str, t: int, calls_per_step: int):
    """Distinct count per call position, from the raw-id ring.

    Calls run in a fixed order within a decode step (draft iterations with T=1,
    then the 48 verify layers, then draft_extend), so `call % calls_per_step`
    identifies the layer once the phase pattern is located.
    """
    recs = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["T"] == t:
                recs.append(r)
    if not recs:
        raise SystemExit(f"no T={t} records in {path}")
    per_pos = {}
    for r in recs:
        pos = r["call"] % calls_per_step
        per_pos.setdefault(pos, []).append(len(set(r["topk_ids"])))
    out = []
    for pos in sorted(per_pos):
        v = sorted(per_pos[pos])
        out.append((pos, len(v), sum(v) / len(v), v[len(v) // 2]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=None)
    ap.add_argument("--dir", default="runs")
    ap.add_argument("--T", type=int, default=None,
                    help="verify row count (default 4 for w4, 16 for w16)")
    ap.add_argument("--ring", default=None)
    ap.add_argument("--calls-per-step", type=int, default=49)
    a = ap.parse_args(argv)
    if a.ring:
        if not os.path.exists(a.ring):
            raise SystemExit(f"analyze_routes.py: no ring snapshot {a.ring} (not published in this repository)")
        t = a.T or 16
        for pos, n, mean, med in ring_by_layer(a.ring, t, a.calls_per_step):
            print(f"pos {pos:3d}  n={n:4d}  mean_distinct={mean:6.2f}  median={med}")
        return 0
    if not a.profile:
        raise SystemExit("analyze_routes.py: --profile or --ring required (needs route_logger snapshots, which are "
                         "not published in this repository)")
    t = a.T or (16 if "16" in a.profile else 4)
    report(a.profile, a.dir, t)
    if t != 1:
        print()
        report(a.profile, a.dir, 1, ORDER)   # the draft (T=1) rows, sanity
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
