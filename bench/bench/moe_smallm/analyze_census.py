#!/usr/bin/env python3
"""P1 census: singleton-route statistics and the tau sweep, offline from a
``route_census`` ring dump.

    python -m moe_smallm.analyze_census runs/census-w16-code-edit.npz [...]
    python -m moe_smallm.analyze_census --glob 'runs/census-w16-*.npz'

A *singleton route* is a (row, expert) pair whose expert is reached by exactly
one row of the call.  Dropping it removes a whole expert from the grouped GEMM
(~1.04 us of GEMM1 + ~0.55x of GEMM2); dropping a non-singleton route saves
nothing.  The top-1 route of a row is never a candidate (a row must keep at
least one expert or the finalize leaves its output row unwritten).
CPU-only.
"""

from __future__ import annotations

import argparse
import glob as globmod
import os

import numpy as np

TAUS = (0.02, 0.03, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10)
E = 512


def load(path):
    z = np.load(path)
    return {k: z[k] for k in ("call", "T", "k", "ids", "w")}


def groups(d):
    """Split a ring into (T, k) groups shaped [n, T, k]."""
    out = {}
    for T, k in sorted({(int(a), int(b)) for a, b in zip(d["T"], d["k"])}):
        sel = (d["T"] == T) & (d["k"] == k)
        n = int(sel.sum())
        R = T * k
        ids = d["ids"][sel][:, :R].astype(np.int32).reshape(n, T, k)
        w = d["w"][sel][:, :R].astype(np.float64).reshape(n, T, k)
        out[(T, k)] = (ids, w, d["call"][sel])
    return out


def multiplicity(ids):
    """[n, T, k] -> how many rows of the same call route to that expert."""
    n = ids.shape[0]
    flat = (np.arange(n, dtype=np.int64)[:, None, None] * E + ids).ravel()
    cnt = np.bincount(flat, minlength=n * E).reshape(n, E)
    # a row that picks the same expert twice cannot happen (top-k is distinct),
    # so #routes to an expert == #rows using it
    return cnt[np.arange(n)[:, None, None], ids]


def ranks(w):
    """[n, T, k] -> 0-based rank by descending weight within the row."""
    order = np.argsort(-w, axis=2, kind="stable")
    r = np.empty_like(order)
    np.put_along_axis(r, order, np.arange(w.shape[2])[None, None, :], axis=2)
    return r


def report(name, ids, w, calls):
    n, T, k = ids.shape
    mult = multiplicity(ids)
    rk = ranks(w)
    single = mult == 1
    rowsum = w.sum(axis=2)
    D = np.array([len(np.unique(ids[i])) for i in range(min(n, 4096))])
    pos_is_rank = float((rk == np.arange(k)[None, None, :]).mean())

    print(f"\n=== {name}   T={T} k={k}  calls={n} "
          f"(call idx {int(calls.min())}..{int(calls.max())}) ===")
    print(f"  D mean={D.mean():.2f} p50={np.percentile(D,50):.0f} "
          f"p90={np.percentile(D,90):.0f}   "
          f"row weight sum: mean={rowsum.mean():.5f} min={rowsum.min():.4f} "
          f"max={rowsum.max():.4f}   stored order == weight rank: {pos_is_rank:.3f}")
    print(f"  singleton routes/call: {single.sum(axis=(1,2)).mean():.2f} "
          f"of {T*k} routes; non-singleton {(~single).sum(axis=(1,2)).mean():.2f}")

    # --- distribution of singleton routes by rank -------------------------
    print("  singleton routes by rank (mean per call, and mean weight):")
    hdr = "    rank      " + "".join(f"{r+1:>8}" for r in range(k))
    print(hdr)
    cnt_r = [(single & (rk == r)).sum(axis=(1, 2)).mean() for r in range(k)]
    print("    count     " + "".join(f"{c:8.2f}" for c in cnt_r))
    wm_r = []
    for r in range(k):
        m = single & (rk == r)
        wm_r.append(w[m].mean() if m.any() else float("nan"))
    print("    mean w    " + "".join(f"{v:8.4f}" for v in wm_r))
    all_r = [(rk == r).sum(axis=(1, 2)).mean() for r in range(k)]
    print("    all routes" + "".join(f"{c:8.2f}" for c in all_r))
    frac = [c / a if a else None for c, a in zip(cnt_r, all_r)]  # None: no route at that rank
    print("    singleton frac" + "".join("     n/a" if v is None else f"{v:8.3f}" for v in frac))

    # --- distribution of singleton routes by weight ------------------------
    edges = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 1.01]
    print("  singleton routes by weight bucket (mean per call):")
    print("    bucket    " + "".join(
        f"{f'<{edges[i+1]:g}':>9}" for i in range(len(edges) - 1)))
    row = []
    for i in range(len(edges) - 1):
        m = single & (w >= edges[i]) & (w < edges[i + 1])
        row.append(m.sum(axis=(1, 2)).mean())
    print("    count     " + "".join(f"{v:9.2f}" for v in row))
    row2 = []
    for i in range(len(edges) - 1):
        m = (~single) & (w >= edges[i]) & (w < edges[i + 1])
        row2.append(m.sum(axis=(1, 2)).mean())
    print("    non-singl " + "".join(f"{v:9.2f}" for v in row2))

    # --- the tau sweep -----------------------------------------------------
    print("  prune rule sweep (drop singleton routes, never rank 1):")
    print("    rule            dD/call   D_after   dD%    mass removed/row   "
          "rows touched  worst row")
    base = D.mean()
    rules = [(f"tau={t:g}", single & (w < t) & (rk > 0)) for t in TAUS]
    for r in (1, 2, 3, 4):
        rules.append((f"rank {k-r+1}-{k}", single & (rk >= k - r)))
    for t in (0.08, 0.09, 0.10):
        for r in (2, 3, 4):
            rules.append((f"rk>={k-r+1} & tau={t:g}",
                          single & (rk >= k - r) & (w < t)))
    rules.append(("all singletons", single & (rk > 0)))
    for label, m in rules:
        dD = m.sum(axis=(1, 2)).mean()
        mass = (w * m).sum(axis=2)          # [n, T] per-row removed mass
        touched = (m.any(axis=2)).mean()
        print(f"    {label:<16}{dD:7.2f}{base-dD:10.2f}"
              f"{100*dD/base:7.1f}{mass.mean():16.4f}"
              f"{touched:14.3f}{mass.max():11.4f}")
    return {"T": T, "k": k, "calls": n, "D": base}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--glob")
    a = ap.parse_args()
    paths = list(a.paths) + (sorted(globmod.glob(a.glob)) if a.glob else [])
    missing = [p for p in paths if not os.path.isfile(p)]
    if not paths or missing:
        ap.error("no census files" + (": missing " + ", ".join(missing) if missing else "") +
                 " (route-census .npz snapshots are not published in this repository)")
    for p in paths:
        d = load(p)
        for (T, k), (ids, w, calls) in groups(d).items():
            report(os.path.basename(p), ids, w, calls)


if __name__ == "__main__":
    main()
