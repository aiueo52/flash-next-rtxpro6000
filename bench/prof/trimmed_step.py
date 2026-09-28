#!/usr/bin/env python3
"""Per-step GPU cost from a torch-profiler trace, per phase.

Primary metric is `busy_ms`: the *interval union* of the phase's kernel intervals, i.e. real
GPU-busy time.  Kernels land on several GPU "streams" (a CUDA graph's internal parallel
branches), so summing durations (`raw_ms`) double-counts by 10-15%.

`legacy_trimmed_ms` is the metric this script printed as `trimmed_ms` before 2026-09-05: each
kernel's duration clipped to 2x the median of that kernel *name* + 5us.  It is kept only so old
logs stay interpretable -- do not optimise against it.  It under-reports GPU-busy time by
0.8-3.6 ms/step here, and the error grows with kernel count, because one kernel name covers many
shapes (`_w8a16_gemv_kernel` runs at 4/5/6/13/24/55us in the draft and has a 420us target-lm_head
instance in verify), so a single median per name deletes real work.  It is computed with the
original cuda_runtime-only attribution so it reproduces historical numbers exactly.

Sanity check the output with: sum(busy_ms) + idle_ms ~= wall_ms (residual = kernels whose launch
site falls outside every phase annotation, reported as `unattributed`).

Usage: trimmed_step.py <trace.json.gz> [more traces...]
"""
import gzip, json, sys, bisect, collections, statistics
PHASES = ("draft", "step[TARGET_VERIFY bs=1]", "draft_extend")

def union_ivals(iv):
    """Merge [(start, end), ...] into disjoint intervals (sorted)."""
    out, cs, ce = [], None, None
    for s, e in sorted(iv):
        if ce is None or s > ce:
            if ce is not None: out.append((cs, ce))
            cs, ce = s, e
        else: ce = max(ce, e)
    if ce is not None: out.append((cs, ce))
    return out

def analyze(p):
    ev = json.load(gzip.open(p, "rt") if p.endswith(".gz") else open(p))["traceEvents"]
    X = [e for e in ev if e.get("ph") == "X"]
    kern = [e for e in X if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    # cuLaunchKernelEx (Triton) is cat "cuda_driver", not "cuda_runtime"; both are launch sites.
    rt = {(e.get("args") or {}).get("correlation"): e
          for e in X if e.get("cat") in ("cuda_runtime", "cuda_driver")}
    rt_legacy = {(e.get("args") or {}).get("correlation"): e
                 for e in X if e.get("cat") == "cuda_runtime"}
    ann = sorted([e for e in X if e.get("cat") == "user_annotation" and e["name"] in PHASES],
                 key=lambda e: e["ts"])
    starts = [a["ts"] for a in ann]
    def phase_of(ts):
        i = bisect.bisect_right(starts, ts) - 1
        best = None
        for j in range(i, max(-1, i - 8), -1):
            a = ann[j]
            if a["ts"] <= ts <= a["ts"] + a["dur"] and (best is None or a["dur"] < best["dur"]): best = a
        return (best["name"], best["ts"]) if best else None
    # collect per phase occurrence: kernel intervals (new) and (name, dur) (legacy)
    per_occ = collections.defaultdict(list)      # occ -> [(ts, ts+dur), ...]
    per_occ_leg = collections.defaultdict(list)  # occ -> [(name, dur), ...]
    n_unattrib = 0
    for k in kern:
        c = (k.get("args") or {}).get("correlation")
        r = rt.get(c)
        if r is None or (ph := phase_of(r["ts"])) is None: n_unattrib += 1
        else: per_occ[ph].append((k["ts"], k["ts"] + k["dur"]))
        rl = rt_legacy.get(c)
        if rl is not None and (phl := phase_of(rl["ts"])) is not None:
            per_occ_leg[phl].append((k["name"], k["dur"]))
    # wall clock per decode step (draft start to next draft start) and GPU idle inside it
    dstarts = sorted(a["ts"] for a in ann if a["name"] == "draft")
    walls = [b - a for a, b in zip(dstarts, dstarts[1:])]
    busy_ivals = union_ivals([(e["ts"], e["ts"] + e["dur"]) for e in kern])
    bstart = [s for s, _ in busy_ivals]
    def busy_in(a, b):
        i = max(0, bisect.bisect_right(bstart, a) - 1)
        t = 0
        while i < len(busy_ivals) and busy_ivals[i][0] < b:
            s_, e_ = busy_ivals[i]; t += max(0, min(e_, b) - max(s_, a)); i += 1
        return t
    idles = [w - busy_in(a, a + w) for a, w in zip(dstarts, walls)]
    out = {}
    if walls:
        out["_wall"] = (statistics.median(walls) / 1e3,
                        statistics.median([w - i for w, i in zip(walls, idles)]) / 1e3,
                        statistics.median(idles) / 1e3)
    out["_unattrib"] = n_unattrib / max(1, len(walls))
    for name in PHASES:
        occs = [v for (n, ts), v in per_occ.items() if n == name]
        if not occs: continue
        occs_leg = [v for (n, ts), v in per_occ_leg.items() if n == name]
        med = collections.defaultdict(list)
        for occ in occs_leg:
            for kn, d in occ: med[kn].append(d)
        med = {kn: statistics.median(v) for kn, v in med.items()}
        legacy = [sum(min(d, 2 * med[kn] + 5) for kn, d in occ) for occ in occs_leg] or [0]
        busy = [sum(e - s for s, e in union_ivals(occ)) for occ in occs]
        raw = [sum(e - s for s, e in occ) for occ in occs]
        nk = [len(occ) for occ in occs]
        out[name] = (len(occs), statistics.median(nk), statistics.median(busy) / 1e3,
                     statistics.median(raw) / 1e3, statistics.median(legacy) / 1e3)
    return out

if len(sys.argv) < 2 or not all(__import__("os").path.isfile(a) for a in sys.argv[1:]):
    sys.exit("usage: trimmed_step.py <trace.json.gz> [more...]  -- needs a torch-profiler trace from prof/profile_decode2.py; traces are not published in this repository")
for p in sys.argv[1:]:
    print(p)
    res = analyze(p)
    tot_k = tot_b = tot_l = 0
    wall = busy = idle = float("nan")
    unattrib = res.pop("_unattrib", 0.0)
    if "_wall" in res:
        wall, busy, idle = res.pop("_wall")
        print(f"  step wall={wall:6.2f}ms  gpu_busy={busy:6.2f}ms  gpu_idle_in_step={idle:5.2f}ms")
    for name, (n, nk, b, raw, leg) in res.items():
        print(f"  {name:28s} occ={n:3d} kernels/step={nk:7.0f} busy_ms={b:6.2f} raw_ms={raw:6.2f} legacy_trimmed_ms={leg:6.2f}")
        tot_k += nk; tot_b += b; tot_l += leg
    print(f"  {'TOTAL':28s} kernels/step={tot_k:7.0f} busy_ms={tot_b:6.2f} wall_ms={wall:6.2f} idle_ms={idle:5.2f} legacy_trimmed_ms={tot_l:6.2f}")
    print(f"  {'(check)':28s} sum(busy)+idle={tot_b + idle:6.2f}ms vs wall={wall:6.2f}ms  residual={wall - tot_b - idle:5.2f}ms ({unattrib:.0f} unattributed kernels/step)")
