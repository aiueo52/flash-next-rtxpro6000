#!/usr/bin/env python3
"""Summarize a torch-profiler chrome trace: per-kernel GPU time, GPU busy vs wall, gaps."""
import gzip, json, sys, collections
if len(sys.argv) < 2 or not all(__import__("os").path.isfile(a) for a in sys.argv[1:2]):
    sys.exit("usage: analyze_trace.py <trace.json[.gz]>  -- needs a torch-profiler trace from prof/profile_decode2.py; traces are not published in this repository")
p = sys.argv[1]
op = gzip.open if p.endswith(".gz") else open
with op(p, "rt") as f: tr = json.load(f)
ev = tr["traceEvents"] if isinstance(tr, dict) else tr
kern = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
cpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("cpu_op", "cuda_runtime", "user_annotation")]
kern.sort(key=lambda e: e["ts"])
if not kern: print("no kernels"); sys.exit()
t_start, t_end = kern[0]["ts"], max(e["ts"] + e["dur"] for e in kern)
wall = t_end - t_start
# GPU busy (union of intervals across streams)
busy = 0; cur_s, cur_e = None, None
for e in kern:
    s, en = e["ts"], e["ts"] + e["dur"]
    if cur_e is None or s > cur_e:
        if cur_e is not None: busy += cur_e - cur_s
        cur_s, cur_e = s, en
    else: cur_e = max(cur_e, en)
busy += cur_e - cur_s
print(f"kernels={len(kern)} wall={wall/1000:.1f}ms gpu_busy={busy/1000:.1f}ms ({100*busy/wall:.1f}%)")
# per name aggregate
agg = collections.defaultdict(lambda: [0, 0.0])
for e in kern:
    a = agg[e["name"][:110]]; a[0] += 1; a[1] += e["dur"]
rows = sorted(agg.items(), key=lambda kv: -kv[1][1])
print(f"{'total_ms':>9} {'count':>6} {'avg_us':>8}  name")
for name, (c, d) in rows[:45]:
    print(f"{d/1000:9.2f} {c:6d} {d/c:8.1f}  {name}")
# gaps > 200us
gaps = []
prev_end = kern[0]["ts"] + kern[0]["dur"]
for e in kern[1:]:
    if e["ts"] > prev_end + 200: gaps.append((e["ts"] - prev_end, e["ts"], e["name"][:60]))
    prev_end = max(prev_end, e["ts"] + e["dur"])
print(f"\ngaps>200us: {len(gaps)} total={sum(g[0] for g in gaps)/1000:.1f}ms")
for g in sorted(gaps, key=lambda x: -x[0])[:15]: print(f"  {g[0]:8.0f}us before {g[2]} @{(g[1]-t_start)/1000:.1f}ms")
# user annotations (step markers)
ann = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "user_annotation"]
names = collections.Counter(e["name"][:60] for e in ann)
print("\nannotations:", names.most_common(12))
