#!/usr/bin/env python3
"""Per-step wall clock from the SSE chunk timeline of an fnbench run jsonl.

Why: `fnbench` derives `effective_forward_per_second` as `decode_tps / accept_length`
(runner.py:126-132), and both inputs are unsound for measuring step time --
`sglang:spec_accept_length` is a *windowed gauge* sampled once after the run (it read 14.12 on a
W16 code-edit run whose true run-average was 11.0), and `usage.completion_tokens` disagrees with
the streamed token count when thinking is on.  That estimator spreads 17.2-27.9 ms across
workloads inside a single W16 config while the real step time is flat at 20-22 ms.

Instead: `client.timeline` (http_client.py:170-181) holds one entry per token-bearing SSE chunk.
At bs=1 with stream_interval=1 that is one entry per decode step, so

    ms/step = decode_seconds / (len(timeline) - 1)

is a direct wall-clock measurement that needs no server metric.  `tok/step` is printed as the
sanity check: it must land at or below the speculation width (<= num_draft_tokens); if it does
not, chunks are not 1:1 with steps for that run and the ms/step is a multiple of the real one.

Usage: sse_wall.py <run.jsonl> [more jsonl...]  [--per-run]
"""
import json, sys, statistics as st

def pct(v, p):
    v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))]

def rows(path):
    for ln in open(path):
        ln = ln.strip()
        if not ln: continue
        try: j = json.loads(ln)
        except json.JSONDecodeError: continue
        c = j.get("client") or {}
        tl = c.get("timeline") or []
        ds = c.get("decode_seconds")
        if len(tl) < 10 or not ds: continue
        n = len(tl) - 1                       # step intervals
        ct = (c.get("usage") or {}).get("completion_tokens")
        gaps = [x[0] for x in tl[1:]]         # per-chunk ms, integer-rounded by the client
        yield {"workload": j.get("workload") or "?", "label": j.get("label"),
               "repeat": j.get("repeat"), "steps": n, "ms_step": ds / n * 1e3,
               "gaps": gaps, "tok_step": (ct / len(tl)) if ct else None,
               "tok_s": (ct / ds) if ct else None}

per_run = "--per-run" in sys.argv
paths = [a for a in sys.argv[1:] if not a.startswith("--")]
if not paths:
    sys.exit("usage: sse_wall.py <fnbench run.jsonl>... [--per-run]  -- needs unstripped fnbench records with client.timeline")
empty = []
for path in paths:
    rs = list(rows(path))
    print(path)
    if not rs:
        print("  (no usable timelines)"); empty.append(path); continue
    print(f"  label={rs[0]['label']}  runs={len(rs)}")
    print(f"  {'workload':14s} {'runs':>4s} {'ms/step':>8s} {'med':>6s} {'p90':>6s} {'tok/step':>9s} {'tok/s':>8s} {'steps':>7s}")
    by = {}
    for r in rs: by.setdefault(r["workload"], []).append(r)
    for w, v in by.items():
        g = [x for r in v for x in r["gaps"]]
        ms = st.mean([r["ms_step"] for r in v])
        tk = [r["tok_step"] for r in v if r["tok_step"]]
        ts = [r["tok_s"] for r in v if r["tok_s"]]
        print(f"  {w:14s} {len(v):4d} {ms:8.2f} {st.median(g):6.1f} {pct(g,.90):6.1f} "
              f"{(st.mean(tk) if tk else float('nan')):9.2f} {(st.mean(ts) if ts else float('nan')):8.1f} {st.mean([r['steps'] for r in v]):7.0f}")
        if per_run:
            for r in v:
                print(f"      repeat={r['repeat']} ms/step={r['ms_step']:6.2f} steps={r['steps']:5d} "
                      f"tok/step={r['tok_step']:5.2f} tok/s={r['tok_s']:6.1f}")
    allms = [r["ms_step"] for r in rs]
    print(f"  {'ALL':14s} {len(rs):4d} {st.mean(allms):8.2f}  (mean over runs; spread {min(allms):.2f}-{max(allms):.2f})")
if empty:
    sys.exit("sse_wall.py: no SSE timelines in " + ", ".join(empty) + " (the published results/runs records are stripped "
             "and keep only client.sse_chunks; timelines are not published)")
