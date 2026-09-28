#!/usr/bin/env python3
"""EXCLUSIVE (critical-path) GPU time per kernel family, from a torch-profiler chrome trace.

Motivation
----------
`trimmed_step.py` reports `busy_ms` = the interval *union* of a phase's kernel intervals, which is
the right total, but it says nothing about *which* kernel owns which microsecond of it.  Summing
raw durations is worse: kernels land on up to 8 GPU streams (a CUDA graph's parallel branches) and
raw sums over-count the union by 10-15 %.

This tool answers the deletion question directly:

    exclusive(k) = the amount of wall time during which k is the ONLY thing running on the GPU.

A kernel that is fully covered by another stream's kernel has exclusive == 0: deleting it would
save nothing.  Time during which N > 1 kernels overlap is charged to *nobody* -- deleting any one
of them saves nothing either, so it is reported only in aggregate (`shared`).  The identity

    sum(exclusive) + shared + gap_no_kernel == wall

holds exactly, so the table is a partition of the step, not an estimate.

Grouping
--------
Families are (readable kernel label, grid shape).  Grid is part of the key on purpose: one kernel
*name* covers many shapes here (`_w8a16_gemv_kernel` runs at 4/5/6/13/24/55 us in the draft and has
a 420 us target-lm_head instance in verify), and a family that mixes them is meaningless.
`gpu_memcpy` / `gpu_memset` events are included.

Phase attribution reuses `trimmed_step.py`'s rule: a kernel belongs to the phase whose
`user_annotation` span contains its *launching* `cuda_runtime` / `cuda_driver` event (CUDA-graph
kernels inherit the phase of the `cudaGraphLaunch` that replayed them).  Exclusive slices are
charged to the phase of the kernel that owns them.

Usage
-----
    exclusive_time.py <trace.json.gz | trace-dir> [...] [--top N] [--phase NAME] [--csv OUT]
    exclusive_time.py <dir> --draft-forwards      # W16 draft phase, per draft forward
"""
import argparse, bisect, collections, glob, gzip, json, os, re, statistics, sys

PHASES = ("draft", "step[TARGET_VERIFY bs=1]", "draft_extend")
PHASE_SHORT = {"draft": "draft", "step[TARGET_VERIFY bs=1]": "verify", "draft_extend": "draft_extend"}
TINY_US = 2.0  # "tiny kernel running alone" threshold

# ---------------------------------------------------------------- kernel labels
# Ordered (regex, label) rules.  First match wins.  Anything unmatched falls back to a truncation
# of the name, so a new kernel shows up as itself rather than being silently merged.
def _cpp_head(name):
    """`void ns::foo<T,U>(args)` -> `ns::foo`.  Cut at the first `<` or `(`."""
    n = name[5:] if name.startswith("void ") else name
    cut = min([i for i in (n.find("<"), n.find("(")) if i > 0] or [len(n)])
    return n[:cut].strip()

_AT_FUNCTOR = re.compile(r"at::native::(?:\(anonymous namespace\)::)?(\w+)")

def _at_native_label(name):
    """`at::native::<outer>_kernel<..., <Functor>, ...>` -- keep the outer kernel plus the functor,
    which is what distinguishes a fill from a copy from an index_put."""
    m = _AT_FUNCTOR.search(name)
    if m is None:
        return _cpp_head(name)
    outer = m.group(1)
    rest = [x.group(1) for x in _AT_FUNCTOR.finditer(name)][1:]
    pick = next((x for x in rest
                 if x not in ("gpu_kernel_impl_nocast", "gpu_index_kernel", "TensorIteratorBase",
                              "OpaqueType")), "")
    if pick == "direct_copy_kernel_cuda":
        pick = "direct_copy"
    return f"at::{outer}<{pick}>" if pick else f"at::{outer}"

RULES = [
    # grouped MoE GEMMs are renamed to GEMM1/GEMM2 later, by position in the MoE chain
    (re.compile(r"^_ZN7cutlass13device_kernel.*GroupProblemShape"),
     lambda n: "cutlass_moe_grouped_gemm"),
    (re.compile(r"cutlass_80_wmma_tensorop_(\w+)"),
     lambda n: "cutlass80_wmma_" + re.search(r"cutlass_80_wmma_tensorop_(\w+)", n).group(1)[:30]),
    (re.compile(r"cublasLt::splitKreduce_kernel"), lambda n: "cublasLt::splitKreduce_kernel"),
    (re.compile(r"gemvx::kernel"), lambda n: "cublas gemvx::kernel"),
    (re.compile(r"tensorrt_llm::kernels::cutlass_kernels::(\w+)"),
     lambda n: "trtllm::" + re.search(r"cutlass_kernels::(\w+)", n).group(1)),
    (re.compile(r"^kernel_cutlass_gdn_decode_bf16state_mtp"),
     lambda n: "flashinfer gdn_decode_bf16state_mtp"),
    (re.compile(r"gdn_decode_bf16_wy_output_only"),
     lambda n: "flashinfer gdn_decode_bf16_wy_output_only"),
    (re.compile(r"flashinfernormkernelsrmsnorm"), lambda n: "flashinfer RMSNormKernel"),
    (re.compile(r"^kernel_mha"), lambda n: "kernel_mha (trtllm-gen)"),
    (re.compile(r"at::native::"), lambda n: _at_native_label(n)),
]

def label(name, cat):
    if cat == "gpu_memcpy":
        return "Memcpy " + name.replace("Memcpy ", "").split(" (")[0].strip()
    if cat == "gpu_memset":
        return "Memset"
    for rx, fn in RULES:
        if rx.search(name):
            return fn(name)
    return _cpp_head(name)[:56] or name[:56]

# ---------------------------------------------------------------- trace loading
def load(path):
    if os.path.isdir(path):
        cands = sorted(glob.glob(os.path.join(path, "*.gz")))
        if not cands:
            raise SystemExit(f"no *.gz in {path}")
        path = cands[0]
    op = gzip.open(path, "rt") if path.endswith(".gz") else open(path)
    ev = json.load(op)["traceEvents"]
    return path, [e for e in ev if e.get("ph") == "X"]

def _innermost(ann, starts, ts, back):
    i = bisect.bisect_right(starts, ts) - 1
    best = None
    for j in range(i, max(-1, i - back), -1):
        a = ann[j]
        if a["ts"] <= ts <= a["ts"] + a["dur"] and (best is None or a["dur"] < best["dur"]):
            best = a
    return best

def phase_map(X):
    """ts -> phase resolver.  Innermost of draft / verify / draft_extend; a kernel launched under
    `scheduler.*` but outside those three is bucketed as `[<enclosing annotation>]` rather than
    dropped, so the exclusive table stays a complete partition of the step."""
    allann = sorted([e for e in X if e.get("cat") == "user_annotation"], key=lambda e: e["ts"])
    ann = [a for a in allann if a["name"] in PHASES]
    starts = [a["ts"] for a in ann]
    astarts = [a["ts"] for a in allann]
    def phase_of(ts):
        b = _innermost(ann, starts, ts, 8)
        if b is not None:
            return b["name"]
        b = _innermost(allann, astarts, ts, 60)
        return "[" + b["name"].replace("scheduler.", "") + "]" if b is not None else "[unannotated]"
    return ann, phase_of

# ---------------------------------------------------------------- core analysis
def analyze(path, window="steps"):
    """Returns a dict of aggregates.  Everything is per decode step (draft-start to draft-start)."""
    path, X = load(path)
    kern = [e for e in X if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    rt = {(e.get("args") or {}).get("correlation"): e
          for e in X if e.get("cat") in ("cuda_runtime", "cuda_driver")}
    ann, phase_of = phase_map(X)
    dstarts = sorted(a["ts"] for a in ann if a["name"] == "draft")
    if len(dstarts) < 2:
        raise SystemExit(f"{path}: fewer than 2 draft annotations")
    t0, t1 = dstarts[0], dstarts[-1]
    steps = len(dstarts) - 1
    wall_per_step = (t1 - t0) / steps

    # Grouped MoE GEMMs: rename by position within the MoE chain.  A chain begins at the routing
    # prologue (`fusedBuildExpertMapsSortFirstTokenAndStrides`); the first grouped GEMM after it is
    # GEMM1 (gate/up), the second is GEMM2 (down).  Done on ts order over the whole trace.
    order = sorted(range(len(kern)), key=lambda i: kern[i]["ts"])
    gemm_ord = {}
    seen = 0
    for i in order:
        n = kern[i]["name"]
        if "fusedBuildExpertMapsSort" in n:
            seen = 0
        elif n.startswith("_ZN7cutlass13device_kernel") and "GroupProblemShape" in n:
            seen += 1
            gemm_ord[i] = seen

    # per-kernel phase + family key
    keys, unattributed = [], 0
    for k in kern:
        c = (k.get("args") or {}).get("correlation")
        r = rt.get(c)
        ph = phase_of(r["ts"]) if r is not None else None
        if ph is None or ph.startswith("["):
            unattributed += 1
            ph = ph or "[no-launch-site]"
        g = (k.get("args") or {}).get("grid")
        grid = "-" if g is None else "[" + ",".join(str(v) for v in g) + "]"
        lab = label(k["name"], k.get("cat"))
        if lab == "cutlass_moe_grouped_gemm":
            lab += str(gemm_ord.get(len(keys), "?"))
        keys.append((PHASE_SHORT.get(ph, ph), lab, grid))

    # restrict to the measurement window
    idx = [i for i, k in enumerate(kern) if k["ts"] + k["dur"] > t0 and k["ts"] < t1]

    raw = collections.Counter(); cnt = collections.Counter()
    durs = collections.defaultdict(list)
    for i in idx:
        k = kern[i]
        d = min(k["ts"] + k["dur"], t1) - max(k["ts"], t0)
        raw[keys[i]] += d
        cnt[keys[i]] += 1
        durs[keys[i]].append(k["dur"])

    # sweep line.  Per-step aggregates are kept separately so the report can show a *median*
    # step: GPU-idle gaps here are dominated by occasional external preemption (desktop
    # compositor / Max-Q power cap), which makes the mean over the window unrepresentative.
    step_bounds = dstarts
    step_gap = [0.0] * steps
    step_tiny = [0.0] * steps
    step_excl = [0.0] * steps
    step_shared = [0.0] * steps
    def add_step(a, b, arr, v):
        i = max(0, bisect.bisect_right(step_bounds, a) - 1)
        if i < steps:
            arr[i] += v
    evts = []
    for i in idx:
        k = kern[i]
        evts.append((max(k["ts"], t0), 1, i))
        evts.append((min(k["ts"] + k["dur"], t1), -1, i))
    evts.sort(key=lambda t: (t[0], t[1]))
    excl = collections.Counter()
    partners = collections.defaultdict(collections.Counter)
    active = set(); prev = t0
    gap = 0.0; shared = 0.0; tiny_alone = 0.0
    conc_hist = collections.Counter()
    for ts, d, i in evts:
        if ts > prev:
            dt = ts - prev
            n = len(active)
            conc_hist[min(n, 8)] += dt
            if n == 0:
                gap += dt; add_step(prev, ts, step_gap, dt)
            elif n == 1:
                j = next(iter(active))
                excl[keys[j]] += dt
                add_step(prev, ts, step_excl, dt)
                if kern[j]["dur"] < TINY_US:
                    tiny_alone += dt; add_step(prev, ts, step_tiny, dt)
            else:
                shared += dt; add_step(prev, ts, step_shared, dt)
                al = list(active)
                for x in al:
                    for y in al:
                        if x != y:
                            partners[keys[x]][keys[y]] += dt / (n - 1)
            prev = ts
        if d == 1:
            active.add(i)
        else:
            active.discard(i)
    if t1 > prev:
        gap += t1 - prev
        conc_hist[0] += t1 - prev
        add_step(prev, t1, step_gap, t1 - prev)

    walls = [b - a for a, b in zip(dstarts, dstarts[1:])]
    return dict(path=path, steps=steps, wall=wall_per_step, excl=excl, raw=raw, cnt=cnt,
                durs=durs, gap=gap, shared=shared, tiny_alone=tiny_alone, partners=partners,
                unattributed=unattributed, conc=conc_hist, nkern=len(idx),
                med=dict(wall=statistics.median(walls), gap=statistics.median(step_gap),
                         tiny=statistics.median(step_tiny), excl=statistics.median(step_excl),
                         shared=statistics.median(step_shared)))

# ---------------------------------------------------------------- draft-forward split
def draft_forwards(path):
    """Split the `draft` phase into individual draft forwards.  The MTP head runs one MoE routing
    prologue (`fusedBuildExpertMapsSortFirstTokenAndStrides`) per forward, so consecutive
    prologues inside a draft annotation delimit the forwards."""
    path, X = load(path)
    kern = sorted([e for e in X if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")],
                  key=lambda e: e["ts"])
    rt = {(e.get("args") or {}).get("correlation"): e
          for e in X if e.get("cat") in ("cuda_runtime", "cuda_driver")}
    ann, phase_of = phase_map(X)
    n_fw = []
    for a in [x for x in ann if x["name"] == "draft"]:
        c = 0
        for k in kern:
            r = rt.get((k.get("args") or {}).get("correlation"))
            if r is not None and a["ts"] <= r["ts"] <= a["ts"] + a["dur"] \
               and "fusedBuildExpertMapsSort" in k["name"]:
                c += 1
        n_fw.append(c)
    return statistics.median(n_fw) if n_fw else 0

# ---------------------------------------------------------------- reporting
def report(res, top=40, phase_filter=None, tag=""):
    steps = res["steps"]; wall = res["wall"]
    tot_excl = sum(res["excl"].values())
    print(f"\n=== {tag or res['path'].split('/')[-2]}  steps={steps}  "
          f"wall={wall:.0f} us/step  kernels={res['nkern']/steps:.0f}/step ===")
    print(f"    exclusive={tot_excl/steps:8.1f}  shared(>=2 concurrent)={res['shared']/steps:8.1f}  "
          f"gap(no kernel)={res['gap']/steps:6.1f}  [sum={(tot_excl+res['shared']+res['gap'])/steps:.1f}]")
    print(f"    tiny(<{TINY_US:.0f}us)-alone={res['tiny_alone']/steps:6.1f} us/step   "
          f"unattributed-phase kernels={res['unattributed']/steps:.1f}/step")
    m = res["med"]
    print(f"    MEDIAN step:  wall={m['wall']:8.1f}  exclusive={m['excl']:8.1f}  shared={m['shared']:7.1f}  "
          f"gap={m['gap']:6.1f}  tiny-alone={m['tiny']:6.1f}")
    ch = res["conc"]; tc = sum(ch.values()) or 1
    print("    concurrency histogram (% of window): " +
          "  ".join(f"{n}:{v/tc*100:4.1f}%" for n, v in sorted(ch.items())))
    per_ph = collections.Counter()
    for k, v in res["excl"].items():
        per_ph[k[0]] += v
    print("    exclusive by phase (us/step): " + "  ".join(
        f"{n}={v/steps:.1f}" for n, v in per_ph.most_common()))
    rows = sorted(res["excl"], key=lambda k: -res["excl"][k])
    if phase_filter:
        rows = [r for r in rows if r[0] == phase_filter]
    tot_excl = sum(res["excl"][r] for r in rows)
    hdr = f"{'phase':12s} {'kernel family':42s} {'grid':14s} {'n/step':>7s} {'med_us':>7s} {'EXCL/st':>8s} {'%step':>6s} {'raw/st':>8s} {'excl/raw':>8s}"
    print(hdr); print("-" * len(hdr))
    shown = 0.0
    for k in rows[:top]:
        e = res["excl"][k] / steps
        r = res["raw"][k] / steps
        shown += e
        print(f"{k[0]:12s} {k[1][:42]:42s} {k[2][:14]:14s} {res['cnt'][k]/steps:7.1f} "
              f"{statistics.median(res['durs'][k]):7.2f} {e:8.1f} {e/wall*100:6.2f} {r:8.1f} "
              f"{(e/r*100 if r else 0):7.1f}%")
    rest = tot_excl / steps - shown
    print(f"{'(tail)':12s} {'%d further families' % max(0, len(rows) - top):42s} {'':14s} {'':7s} {'':7s} {rest:8.1f} {rest/wall*100:6.2f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="*")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--phase", default=None, help="draft | verify | draft_extend")
    ap.add_argument("--draft-forwards", action="store_true")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--partners", type=int, default=0,
                    help="for the top-N exclusive families, show who they overlap with")
    a = ap.parse_args()
    if not a.traces:
        ap.error("no traces given: needs torch-profiler traces from prof/profile_decode2.py "
                 "(traces are not published in this repository)")
    missing = [t for t in a.traces if not os.path.exists(t)]
    if missing:
        ap.error("no such trace: " + ", ".join(missing) + " (torch-profiler traces are not published in this repository)")
    rows_csv = []
    for t in a.traces:
        res = analyze(t)
        tag = t.rstrip("/").split("/")[-1]
        report(res, top=a.top, phase_filter=a.phase, tag=tag)
        if a.partners:
            rows = sorted(res["excl"], key=lambda k: -res["excl"][k])
            if a.phase:
                rows = [r for r in rows if r[0] == a.phase]
            print("\n    overlap partners (us/step of shared time, top 3 each):")
            for k in rows[:a.partners]:
                ps = res["partners"].get(k, collections.Counter())
                tot = sum(ps.values()) / res["steps"]
                top3 = "  ".join(f"{q[1][:30]}{q[2]}={v/res['steps']:.1f}"
                                 for q, v in ps.most_common(3))
                print(f"      {k[1][:40]:40s} {k[2][:12]:12s} shared={tot:7.1f}  {top3}")
        if a.draft_forwards:
            print(f"    draft forwards per step (median, by MoE routing prologue count): {draft_forwards(t):.0f}")
        if a.csv:
            for k, v in res["excl"].items():
                rows_csv.append((tag, k[0], k[1], k[2], res["cnt"][k] / res["steps"],
                                 statistics.median(res["durs"][k]), v / res["steps"],
                                 res["raw"][k] / res["steps"]))
    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["trace", "phase", "family", "grid", "n_per_step", "median_us",
                        "excl_us_per_step", "raw_us_per_step"])
            w.writerows(rows_csv)
        print(f"\nwrote {a.csv}")

if __name__ == "__main__":
    main()
