"""Draft-tail glue in one profiled arm (bench/dt1/arm_dt.sh): per 20-step torch-profiler trace,
  - draft phase: draft forwards per step (= _draft_topk1_finalize count), registry copies captured in
    the draft graph per forward (memcpy32_post + direct_copy, graph-launched; DT-a removes ~6),
    kernels between the draft lm_head GEMV and the topk1 partial (the bf16->fp32 cast; DT-b: 0),
    GPU span of the draft phase by forward count, the eager epilogue after the draft graph (cat,
    mask fill, retrieve fill, build_tree_efficient; DT-t replaces the last three but the mask fill);
  - draft_extend: kernels after the last graph kernel (the eager tail; DT-d/DT-p0 shrink it), their
    summed duration and wall, and the fused select kernel counts.
Compare an A trace with a B trace line by line. All numbers are measured from the trace (us).

  python bench/dt1/dt_trace.py runs/dt1/traces/B1-code-edit
"""
import collections
import glob
import gzip
import json
import os
import statistics
import sys

PHASES = ("draft", "draft_extend")


def phase_of(name):
    if name in PHASES:
        return name
    return "verify" if name.startswith("step[TARGET_VERIFY") else None


def load(path):
    ev = json.load(gzip.open(path, "rt"))
    ev = ev["traceEvents"] if isinstance(ev, dict) else ev
    xs = [e for e in ev if e.get("ph") == "X"]
    kern = sorted(
        (e for e in xs if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")),
        key=lambda e: e["ts"],
    )
    rt = {
        (e.get("args") or {}).get("correlation"): e
        for e in xs
        if e.get("cat") in ("cuda_runtime", "cuda_driver")
    }
    ann = sorted(
        (e for e in xs if e.get("cat") == "user_annotation" and phase_of(e["name"])),
        key=lambda e: e["ts"],
    )
    spans = []
    for a in ann:
        ks = []
        for k in kern:
            r = rt.get((k.get("args") or {}).get("correlation"))
            if r is not None and a["ts"] <= r["ts"] <= a["ts"] + a["dur"]:
                ks.append((k, r["name"] == "cudaGraphLaunch"))
        if ks:
            spans.append((phase_of(a["name"]), ks))
    return spans


def grid0(k):
    g = (k.get("args") or {}).get("grid") or [0]
    return g[0]


def med(v):
    return f"{statistics.median(v):8.2f}" if v else "       -"


def summarize(spans):
    out = collections.defaultdict(list)
    span_by_fw = collections.defaultdict(list)
    names = collections.Counter()
    for name, ks in spans:
        kk = [k for k, _ in ks]
        for k in kk:
            if "_draft_extend_select" in k["name"] or "_chain_tree_topk1" in k["name"]:
                names[k["name"]] += 1
            elif k["name"].startswith("build_tree_efficient"):
                names["build_tree_efficient"] += 1
        span = max(k["ts"] + k["dur"] for k in kk) - min(k["ts"] for k in kk)
        if name == "draft":
            fw = sum("_draft_topk1_finalize" in k["name"] for k in kk)
            if fw == 0:
                continue
            span_by_fw[fw].append(span)
            out["draft_forwards"].append(fw)
            copies = sum(
                g and ("memcpy32_post" in k["name"] or "direct_copy_kernel" in k["name"])
                for k, g in ks
            )
            out["graph_copies_per_forward"].append(copies / fw)
            out["kernels_per_forward"].append(len(kk) / fw)
            graph_end = max(k["ts"] + k["dur"] for k, g in ks if g)
            epi = [k for k, g in ks if not g and k["ts"] >= graph_end - 0.01]
            out["epilogue_kernels"].append(len(epi))
            out["epilogue_wall"].append(
                max((k["ts"] + k["dur"] for k in epi), default=graph_end) - graph_end
            )
            for i, k in enumerate(kk):
                if "_w8a16_gemv_kernel" in k["name"] and grid0(k) >= 1000:
                    for j in range(i + 1, min(i + 6, len(kk))):
                        if "_draft_topk1_partial" in kk[j]["name"]:
                            mid = kk[i + 1 : j]
                            out["lmhead_to_partial_kernels"].append(len(mid))
                            out["lmhead_to_partial_us"].append(
                                kk[j]["ts"] - k["ts"] - k["dur"]
                            )
                            break
        elif name == "draft_extend":
            out["draft_extend_span"].append(span)
            graph = [k for k, g in ks if g]
            if not graph:
                continue
            last = max(k["ts"] + k["dur"] for k in graph)
            tail = [k for k, g in ks if not g and k["ts"] >= last - 0.01]
            out["de_tail_kernels"].append(len(tail))
            out["de_tail_dur_sum"].append(sum(k["dur"] for k in tail))
            out["de_tail_wall"].append(
                max((k["ts"] + k["dur"] for k in tail), default=last) - last
            )
        elif name == "verify":
            out["verify_start"].append(min(k["ts"] for k in kk))
    vs = sorted(out.pop("verify_start", []))
    out["step_period"] = [b - a for a, b in zip(vs, vs[1:])]
    return out, span_by_fw, names


def main(trace_dir):
    files = sorted(glob.glob(os.path.join(trace_dir, "*.trace.json.gz")))
    if not files:
        print(f"no traces in {trace_dir}")
        return
    spans = []
    for f in files:
        spans += load(f)
    out, span_by_fw, names = summarize(spans)
    fw = out.get("draft_forwards", [])
    print(f"draft phases {len(fw)}  draft forwards/step mean {statistics.fmean(fw) if fw else 0:.2f}")
    for key in (
        "graph_copies_per_forward",
        "kernels_per_forward",
        "epilogue_kernels",
        "epilogue_wall",
        "lmhead_to_partial_kernels",
        "lmhead_to_partial_us",
        "de_tail_kernels",
        "de_tail_dur_sum",
        "de_tail_wall",
        "draft_extend_span",
        "step_period",
    ):
        v = out.get(key, [])
        print(f"{key:26s} n {len(v):4d}  median {med(v)}")
    for n in sorted(span_by_fw):
        print(f"draft_span_fw{n:<2d}             n {len(span_by_fw[n]):4d}  median {med(span_by_fw[n])}")
    for k in (
        "_draft_extend_select_partial_kernel",
        "_draft_extend_select_finalize_kernel",
        "_chain_tree_topk1_kernel",
        "_build_tree_efficient",
    ):
        print(f"{k[1:]:26s} count {names.get(k[1:] if k == '_build_tree_efficient' else k, 0)}")


if __name__ == "__main__":
    main(sys.argv[1])
