#!/usr/bin/env python3
"""Enumerate the fusable "glue islands" of one decode step from a chrome trace.

An island is a maximal run of kernels **whose source this project owns** (Triton
kernels in `python/sglang/srt/...`, the sgl-kernel JIT ops, and the torch
elementwise kernels they could absorb) bounded on both sides by an *external*
kernel that cannot be called from inside another kernel: the CUTLASS grouped
GEMMs, the FlashInfer GDN WY kernel, the FlashInfer/TRT-LLM MoE prologue and
activation, the trtllm-gen attention kernel, cuBLAS.

Usage:
  python bench/megakernel/islands.py <trace.json.gz> ["phase"] [occurrence]

Prints, per island of one occurrence of the phase: kernel count, summed kernel
microseconds, and the kernel names, plus a roll-up by island signature (islands
with the same kernel-name sequence are the same island repeated once per layer).
"""

from __future__ import annotations

import bisect
import collections
import gzip
import json
import sys

# Kernels that cannot be a stage of a fused kernel: someone else's persistent /
# TMA / warp-specialised kernel, or a library kernel with no source here.
EXTERNAL = (
    "cutlass::device_kernel",  # CUTLASS grouped GEMM (MoE GEMM1/GEMM2)
    "_ZN7cutlass13device_kernel",
    "cutlass::Kernel2",  # cuBLAS-selected cutlass wmma gemm
    "cublas",
    "gemvx::kernel",
    "flashinfergdn",  # FlashInfer GDN WY
    "fusedBuildExpertMaps",  # TRT-LLM MoE prologue
    "expandInputRows",
    "doActivationKernel",
    "blockExpertPrefixSum",
    "mergeExpertPrefixSum",
    "computeStridesTma",
    "finalizeMoeRouting",
    "kernel_mha",  # trtllm-gen decode attention
    "qsa_index_q_prep",  # sgl-kernel CUDA, big and self-contained
    "qsa_index_k_compress",
    "fast_topk",
    "memset32",  # sits between the two MoE GEMMs; nothing can absorb it
    "kernel_kernel",  # trtllm-gen indexer
    "multi_tensor_apply",
)


def is_external(name: str) -> bool:
    return any(tok in name for tok in EXTERNAL)


def short(name: str) -> str:
    n = name.split("(")[0]
    for pre in ("void ", "at::native::", "(anonymous namespace)::", "sglang::",
                "tensorrt_llm::kernels::cutlass_kernels::", "cutlass::"):
        n = n.replace(pre, "")
    return n[:52]


def load(path, phase, occ):
    ev = json.load(gzip.open(path, "rt") if path.endswith(".gz") else open(path))
    ev = ev["traceEvents"]
    X = [e for e in ev if e.get("ph") == "X"]
    kern = [e for e in X if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    rt = {(e.get("args") or {}).get("correlation"): e
          for e in X if e.get("cat") == "cuda_runtime"}
    ann = sorted(
        [e for e in X if e.get("cat") == "user_annotation" and e["name"] == phase],
        key=lambda e: e["ts"],
    )
    if not ann:
        avail = collections.Counter(
            e["name"][:50] for e in X if e.get("cat") == "user_annotation"
        )
        raise SystemExit(f"phase {phase!r} not found; available: {avail.most_common(10)}")
    a = ann[min(occ, len(ann) - 1)]
    rows = []
    for k in kern:
        r = rt.get((k.get("args") or {}).get("correlation"))
        if r is None:
            continue
        if a["ts"] <= r["ts"] <= a["ts"] + a["dur"]:
            rows.append((r["ts"], k))
    rows.sort(key=lambda x: x[0])
    return [k for _, k in rows]


def main():
    path = sys.argv[1]
    phase = sys.argv[2] if len(sys.argv) > 2 else "step[TARGET_VERIFY bs=1]"
    occ = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    ks = load(path, phase, occ)

    islands, cur = [], []
    ext_us = 0.0
    for k in ks:
        if is_external(k["name"]):
            ext_us += k["dur"]
            if cur:
                islands.append(cur)
                cur = []
        else:
            cur.append(k)
    if cur:
        islands.append(cur)

    glue_us = sum(k["dur"] for isl in islands for k in isl)
    n_glue = sum(len(isl) for isl in islands)
    print(f"{path}\nphase={phase} occ={occ}: {len(ks)} kernels, "
          f"{sum(k['dur'] for k in ks) / 1e3:.2f} ms of kernel time")
    print(f"  external: {len(ks) - n_glue} kernels, {ext_us / 1e3:.2f} ms")
    print(f"  glue:     {n_glue} kernels, {glue_us / 1e3:.2f} ms "
          f"in {len(islands)} islands "
          f"(mean {n_glue / len(islands):.1f} kernels / {glue_us / len(islands):.1f} us)")

    # roll up islands with the same kernel-name signature
    sig = collections.defaultdict(list)
    span = collections.defaultdict(list)
    for isl in islands:
        key = tuple(short(k["name"]) for k in isl)
        sig[key].append(sum(k["dur"] for k in isl))
        span[key].append(
            max(k["ts"] + k["dur"] for k in isl) - min(k["ts"] for k in isl)
        )
    print(f"\n{'n':>4} {'kern':>4} {'sum(med)':>9} {'span(med)':>10} {'span/step':>10}  island")
    rows = sorted(sig.items(), key=lambda kv: -sum(kv[1]))
    tot_span = 0.0
    for names, durs in rows:
        sp = sorted(span[names])
        durs_s = sorted(durs)
        tot_span += sum(sp)
        print(f"{len(durs):4d} {len(names):4d} {durs_s[len(durs_s) // 2]:9.1f} "
              f"{sp[len(sp) // 2]:10.1f} {sum(sp):10.1f}  {' -> '.join(names)[:90]}")
    print(f"\n  total island wall span this phase: {tot_span / 1e3:.2f} ms "
          f"(vs {glue_us / 1e3:.2f} ms summed kernel time)")

    # per-kernel detail for the islands that carry the time
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for isl in islands:
        k = tuple(short(x["name"]) for x in isl)
        for i, x in enumerate(isl):
            per[k][i].append(x["dur"])
    print("\n--- per-kernel medians of the top islands "
          "(<= 1.5 us kernels are marked * : boundary-dominated) ---")
    for names, durs in rows[:8]:
        if len(durs) < 4:
            continue
        cells = []
        for i, nm in enumerate(names):
            d = sorted(per[names][i])
            m = d[len(d) // 2]
            cells.append(f"{nm.split('<')[0][:34]} {m:.1f}{'*' if m <= 1.5 else ''}")
        print(f"\n  [{len(durs)}x, {len(names)} kernels, med {sorted(durs)[len(durs)//2]:.1f} us]")
        for c in cells:
            print(f"      {c}")


if __name__ == "__main__":
    main()
