#!/usr/bin/env python3
"""Fixed-cost map: per functional block, fit t_block(S) = a + b*S over traces taken at several
draft lengths S, and set the intercept a against a physical floor.

Attribution.  Every microsecond of the GPU-side step window is owned by exactly one kernel:
the *oldest* running kernel (earliest start).  With PDL a dependent kernel is resident early and
spins in `griddepcontrol.wait`; charging the overlap to the predecessor keeps the waiter from
stealing its time.  (`FC_ATTR=fair` splits overlaps evenly instead.)  Σ blocks + gap = window.

Step window = first kernel launched from the `draft` annotation of step i .. same of step i+1,
on the GPU clock (the CPU runs ≥ 1 step ahead under the overlap scheduler, so CPU annotations
are not step boundaries on the GPU).

Floors (µs/step, S-independent part):  bytes / 1.6 TB/s for the weights a block streams once per
step (model config + the online FP8/NVFP4 formats of this server), and 0.67 µs per launch in the
chain (PDL launch floor, MEGAKERNEL_SPEC §3.2) where launches dominate.  Per block the floor is
max(bytes floor, launch floor) -- they overlap, so they are not added.

Usage:
    fc_map.py <trace_dir>:<S> [...] [--json OUT]
"""
import argparse, bisect, collections, json, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exclusive_time as ET

BW = 1.6e6          # bytes per µs (1.6 TB/s)
LAUNCH = 0.67       # µs per PDL launch (MEGAKERNEL_SPEC §3.2)
MB = 1e6
# weights streamed per verify step, independent of S (bytes); dims from config.json:
# hidden 2560, 48 layers (36 GDN + 12 full attn), vocab 248320, GDN qkv 10240 + z 6144, out 6144,
# attn q(+gate) 12288 + k 512 + v 512, o 6144, shared expert 640, router 512x2560 bf16,
# HC mix down/up 320x10240 each (FP8), 97 HC mixes/step (2 per layer + the final mixer).
FLOOR_BYTES = {
    "V.lm_head": 248320 * 2560,
    "V.gdn_in_proj": 36 * 16384 * 2560,
    "V.gdn_out_proj": 36 * 2560 * 6144,
    "V.gdn_ba_proj": 36 * 96 * 2560,
    "V.attn_proj": 12 * (13312 * 2560 + 2560 * 6144),
    "V.shared_expert+router": 48 * (3 * 640 * 2560 + 512 * 2560 * 2),
    "V.hc_chain": 97 * 2 * 320 * 10240,
    "V.gdn_core": 36 * 48 * 128 * 128 * 2,              # bf16 recurrent state read
    "V.attn+qsa": 12 * 640 * 2560 * 2,                   # indexer qk proj (bf16); KV is small here
    # MoE at T=1 (S=0): ≤ 10 distinct experts x (gate_up 1280x2560 + down 2560x640) NVFP4 (0.5625 B)
    "V.moe": 48 * 10 * (1280 * 2560 + 2560 * 640) * 0.5625,
    # draft_extend: one MTP forward (hot-vocab lm_head 49152x2560 fp8, attn, fc 2x2560^2 fp8, MoE ~10 experts)
    "X.draft_extend": 49152 * 2560 + (13312 * 2560 + 2560 * 6144) + 2 * 2560 * 2560
                      + 10 * (1280 * 2560 + 2560 * 640) * 0.5625,
}

HC = ("_hc_up", "_hc_down", "_hc_branch_stats", "hc_combine", "grouped_gemma_rmsnorm")
ATTN = ("kernel_mha", "_fused_sigmoid_mul", "_fused_qk_rmsnorm_rope_gate", "_fp8_kv_store",
        "_fa2_valid_counts", "kernel_kernel", "fast_topk", "_compact_kv", "_expand_qsa", "qsa_index",
        "cutlass80_wmma", "splitKreduce", "_mtp_shared_sparse_indices_lookup", "_qsa_graph")

def block(key):
    ph, lab, grid = key
    g = [int(x) for x in grid.strip("[]").split(",")] if grid.startswith("[") else [0]
    P = {"verify": "V", "draft": "D", "draft_extend": "X"}.get(ph, "R")
    if P == "R":
        return "R.gdn_state_update" if "bf16state_mtp" in lab else "R.eager(sample/accept/commit)"
    if P == "X":
        return "X.draft_extend"
    if P == "D":
        return "D.draft_forwards"
    if lab.startswith("_w8a16_gemv"):
        if g[0] == 1940: return "V.lm_head"
        if g in ([512, 1, 1], [256, 1, 1]): return "V.gdn_in_proj"
        if g == [40, 4, 1]: return "V.gdn_out_proj"
        if g == [3, 1, 1]: return "V.gdn_ba_proj"
        if g in ([416, 1, 1], [80, 6, 1]): return "V.attn_proj"
        return "V.shared_expert+router"
    if "_router_triton" in lab or "_shared_expert_gate" in lab or "act_and_mul" in lab:
        return "V.shared_expert+router"
    if "grouped_gemm1" in lab: return "V.moe_gemm1"
    if "grouped_gemm2" in lab: return "V.moe_gemm2"
    if "fusedBuildExpertMaps" in lab or "doActivation" in lab or "trtllm::" in lab:
        return "V.moe_prologue+act"
    if "gdn_decode_bf16_wy" in lab or "conv1d" in lab: return "V.gdn_core"
    if any(h in lab for h in HC): return "V.hc_chain"
    if any(a in lab for a in ATTN): return "V.attn+qsa"
    return "V.glue_other"

def attribute(path, mode=None):
    mode = mode or os.environ.get("FC_ATTR", "oldest")
    path, X = ET.load(path)
    kern = [e for e in X if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    rt = {(e.get("args") or {}).get("correlation"): e for e in X if e.get("cat") in ("cuda_runtime", "cuda_driver")}
    ann, phase_of = ET.phase_map(X)
    ds = sorted(a["ts"] for a in ann if a["name"] == "draft")
    lk = []
    for k in kern:
        r = rt.get((k.get("args") or {}).get("correlation"))
        lk.append(r["ts"] if r else None)
    firsts = []
    srt = sorted((l, k["ts"]) for k, l in zip(kern, lk) if l is not None)
    ls = [x[0] for x in srt]
    for t in ds:
        j = bisect.bisect_left(ls, t)
        firsts.append(min(x[1] for x in srt[j:j + 50]))
    t0, t1 = firsts[1], firsts[-1]
    steps = len(firsts) - 2
    order = sorted(range(len(kern)), key=lambda i: kern[i]["ts"])
    gemm, seen = {}, 0
    for i in order:
        n = kern[i]["name"]
        if "fusedBuildExpertMapsSort" in n: seen = 0
        elif n.startswith("_ZN7cutlass13device_kernel") and "GroupProblemShape" in n:
            seen += 1; gemm[i] = seen
    keys = []
    for i, k in enumerate(kern):
        ph = phase_of(lk[i]) if lk[i] is not None else "[?]"
        ph = ET.PHASE_SHORT.get(ph, ph)
        g = (k.get("args") or {}).get("grid")
        grid = "-" if g is None else "[" + ",".join(map(str, g)) + "]"
        lab = ET.label(k["name"], k.get("cat"))
        if lab == "cutlass_moe_grouped_gemm": lab += str(gemm.get(i, "?"))
        keys.append((ph, lab, grid))
    idx = [i for i, k in enumerate(kern) if k["ts"] + k["dur"] > t0 and k["ts"] < t1]
    ev = []
    for i in idx:
        k = kern[i]
        ev.append((max(k["ts"], t0), 1, i)); ev.append((min(k["ts"] + k["dur"], t1), -1, i))
    ev.sort(key=lambda t: (t[0], t[1]))
    own = collections.Counter(); cnt = collections.Counter(); act = set(); prev = t0; gap = 0.0
    for ts, d, i in ev:
        if ts > prev:
            dt = ts - prev
            if not act: gap += dt
            elif mode == "fair":
                for j in act: own[block(keys[j])] += dt / len(act)
            else:
                own[block(keys[min(act, key=lambda j: (kern[j]["ts"], j))])] += dt
            prev = ts
        (act.add if d == 1 else act.discard)(i)
    gap += max(0.0, t1 - prev)
    for i in idx: cnt[block(keys[i])] += 1
    own = {b: v / steps for b, v in own.items()}
    own["~gap(no kernel)"] = gap / steps
    walls = [b - a for a, b in zip(firsts, firsts[1:])][1:-1] or [(t1 - t0) / steps]
    return dict(own=own, cnt={b: v / steps for b, v in cnt.items()}, steps=steps,
                win=(t1 - t0) / steps, medwall=statistics.median(walls))

def floor_of(b, calls):
    if b.startswith("D."):
        return 0.0, 0.0     # S-1 draft forwards: no S-independent part by construction
    fb = FLOOR_BYTES.get(b)
    if b.startswith("V.moe_gemm"):
        fb = None
    byt = (fb / BW) if fb else 0.0
    return max(byt, LAUNCH * calls), byt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="trace_dir:S")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    import numpy as np
    data = []
    for r in a.runs:
        d, s = r.rsplit(":", 1)
        res = attribute(d)
        data.append((d, int(s), res))
        print(f"{os.path.basename(d.rstrip('/')):42s} S={int(s):2d} steps={res['steps']} "
              f"window={res['win']:.0f} µs  median step={res['medwall']:.0f} µs  gap={res['own']['~gap(no kernel)']:.0f}",
              file=sys.stderr)
    blocks = sorted({b for _, _, r in data for b in r["own"]})
    Sx = np.array([s for _, s, _ in data], float)
    A = np.vstack([np.ones_like(Sx), Sx]).T
    rows = []
    for b in blocks:
        y = np.array([r["own"].get(b, 0.0) for _, _, r in data])
        c = np.array([r["cnt"].get(b, 0.0) for _, _, r in data])
        if len(set(Sx)) > 1:
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        else:
            coef = np.array([y.mean(), 0.0])
        rows.append(dict(block=b, a=float(coef[0]), b=float(coef[1]), calls=float(c.mean()),
                         y=[float(v) for v in y]))
    # MoE floor at T=1 is reported on the combined GEMM row
    for r in rows:
        f, fb = floor_of(r["block"], r["calls"])
        r["floor"], r["bytes_floor"] = f, fb
    moe = [r for r in rows if r["block"].startswith("V.moe_gemm")]
    if moe:
        fm = FLOOR_BYTES["V.moe"] / BW
        tot_a = sum(r["a"] for r in moe)
        for r in moe:
            r["floor"] = fm * (r["a"] / tot_a if tot_a else 0.5)
    rows.sort(key=lambda r: -(r["a"] - r["floor"]))
    print(f"{'block':30s} {'a µs':>8s} {'b µs/S':>8s} {'calls':>7s} {'floor a':>8s} {'a-floor':>8s}   per-trace µs " +
          " ".join(f"S{s}" for _, s, _ in data))
    for r in rows:
        print(f"{r['block']:30s} {r['a']:8.0f} {r['b']:8.1f} {r['calls']:7.1f} {r['floor']:8.0f} {r['a']-r['floor']:8.0f}   " +
              " ".join(f"{v:5.0f}" for v in r["y"]))
    ta = sum(r["a"] for r in rows); tb = sum(r["b"] for r in rows); tf = sum(r["floor"] for r in rows)
    print(f"{'TOTAL':30s} {ta:8.0f} {tb:8.1f} {'':7s} {tf:8.0f} {ta-tf:8.0f}")
    if a.json:
        json.dump(dict(runs=[(d, s, r) for d, s, r in data], rows=rows), open(a.json, "w"), indent=1)

if __name__ == "__main__":
    main()
