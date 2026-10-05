# Sampling fixed cost per step, robust to display time-slice pauses: per-kernel durations are clipped
# to (median of that kernel key + 20 us) before summing the R.eager block per step.
import sys, os, collections, statistics as st
sys.path.insert(0, os.path.expanduser("~/tools/flash-next-bench/prof"))
import fc_map, exclusive_time as ET
def run(d):
    path, X = ET.load(d)
    kern = [e for e in X if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    rt = {(e.get("args") or {}).get("correlation"): e for e in X if e.get("cat") in ("cuda_runtime", "cuda_driver")}
    ann, phase_of = ET.phase_map(X)
    steps = sum(1 for a in ann if a["name"] == "draft")
    keyed = []
    for k in kern:
        r = rt.get((k.get("args") or {}).get("correlation"))
        ph = phase_of(r["ts"]) if r else "[?]"
        ph = ET.PHASE_SHORT.get(ph, ph)
        g = (k.get("args") or {}).get("grid"); grid = "-" if g is None else "[" + ",".join(map(str, g)) + "]"
        lab = ET.label(k["name"], k.get("cat"))
        keyed.append(((ph, lab, grid), k["dur"]))
    by = collections.defaultdict(list)
    for key, dur in keyed: by[key].append(dur)
    med = {k: st.median(v) for k, v in by.items()}
    tot = collections.Counter(); raw = collections.Counter(); per = collections.Counter()
    for key, dur in keyed:
        b = fc_map.block(key)
        tot[b] += min(dur, med[key] + 20); raw[b] += dur
        if b.startswith("R.eager"): per[key[1]] += min(dur, med[key] + 20)
    n = max(steps, 1)
    return n, tot["R.eager(sample/accept/commit)"] / n, raw["R.eager(sample/accept/commit)"] / n, per
for d in sys.argv[1:]:
    n, clip, raw, per = run(d)
    top = sorted(per.items(), key=lambda x: -x[1])[:6]
    print(f"{os.path.basename(d):32s} steps {n:3d}  R.eager clipped {clip:6.1f} us/step (raw {raw:6.1f})  top: " +
          ", ".join(f"{l[:26]} {v/n:.0f}" for l, v in top))
