# Display time-slice pauses per trace: per-kernel excess over (median of its key + 20 us), plus idle gaps > 100 us,
# summed per step window. "clean" = steps with < 50 us of such excess.
import sys, os, bisect, collections, statistics as st
sys.path.insert(0, os.path.expanduser("~/tools/flash-next-bench/prof"))
import exclusive_time as ET
def run(d):
    path, X = ET.load(d)
    kern = sorted([e for e in X if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")], key=lambda k: k["ts"])
    rt = {(e.get("args") or {}).get("correlation"): e for e in X if e.get("cat") in ("cuda_runtime", "cuda_driver")}
    ann, _ = ET.phase_map(X)
    ds = sorted(a["ts"] for a in ann if a["name"] == "draft")
    srt = sorted((rt[(k.get("args") or {}).get("correlation")]["ts"], k["ts"]) for k in kern
                 if (k.get("args") or {}).get("correlation") in rt)
    ls = [x[0] for x in srt]
    firsts = [min(x[1] for x in srt[bisect.bisect_left(ls, t):bisect.bisect_left(ls, t) + 50]) for t in ds]
    by = collections.defaultdict(list)
    key = lambda k: (k["name"], str((k.get("args") or {}).get("grid")))
    for k in kern: by[key(k)].append(k["dur"])
    med = {q: st.median(v) for q, v in by.items()}
    walls, exc = [], []
    for a, b in zip(firsts[1:-1], firsts[2:]):
        ks = [k for k in kern if a <= k["ts"] < b]
        e = sum(max(0.0, k["dur"] - med[key(k)] - 20) for k in ks)
        end = a
        for k in ks:
            if k["ts"] - end > 100: e += k["ts"] - end
            end = max(end, k["ts"] + k["dur"])
        walls.append((b - a) / 1000); exc.append(e / 1000)
    clean = [w for w, e in zip(walls, exc) if e < 0.05]
    return walls, exc, clean
for d in sys.argv[1:]:
    w, e, c = run(d)
    print(f"{os.path.basename(d):34s} steps {len(w):2d} wall med {st.median(w):6.2f} ms  paused {100*sum(e)/sum(w):4.1f}%  "
          f"clean {len(c):2d} med {st.median(c) if c else float('nan'):6.2f}  wall-minus-pause med {st.median([a-b for a,b in zip(w,e)]):6.2f}")
