"""Per-step GPU kernel time from profiler traces, with the display stalls taken out.

Usage: trace_medians.py [--steps 20] LABEL=<trace dir> [LABEL=<trace dir> ...]   (the first arm is the reference)

Another GPU context (the desktop's Xorg/kwin/firefox) preempts our kernels; the time it runs is counted inside
whichever of our kernels was resident, as stalls of 0.2-1.3 ms. A kernel's median ignores those, so
sum(count * median) / steps is the step's kernel time without them. The script prints that per arm and the
kernels that differ most from the reference. Stalls are calls over 3 x median + 100 us (kernels with >= 20 calls),
with the rate per second of trace span and the time they add per step.
"""
import argparse
import collections
import glob
import gzip
import json
import statistics


def load(trace_dir):
    path = sorted(glob.glob(f"{trace_dir}/*.trace.json.gz"))[0]
    events = [e for e in json.load(gzip.open(path))["traceEvents"]
              if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    durations = collections.defaultdict(list)
    for event in events:
        durations[event["name"][:80]].append(event["dur"])
    span = (max(e["ts"] + e["dur"] for e in events) - min(e["ts"] for e in events)) / 1e6
    return durations, span


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("arms", nargs="+")
    args = parser.parse_args()
    arms = [(label, load(path)) for label, path in (a.split("=", 1) for a in args.arms)]
    ref_label, (ref, _) = arms[0]
    print(f"{'arm':10s} {'kernel us/step':>14s} {'vs ' + ref_label:>9s} {'stalls/s':>8s} {'stall us/step':>13s}")
    for label, (durations, span) in arms:
        medians = {name: statistics.median(d) for name, d in durations.items()}
        total = sum(len(durations[name]) * medians[name] for name in durations) / args.steps
        ref_total = sum(len(d) * statistics.median(d) for d in ref.values()) / args.steps
        stalls = [d - medians[name] for name, ds in durations.items() if len(ds) >= 20
                  for d in ds if d > 3 * medians[name] + 100]
        print(f"{label:10s} {total:14.1f} {100 * (total / ref_total - 1):+8.2f}% {len(stalls) / span:8.0f} "
              f"{sum(stalls) / args.steps:13.1f}")
    for label, (durations, _) in arms[1:]:
        rows = []
        for name in set(durations) | set(ref):
            mine, theirs = durations.get(name, []), ref.get(name, [])
            delta = (len(mine) * (statistics.median(mine) if mine else 0)
                     - len(theirs) * (statistics.median(theirs) if theirs else 0)) / args.steps
            rows.append((delta, name, len(theirs), len(mine)))
        rows.sort(key=lambda row: -abs(row[0]))
        print(f"== {label} - {ref_label}, largest per-step changes (us):")
        for delta, name, n_ref, n_mine in rows[:8]:
            print(f"  {delta:+8.1f}  n {n_ref:5d} -> {n_mine:5d}  {name}")


if __name__ == "__main__":
    main()
