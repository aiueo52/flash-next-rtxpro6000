"""proposal_kernels.py <label>=<glob> ...: per label, the count and median GPU time of the RS proposal and verify
kernels over every trace dir the glob matches, and 7 x (partial + finalize medians), the proposal per verify.
Example: B="runs/rs2d-abba/traces/B?-code-edit" K16="runs/rs4/traces/K16-code-edit" (RS2d spec 6).
"""
import glob
import gzip
import json
import statistics
import sys

KERNELS = ("_draft_partial_topk_kernel", "_draft_finalize_kernel", "_chain_sampling_sparse_kernel",
           "_sparse_target_probs_kernel")


def durations(pattern):
    found = {name: [] for name in KERNELS}
    dirs = sorted(glob.glob(pattern))
    for directory in dirs:
        for path in glob.glob(f"{directory}/*.trace.json.gz"):
            for event in json.load(gzip.open(path, "rt"))["traceEvents"]:
                if event.get("ph") == "X" and event.get("cat") == "kernel" and event["name"] in found:
                    found[event["name"]].append(event["dur"])
    return dirs, found


def main():
    for argument in sys.argv[1:]:
        label, pattern = argument.split("=", 1)
        dirs, found = durations(pattern)
        medians = {name: statistics.median(values) if values else 0.0 for name, values in found.items()}
        parts = "  ".join(f"{name.strip('_').removesuffix('_kernel')} n={len(found[name])} med={medians[name]:.2f}"
                          for name in KERNELS)
        proposal = 7 * (medians["_draft_partial_topk_kernel"] + medians["_draft_finalize_kernel"])
        print(f"{label}: {len(dirs)} dirs  {parts}  proposal/verify {proposal:.1f} us", flush=True)


if __name__ == "__main__":
    main()
