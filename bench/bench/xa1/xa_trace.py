"""QSA attention kernels in one profiled server arm (bench/xa1/arm_xa.sh): count and median duration
per role. Production: valid counts + _compact_kv + XQA kernel_mha per attention call. With
SGLANG_OPT_TRITON_DECODE_ATTN=1 all three disappear and _qsa_decode_attn_kernel takes their place
(one launch per call: verify layers, draft extend and every draft decode step).

  python bench/xa1/xa_trace.py runs/xa1/traces/B1-code-edit
"""
import collections
import glob
import gzip
import json
import os
import statistics
import sys

ROLES = {"_qsa_decode_attn_kernel": "triton_decode", "kernel_mha": "xqa_mha",
         "_compact_kv": "compact_kv", "_fa2_valid_counts": "valid_counts"}


def main(trace_dir):
    durs = collections.defaultdict(list)
    grids = collections.defaultdict(collections.Counter)
    files = glob.glob(os.path.join(trace_dir, "*.trace.json.gz"))
    for path in files:
        t = json.load(gzip.open(path))
        for e in t["traceEvents"] if isinstance(t, dict) else t:
            if e.get("cat") != "kernel":
                continue
            for key, role in ROLES.items():
                if key in e["name"]:
                    durs[role].append(e["dur"])
                    grids[role][str(e.get("args", {}).get("grid"))] += 1
    if not files:
        print(f"no traces in {trace_dir}")
        return
    for role in ROLES.values():
        v = durs.get(role, [])
        med = f"{statistics.median(v):6.2f} us" if v else "     -"
        top = ", ".join(f"{g} x{n}" for g, n in grids[role].most_common(4))
        print(f"{role:14s} count {len(v):6d}  median {med}  grids {top}")


if __name__ == "__main__":
    main(sys.argv[1])
