#!/usr/bin/env python3
"""Token-level agreement between two agree_gen.py dumps.

Usage: agree_cmp.py <A.json> <B.json> [prefix_tokens=256] [--model DIR]

Reports, per (workload, repeat) pair: the identical-token fraction over the
first N tokens and the 0-based index of the first divergence.  Also prints the
within-build self-agreement of A (repeat 0 vs repeat 1) as the noise floor.
"""
import json
import sys

MODEL = "/home/user/models/RadixArk/Qwen3.8-Flash-Next-NVFP4-mtpft3"
args, argv = [], iter(sys.argv[1:])
for a in argv:
    if a == "--model":
        MODEL = next(argv, MODEL)
    elif not a.startswith("--"):
        args.append(a)
if len(args) < 2:
    sys.exit(__doc__)
A, B = args[0], args[1]
N = int(args[2]) if len(args) > 2 else 256
for path in (A, B):
    rows = json.load(open(path))
    if any("text" not in r for r in rows):
        sys.exit(f"agree_cmp.py: {path} has no generated text (the published agree dumps keep only sha256_16 per "
                 "generation); token-level agreement needs the agree_gen.py output with text, which is not published in this repository")
import os  # noqa: E402
if not os.path.isdir(MODEL):
    sys.exit(f"agree_cmp.py: tokenizer directory not found: {MODEL} (pass --model DIR)")

from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)


def enc(t):
    return tok(t, add_special_tokens=False)["input_ids"][:N]


def cmp2(a, b):
    ta, tb = enc(a), enc(b)
    n = min(len(ta), len(tb))
    first = next((i for i in range(n) if ta[i] != tb[i]), None)
    if first is None and len(ta) != len(tb):
        first = n
    same = sum(1 for i in range(n) if ta[i] == tb[i])
    denom = max(len(ta), len(tb)) or 1
    return same / denom, first, len(ta), len(tb)


def load(path):
    """(workload, repeat) -> text; refuses a duplicate key."""
    out = {}
    for r in json.load(open(path)):
        k = (r["workload"], r["repeat"])
        if k in out:
            sys.exit(f"agree_cmp.py: {path}: duplicate record for {k}")
        out[k] = r["text"]
    return out


da, db = load(A), load(B)
if set(da) != set(db):
    sys.exit(f"agree_cmp.py: {A} and {B} cover different (workload, repeat) keys: "
             f"only in A {sorted(set(da) - set(db))}, only in B {sorted(set(db) - set(da))}")
if not da:
    sys.exit("agree_cmp.py: no records to compare")

reps = sorted({k[1] for k in da})
floor = {}
if len(reps) > 1:
    print(f"# NOISE FLOOR first ({N} tokens): {A} repeat0 vs repeat1 -- read this first")
    print(f"{'workload':<12}{'agree':>9}{'first_div':>11}")
    for wl in sorted({k[0] for k in da}):
        if (wl, reps[0]) in da and (wl, reps[1]) in da:
            f, first, la, lb = cmp2(da[(wl, reps[0])], da[(wl, reps[1])])
            floor[wl] = f
            print(f"{wl:<12}{f:>9.3f}{('-' if first is None else first):>11}")
    bad = [w for w, f in floor.items() if f < 0.95]
    if bad:
        print(f"  !! self-agreement below 0.95 on {bad}: a build-to-build token"
              f" agreement number is NOT interpretable on those workloads.")
    print()

print(f"# agreement over the first {N} tokens: {A}  vs  {B}")
print(f"{'workload':<12}{'rep':>4}{'agree':>9}{'first_div':>11}{'lenA':>7}{'lenB':>7}")
tot = []
for k in sorted(da):
    f, first, la, lb = cmp2(da[k], db[k])
    tot.append(f)
    print(f"{k[0]:<12}{k[1]:>4}{f:>9.3f}"
          f"{('-' if first is None else first):>11}{la:>7}{lb:>7}")
print(f"{'MEAN':<12}{'':>4}{sum(tot)/len(tot):>9.3f}")

ok = [w for w, f in floor.items() if f >= 0.95]
if ok:
    sel = [t for k, t in zip(sorted(da), tot) if k[0] in ok]
    print(f"\n# restricted to workloads whose noise floor is >= 0.95 {ok}: "
          f"mean agreement {sum(sel)/len(sel):.3f}")
