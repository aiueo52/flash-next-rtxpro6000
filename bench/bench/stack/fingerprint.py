"""Greedy fingerprint across arms: per prompt, (completion tokens, verify calls) from fnbench greedy rows.

Runs of one config that are deterministic give the same pair in every server start. A change that keeps every
token and every draft bit-exact (DT1's SGLANG_OPT_DRAFT_TAIL) must then give the base config's pair too.

  python bench/stack/fingerprint.py --base runs/stack8/B{1,2,3,4}-greedy.jsonl --test runs/stack8/C1-greedy.jsonl
  python bench/stack/fingerprint.py --base runs/stack8/A{1,2,3,4}-greedy.jsonl   # determinism of the base only
"""
import argparse
import collections
import json
import os


def pairs(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r["sampling_mode"] != "greedy":
            continue
        acc = r["server"]["acceptance"]
        out[r["prompt_id"]] = (r["client"]["usage"]["completion_tokens"], int(acc["verify_calls"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="+", required=True)
    ap.add_argument("--test", nargs="*", default=[])
    a = ap.parse_args()
    base = {p: pairs(p) for p in a.base}
    test = {os.path.basename(p): pairs(p) for p in a.test}
    prompts = sorted(set().union(*[set(v) for v in base.values()]))
    n_consistent = n_match = n_checked = 0
    for pid in prompts:
        seen = collections.Counter(v[pid] for v in base.values() if pid in v)
        consistent = len(seen) == 1
        n_consistent += consistent
        cells = " ".join(f"{t}:{v.get(pid)}" for t, v in test.items())
        verdict = ""
        if consistent and test:
            ref = next(iter(seen))
            ok = all(v.get(pid) == ref for v in test.values())
            n_checked += 1
            n_match += ok
            verdict = "MATCH" if ok else "DIFF"
        print(f"{pid:22s} base {dict(seen)} {cells} {verdict}")
    print(f"base consistent on {n_consistent}/{len(prompts)} prompts", end="")
    print(f"; test matches the base on {n_match}/{n_checked} of them" if test else "")


if __name__ == "__main__":
    main()
