"""Greedy outputs for an exactness check: each workload prompt R times, temperature 0, full text kept.

  python bench/stack/exact_client.py --port 8027 --out runs/stack8/exact/X1.jsonl --max-tokens 2048 --repeat 2 \
      workloads/code-edit.txt workloads/prose-en.txt
  python bench/stack/exact_client.py --compare runs/stack8/exact/X1.jsonl runs/stack8/exact/Y1.jsonl ...
"""
import argparse
import hashlib
import json
import os

import requests


def run(port, out, max_tokens, repeat, workloads):
    with open(out, "w") as f:
        for r in range(repeat):
            for w in workloads:
                body = {"model": "flash-next", "messages": [{"role": "user", "content": open(w).read()}],
                        "temperature": 0.0, "max_tokens": max_tokens}
                resp = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=body, timeout=900).json()
                msg = resp["choices"][0]["message"]
                text = (msg.get("reasoning_content") or "") + "\x00" + (msg.get("content") or "")
                row = {"workload": os.path.basename(w), "repeat": r, "tokens": resp["usage"]["completion_tokens"],
                       "sha": hashlib.sha256(text.encode()).hexdigest()[:16], "text": text}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"{row['workload']:16s} r{r} tokens {row['tokens']:5d} sha {row['sha']}", flush=True)


def first_diff(a, b):
    n = min(len(a), len(b))
    return next((i for i in range(n) if a[i] != b[i]), n if len(a) != len(b) else -1)


def compare(paths):
    runs = {p: {(x["workload"], x["repeat"]): x for x in map(json.loads, open(p))} for p in paths}
    ref_path = paths[0]
    for p in paths[1:]:
        same = 0
        for key, x in runs[ref_path].items():
            y = runs[p].get(key)
            if y is None:
                continue
            i = first_diff(x["text"], y["text"])
            same += i < 0
            note = "same" if i < 0 else f"differs at char {i} (tokens {x['tokens']} / {y['tokens']})"
            print(f"{os.path.basename(ref_path)} vs {os.path.basename(p)} {key[0]:16s} r{key[1]}: {note}")
        print(f"{os.path.basename(ref_path)} vs {os.path.basename(p)}: {same}/{len(runs[ref_path])} identical")
    # within one server: repeat 1 against repeat 0
    for p, rows in runs.items():
        pairs = [(rows[k], rows.get((k[0], 1))) for k in rows if k[1] == 0 and (k[0], 1) in rows]
        same = sum(first_diff(a["text"], b["text"]) < 0 for a, b in pairs)
        print(f"{os.path.basename(p)} within the server, repeat 1 vs 0: {same}/{len(pairs)} identical")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8027)
    ap.add_argument("--out")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--compare", nargs="+")
    ap.add_argument("workloads", nargs="*")
    a = ap.parse_args()
    if a.compare:
        compare(a.compare)
    else:
        run(port=a.port, out=a.out, max_tokens=a.max_tokens, repeat=a.repeat, workloads=a.workloads)


if __name__ == "__main__":
    main()
