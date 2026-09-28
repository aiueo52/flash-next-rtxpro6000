#!/usr/bin/env python3
"""Greedy generation on the fnbench workloads, for build-to-build agreement.

Usage: agree_gen.py <outfile.json> [max_tokens=320] [repeats=2]

Temperature 0, thinking off, one request at a time (batch composition changes
which experts are singletons, so a concurrent request would make the pruned
build's own output depend on what else is in flight), and ``/flush_cache``
before every request so repeat 1 does not start from repeat 0's radix-cached
prefix (which chunks the prefill differently and changes its numerics).

Repeats inside one build are the noise floor and must be read first: speculative
decoding plus non-deterministic reductions can make even a greedy run differ
from itself, and on open-ended prose this model's near-ties cascade from the
first divergent token.  If the floor is at the floor, a build-to-build agreement
number means nothing.
"""
import json
import sys
import time
import urllib.request

EP = "http://127.0.0.1:8001/v1/chat/completions"
FLUSH = "http://127.0.0.1:8001/flush_cache"
WL = ["code-edit", "prose-en", "agent-loop", "prose-ja"]
BASE = "/home/user/tools/flash-next-bench/workloads"

out_path = sys.argv[1]
max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 320
repeats = int(sys.argv[3]) if len(sys.argv) > 3 else 2

recs = []
for wl in WL:
    prompt = open(f"{BASE}/{wl}.txt", encoding="utf-8").read()
    for rep in range(repeats):
        body = json.dumps({
            "model": "flash-next",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "top_p": 1,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()
        try:
            urllib.request.urlopen(urllib.request.Request(
                FLUSH, b"", {"Content-Type": "application/json"}), timeout=60).read()
            time.sleep(1.0)
        except Exception as exc:
            print(f"  flush_cache failed: {exc!r}")
        t0 = time.time()
        r = urllib.request.urlopen(urllib.request.Request(
            EP, body, {"Content-Type": "application/json"}), timeout=900)
        d = json.load(r)
        msg = d["choices"][0]["message"]
        recs.append({
            "workload": wl, "repeat": rep,
            "text": msg.get("content") or "",
            "completion_tokens": d["usage"]["completion_tokens"],
            "secs": round(time.time() - t0, 1),
        })
        print(f"{wl}/{rep}: {recs[-1]['completion_tokens']} tok "
              f"in {recs[-1]['secs']}s", flush=True)

json.dump(recs, open(out_path, "w"), ensure_ascii=False, indent=1)
print("saved", out_path)
