#!/usr/bin/env python3
"""Greedy generation dump for the xa1 A/B: prof/agree_gen.py with a --port (it is fixed to 8001).

Same records, so prof/agree_cmp.py compares two dumps: temperature 0, thinking off, one request at a
time, /flush_cache before every request; 4 workloads x repeats x max_tokens.

  python bench/xa1/greedy_dump.py --port 8027 runs/xa1/A1-greedy.json [--max-tokens 320] [--repeats 2]
  python prof/agree_cmp.py runs/xa1/A1-greedy.json runs/xa1/B1-greedy.json 256
"""
import argparse
import json
import time
import urllib.request

WL = ["code-edit", "prose-en", "agent-loop", "prose-ja"]
BASE = "/home/user/tools/flash-next-bench/workloads"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()
    ep = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    flush = f"http://127.0.0.1:{args.port}/flush_cache"
    recs = []
    for wl in WL:
        prompt = open(f"{BASE}/{wl}.txt", encoding="utf-8").read()
        for rep in range(args.repeats):
            body = json.dumps({
                "model": "flash-next",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "top_p": 1,
                "chat_template_kwargs": {"enable_thinking": False},
            }).encode()
            try:
                urllib.request.urlopen(urllib.request.Request(
                    flush, b"", {"Content-Type": "application/json"}), timeout=60).read()
                time.sleep(1.0)
            except Exception as exc:
                print(f"  flush_cache failed: {exc!r}")
            t0 = time.time()
            r = urllib.request.urlopen(urllib.request.Request(
                ep, body, {"Content-Type": "application/json"}), timeout=900)
            d = json.load(r)
            msg = d["choices"][0]["message"]
            recs.append({
                "workload": wl, "repeat": rep,
                "text": msg.get("content") or "",
                "completion_tokens": d["usage"]["completion_tokens"],
                "secs": round(time.time() - t0, 1),
            })
            print(f"{wl}/{rep}: {recs[-1]['completion_tokens']} tok in {recs[-1]['secs']}s",
                  flush=True)
    json.dump(recs, open(args.out, "w"), ensure_ascii=False, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
