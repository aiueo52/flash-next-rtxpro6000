#!/usr/bin/env python3
"""Greedy vs LM-Studio-style sampling on a running SGLang server: per-step wall and acceptance,
optionally with a torch-profiler trace of a decode window.

fnbench cannot send top_k / min_p, so this is a minimal OpenAI-compatible streaming client that
passes them as extra body fields (SGLang's ChatCompletionRequest accepts both).

Usage:
    fc_sampling_probe.py --port 8011 --mode sampling --workload workloads/code-edit.txt \
        [--n 3] [--max-tokens 600] [--profile-dir DIR --profile-steps 20]

Per request it prints decode chunks, completion tokens, tokens/chunk (= tokens per verify step,
bonus included, as in WIDTH_SWEEP_0907) and ms/chunk over the steady part of the stream (the first
SKIP chunks are dropped). One SSE chunk is emitted per verify step on this server.
"""
import argparse, json, os, statistics, threading, time
import requests

SAMPLING = dict(temperature=0.8, top_p=0.95, top_k=40, min_p=0.05)   # LM Studio defaults
GREEDY = dict(temperature=0.0)
SKIP = 8

def run_one(ep, prompt, params, max_tokens, on_chunk=None):
    body = {"model": "flash-next", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True}}
    body.update(params)
    times, usage = [], None
    with requests.post(ep + "/v1/chat/completions", json=body, stream=True, timeout=900) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data:"):
                continue
            d = line[5:].strip()
            if d == b"[DONE]":
                break
            j = json.loads(d)
            if j.get("usage"):
                usage = j["usage"]
            ch = j.get("choices") or []
            if ch:
                delta = ch[0].get("delta", {})
                if delta.get("content") or delta.get("reasoning_content"):
                    times.append(time.time())
                    if on_chunk:
                        on_chunk(len(times))
    return times, usage

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--mode", choices=["greedy", "sampling"], required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--profile-dir", default=None)
    ap.add_argument("--profile-steps", type=int, default=20)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    ep = f"http://127.0.0.1:{a.port}"
    prompt = open(a.workload).read()
    params = SAMPLING if a.mode == "sampling" else GREEDY
    rows = []
    for i in range(a.n):
        trig = {}
        def on_chunk(n, trig=trig):
            # profile once, on the first request, after decode is underway
            if a.profile_dir and i == 0 and n == 30 and not trig:
                trig["t"] = True
                def fire():
                    os.makedirs(a.profile_dir, exist_ok=True)
                    r = requests.post(ep + "/start_profile", json={
                        "output_dir": os.path.abspath(a.profile_dir), "num_steps": a.profile_steps,
                        "activities": ["CPU", "GPU"], "with_stack": False, "record_shapes": False},
                        timeout=120)
                    print(f"  start_profile: {r.status_code}", flush=True)
                threading.Thread(target=fire, daemon=True).start()
        times, usage = run_one(ep, prompt, params, a.max_tokens, on_chunk)
        n = len(times)
        ct = (usage or {}).get("completion_tokens", 0)
        st = times[SKIP:]
        ms = (st[-1] - st[0]) / (len(st) - 1) * 1e3 if len(st) > 2 else float("nan")
        row = dict(mode=a.mode, workload=os.path.basename(a.workload), i=i, chunks=n,
                   completion_tokens=ct, tok_per_step=(ct / n if n else 0), ms_per_step=ms,
                   profiled=bool(a.profile_dir and i == 0))
        rows.append(row)
        print(f"  {a.mode:8s} {row['workload']:14s} #{i} chunks={n:4d} tokens={ct:4d} "
              f"tok/step={row['tok_per_step']:.2f} ms/step={ms:6.2f}{' (profiled)' if row['profiled'] else ''}",
              flush=True)
    clean = [r["ms_per_step"] for r in rows if not r["profiled"]]
    if clean:
        print(f"  => {a.mode} {os.path.basename(a.workload)}: median ms/step (unprofiled) "
              f"{statistics.median(clean):.2f}  tok/step {statistics.mean(r['tok_per_step'] for r in rows):.2f}")
    if a.json_out:
        with open(a.json_out, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    main()
