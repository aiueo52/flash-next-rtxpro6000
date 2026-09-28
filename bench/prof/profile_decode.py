#!/usr/bin/env python3
"""Profile a decode segment on the running SGLang server.
Usage: profile_decode.py <out_dir> [num_steps] [prompt_file]
Starts a streaming chat request, waits until decode is underway, then triggers
/start_profile with num_steps so the server writes a torch-profiler chrome trace.
"""
import json, os, sys, threading, time, requests
EP = "http://127.0.0.1:8001"
out_dir = os.path.abspath(sys.argv[1]); os.makedirs(out_dir, exist_ok=True)
num_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 40
prompt_file = sys.argv[3] if len(sys.argv) > 3 else os.path.expanduser("~/tools/flash-next-bench/workloads/prose-en.txt")
prompt = open(prompt_file).read()
tokens = []
t0 = time.time()
def gen():
    r = requests.post(EP + "/v1/chat/completions", json={
        "model": "flash-next", "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 600, "temperature": 0.7, "top_p": 0.9, "stream": True,
        "stream_options": {"include_usage": True}}, stream=True, timeout=600)
    for line in r.iter_lines():
        if not line or not line.startswith(b"data:"): continue
        d = line[5:].strip()
        if d == b"[DONE]": break
        j = json.loads(d)
        ch = j.get("choices") or []
        if ch:
            delta = ch[0].get("delta", {})
            if delta.get("content") or delta.get("reasoning_content"):
                tokens.append(time.time())
th = threading.Thread(target=gen); th.start()
while len(tokens) < 30 and th.is_alive(): time.sleep(0.05)
print(f"decode underway after {time.time()-t0:.2f}s, {len(tokens)} chunks; starting profile num_steps={num_steps}")
r = requests.post(EP + "/start_profile", json={"output_dir": out_dir, "num_steps": num_steps,
                  "activities": ["CPU", "GPU"], "with_stack": False, "record_shapes": False}, timeout=120)
print("start_profile:", r.status_code, r.text[:200])
th.join()
n = len(tokens)
if n > 2:
    print(f"chunks={n} decode_tps~{(n-1)/(tokens[-1]-tokens[0]):.1f}")
time.sleep(3)
print("files:", os.listdir(out_dir))
