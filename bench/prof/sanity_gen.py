#!/usr/bin/env python3
"""Degeneration check: long-ish greedy generation, report acceptance (completion/spec_verify) and repetition.
Usage: sanity_gen.py <label> [max_tokens]"""
import json, sys, time, urllib.request, collections, re
EP="http://127.0.0.1:8001"
label=sys.argv[1] if len(sys.argv)>1 else ""; mt=int(sys.argv[2]) if len(sys.argv)>2 else 600
ptok=int(sys.argv[3]) if len(sys.argv)>3 else 0
def metrics():
    raw=urllib.request.urlopen(EP+"/metrics",timeout=30).read().decode()
    m=re.search(r'sglang:spec_verify_calls_total\{[^}]*\} ([0-9.e+]+)',raw); return float(m.group(1)) if m else None
filler=("The harbor town of Thornwick kept its records in a ledger bound in oilcloth; every tide, every lamp trimmed, every ship that passed was written down by hand. " * max(0, ptok//32))
prompt=filler+"\n\nWrite a detailed 300-word story about a lighthouse keeper who discovers a strange signal. Use varied vocabulary."
body={"model":"flash-next","messages":[{"role":"user","content":prompt}],"max_tokens":mt,"temperature":0,
      "chat_template_kwargs":{"enable_thinking":True,"reasoning_effort":"medium"}}
b0=metrics(); t0=time.time()
req=urllib.request.Request(EP+"/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
r=json.load(urllib.request.urlopen(req,timeout=600)); dt=time.time()-t0
b1=metrics(); out=(r["choices"][0]["message"].get("content") or "")+(r["choices"][0]["message"].get("reasoning_content") or "")
ct=r["usage"]["completion_tokens"]; sv=(b1-b0) if (b0 is not None and b1 is not None) else None
acc=ct/sv if sv else None
run=max((len(m.group(0)) for m in re.finditer(r'(.)\1+',out)),default=1)
words=out.split(); uniq=len(set(words))/max(1,len(words))
ok = (acc is None or acc<12.0) and run<20 and uniq>0.2
print(f"[{label}] completion={ct} time={dt:.1f}s tps={ct/dt:.0f} accept={acc if acc is None else round(acc,2)} maxrun={run} uniq_words={uniq:.2f} PASS={ok} out={out[:70]!r}")
