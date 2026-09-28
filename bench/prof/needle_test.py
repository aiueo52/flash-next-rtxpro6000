#!/usr/bin/env python3
"""Needle-in-haystack check with thinking disabled. Usage: needle_test.py [approx_tokens] [depth 0-1] [label]"""
if __name__ != "__main__":  # collected by pytest: this is a GPU experiment script, not a unit test
    import pytest
    pytest.skip("GPU experiment script: run it directly (see the docstring)", allow_module_level=True)
import sys, json, time, random, requests
EP="http://127.0.0.1:8001/v1/chat/completions"
target_tokens=int(sys.argv[1]) if len(sys.argv)>1 else 18500
depth=float(sys.argv[2]) if len(sys.argv)>2 else 0.4
label=sys.argv[3] if len(sys.argv)>3 else ""
random.seed(1234)
subjects=["the harbor authority","a regional archive","the observatory staff","the night shift","a cartography guild","the tram depot","the river commission","an apiary cooperative"]
verbs=["recorded","reviewed","catalogued","postponed","inspected","reconciled","transcribed","audited"]
objects=["the tide tables","a set of brass gauges","the quarterly ledgers","several lantern housings","the drainage survey","a batch of glass plates","the timber manifests","the signal logs"]
def para(i):
    s=[]
    for _ in range(6):
        s.append(f"On day {random.randint(1,365)} {random.choice(subjects)} {random.choice(verbs)} {random.choice(objects)}, noting that entry {random.randint(100,9999)} remained consistent with the earlier notes.")
    return " ".join(s)
# ~4 chars/token for English -> target_tokens*4 chars
paras=[]; total=0
while total < target_tokens*3.9:
    p=para(len(paras)); paras.append(p); total+=len(p)
needle="The maintenance passphrase for the east cabinet is AURORA-CEDAR-7319 and must be quoted exactly when asked."
idx=int(len(paras)*depth); paras.insert(idx, needle)
context="\n\n".join(paras)
q="\n\nQuestion: What is the maintenance passphrase for the east cabinet? Reply with the passphrase only."
t0=time.time()
r=requests.post(EP,json={"model":"flash-next","messages":[{"role":"user","content":context+q}],"max_tokens":48,"temperature":0,
    "chat_template_kwargs":{"enable_thinking":False}},timeout=600)
j=r.json(); out=j["choices"][0]["message"]["content"]; u=j.get("usage",{})
ok="AURORA-CEDAR-7319" in (out or "")
print(f"[{label}] prompt_tokens={u.get('prompt_tokens')} completion={u.get('completion_tokens')} time={time.time()-t0:.1f}s PASS={ok} out={out!r:.80}")
