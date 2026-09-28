#!/usr/bin/env python3
"""Deterministic parent-disjoint WA6 streams from v5's pre-existing holdout.
No output tokens, throughput or oracle widths are used as predictor inputs.
"""
import collections, hashlib, importlib.util, json, random, sys
from pathlib import Path
from transformers import AutoTokenizer
B=Path(__file__).resolve().parents[1]; O=B/'specs/wa6'
P=Path('/home/user/mtp-gen-v5/gen.jsonl')
MODEL='/home/user/models/RadixArk/Qwen3.8-Flash-Next-NVFP4-mtpft5'
spec=importlib.util.spec_from_file_location('prior','/home/user/tools/sglang-wa6/python/sglang/srt/speculative/adaptive_request_prior.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def holdout(key):
 return int(hashlib.blake2b(f'20260903:{key}'.encode(),digest_size=8).hexdigest(),16)%10000<1000
rows=[json.loads(l) for l in P.open()]; pools=collections.defaultdict(list);seen=set()
for r in rows:
 domain={'v5-code':'code','v5-agent':'agent','v5-prose-en':'en','v5-prose-ja':'ja'}.get(r['bucket'])
 parent=r['split_key']
 if not domain or not holdout(parent) or parent in seen or r['prompt_tokens']>4096:continue
 seen.add(parent);pools[domain].append(r)
tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
rng=random.Random(2026090806)
for d in pools:rng.shuffle(pools[d])
# Preparation and evaluation blocks of unique parents (sizes are the parameters below; the prompts are private).
# One block is the unit of paired analysis; all arms replay exactly its order.
orders=[['code','en','ja','agent'],['agent','ja','en','code'],['en','code','agent','ja'],['ja','agent','code','en']]
allparents=set();records=[]
for split,nblocks in [('prep',2),('eval',8)]:
 stream=[]
 for block in range(nblocks):
  for bi,budget in enumerate([64,128,256,2048]):
   for domain in orders[(block+bi)%4]:
    r=pools[domain].pop();parent=r['split_key'];assert parent not in allparents;allparents.add(parent)
    prompt=tok.decode(r['prompt_ids'],skip_special_tokens=False)
    row=dict(id=f'{split}-b{block}-{domain}-{budget}',split=split,block=block,domain=domain,budget=budget,parent=parent,pid=r['pid'],family=r['family'],prompt=prompt,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),prompt_tokens=len(r['prompt_ids']),predictor=m.predict_task(prompt),designation=domain if block%2==0 else None)
    # No use of reference generated output/acceptance in sampling or prompts.
    stream.append(row)
 with (O/f'{split}-stream.jsonl').open('x') as f:
  for r in stream:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 records+=stream
conf=collections.Counter((r['domain'],r['predictor']) for r in records if r['split']=='eval')
manifest=dict(source=str(P),source_sha256=hashlib.sha256(P.read_bytes()).hexdigest(),seed=2026090806,heldout_rule='blake2b(20260903:split_key), 8 bytes, modulo10000 <1000 (v5 training holdout)',unique_parents=len(allparents),counts=dict(collections.Counter(r['split'] for r in records)),predictor_version=m.PREDICTOR_VERSION,predictor_confusion={f'{a}->{b}':n for (a,b),n in conf.items()},budgets=[64,128,256,2048],weights='equal request count by domain and budget; primary pooled delivered tokens/time',eos='honor natural EOS; budgets are maxima, actual tokens always recorded',request_designation='even blocks explicit caller domain; odd blocks frozen predictor only; no oracle selection')
(O/'stream-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
