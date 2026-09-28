#!/usr/bin/env python3
"""Native SSE client: exact server token counters, complete request timing, no warmup by domain."""
import hashlib,json,sys,time
from pathlib import Path
import requests
B=Path(__file__).resolve().parents[1]
stream,out,arm=sys.argv[1:];out=Path(out)
rows=[json.loads(l) for l in open(stream)]
def generate(row, rid):
 sampling={'max_new_tokens':row['budget'],'temperature':0,'ignore_eos':False}
 if row.get('designation') is not None:sampling['custom_params']={'adaptive_task':row['designation']}
 body={'text':row['prompt'],'rid':rid,'sampling_params':sampling,'stream':True}
 chunks=[];last={};start=time.perf_counter();wall=time.time()
 with requests.post('http://127.0.0.1:8001/generate',json=body,stream=True,timeout=600) as response:
  response.raise_for_status()
  for line in response.iter_lines(chunk_size=None):
   if not line.startswith(b'data: '):continue
   data=line[6:]
   if data==b'[DONE]':break
   last=json.loads(data)
   if 'error' in last:raise RuntimeError(last)
   meta=last.get('meta_info',{});n=meta.get('completion_tokens',0)
   if n and (not chunks or n>chunks[-1]['tokens']):
    chunks.append({'elapsed':time.perf_counter()-start,'tokens':n,'verifies':meta.get('spec_verify_ct')})
 end=time.perf_counter();meta=last.get('meta_info',{});assert len(chunks)>1,(rid,last)
 first,final=chunks[0],chunks[-1];decode=final['elapsed']-first['elapsed'];tokens=final['tokens']-first['tokens']
 text=last.get('text','')
 return {k:v for k,v in row.items() if k!='prompt'}|dict(rid=rid,arm=arm,wall_start=wall,wall_end=time.time(),total_s=end-start,ttft_s=first['elapsed'],decode_s=decode,decode_tokens=tokens,completion_tokens=final['tokens'],first_chunk_tokens=first['tokens'],tps=tokens/decode,early_s={str(n):next((c['elapsed']-first['elapsed'] for c in chunks if c['tokens']-first['tokens']>=n),None) for n in [16,32,64,128]},chunks=chunks,meta=meta,text=text,text_sha256=hashlib.sha256(text.encode()).hexdigest())
warm={'prompt':'<|im_start|>user\nDescribe the purpose of a notebook in a few sentences.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n','budget':128,'id':'warm','designation':'unknown'}
(out/'warmup.json').write_text(json.dumps(generate(warm,arm+'-warm'))+'\n')
with (out/'requests.jsonl').open('x',buffering=1) as f:
 for i,row in enumerate(rows):
  result=generate(row,arm+'-'+row['id']);f.write(json.dumps(result,ensure_ascii=False)+'\n')
  print(f"{i+1}/{len(rows)} {row['id']} tokens={result['completion_tokens']} decode={result['decode_s']:.3f}s tps={result['tps']:.2f}",flush=True)
