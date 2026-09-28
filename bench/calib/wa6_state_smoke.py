#!/usr/bin/env python3
"""Untimed two-request overlap smoke; exercises retained request state through BS changes."""
import concurrent.futures,json,sys,threading,time
from pathlib import Path
import requests
B=Path(__file__).resolve().parents[1];out=Path(sys.argv[1]);ready=threading.Event()
rs=[json.loads(l) for l in (B/'specs/wa6/prep-stream.jsonl').open()]
a=next(r for r in rs if r['id']=='prep-b1-ja-2048');b=next(r for r in rs if r['id']=='prep-b0-code-256')
def run(row,tag,budget,signal=False):
 chunks=[];last=None
 body={'text':row['prompt'],'rid':out.name+'-state-'+tag,'sampling_params':{'max_new_tokens':budget,'temperature':0,'custom_params':{'adaptive_task':row['domain']}},'stream':True}
 start=time.time()
 with requests.post('http://127.0.0.1:8001/generate',json=body,stream=True,timeout=180) as r:
  r.raise_for_status()
  for line in r.iter_lines(chunk_size=None):
   if not line.startswith(b'data: ') or line[6:]==b'[DONE]':continue
   last=json.loads(line[6:]);chunks.append((time.time(),last.get('meta_info',{}).get('completion_tokens',0)))
   if signal and len(chunks)>=8:ready.set()
 assert last and 'error' not in last,last
 return dict(rid=body['rid'],start=start,end=time.time(),chunks=chunks,meta=last['meta_info'],text=last.get('text',''))
with concurrent.futures.ThreadPoolExecutor(2) as pool:
 fa=pool.submit(run,a,'long-ja',1024,True)
 assert ready.wait(120),'first request did not reach decode'
 fb=pool.submit(run,b,'short-code',256)
 results=[fa.result(),fb.result()]
assert results[1]['start'] < results[0]['end']
assert all(0<r['meta']['completion_tokens']<=limit for r,limit in zip(results,[1024,256]))
(out/'state-smoke.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
print(json.dumps([{'rid':r['rid'],'tokens':r['meta']['completion_tokens'],'retractions':r['meta'].get('num_retractions'),'start':r['start'],'end':r['end']} for r in results],indent=2))

response=requests.post('http://127.0.0.1:8001/v1/chat/completions',json={
 'model':'flash-next','messages':[{'role':'user','content':'Reply with WA6_OK only.'}],
 'max_tokens':24,'temperature':0,'custom_params':{'adaptive_task':'en'},
 'chat_template_kwargs':{'enable_thinking':False}},timeout=120)
response.raise_for_status();payload=response.json()
assert payload.get('choices'),payload
(out/'openai-designation-smoke.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
print('OpenAI-compatible custom_params request accepted:',payload['choices'][0]['message']['content'])
