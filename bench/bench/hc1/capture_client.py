import json
import os
from pathlib import Path
import requests
B=Path('/home/user/tools/flash-next-bench');root=B/'specs/hc1/capture-data'
def active(value):
    p=root/'active.tmp';p.write_text(json.dumps(value));os.replace(p,root/'active.json')
active({})
# Exclude lazy startup/autotune forwards from the recorded BN1 samples.
warm=requests.post('http://127.0.0.1:8001/v1/chat/completions',json=dict(model='flash-next',messages=[dict(role='user',content='Write a detailed practical guide to maintaining a small community bicycle workshop. Discuss tools, storage, inspection and staff handover in connected paragraphs.')],max_tokens=1024,temperature=0,chat_template_kwargs=dict(enable_thinking=False)),timeout=900)
warm.raise_for_status()
print('Unrecorded warmup complete',warm.json().get('usage'),flush=True)
try:
    for domain in ('code-edit','prose-en','prose-ja','agent-loop'):
        folder=B/'workloads/sets'/(domain+'-v1')
        for row in json.loads((folder/'manifest.json').read_text())['prompts']:
            active(dict(id=row['id'],domain=domain))
            r=requests.post('http://127.0.0.1:8001/v1/chat/completions',json=dict(model='flash-next',messages=[dict(role='user',content=(folder/row['file']).read_text())],max_tokens=512,temperature=0,top_p=1,top_k=1,chat_template_kwargs=dict(enable_thinking=False)),timeout=900)
            r.raise_for_status();result=r.json()
            (root/(row['id']+'.response.json')).write_text(json.dumps(result,ensure_ascii=False))
            print(row['id'],result.get('usage'),flush=True)
finally:active({})
