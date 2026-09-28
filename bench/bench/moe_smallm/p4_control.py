#!/usr/bin/env python3
"""Send idle-only reset/dump command to the P4 effective-routing recorder."""
import argparse
import json
from pathlib import Path
import time
import uuid
ap=argparse.ArgumentParser()
ap.add_argument('prefix',type=Path)
ap.add_argument('action',choices=['reset','dump'])
ap.add_argument('--path')
a=ap.parse_args()
nonce=uuid.uuid4().hex
command=Path(str(a.prefix)+'.command.json')
tmp=Path(str(command)+'.tmp')
tmp.write_text(json.dumps(dict(action=a.action,path=a.path,nonce=nonce)))
tmp.replace(command)
ack=Path(str(a.prefix)+'.ack.json')
for _ in range(240):
    if ack.exists():
        result=json.loads(ack.read_text())
        if result.get('nonce')==nonce:
            print(json.dumps(result),flush=True)
            if not result['ok']: raise SystemExit(1)
            break
    time.sleep(.5)
else:
    raise SystemExit('P4 recorder acknowledgement timeout')
