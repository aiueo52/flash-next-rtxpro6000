"""Read-only small progress snapshot for the HC1 coordinator."""
import json
from pathlib import Path
B=Path('/home/user/tools/flash-next-bench');R=B/'specs/hc1';Q=B/'bench/hc1/quality/runs'
arms=('prod','hc256','hc160','hc128','gate-const')
if not R.is_dir():
    raise SystemExit(f"status.py: no HC1 run at {R} (the HC1 run directories are not published in this repository)")
a=next((a for a in arms if not (R/a/'complete.json').exists()),None)
s={'arm':a,'complete_arms':sum((R/a/'complete.json').exists() for a in arms)}
if a:
 p=R/a;s['stage']='lock wait' if not p.exists() else 'startup'
 if (p/'server-info.json').exists():s['stage']='acceptance'
 if (p/'requests.jsonl').exists():s['bn1_requests']=len((p/'requests.jsonl').read_text().splitlines())
 if (p/'needle.log').exists():s['needle']=sum('PASS=True' in x for x in (p/'needle.log').read_text().splitlines())
 if (p/'quality-smoke.log').exists():s['stage']='quality smoke'
 if (p/'quality.log').exists():s['stage']='quality'
 if (Q/a/'summary.json').exists():s['quality']=json.loads((Q/a/'summary.json').read_text())['bench']
 if (p/'server.log').exists():s['chat_http_responses']=(p/'server.log').read_text().count('"POST /v1/chat/completions HTTP/1.1" 200 OK')
 if (p/'memory.jsonl').exists():
  lines=(p/'memory.jsonl').read_text().splitlines()
  if lines:s['free_mib']=json.loads(lines[-1])['free_mib']
 if (p/'invalid.json').exists():s['error']=json.loads((p/'invalid.json').read_text())
print(json.dumps(s))
