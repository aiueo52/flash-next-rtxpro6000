#!/usr/bin/env python3
"""Read-only source/process audit and exact-window telemetry summary."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import statistics as st
import subprocess

B = Path(__file__).resolve().parents[1]
root = B / 'specs/wa5'
if not (root/'source-before.json').is_file() or not (root/'harness-before.json').is_file():
    raise SystemExit(f'wa5_audit.py: needs {root}/source-before.json and harness-before.json from a WA5 run; '
                     'WA5 run directories are not published in this repository')
before = json.loads((root/'source-before.json').read_text())
assert before, 'source-before.json lists no source trees: nothing to audit'
after = {}
for name, old in before.items():
    r = B.parent/name
    git = lambda *a:subprocess.check_output(['git','-C',str(r),*a], text=True)
    after[name] = dict(head=git('rev-parse','HEAD').strip(),branch=git('branch','--show-current').strip(),status=git('status','--short'),diff_cached_sha256=hashlib.sha256(git('diff','--cached').encode()).hexdigest(),hashes={f:hashlib.sha256((r/f).read_bytes()).hexdigest() for f in old['hashes']})
manifest = json.loads((root/'harness-before.json').read_text())
assert manifest, 'harness-before.json lists no files: nothing to audit'
harness = {f:hashlib.sha256((B/f).read_bytes()).hexdigest()==v for f,v in manifest.items()}
ps = subprocess.check_output(['ps','-eo','pid,pgid,comm'],text=True)
(root/'processes-final.txt').write_text(ps)
processes = [line.split() for line in ps.splitlines()[1:]]
arm_audit = {}
for d in sorted(root.glob('wa5-*')):
    assert (d/'complete').exists(), d
    pgid = (d/'server-pgid.txt').read_text().strip()
    remnants = [p for p in processes if p[1]==pgid]
    rows = [json.loads(l) for l in (d/'runs.jsonl').read_text().splitlines()]
    events = [json.loads(l) for l in (d/'events.jsonl').read_text().splitlines()]
    mem = [json.loads(l) for l in (d/'memory.jsonl').read_text().splitlines()]
    windows = {}
    for w in ['code-edit','prose-en','prose-ja','agent-loop']:
        es = [e for e in events if e['workload']==w and e['repeat']>0]
        samples = [r for r in mem if any(e['start']<=dt.datetime.fromisoformat(r['t']).timestamp()<=e['end'] for e in es)]
        windows[w] = dict(samples=len(samples), min_free_mib=min(r['free_mib'] for r in samples),median_sm_mhz=st.median(float(r['gpu'][4]) for r in samples),sm_range_mhz=[min(float(r['gpu'][4]) for r in samples),max(float(r['gpu'][4]) for r in samples)],temperature_range_c=[min(float(r['gpu'][2]) for r in samples),max(float(r['gpu'][2]) for r in samples)],median_power_w=st.median(float(r['gpu'][3]) for r in samples),power_cap_samples=sum(r['gpu'][6]=='Active' for r in samples),thermal_cap_samples=sum(r['gpu'][7]=='Active' for r in samples))
    log = (d/'server.log').read_text()
    arm_audit[d.name] = dict(complete=True, measured=len(rows),warmups=sum(e['repeat']==0 for e in events),needle_pass='PASS=True' in (d/'needle.log').read_text(),other_gpu_app_samples=sum(bool(r.get('other_gpu_app')) for r in mem),remaining_group_processes=remnants,measurement_windows=windows,first_sample=mem[0]['t'],last_sample=mem[-1]['t'],ready_config_lines=[l for l in log.splitlines() if ('adaptive' in l.lower() and any(k in l.lower() for k in ['candidate','config','policy','autotune']))][:30])
result = dict(source_unchanged=after==before,source_after=after,harness_unchanged=all(harness.values()),harness_checks=harness,arms=arm_audit)
(root/'completeness.json').write_text(json.dumps(result,indent=2)+'\n')
assert after==before
assert all(harness.values())
assert len(arm_audit)==4
assert all(a['measured']==12 and a['warmups']==4 and a['needle_pass'] and not a['remaining_group_processes'] for a in arm_audit.values())
print(json.dumps({k:v for k,v in result.items() if k not in ['source_after','harness_checks']},indent=2))
