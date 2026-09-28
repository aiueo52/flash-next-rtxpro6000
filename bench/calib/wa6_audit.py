#!/usr/bin/env python3
"""Read-only final evidence audit; never starts or kills a GPU process."""
import hashlib,json,subprocess
from pathlib import Path
B=Path(__file__).resolve().parents[1];ROOT=B/'specs/wa6';W=Path('/home/user/tools/sglang-wa6')
if not (ROOT/'production-manifest.json').is_file():raise SystemExit(f'wa6_audit.py: needs {ROOT}/production-manifest.json and the WA6 run; WA6 run directories (specs/wa6) are not published in this repository (results/ keeps only stripped records)')
prod={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in json.loads((ROOT/'production-manifest.json').read_text()).items()}
model_changes=[]
model_manifest=json.loads((ROOT/'model-files-stat.json').read_text())
for name,old in model_manifest.items():
 p=Path(name)
 if not p.exists() or dict(resolved=str(p.resolve()),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)!=old:model_changes.append(name)
branch=subprocess.check_output(['git','-C',str(W),'branch','--show-current'],text=True).strip();head=subprocess.check_output(['git','-C',str(W),'rev-parse','HEAD'],text=True).strip()
streams={s:[json.loads(l) for l in (ROOT/f'{s}-stream.jsonl').open()] for s in ['prep','eval']}
parents={s:{r['parent'] for r in rs} for s,rs in streams.items()};assert not parents['prep']&parents['eval'];assert all(len(parents[s])==len(streams[s]) for s in streams)
committed=[]
for p in ROOT.glob('wa6-*/lock-time.json'):
 arm=p.parent;data=json.loads(p.read_text());data.update(label=arm.name,complete=(arm/'complete').exists(),invalid=(arm/'INVALID_CLIENT_TIMING').exists())
 if (arm/'summary.json').exists():
  summary=json.loads((arm/'summary.json').read_text());data.update(steady_min_mib=summary['steady_min_mib'],needle=summary['needle'])
 committed.append(data)
# Disk audit includes the copied runtime cache and any diagnostic subdirectories.
size=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
source={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [W/'python/sglang/srt/speculative'/f for f in ['adaptive_confidence.py','adaptive_runtime_state.py','adaptive_request_prior.py','eagle_worker_v2.py']]+list((B/'calib').glob('wa6_*.py'))}
timed_sources={}
if (ROOT/'final-arms.json').exists():
 for policy,paths in json.loads((ROOT/'final-arms.json').read_text()).items():
  if policy not in ('current','prior'):continue
  for name in paths:
   data=json.loads((Path(name)/'source-hashes.json').read_text())
   timed_sources[Path(name).name]=data
result=dict(model_manifest_files=len(model_manifest),model_stat_changes=model_changes,timed_source_manifests=timed_sources,production_hashes_unchanged=prod,branch=branch,head=head,parent_disjoint=True,parents={k:len(v) for k,v in parents.items()},gpu_lock_hours=sum(x['seconds'] for x in committed)/3600,disk_bytes=size,under_gpu_budget=sum(x['seconds'] for x in committed)<=4*3600,under_disk_budget=size<=120*10**9,arms=committed,final_source_hashes=source)
(ROOT/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('arms','final_source_hashes','timed_source_manifests')},indent=2))
