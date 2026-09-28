#!/usr/bin/env python3
"""Freeze per-width physical mean costs from existing X4 raw traces, CPU only."""
import hashlib, importlib.util, json, statistics
from pathlib import Path
B=Path(__file__).resolve().parents[1]; OUT=B/'specs/wa6'
if not (B/'specs/x4/exclusive_gpu.py').is_file():raise SystemExit(f'wa6_costs.py: needs the X4 raw traces and {B}/specs/x4/exclusive_gpu.py; X4 traces are not published in this repository')
spec=importlib.util.spec_from_file_location('x4cost',B/'specs/x4/exclusive_gpu.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
extra=json.loads((B/'specs/x4/fixed-floor-extra-kernels.json').read_text())
# Mean of observed incremental confidence kernels and D2H. Raw sum is a
# conservative additive charge, not a claim of exclusive removable time.
paths=sorted((B/'specs/x4/x4-20260908-3').glob('*/*.gz'))
charges=[]
for p in paths:
 _,events=m.load(str(p)); ann,phase=m.phase_map(events)
 rt={e.get('args',{}).get('correlation'):e for e in events if e.get('cat') in ('cuda_runtime','cuda_driver')}
 names={x['full_name'] for x in extra}
 ks=[e for e in events if e.get('cat') in ('kernel','gpu_memcpy') and (e['name'] in names or e.get('cat')=='gpu_memcpy' and 'DtoH' in e['name']) and (r:=rt.get(e.get('args',{}).get('correlation'))) and phase(r['ts'])=='draft_extend']
 n=sum(e['name']=='draft_extend' for e in ann)
 charges.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),mean_us=sum(e['dur'] for e in ks)/n,events_per_step=len(ks)/n))
confidence=statistics.mean(r['mean_us'] for r in charges)
rows=[]; table={}
for steps,arm in [(3,'w4'),(7,'w8'),(15,'4')]:
 costs=[]
 for p in sorted((B/f'specs/x4/x4-20260908-{arm}').glob('*/*.gz')):
  r=m.analyze(str(p),window='gpu'); costs.append(r['wall'])
  rows.append(dict(steps=steps,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),intervals=r['steps'],physical_mean_ms=r['wall']/1000,exclusive_ms=sum(r['excl'].values())/r['steps']/1000,shared_ms=r['shared']/r['steps']/1000,gap_ms=r['gap']/r['steps']/1000,gpu_events_per_step=r['nkern']/r['steps']))
 table[str(steps)]=round((statistics.mean(costs)+confidence)/1000,6)
result=dict(step_cost_ms=table,confidence_raw_mean_us=confidence,confidence_traces=charges,fixed_traces=rows,recovery='Included in physical draft-first-kernel to next-draft-first-kernel wall; do not add overlapping recovery again.',aggregation='Equal code/en/agent trace means at each width; Japanese has no historical fixed trace. Raw confidence charge is conservative, not exclusive time. No scheduler cost invented.')
(OUT/'costs.json').write_text(json.dumps(result,indent=2)+'\n')
cfg=json.loads((B/'adaptive/w16_3_7_15_c.json').read_text());cfg['1'].update(step_cost_ms=table,request_prior=True)
(OUT/'prior.json').write_text(json.dumps(cfg,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('fixed_traces','confidence_traces')},indent=2))
