#!/usr/bin/env python3
"""WA6 paired aggregation, diagnostics and frozen opportunity gate. CPU only."""
import argparse,collections,json,random,statistics,sys
from pathlib import Path
B=Path(__file__).resolve().parents[1];ROOT=B/'specs/wa6';NOTE=B/'specs/WA6_REQUEST_PRIOR.md'
def read(p):return [json.loads(l) for l in p.open()]
def metric(rs):
 tokens=sum(r['decode_tokens'] for r in rs);seconds=sum(r['decode_s'] for r in rs)
 return dict(requests=len(rs),total_request_s=sum(r['total_s'] for r in rs),decode_tokens=tokens,completion_tokens=sum(r['completion_tokens'] for r in rs),decode_s=seconds,tps=tokens/seconds,ttft_mean_s=statistics.mean(r['ttft_s'] for r in rs),early32_mean_s=statistics.mean(e) if (e:=[r['early_s']['32'] for r in rs if r['early_s']['32'] is not None]) else None,early32_requests=sum(r['early_s']['32'] is not None for r in rs),verifies=sum(r['meta'].get('spec_verify_ct',0) for r in rs),accept_length=sum(r['completion_tokens'] for r in rs)/max(1,sum(r['meta'].get('spec_verify_ct',0) for r in rs)))
def arm(path):
 rs=read(path/'requests.jsonl');mem=read(path/'memory.jsonl')
 assert len({r['id'] for r in rs})==len(rs)
 assert not (path/'INVALID_CLIENT_TIMING').exists(), 'Excluded invalid harness arm'
 for r in rs:
  if r['decode_s'] >= .5:
   server = r['meta'].get('e2e_latency')
   assert server is not None, 'Missing server/client timing cross-check'
   assert r['total_s']-server < max(.1, .05*server), (r['id'], 'client backlog', r['total_s'],server)
 groups={'all':rs,'short':[r for r in rs if r['budget']<2048],'long':[r for r in rs if r['budget']==2048]}
 for d in ['code','agent','en','ja']:
  groups[d]=[r for r in rs if r['domain']==d]
  for c in ['short','long']:groups[d+'-'+c]=[r for r in groups[d] if (r['budget']==2048)==(c=='long')]
 result=dict(label=path.name,groups={k:metric(v) for k,v in groups.items()},steady_min_mib=min(r['free_mib'] for r in mem if r['phase']=='steady'),needle='PASS=True' in (path/'needle.log').read_text(),lock_seconds=json.loads((path/'lock-time.json').read_text())['seconds'])
 assert result['steady_min_mib'] >= 4096, ('steady VRAM reserve failed', path.name, result['steady_min_mib'])
 if (path/'steps.jsonl').exists():
  steps=read(path/'steps.jsonl');byrid=collections.defaultdict(list)
  for st in steps:
   for rid in st['rids']:byrid[rid].append(st)
  diagnostics={};allst=[]
  for r in rs:
   ss=byrid[r['rid']];assert ss,r['rid'];allst+=ss
   diagnostics[r['id']]=dict(initial_width=ss[0]['steps']+1,width_counts=dict(collections.Counter(str(x['steps']+1) for x in ss)),switches=sum(x['switch_cpu_ms']>0 for x in ss),switch_cpu_ms=sum(x['switch_cpu_ms'] for x in ss),switch_gpu_ms=sum(x.get('switch_gpu_ms') or 0 for x in ss),recovery_wait_gpu_ms=sum(x.get('recovery_wait_gpu_ms') or 0 for x in ss),accept_tokens=sum(sum(x['accepted'])+len(x['accepted']) for x in ss),verifies=len(ss),client_verifies=r['meta'].get('spec_verify_ct'),task=ss[0]['task'],source=ss[0]['source'],steps=ss)
  result['diagnostics']=diagnostics
  result['width_counts']=dict(collections.Counter(str(x['steps']+1) for x in allst))
  result['switch_gpu_ms']=sum(x.get('switch_gpu_ms') or 0 for x in allst);result['recovery_wait_gpu_ms']=sum(x.get('recovery_wait_gpu_ms') or 0 for x in allst);result['switch_timing_missing']=sum(x['switch_cpu_ms']>0 and x.get('switch_gpu_ms') is None for x in allst);result['switches']=sum(x['switch_cpu_ms']>0 for x in allst);result['switch_cpu_ms']=sum(x['switch_cpu_ms'] for x in allst)
 (path/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=[f'\n### Completed {path.name}\n','| cohort | n | emitted tokens | decode s | delivered t/s | tokens/verify | early32 ms |','|---|---:|---:|---:|---:|---:|---:|']
 for g in ['all','short','long','code','agent','en','ja']:
  v=result['groups'][g];early=v['early32_mean_s'];lines.append(f"| {g} | {v['requests']} | {v['completion_tokens']} | {v['decode_s']:.4f} | {v['tps']:.3f} | {v['accept_length']:.3f} | {early*1000 if early is not None else 0:.3f} |")
 lines.append(f"\nSteady minimum {result['steady_min_mib']} MiB; needle {result['needle']}; occupied {result['lock_seconds']/3600:.4f} GPU-hours.")
 if 'width_counts' in result:lines.append(f"Width step occupancy: {result['width_counts']}; {result['switches']} runtime swaps, total CPU recovery-drain/swap submission {result['switch_cpu_ms']:.3f} ms. This is host elapsed time, not synchronized GPU recovery duration; complete client decode time includes execution. GPU event switch total {result['switch_gpu_ms']:.3f} ms, uncovered recovery-fence wait {result['recovery_wait_gpu_ms']:.3f} ms; {result['switch_timing_missing']} timings unavailable.")
 text='\n'.join(lines)+'\n';print(text)
 if f'### Completed {path.name}' not in NOTE.read_text():
  with NOTE.open('a') as f:f.write(text)
 return result

def opportunity(paths):
 data={name:[r for r in read(Path(p)/'requests.jsonl') if r['split']=='prep'] for name,p in paths.items()}
 curr={r['id']:r for r in data['current']};fixed={s:{r['id']:r for r in data[f'fixed{s}']} for s in [4,8,16]}
 diag=json.loads((Path(paths['current'])/'summary.json').read_text())['diagnostics'];cost=json.loads((ROOT/'costs.json').read_text())['step_cost_ms']
 rows=[];base=sum(r['decode_s'] for r in curr.values())
 for id,r in curr.items():
  best=max([4,8,16],key=lambda s:fixed[s][id]['tps'])
  # Diagnostic only: matched fixed requests can follow different trajectories.
  ideal=r['decode_tokens']/fixed[best][id]['tps']
  ss=diag[id]['steps'];wrong=[s for s in ss if s['steps']+1!=best]
  width_time=sum(cost[str(s['steps'])] for s in ss)/1000
  dwell_time=sum(cost[str(s['steps'])] for s in wrong)/1000
  rows.append(dict(id=id,domain=r['domain'],budget=r['budget'],best_fixed_width=best,initial_width=ss[0]['steps']+1,current_s=r['decode_s'],fixed_equivalent_s=ideal,optimistic_saved_s=max(0,r['decode_s']-ideal),wrong_width_step_fraction=len(wrong)/len(ss),measured_cost_wrong_width_s=dwell_time,measured_cost_total_s=width_time))
 saving=sum(r['optimistic_saved_s'] for r in rows)/base
 # This is deliberately generous: if even the fixed per-request oracle cannot
 # recover 3%, stop. Wrong-width occupancy establishes whether dwell is present.
 dwell=sum(r['measured_cost_wrong_width_s'] for r in rows)/sum(r['measured_cost_total_s'] for r in rows)
 result=dict(opportunity_fraction=saving,wrong_width_time_fraction=dwell,continue_prior=saving>=.03 and dwell>=.03,oracle='diagnostic only; fixed runs are different trajectories; positive per-request savings form an optimistic screen, not a shipping policy',rows=rows)
 (ROOT/'opportunity.json').write_text(json.dumps(result,indent=2)+'\n')
 text=f"\n### Preparation stop gate\n\nOptimistic matched-fixed opportunity {saving*100:.3f}%; measured-cost wrong-width occupancy {dwell*100:.3f}%. Continue prior experiment: {result['continue_prior']}. This oracle is diagnostic only; it does not select runtime widths or train the predictor.\n"
 with NOTE.open('a') as f:f.write(text)
 print(text)
if __name__=='__main__':
 dirs=[sys.argv[2]] if sys.argv[1:2]==['arm'] else [x.split('=',1)[-1] for x in sys.argv[2:]]
 if len(sys.argv)<3 or not all((Path(d)/'requests.jsonl').is_file() for d in dirs):
  raise SystemExit('usage: wa6_analyze.py arm <run dir> | wa6_analyze.py opportunity <label>=<run dir>...; needs WA6 run directories with requests.jsonl; WA6 run directories (specs/wa6) are not published in this repository (results/ keeps only stripped records)')
 if sys.argv[1]=='arm':arm(Path(sys.argv[2]))
 else:opportunity(dict(x.split('=',1) for x in sys.argv[2:]))
