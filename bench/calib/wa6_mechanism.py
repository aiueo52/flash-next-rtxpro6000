#!/usr/bin/env python3
"""Post-hoc mechanism diagnostic. Never loaded by the serving policy."""
import collections,json,statistics
from pathlib import Path
from wa6_analyze import read,metric
ROOT=Path(__file__).resolve().parents[1]/'specs/wa6'
if not (ROOT/'sequence-progress.json').is_file():raise SystemExit(f'wa6_mechanism.py: needs {ROOT}/sequence-progress.json; WA6 run directories (specs/wa6) are not published in this repository (results/ keeps only stripped records)')
progress=json.loads((ROOT/'sequence-progress.json').read_text());final=json.loads((ROOT/'final-arms.json').read_text());cost=json.loads((ROOT/'costs.json').read_text())['step_cost_ms']
fixed={s:{r['id']:r for r in read(Path(final[f'fixed{s}'][0])/'requests.jsonl') if r['split']=='eval'} for s in (4,8,16)}
stream=read(ROOT/'eval-stream.jsonl')
transitions={r['id']:('start' if i==0 else stream[i-1]['domain']+'->'+r['domain']) for i,r in enumerate(stream)}
rows=[];summaries={}
for policy in ('current','prior'):
 path=Path(progress[policy+'-diagnostic']);summaries[policy]=json.loads((path/'summary.json').read_text());requests={r['id']:r for r in read(path/'requests.jsonl')}
 for id,diag in summaries[policy]['diagnostics'].items():
  r=requests[id];best=max((4,8,16),key=lambda s:fixed[s][id]['tps'])
  ss=diag['steps'];token_count=0;prefix=[]
  for step in ss:
   if token_count>=min(64,r['decode_tokens']):break
   prefix.append(step);token_count+=sum(step['accepted'])+len(step['accepted'])
  rows.append(dict(policy=policy,id=id,transition=transitions[id],domain=r['domain'],budget=r['budget'],source=diag['source'],task=diag['task'],best_fixed_width=best,initial_width=diag['initial_width'],initial_matches_fixed_best=best==diag['initial_width'],width_counts=diag['width_counts'],early64_wrong_steps=sum(s['steps']+1!=best for s in prefix),early64_total_steps=len(prefix),wrong_width_cost_s=sum(cost[str(s['steps'])] for s in ss if s['steps']+1!=best)/1000,total_width_cost_s=sum(cost[str(s['steps'])] for s in ss)/1000,decode_s=r['decode_s'],decode_tokens=r['decode_tokens'],early32_s=r['early_s']['32']))
groups={}
for policy in ('current','prior'):
 groups[policy]={}
 for group in ('all','short','long','code','agent','en','ja','caller','predictor'):
  rs=[r for r in rows if r['policy']==policy and (group=='all' or r['domain']==group or group=='short' and r['budget']<2048 or group=='long' and r['budget']==2048 or group=='caller' and r['source']=='caller' or group=='predictor' and r['source']=='wa6-prompt-stats-v1')]
  if not rs:continue
  width=collections.Counter()
  for r in rs:width.update(r['width_counts'])
  groups[policy][group]=dict(n=len(rs),initial_best_count=sum(r['initial_matches_fixed_best'] for r in rs),width_counts=dict(width),wrong_width_cost_s=sum(r['wrong_width_cost_s'] for r in rs),total_width_cost_s=sum(r['total_width_cost_s'] for r in rs),early64_wrong_steps=sum(r['early64_wrong_steps'] for r in rs),early64_total_steps=sum(r['early64_total_steps'] for r in rs),early32_mean_s=statistics.mean(v) if (v:=[r['early32_s'] for r in rs if r['early32_s'] is not None]) else None)
transition_groups={}
for policy in ('current','prior'):
 transition_groups[policy]={}
 for transition in sorted(set(transitions.values())):
  rs=[r for r in rows if r['policy']==policy and r['transition']==transition]
  transition_groups[policy][transition]=dict(n=len(rs),initial_best_count=sum(r['initial_matches_fixed_best'] for r in rs),wrong_width_cost_s=sum(r['wrong_width_cost_s'] for r in rs),decode_tokens=sum(r['decode_tokens'] for r in rs),decode_s=sum(r['decode_s'] for r in rs),early32_mean_s=statistics.mean(v) if (v:=[r['early32_s'] for r in rs if r['early32_s'] is not None]) else None)
result=dict(transition_groups=transition_groups,groups=groups,rows=rows,oracle='Fixed per-request best is diagnostic only, obtained after all timings; different greedy trajectories and separated reference runs limit causal interpretation.',switches={p:{k:summaries[p][k] for k in ['switches','switch_cpu_ms','switch_gpu_ms','recovery_wait_gpu_ms','switch_timing_missing']} for p in summaries})
a,b=groups['current']['all'],groups['prior']['all']
current_by_id={r['id']:r for r in rows if r['policy']=='current'}
prior_equivalent_wrong_s=sum(r['wrong_width_cost_s']/r['decode_tokens']*current_by_id[r['id']]['decode_tokens'] for r in rows if r['policy']=='prior')
result['current_token_equivalent_prior_wrong_width_s']=prior_equivalent_wrong_s
result['mechanism_supported']=b['initial_best_count']>a['initial_best_count'] and prior_equivalent_wrong_s<a['wrong_width_cost_s']
(ROOT/'mechanism.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['\n## Width and request-boundary mechanism\n','| policy / cohort | initial width matches fixed best | W4 / W8 / W16 steps | wrong-width modeled time s | steps covering first <=64 delivered decode tokens: wrong / total | early32 ms |','|---|---:|---:|---:|---:|---:|']
for p in groups:
 for g in ('all','short','long','code','agent','en','ja'):
  v=groups[p][g];width='/'.join(str(v['width_counts'].get(str(s),0)) for s in (4,8,16));lines.append(f"| {p}/{g} | {v['initial_best_count']}/{v['n']} | {width} | {v['wrong_width_cost_s']:.3f} | {v['early64_wrong_steps']}/{v['early64_total_steps']} | {1000*v['early32_mean_s']:.3f} |")
lines.append(f"\nMechanism supported by more matched-best initial widths and less token-normalized wrong-width modeled dwell: {result['mechanism_supported']}. Prior wrong-width modeled time at matched current token counts: {prior_equivalent_wrong_s:.3f} s versus current {a['wrong_width_cost_s']:.3f} s. Fixed-best labels are post-hoc diagnostics only; model-based dwell seconds are not substituted for measured client decode time. Full records, classifier outcomes and transition pairs are in mechanism.json.")
text='\n'.join(lines)+'\n';print(text)
with (ROOT.parent/'WA6_REQUEST_PRIOR.md').open('a') as f:f.write(text)
