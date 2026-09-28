#!/usr/bin/env python3
"""Final frozen-policy gate with paired block bootstrap and cohort diagnostics."""
import collections,json,random,statistics,sys
from pathlib import Path
from wa6_analyze import metric,read
ROOT=Path(__file__).resolve().parents[1]/'specs/wa6'
if len(sys.argv)<2 or not Path(sys.argv[1]).is_file():raise SystemExit('usage: wa6_gate.py <final-arms manifest.json>; WA6 run directories (specs/wa6) are not published in this repository (results/ keeps only stripped records)')
manifest=json.load(open(sys.argv[1]))
data={k:[r for p in paths for r in read(Path(p)/'requests.jsonl') if r['split']=='eval'] for k,paths in manifest.items()}
def select(rs,g):
 if g=='all':return rs
 if g in ('caller','predictor'):return [r for r in rs if (r['designation'] is not None)==(g=='caller')]
 if g in ('short','long'):return [r for r in rs if (r['budget']==2048)==(g=='long')]
 return [r for r in rs if r['domain']==g]
def delta(a,b):return 100*(metric(b)['tps']/metric(a)['tps']-1)
groups={};rng=random.Random(6008)
for g in ['all','short','long','code','agent','en','ja','caller','predictor']:
 a,b=select(data['current'],g),select(data['prior'],g)
 aa,bb=metric(a),metric(b);samples=[]
 # The same selected parent block is drawn in both policies. Repeats stay
 # clustered with their parent block; no per-token independence assumption.
 for _ in range(10000):
  available=sorted({r['block'] for r in a});blocks=rng.choices(available,k=len(available))
  ar=[r for block in blocks for r in a if r['block']==block]
  br=[r for block in blocks for r in b if r['block']==block]
  samples.append(delta(ar,br))
 samples.sort()
 by_id=collections.defaultdict(list)
 for row in b:by_id[row['id']].append(row)
 equivalent=sum(row['decode_tokens']*sum(x['decode_s'] for x in by_id[row['id']])/sum(x['decode_tokens'] for x in by_id[row['id']]) for row in a)
 groups[g]=dict(current=aa,prior=bb,tps_delta_pct=delta(a,b),decode_time_reduction_pct=100*(1-equivalent/aa['decode_s']),raw_decode_time_reduction_pct=100*(1-bb['decode_s']/aa['decode_s']),current_token_equivalent_prior_s=equivalent,paired_block_95pct=[samples[249],samples[9749]])
# Equal parent/budget normalized time ratio cross-check, robust to long dominance.
per=collections.defaultdict(lambda:collections.defaultdict(list))
for policy in ('current','prior'):
 for r in data[policy]:per[r['id']][policy].append(r['decode_s']/r['decode_tokens'])
normalized=statistics.mean(statistics.mean(v['prior'])/statistics.mean(v['current']) for v in per.values())
threshold=groups['all']['tps_delta_pct']>=3 or groups['all']['decode_time_reduction_pct']>=3
no_regression=all(groups[d]['tps_delta_pct']>=-2 for d in ('code','agent','en','ja'))
checks=[]
for p in manifest['current']+manifest['prior']:
 s=json.load(open(Path(p)/'summary.json'));checks.append(s['needle'] and s['steady_min_mib']>=4096)
checked=bool(checks) and all(checks)  # no arm summaries checked verifies nothing
result=dict(groups=groups,equal_cell_normalized_time_reduction_pct=100*(1-normalized),numerical_gate=threshold and no_regression and checked,aggregate_threshold=threshold,domain_non_regression=no_regression,reserve_and_needle=checked,mechanism_gate='Requires separate width/initial-choice diagnostic evidence; numerical gate alone is insufficient.',statistical_bound='paired parent-block percentile bootstrap; CI is reported, not an added replacement for the specified point-estimate gate',arms=manifest)
(ROOT/'gate.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['\n## Final paired results\n','| cohort | current t/s | prior t/s | delta | 95% paired block interval | token-normalized decode time reduction |','|---|---:|---:|---:|---:|---:|']
for g,v in groups.items():
 lo,hi=v['paired_block_95pct'];lines.append(f"| {g} | {v['current']['tps']:.3f} | {v['prior']['tps']:.3f} | {v['tps_delta_pct']:+.3f}% | [{lo:+.3f}, {hi:+.3f}]% | {v['decode_time_reduction_pct']:+.3f}% |")
lines.append(f"\nNumerical gate {result['numerical_gate']}; aggregate threshold {threshold}; domain floor {no_regression}; reserve/needle {all(checks)}. Equal-cell normalized time reduction {100*(1-normalized):+.3f}%. Mechanism evidence assessed separately below.")
text='\n'.join(lines)+'\n';print(text)
if '\n## Final paired results\n' not in (ROOT.parent/'WA6_REQUEST_PRIOR.md').read_text():
 with (ROOT.parent/'WA6_REQUEST_PRIOR.md').open('a') as f:f.write(text)
