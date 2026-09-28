#!/usr/bin/env python3
"""BN1 log-only occupancy audit. No inferred exact mixed-width counts."""
import json,re,datetime,collections
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr,spearmanr
B=Path(__file__).resolve().parents[2]
root=B/'specs/bn1/study-v2'
out=B/'specs/wj1/study-20260908'
if not all((root/arm/'server.log').is_file() for arm in ('wa-A1','wa-A2','wa-A3')):
    raise SystemExit(f'wj1_history.py: needs the BN1 study server logs under {root}/wa-A*/server.log; '
                     'server logs are not published in this repository')
rows=[]
for arm in ['wa-A1','wa-A2','wa-A3']:
    lines=(root/arm/'server.log').read_text().splitlines()
    switches=[];logs=[];width=16
    for i,line in enumerate(lines):
        m=re.match(r'\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]',line)
        if not m:continue
        t=datetime.datetime.strptime(m[1],'%Y-%m-%d %H:%M:%S').replace(tzinfo=datetime.timezone(datetime.timedelta(hours=9))).timestamp()
        s=re.search(r'Switch adaptive runtime state: steps (\d+) -> (\d+)',line)
        if s:
            width=int(s[2])+1;switches.append(dict(t=t,width=width,old=int(s[1])+1,line=i+1))
        if 'Decode batch,' in line:logs.append(dict(t=t,width=width,line=i+1))
    for r in map(json.loads,(root/arm/'requests.jsonl').read_text().splitlines()):
        if r['workload']!='prose-ja':continue
        start=datetime.datetime.fromisoformat(r['timestamp']).timestamp();end=start+r['client']['ttft_seconds']+r['client']['decode_seconds']
        n=int(r['server']['acceptance']['verify_calls'])
        near=[s for s in switches if s['t']+1>=start and s['t']<=end]
        certain=[s for s in near if s['t']>=start and s['t']+1<=end]
        initial=next((s['width'] for s in reversed(switches) if s['t']+1<start),16)
        known=collections.Counter()
        if not near:known[initial]=n
        else:
            # Keep complete 40-step logging intervals safely within request;
            # discard an additional adjacent interval around every switch.
            for j in range(1,len(logs)):
                a,b=logs[j-1],logs[j]
                if a['t']<start or b['t']+1>end:continue
                left=logs[max(0,j-2)]['line'];right=logs[min(len(logs)-1,j+1)]['line']
                if any(left<=s['line']<=right for s in switches):continue
                if a['width']==b['width']:known[b['width']]+=40
        assert sum(known.values())<=n,(arm,r['prompt_id'],known,n)
        unresolved=n-sum(known.values())
        row=dict(arm=arm,prompt_id=r['prompt_id'],tps=r['client']['decode_tps'],acceptance=r['server']['acceptance']['tokens_per_verify'],verify_calls=n,
                 width_counts_exact=not near,known_interior_counts={str(w):known[w] for w in [4,8,16]},unresolved_steps=unresolved,
                 switches_certain=len(certain),switches_possible=len(near),path=[initial]+[s['width'] for s in near],switch_events=near)
        rows.append(row)
(out/'historical.json').write_text(json.dumps(rows,indent=2))
md=['\n## Historical BN1 Japanese audit\n', 'No step traces were recorded by BN1. Pure-width rows are exact (total request verify counter plus no switch in the whole-second boundary envelope). Mixed rows show conservative interior 40-step log-interval counts plus unallocated boundary/switch steps U; these are not exact occupancy counts. An extra log interval either side of a switch is excluded to accommodate pipeline ordering. Switches are certain–possible counts under whole-second timestamp uncertainty. W is draft steps + 1.\n', '| Arm | Prompt | t/s | tokens/verify | W4 / W8 / W16 known | U | Switches | Width path |','|---|---|---:|---:|---|---:|---|---|']
for r in rows:
    md.append(f"| {r['arm']} | {r['prompt_id'][-2:]} | {r['tps']:.2f} | {r['acceptance']:.3f} | {' / '.join(str(v) for v in r['known_interior_counts'].values())} | {r['unresolved_steps']} | {r['switches_certain']}–{r['switches_possible']} | {'→'.join(map(str,r['path']))} |")
x=np.array([(r['known_interior_counts']['8']+r['known_interior_counts']['16'])/r['verify_calls'] for r in rows]); y=np.array([r['tps'] for r in rows])
if x.std():
    corr=dict(pearson=float(pearsonr(x,y).statistic),spearman=float(spearmanr(x,y).statistic))
    md.append(f"\nAcross 24 prompt×restart rows, correlation of t/s with the conservative known W8+W16 verify fraction: Pearson r={corr['pearson']:.3f}, Spearman rho={corr['spearman']:.3f}. This lower-bound proxy is descriptive, not exact occupancy or a causal policy estimate; repeated parents are not independent samples.\n")
    (out/'historical-correlation.json').write_text(json.dumps(corr,indent=2))
(out/'historical.md').write_text('\n'.join(md)+'\n')
with (B/'specs/WJ1_WA_JAPANESE.md').open('a') as f:f.write('\n'.join(md)+'\n')
print('\n'.join(md))
