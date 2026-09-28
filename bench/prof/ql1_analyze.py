#!/usr/bin/env python3
"""CPU trace attribution and optimistic QSA removal bounds; no production imports."""
import collections
import contextlib
import json
from pathlib import Path
import statistics as st
import sys
import exclusive_time as ex

B=Path(__file__).resolve().parents[1]
ROOT=B/'prof/traces'
RUN=sys.argv[1] if len(sys.argv)>1 else '20260908-ql1'
_need=[B/'workloads/longctx/manifest.json']+[ROOT/f'ql1-{RUN}-{w}' for w in ('w4','w16')]
if not all(p.exists() for p in _need):
    sys.exit(f"ql1_analyze.py: needs the QL1 run {RUN} (prof/traces/ql1-{RUN}-w4|w16 and workloads/longctx/manifest.json); "
             "QL1 traces and the frozen long-context manifest are not published in this repository")
COMPONENTS=['scan','score_topk','packing','paged_attention','other']
IDEAS=['direct','shared_prefix','query_tiling','score_integration','topk_init_only']

def analyze(path):
    official=ex.analyze(str(path))
    with (path.parent/'exclusive.txt').open('w') as f:
        with contextlib.redirect_stdout(f): ex.report(official,top=300)
    _,X=ex.load(str(path))
    ann,phase=ex.phase_map(X)
    starts=sorted(e['ts'] for e in ann if e['name']=='draft')
    lo,hi=starts[0],starts[-1];steps=len(starts)-1
    rt={(e.get('args') or {}).get('correlation'):e for e in X if e.get('cat') in ('cuda_runtime','cuda_driver')}
    kern=sorted([e for e in X if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset')],key=lambda e:e['ts'])
    cats=[];phases=[];previous=collections.defaultdict(list);initializers=set()
    for i,e in enumerate(kern):
        name=e['name'];stream=e.get('args',{}).get('stream')
        if name=='kernel_kernel':
            cat='scan'
            for j in reversed(previous[stream][-6:]):
                if kern[j]['name']=='kernel_kernel': break
                if 'FillFunctor<float>' in kern[j]['name']:
                    initializers.add(j);break
        elif 'fast_topk' in name:
            cat='score_topk'
            # fast_topk(row_starts=None) allocates a zero-filled int32 row_starts.
            if previous[stream] and 'FillFunctor<int>' in kern[previous[stream][-1]]['name']:
                initializers.add(previous[stream][-1])
        elif name=='_compact_kv':cat='packing'
        elif name.startswith('kernel_mha'):cat='paged_attention'
        else:cat='other'
        cats.append(cat);previous[stream].append(i)
        launch=rt.get(e.get('args',{}).get('correlation'))
        ph=phase(launch['ts']) if launch else '[no-launch-site]'
        phases.append(ex.PHASE_SHORT.get(ph,ph))
    for i in initializers:cats[i]='score_topk'
    # Retain the first recursive pack in each CPU-annotated draft loop. A
    # kernel's launch correlation assigns CUDA graph children to that loop.
    first_pack={}
    for i,e in enumerate(kern):
        if cats[i]!='packing' or phases[i]!='draft':continue
        launch=rt.get(e.get('args',{}).get('correlation'))
        if launch is None:raise ValueError('Cannot assign draft pack to a loop')
        owners=[a for a in ann if a['name']=='draft' and a['ts']<=launch['ts']<=a['ts']+a['dur']]
        if not owners:raise ValueError('No draft annotation owns pack launch')
        owner=min(owners,key=lambda a:a['dur'])['ts']
        first_pack.setdefault(owner,i)
    retained=set(first_pack.values())
    raw=collections.Counter();exclusive=collections.Counter();counts=collections.Counter()
    phase_raw=collections.Counter();phase_exclusive=collections.Counter();events=[]
    families=collections.defaultdict(list)
    for i,e in enumerate(kern):
        a=max(lo,e['ts']);b=min(hi,e['ts']+e['dur'])
        if b<=a:continue
        raw[cats[i]]+=b-a;phase_raw[(phases[i],cats[i])]+=b-a;counts[cats[i]]+=1
        families[(phases[i],cats[i],ex.label(e['name'],e.get('cat')),str(e.get('args',{}).get('grid')))].append(e['dur'])
        events.extend([(a,1,i),(b,-1,i)])
    events.sort();active=set();prev=lo;gap=shared=0
    group_only=collections.Counter()
    # Selected sets overlap by design; never add these proposal budgets together.
    def eligible(i,idea):
        if idea=='direct':return cats[i]=='packing'
        if idea=='shared_prefix':return cats[i]=='packing' and phases[i]=='draft' and i not in retained
        if idea=='query_tiling':return cats[i]=='scan'
        if idea=='score_integration':return cats[i] in ('scan','score_topk')
        if idea=='topk_init_only':return cats[i]=='score_topk'
    ideas=IDEAS
    for ts,d,i in events:
        dt=ts-prev
        if dt:
            if not active:gap+=dt
            elif len(active)==1:
                j=next(iter(active));exclusive[cats[j]]+=dt;phase_exclusive[(phases[j],cats[j])]+=dt
            else:shared+=dt
            for idea in ideas:
                if active and all(eligible(j,idea) for j in active):group_only[idea]+=dt
        prev=ts
        if d==1:active.add(i)
        else:active.remove(i)
    gap+=hi-prev
    assert abs(sum(exclusive.values())+shared+gap-(hi-lo))<0.01
    assert abs(sum(exclusive.values())-sum(official['excl'].values()))<0.01
    fw=ex.draft_forwards(str(path))
    budgets={}
    for idea in ideas:
        raw_budget=sum(min(hi,e['ts']+e['dur'])-max(lo,e['ts']) for i,e in enumerate(kern)
                       if eligible(i,idea) and e['ts']<hi and e['ts']+e['dur']>lo)
        # Sharing retains the first recursive pack. All later prefix packs are
        # assumed free; the required changing tail work is zero-cost in this ceiling.
        # CPU draft markers lead the first captured GPU work by about one
        # iteration. Use all 20 complete GPU iterations for a raw deletion
        # ceiling, rather than understating it with the 19 CPU-start window.
        full_raw=sum(e['dur'] for i,e in enumerate(kern) if eligible(i,idea))
        raw_us=full_raw/len(starts);exclusive_us=group_only[idea]/steps
        frac=raw_us/official['wall']
        budgets[idea]=dict(raw_ceiling_us=raw_us,exclusive_credit_us=exclusive_us,
                           cpu_window_raw_us=raw_budget/steps,
                           raw_ceiling_share_pct=100*frac,max_gain_pct=100*frac/(1-frac),
                           clears_three_pct=frac>=1-1/1.03)
    result=dict(path=str(path),steps=steps,wall_us=official['wall'],median_wall_us=official['med']['wall'],
                raw_us={c:raw[c]/steps for c in COMPONENTS},exclusive_us={c:exclusive[c]/steps for c in COMPONENTS},
                counts_per_step={c:counts[c]/steps for c in COMPONENTS},shared_us=shared/steps,gap_us=gap/steps,
                full_trace_counts_per_step={c:sum(cat==c for cat in cats)/len(starts) for c in COMPONENTS},
                full_trace_raw_us={c:sum(e['dur'] for i,e in enumerate(kern) if cats[i]==c)/len(starts) for c in COMPONENTS},
                phase_raw_us={f'{p}:{c}':v/steps for (p,c),v in phase_raw.items()},
                phase_exclusive_us={f'{p}:{c}':v/steps for (p,c),v in phase_exclusive.items()},
                draft_forwards=fw,budgets=budgets,
                families=[dict(phase=p,component=c,label=l,grid=g,n=len(v)/steps,median_us=st.median(v))
                          for (p,c,l,g),v in families.items() if c!='other'])
    (path.parent/'ql1-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def fit(xs,ys):
    xm=st.mean(xs);ym=st.mean(ys)
    slope=sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/sum((x-xm)**2 for x in xs)
    intercept=ym-slope*xm
    sse=sum((y-intercept-slope*x)**2 for x,y in zip(xs,ys));sst=sum((y-ym)**2 for y in ys)
    return dict(intercept_us=intercept,slope_us_per_ktoken=slope,r2=1-sse/sst if sst else 1,
                rmse_us=(sse/len(xs))**0.5,constant_rmse_us=(sst/len(xs))**0.5,
                last_ratio=ys[-1]/ys[-2] if ys[-2] else None)

manifest=json.loads((B/'workloads/longctx/manifest.json').read_text())
rows=[]
for width in ['w4','w16']:
    arm=ROOT/f'ql1-{RUN}-{width}'
    for prompt in manifest['prompts']:
        if not prompt['perf']:continue
        td=ROOT/f'ql1-{RUN}-{width}-{prompt["id"]}'
        traces=list(td.glob('*.gz'))
        if len(traces)!=1:continue
        r=analyze(traces[0]);r.update(width=width,prompt=prompt['id'],context=prompt['context_tokens'],family=prompt['family'])
        walls=[json.loads(p.read_text()) for p in sorted(arm.glob(prompt['id']+'-wall*.json'))]
        if walls:
            r['wall_tps']=sum(x['timeline'][-1]['tokens']-x['timeline'][0]['tokens'] for x in walls)/sum(x['decode_seconds'] for x in walls)
            r['wall_tps_samples']=[x['decode_tps'] for x in walls]
            r['wall_acceptance']=sum(x['meta_info']['completion_tokens'] for x in walls)/sum(x['meta_info']['spec_verify_ct'] for x in walls)
        rows.append(r)
out=B/'workloads/longctx/results.json'
fits=[]
for width in ['w4','w16']:
    for family in ['doc-a','doc-b','needle-perf']:
        sub=sorted([r for r in rows if r['width']==width and r['family']==family],key=lambda r:r['context'])
        if len(sub)<4:continue
        for metric in ['raw_us','exclusive_us']:
            for c in COMPONENTS:
                f=fit([r['context']/1024 for r in sub],[r[metric][c] for r in sub])
                fits.append(dict(width=width,family=family,metric=metric,component=c,**f))
out.write_text(json.dumps(dict(rows=rows,fits=fits),indent=2)+'\n')
md=['## Measured per-context map (generated)','',
    'Times are microseconds per complete decode interval. Component cells are exclusive / raw; wall is mean / median. '
    'Other exclusive plus shared plus idle complete the wall partition. Throughput is the pooled rate from three unprofiled 128-token requests.', '',
    '| Width | Context | Family | Step mean / median | Decode t/s | Tokens/verify | Scan | Score + top-k | Packing | Paged attention | Other excl | Shared | Gap |',
    '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    cells=[f'{r["exclusive_us"][c]:.1f} / {r["raw_us"][c]:.1f}' for c in COMPONENTS[:4]]
    md.append(f'| {r["width"]} | {r["context"]//1024}k | {r["family"]} | {r["wall_us"]:.1f} / {r["median_wall_us"]:.1f} | {r.get("wall_tps",float("nan")):.1f} | {r.get("wall_acceptance",float("nan")):.3f} | '+ ' | '.join(cells)+f' | {r["exclusive_us"]["other"]:.1f} | {r["shared_us"]:.1f} | {r["gap_us"]:.1f} |')
md += ['', '## Removal ceilings (generated)', '',
       'Each cell: 19-CPU-interval exclusive credit / full-20-GPU-step raw-duration ceiling (microseconds), then ideal throughput gain from the raw ceiling. '
       'Shared prefix retains one recursive copy. Integration includes the entire score-producing scan: this intentionally loose ceiling '
       'cannot separate arithmetic from materialization. Top-k/init-only is shown to expose that uncertainty. These ideas overlap and must not be added.', '',
       '| Width | Context | Family | Direct access | Shared prefix | Query tiling | Score integration | Top-k/init only | 3% threshold us |',
       '|---|---:|---|---:|---:|---:|---:|---:|---:|']
for r in rows:
    cells=[f'{r["budgets"][idea]["exclusive_credit_us"]:.1f} / {r["budgets"][idea]["raw_ceiling_us"]:.1f}; +{r["budgets"][idea]["max_gain_pct"]:.2f}%' for idea in IDEAS]
    md.append(f'| {r["width"]} | {r["context"]//1024}k | {r["family"]} | '+' | '.join(cells)+f' | {r["wall_us"]*(1-1/1.03):.1f} |')
md += ['', '## Context fits (generated)', '',
       'OLS y = intercept + slope * context_k, k=1024. Four points per width/family. R2 is descriptive, not an inferential test. '
       'The ratio compares 96k to 64k; a flat final ratio alone is insufficient to establish saturation.', '',
       '| Width | Family | Metric | Component | Intercept us | Slope us/k | R2 | RMSE us | Constant RMSE | 96k / 64k |',
       '|---|---|---|---|---:|---:|---:|---:|---:|---:|']
for f in fits:
    ratio=f['last_ratio']
    md.append(f'| {f["width"]} | {f["family"]} | {f["metric"]} | {f["component"]} | {f["intercept_us"]:.2f} | {f["slope_us_per_ktoken"]:.3f} | {f["r2"]:.3f} | {f["rmse_us"]:.2f} | {f["constant_rmse_us"]:.2f} | {ratio:.3f} |' if ratio is not None else f'| {f["width"]} | {f["family"]} | {f["metric"]} | {f["component"]} | {f["intercept_us"]:.2f} | {f["slope_us_per_ktoken"]:.3f} | {f["r2"]:.3f} | {f["rmse_us"]:.2f} | {f["constant_rmse_us"]:.2f} | NA |')
(B/'workloads/longctx/tables.md').write_text('\n'.join(md)+'\n')
print(f'Analyzed {len(rows)} traces, {len(fits)} fits; wrote {out}')
