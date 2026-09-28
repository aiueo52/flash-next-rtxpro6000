#!/usr/bin/env python3
"""Aggregate completed QL1 arms and audit measurements; CPU only."""
import hashlib
import json
from pathlib import Path
import statistics as st
import sys

B=Path(__file__).resolve().parents[1];W=B/'workloads/longctx';ROOT=B/'prof/traces'
RUN=sys.argv[1] if len(sys.argv)>1 else '20260908-ql1'
_need=[W/'results.json',W/'frozen.json']+[ROOT/f'ql1-{RUN}-{w}' for w in ('w4','w16')]
if not all(p.exists() for p in _need):
    sys.exit(f"ql1_finalize.py: needs ql1_analyze.py output (workloads/longctx/results.json, frozen.json) and the QL1 run "
             f"{RUN} traces; these are not published in this repository")
data=json.loads((W/'results.json').read_text());rows=data['rows']
assert len(rows)==24, f'Expected 24 analyzed traces, found {len(rows)}'
cs=['scan','score_topk','packing','paged_attention','other']
ideas=['direct','shared_prefix','query_tiling','score_integration']
aggregate=[];audit={};fits=[]

def fit(xs,ys):
    xm=st.mean(xs);ym=st.mean(ys)
    slope=sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/sum((x-xm)**2 for x in xs)
    intercept=ym-slope*xm;residual=sum((y-intercept-slope*x)**2 for x,y in zip(xs,ys));total=sum((y-ym)**2 for y in ys)
    return dict(intercept=intercept,slope=slope,r2=1-residual/total if total else 1,
                rmse=(residual/len(xs))**0.5,constant_rmse=(total/len(xs))**0.5,ratio96_64=ys[-1]/ys[-2])

for width in ['w4','w16']:
    arm=ROOT/f'ql1-{RUN}-{width}';assert (arm/'complete').exists()
    samples=[json.loads(p.read_text()) for p in arm.glob('*-wall[123].json')]
    assert len(samples)==36 and all(s['meta_info']['completion_tokens']==128 for s in samples)
    assert all(s['meta_info']['prompt_tokens']==s['context_tokens'] for s in samples)
    qa=[json.loads(p.read_text()) for p in arm.glob('*-qa.json')]
    assert len(qa)==12
    mem=[json.loads(line) for line in (arm/'memory.jsonl').read_text().splitlines()]
    steady=[x['free_mib'] for x in mem if x['ready']];assert min(steady)>=4096
    coverage=[json.loads(p.read_text()) for p in arm.glob('*-coverage.json')]
    assert len(coverage)==12 and all(c['draft_annotations']==20 and c['kernels']>0 for c in coverage)
    info=json.loads((arm/'server-info-ready.json').read_text())
    assert info['mem_fraction_static']==0.920
    assert info['speculative_num_draft_tokens']==int(width[1:])
    audit[width]=dict(complete=True,wall_samples=len(samples),generated_wall_tokens=sum(s['meta_info']['completion_tokens'] for s in samples),
                     trace_count=len(coverage),draft_annotations=240,complete_trace_intervals=228,
                     exact_needle_pass=sum(q['text'].strip()=='AURORA-CEDAR-7319' for q in qa),needle_total=len(qa),
                     min_steady_free_mib=min(steady),memory_samples=len(steady),
                     kv=json.loads((arm/'kv-budget.json').read_text()),lock_hours=json.loads((arm/'lock-hours.json').read_text()))
    for n in [8192,32768,65536,98304]:
        rs=[r for r in rows if r['width']==width and r['context']==n];assert len(rs)==3
        assert all(r['full_trace_counts_per_step']['scan']==13 for r in rs)
        assert all(r['full_trace_counts_per_step']['packing']==int(width[1:])+11 for r in rs)
        ss=[s for s in samples if s['context_tokens']==n];assert len(ss)==9
        wall=st.mean(r['wall_us'] for r in rs)
        a=dict(width=width,context=n,wall_us=wall,median_wall_us=st.mean(r['median_wall_us'] for r in rs),
               raw={c:st.mean(r['full_trace_raw_us'][c] for r in rs) for c in cs},
               exclusive={c:st.mean(r['exclusive_us'][c] for r in rs) for c in cs},
               shared_us=st.mean(r['shared_us'] for r in rs),gap_us=st.mean(r['gap_us'] for r in rs),
               tps=sum(s['timeline'][-1]['tokens']-s['timeline'][0]['tokens'] for s in ss)/sum(s['decode_seconds'] for s in ss),
               tps_range=[min(s['decode_tps'] for s in ss),max(s['decode_tps'] for s in ss)],
               acceptance=sum(s['meta_info']['completion_tokens'] for s in ss)/sum(s['meta_info']['spec_verify_ct'] for s in ss),
               required_us=wall*(1-1/1.03),budgets={})
        for idea in ideas:
            raw=st.mean(r['budgets'][idea]['raw_ceiling_us'] for r in rs);excl=st.mean(r['budgets'][idea]['exclusive_credit_us'] for r in rs)
            a['budgets'][idea]=dict(raw_us=raw,exclusive_us=excl,share_pct=100*raw/wall,gain_pct=100*raw/(wall-raw),
                                  per_family_gain_range=[min(r['budgets'][idea]['max_gain_pct'] for r in rs),max(r['budgets'][idea]['max_gain_pct'] for r in rs)],
                                  region_reduction_required_pct=100*a['required_us']/raw)
        a['remainder_us']=wall-sum(a['exclusive'][c] for c in cs[:4]);aggregate.append(a)
    sub=[a for a in aggregate if a['width']==width];xs=[a['context']/1024 for a in sub]
    for c in cs:
        fits.append(dict(width=width,component=c,metric='raw',**fit(xs,[a['raw'][c] for a in sub])))
        fits.append(dict(width=width,component=c,metric='exclusive',**fit(xs,[a['exclusive'][c] for a in sub])))
    fits.append(dict(width=width,component='step',metric='wall',**fit(xs,[a['wall_us'] for a in sub])))
frozen=json.loads((W/'frozen.json').read_text())
assert frozen['files'], 'frozen.json lists no source files: nothing to audit'
audit['frozen_source_matches']={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in frozen['files'].items()}
assert all(audit['frozen_source_matches'].values())
audit['analysis_script_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((B/'prof').glob('ql1_*.py'))}
(W/'validation.json').write_text(json.dumps(audit,indent=2)+'\n')
(W/'aggregate.json').write_text(json.dumps(dict(rows=aggregate,fits=fits),indent=2)+'\n')
md=['## Aggregate context map','',
    'Three prompt-family traces are equally weighted for time. Decode t/s pools nine unprofiled 128-token requests. '
    'Component cells show exclusive microseconds (% of mean step). The remainder includes other kernels, all shared intervals, and idle gaps.', '',
    '| Width | Context | Mean step ms | Decode t/s | Tokens/verify | Scan | Score/top-k | Packing | Paged attention | Remainder us |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for a in aggregate:
    cells=[f'{a["exclusive"][c]:.1f} ({100*a["exclusive"][c]/a["wall_us"]:.2f}%)' for c in cs[:4]]
    md.append(f'| {a["width"]} | {a["context"]//1024}k | {a["wall_us"]/1000:.3f} | {a["tps"]:.1f} | {a["acceptance"]:.3f} | '+' | '.join(cells)+f' | {a["remainder_us"]:.1f} |')
md+=['','## Aggregate removable budgets','',
     'Cells: full-20-GPU-step optimistic raw deletion ceiling in microseconds / share of mean CPU-start step / maximum throughput gain. The exclusive table retains the official 19-CPU-interval window. Zero replacement cost and unchanged acceptance; ceilings overlap and are not additive. Per-family raw and exclusive values are in the detailed table.', '',
     '| Width | Context | Direct selected access | Shared recursive prefix | Query-row tiling | Score/top-k integration | Required saving us |',
     '|---|---:|---:|---:|---:|---:|---:|']
for a in aggregate:
    cells=[f'{a["budgets"][i]["raw_us"]:.1f} / {a["budgets"][i]["share_pct"]:.2f}% / +{a["budgets"][i]["gain_pct"]:.2f}%' for i in ideas]
    md.append(f'| {a["width"]} | {a["context"]//1024}k | '+' | '.join(cells)+f' | {a["required_us"]:.1f} |')
md+=['','## Aggregate scaling fits','',
     'y(us) = intercept + slope * context_k, k=1024; four context means. Aggregate raw fits use all 20 complete GPU iterations; exclusive fits retain the official 19-CPU-interval window. Detailed per-family CPU-window fits are reported separately. No extrapolation beyond 96k.', '',
     '| Width | Component | Metric | Intercept us | Slope us/k | R2 | RMSE | Constant RMSE | 96k/64k |',
     '|---|---|---|---:|---:|---:|---:|---:|---:|']
for f in fits:
    md.append(f'| {f["width"]} | {f["component"]} | {f["metric"]} | {f["intercept"]:.2f} | {f["slope"]:.3f} | {f["r2"]:.3f} | {f["rmse"]:.2f} | {f["constant_rmse"]:.2f} | {f["ratio96_64"]:.3f} |')
(W/'summary-tables.md').write_text('\n'.join(md)+'\n')
print(json.dumps(audit,indent=2))
