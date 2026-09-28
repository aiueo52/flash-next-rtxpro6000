#!/usr/bin/env python3
"""Validate WJ1 trace/counter pairing and produce incremental report tables."""
import argparse,collections,datetime,json,math
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr,spearmanr
from paired_ab import analyze
from run_wj1 import check_frozen,occupied_seconds
B=Path(__file__).resolve().parents[2]

def audit(d):
    rows=list(map(json.loads,(d/'requests.jsonl').read_text().splitlines()))
    trace=list(map(json.loads,(d/'trace.jsonl.rank0').read_text().splitlines()))
    result=[]
    for i,r in enumerate(rows):
        start=datetime.datetime.fromisoformat(r['timestamp']).timestamp()
        end=start+r['client']['ttft_seconds']+r['client']['decode_seconds']
        next_start=datetime.datetime.fromisoformat(rows[i+1]['timestamp']).timestamp() if i+1<len(rows) else end+0.25
        ts=[t for t in trace if start<=t['t']<next_start]
        n=int(r['server']['acceptance']['verify_calls'])
        raw_ts=ts
        assert len(raw_ts) in (n,n+1),(d.name,r['prompt_id'],'trace/counter count',len(raw_ts),n)
        # CPU trace precedes req.finished() guard; one overlap tail result can
        # be traced after completion without incrementing request counters.
        tail=raw_ts[n:]
        ts=raw_ts[:n]
        assert not tail or tail[0]['t']>=end-0.02,(r['prompt_id'],'unexpected extra trace inside request')
        assert all(t['bs']==1 and len(t['a'])==1 and t['steps'] in (3,7,15) and 0<=t['a'][0]<=t['steps'] for t in ts)
        assert ts[-1]['t']<=end+0.25,(r['prompt_id'],'timestamp bound')
        emitted=sum(1+t['a'][0] for t in ts);tokens=r['client']['usage']['completion_tokens']
        assert -1<=emitted-tokens<=16,(r['prompt_id'],'emission mismatch',emitted,tokens)
        hist=collections.Counter(t['steps']+1 for t in ts)
        switches=sum(a['steps']!=b['steps'] for a,b in zip(ts,ts[1:]))
        result.append(dict(arm=d.name,domain=r['workload'],prompt_id=r['prompt_id'],tps=r['client']['decode_tps'],acceptance=r['server']['acceptance']['tokens_per_verify'],verify_calls=n,width_counts={str(w):hist[w] for w in (4,8,16)},switches=switches,first_width=ts[0]['steps']+1,last_width=ts[-1]['steps']+1,trace_emission_minus_completion=emitted-tokens,raw_executed_width_counts=dict(collections.Counter(str(t['steps']+1) for t in raw_ts)),excluded_finished_overlap_tail=len(tail),duration=end-start,output_sha256=r['client']['output_sha256']))
    memory=list(map(json.loads,(d/'memory.jsonl').read_text().splitlines()))
    result_all=dict(rows=result,min_free_mib=min(m['free_mib'] for m in memory),min_steady_free_mib=min(m['free_mib'] for m in memory if m['steady']),lock_seconds=json.loads((d/'complete.json').read_text())['lock_seconds'])
    assert result_all['min_free_mib']>=4096
    (d/'width-audit.json').write_text(json.dumps(result_all,indent=2))
    return result_all

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--partial',action='store_true');a=p.parse_args();root=a.root.resolve()
    dirs=[]
    for arm in ('A1','B1','B2','A2'):
        matches=list(root.glob('wj1-*-'+arm+'/complete.json'))
        if not matches:
            if a.partial:continue
            raise SystemExit(f'wj1_report.py: no completed WJ1 arm {arm} under {root} (wj1-*-{arm}/complete.json); '
                             'WJ1 run directories are not published in this repository (use --partial for a run in progress)')
        assert len(matches)==1
        dirs.append(matches[0].parent)
    if not dirs:
        raise SystemExit(f'wj1_report.py: no completed WJ1 arm under {root}; nothing to report')
    audits=[audit(d) for d in dirs]
    md=['\n## WJ1 measured arms\n','Width counts below cover the first N verified results, where N is the exact completed-request verify counter. The existing CPU trace is emitted before the req.finished() guard; an optional single trailing overlap result after completion is recorded separately as executed work and excluded from these counter-aligned counts. Source: batch_result_processor.py _resolve_spec_v2_output and eagle_worker_v2.py on_verify_complete_cpu. Switches count changes between consecutive measured verify steps within a request; a policy decision after its final verify is not a measured switch. Trace token sums may exceed capped client completion by up to one verify width; exact counter acceptance remains primary.\n','| Arm | Domain | t/s geometric mean | acceptance geometric mean | W4 / W8 / W16 verifies | Width % 4 / 8 / 16 | Switches |','|---|---|---:|---:|---|---|---:|']
    for d,au in zip(dirs,audits):
        for dom in ('prose-ja','prose-en'):
            rs=[r for r in au['rows'] if r['domain']==dom];h={w:sum(r['width_counts'][w] for r in rs) for w in ('4','8','16')};n=sum(h.values())
            md.append(f"| {d.name.rsplit('-',1)[1]} | {dom} | {np.exp(np.mean(np.log([r['tps'] for r in rs]))):.2f} | {np.exp(np.mean(np.log([r['acceptance'] for r in rs]))):.3f} | {' / '.join(map(str,h.values()))} | {' / '.join(f'{100*v/n:.2f}' for v in h.values())} | {sum(r['switches'] for r in rs)} |")
    md+=['\n| Arm | Lock seconds | Min free MiB startup+steady | Min free MiB steady |','|---|---:|---:|---:|']
    for d,au in zip(dirs,audits):md.append(f"| {d.name} | {au['lock_seconds']:.1f} | {au['min_free_mib']} | {au['min_steady_free_mib']} |")
    md.append(f'\nOccupied lock seconds: {occupied_seconds(root):.1f}; queue time excluded.\n')
    if len(dirs)==4:
        paired=analyze([d/'requests.jsonl' for d in dirs],schedule='ABBA')
        (root/'paired.json').write_text(json.dumps(paired,indent=2))
        md+=['\n## Paired policy comparison\n','Percent changes are exp(mean log B/A) − 1, equal weighting of eight parents and two blocks. CI is the 20,000-draw two-sided 95% parent bootstrap, conditional on these four restarts.\n','| Domain | Metric | Block 1 mean ln(B1/A1) / % | Block 2 mean ln(B2/A2) / % | Pooled B/A % | 95% CI % | A2/A1 drift % [95% CI] | Drift flag |','|---|---|---:|---:|---:|---|---|---|']
        for dom,metrics in paired['domains'].items():
            for met,v in metrics.items():
                dr=v['A2_over_A1_drift'];bs=v['block_means'];ci=v['ci_pct'];dc=dr['ci_pct']
                md.append(f"| {dom} | {met} | {bs[0]['log_ratio']:+.5f} / {bs[0]['change_pct']:+.2f} | {bs[1]['log_ratio']:+.5f} / {bs[1]['change_pct']:+.2f} | {v['change_pct']:+.2f} | [{ci[0]:+.2f}, {ci[1]:+.2f}] | {dr['change_pct']:+.2f} [{dc[0]:+.2f}, {dc[1]:+.2f}] | {dr['flag']} |")
        md+=['\n| Domain / prompt | ln B1/A1 t/s | ln B2/A2 t/s | Mean ln B/A t/s | ln B1/A1 acceptance | ln B2/A2 acceptance | Mean ln B/A acceptance |','|---|---:|---:|---:|---:|---:|---:|']
        for dom,metrics in paired['domains'].items():
            for t,c in zip(metrics['tps']['per_prompt'],metrics['acceptance']['per_prompt']):
                nums=t['log_ratios']+[t['mean_log_ratio']]+c['log_ratios']+[c['mean_log_ratio']]
                md.append('| '+dom+'/'+t['prompt_id'][-2:]+' | '+' | '.join(f'{v:+.5f}' for v in nums)+' |')
        # Exact per-request raw results and occupancy retained in table.
        md+=['\n## Exact per-request WJ1 occupancy\n','| Arm | Domain / prompt | t/s | tokens/verify | W4 / W8 / W16 | Switches |','|---|---|---:|---:|---|---:|']
        for au in audits:
            for r in au['rows']:
                md.append(f"| {r['arm'].rsplit('-',1)[1]} | {r['domain']}/{r['prompt_id'][-2:]} | {r['tps']:.2f} | {r['acceptance']:.3f} | {' / '.join(map(str,r['width_counts'].values()))} | {r['switches']} |")
        correlations={}
        for arm in ('A','B'):
            rs=[r for au in audits for r in au['rows'] if r['arm'].rsplit('-',1)[1].startswith(arm) and r['domain']=='prose-ja']
            x=[(r['width_counts']['8']+r['width_counts']['16'])/r['verify_calls'] for r in rs];y=[r['tps'] for r in rs]
            correlations[arm]=dict(pearson=float(pearsonr(x,y).statistic),spearman=float(spearmanr(x,y).statistic)) if np.std(x)>0 else None
        (root/'correlations.json').write_text(json.dumps(correlations,indent=2))
        md.append('\nJapanese within-policy descriptive correlations (wide verify fraction vs t/s; repeated parents, no causal attribution): `'+json.dumps(correlations)+'`.\n')
        check_frozen(root)
    (root/'measured.md').write_text('\n'.join(md)+'\n')
    print('\n'.join(md[:20]))
if __name__=='__main__':main()
