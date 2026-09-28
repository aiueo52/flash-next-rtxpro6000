#!/usr/bin/env python3
"""Incremental HC1 tables; candidate-minus-production NI orientation."""
import json
import math
import re
import hashlib
from pathlib import Path
import subprocess
import sys
B=Path('/home/user/tools/flash-next-bench');ROOT=B/'specs/hc1';Q=B/'bench/hc1/quality/runs'
REPORT=B/'specs/HC1_BOTTLENECK_SCREEN.md';ARMS=('prod','hc256','hc160','hc128','gate-const');BENCH=('gsm8k','mmlu','humaneval','jcqa')
if not (ROOT.is_dir() and REPORT.exists()):
    sys.exit(f"report.py: needs the HC1 run directory {ROOT} and {REPORT}; HC1 run outputs are not published in "
             "this repository (its tables are summarized in docs/lab-notes/HC1_BOTTLENECK_SCREEN.md)")
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()]
def tango_bounds(b, c, n, z):
    """Tango (1998) score interval for a paired difference theta = (b - c) / n, where b and c are the two
    discordant counts. Valid for small n and few (even zero) discordances, unlike the Wald interval.
    Returns (lower, upper): the theta values where the score statistic equals +z and -z."""
    def score(t):
        w = b + c - t * (2 * n - b + c)
        q = (w + math.sqrt(max(w * w + 8 * n * c * t * (1 - t), 0.0))) / (4 * n)  # constrained MLE of p(c-cell)
        var = n * (2 * q + t - t * t)
        num = b - c - n * t
        if var <= 0: return 0.0 if num == 0 else math.copysign(math.inf, num)
        return num / math.sqrt(var)
    def solve(target, lo, hi):  # score is decreasing in t
        for _ in range(200):
            mid = (lo + hi) / 2
            if score(mid) > target: lo = mid
            else: hi = mid
        return (lo + hi) / 2
    est = (b - c) / n; eps = 1e-12
    lower = -1.0 if score(-1 + eps) <= z else solve(z, -1 + eps, est)
    upper = 1.0 if score(1 - eps) >= -z else solve(-z, est, 1 - eps)
    return lower, upper
def ni(a,b):
    # One-sided 95% Tango score bounds (same method as the corrected analyzer); a Wald interval collapses with
    # few or zero discordant pairs and would overstate non-inferiority.
    x=sum(a[k]['correct'] and not b[k]['correct'] for k in a)
    y=sum(b[k]['correct'] and not a[k]['correct'] for k in a)
    n=len(a);diff=100*(x-y)/n
    lo,hi=(100*v for v in tango_bounds(x,y,n,1.6448536269514722))
    return dict(diff_pp=diff,lower=lo,upper=hi,status='PASS' if lo>-.5 else 'FAIL' if hi<-.5 else 'INCONCLUSIVE',n=n,wins=x,losses=y)
out=['## Results and verdict',''];data=dict(quality={},acceptance={},ni={},needle={},execution={})
for a in ('capture',)+ARMS:
    p=ROOT/a/'complete.json'
    if p.exists():data['execution'][a]=read(p)
    elif (ROOT/a/'invalid.json').exists():data['execution'][a]=read(ROOT/a/'invalid.json')
    else:data['execution'][a]={'status':'running' if (ROOT/a).exists() else 'pending'}
if (ROOT/'masks.json').exists():
    data['mask_manifest_sha256']=hashlib.sha256((ROOT/'masks.json').read_bytes()).hexdigest()
    expected=read(ROOT/'masks.json')['layers']
    data['load_audit']={}
    for a in ARMS[1:]:
        log=ROOT/a/'server.log'
        if not log.exists():continue
        records=re.findall(r'\[HC1\] (\S+) (.*?) shape=(\[.*?\]) drop_units=(\d+) constant_gates=(\d+)',log.read_text())
        seen={key:(arm,json.loads(shape),int(drop),int(gate)) for arm,key,shape,drop,gate in records}
        data['load_audit'][a]=dict(modules=len(seen),expected=len(expected))
        if (ROOT/a/'complete.json').exists():
            if set(seen)!=set(expected):raise RuntimeError('Incomplete transform load: '+a)
            for key,(arm,shape,drop,gate) in seen.items():
                if arm!=a or shape!=expected[key]['shape']:raise RuntimeError('Transform load mismatch')
                if a=='gate-const':
                    if drop!=0 or gate!=shape[1]//4:raise RuntimeError('Gate mask count mismatch')
                elif drop!=320-int(a[2:]) or gate!=0:raise RuntimeError('Unit mask count mismatch')
for a in ARMS:
    if not (ROOT/a/'complete.json').exists():continue
    info=read(ROOT/a/'server-info.json')
    for key,value in {'mem_fraction_static':.9,'max_running_requests':1,'speculative_num_steps':3,'speculative_num_draft_tokens':4,'disable_cuda_graph':False,'max_total_tokens':131072}.items():
        if info[key]!=value:raise RuntimeError(f'Server config mismatch {a}: {key}')
    memory=rows(ROOT/a/'memory.jsonl')
    steady=[x['free_mib'] for x in memory if x['steady']]
    if not steady or min(steady)<4096:raise RuntimeError('Missing/invalid steady memory observations')
    data['execution'][a]['min_steady_free_mib']=min(steady)
    data['execution'][a]['min_startup_free_mib']=min(x['free_mib'] for x in memory if not x['steady'])
out+=['Execution: '+', '.join(a+': '+('complete' if 'finished' in s else s.get('status','invalid')) for a,s in data['execution'].items())+'.','']
if (ROOT/'activation-statistics.json').exists():
    st=read(ROOT/'activation-statistics.json');out += [f'Calibration captured {len(st)} HC modules, {sum(v["n"] for s in st.values() for v in s["samples"].values())} module-token rows across 32 parents.', '', '| Kept rank | Mean retained contribution | Min / max across modules |','|---|---:|---:|']
    for rank in ('256','160','128'):
        v=[s['retained_contribution'][rank] for s in st.values()]
        out.append(f'| {rank} | {100*sum(v)/len(v):.2f}% | {100*min(v):.2f}% / {100*max(v):.2f}% |')
    rejected={k:sum(v.get('rejected_nonfinite',{}).values()) for k,v in st.items() if sum(v.get('rejected_nonfinite',{}).values())}
    out+=['', 'Excluded nonfinite diagnostic rows: '+json.dumps(rejected,sort_keys=True)+'. All exported activation/gate aggregates passed finite-value checks.', '']
for a in ARMS:
    # Only complete arms whose summary agrees with the per-item results are shown; others stay 'pending'.
    if (ROOT/a/'complete.json').exists():
        summary=read(Q/a/'summary.json');data['quality'][a]=summary['bench']
        if summary['bs']!=1 or summary['conc']!=1:raise RuntimeError('Quality concurrency mismatch')
        for b,n in dict(gsm8k=1319,mmlu=1400,humaneval=164,jcqa=1119).items():
            item=data['quality'][a][b]
            if item['n']!=n or item['errors']!=0:raise RuntimeError(f'Incomplete quality {a}: {b}')
            rr=rows(Q/a/(b+'.jsonl'));expected_ids={r['id'] for r in rows(B/'bench/quality/data'/(b+'.jsonl'))}
            if len(rr)!=n or len(expected_ids)!=n or {r['id'] for r in rr}!=expected_ids:raise RuntimeError('Quality corpus IDs mismatch')
            if sum(r['correct'] for r in rr)!=item['correct']:raise RuntimeError('Quality summary count mismatch')
            if abs(item['acc']-item['correct']/n)>1e-9:raise RuntimeError(f'Quality summary acc mismatch {a}: {b}')
            if item['truncated']!=sum(r['finish']=='length' for r in rr) or sum(r['finish']=='error' for r in rr):
                raise RuntimeError(f'Quality summary truncation/error mismatch {a}: {b}')
            if abs(item['mean_gen_tokens']-sum(r['gen_tokens'] for r in rr)/n)>.05+1e-9:raise RuntimeError(f'Quality summary mean tokens mismatch {a}: {b}')
out+=['### Quality (BS=1, W4, full fixed battery)','', '| Benchmark | '+' | '.join(ARMS)+' |','|---|'+'---|'*len(ARMS)]
for b in BENCH:
    cells=[]
    for a in ARMS:
        s=data['quality'].get(a,{}).get(b)
        cells.append(f"{s['correct']}/{s['n']} ({100*s['correct']/s['n']:.2f}%)" if s else 'pending')
    out.append('| '+b+' | '+' | '.join(cells)+' |')
out+=['','### Quality generation integrity','', '| Arm | Benchmark | Mean tokens | Length-capped | Transport errors |','|---|---|---:|---:|---:|']
for a in ARMS:
    for b in BENCH:
        s=data['quality'].get(a,{}).get(b)
        if s:out.append(f"| {a} | {b} | {s['mean_gen_tokens']:.1f} | {s['truncated']} | {s['errors']} |")
data['humaneval_executor_audit']={}
for a in ARMS:
    p=Q/a/'humaneval.jsonl'
    if not p.exists():continue
    rs=rows(p)
    failures=[r['id'] for r in rs if 'bwrap:' in r.get('err','')]
    kinds={}
    for r in rs:
        if r['correct']:continue
        matches=re.findall(r'^([A-Za-z]+(?:Error|Exception))\b',r.get('err',''),re.M)
        kind=matches[-1] if matches else r['status']
        kinds[kind]=kinds.get(kind,0)+1
    data['humaneval_executor_audit'][a]={'rows':len(rs),'sandbox_startup_failures':failures,'failure_kinds':kinds}
    if failures:raise RuntimeError('HumanEval sandbox failure: '+a)
out+=['', 'HumanEval stderr audit: no bwrap startup errors in completed HumanEval rows. Fixed token caps are part of the primary task; cap-conditioned comparisons cannot replace the fixed-battery NI gate.', '', '| Arm | HumanEval failure kinds |','|---|---|']
for a,s in data['humaneval_executor_audit'].items():out.append('| '+a+' | '+', '.join(k+': '+str(v) for k,v in s['failure_kinds'].items())+' |')
out+=['', 'These are failures of the generated/extracted program under the unchanged executor, not an attribution of each failure to model reasoning versus formatting.', '', '### Candidate-minus-prod non-inferiority (0.5 pp)','', '| Candidate | Benchmark | Delta pp | Lower / upper one-sided 95% pp | NI | Discordant wins/losses |','|---|---|---:|---:|---|---:|']
for a in ARMS[1:]:
    data['ni'][a]={}
    for b in BENCH:
        p,q=Q/a/(b+'.jsonl'),Q/'prod'/(b+'.jsonl')
        if not p.exists() or not q.exists():continue
        la,lb=rows(p),rows(q)
        ra={r['id']:r for r in la};rb={r['id']:r for r in lb}
        if len(ra)!=len(la) or len(rb)!=len(lb):raise RuntimeError('Duplicate quality IDs')
        if set(ra)!=set(rb):raise RuntimeError('Unpaired quality IDs')
        s=ni(ra,rb);data['ni'][a][b]=s
        out.append(f"| {a} | {b} | {s['diff_pp']:+.2f} | {s['lower']:+.2f} / {s['upper']:+.2f} | {s['status']} | {s['wins']}/{s['losses']} |")
    if (ROOT/a/'complete.json').exists() and (ROOT/'prod/complete.json').exists():
        try:
            raw=subprocess.check_output([sys.executable,'-B',str(B/'bench/hc1/quality/analyze.py'),'--margin-pp','0.5','--allow-unmarked',a,a,'prod'],text=True,stderr=subprocess.STDOUT)
            (ROOT/a/'corrected-analyzer.md').write_text(raw)
            audited={}
            for line in raw.splitlines():
                cells=[x.strip() for x in line.strip('|').split('|')]
                if len(cells)==8 and cells[0] in BENCH and cells[1]=='prod' and cells[6] in ('PASS','FAIL','INCONCLUSIVE'):
                    s=data['ni'][a][cells[0]]
                    if cells[2:5]!=[f"{s[k]:+.2f}" for k in ('diff_pp','lower','upper')] or cells[6]!=s['status'] or int(cells[7])!=s['n']:
                        raise RuntimeError('Corrected analyzer NI mismatch')
                    audited[cells[0]]=True
            if set(audited)!=set(BENCH):raise RuntimeError('Missing corrected analyzer NI rows')
            data.setdefault('corrected_analyzer_audit',{})[a]=audited
        except subprocess.CalledProcessError as exc:
            (ROOT/a/'corrected-analyzer-error.txt').write_text(exc.output)
            out.append('Existing analyzer failed in an ancillary table; primary paired NI above is computed with its same corrected formula. See corrected-analyzer-error.txt.')
out+=['', 'Only the copied analyzer\'s final BASE-minus-other non-inferiority section is used for the decision. Its earlier legacy significance VERDICT tests degradation in the opposite comparison direction when BASE is a candidate and is not the HC1 verdict.', '', '### W4 BN1 acceptance and decode t/s (informational, one restart)','', '| Arm | Domain | Tokens/verify (pooled) | Decode t/s (pooled) | Equal-parent accept delta | Equal-parent t/s delta | Requests |','|---|---|---:|---:|---:|---:|---:|']
request={a:rows(ROOT/a/'requests.jsonl') for a in ARMS if (ROOT/a/'requests.jsonl').exists()}
for a,rs in request.items():
    if (ROOT/a/'complete.json').exists() and (len(rs)!=32 or len({r['prompt_id'] for r in rs})!=32):raise RuntimeError('Incomplete BN1 set: '+a)
    data['acceptance'][a]={}
    for domain in ('code-edit','prose-en','prose-ja','agent-loop'):
        rr=[r for r in rs if r['workload']==domain]
        if not rr:continue
        if any(r['server']['acceptance']['status']!='ok' for r in rr):raise RuntimeError('Invalid acceptance')
        toks=sum(r['server']['acceptance']['generation_tokens'] for r in rr)
        verifies=sum(r['server']['acceptance']['verify_calls'] for r in rr)
        seconds=sum(r['client']['decode_seconds'] for r in rr)
        ct=sum(r['client']['usage']['completion_tokens'] for r in rr)
        s=dict(acceptance=toks/verifies,decode_tps=ct/seconds,requests=len(rr),tokens=toks,verify_calls=verifies,decode_seconds=seconds)
        base={r['prompt_id']:r for r in request.get('prod',[]) if r['workload']==domain}
        def delta(fn):return 100*math.expm1(sum(math.log(fn(r)/fn(base[r['prompt_id']])) for r in rr)/len(rr))
        if all(r['prompt_id'] in base for r in rr):
            s['acceptance_delta_percent']=delta(lambda r:r['server']['acceptance']['tokens_per_verify'])
            s['tps_delta_percent']=delta(lambda r:r['client']['decode_tps'])
        data['acceptance'][a][domain]=s
        out.append(f"| {a} | {domain} | {s['acceptance']:.3f} | {s['decode_tps']:.1f} | {s.get('acceptance_delta_percent',float('nan')):+.2f}% | {s.get('tps_delta_percent',float('nan')):+.2f}% | {len(rr)} |")
out+=['','### Acceptance-adjusted planning (W4, not a packed measurement)','', '| Candidate | Domain | Required saving at observed acceptance (historical / current effective T) | Linear packed saving budget | Implied optimistic delta at current effective T |','|---|---|---:|---:|---:|']
for a, budget in (('hc256',.156),('hc160',.390),('hc128',.468),('gate-const',.195)):
    for domain,s in data['acceptance'].get(a,{}).items():
        if 'acceptance_delta_percent' not in s:continue
        ratio=1+s['acceptance_delta_percent']/100
        required=8.919*(1-ratio/1.03)
        base=data['acceptance']['prod'][domain]
        effective=1000*base['decode_seconds']/base['verify_calls']
        current_required=effective*(1-ratio/1.03)
        predicted=100*(ratio/(1-budget/effective)-1)
        out.append(f'| {a} | {domain} | {required:.3f} / {current_required:.3f} ms | {budget:.3f} ms | {predicted:+.2f}% |')
out+=['','This is a counterfactual using historical HC-saving budgets, the prod decode-time/verify proxy, and noisy single-restart acceptance; it is not a packed end-to-end speedup measurement. Gate-const uses the deliberately loose K2<=K1+K2 bound.','']
for a in ARMS:
    p=ROOT/a/'needle.log'
    if p.exists():
        data['needle'][a]=[l for l in p.read_text().splitlines() if 'PASS=' in l]
        if (ROOT/a/'complete.json').exists() and len(data['needle'][a])!=6:raise RuntimeError('Incomplete needle: '+a)
out+=['','Needle results: '+', '.join(a+f": {sum('PASS=True' in l for l in ls)}/{len(ls)}" for a,ls in data['needle'].items())+'.','']
out+=['### Current decision','']
for a in ARMS[1:]:
    result=data['ni'][a]
    if any(s['status']=='FAIL' for s in result.values()):
        failed=', '.join(b for b,s in result.items() if s['status']=='FAIL')
        v=f'Untrained variant fails NI on {failed}; recovery would be required. This alone does not prove training can recover quality.'
        if len(result)<4:v+=' Remaining benchmark results are pending.'
    elif len(result)<4:v='Pending full quality battery; no overall NI verdict.'
    elif all(s['status']=='PASS' for s in result.values()):v='All four NI gates pass untrained.'
    else:v='NI remains inconclusive; untrained preservation is not established.'
    if a in ('hc256','gate-const'):v+=' At unchanged acceptance, standalone >=3% is excluded by the ideal time budget before implementation overhead; any claimed acceptance-driven rescue needs a separate replicated measurement.'
    out.append('- '+a+': '+v)
out+=['','Raw masked throughput includes all original dense work (and gate-const correction overhead); it is not a packed speedup. Single-restart acceptance cannot resolve restart variance. No training recovery or packed latency was measured.','']
text=REPORT.read_text().split('## Results and verdict')[0]+'\n'.join(out)
completed=sum((ROOT/a/'complete.json').exists() for a in ARMS)
text=re.sub(r'^Status:.*$', f'Status: valid capture complete; {completed}/5 evaluation arms complete.' if (ROOT/'capture/complete.json').exists() else 'Status: valid capture pending; evaluation not complete.', text, flags=re.M)
REPORT.write_text(text);(ROOT/'results.json').write_text(json.dumps(data,indent=1)+'\n')
print('Incremental HC1 report updated',flush=True)
