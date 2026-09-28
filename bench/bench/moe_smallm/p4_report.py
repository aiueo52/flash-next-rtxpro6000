"""P4 repeated speed gate; missing cells never pass. No legacy kernel clipping."""
import argparse,gzip,importlib.util,json,math,statistics as st
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('exclusive',ROOT/'prof/exclusive_time.py')
exclusive=importlib.util.module_from_spec(spec);spec.loader.exec_module(exclusive)
def trace_metrics(path):
 ev=json.load(gzip.open(path,'rt'))['traceEvents']
 starts=sorted(e['ts'] for e in ev if e.get('ph')=='X' and e.get('cat')=='user_annotation' and e['name']=='draft')
 steps=sorted((b-a)/1000 for a,b in zip(starts,starts[1:]));assert len(steps)>=10,(path,len(steps))
 trim=len(steps)//10;res=exclusive.analyze(str(path))
 out=dict(trimmed_step_ms=st.mean(steps[trim:len(steps)-trim]),step_median_ms=st.median(steps),steps=len(steps))
 for name in ('cutlass_moe_grouped_gemm1','cutlass_moe_grouped_gemm2'):
  durs=[d for key,values in res['durs'].items() if key[0]=='verify' and key[1]==name for d in values]
  assert durs,(path,name);out[name+'_us']=st.median(durs)
 for name in ('trtllm::fusedBuildExpertMapsSortFirstTokenAndStridesKernel', '_hc_branch_stats_kernel', '_hc_up_kernel'):
  durs=[d for key,values in res['durs'].items() if key[0]=='verify' and key[1]==name for d in values]
  assert durs,(path,name)
  out[name+'_us']=st.median(durs)
  out[name+'_exclusive_us_per_step']=sum(v for key,v in res['excl'].items() if key[0]=='verify' and key[1]==name)/res['steps']
 return out
def exact_accept(record):
 delta=record['server']['metrics']['delta']
 calls=sum(v for k,v in delta.items() if k.startswith('sglang:spec_verify_calls_total{'))
 tokens=sum(v for k,v in delta.items() if k.startswith('sglang:generation_tokens_total{'))
 assert calls>0 and tokens==record['client']['usage']['completion_tokens'], (calls,tokens,record['client']['usage'])
 return tokens/calls
GATE_CELLS={(w,wl) for w in (4,16) for wl in ('code-edit','prose-en','prose-ja','agent-loop')}
def speed_gate(comparisons):
 """PASS needs every (width, workload) cell, |acceptance change| <= 1 % in all of them, and a measured
 step_faster_pct >= 3 for code-edit and prose-en at each width. A missing cell or a missing step metric is
 'not measured' and never passes; an empty comparison list never passes."""
 if {(c['width'],c['workload']) for c in comparisons}!=GATE_CELLS:return False
 for c in comparisons:
  if not abs(c['acceptance_change_pct'])<=1:return False
  if c['workload'] in ('code-edit','prose-en') and not c.get('step_faster_pct',-math.inf)>=3:return False
 return True
def med(v):
 assert v and all(x is not None and math.isfinite(x) for x in v),v
 return st.median(v)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('label');a=ap.parse_args();records=[];traces=[]
 missing=[str(ROOT/'runs/p4'/f'{a.label}-w{w}-{arm}') for w in (4,16) for arm in ('control','contrib','control2')
          if not (ROOT/'runs/p4'/f'{a.label}-w{w}-{arm}'/'COMPLETE').exists()]
 if missing:raise SystemExit('p4_report.py: incomplete or missing P4 arms (P4 run directories are not published in this repository): '+', '.join(missing))
 for width in (4,16):
  for arm in ('control','contrib','control2'):
   directory=ROOT/'runs/p4'/f'{a.label}-w{width}-{arm}';assert (directory/'COMPLETE').exists(),directory
   for workload in ('code-edit','prose-en','prose-ja','agent-loop'):
    fn=[];accepts=[]
    for repeat in range(3):
     fn.extend(json.loads(line) for line in (directory/f'timing/{workload}-r{repeat}.jsonl').read_text().splitlines())
     accepts.append(json.loads((directory/f'timing/{workload}-r{repeat}.accept.json').read_text())['mean'])
    d=json.loads((directory/f'census/{workload}.D.json').read_text())
    assert set(d['by_width'])=={str(width)} and d['calls']%48==0, d
    rec=dict(width=width,arm=arm,workload=workload,tps=med([v['client']['decode_tps'] for v in fn]),acceptance=med([exact_accept(v) for v in fn]),acceptance_log_mean=med(accepts),D=d['by_width'][str(width)]['D'],D_calls=d['by_width'][str(width)]['calls'])
    if workload in ('code-edit','prose-en'):
     rows=[]
     for repeat in range(3):
      path=next((directory/f'timing/trace-{workload}-r{repeat}').glob('*.gz'));row=trace_metrics(path);rows.append(row);traces.append(dict(path=str(path),**row))
     for key in rows[0]:
      if key!='steps':rec[key]=med([r[key] for r in rows])
    records.append(rec)
 comparisons=[]
 for width in (4,16):
  for workload in ('code-edit','prose-en','prose-ja','agent-loop'):
   group={r['arm']:r for r in records if r['width']==width and r['workload']==workload};candidate=group['contrib']
   base={k:(group['control'][k]+group['control2'][k])/2 for k in candidate if k not in ('width','arm','workload','D_calls')}
   c=dict(width=width,workload=workload,acceptance_change_pct=100*(candidate['acceptance']/base['acceptance']-1),tps_change_pct=100*(candidate['tps']/base['tps']-1),delta_D=base['D']-candidate['D'])
   if 'trimmed_step_ms' in base:c['step_faster_pct']=100*(1-candidate['trimmed_step_ms']/base['trimmed_step_ms'])
   comparisons.append(c)
 speed=speed_gate(comparisons)
 result=dict(label=a.label,records=records,comparisons=comparisons,traces=traces,speed_gate=speed,
  step_definition='10% trimmed mean of full draft-to-next-draft wall intervals; median across 3 traces; not legacy per-kernel clipping',control_definition='mean of the two control arm medians',gate_definition='>=3% faster for code-edit and prose-en at each width, acceptance within +/-1% for all four workloads at each width')
 (ROOT/'runs/p4'/f'{a.label}-speed.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['\n## In-server repeated speed results\n',f'Label `{a.label}`. '+result['step_definition']+'. Controls: '+result['control_definition']+'.\n','| W | workload | arm | trimmed step ms | GEMM1 us/call | GEMM2 us/call | measured D | accept length | t/s |','|---|---|---|---:|---:|---:|---:|---:|---:|']
 for r in records:
  cells=[f'{r[k]:.3f}' if k in r else '—' for k in ('trimmed_step_ms','cutlass_moe_grouped_gemm1_us','cutlass_moe_grouped_gemm2_us','D','acceptance','tps')]
  lines.append(f"| {r['width']} | {r['workload']} | {r['arm']} | "+' | '.join(cells)+' |')
 lines+=['\n| W | workload | step faster % | acceptance change % | t/s change % | additional D removed |','|---|---|---:|---:|---:|---:|']
 for c in comparisons:
  cells=[f'{c[k]:+.3f}' if k in c else '—' for k in ('step_faster_pct','acceptance_change_pct','tps_change_pct','delta_D')]
  lines.append(f"| {c['width']} | {c['workload']} | "+' | '.join(cells)+' |')
 lines.append('\nSpeed gate: **'+('PASS' if speed else 'FAIL')+'**. '+result['gate_definition']+'.\n')
 with (ROOT/'specs/P4_CONTRIB_PRUNE_SHIP.md').open('a') as f:f.write('\n'.join(lines))
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
