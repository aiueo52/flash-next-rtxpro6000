#!/usr/bin/env python3
"""Each server is one long blocking flock subprocess; release between arms."""
import datetime,json,subprocess,sys,time
from pathlib import Path
B=Path(__file__).resolve().parents[1];ROOT=B/'specs/wa6'
current=None
plan={}
def run(arm,split,mode,suffix):
 label='wa6-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+suffix
 path=ROOT/label
 cmd=['flock','-w','28800','/home/user/.gpu.lock','/usr/bin/python3',str(B/'calib/wa6_run_arm.py'),arm,label,split,mode]
 print('QUEUE '+' '.join(cmd),flush=True)
 subprocess.run(cmd,cwd=B,check=True)
 subprocess.run(['/usr/bin/python3',str(B/'calib/wa6_analyze.py'),'arm',str(path)],check=True,cwd=B)
 plan[suffix]=str(path);(ROOT/'sequence-progress.json').write_text(json.dumps(plan,indent=2)+'\n')
 return str(path)
current=Path(run('current','prep','diagnostic','prep-current-corrected'))
refs={f'fixed{s}':run(f'fixed{s}','reference','timed',f'fixed{s}-reference') for s in [4,8,16]}
assert (current/'complete').exists(),'preparation current did not complete; inspect retained artifacts'
if not (current/'summary.json').exists():
 subprocess.run(['/usr/bin/python3',str(B/'calib/wa6_analyze.py'),'arm',str(current)],check=True,cwd=B)
subprocess.run(['/usr/bin/python3',str(B/'calib/wa6_analyze.py'),'opportunity','current='+str(current)]+[k+'='+v for k,v in refs.items()],check=True,cwd=B)
if not json.loads((ROOT/'opportunity.json').read_text())['continue_prior']:
 (ROOT/'sequence-stopped.json').write_text(json.dumps({'reason':'below 3% transition/dwell opportunity','refs':refs})+'\n');sys.exit(0)
run('prior','prep','diagnostic','prep-prior')
final={'current':[],'prior':[],**{k:[v] for k,v in refs.items()}}
for arm,name in [('current','A1'),('prior','P1'),('prior','P2'),('current','A2')]:
 final[arm].append(run(arm,'eval','timed',name))
 (ROOT/'final-arms.json').write_text(json.dumps(final,indent=2)+'\n')
for arm in ['current','prior']:run(arm,'eval','diagnostic',arm+'-diagnostic')
subprocess.run(['/usr/bin/python3',str(B/'calib/wa6_gate.py'),str(ROOT/'final-arms.json')],check=True,cwd=B)
(ROOT/'sequence-complete').touch()
