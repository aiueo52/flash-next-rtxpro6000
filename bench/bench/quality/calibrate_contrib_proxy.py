"""Train-only common proxy threshold, followed by untouched temporal holdout.

Fits one threshold per width to reach the additional-D target in every TRAIN
workload. No holdout norms, counts or outcomes enter this threshold choice.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from contrib_policy_sim import load_capture,ContributionReplay,proxy_score,matched_threshold,selection_recall,baseline_mask

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('paths',nargs='*',type=Path,help='P3 target_verify route captures (.npz; not published in this repository)')
ap.add_argument('--artifact-dir',required=True,type=Path)
ap.add_argument('--report',required=True,type=Path)
a=ap.parse_args()
missing=[str(p) for p in a.paths if not p.is_file()]
if not a.paths or missing:
    ap.error('needs P3 target_verify route captures (.npz), which are not published in this repository'
             +(': missing '+', '.join(missing) if missing else ''))
cs=[load_capture(p) for p in a.paths]
rows=[];cal={}
for width in (4,16):
    sub=[c for c in cs if c.width==width]
    table=np.load(a.artifact_dir/f'proxy-w{width}.npz')['norm_table']
    target=3.7 if width==4 else 6.7
    cuts=[]
    for c in sub:
        train=c.subset(~c.split)
        p=ContributionReplay(train,proxy_score(train,table))
        base_removed=(baseline_mask(p.r)/p.r.mult).sum()/len(train.call)
        cuts.append(matched_threshold(p,'joint',base_removed+target))
    cut=max(cuts)
    cal[str(width)]=dict(threshold=cut,workload_thresholds=cuts,workloads=[c.path.name for c in sub])
    for c in sub:
        test=c.subset(c.split)
        p=ContributionReplay(test,proxy_score(test,table))
        drop=p.drop('joint',cut);s=p.stats(drop)
        baseline=p.stats(p.drop('baseline',.08))
        truth=ContributionReplay(test)
        removed=(drop/p.r.mult).sum()/len(test.call)
        oracle_cut=matched_threshold(truth,'joint',removed)
        oracle_drop=truth.drop('joint',oracle_cut)
        oracle=truth.stats(oracle_drop)
        recall=selection_recall(oracle_drop,drop,p.r.mult)
        rows.append(dict(width=width,workload=c.path.stem.split(f'-w{width}-')[-1],threshold=cut,
            delta_D=baseline['D']-s['D'],step_saving_us=(baseline['D']-s['D'])*69.9,
            baseline_p95=baseline['p95'],p95_ratio=s['p95']/baseline['p95'],
            matched_D_oracle_recall=recall['expert_recall'],oracle_p95_ratio=s['p95']/oracle['p95'],**s))
with (a.artifact_dir/'proxy-train-calibrated.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(a.artifact_dir/'proxy-train-calibration.json').write_text(json.dumps(cal,indent=2)+'\n')
lines=['\n## Train-only common proxy threshold: actual temporal holdout\n',
    'One joint-removal proxy threshold per width is the maximum of the four training-workload target-crossing thresholds. Both the norm table and threshold are fit without holdout data. The table below evaluates the fixed setting on the held-out halves; matched-D true-score recall is evaluation only. This still covers repeats of the same four prompts, not a new-prompt quality gate.\n',
    '| W | workload | proxy threshold | D | delta D | true score mean | p95 | p99 | p95 / baseline | estimated us/step | oracle expert recall at matched D |',
    '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['width']} | {r['workload']} | {r['threshold']:.8f} | {r['D']:.3f} | {r['delta_D']:.3f} | {r['mean']:.6f} | {r['p95']:.6f} | {r['p99']:.6f} | {r['p95_ratio']:.3f} | {r['step_saving_us']:.2f} | {r['matched_D_oracle_recall']:.3f} |")
with a.report.open('a') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps(rows,indent=2))
