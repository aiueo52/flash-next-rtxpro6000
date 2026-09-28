#!/usr/bin/env python3
"""Freeze equal-prompt HC unit/gate selection from the BN1 calibration capture."""
import hashlib
import json
import sys
from pathlib import Path
import torch

root=Path(sys.argv[1]); layers={}; stats={}
for path in sorted(root.glob('*.pt')):
    d=torch.load(path,map_location='cpu',weights_only=True)
    samples=d['samples']
    if not torch.isfinite(d['up_norm']).all():
        raise RuntimeError('Nonfinite up norm: '+str(path))
    for key, sample in samples.items():
        for field in ('abs_sum','gate_sum','gate_sq_sum'):
            if not torch.isfinite(sample[field]).all():
                raise RuntimeError(f'Nonfinite calibration: {path} {key} {field}')
    if len(samples)!=32 or any(v['n']<8 for v in samples.values()):
        raise RuntimeError(f'Incomplete calibration {path}: '+str({k:v['n'] for k,v in samples.items()}))
    ma=torch.stack([v['abs_sum']/v['n'] for v in samples.values()]).mean(0)
    gm=torch.stack([v['gate_sum']/v['n'] for v in samples.values()]).mean(0)
    gs=torch.stack([v['gate_sq_sum']/v['n'] for v in samples.values()]).mean(0)
    gv=(gs-gm.square()).clamp(min=0)
    contribution=ma*d['up_norm']
    order=sorted(range(320),key=lambda i:(-float(contribution[i]),i))
    gate_order=sorted(range(len(gm)),key=lambda i:(float(gv[i]),i))[:len(gm)//4]
    layers[d['key']]=dict(shape=[320,d['hc']*d['hs']],keep={str(n):sorted(order[:n]) for n in (256,160,128)},
                          gate_ids=gate_order,gate_means=gm[gate_order].tolist())
    stats[d['key']]=dict(mean_abs_activation=ma.tolist(),up_column_l2=d['up_norm'].tolist(),
                         mean_contribution_l2=contribution.tolist(),gate_mean=gm.tolist(),gate_variance=gv.tolist(),
                         samples={k:dict(n=v['n'],domain=v['domain']) for k,v in samples.items()},
                         rejected_nonfinite=d.get('rejected_nonfinite', {}),
                         retained_contribution={str(n):float(contribution[order[:n]].sum()/contribution.sum()) if contribution.sum()>0 else 1.0 for n in (256,160,128)})
if len(layers)!=100 or sum(k.startswith('mtp.') for k in layers)!=3:
    raise RuntimeError('Missing layers or MTP capture')
output=dict(schema=1,criterion='equal-parent mean |BF16(SiLU(actual K1 / hc))| times FP8-dequantized up-column L2',
            gate_criterion='lowest equal-parent variance of sigmoid gate output; per-gate mean within layer',
            capture_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.pt'))},layers=layers)
(root.parent/'masks.json').write_text(json.dumps(output,indent=1,allow_nan=False)+'\n')
(root.parent/'activation-statistics.json').write_text(json.dumps(stats,indent=1,allow_nan=False)+'\n')
print('Frozen masks for',len(layers),'HC modules',flush=True)
