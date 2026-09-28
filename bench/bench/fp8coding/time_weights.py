"""Read existing X2 traces on CPU; reconstruct exclusive-time weights.

No profiling, GPU access, or server launch. BF16 GEMVs have zero FP8 savings.
"""
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
import collections
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'prof'))
import exclusive_time as e

def classify(phase, family, grid):
    if grid == '[32,10,1]': return 'BF16_router'
    if grid == '[3,1,1]': return 'BF16_ba'
    if phase == 'verify':
        if grid in ('[512,1,1]','[256,1,1]'): return 'linear_attn.in'
        if grid == '[40,4,1]': return 'linear_attn.out'
        if grid == '[1940,1,1]': return 'lm_head.target'
        if grid == '[80,6,1]': return 'attn.out'
        if grid == '[416,1,1]': return 'attn.in'
        if grid in ('[80,5,1]','[80,1,1]'): return 'shared_expert.down'
        if grid in ('[80,10,1]','[20,5,1]'): return 'shared_expert.up'
    if phase in ('draft','draft_extend'):
        if grid in ('[1536,1,1]','[384,1,1]'): return 'lm_head.draft'
        if grid == '[416,1,1]': return 'mtp_dense.in'
        if grid in ('[160,6,1]','[80,6,1]'): return 'mtp_dense.out'
        if grid in ('[160,3,1]','[80,5,1]','[80,1,1]'): return 'mtp_dense.down'
        if grid in ('[40,10,1]','[80,10,1]'): return 'mtp_dense.up'
    raise ValueError((phase,family,grid))

def main():
    rows=[]
    for p in sorted((ROOT/'prof/traces').glob('x20152-*/*.gz')):
        r=e.analyze(str(p))
        for k,v in r['excl'].items():
            if 'w8a16' in k[1]:
                rows.append(dict(trace=str(p),phase=k[0],family=k[1],grid=k[2],
                    exclusive_us=v/r['steps'],calls=r['cnt'][k]/r['steps'],subgroup=classify(*k)))
    out={}
    for profile in ('w4','w16'):
        c=collections.Counter()
        for r in rows:
            if '-'+profile+'-' in r['trace']: c[r['subgroup']]+=r['exclusive_us']/2
        out[profile]=dict(c)
    # The printed map's totals do not exactly equal a fresh sweep's family sum.
    # Keep its published weights authoritative; unlisted residual receives no credit.
    published={
        'w4':{'linear_attn.in':893,'linear_attn.out':470,'lm_head.target':387,
              'lm_head.draft':163+74,'attn.out':145,'BF16_ba':88,'unattributed_residual':165},
        'w16':{'linear_attn.in':893,'linear_attn.out':503,'lm_head.target':428,
               'lm_head.draft':1124+49,'mtp_dense.in':247,'mtp_dense.out':173,
               'attn.out':143,'BF16_ba':93,'unattributed_residual':3}}
    assert sum(published['w4'].values())==2385
    assert sum(published['w16'].values())==3656
    p=ROOT/'bench/fp8coding/results/time_weights.json'
    p.write_text(json.dumps(dict(weights=published,trace_weights=out,families=rows),indent=2)+'\n')
    print(json.dumps(dict(published=published,trace=out),indent=2))

if __name__ == '__main__':main()
