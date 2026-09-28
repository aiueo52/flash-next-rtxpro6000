"""CPU synthetic end-to-end selftest; all fixtures/results live in a temporary directory."""
from contrib_policy_sim import analyze,ContributionReplay,load_capture,baseline_mask
from prune_policy_sim import Replay
from pathlib import Path
import numpy as np
import tempfile
with tempfile.TemporaryDirectory(prefix='p3-synthetic-integration-') as tmp:
    root=Path(tmp);paths=[]
    rng=np.random.default_rng(0)
    for width in (4,16):
        for name in ('code-edit','prose-en','prose-ja','agent-loop'):
            n=192;k=10
            ids=np.tile(np.arange(width*k,dtype=np.int32).reshape(1,width,k),(n,1,1))
            w=rng.uniform(.08,.15,size=(n,width,k)).astype(np.float32)
            w/=w.sum(2,keepdims=True)
            norm=rng.lognormal(-1,1,size=(n,width,k)).astype(np.float32)
            h=np.full((n,width),10,dtype=np.float32)
            replay=Replay(ids,w)
            path=root/f'census-synthetic-w{width}-{name}.npz';paths.append(path)
            np.savez(path,schema_version=1,role='target_verify',method='synthetic-test-only',
                call=np.arange(n),T=np.full(n,width),k=np.full(n,k),total_calls=n,
                layer=np.tile(np.arange(48),4),ids=ids.reshape(n,-1),w=w.reshape(n,-1),
                expert_norm=norm.reshape(n,-1),h_norm=h,input_norm=h,
                singleton=(replay.mult==1).reshape(n,-1),p2_drop=baseline_mask(replay).reshape(n,-1),
                reconstruction_rel=np.zeros((n,width)),unchanged=np.ones(n,bool))
            assert len(load_capture(path).call)==n
    rows,selected=analyze(paths,root/'out',root/'report.md')
    assert len(selected)==8,len(selected)
    assert len(rows)>2400,len(rows)
    assert (root/'out/pareto.csv').stat().st_size>100
    assert (root/'out/proxy-evaluation.json').stat().st_size>100
    assert 'matched-D' in (root/'report.md').read_text()
    print('P3 SYNTHETIC EIGHT-CAPTURE END-TO-END PASS',len(rows),'frontier points; temporary fixtures removed')
