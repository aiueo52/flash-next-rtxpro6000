"""P3 GPU preflight on real NVFP4 weights and recorded unpruned routing."""
import argparse
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from moe_smallm.bench_moe import _load_weights
from moe_smallm import model_shapes as MS
from moe_smallm.runners import ensure_sglang_imports, _duck_dispatch_output, _runner_config, CallInputs, load_autotune_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    from moe_smallm.harness import require_idle_gpu
    require_idle_gpu()
    ensure_sglang_imports()
    from sglang.srt.layers.moe.contrib_recorder import unmix
    from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import FlashInferCutlassMoeQuantInfo, _run_flashinfer_cutlass_impl
    ckpt = MS.DEFAULT_CKPT
    shape = MS.read_checkpoint_shape(ckpt)
    lw = _load_weights(SimpleNamespace(ckpt=ckpt, layer=4, synthetic=False, num_experts=None), shape, 'cuda')
    qi = FlashInferCutlassMoeQuantInfo(quant_type='fp4', w13_weight=lw.w13_weight, w2_weight=lw.w2_weight,
        output_dtype=torch.bfloat16, quant_scales=[lw.w13_input_scale_quant,lw.w13_blockscale_swizzled,lw.g1_alphas,
        lw.w2_input_scale_quant,lw.w2_blockscale_swizzled,lw.g2_alphas], apply_routed_scaling_factor=False)
    load_autotune_cache('auto')
    results=[]
    root=Path(__file__).resolve().parents[2]
    for width in (4,16):
        z=np.load(root/f'runs/census-w{width}-code-edit.npz')
        rows=np.flatnonzero(z['T']==width)[::97][:3]
        for seed,index in enumerate(rows):
            torch.manual_seed(seed)
            ids=torch.from_numpy(z['ids'][index,:width*10].astype(np.int32).reshape(width,10)).cuda()
            w=torch.from_numpy(z['w'][index,:width*10].reshape(width,10).copy()).cuda()
            x=torch.randn(width,shape.hidden,device='cuda',dtype=torch.bfloat16)*0.5
            out=torch.empty_like(x)
            ci=CallInputs(x,ids,w,out)
            kw=dict(dispatch_output=_duck_dispatch_output(ci),quant_info=qi,runner_config=_runner_config(lw),output=out)
            fn=_run_flashinfer_cutlass_impl
            original=fn(**kw).clone()
            fi=unmix(fn,kw)
            torch.cuda.synchronize()
            # Output storage and routing are untouched by the extra calls.
            assert torch.equal(out,original)
            np.testing.assert_array_equal(ids.cpu(),z['ids'][index,:width*10].reshape(width,10))
            recon=(fi.float()*w[...,None]).sum(1)
            rel=((recon-original.float()).norm(dim=-1)/original.float().norm(dim=-1)).max().item()
            assert rel < .025,(width,seed,rel)
            # Independent route-isolation reference: k=1, one row. Changes
            # scheduling but has identical quantized expert math.
            oneci=CallInputs(x[:1],ids[:1,3:4].contiguous(),torch.ones(1,1,device='cuda'),torch.empty_like(x[:1]))
            ref=fn(dispatch_output=_duck_dispatch_output(oneci),quant_info=qi,runner_config=_runner_config(lw),output=oneci.output)
            single_rel=(ref.float()-fi[:1,3].float()).norm()/ref.float().norm()
            assert single_rel.item() < .025,single_rel.item()
            # Replay with fresh input values; no Python callback at graph replay.
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                gf=unmix(fn,kw)
                gn=gf.float().norm(dim=-1)
            x.mul_(.8)
            g.replay()
            eager=unmix(fn,kw).float().norm(dim=-1)
            torch.cuda.synchronize()
            graph_rel=((gn-eager).abs()/eager.clamp_min(1e-12)).max().item()
            assert graph_rel < .01,graph_rel
            results.append(dict(width=width,seed=seed,reconstruction_max_rel=rel,single_route_rel=single_rel.item(),graph_norm_max_rel=graph_rel,output_unchanged=True))
    # Exercise the actual ring/observe path and graph-time writes, not only
    # the route-unmixing helper. Use small temporary buffers and files.
    from sglang.srt.layers.moe import contrib_recorder as cr
    with tempfile.TemporaryDirectory(prefix='p3-recorder-preflight-') as tmp:
        os.environ['SGLANG_MOE_CONTRIB_LOG']=tmp+'/recorder'
        os.environ['SGLANG_MOE_CONTRIB_RING']='4'
        residual=torch.cat([x,x],dim=1)
        cr._CONTEXT=(4,x,residual)
        cr.observe(fn,kw)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            recorded=cr.observe(fn,kw)
        cr._STATE['counter'].zero_()
        cr._STATE['buffers']['call'].fill_(-1)
        x.mul_(1.1)
        residual.mul_(1.2)
        graph.replay()
        torch.cuda.synchronize()
        summary=cr._dump(tmp+'/capture.npz')
        assert summary['stored_calls']==1 and summary['unchanged'],summary
        zz=np.load(tmp+'/capture.npz')
        np.testing.assert_allclose(zz['h_norm'][0],residual.float().norm(dim=-1).cpu(),rtol=1e-6)
        cr._CONTEXT=None
        assert zz['layer'].tolist()==[4]
        assert zz['expert_norm'].shape==(1,160)
        assert np.isfinite(zz['expert_norm']).all()
        results.append(dict(recorder_graph_pass=True,**summary))
    Path(args.output).write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))
    print('P3 GPU PREFLIGHT PASS')

if __name__=='__main__':
    main()
