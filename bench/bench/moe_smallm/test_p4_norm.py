"""GPU test for K0/K2 inverse residual L2 export, with fused HC apply and PDL."""
import json
from types import SimpleNamespace
import torch
from .runners import ensure_sglang_imports

def main():
    ensure_sglang_imports()
    from sglang.srt.layers.hc_mix2_triton import hc_norm_mix2
    from sglang.srt.layers.moe.prune_contrib import target_context, _context
    results=[]
    hc, hs, rank=4,2560,128
    for rows in (4,16):
        x=torch.randn((rows,hc*hs),dtype=torch.bfloat16,device='cuda')
        norm_w=torch.randn(hs,dtype=torch.bfloat16,device='cuda')*.01
        # Grouped norm weights are one HS vector, indexed per branch in this kernel.
        norm_w=norm_w.repeat(hc)
        wd=torch.randn((rank,hc*hs),dtype=torch.bfloat16,device='cuda')*.005
        wu=torch.randn((hc*hs,rank),dtype=torch.bfloat16,device='cuda')*.005
        block=torch.randn((rows,hs),dtype=torch.bfloat16,device='cuda')*.1
        partials=torch.zeros((rows,hc,hc),dtype=torch.float32,device='cuda')
        batch=SimpleNamespace(batch_size=1,forward_mode=SimpleNamespace(is_target_verify=lambda:True))
        for fused in (False,True):
            kw=dict(apply_inputs=(block,partials) if fused else None)
            plain=hc_norm_mix2(x,norm_w,1e-6,wd,wu,hc,hs,**kw)
            with target_context(4,batch,x):
                got=hc_norm_mix2(x,norm_w,1e-6,wd,wu,hc,hs,**kw)
                inv=_context.get()['inv']
            residual=got[3] if fused else x
            expected=1/torch.linalg.vector_norm(residual.float(),dim=-1)
            torch.testing.assert_close(inv,expected,rtol=2e-6,atol=0)
            # Norm export adds no arithmetic to the served normalized tensor.
            assert torch.equal(plain[1],got[1])
            with target_context(4,batch,x):
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    replay=hc_norm_mix2(x,norm_w,1e-6,wd,wu,hc,hs,**kw)
                inv_graph=_context.get()['inv']
            for scale in (.5,2.):
                x.mul_(scale); graph.replay()
                residual=replay[3] if fused else x
                expected=1/torch.linalg.vector_norm(residual.float(),dim=-1)
                torch.testing.assert_close(inv_graph,expected,rtol=2e-6,atol=0)
            results.append(dict(width=rows,fused_apply=fused,graph_replay=True,rtol=2e-6))
    print('P4_NORM_PASS '+json.dumps(results),flush=True)

if __name__=='__main__':main()
