"""Control: is `mixed` reproducible at all on the fp8 path? (K1 uses atomics.)"""
import os, sys, json, torch
sys.path.insert(0, "/home/user/tools/sglang-h2/python")
from sglang.srt.layers.hc_mix2_triton import hc_norm_mix2, quantize_hc_mix2_weights_fp8
from safetensors.torch import safe_open
MODEL="/home/user/models/RadixArk/Qwen3.8-Flash-Next-NVFP4-mtpft3"
idx=json.load(open(os.path.join(MODEL,"model.safetensors.index.json")))["weight_map"]
def get(n):
    with safe_open(os.path.join(MODEL, idx[n]), framework="pt") as fh:
        return fh.get_tensor(n).cuda().bfloat16().contiguous()
P="model.language_model.layers.6.attn_hyper_connection."
norm_w=get(P+"hc_norm.weight"); wd=get(P+"input_mix_weight_down.weight"); wu=get(P+"input_mix_weight_up.weight")
wd8,sd,wu8,su=quantize_hc_mix2_weights_fp8(wd.data,wu.data)
HC,HS=4,2560; K=HC*HS
for M in (1,4,16):
    g=torch.Generator(device="cuda").manual_seed(1234+M)
    x=(torch.randn(M,K,generator=g,device="cuda")*3.0).bfloat16()
    outs=[hc_norm_mix2(x,norm_w,1e-6,wd8,wu8,HC,HS,None,sd,su)[0].clone() for _ in range(6)]
    torch.cuda.synchronize()
    ndiff=sum(0 if torch.equal(outs[0],o) else 1 for o in outs[1:])
    md=max((outs[0].float()-o.float()).abs().max().item() for o in outs[1:])
    print(f"fp8 M={M}: {ndiff}/5 repeats differ from run 0, maxabs={md:.3e}")
    outs=[hc_norm_mix2(x,norm_w,1e-6,wd,wu,HC,HS)[0].clone() for _ in range(6)]
    torch.cuda.synchronize()
    ndiff=sum(0 if torch.equal(outs[0],o) else 1 for o in outs[1:])
    print(f"bf16 M={M}: {ndiff}/5 repeats differ from run 0")
