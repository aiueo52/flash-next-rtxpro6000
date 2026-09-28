"""K0 cost of each prologue variant, isolated (graph-replayed, CUDA-event timed)."""
import os, sys, json, torch
sys.path.insert(0,"/home/user/tools/sglang-h2/python")
import sglang.srt.layers.hc_mix2_triton as T
from sglang.kernels.ops.elementwise.hc_combine import hc_combine_apply, hc_combine_gate
from safetensors.torch import safe_open
MODEL="/home/user/models/RadixArk/Qwen3.8-Flash-Next-NVFP4-mtpft3"
idx=json.load(open(os.path.join(MODEL,"model.safetensors.index.json")))["weight_map"]
def get(n):
    with safe_open(os.path.join(MODEL, idx[n]), framework="pt") as fh:
        return fh.get_tensor(n).cuda().bfloat16().contiguous()
P="model.language_model.layers.6.attn_hyper_connection."
norm_w=get(P+"hc_norm.weight"); inj=get(P+"block_inject_weight.weight")
injp=get("model.language_model.layers.5.mlp_hyper_connection.block_inject_weight.weight")
HC,HS=4,2560; K=HC*HS
cfg=T._DEFAULT_CONFIG
def build(M):
    g=torch.Generator(device="cuda").manual_seed(7)
    return dict(
        x=(torch.randn(M,K,generator=g,device="cuda")*3).bfloat16(),
        applied=torch.empty(M,K,dtype=torch.bfloat16,device="cuda"),
        normed=torch.empty(M,K,dtype=torch.bfloat16,device="cuda"),
        inv=torch.empty(M*HC,dtype=torch.float32,device="cuda"),
        t_raw=torch.zeros(16,320,dtype=torch.float32,device="cuda"),
        gp=torch.empty(M,HC,HC,dtype=torch.float32,device="cuda"),
        y=(torch.randn(M,HS,generator=g,device="cuda")*.5).bfloat16(),
        sh=(torch.randn(M,HS,generator=g,device="cuda")*.5).bfloat16(),
        sg=torch.rand(M,generator=g,device="cuda",dtype=torch.float32),
        part=hc_combine_gate((torch.randn(M,K,generator=g,device="cuda")).bfloat16(), injp.data, HC, HS).contiguous(),
    )
def launch(b,M,fuse_apply,fuse_shared):
    T._hc_branch_stats_kernel[(M*HC,)](
        b["applied"] if fuse_apply else b["x"], norm_w, b["inv"], b["normed"], b["t_raw"],
        inj, b["gp"], b["x"], b["y"], b["part"],
        b["sh"] if fuse_shared else b["x"], b["sg"] if fuse_shared else b["x"],
        M*HC, M*320, K, HS, 1e-6,
        HC=HC, BLOCK_S=cfg.stats_block, ZERO_BLOCK=128, WRITE_NORMED=True,
        SINGLE_TILE=True, FUSE_GATE=True, FUSE_APPLY=fuse_apply,
        FUSE_SHARED=fuse_shared, PREV_SPLITS=b["part"].shape[1],
        num_warps=cfg.stats_warps)
def apply_launch(b,shared):
    hc_combine_apply(b["y"], b["x"], b["part"], HC, HS, out=b["applied"],
                     shared_output=b["sh"] if shared else None,
                     shared_gate=b["sg"] if shared else None)
def timeit(fn, n=2000):
    for _ in range(50): fn()
    torch.cuda.synchronize()
    s,e=torch.cuda.Event(True),torch.cuda.Event(True)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(20): fn()
    torch.cuda.synchronize()
    s.record()
    for _ in range(n//20): g.replay()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)*1000/((n//20)*20)
for M in (4,16):
    b=build(M)
    p =timeit(lambda: launch(b,M,False,False))
    fa=timeit(lambda: launch(b,M,True,False))
    fs=timeit(lambda: launch(b,M,True,True))
    a0=timeit(lambda: apply_launch(b,False))
    a1=timeit(lambda: apply_launch(b,True))
    print(f"M={M:2d}  K0 plain={p:.3f}  K0+apply={fa:.3f} (+{fa-p:.3f})  "
          f"K0+apply+shared={fs:.3f} (+{fs-p:.3f})  |  apply={a0:.3f}  apply+shared={a1:.3f}")
    print(f"      net R7-style = {fa-p-a0:+.3f} us/boundary ; net H2 (shared) = {fs-p-a1:+.3f} us/boundary")
