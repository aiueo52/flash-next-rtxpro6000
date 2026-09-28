"""H2 numerics: the layer->layer apply moved into the next mix's K0.

Compares, on real HC weights pulled out of the checkpoint shards:
  reference : hc_combine_apply(block, resid, partials[, shared, gate]) -> hc_norm_mix2
  fused     : hc_norm_mix2(resid, ..., apply_inputs=(block, partials[, shared, gate]))
"""
import os, sys, torch

sys.path.insert(0, "/home/user/tools/sglang-h2/python")
from sglang.srt.layers.hc_mix2_triton import hc_norm_mix2, quantize_hc_mix2_weights_fp8
from sglang.kernels.ops.elementwise.hc_combine import hc_combine_apply, hc_combine_gate

torch.manual_seed(0)
dev = "cuda"
HC, HS, LR = 4, 2560, 320
K = HC * HS

W = {}
import glob, json
from safetensors.torch import safe_open
MODEL = "/home/user/models/RadixArk/Qwen3.8-Flash-Next-NVFP4-mtpft3"
want = {
    "model.language_model.layers.5.mlp_hyper_connection.block_inject_weight.weight": "inject_prev",
    "model.language_model.layers.6.attn_hyper_connection.hc_norm.weight": "norm_w",
    "model.language_model.layers.6.attn_hyper_connection.input_mix_weight_down.weight": "w_down",
    "model.language_model.layers.6.attn_hyper_connection.input_mix_weight_up.weight": "w_up",
    "model.language_model.layers.6.attn_hyper_connection.block_inject_weight.weight": "inject_next",
}
idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
for name, key in want.items():
    f = idx[name]
    with safe_open(os.path.join(MODEL, f), framework="pt") as fh:
        W[key] = fh.get_tensor(name).to(dev).to(torch.bfloat16).contiguous()
for k, v in W.items():
    print(k, tuple(v.shape), v.dtype)

norm_w = W["norm_w"]
w_down_bf16, w_up_bf16 = W["w_down"], W["w_up"]
wd8, sd, wu8, su = quantize_hc_mix2_weights_fp8(w_down_bf16.data, w_up_bf16.data)
eps = 1e-6

def run(M, fp8, use_shared, gate_early_next):
    g = torch.Generator(device=dev).manual_seed(1234 + M)
    resid = (torch.randn(M, K, generator=g, device=dev) * 3.0).bfloat16()
    normed_prev = (torch.randn(M, K, generator=g, device=dev)).bfloat16()
    block = (torch.randn(M, HS, generator=g, device=dev) * 0.5).bfloat16()
    shared = (torch.randn(M, HS, generator=g, device=dev) * 0.5).bfloat16()
    sgate = torch.rand(M, generator=g, device=dev, dtype=torch.float32)
    partials = hc_combine_gate(normed_prev, W["inject_prev"].data, HC, HS)
    wd, wu, s_d, s_u = (wd8, wu8, sd, su) if fp8 else (w_down_bf16, w_up_bf16, None, None)
    inj = W["inject_next"].data if gate_early_next else None

    # reference: standalone apply, then the mix
    applied_ref = hc_combine_apply(
        block, resid, partials, HC, HS,
        shared_output=shared if use_shared else None,
        shared_gate=sgate if use_shared else None,
    )
    ref = hc_norm_mix2(applied_ref, norm_w, eps, wd, wu, HC, HS, None, s_d, s_u,
                       inject_weight=inj)

    ai = (block, partials) if not use_shared else (block, partials, shared, sgate)
    fus = hc_norm_mix2(resid, norm_w, eps, wd, wu, HC, HS, None, s_d, s_u,
                       inject_weight=inj, apply_inputs=ai)
    applied_fus = fus[-1]
    torch.cuda.synchronize()

    def cmp(name, a, b):
        eq = torch.equal(a, b)
        if eq:
            return f"{name}: EQUAL"
        af, bf = a.float(), b.float()
        d = (af - bf).abs()
        n = int((a != b).sum())
        # ulp distance for bf16
        return (f"{name}: DIFF n={n}/{a.numel()} maxabs={d.max().item():.3e} "
                f"maxrel={(d/bf.abs().clamp_min(1e-30)).max().item():.3e}")

    out = [cmp("applied", applied_ref, applied_fus),
           cmp("mixed  ", ref[0], fus[0]),
           cmp("normed ", ref[1], fus[1])]
    if gate_early_next:
        out.append(cmp("gatepar", ref[2], fus[2]))
    return out

for fp8 in (False, True):
    for use_shared in (False, True):
        for M in (1, 4, 16):
            tag = f"M={M:<3} fp8={int(fp8)} shared={int(use_shared)}"
            for line in run(M, fp8, use_shared, gate_early_next=True):
                print(f"{tag} | {line}")
print("done")
