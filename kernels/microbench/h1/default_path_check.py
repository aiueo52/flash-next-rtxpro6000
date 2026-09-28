"""Prove the flags-off path is byte-identical to the pre-H1 build.

Run once per build (PP = the python/ dir to import), then torch.equal the two
dumps:

    PP=.../sglang-rtxpro6000/python python default_path_check.py base.pt
    PP=.../sglang-h1/python        python default_path_check.py h1.pt

2026-09-06: 30 outputs (5 shapes x M in {1,4,16} x {plain, split-store}) all
byte-identical between e58cf349d3 and the H1 branch with both flags off.
"""
import os, sys, torch
sys.path.insert(0, os.environ["PP"])
import sglang.srt.layers.quantization.w8a16_gemv as gv

dev = "cuda"
torch.manual_seed(0)
out = {}
for tag, N, K in [("fp8_oproj", 2560, 6144), ("fp8_qkvz", 16384, 2560),
                  ("fp8_down", 2560, 640), ("bf16_ba", 96, 2560),
                  ("bf16_router", 512, 2560)]:
    g = torch.Generator(device=dev).manual_seed(7)
    if tag.startswith("fp8"):
        wb = (torch.randn(N, K, device=dev, generator=g) * 0.02).float()
        sc = (wb.abs().amax(dim=1).clamp(min=1e-8) / 448.0).contiguous()
        w = (wb / sc[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous().t().t()
        f = lambda x: gv.w8a16_gemv(x, w, sc.reshape(N, 1))
    else:
        w = (torch.randn(N, K, device=dev, generator=g) * 0.02).to(torch.bfloat16)
        f = lambda x: gv.bf16_gemv(x, w)
    for M in (1, 4, 16):
        x = (torch.randn(M, K, device=dev, generator=g) * 0.1).to(torch.bfloat16)
        out[f"{tag}_M{M}"] = f(x).cpu()
    # split store too
    sp = N // 2
    for M in (1, 4, 16):
        x = (torch.randn(M, K, device=dev, generator=g) * 0.1).to(torch.bfloat16)
        o = torch.empty(M, sp, dtype=torch.bfloat16, device=dev)
        o2 = torch.empty(M, N - sp, dtype=torch.bfloat16, device=dev)
        if tag.startswith("fp8"):
            gv.w8a16_gemv(x, w, sc.reshape(N, 1), out=o, out2=o2, split_n=sp)
        else:
            gv.bf16_gemv(x, w, out=o, out2=o2, split_n=sp)
        out[f"{tag}_split_M{M}"] = (o.cpu(), o2.cpu())
torch.save(out, sys.argv[1])
print("saved", sys.argv[1], len(out), "outputs; BA_SPLIT_GRID=", getattr(gv, "BA_SPLIT_GRID", "n/a"))
