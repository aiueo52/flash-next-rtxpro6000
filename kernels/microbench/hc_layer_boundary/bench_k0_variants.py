"""Which part of the shared fold costs 0.5us? Variants of the K0 prologue."""
import os, sys, json, torch, triton, triton.language as tl
sys.path.insert(0,"/home/user/tools/sglang-h2/python")
from sglang.kernels.triton_pdl import pdl_wait, pdl_trigger
from sglang.kernels.ops.elementwise.hc_combine import hc_combine_gate

@triton.jit
def k0(x_ptr, norm_w_ptr, inv_rms_ptr, normed_ptr, t_raw_ptr, inject_w_ptr,
       gate_partials_ptr, apply_resid_ptr, apply_block_ptr, apply_part_ptr,
       apply_shared_ptr, apply_sgate_ptr, num_tasks, zero_span, K, HS, eps,
       HC: tl.constexpr, BLOCK_S: tl.constexpr, ZERO_BLOCK: tl.constexpr,
       FUSE_APPLY: tl.constexpr, FUSE_SHARED: tl.constexpr, PREV_SPLITS: tl.constexpr,
       VARIANT: tl.constexpr):
    pid = tl.program_id(0); m = pid // HC; c = pid % HC
    base = m * K + c * HS
    offs = tl.arange(0, BLOCK_S); mask_s = offs < HS
    if FUSE_APPLY:
        if VARIANT == 0:      # current: gate reduction first
            total = 0.0
            for ps in tl.static_range(PREV_SPLITS):
                total += tl.load(apply_part_ptr + (m * PREV_SPLITS + ps) * HC + c)
            a = 2.0 / (1.0 + tl.exp(-total / HC))
            y = tl.load(apply_block_ptr + m * HS + offs, mask=mask_s, other=0.0).to(tl.float32)
            if FUSE_SHARED:
                g = tl.load(apply_sgate_ptr + m)
                sh = tl.load(apply_shared_ptr + m * HS + offs, mask=mask_s, other=0.0).to(tl.float32)
                y = (y + g * sh).to(x_ptr.dtype.element_ty).to(tl.float32)
            r = tl.load(apply_resid_ptr + base + offs, mask=mask_s, other=0.0).to(tl.float32)
        elif VARIANT == 1:    # hoist every row load above the gate reduction
            y = tl.load(apply_block_ptr + m * HS + offs, mask=mask_s, other=0.0).to(tl.float32)
            if FUSE_SHARED:
                sh = tl.load(apply_shared_ptr + m * HS + offs, mask=mask_s, other=0.0).to(tl.float32)
                g = tl.load(apply_sgate_ptr + m)
            r = tl.load(apply_resid_ptr + base + offs, mask=mask_s, other=0.0).to(tl.float32)
            total = 0.0
            for ps in tl.static_range(PREV_SPLITS):
                total += tl.load(apply_part_ptr + (m * PREV_SPLITS + ps) * HC + c)
            a = 2.0 / (1.0 + tl.exp(-total / HC))
            if FUSE_SHARED:
                y = (y + g * sh).to(x_ptr.dtype.element_ty).to(tl.float32)
        else:                 # VARIANT 2: hoisted, and no bf16 round-trip (diagnostic only)
            y = tl.load(apply_block_ptr + m * HS + offs, mask=mask_s, other=0.0).to(tl.float32)
            if FUSE_SHARED:
                sh = tl.load(apply_shared_ptr + m * HS + offs, mask=mask_s, other=0.0).to(tl.float32)
                g = tl.load(apply_sgate_ptr + m)
            r = tl.load(apply_resid_ptr + base + offs, mask=mask_s, other=0.0).to(tl.float32)
            total = 0.0
            for ps in tl.static_range(PREV_SPLITS):
                total += tl.load(apply_part_ptr + (m * PREV_SPLITS + ps) * HC + c)
            a = 2.0 / (1.0 + tl.exp(-total / HC))
            if FUSE_SHARED:
                y = y + g * sh
        xb = (r + a * y).to(x_ptr.dtype.element_ty)
        tl.store(x_ptr + base + offs, xb, mask=mask_s)
        x = xb.to(tl.float32)
    else:
        x = tl.load(x_ptr + base + offs, mask=mask_s, other=0.0).to(tl.float32)
    w = tl.load(norm_w_ptr + c * HS + offs, mask=mask_s, other=0.0).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / HS + eps)
    tl.store(inv_rms_ptr + pid, inv_rms)
    nrm = (x * inv_rms * (1.0 + w)).to(normed_ptr.dtype.element_ty)
    tl.store(normed_ptr + base + offs, nrm, mask=mask_s)
    nrm32 = nrm.to(tl.float32)
    for cc in tl.static_range(HC):
        gw = tl.load(inject_w_ptr + cc * K + c * HS + offs, mask=mask_s, other=0.0).to(tl.float32)
        tl.store(gate_partials_ptr + pid * HC + cc, tl.sum(nrm32 * gw, axis=0))
    offs_z = tl.arange(0, ZERO_BLOCK)
    for z0 in range(pid * ZERO_BLOCK, zero_span, num_tasks * ZERO_BLOCK):
        idx = z0 + offs_z
        tl.store(t_raw_ptr + idx, 0.0, mask=idx < zero_span)

HC,HS=4,2560; K=HC*HS
def build(M):
    g=torch.Generator(device="cuda").manual_seed(7)
    return dict(x=(torch.randn(M,K,generator=g,device="cuda")*3).bfloat16(),
        applied=torch.empty(M,K,dtype=torch.bfloat16,device="cuda"),
        normed=torch.empty(M,K,dtype=torch.bfloat16,device="cuda"),
        inv=torch.empty(M*HC,dtype=torch.float32,device="cuda"),
        t_raw=torch.zeros(16,320,dtype=torch.float32,device="cuda"),
        inj=torch.randn(HC,K,device="cuda").bfloat16(),
        nw=torch.randn(K,device="cuda").bfloat16(),
        gp=torch.empty(M,HC,HC,dtype=torch.float32,device="cuda"),
        y=(torch.randn(M,HS,generator=g,device="cuda")*.5).bfloat16(),
        sh=(torch.randn(M,HS,generator=g,device="cuda")*.5).bfloat16(),
        sg=torch.rand(M,generator=g,device="cuda",dtype=torch.float32),
        part=hc_combine_gate((torch.randn(M,K,generator=g,device="cuda")).bfloat16(),
                             torch.randn(HC,K,device="cuda").bfloat16(), HC, HS).contiguous())
def launch(b,M,fa,fs,var):
    k0[(M*HC,)](b["applied"] if fa else b["x"], b["nw"], b["inv"], b["normed"], b["t_raw"],
        b["inj"], b["gp"], b["x"], b["y"], b["part"],
        b["sh"] if fs else b["x"], b["sg"] if fs else b["x"], M*HC, M*320, K, HS, 1e-6,
        HC=HC, BLOCK_S=4096, ZERO_BLOCK=128, FUSE_APPLY=fa, FUSE_SHARED=fs,
        PREV_SPLITS=b["part"].shape[1], VARIANT=var, num_warps=8)
def timeit(fn,n=2000):
    for _ in range(50): fn()
    torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(20): fn()
    torch.cuda.synchronize()
    s,e=torch.cuda.Event(True),torch.cuda.Event(True); s.record()
    for _ in range(n//20): g.replay()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)*1000/n
for M in (4,16):
    b=build(M)
    base=timeit(lambda: launch(b,M,False,False,0))
    fa  =timeit(lambda: launch(b,M,True,False,0))
    print(f"M={M:2d} plain={base:.3f}  +apply={fa:.3f} ({fa-base:+.3f})")
    for v in (0,1,2):
        t=timeit(lambda: launch(b,M,True,True,v))
        print(f"     +apply+shared VARIANT{v} = {t:.3f} ({t-base:+.3f} vs plain, {t-fa:+.3f} vs apply-only)")
