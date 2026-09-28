"""CPU-only Triton front-end compile of the NVFP4 GEMV, for every tile the planner emits.

Catches the class of failure that has already cost three GPU-lock windows (unsupported
globals, constexpr-branch type errors) without touching the GPU: `ASTSource.make_ir`
runs the Python->TTIR front end, which is where all of them surfaced.
"""
import sys, itertools, traceback
sys.path.insert(0, "/home/user/tools/sglang-n1/python")
import torch, triton
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource, make_backend

import sglang.srt.layers.quantization.w4a16_nvfp4_gemv as K

target = GPUTarget("cuda", 120, 32)
backend = make_backend(target)

# 8 pointers, then M,N,K + 7 strides = ints
sig = {}
names = ["x_ptr","q_ptr","s_ptr","g_ptr","y_ptr","y2_ptr","ws_ptr","cnt_ptr"]
for n in names: sig[n] = "*bf16" if n in ("x_ptr","y_ptr","y2_ptr") else ("*fp32" if n in("g_ptr","ws_ptr") else ("*i32" if n=="cnt_ptr" else ("*fp8e4nv" if n=="s_ptr" else "*u8")))
for n in ["M","N","K","stride_xm","stride_xk","stride_qn","stride_sn","stride_ym","stride_yn","stride_y2m"]:
    sig[n] = "i32"

ok = bad = 0
for gather, M, (bn, bk, sp, use_dot, warps, stages) in itertools.product(
    (False, True), (1, 4, 16),
    ((32,128,1,False,4,3), (64,256,1,True,8,3), (16,128,2,True,4,3), (128,256,1,True,8,3),
     (32,512,1,False,4,3), (64,512,1,True,8,3)),
):
    ud = use_dot or M > 1
    cst = {"SPLIT_N":0, "PER_CHANNEL":False, "BLOCK_N":bn, "BLOCK_K":bk,
           "M_PAD":16 if ud else 1, "SPLITS":sp, "USE_DOT":ud,
           "SCALE_GATHER":gather, "USE_PDL":False}
    tag = f"gather={int(gather)} M={M:2d} BN={bn:3d} BK={bk:3d} sp={sp} dot={int(ud)} w={warps} st={stages}"
    try:
        src = ASTSource(fn=K._w4a16_nvfp4_gemv_kernel, signature=sig, constexprs=cst)
        opts = backend.parse_options({"num_warps": warps, "num_stages": stages})
        import triton.compiler.compiler as C
        C.compile(src, target=target, options=opts.__dict__)
        ok += 1
    except Exception as e:
        bad += 1
        msg = [l for l in str(e).splitlines() if l.strip()]
        print(f"FAIL {tag}\n     {type(e).__name__}: {msg[-1][:170] if msg else ''}")
print(f"\n{ok} compiled, {bad} failed")
