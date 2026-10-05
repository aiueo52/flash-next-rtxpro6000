#!/bin/bash
# smoke_stack.sh -- one `on` arm without the BN1 grid under the GPU lock (STACK_ENV / STACK_FI / SERVER_ENV
# pass through; SMOKE_LABEL names the arm and log, default smoke-stack), then the pass check: no server errors
# and evidence for each change that is on.
cd ~/tools/flash-next-bench
L=${SMOKE_LABEL:-smoke-stack}; S=runs/stack/$L.log; mkdir -p runs/stack
flock -w 28800 ~/.gpu.lock systemd-run --user --scope -q -p MemoryMax=110G -p MemorySwapMax=0 \
    env OUTDIR=runs/stack bash bench/stack/arm_stack.sh $L on none > $S 2>&1
echo "[smoke-stack] done $(date +%T)"
grep -E "start|server up|mapped|error lines|enabled lines|fallback|rt1 _router|dg1 gemv_k1|fg1 main stream|sparse-verify|top-k kernels|died|not ready" $S | cut -c1-240
ok=1
need() { grep -qE "$1" $S || { echo "[smoke-stack] missing: $1"; ok=0; }; }
need "error lines in server log: 0$"
E=$(grep -m1 "\] start" $S)
case $E in *SGLANG_ROUTER_FAST_TOPK=1*) need "rt1 _router_softmax_fast32_kernel: n=[1-9]"; need "rt1 _router_triton_kernel: n=0 " ;; esac
case $E in *SGLANG_OPT_DRAFT_MOE_GEMV=1*) need "draft MoE GEMV enabled lines: [1-9]"; need "GEMV fallback warnings: 0$"; need "dg1 gemv_k1 +count +[1-9]" ;; esac
case $E in *SGLANG_OPT_GDN_FRONT_OVERLAP=1*) need "gates on the alt stream [1-9]" ;; esac
case $E in *SGLANG_OPT_SPEC_SPARSE_VERIFY=1*) need "sv1 code-edit sparse-verify kernels in trace: .*_sparse_target_probs_kernel" ;; esac
case $E in *SGLANG_OPT_SPEC_SPARSE_TOPK=1*) need "sv2 code-edit top-k kernels in trace: .*flashinfer::sampling::" ;; esac
case $E in *FLASHINFER_WORKSPACE_BASE=*) need "mapped: .*/sglang-${STACK_FI:-rq2u2h}/" ;; esac
[ $ok = 1 ] && echo "[smoke-stack] CLEAN" || echo "[smoke-stack] NOT CLEAN"
