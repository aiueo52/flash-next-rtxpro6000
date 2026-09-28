# Glue-E: remaining per-step glue in the Qwen3.8-Flash-Next decode step (spec)

Context: SGLang fork, RTX PRO 6000 Blackwell (sm_120). Decode step = NEXTN speculative decoding
(W16: 15 draft steps + verify of 16 tokens; W4: 3 + 4). Every kernel inside the CUDA graph costs
~1 us of fixed time plus its own duration. Ground truth for what runs per step: the eager+stack
profile `~/tools/flash-next-bench/prof/traces/stack-w16-w16-code-edit/*.gz`, aggregated with
`~/tools/flash-next-bench/prof/stack_agg.py <trace> "step[TARGET_VERIFY bs=1]" 5 45` (and phase
`draft`). Graph-mode traces for timing: `traces/gemv3b-w16-*` and `traces/gemv3b-w4-*` (per-kernel
stats: `prof/gridstats.py <trace> <name-substring>`).

## Items (all flag-gated, default OFF unless stated; keep bit-exactness where claimed)
1. **GDN verify q/k/v copies (108 kernels/step at W16, ~150-200 us).** Vendored FlashInfer wrapper
   `~/tools/sglang-rtxpro6000/.venv/lib/python3.12/site-packages/flashinfer/gdn_kernels/gdn_decode_bf16_wy_output_only.py`
   (~L1961-2300): at `T == T_KERNEL (16)` it does `q/k/v/a/b.contiguous()`; the q/k/v are strided
   column slices of the fused conv output, so three copies per GDN layer x 36 layers. The wrapper
   already supports a strided read (`_STRIDED_QKV`, env `FLASHINFER_GDN_WY_STRIDED_QKV=1`,
   `_qkv_rs` row stride, "fully static descriptor") but only on the native path (T in {4, 8}).
   Task: (a) make the strided read available when `T == T_KERNEL` too (n_valid = 16; same kernel,
   no smem-tail zeroing needed) — produce a **patch file** (`git diff`-style, against the vendored
   file) plus an offline test that imports a patched COPY of the file from a temp path and proves
   bit-identical outputs vs the unpatched wrapper for T=16 and T=4 (B=1, real shapes: H=16 heads,
   HV=48, K=V=128, conv_dim = q/k/v row stride from the caller in
   `python/sglang/srt/layers/attention/linear/gdn_backend.py` / `gdn_flashinfer.py`).
   (b) Say whether enabling `FLASHINFER_GDN_WY_STRIDED_QKV=1` at W4 (T=4, native path) is already
   safe as-is (read the code: why is it OFF by default? any dynamic-shape/compile caveat?).
   **Do NOT edit the file inside the venv in place** — a server that another agent is running
   re-imports it on restart; the supervisor applies the patch during a quiet GPU window.
2. **HC combine 2 -> 1 kernel (96 pairs/step, 331 us).** `python/sglang/srt/layers/hyperconnection.py:455
   combine()` launches `hc_combine_gate` then `hc_combine_apply` (sgl-kernel CUDA ops). Write one
   Triton kernel doing both (read the two kernels' semantics from the sgl-kernel source or the Python
   fallback path; hc_count 4, hidden 2560, rows 1..16). Gate with `SGLANG_HC_COMBINE_FUSED=1`.
   Must be bit-identical or within 1 bf16 ulp with a documented reason; bench with CUPTI medians
   (see methodology below) vs the two-kernel path.
3. **Draft logits copy (14/step at W16, 38 us).** `python/sglang/srt/layers/logits_processor.py:875
   _copy_logits_to_buffer` copies the draft-head output into a shared buffer every draft step. Make
   the W8A16 gemv write straight into that buffer (the `w8a16_gemv`/`bf16_gemv` wrappers in
   `layers/quantization/w8a16_gemv.py` allocate `y`; add an optional `out=` argument and plumb it from
   `eagle_worker_v2.py`'s draft head path) or otherwise remove the copy. Bit-identical.
4. (Analysis only, no code) The router GEMM runs as cuBLAS gemm + splitKreduce (2 kernels x 61/step,
   119 us). `SGLANG_ROUTER_GEMV=1` routes it through `bf16_gemv` (retuned table entry
   `(2, False, 512, 2560)`). The supervisor will A/B it in-server; you just confirm the flag still
   wires correctly after the gemv3 retune (`python/sglang/srt/models/qwen2_moe.py
   _forward_router_logits`) and that `bias is None` holds for this model.

## Bench methodology (mandatory for any timing claim)
CUPTI kernel durations from `torch.profiler` over CUDA-graph replay with a >= 4x L2 rotating working
set: reuse `bench/w8a16v2/harness.py::cupti_kernel_us` from worktree `~/tools/sglang-qsa-ring`.
Report medians; re-measure winners head-to-head in one process. CUDA-event timing is not comparable.

## GPU protocol (strict)
The GPU is currently owned by an extraction job. Do all reading/writing/CPU tests now. Before ANY
GPU use wait until `$SCRATCH/GPU_FREE_FOR_KERNELS`
exists (`until [ -f <path> ]; do sleep 60; done`), then wrap EVERY GPU command in
`flock $SCRATCH/gpu.lock <cmd>`
(another kernel agent shares the GPU under the same lock). Never start an SGLang server. Never edit
`~/tools/sglang-rtxpro6000` (main tree). Env for GPU runs: `. ~/tools/sglang-qsa-ring/bench/w8a16v2/env.sh`
but with `PYTHONPATH` pointing at YOUR worktree.

## Deliverables
Branch `opus/glue-e` in worktree `~/tools/sglang-draft-fuse` (create it from commit 61cf5390bc:
`git -C ~/tools/sglang-draft-fuse checkout -B opus/glue-e 61cf5390bc`), commits per item, tests under
`test/srt/layers/`, benches under `bench/glue_e/`, the vendored-file patch as `bench/glue_e/gdn_wy_T16_strided.patch`,
and a compact final report: per item what changed, correctness evidence, CUPTI before/after, expected
per-step saving at W16 and W4, and exactly how the supervisor enables each item.
