# Draft-loop glue: the bookkeeping and elementwise kernels of the NEXTN draft step (spec)

Context: SGLang fork `~/tools/sglang-rtxpro6000` (branch `codex/perf-v1`), Qwen3.8-Flash-Next
NVFP4 on RTX PRO 6000 Blackwell (sm_120), NEXTN topk=1, bs=1. Successor to
`OVERHEAD_REPORT.md` items **F3** (hoist per-iteration QSA/MTP index bookkeeping out of the
draft body) and **F5** (fuse the ~30 tail glue kernels). Written 2026-09-05, CPU-only: every
number below is a median over the 19-20 phase occurrences of an existing chrome trace, no GPU
was used and no server was started.

Traces used (all `code-edit`, the fastest-moving workload):

| config | trace | S | T | draft busy | verify busy | extend busy | step wall |
|---|---|--:|--:|--:|--:|--:|--:|
| W16 | `prof/traces/prof-base32k-w16-code-edit/` | 15 | 16 | 4.44 ms | 15.10 | 0.55 | 20.47 ms |
| W8  | `prof/traces/prof-s7-w4-code-edit/` | 7 | 8 | 2.10 ms | 11.80 | 0.54 | 14.90 ms |
| W4  | `prof/traces/prof-base32k-w4-code-edit/` | 3 | 4 | 0.58 ms | 10.56 | 0.45 | 11.95 ms |

Reader scripts: `prof/trimmed_step.py` (per-phase `busy_ms`), `prof/gridstats.py`
(per-(name, grid) medians), `prof/stack_agg.py` (python call sites, eager trace
`traces/stack-w16-w16-code-edit/`). The tables below were produced with a per-phase
launch-order variant of `gridstats.py`; the raw dumps are reproducible from the traces alone.

---

## 0. Shape of one draft phase

At W16 the `draft` phase holds **787 kernels**. They decompose exactly:

```
  9 kernels   prologue          (cache-loc assign, registry copies, index fills)     ~13 us
 28 kernels   QSA metadata      14 x (_qsa_graph_layout + _qsa_graph_row_metadata)   155 us   <-- eager, NOT in the graph
  2 kernels   pre-loop          hot_token_id[topk_index], one extra memcpy32_post      3 us
744 kernels   14 x 53-kernel    the captured draft body, one CUDA graph replay      4230 us
  4 kernels   epilogue          cat + tree-mask fill + build_tree_efficient           27 us
```

Two facts that change the F3 framing:

1. **The 28 QSA metadata kernels are NOT inside the CUDA graph.** They are eager Triton
   launches issued by `QwenSparseMultiStepDraftBackend.init_forward_metadata_out_graph`
   (`qwen_sparse_attn_backend.py:1945-1982`), which loops over the `speculative_num_steps - 1`
   per-step `attn_backends` and calls `_replay_cuda_graph_metadata` for each. This matches the
   host counts in `OVERHEAD_REPORT.md` §4: `cuLaunchKernelEx` per step is 18 (W4) / 26 (W8) /
   43 (W16), i.e. **exactly +2 per extra draft step**. So this bucket can be changed without
   re-capturing anything, and its 155 us at W16 sit on the GPU critical path *before* the
   single `cudaGraphLaunch` — pure serial dead time at the head of the phase.
2. The remaining glue is inside the graph body and has to be attacked with kernel work.

Group totals for the draft phase (sum of `median x count`; the phase's kernel-interval union is
4.44 ms at W16 and 0.58 ms at W4, so these sums are ~5 % high from cross-stream overlap):

| group | W16 us/step | W16 kernels/step | W4 us/step | W4 kernels/step |
|---|--:|--:|--:|--:|
| gemv (dense projections) | 1821.0 | 126 | 260.6 | 18 |
| MoE (experts + glue) | 939.2 | 154 | 127.4 | 20 |
| attention / QSA index | 569.9 | 112 | 88.9 | 16 |
| HC (hyperconnection) | 516.3 | 182 | 73.3 | 26 |
| **bookkeeping (index / arange / copies / fills)** | **204.4** | **126** | **42.7** | **30** |
| norms / elementwise glue | 157.4 | 85 | 24.0 | 13 |
| (memcpy DtoD, unclassified) | 2.1 | 2 | 1.9 | 2 |
| total | 4210.3 | 787 | 618.8 | 125 |

Bookkeeping + norm/elementwise glue = **362 us/step at W16** (1.8 % of the 20.47 ms step) and
**67 us at W4** (0.6 %). The QSA index bookkeeping counted under "attention/QSA" adds another
155 us at W16. That is the honest ceiling for F3+F5 inside the draft phase — but the same
kernels also run in `verify` and `draft_extend`, and two of them (DG-1, DG-3) are worth more
outside the draft loop than inside it.

---

## 1. Every kernel of one W16 draft phase, in launch order

`n/step` is per **decode** step; the body kernels run 14x (once per inner draft iteration,
`n_inner = speculative_num_steps - 1`, `eagle_worker_v2.py:643`). Divide by 14 for per-draft-step
cost. `med us` is the median duration of that (name, grid) pair over the whole trace. W4 column
is the same (name, grid) pair in `prof-base32k-w4-code-edit` (2 inner iterations); `-` means the
shape does not occur at W4.

Rows 0-8 are the prologue, 9-10 the eager QSA metadata, 11-12 the pre-loop pair, 12-53 the
captured body (54 launches per iteration), 54-57 the epilogue.

| # | group | kernel | grid | n/step (W16) | med us | W16 us/step | W4 us/step | python call site |
|--:|---|---|---|--:|--:|--:|--:|---|
| 0 | book | `assign_draft_cache_locs_contiguous` | `[1,1,1]` | 1 | 1.46 | 1.5 | 1.4 | eagle_worker_common.py:245 (kernels/ops/speculative/cache_locs.py) |
| 1 | book | `Memcpy DtoD` | `-` | 2 | 1.04 | 2.1 | 1.9 | eagle_draft_cuda_graph_runner.py:588-603 hidden_states/draft_probs copy_ |
| 2 | book | `vectorized_elementwise_kernel<2, FillFunctor<l` | `[1,1,1]` | 2 | 0.88 | 1.8 | 1.2 | eagle_worker_common.py prepare_for_draft (index fills) |
| 3 | book | `vectorized_elementwise_kernel<2, CUDAFunctor_a` | `[1,1,1]` | 1 | 1.42 | 1.4 | 1.2 | eagle_worker_common.py prepare_for_draft (seq_lens+1) |
| 4 | book | `elementwise_kernel<128, 2, gpu_kernel_impl_noc` | `[1,1,1]` | 15 | 1.25 | 18.7 | 1.3 | per-iteration index/scalar op (DG-6) |
| 5 | book | `multi_tensor_apply_kernel<` | `[5,1,1]` | 1 | 1.68 | 1.7 | 1.5 | cuda_graph_buffer_registry.py:51 _grouped_foreach_copy_ |
| 6 | book | `multi_tensor_apply_kernel<` | `[1,1,1]` | 1 | 1.58 | 1.6 | 1.4 | cuda_graph_buffer_registry.py:51 _grouped_foreach_copy_ |
| 7 | book | `unrolled_elementwise_kernel<direct_copy_kernel` | `[1,1,1]` | 1 | 1.36 | 1.4 | 1.2 | prologue: spec_info staging copy (prepare_for_draft) |
| 8 | book | `elementwise_kernel<128, 2, gpu_kernel_impl_noc` | `[1,1,1]` | 1 | 1.41 | 1.4 | 1.3 | per-iteration index/scalar op (DG-6) |
| 9 | attn/QSA | `_qsa_graph_layout_kernel` | `[2,1,1]` | 14 | 1.06 | 14.8 | 2.0 | qsa/graph_metadata.py:208 <- qwen_sparse_attn_backend.py:1974 (x steps) |
| 10 | attn/QSA | `_qsa_graph_row_metadata_kernel` | `[1,1,1]` | 14 | 10.03 | 140.4 | 28.8 | qsa/graph_metadata.py:226 <- qwen_sparse_attn_backend.py:1974 (x steps) |
| 11 | book | `index_elementwise_kernel<128, 4, gpu_index_ker` | `[1,1,1]` | 1 | 2.40 | 2.4 | 2.2 | eagle_worker_v2.py:704 `hot_token_id[topk_index]` (once, pre-loop) |
| 12 | book | `memcpy32_post` | `[1,1,1]` | 43 | 1.02 | 44.0 | 7.2 | UNATTRIBUTED small DtoD inside the graph body (DG-6) |
| 13 | glue | `indexSelectSmallIndex<c10::BFloat16, l` | `[20,1,1]` | 14 | 1.92 | 26.9 | 3.6 | unquant.py:326 embedding (model.embed_tokens) |
| 14 | glue | `flashinfer RMSNormKernel` | `[1,1,1]` | 14 | 2.08 | 29.1 | 4.8 | qwen4_exp_mtp.py:229/231 pre_fc_norm |
| 15 | glue | `flashinfer RMSNormKernel` | `[1,1,1]` | 14 | 2.46 | 34.5 | 4.8 | qwen4_exp_mtp.py:229/231 pre_fc_norm |
| 16 | gemv | `gemvx::kernel<int, int, __nv_bfloat` | `[320,1,1]` | 14 | 11.50 | 161.1 | 24.1 | qwen4_exp_mtp.py:254 fc_embedding (nn.Linear 2560x2560 bf16) |
| 17 | gemv | `cutlass_80_wmma_bf16_32x32` | `[8,10,20]` | 14 | 11.10 | 155.5 | 22.0 | qwen4_exp_mtp.py:255 fc_hidden (nn.Linear on the 4 HC branches) |
| 18 | gemv | `splitKreduce_kernel<32, 16, int, float, __nv_b` | `[80,1,1]` | 14 | 1.60 | 22.4 | 3.1 | qwen4_exp_mtp.py:255 fc_hidden split-K epilogue |
| 19 | glue | `elementwise_kernel<128, 4, gpu_kernel_impl_noc` | `[20,1,1]` | 14 | 2.02 | 28.2 | 4.0 | qwen4_exp_mtp.py:256 input_embeds.unsqueeze(-2)+encoder_inputs |
| 20 | HC | `_hc_branch_stats_kernel` | `[4,1,1]` | 42 | 1.50 | 63.2 | 9.0 | hc_mix2_triton.py K0 (hyperconnection.py mix()) |
| 21 | HC | `_hc_down_kernel` | `[20,5,1]` | 42 | 3.97 | 166.7 | 23.8 | hc_mix2_triton.py K1 |
| 22 | HC | `_hc_up_kernel` | `[160,1,1]` | 42 | 4.77 | 200.3 | 28.6 | hc_mix2_triton.py K2 |
| 23 | attn/QSA | `_mtp_shared_sparse_indices_lookup_kernel` | `[1,9,1]` | 14 | 5.15 | 72.1 | 9.0 | qwen_sparse_attn_backend.py:232 QSAMTPSharedSparseIndices.lookup |
| 24 | gemv | `_w8a16_gemv_kernel` | `[416,1,1]` | 14 | 23.26 | 325.7 | 47.2 | w8a16_gemv.py -- draft attn qkv/in_proj (FP8) |
| 25 | attn/QSA | `_fused_qk_rmsnorm_rope_gate_kernel` | `[1,26,1]` | 14 | 1.89 | 26.4 | 4.0 | qsa attn: fused q/k norm + rope + gate |
| 26 | attn/QSA | `_fp8_kv_store_kernel` | `[1,2,1]` | 14 | 0.90 | 12.5 | 2.2 | KV fp8 fused store (SGLANG_KV_FP8_FUSED_STORE) |
| 27 | attn/QSA | `_fa2_valid_counts` | `[1,1,1]` | 14 | 1.22 | 17.0 | 2.4 | qsa/sparse_attn.py:349 |
| 28 | attn/QSA | `_compact_kv` | `[1,2,130]` | 14 | 3.74 | 52.4 | - | qsa/sparse_attn.py:429 |
| 29 | attn/QSA | `kernel_mha(unsigned int, float, float const*, ` | `[9,2,1]` | 14 | 16.74 | 234.3 | 33.6 | flashinfer/xqa.py:94 (trtllm-gen decode) |
| 30 | glue | `_fused_sigmoid_mul_kernel` | `[1,6,1]` | 14 | 0.99 | 13.9 | 2.0 | attn output gate |
| 31 | gemv | `_w8a16_gemv_kernel` | `[160,6,1]` | 14 | 12.58 | 176.1 | 24.8 | w8a16_gemv.py -- draft attn o_proj (FP8, split-K 6) |
| 32 | HC | `hc_combine_gate_kernel<4l, 2560l, true, __nv_b` | `[1,32,1]` | 28 | 1.95 | 54.7 | 7.4 | hyperconnection.py:564 combine() |
| 33 | HC | `hc_combine_apply_kernel<4l, 2560l, true, __nv_` | `[1,8,1]` | 28 | 1.12 | 31.4 | 4.5 | hyperconnection.py:564 combine() |
| 34 | book | `memcpy32_post` | `[5,1,1]` | 14 | 0.86 | 12.1 | 1.8 | UNATTRIBUTED small DtoD inside the graph body (DG-6) |
| 35 | gemv | `_w8a16_gemv_kernel` | `[32,10,1]` | 14 | 4.48 | 62.7 | 10.5 | w8a16_gemv.py -- MoE router gate GEMM (SGLANG_ROUTER_GEMV=1, N=512) |
| 36 | gemv | `_w8a16_gemv_kernel` | `[80,10,1]` | 14 | 5.95 | 83.3 | - | shared-expert gate_up (W4 fuses silu: `_w8a16_gemv_silu_kernel` [40,10,1] 6.10us) |
| 37 | MoE | `_router_triton_kernel` | `[1,1,1]` | 14 | 4.45 | 62.3 | 8.4 | kernels/ops/moe/moe_fused_gate.py:350 (gate softmax+topk, 1 CTA) |
| 38 | MoE | `act_and_mul_kernel<__nv_bfloat16, (ActivationK` | `[1,1,1]` | 14 | 1.95 | 27.3 | - | activation.py:92 shared-expert silu*up (W16: gate_up NOT fused) |
| 39 | MoE | `trtllm::blockExpertPrefixSumKernel<512>(i` | `[512,1,1]` | 14 | 3.26 | 45.7 | 4.7 | flashinfer/fused_moe/core.py:510 cutlass_fused_moe glue |
| 40 | gemv | `_w8a16_gemv_kernel` | `[160,3,1]` | 14 | 4.61 | 64.5 | 7.2 | shared-expert down_proj (FP8, split-K 3) |
| 41 | MoE | `trtllm::globalExpertPrefixSumKernel<512>(` | `[1,1,1]` | 14 | 1.76 | 24.6 | 2.9 | flashinfer/fused_moe/core.py:510 cutlass_fused_moe glue |
| 42 | MoE | `trtllm::mergeExpertPrefixSumKernel(int const*,` | `[512,1,1]` | 14 | 1.60 | 22.4 | 3.2 | flashinfer/fused_moe/core.py:510 cutlass_fused_moe glue |
| 43 | MoE | `trtllm::expandInputRowsKernel<__nv_bfloat` | `[10,1,1]` | 14 | 4.54 | 63.6 | 9.1 | flashinfer/fused_moe/core.py:510 cutlass_fused_moe glue |
| 44 | MoE | `trtllm::computeStridesTmaWarpSpecializedK` | `[16,1,1]` | 14 | 2.88 | 40.3 | 5.9 | flashinfer/fused_moe/core.py:510 cutlass_fused_moe glue |
| 45 | MoE | `cutlass GemmUniversal<GroupProblemShape> (NVFP4)` | `[1,188,1]` | 14 | 21.52 | 301.3 | 41.3 | flashinfer/fused_moe/core.py:510 grouped NVFP4 GEMM |
| 46 | MoE | `trtllm::doActivationKernel<__nv_fp4_e2m1,` | `[10,1,1]` | 14 | 3.58 | 50.2 | 7.2 | flashinfer/fused_moe/core.py:510 cutlass_fused_moe glue |
| 47 | MoE | `memset32` | `[1,1,1]` | 14 | 0.80 | 11.2 | 1.6 | flashinfer/fused_moe/core.py:510 workspace zero |
| 48 | MoE | `cutlass GemmUniversal<GroupProblemShape> (NVFP4)` | `[1,188,1]` | 14 | 20.74 | 290.3 | 41.3 | flashinfer/fused_moe/core.py:510 grouped NVFP4 GEMM |
| 49 | glue | `_fused_gate_sigmoid_mul_add_kernel` | `[1,1,1]` | 14 | 1.60 | 22.4 | 3.2 | MoE output gate + residual add |
| 50 | gemv | `_w8a16_gemv_kernel` | `[1024,1,1]` | 14 | 54.98 | 769.7 | 109.5 | draft lm_head over the 32768-row hot vocab (roofline) |
| 51 | book | `unrolled_elementwise_kernel<direct_copy_kernel` | `[64,1,1]` | 14 | 2.50 | 34.9 | 5.0 | logits_processor.py:930 _copy_logits_to_buffer |
| 52 | book | `_draft_topk1_partial_argmax_kernel` | `[1,4,1]` | 14 | 1.86 | 26.0 | 3.6 | kernels/ops/speculative/topk1.py:137 |
| 53 | book | `_draft_topk1_finalize_kernel` | `[1,1,1]` | 14 | 2.21 | 30.9 | 3.8 | kernels/ops/speculative/topk1.py:150 (+positions.add_) |
| 54 | book | `vectorized_elementwise_kernel<2, CUDAFunctorOn` | `[1,1,1]` | 1 | 1.39 | 1.4 | 1.2 | eagle_utils.py:151 build_tree_kernel_efficient |
| 55 | book | `CatArrayBatchedCopy_alignedK_contig<at` | `[1,2,1]` | 1 | 1.86 | 1.9 | 1.8 | eagle_utils.py:165 cat(bonus_tokens, draft_tokens) |
| 56 | book | `vectorized_elementwise_kernel<4, FillFunctor<b` | `[2049,1,1]` | 1 | 1.79 | 1.8 | - | eagle_utils.py:181 tree_mask.fill_(True) |
| 57 | book | `build_tree_efficient(long*, long*, long*, bool` | `[1,1,1]` | 1 | 21.94 | 21.9 | 3.8 | eagle_utils.py:151 build_tree_kernel_efficient |

Notes on the table:

* Rows 9-10 (`_qsa_graph_layout` + `_qsa_graph_row_metadata`) all run **back to back at the head
  of the phase**, occupying 0-196 us of the phase timeline before any model kernel starts.
* Row 12 `memcpy32_post [1,1,1]` is 43 launches: 3 per iteration plus one before the loop. Row 34
  is a 4th per iteration at `[5,1,1]` (1280 32-bit words = 5120 B = one 2560-wide bf16 row).
  `memcpy32_post` is how a graph-captured small DtoD `cudaMemcpyAsync` lowers; these are
  `Tensor.copy_` / `.contiguous()` calls inside the draft body (see DG-6 — they are not yet
  attributed to a line, the graph-mode trace has no python stacks for graph nodes).
* Row 36 vs W4: at W16 `serve-fast.sh` forces `SGLANG_SHARED_GATEUP_FUSED=0` (long-chain
  acceptance regression measured 2026-09-04), so the shared expert pays gate_up + a separate
  `act_and_mul` (row 38) instead of W4's fused `_w8a16_gemv_silu_kernel`. Deliberate — not an item.
* Row 50 is the draft lm_head over the 32768-row hot vocab: 84 MB of FP8 at 55 us is roofline,
  and `TOKEN_MAP` is the only lever (report F2).
* Row 51 is still present although `SGLANG_DRAFT_LOGITS_OUT=1` is a `serve-fast.sh` default —
  see DG-5.

## 2. Every kernel of one W16 draft_extend phase

98 kernels, 0.55 ms busy, one occurrence per decode step (no inner loop). Same trace.

| # | group | kernel | grid | n | med us | us/step |
|--:|---|---|---|--:|--:|--:|
| 0 | book | `vectorized_elementwise_kernel<4, CUDAFunctorOnSel` | `[1,1,1]` | 1 | 1.42 | 1.4 |
| 1 | book | `elementwise_kernel_with_index<int, ara` | `[1,1,1]` | 1 | 1.10 | 1.1 |
| 2 | book | `unrolled_elementwise_kernel<CUDAFunctor_add<long>` | `[1,1,1]` | 1 | 1.60 | 1.6 |
| 3 | book | `vectorized_elementwise_kernel<2, CUDAFunctorOnSel` | `[1,1,1]` | 5 | 1.31 | 6.6 |
| 4 | book | `unrolled_elementwise_kernel<direct_copy_kernel_cu` | `[1,1,1]` | 4 | 1.47 | 5.9 |
| 5 | book | `vectorized_elementwise_kernel<2, (anonymous names` | `[1,1,1]` | 2 | 1.41 | 2.8 |
| 6 | book | `unrolled_elementwise_kernel<direct_copy_kernel_cu` | `[1,1,1]` | 1 | 1.58 | 1.6 |
| 7 | book | `vectorized_elementwise_kernel<4, FillFunctor<int>` | `[1,1,1]` | 2 | 1.42 | 2.8 |
| 8 | book | `compute_position_kernel` | `[1,1,1]` | 1 | 1.15 | 1.2 |
| 9 | book | `vectorized_elementwise_kernel<2, FillFunctor<long` | `[1,1,1]` | 2 | 1.02 | 2.0 |
| 10 | book | `elementwise_kernel<128, 2, gpu_kernel_impl_nocast` | `[1,1,1]` | 1 | 1.60 | 1.6 |
| 11 | book | `elementwise_kernel<128, 2, gpu_kernel_impl_nocast` | `[1,1,1]` | 1 | 1.31 | 1.3 |
| 12 | book | `multi_tensor_apply_kernel<` | `[5,1,1]` | 1 | 1.52 | 1.5 |
| 13 | book | `multi_tensor_apply_kernel<` | `[3,1,1]` | 1 | 1.49 | 1.5 |
| 14 | book | `Memcpy DtoD` | `-` | 1 | 1.62 | 1.6 |
| 15 | attn/QSA | `_qsa_graph_layout_kernel` | `[2,1,1]` | 1 | 2.21 | 2.2 |
| 16 | attn/QSA | `_qsa_graph_row_metadata_kernel` | `[16,1,1]` | 1 | 21.62 | 21.6 |
| 17 | glue | `indexSelectSmallIndex<c10::BFloat16, l` | `[20,1,1]` | 1 | 8.82 | 8.8 |
| 18 | glue | `flashinfer RMSNormKernel` | `[4,1,1]` | 1 | 2.72 | 2.7 |
| 19 | glue | `flashinfer RMSNormKernel` | `[16,1,1]` | 1 | 2.78 | 2.8 |
| 20 | gemv | `cutlass_80_wmma_bf16_32x32` | `[8,10,20]` | 1 | 11.49 | 11.5 |
| 21 | gemv | `splitKreduce_kernel<32, 16, int, float, __nv_bfloat1` | `[80,1,1]` | 1 | 3.62 | 3.6 |
| 22 | gemv | `cutlass_80_tensorop_bf16_128x64` | `[8,3,8]` | 1 | 14.06 | 14.1 |
| 23 | gemv | `splitKreduce_kernel<32, 16, int, __nv_bfloat16, __nv` | `[80,4,1]` | 1 | 5.17 | 5.2 |
| 24 | glue | `elementwise_kernel<128, 4, gpu_kernel_impl_nocast` | `[320,1,1]` | 1 | 2.24 | 2.2 |
| 25 | HC | `_hc_branch_stats_kernel` | `[64,1,1]` | 3 | 1.70 | 5.1 |
| 26 | HC | `_hc_down_kernel` | `[20,5,1]` | 3 | 4.93 | 14.8 |
| 27 | HC | `_hc_up_kernel` | `[160,1,1]` | 3 | 6.46 | 19.4 |
| 28 | gemv | `cutlass_80_wmma_bf16_32x32` | `[8,3,20]` | 1 | 11.20 | 11.2 |
| 29 | gemv | `_w8a16_gemv_kernel` | `[416,1,1]` | 1 | 29.22 | 29.2 |
| 30 | gemv | `splitKreduce_kernel<32, 16, int, float, __nv_bfloat1` | `[20,1,1]` | 1 | 8.06 | 8.1 |
| 31 | attn/QSA | `qsa_index_q_prep_kernel<__nv_bfloat16, 128, true, tr` | `[16,1,1]` | 1 | 11.50 | 11.5 |
| 32 | attn/QSA | `qsa_index_k_compress_kernel<__nv_bfloat16, 128, true` | `[4,1,1]` | 1 | 5.06 | 5.1 |
| 33 | attn/QSA | `_fused_qk_rmsnorm_rope_gate_kernel` | `[16,26,1]` | 1 | 3.52 | 3.5 |
| 34 | book | `vectorized_elementwise_kernel<4, FillFunctor<floa` | `[1024,1,1]` | 1 | 2.62 | 2.6 |
| 35 | attn/QSA | `kernel_kernel` | `[16,1024,1]` | 1 | 13.95 | 14.0 |
| 36 | attn/QSA | `fast_topk_detail::fast_topk_kernel<512, true>(fast_t` | `[16,1,1]` | 1 | 8.43 | 8.4 |
| 37 | attn/QSA | `_expand_qsa_block_indices_kernel` | `[16,1,1]` | 1 | 1.87 | 1.9 |
| 38 | book | `cub::DeviceScanInitKernel<at_cuda_detail::c` | `[1,1,1]` | 1 | 0.86 | 0.9 |
| 39 | book | `cub::DeviceScanKernel<at_cuda_detail::cub::` | `[1,1,1]` | 1 | 1.46 | 1.5 |
| 40 | book | `vectorized_elementwise_kernel<2, CUDAFunctor_add<` | `[1,1,1]` | 2 | 1.07 | 2.1 |
| 41 | book | `vectorized_elementwise_kernel<4, compare_scalar_k` | `[1,1,1]` | 2 | 1.15 | 2.3 |
| 42 | book | `vectorized_elementwise_kernel<4, BinaryFunctor<bo` | `[1,1,1]` | 1 | 1.20 | 1.2 |
| 43 | book | `vectorized_elementwise_kernel<2, (anonymous names` | `[1,1,1]` | 1 | 1.23 | 1.2 |
| 44 | glue | `indexSelectSmallIndex<int, long, unsig` | `[1,1,1]` | 1 | 4.05 | 4.0 |
| 45 | glue | `index_elementwise_kernel<128, 4, gpu_index_kernel` | `[1,1,1]` | 1 | 2.98 | 3.0 |
| 46 | glue | `indexSelectSmallIndex<int, long, unsig` | `[17,1,1]` | 1 | 1.52 | 1.5 |
| 47 | glue | `index_elementwise_kernel<128, 4, index_copy_kerne` | `[5,1,1]` | 1 | 2.34 | 2.3 |
| 48 | glue | `index_elementwise_kernel<128, 4, index_copy_kerne` | `[1,1,1]` | 1 | 1.23 | 1.2 |
| 49 | attn/QSA | `_fp8_kv_store_kernel` | `[16,2,1]` | 1 | 1.02 | 1.0 |
| 50 | attn/QSA | `_fa2_valid_counts` | `[16,1,1]` | 1 | 1.30 | 1.3 |
| 51 | attn/QSA | `_compact_kv` | `[16,2,129]` | 1 | 14.86 | 14.9 |
| 52 | attn/QSA | `kernel_mha(unsigned int, float, float const*, Vec<__` | `[5,2,16]` | 1 | 28.40 | 28.4 |
| 53 | glue | `_fused_sigmoid_mul_kernel` | `[16,6,1]` | 1 | 1.95 | 2.0 |
| 54 | gemv | `_w8a16_gemv_kernel` | `[80,6,1]` | 1 | 15.28 | 15.3 |
| 55 | HC | `hc_combine_gate_kernel<4l, 2560l, true, __nv_bfloat1` | `[16,32,1]` | 2 | 2.35 | 4.7 |
| 56 | HC | `hc_combine_apply_kernel<4l, 2560l, true, __nv_bfloat` | `[16,8,1]` | 2 | 1.46 | 2.9 |
| 57 | book | `memcpy32_post` | `[80,1,1]` | 1 | 2.83 | 2.8 |
| 58 | gemv | `_w8a16_gemv_kernel` | `[32,10,1]` | 1 | 5.95 | 6.0 |
| 59 | gemv | `_w8a16_gemv_kernel` | `[80,10,1]` | 1 | 8.14 | 8.1 |
| 60 | MoE | `_router_triton_kernel` | `[16,1,1]` | 1 | 4.97 | 5.0 |
| 61 | MoE | `act_and_mul_kernel<__nv_bfloat16, (ActivationKind)0,` | `[3,1,1]` | 1 | 2.29 | 2.3 |
| 62 | MoE | `trtllm::blockExpertPrefixSumKernel<512>(i` | `[512,1,1]` | 1 | 3.86 | 3.9 |
| 63 | gemv | `_w8a16_gemv_kernel` | `[80,1,1]` | 1 | 4.58 | 4.6 |
| 64 | MoE | `trtllm::globalExpertPrefixSumKernel<512>(` | `[1,1,1]` | 1 | 1.68 | 1.7 |
| 65 | MoE | `trtllm::mergeExpertPrefixSumKernel(int const*,` | `[512,1,1]` | 1 | 2.14 | 2.1 |
| 66 | MoE | `trtllm::expandInputRowsKernel<__nv_bfloat` | `[160,1,1]` | 1 | 4.78 | 4.8 |
| 67 | MoE | `trtllm::computeStridesTmaWarpSpecializedK` | `[16,1,1]` | 1 | 3.02 | 3.0 |
| 68 | MoE | `cutlass GemmUniversal<GroupProblemShape> (NVFP4)` | `[1,188,1]` | 1 | 69.23 | 69.2 |
| 69 | MoE | `trtllm::doActivationKernel<__nv_fp4_e2m1,` | `[160,1,1]` | 1 | 4.35 | 4.4 |
| 70 | MoE | `memset32` | `[10,1,1]` | 1 | 0.99 | 1.0 |
| 71 | MoE | `cutlass GemmUniversal<GroupProblemShape> (NVFP4)` | `[1,188,1]` | 1 | 50.86 | 50.9 |
| 72 | glue | `_fused_gate_sigmoid_mul_add_kernel` | `[16,1,1]` | 1 | 2.06 | 2.1 |
| 73 | gemv | `_w8a16_gemv_kernel` | `[256,1,1]` | 1 | 60.69 | 60.7 |
| 74 | glue | `index_elementwise_kernel<128, 4, gpu_index_kernel` | `[64,1,1]` | 1 | 5.97 | 6.0 |
| 75 | glue | `index_elementwise_kernel<128, 4, gpu_index_kernel` | `[20,1,1]` | 1 | 6.11 | 6.1 |
| 76 | book | `reduce_kernel<512, 1, ReduceOp<float, at::native:` | `[1,1,1]` | 1 | 8.56 | 8.6 |
| 77 | book | `vectorized_elementwise_kernel<4, FillFunctor<floa` | `[1,1,1]` | 1 | 1.09 | 1.1 |

`draft_extend` is 0.55 ms at W16 and 0.45 ms at W4 and does not scale with S (report §5). Its
`_qsa_graph_row_metadata_kernel` instance (row 16, 21.6 us at 16 rows) is the *same* kernel as
the draft loop's and is covered by DG-1; its 36 bookkeeping kernels (57 us) are the eager
draft-extend index prep (arange / cumsum / compare / index_copy over 16 rows) — real but small,
and the whole phase is 2.7 % of the step, so it is not a priority target.

---

## 3. What in the bookkeeping / glue buckets is actually removable

For each: what is recomputed, why it is loop-invariant or trivially incremental, kernels and us
removed per **decode** step, and how to prove it.

### DG-1 — QSA graph metadata: the page table is rebuilt 4096 entries at a time by one warp

**Code**: `python/sglang/srt/layers/attention/qsa/graph_metadata.py:112-168`
(`_qsa_graph_row_metadata_kernel`), launched at `:226` from `launch_graph_metadata`, called per
draft step by `qwen_sparse_attn_backend.py:1945-1982`
(`init_forward_metadata_out_graph` -> `_replay_cuda_graph_metadata` -> `_replay_cuda_graph_metadata_gpu`).

**What is wrong**: the kernel is launched `[(num_rows,)]` with `num_warps=1`, and its page-table
section is

```python
for p0 in range(0, max_pages, PAGE_BLOCK):        # PAGE_BLOCK = 128
    idx = p0 + offs
    valid = idx < tl.minimum(max_pages, row_width_pages)
    loc = tl.load(req_to_token_ptr + token_row + idx * FULL_PAGE, mask=valid, other=0)
    tl.store(table_row + idx, tl.maximum(loc // FULL_PAGE, 0), mask=valid)
```

`max_pages = ceil(ceil(max_context_len / ratio) / (page_size / ratio)) = max_context_len / page_size
= 262144 / 64 = 4096` (`qwen_sparse_attn_backend.py:903-927`), and `row_width_pages =
req_to_token.stride(0) // FULL_PAGE = 4096` too, so **every one of the 4096 entries is rewritten on
every call**, by a single warp, in 32 serial trips, each gathering 128 words at a 256-byte stride.
The measured cost is therefore independent of the row count:

| launch | phase | med us | n/step (W16) | n/step (W4) |
|---|---|--:|--:|--:|
| `_qsa_graph_row_metadata_kernel [1,1,1]` | draft (one per inner step) | 10.0 | 14 | 2 (14.4 us med) |
| `_qsa_graph_row_metadata_kernel [16,1,1]` (W16) / `[4,1,1]` (W4) | verify + draft_extend | 20.4 / 21.2 | 2 | 2 |
| `_qsa_graph_layout_kernel [2,1,1]` | all | 1.1 | 16 | 4 |

Totals: **199 us/step at W16, 103 us/step at W4** (`gridstats.py <trace> _qsa_graph`). Note W4 is
dominated by the two verify/extend calls, so this item is *not* draft-loop-specific.

**Fix DG-1a (bit-exact, implemented)** — move the page dimension onto `program_id(1)`: grid
`(num_rows, cdiv(max_pages, PAGE_BLOCK))`, each program writes one `PAGE_BLOCK` slice, the scalar
stores (`compressed_lens`, `write_locs`, `logical_positions`, `state_slots`, `ring_locs`) guarded
to `program_id(1) == 0`. Identical values at identical addresses => bit-exact. 32x the CTAs, so
the latency of the strided gather is hidden; expect ~1.5-2 us per launch.
Arithmetic: at W16 the 16 row-metadata launches cost 140.4 (draft) + 40.8 (verify/extend) us and
should land at ~2 us each, i.e. 181 -> ~32 us; at W4 the four launches cost 28.9 + 42.4 us and
should land at ~10 us total. The layout kernel is untouched (17 / 14 us).
Expected: **W16 -0.15 ms/step (range -0.12..-0.17), W4 -0.07 ms/step (range -0.06..-0.09)**.
Flag `SGLANG_QSA_META_PAGE_PARALLEL=1`.

**Fix DG-1b (NOT bit-exact)** — bound the loop at `cdiv(seq_len, FULL_PAGE)`. The scoring kernel
walks pages while `page * page_size < compressed_length` (`qsa/metadata.py:204-215` hands it
`graph_compressed_lengths` and `page_table`), so entries at or above that bound are dead; today
512 of 4096 entries matter at 32k context. Not bit-exact *on the buffer* (the tail keeps stale
values), so it needs its own A/B and a re-read of the indexer's page bound. After DG-1a the
residual is only ~20 us/step, so this is worth ~-0.01 ms at 32k — but it grows with context and
should be revisited at 128k+. Flag `SGLANG_QSA_META_PAGE_BOUND=1`.

**Fix DG-1c (loop-invariance, not yet implemented)** — the 14 draft-step page tables are
*identical*. `assign_draft_cache_locs_contiguous` (the first kernel of the phase) has already
written every draft slot into `req_to_token` before the metadata kernels run, and the page-table
expression reads only `req_to_token[req, idx*64]` with no dependence on `seq_len`. Only the five
scalar buffers differ per step (all are `f(seq_len + i)`). So one launch with a step dimension
suffices: build the page table once, write the 14 rows of scalars in one grid. Residual after
DG-1a is 14 x ~1.5 (row) + 14 x 1.1 (layout) = ~36 us -> ~4 us, i.e. a further **-0.03 ms at W16**.
Requires the per-step buffers to be one contiguous `[steps, rows, max_pages]` allocation in
`init_cuda_graph_state` (each step's backend takes a view) — a real refactor of
`qwen_sparse_attn_backend.py:900-960`. Do it after DG-1a has landed and been measured.

### DG-2 — the MTP entry projections are two untuned bf16 cuBLAS GEMMs

**Code**: `python/sglang/srt/models/qwen4_exp_mtp.py:197-256` (`_fuse_residual_linear_shared`),
`fc_embedding` / `fc_hidden` built as plain `nn.Linear(2560, 2560, bias=False)` at `:178-182`.

Per decode step at W16 (per iteration in brackets):

| kernel | grid | W16 us/step | W4 us/step |
|---|---|--:|--:|
| `flashinfer RMSNormKernel` x2 (`pre_fc_norm_embedding`, `pre_fc_norm_hidden`) | `[1,1,1]` | 63.6 | 9.6 |
| `gemvx::kernel` (`fc_embedding`, M=1) | `[320,1,1]` | 161.1 [11.5] | 24.1 |
| `cutlass_80_wmma_bf16_32x32` (`fc_hidden`, M=4 HC branches) | `[8,10,20]` | 155.5 [11.1] | 22.0 |
| `splitKreduce_kernel` (fc_hidden split-K epilogue) | `[80,1,1]` | 22.4 | 3.1 |
| `elementwise_kernel<128,4>` (`input_embeds.unsqueeze(-2) + encoder_inputs`) | `[20,1,1]` | 28.2 | 4.0 |
| **total** | | **430.8** | **62.8** |

Two 13.1 MB bf16 weight reads in 22.6 us is 1.16 TB/s — well under what `w8a16_gemv`/`bf16_gemv`
reach on this box, and the `splitKreduce` + broadcast-add are pure launch tax. Two flags already
exist and are **off**: `SGLANG_MTP_FC_GEMV=1` (`qwen4_exp_mtp.py:30`, routes both through
`bf16_gemv`, gate `rows * hc_count <= 16` holds at bs=1) and `SGLANG_MTP_ENTRY_FUSED=1`
(`layers/mtp_entry.py:187`, one kernel for norms + both projections + the add).
`serve-fast.sh:14` says they measured slower — **but that A/B predates the gemv3/gemv3b retune
(2026-09-03 19:49) and the split-K scratch fix `ac2e8cb9d2`**. Re-running it is config-only.

Expected if the two GEMMs hit 1.5 TB/s and the epilogue kernels fold: **W16 -0.10..-0.20 ms,
W4 -0.015..-0.03 ms**. Numerics: not bit-exact (different reduction order); gate on acceptance
length, exactly like `SGLANG_HC_MIX2_FP8`.

### DG-3 — trtllm MoE launch metadata: 4 kernels of prefix-sum + stride setup for one token

**Code**: `flashinfer/fused_moe/core.py:510` `cutlass_fused_moe`, reached from the draft MoE in
`qwen4_exp.py`. Per decode step in the **draft** phase at W16:

| kernel | grid | W16 us/step | W4 us/step |
|---|---|--:|--:|
| `trtllm::blockExpertPrefixSumKernel<512>` | `[512,1,1]` | 45.7 | 4.7 |
| `trtllm::globalExpertPrefixSumKernel<512>` | `[1,1,1]` | 24.6 | 2.9 |
| `trtllm::mergeExpertPrefixSumKernel` | `[512,1,1]` | 22.4 | 3.2 |
| `trtllm::computeStridesTmaWarpSpecializedKernel` | `[16,1,1]` | 40.3 | 5.9 |
| subtotal (launch metadata) | | **133.0** | **16.7** |
| `trtllm::expandInputRowsKernel` | `[10,1,1]` | 63.6 | 9.1 |
| `trtllm::doActivationKernel<fp4>` | `[10,1,1]` | 50.2 | 7.2 |
| `memset32` (workspace zero) | `[1,1,1]` | 11.2 | 1.6 |
| vs the two `cutlass GemmUniversal<GroupProblemShape>` they set up | `[1,188,1]` | 591.6 | 82.6 |

**512 CTAs of prefix-sum over the expert histogram of a single token**, three times, plus a
16-CTA TMA descriptor build — 133 us/step for launch metadata against 592 us of actual GEMM. The
work is O(num_experts), not O(tokens); a single-CTA specialisation for `M <= 16` should land at
~10 us. Expected **W16 -0.12 ms, W4 -0.014 ms in the draft alone**, and the same glue in the
verify phase is 1.02-1.43 ms/step (report §5.1 "MoE glue") where the same fix is worth
**another -0.3..-0.5 ms**. This is the largest item in the whole spec and also the hardest: the
replacement must reproduce trtllm's `expert_first_token_offset` / TMA descriptor layout exactly.

### DG-4 — `_mtp_shared_sparse_indices_lookup`: 2048 frozen columns recopied every iteration

**Code**: `python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py:51-82` (kernel),
`:208-256` (`QSAMTPSharedSparseIndices.lookup`, capture at `:194`), called once per draft iteration from
`qwen4_exp.py:1506`.

`indices[slot]` is the frozen target-aligned selection captured during `draft_extend`; it does
not change inside the draft loop (`capture()` is only called when
`should_capture_mtp_sparse_indices` is true, which is false in decode). Only the last
`tail_width` columns depend on `current_positions`, which advances by one per iteration:

```python
tail_value = base + tail_offset
tail_value = tl.where(tail_value <= position, tail_value, -1)
value = tl.where(columns >= tail_start, tail_value, frozen)
```

Cost: `[1,9,1]` x 5.15 us x 14 = **72.1 us/step at W16**, 9.0 us at W4. `num_columns =
token_topk + tail_width` (~2064), so 2048 of 2064 columns are an identical copy 14 times over.

**Fix**: give `lookup()` a persistent output buffer (allocated with the other graph buffers) and
launch the *full* column grid only on the first draft iteration; for the rest launch only the
column blocks from `tail_start // BLOCK` upward (add a `col_block_start` scalar to the kernel).
Bit-exact by construction — same kernel, same values. Expect 72.1 -> ~20 us, **W16 -0.05 ms**,
W4 -0.006 ms.
**Prerequisite check**: `IndexTopKShareState.publish` (`layers/attention/index_topk_share.py:51-64`)
can store the returned tensor into `spec_info.dsa_topk_indices`; confirm that on the
`should_reuse_mtp_sparse_indices` path nothing retains the buffer past the iteration, otherwise
the persistent buffer aliases across decode steps. This is why it is spec-only here.

### DG-5 — the draft logits copy is still in the trace

**Code**: `python/sglang/srt/layers/logits_processor.py:930` `_copy_logits_to_buffer`.
`unrolled_elementwise_kernel<direct_copy_kernel_cuda>` `[64,1,1]` x 2.50 us x 14 = **34.9 us/step
at W16** (5.0 at W4). Grid 64 x 128 threads x 4 = 32768 = exactly the hot vocab, so this is the
draft lm_head output being copied into `next_token_logits_buffer`.

`SGLANG_DRAFT_LOGITS_OUT=1` is a `serve-fast.sh` default and `test/srt/layers/test_draft_logits_out.py`
exists, yet the copy is still in the 2026-09-04 traces. **First task is to find why the gate
declines**: `TestDirectLogitsBufferGate` in that file enumerates the conditions
(`next_token_logits_buffer` present, no TP all-gather, vocab match). Likely candidates: the hot
token map changes the head's N so the buffer's last dim no longer matches, or the draft head
path does not pass `out=`. If it can be re-enabled: **W16 -0.035 ms, W4 -0.005 ms**, bit-exact
(the GEMV rounds its fp32 accumulator to bf16 before widening either way).

### DG-6 — the unattributed in-graph small copies

Per draft iteration the captured body contains, immediately after `_draft_topk1_finalize` and
before the embedding lookup:

| kernel | grid | per iter | W16 us/step | W4 us/step |
|---|---|--:|--:|--:|
| `memcpy32_post` | `[1,1,1]` (<= 1 KB) | 3 | 44.0 | 7.2 |
| `memcpy32_post` | `[5,1,1]` (5120 B = one 2560-wide bf16 row) | 1 | 12.1 | 1.8 |
| `elementwise_kernel<128,2, gpu_kernel_impl_nocast>` | `[1,1,1]` | 1 | 17.5 | 2.5 |
| total | | 5 | **73.6** | **11.5** |

These are `Tensor.copy_` / `.contiguous()` inside the draft body (`memcpy32_post` is how a
graph-captured small DtoD memcpy lowers). They are **not attributed to a source line yet**: the
graph-mode trace carries no python stack for graph nodes, and the eager trace
(`traces/stack-w16-w16-code-edit/`) takes a different metadata path (`_qsa_write_plan`,
420 kernels), so its attribution does not transfer.

**Step 1 of this item is identification, not optimisation.** Do it by profiling the *capture*,
not the replay: `torch.profiler` with `with_stack=True` active while
`EAGLEDraftCudaGraphRunner` captures the draft graph records the python frames of every node.
Add a one-shot `SGLANG_PROFILE_DRAFT_CAPTURE=1` hook around the capture call and dump.
Cheap alternative that costs nothing: grep the draft body path for `.contiguous()` — the
`[5,1,1]` copy is one 2560-wide bf16 row, and `eagle_worker_v2.py:762-766` already contains a
conditional `out_cache_loc = out_cache_loc.contiguous()` for a different architecture.
Optimistic ceiling once identified: **W16 -0.05 ms, W4 -0.008 ms**.

### DG-7 — `build_tree_efficient` builds a chain with 16 threads

**Code**: `python/sglang/srt/speculative/eagle_utils.py:151` `build_tree_kernel_efficient`.
Epilogue of the draft phase: `build_tree_efficient` `[1,1,1]` block `[16,1,1]` = **21.9 us/step at
W16** (3.8 at W4), plus `tree_mask.fill_(True)` `[2049,1,1]` 1.8 us and the
`cat(bonus_tokens, draft_tokens)` 1.9 us. One CTA of 16 threads for a `T x (seq_len + T)` mask.

At topk=1 the tree is a chain, so the tree part of the mask is a constant lower-triangular
`T x T` block and the prefix part is all-ones — exactly the invariance
`_rebuild_topk1_chain_buffers` (`base_spec_worker.py:112`) already exploits for `parent_list`
and `top_scores_index`. Prefill the constant `T x T` block once at capture and have the per-step
kernel write only the `seq_len` offset. Expected **W16 -0.02 ms**, W4 -0.003. Bit-exact.
Small, self-contained, good warm-up task.

### DG-8 — `_router_triton_kernel` runs the expert gate in one CTA

`kernels/ops/moe/moe_fused_gate.py:350`, grid `[1,1,1]`, 4.45 us x 14 = **62.3 us/step at W16**
(8.4 at W4). It follows the router gate GEMM (`_w8a16_gemv_kernel [32,10,1]`, N=512 experts,
`SGLANG_ROUTER_GEMV=1`). One CTA doing softmax + group top-k over 512 logits for one token, at
4.45 us, is latency not work. Either widen the grid or fold the gate into the GEMV's epilogue
(the same trick as `SGLANG_SHARED_GATEUP_FUSED`). Expected **W16 -0.03 ms**, W4 -0.004.

### DG-9 — `_fa2_valid_counts` folded into the metadata kernel

`qsa/sparse_attn.py:349`, grid `[1,1,1]`, 1.22 us x 14 = 17.0 us/step at W16. It derives valid
counts / `cu_seqlens_k` from the compressed lengths that DG-1's kernel just wrote. Fold it into
the same kernel (it is `f(compressed_lens)` only). **W16 -0.017 ms**, W4 -0.002. Bit-exact.
Do it as part of DG-1c.

### Measured and explicitly NOT worth doing

| candidate | measurement |
|---|---|
| fuse the two entry RMSNorms into one | 63.6 us/step at W16, but they are two different normalized widths (2560 and 4x2560); covered by DG-2's `mtp_entry_fused` instead |
| fuse `_fused_sigmoid_mul` (13.9) + `_fused_gate_sigmoid_mul_add` (22.4) + `act_and_mul` (27.3) | already single fused kernels at 1.0-2.0 us each; ~1 us of launch tax apiece, nothing to win |
| the `[5,1,1]` MoE-side `memcpy32_post` | 12.1 us/step; part of DG-6, do not chase separately |
| `_compact_kv` (52.4 us/step) | real gather work (`[1,2,130]` = 130 topk blocks), not glue |
| `assign_draft_cache_locs_contiguous` + prologue | 13 us/step total, once per phase |
| enabling `SGLANG_SHARED_GATEUP_FUSED` at W16 | 25 us/step of kernel time, already rejected 2026-09-04 for a 0.3 acceptance-length regression on agent-loop |

---

## 4. Ranked work items

Expected saving is **per decode step**, from the trace medians above. "Effort" is my estimate for
one focused agent session.

| # | item | W16 | W4 | effort | risk | bit-exact |
|---|---|--:|--:|---|---|---|
| **DG-1a** | page-parallel `_qsa_graph_row_metadata_kernel` | **-0.15** | **-0.07** | done (this branch) | none | yes |
| **DG-3** | single-CTA MoE launch metadata for `M <= 16` (draft) | **-0.12** | -0.014 | high (2-3 d) | correctness | no (same values, different kernel) |
| **DG-2** | re-A/B `SGLANG_MTP_FC_GEMV` / `SGLANG_MTP_ENTRY_FUSED` after the gemv3 retune | **-0.10..-0.20** | -0.02..-0.03 | config only | acceptance | no |
| **DG-4** | tail-only refresh of the MTP shared sparse indices | **-0.05** | -0.006 | medium | aliasing | yes |
| **DG-6** | identify + remove the 5 in-graph small copies per iteration | -0.05 | -0.008 | medium (identify first) | correctness | depends |
| **DG-5** | re-enable `SGLANG_DRAFT_LOGITS_OUT` on the draft head | -0.035 | -0.005 | low | none | yes |
| **DG-1c** | one batched metadata launch for all draft steps | -0.03 | -0.005 | medium-high | correctness | yes |
| **DG-8** | widen / fold `_router_triton_kernel` | -0.03 | -0.004 | low-medium | none | no |
| **DG-7** | constant chain tree-mask at topk=1 | -0.02 | -0.003 | low | correctness | yes |
| **DG-9** | fold `_fa2_valid_counts` into the metadata kernel | -0.017 | -0.002 | low (with DG-1c) | none | yes |
| **DG-1b** | bound the page build by `seq_len` | -0.01 | -0.005 | done (this branch) | correctness | **no** |
| | **sum if everything lands** | **-0.61..-0.71** | **-0.15..-0.16** | | | |

Against the 20.47 ms W16 / 11.95 ms W4 step that is **3.0-3.5 % (W16)** and **1.3 % (W4)**.
DG-3's verify-side twin is worth another -0.3..-0.5 ms at W16 and is the real prize; it is listed
separately because it is not draft-loop work.

### Implementation order

1. **DG-1a** — flagged and committed on `opus/draft-glue`. Validate first: it is the biggest,
   the safest, and it also de-risks the measurement (it removes 155 us of serial GPU dead time
   that sits before the draft graph launch, plus 44 us in verify/draft_extend).
2. **DG-2** — config-only A/B, no code. Run it in the same GPU window as (1).
3. **DG-5** then **DG-7** — small, bit-exact, good for shaking out the harness.
4. **DG-4** — after the `IndexTopKShareState.publish` aliasing check.
5. **DG-6** — identification pass first, then decide.
6. **DG-1c + DG-9** — the buffer refactor, once DG-1a's number is confirmed.
7. **DG-3** — last, and do the draft and verify sides together.

### Env flags

| flag | default | meaning |
|---|---|---|
| `SGLANG_QSA_META_PAGE_PARALLEL` | `0` | DG-1a. Page dimension on `program_id(1)`. Bit-exact. |
| `SGLANG_QSA_META_PAGE_BOUND` | `0` | DG-1b. Stop the page build at `cdiv(seq_len, 64)`. **Not** bit-exact on the buffer. |
| `SGLANG_MTP_FC_GEMV` | `0` | DG-2, exists. `fc_embedding`/`fc_hidden` through `bf16_gemv`. |
| `SGLANG_MTP_ENTRY_FUSED` | `0` | DG-2, exists. One kernel for the whole MTP entry. |
| `SGLANG_DRAFT_LOGITS_OUT` | `1` in `serve-fast.sh` | DG-5, exists but apparently inert on this path. |
| `SGLANG_QSA_MTP_INDEX_TAIL_ONLY` | (to add) | DG-4. |
| `SGLANG_MOE_META_SMALL_M` | (to add) | DG-3. |
| `SGLANG_TOPK1_CONST_TREE_MASK` | (to add) | DG-7. |

Every new item must be flag-gated, default OFF, and must reproduce the old path exactly when off
(for DG-1a this was checked by diffing the compiled sm_120 TTIR of the flag-off kernel against
the pre-change kernel — see §5).

---

## 5. Validation protocol (mandatory)

### 5.0 Step-time claims use `busy_ms`, never `legacy_trimmed_ms`

`prof/trimmed_step.py` prints three per-phase numbers. **`busy_ms` is the only one that may be
quoted**: it is the interval union of the phase's kernel intervals, i.e. real GPU-busy time.
`raw_ms` sums durations and double-counts 10-15 % because a CUDA graph's internal branches land
on up to 8 GPU streams. `legacy_trimmed_ms` is the pre-2026-09-05 metric, kept only so old logs
stay readable: it clips each kernel to `2 x median(by name) + 5 us`, and one kernel *name* covers
many shapes here (`_w8a16_gemv_kernel` runs at 4/5/6/13/24/55 us in the draft alone), so it
**under-reports GPU-busy time by 0.8-3.6 ms/step and the error grows with kernel count**. Every
item in this spec removes kernels, which is exactly the direction in which `legacy_trimmed_ms`
lies. A change that "improves `legacy_trimmed_ms`" has proven nothing.

Sanity check on every profile: the script's `(check)` line, `sum(busy) + idle ~= wall`, must
close to within 0.4 ms.

### 5.1 Unit test (no server)

```
cd ~/tools/sglang-draft-glue
CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD/python ~/tools/sglang-rtxpro6000/.venv/bin/python \
  test/srt/layers/test_qsa_graph_metadata_pages.py      # planner cases, CPU only
PYTHONPATH=$PWD/python ~/tools/sglang-rtxpro6000/.venv/bin/python \
  test/srt/layers/test_qsa_graph_metadata_pages.py      # + bit-exactness + torch reference, needs CUDA
```

The GPU cases assert that every one of the six output buffers is `torch.equal` between
`PAGE_PARALLEL=False` and `PAGE_PARALLEL=True` at `(rows, max_pages)` =
(1, 4096), (4, 4096), (16, 4096), (3, 129), and that the parallel kernel matches an independent
torch reference. **Both must pass before any in-server run.**

Compile-only check that needs no GPU at all (catches IR errors and proves the flag-off path is
unchanged) — this is how DG-1a was verified on CPU:

```
PATH=$HOME/tools/mamba/envs/cuda13/bin:$PATH CUDA_VISIBLE_DEVICES="" \
  ~/tools/sglang-rtxpro6000/.venv/bin/python -c '
from triton.compiler import ASTSource; from triton.backends.compiler import GPUTarget
import triton; ...  # ASTSource(fn=_qsa_graph_row_metadata_kernel, ...)
triton.compile(src, target=GPUTarget("cuda", 120, 32))'
```

For any new item, add the equivalent: a bit-exactness test against the old path on dumped inputs.
`~/mtp-parity-a/` holds `draft-<pid>-<n>.safetensors` server dumps (input ids, positions, target
HC state, draft logits / `own_hc` / `mixed` hidden states); the development tool
`parity/compare.py` (not published in `mtp-train/`) reads them (`--ref DIR` compares a standalone head against the dumps, `--cmp A B` compares two
server runs matched by the hash of `(input_ids, positions)`). Use `--cmp` to compare a flag-on
and a flag-off dump run: for a bit-exact item the top-1 agreement must be 100 % and the logit
max-abs difference exactly 0.

### 5.2 CUPTI profile (server, no bench)

```
cd ~/tools/flash-next-bench/prof
SERVER_ENV="SGLANG_QSA_META_PAGE_PARALLEL=1" MODE=prof ./validate.sh dg1a w16
SERVER_ENV="SGLANG_QSA_META_PAGE_PARALLEL=1" MODE=prof ./validate.sh dg1a w4
```

`validate.sh` starts `serve-fast.sh <profile>`, takes 20-step traces for `code-edit` and
`prose-en` into `prof/traces/<label>-<profile>-<workload>/`, runs `trimmed_step.py`, and kills the
server. Compare against the baselines in `OVERHEAD_REPORT.md` §6 note F0 (W16 code-edit: draft
4.44 / verify 15.10 / extend 0.55 / busy 20.09 / wall 20.47; W4 code-edit: 0.58 / 10.56 / 0.45 /
11.59 / 11.95). For DG-1a the draft `busy_ms` must drop by ~0.11 and verify+extend by ~0.04 at
W16; also confirm directly with

```
python gridstats.py traces/dg1a-w16-code-edit/*.gz _qsa_graph 20   # expect total/step ~15-25 us, was 239.5
```

Because the item removes serial GPU time from a phase whose in-trace step-wall spread is already
+-1.8 ms (Max-Q power cap / compositor preemption, report F4), **quote the per-phase `busy_ms`,
not the wall**, and take both workloads.

### 5.3 Acceptance + needle (full)

```
SERVER_ENV="SGLANG_QSA_META_PAGE_PARALLEL=1" MODE=full ./validate.sh dg1a w16
SERVER_ENV="SGLANG_QSA_META_PAGE_PARALLEL=1" MODE=full ./validate.sh dg1a w4
```

This adds `fnbench run --workloads code-edit,prose-en,agent-loop --repeats 2 --sampling greedy`
into `runs/dg1a-<profile>.jsonl` and `needle_test.py 18500 0.4`. Gates:

* acceptance length unchanged — greedy W4 3.7-3.85 (code) / 2.2-2.3 (prose) / 3.1 (agent);
  W16 code 9-10.8, prose 2.4-2.5. For a bit-exact item any movement outside noise means a bug.
* needle 18.5k **PASS**.
* per-step wall from the SSE timeline, not `accept_len / tps`:
  `python prof/sse_wall.py runs/dg1a-w16.jsonl` (baselines: W16 base32k code-edit 21.80,
  prose-en 20.11, agent-loop 20.74 ms; W4 12.45 / 11.90 / 12.06 ms).

Session-to-session drift on this box is ~6 %, so a winner must be re-measured head-to-head in one
sitting before it goes into `serve-fast.sh`.

### 5.4 GPU protocol

The GPU is shared with a human user. Never start a server or a benchmark without checking that
the box is free, run the whole `validate.sh` under whatever lock the supervisor names, and never
run two of them concurrently. Re-apply `nvidia-smi -pl 325` after any reboot (report F4) and
prefer a quiet desktop for the measurement runs.

---

## 6. What is on branch `opus/draft-glue`

Worktree `~/tools/sglang-draft-glue`, branch `opus/draft-glue` off `ea52699963`. One commit,
`f21ef2f0f6`:

* `python/sglang/srt/layers/attention/qsa/graph_metadata.py`
  * `SGLANG_QSA_META_PAGE_PARALLEL` (DG-1a) and `SGLANG_QSA_META_PAGE_BOUND` (DG-1b), both
    default OFF, read through `functools.lru_cache`d helpers.
  * `_qsa_graph_row_metadata_kernel` gains `PAGE_PARALLEL` / `PAGE_BOUND` `tl.constexpr`
    parameters. With `PAGE_PARALLEL` off the constexpr short-circuit erases the guard at trace
    time; the compiled sm_120 TTIR of the flag-off kernel is identical to the pre-change kernel
    apart from the loop-invariant `page_limit` being hoisted out of the page loop.
  * `qsa_row_metadata_grid()` — the pure-python launch planner, so the grid arithmetic is
    testable without a GPU.
* `test/srt/layers/test_qsa_graph_metadata_pages.py` — 5 CPU cases (planner) + 2 CUDA cases
  (bit-exactness vs the serial launch on four shapes, and a torch reference for the stored
  values). CPU cases pass; CUDA cases skip on this machine.

Nothing in `~/tools/sglang-rtxpro6000` was touched. No server was started, no GPU work was run;
the Triton kernels were only *compiled* for sm_120 (ptxas, no device).
