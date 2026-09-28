# Patch series index

Base: `jpezzulli/sglang-rtxpro6000` @ `16e5682aad` (branch `pennyroyal-main-sm120-final`). "original" is the commit hash on the author's development branch, which the documentation cites; patch 0040 folds a side-branch merge (its side-branch commits are listed in the second column). Dates are author dates (JST). The patches were regenerated for publication (see `../README.md`): results files with generated text were stripped, the development shell scripts under `dbg/` and `prof-pdl/` were left out (so the file counts of 0061–0066 and 0074 are lower than in the original commits), and names of the author's other projects genericised; subjects, authorship dates and runtime code are unchanged.

| # | original | date | subject | files |
|---:|---|---|---|---:|
| 0001 | 4fe6296192 | 08-28 | local: harden recover_final_state (int64 slot index cast + dense-pitch assert) | 1 |
| 0002 | 6d229a69f0 | 09-01 | feat: NGRAM_CHAIN request-local suffix-match chain draft provider | 7 |
| 0003 | 2ffd61a800 | 09-01 | feat: opt-in per-category FP8 for qwen4_exp dense path | 9 |
| 0004 | 5058a8582d | 09-02 | perf: Triton W8A16 dense GEMV, hot-vocab draft head, adaptive logits buffer sizing | 5 |
| 0005 | aa61a26bad | 09-02 | fix(qsa): size the pending index-key ring for the speculative verify window | 7 |
| 0006 | c0eb3f7276 | 09-02 | perf: FP8 hot-vocab draft head via W8A16, Triton in_proj_ba for Qwen3_5GatedDeltaNet | 2 |
| 0007 | c03716d161 | 09-02 | qwen3_5: SGLANG_GDN_ALT_STREAM=0 switch to keep in_proj_ba on the main stream | 1 |
| 0008 | 7a62299ce3 | 09-02 | ngram_chain: implement create_future_map and fall back to EAGLE enum helpers | 1 |
| 0009 | 4bf52fbfcd | 09-02 | ngram_chain: log hit/miss counts every 200 draft calls | 1 |
| 0010 | 9d618c7f99 | 09-02 | hc_mix: optional weight-only FP8 path for the persistent mix kernel (SGLANG_HC_MIX_FP8=1) | 2 |
| 0011 | 3a896dfe84 | 09-02 | wip: mtp_dense FP8 category plumbing (not yet effective for the in-model MTP draft) | 2 |
| 0012 | 1b43a90af3 | 09-02 | fix(adaptive): make --speculative-adaptive sound for qwen4_exp (GDN recovery fence, per-state QSA index share, per-state recovery graphs) | 9 |
| 0013 | 058765b624 | 09-02 | fix(adaptive): tolerate EagerRunner (no capture_bs) when scheduling GDN recovery-graph capture | 1 |
| 0014 | 0f006a6a98 | 09-02 | wip(adaptive): private workspace + private input buffers while building extra runtime states (diagnostic; did not fix the acceptance drop) | 4 |
| 0015 | 6d77665af7 | 09-03 | qsa: build per-step draft seq_lens as one [steps, bs] table | 1 |
| 0016 | fcc43cdf2a | 09-03 | spec: remove per-draft-step glue in the topk=1 chain with a hot-vocab map | 4 |
| 0017 | 4ddfa5a229 | 09-03 | wip(hc): fused combine+norm+mix persistent Triton kernel (SGLANG_HC_FUSED=1, default off) | 5 |
| 0018 | f30507e31f | 09-03 | gdn: cache detached A_log/dt_bias views so FlashInfer's bf16 cast cache hits | 1 |
| 0019 | 075d071988 | 09-03 | glue: cached router zero-bias, fused FP8 KV scale+cast+store, router GEMV and MTP-entry kernels (flag-gated) | 10 |
| 0020 | 45aab312cb | 09-03 | kv: fused FP8 store accepts token-strided K/V sources | 3 |
| 0021 | aeef3d2ab5 | 09-03 | wip(hc): round-4 fused HC kernel (weight prefetch above barriers, end-of-kernel t_raw clear, pipelined PB) | 1 |
| 0022 | d4faa135d7 | 09-03 | mtp: make the mtp_dense weight-only FP8 category reach the in-model MTP draft | 4 |
| 0023 | 0297a12070 | 09-03 | mamba: 64-wide channel blocks for small-grid causal_conv1d_update | 1 |
| 0024 | 65192006d4 | 09-03 | moe/mtp: route the router gate and MTP entry projections through the skinny BF16 Triton GEMV (flag-gated) | 2 |
| 0025 | b20ce6eebb | 09-03 | hc: three-kernel norm+mix (K0 stats/normalize/zero, K1 split-K down-proj with atomics, K2 up-proj+gate+mean), SGLANG_HC_MIX2=1 | 4 |
| 0026 | 1eb2831e5c | 09-03 | w8a16 gemv: per-shape tuned tiles + in-launch fixup split-K (no extra launch) | 1 |
| 0027 | a63e00ba95 | 09-03 | w8a16 gemv: retune for the [N, K] weight layout the server actually passes | 11 |
| 0028 | 257b3adb6c | 09-03 | w8a16 gemv: pin in_proj_ba and attention-qkv M=16 tiles to the server-measured winners | 1 |
| 0029 | 61cf5390bc | 09-03 | w8a16 gemv: shared-expert down keeps the split-5 tile at M=4 (W4 server 4.0us vs 5.2us) | 1 |
| 0030 | 32fe4f5953 | 09-03 | gdn: strided q/k/v read at the full T=16 tile (patch for the vendored WY wrapper) | 4 |
| 0031 | 06b6f7f709 | 09-03 | hc: one-launch combine (gate + apply in a single Triton kernel), SGLANG_HC_COMBINE_FUSED=1 | 4 |
| 0032 | 5a7572cfc0 | 09-03 | logits: let the W8A16 lm_head GEMV write the next-token buffer directly (SGLANG_DRAFT_LOGITS_OUT=1) | 5 |
| 0033 | f9ba2792e5 | 09-03 | hc: rename the fused-combine grid-size arg off Triton's reserved num_ctas name | 1 |
| 0034 | d3bfaa971a | 09-03 | hc: fused-combine support check covers every tensor the kernel reads compactly | 4 |
| 0035 | 425abd9e5c | 09-03 | hc: reshape the fused combine around per-branch column slices | 2 |
| 0036 | bc834c3be9 | 09-03 | hc: record the CPU-emulated ulp gap in the fused-combine docstring | 1 |
| 0037 | 9e841cb79f | 09-03 | bench: isolated router-gate picture for the SGLANG_ROUTER_GEMV A/B | 1 |
| 0038 | 921554538b | 09-03 | bench: size the GDN WY working set to the state slots the kernel actually reads | 2 |
| 0039 | d35215617c | 09-04 | hc: cheaper barrier poll, and record the fused combine as a measured regression | 1 |
| 0040 | 1c0bd58a57 (merge of f685bf517d, c964be9ebc, b5301df256, 081e8d9b09) | 09-04 | Merge opus/hc-fp8: FP8 HC mix weights and fused shared-expert gate_up+silu | 15 |
| 0041 | ac2e8cb9d2 | 09-04 | w8a16 gemv: per-stream split-K scratch slots (fixes garbage output with SGLANG_ROUTER_GEMV=1) | 2 |
| 0042 | ea52699963 | 09-05 | qwen4-exp mtp: placeholder draft vocab weights (SGLANG_DRAFT_SKIP_VOCAB_WEIGHTS) | 2 |
| 0043 | f21ef2f0f6 | 09-05 | qsa graph metadata: page-parallel row-metadata kernel (SGLANG_QSA_META_PAGE_PARALLEL) | 2 |
| 0044 | 51afe49c41 | 09-05 | mtp entry: precompute the embedding-side projection per token (SGLANG_MTP_EMBED_TABLE) | 2 |
| 0045 | 56383f08a1 | 09-05 | gdn: chain-parallel verify conv1d (SGLANG_GDN_CONV_CHAIN_PARALLEL) | 2 |
| 0046 | cf12af5ddc | 09-05 | test: cover the transposed conv-state layout (kda_backend view) | 1 |
| 0047 | 64b662b5f6 | 09-05 | w8a16_gemv: optional two-destination column split in the epilogue | 1 |
| 0048 | 5ec47f104c | 09-05 | R2: GDN verify writes a/b straight into the RecoverSSM stash | 3 |
| 0049 | 50e1e488b8 | 09-05 | R5: GDN projections write mixed_qkv/z/b/a directly, dropping the re-layout copy | 2 |
| 0050 | 7607c2c977 | 09-05 | R5: match the dual-stream gate of _forward_input_proj exactly | 1 |
| 0051 | cdd86329b8 | 09-05 | R5/R2: guard the flag entry against the non-Tensor (aiter tuple) input | 1 |
| 0052 | f72b65712f | 09-05 | w8a16_gemv: single store for n blocks that do not straddle the split | 1 |
| 0053 | ccf9d2faba | 09-05 | Revert "w8a16_gemv: single store for n blocks that do not straddle the split" | 1 |
| 0054 | f8cc4d58cf | 09-05 | hc boundary: compute the combine gate at mix time (SGLANG_HC_GATE_EARLY) | 5 |
| 0055 | b62750e80c | 09-05 | qwen4-exp moe: shared-expert gate early, join folded into HC apply (SGLANG_SHARED_GATE_EARLY) | 3 |
| 0056 | 0b0f8f32ec | 09-05 | hc boundary: fold the combine apply into the next mix's K0 (SGLANG_HC_APPLY_MIX_FUSED) | 4 |
| 0057 | 610ddd5d60 | 09-05 | qwen4-exp moe: launch the early shared gate inside the dual-stream region | 1 |
| 0058 | 2cdfe3ef69 | 09-05 | hc combine apply: make the gate partial-slot count a template parameter | 2 |
| 0059 | 8d3cf2306c | 09-05 | hc gate early: record that the K0-fused gate is a measured loss | 2 |
| 0060 | d723dc04c7 | 09-05 | qwen4-exp moe: defer the shared join at decode widths only | 1 |
| 0061 | 2039efca83 | 09-05 | fix(adaptive): stop the shared draft-extend backend being re-sized per state | 9 |
| 0062 | 558df950cb | 09-05 | fix(adaptive): drop post-switch stale accept samples (ping-pong) | 4 |
| 0063 | e53cbebc04 | 09-05 | adaptive: post-switch grace window + retuned two-state config | 8 |
| 0064 | e40701639b | 09-05 | adaptive: measure it -- building a second runtime state costs the base profile | 1 |
| 0065 | b51c0dcdd2 | 09-05 | fix(adaptive): give every runtime state its own draft-extend backend | 5 |
| 0066 | 926e3661fb | 09-05 | adaptive: re-seed the EMA on step-down only; ship the validated config | 6 |
| 0067 | 50b3487a04 | 09-05 | pdl: SGLANG_TRITON_PDL flag + gdc wait/trigger helpers for Triton kernels | 1 |
| 0068 | 7081a094f4 | 09-05 | pdl: w8a16_gemv + w8a16_gemv_silu wait/trigger (229+48 launches per W4 verify step) | 1 |
| 0069 | c0a3a4c452 | 09-05 | pdl: hc_mix2 K0/K1/K2 wait/trigger (97 launches each per W4 verify step) | 1 |
| 0070 | b819f27cfa | 09-05 | pdl: fused_qkvzba_split_reshape_cat_contiguous wait/trigger | 1 |
| 0071 | 661e87a643 | 09-05 | pdl: fla layer-norm trigger + _fused_sigmoid_mul_kernel wait/trigger | 2 |
| 0072 | d8f89935c0 | 09-05 | Revert the fla layer-norm trigger: it was already there | 1 |
| 0073 | f44d35d20a | 09-05 | pdl: default USE_PDL to False so the bench/ direct callers keep working | 2 |
| 0074 | e58cf349d3 | 09-06 | pdl: measurement harness and results | 11 |
| 0075 | 81d4e5d8e5 | 09-06 | hc layer boundary: defer the MoE combine into the next layer's mix K0 | 4 |
| 0076 | b9cf17830b | 09-06 | hc layer boundary: issue the prologue's row loads before the gate reduction | 1 |
| 0077 | bb3c5ac76c | 09-06 | hc layer boundary: the measurement scripts, and what they said | 5 |
| 0078 | 62efb3bc65 | 09-06 | H1-C: split-aware GEMV grid so no CTA straddles a two-destination split | 1 |
| 0079 | d042e1845f | 09-06 | H1-B: fold the GDN gated RMSNorm into the out_proj GEMV's A-load | 4 |
| 0080 | 2f7130aff9 | 09-06 | H1: record the in-server A/B for the two GEMV fusion flags | 2 |
| 0081 | ef31f26346 | 09-06 | H1: default-path byte-identity check against the pre-H1 build | 1 |
| 0082 | 06f2563c39 | 09-06 | C1 step 1: make the draft's real top-1 probability available | 7 |
| 0083 | bfb1731792 | 09-06 | C1: never stall the CPU run-ahead to read a confidence | 1 |
| 0084 | de1da18523 | 09-06 | C1: decide over the confidence MIXTURE, not the latest confidence | 1 |
| 0085 | 1d34c8a608 | 09-06 | C1: pack the confidence buckets against 1.0 | 1 |
| 0086 | 050cbcabc4 | 09-06 | C1: fix the tracer's buffering, and timestamp every row | 1 |
| 0087 | 7473a2ef65 | 09-06 | C1: exact downward prediction, hazard upward, and anti-oscillation | 1 |
| 0088 | 5cf83aa321 | 09-06 | C1: asymmetric switch margin (the up estimate is the extrapolated one) | 1 |
| 0089 | 0794e3f51f | 09-06 | N1 stage A: W4A16 NVFP4 GEMV kernel, quantiser, numerics check and FP8 head-to-head bench | 3 |
| 0090 | 29d7425c46 | 09-06 | N1: tighten the tile sweep, add a scale-broadcast fallback | 2 |
| 0091 | 94ca6842d1 | 09-06 | N1: fix two over-strict assertions in the numerics check | 1 |
| 0092 | b20633a9b1 | 09-06 | N1: kernel compiles clean; add a CPU-only Triton front-end check | 3 |
| 0093 | 173f0e6bc6 | 09-06 | N1: apply the bit trick's 2^14 in the epilogue | 1 |
| 0094 | 0a293bca95 | 09-06 | N1 stage A: kernel measured at 83.3% of roof, 1.487x over FP8 at the draft head | 1 |
| 0095 | b976aab485 | 09-06 | N1 stage A integration: SGLANG_MTP_LMHEAD_NVFP4=1 | 2 |
| 0096 | 5af9a8b6f3 | 09-06 | N1 stage B: SGLANG_LMHEAD_NVFP4=1 for the target lm_head, and a Parameter fix | 1 |
| 0097 | 844c7d211e | 09-06 | N1: make the wide-M fallback CUDA-graph safe | 2 |
| 0098 | 3390966093 | 09-06 | N1 stage C: NVFP4 for the GDN qkvz in_proj and attention qkv | 2 |
| 0099 | 6e75b5fe9f | 09-06 | N1: fix the stage B OOM and the stage C silent miss on attention qkv | 3 |
| 0100 | 4d62a84e77 | 09-06 | P1: singleton-route pruning for the MoE grouped GEMM | 2 |
| 0101 | 1c7630d2ee | 09-06 | P1: pass the -1 sentinel as a tl.constexpr kernel arg | 1 |
| 0102 | 57df694fca | 09-06 | P1: O(N) prune kernel -- device histogram for the singleton count, k x k rank | 1 |
| 0103 | 14d4c4c985 | 09-06 | P2: hand the singleton-route prune to FlashInfer's fused routing prologue | 2 |
| 0104 | 446c801189 | 09-07 | adaptive runtime: run target warmup/autotune before capturing candidate graphs (SGLANG_ADAPTIVE_TARGET_AUTOTUNE=1, default off; 49 tests) | 3 |
| 0105 | 7b4d539f9b | 09-07 | adaptive_confidence: config-driven max_grace_batches, down_margin (per-target), adjacent_only_promotion (inert by default; 54 unit tests) | 2 |
