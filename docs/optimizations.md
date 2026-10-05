# Adopted optimisations

This page covers every change that went into the production launcher
(`serve-fast.sh`) for single-stream decode of Qwen3.8-Flash-Next
(`RadixArk/Qwen3.8-Flash-Next-NVFP4`) on one RTX PRO 6000 Blackwell Max-Q.
The base is the SGLang fork `jpezzulli/sglang-rtxpro6000`. The changes are:

* a patch series of about 105 commits on that fork (121 since 2026-10-02);
* six patches to FlashInfer 0.6.17's C++ sources (eight since 2026-10-02);
* one patch to FlashInfer's vendored GDN WY decode wrapper;
* since 2026-10-02, one patch to FlashInfer's JIT loader (`FLASHINFER_P2_NO_NINJA`, load a copied cache as-is).

Sections A-G describe the 2026-09-08 state. Section H adds the 2026-10-02 round.

Conventions:

* **W4 / W8 / W16**: NEXTN linear-chain speculation with 3 / 7 / 15 draft
  steps and a 4 / 8 / 16-token verify.
* **wa**: the adaptive-width profile.
* Workloads: `code-edit`, `prose-en`, `prose-ja`, `agent-loop` (the public
  prompts in `bench/workloads/`). "Needle" is the synthetic 18.5k-token
  retrieval test in `bench/prof/needle_test.py`; "sanity" is the
  degeneration check in `bench/prof/sanity_gen.py` (its prompt is in that
  file). A triple
  "a / b / c" of t/s is code-edit / prose-en / agent-loop unless stated
  otherwise.
* "Acceptance" = tokens emitted per verify, including the bonus token.
* All numbers are greedy unless stated otherwise.
* Per-kernel timings are CUPTI medians from in-server traces.
* Client t/s is noisy by ±10-20 % (see `measurement.md`), so kernel-level
  numbers carry the decision wherever they exist.
* Commit hashes refer to the fork's performance branch. Spec names refer to
  the benchmark repository's `lab-notes/` directory.

Contents:

* A. Speculative-decoding side
* B. Dense-weight GEMV and quantisation
* C. Kernel and glue fusion, launch reduction
* D. FlashInfer MoE C++ patches
* E. MoE singleton-route pruning (P1/P2) and the quality audit
* F. Adaptive width
* G. Runtime and operations
* H. The 2026-10 round: fixed-cost cuts, sampling fidelity, rejection sampling, 524k context
* Appendix: production flags

---

## A. Speculative-decoding side

### A1. Reduced draft vocabulary (FR-Spec-style hot-token map): `hot_32768`, then the blended `hot2_49152`

* **Why.** The stock EAGLE pipeline reads the full BF16 lm_head (248,320 ×
  2,560 = 1.27 GB) four times per step: two draft steps, one draft_extend and
  one verify. That cost 3.5-4 ms of an 18-19 ms step. The fork already had the
  plumbing for `--speculative-token-map`: the draft head is sliced to a hot
  vocabulary, and the top-k indices are mapped back through `hot_token_id`.
* **How.**
  * `hot_32768` was built from token frequencies on the generation side of
    the author's own agent sessions and chat logs, which are not published
    (neither are any statistics of them).
  * The 33 special tokens are always included.
* **How (v2, 2026-09-04).** `hot2_49152` mixes corpus frequency 50/50 with the
  **target model's own argmax tokens** on self-generated continuations. It
  replaced `hot_32768` in production from 2026-09-04 on.
* **Flag.** `--speculative-token-map <file>`, selected by `TOKEN_MAP=` in the
  launcher. `TOKEN_MAP=none` disables it.
* **Effect, `hot_32768`** (2026-09-02, W4, recommended sampling):

  | Workload | Change |
  |---|---|
  | agent-loop | 174 -> 211 t/s (+21 %) |
  | long-ctx | +16 % |
  | prose-ja | +11 % |
  | code-edit | 210 -> 224 t/s (+6 %) |
  | prose-en | −1 % |

  Step time went from 19 ms to 15-16.6 ms.
* **Effect, `hot2_49152`.** Its acceptance was evaluated offline on the
  author's private data; no result or conclusion of that evaluation is
  published.

  Measured CUPTI step time: W4 10.53 vs 10.64 ms, W16 18.06 vs 18.12 ms.
  * **Conflict:** a later trace re-analysis (`../bench/prof/OVERHEAD_REPORT.md`,
    2026-09-05) attributes +0.26 ms/step (W16) and +0.06 ms (W4) of extra draft
    time to the larger head. The map stayed in production.
  * A later map-size sweep (G3, 2026-09-06) was also run on private data; no
    result or conclusion of it is published. Production kept 49,152 rows.
* **Cost of any restriction.** The map's effect on acceptance was modelled in
  the offline evaluator on private data (not published). The full vocabulary
  was still slower overall (see `rejected.md`).
* **Correctness.** Draft-only change: the target still verifies every token
  with the full vocabulary. With singleton pruning on, a different draft can
  change which experts the target drops (see `measurement.md` §2), so the
  output is not guaranteed identical. Needle 18.5k PASS.
* **Source.** Project log (2026-09-02, 2026-09-04); `../bench/prof/OVERHEAD_REPORT.md`.

### A2. FP8 hot-vocabulary draft head

* **Why.** With the target lm_head in FP8, the first hot-vocab implementation
  dequantised the hot rows to a BF16 copy. For that shape cuBLAS picked a
  300-450 µs WMMA kernel running at about 0.5 TB/s.
* **How.** `eagle_worker_v2.init_lm_head` slices the hot rows directly in FP8.
  It keeps the transposed `[K, H]` layout and the per-channel scales, and
  reuses the target's `Fp8LinearMethod`, so the draft head runs through the
  Triton W8A16 GEMV (B1). This is always on when the target lm_head is FP8
  (`--qwen4-exp-dense-fp8 ...,lm_head`).
* **Effect.** Draft head 55 µs (32k rows) and later 78-81 µs (49k rows), at
  96 % of the read roof.
  * The "v3" configuration (this plus the QSA ring fix plus in_proj_ba) had
    the lowest stall-trimmed GPU time at the time: **13.6 ms/step, versus 14.8
    ms for W8A16 alone and 15.6 ms for the baseline** (2026-09-02, W4). Raw
    t/s hid this because of desktop preemption noise.
* **Correctness.** Needle PASS. Acceptance unchanged within noise.
* **Source.** `c0eb3f7276`, `5058a8582d`.

### A3. QSA pending index-key ring generalisation (enables draft widths > 4)

* **Why.** `QwenSparseAttnBackend` required `speculative_num_draft_tokens ≤
  compress_ratio (4)`. The per-request ring of pending index keys held exactly
  one compression group, indexed by `position % ratio`. A W-token verify
  writes W entries in one forward, so any group completing inside the window
  aliased its members. At W = 4 only `L % ratio == 0` was even correct. The
  guard therefore blocked W8, W16, n-gram chains and adaptive width.
* **How.**
  * Size the ring as `R = ratio × (ceil(Wmax/ratio) + 1)` from the runtime
    maximum draft width.
  * Thread `ring_size` through the metadata builders, the Triton graph-metadata
    kernel (a `RING` constexpr) and the backend guard.
  * Map padding rows to the reserved slot 0 on the host path.
  * No CUDA kernel changes. 6 files, about 100 lines.
* **Effect.** The same day, static W8 ran at code-edit 348-367 t/s (acceptance
  6.9 of 8) and static W16 at 429-433 t/s (acceptance 11.6 of 16). Every wide
  profile and adaptive policy afterwards depends on this.
* **Correctness.** 13 CPU tests. Needle PASS at W4 and W8.
* **Source.** `aa61a26bad`. Project log 2026-09-02.

### A4. Static W8/W16 profiles

* **Why.** The best width depends on the workload. Mechanical code editing
  accepts long chains; prose saturates at about 2.5-3 tokens per verify.
* **How.** The launcher profiles `w4`, `w8` and `w16` pass `--speculative-num-steps
  3/7/15` and `--speculative-num-draft-tokens 4/8/16`.
  * `MAMBA_SLOTS=10` keeps the KV/mamba reservation within budget at W16.
    The default of 24 failed with "no GPU memory for the KV cache".
  * W16 runs at `--mem-fraction-static 0.93`.
  * With top-k = 1 the fork forces `num_draft_tokens = num_steps + 1`, so only
    the steps value matters.
* **Effect.**
  * 2026-09-02: code-edit W4 241-268 / W8 348-367 / W16 429-433 t/s; prose-en
    147-163 / 112-116 / 100-110 t/s.
  * 2026-09-07 width sweep on the full stack (code-edit / prose-en / prose-ja /
    agent-loop):

    | Profile | t/s |
    |---|---|
    | W4 | 363 / 264 / 242 / 312 |
    | W8 | 533 / 231 / 210 / 365 |
    | W16 | 620 / 182 / 153 / 298 |

    So W16 is +70 % on code, W8 is +15 % on agent-loop, and W4 wins on prose.
    These fixed profiles led to the adaptive wa profile (F).
* **Correctness.** Needle PASS at every width.
* **Source.** `lab-notes/WIDTH_SWEEP_0907.md`; project log.

### A5. QSA per-step `seq_lens` table

* **Why.** Before the draft CUDA graph, the multi-step draft backend rebuilt
  `(seq_lens + step + 1).to(int32)` for every draft step. That was 3 eager
  launches per step, 42 at `num_steps = 15`, all on the GPU-idle critical
  path: about 250 µs per W16 decode step.
* **How.** One `[steps, bs]` int32 table (2 launches) feeds every step's
  metadata. Host lengths are derived once and offset per step. Always on.
* **Effect.** W16 draft phase 1,038 -> 998 kernels, step wall 25.3 -> 24.3 ms
  (2026-09-03).
* **Correctness.** Same values by construction. No acceptance change beyond
  run-to-run noise was seen.
* **Source.** `6d77665af7`.

### A6. Draft top-k direct chain with the hot map, plus a fused QSA index lookup

* **Why.** With a token map active, the top-1 chain fell back to a generic
  path: per-step `select_top_k_tokens` plus a separate gather. The QSA shared
  sparse-index lookup for the draft was about 10 small torch ops per draft
  step.
* **How.**
  * `draft_topk1_postprocess` takes an optional `hot_token_id` map and applies
    it inside the finalize kernel, so the direct chain path also covers
    `--speculative-token-map`.
  * One Triton kernel replaces the gather/cast/arange/where/slice-assign
    sequence of the QSA lookup.
  * Always on.
* **Effect** (2026-09-03):
  * W16 draft 998 -> 800 kernels; total 2,937 kernels / 21.7 ms trimmed; wall
    22.5 ms (−11 %).
  * W4 draft 162 -> 128 kernels, with no measurable step change.
* **Correctness.** GPU tests 8 + 9 pass. Needle PASS. No acceptance change
  beyond run-to-run spread was seen: W4 3.65-3.72 / 2.15-2.26 / 3.0-3.07; W16 code 9.4-11.0.
* **Source.** `fcc43cdf2a`.

### A7. MTP dense FP8 (`mtp_dense` category plus the `modelopt_fp4` draft path)

* **Why.** A W16 profile showed the draft phase at 6.6 ms of a 25.4 ms step.
  One draft forward was 70 kernels and about 400 µs, of which BF16 dense GEMMs
  were 111 µs. `--speculative-draft-model-quantization fp8` had no effect.
* **How.** It took three attempts to find the real cause:
  1. A new FP8 category `mtp_dense` (MTP q/k/v/o, fc, shared expert) applied to
     0 modules.
  2. `_mtp_quant_config` forced `quant_config=None` for the MTP layer of the
     serialized NVFP4 checkpoint.
  3. The actual culprit: under `--speculative-draft-model-quantization
     modelopt_fp4` the draft is built with the non-serialised
     `_ModelOptFp4OnlineConfig`, whose `get_quant_method` left every Linear
     unquantised.
  
  The fix marks the draft config. The online adapter then routes the marked
  Linear layers through `Qwen4ExpDenseFp8LinearMethod` (W8A16 GEMV). The MTP
  router stays BF16, because FP8 routing lowered acceptance.
  * Launcher: `--qwen4-exp-dense-fp8 shared_expert,attn,linear_attn,lm_head,mtp_dense`
    and `--speculative-draft-model-quantization modelopt_fp4`. The latter also
    quantises the MTP experts to NVFP4 online.
* **Effect.** W16 code-edit draft phase 4.93 -> 4.18 ms/step. Total trimmed
  21.5 -> 20.6 ms, wall 22.6 ms. The 4 modules shrank from 109 MB to 55 MB
  (2026-09-03).
* **Correctness.** Greedy acceptance showed no shift beyond run-to-run spread:
  code 10.8-11.5, prose 2.7, agent 5.2-5.6. Needle PASS.
* **Lesson.** A quantisation category can silently apply to zero modules.
  Always log the module counts.
* **Source.** `d4faa135d7` (earlier `3a896dfe84`).

### A8. Draft logits written directly into the next-token buffer

* **Why.** `_copy_logits_to_buffer` widened the draft head's BF16 logits into
  the shared FP32 buffer with a separate kernel: 14 launches and about 38 µs per
  W16 step.
* **How.** `w8a16_gemv` / `bf16_gemv` accept `out=` and store
  `acc.to(bf16).to(out dtype)`. The BF16 rounding is kept, so the buffer holds
  exactly the old bytes. `LogitsProcessor` offers the buffer only when every
  intermediate step is a no-op for the batch. Flag `SGLANG_DRAFT_LOGITS_OUT=1`.
* **Effect.** −23 µs/step (W16).
  * **Note:** a later trace analysis (`lab-notes/DRAFT_GLUE_SPEC.md`, item DG-5)
    still found a 34.9 µs/step copy with the flag on. That suggests the gate
    declines in some path. Unresolved.
* **Correctness.** Bit-identical at M ∈ {1, 2, 4, 16} (test).
* **Source.** `5a7572cfc0`; `lab-notes/GLUE_E_SPEC.md`.

### A9. R3: MTP embedding-side projection as a precomputed table

* **Why.** The MTP entry adds `fc_embedding(pre_fc_norm_embedding(embed(id)))`,
  which depends only on the token id and fixed weights. Yet it was recomputed
  on every draft iteration: 15 times per step at W16. An external design review
  (2026-09-05) ranked it the largest remaining micro-lever.
* **How.**
  * `SGLANG_MTP_EMBED_TABLE=1` materialises the term for the whole vocabulary,
    248,320 × 2,560 BF16 = 1.18 GB, in 2.2-2.8 s at load.
  * Rows are built one at a time through the same M = 1 cuBLAS call that decode
    uses, because cuBLAS changes kernel (and low-order bits) with M.
  * A post-build check compares 256 random rows against the live path. If any
    row differs, the table is dropped.
  * The table covers the full vocabulary, because the first token of each chain
    comes from the target and can be any id.
  * The table is allocated after the KV pool, so W4 needs its memory fraction
    lowered by about 0.013.
* **Effect** (same session, flag off -> on):

  | Profile | Draft busy (ms/step) | Kernels |
  |---|---|---|
  | W16 | 5.00 -> 4.76 | 745 -> 717 |
  | W4 | 0.62 -> 0.59 | 119 -> 115 |

  The `gemvx` kernel at W16 disappears (−166 µs).
* **Correctness.** 256/256 sampled rows bit-identical (a sample, not every
  row). Needle PASS; acceptance showed no shift beyond run-to-run spread:
  W16 10.49 / 3.02 / 4.66, W4 3.91 / 2.49 / 3.04.
* **Source.** `51afe49c41`; project log 2026-09-05.

### A10. MTP head fine-tuning (v3, then v5)

*The fine-tuned head weights and the training data are **not published**,
because the training prompts came from private sources. Only the method is
described.*

* **Why.** Acceptance is the other factor in t/s. The stock MTP layer is one
  decoder layer (QSA attention, 512-expert MoE with top-10, shared expert, HC;
  about 2.6 B parameters). Its input is the target's final-layer HC state plus
  the next token's embedding.
* **Method.**
  * The target body is frozen. The MTP layer is retrained on the target's own
    hidden states, dumped from the serving stack with a prefill hook.
  * Labels are the target's greedy argmax, with the top-8 logits and the
    log-sum-exp stored for distillation.
  * Loss = (1 − α)·CE + α·(top-k soft-target KL), with α = 0.5 and temperature 1,
    plus a K = 3 self-rollout loss (weight 0.5) so that recursive chain
    positions are trained too. LR 1e-5, about 1,000 steps from the original
    head; the checkpoint from around step 400 is the one that was served.
  * Checkpoints are selected by an offline "renewal" evaluator. It
    replays the server's verify-and-advance process on held-out (private)
    self-generated sequences and reports tokens per verify, the same quantity
    the server measures. Its results, and how well it tracked the server, come
    from private data and are not published (`measurement.md` §3). Server A/B
    on the public workloads was the final gate.
* **v1 and v2** (plain CE; then prose data removed and a rollout loss added)
  showed no server effect. See `rejected.md`.
* **v3 effect** (adopted 2026-09-05 07:40; server A/B, `hot2_49152` on both
  arms, 4 repeats):
  * W4 prose-en acceptance 2.39 -> 2.54 (+6 %), 217 -> 230 t/s.
  * W4 agent-loop 3.03 -> 3.24 (+7 %), 270 -> 289 t/s.
  * W4 prose-ja +2 %; code ±0.
  * W16 neutral.
* **v5** (adopted 2026-09-07 00:25). Same recipe, broader re-extraction:
  more documents and prompt families than v3 (private data; sizes not
  published), partly sampled rather than greedy. The step-600 checkpoint is the
  one that was served; checkpoints were ranked with the offline renewal
  evaluator (private data, no results published).
  * Server A/B against v3, acceptance per workload:

    | Profile | code | prose-en | prose-ja | agent |
    |---|--:|--:|--:|--:|
    | W4 | +0.8 % | +4.5 % | +4.0 % | +3.2 % |
    | W16 | +6.5 % | +7.5 % | +3.1 % | +1.6 % |

    The W4 code figure is from n = 6; an n = 2 run had read −2.3 %.
  * All four switching gates passed. t/s differences were inside session
    noise, so the decision was made on acceptance.
* **Correctness.** The target verifies every drafted token, but with the
  default pruning (P1/P2, and batch-union pruning E) the draft changes which
  experts are pruned, so the target's output can change. That is why the Q1
  audit ran with the v5 head. Needle PASS on all arms.
* **Source.** Project log (2026-09-04 to 09-07); `serve-fast.sh` comments.

---

## B. Dense-weight GEMV and quantisation

### B1. Triton W8A16 skinny GEMV (v1, then v2 and v3 retunes, and the per-stream split-K scratch fix)

* **Why.**
  * The checkpoint's dense modules are BF16: about 7.2 GB of the ~9.8 GB read
    per decoded token.
  * The existing FP8 path (CUTLASS W8A8) was 35 % *slower*: 386 GEMM calls per
    step at about 44 µs each (see `rejected.md`).
  * FP8 Marlin (W8A16) is not in the sgl-kernel wheel for this GPU.
* **How (v1, 2026-09-02).**
  * A Triton GEMV for M ≤ 16 with FP8 E4M3 weights, per-output-channel FP32
    scale, BF16 activations and FP32 accumulation.
  * Hooked into `Fp8LinearMethod.apply` behind `SGLANG_FP8_W8A16_GEMV=1`.
  * Categories: `--qwen4-exp-dense-fp8 shared_expert,attn,linear_attn,lm_head`.
    The HC mixers are excluded, because their fused BF16 kernel was faster at
    the time. The QSA indexer is hard-excluded for quality.
  * Microbench: 10,240 × 2,560 in 18.9 µs (1.38 TB/s); lm_head 248,320 × 2,560
    in 420 µs versus 860 µs for BF16 cuBLAS; relative error about 1e-3.
    * Square 2,560 × 2,560 was slower than BF16 (9.8 vs 8.0 µs). Small
      matrices do not benefit.
* **Effect v1** (W8A16 plus `hot_32768`, W4, recommended sampling): agent-loop
  **244 t/s (+40 %)** / prose-ja 165 (+40 %) / long-ctx 203 (+33 %) / code-edit
  261 (+24 %) / prose-en 163 (+24 %). Step 12.5-14 ms.
* **v2** (`1eb2831e5c`, 2026-09-03).
  * Per-shape tile table.
  * An **in-launch fix-up split-K** for small-N shapes: the last CTA per
    n-block reduces the FP32 partials, and the counters reset themselves so
    CUDA-graph replay is safe.
  * In-server, the W8A16 family went 6.99 -> 5.81 ms/step at W16
    (shared_down 7 -> 4.1 µs, gate_up 7.9 -> 6.6, out_proj 14 -> 13.1).
* **v3** (`a63e00ba95`, `257b3adb6c`, `61cf5390bc`).
  * The v2 shape table **never matched in the server**. `Fp8LinearMethod`
    passes `[N, K]`-contiguous weights (stride `(K, 1)`); the table was keyed
    for `[K, N]`. So every FP8 shape had been running the planner's fallback
    tile.
  * That was also why the tuning microbenchmarks had disagreed with the server
    by 1.2-3×.
  * v3 first reproduced the in-server medians to within ±22 % on every grid,
    then retuned per shape and per M bucket. M = 1 uses a broadcast (no
    `tl.dot`) path.
  * Two shapes were pinned to the server-measured winners after an A/B:
    in_proj_ba, and attention qkv at M = 16.
  * Effect: W8A16 total 5.0 -> 4.7 ms/step at W16, and −0.3 ms at W4.
    Per shape: qkvz 32.4 -> 29.9 µs, draft head 65 -> 55, shared down 4.1 -> 3.8.
* **Split-K scratch bug** (`ac2e8cb9d2`, 2026-09-04).
  * With every flag on, output degraded to `A!!!!` (NaN, then argmax 0).
    Bisection isolated `SGLANG_ROUTER_GEMV=1`.
  * Cause: the MoE block runs the shared expert on an alternate stream,
    concurrently with the router and routed experts. Two split-K launches
    shared one scratch buffer and one counter array and corrupted each other.
  * Fix: a `scratch_slot(1)` context for alt-stream work, with both slots
    preallocated before graph capture (`b5301df256`).
  * Lesson: fix-up-style split-K needs per-stream workspaces.
* **Correctness.** Per-shape verification against an FP32 reference, plus a
  bit-identical graph replay. Needle PASS. No acceptance change beyond
  run-to-run noise was seen.
* **Source.** Commits above; project log 2026-09-02/03.

### B2. Triton `in_proj_ba` GEMV

* **Why.** The tiny GDN `in_proj_ba` projection (96 × 2,560 BF16, about 0.5 MB)
  made cuBLAS pick a kernel of about 39 µs.
* **How.** `SGLANG_GDN_BA_TRITON_GEMV=1` routes it through the skinny BF16
  Triton GEMV. The first patch hooked `qwen3_next.py`, but this model's GDN
  class is `Qwen3_5GatedDeltaNet` in `qwen3_5.py`, so the flag initially did
  nothing. It was fixed in the second commit.
* **Effect.** About −6 µs per GDN layer (36 layers). After v3 tuning, 6.7 µs per
  call in the server.
* **Correctness.** Same arithmetic class as the other GEMVs. Needle PASS.
* **Source.** `5058a8582d`, `c0eb3f7276`.

### B3. Router gate as a BF16 Triton GEMV

* **Why.** The MoE router (512 × 2,560 BF16, no bias) ran as cuBLAS GEMM plus
  a split-K reduce: 2 kernels × 61 per step, 119 µs.
* **How.** `SGLANG_ROUTER_GEMV=1` uses `bf16_gemv` (same weights, FP32
  accumulate, BF16 logits).
  * A first dedicated router kernel was slower than cuBLAS (7.5 vs 5.0 µs).
  * The first `bf16_gemv` wiring made W4 worse, because the old GEMV launched
    only 16 CTAs at N = 512.
  * After the v3 table it ran at M = 16 in 6.18 -> 4.42 µs.
* **Effect.** −86 µs/step (M = 16), and 63 fewer kernels per step (no
  split-K). It now sits entirely on the alternate stream: 0.1 µs of exclusive
  time out of 340 µs raw at W16.
* **Correctness.** A 4-repeat W16 A/B showed no acceptance effect of this flag
  beyond noise (agent-loop 4.53 with only the router off, versus 4.49-4.85
  otherwise, within noise). The alt-stream scratch bug in B1 was found through
  this flag.
* **Source.** `075d071988`, `65192006d4`, `9e841cb79f`; project log 2026-09-04.

### B4. Shared-expert `gate_up` GEMV + SiLU×up epilogue (W4 only)

* **Why.** The shared expert ran `w8a16_gemv` for gate_up and then a separate
  `act_and_mul`: 48 launches per step at W4 (62 at W16), each reading 2H values
  and writing H.
* **How.** `SGLANG_SHARED_GATEUP_FUSED=1`. One CTA owns columns n and n + H, so
  the same k loop fills two accumulators and SiLU(gate)·up becomes the
  epilogue. The split-K fix-up keeps two partials per (n-block, split). The
  tile was re-tuned: halve the split, double the n-block.
* **Effect.**

  | M | gemv + act | fused |
  |--:|--:|--:|
  | 1 | 5.92 µs | 4.42 µs |
  | 4 | 6.75 µs | 4.99 µs |
  | 16 | 7.07 µs | 4.96 µs |

  At M ≥ 4 the fused kernel is no slower than the plain GEMV alone.
* **Why it is off at W16.** In a 4-repeat W16 A/B on 2026-09-04, agent-loop
  acceptance was 4.78 with only this flag off, against 4.49 with everything on:
  about −7 % on long chains. The step saving was only about 0.1 ms.
  * The fused path is actually *closer* to an FP32 reference (≤ 1.9e-6 vs 3.1e-2
    for the unfused path, which rounds gate and up to BF16 first). The
    acceptance drop is therefore a numerics-sensitivity effect in the long
    chain, not an error.
  * Given the ±15 % acceptance noise at W16, this effect is only moderately
    certain. It was kept off at W16 and in wa (`W16_GATEUP=0`).
* **Correctness.** Within 1 BF16 ulp mean and 4 max of the unfused path. No
  W4 acceptance change beyond run-to-run noise was seen. Needle PASS.
* **Source.** `c964be9ebc`, `081e8d9b09`; project log 2026-09-04.

---

## C. Kernel and glue fusion, launch reduction

Context: by 2026-09-03 a decode step was 2,100-3,200 kernels inside CUDA graphs
with about 0.15 ms of total GPU idle. The cost of a small kernel is its own
ramp, not a launch gap. A deleted kernel saves roughly 1 µs of critical path,
and only if its work is not simply moved elsewhere.

### C1. `A_log` / `dt_bias` view cache

* **Why.** `A_log.detach().float()` built a new tensor object on every verify
  call. That defeated FlashInfer's identity-keyed `_cached_bf16`, so the cast
  kernel ran once per GDN layer per step: 36 launches, about 45 µs.
* **How.** Cache the detached views. Always on.
* **Effect.** −36 kernels/step (about −45 µs).
* **Correctness.** Identical values.
* **Source.** `f30507e31f`.

### C2. Zero correction-bias cache

* **Why.** `fused_topk` called `torch.zeros` for the correction bias in every
  MoE layer: 48 fill kernels per step.
* **How.** Reuse one zero tensor per (N, device). Always on.
* **Effect.** −48 kernels/step.
* **Correctness.** Semantically identical.
* **Source.** `075d071988`.

### C3. Fused FP8 KV store

* **Why.** Writing BF16 K/V into the FP8 E4M3 KV cache took `div_, div_, to,
  to` plus `store_cache`: 5 kernels, 26 call sites per step.
* **How.** `SGLANG_KV_FP8_FUSED_STORE=1`: one Triton kernel for N ≤ 64.
  * 0-d tensor scales are rounded to BF16 first, as torch promotion does.
    This rounding fix was needed for bit-exactness.
  * A later commit accepts token-strided K/V views (verify passes slices of the
    fused qkv buffer), which otherwise fell back to the 5-kernel path.
* **Effect.** State after the glue round (2026-09-03):

  | Profile | Kernels | Trimmed ms | Wall ms |
  |---|--:|--:|--:|
  | W16 | 2,730 | 21.3-22.0 | 23.3 |
  | W4 | 2,118 | 12.6-13.0 | 13.1-13.7 |
* **Correctness.** Bit-exact (10 tests, then 13). Needle PASS.
* **Source.** `075d071988`, `45aab312cb`.

### C4. `causal_conv1d_update` with 64-wide channel blocks

* **Why.** At batch 1 with dim = 10,240, the 256-wide block left only 40 CTAs
  for the serial per-timestep loop. The 16-token verify update took 18.8 µs per
  GDN layer.
* **How.** Use `BLOCK_N = 64` (160 CTAs) when `batch × cdiv(dim, 256) < 128`.
  Override with `SGLANG_CONV1D_UPDATE_BLOCK_N`. Always on.
* **Effect.** 18.8 -> 6.2 µs per layer. W16 0.76 -> 0.26 ms/step.
* **Correctness.** Identical math.
* **Source.** `0297a12070`.

### C5. R4: chain-parallel verify conv1d

* **Why.** The target-verify causal conv1d walks the T draft tokens serially
  inside one CTA. At T = 16 it cost about 2.5× the T = 4 case purely from loop
  length (261 vs 106 µs per step over 36 calls).
* **How.** `SGLANG_GDN_CONV_CHAIN_PARALLEL=1`.
  * With top-k = 1 the draft tokens form a line, so each output position is
    independent given the incoming conv state.
  * A third grid axis covers `BLOCK_T = 4` token tiles. The per-output
    accumulation order, the FP32 accumulator, SiLU and rounding are unchanged.
  * Only the tile-0 CTA reads and rolls the conv state, so there is one writer.
  * The tree path, rolling accept, circular cache and seqlen == 1 keep the old
    kernel.
* **Effect.** In-server per call 6.2 -> 3.4 µs at W16 (−0.10 ms/step); 2.7 ->
  2.5 µs at W4 (−7 µs).
* **Correctness.** 84-85 `torch.equal` cases (outputs, final state,
  intermediate window). Acceptance, needle and sanity PASS.
* **Source.** `56383f08a1`, `cf12af5ddc`.

### C6. GDN strided q/k/v in the vendored FlashInfer WY wrapper

* **Why.** At `T == T_KERNEL` (the W16 verify) the vendored wrapper
  `.contiguous()`-materialised q, k and v. That is three GMEM-to-GMEM copies per
  GDN layer: 108 kernels and about 200 µs per W16 step. The kernel already
  reads through base pointers and strides it rebuilds itself, and it has a
  `qkv_row_stride` knob used on the T ∈ {4, 8} path.
* **How.**
  * A patch to `flashinfer/gdn_kernels/gdn_decode_bf16_wy_output_only.py` in the
    venv. The file is patched in place; a `.bak` is kept.
  * It reads q/k/v straight from the fused conv-output column slices (token
    stride 10,240).
  * It tightens the guard: shared token stride, canonical head/feature/batch
    strides, 16 B alignment.
  * Enabled by `FLASHINFER_GDN_WY_STRIDED_QKV=1`. At W4 the environment variable
    alone engages the existing native path.
* **Effect.** W16 −200 µs/step, W4 −221 µs/step (2026-09-04).
* **Correctness.** Bit-identical at T = 16 and T = 4 on the real shapes. The test
  includes a compile-cache probe, so a silent fallback cannot make the check
  vacuous.
* **Source.** `32fe4f5953` (the patch is shipped as a file and a test);
  `lab-notes/GLUE_E_SPEC.md`.

### C7. R2 + R5: GDN projections write consumer layouts directly

* **Why.**
  * (R5) The fused split/reshape/cat kernel after the GDN projections is a pure
    row compaction. It exists only because q/k/v, z, b and a need their own
    row strides: 36 launches per verify step, about 2.8 µs each.
  * (R2) With `gdn_mtp_cache_mode=none`, verify copied a/b into the persistent
    per-layer recovery stash: 72 device-to-device copies per step.
* **How.**
  * `w8a16_gemv` / `bf16_gemv` gained a two-destination epilogue (`out2=`,
    `split_n=`). Only the store changes, not the k loop, accumulation order,
    scale index or rounding.
  * `SGLANG_GDN_PROJ_DIRECT_LAYOUT=1`: the qkvz projection writes `mixed_qkv`
    and `z`, and `ba` writes `b` and `a`, directly.
  * `SGLANG_GDN_AB_STASH_DIRECT=1`: the backend hands the stash slice to the
    producing kernel. Buffer addresses are unchanged, so the captured recovery
    graphs keep their pointers.
* **Effect.** −108 kernels/step; −0.159 ms at W4 and −0.124 ms at W16
  (attributed).
  * The in_proj_ba split costs +0.9 µs/call at M = 16, because the 48-column
    split straddles a 32-wide tile. A "single store for non-straddling blocks"
    variant was measured slower and reverted (`ccf9d2faba`). A split-aware grid
    (H1-C) recovered only about 0.2 µs and stays off.
* **Correctness.** `torch.equal` on real layer-0 weights at M = 1/4/16 for all
  four outputs and the stash. Acceptance and needle PASS.
* **Source.** `64b662b5f6`, `5ec47f104c`, `50e1e488b8`, `7607c2c977`, `cdd86329b8`,
  `ccf9d2faba`.

### C8. Three-kernel HC norm+mix (MIX2), then FP8 mix weights

* **Why.** Each hyper-connection boundary normalises 4 residual branches
  (4 × 2,560) and applies a low-rank mix with rank 320. That is two 6.5 MB BF16
  matrices, about 97 calls per step. The old chain was a grouped RMSNorm (2.1 µs)
  plus a persistent mix kernel (17.7 µs) = 19.8 µs per call, and the persistent
  kernel was barrier/atomic-bound.
* **How (MIX2, `SGLANG_HC_MIX2=1`).** Three non-persistent kernels:
  * K0: statistics, normalise, zero the accumulator;
  * K1: split-K down-projection with atomics;
  * K2: up-projection, gate and mean.
* **How (FP8, `SGLANG_HC_MIX2_FP8=1`).**
  * E4M3 weights with one FP32 scale per output row, quantised at weight-load
    time. They cannot be quantised during capture, which would bake the
    quantiser into the graph.
  * K1 scales its FP32 partial before the atomic add; K2 scales its finished
    accumulator.
  * FP8 alone reached only 11.4 µs. The gain came from the wider n-block that FP8
    allows: 64 low-rank columns per 2 KB tile instead of 16, which cuts K1's CTA
    count by 4×.
* **Effect.**
  * MIX2: 19.8 -> 14.3 µs per call (K0 1.6 + K1 6.3 + K2 6.4). W16 −0.49
    ms/step, W4 −0.26 ms. That is close to the ~11.5 µs needed just to stream
    13 MB.
  * FP8: 13.7 -> 9.9 µs per call (−27.6 %); expected W16 −0.45, W4 −0.26 ms.
* **Correctness.**
  * `normed` is bit-exact.
  * Mixed output: ≤ 1e-3 (MIX2); ≤ 2 BF16 ulp of output RMS, 59 % of elements
    bit-identical (FP8).
  * A W16 4-repeat A/B with only HC FP8 off gave acceptance 9.7 / 2.51 / 4.44
    against all-on 10.2 / 2.48 / 4.49, i.e. no difference beyond the spread
    of that single A/B.
  * K1's atomics make the mix non-deterministic run to run, even unfused.
* **Source.** `b20ce6eebb`, `f685bf517d`, `081e8d9b09`;
  `lab-notes/HC_MIX2_FP8_SPEC.md`.

### C9. R1 / R6 / R7: HC boundary restructuring

* **Why.** At each HC boundary, the combine's gate depends only on the normed
  residual and fixed weights, yet it ran after the block output was ready. It
  cost 2.0 µs × 102 boundaries per step at W4 on the critical path. The MoE's
  shared-expert gate and join (`fused_gate_sigmoid_mul_add`, 1.7-2.1 µs × 51)
  were also on the critical path. Finally, the combine's apply writes the branch
  residual, and the next K0 immediately reads it back.
* **How.**
  * **R1** (`SGLANG_HC_GATE_EARLY=2`): launch the same split gate kernel right
    after the mix. This mode is bit-identical. Mode 1, which fuses the gate into
    K0, was a measured *loss* (+114 µs/step at W4: K0 1.5 -> 4.3 µs because its
    grid is small).
  * **R6** (`SGLANG_SHARED_GATE_EARLY=1`):
    * Compute the shared-expert gate inside the dual-stream region, on the
      short shared-expert branch.
    * Hand `(shared_output, gate)` to the HC combine, whose apply forms
      `bf16(routed + gate·shared)` and the residual update in one kernel.
    * This is bit-identical.
    * Held only at decode widths (≤ 32 rows). An 8k prefill chunk otherwise
      pinned 2 GB and caused an OOM (`d723dc04c7`).
  * **R7** (`SGLANG_HC_APPLY_MIX_FUSED=1`): fold the combine apply into the next
    mix's K0 as a prologue, since K0's grid is one CTA per (row, branch), which
    is exactly what the apply writes. This covers the attention -> MoE boundary
    inside a layer. Result is within 1 ulp at M = 16.
* **Effect.** Kernels W4 1,857 -> 1,806, W16 2,508 -> 2,445. Attributed −100
  µs/step (W4) and −90 µs (W16).
* **Correctness.** `torch.equal` for R1 (mode 2) and R6; R7 ≤ 1 ulp. Acceptance
  in band: W4 3.78 / 2.54 / 3.23, W16 10.59 / 2.97 / 4.99. Needle and sanity
  PASS.
* **Source.** `f8cc4d58cf`, `b62750e80c`, `0b0f8f32ec`, `610ddd5d60`,
  `2cdfe3ef69`, `8d3cf2306c`, `d723dc04c7`. The idea came from an external
  design review (2026-09-05).

### C10. H2: layer-to-layer HC apply folded into the next layer's K0

* **Why.** R7 covered only the boundary inside a layer. The MoE combine feeding
  the *next* layer's attention mix is the same shape of work, but it must
  survive a trip through the model loop.
* **How.** `SGLANG_HC_LAYER_APPLY_FUSED=1`.
  * `_PendingHCCombine` carries the combine inputs to the next layer's
    `combine_then_mix`, which runs the apply as K0's prologue, including R6's
    shared-expert fold (`FUSE_SHARED`, with the same intermediate BF16 rounding).
  * Two boundaries keep the plain apply:
    * layer 1, the only per-layer-embedding (PLE) layer, whose query reads the
      combined residual;
    * the last layer, whose `hc_hidden_states` the MTP head reads.
  * That leaves 46 of 48 boundaries fused. Row loads were reordered ahead of
    the gate reduction (−0.18 µs per launch at M = 16).
* **Effect** (same build, flag off -> on):

  | Profile | apply launches | K0 | Net | Verify kernels |
  |---|---|---|---|---|
  | W4 | 51 -> 5 (64.6 -> 6.0 µs) | 193 -> 210 µs | **−42 µs/step** | 1,399 -> 1,353 |
  | W16 | 63 -> 17 | | **−47 µs/step** | 1,447 -> 1,401 |
* **Correctness.** Real weights, M ∈ {1, 4, 16} × {BF16, FP8} × {with, without
  shared fold}: `applied`, `normed` and the gate partials are bit-identical in
  12/12 cases. `mixed` differs in 2/12, which is the known K1 atomics
  nondeterminism (5/5 repeats of the unfused call already differ). Acceptance
  in band, needle 18.5k and sanity PASS. Not compatible with
  `--enable-torch-compile`.
* **Source.** `81d4e5d8e5`, `b9cf17830b`, `bb3c5ac76c`.

### C11. H1-B: GDN gated RMSNorm folded into the `out_proj` GEMV

* **Why.** The GDN block's `RMSNormGated` (per token and head, head_v_dim 128)
  was a 192-CTA one-warp kernel. It cost 1.9-2.0 µs in the server × 36 per step,
  almost entirely on the critical path: 57 µs/step exclusive at W4, 68 at W16.
* **How.** `SGLANG_NORM_INTO_GEMV=1`.
  * With `BLOCK_K == head_v_dim`, each GEMV A-tile is exactly one norm group,
    so the row statistics are recomputed from the tile already in registers.
    The `z` tile is L2-resident.
  * Applied only for M ≥ 4. At M = 1 the broadcast path is not fused, and fusing
    there was a loss.
* **Effect.** −1.06 µs/call median × 36 = **−36 µs/step** (−0.35 % W4, −0.21 %
  W16). −36 kernels per step. The whole-step time could not resolve it: MoE GEMM
  drift alone is ±0.5 ms between sessions.
* **Correctness.**
  * The fold itself is 0 ulp.
  * The profitable tile (BLOCK_N 64, SPLITS 4 instead of 32/6) re-partitions
    split-K. That moves ≤ 6 ulp on 0.03 % of elements at M = 16 (≤ 1 ulp at
    M = 4), the same class of change as the existing per-M tile table.
  * MODE=full with the flag on:

    | Profile | code-edit | prose-en | agent-loop |
    |---|---|---|---|
    | W4 | 373.7 t/s / 3.81 | 266.1 / 2.59 | 326.6 / 3.27 |
    | W16 | 578.8 / 9.89 | 183.3 / 3.10 | 288.2 / 4.90 |

  * Needle PASS.
  * With both H1 flags off, the output is 30/30 byte-identical to the pre-H1
    build.
* **Source.** `d042e1845f`, `2f7130aff9`, `ef31f26346`.

### C12. QSA graph row-metadata kernel made page-parallel (DG-1a)

* **Why.** `_qsa_graph_row_metadata_kernel` launched one program per row with
  `num_warps=1`. It walked the whole page table (4,096 pages at page size 64 and
  262,144 context) in 32 serial strided trips. So it cost about 10 µs warm and
  21 µs cold regardless of row count, once per draft step plus twice more per
  step: 199 µs of a 20.5 ms W16 step.
* **How.** `SGLANG_QSA_META_PAGE_PARALLEL=1`.
  * The page dimension moves to `program_id(1)`: one program per page block.
  * Scalar stores are guarded to `program_id(1) == 0`, so the same values go
    to the same addresses.
  * With the flag off, the TTIR is the previous kernel's (only a hoisted
    invariant differs).
* **Effect.** W16 240 -> 49 µs/step, W4 103 -> 13 µs/step.
* **Correctness.** Bit-exact by construction (identical values to identical
  addresses), and checked against the serial launch on four shapes. Needle
  PASS; no acceptance change beyond run-to-run noise was seen.
* **Source.** `f21ef2f0f6`; `lab-notes/DRAFT_GLUE_SPEC.md`.

### C13. Programmatic dependent launch (PDL) for the fork's Triton kernels

* **Why.** The fork's JIT CUDA kernels, and some upstream Triton kernels,
  already carried PDL, which lets a kernel's launch and prologue overlap its
  predecessor's drain. The fork's own Triton kernels did not. On the W4 verify
  side they are a large share:
  * `_w8a16_gemv_kernel`: 229 of about 1,650 kernels, 3.46 ms;
  * `_w8a16_gemv_silu`: 48 kernels;
  * the three HC mix kernels: 291 kernels, 1.37 ms.
* **How.** `SGLANG_TRITON_PDL=1` gates both halves: the `launch_pdl` attribute
  and a `USE_PDL` constexpr that decides whether `griddepcontrol` instructions
  are traced. With the flag off, PTX and launch config are identical to before.
  * Waits go at the top of kernels whose loads all depend on the predecessor.
  * Triggers go after the last store. K0's trigger follows the accumulator
    clear, so K1's atomics cannot land first.
* **Effect.**
  * W4: −0.63 / −0.56 ms per step (−5.5 %), all in verify; kernel count
    unchanged.
  * W16 was neutral in a noisy first test. It was re-measured headless on
    2026-09-06 with compositing off, 2 × 2 on/off: **−0.2 ms (−1.2 %), same sign
    in both pairs**. After that it was enabled for all profiles.
  * PDL buys overlap, not only launch latency: the W4 gain was about 4× the
    launch-latency model's 162 µs.
  * Side effect at W16: the attention kernel's exclusive time dropped from
    189 + 217 µs to 16 + 10 µs.
* **Correctness.** 57-tensor bit-exactness check, full-step greedy equality
  harness, needle and sanity PASS.
  * Caveat: with PDL on, per-kernel durations of small kernels that overlap a
    predecessor include spin time. Compare `busy_ms`, not per-kernel medians.
* **Source.** `50b3487a04`, `7081a094f4`, `c0a3a4c452`, `b819f27cfa`, `661e87a643`,
  `d8f89935c0`, `f44d35d20a`, `e58cf349d3`; `lab-notes/MEGAKERNEL_SPEC.md` (K1
  measurements).

---

## D. FlashInfer MoE C++ patches (FlashInfer 0.6.17, sm_120 CUTLASS fused MoE)

These patches apply in place to the venv's
`flashinfer/data/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh`
and `.../cutlass_kernels/include/moe_kernels.h`, in this order: `a0`, `a3`,
`g2-1`, `g2-2`, `g1-pack-only`, `p2`. JIT rebuild takes about 100 s.

`uv` installs by hard-linking, so the files initially shared inodes with other
venvs and the uv cache. Break the links after patching.

The MoE chain per call before these patches was 8 kernels:

* 3 prefix-sum kernels for routing;
* `expandInputRows` (permute plus NVFP4 re-quantisation);
* `computeStrides`;
* CUTLASS GEMM1;
* `doActivation` (SwiGLU plus FP4 re-quantisation);
* CUTLASS GEMM2, whose epilogue does the finalize scatter.

Glue was about 21.5 µs per call, over 51 (W4) and 63 (W16) MoE calls per step.
sm_120 CUTLASS offers no split-K and no M tile below 128, so the GEMM itself had
little to tune (see `lab-notes/MOE_SMALLM_SPEC.md`).

### D1. A0: fused routing prologue enabled for top-10 / 512 experts

* **Why.** FlashInfer has a one-kernel routing prologue
  (`fusedBuildExpertMapsSortFirstToken`), but it was disabled for this model:
  * `expert_log = 10` fell outside the dispatch (≤ 9);
  * top-k = 10 was not in the switch {1, 2, 4, 6, 8}.
  
  So the 3-kernel path ran.
* **How.** Extend both gates.
  * Enabling them naively was *slower*: 15.05 µs versus 9.4 µs for the 3-kernel
    path. `cub::BlockRadixRank` needs 65,696 B of temp storage at 10 radix bits.
  * The patch uses `cub::BlockRadixRankMatch` (4,256 B) for ≥ 10-bit digits:
    **3.36 µs/call**. Configurations with `expert_log ≤ 9` keep the stock ranker
    bit-for-bit.
* **Kill switch.** `FLASHINFER_MOE_FUSED_PROLOGUE=0`.

### D2. A3: `computeStrides` folded into the prologue

* **How.** The prologue re-runs the cheap rank in every block
  (grid = ceil(E/32)) and publishes expert offsets through shared memory, so the
  TMA-descriptor phase stays parallel with no grid barrier.
* **Kill switch.** `FLASHINFER_MOE_FUSED_STRIDES=0`.
* **Effect, A0 + A3** (2026-09-05):
  * Routing bookkeeping W4 0.57 -> 0.41 ms/step, W16 0.68 -> 0.45 ms/step.
  * Kernels W4 2,017 -> 1,864, W16 2,728 -> 2,539.
* **Correctness.** Deterministic mode (`SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0`,
  `--autotune-cache none`): upstream, A0 and A0 + A3 are bit-identical at
  T = 1, 2, 4, 8, 16 on a real 512-expert layer. Production mode is within the
  unpatched path's own run-to-run envelope (about 1e-4). Acceptance, needle and
  sanity PASS.

### D3. G2-1: finalize memset folded into `doActivation`

* **How.** `doActivationKernel` zeroes the fused-finalize output in its own
  grid, before `cudaGridDependencySynchronize()`, so GEMM2 skips its
  `cudaMemsetAsync`. Default on; kill switch `FLASHINFER_MOE_FOLD_MEMSET=0`.

### D4. G2-2: `expandInputRows` folded into the prologue

* **How.**
  * The prologue grid becomes ceil(E/32) descriptor blocks plus
    `num_tokens × k` row blocks.
  * Row blocks read the inverse permutation from shared memory, so there is
    no binary search and no global round trip.
  * The row is staged into registers so all ten 16-byte loads are in flight.
  * NVFP4 quantisation keeps the upstream lane pairing, so scale factors are
    bit-identical.
  * Default on; kill switch `FLASHINFER_MOE_FOLD_EXPAND=0`.
* **Effect, G2-1 + G2-2** (4 in-server runs):
  * Glue 14.16-14.34 -> **10.34 µs per MoE call (−3.90 µs)**, GEMMs unchanged.
  * Launches W4 1,698 -> 1,596, W16 2,337 -> 2,211.
  * **−0.19 ms/step (W4), −0.25 ms/step (W16).**
  * Measured on raw time the prologue got slower (9.5 -> 11.9 µs). The
    exclusive-time view shows the true gain, since MoE-glue exclusive time fell
    446 -> 377 µs (W4) and 654 -> 549 µs (W16). A raw-time review would have
    rejected this change.
* **Correctness.** Bit-identical in deterministic mode for every flag
  combination (real 512-expert layer, T = 4 and 16, 4 calls each).
  Production mode within the control envelope (max_abs 2.441e-4, the same as
  off vs off). Acceptance and needle PASS.

### D5. G1-pack: pack the active experts into CUTLASS groups [0, D)

* **Why.**
  * The GEMM1 main loop already streams at about 100 % of the measured read roof
    (`GEMM1 ≈ 9.4 + 1.036·D µs`). The supposed "wave-quantisation staircase" did
    not exist.
  * The remaining loss is a per-call ramp. Part of it is the persistent tile
    scheduler's linear scan over all 512 groups.
* **How.**
  * Each descriptor block computes rank(e) among the active experts and writes
    expert e's group at index rank(e).
  * The group count drops from 512 to `expanded_num_rows` (40 at W4, 160 at
    W16).
  * Everything CUTLASS indexes per group moves with it: alpha, SF pointers and
    layouts, and the finalize epilogue's source-token and router-scale arrays.
  * Opt-in: `FLASHINFER_MOE_PACK_GROUPS=1`, which the launcher sets.
  * A bug was fixed during the rebase onto G2: an active expert with an id above
    D must also empty its own slot.
* **Effect** (on top of G2):
  * W4 −2.55 ± 0.14 µs/call = **−0.125 ms/step**.
  * W16 −1.32 ± 0.11 µs/call = **−0.083 ms/step**.
  * G2-2 hides the rank computation inside the prologue, so the gate threshold
    could drop from 4 to 1.
* **Correctness.** Bit-identical in deterministic mode (12/12 tensors,
  T = 1/4/16). Production mode max_abs 2.441e-4, the same as the off-vs-off
  control.
* **Combined G2 + G1-pack:** W4 −0.31 ms/step, W16 −0.33 ms/step.

**Source for D1-D5.** `../patches/README.md`,
`lab-notes/MOE_SMALLM_SPEC.md`, `lab-notes/G1_LOG.md`, `lab-notes/G2_LOG.md`. The sixth
patch (P2) is described in E.

---

## E. MoE singleton-route pruning (P1, then P2 inside the FlashInfer prologue)

*This is the only adopted change that alters the target model's arithmetic.*

### E1. Why

* Expert-weight bytes are the largest item in the step: the MoE GEMM was 31 %
  (W4) and 39 % (W16) of critical-path time on 2026-09-06.
* They scale with **D**, the number of distinct experts read per layer per
  verify call. Each distinct expert costs about 1.2-1.6 µs of GEMM1 + GEMM2.
* A route census (2026-09-06, about 12-16k calls per workload) showed:
  * **The router is nearly flat.** Mean weight by rank goes from about
    0.18-0.21 (rank 1) to 0.06-0.07 (rank 10).
  * **Many experts are singletons**, used by exactly one row of the verify
    batch: about 34-45 of about 57-73 at W16. The singleton rate rises with rank.
* Dropping a route to an expert that another row also uses saves nothing.
  Dropping a low-weight singleton route saves a whole expert read.

### E2. How

* The rule drops a route when all of these hold:
  * it is the only route to its expert in the call;
  * its routing weight is below τ;
  * it is not the row's top-1 (`MIN_RANK = 1`);
  * the batch has 2-64 rows, so the T = 1 draft is never touched.
* Surviving weights are **not** renormalised. Production τ = **0.08**.
* **P1** (`prune_singleton.py`): a separate Triton kernel on the [M, k] top-k
  tensors, in place, with no host sync and CUDA-graph safe.
  * Dropped routes get id −1. That is not `num_experts`, because that slot is
    the fused shared expert. FlashInfer sorts −1 past the valid tokens and the
    finalize skips it.
  * The first kernel was an O(N²) compare at 21 µs/call, which ate the whole
    saving. The rewrite is O(N) (a per-row k × k rank plus an atomic histogram)
    at 2.5-4.2 µs/call.
  * Even so, inserting any kernel in front of the routing prologue cost the
    prologue its ~54 % overlap with preceding kernels. At W16 this cancelled
    most of the gain.
* **P2** (`p2-prune-in-prologue.patch`, the sixth FlashInfer patch): the same
  rule is evaluated inside `fusedBuildExpertMapsSortFirstTokenAndStridesKernel`,
  which already holds the top-k ids in registers.
  * A block-local shared histogram gives the pre-prune per-expert count.
  * A k × k register compare gives the rank.
  * Dropped routes are re-keyed to the "not on this node" bucket *before* the
    radix rank, so permutation, offsets, G1 packing and G2-2 row expansion all
    see the pruned set unchanged.
  * Block 0 writes −1 / 0 back for the unfused finalize.
* **Flags.** `SGLANG_MOE_PRUNE_SINGLETON_TAU=0.08` with
  `SGLANG_MOE_PRUNE_IN_PROLOGUE=1`. These export the `FLASHINFER_MOE_PRUNE_*`
  variables and turn the P1 kernel into a no-op. Without the C++ patch, the
  in-prologue path prunes nothing. `PRUNE_TAU=0` disables pruning. Gated off
  when expert parallelism > 1.

### E3. Effect

* **Implied D** after pruning: W4 about 29 -> 19-20, W16 about 68-76 -> 50-52.
  Routing mass dropped per row: about 7 % at W16 and 14 % at W4.
* **P1 at W4** (2026-09-06, τ = 0.08):
  * Step 10.53 -> 9.37 ms (code-edit, −11.0 %) and 10.46 -> 9.46 ms (prose-en,
    −9.6 %).
  * t/s code +9.9 %, prose-en +5.1 %, agent-loop +14.5 %, prose-ja +13.3 %.
  * Adopted for W4 and wa at 21:30. Post-merge smoke: W4 388 / 272 / 346 t/s
    (before: 363 / 254 / 317); wa 638 / 269 / 306.
* **P1 at W16**: GEMM −1.2 ms/step, but the lost prologue overlap left a net of
  −0.6 % to −4.0 %. On hold.
* **P2** (same hour, off -> P2):
  * Prologue exclusive time W16 256.5 (off) / 484.2 (P1) / 290.4 µs (P2).
  * **W16 step 18.4 -> 16.7 ms (−9.5 % / −8.6 %)**; **W4 −8.9 % / −11.7 %**.
  * t/s and acceptance:

    | Profile | Workload | t/s | Acceptance |
    |---|---|---|---|
    | W16 | code | 492 -> 543 | 9.12 -> 8.92 |
    | W16 | prose-en | 162 -> 179 | |
    | W16 | agent | 250 -> 310 | 4.65 -> 5.11 |
    | W16 | prose-ja | 147 -> 177 | |
    | W4 | code | 357 -> 400 | |
    | W4 | prose-en | 259 -> 287 | |
    | W4 | agent | 314 -> 355 | |
    | W4 | prose-ja | 235 -> 258 | |

* **Production smoke with P2 on all profiles** (2026-09-06 23:42, W4 / W16 / wa):
  * code-edit 399 / 671 / 595;
  * prose-en 280 / 186 / 282;
  * agent-loop 349 / 314 / 326 t/s.
* **2026-09-07 exclusive-time map.** P2 accounts for all of the 09-06 -> 09-07
  improvement at W4 and 83 % of it at W16. GEMM1 + GEMM2 exclusive time fell
  836 µs (W4) and 1,484 µs (W16).

### E4. Correctness evidence

* **P2 vs P1.** Same routes dropped (753 at W16 and 474 at W4 in 48 recorded
  calls), and the MoE output is **bit-identical 48/48** in deterministic mode.
  τ = 0 on the patched build is bit-identical to the unpatched build.
* **Functional checks.** Synthetic needle 18.5k PASS (×4 and later).
  Acceptance on the public workloads stayed inside the ±4-6 % run-to-run band.
  Code-edit greedy output matched the unpruned build for 256 tokens.
* **Greedy agreement is not a usable gate.** The MoE finalize epilogue
  scatter-adds with float atomics, so even the unpruned build does not reproduce
  its own prose output past the first few tokens.
* **Q1 quality audit** (`lab-notes/Q1_QUALITY_AUDIT.md`, 2026-09-07). Setup: wa
  profile, v5 head, greedy, thinking off, four arms:
  * `prod` (τ = 0.08);
  * `noprune` (τ = 0);
  * `legacy` (τ = 0 with the day's optimisations off);
  * `prod2` (a repeat of prod).

  | Benchmark | prod | noprune | legacy | prod2 |
  |---|--:|--:|--:|--:|
  | GSM8K (1,319) | 88.63 % | 89.31 % | 89.46 % | 89.31 % |
  | MMLU (14 × 100) | 85.36 % | 85.07 % | 84.93 % | 84.86 % |
  | HumanEval (164) | 96.95 % | 96.34 % | 96.95 % | 95.73 % |
  | JCommonsenseQA (1,119) | 97.14 % | 97.14 % | 97.14 % | 97.14 % |
  | Needle | 6/6 | 6/6 | 6/6 | 6/6 |

  * McNemar p ≥ 0.266 for every paired comparison (smallest: GSM8K prod vs
    legacy, p = 0.266; prod vs noprune p ≥ 0.362). The prod-vs-noprune
    differences are the same size as the prod-vs-prod2 scatter.
  * On GSM8K questions that finished in both arms (n = 1,134), prod 98.41 % vs
    noprune 98.59 % (p = 0.625). About 11 % of GSM8K chains hit the 512-token
    cap in each arm.
  * Conclusion: no statistically significant difference (all p > 0.05; rough
    detection limits ±1.5 pp for GSM8K/MMLU and ±3 pp for HumanEval). This is
    not a no-degradation verdict; see the non-inferiority caveat below.
  * Caveats found afterwards:
    * the audit ran with up to 2 concurrent requests, which weakens pruning
      compared with bs = 1, so it is not a conservative test;
    * the harness's regression flag had an inverted sign (the verdict is
      unchanged, since all p > 0.05);
    * "no significant difference" is not non-inferiority. The one-sided
      non-inferiority re-analysis (0.5 pp margin, Tango score bounds) is
      INCONCLUSIVE for GSM8K (lower bound −1.80 pp vs noprune) and HumanEval
      (−1.02 pp) and PASS for MMLU and JCommonsenseQA
      (`results/quality-q1/q2_quality_analysis.md`).
    
    Later gates use a one-sided non-inferiority test with a 0.5 pp margin at
    bs = 1.
* **Behavioural consequence.** Singleton status depends on the whole verify
  batch. At bs > 1, or with a different draft, the target's output for a request
  can depend on what else is in the batch.

**Source.** `4d62a84e77`, `1c7630d2ee`, `57df694fca`, `14d4c4c985`;
`lab-notes/P1_LOG.md`, `lab-notes/P2_LOG.md`, `lab-notes/Q1_QUALITY_AUDIT.md`,
`../patches/README.md`.

---

## F. Adaptive width (wa)

One server that chooses the draft width per step from {3, 15} and later
{3, 7, 15}. It replaces hand-picking W4 or W16 per use case.

### F1. Making `--speculative-adaptive` work on this model

* **Why.** Stock adaptive speculation first collapsed acceptance to 1.3-1.8,
  flip-flopping 3 ↔ 1 every step, and then crashed with an illegal memory
  access (2026-09-02).
* **Fixes (2026-09-02), `1b43a90af3` / `058765b624`, CPU tests 18 pass.** Four
  causes were found and fixed:
  * bs = 1 hysteresis too narrow;
  * QSA index sharing disabled under adaptive;
  * the GDN recovery stream not drained before a state swap;
  * no recovery graphs for additional states.
  
  Also fixed: the shared logits buffer is sized for the widest candidate
  (`5058a8582d`).
  
  After this there were no crashes and needle passed. But *building* a second
  runtime state still cut the initial state's acceptance by about 35 %.
* **Root cause (2026-09-05).** For compressed-QSA draft models,
  `create_draft_extend_backend()` returned the draft runner's own attention
  backend. So every adaptive state's draft-extend runner shared one
  `QwenSparseAttnBackend`, which caused two failures:
  1. Illegal memory access: each state's `init_cuda_graph_state` re-allocated
     the metadata tensors that the already-captured steps = 15 graphs point to.
     Fixed by `init_cuda_graph_state_no_shrink` (`2039efca83`).
  2. Silent acceptance loss: the backend's `_cuda_graph_metadata` is keyed by
     (mode, bs) but not by token width. The steps = 3 capture overwrote the
     launch state's 16-row metadata with a 4-row version, so long chains read
     stale sparse selections.
     * Measured with the steps = 3 state merely built: code-edit acceptance
       9.28 -> 6.19 and 488 -> 304 t/s.
     * Fixed by giving each state a private draft-extend backend, +0.14 GB
       (`b51c0dcdd2`).
* **Controller fixes.**
  * Drop post-switch accept samples that are longer than the live chain. The
    first working run switched 489 times in 90 s (`558df950cb`).
  * A post-switch grace window (`e53cbebc04`).
  * Re-seed the EMA on step-down only (`926e3661fb`). Switches went 24 -> 10
    per benchmark.
  * Config `adaptive_w16_3_15.json`.
* **Effect** (2026-09-05 21:31-21:40, same window):

  | Profile | code | prose-en | prose-ja | agent |
  |---|---|---|---|---|
  | wa | 463 | 229 | 205 | 306 |
  | W4 | 318 | 220 | 208 | 282 |
  | W16 | 528 | 156 | 170 | 235 |

  Agent-loop beats both fixed profiles; code-edit is −12 % against W16. This
  became the `wa` profile at `--mem-fraction-static 0.925`. The extra state
  costs about 0.67 GB.
* **Correctness.** Needle PASS on all arms. Unit tests model the pipeline lag
  and bound the switches.

### F2. C1: confidence-mixture policy

* **Why.** Code-edit's acceptance at S = 15 is bimodal: either full chains or 0-1
  tokens. A threshold on one EMA of the mean keeps dipping below the step-down
  line.
* **Structural constraint.** The step count must be decided *before* the draft
  runs, because the whole 15-step draft is one CUDA graph. The only signal
  available then is the top-1 probability of chain position 0, from the previous
  draft_extend. The top-k = 1 fast path had hard-coded it to 1.0.
* **How.**
  1. The split-argmax kernel also emits `sum(exp(v − m))`, so the finalize
     computes the real p_top1. A constexpr keeps the default path bit-identical.
  2. The probability is staged to pinned host memory without ever blocking CPU
     run-ahead. The policy reads the newest *completed* event.
  3. The policy picks
     `argmax_S Σ_b w_b · (1 + E[accepted | bucket b, S]) / step_time(S)`.
     * w_b is an occupancy EMA over confidence buckets, with edges
       (0.8, 0.95, 0.99, 0.999) packed against 1.0, where the mass is.
     * Downward predictions are **exact**, because a shorter greedy chain is a
       prefix: min(accepted, S').
     * Upward predictions use a hazard extrapolation with `tail_bias`, which
       measured 8-17 % optimistic, and an asymmetric `up_margin`.
     * Grace backoff prevents ping-pong where two widths tie.
  
  Flag `SGLANG_ADAPTIVE_POLICY=confidence`, config `../bench/adaptive/w16_conf.json`.
* **Evidence the signal is real.** corr(p0, accepted) = 0.37-0.53 across
  workloads (0.63 on the first W4 trace). On code-edit, mean accepted rises
  2.93 -> 11.70 across the buckets.
* **Effect** (2026-09-06, same hour, 4 workloads × 2 repeats):

  | Workload | EMA policy | Confidence policy | Change | vs better fixed profile |
  |---|---|---|---|---|
  | code-edit | 480 | **586** | **+22.1 %** | +18.9 % |
  | prose-en | | | −1.7 % | +4.0 % |
  | prose-ja | | | +1.4 % | +2.3 % |
  | agent-loop | | | +1.0 % | +2.1 % |

  Switches per benchmark: 4 (EMA: 13). Offline 3,000-batch replay: worst case
  0.996 of the better fixed profile, against 0.957 for the EMA policy. Removing
  the confidence buckets drops the worst case to 0.860, so the buckets matter.
* **Correctness.** Needle PASS. Free VRAM 4.59 GB at fraction 0.925.
* **Source.** `06f2563c39`, `bfb1731792`, `de1da18523`, `1d34c8a608`,
  `050cbcabc4`, `7473a2ef65`, `5cf83aa321`; `lab-notes/C1_LOG.md`;
  `../bench/adaptive/README-confidence.txt`.

### F3. X3: the non-initial adaptive state skipped the target autotune

* **Why.** In adaptive mode a width of 7 cost 1.45× the W4 step time, against
  1.23× for fixed profiles, so three-width policies could not pay (WA2/WA3).
* **Root cause.**
  * `BaseRunner.warmup` returns early once the shared model is marked warmed
    up, so additional target graph runners skipped autotune.
  * By then FlashInfer's autotuner had cleared its file configs while loading
    the *draft's* cache.
  * So the non-initial state's verify graph captured generic CUTLASS MoE
    tactics plus a separate `finalizeMoeRoutingKernel` in each of 48 layers.
    GEMM1 went 43.3 -> 48.7 µs and GEMM2 25.3 -> 37.3 µs, plus 9.1 µs of
    finalize per layer.
* **How.** `SGLANG_ADAPTIVE_TARGET_AUTOTUNE=1` runs the target warmup/autotune,
  with the candidate attention backend installed, before capturing each
  candidate's graphs. It is set in the wa profile only; the global default is
  off. 49 CPU tests.
* **Effect.** Adaptive state pinned at S = 7:
  * trimmed 13.39 -> 11.96 ms (−10.7 %), against 11.77 ms for fixed W8;
  * wall −13.4 % (code), −9.1 % (prose), −17.1 % (agent).
  * The production [3, 15] profile was untouched on its initial S = 15 side.
  * Production smoke (2026-09-07 23:38): startup, graph capture and needle
    normal.
  * At the same time no 325 W power-cap throttling was seen in settled decode:
    0/39 trace-aligned samples (nvidia-smi, 100 ms polling) capped, per-arm
    median SM 2,242-2,287 MHz. Throttling shorter than the polling interval is
    not ruled out.
* **Source.** `446c801189`; `lab-notes/X3_ADAPTIVE_COST.md`; `lab-notes/WA4_LOG.md`.

### F4. WA5: three-width policy [3, 7, 15]

* **Why.** The fixed-width sweep showed agent-loop is best at W8 (+15 % over
  W4), while prose wants W4 and code wants W16.
* **How.** Config `../bench/adaptive/w16_3_7_15_c.json` (policy commit `7b4d539f9b`,
  inert by default, 54 unit tests):
  * `switch_margin 0.10`, `up_margin 0.04`;
  * per-destination `down_margin {3: 0.03, 7: 0.15}`;
  * `max_grace_batches 80` (without a cap, reversal grace grew to 640-1,280
    batches);
  * `adjacent_only_promotion` (only the next larger width may be promoted to).
  
  The step-cost model was refitted from the width sweep:
  `SGLANG_ADAPTIVE_STEP_A/B = 7.943 / 0.5554`.
* **Path to adoption.**
  * WA2 and WA3 (2026-09-07) failed. The uncapped grace and single margin made
    selection unstable, and then the X3 bug made W8 expensive.
  * WA4 (after X3) passed for agent-loop (+19 %) but showed code and prose
    losses. X4 could not reproduce them after a restart; they were attributed to
    other GPU applications running on the desktop.
* **Effect** (WA5, quiet-night ABAB, 2026-09-08 01:56-02:10, pooled C vs A,
  where A = [3, 15]):
  * **agent-loop +14.4 %**: +13.7 % and +15.0 % in the two pairs, at W8 100 % of
    the time, acceptance 4.64-4.67;
  * code-edit +0.7 %, prose-en +0.4 %, prose-ja +0.2 %;
  * needle 4/4.
  
  One code repeat took a 200-step detour (W16 -> W4 -> W8 -> W16) and scored
  −0.73 %. The review recommended holding for that reason. It was adopted
  because the agent gain reproduced in both pairs and is far above the
  measurement MDE.
* **Production smoke** (2026-09-08 02:25, two repeats): code-edit 669 / 659,
  prose-en 287 / 259, agent-loop **430 / 362** t/s. Needle PASS.
* **Follow-up (WJ1).** A 3-candidate vs 2-candidate ABBA on prose found Japanese
  inconclusive (+4.4 %, CI −3.3 to +16.9) and English −8.3 % for the 2-candidate
  config. The 3-candidate config stayed.
* **Revert.** `ADAPTIVE_CONFIG=.../w16_conf.json SGLANG_ADAPTIVE_STEP_A=9.74
  SGLANG_ADAPTIVE_STEP_B=0.70`.
* **Source.** `7b4d539f9b`; `lab-notes/WA5_LOG.md`, `lab-notes/WA2_LOG.md`,
  `lab-notes/WA4_LOG.md`, `lab-notes/X4_THREE_CANDIDATE_COST.md`,
  `lab-notes/WJ1_WA_JAPANESE.md`, `lab-notes/WIDTH_SWEEP_0907.md`.

---

## G. Runtime and operations

### G1. Placeholder draft vocabulary weights (`SGLANG_DRAFT_SKIP_VOCAB_WEIGHTS=1`)

* **Why.** The NEXTN draft allocated a full [248,320, 2,560] BF16
  `embed_tokens` and an identical `lm_head` (2.37 GB). No checkpoint tensor
  fills them, and `init_lm_head` replaces them with the target's tensors. On a
  96 GB card with the target resident, that pair was the start-up memory peak.
  The 1.19 GiB embedding allocation is what ran out of memory when the desktop
  used about 0.8 GB more than usual.
* **How.** Build both modules on the meta device with a 1-row placeholder. They
  keep their layout metadata and quant method.
* **Effect.** 2.40 GB less resident.
* **Correctness.** Needle and sanity PASS; greedy acceptance prose-en 2.58,
  agent 3.19 vs 2.54 / 3.24, a difference within run-to-run noise.
* **Caveat.** The freed bytes go to the KV pool, so W16 at mem fraction 0.97
  started but ran out of memory on the first request. Fractions were lowered
  (G3).
* **Source.** `ea52699963`.

### G2. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (launcher default)

* **Why.** From 2026-09-07 00:15 every start failed the KV-pool check even
  with about 88.8 GB free. After the draft loaded, the caching allocator held
  about 9 GB of inactive split blocks and released 4.2 GB instead of 10.8 GB.
  Headroom was only 0.1-0.3 GB, so small changes in desktop VRAM (about 1.2 GB
  less free that night) decided the outcome.
* **How.** Expandable segments made the KV pool identical to known-good runs.
  Setting it empty restores the native allocator.
* **Correctness.** Allocator-only. the Q1 runs logged identical pool sizes, and later
  traces showed kernel medians within 1 %.

### G3. Memory fractions, mamba slots and desktop headroom

* W4 0.935, W16 0.93, wa 0.925. When the desktop is heavy, the experiment
  harness used 0.92 or 0.90.
* These values keep at least 4 GB free for the desktop in steady state. The
  launcher had used W16 0.945 with the embedding table, which left 3.4 GB free
  and led to a KWin/plasma crash on 2026-09-05.
* The embedding table (A9) and the extra adaptive states are allocated after
  the KV pool. So every new feature that allocates late needs the fraction
  re-checked.
* `MAMBA_SLOTS=10` for W8/W16/wa (the default 24 over-reserves for the
  speculative state).

### G4. Display refresh-rate switch (opt-in, `SERVE_DISPLAY_HZ=60`)

* **Why.** The desktop compositor shares this GPU. At 4K 160 Hz, compositing
  inflated the W4 step by about 7.5 %. Dropping to 60 Hz recovered −5.5 % to
  −6.7 % of step time (CUPTI A/B, 2026-09-05).
* **How.** The launcher switches the primary output's rate while the server
  runs. A double-forked watcher restores it when the server exits; the earlier
  version was killed by SGLang's crash path. Off by default.
* The real fix is a separate display GPU.

### G5. Other operational rules that affected results

* One server at a time, under a file lock.
* Kill by process group, then confirm that `nvidia-smi` shows the memory
  released before the next start. Orphaned servers caused two failed
  quality-audit arms.
* Wait a few seconds after killing a server; otherwise the KV pool size varied
  from 5k to 180k tokens.

## H. The 2026-10 round: fixed-cost cuts, sampling fidelity, rejection sampling, 524k context

Everything in this section shipped on 2026-10-02 as SGLang patches 0106-0121 and FlashInfer patches 08-10
(`patches/sglang/SERIES.md`, `patches/flashinfer/`). Every flag is a `${VAR:-1}` default in
`launch/serve-fast.sh`; setting it to 0 restores the 2026-09-08 behaviour of that piece. The raw results are in
`results/runs-1002/`, the lab notes in `lab-notes/` (`STACK_2026-10-01.md` is the overview).

Two things changed in how this round was measured (details in `measurement.md`, "The 2026-10 protocol"):

* **Sampling counts as much as greedy.** Until September every decision was greedy. LM Studio, which is how the
  server is actually used, samples with T 0.8, top_p 0.95, top_k 40, min_p 0.05. `fnbench --sampling lmstudio`
  sends exactly that, and every A/B in this section reports both modes. The two can move in opposite directions.
* **Eight server starts per A/B** (A1 B1 B2 A2 B3 A3 A4 B4), analysed with an ANCOVA that reports ms/token,
  tok/step and ms/step at fixed acceptance (`ms/step|a`). One server start moves ms/step by 1-3 % on its own, so
  four starts could not resolve 1 % changes (FG1 below).

Unless a row says otherwise, numbers are on the `wa` profile with the private MTP head and token map (see the
README's results note). Kernel numbers are CUPTI medians or CUDA-graph microbenchmarks; "clipped R.eager" is the
eager (non-graph) tail of a step from in-server traces with display stalls clipped.

### H1. The fixed-cost stack (RT1, SV1, SV2, FG1, DG1, RQ2 u2h)

A fixed-cost map of the production step (`FC_fc-map_2026-10-01.md`, `FC_fc-moe_2026-10-01.md`,
`FC_fc-glue_2026-10-01.md`) listed the per-step costs that do not scale with the weights read. Six pieces came
out of it. Each was checked on its own (bit-exactness or a stated output difference, kernel timing, an in-server
smoke), then all six were judged together in one 8-start ABBA.

| piece | what | own measurement | output |
|---|---|---|---|
| **RT1** `SGLANG_ROUTER_FAST_TOPK` (0110-0111) | router softmax top-k on 32-bit packed keys | router share 3.8 → 1.5-1.6 µs per call, about −0.13 ms/step at wa (55 calls) | bit-exact (12 cases × 65,536 rows) |
| **SV1** `SGLANG_OPT_SPEC_SPARSE_VERIFY` (0108) | sampling verify on the top-KP logits (KP 64 for top_k 40) instead of five full-vocabulary passes | sampling eager tail −205 µs/step in a 4-start ABBA, tok/step unchanged | rounding-level; greedy untouched |
| **SV2** `SGLANG_OPT_SPEC_SPARSE_TOPK` (0113) | FlashInfer radix top-k (sorted, deterministic) for SV1's top-KP | −32..−38 µs per call; 2 top-k launches per sampling step instead of 28 | same values as `torch.topk` |
| **FG1** `SGLANG_OPT_GDN_FRONT_OVERLAP` (0109) | GDN verify front: the combine gate forked onto the alt stream, overlapping the HC mix kernels | −3.57 µs per layer at M=4 (microbenchmark), about −0.13 ms/step at W4; its own ABBA was inconclusive (`rejected.md`) | bit-exact |
| **DG1** `SGLANG_OPT_DRAFT_MOE_GEMV` (0112) | the draft's one-token MoE as two Triton W4A16 NVFP4 GEMV kernels instead of the CUTLASS FP4 chain | 49.4 → 29.8 µs per call; −0.27..−0.34 ms per step at 15 draft steps | drafts change (bf16 instead of FP4 activations); target untouched |
| **RQ2 u2h** (FlashInfer 08, 10) | MoE routing prologue: expand row staged with `cp.async`, rolled quantize loop, scale-factor address hoisted | in-server prologue 12.4 → 7.8 µs; 4-start ABBA ms/step −3.38 % [−5.35, −1.36] greedy, −3.28 % sampling (request-level CI) | bit-exact by construction |

**The stack ABBA** (`results/runs-1002/stack8/`, `TABLES.md` §6). All six against the 2026-09-08 production,
min_p off in both arms, 8 starts, 4 workloads × both modes. B/A, 95 % CI at arm level (5 degrees of freedom):

| mode | ms/token | tok/step | ms/step\|a |
|---|---|---|---|
| LM Studio sampling | **−9.55 %** [−13.63, −5.27] | +3.13 % [+1.05, +5.24] | −7.57 % [−11.19, −3.80] |
| greedy | **−4.59 %** [−8.89, −0.08] | −1.18 % [−5.82, +3.70] | −5.40 % [−8.32, −2.40] |

Client t/s per workload (geometric mean per arm, then the mean of the 4 arms; `TABLES.md` uses arithmetic
means and differs by 1-6 t/s):

| mode | code-edit | prose-en | prose-ja | agent-loop |
|---|---|---|---|---|
| LM Studio | 511 → 580 (+13.6 %) | 203 → 222 | 221 → 241 | 286 → 317 |
| greedy | 555 → 569 | 237 → 248 | 268 → 275 | 323 → 349 |

* Sampling gains more because SV1/SV2 only act on sampling steps. In clipped R.eager, the sampling eager tail
  fell from 376-391 µs/step (A arms) to 119-139 µs/step (B arms).
* No piece targets acceptance. DG1 changes the drafts, so the tok/step rise under sampling may be real, but it
  was not tested on its own.
* DT1 (draft-tail glue) was built in the same round and is not shipped (`rejected.md`).

### H2. min_p in the speculative verify (fidelity fix)

* **What was wrong.** The speculative sampling verify ignored `min_p`, both target-only and with rejection
  sampling. With LM Studio's min_p 0.05, a speculative server could emit tokens below 0.05 × p_max, which the
  non-speculative sampler excludes. Found while mapping the sampling cost (`FC_fc-map_2026-10-01.md`).
* **Fix.** Patch 0107, `SGLANG_SPEC_MIN_P=1`: min_p is applied in the verify and in the RS draft proposal.
* **Cost.** None measurable (`RS1_REJECTION_SAMPLING.md`, the min_p ABBA). It ships for correctness, not speed.
  The stack, ST1 and XA1 ABBAs ran with min_p off in both arms; the RS ABBAs with it on in both arms.

### H3. ST1 + XA1: the draft's valid prefix, and a Triton decode attention

* **ST1** `SGLANG_ENABLE_QSA_SHARED_TAIL_PREFIX` is a **correctness fix**. Index-shared draft rows broke the
  attention's valid-prefix contract, so the draft's attention silently dropped its newest positions. XA1's
  kernel work found it (its §1.3). With pruning off (`PRUNE_TAU=0`) output is exact either way, because the target verifies every token (with the
  default pruning the target's routing, and so its output, can depend on the draft; see the README); the open
  question was whether acceptance would rise. It did not beyond noise. 8 starts, request-level CI:
  LM Studio ms/token −2.09 % [−4.35, +0.21], tok/step +0.38 %, ms/step|a −1.68 % [−3.16, −0.18]; greedy ms/token
  −0.64 %, tok/step −3.49 % [−7.81, +1.03]. Arm-level CIs are about 3× wider and all cross 0, and the night arms
  alone show LM Studio ms/token +0.18 %. So ST1 is **not claimed as a speed-up**; it ships only because XA1 was
  measured on top of it and relies on its layout.
* **XA1** `SGLANG_OPT_TRITON_DECODE_ATTN`: the QSA decode attention as one split-KV Triton kernel instead of
  the production XQA path: 12.2 µs against 16.2-16.4 µs per call in server traces (microbenchmark savings
  −6.2..−14 µs per call). 8-start A/B with ST1 in both arms (`results/runs-1002/xa1-st1/`):
  * LM Studio ms/step|a −2.52 % [−3.54, −1.49] (arm level [−5.52, +0.35]); pooled t/s +2.82 % [+0.01, +5.70]
    (arm level [−0.88, +6.66]);
  * greedy t/s +1.17 % [−1.33, +3.75], ms/step|a −0.44 % (unresolved);
  * no workload's tok/step CI excludes 0.
  The speed claim rests on the kernel timing and the request-level sampling result; the arm-level CIs cross 0.

### H4. The RS package: sparse rejection sampling with a K = 16 draft support

Rejection sampling (RS) lets a sampling request accept a draft token with probability min(1, p/q) instead of
requiring an exact match with the target's sample, which raises acceptance under sampling. Greedy requests are
unaffected in principle.

* **RS1** (dense RS over the hot vocabulary, patch 0106) raised acceptance but its per-step cost ate the gain on
  code (rejected, `rejected.md`). **RS2** (patch 0116, `SGLANG_OPT_SPEC_SPARSE_RS`) keeps the draft proposal
  sparse: q lives on the draft's top-K ids, after the request's own top_k/top_p/min_p filters. RS2 alone was
  rejected too (code-edit slower).
* What shipped is a **package** on top of RS2, measured as one unit against the stack's target-only verify:
  * RS2d draft sharpening (0117): `SGLANG_RS_DRAFT_TEMP_SCALE=0.7`, `SGLANG_RS_DRAFT_ONEHOT_ABOVE=0.9`;
  * G1 greedy fast path in the proposal (0118): `SGLANG_RS_GREEDY_FAST=1`;
  * RS3 block verification (0119-0120): `SGLANG_RS_BLOCK_VERIFY=1`, vectorised over the chain (+8.5 µs per
    verify at most; the sequential version cost +58 µs);
  * `--speculative-use-rejection-sampling` (added by the launcher when RS is on; never for ngram).
* **Exactness.** Each step was checked against Σ min(p, q) and per-position chi-square / Fisher tests on the GPU
  (e.g. RS2: accept@0 0.73396 against 0.73481, SE 0.00156, Fisher p = 0.837).
* **Package ABBA** (`results/runs-1002/rs2d-abba/`), 8 starts, request CI (arm CI in brackets after):
  * LM Studio t/s pooled **+6.53 %** [+4.60, +8.51] (arm [+1.36, +11.97]); agent-loop +6.17 %, prose-en +8.52 %,
    prose-ja +10.01 %;
  * code-edit +1.63 % [−2.49, +5.90]: **unresolved**, and below the rule's −2 % floor at the lower bound;
  * greedy ms/step|a −2.18 % [−3.30, −1.04] (arm [−8.05, +4.12]). Nothing in the package should make a greedy
    step cheaper; read this as drift between arms.
  * By the pre-registered rule the package was **not adopted**. The author shipped it anyway for the prose and
    agent gains, accepting the unresolved code-edit result. Turn it off with `SGLANG_OPT_SPEC_SPARSE_RS=0` (`launch/serve-fast.sh` then also clears `SGLANG_RS_BLOCK_VERIFY`; with
    `launch/as-measured/serve-fast.sh` set `SGLANG_RS_BLOCK_VERIFY=0` yourself, or start-up fails).
* **K = 16** (`SGLANG_RS_DRAFT_TOPK=16`, was 64 in the ABBA): after LM Studio's filters q has only 6.7-15.1
  nonzero ranks per verify, so 16 loses nothing measurable offline. It saves 11.7 µs per draft step; the
  support computation went from 21.10 to 8.22 µs per call (fp32), about 82 µs per verify at 7 draft steps and
  176 µs at 15 (`results/runs-1002/rs4-k16/`). Offline on the dump, the acceptance change is −0.005 %. It joined the package after a
  one-arm smoke (kernel medians against the ABBA's B traces), without its own ABBA, by a rule fixed beforehand.

### H5. 524,288-token context on wa

* **How.** Above 262,144 tokens the launcher turns on factor-2 YaRN, and `wa` raises its memory fraction to
  `WA_LONG_MEM_FRACTION` (0.96). The default context stays 262,144; 524,288 is opt-in via `CONTEXT_LENGTH`.
* **Needles** (thinking off, temperature 0, `results/runs-1002/long-524k/`): 16k PASS in 8.1 s; 185k at depth
  0.5 PASS in 17.5 s; 480k at depth 0.25 PASS in 70.7 s; 480k at depth 0.75 PASS in 71.4 s.
* Prefill about 6.8k tok/s at 480k. Decode with LM Studio sampling 185.5 t/s at 480k context, 284.4 t/s at 2k
  (same request shape).
* GPU memory in use peaked at 96,220 MiB (1,667 MiB free), with the desktop holding about 6.8 GB of it. At the
  normal 0.925 fraction the KV pool would have been capped below 524,288 with that desktop.
* Short prompts with YaRN on, one run each, against the 262,144 smoke: −25 % to +34 % per workload and mode.
  Outputs differ under YaRN, so this is noise-level evidence, not a measured cost. W4 and W16 were not tested
  above 262,144.
* 1,048,576 tokens do not fit: KV and state need about 10 GB more than the GPU has (13.5 KB per token), and the
  model only promises 262,144. YaRN factor 4 is outside the fork's qualified range.

### H6. Shipped build smokes

One server start per profile with the shipped flags at context 262,144 (the worst case for memory), one prompt
per workload and mode, plus a check script that every flag logged its enable line and no error appeared
(`results/runs-1002/ship/`, `TABLES.md` §7). LM Studio t/s, code-edit / prose-en / prose-ja / agent-loop:

| profile | LM Studio t/s |
|---|---|
| wa | 524 / 226 / 335 / 317 |
| W4 | 348 / 235 / 241 / 304 |
| W16 | 539 / 149 / 152 / 242 |

These are single prompts, so they show that the build works, not how fast it is. The shipped build was measured
properly on 2026-10-06 (32 prompts, `TABLES.md` §8, README "2026-10 update").

Every A/B in this section ran `wa`. W4 and W16 ran the new pieces for the first time in these smokes; W4 was
then measured on 2026-10-06 (`TABLES.md` §8, greedy only), W16 has not been. A code reading found no adaptive-only assumption in the shipped diff (W4 runs
wa's width-3 code, W16 its width-15 code). The wa width model (`SGLANG_ADAPTIVE_STEP_A/B`) was fitted to the
September step cost and has not been re-fitted to the cheaper steps.

### H7. Known issues found in the pre-publication review (2026-10-06, not fixed yet)

The first and third cannot occur in the measured setting (one running request, SGLang's own contiguous hidden
states). The second can, with any batch size, but only in a batch that holds a sampled request with a `top_k` (an
all-greedy batch takes the argmax path; in a mixed batch the greedy rows go through the sparse path too) and
only when more logits tie exactly at the cut than the margin; we did not check how often:

* **RS greedy fast path with mixed batches.** With `SGLANG_RS_GREEDY_FAST=1` a greedy request's draft support
  repeats one token id. If a batch mixes it with a request that needs the dense verify fallback (for example
  `top_k=-1`), the dense scatter of the support (`eagle_utils.py`, `draft_probs.scatter_`) can let a zero
  overwrite the probability 1, which biases rejection sampling for that request. Needs ≥ 2 running requests.
  Fix: `scatter_add_` or unique support ids.
* **Sparse verify cuts boundary ties.** SV1 keeps a fixed `KP` (≥ 8 entries past the largest `top_k`); if more
  logits tie exactly at the top-k boundary than that margin, the extra ones are dropped, so the target
  distribution differs slightly from the dense path (documented in `sparse_verify.py`, not in the launcher).
  Fix: fall back to the dense verify when a tie reaches the boundary.
* **FlashInfer patch 08 assumes 16-byte aligned input.** The RQ2 expand-row staging uses 16-byte `cp.async`;
  FlashInfer itself only requires 4-byte alignment for BF16 input, so a sliced input view (for example
  `storage_offset=2`) would fault. Fix: gate the fused path on `input.data_ptr() % 16 == 0`.

---

## Appendix: production flags (as of 2026-09-08; the 2026-10 additions at the end)

| Flag / argument | Section | Profiles |
|---|---|---|
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | G2 | all |
| `SGLANG_FP8_W8A16_GEMV=1` | B1 | all |
| `SGLANG_GDN_BA_TRITON_GEMV=1` | B2 | all |
| `SGLANG_KV_FP8_FUSED_STORE=1` | C3 | all |
| `SGLANG_HC_MIX2=1`, `SGLANG_HC_MIX2_FP8=1` | C8 | all |
| `FLASHINFER_GDN_WY_STRIDED_QKV=1` (+ vendored patch) | C6 | all |
| `SGLANG_DRAFT_LOGITS_OUT=1` | A8 | all |
| `SGLANG_ROUTER_GEMV=1` | B3 | all |
| `SGLANG_SHARED_GATEUP_FUSED=1` | B4 | W4 only (0 for W16/wa) |
| `SGLANG_DRAFT_SKIP_VOCAB_WEIGHTS=1` | G1 | all |
| `SGLANG_QSA_META_PAGE_PARALLEL=1` | C12 | all |
| `SGLANG_MTP_EMBED_TABLE=1` | A9 | all |
| `SGLANG_GDN_CONV_CHAIN_PARALLEL=1` | C5 | all |
| `SGLANG_GDN_PROJ_DIRECT_LAYOUT=1`, `SGLANG_GDN_AB_STASH_DIRECT=1` | C7 | all |
| `SGLANG_HC_GATE_EARLY=2`, `SGLANG_SHARED_GATE_EARLY=1`, `SGLANG_HC_APPLY_MIX_FUSED=1` | C9 | all |
| `SGLANG_HC_LAYER_APPLY_FUSED=1` | C10 | all |
| `SGLANG_NORM_INTO_GEMV=1` | C11 | all |
| `SGLANG_MOE_PRUNE_SINGLETON_TAU=0.08`, `SGLANG_MOE_PRUNE_IN_PROLOGUE=1` (+ FlashInfer P2 patch) | E | all |
| `FLASHINFER_MOE_PACK_GROUPS=1` (+ G1 patch); A0/A3/G2 folds on by default in the patched C++ | D | all |
| `SGLANG_TRITON_PDL=1` | C13 | all |
| `--speculative-token-map hot2_49152` | A1 | all |
| `--qwen4-exp-dense-fp8 shared_expert,attn,linear_attn,lm_head,mtp_dense` | A2, A7, B1 | all |
| `--speculative-draft-model-quantization modelopt_fp4` | A7 | all |
| v5 fine-tuned MTP head (not published) | A10 | all |
| `--speculative-num-steps 15 --speculative-num-draft-tokens 16`, `MAMBA_SLOTS=10` | A4 | W16, wa |
| `--speculative-adaptive`, `SGLANG_ADAPTIVE_POLICY=confidence`, `w16_3_7_15_c.json`, `SGLANG_ADAPTIVE_STEP_A/B=7.943/0.5554`, `SGLANG_ADAPTIVE_TARGET_AUTOTUNE=1` | F | wa |
| `--mem-fraction-static` 0.935 / 0.93 / 0.925 | G3 | W4 / W16 / wa |
| `SGLANG_ROUTER_FAST_TOPK=1`, `SGLANG_OPT_SPEC_SPARSE_VERIFY=1`, `SGLANG_OPT_SPEC_SPARSE_TOPK=1`, `SGLANG_OPT_GDN_FRONT_OVERLAP=1`, `SGLANG_OPT_DRAFT_MOE_GEMV=1` (+ FlashInfer 08, 10) | H1 | all (2026-10-02) |
| `SGLANG_SPEC_MIN_P=1` | H2 | all (2026-10-02) |
| `SGLANG_ENABLE_QSA_SHARED_TAIL_PREFIX=1`, `SGLANG_OPT_TRITON_DECODE_ATTN=1` | H3 | all (2026-10-02) |
| `SGLANG_OPT_SPEC_SPARSE_RS=1`, `SGLANG_RS_DRAFT_TOPK=16`, `SGLANG_RS_DRAFT_TEMP_SCALE=0.7`, `SGLANG_RS_DRAFT_ONEHOT_ABOVE=0.9`, `SGLANG_RS_GREEDY_FAST=1`, `SGLANG_RS_BLOCK_VERIFY=1`, `--speculative-use-rejection-sampling` | H4 | all but ngram (2026-10-02) |
| `CONTEXT_LENGTH` > 262144: factor-2 YaRN, `WA_LONG_MEM_FRACTION=0.96` | H5 | wa (opt-in) |
