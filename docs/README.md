# Documentation

| file | contents |
|---|---|
| [`optimizations.md`](optimizations.md) | every adopted change: what, why, how, measured effect, source |
| [`rejected.md`](rejected.md) | experiments that did not ship, with numbers and the reason |
| [`roofline.md`](roofline.md) | bandwidth floor, step-time model, exclusive-time maps, why 1000 t/s is out of reach |
| [`measurement.md`](measurement.md) | metrics, noise sources, the BN1 benchmark protocol, quality gates |
| [`timeline.md`](timeline.md) | dated record, 2026-08-28 to 2026-09-08 |
| [`reproduce.md`](reproduce.md) | how to rebuild the stack and re-run the benchmarks |
| [`train-your-own-mtp-head.md`](train-your-own-mtp-head.md) | fine-tune the MTP draft head on your own data: dump hook, corpus, self-generation, training recipe, renewal evaluation, write-back, server A/B |
| [`lab-notes/`](lab-notes/) | the original engineering notes, lightly sanitised (paths; all results of evaluations on private data removed) |

The first five pages were written after the fact from the project log and the lab notes. Numbers that
come with a result file under `../results/` can be regenerated; the rest are quoted from the notes.
Byte-floor figures in `roofline.md` are our own arithmetic, not measurements. Commit hashes in the
notes refer to the original working branch; `../patches/sglang/SERIES.md` maps them to patch numbers.

## Lab notes

Written during the work, for the people (and AI coding assistants) doing it, so they are terse and use
internal task IDs. Some mention local branch names or scripts that are not part of this repository.
Paths were replaced with `$HOME/...` / `$SCRATCH/...`. Results of evaluations on the author's private data
(MTP-training dumps and prompts, agent/chat session logs), both numbers and conclusions, were removed. Where such
an evaluation was run, the text says what was evaluated and whether the candidate reached production, and nothing
about its outcome.
Results of a 12-task spot-check whose questions came from another, private evaluation project were removed as well.
Results measured on the public workloads in `../bench/workloads` and the public quality benchmarks are unchanged.
The docs pages above follow the same rule.

| note | title |
|---|---|
| [`A8_FEASIBILITY`](lab-notes/A8_FEASIBILITY.md) | A8 feasibility: fuse `silu(g)*u` + NVFP4 quant into the sm120 grouped-GEMM1 epilogue |
| [`BN1_BENCH_NOISE`](lab-notes/BN1_BENCH_NOISE.md) | BN1 — held-out prompt sets and paired block benchmark |
| [`C1_LOG`](lab-notes/C1_LOG.md) | C1 — confidence-driven speculative step choice (Qwen3.8-Flash-Next) |
| [`DH3_SHORTLIST_KERNEL`](lab-notes/DH3_SHORTLIST_KERNEL.md) | DH3 measured result |
| [`DH4_SHORTLIST_SERVER`](lab-notes/DH4_SHORTLIST_SERVER.md) | DH4 shortlist server integration |
| [`DH7_SELECTOR_REPAIR`](lab-notes/DH7_SELECTOR_REPAIR.md) | DH7 — production-state selector repair and restricted W16 eligibility |
| [`DRAFT_GLUE_SPEC`](lab-notes/DRAFT_GLUE_SPEC.md) | Draft-loop glue: the bookkeeping and elementwise kernels of the NEXTN draft step (spec) |
| [`DRAFT_V2_SPEC`](lab-notes/DRAFT_V2_SPEC.md) | Draft v2: can a stronger draft raise acceptance 10-20 %, and what would it take? |
| [`E0_MTP_FC_FP8`](lab-notes/E0_MTP_FC_FP8.md) | E0 — MTP fc_embedding / fc_hidden FP8 |
| [`E1_EXPERT_WIDTH_SCREEN`](lab-notes/E1_EXPERT_WIDTH_SCREEN.md) | E1 — expert intermediate width quality screen |
| [`EXCLUSIVE_TIME_MAP`](lab-notes/EXCLUSIVE_TIME_MAP.md) | X1 — Exclusive (critical-path) time map of the Qwen3.8-Flash-Next decode step |
| [`EXCLUSIVE_TIME_MAP_0907`](lab-notes/EXCLUSIVE_TIME_MAP_0907.md) | X2 — Exclusive (critical-path) time map, 2026-09-07 re-take |
| [`FP8D_FUSED_DECODE`](lab-notes/FP8D_FUSED_DECODE.md) | FP8D — fused byte-Huffman dense GEMV feasibility |
| [`FP8_CODING_SURVEY`](lab-notes/FP8_CODING_SURVEY.md) | FP8C — CPU lossless coding survey of serving dense FP8 weights |
| [`G1_LOG`](lab-notes/G1_LOG.md) | G1 — recover the MoE grouped-GEMM tail-wave loss on sm_120 |
| [`G2_LOG`](lab-notes/G2_LOG.md) | G2 log — deleting launches from the FlashInfer CUTLASS fused-MoE chain |
| [`GLUE_E_SPEC`](lab-notes/GLUE_E_SPEC.md) | Glue-E: remaining per-step glue in the Qwen3.8-Flash-Next decode step (spec) |
| [`HC1_BOTTLENECK_SCREEN`](lab-notes/HC1_BOTTLENECK_SCREEN.md) | HC1 bottleneck screen — complete |
| [`HC_MIX2_FP8_SPEC`](lab-notes/HC_MIX2_FP8_SPEC.md) | HC mix2: FP8 weights for the low-rank mix (spec) |
| [`J1_LOG`](lab-notes/J1_LOG.md) | J1 — deleting `_hc_branch_stats_kernel` (K0) from the HC mix chain: **NO-GO** |
| [`L2_LOG`](lab-notes/L2_LOG.md) | L2 — log: L2 prefetch of the next layer's dense weights (task L2) |
| [`L2_PREFETCH_SPEC`](lab-notes/L2_PREFETCH_SPEC.md) | L2 — hiding the dense-weight DRAM stream in the 128 MiB L2: feasibility and verdict |
| [`MEGAKERNEL_SPEC`](lab-notes/MEGAKERNEL_SPEC.md) | K2 — Persistent "glue island" kernels for the Qwen3.8-Flash-Next decode step |
| [`MOE_SMALLM_SPEC`](lab-notes/MOE_SMALLM_SPEC.md) | Routed-expert grouped GEMM at small M: call path, bytes model, and what is left to win |
| [`N1_LOG`](lab-notes/N1_LOG.md) | N1 — Dense NVFP4 for the Qwen3.8-Flash-Next serving path |
| [`P1_LOG`](lab-notes/P1_LOG.md) | P1 — selective expert-route pruning in the verify step |
| [`P2_LOG`](lab-notes/P2_LOG.md) | P2 — singleton-route pruning folded into FlashInfer's fused routing prologue |
| [`P3_CONTRIB_PRUNE`](lab-notes/P3_CONTRIB_PRUNE.md) | P3 — contribution-aware expert pruning |
| [`P4_CONTRIB_PRUNE_SHIP`](lab-notes/P4_CONTRIB_PRUNE_SHIP.md) | P4 contribution pruning ship candidate |
| [`Q1_QUALITY_AUDIT`](lab-notes/Q1_QUALITY_AUDIT.md) | Q1 — Quality audit of the Qwen3.8-Flash-Next production serving stack (2026-09-07) |
| [`QL1_LONG_CONTEXT_MAP`](lab-notes/QL1_LONG_CONTEXT_MAP.md) | QL1 long-context QSA map |
| [`S1_SPARSE_FP4`](lab-notes/S1_SPARSE_FP4.md) | S1 — sparse block-scaled FP4 expert feasibility |
| [`WA2_LOG`](lab-notes/WA2_LOG.md) | WA2: three-width confidence adaptive speculation |
| [`WA4_LOG`](lab-notes/WA4_LOG.md) | WA4: three-width confidence policy after X3 target autotune fix |
| [`WA5_LOG`](lab-notes/WA5_LOG.md) | WA5: quiet-night three-width confidence policy re-test |
| [`WA6_REQUEST_PRIOR`](lab-notes/WA6_REQUEST_PRIOR.md) | WA6 — request-boundary priors with measured width costs |
| [`WA7_JA_PRIOR`](lab-notes/WA7_JA_PRIOR.md) | WA7 — Japanese-only request pin |
| [`WIDTH_SWEEP_0907`](lab-notes/WIDTH_SWEEP_0907.md) | WS1: fixed speculation-width frontier, production stack (2026-09-07) |
| [`WJ1_WA_JAPANESE`](lab-notes/WJ1_WA_JAPANESE.md) | WJ1 — Japanese prose diagnostic |
| [`X3_ADAPTIVE_COST`](lab-notes/X3_ADAPTIVE_COST.md) | X3 adaptive runtime step cost and power diagnostic |
| [`X4_THREE_CANDIDATE_COST`](lab-notes/X4_THREE_CANDIDATE_COST.md) | X4: remaining three-candidate runtime cost |
