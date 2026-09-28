# Rejected and NO-GO experiments

Every entry below was built far enough to measure. Each gives the numbers that
settled it and what it taught. Some entries were evaluated only on the author's
private data; for those, the entry says what was done and whether the candidate
is in the production configuration, and gives no result or conclusion of that
evaluation. Conventions are the same as in
`optimizations.md`:

* W4 / W8 / W16 = 3 / 7 / 15 draft steps;
* wa = adaptive width;
* a t/s triple is code-edit / prose-en / agent-loop unless labelled.

Some entries were later partly revived or superseded; each such entry says so.

Contents:

1. Speculative-decoding side
2. Dense weights and quantisation
3. Kernel fusion and launch reduction
4. MoE
5. Adaptive width
6. Findings that were not rejections

---

## 1. Speculative-decoding side

### 1.1 NGRAM_CHAIN: request-local n-gram suffix drafting, compared with wide MTP

* **Idea.** Draft the next W tokens by suffix-matching the request's own
  history (prompt plus output), skipping the MTP draft on a hit and falling
  back to MTP on a miss.
* **Offline replay.** The idea was first screened with an offline replay
  (`sim/`) over the author's own private data; no result or conclusion of that
  replay is published.
* **Measured** (2026-09-02, W16 with the QSA ring fix):

  | Workload | Accepted per verify | Step time | t/s |
  |---|--:|--:|--:|
  | code-edit | 6.76 | 20.5 ms | 314-347 |
  | agent-loop | 4.56 | 22 ms | 200-214 |
  | prose-en | 2.41 | | 110 |

  Static MTP at W16 reached code-edit 429-433 t/s with acceptance 11.6. The
  measured hit rate was 0.37-0.49, but on hits the match
  length did not beat MTP-15, and misses paid for a full 15-step MTP draft.
  A later design study put it at code −22 %, agent +8 %, prose +9 % against the
  MTP profiles.
* **Lesson.** In these server runs a strong native MTP head at wide width beat
  n-gram lookup. A measured hit rate of 37-49 % did not translate into
  acceptance once the fallback cost was included.
* **Source.** `6d229a69f0`, `7a62299ce3`, `4bf52fbfcd`; project log 2026-09-02;
  `lab-notes/DRAFT_V2_SPEC.md`.

### 1.2 Stock `--speculative-adaptive`, as shipped

* **Measured** (2026-09-02):
  * Acceptance collapsed to 1.3-1.8: steps flip-flopped 3 ↔ 1 every step and 7
    was almost never used.
  * t/s: prose-en 95, agent-loop 117.
  * The server crashed on the second agent-loop request.
* **After the first fixes.** Four fixes went in (hysteresis, per-state QSA
  index sharing, draining the GDN recovery stream, per-state recovery graphs).
  With those there were no crashes, but merely *building* a second state still
  cut acceptance by about 35 % (code-edit 3.84 -> 2.1-2.5).
* **Later resolved.** The cause was a draft-extend backend shared across
  states, which led to an illegal memory access and stale per-width metadata.
  The fixed version became the `wa` profile (`optimizations.md` §F1).
* **Lesson.** Subsystems that share state objects across CUDA-graph states fail
  silently. The regression only showed up as acceptance, and it grew with chain
  length.

### 1.3 Full-vocabulary draft, and the 45k token map

* **Full vocabulary** (2026-09-04, W16, 4 greedy repeats):
  * Acceptance rose, as expected: prose-en 2.61 -> 3.05.
  * But throughput fell, code-edit 520 -> 357 t/s. The full-vocab draft head
    (248k rows) did not go through the FP8 W8A16 GEMV path.
* **`hot_49152` v1** (a frequency-only map that came out below 49,152 rows, i.e. not a power of two):
  * W4 prose-en +5 % t/s.
  * W16 code-edit and prose-ja −10 %.
  * The suspected cause was that a non-power-of-two row count missed the GEMV
    tile table.
* **Superseded by.** The blended `hot2_49152` (exactly 49,152 rows), which
  replaced both. A later map-size sweep was run on the author's private data; no
  result or conclusion of it is published. Production kept 49,152 rows.
* **Lesson.** Draft-head size is a trade between the per-forward GEMV and
  acceptance.

### 1.4 MTP head v1 and v2

* **v1** (2026-09-04):
  * Method: teacher-forced single-step CE, from a corpus-plus-self-generated
    dump (private).
  * An offline metric on the private dump was computed; its result is not
    published.
  * Server: no acceptance change beyond run-to-run noise (W4 code 3.76 / 3.67, prose 2.28 / 2.15,
    agent 3.17 / 3.30).
* **v2**:
  * Method: prose-en corpus removed, K = 3 self-rollout loss added, warm start.
  * Offline evaluation on private data (not published).
  * Server, 4 repeats:

    | Profile | v2 (code / prose / agent) | Original head |
    |---|---|---|
    | W16 | 9.89 / 2.56 / 4.47 | 10.5 / 2.52 / 4.85 |
    | W4 | 3.70 / 2.32 / 3.11 | 3.79 / 2.27 / 3.00 |

    No measurable effect.
* **Later offline metric.** From v3 on, candidates were scored with a "renewal"
  evaluator that replays the server's verify-and-advance process on held-out
  self-generated sequences (private data). The evaluator's numbers and its
  comparison with the server are not published.
  Server A/B on the public workloads remained the final gate. Soft-target
  distillation (v3) was then the first recipe that moved the server
  (`optimizations.md` §A10).

### 1.5 V4: full-head retrain on the same data

* **Setup** (2026-09-06). Retrain the whole MTP head from v3 weights for 3,000
  steps × 24,576 tokens, three times v3's budget. The re-extraction added no
  new documents.
* **Result.** Evaluated offline on the author's private data only; no result or
  conclusion of that evaluation is published. V4 was never served in
  production.
* **Next step.** The following head (v5) was trained on a broader
  re-extraction.

### 1.6 MT1 and MT2: training on serving-state data

* **MT1** (2026-09-07).
  * Dump: bs = 1 verify states from the wa server. Retrain from v5.
  * Offline evaluation on private data (not published).
  * Server A/B:
    * W4: code +2.7, prose-en +0.9, prose-ja +10.6, agent +1.1 %. The W4 gate
      needed +2 % on target workloads.
    * **W16 all negative: −13.6 / −8.0 / −2.8 / −13.7 %.**
  * A later review traced the W16 loss to the training rollout, which omits the
    chain's own intermediate KV (a "rollout contract" mismatch) rather than to a
    lack of W16 data.
* **MT2** (2026-09-08).
  * Cached long rollout (depth 15) that stacks real selected-prefix KV, plus a
    50/50 serving/v5 replay mix.
  * Evaluated offline on private data against a pre-registered gate; no result
    or conclusion of that evaluation is published. MT2 was never served.
* **Outcome.** No head after v5 went into production; v5 is the final head.

### 1.7 EAGLE-3-style strong draft: design study, phase 0 and phase 1

* **Design study** (`lab-notes/DRAFT_V2_SPEC.md`, 2026-09-05).
  * The chain acceptance model fitted a per-position rate p to server
    acceptance on the public workloads (code p ≈ 0.91-0.96, agent 0.80-0.86,
    prose 0.67-0.73). On that basis a "flat-p" ceiling suggested **+24-40 % at
    W16** and almost nothing at W4.
  * Break-even for a heavier draft: an m× slower draft needs an acceptance
    multiplier of 1 + 0.089(m − 1) at W4 and 1 + 0.249(m − 1) at W16.
  * Recommended path:
    1. Per-position (PosS-style) entry adapters first.
    2. Then EAGLE-3-style multi-layer tap fusion on the existing MTP head: taps
       at QSA layers 3 and 23, with zero-initialised projections so day-0
       output is bit-identical.
  * Not recommended:
    * SGLang's stock EAGLE3 mode, which discards `--speculative-token-map`,
      expects a Llama-shaped draft, and has no aux-capture wiring for this
      model;
    * a from-scratch EAGLE-3 draft, which would discard the 2.6 B pretrained
      head;
    * tree drafting at W4, which is MoE-bound: break-even +11.3 % against a cap
      of +6 % for code;
    * n-gram chains (1.1).
* **Phase 0** (per-position acceptance, measured offline on the author's
  private data). No result or conclusion is published. The PosS-style adapters
  were not built.
* **Phase 1** (EAGLE-3 tap fusion, small pilot dataset, zero-initialised 13.1 M
  projection, frozen head). Evaluated offline on private data against a
  +3 % / +6 % (k = 3 / 7) gate; no result or conclusion is published. Tap fusion
  is not part of the production configuration (the training code keeps it as
  an off-by-default option).
* **Architectural note.** The MTP entry already consumes all four
  hyper-connection residual streams (4 × 2,560). EAGLE-3's reported +14 % comes
  from going from one stream to a fusion, and this model already has that.
* **Side finding.** With a frozen head and only the entry trained, the rollout
  loss's gradient is exactly zero (inputs are detached per step).

### 1.8 DH1-DH7: dynamic draft-head shortlist

* **Idea.** The draft head (49,152 FP8 rows) costs about 81 µs of each
  220-230 µs draft forward, × 14 at W16. Replace it with a cheap selector, a
  top-K shortlist and rescoring of the shortlist with the real FP8 rows,
  falling back to the full head when the selector is unsure.
  * Budget: at most 36.2 µs per forward for +3.9 % of step at W16.
* **DH1.** Untrained selectors (low-rank SVD, unigram prior, random
  projection) were screened against a quality and a time gate on private data;
  no result or conclusion of that screen is published. DH2 went on to a learned
  selector.
* **DH2.** A learned rank-128 selector with K = 1,024.
  * Its shortlist quality and its eager-mode cost were measured offline with
    private data as input; no result or conclusion is published.
* **DH3.** A fused path inside one CUDA graph: two selector GEMVs, a 64-CTA
  approximate top-K, gather-rescore-confidence, and a graph IF node for the
  fallback.
  * Its standalone kernel timings used private DH2 states as input and are not
    published. The in-server cost on the public workloads is under DH4.
* **DH4** (in-server).
  * Rescoring follows the production arithmetic (sequential 256-wide FP32
    chunks, then BF16 rounding). In-server cost 35 µs per forward; no fallback
    executed in the traced runs (the frozen threshold is −∞).
  * t/s at W16: code +3.5 %, prose-en +13.4 %, prose-ja +8.8 %, agent +0.9 %.
  * **Acceptance at W16: code −5.1 %, agent −8.8 %** (gate ±1 %). Rare tokens
    fell outside the shortlist, and the calibrated threshold did not fire in
    the traced runs.
  * Also found: a `dataclasses.replace` dropped the shortlist context, so the
    first test arm was silently a no-op.
* **DH5.** K = 2,048 and a margin < 0.25 threshold, giving 3-12 % fallback.
  * W16 against DH4-era controls: +8.6 % to +14.8 % t/s, acceptance +0.5 % to
    +6.1 %.
  * W4 failed: prose-ja −4.1 %.
  * The confidence map was wrong on some tokens: 0.92 confident while the true
    probability was ≤ 0.034.
* **DH6** (same-session ABAB, W16).
  * code +13.5 %, prose-en +5.4 %, agent +6.7 %.
  * **prose-ja t/s −4.4 %, acceptance −10.5 %** (pairs −14.7 % and −5.9 %).
    NO-GO.
* **DH7.** The selector was retrained on *real* captured draft-head inputs
  and restricted to caller-marked code requests at W16.
  * It was evaluated on the author's private prompts; no result or conclusion
    of that evaluation is published. The shortlist is not part of the
    production configuration.
* **Lesson.** In-server, the losses concentrated at the rejection boundary:
  rare tokens fell outside the shortlist (DH4), and a frozen confidence map
  could not detect a missing strong candidate (DH5).
* **Source.** `lab-notes/DH3_SHORTLIST_KERNEL.md`, `lab-notes/DH4_SHORTLIST_SERVER.md`,
  `lab-notes/DH7_SELECTOR_REPAIR.md`; project log 2026-09-07/08.

### 1.9 E0: FP8 for the MTP `fc_hidden` / `fc_embedding` projections

* **Measured** (2026-09-08).
  * `fc_hidden` 12.66 -> 7.33 µs per call, which is 75 µs/step at W16 (0.45 %
    of step).
  * `fc_embedding` was already replaced by the precomputed table (R3), so FP8
    changes nothing there.
  * The acceptance gate (±0.5 %) failed in 7/8 cells, but control-to-control
    drift (W16 code acceptance 11.4 vs 9.7) was far larger than the effect.
* **Verdict.** Not worth it. The whole prize was below 1 %.
* **Source.** `lab-notes/E0_MTP_FC_FP8.md`.

### 1.10 MTP entry fusion and `MTP_FC_GEMV`

* **Fused entry** (norm × 2 + GEMM × 2 + add in one Triton kernel;
  `SGLANG_MTP_ENTRY_FUSED`).
  * The first version exceeded shared memory (245 KB).
  * The working version was slower than the reference: 94-157 µs against
    63-75 µs.
* **`SGLANG_MTP_FC_GEMV`** (the entry's `fc` projections through the skinny
  BF16 GEMV). It made no difference or made things worse. The two 13 MB `fc`
  matrices stay L2-resident across the 15 draft iterations, so cuBLAS was
  already fast. Re-evaluated on 2026-09-05 with the same result.
* R3 (the embedding table) later removed half of the entry cost more cheaply.

### 1.11 Draft-model quantisation flag

`--speculative-draft-model-quantization fp8` had no effect. The in-model MTP
inherits the target's ModelOpt config, and the draft phase stayed at 6.75 ms.
It was solved differently (`optimizations.md` §A7).

---

## 2. Dense weights and quantisation

### 2.1 Dense FP8 W8A8 through CUTLASS (the fork's opt-in category path)

* **Measured** (2026-09-02, W4):
  * 29-36 % slower on the workloads recorded: code-edit 210 -> 134 t/s,
    prose-ja 117 -> 83; step 19 -> 28 ms.
  * No acceptance change beyond run-to-run noise was seen.
* **Cause.**
  * The CUTLASS W8A8 FP8 GEMM has a fixed cost of about 40 µs even for skinny
    shapes. 386 calls per step × about 44 µs = 17 ms.
  * The fused BF16 HC mix kernel was also replaced by unfused FP8 GEMMs.
* **Lesson.** Decode at M ≤ 16 needs a weight-only GEMV (W8A16), not a GEMM.
  This led to the Triton W8A16 GEMV.

### 2.2 FP8 router gates (`mlp_gates` category)

* **Measured** (2026-09-02, recommended sampling):
  * Verify time unchanged.
  * Acceptance code-edit 11.6 -> 9.2, prose 2.35 -> 2.26.
  * Needle PASS.
* **Caveat.** This was measured before the switch to greedy comparisons, and
  sampling alone moved code-edit acceptance 7.2-11.7. The acceptance loss is
  therefore not certain.
* **Result.** The router stayed BF16, and the MTP router too. The BF16 router
  later moved to a Triton GEMV; a 4-repeat W16 A/B showed no acceptance effect
  of that beyond noise.

### 2.3 First split-K W8A16 GEMV attempt

* **Measured.** Split-K plus a separate reduce kernel made the in-server
  W8A16 total worse: 5.94 -> 6.81 ms at W16 and 4.38 -> 4.91 ms at W4.
* **Cause.** The microbenchmark's "before" figures were 2-3× slower than the
  server (for example out_proj 48 µs against 14 µs in the server), so the tuning
  targeted the wrong regime. The root cause was found a day later: the
  benchmark used `[K, N]` weights, while the server passes `[N, K]`.
* **What worked instead.** In-launch fix-up split-K (v2), followed by the v3
  retune in the correct layout.
* **Lesson.** A benchmark must reproduce the in-server per-shape medians before
  its tuning is trusted.

### 2.4 N1: NVFP4 (W4A16) dense GEMV, stages A, B and C

* **Kernel** (2026-09-06).
  * A new Triton W4A16 NVFP4 GEMV: nibble unpack via `tl.join`, fp16 bit-trick
    dequant, `BLOCK_K = 512` so one row's block scales fill exactly one 32 B
    sector.
  * Bandwidth: 83 % of roof on the draft head (1.49× faster than FP8), 88.5 %
    on the target lm_head (1.57×).
  * Weaker shapes: GDN out_proj only 47-52 % of roof (1.08-1.13×); qkvz and
    attention qkv 58-77 %.
  * NVFP4 moves 0.5625 B per element against 1.0, a 1.78× byte advantage.
    Break-even therefore needs 55 % of roof, not 40 %.
  * The FP8 kernels were already at 96-100 % of roof on the lm_heads.
* **Stage A** (draft hot-vocab head).
  * An offline acceptance gate was run first on private data; its result is
    not published.
  * Kernel in-server 81.4 -> 50.4 µs; W16 step −2.1 %.
  * In the server A/B (same hour), acceptance fell:
    W16 code 11.77 -> 10.05, agent 5.75 -> 5.20; W4 prose-en 2.58 -> 2.46.
    Net negative. Left off (`SGLANG_MTP_LMHEAD_NVFP4=0`).
* **Stage B** (target lm_head).
  * 400 -> 235 µs (1.70×), W4 step −1.4 %, frees 278 MB.
  * Synthetic needle PASS. Greedy agreement 1.000 over the first
    256 tokens on code-edit, 2 repeats; the other workloads could not be
    compared, since two runs of the unchanged build already disagreed there.
  * Same-hour A/B with stages A and B on: acceptance W4 prose-en 2.58 -> 2.38,
    agent 3.22 -> 3.05; W16 agent 5.75 -> 5.08. That is a net t/s loss (agent
    −5 % to −12 %). Off (`SGLANG_LMHEAD_NVFP4=0`).
* **Stage C** (GDN qkvz, attention qkv).
  * qkvz 29.3 -> 19.1 µs, but the step did not move (10.50 -> 10.55 ms).
  * The GEMV had been hiding MoE GEMMs that run at the expert-bandwidth limit.
    Exposing them cost +519 µs. A missing split-output path added another
    +115 µs.
* **Bugs found only in the server**, each of which produced plausible numbers
  rather than an error:
  * a dequant fallback that was not graph-safe;
  * deleting the `lm_head.weight` Parameter caused a silent fallback to a dense
    matmul;
  * a stage-B OOM that silently ran FP8 and "passed";
  * `QKVParallelLinear` missing its prefix, so only 36 of 48 layers were
    quantised.
* **Lesson.** Quantising the *target's* output layer changes what the draft was
  trained to predict. The server A/B on the public workloads is the gate that
  counts. On a GPU
  where the big GEMVs are already at the bandwidth roof, fewer bytes only help
  if nothing else is hidden underneath.
* **Source.** `lab-notes/N1_LOG.md`; commits `0794e3f51f` to `6e75b5fe9f`.

### 2.5 FP8C / FP8D: lossless coding of FP8 dense weights

* **FP8C survey** (CPU).
  * Entropy per FP8 byte is 6.53-6.73 bits; the redundancy is mostly in the
    exponent.
  * zlib-9 saves 16.1 %. A byte Huffman code with a restart every 256 weights
    saves 13.2 % (W4-weighted) and 14.7 % (W16-weighted). That clears the 11 % /
    13 % thresholds.
  * But decoding must be fused into the GEMV at about 1.85 TB/s of output. The
    margin left is only 2.2 % / 1.7 % of dense time. Published GPU entropy
    decoders run at 250-600 GB/s.
* **FP8D.** A fused lane-serial Huffman GEMV was built (Triton plus inline PTX,
  bit-exact).
  * Target lm_head: **397 µs -> 13,861 µs (35-39× slower)**, an effective
    46 GB/s.
  * Killed. A parallel-decode design was not attempted.
* **Source.** `lab-notes/FP8_CODING_SURVEY.md`, `lab-notes/FP8D_FUSED_DECODE.md`.

---

## 3. Kernel fusion and launch reduction

### 3.1 HC fused persistent kernel ("B"): combine + norm + mix in one kernel

* **What was tried.** A single persistent Triton kernel for combine, per-branch
  RMSNorm and the rank-320 low-rank mix, using grid barriers. Four rounds of
  work (`SGLANG_HC_FUSED=1`): weight prefetch above the barriers, a tail-CTA
  clear, a pipelined reduction.
* **Measured.**
  * Numerics matched: mixed max 2e-3, residual within 1 ulp.
  * Time was only at parity with the old chain (17.7-24 µs vs 18.5-22.7 µs), and
    no faster in the server at 22.0 ms/step.
  * The old mix kernel ran 18.5 µs cold against 12.5 µs L2-hot. That gap is the
    structural cost.
* **What worked instead.** The three-kernel non-persistent MIX2 (14.3 µs, then
  9.9 µs in FP8).
* **Earlier variant.** `SGLANG_HC_MIX_FP8`, FP8 weights in the old persistent
  kernel, reached only 13.6 -> 12.3 µs because the kernel is barrier-bound.

### 3.2 HC combine fusion (gate + apply in one Triton kernel)

* **Idea.** 96 boundaries × 2 launches per step looked launch-bound.
* **Measured.** The CUDA pair costs 2.14 µs per call (1 row) and 2.85 µs
  (16 rows). The fused Triton kernel cost 3.26 / 3.74 µs; with its barrier
  removed entirely it still cost 2.53 / 2.88 µs. In the server: **+130
  µs/step**.
* **Cause.** sgl-kernel already gives the apply kernel PDL, so the pair never
  paid two full launches. There was no launch overhead left to reclaim.
* **Result.** Kept off (`SGLANG_HC_COMBINE_FUSED`) as a recorded negative
  result.
* **Related loss.** Moving the combine gate *into* K0 (`SGLANG_HC_GATE_EARLY=1`)
  cost +114 µs/step, because K0's grid is too small to absorb four extra
  dependent loads.

### 3.3 K2: persistent megakernel with grid barriers ("glue islands")

* **Measured cost per dependent stage** at 188 CTAs:

  | Mechanism | Cost |
  |---|--:|
  | Separate kernel | 0.823 µs |
  | Separate kernel with PDL | 0.668 µs |
  | Grid-wide barrier | 1.264 µs |
  | Per-CTA flag | 1.256 µs |

* **Result.**
  * Over the 1,011 (W4) and 1,404 (W16) serial boundaries per step, a
    megakernel would *add* 0.45-0.62 ms.
  * Even a free barrier caps the gain at −0.14 / −0.20 ms.
  * A measured HC-island megakernel was 18-26 % slower than the split version.
  * GEMVs cannot be stages either: a 188-CTA persistent grid loses 11-13 % of
    bandwidth and loses split-K, making them 1.8-2.2× slower.
* **What worked instead.** PDL (K1 in the same study), −5.5 % at W4.
* **Lesson.** On this GPU a barrier costs more than the launch it replaces.
* **Source.** `lab-notes/MEGAKERNEL_SPEC.md`.

### 3.4 J1: folding the HC statistics kernel (K0) into K1

* **Idea.** K0 had been costed as a pure launch worth 1.41 µs.
* **Measured.** The chain got slower: 9.59 -> 13.52 µs (M = 4, plain boundary)
  and 10.13 -> 19.10 µs (fused boundary). Per step that is +0.23 to +0.83 ms.
* **Cause.** K0 is a layout producer. It writes the normalised BF16 tile that
  K1's `tl.dot` reads, and it runs the previous boundary's apply. Rebuilding
  that inside K1 costs +1.67 µs, and re-reading the branch for the statistics
  costs +2.40 µs, against −1.60 µs for the deleted launch.
* **Lesson.** Exclusive time says what a kernel occupies, not what deleting it
  saves. Ask what the consumer loses.
* **Source.** `lab-notes/J1_LOG.md`.

### 3.5 J2: draft attention grid

* **Idea.** Speed up the draft attention (`kernel_mha`) by splitting its grid,
  estimated at −145 µs/step at W16.
* **Measured.**
  * On SM120 the draft attention resolves to the XQA kernel with an 18-CTA grid:
    2,067 selected columns = 9 tiles.
  * A split sweep from 2 to 32 found the current 9 fastest (11.27 µs).
  * Time was flat from KV length 256 to 2,067, i.e. a fixed cost of about 10 µs.
* **Result.** Already optimal. The next day PDL hid 93-96 % of this kernel
  anyway.

### 3.6 H1-C: split-aware GEMV grid for `in_proj_ba`

* **Measured.** Bit-exact, but it recovered only about 0.2 of the 0.9 µs per
  call that the two-destination store costs at M = 16. That is below the 1 µs
  bar, so the flag stays off.
* An earlier "single store for non-straddling blocks" branch was measured
  slower in the server (6.7 -> 8.0 µs) and reverted.

### 3.7 L2 prefetch and L2 persisting pin

* **Prefetch of the next layer's dense weights** (`SGLANG_L2_PREFETCH=1`).
  * The GEMVs did hit L2: qkvz 29.3 -> 18.3 µs.
  * But the "idle" glue windows are not DRAM-idle. The HC mix kernels stretched
    2.3× and the routing prologue slowed 14 %.
  * Net: W16 +4.9-5.3 % slower, W4 +13 % (partly drift).
* **Pinning the draft head in the persisting L2 carve-out**
  (`SGLANG_L2_PIN_DRAFT_HEAD=1`).
  * At W4 the draft forward went 82 -> 29 µs, but the step was 8 % slower: a
    34 µs per-step touch, and MoE GEMMs 6-12 % slower with only 48 MiB of normal
    L2 left.
  * At W16 the head stayed cold. **Any other CUDA context, including the desktop
    compositor, wipes the persisting set** within about 3.5 ms.
* **Device facts.** Persisting carve-out at most 80 MiB. L2 hits give dense
  GEMVs only 1.4-1.8×.
* **Revisit.** Worth another try only on a GPU with no display attached.
* **Source.** `lab-notes/L2_PREFETCH_SPEC.md`, `lab-notes/L2_LOG.md`.

### 3.8 HC1: narrower hyper-connection bottleneck (untrained)

* **Screen.** The rank-320 HC mix was truncated to 256, 160 or 128 (keeping the
  top units by contribution), plus a "gate-const" variant that replaces the
  lowest-variance gates with their mean. The quality battery covered 20,010
  items (candidate − prod, non-inferiority margin 0.5 pp):

  | Variant | Result |
  |---|---|
  | hc256 | MMLU −2.0 pp FAIL |
  | hc160 | GSM8K −2.7, MMLU −7.7, HumanEval −14.0, JCQA −2.6 pp: all FAIL |
  | hc128 | HumanEval −32 pp |
  | gate-const | all inconclusive (−0.1 to −1.4 pp) |

* **Ideal gains** with perfect packing: hc256 +1.5-1.8 %, hc160 +3.7-4.6 %,
  gate-const ≤ +2.2 %.
* **Result.** Only hc160 is worth anything, and only with recovery training.
  That is deferred.
* **Source.** `lab-notes/HC1_BOTTLENECK_SCREEN.md`.

---

## 4. MoE

### 4.1 Custom Triton MoE kernel

* **Estimate.**
  * A split-K custom kernel would be slower than CUTLASS if Triton streamed at
    ≤ 1,150 GB/s on this box (77 vs 66 µs at W4); that rate is an assumption,
    not a measurement from this project. W16 is compute-bound for non-tensor-core
    code.
* **Verdict.** NO-GO before implementation.
* **Also blocked.** `--moe-runner-backend flashinfer_trtllm` does not support
  sm_120 (trtllm-gen cubins exist only for sm100/103/107), and the CuTe DSL
  path needs tcgen05.
* **Source.** `lab-notes/MOE_SMALLM_SPEC.md`.

### 4.2 A8: SwiGLU + NVFP4 quantisation in the GEMM1 epilogue

* **Idea.** Delete `doActivationKernel` by doing its work in the GEMM1 epilogue.
* **Finding.** GEMM1 runs with `swap_ab` (tactic 16/17), so gate/up pairs lie
  along CUTLASS M. Fusing would need:
  * a new half-width FP4 store visitor;
  * TRT-LLM's swizzled scale layout;
  * bit-exact quantiser numerics;
  * a 97-TU JIT rebuild.
  
  Estimated 4-8 days for about 1.5 %.
* **The kernel's true cost.** `doActivation` is 3.6 µs, of which about 0.31 µs
  is real memory work. Its exclusive time is only 115-169 µs/step, because
  26-33 % already hides under GEMM2.
* **Cheap variant (A8-lite).** Offsets in shared memory, deferred PDL wait:
  −0.52 µs/call at T = 4, −0.19 at T = 16. That is about 0.01-0.03 ms/step.
  Rejected.
* **What worked instead.** Folding the memset into `doActivation` (G2-1).
* **Source.** `lab-notes/A8_FEASIBILITY.md`.

### 4.3 G1 split-K for the grouped GEMM

* **Implemented.** Extra CUTLASS groups over K slices, with a patched
  block-scale stride, reduced in `doActivation`.
* **Measured.**
  * W4: GEMM1 −1.14 µs, but reduction +2.24 µs, a net −0.60 ± 0.50 µs (a wash).
  * W16 (planner picks s = 1): **+2.32 µs per call**, from loading one scalar
    in the activation kernel.
* **Result.** Off. Only group packing was shipped.
* **Finding.** No "wave-quantisation staircase" exists. GEMM1 is linear in D
  when warm, and the main loop is at the read roof.
* **Source.** `lab-notes/G1_LOG.md`.

### 4.4 G3: top-k reduction in the draft MoE

* **Result.**
  * The cost ceiling was small: the draft MoE is 22 % of the draft phase, which
    is 23 % of a W16 step. Even top-10 -> 2 would gain at most +1.6 %.
  * The acceptance cost was evaluated with the offline evaluator on the
    author's private data against a +3 % gate; no result or conclusion of that
    evaluation is published. G3 is not part of the production configuration.
* **Side finding.** Verify visits 3,634 experts per step against 140 for the
  draft at W16, so any per-expert saving is worth 26× more on the verify side.

### 4.5 P3 / P4: contribution-aware pruning

* **P3** (CPU study).
  * Expert output norms were recorded over 316 k MoE calls (27.9 M routes).
  * Scoring s = w·‖F_e(h)‖/‖h‖ with a per-expert norm-table proxy removes 4-12
    more distinct experts than τ = 0.08 while having a smaller removed-score p95.
    Estimated −3.2 % to −5.1 % step.
  * The added scoring had only 23-36 µs/step of budget.
* **P4** (implemented in the FlashInfer prologue).
  * Correct: 0 mismatches over 55.7 M route comparisons; P2-compatible
    bit-identical.
  * D fell by 4-11 more.
  * But the prologue grew from 12.4 to 18.0 µs per call, which is +265 to
    +293 µs/step exclusive. Step change −0.9 % to +2.1 %.
  * **Acceptance −16 % on W16 code and agent**: the target's outputs changed
    enough to diverge from the draft.
  * NO-SHIP. The quality battery was not run.
* **Lesson.** Pruning that changes more of the target's computation costs
  acceptance, because the draft was trained on the unpruned target. And
  "moving work into the prologue" still leaves it on the critical path.
* **Source.** `lab-notes/P3_CONTRIB_PRUNE.md`, `lab-notes/P4_CONTRIB_PRUNE_SHIP.md`.

### 4.6 T10: raising the pruning threshold τ 0.08 -> 0.10 (2026-09-09)

* **Why it looked attractive.** The 09-07 map projected τ = 0.10 at −5.3 %
  (W16) and −5.7 % (W4) of step time. But the dropped routing mass per row rises
  steeply: mean 14 % -> 38 %, p95 38 % -> 62 % on W4 code (census replay).
  A 20 % per-row mass cap removes the gain entirely.
* **Shipping gate** (quality battery plus A/B):
  * MMLU and JCQA: no difference between the arms (project log; the per-item
    files of this run are not published).
  * GSM8K +0.09 pp on the questions both arms finished, −0.53 pp overall.
    Inconclusive: truncation became a coin flip.
  * HumanEval (164) is underpowered for a 0.5 pp margin when the difference
    is small; non-inferiority was not established in this comparison.
  * **Behaviour changed in ways τ alone does not explain.** Agent-loop requests
    hitting `max_tokens` went from 2/8 to 7/8, and one prose-ja block's
    acceptance rose 23 %.
  * The t/s gains (+4 % to +17 %) were therefore not trusted.
* **Verdict.** NO SHIP; production stays at τ = 0.08.
* **Next time.** Use a HumanEval margin of 2 pp (or add MBPP), gate on the
  truncation rate and the output-length distribution, and match output lengths
  in A/Bs.
* **Source.** Project log 2026-09-09. The T10 spec file is not in this document
  set.

### 4.7 S1: 2:4-style sparse FP4 experts (untrained)

* **Quality** (4-of-8 magnitude masking with dense kernels, 12,006 scored items):

  | Variant | GSM8K | MMLU | HumanEval | JCQA |
  |---|--:|--:|--:|--:|
  | sparse-all | −3.6 pp | −17.4 pp | −6.7 pp | −1.8 pp |
  | sparse-down only | −2.1 pp FAIL | −2.9 pp FAIL | | |

  English sanity output from sparse-all was broken: the repeated 4-gram rate
  went from 3 % to 92 %.
* **Speed.** CUTLASS sm120 sparse NVFP4 was *slower* than dense at M ≤ 16:
  gate/up +12 %, down +18 % at D = 19. It also needs block-32 scales, while the
  checkpoint uses block-16 scales and 89.6 % of adjacent scale pairs differ.
* **Revisit.** Would require recovery training.
* **Source.** `lab-notes/S1_SPARSE_FP4.md`.

### 4.8 E1: narrower experts (intermediate 640 -> 512 / 576, untrained masks)

* **Quality.**
  * w512: GSM8K −2.5, MMLU −10.0, HumanEval −3.1 pp. FAIL.
  * w576: MMLU −3.4 and HumanEval −2.4 pp FAIL; the rest inconclusive.
  * Many losses were format failures: truncated or unparsable answers.
* **Ideal gain** with physical packing: w512 +3.0 % (W4) and +4.8 % (W16).
  w576 cannot reach 3 %.
* **Verdict.** Needs recovery training, estimated at GPU-days. Deferred.
* **Source.** `lab-notes/E1_EXPERT_WIDTH_SCREEN.md`.

---

## 5. Adaptive width

### 5.1 Fixed W8, and W6

* **2026-09-05** (v3 head): a clean W8 re-measurement gave agent-loop 269 t/s,
  less than W4's 289, with served acceptance 4.33.
* **2026-09-07 width sweep** (full stack, v5 head): fixed W8 gave agent-loop
  +15 % over W4 (365 vs 312 t/s), but −12 % on prose.
* **W6** has the same step time as W8 (12.70 vs 12.82 ms) and lower acceptance
  on every measured workload, so it was not the best fixed width on any of them.
* **Result.** W8 became a candidate inside the adaptive policy (WA5), not a
  launcher default.

### 5.2 WA2, WA3 and WA4: early three-width [3, 7, 15] attempts

* **WA2 iteration 1.**
  * Setup: the two-state config with 7 added and a refitted step model.
  * The upward threshold (1.12 × 1.30 = 1.456×) was structurally unreachable:
    even a fully accepting chain predicts at most 1.42× (S3 -> S7).
  * code-edit got stuck at W8: −17.5 %.
* **WA2 iteration 2** (margins lowered to 1.134×).
  * Reversal grace doubled unboundedly (40 -> 640 batches), so code oscillated
    15 ↔ 7 and a 640-batch hold at W8 spilled into prose.
  * −1.8 % / −15.8 % code, −10 % prose-en.
* **WA3** (per-destination `down_margin`, `max_grace 80`, adjacent-only
  promotion).
  * Width selection was now correct.
  * But W8 cost 1.45× W4 inside the adaptive server, against 1.23× for fixed
    profiles. agent −5.3 % / −0.7 %.
  * Declared the last iteration; this led to X3.
* **WA4** (after the X3 fix).
  * The two-candidate fix alone passed (+7.5 / +3.8 / +1.7 / +5.3 % against
    pooled controls). But control drift was +8 % to +23 %, with other GPU applications
    running on the desktop.
  * Three candidates: agent +19 %, code −8 %, prose-en −7.5 %. FAIL.
  * X4 then showed the same-width penalty did not reproduce after a restart.
  * WA5, re-run on a quiet night, passed and was adopted (`optimizations.md`
    §F4).
* **Lesson.** One scalar margin cannot separate a real upward move from a false
  one. Grace windows need a cap. And a measured "penalty" can be environmental.
  Always check it on a quiet machine before engineering around it.

### 5.3 WA6 and WA7: request-level width priors

* **WA6.** Start each request's controller from a per-domain prior (named by
  the caller, or from a frozen token-only predictor), with a measured per-width
  cost table. It was measured (ABBA) on a mixed request stream built from the
  author's private prompts; no result or conclusion of that comparison is
  published. WA6 is not part of the production configuration.
* **WA7.** Pin Japanese requests (kana + Han ratio ≥ 0.5) to S = 3.
  * Japanese +2.7 % (CI −5.7 to +9.3); other domains moved +3 % to +5 % in
    directions that could not be caused by the change.
  * Control drift was −4 % to −13 % in the daytime.
* **Verdict.** WA7 FAIL (public BN1 sets). Neither width prior is in
  production, which keeps the WA5 controller.
* **Source.** `lab-notes/WA6_REQUEST_PRIOR.md`, `lab-notes/WA7_JA_PRIOR.md`,
  `lab-notes/WJ1_WA_JAPANESE.md`.

### 5.4 Draft early exit (C1 option 2)

* **Idea.** Stop a W16 draft early when confidence drops.
* **Why it cannot work.** Verify rows are always real tokens, and padded verify
  rows still route through the MoE: there is no row-validity mask in the
  non-EP MoE path or the FP4 MoE kernel. So early exit cannot shrink D; it
  reduces to "replay a shorter graph", which is exactly choosing S = 3.
* **Side finding.** CUDA-graph batch-padding rows are not zeroed and do
  activate experts.

---

## 6. Findings that were not rejections

### 6.1 QL1: long-context cost map

* **What it measured.** Contexts of 8k / 32k / 64k / 96k tokens at W4 and W16.
* **Step time grows only 3.7-4.6 %.**

  | Profile | 8k | 96k |
  |---|--:|--:|
  | W4 | 9.43 ms | 9.77 ms |
  | W16 | 17.0 ms | 17.8 ms |

  QSA selects a fixed 2,048 tokens, so packing and attention stay flat. Only
  the index scan and score/top-k grow.
* **Headroom.**
  * At 96k W16 the index scan is 2.8 % of the step.
  * Deleting score/top-k entirely would give at most +4.8 % (W4) and +5.0 %
    (W16) at 96k, and under 3 % at 32k or below.
* **Result.** A bounded prototype is justified only for ≥ 64k contexts. It is
  not a priority for short and medium use.
* **Source.** `lab-notes/QL1_LONG_CONTEXT_MAP.md`.

### 6.2 T = 6 / T = 10 verify-graph fallback

* **What happens.** Non-power-of-two verify widths hit an indexing fallback
  inside the verify graph: 109 extra `index_elementwise` launches, more
  memcpys, and a slower prefix sum. That costs **+0.9 ms (T = 6) and +2.8 ms
  (T = 10)** against the linear step model.
* **How it was found.** Width S = 5 and S = 9 measurements sat well above the
  fitted line. Separately, with top-k = 1 the fork forces
  `num_draft_tokens = steps + 1`, which silently overrode a mislabelled "W8
  draft with W4 verify" configuration.
* **Consequence.** All width policies use S ∈ {3, 7, 15}.
* **Source.** `../bench/prof/OVERHEAD_REPORT.md`.

### 6.3 "Host overhead grows with draft steps": a measurement artefact

The old trimmed metric under-reported GPU busy time by 0.8-3.6 ms, and the error
grew with kernel count. It is described in `measurement.md` §4. Measured
properly, GPU idle is 0.15-0.21 ms at every width.

### 6.4 Compositor interference

Compositor interference is real: about 7.5 % of the W4 step at 4K 160 Hz. It
has no complete software fix. Stream priorities, MPS and green contexts do not
isolate the compute context from the compositor. MIG would require disabling
display output and would leave only 48 GB. See `measurement.md` §5.
