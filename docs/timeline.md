# Timeline (2026-08-28 to 2026-10-02)

A dated record of how single-stream (batch size 1) decode of Qwen3.8-Flash-Next
(NVFP4 checkpoint `RadixArk/Qwen3.8-Flash-Next-NVFP4`) on one RTX PRO 6000
Blackwell Max-Q moved from roughly 120-210 t/s to 260-670 t/s. The work was
done by one person, implemented with AI coding assistants under human review,
and every change was validated on the GPU before it was adopted.

How to read the numbers:

* Workloads are the benchmark harness's prompt sets: `code-edit` (mechanical
  code editing, highly predictable), `prose-en`, `prose-ja` (English and
  Japanese prose), `agent-loop` (tool-using agent turns) and, early on,
  `long-ctx`. Unless a line says otherwise, a triple such as `399/280/349` is
  **code-edit / prose-en / agent-loop** client decode t/s.
* Profiles: **W4** = NEXTN chain with 3 draft steps and a 4-token verify,
  **W8** = 7 steps / 8 tokens, **W16** = 15 steps / 16 tokens, **wa** = one
  server that picks the width adaptively per request (first {3,15}, later
  {3,7,15}).
* "greedy" means temperature 0. Numbers from before 2026-09-02 afternoon use the
  model's recommended sampling settings, which makes acceptance noisier.
* Client t/s on this machine moves by ±10-20 % between runs because the
  desktop shares the GPU (see `measurement.md`). Treat single t/s figures as
  indicative. The kernel-level step times are the more reliable signal.
* Reasoning ("thinking") tokens count as decoded tokens in every t/s figure.
* Commit hashes are those of the original working branch; `../patches/sglang/SERIES.md`
  maps each one to its patch number. Times are JST.
* Rows backed by published result files are marked with the file name. Other
  numbers come from the project log and lab notes (`lab-notes/`) and have no
  raw file in this repository.

## 2026-08-27 / 08-28: base

* The fork `jpezzulli/sglang-rtxpro6000` at commit `16e5682aad` (2026-08-27) is
  the base of the patch series.
* 08-28: one hardening commit before the project proper
  (`recover_final_state`: int64 slot-index cast plus a dense-pitch assert).

## 2026-09-01: plan and CPU-only preparation (GPU busy with other work)

* Starting point: the stock fork's NEXTN/EAGLE linear-chain speculative decode
  (3 draft steps). The measured baseline is the 09-02 row below.
* Offline replays of request-local n-gram suffix drafting were run on the
  author's private data (results not published). An `NGRAM_CHAIN` draft provider (`6d229a69f0`) and an opt-in per-category
  dense FP8 path (`2ffd61a800`) were written and tested on CPU only.
* Structural facts established: the "n-gram table" in the model is a hashed
  input embedding and cannot be used for drafting; the linear draft chain has
  no inherent width limit other than a QSA guard found the next day.

## 2026-09-02: first GPU day

Every run this day used the default 300 W power limit; 325 W was set from 2026-09-03 on.

* **Baseline**, NEXTN steps 3, recommended sampling, 300 W power limit:
  code-edit 211 / prose-ja 117 / prose-en 129 / agent-loop 174 / long-ctx
  154 t/s (mean of 3, `results/runs/baseline-nextn.jsonl`). Step time was about
  18-19 ms for every workload, about 2,500 kernels per step, and the GPU was 98 %
  busy, so the limit was kernel work and not the host.
* Dense FP8 via CUTLASS W8A8 **rejected**: about 35 % slower (step 19 -> 28 ms).
* Reduced draft vocabulary `hot_32768`: agent-loop 211 (+21 %), code-edit 224.
* Triton W8A16 GEMV + `hot_32768`: agent-loop **244** / prose-ja 165 / long-ctx
  203 / code-edit **261** / prose-en 163, step 12.5-14 ms (`5058a8582d`).
* QSA pending index-key ring generalised (`aa61a26bad`). Draft widths above 4
  became possible. Static W8 then gave code-edit 348-367 t/s (acceptance
  6.9 of 8), and static W16 gave code-edit 429-433 t/s (acceptance 11.6 of 16),
  while prose and agent work were better at W4. From this point the launcher
  had per-use-case profiles.
* Rejected the same day: stock `--speculative-adaptive` (acceptance collapsed,
  then a crash), `NGRAM_CHAIN` at width 16 (worse than MTP-15), and FP8 router
  gates.
* Greedy reference numbers set for later comparisons: W4 code-edit 271
  (acceptance 3.68, 13.6 ms) / prose-en 172 (2.28) / agent-loop 220-231 (3.03);
  W16 code-edit 345-393 (9.6-11.4, 26-29 ms).

## 2026-09-03: glue fusion and the large kernels

* Trimmed-kernel baseline: **W16 code-edit 3,175 kernels/step, 22.75 ms
  (wall 25.3 ms)**; **W4 prose 2,299 kernels, 12.87 ms (wall 15.3 ms)**.
* Adopted: QSA per-step seq_lens table, direct top-k chain with the hot-vocab
  map and a fused QSA index lookup, A_log cache, zero-bias cache, fused FP8 KV
  store, MTP dense FP8.
  Mid-day state: W16 3,175 -> 2,713 kernels, trimmed 22.75 -> 20.6 ms, wall
  25.3 -> 22.6 ms, greedy code-edit **454/443 t/s**; W4 2,299 -> 2,113
  kernels, wall 15.3 -> 13.4 ms.
* Then conv1d `BLOCK_N` 64, three-kernel HC mix (MIX2) and W8A16 GEMV v2.
  End of the morning: **W16 trimmed 19.35 ms / wall 21.1-21.5 ms; W4 trimmed
  11.5-11.7 ms / wall 14.1 ms**, with needle PASS and no acceptance change
  beyond run-to-run noise.
* Evening: GEMV v3 retune after finding that the v2 shape table never matched
  the weight layout the server actually passes. The GDN strided q/k/v patch,
  HC-mix FP8 weights, fused shared-expert gate_up+SiLU, direct draft-logits
  write and router BF16 GEMV were implemented and merged in the early hours of
  09-04.
* Rejected: HC fused persistent kernel, MTP entry fusion, and the first
  split-K W8A16 attempt.

## 2026-09-04: merge, a split-K bug, MTP head training v1-v3

* After merging all of the above, output turned to garbage with every flag on.
  Bisection found that the split-K scratch was shared across two concurrent
  streams. Fixed with per-stream scratch slots (`ac2e8cb9d2`).
* **W4 validated (09:03)**: kernels 2,219 -> 2,009, trimmed 11.5 -> 10.7 ms, wall
  11.5 ms, t/s code-edit 268 -> **319** / prose-en 168 -> **208** / agent-loop
  226 -> **287**, no acceptance change beyond run-to-run noise.
* **W16**: trimmed 19.9 -> 18.5 ms. A 4-repeat A/B found that the fused
  shared-expert gate_up lowered long-chain acceptance, so W16 keeps it off.
  Resulting agent-loop: 225-227 t/s.
* MTP head fine-tuning v1 and v2 had no measurable effect in the server. From
  here on, candidate heads were also scored offline with a "renewal" evaluator
  (a replay of the server's verify-and-advance process on the author's private
  data); its results are not published.
* Blended draft vocabulary `hot2_49152` in production from 19:10, same kernel
  step time. Its offline acceptance evaluation used private data and is not
  published.
* v3 head (soft-target distillation plus rollout loss) trained overnight.

## 2026-09-05: v3 head, MoE prologue patches, HC boundary, working adaptive width

* **v3 head adopted (07:40)**: W4 prose-en 2.54 acceptance / **230 t/s** (was
  217), agent-loop **289** (was 270); W16 neutral. Morning production:
  W4 code/prose-en/agent 332/230/289, W16 470/152/238.
* Draft vocab weight skipping (-2.4 GB resident). Step-time model fitted:
  `9.74 + 0.70*S` ms. Found that T=6 and T=10 verify widths fall back inside
  the verify graph.
* QSA metadata page-parallel (DG-1a), FlashInfer A0+A3 prologue patches
  (13:00), R3 MTP embedding table (14:00), R4 chain-parallel conv1d (17:00),
  R2+R5 GDN direct layouts (17:30), R1/R6/R7 HC boundary (20:00).
* Measured compositor interference: turning desktop compositing off shortened
  the W4 step by 7.5 %.
* `--speculative-adaptive` fixed (a shared draft-extend backend was the root
  cause of both the crash and the acceptance loss) and shipped as the `wa`
  profile (22:30). Code-edit 463 / prose-en 229 / prose-ja 205 / agent-loop 306
  t/s against W4 318/220/208/282 and W16 528/156/170/235.
* Rejected: A8 gated epilogue, K2 persistent megakernel.
* Evaluated offline on private data, no results or conclusions published, and
  not put into production: G3 draft-MoE top-k (23:30), EAGLE-3-style tap
  fusion (phase 1), position-specific adapters (phase 0).

## 2026-09-06: MoE glue, PDL, H1/H2, C1, singleton pruning

* FlashInfer G2-1/G2-2 and G1 group packing applied. PDL for the fork's Triton
  kernels.
* **final0906 (01:00, quiet night)**: W4 1,857 -> 1,596 kernels, wall
  10.5/9.9 ms, **364/266/324** t/s; W16 2,508 -> 2,211 kernels, wall 17.6/17.4
  ms, **512/175/314**; wa **576/256/328**.
* H2 layer-apply fold and H1-B norm-into-GEMV. **final0906b (02:02)**: W4 1,514
  kernels/step, **380/256/328**; W16 2,129 kernels, **566/182/278**; wa
  **550/264/334**. Compared with the previous morning that is +11-17 % at W4 and
  +17-20 % at W16 (quiet-night values).
* First exclusive-time map (02:30). By then no single remaining item was worth
  more than 2.2 %.
* Rejected overnight and during the day: J1, J2, G1 split-K, N1 NVFP4 dense
  GEMV (all stages). A V4 head retrain was evaluated offline on private data
  (no results or conclusions published) and never served.
* **C1 confidence policy (16:00)**: wa code-edit 480 -> **586** t/s, other
  workloads within ±3 %, 3-4 switches per benchmark.
* **P1 singleton pruning (21:30, W4/wa)**: W4 **388/272/346** (was 363/254/317);
  wa **638/269/306**. This is the first change that alters the target model's
  arithmetic.
* **P2 prune-in-prologue (23:00)**, all profiles: W16 step 18.4 -> 16.7 ms
  (-9.5 %). **Production smoke (23:35-23:41): W4 399/280/349, W16 671/187/314,
  wa 595/282/326** (`results/runs/p2m2332-*.jsonl`), needle PASS.

## 2026-09-07: v5 head, quality audit, width sweep, X3

* v5 head (broader re-extraction) passed server A/B (00:05). Production switched
  to it at 00:25.
* **Q1 quality audit (02:35)**: pruning on, pruning off and legacy builds show
  no statistically significant quality difference (GSM8K, MMLU, HumanEval,
  JCommonsenseQA, needle); non-inferiority at 0.5 pp was later found
  INCONCLUSIVE for GSM8K and HumanEval.
* `expandable_segments` allocator made the default (02:40) after start-up OOMs.
* Second exclusive-time map: median step **W4 8.92 ms (-10.5 % vs 09-06), W16
  16.08 ms (-7.7 %)**.
* Width sweep (WS1, 14:04, monitor on): best fixed width is W4 for prose, W8 for
  agent-loop (+15 %) and W16 for code (+70 %, 620 t/s).
* Three-width adaptive attempts WA2 and WA3 failed. X3 found that non-initial
  adaptive states skipped the target autotune. The fix went to production
  (23:40). Smoke with other GPU applications running on the desktop: code-edit 630/450,
  prose-en 251/235, agent-loop 311/372 t/s (two repeats each).
* MT1 serving-state head training: FAIL.

## 2026-09-08: three-width policy, measurement protocol, closing experiments

* **WA5 (02:15)**: in a quiet-night ABAB the three-width policy [3,7,15] gave
  agent-loop **+14.4 %** (both pairs), code-edit +0.7 %, prose within
  ±0.4 %. Adopted at 02:20.
  **Final production smoke (02:25, wa, two repeats): code-edit 669/659,
  prose-en 287/259, agent-loop 430/362 t/s** (`results/runs/wa5prod0221-wa.jsonl`),
  needle PASS.
* BN1 benchmark-noise study (10:08) set the measurement protocol used from then
  on. Held-out prompt sets, ABBA and bootstrap CIs showed that one-prompt t/s
  A/Bs cannot resolve 3 % effects.
* Closed with NO-GO or needs-more-work verdicts: draft-head shortlist (DH4-DH6),
  E0 MTP fc FP8, S1 sparse FP4, WA7 request prior, FP8D lossless coding, HC1
  bottleneck, E1 expert width. MT2, DH7 and WA6 were evaluated on private data
  (no results or conclusions published) and are not in production. QL1 mapped
  long-context costs. Details are in `rejected.md`.
* 19:45: work stopped. Final production configuration: v5 head, expandable
  segments, X3 autotune fix, wa with three widths [3,7,15] (step model
  `7.943 + 0.5554*S`), singleton pruning τ = 0.08 in the FlashInfer
  prologue.

## After the window

* 2026-09-09 (night): a τ = 0.10 shipping gate (T10) returned NO SHIP. See
  `rejected.md`.

* The production SGLang build (patch 0105) then stayed unchanged until 2026-10-02.

## 2026-10-01 / 10-02: fixed-cost cuts, sampling, rejection sampling

This round measured LM Studio's sampling settings next to greedy, and judged every server A/B with 8 server
starts and an ANCOVA (`measurement.md`, "The 2026-10 protocol"). Its numbers come from the BN1 held-out prompts (4 or 8
per workload, `measurement.md` §6) and are not comparable with the single-prompt smokes in the table below. Details:
`optimizations.md` §H, `rejected.md` §7.

* **10-01 early morning**: fixed-cost maps of the production step (fc-map, fc-moe, fc-glue). They found that the
  speculative verify ignored `min_p`, and that a sampling step cost 250-300 µs more than a greedy one. A host-RAM
  exhaustion freeze and a GPU hang (driver module hot-loaded after boot) cost a reboot; new rules: at least 110 GB
  of free host RAM before a server, compiles in their own capped scope.
* **10-01 morning to 09:50**: RS1 (dense rejection sampling) ABBAs without and with the min_p fix. Not adopted
  (code-edit −7.7 % / −3.6 % t/s); the min_p fix kept as a fidelity fix.
* **10-01 late morning to 13:42**: RT1, SV1, SV2, FG1, DG1 built and checked one by one; RQ2 u2h (FlashInfer
  prologue) ABBA −3.3 % ms/step. FG1's own 4-start ABBA was inconclusive (12:33-13:01), so the single-piece ABBAs
  of RT1 and DG1 were cancelled.
* **10-01 13:42-16:15**: **the stack ABBA, 8 starts: LM Studio ms/token −9.55 %, greedy −4.59 %** (arm-level CIs
  exclude 0 in both modes).
* **10-01 16:50 to 10-02 02:19**: ST1 (8 starts, no speed-up) and XA1 on ST1 (8 starts, LM Studio t/s +2.8 %,
  request level). DT1 was built; its exactness check found that greedy text is not reproducible across starts,
  so it stays off.
* **10-01 evening to 23:55**: RS2 sparse RS, 8 starts: faster on prose and agent, slower on code-edit; not
  adopted. A verify dump picked the RS2d knobs and estimated RS3 block verification offline (+0.20 % and
  +0.59 % tok/step).
* **10-02 02:19-06:40**: the RS package ABBA, 8 starts: LM Studio t/s +6.5 % pooled, code-edit unresolved
  (+1.6 % [−2.5, +5.9]). Not adopted by the rule; K = 16 joined it at 06:40.
* **10-02 morning**: **shipped**: the stack, ST1 + XA1, min_p on, and the RS package with K = 16 (patch 0121,
  FlashInfer 08-10), in a new worktree with a frozen FlashInfer copy; the old build kept as the rollback. Smokes
  on wa, W4 and W16 clean.
* **10-02 21:40-22:00**: 524,288-token context on wa with factor-2 YaRN: all needles up to 480k PASS; shipped as
  opt-in. 1M does not fit.

## Summary of production numbers

| Date / time | Config | code-edit | prose-en | agent-loop | Notes |
|---|---|--:|--:|--:|---|
| 09-02 am | baseline NEXTN steps 3 | 211 | 129 | 174 | recommended sampling, 300 W |
| 09-02 | W8A16 + hot_32768 (W4) | 261 | 163 | 244 | recommended sampling |
| 09-02 | static W16 | 429-433 | 100-104 | 189-192 | |
| 09-03 | W16 after glue round, greedy | 454 / 443 | | | acceptance 11.5 / 10.8 |
| 09-04 09:03 | W4 final4, greedy | 319 | 208 | 287 | |
| 09-05 am | W4 / W16 (v3 head) | 332 / 470 | 230 / 152 | 289 / 238 | |
| 09-05 22:30 | wa {3,15} EMA | 463 | 229 | 306 | same window: W4 318/220/282, W16 528/156/235 |
| 09-06 01:00 | W4 / W16 / wa | 364 / 512 / 576 | 266 / 175 / 256 | 324 / 314 / 328 | quiet night |
| 09-06 02:02 | W4 / W16 / wa | 380 / 566 / 550 | 256 / 182 / 264 | 328 / 278 / 334 | quiet night |
| 09-06 23:3x | W4 / W16 / wa (P2 pruning) | 399 / 671 / 595 | 280 / 187 / 282 | 349 / 314 / 326 | quiet night |
| 09-07 14:04 | fixed-width sweep W4 / W8 / W16 | 363 / 533 / 620 | 264 / 231 / 182 | 312 / 365 / 298 | monitor on, v5 head; `results/runs/ws11337-*.jsonl` |
| 09-08 02:25 | **wa [3,7,15], final** | **669 / 659** | **287 / 259** | **430 / 362** | two repeats, quiet night; `results/runs/wa5prod0221-wa.jsonl` |

All rows are greedy from 09-02 afternoon onward. Rows taken at different times
of day are not directly comparable. See `measurement.md`.
