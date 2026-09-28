# WA5: quiet-night three-width confidence policy re-test

Status: COMPLETE — 2026-09-08 JST. Specified pooled C-vs-A gate PASS; conservative individual-repeat check FAIL for C2 code-edit (-0.73% vs pooled A). Recommendation: hold automatic production adoption; exact unapplied change documented. Four arms/48 measurements/4 needles complete; production unchanged. Protocol was registered before GPU execution.

## Protocol and preflight

- Read WA4_LOG.md, X4_THREE_CANDIDATE_COST.md, WA2_LOG.md Iteration 3 and C1_LOG.md. User reports another GPU application was running during WA4; X4 did not reproduce the same-width penalty. This is motivation, not retrospective proof of causality.
- Order A1 -> C1 -> A2 -> C2, per-arm `flock -w 28800 $HOME/.gpu.lock`. Fresh timestamped labels, exclusive creation. Lock is released after each arm including cleanup. No extra policy iteration if the gate fails.
- A: production codex/perf-v1 at 446c801189165380465a73782c93658576283372, actual launcher wa defaults (w16_conf.json, X3 autotune=1, STEP_A/B=9.74/0.70).
- C: existing sglang-wa4 / codex/wa4-three-widths, same HEAD and X3 fix plus the existing staged two-file WA2 policy patch; config adaptive/w16_3_7_15_c.json, STEP_A/B=7.943/0.5554, autotune=1. No new branch, commit or push.
- All arms: MEM_FRACTION=W16_MEM_FRACTION=WA_MEM_FRACTION=0.920, MAX_TOTAL_TOKENS=131072, SERVE_DISPLAY_HZ empty. Single stream, workload order code-edit/prose-en/prose-ja/agent-loop, one warmup and n=3 greedy measured requests per workload; needle 18500 / depth 0.4 once per arm.
- Identical trace/debug instrumentation. Width histograms count measured verified decode steps, W=S+1, exact client time windows exclude warmups/needle. Validate trace emissions within 32 tokens of client totals and report SSE acceptance cross-check.
- Primary gate: equal-weight mean of C1/C2 versus equal-weight mean of A1/A2 (n=6 per workload per policy): code-edit >=0%, prose-en/prose-ja >=-2%, agent-loop >=+5%, both C needles PASS. Also report each C arm versus pooled A, both paired comparisons C1/A1 and C2/A2, and control drift. For adoption, both C instances must individually satisfy the stated pooled-A thresholds; pooled values alone cannot conceal a failing instance.
- Production, launcher and venv mounted read-only with bwrap; WA4 overlay read-only. Hardcoded production .cache maps to a WA5-private copy of the prior WA4 cache. Global FlashInfer $HOME/.cache/sglang remains shared as in X4; caches are not frozen between arms. PYTHONDONTWRITEBYTECODE=1. Source/harness hashes recorded in specs/wa5/.
- GPU telemetry every 2 s: free VRAM, utilization, temperature, power, clocks and power/thermal flags; presence of another GPU application recorded. Full NVIDIA/process snapshots before/after. Quiet means no observed other GPU application or competing compute job, not an assertion of no desktop scheduling. Display/clock/power settings unchanged.
- Require >=4096 MiB free at steady state; startup dips recorded separately. Abort owned server group if <4096 MiB persists 10 s, and invalidate the reserve gate for any observed steady dip below 4096. Only owned process groups terminated; no pkill/pgrep.
- Initial sandbox nvidia-smi could not access driver; authorized outside-sandbox read-only preflight succeeded: RTX PRO 6000, ~1.6 GiB desktop allocation, 1% GPU utilization, no compute job or other GPU application. No driver/system change.

## Execution

### Starting wa5-20260908-015812-A1

Arm A, 2026-09-08T01:56:47.113353+09:00; per-arm flock held.

wa5-20260908-015812-A1: ready; 10856 MiB free VRAM.

CPU preflight: 59 adaptive tests PASS (0.705 s), including confidence policy and target autotune tests. Existing staged diff check PASS; no policy code changed. Evidence: wa5/cpu-tests.txt. A1 label was allocated manually at preparation time; authoritative start timestamps are the actual log/telemetry times (01:56:47 JST), not the label timestamp. Subsequent labels use the current clock.

wa5-20260908-015812-A1: owned process groups cleaned up; lock released on command exit.

## Arm wa5-20260908-015812-A1

| workload | t/s mean (min-max), n=3 | acceptance (trace / SSE) | W4 count (%) | W8 count (%) | W16 count (%) |
|---|---:|---:|---:|---:|---:|
| code-edit | 577.71 (530.32-663.36) | 10.463 / 10.352 | 0 (0.00%) | 0 (0.00%) | 703 (100.00%) |
| prose-en | 262.98 (259.83-267.59) | 2.518 / 2.513 | 1431 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| prose-ja | 255.86 (247.16-266.36) | 2.402 / 2.403 | 1501 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| agent-loop | 337.78 (322.83-348.76) | 3.480 / 3.458 | 962 (91.44%) | 0 (0.00%) | 90 (8.56%) |

Needle: `[wa5-20260908-015812-A1] prompt_tokens=17307 completion=12 time=1.4s PASS=True out='AURORA-CEDAR-7319'`

Steady-state free VRAM: ready 10856 MiB; minimum 8731 MiB (8.526 GiB); median 9432 MiB. Startup minimum 6503 MiB, reported separately. Samples every 2 s; cleanup excluded. Steady 4 GiB reserve: PASS.

Trace cross-check, sum(1 + accepted drafts) minus completion_tokens by repeat: code-edit: [26, 25, 7]; prose-en: [3, 1, -1]; prose-ja: [3, 1, -1]; agent-loop: [3, 2, 13].

Artifacts: `specs/wa5/wa5-20260908-015812-A1/` (server, trace, exact request boundaries, fnbench JSONL, needle, VRAM, summary).

### Starting wa5-20260908-020008-C1

Arm C, 2026-09-08T02:00:08.381903+09:00; per-arm flock held.

wa5-20260908-020008-C1: ready; 9585 MiB free VRAM.

wa5-20260908-020008-C1: owned process groups cleaned up; lock released on command exit.

## Arm wa5-20260908-020008-C1

| workload | t/s mean (min-max), n=3 | acceptance (trace / SSE) | W4 count (%) | W8 count (%) | W16 count (%) |
|---|---:|---:|---:|---:|---:|
| code-edit | 604.32 (523.28-656.08) | 10.762 / 10.654 | 0 (0.00%) | 70 (10.14%) | 620 (89.86%) |
| prose-en | 268.11 (262.56-276.78) | 2.585 / 2.578 | 1316 (94.27%) | 80 (5.73%) | 0 (0.00%) |
| prose-ja | 254.33 (246.54-269.05) | 2.385 / 2.382 | 1512 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| agent-loop | 384.11 (356.96-423.33) | 4.667 / 4.639 | 0 (0.00%) | 777 (100.00%) | 0 (0.00%) |

Needle: `[wa5-20260908-020008-C1] prompt_tokens=17307 completion=12 time=1.4s PASS=True out='AURORA-CEDAR-7319'`

Steady-state free VRAM: ready 9585 MiB; minimum 7638 MiB (7.459 GiB); median 8165 MiB. Startup minimum 6465 MiB, reported separately. Samples every 2 s; cleanup excluded. Steady 4 GiB reserve: PASS.

Trace cross-check, sum(1 + accepted drafts) minus completion_tokens by repeat: code-edit: [24, 28, 21]; prose-en: [4, -1, 0]; prose-ja: [1, -1, -1]; agent-loop: [4, -1, 0].

Artifacts: `specs/wa5/wa5-20260908-020008-C1/` (server, trace, exact request boundaries, fnbench JSONL, needle, VRAM, summary).

### Starting wa5-20260908-020336-A2

Arm A, 2026-09-08T02:03:36.774308+09:00; per-arm flock held.

### Pair 1 interim checkpoint

| workload | C1 vs A1 |
|---|---:|
| code-edit | +4.61% |
| prose-en | +1.95% |
| prose-ja | -0.60% |
| agent-loop | +13.72% |

Pair 1 clears all throughput thresholds against A1, and both needles/reserve pass; final gate awaits A2/C2. C1 measured agent is 100% W8 and Japanese prose 100% W4. Unlike WA4 C, code includes 70 W8 steps (10.14%) and English prose 80 W8 steps (5.73%). Code batch345 demotes 15->7 on 343/294=1.1667 >1.15, then batch415 promotes back on 694/600=1.1567 >1.144. English batch1865 promotes 3->7 on 380/318=1.1950 >1.144, then reverses at batch1945 after an 80-batch hold when 298/280=1.0643 >1.03. These observed excursions do not cause a pair-1 aggregate threshold failure. No config retuning. Exact evidence: C1 diagnostics.json and server.log.

wa5-20260908-020336-A2: ready; 10856 MiB free VRAM.

wa5-20260908-020336-A2: owned process groups cleaned up; lock released on command exit.

## Arm wa5-20260908-020336-A2

| workload | t/s mean (min-max), n=3 | acceptance (trace / SSE) | W4 count (%) | W8 count (%) | W16 count (%) |
|---|---:|---:|---:|---:|---:|
| code-edit | 605.47 (527.63-646.11) | 10.939 / 10.824 | 0 (0.00%) | 0 (0.00%) | 672 (100.00%) |
| prose-en | 276.96 (269.24-284.83) | 2.647 / 2.640 | 1362 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| prose-ja | 243.14 (239.93-246.15) | 2.282 / 2.301 | 1576 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| agent-loop | 327.83 (319.28-334.29) | 3.145 / 3.129 | 1148 (100.00%) | 0 (0.00%) | 0 (0.00%) |

Needle: `[wa5-20260908-020336-A2] prompt_tokens=17307 completion=12 time=1.5s PASS=True out='AURORA-CEDAR-7319'`

Steady-state free VRAM: ready 10856 MiB; minimum 8772 MiB (8.566 GiB); median 9432 MiB. Startup minimum 6505 MiB, reported separately. Samples every 2 s; cleanup excluded. Steady 4 GiB reserve: PASS.

Trace cross-check, sum(1 + accepted drafts) minus completion_tokens by repeat: code-edit: [26, 19, 30]; prose-en: [4, -1, 1]; prose-ja: [-1, -1, -1]; agent-loop: [6, 2, 1].

Artifacts: `specs/wa5/wa5-20260908-020336-A2/` (server, trace, exact request boundaries, fnbench JSONL, needle, VRAM, summary).

### Starting wa5-20260908-020710-C2

Arm C, 2026-09-08T02:07:10.074146+09:00; per-arm flock held.

### A2 checkpoint

| workload | A2 vs A1 | mean A | C1 vs mean A |
|---|---:|---:|---:|
| code-edit | +4.81% | 591.59 | +2.15% |
| prose-en | +5.32% | 269.97 | -0.69% |
| prose-ja | -4.97% | 249.50 | +1.94% |
| agent-loop | -2.94% | 332.81 | +15.42% |

C1 passes each threshold against the now-complete A mean. Final C2 remains pending. A1 sole steady power-cap sample at 01:59:47.894 is outside all fnbench request windows, following agent completion.

wa5-20260908-020710-C2: ready; 9624 MiB free VRAM.

wa5-20260908-020710-C2: owned process groups cleaned up; lock released on command exit.

## Arm wa5-20260908-020710-C2

| workload | t/s mean (min-max), n=3 | acceptance (trace / SSE) | W4 count (%) | W8 count (%) | W16 count (%) |
|---|---:|---:|---:|---:|---:|
| code-edit | 587.24 (446.71-681.25) | 10.174 / 10.062 | 80 (10.26%) | 120 (15.38%) | 580 (74.36%) |
| prose-en | 274.06 (264.92-287.12) | 2.613 / 2.607 | 1380 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| prose-ja | 245.60 (244.15-247.71) | 2.313 / 2.321 | 1558 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| agent-loop | 377.15 (366.91-383.86) | 4.641 / 4.607 | 0 (0.00%) | 778 (100.00%) | 0 (0.00%) |

Needle: `[wa5-20260908-020710-C2] prompt_tokens=17307 completion=12 time=1.4s PASS=True out='AURORA-CEDAR-7319'`

Steady-state free VRAM: ready 9624 MiB; minimum 7478 MiB (7.303 GiB); median 8202 MiB. Startup minimum 6505 MiB, reported separately. Samples every 2 s; cleanup excluded. Steady 4 GiB reserve: PASS.

Trace cross-check, sum(1 + accepted drafts) minus completion_tokens by repeat: code-edit: [20, 30, 26]; prose-en: [0, 0, 1]; prose-ja: [3, -1, 2]; agent-loop: [5, 3, 0].

Artifacts: `specs/wa5/wa5-20260908-020710-C2/` (server, trace, exact request boundaries, fnbench JSONL, needle, VRAM, summary).

## Final throughput and gate

| workload | A1 t/s | C1 t/s | A2 t/s | C2 t/s | A mean | C mean | C vs A | threshold | pooled gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| code-edit | 577.71 | 604.32 | 605.47 | 587.24 | 591.59 | 595.78 | +0.71% | +0% | PASS |
| prose-en | 262.98 | 268.11 | 276.96 | 274.06 | 269.97 | 271.09 | +0.41% | -2% | PASS |
| prose-ja | 255.86 | 254.33 | 243.14 | 245.60 | 249.50 | 249.96 | +0.19% | -2% | PASS |
| agent-loop | 337.78 | 384.11 | 327.83 | 377.15 | 332.81 | 380.63 | +14.37% | +5% | PASS |

### Individual C versus pooled A and paired drift

| workload | C1 vs mean A | C2 vs mean A | C1 vs A1 | C2 vs A2 | A2 vs A1 | C2 vs C1 |
|---|---:|---:|---:|---:|---:|---:|
| code-edit | +2.15% | -0.73% | +4.61% | -3.01% | +4.81% | -2.83% |
| prose-en | -0.69% | +1.52% | +1.95% | -1.05% | +5.32% | +2.22% |
| prose-ja | +1.94% | -1.56% | -0.60% | +1.01% | -4.97% | -3.43% |
| agent-loop | +15.42% | +13.32% | +13.72% | +15.04% | -2.94% | -1.81% |

All figures use arithmetic means of the three measured request decode throughputs per arm; the pooled policy means weight both instances equally (six requests each). Individual C gates use the same pooled A baseline. Pair deltas diagnose drift and do not replace the registered baseline.

Verdict: {"pooled_pass": true, "individual_pass": {"C1": true, "C2": false}, "needle_pass": true, "reserve_pass": true, "adoption_pass": false}.

### VRAM and needle

| arm | ready MiB | steady median MiB | steady min MiB | reserve | needle |
|---|---:|---:|---:|---|---|
| A1 | 10856 | 9432 | 8731 | PASS | PASS |
| C1 | 9585 | 8165 | 7638 | PASS | PASS |
| A2 | 10856 | 9432 | 8772 | PASS | PASS |
| C2 | 9624 | 8202 | 7478 | PASS | PASS |

## Final mechanism and repeat-stability assessment

**Specified pooled gate: PASS. Conservative repeat-stability check: FAIL (C2 code-edit only).** The pre-run protocol additionally required both C instances to pass individually versus pooled A for an adoption recommendation. That stricter check is an assistant-added robustness criterion, not a replacement for the user-specified pooled C-versus-A gate. Consequently the headline benchmark gate remains PASS; the recommendation below is more conservative than that numerical gate. No fifth server, additional tuning or changed threshold was used.

| arm | workload | mean acceptance | effective step ms | measured W4/W8/W16 counts |
|---|---|---:|---:|---|
| A1 | code-edit | 10.463 | 18.083 | 0/0/703 |
| A1 | prose-en | 2.518 | 9.576 | 1431/0/0 |
| A1 | prose-ja | 2.402 | 9.390 | 1501/0/0 |
| A1 | agent-loop | 3.480 | 10.291 | 962/0/90 |
| C1 | code-edit | 10.762 | 17.735 | 0/70/620 |
| C1 | prose-en | 2.585 | 9.639 | 1316/80/0 |
| C1 | prose-ja | 2.385 | 9.375 | 1512/0/0 |
| C1 | agent-loop | 4.667 | 12.143 | 0/777/0 |
| A2 | code-edit | 10.939 | 18.053 | 0/0/672 |
| A2 | prose-en | 2.647 | 9.558 | 1362/0/0 |
| A2 | prose-ja | 2.282 | 9.388 | 1576/0/0 |
| A2 | agent-loop | 3.145 | 9.593 | 1148/0/0 |
| C2 | code-edit | 10.174 | 17.011 | 80/120/580 |
| C2 | prose-en | 2.613 | 9.534 | 1380/0/0 |
| C2 | prose-ja | 2.313 | 9.419 | 1558/0/0 |
| C2 | agent-loop | 4.641 | 12.304 | 0/778/0 |

Effective ms is the mean across requests of `1000 * trace tokens per verify / client decode tps`; it is not a CUPTI/kernel measurement. Mixed-width rows cannot be read as one physical width cost.

C2 code-edit repeat 3 produces 446.71 t/s, acceptance 6.452 tokens/verify, versus repeats 1/2 at 633.76/681.25 t/s with acceptance 11.415/12.656. Trace repeat 3 contains 80 W4, 120 W8 and 176 W16 verify steps; A1/A2 code is entirely W16. All 200 narrower C2 code steps occur in this measured request. The exact decision path is:

- Batch 845: S15 -> S3. Forecast W4=309 versus W16=293 t/s, ratio 1.0546, clears the configured down-to-3 threshold 1.03. The legacy production threshold 1.12 would reject this same rounded snapshot (not a full-trajectory counterfactual). The reversal hold lasts 80 batches.
- Batch 925: S3 -> S7. W8/W4=429/353=1.2153 clears 1.144. W16 is also estimated at 440 t/s, but adjacent_only_promotion prevents a direct S3 -> S15 jump. A further 80-batch hold applies.
- Batch 1005: promotion to W16 remains blocked: 609/579=1.0518 <1.144. The controller remains W8 through batch 1035.
- Batch 1045: S7 -> S15 after 696/604=1.1523 >1.144; total narrower-width dwell is 200 decode steps.

Thus the observed weak repeat has a policy/trajectory failure mode: a low-acceptance passage permits a W4 demotion, after which grace plus adjacent promotion and the upward margin delay recovery to W16. Its mixed-width effective step is 14.444 ms, below the two W16-only C2 requests at 18.012/18.578 ms; lower token yield per verify accompanies the throughput loss. This is not the WA4 pattern of 100% correct widths plus unexplained multi-ms same-width cost. It does not prove how fast that exact request would have been under a counterfactual fixed-W16 trajectory, nor assign all -0.73% to the switches. The small sample and low-acceptance passage remain relevant.

C1 code also briefly demotes to W8 (70 steps) and C1 English prose briefly promotes to W8 (80 steps), as documented at the pair-1 checkpoint. Both C agent-loop instances remain W8 for 100% of measured steps: 777 and 778 verifies. The agent improvement repeats in the adjacent pairs (+13.72%/+15.04%) and against pooled A (+15.42%/+13.32%). Both prose pooled deltas are positive, and C2 English is entirely W4. The historical WA4 code/English same-width losses are not reproduced as an invariant third-candidate penalty; no new profiler evidence was collected to identify historical another GPU application causality.


## Recommendation and exact unapplied production delta

Recommendation: **hold the automatic production default change**. The requested pooled gate passes, and the agent gain is repeated, but code has only +0.71% pooled headroom and C2 loses -0.73% versus pooled A / -3.01% versus its adjacent A2 while exhibiting recoverable narrow-width dwell. The original WA4 environmental suspicion no longer explains all observed weaknesses; this run has a concrete policy-decision concern. This is a conservative recommendation despite pooled PASS, not a claim that the requested pooled gate failed. Stop here: no config d, fifth arm, new branch, commit, push or production edit.

Because the specified pooled gate passes, the exact production change is recorded below for review. It was NOT applied or executed.

1. Integrate the policy patch currently staged in `sglang-wa4` / `codex/wa4-three-widths` into production `codex/perf-v1`. Both branches currently point to `446c801189165380465a73782c93658576283372`; merging the current WA4 HEAD alone would be a no-op and would omit the policy. The staged patch has stable patch-id `f9d7c0bcde4f48a227f718270b721dcfe6e3c422`, exactly equal to existing WA2 commit `2671ec1b00b8428e1da032b1c6af33b72bcbb3fe`. Future integration can cherry-pick that existing policy commit onto production, or commit the already-staged identical patch on the existing WA4 branch and merge that resulting commit. No nonexistent WA4 commit hash is invented. X3 `446c801189` is already present; no second X3 integration is needed. Changed files are `python/sglang/srt/speculative/adaptive_confidence.py` and `test/registered/unit/spec/test_adaptive_confidence.py`.
2. In `serve-fast.sh`, only the `wa)` case needs these profile-scoped environment defaults before the existing `exec`, while retaining its existing X3 flag:

```sh
SGLANG_ADAPTIVE_TARGET_AUTOTUNE="${SGLANG_ADAPTIVE_TARGET_AUTOTUNE:-1}" \
SGLANG_ADAPTIVE_STEP_A="${SGLANG_ADAPTIVE_STEP_A:-7.943}" \
SGLANG_ADAPTIVE_STEP_B="${SGLANG_ADAPTIVE_STEP_B:-0.5554}" \
```

Replace the wa case's existing config argument with:

```sh
--speculative-adaptive-config "${ADAPTIVE_CONFIG:-$HOME/tools/flash-next-bench/adaptive/w16_3_7_15_c.json}"
```

For an exact single-line proposed `wa)` case and unified diff, see `wa5/proposed-serve-fast.patch`. This is a text artifact only. Keep the environment values profile-scoped, not global STEP defaults. The existing config file hash is `ce61929aac08d41ca7d81a63580ac4bc1f099fca95d2f99937712b9725938faa`. Tracing/debug remain benchmark-only. Reverting just ADAPTIVE_CONFIG is insufficient to reproduce A if the new STEP defaults remain; an exact A fallback also sets STEP_A/B=9.74/0.70.

Risk note: all throughput/decision evidence is bs=1. Slot 2 remains pinned to S15 and equivalent to production, but bs>1 operation, batching transitions and concurrency performance were not exercised. The only retrieval check is a 17,307-token prompt (four PASS results), not saturated long-context correctness/performance. Every arm used fraction 0.920 plus MAX_TOTAL_TOKENS=131072; launcher-default 0.925 and uncapped KV capacity are untested and are not validated by this proposal. X4 measured the third candidate at approximately 1.20 GiB extra allocation; WA5's paired ready-memory differences are 1,271 and 1,232 MiB (1.241/1.203 GiB), consistent with that scale but including desktop allocation variation. Combined with the observed code downshift/recovery path and debug-trace instrumentation, these constraints limit how broadly the pooled PASS can be applied.


## Quiet-condition audit and completion

| arm | other-GPU-app samples | measured power-cap / thermal-cap samples | measured SM median by code/en/ja/agent (MHz) | measured temperature range C |
|---|---:|---|---|---|
| A1 | 0 | 0/0 | 2280/2310/2310/2310 | 45-71 |
| C1 | 0 | 0/0 | 2287/2310/2310/2310 | 52-75 |
| A2 | 0 | 0/0 | 2287/2310/2310/2310 | 54-75 |
| C2 | 0 | 0/0 | 2287/2310/2310/2310 | 53-75 |

Two-second telemetry found no other GPU application and no power/thermal cap active inside measured request windows. These are sampled observations, not continuous proof of no brief stalls. Pre-start compute lists were empty under the per-arm shared lock; NVIDIA snapshots retain the normal desktop processes. No display, power or clock setting was changed. A1 post-workload power-cap sample remains included in the raw record.

All four arms completed in A1 -> C1 -> A2 -> C2 order, approximately 01:56:47-02:10:25 JST (same-hour span, about 14 minutes; no shared-lock queue delay). All 48 measured requests, 16 warmups, 4 needle tests and all request/trace validation checks completed. Minimum steady free VRAM 7478 MiB (7.303 GiB); every arm passes >=4096 MiB. 59 CPU adaptive tests pass. Final source hashes, branch/HEAD/status, staged patch hash and benchmark inputs match the preflight snapshots; production/launcher/venv were mounted read-only and no installation was performed. All four owned server process groups have zero remaining processes. GPU runs have stopped. Evidence: `wa5/gate.json`, `wa5/repeat-analysis.json`, `wa5/completeness.json`, per-arm `diagnostics.json`, command/environment, server/trace/events, fnbench/needle and GPU telemetry files.

Final 02:13 JST GPU check: another SGLang task is now using the GPU (process group 3567023), after WA5 released its lock. It is not one of the four recorded WA5 process groups and was left untouched. Rechecked all four WA5 groups: zero surviving processes. Evidence: wa5/process-cleanup-final.json. The proposed launcher diff passed bash -n on in-memory text; the actual launcher was not edited.
